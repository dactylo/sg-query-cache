"""Wrap synchronous SG reads with exact caching and conservative write fences."""

from __future__ import annotations

from collections.abc import Callable, Generator, Mapping
from contextlib import contextmanager, suppress
from importlib import import_module
from os import environ as process_environ
from threading import Lock
from typing import TYPE_CHECKING, Any, Self, TypeAlias, cast

from sg_query_cache._cache import QueryCache, client_binding
from sg_query_cache._client_common import (
    ClientContext,
    EntityPlanContext,
    EntityRead,
    WritePlan,
    batch_write_plan,
    client_options,
    create_write_plan,
    entity_calls,
    entity_read_plan,
    exception_types,
    lifecycle_changes,
    parse_schema_view,
    population_callback,
    require_context_namespace,
    schema_changes,
    schema_entity_read_call,
    schema_field_read_call,
    schema_plan,
    schema_read_call,
    schema_view_call,
    site_call,
    site_changes,
    ttl_only_plan,
    unconditional_entity_plan,
    update_write_plan,
    work_schedule_read_call,
)
from sg_query_cache._dependencies import DependencyAnalyzer, SchemaView
from sg_query_cache._local import LocalBackend
from sg_query_cache._metrics import MetricsSink, NullMetrics
from sg_query_cache._model import (
    DependencyPlan,
    ExactCall,
    Identity,
)

if TYPE_CHECKING:
    from contextlib import AbstractContextManager
    from datetime import datetime

    from sg_query_cache._backend import CacheBackend
    from sg_query_cache._cache_common import PolicyMode
    from sg_query_cache._config import CacheSettings
    from sg_query_cache.valkey import ValkeyBackend

_Call: TypeAlias = Callable[[], object]


class CachedShotgun:
    """Cache explicitly supported synchronous SG calls for one exact identity.

    Wrapped SG methods preserve the raw client's argument and result contracts.
    Cache failures fall back to SG reads; SG failures propagate. Writes attempt
    fences before dispatch and after success or a configured indeterminate
    transport failure. Composite create, update, and batch writes also fence
    after other ordinary failures.

    Direct construction owns neither the raw client nor the injected cache.
    `open` owns the cache layers it constructs; the raw client remains
    caller-owned. Keep the immutable response context consistent with the raw
    client's effective configuration. Policy scopes are thread-local. Use
    `raw_client` for unsupported operations; those calls bypass these fences.
    """

    __slots__ = (
        '_analyzer',
        '_cache',
        '_client',
        '_context',
        '_context_lock',
        '_identity_value',
        '_indeterminate_write_exceptions',
        '_settings',
    )

    def __init__(
        self,
        client: object,
        cache: QueryCache,
        context: ClientContext,
        *,
        indeterminate_write_exceptions: tuple[type[Exception], ...] = (),
    ) -> None:
        """Bind a caller-owned client and cache to one response context.

        Args:
            client: Caller-owned thread-safe synchronous SG client.
            cache: Query cache already bound to the effective caller identity.
            context: Immutable site and response-affecting client context.
            indeterminate_write_exceptions: Explicit transport exception
                classes whose outcomes require a conservative post-write fence.

        Raises:
            TypeError: Indeterminate write exceptions are not a tuple of
                Exception subclasses.
            ValueError: Context and shared cache namespaces disagree.
        """
        settings, identity = client_binding(cache)
        require_context_namespace(settings, context)
        self._analyzer = DependencyAnalyzer(
            enable_additional_filter_presets=settings.enable_additional_filter_presets
        )
        self._cache = cache
        self._client = client
        self._context = context
        self._context_lock = Lock()
        self._identity_value = identity
        self._indeterminate_write_exceptions = exception_types(indeterminate_write_exceptions)
        self._settings = settings

    def _identity(self) -> Identity:
        """Return the caller-supplied identity already bound to the cache."""
        return self._identity_value

    def _read(
        self,
        call: ExactCall,
        plan: DependencyPlan,
        fetch: _Call,
        context: ClientContext,
        returned_fields: Mapping[str, str] | None = None,
    ) -> object:
        """Execute one exact read and safely apply response-derived age caps."""
        return self._cache.read(
            site_call(call, context),
            plan,
            fetch,
            client_options(context, call),
            _population_decision=population_callback(returned_fields, self._settings),
        )

    def _read_direct(
        self,
        method: str,
        fetch: _Call,
        context: ClientContext,
        *,
        reason: str = 'unsupported_argument',
    ) -> object:
        """Bypass the cache, without building a call key, for one stated reason."""
        call = ExactCall(method, ())
        return self._cache.read(
            call,
            DependencyPlan('bypass', (), reason),
            fetch,
            client_options(context, call),
        )

    def _schema_plan(self, context: ClientContext) -> DependencyPlan:
        """Build the fixed schema plan for the current exact identity."""
        return schema_plan(self._identity(), context.site_namespace, self._settings)

    def _schema_read(
        self,
        project_entity: dict[str, object] | None,
        context: ClientContext,
    ) -> object:
        """Read full schema through the nonrecursive cache boundary."""
        method = cast('Any', self._client).schema_read

        def fetch() -> object:
            return method(project_entity)

        call = schema_read_call(project_entity)
        if call is None:
            return self._read_direct('schema_read', fetch, context)
        return self._read(
            call,
            self._schema_plan(context),
            fetch,
            context,
        )

    def _schema_view(self, context: ClientContext) -> SchemaView:
        """Load the identity-scoped compact schema view through the parsed memo."""
        method = cast('Any', self._client).schema_read
        call = schema_view_call(context)
        value = self._cache.read_parsed(
            call,
            self._schema_plan(context),
            lambda: SchemaView.from_api_response(method(None)).to_compact(),
            client_options(context, call),
            parse_schema_view,
        )
        if not isinstance(value, SchemaView):
            return SchemaView.from_api_response(None)
        return value

    def _schema_view_or_empty(self, context: ClientContext) -> SchemaView:
        """Return an empty conservative view when schema acquisition fails."""
        try:
            return self._schema_view(context)
        except Exception:  # noqa: BLE001 - unavailable schema must not block the read
            return SchemaView.from_api_response(None)

    def _ttl_only_plan(self, method: str, context: ClientContext) -> DependencyPlan:
        """Build a disabled-by-default universal plan for one explicit method."""
        return ttl_only_plan(method, self._identity(), context.site_namespace, self._settings)

    def _write_with_fence(self, plan: WritePlan, call: _Call) -> object:
        """Fence one write before dispatch and after nondefinite outcomes."""
        self._cache.invalidate(plan.changes)
        try:
            value = call()
        except Exception as error:
            if plan.repeat_on_exception or isinstance(
                error,
                self._indeterminate_write_exceptions,
            ):
                self._cache.invalidate(plan.indeterminate_changes or plan.changes)
            raise
        self._cache.invalidate(
            plan.post_changes(value) if plan.post_changes is not None else plan.changes
        )
        return value

    def bypass(self) -> AbstractContextManager[None]:
        """Bypass reads and population within the current thread-local scope."""
        return self._cache.bypass()

    @property
    def context(self) -> ClientContext:
        """Return the immutable context used by subsequent operations."""
        with self._context_lock:
            return self._context

    def _entity_read(self, read: EntityRead, fetch: _Call) -> object:
        """Plan and read one `find` or `find_one`, reading directly when unnormalizable."""
        context = self.context
        analysis_call, call = entity_calls(read)
        if analysis_call is None or call is None:
            return self._read_direct(read.method, fetch, context)
        plan = unconditional_entity_plan(
            self._identity(), self._settings, read.entity_type, lambda: self._cache.bypass_active
        )
        returned_fields = None
        if plan is None:
            schema = self._schema_view_or_empty(context)
            plan, returned_fields = entity_read_plan(
                analysis_call,
                schema,
                read.entity_type,
                read.fields,
                EntityPlanContext(
                    self._analyzer, self._identity(), self._settings, context.site_namespace
                ),
            )
        return self._read(call, plan, fetch, context, returned_fields)

    def activity_stream_read(  # noqa: PLR0913, PLR0917 - exact official SG signature
        self,
        entity_type: str,
        entity_id: int,
        entity_fields: dict[str, object] | None = None,
        min_id: int | None = None,
        max_id: int | None = None,
        limit: int | None = None,
    ) -> dict[str, object]:
        """Always read an activity stream directly.

        Bypass cache lookup and population because this composite response has
        no proven URL lifetime for caching.

        Arguments, results and SG errors are those of
        `Shotgun.activity_stream_read`.

        Raises:
            RuntimeError: The cache has been closed.
        """
        context = self.context
        method = cast('Any', self._client).activity_stream_read

        def fetch() -> object:
            return method(entity_type, entity_id, entity_fields, min_id, max_id, limit)

        return cast(
            'dict[str, object]',
            self._read_direct(
                'activity_stream_read', fetch, context, reason='composite_url_unproven'
            ),
        )

    def batch(self, requests: list[dict[str, object]]) -> list[dict[str, object] | bool]:
        """Execute one ordered batch behind its complete conservative fence.

        Attempt invalidation before SG dispatch and after success or a
        configured indeterminate transport failure. Failed cache fences do not
        prevent SG dispatch.

        Writes containing implicit uploads use a site-wide fence and repeat it
        after other ordinary failures as well.

        Arguments, results and SG errors are those of `Shotgun.batch`.

        Raises:
            RuntimeError: The cache has been closed.
        """
        method = cast('Any', self._client).batch
        return cast(
            'list[dict[str, object] | bool]',
            self._write_with_fence(
                batch_write_plan(self.context.site_namespace, requests),
                lambda: method(requests),
            ),
        )

    def cache_policy(
        self,
        mode: PolicyMode | None = None,
        maximum_age: float | None = None,
        *,
        maximum_age_seconds: float | None = None,
    ) -> AbstractContextManager[None]:
        """Apply one validated thread-local cache policy scope.

        Passing non-`None` values for both age spellings raises `TypeError`.
        Entering the returned context validates its arguments. Nonfinite,
        boolean, or nonnumeric ages raise `TypeError`; unsupported modes or
        nonpositive ages raise `ValueError`.

        Args:
            mode: Optional conservative downgrade to event-aware, TTL-only, or
                bypass.
            maximum_age: Positional or keyword alias for `maximum_age_seconds`.
            maximum_age_seconds: Optional positive age in seconds, combined by
                minimum with all active scopes and per-call ages. The selected
                explicit age replaces configured ages before safety caps apply.

        Returns:
            A nestable context manager delegated to the query cache.
        """
        if maximum_age is not None and maximum_age_seconds is not None:
            raise TypeError('maximum_age and maximum_age_seconds cannot both be provided')
        age = maximum_age if maximum_age_seconds is None else maximum_age_seconds
        return self._cache.policy(mode, age)

    def create(
        self,
        entity_type: str,
        data: dict[str, object],
        return_fields: list[str] | None = None,
    ) -> dict[str, object]:
        """Create one entity behind conservative membership fences.

        Attempt invalidation before SG dispatch and after success or a
        configured indeterminate transport failure. Failed cache fences do not
        prevent SG dispatch.

        Writes containing implicit uploads use a site-wide fence and repeat it
        after other ordinary failures as well.

        Arguments, results and SG errors are those of `Shotgun.create`.

        Raises:
            RuntimeError: The cache has been closed.
        """
        method = cast('Any', self._client).create
        return cast(
            'dict[str, object]',
            self._write_with_fence(
                create_write_plan(self.context.site_namespace, entity_type, data),
                lambda: method(entity_type, data, return_fields),
            ),
        )

    def delete(self, entity_type: str, entity_id: int) -> bool:
        """Delete one entity behind conservative lifecycle fences.

        Attempt invalidation before SG dispatch and after success or a
        configured indeterminate transport failure. Failed cache fences do not
        prevent SG dispatch.

        Arguments, results and SG errors are those of `Shotgun.delete`.

        Raises:
            RuntimeError: The cache has been closed.
        """
        changes = lifecycle_changes(self.context.site_namespace, entity_type, entity_id)
        method = cast('Any', self._client).delete
        return cast(
            'bool',
            self._write_with_fence(WritePlan(changes), lambda: method(entity_type, entity_id)),
        )

    def find(  # noqa: PLR0913, PLR0917 - exact official SG signature
        self,
        entity_type: str,
        filters: object,
        fields: list[str] | None = None,
        order: list[dict[str, str]] | None = None,
        filter_operator: str | None = None,
        limit: int = 0,
        retired_only: bool = False,
        page: int = 0,
        include_archived_projects: bool = True,
        additional_filter_presets: object = None,
    ) -> list[dict[str, object]]:
        """Return matching entities through the exact query cache.

        Use exact query caching when dependency analysis and active policy
        admit the call. Unnormalizable or unsupported calls bypass lookup and
        population. Cache failures fall back to SG.

        Arguments, results and SG errors are those of `Shotgun.find`.

        Returns:
            An isolated cache hit or the direct SG result, with the upstream
            result contract.

        Raises:
            RuntimeError: The cache has been closed.
        """
        method = cast('Any', self._client).find
        values = (
            entity_type,
            filters,
            fields,
            order,
            filter_operator,
            limit,
            retired_only,
            page,
            include_archived_projects,
            additional_filter_presets,
        )
        return cast(
            'list[dict[str, object]]',
            self._entity_read(
                EntityRead('find', entity_type, fields, values),
                lambda: method(*values),
            ),
        )

    def find_one(  # noqa: PLR0913, PLR0917 - exact official SG signature
        self,
        entity_type: str,
        filters: object,
        fields: list[str] | None = None,
        order: list[dict[str, str]] | None = None,
        filter_operator: str | None = None,
        retired_only: bool = False,
        include_archived_projects: bool = True,
        additional_filter_presets: object = None,
    ) -> dict[str, object] | None:
        """Return the first matching entity through its distinct exact key.

        Use exact query caching when dependency analysis and active policy
        admit the call. Unnormalizable or unsupported calls bypass lookup and
        population. Cache failures fall back to SG.

        Arguments, results and SG errors are those of `Shotgun.find_one`.

        Returns:
            An isolated cache hit or the direct SG result, with the upstream
            result contract.

        Raises:
            RuntimeError: The cache has been closed.
        """
        method = cast('Any', self._client).find_one
        values = (
            entity_type,
            filters,
            fields,
            order,
            filter_operator,
            retired_only,
            include_archived_projects,
            additional_filter_presets,
        )
        return cast(
            'dict[str, object] | None',
            self._entity_read(
                EntityRead('find_one', entity_type, fields, values),
                lambda: method(*values),
            ),
        )

    def follow(self, user: dict[str, object], entity: dict[str, object]) -> dict[str, object]:
        """Follow one entity behind a conservative site fence.

        Attempt invalidation before SG dispatch and after success or a
        configured indeterminate transport failure. Failed cache fences do not
        prevent SG dispatch.

        Arguments, results and SG errors are those of `Shotgun.follow`.

        Raises:
            RuntimeError: The cache has been closed.
        """
        changes = site_changes(self.context.site_namespace)
        method = cast('Any', self._client).follow
        return cast(
            'dict[str, object]',
            self._write_with_fence(WritePlan(changes), lambda: method(user, entity)),
        )

    def note_thread_read(
        self,
        note_id: int,
        entity_fields: dict[str, object] | None = None,
    ) -> list[dict[str, object]]:
        """Always read a note thread directly.

        Bypass cache lookup and population because this composite response has
        no proven URL lifetime for caching.

        Arguments, results and SG errors are those of
        `Shotgun.note_thread_read`.

        Raises:
            RuntimeError: The cache has been closed.
        """
        context = self.context
        method = cast('Any', self._client).note_thread_read

        def fetch() -> object:
            return method(note_id, entity_fields)

        return cast(
            'list[dict[str, object]]',
            self._read_direct('note_thread_read', fetch, context, reason='composite_url_unproven'),
        )

    def preferences_read(self, prefs: list[str] | None = None) -> dict[str, object]:
        """Always read preferences directly.

        Bypass cache lookup and population because this composite response has
        no proven URL lifetime for caching.

        Arguments, results and SG errors are those of
        `Shotgun.preferences_read`.

        Raises:
            RuntimeError: The cache has been closed.
        """
        context = self.context
        method = cast('Any', self._client).preferences_read

        def fetch() -> object:
            return method(prefs)

        return cast(
            'dict[str, object]',
            self._read_direct('preferences_read', fetch, context, reason='composite_url_unproven'),
        )

    @classmethod
    def open(  # noqa: PLR0913 - exact prescribed factory signature
        cls,
        client: object,
        settings: CacheSettings,
        identity: Identity,
        context: ClientContext,
        *,
        metrics: MetricsSink | None = None,
        environ: Mapping[str, object] | None = None,
        indeterminate_write_exceptions: tuple[type[Exception], ...] = (),
    ) -> AbstractContextManager[Self]:
        """Construct and own configured cache layers around a caller-owned client.

        Args:
            client: Caller-owned thread-safe synchronous SG client.
            settings: Validated local and optional Valkey cache settings.
            identity: Exact effective SG identity namespace.
            context: Complete response-affecting client context.
            metrics: Optional synchronous metrics destination. `None` discards
                metrics and skips correlation-secret construction.
            environ: Environment used for metrics correlation and role
                passwords; `None` uses the process environment.
            indeterminate_write_exceptions: Explicit transport exceptions whose
                server outcome is unknown.

        Returns:
            A context manager that constructs resources on entry and closes
            only its constructed cache on exit. A body error takes precedence
            over ordinary cleanup failure. Close does not wait for active cache
            operations.

        Raises:
            ImportError: Shared settings require the optional Valkey package.
            TypeError: Indeterminate write exception configuration is
                malformed.
            ValueError: The context namespace or a required configured secret
                is invalid.
            Exception: Backend construction fails, or cleanup fails without a
                higher-priority body error.
        """

        @contextmanager
        def managed() -> Generator[Self]:
            environment = process_environ if environ is None else environ
            sink = (
                NullMetrics()
                if metrics is None
                else QueryCache.build_metrics(settings, environment, metrics)
            )
            local: LocalBackend | None = None
            shared: CacheBackend | None = None
            cache: QueryCache | None = None
            try:
                if settings.l1.enabled:
                    local = LocalBackend(settings, metrics=sink)
                if settings.l2.enabled:
                    valkey = import_module('sg_query_cache.valkey')
                    backend_type = cast('type[ValkeyBackend]', valkey.ValkeyBackend)
                    shared = backend_type.from_settings(settings, environment, sink)
                cache = QueryCache(settings, identity, l1=local, l2=shared, metrics=sink)
                yield cls(
                    client,
                    cache,
                    context,
                    indeterminate_write_exceptions=indeterminate_write_exceptions,
                )
            except BaseException:
                resources = (cache,) if cache is not None else (local, shared)
                for resource in resources:
                    if resource is not None:
                        with suppress(Exception):
                            resource.close()
                raise
            cache.close()

        return managed()

    @property
    def raw_client(self) -> object:
        """Return the caller-owned client for unsupported explicit operations."""
        return self._client

    def replace_context(self, context: ClientContext) -> None:
        """Replace response-affecting state after invalidating prior local reachability.

        A changed context also advances the site generation in shared Valkey,
        invalidating cached reads for every process using that site. An equal
        context leaves the cache unchanged. The raw SG client is not
        reconfigured.

        Args:
            context: Complete replacement context.

        Raises:
            RuntimeError: The cache has closed.
            ValueError: A shared cache namespace disagrees with the context.
        """
        self._cache.ensure_open()
        require_context_namespace(self._settings, context)
        with self._context_lock:
            previous = self._context
            if previous == context:
                return
            self._cache.invalidate(site_changes(previous.site_namespace))
            self._context = context

    def revive(self, entity_type: str, entity_id: int) -> bool:
        """Revive one entity behind conservative lifecycle fences.

        Attempt invalidation before SG dispatch and after success or a
        configured indeterminate transport failure. Failed cache fences do not
        prevent SG dispatch.

        Arguments, results and SG errors are those of `Shotgun.revive`.

        Raises:
            RuntimeError: The cache has been closed.
        """
        changes = lifecycle_changes(self.context.site_namespace, entity_type, entity_id)
        method = cast('Any', self._client).revive
        return cast(
            'bool',
            self._write_with_fence(WritePlan(changes), lambda: method(entity_type, entity_id)),
        )

    def schema_entity_read(
        self, project_entity: dict[str, object] | None = None
    ) -> dict[str, dict[str, object]]:
        """Return cached entity schema for an optional project.

        Use identity-scoped schema caching subject to active cache policy.
        Calls that cannot be normalized read directly; cache failures fall back
        to SG.

        Arguments, results and SG errors are those of
        `Shotgun.schema_entity_read`.

        Returns:
            An isolated cache hit or the direct SG result, with the upstream
            result contract.

        Raises:
            RuntimeError: The cache has been closed.
        """
        context = self.context
        method = cast('Any', self._client).schema_entity_read

        def fetch() -> object:
            return method(project_entity)

        call = schema_entity_read_call(project_entity)
        if call is None:
            return cast(
                'dict[str, dict[str, object]]',
                self._read_direct('schema_entity_read', fetch, context),
            )
        return cast(
            'dict[str, dict[str, object]]',
            self._read(
                call,
                self._schema_plan(context),
                fetch,
                context,
            ),
        )

    def schema_field_create(
        self,
        entity_type: str,
        data_type: str,
        display_name: str,
        properties: dict[str, object] | None = None,
    ) -> str:
        """Create one field behind schema-generation fences.

        Attempt invalidation before SG dispatch and after success or a
        configured indeterminate transport failure. Failed cache fences do not
        prevent SG dispatch.

        Arguments, results and SG errors are those of
        `Shotgun.schema_field_create`.

        Raises:
            RuntimeError: The cache has been closed.
        """
        changes = schema_changes(self.context.site_namespace)
        method = cast('Any', self._client).schema_field_create
        return cast(
            'str',
            self._write_with_fence(
                WritePlan(changes),
                lambda: method(entity_type, data_type, display_name, properties),
            ),
        )

    def schema_field_delete(self, entity_type: str, field_name: str) -> bool:
        """Delete one field behind schema-generation fences.

        Attempt invalidation before SG dispatch and after success or a
        configured indeterminate transport failure. Failed cache fences do not
        prevent SG dispatch.

        Arguments, results and SG errors are those of
        `Shotgun.schema_field_delete`.

        Raises:
            RuntimeError: The cache has been closed.
        """
        changes = schema_changes(self.context.site_namespace)
        method = cast('Any', self._client).schema_field_delete
        return cast(
            'bool',
            self._write_with_fence(WritePlan(changes), lambda: method(entity_type, field_name)),
        )

    def schema_field_read(
        self,
        entity_type: str,
        field_name: str | None = None,
        project_entity: dict[str, object] | None = None,
    ) -> dict[str, dict[str, object]]:
        """Return cached field schema for an entity type and optional project.

        Use identity-scoped schema caching subject to active cache policy.
        Calls that cannot be normalized read directly; cache failures fall back
        to SG.

        Arguments, results and SG errors are those of
        `Shotgun.schema_field_read`.

        Returns:
            An isolated cache hit or the direct SG result, with the upstream
            result contract.

        Raises:
            RuntimeError: The cache has been closed.
        """
        context = self.context
        method = cast('Any', self._client).schema_field_read

        def fetch() -> object:
            return method(entity_type, field_name, project_entity)

        call = schema_field_read_call(entity_type, field_name, project_entity)
        if call is None:
            return cast(
                'dict[str, dict[str, object]]',
                self._read_direct('schema_field_read', fetch, context),
            )
        return cast(
            'dict[str, dict[str, object]]',
            self._read(
                call,
                self._schema_plan(context),
                fetch,
                context,
            ),
        )

    def schema_field_update(
        self,
        entity_type: str,
        field_name: str,
        properties: dict[str, object],
        project_entity: dict[str, object] | None = None,
    ) -> bool:
        """Update one field behind schema-generation fences.

        Attempt invalidation before SG dispatch and after success or a
        configured indeterminate transport failure. Failed cache fences do not
        prevent SG dispatch.

        Arguments, results and SG errors are those of
        `Shotgun.schema_field_update`.

        Raises:
            RuntimeError: The cache has been closed.
        """
        changes = schema_changes(self.context.site_namespace)
        method = cast('Any', self._client).schema_field_update
        return cast(
            'bool',
            self._write_with_fence(
                WritePlan(changes),
                lambda: method(entity_type, field_name, properties, project_entity),
            ),
        )

    def schema_read(
        self, project_entity: dict[str, object] | None = None
    ) -> dict[str, dict[str, dict[str, object]]]:
        """Return cached full schema for an optional project.

        Use identity-scoped schema caching subject to active cache policy.
        Calls that cannot be normalized read directly; cache failures fall back
        to SG.

        Arguments, results and SG errors are those of `Shotgun.schema_read`.

        Returns:
            An isolated cache hit or the direct SG result, with the upstream
            result contract.

        Raises:
            RuntimeError: The cache has been closed.
        """
        return cast(
            'dict[str, dict[str, dict[str, object]]]',
            self._schema_read(project_entity, self.context),
        )

    def share_thumbnail(
        self,
        entities: list[dict[str, object]],
        thumbnail_path: str | None = None,
        source_entity: dict[str, object] | None = None,
        filmstrip_thumbnail: bool = False,
        **kwargs: object,
    ) -> int:
        """Share a thumbnail behind a conservative site fence.

        Attempt invalidation before SG dispatch and after success or a
        configured indeterminate transport failure. Failed cache fences do not
        prevent SG dispatch.

        Arguments, results and SG errors are those of
        `Shotgun.share_thumbnail`.

        Raises:
            RuntimeError: The cache has been closed.
        """
        changes = site_changes(self.context.site_namespace)
        method = cast('Any', self._client).share_thumbnail
        return cast(
            'int',
            self._write_with_fence(
                WritePlan(changes),
                lambda: method(
                    entities,
                    thumbnail_path,
                    source_entity,
                    filmstrip_thumbnail,
                    **kwargs,
                ),
            ),
        )

    def summarize(  # noqa: PLR0913, PLR0917 - exact official SG signature
        self,
        entity_type: str,
        filters: object,
        summary_fields: list[dict[str, str]],
        filter_operator: str | None = None,
        grouping: list[dict[str, str]] | None = None,
        include_archived_projects: bool = True,
    ) -> dict[str, object]:
        """Always summarize entities directly.

        Bypass cache lookup and population because this composite response has
        no proven URL lifetime for caching.

        Arguments, results and SG errors are those of `Shotgun.summarize`.

        Raises:
            RuntimeError: The cache has been closed.
        """
        context = self.context
        method = cast('Any', self._client).summarize

        def fetch() -> object:
            return method(
                entity_type,
                filters,
                summary_fields,
                filter_operator,
                grouping,
                include_archived_projects,
            )

        return cast(
            'dict[str, object]',
            self._read_direct('summarize', fetch, context, reason='composite_url_unproven'),
        )

    def text_search(
        self,
        text: str,
        entity_types: dict[str, object],
        project_ids: list[object] | None = None,
        limit: int | None = None,
    ) -> dict[str, object]:
        """Always search text directly.

        Bypass cache lookup and population because this composite response has
        no proven URL lifetime for caching.

        Arguments, results and SG errors are those of `Shotgun.text_search`.

        Raises:
            RuntimeError: The cache has been closed.
        """
        context = self.context
        method = cast('Any', self._client).text_search

        def fetch() -> object:
            return method(text, entity_types, project_ids, limit)

        return cast(
            'dict[str, object]',
            self._read_direct('text_search', fetch, context, reason='composite_url_unproven'),
        )

    def unfollow(self, user: dict[str, object], entity: dict[str, object]) -> dict[str, object]:
        """Unfollow one entity behind a conservative site fence.

        Attempt invalidation before SG dispatch and after success or a
        configured indeterminate transport failure. Failed cache fences do not
        prevent SG dispatch.

        Arguments, results and SG errors are those of `Shotgun.unfollow`.

        Raises:
            RuntimeError: The cache has been closed.
        """
        changes = site_changes(self.context.site_namespace)
        method = cast('Any', self._client).unfollow
        return cast(
            'dict[str, object]',
            self._write_with_fence(WritePlan(changes), lambda: method(user, entity)),
        )

    def update(
        self,
        entity_type: str,
        entity_id: int,
        data: dict[str, object],
        multi_entity_update_modes: dict[str, object] | None = None,
    ) -> dict[str, object]:
        """Update one known entity behind value and membership fences.

        Attempt invalidation before SG dispatch and after success or a
        configured indeterminate transport failure. Failed cache fences do not
        prevent SG dispatch.

        Writes containing implicit uploads use a site-wide fence and repeat it
        after other ordinary failures as well.

        Arguments, results and SG errors are those of `Shotgun.update`.

        Raises:
            RuntimeError: The cache has been closed.
        """
        method = cast('Any', self._client).update
        return cast(
            'dict[str, object]',
            self._write_with_fence(
                update_write_plan(self.context.site_namespace, entity_type, entity_id, data),
                lambda: method(entity_type, entity_id, data, multi_entity_update_modes),
            ),
        )

    def update_project_last_accessed(
        self,
        project: dict[str, object],
        user: dict[str, object] | None = None,
    ) -> None:
        """Update project access state behind a conservative site fence.

        Attempt invalidation before SG dispatch and after success or a
        configured indeterminate transport failure. Failed cache fences do not
        prevent SG dispatch.

        Arguments, results and SG errors are those of
        `Shotgun.update_project_last_accessed`.

        Raises:
            RuntimeError: The cache has been closed.
        """
        changes = site_changes(self.context.site_namespace)
        method = cast('Any', self._client).update_project_last_accessed
        self._write_with_fence(WritePlan(changes), lambda: method(project, user))

    def upload(  # noqa: PLR0913, PLR0917 - exact official SG signature
        self,
        entity_type: str,
        entity_id: int,
        path: str,
        field_name: str | None = None,
        display_name: str | None = None,
        tag_list: str | None = None,
        created_at: datetime | None = None,
    ) -> int:
        """Upload one file behind a site-wide Attachment creation fence.

        Attempt invalidation before SG dispatch and after success or a
        configured indeterminate transport failure. Failed cache fences do not
        prevent SG dispatch.

        Arguments, results and SG errors are those of `Shotgun.upload`.

        Raises:
            RuntimeError: The cache has been closed.
        """
        changes = site_changes(self.context.site_namespace)
        method = cast('Any', self._client).upload
        return cast(
            'int',
            self._write_with_fence(
                WritePlan(changes),
                lambda: method(
                    entity_type,
                    entity_id,
                    path,
                    field_name,
                    display_name,
                    tag_list,
                    created_at,
                ),
            ),
        )

    def upload_filmstrip_thumbnail(
        self,
        entity_type: str,
        entity_id: int,
        path: str,
        **kwargs: object,
    ) -> int:
        """Upload a filmstrip thumbnail behind a site-wide Attachment fence.

        Attempt invalidation before SG dispatch and after success or a
        configured indeterminate transport failure. Failed cache fences do not
        prevent SG dispatch.

        Arguments, results and SG errors are those of
        `Shotgun.upload_filmstrip_thumbnail`.

        Raises:
            RuntimeError: The cache has been closed.
        """
        changes = site_changes(self.context.site_namespace)
        method = cast('Any', self._client).upload_filmstrip_thumbnail
        return cast(
            'int',
            self._write_with_fence(
                WritePlan(changes),
                lambda: method(entity_type, entity_id, path, **kwargs),
            ),
        )

    def upload_thumbnail(
        self,
        entity_type: str,
        entity_id: int,
        path: str,
        **kwargs: object,
    ) -> int:
        """Upload a thumbnail behind a site-wide Attachment creation fence.

        Attempt invalidation before SG dispatch and after success or a
        configured indeterminate transport failure. Failed cache fences do not
        prevent SG dispatch.

        Arguments, results and SG errors are those of
        `Shotgun.upload_thumbnail`.

        Raises:
            RuntimeError: The cache has been closed.
        """
        changes = site_changes(self.context.site_namespace)
        method = cast('Any', self._client).upload_thumbnail
        return cast(
            'int',
            self._write_with_fence(
                WritePlan(changes),
                lambda: method(entity_type, entity_id, path, **kwargs),
            ),
        )

    def user_subscriptions_create(self, users: list[dict[str, object]]) -> bool:
        """Create user subscriptions behind a conservative site fence.

        Attempt invalidation before SG dispatch and after success or a
        configured indeterminate transport failure. Failed cache fences do not
        prevent SG dispatch.

        Arguments, results and SG errors are those of
        `Shotgun.user_subscriptions_create`.

        Raises:
            RuntimeError: The cache has been closed.
        """
        changes = site_changes(self.context.site_namespace)
        method = cast('Any', self._client).user_subscriptions_create
        return cast('bool', self._write_with_fence(WritePlan(changes), lambda: method(users)))

    def work_schedule_read(
        self,
        start_date: str,
        end_date: str,
        project: dict[str, object] | None = None,
        user: dict[str, object] | None = None,
    ) -> dict[str, object]:
        """Read work schedules directly unless TTL-only caching is enabled.

        Cache only when this method is enabled in `enabled_ttl_only_methods`,
        the call can be normalized, and identity and policy permit caching.
        Cache failures fall back to SG.

        Arguments, results and SG errors are those of
        `Shotgun.work_schedule_read`.

        Returns:
            An isolated cache hit or the direct SG result, with the upstream
            result contract.

        Raises:
            RuntimeError: The cache has been closed.
        """
        context = self.context
        method = cast('Any', self._client).work_schedule_read

        def fetch() -> object:
            return method(start_date, end_date, project, user)

        call = work_schedule_read_call(start_date, end_date, project, user)
        if call is None:
            return cast(
                'dict[str, object]',
                self._read_direct('work_schedule_read', fetch, context),
            )
        return cast(
            'dict[str, object]',
            self._read(
                call,
                self._ttl_only_plan('work_schedule_read', context),
                fetch,
                context,
            ),
        )

    def work_schedule_update(  # noqa: PLR0913, PLR0917 - exact official SG signature
        self,
        date: str,
        working: bool,
        description: str | None = None,
        project: dict[str, object] | None = None,
        user: dict[str, object] | None = None,
        recalculate_field: str | None = None,
    ) -> dict[str, object]:
        """Update work schedule state behind a conservative site fence.

        Attempt invalidation before SG dispatch and after success or a
        configured indeterminate transport failure. Failed cache fences do not
        prevent SG dispatch.

        Arguments, results and SG errors are those of
        `Shotgun.work_schedule_update`.

        Raises:
            RuntimeError: The cache has been closed.
        """
        changes = site_changes(self.context.site_namespace)
        method = cast('Any', self._client).work_schedule_update
        return cast(
            'dict[str, object]',
            self._write_with_fence(
                WritePlan(changes),
                lambda: method(
                    date,
                    working,
                    description,
                    project,
                    user,
                    recalculate_field,
                ),
            ),
        )
