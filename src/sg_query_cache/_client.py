"""Wrap synchronous SG reads with exact caching and conservative write fences."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from threading import Lock
from typing import TYPE_CHECKING, Any, TypeAlias, cast

from sg_query_cache._cache import client_binding
from sg_query_cache._client_common import (
    IMPLICIT_UPLOAD_FIELDS,
    ClientContext,
    EntityPlanContext,
    batch_changes,
    batch_has_create,
    batch_post_changes,
    bound_call,
    client_options,
    create_changes,
    create_post_changes,
    entity_calls,
    entity_read_plan,
    exception_types,
    freeze,
    key_freeze,
    lifecycle_changes,
    opaque,
    parse_schema_view,
    schema_changes,
    schema_plan,
    site_changes,
    ttl_only_plan,
    unconditional_entity_plan,
    update_changes,
)
from sg_query_cache._dependencies import DependencyAnalyzer, SchemaView
from sg_query_cache._model import (
    DependencyPlan,
    DependencyTag,
    ExactCall,
    GenerationChange,
    Identity,
)
from sg_query_cache._urls import AgeDecision, cap_response_age

if TYPE_CHECKING:
    from contextlib import AbstractContextManager

    from sg_query_cache._cache import QueryCache
    from sg_query_cache._cache_common import PolicyMode

_Call: TypeAlias = Callable[[], object]
_PostChanges: TypeAlias = Callable[[object], tuple[GenerationChange, ...]]


class CachedShotgun:
    """Cache explicitly supported synchronous SG calls for one exact identity.

    Wrapped SG methods preserve the raw client's argument and result contracts.
    Cache failures fall back to SG reads; SG failures propagate. Writes attempt
    fences before dispatch and after success or a configured indeterminate
    transport failure. Composite create, update, and batch writes also fence
    after other ordinary failures.

    The wrapper owns neither the raw client nor the injected cache. Callers
    close both resources and keep the immutable response context consistent
    with the raw client's effective configuration. Policy scopes are
    thread-local. Use `raw_client` for unsupported operations; those calls
    bypass these fences.
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
        if settings.l2.enabled and context.site_namespace != settings.site_namespace:
            raise ValueError('shared cache context must use the configured site namespace')
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

    def _read_direct(self, method: str, fetch: _Call, context: ClientContext) -> object:
        """Bypass the cache for a call whose arguments cannot be normalized."""
        call = ExactCall(method, ())
        return self._cache.read(
            call,
            DependencyPlan('bypass', (), 'unsupported_argument'),
            fetch,
            client_options(context, call),
        )

    def _schema_plan(self, context: ClientContext) -> DependencyPlan:
        """Build the fixed schema plan for the current exact identity."""
        return schema_plan(self._identity(), context.site_namespace, self._settings)

    def _schema_view(self, context: ClientContext) -> SchemaView:
        """Load the identity-scoped compact schema view through the parsed memo."""
        method = cast('Any', self._client).schema_read
        call = ExactCall(
            'schema_view',
            (('project_entity', None), ('site_url', context.site_url)),
        )
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

    def bypass(self) -> AbstractContextManager[None]:
        """Bypass reads and population within the current thread-local scope."""
        return self._cache.bypass()

    @property
    def context(self) -> ClientContext:
        """Return the immutable context used by subsequent operations."""
        with self._context_lock:
            return self._context

    def _read(
        self,
        call: ExactCall,
        plan: DependencyPlan,
        fetch: _Call,
        context: ClientContext,
        returned_fields: Mapping[str, str] | None = None,
    ) -> object:
        """Execute one exact read and safely apply response-derived age caps."""
        options = client_options(context, call)
        exact_call = ExactCall(
            call.method,
            (*call.arguments, ('site_url', context.site_url)),
        )
        population_decision = None
        if returned_fields and {'image', 'url'} & set(returned_fields.values()):

            def decide(value: object) -> AgeDecision:
                return cap_response_age(
                    value,
                    returned_fields,
                    datetime.now(tz=UTC),
                    self._settings,
                )

            population_decision = decide
        return self._cache.read(
            exact_call,
            plan,
            fetch,
            options,
            _population_decision=population_decision,
        )

    def _schema_read(
        self,
        project_entity: dict[str, object] | None,
        context: ClientContext,
    ) -> object:
        """Read full schema through the nonrecursive cache boundary."""
        method = cast('Any', self._client).schema_read

        def fetch() -> object:
            return method(project_entity)

        call = bound_call(
            lambda: ExactCall('schema_read', (('project_entity', key_freeze(project_entity)),))
        )
        if call is None:
            return self._read_direct('schema_read', fetch, context)
        return self._read(
            call,
            self._schema_plan(context),
            fetch,
            context,
        )

    def _write_with_fence(
        self,
        changes: tuple[GenerationChange, ...],
        call: _Call,
        *,
        post_changes: _PostChanges | None = None,
        indeterminate_changes: tuple[GenerationChange, ...] | None = None,
        repeat_on_exception: bool = False,
    ) -> object:
        """Fence one write before dispatch and after nondefinite outcomes."""
        self._cache.invalidate(changes)
        try:
            value = call()
        except Exception as error:
            if repeat_on_exception or isinstance(
                error,
                self._indeterminate_write_exceptions,
            ):
                self._cache.invalidate(indeterminate_changes or changes)
            raise
        self._cache.invalidate(post_changes(value) if post_changes is not None else changes)
        return value

    def batch(self, requests: list[dict[str, object]]) -> list[dict[str, object] | bool]:
        """Execute one ordered batch behind its complete conservative fence."""
        site_namespace = self.context.site_namespace
        composite = any(
            isinstance(request.get('data'), Mapping)
            and not IMPLICIT_UPLOAD_FIELDS.isdisjoint(
                cast('Mapping[str, object]', request['data'])
            )
            for request in requests
        )
        changes = (
            site_changes(site_namespace) if composite else batch_changes(site_namespace, requests)
        )
        indeterminate_changes = (
            site_changes(site_namespace) if composite or batch_has_create(requests) else changes
        )
        method = cast('Any', self._client).batch
        return cast(
            'list[dict[str, object] | bool]',
            self._write_with_fence(
                changes,
                lambda: method(requests),
                post_changes=(
                    None
                    if composite
                    else lambda value: batch_post_changes(
                        site_namespace,
                        requests,
                        changes,
                        value,
                    )
                ),
                indeterminate_changes=indeterminate_changes,
                repeat_on_exception=composite,
            ),
        )

    def delete(self, entity_type: str, entity_id: int) -> bool:
        """Delete one entity behind conservative lifecycle fences."""
        changes = lifecycle_changes(self.context.site_namespace, entity_type, entity_id)
        method = cast('Any', self._client).delete
        return cast(
            'bool', self._write_with_fence(changes, lambda: method(entity_type, entity_id))
        )

    def follow(self, user: dict[str, object], entity: dict[str, object]) -> dict[str, object]:
        """Follow one entity behind a conservative site fence."""
        changes = site_changes(self.context.site_namespace)
        method = cast('Any', self._client).follow
        return cast(
            'dict[str, object]', self._write_with_fence(changes, lambda: method(user, entity))
        )

    @property
    def raw_client(self) -> object:
        """Return the caller-owned client for unsupported explicit operations."""
        return self._client

    def activity_stream_read(  # noqa: PLR0913, PLR0917 - exact official SG signature
        self,
        entity_type: str,
        entity_id: int,
        entity_fields: dict[str, object] | None = None,
        min_id: int | None = None,
        max_id: int | None = None,
        limit: int | None = None,
    ) -> dict[str, object]:
        """Always read an activity stream directly."""
        context = self.context
        method = cast('Any', self._client).activity_stream_read

        def fetch() -> object:
            return method(entity_type, entity_id, entity_fields, min_id, max_id, limit)

        call = bound_call(
            lambda: ExactCall(
                'activity_stream_read',
                (
                    ('entity_type', entity_type),
                    ('entity_id', entity_id),
                    ('entity_fields', opaque(entity_fields)),
                    ('min_id', min_id),
                    ('max_id', max_id),
                    ('limit', limit),
                ),
            )
        )
        if call is None:
            return cast(
                'dict[str, object]',
                self._read_direct('activity_stream_read', fetch, context),
            )
        return cast(
            'dict[str, object]',
            self._read(
                call,
                self._ttl_only_plan('activity_stream_read', context),
                fetch,
                context,
            ),
        )

    def cache_policy(
        self,
        mode: PolicyMode | None = None,
        maximum_age: float | None = None,
    ) -> AbstractContextManager[None]:
        """Apply one validated thread-local cache policy scope.

        Entering the returned context validates its arguments. Nonfinite,
        boolean, or nonnumeric ages raise `TypeError`; unsupported modes or
        nonpositive ages raise `ValueError`.

        Args:
            mode: Optional conservative downgrade to event-aware, TTL-only, or
                bypass.
            maximum_age: Optional positive age in seconds, combined by minimum
                with all active scopes and per-call ages. The selected explicit
                age replaces configured ages before safety caps are applied.

        Returns:
            A nestable context manager delegated to the query cache.
        """
        return self._cache.policy(mode, maximum_age)

    def create(
        self,
        entity_type: str,
        data: dict[str, object],
        return_fields: list[str] | None = None,
    ) -> dict[str, object]:
        """Create one entity behind conservative membership fences."""
        site_namespace = self.context.site_namespace
        composite = not IMPLICIT_UPLOAD_FIELDS.isdisjoint(data)
        changes = (
            site_changes(site_namespace)
            if composite
            else create_changes(site_namespace, entity_type, data)
        )
        method = cast('Any', self._client).create
        return cast(
            'dict[str, object]',
            self._write_with_fence(
                changes,
                lambda: method(entity_type, data, return_fields),
                post_changes=(
                    None
                    if composite
                    else lambda value: create_post_changes(
                        site_namespace,
                        entity_type,
                        changes,
                        value,
                    )
                ),
                indeterminate_changes=site_changes(site_namespace),
                repeat_on_exception=composite,
            ),
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
        """Return matching entities through the exact query cache."""
        context = self.context
        method = cast('Any', self._client).find

        def fetch() -> object:
            return method(
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

        analysis_call, call = entity_calls(
            'find',
            (
                ('entity_type', entity_type),
                ('filters', filters),
                ('fields', fields),
                ('order', order),
                ('filter_operator', filter_operator),
                ('limit', limit),
                ('retired_only', retired_only),
                ('page', page),
                ('include_archived_projects', include_archived_projects),
                ('additional_filter_presets', additional_filter_presets),
            ),
        )
        if analysis_call is None or call is None:
            return cast('list[dict[str, object]]', self._read_direct('find', fetch, context))
        plan = unconditional_entity_plan(
            self._identity(), self._settings, entity_type, lambda: self._cache.bypass_active
        )
        returned_fields = None
        if plan is None:
            schema = self._schema_view_or_empty(context)
            plan, returned_fields = entity_read_plan(
                analysis_call,
                schema,
                entity_type,
                fields,
                EntityPlanContext(
                    self._analyzer, self._identity(), self._settings, context.site_namespace
                ),
            )
        value = self._read(
            call,
            plan,
            fetch,
            context,
            returned_fields,
        )
        return cast('list[dict[str, object]]', value)

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
        """Return the first matching entity through its distinct exact key."""
        context = self.context
        method = cast('Any', self._client).find_one

        def fetch() -> object:
            return method(
                entity_type,
                filters,
                fields,
                order,
                filter_operator,
                retired_only,
                include_archived_projects,
                additional_filter_presets,
            )

        analysis_call, call = entity_calls(
            'find_one',
            (
                ('entity_type', entity_type),
                ('filters', filters),
                ('fields', fields),
                ('order', order),
                ('filter_operator', filter_operator),
                ('retired_only', retired_only),
                ('include_archived_projects', include_archived_projects),
                ('additional_filter_presets', additional_filter_presets),
            ),
        )
        if analysis_call is None or call is None:
            return cast(
                'dict[str, object] | None',
                self._read_direct('find_one', fetch, context),
            )
        plan = unconditional_entity_plan(
            self._identity(), self._settings, entity_type, lambda: self._cache.bypass_active
        )
        returned_fields = None
        if plan is None:
            schema = self._schema_view_or_empty(context)
            plan, returned_fields = entity_read_plan(
                analysis_call,
                schema,
                entity_type,
                fields,
                EntityPlanContext(
                    self._analyzer, self._identity(), self._settings, context.site_namespace
                ),
            )
        value = self._read(
            call,
            plan,
            fetch,
            context,
            returned_fields,
        )
        return cast('dict[str, object] | None', value)

    def note_thread_read(
        self,
        note_id: int,
        entity_fields: dict[str, object] | None = None,
    ) -> list[dict[str, object]]:
        """Always read a note thread directly."""
        context = self.context
        method = cast('Any', self._client).note_thread_read

        def fetch() -> object:
            return method(note_id, entity_fields)

        call = bound_call(
            lambda: ExactCall(
                'note_thread_read',
                (('note_id', note_id), ('entity_fields', opaque(entity_fields))),
            )
        )
        if call is None:
            return cast(
                'list[dict[str, object]]',
                self._read_direct('note_thread_read', fetch, context),
            )
        return cast(
            'list[dict[str, object]]',
            self._read(
                call,
                self._ttl_only_plan('note_thread_read', context),
                fetch,
                context,
            ),
        )

    def preferences_read(self, prefs: list[str] | None = None) -> dict[str, object]:
        """Always read preferences directly."""
        context = self.context
        method = cast('Any', self._client).preferences_read

        def fetch() -> object:
            return method(prefs)

        call = bound_call(lambda: ExactCall('preferences_read', (('prefs', freeze(prefs)),)))
        if call is None:
            return cast(
                'dict[str, object]',
                self._read_direct('preferences_read', fetch, context),
            )
        return cast(
            'dict[str, object]',
            self._read(
                call,
                self._ttl_only_plan('preferences_read', context),
                fetch,
                context,
            ),
        )

    def replace_context(self, context: ClientContext) -> None:
        """Replace response-affecting state after invalidating prior local reachability.

        A changed context also advances the site generation in shared Valkey,
        invalidating cached reads for every process using that site. An equal
        context leaves the cache unchanged. The raw SG client is not
        reconfigured.

        Args:
            context: Complete replacement context.

        Raises:
            ValueError: A shared cache namespace disagrees with the context.
        """
        settings = self._settings
        if settings.l2.enabled and context.site_namespace != settings.site_namespace:
            raise ValueError('shared cache context must use the configured site namespace')
        with self._context_lock:
            previous = self._context
            if previous == context:
                return
            self._cache.invalidate(
                (GenerationChange(DependencyTag.site(previous.site_namespace)),)
            )
            self._context = context

    def revive(self, entity_type: str, entity_id: int) -> bool:
        """Revive one entity behind conservative lifecycle fences."""
        changes = lifecycle_changes(self.context.site_namespace, entity_type, entity_id)
        method = cast('Any', self._client).revive
        return cast(
            'bool', self._write_with_fence(changes, lambda: method(entity_type, entity_id))
        )

    def schema_entity_read(
        self, project_entity: dict[str, object] | None = None
    ) -> dict[str, dict[str, object]]:
        """Return cached entity schema for an optional project."""
        context = self.context
        method = cast('Any', self._client).schema_entity_read

        def fetch() -> object:
            return method(project_entity)

        call = bound_call(
            lambda: ExactCall(
                'schema_entity_read',
                (('project_entity', key_freeze(project_entity)),),
            )
        )
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

    def schema_field_delete(self, entity_type: str, field_name: str) -> bool:
        """Delete one field behind schema-generation fences."""
        changes = schema_changes(self.context.site_namespace)
        method = cast('Any', self._client).schema_field_delete
        return cast(
            'bool',
            self._write_with_fence(changes, lambda: method(entity_type, field_name)),
        )

    def schema_field_read(
        self,
        entity_type: str,
        field_name: str | None = None,
        project_entity: dict[str, object] | None = None,
    ) -> dict[str, dict[str, object]]:
        """Return cached field schema for an entity type and optional project."""
        context = self.context
        method = cast('Any', self._client).schema_field_read

        def fetch() -> object:
            return method(entity_type, field_name, project_entity)

        call = bound_call(
            lambda: ExactCall(
                'schema_field_read',
                (
                    ('entity_type', entity_type),
                    ('field_name', field_name),
                    ('project_entity', key_freeze(project_entity)),
                ),
            )
        )
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

    def schema_field_create(
        self,
        entity_type: str,
        data_type: str,
        display_name: str,
        properties: dict[str, object] | None = None,
    ) -> str:
        """Create one field behind schema-generation fences."""
        changes = schema_changes(self.context.site_namespace)
        method = cast('Any', self._client).schema_field_create
        return cast(
            'str',
            self._write_with_fence(
                changes,
                lambda: method(entity_type, data_type, display_name, properties),
            ),
        )

    def schema_field_update(
        self,
        entity_type: str,
        field_name: str,
        properties: dict[str, object],
        project_entity: dict[str, object] | None = None,
    ) -> bool:
        """Update one field behind schema-generation fences."""
        changes = schema_changes(self.context.site_namespace)
        method = cast('Any', self._client).schema_field_update
        return cast(
            'bool',
            self._write_with_fence(
                changes,
                lambda: method(entity_type, field_name, properties, project_entity),
            ),
        )

    def schema_read(
        self, project_entity: dict[str, object] | None = None
    ) -> dict[str, dict[str, dict[str, object]]]:
        """Return cached full schema for an optional project."""
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
        """Share a thumbnail behind a conservative site fence."""
        changes = site_changes(self.context.site_namespace)
        method = cast('Any', self._client).share_thumbnail
        return cast(
            'int',
            self._write_with_fence(
                changes,
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
        """Always summarize entities directly."""
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

        call = bound_call(
            lambda: ExactCall(
                'summarize',
                (
                    ('entity_type', entity_type),
                    ('filters', freeze(filters)),
                    ('summary_fields', opaque(summary_fields)),
                    ('filter_operator', filter_operator),
                    ('grouping', opaque(grouping)),
                    ('include_archived_projects', include_archived_projects),
                ),
            )
        )
        if call is None:
            return cast('dict[str, object]', self._read_direct('summarize', fetch, context))
        return cast(
            'dict[str, object]',
            self._read(
                call,
                self._ttl_only_plan('summarize', context),
                fetch,
                context,
            ),
        )

    def text_search(
        self,
        text: str,
        entity_types: dict[str, object],
        project_ids: list[object] | None = None,
        limit: int | None = None,
    ) -> dict[str, object]:
        """Always search text directly."""
        context = self.context
        method = cast('Any', self._client).text_search

        def fetch() -> object:
            return method(text, entity_types, project_ids, limit)

        call = bound_call(
            lambda: ExactCall(
                'text_search',
                (
                    ('text', text),
                    ('entity_types', opaque(entity_types)),
                    ('project_ids', freeze(project_ids)),
                    ('limit', limit),
                ),
            )
        )
        if call is None:
            return cast('dict[str, object]', self._read_direct('text_search', fetch, context))
        return cast(
            'dict[str, object]',
            self._read(
                call,
                self._ttl_only_plan('text_search', context),
                fetch,
                context,
            ),
        )

    def unfollow(self, user: dict[str, object], entity: dict[str, object]) -> dict[str, object]:
        """Unfollow one entity behind a conservative site fence."""
        changes = site_changes(self.context.site_namespace)
        method = cast('Any', self._client).unfollow
        return cast(
            'dict[str, object]', self._write_with_fence(changes, lambda: method(user, entity))
        )

    def update(
        self,
        entity_type: str,
        entity_id: int,
        data: dict[str, object],
        multi_entity_update_modes: dict[str, object] | None = None,
    ) -> dict[str, object]:
        """Update one known entity behind value and membership fences."""
        site_namespace = self.context.site_namespace
        composite = not IMPLICIT_UPLOAD_FIELDS.isdisjoint(data)
        changes = (
            site_changes(site_namespace)
            if composite
            else update_changes(site_namespace, entity_type, entity_id, data)
        )
        method = cast('Any', self._client).update
        return cast(
            'dict[str, object]',
            self._write_with_fence(
                changes,
                lambda: method(entity_type, entity_id, data, multi_entity_update_modes),
                repeat_on_exception=composite,
            ),
        )

    def update_project_last_accessed(
        self,
        project: dict[str, object],
        user: dict[str, object] | None = None,
    ) -> None:
        """Update project access state behind a conservative site fence."""
        changes = site_changes(self.context.site_namespace)
        method = cast('Any', self._client).update_project_last_accessed
        self._write_with_fence(changes, lambda: method(project, user))

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
        """Upload one file behind a site-wide Attachment creation fence."""
        changes = site_changes(self.context.site_namespace)
        method = cast('Any', self._client).upload
        return cast(
            'int',
            self._write_with_fence(
                changes,
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
        """Upload a filmstrip thumbnail behind a site-wide Attachment fence."""
        changes = site_changes(self.context.site_namespace)
        method = cast('Any', self._client).upload_filmstrip_thumbnail
        return cast(
            'int',
            self._write_with_fence(
                changes,
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
        """Upload a thumbnail behind a site-wide Attachment creation fence."""
        changes = site_changes(self.context.site_namespace)
        method = cast('Any', self._client).upload_thumbnail
        return cast(
            'int',
            self._write_with_fence(
                changes,
                lambda: method(entity_type, entity_id, path, **kwargs),
            ),
        )

    def user_subscriptions_create(self, users: list[dict[str, object]]) -> bool:
        """Create user subscriptions behind a conservative site fence."""
        changes = site_changes(self.context.site_namespace)
        method = cast('Any', self._client).user_subscriptions_create
        return cast('bool', self._write_with_fence(changes, lambda: method(users)))

    def work_schedule_read(
        self,
        start_date: str,
        end_date: str,
        project: dict[str, object] | None = None,
        user: dict[str, object] | None = None,
    ) -> dict[str, object]:
        """Read work schedules directly unless TTL-only caching is enabled."""
        context = self.context
        method = cast('Any', self._client).work_schedule_read

        def fetch() -> object:
            return method(start_date, end_date, project, user)

        call = bound_call(
            lambda: ExactCall(
                'work_schedule_read',
                (
                    ('start_date', start_date),
                    ('end_date', end_date),
                    ('project', key_freeze(project)),
                    ('user', key_freeze(user)),
                ),
            )
        )
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
        """Update work schedule state behind a conservative site fence."""
        changes = site_changes(self.context.site_namespace)
        method = cast('Any', self._client).work_schedule_update
        return cast(
            'dict[str, object]',
            self._write_with_fence(
                changes,
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
