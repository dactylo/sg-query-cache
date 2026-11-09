"""Wrap native asynchronous SG calls with exact caching and conservative fences."""

from __future__ import annotations

from asyncio import CancelledError, current_task, sleep
from collections.abc import AsyncGenerator, Awaitable, Callable, Mapping
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass
from importlib import import_module
from os import environ as process_environ
from time import monotonic
from typing import TYPE_CHECKING, Any, Self, TypeAlias, cast

from sg_query_cache._async_cache import AsyncQueryCache, client_binding
from sg_query_cache._async_local import AsyncLocalBackend
from sg_query_cache._backend import BackendUnavailable
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
from sg_query_cache._metrics import MetricsSink, NullMetrics
from sg_query_cache._model import (
    DependencyPlan,
    ExactCall,
    Identity,
)

if TYPE_CHECKING:
    from contextlib import AbstractContextManager
    from datetime import datetime

    from sg_query_cache._async_backend import AsyncCacheBackend
    from sg_query_cache._async_support import OperationToken
    from sg_query_cache._cache_common import PolicyMode
    from sg_query_cache._config import CacheSettings
    from sg_query_cache.valkey_async import AsyncValkeyBackend

_AsyncCall: TypeAlias = Callable[[], Awaitable[object]]


@dataclass(frozen=True, slots=True)
class _ReadRequest:
    """Keep one planned wrapper read compact across its async cache boundary."""

    call: ExactCall
    plan: DependencyPlan
    context: ClientContext
    returned_fields: Mapping[str, str] | None = None


class AsyncCachedShotgun:
    """Cache explicitly supported native asynchronous SG calls.

    The wrapper and raw client are caller-owned when constructed directly.
    `open` instead owns the cache layers it constructs, while the raw client
    always remains caller-owned.

    Wrapped SG methods preserve the native client's argument and result
    contracts.
    Cache failures fall back to SG reads; SG failures propagate. Writes attempt
    fences before dispatch and after success or an indeterminate outcome.
    Cancellation during a dispatched write is indeterminate.

    Operations require an open cache on its owning event loop and raise
    `RuntimeError` when admission is rejected. Lexical policy scopes are
    task-local;
    child tasks inherit the snapshot present at creation. Keep the response
    context consistent with the raw client's effective configuration.
    """

    __slots__ = (
        '_analyzer',
        '_cache',
        '_client',
        '_context',
        '_identity_value',
        '_indeterminate_write_exceptions',
        '_settings',
    )

    def __init__(
        self,
        client: object,
        cache: AsyncQueryCache,
        context: ClientContext,
        *,
        indeterminate_write_exceptions: tuple[type[Exception], ...] = (),
    ) -> None:
        """Bind a caller-owned client and cache without acquiring resources.

        Args:
            client: Caller-owned native asynchronous SG client.
            cache: Caller-owned async cache with the effective identity.
            context: Complete response-affecting client context.
            indeterminate_write_exceptions: Explicit transport exceptions whose
                server outcome is unknown.

        Raises:
            TypeError: Indeterminate write exceptions are not a tuple of
                Exception subclasses.
            ValueError: Shared cache and context namespaces disagree.
        """
        settings, identity = client_binding(cache)
        require_context_namespace(settings, context)
        self._analyzer = DependencyAnalyzer(
            enable_additional_filter_presets=settings.enable_additional_filter_presets
        )
        self._cache = cache
        self._client = client
        self._context = context
        self._identity_value = identity
        self._indeterminate_write_exceptions = exception_types(indeterminate_write_exceptions)
        self._settings = settings

    async def __aenter__(self) -> Self:
        """Enter a non-owning wrapper context after validating cache admission.

        Returns:
            This wrapper without acquiring ownership of the cache or client.

        Raises:
            RuntimeError: The cache is closing, closed, or bound to another
                event loop.
        """
        with self._cache.admit():
            return self

    async def __aexit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc: BaseException | None,
        _traceback: object,
    ) -> None:
        """Leave a non-owning wrapper context without closing caller resources."""

    def _identity(self) -> Identity:
        """Return the caller-supplied identity already bound to the cache."""
        return self._identity_value

    async def _read(
        self,
        request: _ReadRequest,
        fetch: _AsyncCall,
        token: OperationToken,
    ) -> object:
        """Execute one exact read and safely apply response-derived age caps."""
        return await self._cache.read_admitted(
            site_call(request.call, request.context),
            request.plan,
            fetch,
            client_options(request.context, request.call),
            token=token,
            _population_decision=population_callback(request.returned_fields, self._settings),
        )

    async def _read_direct(
        self,
        method: str,
        fetch: _AsyncCall,
        context: ClientContext,
        token: OperationToken,
        *,
        reason: str = 'unsupported_argument',
    ) -> object:
        """Bypass the cache, without building a call key, for one stated reason."""
        call = ExactCall(method, ())
        return await self._cache.read_admitted(
            call,
            DependencyPlan('bypass', (), reason),
            fetch,
            client_options(context, call),
            token=token,
        )

    def _schema_plan(self, context: ClientContext) -> DependencyPlan:
        """Build the fixed schema plan for the current exact identity."""
        return schema_plan(self._identity(), context.site_namespace, self._settings)

    async def _schema_read(
        self,
        project_entity: dict[str, object] | None,
        context: ClientContext,
        token: OperationToken,
    ) -> object:
        """Read full schema through the nonrecursive cache boundary."""
        method = cast('Any', self._client).schema_read

        async def fetch() -> object:
            return await method(project_entity)

        call = schema_read_call(project_entity)
        if call is None:
            return await self._read_direct('schema_read', fetch, context, token)
        return await self._read(
            _ReadRequest(call, self._schema_plan(context), context), fetch, token
        )

    async def _schema_view(self, context: ClientContext, token: OperationToken) -> SchemaView:
        """Load the identity-scoped compact schema view through the parsed memo."""
        method = cast('Any', self._client).schema_read
        call = schema_view_call(context)

        async def fetch() -> object:
            return SchemaView.from_api_response(await method(None)).to_compact()

        value = await self._cache.read_parsed_admitted(
            call,
            self._schema_plan(context),
            fetch,
            client_options(context, call),
            parse_schema_view,
            token=token,
        )
        if not isinstance(value, SchemaView):
            return SchemaView.from_api_response(None)
        return value

    async def _schema_view_or_empty(
        self, context: ClientContext, token: OperationToken
    ) -> SchemaView:
        """Return an empty conservative view when schema acquisition fails."""
        try:
            return await self._schema_view(context, token)
        except Exception:  # noqa: BLE001 - unavailable schema must not block the read
            return SchemaView.from_api_response(None)

    def _ttl_only_plan(self, method: str, context: ClientContext) -> DependencyPlan:
        """Build a disabled-by-default universal plan for one explicit method."""
        return ttl_only_plan(method, self._identity(), context.site_namespace, self._settings)

    async def _write_with_fence(
        self,
        plan: WritePlan,
        call: _AsyncCall,
        token: OperationToken,
    ) -> object:
        """Fence one write before dispatch and after nondefinite outcomes."""
        await self._cache.invalidate_admitted(plan.changes, token=token)
        await sleep(0)
        try:
            value = await call()
        except CancelledError:
            with suppress(CancelledError):
                await self._cache.invalidate_admitted(
                    plan.indeterminate_changes or plan.changes,
                    token=token,
                )
            raise
        except Exception as error:
            if plan.repeat_on_exception or isinstance(
                error,
                self._indeterminate_write_exceptions,
            ):
                await self._cache.invalidate_admitted(
                    plan.indeterminate_changes or plan.changes,
                    token=token,
                )
            raise
        await self._cache.invalidate_admitted(
            plan.post_changes(value) if plan.post_changes is not None else plan.changes,
            token=token,
        )
        return value

    def bypass(self) -> AbstractContextManager[None]:
        """Bypass reads and population within the current task-local scope."""
        return self._cache.bypass()

    @property
    def context(self) -> ClientContext:
        """Return the immutable context used by subsequent operations."""
        return self._context

    async def _entity_read(
        self, read: EntityRead, fetch: _AsyncCall, token: OperationToken
    ) -> object:
        """Plan and read one `find` or `find_one`, reading directly when unnormalizable."""
        context = self.context
        analysis_call, call = entity_calls(read)
        if analysis_call is None or call is None:
            return await self._read_direct(read.method, fetch, context, token)
        plan = unconditional_entity_plan(
            self._identity(), self._settings, read.entity_type, lambda: self._cache.bypass_active
        )
        returned_fields = None
        if plan is None:
            schema = await self._schema_view_or_empty(context, token)
            plan, returned_fields = entity_read_plan(
                analysis_call,
                schema,
                read.entity_type,
                read.fields,
                EntityPlanContext(
                    self._analyzer, self._identity(), self._settings, context.site_namespace
                ),
            )
        return await self._read(_ReadRequest(call, plan, context, returned_fields), fetch, token)

    async def activity_stream_read(  # noqa: PLR0913, PLR0917 - exact official SG signature
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

        Admit the complete operation, including planning and cleanup, on the
        cache's owning event loop.

        Arguments, results and SG errors are those of
        `AsyncShotgun.activity_stream_read`.

        Raises:
            RuntimeError: The cache is closing, closed, or bound to another
                event loop.
            CancelledError: The caller cancels the operation.
        """
        with self._cache.admit() as token:
            context = self.context
            method = cast('Any', self._client).activity_stream_read

            async def fetch() -> object:
                return await method(entity_type, entity_id, entity_fields, min_id, max_id, limit)

            return cast(
                'dict[str, object]',
                await self._read_direct(
                    'activity_stream_read', fetch, context, token, reason='composite_url_unproven'
                ),
            )

    async def batch(self, requests: list[dict[str, object]]) -> list[dict[str, object] | bool]:
        """Execute one ordered batch behind its complete conservative fence.

        Attempt invalidation before SG dispatch and after success or a
        configured indeterminate transport failure. Failed cache fences do not
        prevent SG dispatch.

        Writes containing implicit uploads use a site-wide fence and repeat it
        after other ordinary failures as well.

        Cancellation during SG dispatch attempts a conservative post-write
        fence before propagating.

        Admit the complete operation, including planning and cleanup, on the
        cache's owning event loop.

        Arguments, results and SG errors are those of `AsyncShotgun.batch`.

        Raises:
            RuntimeError: The cache is closing, closed, or bound to another
                event loop.
            CancelledError: The caller cancels the operation.
        """
        with self._cache.admit() as token:
            method = cast('Any', self._client).batch
            return cast(
                'list[dict[str, object] | bool]',
                await self._write_with_fence(
                    batch_write_plan(self.context.site_namespace, requests),
                    lambda: method(requests),
                    token,
                ),
            )

    def cache_policy(
        self,
        mode: PolicyMode | None = None,
        maximum_age: float | None = None,
        *,
        maximum_age_seconds: float | None = None,
    ) -> AbstractContextManager[None]:
        """Apply one validated task-local cache policy scope.

        Passing non-`None` values for both age spellings raises `TypeError`.
        Entering the returned context validates its arguments. Nonfinite,
        boolean, or nonnumeric ages raise `TypeError`; unsupported modes or
        nonpositive ages raise `ValueError`. Child tasks inherit the scope
        snapshot present when they are created.

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

    async def create(
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

        Cancellation during SG dispatch attempts a conservative post-write
        fence before propagating.

        Admit the complete operation, including planning and cleanup, on the
        cache's owning event loop.

        Arguments, results and SG errors are those of `AsyncShotgun.create`.

        Raises:
            RuntimeError: The cache is closing, closed, or bound to another
                event loop.
            CancelledError: The caller cancels the operation.
        """
        with self._cache.admit() as token:
            method = cast('Any', self._client).create
            return cast(
                'dict[str, object]',
                await self._write_with_fence(
                    create_write_plan(self.context.site_namespace, entity_type, data),
                    lambda: method(entity_type, data, return_fields),
                    token,
                ),
            )

    async def delete(self, entity_type: str, entity_id: int) -> bool:
        """Delete one entity behind conservative lifecycle fences.

        Attempt invalidation before SG dispatch and after success or a
        configured indeterminate transport failure. Failed cache fences do not
        prevent SG dispatch.

        Cancellation during SG dispatch attempts a conservative post-write
        fence before propagating.

        Admit the complete operation, including planning and cleanup, on the
        cache's owning event loop.

        Arguments, results and SG errors are those of `AsyncShotgun.delete`.

        Raises:
            RuntimeError: The cache is closing, closed, or bound to another
                event loop.
            CancelledError: The caller cancels the operation.
        """
        with self._cache.admit() as token:
            changes = lifecycle_changes(self.context.site_namespace, entity_type, entity_id)
            method = cast('Any', self._client).delete
            return cast(
                'bool',
                await self._write_with_fence(
                    WritePlan(changes),
                    lambda: method(entity_type, entity_id),
                    token,
                ),
            )

    async def find(  # noqa: PLR0913, PLR0917 - exact official SG signature
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

        Admit the complete operation, including planning and cleanup, on the
        cache's owning event loop.

        Arguments, results and SG errors are those of `AsyncShotgun.find`.

        Returns:
            An isolated cache hit or the direct SG result, with the upstream
            result contract.

        Raises:
            RuntimeError: The cache is closing, closed, or bound to another
                event loop.
            CancelledError: The caller cancels the operation.
        """
        with self._cache.admit() as token:
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
                await self._entity_read(
                    EntityRead('find', entity_type, fields, values),
                    lambda: method(*values),
                    token,
                ),
            )

    async def find_one(  # noqa: PLR0913, PLR0917 - exact official SG signature
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

        Admit the complete operation, including planning and cleanup, on the
        cache's owning event loop.

        Arguments, results and SG errors are those of `AsyncShotgun.find_one`.

        Returns:
            An isolated cache hit or the direct SG result, with the upstream
            result contract.

        Raises:
            RuntimeError: The cache is closing, closed, or bound to another
                event loop.
            CancelledError: The caller cancels the operation.
        """
        with self._cache.admit() as token:
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
                await self._entity_read(
                    EntityRead('find_one', entity_type, fields, values),
                    lambda: method(*values),
                    token,
                ),
            )

    async def follow(
        self, user: dict[str, object], entity: dict[str, object]
    ) -> dict[str, object]:
        """Follow one entity behind a conservative site fence.

        Attempt invalidation before SG dispatch and after success or a
        configured indeterminate transport failure. Failed cache fences do not
        prevent SG dispatch.

        Cancellation during SG dispatch attempts a conservative post-write
        fence before propagating.

        Admit the complete operation, including planning and cleanup, on the
        cache's owning event loop.

        Arguments, results and SG errors are those of `AsyncShotgun.follow`.

        Raises:
            RuntimeError: The cache is closing, closed, or bound to another
                event loop.
            CancelledError: The caller cancels the operation.
        """
        with self._cache.admit() as token:
            changes = site_changes(self.context.site_namespace)
            method = cast('Any', self._client).follow
            return cast(
                'dict[str, object]',
                await self._write_with_fence(
                    WritePlan(changes), lambda: method(user, entity), token
                ),
            )

    async def note_thread_read(
        self,
        note_id: int,
        entity_fields: dict[str, object] | None = None,
    ) -> list[dict[str, object]]:
        """Always read a note thread directly.

        Bypass cache lookup and population because this composite response has
        no proven URL lifetime for caching.

        Admit the complete operation, including planning and cleanup, on the
        cache's owning event loop.

        Arguments, results and SG errors are those of
        `AsyncShotgun.note_thread_read`.

        Raises:
            RuntimeError: The cache is closing, closed, or bound to another
                event loop.
            CancelledError: The caller cancels the operation.
        """
        with self._cache.admit() as token:
            context = self.context
            method = cast('Any', self._client).note_thread_read

            async def fetch() -> object:
                return await method(note_id, entity_fields)

            return cast(
                'list[dict[str, object]]',
                await self._read_direct(
                    'note_thread_read', fetch, context, token, reason='composite_url_unproven'
                ),
            )

    @classmethod
    @asynccontextmanager
    async def open(  # noqa: PLR0913 - exact prescribed local factory signature
        cls,
        client: object,
        settings: CacheSettings,
        identity: Identity,
        context: ClientContext,
        *,
        metrics: MetricsSink | None = None,
        environ: Mapping[str, object] | None = None,
        indeterminate_write_exceptions: tuple[type[Exception], ...] = (),
    ) -> AsyncGenerator[Self]:
        """Construct and own configured async cache layers around a caller-owned client.

        Args:
            client: Caller-owned native asynchronous SG client.
            settings: Validated local and optional Valkey cache settings.
            identity: Exact effective SG identity namespace.
            context: Complete response-affecting client context.
            metrics: Optional synchronous metrics destination. `None` discards
                metrics and skips correlation-secret construction.
            environ: Environment used for metrics correlation and role
                passwords;
                `None` uses the process environment.
            indeterminate_write_exceptions: Explicit transport exceptions whose
                server outcome is unknown.

        Returns:
            An async context manager that constructs resources on entry and
            owns only its constructed cache. Entry and exit must use the same
            event loop.
            Exit joins admitted work before closing the cache. Caller
            cancellation takes precedence over a body error, which takes
            precedence over ordinary cleanup failure.

        Raises:
            ImportError: Shared settings require the optional Valkey package.
            TypeError: Indeterminate write exception configuration is
                malformed.
            ValueError: The context namespace or a required configured secret
                is invalid.
            Exception: Backend construction fails, or cleanup fails without a
                higher-priority body error or caller cancellation.
        """
        sink = (
            NullMetrics()
            if metrics is None
            else AsyncQueryCache.build_metrics(
                settings,
                process_environ if environ is None else environ,
                metrics,
            )
        )
        backend: AsyncLocalBackend | None = None
        shared: AsyncCacheBackend | None = None
        cache: AsyncQueryCache | None = None
        try:
            if settings.l1.enabled:
                backend = AsyncLocalBackend(settings, monotonic, sink)
            if settings.l2.enabled:
                valkey = import_module('sg_query_cache.valkey_async')
                backend_type = cast('type[AsyncValkeyBackend]', valkey.AsyncValkeyBackend)
                shared = await backend_type.from_settings(
                    settings,
                    process_environ if environ is None else environ,
                    sink,
                )
            cache = AsyncQueryCache(
                settings,
                identity,
                l1=backend,
                l2=shared,
                metrics=sink,
            )
            wrapper = cls(
                client,
                cache,
                context,
                indeterminate_write_exceptions=indeterminate_write_exceptions,
            )
        except BaseException as error:
            resources = (cache,) if cache is not None else (shared, backend)
            await _close_partial(resources, error, sink)
            raise
        try:
            yield wrapper
        except BaseException as error:
            await _close_partial((cache,), error, sink)
            raise
        await cache.close()

    async def preferences_read(self, prefs: list[str] | None = None) -> dict[str, object]:
        """Always read preferences directly.

        Bypass cache lookup and population because this composite response has
        no proven URL lifetime for caching.

        Admit the complete operation, including planning and cleanup, on the
        cache's owning event loop.

        Arguments, results and SG errors are those of
        `AsyncShotgun.preferences_read`.

        Raises:
            RuntimeError: The cache is closing, closed, or bound to another
                event loop.
            CancelledError: The caller cancels the operation.
        """
        with self._cache.admit() as token:
            context = self.context
            method = cast('Any', self._client).preferences_read

            async def fetch() -> object:
                return await method(prefs)

            return cast(
                'dict[str, object]',
                await self._read_direct(
                    'preferences_read', fetch, context, token, reason='composite_url_unproven'
                ),
            )

    @property
    def raw_client(self) -> object:
        """Return the caller-owned client for unsupported explicit operations."""
        return self._client

    async def replace_context(self, context: ClientContext) -> None:
        """Atomically publish a replacement and fence prior local reachability.

        The new context is published before the fence awaits and remains
        installed if that await is cancelled. This does not reconfigure the raw
        SG client. A changed context also advances the site generation in
        shared Valkey, invalidating cached reads for every process using that
        site. An equal context leaves the cache unchanged.

        Args:
            context: Complete replacement context.

        Raises:
            RuntimeError: The cache is closing, closed, or bound to another
                loop.
            ValueError: Shared cache and replacement namespaces disagree.
        """
        with self._cache.admit() as token:
            require_context_namespace(self._settings, context)
            previous = self._context
            if previous == context:
                return
            self._context = context
            await self._cache.invalidate_admitted(
                site_changes(previous.site_namespace),
                token=token,
            )

    async def revive(self, entity_type: str, entity_id: int) -> bool:
        """Revive one entity behind conservative lifecycle fences.

        Attempt invalidation before SG dispatch and after success or a
        configured indeterminate transport failure. Failed cache fences do not
        prevent SG dispatch.

        Cancellation during SG dispatch attempts a conservative post-write
        fence before propagating.

        Admit the complete operation, including planning and cleanup, on the
        cache's owning event loop.

        Arguments, results and SG errors are those of `AsyncShotgun.revive`.

        Raises:
            RuntimeError: The cache is closing, closed, or bound to another
                event loop.
            CancelledError: The caller cancels the operation.
        """
        with self._cache.admit() as token:
            changes = lifecycle_changes(self.context.site_namespace, entity_type, entity_id)
            method = cast('Any', self._client).revive
            return cast(
                'bool',
                await self._write_with_fence(
                    WritePlan(changes),
                    lambda: method(entity_type, entity_id),
                    token,
                ),
            )

    async def schema_entity_read(
        self, project_entity: dict[str, object] | None = None
    ) -> dict[str, dict[str, object]]:
        """Return cached entity schema for an optional project.

        Use identity-scoped schema caching subject to active cache policy.
        Calls that cannot be normalized read directly; cache failures fall back
        to SG.

        Admit the complete operation, including planning and cleanup, on the
        cache's owning event loop.

        Arguments, results and SG errors are those of
        `AsyncShotgun.schema_entity_read`.

        Returns:
            An isolated cache hit or the direct SG result, with the upstream
            result contract.

        Raises:
            RuntimeError: The cache is closing, closed, or bound to another
                event loop.
            CancelledError: The caller cancels the operation.
        """
        with self._cache.admit() as token:
            context = self.context
            method = cast('Any', self._client).schema_entity_read

            async def fetch() -> object:
                return await method(project_entity)

            call = schema_entity_read_call(project_entity)
            if call is None:
                return cast(
                    'dict[str, dict[str, object]]',
                    await self._read_direct('schema_entity_read', fetch, context, token),
                )
            return cast(
                'dict[str, dict[str, object]]',
                await self._read(
                    _ReadRequest(call, self._schema_plan(context), context),
                    fetch,
                    token,
                ),
            )

    async def schema_field_create(
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

        Cancellation during SG dispatch attempts a conservative post-write
        fence before propagating.

        Admit the complete operation, including planning and cleanup, on the
        cache's owning event loop.

        Arguments, results and SG errors are those of
        `AsyncShotgun.schema_field_create`.

        Raises:
            RuntimeError: The cache is closing, closed, or bound to another
                event loop.
            CancelledError: The caller cancels the operation.
        """
        with self._cache.admit() as token:
            changes = schema_changes(self.context.site_namespace)
            method = cast('Any', self._client).schema_field_create
            return cast(
                'str',
                await self._write_with_fence(
                    WritePlan(changes),
                    lambda: method(entity_type, data_type, display_name, properties),
                    token,
                ),
            )

    async def schema_field_delete(self, entity_type: str, field_name: str) -> bool:
        """Delete one field behind schema-generation fences.

        Attempt invalidation before SG dispatch and after success or a
        configured indeterminate transport failure. Failed cache fences do not
        prevent SG dispatch.

        Cancellation during SG dispatch attempts a conservative post-write
        fence before propagating.

        Admit the complete operation, including planning and cleanup, on the
        cache's owning event loop.

        Arguments, results and SG errors are those of
        `AsyncShotgun.schema_field_delete`.

        Raises:
            RuntimeError: The cache is closing, closed, or bound to another
                event loop.
            CancelledError: The caller cancels the operation.
        """
        with self._cache.admit() as token:
            changes = schema_changes(self.context.site_namespace)
            method = cast('Any', self._client).schema_field_delete
            return cast(
                'bool',
                await self._write_with_fence(
                    WritePlan(changes),
                    lambda: method(entity_type, field_name),
                    token,
                ),
            )

    async def schema_field_read(
        self,
        entity_type: str,
        field_name: str | None = None,
        project_entity: dict[str, object] | None = None,
    ) -> dict[str, dict[str, object]]:
        """Return cached field schema for an entity type and optional project.

        Use identity-scoped schema caching subject to active cache policy.
        Calls that cannot be normalized read directly; cache failures fall back
        to SG.

        Admit the complete operation, including planning and cleanup, on the
        cache's owning event loop.

        Arguments, results and SG errors are those of
        `AsyncShotgun.schema_field_read`.

        Returns:
            An isolated cache hit or the direct SG result, with the upstream
            result contract.

        Raises:
            RuntimeError: The cache is closing, closed, or bound to another
                event loop.
            CancelledError: The caller cancels the operation.
        """
        with self._cache.admit() as token:
            context = self.context
            method = cast('Any', self._client).schema_field_read

            async def fetch() -> object:
                return await method(entity_type, field_name, project_entity)

            call = schema_field_read_call(entity_type, field_name, project_entity)
            if call is None:
                return cast(
                    'dict[str, dict[str, object]]',
                    await self._read_direct('schema_field_read', fetch, context, token),
                )
            return cast(
                'dict[str, dict[str, object]]',
                await self._read(
                    _ReadRequest(call, self._schema_plan(context), context),
                    fetch,
                    token,
                ),
            )

    async def schema_field_update(
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

        Cancellation during SG dispatch attempts a conservative post-write
        fence before propagating.

        Admit the complete operation, including planning and cleanup, on the
        cache's owning event loop.

        Arguments, results and SG errors are those of
        `AsyncShotgun.schema_field_update`.

        Raises:
            RuntimeError: The cache is closing, closed, or bound to another
                event loop.
            CancelledError: The caller cancels the operation.
        """
        with self._cache.admit() as token:
            changes = schema_changes(self.context.site_namespace)
            method = cast('Any', self._client).schema_field_update
            return cast(
                'bool',
                await self._write_with_fence(
                    WritePlan(changes),
                    lambda: method(entity_type, field_name, properties, project_entity),
                    token,
                ),
            )

    async def schema_read(
        self, project_entity: dict[str, object] | None = None
    ) -> dict[str, dict[str, dict[str, object]]]:
        """Return cached full schema for an optional project.

        Use identity-scoped schema caching subject to active cache policy.
        Calls that cannot be normalized read directly; cache failures fall back
        to SG.

        Admit the complete operation, including planning and cleanup, on the
        cache's owning event loop.

        Arguments, results and SG errors are those of
        `AsyncShotgun.schema_read`.

        Returns:
            An isolated cache hit or the direct SG result, with the upstream
            result contract.

        Raises:
            RuntimeError: The cache is closing, closed, or bound to another
                event loop.
            CancelledError: The caller cancels the operation.
        """
        with self._cache.admit() as token:
            return cast(
                'dict[str, dict[str, dict[str, object]]]',
                await self._schema_read(project_entity, self.context, token),
            )

    async def share_thumbnail(
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

        Cancellation during SG dispatch attempts a conservative post-write
        fence before propagating.

        Admit the complete operation, including planning and cleanup, on the
        cache's owning event loop.

        Arguments, results and SG errors are those of
        `AsyncShotgun.share_thumbnail`.

        Raises:
            RuntimeError: The cache is closing, closed, or bound to another
                event loop.
            CancelledError: The caller cancels the operation.
        """
        with self._cache.admit() as token:
            changes = site_changes(self.context.site_namespace)
            method = cast('Any', self._client).share_thumbnail
            return cast(
                'int',
                await self._write_with_fence(
                    WritePlan(changes),
                    lambda: method(
                        entities,
                        thumbnail_path,
                        source_entity,
                        filmstrip_thumbnail,
                        **kwargs,
                    ),
                    token,
                ),
            )

    async def summarize(  # noqa: PLR0913, PLR0917 - exact official SG signature
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

        Admit the complete operation, including planning and cleanup, on the
        cache's owning event loop.

        Arguments, results and SG errors are those of `AsyncShotgun.summarize`.

        Raises:
            RuntimeError: The cache is closing, closed, or bound to another
                event loop.
            CancelledError: The caller cancels the operation.
        """
        with self._cache.admit() as token:
            context = self.context
            method = cast('Any', self._client).summarize

            async def fetch() -> object:
                return await method(
                    entity_type,
                    filters,
                    summary_fields,
                    filter_operator,
                    grouping,
                    include_archived_projects,
                )

            return cast(
                'dict[str, object]',
                await self._read_direct(
                    'summarize', fetch, context, token, reason='composite_url_unproven'
                ),
            )

    async def text_search(
        self,
        text: str,
        entity_types: dict[str, object],
        project_ids: list[object] | None = None,
        limit: int | None = None,
    ) -> dict[str, object]:
        """Always search text directly.

        Bypass cache lookup and population because this composite response has
        no proven URL lifetime for caching.

        Admit the complete operation, including planning and cleanup, on the
        cache's owning event loop.

        Arguments, results and SG errors are those of
        `AsyncShotgun.text_search`.

        Raises:
            RuntimeError: The cache is closing, closed, or bound to another
                event loop.
            CancelledError: The caller cancels the operation.
        """
        with self._cache.admit() as token:
            context = self.context
            method = cast('Any', self._client).text_search

            async def fetch() -> object:
                return await method(text, entity_types, project_ids, limit)

            return cast(
                'dict[str, object]',
                await self._read_direct(
                    'text_search', fetch, context, token, reason='composite_url_unproven'
                ),
            )

    async def unfollow(
        self, user: dict[str, object], entity: dict[str, object]
    ) -> dict[str, object]:
        """Unfollow one entity behind a conservative site fence.

        Attempt invalidation before SG dispatch and after success or a
        configured indeterminate transport failure. Failed cache fences do not
        prevent SG dispatch.

        Cancellation during SG dispatch attempts a conservative post-write
        fence before propagating.

        Admit the complete operation, including planning and cleanup, on the
        cache's owning event loop.

        Arguments, results and SG errors are those of `AsyncShotgun.unfollow`.

        Raises:
            RuntimeError: The cache is closing, closed, or bound to another
                event loop.
            CancelledError: The caller cancels the operation.
        """
        with self._cache.admit() as token:
            changes = site_changes(self.context.site_namespace)
            method = cast('Any', self._client).unfollow
            return cast(
                'dict[str, object]',
                await self._write_with_fence(
                    WritePlan(changes), lambda: method(user, entity), token
                ),
            )

    async def update(
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

        Cancellation during SG dispatch attempts a conservative post-write
        fence before propagating.

        Admit the complete operation, including planning and cleanup, on the
        cache's owning event loop.

        Arguments, results and SG errors are those of `AsyncShotgun.update`.

        Raises:
            RuntimeError: The cache is closing, closed, or bound to another
                event loop.
            CancelledError: The caller cancels the operation.
        """
        with self._cache.admit() as token:
            method = cast('Any', self._client).update
            return cast(
                'dict[str, object]',
                await self._write_with_fence(
                    update_write_plan(self.context.site_namespace, entity_type, entity_id, data),
                    lambda: method(entity_type, entity_id, data, multi_entity_update_modes),
                    token,
                ),
            )

    async def update_project_last_accessed(
        self,
        project: dict[str, object],
        user: dict[str, object] | None = None,
    ) -> None:
        """Update project access state behind a conservative site fence.

        Attempt invalidation before SG dispatch and after success or a
        configured indeterminate transport failure. Failed cache fences do not
        prevent SG dispatch.

        Cancellation during SG dispatch attempts a conservative post-write
        fence before propagating.

        Admit the complete operation, including planning and cleanup, on the
        cache's owning event loop.

        Arguments, results and SG errors are those of
        `AsyncShotgun.update_project_last_accessed`.

        Raises:
            RuntimeError: The cache is closing, closed, or bound to another
                event loop.
            CancelledError: The caller cancels the operation.
        """
        with self._cache.admit() as token:
            changes = site_changes(self.context.site_namespace)
            method = cast('Any', self._client).update_project_last_accessed
            await self._write_with_fence(WritePlan(changes), lambda: method(project, user), token)

    async def upload(  # noqa: PLR0913, PLR0917 - exact official SG signature
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

        Cancellation during SG dispatch attempts a conservative post-write
        fence before propagating.

        Admit the complete operation, including planning and cleanup, on the
        cache's owning event loop.

        Arguments, results and SG errors are those of `AsyncShotgun.upload`.

        Raises:
            RuntimeError: The cache is closing, closed, or bound to another
                event loop.
            CancelledError: The caller cancels the operation.
        """
        with self._cache.admit() as token:
            changes = site_changes(self.context.site_namespace)
            method = cast('Any', self._client).upload
            return cast(
                'int',
                await self._write_with_fence(
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
                    token,
                ),
            )

    async def upload_filmstrip_thumbnail(
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

        Cancellation during SG dispatch attempts a conservative post-write
        fence before propagating.

        Admit the complete operation, including planning and cleanup, on the
        cache's owning event loop.

        Arguments, results and SG errors are those of
        `AsyncShotgun.upload_filmstrip_thumbnail`.

        Raises:
            RuntimeError: The cache is closing, closed, or bound to another
                event loop.
            CancelledError: The caller cancels the operation.
        """
        with self._cache.admit() as token:
            changes = site_changes(self.context.site_namespace)
            method = cast('Any', self._client).upload_filmstrip_thumbnail
            return cast(
                'int',
                await self._write_with_fence(
                    WritePlan(changes),
                    lambda: method(entity_type, entity_id, path, **kwargs),
                    token,
                ),
            )

    async def upload_thumbnail(
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

        Cancellation during SG dispatch attempts a conservative post-write
        fence before propagating.

        Admit the complete operation, including planning and cleanup, on the
        cache's owning event loop.

        Arguments, results and SG errors are those of
        `AsyncShotgun.upload_thumbnail`.

        Raises:
            RuntimeError: The cache is closing, closed, or bound to another
                event loop.
            CancelledError: The caller cancels the operation.
        """
        with self._cache.admit() as token:
            changes = site_changes(self.context.site_namespace)
            method = cast('Any', self._client).upload_thumbnail
            return cast(
                'int',
                await self._write_with_fence(
                    WritePlan(changes),
                    lambda: method(entity_type, entity_id, path, **kwargs),
                    token,
                ),
            )

    async def user_subscriptions_create(self, users: list[dict[str, object]]) -> bool:
        """Create user subscriptions behind a conservative site fence.

        Attempt invalidation before SG dispatch and after success or a
        configured indeterminate transport failure. Failed cache fences do not
        prevent SG dispatch.

        Cancellation during SG dispatch attempts a conservative post-write
        fence before propagating.

        Admit the complete operation, including planning and cleanup, on the
        cache's owning event loop.

        Arguments, results and SG errors are those of
        `AsyncShotgun.user_subscriptions_create`.

        Raises:
            RuntimeError: The cache is closing, closed, or bound to another
                event loop.
            CancelledError: The caller cancels the operation.
        """
        with self._cache.admit() as token:
            changes = site_changes(self.context.site_namespace)
            method = cast('Any', self._client).user_subscriptions_create
            return cast(
                'bool',
                await self._write_with_fence(WritePlan(changes), lambda: method(users), token),
            )

    async def work_schedule_read(
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

        Admit the complete operation, including planning and cleanup, on the
        cache's owning event loop.

        Arguments, results and SG errors are those of
        `AsyncShotgun.work_schedule_read`.

        Returns:
            An isolated cache hit or the direct SG result, with the upstream
            result contract.

        Raises:
            RuntimeError: The cache is closing, closed, or bound to another
                event loop.
            CancelledError: The caller cancels the operation.
        """
        with self._cache.admit() as token:
            context = self.context
            method = cast('Any', self._client).work_schedule_read

            async def fetch() -> object:
                return await method(start_date, end_date, project, user)

            call = work_schedule_read_call(start_date, end_date, project, user)
            if call is None:
                return cast(
                    'dict[str, object]',
                    await self._read_direct('work_schedule_read', fetch, context, token),
                )
            return cast(
                'dict[str, object]',
                await self._read(
                    _ReadRequest(
                        call,
                        self._ttl_only_plan('work_schedule_read', context),
                        context,
                    ),
                    fetch,
                    token,
                ),
            )

    async def work_schedule_update(  # noqa: PLR0913, PLR0917 - exact official SG signature
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

        Cancellation during SG dispatch attempts a conservative post-write
        fence before propagating.

        Admit the complete operation, including planning and cleanup, on the
        cache's owning event loop.

        Arguments, results and SG errors are those of
        `AsyncShotgun.work_schedule_update`.

        Raises:
            RuntimeError: The cache is closing, closed, or bound to another
                event loop.
            CancelledError: The caller cancels the operation.
        """
        with self._cache.admit() as token:
            changes = site_changes(self.context.site_namespace)
            method = cast('Any', self._client).work_schedule_update
            return cast(
                'dict[str, object]',
                await self._write_with_fence(
                    WritePlan(changes),
                    lambda: method(
                        date,
                        working,
                        description,
                        project,
                        user,
                        recalculate_field,
                    ),
                    token,
                ),
            )


def _record_partial_cleanup_failure(metrics: MetricsSink, error: Exception | None) -> None:
    """Report an owned cleanup failure with its optional client category."""
    labels = {'backend_operation': 'factory_close', 'outcome': 'failure'}
    if isinstance(error, BackendUnavailable) and error.failure_class is not None:
        labels['failure_class'] = error.failure_class
    with suppress(Exception):
        metrics.increment('backend_failure', labels)


async def _close_partial(
    resources: tuple[AsyncQueryCache | AsyncCacheBackend | None, ...],
    error: BaseException,
    metrics: MetricsSink,
) -> None:
    """Join each partially constructed layer with the original error precedence."""
    task = current_task()
    if task is None:
        raise RuntimeError('partial cleanup requires an asyncio task')
    cancellation = error if isinstance(error, CancelledError) else None
    for resource in resources:
        if resource is not None:
            # Deliver pending requests separately from a resource's cancelled
            # child; a stale handled request is not new caller cancellation.
            try:
                await sleep(0)
            except CancelledError as pending_cancellation:
                cancellation = cancellation or pending_cancellation
            requests = task.cancelling()
            failed = False
            cleanup_failure: Exception | None = None
            try:
                await resource.close()
            except CancelledError as cleanup_cancellation:
                failed = True
                if task.cancelling() > requests:
                    cancellation = cancellation or cleanup_cancellation
            except Exception as cleanup_error:  # noqa: BLE001 - record lower-priority owned cleanup failure
                failed = True
                cleanup_failure = cleanup_error
            if failed:
                _record_partial_cleanup_failure(metrics, cleanup_failure)
    if cancellation is not None:
        raise cancellation from None
