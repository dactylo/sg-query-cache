# Same-file collaborators use private implementation members.
# pyright: reportPrivateUsage=false

"""Provide the optional asynchronous two-role Valkey cache backend.

Use `AsyncValkeyBackend.from_settings` to own configured result and metadata
roles, or `AsyncValkeyMetadataStore.from_settings` for metadata alone. Both
factories are awaited and bind to that loop. Close is terminal and joins owned
cleanup. Import this module explicitly; the package root keeps Valkey optional.
"""

from asyncio import (
    CancelledError,
    Future,
    Task,
    create_task,
    gather,
    get_running_loop,
    timeout,
)
from collections.abc import AsyncGenerator, Awaitable, Callable, Generator, Iterable, Mapping
from contextlib import asynccontextmanager, contextmanager, suppress
from functools import partial
from itertools import chain
from secrets import token_hex
from time import monotonic
from typing import Protocol, Self, TypeVar, cast
from weakref import WeakKeyDictionary

from typing_extensions import override
from valkey.asyncio import Connection, Sentinel, Valkey
from valkey.asyncio.connection import AbstractConnection, SSLConnection
from valkey.asyncio.connection import ConnectionPool as MaintainedConnectionPool
from valkey.asyncio.retry import Retry
from valkey.asyncio.sentinel import SentinelConnectionPool as MaintainedSentinelPool
from valkey.asyncio.sentinel import SentinelManagedConnection
from valkey.backoff import NoBackoff
from valkey.exceptions import ConnectionError as ValkeyConnectionError
from valkey.exceptions import ReadOnlyError, ValkeyError
from valkey.exceptions import TimeoutError as ValkeyTimeoutError

from sg_query_cache._async_support import (
    Admission,
    CleanupOutcome,
    LoopAffinity,
    enter_cleanup,
    join_shared_close,
    joined_cleanup,
)
from sg_query_cache._async_valkey_control import (
    AsyncValkeyControl,
    ControlClient,
    PinnedControlClient,
    ResultClient,
    ScriptClient,
    execute_commit,
    execute_script,
)
from sg_query_cache._backend import (
    BackendCapacity,
    BackendProtocolError,
    BackendUnavailable,
    StaleIncarnation,
)
from sg_query_cache._backend_validation import (
    encoded_payload,
    fill_lease,
    flush_reason,
    generation_changes,
    metric_flush_reason,
    opaque_key,
)
from sg_query_cache._config import (
    CacheSettings,
    ValkeyRoleSettings,
    parse_sentinel_address,
)
from sg_query_cache._keys import KeyBuilder
from sg_query_cache._metrics import MetricsSink
from sg_query_cache._model import (
    CacheHit,
    DependencyTag,
    FillLease,
    GenerationChange,
    GenerationVector,
)
from sg_query_cache._valkey_common import CircuitBreaker as _CircuitBreaker
from sg_query_cache._valkey_common import (
    Incarnation,
    LookupOutcome,
    RoleEpoch,
    ScriptArgument,
    decode_change_reply,
    decode_release_reply,
    decode_reserve_reply,
    decode_resolve_reply,
    finalize_lookup,
    interpret_lookup,
    lookup_operands,
    metadata_operands,
    milliseconds,
    release_operands,
    require_enabled_l2,
    reserve_operands,
    role_password,
    validated_resolve_tags,
)
from sg_query_cache._valkey_proof import require_primary_reply
from sg_query_cache._valkey_scripts import (
    CHANGE_SCRIPT,
    CHANGE_SCRIPT_SHA,
    LOOKUP_SCRIPT,
    LOOKUP_SCRIPT_SHA,
    RELEASE_SCRIPT,
    RELEASE_SCRIPT_SHA,
    RESERVE_SCRIPT,
    RESERVE_SCRIPT_SHA,
    RESOLVE_SCRIPT,
    RESOLVE_SCRIPT_SHA,
    translate_client_error,
)

_T = TypeVar('_T')


class _MetadataStore(Protocol):
    """Provide generation metadata and its independent resource lifecycle."""

    async def change(self, changes: tuple[GenerationChange, ...]) -> None:
        """Replace the ordered dependency generations."""
        ...

    async def close(self) -> None:
        """Release metadata resources."""
        ...

    async def flush_site(self, site_namespace: str, reason: str) -> None:
        """Replace the site's permanent generation."""
        ...

    async def resolve(self, tags: tuple[DependencyTag, ...]) -> GenerationVector:
        """Resolve ordered dependency generations."""
        ...


class _ResultRole(ScriptClient, ResultClient, Protocol):
    """Supply result commands, observed execution, refresh and terminal cleanup."""

    async def close(self) -> None:
        """Release this role's resources."""
        ...

    async def refresh(self) -> None:
        """Discard the current pool and construct a fresh connection."""
        ...


class _ValkeyClient(ControlClient, ResultClient, Protocol):
    """Supply loop-native role operations and terminal cleanup."""

    async def close(self) -> None:
        """Close the role and every discovery client it owns."""
        ...


class _Lifecycle:
    """Own backend admission, operation budgets and one shared terminal close."""

    def __init__(self, timeout_seconds: float) -> None:
        """Retain loop-owned obligations before any close task can be created."""
        self._admission = Admission()
        self._timeout_seconds = timeout_seconds
        self._task: Task[CleanupOutcome[None]] | None = None
        self._completion: Future[CleanupOutcome[None]] | None = None
        self._pending: list[Callable[[], Awaitable[None]]] = []
        self._settling = False

    @asynccontextmanager
    async def _budget(self) -> AsyncGenerator[None]:
        """Apply the whole operation's cooperative budget."""
        try:
            async with timeout(self._timeout_seconds):
                yield
        except TimeoutError:
            raise BackendUnavailable('Valkey operation deadline expired') from None

    async def _close_all(self) -> CleanupOutcome[None]:
        """Join operations and settle every explicitly retained close obligation."""
        idle = await joined_cleanup(self._admission.wait_idle)
        if idle.error is not None:
            # Independent cancellation is failed cleanup, not proof of idleness.
            await self._admission.wait_idle()
        deadline = get_running_loop().time() + self._timeout_seconds
        outcome = await _close_resources(tuple(self._pending), deadline=deadline)
        self._pending.clear()
        self._admission.finish_close()
        failure = idle.error or outcome.error
        if isinstance(failure, TimeoutError):
            failure = BackendUnavailable('Valkey close deadline expired')
        return CleanupOutcome(
            error=failure,
            cancellation=idle.cancellation or outcome.cancellation,
        )

    @contextmanager
    def _operation(self) -> Generator[None]:
        """Admit the whole operation before its first suspension."""
        with self._admission.operation():
            yield

    async def _settle_close(
        self,
        task: Task[CleanupOutcome[None]],
        completion: Future[CleanupOutcome[None]],
    ) -> CancelledError | None:
        """Claim unfinished ownership directly and publish one shared failure."""
        if completion.done() or self._settling:
            return None
        self._settling = True
        if task.cancelled():
            # Direct outer settlement cannot be cancelled before coroutine entry.
            settled = await self._close_all()
            completion.set_result(
                CleanupOutcome(
                    error=BackendUnavailable('Valkey close task was interrupted'),
                )
            )
            return settled.cancellation
        settled = task.result()
        if settled.cancellation is not None and settled.error is None:
            settled = CleanupOutcome[None](
                error=BackendUnavailable('Valkey close was interrupted')
            )
        completion.set_result(settled)
        return None

    async def _close(self, resources: tuple[Callable[[], Awaitable[None]], ...]) -> None:
        """Join shared cleanup and synchronously claim any pre-entry settlement."""
        if self._admission.begin_close():
            self._pending = list(resources)
            self._completion = get_running_loop().create_future()
            self._task = create_task(self._close_all())
        elif self._admission.closed:
            return
        task, completion = self._task, self._completion
        if task is None or completion is None:
            raise RuntimeError('close task was not published')
        await join_shared_close(task, completion, lambda: self._settle_close(task, completion))


async def _close_resources(
    resources: tuple[Callable[[], Awaitable[None]], ...],
    *,
    deadline: float | None = None,
) -> CleanupOutcome[None]:
    """Attempt every resource and retain the first ordinary cleanup failure."""
    failure: Exception | None = None
    cancellation: CancelledError | None = None
    for resource_close in resources:
        outcome = await enter_cleanup(resource_close, deadline=deadline)
        failure = failure or outcome.error
        cancellation = cancellation or outcome.cancellation
    return CleanupOutcome(error=failure, cancellation=cancellation)


async def _cleanup_construction(
    resources: tuple[Callable[[], Awaitable[None]], ...],
    error: BaseException,
    metrics: MetricsSink | None = None,
) -> None:
    """Settle owned resources with caller cancellation before construction failure."""
    outcome = await _close_resources(resources)
    if outcome.error is not None and metrics is not None:
        with suppress(Exception):
            metrics.increment(
                'backend_failure',
                {'backend_operation': 'construction_close', 'outcome': 'failure'},
            )
    if isinstance(error, CancelledError):
        return
    if outcome.cancellation is not None:
        raise outcome.cancellation from None


class _ConnectionGeneration:
    """Weakly assign role-unique identities to connections on one event loop."""

    def __init__(self, *, prove_primary: bool = False) -> None:
        """Create an unbound tracker without locks or connection I/O."""
        self._prove_primary = prove_primary
        self._affinity = LoopAffinity()
        self._by_connection: WeakKeyDictionary[AbstractConnection, int] = WeakKeyDictionary()
        self._generation_count = 0

    def _connected(self, connection: AbstractConnection) -> None:
        """Record one successful connection under a new role-unique generation without awaiting."""
        self._affinity.check()
        self._generation_count += 1
        self._by_connection[connection] = self._generation_count

    def _for_connection(self, connection: AbstractConnection | None) -> int | None:
        """Sample one connection's latest role-unique generation without I/O."""
        self._affinity.check()
        return None if connection is None else self._by_connection.get(connection)

    @property
    def _generation(self) -> int:
        """Sample the aggregate diagnostic generation without I/O."""
        self._affinity.check()
        return self._generation_count

    async def _initialize(self, connection: AbstractConnection) -> None:
        """Complete setup, prove HA primaries, and register a weak reconnect callback."""
        self._affinity.check()
        await connection.on_connect()
        if self._prove_primary:
            await _prove_connection(connection)
        connection.register_connect_callback(self._connected)


class _Connection(Connection):
    """Keep command writes and their cancellation in the caller's task."""

    @override
    async def send_packed_command(
        self, command: bytes | str | Iterable[bytes], check_health: bool = True
    ) -> None:
        if not self.is_connected:
            await self.connect()
        elif check_health:
            await self.check_health()
        try:
            if isinstance(command, str):
                command = command.encode()
            if isinstance(command, bytes):
                command = [command]
            # Valkey 6.1 uses wait_for here. On Python 3.11 its completed-child
            # race can swallow caller cancellation before the response read.
            async with timeout(self.socket_timeout or None):
                await self._send_packed_command(command)
        except TimeoutError:
            await self.disconnect(nowait=True)
            raise ValkeyTimeoutError('Timeout writing to socket') from None
        except OSError as error:
            await self.disconnect(nowait=True)
            raise ValkeyConnectionError('Valkey socket write failed') from error
        except BaseException:
            await self.disconnect(nowait=True)
            raise


class _SSLConnection(_Connection, SSLConnection):
    """Retain maintained TLS setup with cancellation-safe command writes."""


class _SentinelConnection(SentinelManagedConnection, _Connection):
    """Let maintained demotion handling reach the role owner without graceful shutdown."""

    @override
    async def disconnect(self, nowait: bool = True) -> None:
        """Default socket shutdown to nowait so demotion reaches role invalidation."""
        await super().disconnect(nowait=nowait)


class _SentinelSSLConnection(_SentinelConnection, SSLConnection):
    """Retain maintained TLS connection setup with non-graceful demotion shutdown."""


class _ConnectionPool(MaintainedConnectionPool):
    """Close interrupted acquisitions and discard sockets without graceful waits."""

    def __init__(
        self,
        *,
        connection_class: type[AbstractConnection] = _Connection,
        max_connections: int | None = None,
        **connection_kwargs: object,
    ) -> None:
        """Build a maintained pool with command mutation retries disabled."""
        connection_kwargs.update(
            retry=Retry(NoBackoff(), 0), retry_on_timeout=False, retry_on_error=[]
        )
        super().__init__(
            connection_class=connection_class,
            max_connections=max_connections,
            **connection_kwargs,
        )

    async def _settle_disconnect(self) -> None:
        """Attempt every detached socket after independently interrupted cleanup."""
        for connection in tuple(chain(self._available_connections, self._in_use_connections)):
            # nowait disconnect uses the maintained socket-close path without
            # suspending for shutdown. Preserve the original cleanup outcome.
            with suppress(Exception):
                await connection.disconnect(nowait=True)

    async def _settle_release(self, connection: AbstractConnection) -> None:
        """Return a still-owned pin after interrupted release through the maintained API."""
        if connection in self._in_use_connections:
            # Maintained release performs only synchronous set/list updates.
            await super().release(connection)

    @override
    async def disconnect(self, inuse_connections: bool = True) -> None:
        connections = tuple(
            chain(
                self._available_connections,
                self._in_use_connections if inuse_connections else (),
            )
        )
        results = await gather(
            *(connection.disconnect(nowait=True) for connection in connections),
            return_exceptions=True,
        )
        for result in results:
            if isinstance(result, BaseException):
                raise result

    @override
    async def ensure_connection(self, connection: AbstractConnection) -> None:
        try:
            await super().ensure_connection(connection)
        except CancelledError:
            # The maintained get_connection handler releases this exact borrow
            # after we close any socket established before setup was interrupted.
            await _disconnect_interrupted(connection)
            raise


class _SentinelConnectionPool(MaintainedSentinelPool, _ConnectionPool):
    """Own Sentinel acquisition and demotion cleanup for plaintext and TLS primaries."""

    def __init__(
        self, service_name: str, sentinel_manager: object, **connection_kwargs: object
    ) -> None:
        """Select owned per-connection shutdown before maintained Sentinel construction."""
        connection_kwargs['connection_class'] = (
            _SentinelSSLConnection if connection_kwargs.get('ssl', False) else _SentinelConnection
        )
        super().__init__(service_name, sentinel_manager, **connection_kwargs)


class _RawClient(Protocol):
    """Type the maintained asyncio surface without its synchronous pool annotation."""

    connection: AbstractConnection | None
    connection_pool: _ConnectionPool | _SentinelConnectionPool

    async def aclose(self, close_connection_pool: bool = False) -> None:
        """Release the pin and optionally disconnect its whole pool."""
        ...

    def client(self) -> Self:
        """Create an uninitialized single-connection child client."""
        ...

    async def execute_command(self, *args: ScriptArgument) -> object:
        """Execute one command through maintained send/read and retry handling."""
        ...

    async def initialize(self) -> Self:
        """Acquire and initialize a maintained pinned connection."""
        ...


class _RoleClient:
    """Own a replaceable role pool and prevent detached pins from issuing commands."""

    def __init__(
        self,
        factory: Callable[[], _RawClient],
        connections: _ConnectionGeneration,
        endpoint_probe: Callable[[_RawClient], Awaitable[object]] | None = None,
    ) -> None:
        """Retain construction and discovery dependencies without performing I/O."""
        self._affinity = LoopAffinity()
        self._client: _RawClient | None = None
        self._closed = False
        self._connections = connections
        self._endpoint: object | None = None
        self._endpoint_probe = endpoint_probe
        self._factory = factory
        self._pool_epoch = 0

    def _active(self) -> _RawClient:
        """Return the current pool without awaiting or reviving a closed role."""
        self._affinity.check()
        if self._closed:
            raise RuntimeError('async Valkey role is closed')
        if self._client is None:
            self._client = self._factory()
        return self._client

    def _connection_generation(self, client: _RawClient) -> int | None:
        """Sample a borrowed connection's latest role-unique generation without I/O."""
        return self._connections._for_connection(client.connection)

    async def _invalidate(self, expected: _RawClient | None = None) -> None:
        """Detach an affected pool before joining attempt-all connection cleanup."""
        self._affinity.check()
        client = self._client
        if client is None or (expected is not None and client is not expected):
            return
        self._client = None
        self._pool_epoch += 1
        outcome = await joined_cleanup(lambda: client.aclose(close_connection_pool=True))
        if outcome.error is not None:
            await client.connection_pool._settle_disconnect()
        if outcome.cancellation is not None:
            raise outcome.cancellation
        if outcome.error is not None:
            translate_client_error(outcome.error)

    async def _invoke_on(self, pool: _RawClient, operation: Callable[[], Awaitable[_T]]) -> _T:
        """Run once against a current pool and discard reconnect-class failures."""
        self._affinity.check()
        if self._closed or self._client is not pool:
            raise BackendUnavailable('Valkey pin belongs to a discarded pool')
        try:
            return await operation()
        except (ReadOnlyError, ValkeyConnectionError, ValkeyTimeoutError):
            await self._invalidate(pool)
            raise

    def _role_epoch(self) -> tuple[int, object | None]:
        """Sample pool and endpoint state without the aggregate diagnostic counter."""
        self._affinity.check()
        return self._pool_epoch, self._endpoint

    async def close(self) -> None:
        """Make the role terminal and detach its pool before cleanup awaits."""
        self._affinity.check()
        self._closed = True
        await self._invalidate()

    def epoch(self) -> RoleEpoch:
        """Sample the immutable pool, connection, and endpoint epoch without I/O."""
        self._affinity.check()
        return self._pool_epoch, self._connections._generation, self._endpoint

    @asynccontextmanager
    async def pin(self) -> AsyncGenerator['_PinnedRoleClient']:
        """Borrow one maintained single-connection client and join its release."""
        pool = self._active()
        pinned = pool.client()
        wrapped = _PinnedRoleClient(self, pool, pinned)
        cancellation: CancelledError | None = None
        try:
            failure: Exception | None = None
            try:
                await pinned.initialize()
            except ValkeyError as error:
                failure = error
            if failure is not None:
                if isinstance(failure, (ReadOnlyError, ValkeyConnectionError, ValkeyTimeoutError)):
                    await self._invalidate(pool)
                translate_client_error(failure)
            if self._client is not pool:
                raise BackendUnavailable('Valkey pool changed during acquisition')
            yield wrapped
        except CancelledError as error:
            cancellation = error
            raise
        except (ReadOnlyError, ValkeyConnectionError, ValkeyTimeoutError):
            await self._invalidate(pool)
            raise
        finally:
            wrapped._finish()
            if self._client is not pool and pinned.connection is not None:
                with suppress(Exception):
                    await pinned.connection.disconnect(nowait=True)
            await _close_pin(pinned, cancellation)

    async def refresh(self) -> None:
        """Detach the affected pool and construct a fresh lazy replacement."""
        await self._invalidate()
        self._active()

    async def refresh_endpoint(self) -> None:
        """Discover and publish the current endpoint outside pure epoch sampling."""
        self._affinity.check()
        probe = self._endpoint_probe
        if probe is None:
            return
        client = self._active()
        failure: Exception | None = None
        endpoint: object | None = None
        try:
            endpoint = await probe(client)
        except (BackendCapacity, BackendProtocolError, BackendUnavailable, StaleIncarnation):
            raise
        except Exception as error:  # noqa: BLE001 - closed discovery boundary
            failure = error
        if failure is not None:
            await self._invalidate(client)
            translate_client_error(failure)
        if self._client is not client:
            raise BackendUnavailable('Valkey pool changed during discovery')
        changed = self._endpoint is not None and endpoint != self._endpoint
        self._endpoint = endpoint
        if changed:
            await self._invalidate(client)

    async def run_operation(self, operation: Callable[[PinnedControlClient], Awaitable[_T]]) -> _T:
        """Check one result command's endpoint and physical delivery span."""
        before = self._role_epoch()
        await self.refresh_endpoint()
        # Initial discovery is admission, not a change of an existing endpoint.
        if self._role_epoch() != before and (
            before[1] is not None or self._pool_epoch != before[0]
        ):
            raise BackendUnavailable('Valkey role changed before operation')
        async with self.pin() as pinned:
            pool = self._active()
            initial = self._role_epoch(), pinned.connection_generation()
            cancelled = False
            try:
                return await operation(pinned)
            except CancelledError:
                cancelled = True
                raise
            finally:
                if not cancelled:
                    await self.refresh_endpoint()
                    if (self._role_epoch(), pinned.connection_generation()) != initial:
                        await self._invalidate(pool)
                        raise BackendUnavailable('Valkey role changed during operation')

    async def evalsha(self, sha: str, numkeys: int, *keys_and_args: ScriptArgument) -> object:
        """Execute one cached script inside an observed role delivery span."""
        return await self.run_operation(
            lambda client: client.evalsha(sha, numkeys, *keys_and_args)
        )

    async def exists(self, key: str) -> object:
        """Read key existence inside an observed role delivery span."""
        return await self.run_operation(lambda client: client.exists(key))

    async def get(self, key: str) -> object:
        """Read one raw value inside an observed role delivery span.

        Returns:
            The raw stored value, or `None` when the key is absent.
        """
        return await self.run_operation(lambda client: client.get(key))

    async def info(self, section: str) -> object:
        """Read server information inside an observed role delivery span."""
        return await self.run_operation(lambda client: client.info(section))

    async def script_load(self, script: bytes | str) -> object:
        """Load a script and return its raw server digest inside an observed role span."""
        return await self.run_operation(lambda client: client.script_load(script))


class _PinnedRoleClient:
    """Route a maintained single-connection client through its role ownership."""

    def __init__(self, owner: _RoleClient, pool: _RawClient, client: _RawClient) -> None:
        """Retain the pool identity for overlap-safe invalidation."""
        self._client = client
        self._owner = owner
        self._pool = pool
        self._finished = False

    def _finish(self) -> None:
        """Revoke this pin synchronously before release can suspend."""
        self._finished = True

    async def _invoke(self, *args: ScriptArgument) -> object:
        """Refuse detached pins and preserve maintained command cancellation handling."""
        if self._finished:
            raise BackendUnavailable('Valkey pin has left its scope')
        try:
            return await self._owner._invoke_on(
                self._pool,
                lambda: self._client.execute_command(*args),
            )
        except CancelledError:
            # A pinned reconnect occurs before maintained send/read guards and
            # bypasses pool acquisition. Settle only this connection on cancel.
            if self._client.connection is not None:
                await _disconnect_interrupted(self._client.connection)
            raise

    def connection_generation(self) -> int | None:
        """Sample this pin's latest role-unique connection generation without I/O.

        Returns:
            The role-wide generation recorded when this physical connection
            last connected, or `None` when it has no tracking record.
        """
        return self._owner._connection_generation(self._client)

    async def evalsha(self, sha: str, numkeys: int, *keys_and_args: ScriptArgument) -> object:
        """Execute one cached script on the borrowed connection."""
        return await self._invoke('EVALSHA', sha, numkeys, *keys_and_args)

    async def exists(self, key: str) -> object:
        """Read key existence on the borrowed connection."""
        return await self._invoke('EXISTS', key)

    async def get(self, key: str) -> object:
        """Read one raw string value on the borrowed connection.

        Returns:
            The raw stored value, or `None` when the key is absent.
        """
        return await self._invoke('GET', key)

    async def info(self, section: str) -> object:
        """Read one parsed information section on the borrowed connection."""
        return await self._invoke('INFO', section)

    async def script_load(self, script: bytes | str) -> object:
        """Load one script and return its raw server digest on the borrowed connection."""
        return await self._invoke('SCRIPT LOAD', script)


class _SentinelAPI(Protocol):
    """Type maintained Sentinel ownership and async primary discovery."""

    sentinels: list[_RawClient]

    def master_for(self, service_name: str, **kwargs: object) -> _RawClient:
        """Construct a lazy maintained primary client for one service."""
        ...


class _ConfiguredRole(_RoleClient):
    """Retain separate discovery resources for a configured primary role."""

    def __init__(
        self,
        factory: Callable[[], _RawClient],
        tracker: _ConnectionGeneration,
        role: ValkeyRoleSettings,
        discovery: list[_RawClient],
    ) -> None:
        """Own primary replacement and the role's stable discovery clients."""
        super().__init__(
            factory,
            tracker,
            _sentinel_endpoint if role.mode == 'sentinel' else None,
        )
        self._discovery = discovery
        self._active()

    @override
    async def close(self) -> None:
        resources = [super().close]
        discovery, self._discovery = self._discovery, []
        resources.extend(partial(_close_raw, client) for client in discovery)
        outcome = await _close_resources(tuple(resources))
        if outcome.cancellation is not None:
            raise outcome.cancellation
        if outcome.error is not None:
            translate_client_error(outcome.error)


class AsyncValkeyMetadataStore:
    """Own metadata-role control and generation operations without result access.

    Direct construction binds to the first operation's event loop. Terminal
    close joins admitted operations and all owned discovery resources.

    Operations follow the `AsyncCacheBackend` contracts for validation,
    generation fencing and normalized backend failures. The instance owns its
    supplied role resources. Failures use `BackendCapacity` for capacity
    rejection and `BackendProtocolError` for malformed inputs or replies, and
    `BackendUnavailable` for unavailable roles. Stale metadata can raise
    `StaleIncarnation`.

    All operations and close must run on the owning event loop. A different
    loop or a new operation after closing begins raises `RuntimeError`.
    Concurrent close calls join cleanup; later close calls do nothing.
    Cancellation joins owned work before returning.
    """

    __slots__ = (
        '_breaker',
        '_client',
        '_control',
        '_lifecycle',
        '_metadata_pressure_started',
        '_metrics',
        '_settings',
    )

    def __init__(
        self,
        client: _ValkeyClient,
        settings: CacheSettings,
        metrics: MetricsSink,
    ) -> None:
        """Own metadata validation, telemetry and terminal resource cleanup.

        Args:
            client: Dedicated metadata role whose resources this store owns.
            settings: Validated settings used for namespace and operation
                bounds.
            metrics: Prompt synchronous telemetry destination.
        """
        self._breaker = _CircuitBreaker(
            'metadata',
            threshold=settings.limits.circuit_failure_threshold,
            probe_seconds=settings.limits.circuit_probe_seconds,
            metrics=metrics,
        )
        self._client = client
        self._lifecycle = _Lifecycle(settings.limits.operation_timeout_seconds)
        self._control = AsyncValkeyControl(
            client,
            key_prefix=settings.l2.valkey.metadata.key_prefix,
            validation_seconds=settings.limits.incarnation_validation_seconds,
            metrics=metrics,
        )
        self._metadata_pressure_started: float | None = None
        self._metrics = metrics
        self._settings = settings

    def _increment(self, name: str, labels: Mapping[str, str]) -> None:
        """Emit one best-effort counter without affecting backend behavior."""
        with suppress(Exception):
            self._metrics.increment(name, labels)

    async def _change(self, changes: tuple[GenerationChange, ...]) -> None:
        """Perform one unguarded atomic generation change."""
        tags = tuple(change.tag for change in changes)

        async def operation(client: PinnedControlClient, incarnation: Incarnation) -> None:
            operands = metadata_operands(
                self._settings.l2.valkey.metadata.key_prefix,
                incarnation,
                tags,
                self._settings.limits.scoped_generation_retention_seconds,
            )
            reply = await execute_script(
                client,
                script=CHANGE_SCRIPT,
                sha=CHANGE_SCRIPT_SHA,
                keys=operands.keys,
                args=operands.args,
            )
            _replaced, absent = decode_change_reply(reply)
            for tag in tags:
                self._increment('cache_invalidation', {'tag_kind': tag.kind.value})
            if absent:
                self._increment(
                    'cache_scoped_generation',
                    {'outcome': 'absent', 'tag_kind': 'scoped'},
                )

        await self._control.run(operation)

    def _metadata_capacity_failure(self, fresh: bool) -> None:
        """Record one rejected metadata write and preserve probe state."""
        self._increment('cache_metadata_write_rejection', {'reason': 'noeviction'})
        if self._metadata_pressure_started is None:
            self._metadata_pressure_started = monotonic()
        if fresh:
            self._breaker.succeed()

    def _observe(
        self,
        name: str,
        value: float,
        labels: Mapping[str, str],
    ) -> None:
        """Emit one best-effort observation without affecting backend behavior."""
        with suppress(Exception):
            self._metrics.observe(name, value, labels)

    async def _guard(self, name: str, operation: Callable[[], Awaitable[_T]]) -> _T:
        """Run one metadata operation through its role breaker."""
        started_at = monotonic()
        fresh = self._breaker.admit()
        try:
            async with self._lifecycle._budget():
                if fresh:
                    await self._client.refresh()
                result = await operation()
        except BackendCapacity:
            self._metadata_capacity_failure(fresh)
            raise
        except StaleIncarnation:
            if fresh:
                self._breaker.succeed()
            raise
        except (BackendProtocolError, BackendUnavailable):
            self._increment(
                'backend_failure',
                {'backend_operation': name, 'outcome': 'failure'},
            )
            self._breaker.fail()
            raise
        except BaseException:
            if fresh:
                self._breaker.fail()
            raise
        self._breaker.succeed()
        self._observe(
            'backend_latency_seconds',
            monotonic() - started_at,
            {'backend_operation': name},
        )
        if self._metadata_pressure_started is not None:
            self._observe(
                'cache_metadata_recovery_seconds',
                monotonic() - self._metadata_pressure_started,
                {'outcome': 'recovered'},
            )
            self._metadata_pressure_started = None
        return result

    async def _resolve(self, tags: tuple[DependencyTag, ...]) -> GenerationVector:
        """Perform one unguarded metadata vector resolution."""

        async def operation(
            client: PinnedControlClient, incarnation: Incarnation
        ) -> GenerationVector:
            operands = metadata_operands(
                self._settings.l2.valkey.metadata.key_prefix,
                incarnation,
                tags,
                self._settings.limits.scoped_generation_retention_seconds,
            )
            reply = await execute_script(
                client,
                script=RESOLVE_SCRIPT,
                sha=RESOLVE_SCRIPT_SHA,
                keys=operands.keys,
                args=operands.args,
            )
            tokens = decode_resolve_reply(reply, expected_count=len(tags))
            return GenerationVector(incarnation.nonce, tuple(zip(tags, tokens, strict=True)))

        return await self._control.run(operation)

    async def change(self, changes: tuple[GenerationChange, ...]) -> None:
        """Replace all applicable generations in one metadata script.

        Args:
            changes: Ordered global and replace-if-materialized scoped changes.

        Raises:
            BackendProtocolError: The change tuple is malformed.
            StaleIncarnation: Control state changed or its counter requires
                rotation.
        """
        with self._lifecycle._operation():
            validated = generation_changes(changes)
            return await self._guard('change', lambda: self._change(validated))

    async def close(self) -> None:
        """Join admitted work before terminal metadata-role cleanup.

        Concurrent callers join the same terminal cleanup. Calls after
        completion do nothing, including after a cleanup failure. Caller
        cancellation is deferred until owned work and cleanup settle. Cleanup
        may outlast the ordinary operation budget and reports its first failure
        after all owned resources have been attempted.

        Raises:
            RuntimeError: The call uses another event loop or attempts to close
                from work admitted by this owner.
            BackendCapacity: A role close reports a normalized capacity
                failure.
            BackendProtocolError: A role close reports a malformed response.
            BackendUnavailable: A role close cannot complete.
        """
        await self._lifecycle._close((self._client.close,))

    async def flush_site(self, site_namespace: str, reason: str) -> None:
        """Replace one site's permanent generation through the metadata role.

        Args:
            site_namespace: Stable configured site namespace.
            reason: Short operator-facing reason code.
        """
        with self._lifecycle._operation():
            try:
                tag = DependencyTag.site(site_namespace)
            except (TypeError, ValueError):
                raise BackendProtocolError('site namespace is invalid') from None
            flush_reason(reason)
            await self.change((GenerationChange(tag),))
            self._increment('cache_site_flush', {'reason': metric_flush_reason(reason)})

    @classmethod
    async def from_settings(
        cls, settings: CacheSettings, environ: Mapping[str, object], metrics: MetricsSink
    ) -> Self:
        """Construct and own the configured metadata role and discovery resources.

        Args:
            settings: Shared settings with enabled Valkey L2 configuration.
            environ: Environment mapping containing the configured role
                passwords.
            metrics: Safe synchronous metrics sink.

        Returns:
            A metadata-only store bound to the current event loop. Network
            connection and primary validation are deferred to operations.
            Failed construction joins cleanup of partially acquired resources
            before propagating.

        Raises:
            TypeError: Constructor inputs are malformed.
            ValueError: L2 is disabled or a configured password is absent or
                empty.
            BackendCapacity: Resource construction reports capacity rejection.
            BackendProtocolError: Resource construction returns malformed
                state.
            BackendUnavailable: A configured resource cannot be constructed.
        """
        require_enabled_l2(settings)
        role = settings.l2.valkey.metadata
        client = await _create_client(role, settings, role_password(role, environ), metrics)
        try:
            return cls(client, settings, metrics)
        except BaseException as error:
            await _cleanup_construction((client.close,), error, metrics)
            raise

    async def resolve(self, tags: tuple[DependencyTag, ...]) -> GenerationVector:
        """Resolve one complete dependency vector in one metadata script.

        Args:
            tags: Ordered unique validated dependency tags.

        Returns:
            Every tag and token under the active incarnation.

        Raises:
            BackendProtocolError: The vector is malformed or exceeds its
                bounds.
            StaleIncarnation: Control state changed or its counter requires
                rotation.
        """
        with self._lifecycle._operation():
            validated = validated_resolve_tags(tags, self._settings)
            self._observe(
                'cache_dependency_vector_tags',
                float(len(validated)),
                {'layer': 'l2'},
            )
            self._observe(
                'cache_dependency_vector_bytes',
                float(KeyBuilder.worst_case_vector_bytes(validated)),
                {'layer': 'l2'},
            )
            vector = await self._guard('resolve', lambda: self._resolve(validated))
            self._increment('cache_vector_resolution', {'layer': 'l2'})
            return vector


class AsyncValkeyBackend:
    """Store shared results and metadata in separate Valkey roles.

    Direct construction binds to the first operation's event loop. Close is
    terminal, joins admitted operations and attempts both owned role closes.

    Operations follow the `AsyncCacheBackend` contracts for validation,
    generation fencing, result admission, and normalized backend failures. The
    instance owns its supplied role resources. Failures use `BackendCapacity`
    for capacity rejection and `BackendProtocolError` for malformed inputs or
    replies, and `BackendUnavailable` for unavailable roles. Stale metadata can
    raise `StaleIncarnation`.

    All operations and close must run on the owning event loop. A different
    loop or a new operation after closing begins raises `RuntimeError`.
    Concurrent close calls join cleanup; later close calls do nothing.
    Cancellation joins owned work before returning.
    """

    __slots__ = (
        '_lifecycle',
        '_metadata',
        '_metrics',
        '_result_breaker',
        '_result_client',
        '_settings',
    )

    def __init__(
        self,
        result_client: _ResultRole,
        metadata: _MetadataStore,
        settings: CacheSettings,
        metrics: MetricsSink,
    ) -> None:
        """Own separate result and metadata resources on their first loop.

        Args:
            result_client: Dedicated result role whose resources this backend
                owns.
            metadata: Metadata store whose close this backend owns.
            settings: Validated settings used for namespaces and operation
                bounds.
            metrics: Prompt synchronous telemetry destination.
        """
        self._lifecycle = _Lifecycle(settings.limits.operation_timeout_seconds)
        self._metadata = metadata
        self._metrics = metrics
        self._result_breaker = _CircuitBreaker(
            'result',
            threshold=settings.limits.circuit_failure_threshold,
            probe_seconds=settings.limits.circuit_probe_seconds,
            metrics=metrics,
        )
        self._result_client = result_client
        self._settings = settings

    async def _commit(self, lease: FillLease, payload: bytes, deadline: float) -> bool:
        """Perform one unguarded token-checked result commit."""
        try:
            return await execute_commit(
                self._result_client,
                prefix=self._settings.l2.valkey.result.key_prefix,
                lease=lease,
                payload=payload,
                deadline=deadline,
                clock=monotonic,
            )
        except BackendCapacity:
            return False

    def _increment(self, name: str, labels: Mapping[str, str]) -> None:
        """Emit one best-effort counter without affecting backend behavior."""
        with suppress(Exception):
            self._metrics.increment(name, labels)

    async def _lookup(self, key: str) -> LookupOutcome:
        """Perform one unguarded result lookup."""
        operands = lookup_operands(self._settings.l2.valkey.result.key_prefix, key)
        # Conservatively include discovery, command delivery, and reply settlement.
        # Pre-command or NOSCRIPT delay may cause a safe early miss.
        lookup_started_at = monotonic()
        reply = await execute_script(
            self._result_client,
            script=LOOKUP_SCRIPT,
            sha=LOOKUP_SCRIPT_SHA,
            keys=operands.keys,
            args=operands.args,
        )
        return interpret_lookup(
            reply, self._settings.limits.max_serialized_result_bytes, lookup_started_at, monotonic
        )

    def _observe(
        self,
        name: str,
        value: float,
        labels: Mapping[str, str],
    ) -> None:
        """Emit one best-effort observation without affecting backend behavior."""
        with suppress(Exception):
            self._metrics.observe(name, value, labels)

    async def _release(self, lease: FillLease) -> bool:
        """Perform one unguarded result-role lease release."""
        operands = release_operands(self._settings.l2.valkey.result.key_prefix, lease)
        reply = await execute_script(
            self._result_client,
            script=RELEASE_SCRIPT,
            sha=RELEASE_SCRIPT_SHA,
            keys=operands.keys,
            args=operands.args,
        )
        return decode_release_reply(reply)

    async def _reserve(self, key: str, ttl: int, token: str) -> FillLease | None:
        """Perform one unguarded result-role lease reservation."""
        operands = reserve_operands(self._settings.l2.valkey.result.key_prefix, key, ttl, token)
        reply = await execute_script(
            self._result_client,
            script=RESERVE_SCRIPT,
            sha=RESERVE_SCRIPT_SHA,
            keys=operands.keys,
            args=operands.args,
        )
        if not decode_reserve_reply(reply):
            return None
        return FillLease(key, token)

    async def _result_guard(self, name: str, operation: Callable[[], Awaitable[_T]]) -> _T:
        """Run one result operation through its independent role breaker."""
        started_at = monotonic()
        fresh = self._result_breaker.admit()
        try:
            async with self._lifecycle._budget():
                if fresh:
                    await self._result_client.refresh()
                result = await operation()
        except (BackendCapacity, StaleIncarnation):
            if fresh:
                self._result_breaker.succeed()
            raise
        except (BackendProtocolError, BackendUnavailable):
            self._increment(
                'backend_failure',
                {'backend_operation': name, 'outcome': 'failure'},
            )
            self._result_breaker.fail()
            raise
        except BaseException:
            if fresh:
                self._result_breaker.fail()
            raise
        self._result_breaker.succeed()
        self._observe(
            'backend_latency_seconds',
            monotonic() - started_at,
            {'backend_operation': name},
        )
        return result

    async def change(self, changes: tuple[GenerationChange, ...]) -> None:
        """Replace all applicable generations in caller order.

        Args:
            changes: Ordered generation replacements to acknowledge together.
        """
        with self._lifecycle._operation():
            await self._metadata.change(changes)

    async def close(self) -> None:
        """Join admitted work and attempt both owned role closes exactly once.

        Concurrent callers join the same terminal cleanup. Calls after
        completion do nothing, including after a cleanup failure. Caller
        cancellation is deferred until owned work and cleanup settle. Cleanup
        may outlast the ordinary operation budget and reports its first failure
        after all owned resources have been attempted.

        Raises:
            RuntimeError: The call uses another event loop or attempts to close
                from work admitted by this owner.
            BackendCapacity: A role close reports a normalized capacity
                failure.
            BackendProtocolError: A role close reports a malformed response.
            BackendUnavailable: A role close cannot complete.
        """
        await self._lifecycle._close((self._result_client.close, self._metadata.close))

    async def flush_site(self, site_namespace: str, reason: str) -> None:
        """Replace one site's permanent site generation.

        Args:
            site_namespace: Stable configured site namespace.
            reason: Short operator-facing reason code.
        """
        with self._lifecycle._operation():
            await self._metadata.flush_site(site_namespace, reason)

    @classmethod
    async def from_settings(
        cls, settings: CacheSettings, environ: Mapping[str, object], metrics: MetricsSink
    ) -> Self:
        """Construct and own both configured Valkey roles and discovery resources.

        Args:
            settings: Shared settings with enabled Valkey L2 configuration.
            environ: Environment mapping containing the configured role
                passwords.
            metrics: Safe synchronous metrics sink.

        Returns:
            A two-role backend bound to the current event loop. Network
            connection and primary validation are deferred to operations.
            Failed construction joins cleanup of partially acquired resources
            before propagating.

        Raises:
            TypeError: Constructor inputs are malformed.
            ValueError: L2 is disabled or a configured password is absent or
                empty.
            BackendCapacity: Resource construction reports capacity rejection.
            BackendProtocolError: Resource construction returns malformed
                state.
            BackendUnavailable: A configured resource cannot be constructed.
        """
        require_enabled_l2(settings)
        metadata_role = settings.l2.valkey.metadata
        result_role = settings.l2.valkey.result
        metadata_password = role_password(metadata_role, environ)
        result_password = role_password(result_role, environ)
        resources: list[Callable[[], Awaitable[None]]] = []
        try:
            metadata_client = await _create_client(
                metadata_role, settings, metadata_password, metrics
            )
            resources.append(metadata_client.close)
            result_client = await _create_client(result_role, settings, result_password, metrics)
            resources.append(result_client.close)
            metadata = AsyncValkeyMetadataStore(metadata_client, settings, metrics)
            return cls(result_client, metadata, settings, metrics)
        except BaseException as error:
            await _cleanup_construction(tuple(resources), error, metrics)
            raise

    async def lookup(self, key: str) -> CacheHit | None:
        """Read valid encoded bytes and their server-authoritative remaining age.

        Args:
            key: Opaque generation-qualified result key.

        Returns:
            An isolated hit, or `None` for absence, expiry, or malformed data.
        """
        with self._lifecycle._operation():
            normalized = opaque_key(key)
            outcome = finalize_lookup(
                await self._result_guard('lookup', lambda: self._lookup(normalized)), monotonic
            )
            if outcome.reason is not None:
                self._increment('cache_miss', {'layer': 'l2', 'reason': outcome.reason})
            return outcome.hit

    async def release(self, lease: FillLease) -> bool:
        """Remove a result-role fill lease only when its token remains current.

        Args:
            lease: Exact key and random token returned by `reserve`.

        Returns:
            Whether the current lease was removed.
        """
        with self._lifecycle._operation():
            validated = fill_lease(lease)
            return await self._result_guard('release', lambda: self._release(validated))

    async def commit(self, lease: FillLease, payload: bytes, remaining_seconds: float) -> bool:
        """Store bytes only while the exact result-role lease remains current.

        Oversized payloads release the matching lease and return false. A
        script capacity-rejection reply, an obsolete lease, or exhaustion of
        the entry-anchored lifetime also returns false.

        Args:
            lease: Exact key and random token returned by `reserve`.
            payload: Closed-codec result bytes.
            remaining_seconds: Authoritative remaining lifetime in seconds, at
                least one millisecond after flooring and at most 2**63 - 1
                milliseconds.

        Returns:
            Whether the result was admitted under the current lease.

        Raises:
            BackendProtocolError: The lifetime cannot be represented as a
                positive signed 64-bit millisecond duration.
        """
        entered_at = monotonic()
        with self._lifecycle._operation():
            validated_lease = fill_lease(lease)
            validated_payload = encoded_payload(payload)
            milliseconds(remaining_seconds, 'remaining lifetime')
            deadline = entered_at + remaining_seconds
            if len(validated_payload) > self._settings.limits.max_serialized_result_bytes:
                await self.release(validated_lease)
                return False
            committed = await self._result_guard(
                'commit',
                lambda: self._commit(validated_lease, validated_payload, deadline),
            )
            self._increment(
                'cache_commit',
                {'layer': 'l2', 'outcome': 'committed' if committed else 'rejected'},
            )
            if committed:
                self._observe(
                    'cache_serialized_bytes',
                    float(len(validated_payload)),
                    {'layer': 'l2', 'outcome': 'admitted'},
                )
            else:
                self._increment('cache_lease_rejection', {'layer': 'l2'})
            return committed

    async def reserve(self, key: str, lease_seconds: float) -> FillLease | None:
        """Reserve temporary fill ownership with a random result-role token.

        Args:
            key: Opaque generation-qualified result key.
            lease_seconds: Maximum ownership duration in seconds, floored to
                between 1 and 2**63 - 1 milliseconds.

        Returns:
            A fresh lease, or `None` while another lease remains current.

        Raises:
            BackendProtocolError: The lifetime cannot be represented as a
                positive signed 64-bit millisecond duration.
        """
        with self._lifecycle._operation():
            normalized = opaque_key(key)
            ttl = milliseconds(lease_seconds, 'lease duration')
            try:
                token = token_hex(32)
            except Exception:  # noqa: BLE001 - closed entropy boundary
                raise BackendUnavailable('Valkey lease entropy is unavailable') from None
            lease = await self._result_guard(
                'reserve', lambda: self._reserve(normalized, ttl, token)
            )
            if lease is None:
                self._increment('cache_lease_rejection', {'layer': 'l2'})
            return lease

    async def resolve(self, tags: tuple[DependencyTag, ...]) -> GenerationVector:
        """Resolve one complete dependency vector in one metadata phase.

        Args:
            tags: Ordered unique dependency tags.

        Returns:
            Every requested tag and current token.
        """
        with self._lifecycle._operation():
            return await self._metadata.resolve(tags)


async def _close_pin(client: _RawClient, cancellation: CancelledError | None) -> None:
    """Join maintained pin release before propagating cancellation or failure."""
    connection = client.connection
    outcome = await joined_cleanup(lambda: client.aclose(close_connection_pool=False))
    # aclose clears connection before release, so the outer owner settles only
    # when the owned cleanup failed. After a completed release, another pin may
    # already have re-borrowed this connection.
    client.connection = None
    if connection is not None and outcome.error is not None:
        await client.connection_pool._settle_release(connection)
    if cancellation is not None:
        raise cancellation
    if outcome.cancellation is not None:
        raise outcome.cancellation
    if outcome.error is not None:
        translate_client_error(outcome.error)


async def _close_raw(client: _RawClient) -> None:
    """Close one discovery pool and settle interrupted maintained cleanup."""
    outcome = await joined_cleanup(lambda: client.aclose(close_connection_pool=True))
    if outcome.error is not None:
        await client.connection_pool._settle_disconnect()
    if outcome.cancellation is not None:
        raise outcome.cancellation
    if outcome.error is not None:
        translate_client_error(outcome.error)


def _construct_resource(factory: Callable[[], _T]) -> _T:
    """Redact ordinary third-party allocation failures on initial and replacement pools."""
    failure: Exception | None = None
    try:
        return factory()
    except Exception as error:  # noqa: BLE001 - redact the external resource construction boundary
        failure = error
    translate_client_error(failure)
    raise BackendUnavailable('Valkey resource construction failed')


async def _create_client(
    role: ValkeyRoleSettings,
    settings: CacheSettings,
    password: str,
    metrics: MetricsSink,
) -> _ConfiguredRole:
    """Construct one lazy role and own each separate Sentinel discovery client."""
    tracker = _ConnectionGeneration(prove_primary=role.mode != 'standalone')
    service_name = role.service_name
    if role.mode == 'sentinel' and service_name is None:
        raise BackendProtocolError('Valkey Sentinel service name is absent')
    discovery: list[_RawClient] = []
    kwargs: dict[str, object] = {
        'db': role.database,
        'username': role.username,
        'password': password,
        'socket_connect_timeout': settings.limits.connect_timeout_seconds,
        'socket_timeout': settings.limits.operation_timeout_seconds,
        'decode_responses': False,
    }
    if role.tls:
        kwargs['ssl_check_hostname'] = True
    try:
        sentinel: _SentinelAPI | None = None
        if role.mode == 'sentinel':
            sentinel = _construct_resource(
                lambda: cast('_SentinelAPI', cast('object', Sentinel((), min_other_sentinels=2)))
            )
            discovery_kwargs = {key: value for key, value in kwargs.items() if key != 'db'}
            for address in role.sentinels:
                host, port = parse_sentinel_address(address)
                pool = _construct_resource(
                    partial(
                        _ConnectionPool,
                        connection_class=_SSLConnection if role.tls else _Connection,
                        max_connections=None,
                        host=host,
                        port=port,
                        **discovery_kwargs,
                    )
                )
                discovery.append(
                    _construct_resource(
                        lambda pool=pool: cast(
                            '_RawClient', cast('object', Valkey.from_pool(pool))
                        )
                    )
                )
            sentinel.sentinels = discovery

        def create() -> _RawClient:
            if sentinel is not None:
                return sentinel.master_for(
                    cast('str', service_name),
                    connection_pool_class=_SentinelConnectionPool,
                    ssl=role.tls,
                    valkey_connect_func=tracker._initialize,
                    **kwargs,
                )
            pool = _ConnectionPool(
                connection_class=_SSLConnection if role.tls else _Connection,
                max_connections=None,
                host=role.host,
                port=role.port,
                valkey_connect_func=tracker._initialize,
                **kwargs,
            )
            return cast('_RawClient', cast('object', Valkey.from_pool(pool)))

        result = _ConfiguredRole(partial(_construct_resource, create), tracker, role, discovery)
    except BaseException as error:
        resources = tuple(partial(_close_raw, client) for client in discovery)
        await _cleanup_construction(resources, error, metrics)
        raise
    else:
        return result


async def _disconnect_interrupted(connection: AbstractConnection) -> None:
    """Settle the exact interrupted connection before its owner propagates cancellation."""
    await joined_cleanup(lambda: connection.disconnect(nowait=True))
    with suppress(Exception):
        await connection.disconnect(nowait=True)


async def _prove_connection(connection: AbstractConnection) -> None:
    """Prove a new socket reaches a writable primary, closing it without a graceful wait."""
    try:
        await connection.send_command('INFO', 'replication')
        require_primary_reply(await connection.read_response())
    except CancelledError:
        # Acquisition and pinned-reconnect handlers already settle interrupted sockets.
        raise
    except BaseException:
        with suppress(Exception):
            await connection.disconnect(nowait=True)
        raise


async def _sentinel_endpoint(client: _RawClient) -> object:
    """Discover the current primary through the maintained owned Sentinel pool."""
    pool = client.connection_pool
    if not isinstance(pool, _SentinelConnectionPool):
        raise BackendProtocolError('Valkey Sentinel pool is invalid')
    return await pool.get_master_address()
