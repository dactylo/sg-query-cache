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
    Timeout,
    create_task,
    gather,
    get_running_loop,
    timeout,
    timeout_at,
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

try:
    import valkey
    from valkey.asyncio import Connection, Sentinel, Valkey
    from valkey.asyncio.connection import AbstractConnection, SSLConnection
    from valkey.asyncio.connection import ConnectionPool as MaintainedConnectionPool
    from valkey.asyncio.retry import Retry
    from valkey.asyncio.sentinel import SentinelConnectionPool as MaintainedSentinelPool
    from valkey.asyncio.sentinel import SentinelManagedConnection
    from valkey.backoff import NoBackoff
    from valkey.exceptions import ConnectionError as ValkeyConnectionError
    from valkey.exceptions import ReadOnlyError
    from valkey.exceptions import TimeoutError as ValkeyTimeoutError
except ModuleNotFoundError as error:
    if error.name != 'valkey':
        raise
    raise ImportError(
        "sg_query_cache.valkey_async requires the 'valkey' extra: "
        "pip install 'sg-query-cache[valkey]'"
    ) from error
else:
    from sg_query_cache._async_support import (
        PUBLISHED_DEADLINE,
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
        BACKEND_ERRORS,
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
        client_failure_class,
        translate_client_error,
    )
del valkey

_T = TypeVar('_T')


__all__ = ['AsyncValkeyBackend', 'AsyncValkeyMetadataStore']


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

    async def _close_all(self) -> CleanupOutcome[None]:
        """Join operations and settle every explicitly retained close obligation."""
        idle = await joined_cleanup(self._admission.wait_idle)
        if idle.error is not None:
            # Independent cancellation is failed cleanup, not proof of idleness.
            await self._admission.wait_idle()
        deadline = get_running_loop().time() + self._timeout_seconds
        record = PUBLISHED_DEADLINE.get()
        if record is not None:
            deadline = min(deadline, record.deadline)
        outcome = await _close_resources(tuple(self._pending), deadline=deadline)
        self._pending.clear()
        self._admission.finish_close()
        failure = idle.error or outcome.error
        if isinstance(failure, TimeoutError):
            failure = BackendUnavailable('Valkey close deadline expired', failure_class='timeout')
        return CleanupOutcome(
            error=failure,
            cancellation=idle.cancellation or outcome.cancellation,
        )

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

    @asynccontextmanager
    async def budget(self) -> AsyncGenerator[Timeout]:
        """Apply the whole operation's cooperative budget.

        Yields:
            The owning timer, whose expiry remains available during settlement.

        Raises:
            BackendUnavailable: The operation exceeds its deadline.
        """
        deadline = get_running_loop().time() + self._timeout_seconds
        record = PUBLISHED_DEADLINE.get()
        if record is not None:
            deadline = min(deadline, record.deadline)
        timer = timeout_at(deadline)
        try:
            async with timer:
                try:
                    yield timer
                finally:
                    if timer.expired() and record is not None:
                        record.expired = True
        except TimeoutError:
            raise BackendUnavailable(
                'Valkey operation deadline expired', failure_class='timeout'
            ) from None

    async def close(self, resources: tuple[Callable[[], Awaitable[None]], ...]) -> None:
        """Join shared cleanup and synchronously claim any pre-entry settlement.

        Args:
            resources: Owned cleanup callbacks, attempted in order by the first
                closer.

        Raises:
            RuntimeError: Called by an admitted operation or from another event
                loop.
            BackendUnavailable: Cleanup exceeds its deadline or is interrupted.
            Exception: A retained cleanup callback fails.
            CancelledError: The caller is cancelled; shared cleanup is joined
                first.
        """
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

    @contextmanager
    def operation(self) -> Generator[None]:
        """Admit the whole operation before its first suspension.

        Yields:
            Control while the operation is admitted and terminal close must
            wait.

        Raises:
            RuntimeError: Closing, closed, outside a task, or on another event
                loop.
        """
        with self._admission.operation():
            yield


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
        labels = {'backend_operation': 'construction_close', 'outcome': 'failure'}
        if (
            isinstance(outcome.error, BackendUnavailable)
            and outcome.error.failure_class is not None
        ):
            labels['failure_class'] = outcome.error.failure_class
        with suppress(Exception):
            metrics.increment(
                'backend_failure',
                labels,
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

    def for_connection(self, connection: AbstractConnection | None) -> int | None:
        """Sample one connection's latest role-unique generation without I/O.

        Returns:
            The tracked generation, or `None` for an absent or untracked
            connection.

        Raises:
            RuntimeError: No event loop is running, or the tracker belongs to
                another loop.
        """
        self._affinity.check()
        return None if connection is None else self._by_connection.get(connection)

    @property
    def generation(self) -> int:
        """Sample the aggregate diagnostic generation without I/O.

        Raises:
            RuntimeError: No event loop is running, or the tracker belongs to
                another loop.
        """
        self._affinity.check()
        return self._generation_count

    async def initialize(self, connection: AbstractConnection) -> None:
        """Complete setup, prove HA primaries, and register a weak reconnect callback.

        Args:
            connection: Newly connected socket whose maintained setup must
                finish.

        Raises:
            RuntimeError: No event loop is running, or the tracker belongs to
                another loop.
            ValkeyConnectionError: Connection setup fails, or the primary proof
                is malformed or reports a non-primary role.
            ValkeyTimeoutError: Connection setup times out.
        """
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
        """Write a packed command while preserving caller cancellation.

        Args:
            command: Encoded protocol data, text to encode, or ordered byte
                chunks.
            check_health: Whether to check an already-connected socket before
                writing.

        Raises:
            ValkeyTimeoutError: Connecting, checking health, or writing times
                out.
            ValkeyConnectionError: The connection cannot be established or
                used.
        """
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
        self.borrow_tokens: WeakKeyDictionary[AbstractConnection, object] = WeakKeyDictionary()

    @override
    async def disconnect(self, inuse_connections: bool = True) -> None:
        """Attempt every selected socket close before propagating a failure.

        Args:
            inuse_connections: Include borrowed connections as well as idle
                ones.

        Raises:
            Exception: A selected connection fails to disconnect.
        """
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
        """Prepare a usable connection and close it if setup is cancelled.

        Raises:
            ValkeyConnectionError: The connection cannot be established or has
                unread data.
            ValkeyTimeoutError: Connection setup or readiness checking times
                out.
        """
        try:
            await super().ensure_connection(connection)
        except CancelledError:
            # The maintained get_connection handler releases this exact borrow
            # after we close any socket established before setup was interrupted.
            await _disconnect_interrupted(connection)
            raise

    @override
    def get_available_connection(self) -> AbstractConnection:
        """Borrow an idle or new connection and assign its ownership token.

        Raises:
            ValkeyConnectionError: The pool has reached its connection limit.
        """
        connection = super().get_available_connection()
        self.borrow_tokens[connection] = object()
        return connection

    async def settle_disconnect(self) -> None:
        """Attempt every detached socket after independently interrupted cleanup."""
        for connection in tuple(chain(self._available_connections, self._in_use_connections)):
            # nowait disconnect uses the maintained socket-close path without
            # suspending for shutdown. Preserve the original cleanup outcome.
            with suppress(Exception):
                await connection.disconnect(nowait=True)

    async def settle_release(self, connection: AbstractConnection, token: object | None) -> None:
        """Return a still-owned pin after interrupted release through the maintained API.

        Args:
            connection: Borrowed connection to release.
            token: Ownership token captured before cleanup; an absent or stale
                token leaves the connection untouched.
        """
        if (
            token is not None
            and self.borrow_tokens.get(connection) is token
            and connection in self._in_use_connections
        ):
            # Maintained release performs only synchronous set/list updates.
            await super().release(connection)


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
        """Release a backend pin or its owned pool during joined cleanup.

        Arguments, results and client errors are those of
        `valkey.asyncio.Valkey.aclose`.
        """
        ...

    def client(self) -> Self:
        """Create a lazy single-connection child for a backend operation.

        Arguments, results and client errors are those of
        `valkey.asyncio.Valkey.client`.

        Returns:
            A child sharing the owning pool, retained only for the backend
            operation and closed when that operation finishes.
        """
        ...

    async def execute_command(self, *args: ScriptArgument) -> object:
        """Dispatch a backend command through maintained send and read handling.

        Arguments, results and client errors are those of
        `valkey.asyncio.Valkey.execute_command`.
        """
        ...

    async def initialize(self) -> Self:
        """Acquire the physical connection before yielding a backend pin.

        Arguments, results and client errors are those of
        `valkey.asyncio.Valkey.initialize`.
        """
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
            await client.connection_pool.settle_disconnect()
        if outcome.cancellation is not None:
            raise outcome.cancellation
        if outcome.error is not None:
            translate_client_error(outcome.error)

    def _role_epoch(self) -> tuple[int, object | None]:
        """Sample pool and endpoint state without the aggregate diagnostic counter."""
        self._affinity.check()
        return self._pool_epoch, self._endpoint

    async def close(self) -> None:
        """Make the role terminal and join cleanup of its detached pool.

        New operations remain rejected even when cleanup fails. Repeated close
        calls do not construct a pool.

        Pool release uses `valkey.asyncio.Valkey.aclose`, which owns the
        upstream release contract. This boundary normalizes client failures.

        Raises:
            BackendCapacity: Pool cleanup is rejected for capacity.
            BackendProtocolError: Pool cleanup violates the backend protocol.
            BackendUnavailable: Pool cleanup cannot complete reliably.
            RuntimeError: The role belongs to another event loop.
            CancelledError: The caller cancels close; cleanup is joined before
                propagation.
        """
        self._affinity.check()
        self._closed = True
        await self._invalidate()

    def connection_generation(self, client: _RawClient) -> int | None:
        """Sample a borrowed connection's latest role-unique generation without I/O.

        Returns:
            The tracked generation, or `None` for an absent or untracked
            connection.

        Raises:
            RuntimeError: No event loop is running, or the tracker belongs to
                another loop.
        """
        return self._connections.for_connection(client.connection)

    def epoch(self) -> RoleEpoch:
        """Sample the immutable pool, connection, and endpoint epoch without I/O."""
        self._affinity.check()
        return self._pool_epoch, self._connections.generation, self._endpoint

    async def invoke_on(self, pool: _RawClient, operation: Callable[[], Awaitable[_T]]) -> _T:
        """Run once against a current pool and discard reconnect-class failures.

        Args:
            pool: Pool retained by the caller's pin; it must still be current.
            operation: Command callback executed once against that pool.

        Returns:
            The callback's result.

        Raises:
            RuntimeError: No event loop is running, or the role belongs to
                another loop.
            BackendUnavailable: The pool was discarded or the role has closed.
            Exception: The command or subsequent pool cleanup fails.
            CancelledError: Command execution or cleanup is cancelled.
        """
        self._affinity.check()
        if self._closed or self._client is not pool:
            raise BackendUnavailable('Valkey pin belongs to a discarded pool')
        try:
            return await operation()
        except (ReadOnlyError, ValkeyConnectionError, ValkeyTimeoutError):
            await self._invalidate(pool)
            raise

    @asynccontextmanager
    async def pin(self) -> AsyncGenerator['_PinnedRoleClient']:
        """Borrow one physical connection for a complete backend operation.

        Borrowing follows `valkey.asyncio.Valkey.client`; release follows
        `valkey.asyncio.Valkey.aclose`. Those methods own their upstream
        contracts.

        Join release before propagating cancellation.

        Yields:
            A command wrapper valid only inside this context. Its borrowed
            child is released on exit.

        Raises:
            BackendCapacity: Valkey rejects required discovery, setup, or
                cleanup for capacity.
            BackendProtocolError: Required discovery, setup, or cleanup
                violates the backend protocol.
            BackendUnavailable: The role changes or its connection state cannot
                be established or settled.
            RuntimeError: The role belongs to another event loop or required
                pool access finds it closed.
            CancelledError: The caller cancels the operation; connection
                cleanup is joined before propagation.
        """
        pool = self._active()
        pinned = pool.client()
        wrapped = _PinnedRoleClient(self, pool, pinned)
        cancellation: CancelledError | None = None
        try:
            failure: Exception | None = None
            try:
                await pinned.initialize()
            except Exception as error:  # noqa: BLE001 - closed external-client boundary
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
            wrapped.finish()
            if self._client is not pool and pinned.connection is not None:
                with suppress(Exception):
                    await pinned.connection.disconnect(nowait=True)
            await _close_pin(pinned, cancellation)

    async def refresh(self) -> None:
        """Discard the current pool and construct its lazy replacement.

        The replacement is constructed after detachment; it is not a borrowed
        pin.

        Pool release uses `valkey.asyncio.Valkey.aclose`, which owns the
        upstream release contract. This boundary normalizes client failures.

        Raises:
            BackendCapacity: Valkey rejects required discovery, setup, or
                cleanup for capacity.
            BackendProtocolError: Required discovery, setup, or cleanup
                violates the backend protocol.
            BackendUnavailable: The role changes or its connection state cannot
                be established or settled.
            RuntimeError: The role belongs to another event loop or required
                pool access finds it closed.
            CancelledError: The caller cancels the operation; connection
                cleanup is joined before propagation.
        """
        await self._invalidate()
        self._active()

    async def refresh_endpoint(self) -> None:
        """Refresh primary discovery and discard a pool whose endpoint changed.

        Do nothing when the role has no endpoint probe.

        Discovery arguments, results and upstream errors are owned by
        `valkey.asyncio.sentinel.SentinelConnectionPool.get_master_address`;
        this boundary normalizes discovery failures.

        Raises:
            BackendCapacity: Valkey rejects required discovery, setup, or
                cleanup for capacity.
            BackendProtocolError: Required discovery, setup, or cleanup
                violates the backend protocol.
            BackendUnavailable: The role changes or its connection state cannot
                be established or settled.
            RuntimeError: The role belongs to another event loop or required
                pool access finds it closed.
            CancelledError: The caller cancels the operation; connection
                cleanup is joined before propagation.
        """
        self._affinity.check()
        probe = self._endpoint_probe
        if probe is None:
            return
        client = self._active()
        failure: Exception | None = None
        endpoint: object | None = None
        try:
            endpoint = await probe(client)
        except BACKEND_ERRORS:
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
        """Run result work within one verified physical-connection span.

        Borrowing uses `valkey.asyncio.Valkey.client`; discovery uses
        `valkey.asyncio.sentinel.SentinelConnectionPool.get_master_address`.
        Those methods own their upstream contracts. Callback command contracts
        belong to the corresponding `valkey.asyncio.Valkey` methods.

        Callback failures propagate after successful settlement; settlement
        failures take precedence. Cancellation skips settlement and joins pin
        release.

        Args:
            operation: Callback receiving a pinned client valid only for this
                call. It may issue multiple commands. Await its result before
                settlement.

        Returns:
            The callback result after endpoint and connection state settle
            unchanged.

        Raises:
            BackendCapacity: Valkey rejects required discovery, setup, or
                cleanup for capacity.
            BackendProtocolError: Required discovery, setup, or cleanup
                violates the backend protocol.
            BackendUnavailable: The role changes or its connection state cannot
                be established or settled.
            Exception: The callback fails and role settlement succeeds.
            RuntimeError: The role belongs to another event loop or required
                pool access finds it closed.
            CancelledError: The caller cancels the operation; connection
                cleanup is joined before propagation.
        """
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
            outcome: tuple[_T, ...] = ()
            operation_failure: BaseException | None = None
            try:
                outcome = (await operation(pinned),)
            except BaseException as error:
                if isinstance(error, CancelledError):
                    raise
                operation_failure = error
            try:
                await self.refresh_endpoint()
            except BaseException:
                del operation_failure
                raise
            if (self._role_epoch(), pinned.connection_generation()) != initial:
                failure_class = client_failure_class(
                    operation_failure if isinstance(operation_failure, Exception) else None
                )
                del operation_failure
                await self._invalidate(pool)
                raise BackendUnavailable(
                    'Valkey role changed during operation', failure_class=failure_class
                ) from None
            if operation_failure is not None:
                raise operation_failure
            if not outcome:
                raise BackendUnavailable('Valkey operation did not return an outcome')
            return outcome[0]

    async def evalsha(self, sha: str, numkeys: int, *keys_and_args: ScriptArgument) -> object:
        """Execute a cached backend script within a verified role span.

        Arguments, results and command errors are those of
        `valkey.asyncio.Valkey.evalsha`.

        Discovery, connection acquisition and final settlement follow
        `run_operation`;
        reject a result if the role or physical connection changes.

        Raises:
            BackendCapacity: Valkey rejects required discovery, setup, or
                cleanup for capacity.
            BackendProtocolError: Required discovery, setup, or cleanup
                violates the backend protocol.
            BackendUnavailable: The role changes or its connection state cannot
                be established or settled.
            RuntimeError: The role belongs to another event loop or required
                pool access finds it closed.
            CancelledError: The caller cancels the operation; connection
                cleanup is joined before propagation.
        """
        return await self.run_operation(
            lambda client: client.evalsha(sha, numkeys, *keys_and_args)
        )

    async def exists(self, key: str) -> object:
        """Check control-key presence within a verified role span.

        Arguments, results and command errors are those of
        `valkey.asyncio.Valkey.exists`.

        Discovery, connection acquisition and final settlement follow
        `run_operation`;
        reject a result if the role or physical connection changes.

        Raises:
            BackendCapacity: Valkey rejects required discovery, setup, or
                cleanup for capacity.
            BackendProtocolError: Required discovery, setup, or cleanup
                violates the backend protocol.
            BackendUnavailable: The role changes or its connection state cannot
                be established or settled.
            RuntimeError: The role belongs to another event loop or required
                pool access finds it closed.
            CancelledError: The caller cancels the operation; connection
                cleanup is joined before propagation.
        """
        return await self.run_operation(lambda client: client.exists(key))

    async def get(self, key: str) -> object:
        """Read the control record within a verified role span.

        Arguments, results and command errors are those of
        `valkey.asyncio.Valkey.get`.

        Discovery, connection acquisition and final settlement follow
        `run_operation`;
        reject a result if the role or physical connection changes.

        Raises:
            BackendCapacity: Valkey rejects required discovery, setup, or
                cleanup for capacity.
            BackendProtocolError: Required discovery, setup, or cleanup
                violates the backend protocol.
            BackendUnavailable: The role changes or its connection state cannot
                be established or settled.
            RuntimeError: The role belongs to another event loop or required
                pool access finds it closed.
            CancelledError: The caller cancels the operation; connection
                cleanup is joined before propagation.
        """
        return await self.run_operation(lambda client: client.get(key))

    async def info(self, section: str) -> object:
        """Read replication information within a verified role span.

        Arguments, results and command errors are those of
        `valkey.asyncio.Valkey.info`.

        Discovery, connection acquisition and final settlement follow
        `run_operation`;
        reject a result if the role or physical connection changes.

        Raises:
            BackendCapacity: Valkey rejects required discovery, setup, or
                cleanup for capacity.
            BackendProtocolError: Required discovery, setup, or cleanup
                violates the backend protocol.
            BackendUnavailable: The role changes or its connection state cannot
                be established or settled.
            RuntimeError: The role belongs to another event loop or required
                pool access finds it closed.
            CancelledError: The caller cancels the operation; connection
                cleanup is joined before propagation.
        """
        return await self.run_operation(lambda client: client.info(section))

    async def script_load(self, script: bytes | str) -> object:
        """Load a backend script within a verified role span.

        Arguments, results and command errors are those of
        `valkey.asyncio.Valkey.script_load`.

        Discovery, connection acquisition and final settlement follow
        `run_operation`;
        reject a result if the role or physical connection changes.

        Raises:
            BackendCapacity: Valkey rejects required discovery, setup, or
                cleanup for capacity.
            BackendProtocolError: Required discovery, setup, or cleanup
                violates the backend protocol.
            BackendUnavailable: The role changes or its connection state cannot
                be established or settled.
            RuntimeError: The role belongs to another event loop or required
                pool access finds it closed.
            CancelledError: The caller cancels the operation; connection
                cleanup is joined before propagation.
        """
        return await self.run_operation(lambda client: client.script_load(script))


class _PinnedRoleClient:
    """Route a maintained single-connection client through its role ownership."""

    def __init__(self, owner: _RoleClient, pool: _RawClient, client: _RawClient) -> None:
        """Retain the pool identity for overlap-safe invalidation."""
        self._client = client
        self._owner = owner
        self._pool = pool
        self._finished = False

    async def _invoke(self, *args: ScriptArgument) -> object:
        """Refuse detached pins and preserve maintained command cancellation handling."""
        if self._finished:
            raise BackendUnavailable('Valkey pin has left its scope')
        try:
            return await self._owner.invoke_on(
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
        return self._owner.connection_generation(self._client)

    async def evalsha(self, sha: str, numkeys: int, *keys_and_args: ScriptArgument) -> object:
        """Execute a cached backend script on the borrowed connection.

        Arguments, results and command errors are those of
        `valkey.asyncio.Valkey.evalsha`.

        Reconnect-class command failures invalidate the owning role pool.

        Raises:
            BackendCapacity: Cleanup after a reconnect-class failure is
                rejected for capacity.
            BackendProtocolError: Cleanup after a reconnect-class failure
                violates the backend protocol.
            BackendUnavailable: The pin has left its scope, its pool was
                discarded, the role closed, or cleanup failed.
            RuntimeError: The role belongs to another event loop.
            CancelledError: The caller cancels the command; the interrupted
                connection is settled before propagation.
        """
        return await self._invoke('EVALSHA', sha, numkeys, *keys_and_args)

    async def exists(self, key: str) -> object:
        """Check control-key presence on the borrowed connection.

        Arguments, results and command errors are those of
        `valkey.asyncio.Valkey.exists`.

        Reconnect-class command failures invalidate the owning role pool.

        Raises:
            BackendCapacity: Cleanup after a reconnect-class failure is
                rejected for capacity.
            BackendProtocolError: Cleanup after a reconnect-class failure
                violates the backend protocol.
            BackendUnavailable: The pin has left its scope, its pool was
                discarded, the role closed, or cleanup failed.
            RuntimeError: The role belongs to another event loop.
            CancelledError: The caller cancels the command; the interrupted
                connection is settled before propagation.
        """
        return await self._invoke('EXISTS', key)

    def finish(self) -> None:
        """Revoke this pin synchronously before release can suspend."""
        self._finished = True

    async def get(self, key: str) -> object:
        """Read the control record on the borrowed connection.

        Arguments, results and command errors are those of
        `valkey.asyncio.Valkey.get`.

        Reconnect-class command failures invalidate the owning role pool.

        Raises:
            BackendCapacity: Cleanup after a reconnect-class failure is
                rejected for capacity.
            BackendProtocolError: Cleanup after a reconnect-class failure
                violates the backend protocol.
            BackendUnavailable: The pin has left its scope, its pool was
                discarded, the role closed, or cleanup failed.
            RuntimeError: The role belongs to another event loop.
            CancelledError: The caller cancels the command; the interrupted
                connection is settled before propagation.
        """
        return await self._invoke('GET', key)

    async def info(self, section: str) -> object:
        """Read replication information on the borrowed connection.

        Arguments, results and command errors are those of
        `valkey.asyncio.Valkey.info`.

        Reconnect-class command failures invalidate the owning role pool.

        Raises:
            BackendCapacity: Cleanup after a reconnect-class failure is
                rejected for capacity.
            BackendProtocolError: Cleanup after a reconnect-class failure
                violates the backend protocol.
            BackendUnavailable: The pin has left its scope, its pool was
                discarded, the role closed, or cleanup failed.
            RuntimeError: The role belongs to another event loop.
            CancelledError: The caller cancels the command; the interrupted
                connection is settled before propagation.
        """
        return await self._invoke('INFO', section)

    async def script_load(self, script: bytes | str) -> object:
        """Load a backend script on the borrowed connection.

        Arguments, results and command errors are those of
        `valkey.asyncio.Valkey.script_load`.

        Reconnect-class command failures invalidate the owning role pool.

        Raises:
            BackendCapacity: Cleanup after a reconnect-class failure is
                rejected for capacity.
            BackendProtocolError: Cleanup after a reconnect-class failure
                violates the backend protocol.
            BackendUnavailable: The pin has left its scope, its pool was
                discarded, the role closed, or cleanup failed.
            RuntimeError: The role belongs to another event loop.
            CancelledError: The caller cancels the command; the interrupted
                connection is settled before propagation.
        """
        return await self._invoke('SCRIPT LOAD', script)


class _SentinelAPI(Protocol):
    """Type maintained Sentinel ownership and async primary discovery."""

    sentinels: list[_RawClient]

    def master_for(self, service_name: str, **kwargs: object) -> _RawClient:
        """Construct the lazy primary client owned by a backend role.

        Arguments, results and client errors are those of
        `valkey.asyncio.sentinel.Sentinel.master_for`.
        """
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
        """Close the primary role and all retained discovery clients.

        Raises:
            BackendCapacity: A client reports capacity rejection during
                cleanup.
            BackendProtocolError: A client reports invalid protocol data during
                cleanup.
            BackendUnavailable: An owned resource cannot be closed.
        """
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

    owns_published_deadline = True

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

    def _settle_guard(
        self,
        name: str,
        error: BaseException | None,
        *,
        fresh: bool,
        timer: Timeout | None,
    ) -> None:
        """Settle the admitted operation from its own timer before its outcome."""
        expired = timer is not None and timer.expired()
        if expired or isinstance(error, (BackendProtocolError, BackendUnavailable)):
            failure_class = error.failure_class if isinstance(error, BackendUnavailable) else None
            if expired:
                failure_class = 'timeout'
            labels = {'backend_operation': name, 'outcome': 'failure'}
            if failure_class is not None:
                labels['failure_class'] = failure_class
            self._increment('backend_failure', labels)
            self._breaker.fail(failure_class)
        elif isinstance(error, BackendCapacity):
            self._metadata_capacity_failure(fresh)
        elif error is None or (fresh and isinstance(error, StaleIncarnation)):
            self._breaker.succeed()
        elif fresh:
            if isinstance(error, CancelledError):
                self._breaker.abandon()
            else:
                self._breaker.fail()

    async def _guard(self, name: str, operation: Callable[[], Awaitable[_T]]) -> _T:
        """Run one metadata operation through its role breaker."""
        started_at = monotonic()
        fresh = self._breaker.admit()
        timer: Timeout | None = None
        try:
            async with self._lifecycle.budget() as timer:
                if fresh:
                    await self._client.refresh()
                result = await operation()
        except BaseException as error:
            self._settle_guard(name, error, fresh=fresh, timer=timer)
            raise
        self._settle_guard(name, None, fresh=fresh, timer=timer)
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

    async def change(self, changes: tuple[GenerationChange, ...]) -> None:
        """Replace all applicable generations in one metadata script.

        Args:
            changes: Ordered global and replace-if-materialized scoped changes.

        Raises:
            BackendCapacity: Required generation metadata cannot be retained.
            BackendProtocolError: The change tuple or a backend reply is
                malformed.
            BackendUnavailable: Metadata validation or the change cannot
                complete reliably.
            StaleIncarnation: Control state changed or its counter requires
                rotation.
            RuntimeError: The store is closing, closed, or bound to another
                event loop.
            CancelledError: The caller cancels the operation.
        """
        with self._lifecycle.operation():
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
        await self._lifecycle.close((self._client.close,))

    async def flush_site(self, site_namespace: str, reason: str) -> None:
        """Replace one site's permanent generation through the metadata role.

        Args:
            site_namespace: Stable configured site namespace.
            reason: Nonempty operator-facing ASCII reason of at most 64
                characters.

        Raises:
            BackendCapacity: Required global metadata cannot be retained.
            BackendProtocolError: The namespace, reason, or backend reply is
                malformed.
            BackendUnavailable: Metadata validation or the flush cannot
                complete reliably.
            StaleIncarnation: Control state changed or its counter requires
                rotation.
            RuntimeError: The store is closing, closed, or bound to another
                event loop.
            CancelledError: The caller cancels the operation.
        """
        with self._lifecycle.operation():
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
        client = await _create_client(
            role, settings, role_password(role, environ, 'metadata'), metrics
        )
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
            Every tag and token in request order under the active incarnation.

        Raises:
            BackendCapacity: Metadata capacity cannot retain the vector.
            BackendProtocolError: The vector or backend reply is malformed, or
                the vector exceeds its bounds.
            BackendUnavailable: Metadata validation or resolution cannot
                complete reliably.
            StaleIncarnation: Control state changed or its counter requires
                rotation.
            RuntimeError: The store is closing, closed, or bound to another
                event loop.
            CancelledError: The caller cancels the operation.
        """
        with self._lifecycle.operation():
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

    owns_published_deadline = True

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

    def _settle_guard(
        self,
        name: str,
        error: BaseException | None,
        *,
        fresh: bool,
        timer: Timeout | None,
    ) -> None:
        """Settle the admitted operation from its own timer before its outcome."""
        expired = timer is not None and timer.expired()
        if expired or isinstance(error, (BackendProtocolError, BackendUnavailable)):
            failure_class = error.failure_class if isinstance(error, BackendUnavailable) else None
            if expired:
                failure_class = 'timeout'
            labels = {'backend_operation': name, 'outcome': 'failure'}
            if failure_class is not None:
                labels['failure_class'] = failure_class
            self._increment('backend_failure', labels)
            self._result_breaker.fail(failure_class)
        elif error is None or (fresh and isinstance(error, (BackendCapacity, StaleIncarnation))):
            self._result_breaker.succeed()
        elif fresh:
            if isinstance(error, CancelledError):
                self._result_breaker.abandon()
            else:
                self._result_breaker.fail()

    async def _result_guard(self, name: str, operation: Callable[[], Awaitable[_T]]) -> _T:
        """Run one result operation through its independent role breaker."""
        started_at = monotonic()
        fresh = self._result_breaker.admit()
        timer: Timeout | None = None
        try:
            async with self._lifecycle.budget() as timer:
                if fresh:
                    await self._result_client.refresh()
                result = await operation()
        except BaseException as error:
            self._settle_guard(name, error, fresh=fresh, timer=timer)
            raise
        self._settle_guard(name, None, fresh=fresh, timer=timer)
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

        Raises:
            BackendCapacity: Required generation metadata cannot be retained.
            BackendProtocolError: The change tuple or backend reply is
                malformed.
            BackendUnavailable: Metadata validation or the change cannot
                complete reliably.
            StaleIncarnation: The metadata incarnation is no longer current.
            RuntimeError: The backend is closing, closed, or bound to another
                event loop.
            CancelledError: The caller cancels the operation.
        """
        with self._lifecycle.operation():
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
        await self._lifecycle.close((self._result_client.close, self._metadata.close))

    async def flush_site(self, site_namespace: str, reason: str) -> None:
        """Replace one site's permanent site generation.

        Args:
            site_namespace: Stable configured site namespace.
            reason: Nonempty operator-facing ASCII reason of at most 64
                characters.

        Raises:
            BackendCapacity: Required global metadata cannot be retained.
            BackendProtocolError: The namespace, reason, or backend reply is
                malformed.
            BackendUnavailable: Metadata validation or the flush cannot
                complete reliably.
            StaleIncarnation: The metadata incarnation is no longer current.
            RuntimeError: The backend is closing, closed, or bound to another
                event loop.
            CancelledError: The caller cancels the operation.
        """
        with self._lifecycle.operation():
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
        metadata_password = role_password(metadata_role, environ, 'metadata')
        result_password = role_password(result_role, environ, 'result')
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
        """Read valid encoded bytes with conservatively debited remaining age.

        Args:
            key: Nonempty UTF-8 generation-qualified result key.

        Returns:
            An isolated hit carrying a fresh decoded value and the server TTL
            after elapsed lookup processing is deducted, or `None` for absence,
            expiry, oversized values, or malformed data.

        Raises:
            BackendCapacity: The result role rejects required work for
                capacity.
            BackendProtocolError: The key or backend reply is malformed.
            BackendUnavailable: The result role is unavailable or the lookup
                cannot complete reliably.
            RuntimeError: The backend is closing, closed, or bound to another
                event loop.
            CancelledError: The caller cancels the operation.
        """
        with self._lifecycle.operation():
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
            Whether the current lease was removed; `False` if it is absent or
            belongs to another token.

        Raises:
            BackendCapacity: The result role rejects required work for
                capacity.
            BackendProtocolError: The lease or backend reply is malformed.
            BackendUnavailable: The result role is unavailable or release
                cannot complete reliably.
            RuntimeError: The backend is closing, closed, or bound to another
                event loop.
            CancelledError: The caller cancels the operation.
        """
        with self._lifecycle.operation():
            validated = fill_lease(lease)
            return await self._result_guard('release', lambda: self._release(validated))

    async def commit(self, lease: FillLease, payload: bytes, remaining_seconds: float) -> bool:
        """Store bytes only while the exact result-role lease remains current.

        Elapsed work is deducted from the lifetime measured at method entry.
        An oversized payload triggers an attempt to release its lease.

        Args:
            lease: Exact key and random token returned by `reserve`.
            payload: Closed-codec result bytes.
            remaining_seconds: Finite remaining lifetime in seconds whose
                floored milliseconds fit a positive signed 64-bit integer.

        Returns:
            Whether the result was admitted under the current lease. Return
            `False` for a lost lease, an exhausted lifetime, an oversized
            payload, or capacity rejection during the commit operation.

        Raises:
            BackendCapacity: Role preparation or oversized-payload lease
                cleanup encounters a capacity failure.
            BackendProtocolError: An argument or backend reply violates the
                backend contract.
            BackendUnavailable: The result role is unavailable or commit or
                oversized-payload lease cleanup cannot complete reliably.
            RuntimeError: The backend is closing, closed, or bound to another
                event loop.
            CancelledError: The caller cancels the operation.
        """
        entered_at = monotonic()
        with self._lifecycle.operation():
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
            key: Nonempty UTF-8 generation-qualified result key.
            lease_seconds: Finite ownership duration in seconds whose floored
                milliseconds fit a positive signed 64-bit integer.

        Returns:
            A fresh lease, or `None` while another lease remains current.

        Raises:
            BackendCapacity: The result role cannot retain the lease.
            BackendProtocolError: An argument or backend reply violates the
                backend contract.
            BackendUnavailable: Entropy is unavailable, the result role is
                unavailable, or reservation cannot complete reliably.
            RuntimeError: The backend is closing, closed, or bound to another
                event loop.
            CancelledError: The caller cancels the operation.
        """
        with self._lifecycle.operation():
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
            Every requested tag and current token in request order under one
            metadata incarnation.

        Raises:
            BackendCapacity: Metadata capacity cannot retain the vector.
            BackendProtocolError: The vector or backend reply is malformed, or
                the vector exceeds its bounds.
            BackendUnavailable: Metadata validation or resolution cannot
                complete reliably.
            StaleIncarnation: The metadata incarnation is no longer current.
            RuntimeError: The backend is closing, closed, or bound to another
                event loop.
            CancelledError: The caller cancels the operation.
        """
        with self._lifecycle.operation():
            return await self._metadata.resolve(tags)


async def _close_pin(client: _RawClient, cancellation: CancelledError | None) -> None:
    """Join maintained pin release before propagating cancellation or failure."""
    connection = client.connection
    token = (
        client.connection_pool.borrow_tokens.get(connection) if connection is not None else None
    )
    outcome = await joined_cleanup(lambda: client.aclose(close_connection_pool=False))
    # aclose clears connection before release. Settle failed cleanup only while
    # the token still identifies this borrow; release may have succeeded before
    # a later failure, allowing another pin to acquire the same connection.
    client.connection = None
    if connection is not None and outcome.error is not None:
        await client.connection_pool.settle_release(connection, token)
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
        await client.connection_pool.settle_disconnect()
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
                    valkey_connect_func=tracker.initialize,
                    **kwargs,
                )
            pool = _ConnectionPool(
                connection_class=_SSLConnection if role.tls else _Connection,
                max_connections=None,
                host=role.host,
                port=role.port,
                valkey_connect_func=tracker.initialize,
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
