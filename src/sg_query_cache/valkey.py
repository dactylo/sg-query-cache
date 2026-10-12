# Same-file collaborators use private implementation members.
# pyright: reportPrivateUsage=false

"""Provide the optional synchronous two-role Valkey cache backend.

Importing this module requires the optional `valkey` dependency. Construct a
configured backend with `ValkeyBackend.from_settings`, supplying validated
cache settings and the environment mapping containing the role passwords.
The result and metadata roles use separate clients; metadata-only consumers can
construct `ValkeyMetadataStore` instead.

Close the owning backend or metadata store when finished. Service layout,
eviction policies, and deployment requirements are documented in
`docs/operations.md`.
"""

from collections.abc import Callable, Generator, Mapping
from contextlib import contextmanager, suppress
from secrets import token_hex
from threading import Lock
from time import monotonic
from typing import Protocol, Self, TypeAlias, TypeVar, cast
from weakref import WeakKeyDictionary, WeakSet

from valkey import Sentinel, Valkey
from valkey.backoff import NoBackoff
from valkey.exceptions import ConnectionError as ValkeyConnectionError
from valkey.exceptions import ReadOnlyError
from valkey.exceptions import TimeoutError as ValkeyTimeoutError
from valkey.retry import Retry

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
from sg_query_cache._valkey_common import (
    CircuitBreaker,
    Incarnation,
    LookupOutcome,
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
from sg_query_cache._valkey_control import ControlClient, PinnedControlClient, ValkeyControl
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
    ResultClient,
    ScriptClient,
    execute_commit,
    execute_script,
    translate_client_error,
)

_T = TypeVar('_T')

_Clock: TypeAlias = Callable[[], float]


class _CircuitBreaker(CircuitBreaker):
    """Serialize the shared circuit state machine for synchronous callers."""

    def __init__(
        self,
        role: str,
        *,
        threshold: int,
        probe_seconds: float,
        metrics: MetricsSink,
        clock: _Clock = monotonic,
    ) -> None:
        """Initialize the shared breaker and its role-local lock."""
        super().__init__(
            role, threshold=threshold, probe_seconds=probe_seconds, metrics=metrics, clock=clock
        )
        self._lock = Lock()

    def admit(self) -> bool:
        """Admit one operation and report whether it needs a fresh pool."""
        with self._lock:
            return super().admit()

    def fail(self) -> None:
        """Count one failure or reopen a failed probe."""
        with self._lock:
            super().fail()

    def succeed(self) -> None:
        """Close a successful probe and reset consecutive failures."""
        with self._lock:
            super().succeed()


class _MetadataStore(Protocol):
    """Provide generation metadata and its independent resource lifecycle."""

    def change(self, changes: tuple[GenerationChange, ...]) -> None:
        """Replace the ordered dependency generations."""
        ...

    def close(self) -> None:
        """Release metadata resources."""
        ...

    def flush_site(self, site_namespace: str, reason: str) -> None:
        """Replace the site's permanent generation."""
        ...

    def resolve(self, tags: tuple[DependencyTag, ...]) -> GenerationVector:
        """Resolve ordered dependency generations."""
        ...


class _ResultRole(ScriptClient, ResultClient, Protocol):
    """Supply result commands, observed execution, refresh and terminal cleanup."""

    def close(self) -> None:
        """Release this role's resources."""
        ...

    def refresh(self) -> None:
        """Discard the current pool and construct a fresh connection."""
        ...


class _ValkeyClient(ControlClient, ResultClient, Protocol):
    """Provide the synchronous Valkey commands owned by this backend."""

    def close(self) -> None:
        """Release the maintained client's connection pool."""
        ...

    def refresh(self) -> None:
        """Discard the current pool and construct a fresh connection."""
        ...


class _RawValkeyClient(PinnedControlClient, Protocol):
    """Describe one concrete maintained-client connection pool."""

    def client(self) -> '_PinnedRawValkeyClient':
        """Borrow the maintained client's single-connection child.

        Returns:
            A child sharing this client's pool. Close the child when finished
            to return its borrowed connection.
        """
        ...

    def close(self) -> None:
        """Release the concrete connection pool."""
        ...


class _PinnedRawValkeyClient(PinnedControlClient, Protocol):
    """Describe one maintained client pinned to a physical connection."""

    def close(self) -> None:
        """Return the borrowed connection to its owning pool."""
        ...


class _ReconnectObservable(Protocol):
    """Publish maintained-client physical reconnect events."""

    def connect(self) -> None:
        """Establish the physical connection before its first command."""
        ...

    def disconnect(self) -> None:
        """Close the current socket before observed reconnection."""
        ...

    def register_connect_callback(self, callback: Callable[[object], None]) -> None:
        """Register a weakly held callback for successful connection establishment.

        Args:
            callback: Bound method receiving the physical connection after an
                initial connection or reconnect. Its owner must remain alive
                for notifications to continue.
        """
        ...


class _SetupConnection(Protocol):
    """Initialize one maintained-client connection after socket establishment."""

    def disconnect(self) -> None:
        """Close the connection's socket."""
        ...

    def on_connect(self) -> None:
        """Authenticate, select the database, and finish protocol setup."""
        ...

    def read_response(self) -> object:
        """Read one raw protocol response from the connection."""
        ...

    def send_command(self, *args: object) -> None:
        """Send one command on the established connection."""
        ...


class _ConnectionGeneration:
    """Give each physical connection and reconnect a role-unique identity."""

    __slots__ = ('__weakref__', '_by_connection', '_generation_count', '_lock', '_registered')

    def __init__(self) -> None:
        """Initialize weak connection tracking and the synchronized generation count."""
        self._by_connection: WeakKeyDictionary[object, int] = WeakKeyDictionary()
        self._generation_count = 0
        self._lock = Lock()
        self._registered: WeakSet[object] = WeakSet()

    def _connected(self, connection: object) -> None:
        """Record one successful physical connection under a new role-unique generation."""
        with self._lock:
            self._registered.add(connection)
            self._generation_count += 1
            self._by_connection[connection] = self._generation_count

    def _borrowed(self, client: object) -> bool:
        """Register a newly borrowed physical connection for reconnects."""
        connection = getattr(client, 'connection', None)
        register = getattr(connection, 'register_connect_callback', None)
        if connection is None or not callable(register):
            return False
        with self._lock:
            if connection in self._registered:
                return False
            self._registered.add(connection)
            self._by_connection[connection] = 0
        observable = cast('_ReconnectObservable', connection)
        observable.register_connect_callback(self._connected)
        return True

    def _for_client(self, client: object) -> int | None:
        """Return one borrowed connection's latest role-unique generation."""
        connection = getattr(client, 'connection', None)
        if connection is None:
            return None
        with self._lock:
            return self._by_connection.get(connection)

    @property
    def _generation(self) -> int:
        """Return the successful-connection generation."""
        with self._lock:
            return self._generation_count


class _SentinelPool(Protocol):
    """Expose the currently discovered Sentinel primary address."""

    def get_master_address(self) -> tuple[str, int]:
        """Discover and return the writable primary address."""
        ...


class _SentinelPooledClient(Protocol):
    """Expose a Sentinel-backed maintained-client pool."""

    connection_pool: _SentinelPool


class _RoleClient:
    """Reconnect one role through a retained provider-neutral factory."""

    __slots__ = (
        '_client',
        '_connections',
        '_endpoint',
        '_endpoint_probe',
        '_factory',
        '_lock',
        '_pool_epoch',
    )

    def _active(self) -> _RawValkeyClient:
        """Return the active pool, constructing it after invalidation."""
        with self._lock:
            if self._client is None:
                self._client = self._factory()
            return self._client

    def __init__(
        self,
        factory: Callable[[], _RawValkeyClient],
        connections: _ConnectionGeneration,
        endpoint_probe: Callable[[_RawValkeyClient], object] | None = None,
    ) -> None:
        """Retain the role factory and track its initial connection pool."""
        self._client: _RawValkeyClient | None = None
        self._connections = connections
        self._endpoint: object | None = None
        self._endpoint_probe = endpoint_probe
        self._factory = factory
        self._lock = Lock()
        self._pool_epoch = 0
        self._active()
        if endpoint_probe is not None:
            self.refresh_endpoint()

    def _connection_generation(self, client: _PinnedRawValkeyClient) -> int | None:
        """Return one pin's latest role-unique connection generation."""
        return self._connections._for_client(client)

    def _invalidate(self, expected: _RawValkeyClient | None = None) -> None:
        """Detach and close a pool after reconnect-class failure."""
        with self._lock:
            if expected is not None and self._client is not expected:
                return
            client = self._client
            if client is None:
                return
            self._client = None
            self._pool_epoch += 1
        with suppress(Exception):
            client.close()

    def _invoke_on(
        self,
        client: _RawValkeyClient,
        operation: Callable[[], _T],
    ) -> _T:
        """Run one command against a known pool and invalidate its failures."""
        try:
            return operation()
        except (ReadOnlyError, ValkeyConnectionError, ValkeyTimeoutError):
            self._invalidate(client)
            raise

    def _role_epoch(self) -> tuple[int, object | None]:
        """Return pool and endpoint state without the diagnostic aggregate."""
        with self._lock:
            pool_epoch = self._pool_epoch
            endpoint = self._endpoint
        return pool_epoch, endpoint

    def close(self) -> None:
        """Release the active pool without constructing a replacement."""
        with self._lock:
            client = self._client
            self._client = None
        if client is not None:
            client.close()

    def epoch(self) -> tuple[int, int, object | None]:
        """Return recorded role state without performing network I/O.

        Returns:
            The pool invalidation count, aggregate successful-connection count,
            and last recorded Sentinel primary address. The address is `None`
            when no endpoint has been recorded.
        """
        pool_epoch, endpoint = self._role_epoch()
        return pool_epoch, self._connections._generation, endpoint

    @contextmanager
    def pin(self) -> Generator['_PinnedRoleClient']:
        """Borrow one observed physical connection for a context-managed operation.

        Yields:
            A command wrapper valid within the context. Its borrowed child is
            closed on exit to return the connection to the owning pool.

        Raises:
            BackendCapacity: Valkey rejects connection setup for capacity.
            BackendProtocolError: Borrowing or setup violates the client
                protocol.
            BackendUnavailable: A physical connection cannot be borrowed or
                established.
        """
        client = self._active()
        failure: Exception | None = None
        pinned: _PinnedRawValkeyClient | None = None
        try:
            pinned = client.client()
            newly_observed = self._connections._borrowed(pinned)
            connection = getattr(pinned, 'connection', None)
            connect = getattr(connection, 'connect', None)
            disconnect = getattr(connection, 'disconnect', None)
            if newly_observed and callable(connect) and callable(disconnect):
                disconnect()
                connect()
        except Exception as error:  # noqa: BLE001 - closed single-connection borrow boundary
            failure = error
        if failure is not None:
            self._invalidate(client)
            translate_client_error(failure)
        if pinned is None:
            raise BackendUnavailable('Valkey single-connection borrow failed')
        wrapped = _PinnedRoleClient(self, client, pinned)
        try:
            yield wrapped
        finally:
            if pinned is not client:
                with suppress(Exception):
                    pinned.close()

    def refresh(self) -> None:
        """Discard the old pool and eagerly construct its replacement.

        Raises:
            BackendCapacity: Valkey rejects replacement setup for capacity.
            BackendProtocolError: Replacement setup violates the client
                protocol.
            BackendUnavailable: A replacement client cannot be established.
        """
        self._invalidate()
        failure: Exception | None = None
        try:
            self._active()
        except (BackendCapacity, BackendProtocolError, BackendUnavailable):
            raise
        except Exception as error:  # noqa: BLE001 - closed client-factory boundary
            failure = error
        if failure is not None:
            translate_client_error(failure)

    def refresh_endpoint(self) -> None:
        """Refresh Sentinel discovery and invalidate a pool whose primary changed.

        Do nothing when this role has no endpoint probe.

        Raises:
            BackendCapacity: Valkey rejects discovery or setup for capacity.
            BackendProtocolError: Discovery or setup violates the client
                protocol.
            BackendUnavailable: The primary cannot be discovered reliably.
        """
        probe = self._endpoint_probe
        if probe is None:
            return
        client = self._active()
        failure: Exception | None = None
        endpoint: object | None = None
        try:
            endpoint = probe(client)
        except Exception as error:  # noqa: BLE001 - closed discovery boundary
            failure = error
        if failure is not None:
            self._invalidate(client)
            translate_client_error(failure)
        with self._lock:
            changed = self._endpoint is not None and endpoint != self._endpoint
            self._endpoint = endpoint
        if changed:
            self._invalidate(client)

    def run_operation(self, operation: Callable[[PinnedControlClient], _T]) -> _T:
        """Run result work within one verified physical-connection span.

        Callback exceptions propagate after successful settlement; a settlement
        failure takes precedence over a callback exception.

        Args:
            operation: Callback receiving a pinned client for the duration of
                the call. It may issue multiple commands.

        Returns:
            The callback's result after endpoint and connection state settle
            without a role change.

        Raises:
            BackendCapacity: Valkey rejects discovery or connection setup for
                capacity.
            BackendProtocolError: Discovery or connection setup violates the
                client protocol.
            BackendUnavailable: The role changes before or during the
                operation, or its connection state cannot be settled.
        """
        before_refresh = self._role_epoch()
        self.refresh_endpoint()
        if self._role_epoch() != before_refresh:
            raise BackendUnavailable('Valkey role changed before operation')
        with self.pin() as client:
            initial_role_epoch = self._role_epoch()
            initial_connection_generation = client.connection_generation()
            outcome: tuple[_T, ...] = ()
            operation_failure: Exception | None = None
            try:
                outcome = (operation(client),)
            except Exception as error:  # noqa: BLE001 - closed role-operation boundary
                operation_failure = error
            settlement_failure: Exception | None = None
            try:
                self.refresh_endpoint()
            except (BackendProtocolError, BackendUnavailable) as error:
                settlement_failure = error
            if (
                self._role_epoch() != initial_role_epoch
                or client.connection_generation() != initial_connection_generation
            ):
                self._invalidate()
                raise BackendUnavailable('Valkey role changed during operation')
            if settlement_failure is not None:
                raise settlement_failure
            if operation_failure is not None:
                raise operation_failure
            if not outcome:
                raise BackendUnavailable('Valkey operation did not return an outcome')
            return outcome[0]

    def evalsha(self, sha: str, numkeys: int, *keys_and_args: ScriptArgument) -> object:
        """Execute a cached script and return its raw reply through the active pool.

        Role validation and failures follow `run_operation`.

        Args:
            sha: SHA-1 digest of the script to execute.
            numkeys: Number of leading `keys_and_args` entries interpreted as
                keys.
            *keys_and_args: Keys followed by script arguments in caller order.

        Raises:
            NoScriptError: The server does not have the requested script
                cached.
        """
        return self.run_operation(lambda client: client.evalsha(sha, numkeys, *keys_and_args))

    def exists(self, key: str) -> object:
        """Read key existence through the active pool."""
        return self.run_operation(lambda client: client.exists(key))

    def get(self, key: str) -> object:
        """Read one raw string value through the active pool.

        Role validation and failures follow `run_operation`.

        Returns:
            The stored bytes, or `None` when the key is absent.
        """
        return self.run_operation(lambda client: client.get(key))

    def info(self, section: str) -> object:
        """Read one information section through the active pool."""
        return self.run_operation(lambda client: client.info(section))

    def script_load(self, script: bytes | str) -> object:
        """Load one script and return its raw server digest through the active pool.

        Role validation and failures follow `run_operation`.
        """
        return self.run_operation(lambda client: client.script_load(script))


class ValkeyMetadataStore:
    """Own metadata-role control and generation operations without result access.

    The store owns its metadata client and releases it through `close`.
    Pair it with `ValkeyBackend` when result storage is also required.
    Operations follow the `CacheBackend` validation, generation-fencing, and
    normalized-failure contracts. Stop using the store after `close`;
    subsequent operations are not uniformly rejected.
    """

    __slots__ = (
        '_breaker',
        '_client',
        '_closed',
        '_control',
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
        """Retain the metadata client and initialize control and failure tracking.

        Args:
            client: Dedicated metadata-role maintained client.
            settings: Validated shared cache settings.
            metrics: Safe metrics sink.
        """
        self._breaker = _CircuitBreaker(
            'metadata',
            threshold=settings.limits.circuit_failure_threshold,
            probe_seconds=settings.limits.circuit_probe_seconds,
            metrics=metrics,
        )
        self._client = client
        self._closed = False
        self._control = ValkeyControl(
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

    def _change(self, changes: tuple[GenerationChange, ...]) -> None:
        """Perform one unguarded atomic generation change."""
        tags = tuple(change.tag for change in changes)

        def operation(client: PinnedControlClient, incarnation: Incarnation) -> None:
            operands = metadata_operands(
                self._settings.l2.valkey.metadata.key_prefix,
                incarnation,
                tags,
                self._settings.limits.scoped_generation_retention_seconds,
            )
            reply = execute_script(
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

        self._control.run(operation)

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

    def _guard(self, name: str, operation: Callable[[], _T]) -> _T:
        """Run one metadata operation through its role breaker."""
        started_at = monotonic()
        fresh = self._breaker.admit()
        try:
            if fresh:
                self._client.refresh()
            result = operation()
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

    def _resolve(self, tags: tuple[DependencyTag, ...]) -> GenerationVector:
        """Perform one unguarded metadata vector resolution."""

        def operation(client: PinnedControlClient, incarnation: Incarnation) -> GenerationVector:
            operands = metadata_operands(
                self._settings.l2.valkey.metadata.key_prefix,
                incarnation,
                tags,
                self._settings.limits.scoped_generation_retention_seconds,
            )
            reply = execute_script(
                client,
                script=RESOLVE_SCRIPT,
                sha=RESOLVE_SCRIPT_SHA,
                keys=operands.keys,
                args=operands.args,
            )
            tokens = decode_resolve_reply(reply, expected_count=len(tags))
            return GenerationVector(incarnation.nonce, tuple(zip(tags, tokens, strict=True)))

        return self._control.run(operation)

    def change(self, changes: tuple[GenerationChange, ...]) -> None:
        """Replace all applicable generations in one metadata script.

        Args:
            changes: Ordered global and replace-if-materialized scoped changes.

        Raises:
            BackendCapacity: Required metadata cannot be retained.
            BackendProtocolError: The change tuple or a backend reply is
                malformed.
            BackendUnavailable: The role circuit is open or the change cannot
                complete reliably.
            StaleIncarnation: Control state changed or its counter requires
                rotation.
        """
        validated = generation_changes(changes)
        return self._guard('change', lambda: self._change(validated))

    def close(self) -> None:
        """Attempt metadata-client cleanup once.

        Repeated calls have no effect, including after a cleanup failure.

        Raises:
            BackendCapacity: The client reports a capacity failure during
                cleanup.
            BackendProtocolError: The client reports a protocol failure during
                cleanup.
            BackendUnavailable: The client cannot complete cleanup reliably.
        """
        if self._closed:
            return
        self._closed = True
        _close_client(self._client)

    def flush_site(self, site_namespace: str, reason: str) -> None:
        """Replace one site's permanent generation through the metadata role.

        Args:
            site_namespace: Stable configured site namespace.
            reason: Nonempty ASCII reason code of at most 64 characters.

        Raises:
            BackendCapacity: Required metadata cannot be retained.
            BackendProtocolError: The site namespace, reason, or a backend
                reply is malformed.
            BackendUnavailable: The role circuit is open or the flush cannot
                complete reliably.
            StaleIncarnation: Control state changed or its counter requires
                rotation.
        """
        try:
            tag = DependencyTag.site(site_namespace)
        except (TypeError, ValueError):
            raise BackendProtocolError('site namespace is invalid') from None
        flush_reason(reason)
        self.change((GenerationChange(tag),))
        self._increment('cache_site_flush', {'reason': metric_flush_reason(reason)})

    @classmethod
    def from_settings(
        cls,
        settings: CacheSettings,
        environ: Mapping[str, object],
        metrics: MetricsSink,
    ) -> Self:
        """Create a metadata-only store from validated role settings.

        Sentinel and managed role preparation can perform I/O before returning.
        If initial primary validation fails, cleanup of that client is
        attempted before the failure propagates.

        Args:
            settings: Shared settings with enabled Valkey L2 configuration.
            environ: Environment mapping containing the configured metadata
                role password.
            metrics: Safe metrics sink.

        Returns:
            A store owning its metadata-role client, to be closed when
            finished.

        Raises:
            TypeError: `settings` is not a `CacheSettings` instance.
            ValueError: Valkey L2 is disabled or its password is absent or
                empty.
            BackendCapacity: Valkey rejects client setup for capacity.
            BackendProtocolError: Client setup or primary validation violates
                the backend protocol.
            BackendUnavailable: Client setup or primary discovery cannot
                complete reliably.
        """
        require_enabled_l2(settings)
        role = settings.l2.valkey.metadata
        client = _create_client(role, settings, role_password(role, environ))
        return cls(client, settings, metrics)

    def resolve(self, tags: tuple[DependencyTag, ...]) -> GenerationVector:
        """Resolve one complete dependency vector in one metadata script.

        Args:
            tags: Ordered unique dependency tags within the configured vector
                bounds.

        Returns:
            Every requested tag and token in request order under the active
            incarnation.

        Raises:
            BackendCapacity: Required metadata cannot be retained.
            BackendProtocolError: The vector exceeds its bounds or an input or
                backend reply is malformed.
            BackendUnavailable: The role circuit is open or resolution cannot
                complete reliably.
            StaleIncarnation: Control state changed or its counter requires
                rotation.
        """
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
        vector = self._guard('resolve', lambda: self._resolve(validated))
        self._increment('cache_vector_resolution', {'layer': 'l2'})
        return vector

    def rotate_incarnation(self) -> None:
        """Rotate the freshly observed metadata incarnation once.

        Rotation invalidates old result names; it does not reclaim retained
        metadata.

        Raises:
            BackendCapacity: The replacement control state cannot be retained.
            BackendProtocolError: Control replies are malformed.
            BackendUnavailable: The store is closed, its role is unavailable,
                or the incarnation nonce did not change.
        """
        if self._closed:
            raise BackendUnavailable('Valkey metadata store is closed')

        def operation() -> None:
            observed = self._control.validate(force=True)
            replacement = self._control.rotate(observed)
            if replacement.nonce == observed.nonce:
                raise BackendUnavailable('Valkey incarnation did not rotate')

        self._guard('metadata', operation)


class ValkeyBackend:
    """Store shared results and metadata in separate Valkey roles.

    Result operations use caller-supplied generation-qualified keys; metadata
    operations use the separate metadata store. The backend owns its result
    client and metadata store and attempts cleanup of both through `close`.
    Operations follow the `CacheBackend` validation, generation-fencing,
    result-admission, and normalized-failure contracts. Stop using the backend
    after `close`; subsequent operations are not uniformly rejected.
    """

    __slots__ = (
        '_closed',
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
        """Retain separate result and metadata owners with result failure tracking.

        Args:
            result_client: Dedicated result-role maintained client.
            metadata: Metadata-only store owning its separate connection.
            settings: Validated shared cache settings.
            metrics: Safe metrics sink.
        """
        self._closed = False
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

    def _commit(self, lease: FillLease, payload: bytes, deadline: float) -> bool:
        """Perform one unguarded token-checked result commit."""
        try:
            return execute_commit(
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

    def _lookup(self, key: str) -> LookupOutcome:
        """Perform one unguarded result lookup."""
        operands = lookup_operands(self._settings.l2.valkey.result.key_prefix, key)
        # Conservatively include discovery, command delivery, and reply settlement.
        # Pre-command or NOSCRIPT delay may cause a safe early miss.
        lookup_started_at = monotonic()
        reply = execute_script(
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

    def _release(self, lease: FillLease) -> bool:
        """Perform one unguarded result-role lease release."""
        operands = release_operands(self._settings.l2.valkey.result.key_prefix, lease)
        reply = execute_script(
            self._result_client,
            script=RELEASE_SCRIPT,
            sha=RELEASE_SCRIPT_SHA,
            keys=operands.keys,
            args=operands.args,
        )
        return decode_release_reply(reply)

    def _reserve(self, key: str, ttl: int, token: str) -> FillLease | None:
        """Perform one unguarded result-role lease reservation."""
        operands = reserve_operands(self._settings.l2.valkey.result.key_prefix, key, ttl, token)
        reply = execute_script(
            self._result_client,
            script=RESERVE_SCRIPT,
            sha=RESERVE_SCRIPT_SHA,
            keys=operands.keys,
            args=operands.args,
        )
        if not decode_reserve_reply(reply):
            return None
        return FillLease(key, token)

    def _result_guard(self, name: str, operation: Callable[[], _T]) -> _T:
        """Run one result operation through its independent role breaker."""
        started_at = monotonic()
        fresh = self._result_breaker.admit()
        try:
            if fresh:
                self._result_client.refresh()
            result = operation()
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

    def change(self, changes: tuple[GenerationChange, ...]) -> None:
        """Replace all applicable generations in caller order.

        Args:
            changes: Ordered global and replace-if-materialized scoped changes
                to acknowledge together.

        Raises:
            BackendCapacity: Required metadata cannot be retained.
            BackendProtocolError: The change tuple or a backend reply is
                malformed.
            BackendUnavailable: The metadata role is unavailable or the change
                cannot complete reliably.
            StaleIncarnation: Control state changed or its counter requires
                rotation.
        """
        self._metadata.change(changes)

    def close(self) -> None:
        """Attempt cleanup of both owned roles once.

        Attempt metadata cleanup even when result-client cleanup fails, then
        raise the first cleanup failure. Repeated calls have no effect,
        including after a failure. Stop using the backend after close; this
        method does not prevent later operations.

        Raises:
            BackendCapacity: A client reports a capacity failure during
                cleanup.
            BackendProtocolError: A client reports a protocol failure during
                cleanup.
            BackendUnavailable: A client cannot complete cleanup reliably.
        """
        if self._closed:
            return
        self._closed = True
        failure: RuntimeError | None = None
        try:
            _close_client(self._result_client)
        except (BackendCapacity, BackendProtocolError, BackendUnavailable) as error:
            failure = error
        try:
            self._metadata.close()
        except (BackendCapacity, BackendProtocolError, BackendUnavailable) as error:
            if failure is None:
                failure = error
        if failure is not None:
            raise failure from None

    def flush_site(self, site_namespace: str, reason: str) -> None:
        """Replace one site's permanent site generation.

        Args:
            site_namespace: Stable configured site namespace.
            reason: Nonempty ASCII reason code of at most 64 characters.

        Raises:
            BackendCapacity: Required metadata cannot be retained.
            BackendProtocolError: The site namespace, reason, or a backend
                reply is malformed.
            BackendUnavailable: The metadata role is unavailable or the flush
                cannot complete reliably.
            StaleIncarnation: Control state changed or its counter requires
                rotation.
        """
        self._metadata.flush_site(site_namespace, reason)

    @classmethod
    def from_settings(
        cls,
        settings: CacheSettings,
        environ: Mapping[str, object],
        metrics: MetricsSink,
    ) -> Self:
        """Create both configured role clients from validated settings.

        Sentinel and managed role preparation can perform I/O before returning.
        If result-client construction raises a backend capacity, protocol, or
        availability error, attempt cleanup of the already created metadata
        client before propagating that error.

        Args:
            settings: Shared settings with enabled Valkey L2 configuration.
            environ: Environment mapping containing both configured role
                passwords.
            metrics: Safe metrics sink.

        Returns:
            A two-role backend owning both clients, to be closed when finished.

        Raises:
            TypeError: `settings` is not a `CacheSettings` instance.
            ValueError: Valkey L2 is disabled or a role password is absent or
                empty.
            BackendCapacity: Valkey rejects client setup for capacity.
            BackendProtocolError: Client setup or primary validation violates
                the backend protocol.
            BackendUnavailable: Client setup or primary discovery cannot
                complete reliably.
        """
        require_enabled_l2(settings)
        metadata_role = settings.l2.valkey.metadata
        result_role = settings.l2.valkey.result
        metadata_password = role_password(metadata_role, environ)
        result_password = role_password(result_role, environ)
        metadata_client = _create_client(metadata_role, settings, metadata_password)
        construction_failure: RuntimeError | None = None
        result_client: _ResultRole | None = None
        try:
            result_client = _create_client(result_role, settings, result_password)
        except (BackendCapacity, BackendProtocolError, BackendUnavailable) as error:
            construction_failure = error
        if construction_failure is not None:
            with suppress(BackendCapacity, BackendProtocolError, BackendUnavailable):
                _close_client(metadata_client)
            raise construction_failure from None
        if result_client is None:
            _close_client(metadata_client)
            raise BackendUnavailable('Valkey result client construction failed')
        metadata = ValkeyMetadataStore(metadata_client, settings, metrics)
        return cls(result_client, metadata, settings, metrics)

    def lookup(self, key: str) -> CacheHit | None:
        """Read valid encoded bytes with conservatively debited remaining age.

        Args:
            key: Nonempty UTF-8 generation-qualified result key.

        Returns:
            An isolated hit with its fresh decoded value and remaining server
            TTL after elapsed lookup processing is deducted, or `None` for
            absence, expiry, oversized values, or malformed data.

        Raises:
            BackendProtocolError: The key or a backend reply is malformed.
            BackendUnavailable: The result role is unavailable or the lookup
                cannot complete reliably.
        """
        normalized = opaque_key(key)
        outcome = finalize_lookup(
            self._result_guard('lookup', lambda: self._lookup(normalized)), monotonic
        )
        if outcome.reason is not None:
            self._increment('cache_miss', {'layer': 'l2', 'reason': outcome.reason})
        return outcome.hit

    def release(self, lease: FillLease) -> bool:
        """Remove a result-role fill lease only when its token remains current.

        Args:
            lease: Exact key and random token returned by `reserve`.

        Returns:
            Whether the current lease was removed; `False` if it is absent or
            belongs to another token.

        Raises:
            BackendProtocolError: The lease or a backend reply is malformed.
            BackendUnavailable: The result role is unavailable or release
                cannot complete reliably.
        """
        validated = fill_lease(lease)
        return self._result_guard('release', lambda: self._release(validated))

    def commit(self, lease: FillLease, payload: bytes, remaining_seconds: float) -> bool:
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
            payload, or result-capacity rejection.

        Raises:
            BackendProtocolError: An argument or a backend reply violates the
                backend contract.
            BackendUnavailable: The result role is unavailable or commit or
                oversized-payload lease cleanup cannot complete reliably.
        """
        entered_at = monotonic()
        validated_lease = fill_lease(lease)
        validated_payload = encoded_payload(payload)
        milliseconds(remaining_seconds, 'remaining lifetime')
        deadline = entered_at + remaining_seconds
        if len(validated_payload) > self._settings.limits.max_serialized_result_bytes:
            self.release(validated_lease)
            return False
        committed = self._result_guard(
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

    def reserve(self, key: str, lease_seconds: float) -> FillLease | None:
        """Reserve temporary fill ownership with a random result-role token.

        Args:
            key: Nonempty UTF-8 generation-qualified result key.
            lease_seconds: Finite ownership duration in seconds whose floored
                milliseconds fit a positive signed 64-bit integer.

        Returns:
            A fresh lease, or `None` while another lease remains current.

        Raises:
            BackendCapacity: Valkey cannot retain the lease.
            BackendProtocolError: An argument or a backend reply violates the
                backend contract.
            BackendUnavailable: Lease entropy is unavailable, the result role
                is unavailable, or reservation cannot complete reliably.
        """
        normalized = opaque_key(key)
        ttl = milliseconds(lease_seconds, 'lease duration')
        try:
            token = token_hex(32)
        except Exception:  # noqa: BLE001 - closed entropy boundary
            raise BackendUnavailable('Valkey lease entropy is unavailable') from None
        lease = self._result_guard('reserve', lambda: self._reserve(normalized, ttl, token))
        if lease is None:
            self._increment('cache_lease_rejection', {'layer': 'l2'})
        return lease

    def resolve(self, tags: tuple[DependencyTag, ...]) -> GenerationVector:
        """Resolve one complete dependency vector in one metadata phase.

        Args:
            tags: Ordered unique dependency tags within the configured vector
                bounds.

        Returns:
            Every requested tag and current token in request order under one
            active incarnation.

        Raises:
            BackendCapacity: Required metadata cannot be retained.
            BackendProtocolError: The vector exceeds its bounds or an input or
                backend reply is malformed.
            BackendUnavailable: The metadata role is unavailable or resolution
                cannot complete reliably.
            StaleIncarnation: Control state changed or its counter requires
                rotation.
        """
        return self._metadata.resolve(tags)


class _PinnedRoleClient:
    """Route one pinned client through its owning pool's failure boundary."""

    __slots__ = ('_client', '_owner', '_pool')

    def __init__(
        self,
        owner: _RoleClient,
        pool: _RawValkeyClient,
        client: _PinnedRawValkeyClient,
    ) -> None:
        """Retain one physical borrow and its owning role and pool."""
        self._client = client
        self._owner = owner
        self._pool = pool

    def _invoke(self, operation: Callable[[], _T]) -> _T:
        """Run one pinned command with complete-pool invalidation on failure."""
        return self._owner._invoke_on(self._pool, operation)

    def connection_generation(self) -> int | None:
        """Return this pin's latest role-unique connection generation.

        Returns:
            The role-wide generation recorded when this physical connection
            last connected, or `None` when it has no tracking record.
        """
        return self._owner._connection_generation(self._client)

    def evalsha(self, sha: str, numkeys: int, *keys_and_args: ScriptArgument) -> object:
        """Execute a cached script and return its raw reply on the pinned connection.

        Args:
            sha: SHA-1 digest of the script to execute.
            numkeys: Number of leading `keys_and_args` entries interpreted as
                keys.
            *keys_and_args: Keys followed by script arguments in caller order.

        Raises:
            NoScriptError: The server does not have the requested script
                cached.
        """
        return self._invoke(lambda: self._client.evalsha(sha, numkeys, *keys_and_args))

    def exists(self, key: str) -> object:
        """Read key existence on the pinned connection."""
        return self._invoke(lambda: self._client.exists(key))

    def get(self, key: str) -> object:
        """Read one raw string value on the pinned connection.

        Returns:
            The stored bytes, or `None` when the key is absent.
        """
        return self._invoke(lambda: self._client.get(key))

    def info(self, section: str) -> object:
        """Read one information section on the pinned connection."""
        return self._invoke(lambda: self._client.info(section))

    def script_load(self, script: bytes | str) -> object:
        """Load one script and return its raw server digest on the pinned connection."""
        return self._invoke(lambda: self._client.script_load(script))


def _close_client(client: _ResultRole) -> None:
    """Close one maintained client through the redacted failure boundary."""
    failure: Exception | None = None
    try:
        client.close()
    except Exception as error:  # noqa: BLE001 - closed external-client boundary
        failure = error
    if failure is not None:
        translate_client_error(failure)


def _prove_connection(connection: _SetupConnection) -> None:
    """Prove a new socket reaches a writable primary, closing it before any failure escapes."""
    try:
        connection.send_command('INFO', 'replication')
        require_primary_reply(connection.read_response())
    except BaseException:
        with suppress(Exception):
            connection.disconnect()
        raise


def _create_raw_client(
    role: ValkeyRoleSettings,
    settings: CacheSettings,
    password: str,
    connections: _ConnectionGeneration,
) -> _RawValkeyClient:
    """Construct one fresh direct or Sentinel-discovered maintained client."""
    service_name = role.service_name
    if role.mode == 'sentinel' and service_name is None:
        raise BackendProtocolError('Valkey Sentinel service name is absent')
    failure: Exception | None = None
    client: Valkey | None = None

    def observe_connection(connection: object) -> None:
        setup = cast('_SetupConnection', connection)
        setup.on_connect()
        if role.mode != 'standalone':
            _prove_connection(setup)
        connections._connected(connection)

    try:
        retry = Retry(NoBackoff(), 0)
        if role.mode == 'sentinel':
            kwargs: dict[str, object] = {
                'db': role.database,
                'ssl': role.tls,
                'username': role.username,
                'password': password,
                'socket_connect_timeout': settings.limits.connect_timeout_seconds,
                'socket_timeout': settings.limits.operation_timeout_seconds,
                'decode_responses': False,
                'retry': retry,
                'retry_on_timeout': False,
            }
            if role.tls:
                kwargs['ssl_check_hostname'] = True
            sentinel_kwargs = {key: value for key, value in kwargs.items() if key != 'db'}
            kwargs['valkey_connect_func'] = observe_connection
            sentinel = Sentinel(
                tuple(parse_sentinel_address(value) for value in role.sentinels),
                min_other_sentinels=2,
                sentinel_kwargs=sentinel_kwargs,
                **kwargs,
            )
            client = sentinel.master_for(cast('str', service_name))
        else:
            client = Valkey(
                host=role.host,
                port=role.port,
                db=role.database,
                ssl=role.tls,
                ssl_check_hostname=role.tls,
                username=role.username,
                password=password,
                socket_connect_timeout=settings.limits.connect_timeout_seconds,
                socket_timeout=settings.limits.operation_timeout_seconds,
                decode_responses=False,
                retry=retry,
                retry_on_timeout=False,
                valkey_connect_func=observe_connection,
            )
    except Exception as error:  # noqa: BLE001 - closed external-client boundary
        failure = error
    if failure is not None:
        translate_client_error(failure)
    if client is None:
        raise BackendUnavailable('Valkey client construction failed')
    return client


def _create_validated_primary(
    role: ValkeyRoleSettings,
    settings: CacheSettings,
    password: str,
    connections: _ConnectionGeneration,
) -> _RawValkeyClient:
    """Construct an HA client whose first connection proves a writable primary."""
    client = _create_raw_client(role, settings, password, connections)
    failure: Exception | None = None
    try:
        client.client().close()
    except Exception as error:  # noqa: BLE001 - closed external-client boundary
        failure = error
    if failure is not None:
        with suppress(Exception):
            client.close()
        translate_client_error(failure)
    return client


def _sentinel_primary(client: _RawValkeyClient) -> object:
    """Return the primary Sentinel currently resolves for one role client."""
    pooled = cast('_SentinelPooledClient', cast('object', client))
    return pooled.connection_pool.get_master_address()


def _create_client(
    role: ValkeyRoleSettings,
    settings: CacheSettings,
    password: str,
) -> _ValkeyClient:
    """Construct one reconnectable role client without credential-bearing URLs."""
    endpoint_probe = _sentinel_primary if role.mode == 'sentinel' else None
    connections = _ConnectionGeneration()

    def factory() -> _RawValkeyClient:
        if role.mode == 'standalone':
            return _create_raw_client(role, settings, password, connections)
        return _create_validated_primary(role, settings, password, connections)

    return _RoleClient(
        factory,
        connections,
        endpoint_probe,
    )
