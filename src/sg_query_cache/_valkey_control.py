"""Validate and rotate Valkey metadata-primary control state."""

from collections.abc import Callable
from contextlib import AbstractContextManager, suppress
from secrets import token_hex
from threading import Lock
from time import monotonic
from typing import Protocol, TypeAlias, TypeVar

from sg_query_cache._backend import (
    BackendCapacity,
    BackendProtocolError,
    BackendUnavailable,
    StaleIncarnation,
)
from sg_query_cache._metrics import MetricsSink, NullMetrics
from sg_query_cache._valkey_common import (
    Incarnation,
    RoleEpoch,
    ScriptRotationError,
    ValidationState,
    cached_incarnation,
    can_publish_validation,
    control_create_operands,
    control_key,
    control_rotate_operands,
    decode_control_record,
    decode_control_reply,
    decode_exists,
    jittered_interval,
    new_incarnation,
    next_validation_state,
    parse_record,
    primary_replication_id,
    read_validation_time,
    span_epoch,
    validation_interval,
    validation_is_current,
)
from sg_query_cache._valkey_scripts import (
    CONTROL_CREATE_SCRIPT,
    CONTROL_CREATE_SCRIPT_SHA,
    CONTROL_ROTATE_SCRIPT,
    CONTROL_ROTATE_SCRIPT_SHA,
    ScriptClient,
    execute_script,
    translate_client_error,
)

_T = TypeVar('_T')

_Clock: TypeAlias = Callable[[], float]
_JitterFactory: TypeAlias = Callable[[float], float]
_NonceFactory: TypeAlias = Callable[[], str]

_MAX_VALIDATION_ATTEMPTS = 4


class PinnedControlClient(ScriptClient, Protocol):
    """Read the commands needed to validate writable primary state."""

    def exists(self, key: str) -> object:
        """Return whether one key exists."""
        ...

    def get(self, key: str) -> object:
        """Return one raw string value."""
        ...

    def info(self, section: str) -> object:
        """Return one server information section."""
        ...


class ControlClient(PinnedControlClient, Protocol):
    """Pin all control validation commands to one physical connection."""

    def epoch(self) -> RoleEpoch:
        """Return the current pure role epoch."""
        ...

    def pin(self) -> AbstractContextManager[PinnedControlClient]:
        """Borrow one physical connection for a complete validation."""
        ...

    def refresh(self) -> None:
        """Discard the current pool and construct a fresh connection."""
        ...

    def refresh_endpoint(self) -> None:
        """Explicitly refresh Sentinel discovery when applicable."""
        ...


class ValkeyControl:
    """Validate one metadata primary and its control record."""

    __slots__ = (
        '_client',
        '_clock',
        '_control_key',
        '_key_prefix',
        '_metrics',
        '_nonce_factory',
        '_state',
        '_state_lock',
        '_validation_interval',
    )

    def __init__(  # noqa: PLR0913 - explicit validation dependencies remain injectable
        self,
        client: ControlClient,
        *,
        key_prefix: str,
        validation_seconds: float,
        clock: _Clock = monotonic,
        jitter_factory: _JitterFactory = jittered_interval,
        nonce_factory: _NonceFactory = lambda: token_hex(16),
        metrics: MetricsSink | None = None,
    ) -> None:
        """Initialize cached control validation without opening a connection.

        Args:
            client: One physical synchronous metadata-role client.
            key_prefix: Validated configured metadata key namespace.
            validation_seconds: Positive interval for cached validation.
            clock: Local monotonic clock used only to schedule validation.
            jitter_factory: Source of a positive jittered validation interval.
            nonce_factory: Cryptographic lowercase hexadecimal nonce source.
            metrics: Optional safe metrics destination.

        Raises:
            BackendProtocolError: The prefix or validation interval is invalid.
            BackendUnavailable: The jitter source fails or returns an invalid
                interval.
        """
        interval = validation_interval(key_prefix, validation_seconds, jitter_factory)
        self._client = client
        self._clock = clock
        self._control_key = control_key(key_prefix)
        self._key_prefix = key_prefix
        self._metrics = NullMetrics() if metrics is None else metrics
        self._nonce_factory = nonce_factory
        self._state = ValidationState(None, None, None, 0)
        self._state_lock = Lock()
        self._validation_interval = interval

    def _call(self, operation: Callable[[], object]) -> object:
        """Run one direct client command through the redacted boundary."""
        failure: Exception | None = None
        try:
            return operation()
        except Exception as error:  # noqa: BLE001 - closed external-client boundary
            failure = error
        translate_client_error(failure)
        raise BackendUnavailable('Valkey operation failed')

    def _clear_validation(self) -> None:
        """Forget locally cached validation after rotation or role drift."""
        with self._state_lock:
            current = self._state
            self._state = ValidationState(None, None, None, current.revision + 1)

    def _create(
        self,
        client: PinnedControlClient,
        replication_id: str,
    ) -> tuple[str, bytes | None]:
        """Atomically create absent control state or observe its winner."""
        incarnation = new_incarnation(self._key_prefix, replication_id, self._nonce_factory)
        operands = control_create_operands(self._control_key, incarnation)
        reply = execute_script(
            client,
            script=CONTROL_CREATE_SCRIPT,
            sha=CONTROL_CREATE_SCRIPT_SHA,
            keys=operands.keys,
            args=operands.args,
        )
        return decode_control_reply(reply)

    def _discard_pool(self) -> None:
        """Discard the complete maintained-client pool after role drift."""
        self._clear_validation()
        self._client.refresh()

    def _increment(self, outcome: str) -> None:
        """Emit one best-effort incarnation transition."""
        with suppress(Exception):
            self._metrics.increment('cache_incarnation', {'outcome': outcome})

    def _publish(self, snapshot: ValidationState, published: ValidationState) -> bool:
        """Install a prepared validation only if no transition happened since the snapshot."""
        with self._state_lock:
            if not can_publish_validation(snapshot, self._state):
                return False
            self._state = published
        return True

    def _read_control(self, client: PinnedControlClient) -> bytes | None:
        """Read the raw active control record."""
        raw = self._call(lambda: client.get(self._control_key))
        return decode_control_record(raw)

    def _is_active(self, client: PinnedControlClient, candidate: Incarnation) -> bool:
        """Recheck one complete control-record and counter relationship."""
        if self._read_control(client) != candidate.record:
            return False
        return decode_exists(self._call(lambda: client.exists(candidate.counter_key)))

    def _read_replication_id(self, client: PinnedControlClient) -> str:
        """Read and validate the writable primary's replication identity."""
        raw = self._call(lambda: client.info('replication'))
        return primary_replication_id(raw)

    def _refresh_endpoint(self) -> None:
        """Refresh Sentinel discovery through the maintained role client."""
        self._client.refresh_endpoint()

    def _rotate(
        self,
        client: PinnedControlClient,
        observed: bytes | None,
        replication_id: str,
    ) -> tuple[str, bytes | None]:
        """Compare-and-swap one complete observed record to fresh state."""
        incarnation = new_incarnation(self._key_prefix, replication_id, self._nonce_factory)
        operands = control_rotate_operands(self._control_key, observed, incarnation)
        reply = execute_script(
            client,
            script=CONTROL_ROTATE_SCRIPT,
            sha=CONTROL_ROTATE_SCRIPT_SHA,
            keys=operands.keys,
            args=operands.args,
        )
        return decode_control_reply(reply)

    def _candidate(
        self,
        client: PinnedControlClient,
        replication_id: str,
        observed: bytes | None,
    ) -> Incarnation | None:
        """Return locally valid or atomically created replacement state."""
        if observed is None:
            status, raw_candidate = self._create(client, replication_id)
            if status == 'created' and raw_candidate is not None:
                return parse_record(raw_candidate, self._key_prefix)
            return None
        parsed: Incarnation | None = None
        with suppress(BackendProtocolError):
            parsed = parse_record(observed, self._key_prefix)
        if parsed is not None and parsed.replication_id == replication_id:
            counter_key = parsed.counter_key
            if decode_exists(self._call(lambda: client.exists(counter_key))):
                return parsed
        status, raw_candidate = self._rotate(client, observed, replication_id)
        if status == 'rotated' and raw_candidate is not None:
            self._increment('rotated')
            return parse_record(raw_candidate, self._key_prefix)
        self._increment('mismatch')
        return None

    def _execute(
        self,
        client: PinnedControlClient,
        candidate: Incarnation,
        operation: Callable[[PinnedControlClient, Incarnation], _T],
    ) -> tuple[tuple[_T, ...], Exception | None, bool]:
        """Capture one operation outcome and any required rotation."""
        try:
            return (operation(client, candidate),), None, False
        except ScriptRotationError:
            pass
        except Exception as error:  # noqa: BLE001 - closed caller-operation boundary
            return (), error, False
        try:
            replication_id = self._read_replication_id(client)
            self._rotate(client, candidate.record, replication_id)
        except (BackendCapacity, BackendProtocolError, BackendUnavailable) as error:
            return (), error, False
        return (), None, True

    def _settle(self, client: PinnedControlClient, initial_epoch: RoleEpoch) -> None:
        """Settle one returned or raised operation outcome against its epoch."""
        failure: Exception | None = None
        try:
            self._refresh_endpoint()
        except (BackendProtocolError, BackendUnavailable) as error:
            failure = error
        if span_epoch(self._client.epoch(), client) != initial_epoch:
            self._discard_pool()
            raise BackendUnavailable('Valkey role changed during operation')
        if failure is not None:
            raise failure

    def _snapshot(self) -> ValidationState:
        """Read one consistent view of the cached validation."""
        with self._state_lock:
            return self._state

    def _validate(
        self,
        client: PinnedControlClient,
        force: bool,
        epoch: RoleEpoch,
    ) -> tuple[Incarnation, float | None, ValidationState]:
        """Build one candidate through an already pinned connection."""
        snapshot = self._snapshot()
        now = read_validation_time(self._clock)
        cached = (
            None if force else cached_incarnation(snapshot, epoch, now, self._validation_interval)
        )
        if cached is not None:
            return cached, None, snapshot

        for _attempt in range(_MAX_VALIDATION_ATTEMPTS):
            replication_id = self._read_replication_id(client)
            observed = self._read_control(client)
            candidate = self._candidate(client, replication_id, observed)
            if candidate is None:
                continue
            if self._read_replication_id(client) != replication_id:
                continue
            if not self._is_active(client, candidate):
                continue
            return candidate, now, snapshot
        raise BackendUnavailable('Valkey control state did not stabilize')

    def run(
        self,
        operation: Callable[[PinnedControlClient, Incarnation], _T],
        *,
        force: bool = False,
    ) -> _T:
        """Validate, execute, and settle one fenced metadata operation.

        Args:
            operation: Fenced operation to execute exactly once on the
                validated physical connection.
            force: Ignore a still-current local validation interval.

        Returns:
            The settled operation result.

        Raises:
            BackendCapacity: Metadata capacity rejects control-state creation
                or rotation.
            BackendProtocolError: Server control or epoch data is malformed.
            BackendUnavailable: Validation cannot stabilize or the role changes
                after operation execution.
            StaleIncarnation: The operation requires incarnation rotation.
        """
        for _attempt in range(_MAX_VALIDATION_ATTEMPTS):
            self._refresh_endpoint()
            drifted_before_operation = False
            with self._client.pin() as client:
                initial_epoch = span_epoch(self._client.epoch(), client)
                candidate, validated_at, snapshot = self._validate(client, force, initial_epoch)
                if span_epoch(self._client.epoch(), client) != initial_epoch:
                    drifted_before_operation = True
                else:
                    outcome, operation_failure, rotation_required = self._execute(
                        client,
                        candidate,
                        operation,
                    )
                    self._settle(client, initial_epoch)
                    if rotation_required:
                        self._increment('old_incarnation_rejected')
                        self._clear_validation()
                        raise StaleIncarnation('Valkey metadata incarnation rotated') from None
                    if operation_failure is not None:
                        if isinstance(operation_failure, StaleIncarnation):
                            self._increment('old_incarnation_rejected')
                            self._clear_validation()
                        raise operation_failure
                    if not outcome:
                        raise BackendUnavailable('Valkey operation did not return an outcome')
                    if validated_at is not None:
                        join = not force and validation_is_current(
                            snapshot, initial_epoch, validated_at, self._validation_interval
                        )
                        published = next_validation_state(
                            snapshot, candidate, initial_epoch, validated_at, join=join
                        )
                        if self._publish(snapshot, published):
                            self._increment('validated')
                    return outcome[0]
            if drifted_before_operation:
                self._discard_pool()
                continue
        raise BackendUnavailable('Valkey connection changed during validation')

    def validate(self, force: bool = False) -> Incarnation:
        """Return control state validated against the active metadata primary.

        Args:
            force: Ignore a still-current local validation interval.

        Returns:
            The active metadata incarnation.

        Raises:
            BackendCapacity: Metadata capacity rejects control-state creation
                or rotation.
            BackendProtocolError: Server control or replication data is
                malformed.
            BackendUnavailable: Validation cannot reach a stable writable
                primary.
        """
        return self.run(lambda _client, incarnation: incarnation, force=force)

    def rotate(self, observed: Incarnation) -> Incarnation:
        """Rotate the observed incarnation and adopt the winning state.

        Args:
            observed: Complete incarnation whose counter cannot continue.

        Returns:
            The freshly validated winning incarnation.

        Raises:
            BackendCapacity: Metadata capacity rejects control-state creation
                or rotation.
            BackendProtocolError: Control replies are malformed.
            BackendUnavailable: Primary validation cannot complete.
        """

        def rotate_active(client: PinnedControlClient, _active: Incarnation) -> bool:
            replication_id = self._read_replication_id(client)
            status, _candidate = self._rotate(client, observed.record, replication_id)
            return status == 'rotated'

        if self.run(rotate_active, force=True):
            self._increment('rotated')
        self._clear_validation()
        return self.validate(force=True)
