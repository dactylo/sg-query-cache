"""Validate and rotate Valkey metadata-primary control state."""

from asyncio import CancelledError
from collections.abc import Awaitable, Callable
from contextlib import AbstractAsyncContextManager, suppress
from math import floor
from secrets import token_hex
from time import monotonic
from typing import Protocol, TypeVar

from valkey.exceptions import NoScriptError

from sg_query_cache._async_support import LoopAffinity
from sg_query_cache._backend import (
    BackendCapacity,
    BackendProtocolError,
    BackendUnavailable,
    StaleIncarnation,
)
from sg_query_cache._metrics import MetricsSink, NullMetrics
from sg_query_cache._model import FillLease
from sg_query_cache._valkey_common import (
    Incarnation,
    RoleEpoch,
    ScriptArgument,
    ScriptRotationError,
    ValidationState,
    cached_incarnation,
    can_publish_validation,
    commit_operands,
    control_create_operands,
    control_key,
    control_rotate_operands,
    decode_commit_reply,
    decode_control_record,
    decode_control_reply,
    decode_digest,
    decode_exists,
    jittered_interval,
    new_incarnation,
    next_validation_state,
    parse_record,
    primary_replication_id,
    read_validation_time,
    require_loaded_digest,
    span_epoch,
    validation_interval,
    validation_is_current,
)
from sg_query_cache._valkey_scripts import (
    COMMIT_SCRIPT,
    COMMIT_SCRIPT_SHA,
    CONTROL_CREATE_SCRIPT,
    CONTROL_CREATE_SCRIPT_SHA,
    CONTROL_ROTATE_SCRIPT,
    CONTROL_ROTATE_SCRIPT_SHA,
    translate_client_error,
)

_T = TypeVar('_T')

_MAX_VALIDATION_ATTEMPTS = 4


class ScriptClient(Protocol):
    """Execute and load cached scripts on one asynchronous connection."""

    async def evalsha(self, sha: str, numkeys: int, *args: ScriptArgument) -> object:
        """Execute one cached script with ordered keys and arguments."""
        ...

    async def script_load(self, script: bytes | str) -> object:
        """Load one exact script and return its digest."""
        ...


class PinnedControlClient(ScriptClient, Protocol):
    """Read the commands needed to validate writable primary state."""

    async def exists(self, key: str) -> object:
        """Return whether one key exists."""
        ...

    async def get(self, key: str) -> object:
        """Return one raw string value."""
        ...

    async def info(self, section: str) -> object:
        """Return one server information section."""
        ...


class ResultClient(Protocol):
    """Run result work on one prepared pin through role settlement."""

    async def run_operation(self, operation: Callable[[ScriptClient], Awaitable[_T]]) -> _T:
        """Execute result work within one observed physical-connection span."""
        ...


class ControlClient(PinnedControlClient, Protocol):
    """Pin all control validation commands to one physical connection."""

    def epoch(self) -> RoleEpoch:
        """Return the current pure role epoch."""
        ...

    def pin(self) -> AbstractAsyncContextManager[PinnedControlClient]:
        """Borrow one physical connection for a complete validation."""
        ...

    async def refresh(self) -> None:
        """Discard the current pool and construct a fresh connection."""
        ...

    async def refresh_endpoint(self) -> None:
        """Explicitly refresh Sentinel discovery when applicable."""
        ...


class AsyncValkeyControl:
    """Validate one metadata primary and its control record.

    Each call pins a physical connection and publishes cached validation only
    if its revision remains current. The first call binds the owning event
    loop.
    """

    __slots__ = (
        '_affinity',
        '_client',
        '_clock',
        '_control_key',
        '_key_prefix',
        '_metrics',
        '_nonce_factory',
        '_state',
        '_validation_interval',
    )

    def __init__(  # noqa: PLR0913 - explicit validation dependencies remain injectable
        self,
        client: ControlClient,
        *,
        key_prefix: str,
        validation_seconds: float,
        clock: Callable[[], float] = monotonic,
        jitter_factory: Callable[[float], float] = jittered_interval,
        nonce_factory: Callable[[], str] = lambda: token_hex(16),
        metrics: MetricsSink | None = None,
    ) -> None:
        """Initialize loop-owned validation with prompt injectable sources.

        Args:
            client: Metadata role providing an asynchronous physical pin.
            key_prefix: Validated configured metadata key namespace.
            validation_seconds: Positive nominal cached-validation interval in
                seconds;
                the default jitter varies it by up to ten percent.
            clock: Local monotonic clock used only to schedule validation.
            jitter_factory: Source of a positive jittered validation interval.
            nonce_factory: Cryptographic lowercase hexadecimal nonce source.
            metrics: Optional synchronous telemetry destination.

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
        self._affinity = LoopAffinity()
        self._validation_interval = interval

    async def _call(self, operation: Callable[[], Awaitable[object]]) -> object:
        """Run one direct client command through the redacted boundary."""
        failure: Exception | None = None
        try:
            return await operation()
        except Exception as error:  # noqa: BLE001 - closed external-client boundary
            failure = error
        translate_client_error(failure)
        raise BackendUnavailable('Valkey operation failed')

    def _clear_validation(self) -> None:
        """Forget locally cached validation after rotation or role drift."""
        current = self._state
        self._state = ValidationState(None, None, None, current.revision + 1)

    async def _create(
        self,
        client: PinnedControlClient,
        replication_id: str,
    ) -> tuple[str, bytes | None]:
        """Atomically create absent control state or observe its winner."""
        incarnation = new_incarnation(self._key_prefix, replication_id, self._nonce_factory)
        operands = control_create_operands(self._control_key, incarnation)
        reply = await execute_script(
            client,
            script=CONTROL_CREATE_SCRIPT,
            sha=CONTROL_CREATE_SCRIPT_SHA,
            keys=operands.keys,
            args=operands.args,
        )
        return decode_control_reply(reply)

    async def _discard_pool(self) -> None:
        """Discard the complete maintained-client pool after role drift."""
        self._clear_validation()
        await self._client.refresh()

    def _increment(self, outcome: str) -> None:
        """Emit one best-effort incarnation transition."""
        with suppress(Exception):
            self._metrics.increment('cache_incarnation', {'outcome': outcome})

    def _publish(self, snapshot: ValidationState, published: ValidationState) -> bool:
        """Install a prepared validation only if no transition happened since the snapshot."""
        if not can_publish_validation(snapshot, self._state):
            return False
        self._state = published
        return True

    async def _read_control(self, client: PinnedControlClient) -> bytes | None:
        """Read the raw active control record."""
        raw = await self._call(lambda: client.get(self._control_key))
        return decode_control_record(raw)

    async def _is_active(self, client: PinnedControlClient, candidate: Incarnation) -> bool:
        """Recheck one complete control-record and counter relationship."""
        if await self._read_control(client) != candidate.record:
            return False
        return decode_exists(await self._call(lambda: client.exists(candidate.counter_key)))

    async def _read_replication_id(self, client: PinnedControlClient) -> str:
        """Read and validate the writable primary's replication identity."""
        raw = await self._call(lambda: client.info('replication'))
        return primary_replication_id(raw)

    async def _refresh_endpoint(self) -> None:
        """Refresh Sentinel discovery explicitly when the client supports it."""
        await self._client.refresh_endpoint()

    async def _rotate(
        self,
        client: PinnedControlClient,
        observed: bytes | None,
        replication_id: str,
    ) -> tuple[str, bytes | None]:
        """Compare-and-swap one complete observed record to fresh state."""
        incarnation = new_incarnation(self._key_prefix, replication_id, self._nonce_factory)
        operands = control_rotate_operands(self._control_key, observed, incarnation)
        reply = await execute_script(
            client,
            script=CONTROL_ROTATE_SCRIPT,
            sha=CONTROL_ROTATE_SCRIPT_SHA,
            keys=operands.keys,
            args=operands.args,
        )
        return decode_control_reply(reply)

    async def _candidate(
        self,
        client: PinnedControlClient,
        replication_id: str,
        observed: bytes | None,
    ) -> Incarnation | None:
        """Return locally valid or atomically created replacement state."""
        if observed is None:
            status, raw_candidate = await self._create(client, replication_id)
            if status == 'created' and raw_candidate is not None:
                return parse_record(raw_candidate, self._key_prefix)
            return None
        parsed: Incarnation | None = None
        with suppress(BackendProtocolError):
            parsed = parse_record(observed, self._key_prefix)
        if parsed is not None and parsed.replication_id == replication_id:
            counter_key = parsed.counter_key
            if decode_exists(await self._call(lambda: client.exists(counter_key))):
                return parsed
        status, raw_candidate = await self._rotate(client, observed, replication_id)
        if status == 'rotated' and raw_candidate is not None:
            self._increment('rotated')
            return parse_record(raw_candidate, self._key_prefix)
        self._increment('mismatch')
        return None

    async def _execute(
        self,
        client: PinnedControlClient,
        candidate: Incarnation,
        operation: Callable[[PinnedControlClient, Incarnation], Awaitable[_T]],
    ) -> tuple[tuple[_T, ...], Exception | None, bool]:
        """Capture one operation outcome and any required rotation."""
        try:
            return (await operation(client, candidate),), None, False
        except ScriptRotationError:
            pass
        except Exception as error:  # noqa: BLE001 - closed caller-operation boundary
            return (), error, False
        try:
            replication_id = await self._read_replication_id(client)
            await self._rotate(client, candidate.record, replication_id)
        except (BackendCapacity, BackendProtocolError, BackendUnavailable) as error:
            return (), error, False
        return (), None, True

    async def _settle(self, client: PinnedControlClient, initial_epoch: RoleEpoch) -> None:
        """Settle one returned or raised operation outcome against its epoch."""
        failure: Exception | None = None
        try:
            await self._refresh_endpoint()
        except (BackendProtocolError, BackendUnavailable) as error:
            failure = error
        if span_epoch(self._client.epoch(), client) != initial_epoch:
            await self._discard_pool()
            raise BackendUnavailable('Valkey role changed during operation')
        if failure is not None:
            raise failure

    def _snapshot(self) -> ValidationState:
        """Read one consistent view of the cached validation."""
        return self._state

    async def _validate(
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
            replication_id = await self._read_replication_id(client)
            observed = await self._read_control(client)
            candidate = await self._candidate(client, replication_id, observed)
            if candidate is None:
                continue
            if await self._read_replication_id(client) != replication_id:
                continue
            if not await self._is_active(client, candidate):
                continue
            return candidate, now, snapshot
        raise BackendUnavailable('Valkey control state did not stabilize')

    async def _run(
        self,
        operation: Callable[[PinnedControlClient, Incarnation], Awaitable[_T]],
        *,
        force: bool = False,
    ) -> _T:
        """Execute one complete validated delivery span without mutation retry."""
        self._affinity.check()
        for _attempt in range(_MAX_VALIDATION_ATTEMPTS):
            await self._refresh_endpoint()
            drifted_before_operation = False
            async with self._client.pin() as client:
                initial_epoch = span_epoch(self._client.epoch(), client)
                candidate, validated_at, snapshot = await self._validate(
                    client, force, initial_epoch
                )
                if span_epoch(self._client.epoch(), client) != initial_epoch:
                    drifted_before_operation = True
                else:
                    outcome, operation_failure, rotation_required = await self._execute(
                        client,
                        candidate,
                        operation,
                    )
                    await self._settle(client, initial_epoch)
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
                await self._discard_pool()
                continue
        raise BackendUnavailable('Valkey connection changed during validation')

    async def run(
        self,
        operation: Callable[[PinnedControlClient, Incarnation], Awaitable[_T]],
        *,
        force: bool = False,
    ) -> _T:
        """Validate, execute once, and invalidate interrupted control observations.

        Args:
            operation: Fenced coroutine executed on the validated physical pin.
            force: Ignore the cached validation interval.

        Returns:
            The operation's result after epoch settlement and pin release.

        Raises:
            BackendCapacity: Metadata capacity rejects control-state creation
                or rotation.
            BackendProtocolError: Server control or epoch data is malformed.
            BackendUnavailable: Validation cannot stabilize or the role changes
                after operation execution.
            StaleIncarnation: The operation requires incarnation rotation.
            RuntimeError: The call uses a different event loop from the owner.
        """
        self._affinity.check()
        revision = self._state.revision
        try:
            return await self._run(operation, force=force)
        except CancelledError:
            self._clear_validation()
            raise
        except BackendUnavailable:
            if self._state.revision == revision:
                self._clear_validation()
            raise

    async def _validate_incarnation(self, force: bool = False) -> Incarnation:
        """Return control state validated against the active metadata primary."""

        async def identity(_client: PinnedControlClient, incarnation: Incarnation) -> Incarnation:
            return incarnation

        return await self.run(identity, force=force)

    async def _rotate_incarnation(self, observed: Incarnation) -> Incarnation:
        """Rotate the observed incarnation and adopt the winning state."""

        async def rotate_active(client: PinnedControlClient, _active: Incarnation) -> bool:
            replication_id = await self._read_replication_id(client)
            status, _candidate = await self._rotate(client, observed.record, replication_id)
            return status == 'rotated'

        if await self.run(rotate_active, force=True):
            self._increment('rotated')
        self._clear_validation()
        return await self._validate_incarnation(force=True)


async def execute_commit(  # noqa: PLR0913 - explicit commit deadline inputs
    client: ResultClient,
    *,
    prefix: str,
    lease: FillLease,
    payload: bytes,
    deadline: float,
    clock: Callable[[], float],
) -> bool:
    """Commit on one prepared pin, debiting preparation and one script load.

    Args:
        client: Result role that observes the complete operation span.
        prefix: Configured result-role key prefix.
        lease: Exact fill ownership to compare atomically.
        payload: Original encoded result envelope, retained unchanged.
        deadline: Expiration anchored at backend commit entry in `clock` time.
        clock: Backend monotonic clock used immediately before each attempt.

    Returns:
        Whether the live lease admitted the result before deadline exhaustion.

    Raises:
        BackendCapacity: Valkey rejects a command for capacity.
        BackendProtocolError: A command, loaded digest, or reply is malformed.
        BackendUnavailable: The client cannot complete or settle the operation.
    """

    async def operation(pinned: ScriptClient) -> bool:
        async def attempt() -> bool:
            ttl = floor((deadline - clock()) * 1000)
            if ttl <= 0:
                return False
            operands = commit_operands(prefix, lease, payload, ttl)
            reply = await pinned.evalsha(
                COMMIT_SCRIPT_SHA,
                len(operands.keys),
                *operands.keys,
                *operands.args,
            )
            return decode_commit_reply(reply)

        try:
            return await attempt()
        except NoScriptError:
            pass
        loaded = decode_digest(await pinned.script_load(COMMIT_SCRIPT))
        require_loaded_digest(loaded, COMMIT_SCRIPT_SHA)
        try:
            return await attempt()
        except NoScriptError:
            pass
        raise BackendUnavailable('Valkey script cache remained unavailable') from None

    failure: Exception | None = None
    try:
        return await client.run_operation(operation)
    except (BackendCapacity, BackendProtocolError, BackendUnavailable, StaleIncarnation):
        raise
    except Exception as error:  # noqa: BLE001 - closed external-client boundary
        failure = error
    translate_client_error(failure)
    raise BackendUnavailable('Valkey operation failed')


async def execute_script(
    client: ScriptClient,
    *,
    script: str,
    sha: str,
    keys: tuple[str, ...],
    args: tuple[ScriptArgument, ...],
) -> object:
    """Execute one script, allowing only one safe load after `NOSCRIPT`.

    Args:
        client: Loop-owned role or pinned client.
        script: Exact Lua source bound to the supplied digest.
        sha: Expected lowercase SHA-1 digest.
        keys: Ordered keys preceding script arguments.
        args: Ordered script arguments.

    Returns:
        Raw reply for operation-specific decoding.

    Raises:
        BackendCapacity: Valkey rejects a command for capacity.
        BackendProtocolError: A command, loaded digest, or reply is malformed.
        BackendUnavailable: The client cannot complete or settle the operation.
    """
    failure: Exception | None = None
    try:
        return await client.evalsha(sha, len(keys), *keys, *args)
    except NoScriptError:
        pass
    except (BackendCapacity, BackendProtocolError, BackendUnavailable, StaleIncarnation):
        raise
    except Exception as error:  # noqa: BLE001 - closed external-client boundary
        failure = error
    if failure is not None:
        translate_client_error(failure)
    try:
        loaded = decode_digest(await client.script_load(script))
        require_loaded_digest(loaded, sha)
        return await client.evalsha(sha, len(keys), *keys, *args)
    except NoScriptError:
        raise BackendUnavailable('Valkey script cache remained unavailable') from None
    except (BackendCapacity, BackendProtocolError, BackendUnavailable, StaleIncarnation):
        raise
    except Exception as error:  # noqa: BLE001 - closed external-client boundary
        failure = error
    translate_client_error(failure)
    raise BackendUnavailable('Valkey operation failed')
