"""Share pure Valkey records, reply validation, and script operands."""

from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass, replace
from math import floor, isfinite
from secrets import randbelow
from time import monotonic
from typing import TypeAlias, cast

from sg_query_cache._backend import (
    GLOBAL_TAG_KINDS,
    BackendCapacity,
    BackendProtocolError,
    BackendUnavailable,
    StaleIncarnation,
)
from sg_query_cache._backend_validation import (
    require_vector_byte_bound,
    validate_vector_tags,
    vector_bytes,
)
from sg_query_cache._codec import CodecError, decode_value
from sg_query_cache._config import VALKEY_MAX_DEPENDENCY_TAGS_V1, CacheSettings, ValkeyRoleSettings
from sg_query_cache._keys import KeyBuilder
from sg_query_cache._metrics import MetricsSink
from sg_query_cache._model import CacheHit, DecodedPayload, DependencyTag, FillLease

RoleEpoch: TypeAlias = tuple[int, int, object | None]
ScriptArgument: TypeAlias = bytes | int | str

_MAX_COUNTER = 2**63 - 1
_MAX_REPLICATION_ID_LENGTH = 128
_MAX_STATUS_LENGTH = 32
_NONCE_LENGTH = 32
_RECORD_PART_COUNT = 3
_SCRIPT_DIGEST_LENGTH = 40
_THREE_ITEM_REPLY = 3
_TWO_ITEM_REPLY = 2


class CircuitBreaker:
    """Coordinate prompt breaker transitions under synchronization owned by the caller."""

    __slots__ = (
        '_clock',
        '_failures',
        '_metrics',
        '_opened_at',
        '_probe_seconds',
        '_role',
        '_state',
        '_threshold',
    )

    def __init__(
        self,
        role: str,
        *,
        threshold: int,
        probe_seconds: float,
        metrics: MetricsSink,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        """Initialize breaker counters and transition telemetry.

        Args:
            role: Low-cardinality role label used for transition metrics.
            threshold: Consecutive failures required to open the circuit.
            probe_seconds: Delay before an open circuit admits one probe.
            metrics: Best-effort transition metrics sink.
            clock: Monotonic seconds used to schedule the next probe.
        """
        self._clock = clock
        self._failures = 0
        self._metrics = metrics
        self._opened_at: float | None = None
        self._probe_seconds = probe_seconds
        self._role = role
        self._state = 'closed'
        self._threshold = threshold

    def _record(self, outcome: str) -> None:
        """Emit one low-cardinality state transition."""
        with suppress(Exception):
            self._metrics.increment(
                'backend_failure',
                {'backend_operation': self._role, 'outcome': outcome},
            )

    def admit(self) -> bool:
        """Admit one operation and report whether it must use a fresh pool.

        Raises:
            BackendUnavailable: The circuit is open or another probe is active.
        """
        if self._state == 'closed':
            return False
        now = self._clock()
        if (
            self._state == 'open'
            and self._opened_at is not None
            and now - self._opened_at >= self._probe_seconds
        ):
            self._state = 'half_open'
            self._record('half_open')
            return True
        raise BackendUnavailable('Valkey circuit breaker is open')

    def fail(self) -> None:
        """Count one backend failure or reopen a failed probe."""
        if self._state == 'half_open':
            self._state = 'open'
            self._opened_at = self._clock()
            self._record('reopened')
            return
        if self._state == 'open':
            return
        self._failures += 1
        if self._failures >= self._threshold:
            self._state = 'open'
            self._opened_at = self._clock()
            self._record('open')

    def succeed(self) -> None:
        """Close a successful probe and reset consecutive failures."""
        transitioned = self._state == 'half_open'
        self._failures = 0
        self._opened_at = None
        self._state = 'closed'
        if transitioned:
            self._record('closed')


@dataclass(frozen=True, slots=True)
class LookupOutcome:
    """Carry a decoded hit or miss reason and its optional expiration.

    Attributes:
        hit: Result hit, or `None` for a miss.
        reason: Bounded metric reason for a miss, or `None` for a hit.
        deadline: Expiration in the caller's monotonic clock, when known.
    """

    hit: CacheHit | None
    reason: str | None
    deadline: float | None = None


class ScriptRotationError(RuntimeError):
    """Report that generation allocation requires incarnation rotation."""


@dataclass(frozen=True, slots=True)
class Incarnation:
    """Identify one validated metadata replication history and token counter.

    Attributes:
        replication_id: Live metadata-primary replication identity.
        nonce: Random lowercase hexadecimal incarnation namespace.
        counter_key: Incarnation-scoped generation counter key.
        record: Complete compare-and-swap control record.
    """

    replication_id: str
    nonce: str
    counter_key: str
    record: bytes

    def __post_init__(self) -> None:
        """Validate the replication identity and exact serialized control record.

        Raises:
            BackendProtocolError: The identity or counter key is malformed, or
                the record disagrees with those fields.
            BackendUnavailable: The incarnation nonce is not 128-bit lowercase
                hexadecimal.
        """
        _validate_replication_id(self.replication_id)
        _validate_nonce(self.nonce)
        if not self.counter_key or not self.counter_key.isascii() or '|' in self.counter_key:
            raise BackendProtocolError('Valkey counter key is malformed')
        if self.record != _record_bytes(self.replication_id, self.nonce, self.counter_key):
            raise BackendProtocolError('Valkey control record is malformed')


@dataclass(frozen=True, slots=True)
class ValidationState:
    """Hold one immutable snapshot of locally cached validation.

    Attributes:
        incarnation: Validated incarnation, or `None` after clearing.
        role_epoch: Epoch associated with the incarnation, or `None`.
        validated_at: Monotonic validation time in seconds, or `None`.
        revision: Local revision advanced by each publication or clear.
        covered: Connection values covered by this validation interval.
    """

    incarnation: Incarnation | None
    role_epoch: RoleEpoch | None
    validated_at: float | None
    revision: int
    covered: frozenset[int] = frozenset()


@dataclass(frozen=True, slots=True)
class ScriptOperands:
    """Hold ordered keys and arguments for one atomic script invocation."""

    keys: tuple[str, ...]
    args: tuple[ScriptArgument, ...]


def _counter_key(prefix: str, nonce: str) -> str:
    """Build the version 1 incarnation counter key."""
    return f'{prefix}:v1:counter:{nonce}'


def _decode_bytes(value: object, owner: str) -> bytes:
    """Require one immutable byte-string response value."""
    if not isinstance(value, bytes):
        raise BackendProtocolError(f'Valkey {owner} reply is malformed')
    return value


def _decode_counter(value: object) -> int:
    """Require one positive signed 63-bit generation token."""
    if isinstance(value, bool):
        raise BackendProtocolError('Valkey script reply is malformed')
    token: int | None = None
    if isinstance(value, int):
        token = value
    elif isinstance(value, bytes | str):
        if isinstance(value, bytes):
            text = None
            with suppress(UnicodeDecodeError):
                text = value.decode('ascii')
            if text is None:
                raise BackendProtocolError('Valkey script reply is malformed')
        else:
            text = value
        if text and text.isascii() and text.isdecimal():
            try:
                token = int(text)
            except (ValueError, OverflowError):
                token = None
    if token is None or token <= 0 or token > _MAX_COUNTER:
        raise BackendProtocolError('Valkey script reply is malformed')
    return token


def _decode_nonnegative_int(value: object) -> int:
    """Require one nonnegative integer response field."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise BackendProtocolError('Valkey script reply is malformed')
    return value


def _decode_role_epoch(value: object) -> RoleEpoch:
    """Return a pool and connection epoch with an opaque endpoint identity."""
    if not isinstance(value, tuple):
        raise BackendProtocolError('Valkey role epoch is malformed')
    items = cast('tuple[object, ...]', value)
    if len(items) != _RECORD_PART_COUNT:
        raise BackendProtocolError('Valkey role epoch is malformed')
    pool_epoch, connection_generation, endpoint = items
    if (
        isinstance(pool_epoch, bool)
        or not isinstance(pool_epoch, int)
        or pool_epoch < 0
        or isinstance(connection_generation, bool)
        or not isinstance(connection_generation, int)
        or connection_generation < 0
    ):
        raise BackendProtocolError('Valkey role epoch is malformed')
    return pool_epoch, connection_generation, endpoint


def _decode_status(value: object) -> str:
    """Require one short ASCII script status."""
    if isinstance(value, bytes):
        try:
            value = value.decode('ascii')
        except UnicodeDecodeError:
            value = None
    if (
        not isinstance(value, str)
        or not value
        or not value.isascii()
        or len(value) > _MAX_STATUS_LENGTH
    ):
        raise BackendProtocolError('Valkey script reply is malformed')
    return value


def _lease_key(prefix: str, key: str) -> str:
    """Map one opaque logical lease key into the configured namespace."""
    return f'{prefix}:v1:lease:{key}'


def _raise_metadata_status(status: str, item_count: int) -> None:
    """Translate a one-element metadata failure status."""
    if item_count != 1:
        return
    if status in ('missing-counter', 'stale'):
        raise StaleIncarnation('Valkey metadata incarnation is stale')
    if status == 'capacity':
        raise BackendCapacity('Valkey metadata capacity rejected the operation')
    if status == 'rotation-required':
        raise ScriptRotationError('Valkey generation counter requires rotation')
    if status == 'too-many-tags':
        raise BackendProtocolError('dependency vector exceeds the tag bound')


def _record_bytes(replication_id: str, nonce: str, counter_key: str) -> bytes:
    """Encode one validated complete compare-and-swap record."""
    return f'{replication_id}|{nonce}|{counter_key}'.encode('ascii')


def _reply_items(reply: object) -> tuple[object, ...]:
    """Require one nonempty array response."""
    if not isinstance(reply, list | tuple) or not reply:
        raise BackendProtocolError('Valkey script reply is malformed')
    return tuple(cast('list[object] | tuple[object, ...]', reply))


def _decode_binary_status(
    reply: object,
    *,
    accepted: str,
    rejected: str,
    owner: str,
) -> bool:
    """Decode a one-element accepted-or-rejected response."""
    items = _reply_items(reply)
    if len(items) != 1:
        raise BackendProtocolError(f'Valkey {owner} reply is malformed')
    status = _decode_status(items[0])
    if status == accepted:
        return True
    if status == rejected:
        return False
    raise BackendProtocolError(f'Valkey {owner} reply is malformed')


def _decode_lookup_reply(reply: object) -> tuple[bytes, int] | None:
    """Decode an atomic result value and its authoritative TTL."""
    items = _reply_items(reply)
    status = _decode_status(items[0])
    if status == 'miss' and len(items) == 1:
        return None
    if status == 'hit' and len(items) == _THREE_ITEM_REPLY:
        ttl = _decode_nonnegative_int(items[2])
        if ttl == 0:
            raise BackendProtocolError('Valkey lookup reply is malformed')
        return _decode_bytes(items[1], 'lookup'), ttl
    raise BackendProtocolError('Valkey lookup reply is malformed')


def _result_key(prefix: str, key: str) -> str:
    """Map one opaque logical result key into the configured namespace."""
    return f'{prefix}:v1:result:{key}'


def _same_role(left: RoleEpoch, right: RoleEpoch) -> bool:
    """Compare pool and endpoint components, ignoring the connection component."""
    return left[0] == right[0] and left[2] == right[2]


def _tag_key(prefix: str, incarnation: str, tag: DependencyTag) -> str:
    """Build one incarnation-scoped opaque metadata tag key."""
    return f'{prefix}:v1:metadata:{incarnation}:{KeyBuilder.tag_key(tag)}'


def _validate_nonce(value: object) -> str:
    """Require a 128-bit lowercase hexadecimal incarnation nonce."""
    if (
        not isinstance(value, str)
        or len(value) != _NONCE_LENGTH
        or value.lower() != value
        or any(character not in '0123456789abcdef' for character in value)
    ):
        raise BackendUnavailable('Valkey incarnation entropy is unavailable')
    return value


def _validate_replication_id(value: object) -> str:
    """Require one nonempty printable ASCII replication identity."""
    if (
        not isinstance(value, str)
        or not value
        or len(value) > _MAX_REPLICATION_ID_LENGTH
        or not value.isascii()
        or any(not character.isprintable() or character in '|\r\n' for character in value)
    ):
        raise BackendProtocolError('Valkey replication identity is malformed')
    return value


def can_publish_validation(snapshot: ValidationState, current: ValidationState) -> bool:
    """Accept publication only when no newer publication or clear changed the revision."""
    return current.revision == snapshot.revision


def commit_operands(prefix: str, lease: FillLease, payload: bytes, ttl: int) -> ScriptOperands:
    """Build token-checked commit operands with the lease key before the result key.

    Args:
        prefix: Configured result-role key namespace.
        lease: Validated fill ownership.
        payload: Encoded result bytes.
        ttl: Positive result lifetime in milliseconds.
    """
    return ScriptOperands(
        (_lease_key(prefix, lease.key), _result_key(prefix, lease.key)),
        (lease.token, payload, ttl),
    )


def control_key(prefix: str) -> str:
    """Build the version 1 metadata control key."""
    return f'{prefix}:v1:control'


def control_create_operands(control_key: str, incarnation: Incarnation) -> ScriptOperands:
    """Build create-if-absent operands for a complete fresh incarnation."""
    return ScriptOperands((control_key, incarnation.counter_key), (incarnation.record,))


def control_rotate_operands(
    control_key: str,
    observed: bytes | None,
    incarnation: Incarnation,
) -> ScriptOperands:
    """Build compare-and-swap operands, encoding an absent observed record as empty bytes."""
    return ScriptOperands(
        (control_key, incarnation.counter_key),
        (b'' if observed is None else observed, incarnation.record),
    )


def decode_change_reply(reply: object) -> tuple[int, int]:
    """Return the replaced and absent counts from a generation-change reply.

    Raises:
        BackendCapacity: Metadata allocation was rejected for capacity.
        StaleIncarnation: The script observed a different incarnation.
        ScriptRotationError: The generation counter requires rotation.
        BackendProtocolError: The reply or expected result shape is malformed.
    """
    items = _reply_items(reply)
    status = _decode_status(items[0])
    if status == 'ok' and len(items) == _THREE_ITEM_REPLY:
        return _decode_nonnegative_int(items[1]), _decode_nonnegative_int(items[2])
    _raise_metadata_status(status, len(items))
    raise BackendProtocolError('Valkey change reply is malformed')


def decode_commit_reply(reply: object) -> bool:
    """Return true for committed, false for capacity or rejected, and reject other replies.

    Raises:
        BackendProtocolError: The reply has an unsupported shape or value.
    """
    items = _reply_items(reply)
    if len(items) != 1:
        raise BackendProtocolError('Valkey commit reply is malformed')
    status = _decode_status(items[0])
    if status == 'committed':
        return True
    if status in ('capacity', 'rejected'):
        return False
    raise BackendProtocolError('Valkey commit reply is malformed')


def decode_control_record(raw: object) -> bytes | None:
    """Return bytes unchanged, encode ASCII text, or preserve an absent record.

    Raises:
        BackendProtocolError: The value is neither bytes, text, nor `None`, or
            text contains non-ASCII characters.
    """
    if raw is None:
        return None
    if isinstance(raw, str):
        try:
            return raw.encode('ascii')
        except UnicodeEncodeError:
            raise BackendProtocolError('Valkey control record is malformed') from None
    if not isinstance(raw, bytes):
        raise BackendProtocolError('Valkey control record is malformed')
    return raw


def decode_control_reply(reply: object) -> tuple[str, bytes | None]:
    """Decode a control-state create or compare-and-swap reply.

    Returns:
        A `created`, `rotated`, or `winner` status with record bytes, or
        `("retry", None)` when the compare-and-swap must be retried.

    Raises:
        BackendCapacity: Metadata capacity rejected control-state storage.
        BackendProtocolError: The status, record, or reply shape is malformed.
    """
    items = _reply_items(reply)
    status = _decode_status(items[0])
    if status in ('created', 'rotated', 'winner') and len(items) == _TWO_ITEM_REPLY:
        return status, _decode_bytes(items[1], 'control')
    if status == 'retry' and len(items) == 1:
        return status, None
    if status == 'capacity' and len(items) == 1:
        raise BackendCapacity('Valkey metadata capacity rejected control state')
    raise BackendProtocolError('Valkey control reply is malformed')


def decode_digest(value: object) -> str:
    """Return a lowercase SHA-1 digest from ASCII bytes or text.

    Raises:
        BackendProtocolError: The reply has an unsupported shape or value.
    """
    if isinstance(value, bytes):
        try:
            value = value.decode('ascii')
        except UnicodeDecodeError:
            value = None
    if (
        not isinstance(value, str)
        or len(value) != _SCRIPT_DIGEST_LENGTH
        or value.lower() != value
        or any(character not in '0123456789abcdef' for character in value)
    ):
        raise BackendProtocolError('Valkey script-load reply is malformed')
    return value


def decode_exists(value: object) -> bool:
    """Return whether an exact integer EXISTS reply is one; reject booleans.

    Raises:
        BackendProtocolError: The reply has an unsupported shape or value.
    """
    if isinstance(value, bool) or not isinstance(value, int) or value not in (0, 1):
        raise BackendProtocolError('Valkey existence reply is malformed')
    return value == 1


def decode_release_reply(reply: object) -> bool:
    """Return true for released and false for retained lease replies.

    Raises:
        BackendProtocolError: The reply has an unsupported shape or value.
    """
    return _decode_binary_status(reply, accepted='released', rejected='retained', owner='release')


def decode_reserve_reply(reply: object) -> bool:
    """Return true for reserved and false for busy lease replies.

    Raises:
        BackendProtocolError: The reply has an unsupported shape or value.
    """
    return _decode_binary_status(reply, accepted='reserved', rejected='busy', owner='reserve')


def decode_resolve_reply(reply: object, *, expected_count: int) -> tuple[int, ...]:
    """Return an ordered generation vector with exactly the expected token count.

    Args:
        reply: Raw atomic resolve-script response.
        expected_count: Nonnegative integer number of requested tags.

    Raises:
        BackendCapacity: Metadata allocation was rejected for capacity.
        StaleIncarnation: The script observed a different incarnation.
        ScriptRotationError: The generation counter requires rotation.
        BackendProtocolError: The reply or expected result shape is malformed.
    """
    if isinstance(expected_count, bool) or expected_count < 0:
        raise BackendProtocolError('expected generation count is invalid')
    items = _reply_items(reply)
    status = _decode_status(items[0])
    if status == 'ok' and len(items) == expected_count + 1:
        return tuple(_decode_counter(value) for value in items[1:])
    _raise_metadata_status(status, len(items))
    raise BackendProtocolError('Valkey resolve reply is malformed')


def finalize_lookup(outcome: LookupOutcome, clock: Callable[[], float]) -> LookupOutcome:
    """Debit post-reply guard processing before returning one lookup outcome."""
    hit = outcome.hit
    deadline = outcome.deadline
    if hit is None or deadline is None:
        return outcome
    remaining_seconds = deadline - clock()
    if remaining_seconds <= 0:
        return LookupOutcome(None, 'expired')
    return replace(outcome, hit=replace(hit, remaining_seconds=remaining_seconds))


def interpret_lookup(
    reply: object, maximum_bytes: int, started_at: float, clock: Callable[[], float]
) -> LookupOutcome:
    """Decode a lookup reply and debit its lifetime through the supplied clock.

    Args:
        reply: Raw atomic lookup response.
        maximum_bytes: Maximum accepted serialized result size.
        started_at: Monotonic time before discovery and command delivery.
        clock: Same monotonic clock used for `started_at`.

    Returns:
        Decoded hit with remaining lifetime and absolute deadline, or a bounded
        miss reason.

    Raises:
        BackendProtocolError: The script response has an invalid shape.
    """
    decoded = _decode_lookup_reply(reply)
    if decoded is None:
        return LookupOutcome(None, 'absence')
    payload, ttl_milliseconds = decoded
    if len(payload) > maximum_bytes:
        return LookupOutcome(None, 'oversized')
    try:
        value = decode_value(payload)
    except CodecError:
        return LookupOutcome(None, 'malformed')
    deadline = started_at + ttl_milliseconds / 1000
    remaining_seconds = deadline - clock()
    if remaining_seconds <= 0:
        return LookupOutcome(None, 'expired')
    return LookupOutcome(
        CacheHit(payload, remaining_seconds, decoded=DecodedPayload(value)),
        None,
        deadline,
    )


def jittered_interval(seconds: float) -> float:
    """Return an independent interval within ten percent of the nominal seconds."""
    return seconds * (0.9 + randbelow(2001) / 10_000)


def lookup_operands(prefix: str, key: str) -> ScriptOperands:
    """Build atomic result lookup operands in the configured namespace."""
    return ScriptOperands((_result_key(prefix, key),), ())


def milliseconds(value: object, owner: str) -> int:
    """Floor seconds to a positive signed 64-bit count of milliseconds.

    Args:
        value: Finite non-boolean integer or float duration in seconds.
        owner: Safe operation label used in validation errors.

    Returns:
        The floored duration, between 1 and 2**63 - 1 milliseconds.

    Raises:
        BackendProtocolError: The value has the wrong type, is non-finite,
            rounds below one millisecond, or exceeds the integer bound.
    """
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise BackendProtocolError(f'{owner} must be finite')
    try:
        seconds = float(value)
        milliseconds = floor(seconds * 1000)
    except (OverflowError, ValueError):
        raise BackendProtocolError(f'{owner} must be finite') from None
    if not isfinite(seconds):
        raise BackendProtocolError(f'{owner} must be finite')
    if milliseconds <= 0:
        raise BackendProtocolError(f'{owner} is below Valkey precision')
    if milliseconds > 2**63 - 1:
        raise BackendProtocolError(f'{owner} exceeds Valkey precision')
    return milliseconds


def metadata_operands(
    prefix: str,
    incarnation: Incarnation,
    tags: Sequence[DependencyTag],
    retention_seconds: float,
) -> ScriptOperands:
    """Build shared resolve/change operands with ordered tags and global-retention flags.

    Args:
        prefix: Configured metadata-role key namespace.
        incarnation: Validated control record and counter.
        tags: Ordered dependency tags aligned with returned generation tokens.
        retention_seconds: Scoped-generation retention in seconds, flooring to
            between 1 and 2**63 - 1 milliseconds.

    Raises:
        BackendProtocolError: The retention duration cannot be represented
            within the positive millisecond bound.
    """
    return ScriptOperands(
        (
            control_key(prefix),
            incarnation.counter_key,
            *(_tag_key(prefix, incarnation.nonce, tag) for tag in tags),
        ),
        (
            incarnation.record,
            milliseconds(retention_seconds, 'scoped-generation retention'),
            *('1' if tag.kind in GLOBAL_TAG_KINDS else '0' for tag in tags),
        ),
    )


def new_incarnation(
    key_prefix: str, replication_id: str, nonce_factory: Callable[[], str]
) -> Incarnation:
    """Build fresh control state from validated entropy for one key prefix.

    Raises:
        BackendUnavailable: The nonce source fails or does not return
            128-bit lowercase hexadecimal entropy.
    """
    try:
        nonce = nonce_factory()
    except Exception:  # noqa: BLE001 - closed entropy boundary
        raise BackendUnavailable('Valkey incarnation entropy is unavailable') from None
    _validate_nonce(nonce)
    key = _counter_key(key_prefix, nonce)
    return Incarnation(replication_id, nonce, key, _record_bytes(replication_id, nonce, key))


def next_validation_state(
    snapshot: ValidationState,
    candidate: Incarnation,
    role_epoch: RoleEpoch,
    validated_at: float,
    *,
    join: bool,
) -> ValidationState:
    """Prepare a new validation or join coverage without extending its original lifetime.

    Args:
        snapshot: Immutable validation observed before the full validation
            began.
        candidate: Incarnation obtained through full validation on the pinned
            connection.
        role_epoch: Pool, connection, and endpoint values for the validated
            pin.
        validated_at: Monotonic time in seconds when full validation began.
        join: Whether to add coverage if the incarnation is unchanged. Callers
            pass true only for an unforced validation whose snapshot was
            current when it began.

    Returns:
        A state with the next revision and the validated connection covered.
        A join keeps the original validation time and role epoch; a new
        validation covers only its own connection.
    """
    revision = snapshot.revision + 1
    if (
        join
        and snapshot.incarnation == candidate
        and snapshot.role_epoch is not None
        and snapshot.validated_at is not None
    ):
        return ValidationState(
            candidate,
            snapshot.role_epoch,
            snapshot.validated_at,
            revision,
            snapshot.covered | {role_epoch[1]},
        )
    return ValidationState(
        candidate, role_epoch, validated_at, revision, frozenset({role_epoch[1]})
    )


def parse_record(record: bytes, key_prefix: str) -> Incarnation:
    """Decode a complete ASCII control record for the configured namespace.

    Args:
        record: Serialized replication identity, nonce, and counter key.
        key_prefix: Namespace that must own the incarnation counter key.

    Returns:
        A validated incarnation preserving the original record bytes.

    Raises:
        BackendProtocolError: The record, identity, or counter key is malformed
            or belongs to a different namespace.
        BackendUnavailable: The nonce is not 128-bit lowercase hexadecimal.
    """
    parts = record.split(b'|')
    if len(parts) != _RECORD_PART_COUNT:
        raise BackendProtocolError('Valkey control record is malformed')
    try:
        replication_id, nonce, parsed_counter_key = (part.decode('ascii') for part in parts)
    except UnicodeDecodeError:
        raise BackendProtocolError('Valkey control record is malformed') from None
    if parsed_counter_key != _counter_key(key_prefix, nonce):
        raise BackendProtocolError('Valkey control record is malformed')
    return Incarnation(replication_id, nonce, parsed_counter_key, record)


def primary_replication_id(raw: object) -> str:
    """Return the replication identity from a writable primary's INFO mapping.

    String and ASCII-byte field names and values are accepted.

    Raises:
        BackendProtocolError: The mapping, identity, or encoded text is
            malformed.
        BackendUnavailable: The reported role is not `master`.
    """
    if not isinstance(raw, Mapping):
        raise BackendProtocolError('Valkey replication information is malformed')
    info = cast('Mapping[object, object]', raw)
    role = info.get('role', info.get(b'role'))
    replid = info.get('master_replid', info.get(b'master_replid'))
    if isinstance(role, bytes):
        decoded_role = None
        with suppress(UnicodeDecodeError):
            decoded_role = role.decode('ascii')
        if decoded_role is None:
            raise BackendProtocolError('Valkey replication information is malformed')
        role = decoded_role
    if role != 'master':
        raise BackendUnavailable('Valkey metadata role is not writable')
    if isinstance(replid, bytes):
        decoded_replid = None
        with suppress(UnicodeDecodeError):
            decoded_replid = replid.decode('ascii')
        if decoded_replid is None:
            raise BackendProtocolError('Valkey replication information is malformed')
        replid = decoded_replid
    if not isinstance(replid, str):
        raise BackendProtocolError('Valkey replication information is malformed')
    return _validate_replication_id(replid)


def read_validation_time(clock: Callable[[], float]) -> float:
    """Read finite validation seconds or raise `BackendUnavailable` for an invalid clock."""
    try:
        value = clock()
        normalized = float(value)
    except Exception:  # noqa: BLE001 - closed injected-clock boundary
        raise BackendUnavailable('Valkey validation clock is unavailable') from None
    if isinstance(value, bool) or not isfinite(normalized):
        raise BackendUnavailable('Valkey validation clock is unavailable')
    return normalized


def release_operands(prefix: str, lease: FillLease) -> ScriptOperands:
    """Build token-checked release operands in the configured namespace."""
    return ScriptOperands((_lease_key(prefix, lease.key),), (lease.token,))


def require_enabled_l2(settings: object) -> CacheSettings:
    """Require enabled validated Valkey L2 settings.

    Raises:
        TypeError: The value is not a `CacheSettings` instance.
        ValueError: Valkey L2 is not enabled.
    """
    if not isinstance(settings, CacheSettings):
        raise TypeError('Valkey backend requires CacheSettings')
    if not settings.l2.enabled or settings.l2.backend != 'valkey':
        raise ValueError('Valkey backend requires enabled Valkey L2 settings')
    return settings


def require_loaded_digest(loaded: str, expected: str) -> None:
    """Require the server to report the digest bound to the local source.

    Raises:
        BackendProtocolError: The reported digest differs from the expected
            digest.
    """
    if loaded != expected:
        raise BackendProtocolError('Valkey loaded an unexpected script digest')


def reserve_operands(prefix: str, key: str, ttl: int, token: str) -> ScriptOperands:
    """Build lease reservation operands with token before expiry.

    Args:
        prefix: Configured result-role key namespace.
        key: Validated logical result key.
        ttl: Positive lease lifetime in milliseconds.
        token: Fresh validated lease ownership token.
    """
    return ScriptOperands((_lease_key(prefix, key),), (token, ttl))


def role_password(role: ValkeyRoleSettings, environ: Mapping[str, object]) -> str:
    """Resolve a nonempty password from the role's configured environment key.

    Raises:
        ValueError: The configured value is absent, not a string, or empty.
    """
    password = environ.get(role.password_env)
    if not isinstance(password, str) or not password:
        raise ValueError('Valkey role password is unset or empty')
    return password


def span_epoch(epoch: object, pinned: object) -> RoleEpoch:
    """Combine stable role state with one pin's connection identity.

    Returns:
        Pool epoch, per-pin connection identity when available (otherwise the
        aggregate generation), and the discovered endpoint.

    Raises:
        BackendProtocolError: The role epoch or pin generation is malformed.
    """
    pool_epoch, aggregate_generation, endpoint = _decode_role_epoch(epoch)
    connection_generation = getattr(pinned, 'connection_generation', None)
    if not callable(connection_generation):
        return pool_epoch, aggregate_generation, endpoint
    value: object = connection_generation()
    if value is None:
        return pool_epoch, aggregate_generation, endpoint
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise BackendProtocolError('Valkey connection generation is malformed')
    return pool_epoch, value, endpoint


def validated_resolve_tags(
    tags: object,
    settings: CacheSettings,
) -> tuple[DependencyTag, ...]:
    """Validate an ordered dependency tuple against configured and version-one bounds.

    Args:
        tags: Immutable tuple of unique dependency tags.
        settings: Settings supplying the tag-count and encoded-vector limits.

    Returns:
        The original tuple, within the configured bounds and the version-one
        ceiling of 32 tags.

    Raises:
        BackendProtocolError: The tuple is malformed, contains duplicates,
            cannot be encoded, or exceeds a count or byte bound.
    """
    validated = validate_vector_tags(
        tags, min(settings.limits.max_dependency_tags, VALKEY_MAX_DEPENDENCY_TAGS_V1)
    )
    require_vector_byte_bound(vector_bytes(validated), settings.limits.max_dependency_vector_bytes)
    return validated


def validation_interval(
    key_prefix: str, validation_seconds: float, jitter_factory: Callable[[float], float]
) -> float:
    """Validate configuration and obtain a finite positive jittered interval.

    Args:
        key_prefix: Nonempty ASCII metadata namespace.
        validation_seconds: Finite positive non-boolean nominal seconds.
        jitter_factory: Source called with the nominal interval to obtain the
            actual positive interval.

    Returns:
        The finite positive jittered interval in seconds.

    Raises:
        BackendProtocolError: The prefix or nominal interval is invalid.
        BackendUnavailable: The jitter source fails or returns an invalid
            interval.
    """
    prefix_value = cast('object', key_prefix)
    if not isinstance(prefix_value, str) or not prefix_value or not prefix_value.isascii():
        raise BackendProtocolError('Valkey key prefix is invalid')
    interval_value = cast('object', validation_seconds)
    if (
        isinstance(interval_value, bool)
        or not isinstance(interval_value, int | float)
        or not isfinite(interval_value)
        or interval_value <= 0
    ):
        raise BackendProtocolError('Valkey validation interval must be positive')
    try:
        interval = float(jitter_factory(float(validation_seconds)))
    except Exception:  # noqa: BLE001 - closed injected-jitter boundary
        raise BackendUnavailable('Valkey validation jitter is unavailable') from None
    if not isfinite(interval) or interval <= 0:
        raise BackendUnavailable('Valkey validation jitter is unavailable')
    return interval


def validation_is_current(
    snapshot: ValidationState,
    epoch: RoleEpoch,
    now: float,
    interval: float,
) -> bool:
    """Check the validation lifetime and role without considering connection coverage.

    Args:
        snapshot: Immutable locally published validation.
        epoch: Current pool, connection, and endpoint values.
        now: Current monotonic time in seconds.
        interval: Positive validation lifetime in seconds.

    Returns:
        Whether an uncleared validation remains within its original interval on
        the same pool and endpoint.
    """
    return (
        snapshot.incarnation is not None
        and snapshot.role_epoch is not None
        and snapshot.validated_at is not None
        and _same_role(epoch, snapshot.role_epoch)
        and now >= snapshot.validated_at
        and now - snapshot.validated_at < interval
    )


def cached_incarnation(
    snapshot: ValidationState,
    epoch: RoleEpoch,
    now: float,
    interval: float,
) -> Incarnation | None:
    """Return current validation for a covered connection on the same pool and endpoint.

    Args:
        snapshot: Immutable locally published validation.
        epoch: Current pure role epoch.
        now: Current monotonic time in seconds.
        interval: Positive validation lifetime in seconds.
    """
    if validation_is_current(snapshot, epoch, now, interval) and epoch[1] in snapshot.covered:
        return snapshot.incarnation
    return None
