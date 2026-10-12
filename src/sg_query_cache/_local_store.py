# Same-file collaborators use private implementation members.
# pyright: reportPrivateUsage=false

"""Own bounded local records and callback-free state transitions."""

from collections import OrderedDict
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass, field
from math import isfinite

from sg_query_cache._backend import (
    GLOBAL_TAG_KINDS,
    BackendCapacity,
    BackendProtocolError,
    BackendUnavailable,
    MonotonicClock,
)
from sg_query_cache._codec import CodecError, encode_value
from sg_query_cache._config import CacheSettings
from sg_query_cache._keys import KeyBuilder, NormalizationError
from sg_query_cache._model import (
    DependencyTag,
    FillLease,
    GenerationChange,
    GenerationVector,
)

_MAINTENANCE_OVERHEAD_BYTES = 64
_MAINTENANCE_STEP = 64
_MAX_COUNTER = 2**63 - 1
_METADATA_OVERHEAD_BYTES = 96
_RESULT_OVERHEAD_BYTES = 128


@dataclass(frozen=True, slots=True)
class _GenerationRecord:
    """Retain a generation token and its optional retirement deadline."""

    _retirement_deadline: float | None
    _key_size: int
    _size: int
    _token: int


@dataclass(frozen=True, slots=True)
class _LeaseRecord:
    """Retain a lease token and its absolute expiry."""

    _deadline: float
    _size: int
    _token: str


@dataclass(frozen=True, slots=True)
class ResultSnapshot:
    """Carry immutable result bytes whose object identity fences removal.

    Attributes:
        expiry: Absolute expiration in the owning backend's monotonic seconds.
        payload: Encoded result bytes.
        size: Accounted result bytes, including the key and fixed index
            allowances.
    """

    expiry: float
    payload: bytes
    size: int


@dataclass(slots=True)
class StoreOperation:
    """Carry sampled monotonic time and captured metrics for one transition.

    Events contain a metric name, an optional observation value, and labels.
    The wrapper emits them after leaving the store and its synchronization.
    """

    now: float
    events: list[tuple[str, float | None, dict[str, str]]] = field(
        default_factory=list[tuple[str, float | None, dict[str, str]]]
    )


class LocalStore:
    """Own local records, separate maintenance order, and incremental byte totals.

    Callers serialize transitions and supply validated inputs and sampled time.
    No transition invokes caller callbacks. Close alone may clear all records
    in linear time; normal maintenance visits at most 64 records per queue.
    Accounted bytes include each index's fixed allowance, not exact heap usage.

    Time-checked transitions raise `BackendUnavailable` after close or if the
    sample moves backward. Resolving or changing generations and reserving
    leases may raise `BackendCapacity` without publishing the requested new
    state.
    Result commit may evict older results; a matching lease is consumed even
    when the result exceeds the local capacity or payload limit.
    """

    def _allocate_generation(self, counter: int) -> int:
        """Return the next nonrepeating generation token."""
        if counter >= _MAX_COUNTER:
            raise BackendCapacity('generation token capacity is exhausted')
        return counter + 1

    def _generation_key_size(self, tag: DependencyTag) -> int:
        """Validate and measure a newly retained generation key."""
        key_size: int | None = None
        with suppress(CodecError, NormalizationError):
            key_size = len(KeyBuilder.tag_key(tag).encode('ascii'))
        if key_size is None:
            raise BackendProtocolError('dependency tag cannot be encoded safely')
        return key_size

    def _generation_size(self, key_size: int, token: int, deadline: float | None) -> int:
        """Measure encoded generation value and every retained index."""
        value_size: int | None = None
        with suppress(CodecError):
            value_size = len(encode_value((token, deadline)))
        if value_size is None:
            raise BackendProtocolError('dependency tag cannot be encoded safely')
        return (
            key_size
            + value_size
            + _METADATA_OVERHEAD_BYTES
            + (0 if deadline is None else _MAINTENANCE_OVERHEAD_BYTES)
        )

    def __init__(self, settings: CacheSettings, incarnation: str, now: float) -> None:
        """Initialize permanent generations within the configured capacity.

        Args:
            settings: Validated storage limits and site namespace.
            incarnation: Fresh entropy supplied by the owning backend.
            now: Initial finite monotonic sample.

        Raises:
            BackendCapacity: Permanent generations exceed metadata capacity.
        """
        self._closed = False
        self._last_now = now
        self._incarnation = incarnation
        self._generation_counter = 0
        self._lease_counter = 0
        self._generation_bytes = 0
        self._global_bytes = 0
        self._lease_bytes = 0
        self._result_bytes = 0
        self._generations: dict[DependencyTag, _GenerationRecord] = {}
        self._leases: dict[str, _LeaseRecord] = {}
        self._results: OrderedDict[str, ResultSnapshot] = OrderedDict()
        self._generation_queue: OrderedDict[DependencyTag, None] = OrderedDict()
        self._lease_queue: OrderedDict[str, None] = OrderedDict()
        self._result_queue: OrderedDict[str, None] = OrderedDict()
        self._metadata_budget_bytes = settings.l1.metadata_budget_bytes
        self._result_budget_bytes = settings.l1.result_budget_bytes
        self._max_serialized_result_bytes = settings.limits.max_serialized_result_bytes
        self._retention_seconds = settings.limits.scoped_generation_retention_seconds
        for tag in (
            DependencyTag.site(settings.site_namespace),
            DependencyTag.schema(settings.site_namespace),
            DependencyTag.authorization(settings.site_namespace),
            DependencyTag.human_visibility(settings.site_namespace),
        ):
            self._generation_counter = self._allocate_generation(self._generation_counter)
            key_size = self._generation_key_size(tag)
            size = self._generation_size(key_size, self._generation_counter, None)
            self._generations[tag] = _GenerationRecord(
                None, key_size, size, self._generation_counter
            )
            self._generation_bytes += size
        self._global_bytes = self._generation_bytes
        if self._generation_bytes > self._metadata_budget_bytes:
            raise BackendCapacity('metadata capacity cannot retain permanent generations')

    def _publish_generations(
        self, staged: dict[DependencyTag, _GenerationRecord], counter: int
    ) -> None:
        """Publish addressed generation replacements and incremental totals."""
        for tag, replacement in staged.items():
            previous = self._generations.get(tag)
            delta = replacement._size - (0 if previous is None else previous._size)
            self._generations[tag] = replacement
            self._generation_bytes += delta
            if tag.kind in GLOBAL_TAG_KINDS:
                self._global_bytes += delta
            elif previous is None:
                self._generation_queue[tag] = None
        self._generation_counter = counter

    def _remove_generation(self, tag: DependencyTag, op: StoreOperation) -> None:
        """Retire one scoped generation and both accounting indexes."""
        record = self._generations.pop(tag)
        del self._generation_queue[tag]
        self._generation_bytes -= record._size
        _increment(
            op, 'cache_scoped_generation', {'outcome': 'expired', 'tag_kind': tag.kind.value}
        )
        if record._retirement_deadline is not None:
            _observe(
                op,
                'cache_metadata_recovery_seconds',
                max(0.0, op.now - record._retirement_deadline),
                {'outcome': 'expiration'},
            )

    def _expire_generation(self, tag: DependencyTag, op: StoreOperation) -> None:
        """Check an addressed generation independently of maintenance progress."""
        record = self._generations.get(tag)
        if (
            record is not None
            and record._retirement_deadline is not None
            and record._retirement_deadline <= op.now
        ):
            self._remove_generation(tag, op)

    def _remove_lease(self, key: str) -> None:
        """Remove a lease and its maintenance index immediately."""
        record = self._leases.pop(key)
        del self._lease_queue[key]
        self._lease_bytes -= record._size

    def _expire_lease(self, key: str, now: float) -> None:
        """Check an addressed lease independently of maintenance progress."""
        record = self._leases.get(key)
        if record is not None and record._deadline <= now:
            self._remove_lease(key)

    def _remove_result(self, key: str, outcome: str, op: StoreOperation) -> None:
        """Remove result storage and both ordering indexes immediately."""
        record = self._results.pop(key)
        del self._result_queue[key]
        self._result_bytes -= record.size
        _observe(
            op,
            'cache_serialized_bytes',
            float(len(record.payload)),
            {'layer': 'local_l1', 'outcome': outcome},
        )

    def _expire_result(self, key: str, op: StoreOperation) -> bool:
        """Check an addressed result independently of maintenance progress."""
        record = self._results.get(key)
        if record is not None and record.expiry <= op.now:
            self._remove_result(key, 'expired', op)
            return True
        return False

    def _require_metadata_capacity(
        self, generation_bytes: int, lease_bytes: int, op: StoreOperation
    ) -> None:
        """Reject staged metadata mutations without publishing requested state."""
        if generation_bytes + lease_bytes > self._metadata_budget_bytes:
            _increment(op, 'cache_metadata_write_rejection', {'reason': 'capacity'})
            raise BackendCapacity('backend metadata capacity is exhausted')

    def check_time(self, now: float) -> None:
        """Check open state and a finite, wrapper-validated monotonic sample."""
        if self._closed:
            raise BackendUnavailable('backend is closed')
        if now < self._last_now:
            raise BackendUnavailable('backend monotonic clock moved backward')
        self._last_now = now

    def _maintain(self, op: StoreOperation) -> None:
        """Advance each separate maintenance queue by one bounded rotation."""
        self.check_time(op.now)
        for _ in range(min(_MAINTENANCE_STEP, len(self._result_queue))):
            key = next(iter(self._result_queue))
            if not self._expire_result(key, op):
                self._result_queue.move_to_end(key)
        for _ in range(min(_MAINTENANCE_STEP, len(self._lease_queue))):
            key = next(iter(self._lease_queue))
            self._expire_lease(key, op.now)
            if key in self._lease_queue:
                self._lease_queue.move_to_end(key)
        for _ in range(min(_MAINTENANCE_STEP, len(self._generation_queue))):
            tag = next(iter(self._generation_queue))
            self._expire_generation(tag, op)
            if tag in self._generation_queue:
                self._generation_queue.move_to_end(tag)

    def close(self) -> None:
        """Clear all records in O(N); repeated terminal close is harmless."""
        self._closed = True
        self._generations.clear()
        self._leases.clear()
        self._results.clear()
        self._generation_queue.clear()
        self._lease_queue.clear()
        self._result_queue.clear()
        self._generation_bytes = self._global_bytes = self._lease_bytes = self._result_bytes = 0

    def commit(
        self, lease: FillLease, payload: bytes, duration: float, op: StoreOperation
    ) -> bool:
        """Admit encoded bytes while the exact addressed lease is still live.

        Args:
            lease: Validated ownership token and opaque key.
            payload: Immutable bytes within the closed backend input contract.
            duration: Positive authoritative lifetime remaining at op.now.
            op: Time sample and captured metric events for this transition.

        Returns:
            Whether the result was admitted under the current lease.
        """
        self._maintain(op)
        self._expire_lease(lease.key, op.now)
        record = self._leases.get(lease.key)
        if record is None or record._token != lease.token:
            _increment(op, 'cache_lease_rejection', {'layer': 'local_l1'})
            _increment(op, 'cache_commit', {'layer': 'local_l1', 'outcome': 'rejected'})
            return False
        self._remove_lease(lease.key)
        size = (
            len(lease.key.encode('utf-8'))
            + len(payload)
            + _RESULT_OVERHEAD_BYTES
            + _MAINTENANCE_OVERHEAD_BYTES
        )
        if len(payload) > self._max_serialized_result_bytes or size > self._result_budget_bytes:
            _increment(op, 'cache_commit', {'layer': 'local_l1', 'outcome': 'rejected'})
            return False
        previous = self._results.get(lease.key)
        available = self._result_bytes - (0 if previous is None else previous.size)
        while available + size > self._result_budget_bytes:
            candidate = next(iter(self._results))
            if candidate == lease.key:
                self._results.move_to_end(candidate)
                continue
            available -= self._results[candidate].size
            self._remove_result(candidate, 'evicted', op)
        if previous is not None:
            self._result_bytes -= previous.size
            del self._results[lease.key]
        else:
            self._result_queue[lease.key] = None
        self._results[lease.key] = ResultSnapshot(op.now + duration, payload, size)
        self._result_bytes += size
        _observe(
            op,
            'cache_serialized_bytes',
            float(len(payload)),
            {'layer': 'local_l1', 'outcome': 'admitted'},
        )
        _increment(op, 'cache_commit', {'layer': 'local_l1', 'outcome': 'committed'})
        return True

    @property
    def generation_bytes(self) -> int:
        """Return accounted generation bytes, including maintenance indexes."""
        return self._generation_bytes

    @property
    def lease_bytes(self) -> int:
        """Return accounted lease bytes, including maintenance indexes."""
        return self._lease_bytes

    def lookup(self, key: str, op: StoreOperation) -> ResultSnapshot | None:
        """Capture an immutable snapshot, checking addressed expiry immediately."""
        record = self._results.get(key)
        expired = record is not None and record.expiry <= op.now
        self._maintain(op)
        self._expire_result(key, op)
        record = self._results.get(key)
        if record is None:
            _increment(
                op,
                'cache_miss',
                {'layer': 'local_l1', 'reason': 'expired' if expired else 'absence'},
            )
            return None
        self._results.move_to_end(key)
        return record

    def metadata_bytes(self, op: StoreOperation) -> None:
        """Capture incremental global and scoped generation capacity totals."""
        _observe(
            op, 'generation_metadata_bytes', float(self._global_bytes), {'tag_kind': 'global'}
        )
        _observe(
            op,
            'generation_metadata_bytes',
            float(self._generation_bytes - self._global_bytes),
            {'tag_kind': 'scoped'},
        )

    def _change(self, changes: tuple[GenerationChange, ...], op: StoreOperation) -> None:
        """Stage and publish only addressed generations in caller order."""
        staged: dict[DependencyTag, _GenerationRecord] = {}
        counter = self._generation_counter
        delta = 0
        events: list[tuple[str, float | None, dict[str, str]]] = []
        for change in changes:
            tag = change.tag
            self._expire_generation(tag, op)
            record = staged.get(tag, self._generations.get(tag))
            if record is None and tag.kind not in GLOBAL_TAG_KINDS:
                events.append(
                    (
                        'cache_scoped_generation',
                        None,
                        {'outcome': 'absent', 'tag_kind': tag.kind.value},
                    )
                )
                continue
            counter = self._allocate_generation(counter)
            deadline = None if record is None else record._retirement_deadline
            key_size = self._generation_key_size(tag) if record is None else record._key_size
            replacement = _GenerationRecord(
                deadline, key_size, self._generation_size(key_size, counter, deadline), counter
            )
            delta += replacement._size - (0 if record is None else record._size)
            staged[tag] = replacement
            events.append(('cache_invalidation', None, {'tag_kind': tag.kind.value}))
        self._require_metadata_capacity(self._generation_bytes + delta, self._lease_bytes, op)
        self._publish_generations(staged, counter)
        op.events.extend(events)
        self.metadata_bytes(op)

    def change(self, changes: tuple[GenerationChange, ...], op: StoreOperation) -> None:
        """Replace applicable addressed generations atomically in caller order."""
        self._maintain(op)
        self._change(changes, op)

    def flush_site(self, tag: DependencyTag, reason: str, op: StoreOperation) -> None:
        """Replace a validated site generation in one maintenance transition."""
        self._maintain(op)
        self._change((GenerationChange(tag),), op)
        _increment(op, 'cache_site_flush', {'reason': reason})

    def release(self, lease: FillLease, op: StoreOperation) -> bool:
        """Remove only an exact current unexpired lease."""
        self._maintain(op)
        self._expire_lease(lease.key, op.now)
        record = self._leases.get(lease.key)
        if record is None or record._token != lease.token:
            return False
        self._remove_lease(lease.key)
        return True

    def remove_snapshot(self, key: str, snapshot: ResultSnapshot) -> bool:
        """Remove captured result bytes only if their record is still current."""
        if self._results.get(key) is not snapshot:
            return False
        self._results.pop(key)
        del self._result_queue[key]
        self._result_bytes -= snapshot.size
        return True

    def reserve(
        self, key: str, duration: float, entropy: str, op: StoreOperation
    ) -> FillLease | None:
        """Reserve an absent or expired addressed lease with supplied fresh entropy.

        Args:
            key: Validated opaque result key.
            duration: Positive lease lifetime in seconds from `op.now`.
            entropy: Fresh cryptographic hex sampled by the wrapper.
            op: Time sample and captured metric events for this transition.

        Returns:
            The new lease, or `None` if a live owner already exists.
        """
        self._maintain(op)
        self._expire_lease(key, op.now)
        if key in self._leases:
            _increment(op, 'cache_lease_rejection', {'layer': 'local_l1'})
            return None
        if self._lease_counter >= _MAX_COUNTER:
            raise BackendCapacity('lease token capacity is exhausted')
        counter = self._lease_counter + 1
        token = f'{counter:016x}{entropy}'
        deadline = op.now + duration
        size = (
            len(key.encode('utf-8'))
            + len(encode_value((token, deadline)))
            + _METADATA_OVERHEAD_BYTES
            + _MAINTENANCE_OVERHEAD_BYTES
        )
        self._require_metadata_capacity(self._generation_bytes, self._lease_bytes + size, op)
        self._lease_counter = counter
        self._leases[key] = _LeaseRecord(deadline, size, token)
        self._lease_queue[key] = None
        self._lease_bytes += size
        return FillLease(key, token)

    def resolve(self, tags: tuple[DependencyTag, ...], op: StoreOperation) -> GenerationVector:
        """Resolve an ordered validated vector after bounded queue maintenance."""
        self._maintain(op)
        staged: dict[DependencyTag, _GenerationRecord] = {}
        counter = self._generation_counter
        delta = 0
        created: list[DependencyTag] = []
        entries: list[tuple[DependencyTag, int]] = []
        for tag in tags:
            self._expire_generation(tag, op)
            record = self._generations.get(tag)
            deadline = None if tag.kind in GLOBAL_TAG_KINDS else op.now + self._retention_seconds
            if record is None:
                counter = self._allocate_generation(counter)
                token = counter
                key_size = self._generation_key_size(tag)
                if deadline is not None:
                    created.append(tag)
            else:
                token = record._token
                key_size = record._key_size
            replacement = _GenerationRecord(
                deadline, key_size, self._generation_size(key_size, token, deadline), token
            )
            delta += replacement._size - (0 if record is None else record._size)
            staged[tag] = replacement
            entries.append((tag, token))
        self._require_metadata_capacity(self._generation_bytes + delta, self._lease_bytes, op)
        self._publish_generations(staged, counter)
        for tag in created:
            _increment(
                op, 'cache_scoped_generation', {'outcome': 'created', 'tag_kind': tag.kind.value}
            )
        self.metadata_bytes(op)
        _increment(op, 'cache_vector_resolution', {'layer': 'local_l1'})
        return GenerationVector(self._incarnation, tuple(entries))

    @property
    def result_bytes(self) -> int:
        """Return accounted result bytes, including both ordering indexes."""
        return self._result_bytes


def _increment(op: StoreOperation, name: str, labels: dict[str, str]) -> None:
    """Capture one counter event for later emission."""
    op.events.append((name, None, labels))


def _observe(op: StoreOperation, name: str, value: float, labels: dict[str, str]) -> None:
    """Capture one observation event for later emission."""
    op.events.append((name, value, labels))


def _unavailable_primitive(operation: Callable[[], object], message: str) -> object:
    """Replace ordinary primitive failures with a safe backend error."""
    sentinel = object()
    result = sentinel
    with suppress(Exception):
        result = operation()
    if result is sentinel:
        raise BackendUnavailable(message)
    return result


def clock_read(clock: MonotonicClock) -> float:
    """Read finite monotonic seconds or raise `BackendUnavailable` for an invalid clock."""
    value = _unavailable_primitive(clock, 'backend monotonic clock is unavailable')
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise BackendUnavailable('backend monotonic clock is unavailable')
    normalized: float | None = None
    with suppress(Exception):
        normalized = float(value)
    if normalized is None or not isfinite(normalized):
        raise BackendUnavailable('backend monotonic clock is unavailable')
    return normalized


def entropy_hex(byte_count: int, source: Callable[[int], str]) -> str:
    """Read lowercase hexadecimal entropy through the backend failure boundary.

    Args:
        byte_count: Byte count passed to the source; its output must contain
            exactly twice this many hexadecimal characters.
        source: Token source invoked once.

    Returns:
        The validated lowercase hexadecimal token.

    Raises:
        BackendUnavailable: The source fails or returns a malformed token.
    """
    value = _unavailable_primitive(
        lambda: source(byte_count), 'backend entropy source is unavailable'
    )
    if (
        not isinstance(value, str)
        or len(value) != byte_count * 2
        or any(character not in '0123456789abcdef' for character in value)
    ):
        raise BackendUnavailable('backend entropy source is unavailable')
    return value


def positive_seconds(value: object, owner: str) -> float:
    """Require finite positive non-boolean seconds or raise `BackendProtocolError`."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise BackendProtocolError(f'{owner} must be finite')
    normalized: float | None = None
    with suppress(Exception):
        normalized = float(value)
    if normalized is None or not isfinite(normalized):
        raise BackendProtocolError(f'{owner} must be finite')
    if normalized <= 0:
        raise BackendProtocolError(f'{owner} must be positive')
    return normalized
