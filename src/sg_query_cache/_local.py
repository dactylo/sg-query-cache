"""Implement the synchronized wrapper around bounded local storage."""

from collections.abc import Generator, Mapping
from contextlib import contextmanager, suppress
from secrets import token_hex
from threading import RLock

from sg_query_cache._backend import BackendProtocolError, MonotonicClock
from sg_query_cache._backend_validation import (
    encoded_payload,
    fill_lease,
    flush_reason,
    generation_changes,
    metric_flush_reason,
    opaque_key,
    require_vector_byte_bound,
    validate_vector_tags,
    vector_bytes,
)
from sg_query_cache._codec import CodecError, decode_value
from sg_query_cache._config import CacheSettings
from sg_query_cache._local_store import (
    LocalStore,
    StoreOperation,
    entropy_hex,
)
from sg_query_cache._local_store import (
    clock_read as _clock_read,
)
from sg_query_cache._local_store import (
    positive_seconds as _positive_seconds,
)
from sg_query_cache._metrics import MetricsSink
from sg_query_cache._model import (
    CacheHit,
    DecodedPayload,
    DependencyTag,
    FillLease,
    GenerationChange,
    GenerationVector,
)


class LocalBackend:
    """Provide one thread-safe byte-bounded in-process cache backend.

    Store transitions and monotonic clock sampling share one reentrant lock.
    Immutable snapshots decode outside it; telemetry is emitted after each
    transition. Result eviction never consumes metadata capacity.
    """

    __slots__ = (
        '_clock',
        '_lock',
        '_max_dependency_tags',
        '_max_dependency_vector_bytes',
        '_metrics',
        '_store',
    )

    def _emit(self, op: StoreOperation) -> None:
        """Emit captured metrics outside the store lock with exception isolation."""
        for name, value, labels in op.events:
            with suppress(Exception):
                if value is None:
                    self._metrics.increment(name, labels)
                else:
                    self._metrics.observe(name, value, labels)

    def __init__(
        self, settings: CacheSettings, clock: MonotonicClock, metrics: MetricsSink
    ) -> None:
        """Initialize the bounded store and its permanent generation records.

        Args:
            settings: Validated cache settings supplying L1 and operational
                bounds.
            clock: Backend-authoritative monotonic clock.
            metrics: Safe metrics boundary retained for owning state
                transitions.

        Raises:
            BackendCapacity: The metadata budget cannot retain permanent
                globals.
            BackendUnavailable: The initial clock or entropy source is invalid.
        """
        self._clock = clock
        self._lock = RLock()
        self._metrics = metrics
        self._max_dependency_tags = settings.limits.max_dependency_tags
        self._max_dependency_vector_bytes = settings.limits.max_dependency_vector_bytes
        now = _clock_read(clock)
        self._store = LocalStore(settings, entropy_hex(16, token_hex), now)
        op = StoreOperation(now)
        self._store.metadata_bytes(op)
        self._emit(op)

    def _increment(self, name: str, labels: Mapping[str, str]) -> None:
        """Emit a best-effort counter outside a state transition."""
        with suppress(Exception):
            self._metrics.increment(name, labels)

    def _observe(self, name: str, value: float, labels: Mapping[str, str]) -> None:
        """Emit a best-effort observation outside a state transition."""
        with suppress(Exception):
            self._metrics.observe(name, value, labels)

    @contextmanager
    def _transition(self) -> Generator[StoreOperation]:
        """Serialize clock observation and store state, then emit captured events."""
        op: StoreOperation | None = None
        try:
            with self._lock:
                op = StoreOperation(_clock_read(self._clock))
                yield op
        finally:
            if op is not None:
                self._emit(op)

    def change(self, changes: tuple[GenerationChange, ...]) -> None:
        """Replace all applicable generations in caller order.

        Args:
            changes: Ordered generation replacements to acknowledge together.

        Raises:
            BackendCapacity: Required global metadata cannot be retained.
            BackendProtocolError: The ordered change tuple is malformed.
            BackendUnavailable: The backend or clock is unavailable.
        """
        validated = generation_changes(changes)
        with self._transition() as op:
            self._store.change(validated, op)

    def close(self) -> None:
        """Release backend resources; repeated calls have no effect."""
        with self._lock:
            self._store.close()

    def commit(self, lease: FillLease, payload: bytes, remaining_seconds: float) -> bool:
        """Store encoded bytes only while the exact lease remains current.

        Args:
            lease: Exact key and token returned by `reserve`.
            payload: Closed-codec bytes to retain.
            remaining_seconds: Positive authoritative lifetime remaining.

        Returns:
            Whether the lease remained current and the result was admitted.

        Raises:
            BackendProtocolError: An argument is outside the backend contract.
            BackendUnavailable: The backend or clock is unavailable.
        """
        entry_time = _clock_read(self._clock)
        validated_lease = fill_lease(lease)
        validated_payload = encoded_payload(payload)
        duration = _positive_seconds(remaining_seconds, 'remaining lifetime')
        deadline = entry_time + duration
        opaque_key(validated_lease.key)
        with self._transition() as op:
            self._store.check_time(op.now)
            remaining = deadline - op.now
            if remaining <= 0:
                return False
            return self._store.commit(validated_lease, validated_payload, remaining, op)

    def flush_site(self, site_namespace: str, reason: str) -> None:
        """Replace one site's permanent site generation.

        Args:
            site_namespace: Stable configured site namespace.
            reason: Short operator-facing reason code.

        Raises:
            BackendCapacity: Required global metadata cannot be retained.
            BackendProtocolError: The site namespace or reason is malformed.
            BackendUnavailable: The backend or clock is unavailable.
        """
        try:
            tag = DependencyTag.site(site_namespace)
        except (TypeError, ValueError) as error:
            raise BackendProtocolError('site namespace is invalid') from error
        flush_reason(reason)
        with self._transition() as op:
            self._store.flush_site(tag, metric_flush_reason(reason), op)

    def lookup(self, key: str) -> CacheHit | None:
        """Read valid encoded bytes with backend-authoritative remaining age.

        Args:
            key: Opaque generation-qualified result key.

        Returns:
            An isolated hit, or `None` for absence, expiry, or corruption.

        Raises:
            BackendProtocolError: The opaque key is malformed.
            BackendUnavailable: The backend or clock is unavailable.
        """
        normalized_key = opaque_key(key)
        with self._transition() as op:
            snapshot = self._store.lookup(normalized_key, op)
        if snapshot is None:
            return None
        try:
            decoded = decode_value(snapshot.payload)
        except CodecError:
            with self._lock:
                self._store.remove_snapshot(normalized_key, snapshot)
            self._increment('cache_miss', {'layer': 'local_l1', 'reason': 'malformed'})
            return None
        with self._lock:
            now = _clock_read(self._clock)
            self._store.check_time(now)
            remaining = snapshot.expiry - now
            if remaining <= 0:
                self._store.remove_snapshot(normalized_key, snapshot)
        if remaining <= 0:
            self._increment('cache_miss', {'layer': 'local_l1', 'reason': 'expired'})
            return None
        return CacheHit(snapshot.payload, remaining, decoded=DecodedPayload(decoded))

    def release(self, lease: FillLease) -> bool:
        """Remove a fill lease only when its exact token remains current.

        Args:
            lease: Exact key and token returned by `reserve`.

        Returns:
            Whether the current lease was removed.

        Raises:
            BackendProtocolError: The lease is malformed.
            BackendUnavailable: The backend or clock is unavailable.
        """
        validated = fill_lease(lease)
        opaque_key(validated.key)
        with self._transition() as op:
            return self._store.release(validated, op)

    def reserve(self, key: str, lease_seconds: float) -> FillLease | None:
        """Acquire temporary fill ownership for one opaque key.

        Args:
            key: Opaque generation-qualified result key.
            lease_seconds: Positive ownership duration.

        Returns:
            A fresh lease, or `None` while another lease remains current.

        Raises:
            BackendCapacity: Metadata capacity cannot retain a new lease.
            BackendProtocolError: An argument is outside the backend contract.
            BackendUnavailable: The backend or clock is unavailable.
        """
        normalized_key = opaque_key(key)
        duration = _positive_seconds(lease_seconds, 'lease duration')
        entropy = entropy_hex(16, token_hex)
        with self._transition() as op:
            return self._store.reserve(normalized_key, duration, entropy, op)

    def resolve(self, tags: tuple[DependencyTag, ...]) -> GenerationVector:
        """Resolve one complete bounded dependency vector in request order.

        Args:
            tags: Ordered unique dependency tags.

        Returns:
            Every requested tag and current token under one incarnation.

        Raises:
            BackendCapacity: Metadata capacity cannot retain the vector.
            BackendProtocolError: The vector is malformed or exceeds its
                bounds.
            BackendUnavailable: The backend or clock is unavailable.
        """
        validated_tags = validate_vector_tags(tags, self._max_dependency_tags)
        encoded_bytes = vector_bytes(validated_tags)
        self._observe(
            'cache_dependency_vector_tags', float(len(validated_tags)), {'layer': 'local_l1'}
        )
        self._observe('cache_dependency_vector_bytes', float(encoded_bytes), {'layer': 'local_l1'})
        require_vector_byte_bound(encoded_bytes, self._max_dependency_vector_bytes)
        with self._transition() as op:
            return self._store.resolve(validated_tags, op)

    def storage_usage(self) -> dict[str, int]:
        """Return a fresh snapshot of accounted capacity bytes without maintenance.

        Returns:
            Result, generation, and lease bytes under `result_bytes`,
            `generation_bytes`, and `lease_bytes`; all zero after close.
            These are capacity accounting totals, not Python heap sizes.
        """
        with self._lock:
            return {
                'result_bytes': self._store.result_bytes,
                'generation_bytes': self._store.generation_bytes,
                'lease_bytes': self._store.lease_bytes,
            }
