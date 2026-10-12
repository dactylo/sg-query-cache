# Same-file collaborators use private implementation members.
# pyright: reportPrivateUsage=false

"""Compose the supported local, shared, and hierarchical cache read paths."""

from __future__ import annotations

from collections.abc import Callable, Generator, Mapping
from contextlib import contextmanager, suppress
from dataclasses import dataclass, replace
from random import uniform
from threading import Condition, Lock, local
from time import monotonic, sleep
from typing import TYPE_CHECKING, Literal, TypeAlias

from sg_query_cache._backend import (
    BackendCapacity,
    BackendProtocolError,
    BackendUnavailable,
    StaleIncarnation,
)
from sg_query_cache._cache_common import (
    DecodedHit,
    PolicyFrame,
    PolicyMode,
    PopulationCallback,
    StaleHitError,
    TimedResult,
    WaitOutcome,
    WaitRecord,
    apply_policy_frames,
    bound_reason,
    decode_hit,
    encode_result,
    merge_changes,
    minimum_policy_age,
    policy_bypass_active,
    policy_frame,
    population_limit,
)
from sg_query_cache._codec import CodecBoundError, CodecError
from sg_query_cache._config import CacheLayer, CacheSettings, TTLContext, select_maximum_age
from sg_query_cache._keys import KeyBuilder, NormalizationError
from sg_query_cache._metrics import MetricsSink, NullMetrics, SafeMetrics

if TYPE_CHECKING:
    from sg_query_cache._backend import CacheBackend
    from sg_query_cache._model import (
        DependencyPlan,
        DependencyTag,
        ExactCall,
        FillLease,
        GenerationChange,
        GenerationVector,
        Identity,
    )

_FenceLayer: TypeAlias = Literal['l1', 'l2']
_Fetch: TypeAlias = Callable[[], object]
Parse: TypeAlias = Callable[[object], object]

_BACKEND_ERRORS = (BackendCapacity, BackendProtocolError, BackendUnavailable, StaleIncarnation)
_BACKEND_FAILURE = 'backend_failure'
_CODEC_FAILURE = 'codec_failure'
_CORRELATION_ERROR = 'configured metrics correlation secret is unavailable or invalid'
_MISS = object()


class _CachePathError(Exception):
    """Redirect one unsafe cache path to a direct fetch."""

    __slots__ = ('_reason',)

    def __init__(self, reason: str) -> None:
        """Retain the safe reason for bypassing this cache path."""
        super().__init__()
        self._reason = reason


@dataclass(frozen=True, slots=True)
class _FenceState:
    """Hold one layer's retained unacknowledged changes with a revision."""

    _retained: tuple[GenerationChange, ...]
    _revision: int


@dataclass(slots=True)
class _Flight:
    """Coordinate one bounded same-key local fill."""

    _condition: Condition
    _done: bool = False
    _participants: int = 1


@dataclass(frozen=True, slots=True)
class _LayerHit:
    """Pair a decoded hit with its authoritative deadline and observation time."""

    _decoded: DecodedHit
    _deadline: float
    _observed_at: float


@dataclass(frozen=True, slots=True)
class _LayerMiss:
    """Retain the original generation vector and qualified result key."""

    _key: str
    _vector: GenerationVector


@dataclass(frozen=True, slots=True)
class _ReadOutcome:
    """Return one value with the keys and deadlines the parsed-view memo needs."""

    _value: object
    _first_key: str
    _shared_key: str | None
    _origin_at: float
    _first_deadline: float
    _deep_deadline: float


@dataclass(frozen=True, slots=True)
class _MemoEntry:
    """Hold one parsed value with its validity keys, deadlines, and parser."""

    _first_key: str
    _shared_key: str | None
    _origin_at: float
    _first_deadline: float
    _deep_deadline: float
    _parse: Parse
    _value: object


@dataclass(frozen=True, slots=True)
class _SharedOutcome:
    """Return a shared value with the timing needed for a later local promotion."""

    _promotion_deadline: float | None
    _sg_duration: float | None
    _value: object
    _origin_elapsed: float | None
    _observed_at: float
    _key: str


class _PolicyState(local):
    """Hold an independent policy stack for each calling thread."""

    _stack: list[PolicyFrame]

    def __init__(self) -> None:
        """Initialize this thread's independent policy stack."""
        self._stack = []


class _MetricsState(local):
    """Hold independent correlation and request wait stacks for each calling thread."""

    _stack: list[str | None]
    _waits: list[WaitRecord]

    def __init__(self) -> None:
        """Initialize this thread's independent metric stacks."""
        self._stack = []
        self._waits = []


class QueryCache:
    """Compose one exact supported cache topology around caller-owned fetches.

    The cache closes its configured backends when `close` is called; it never
    owns or closes a fetch callback's SG client. Coordinate shutdown with
    callers, because close does not wait for active reads.

    Lexical policy scopes are thread-local. Backend failures on read paths fall
    back to the caller's fetch, whose failures propagate. Ordinary cache hits
    are isolated decoded values; parsed memo results must be immutable and may
    be shared across calls.
    """

    __slots__ = (
        '_fence_lock',
        '_fences',
        '_flight_lock',
        '_flights',
        '_identity',
        '_key_builder',
        '_l1',
        '_l2',
        '_memo',
        '_memo_lock',
        '_metrics',
        '_metrics_state',
        '_policy_state',
        '_settings',
    )

    def __init__(  # noqa: PLR0913, PLR0917 - required public topology interface
        self,
        settings: CacheSettings,
        identity: Identity,
        key_builder: KeyBuilder | None = None,
        l1: CacheBackend | None = None,
        l2: CacheBackend | None = None,
        metrics: MetricsSink | None = None,
    ) -> None:
        """Validate the topology and retain caller-supplied cache resources.

        Args:
            settings: Immutable cache policy and operational bounds.
            identity: Exact effective SG identity namespace.
            key_builder: Optional exact-call and qualified-key builder. The
                default is the package's standard builder.
            l1: Optional local cache backend selected by `settings`.
            l2: Optional shared cache backend selected by `settings`.
            metrics: Optional safe metrics destination.

        Raises:
            ValueError: Backend wiring does not match one supported topology.
        """
        expected = (settings.l1.enabled, settings.l2.enabled)
        supplied = (l1 is not None, l2 is not None)
        if supplied != expected or (l1 is not None and l1 is l2):
            raise ValueError('backend wiring does not match a supported cache topology')
        self._fence_lock = Lock()
        self._fences: dict[_FenceLayer, _FenceState] = {
            'l1': _FenceState((), 0),
            'l2': _FenceState((), 0),
        }
        self._flight_lock = Lock()
        self._flights: dict[str, _Flight] = {}
        self._identity = identity
        self._key_builder = KeyBuilder() if key_builder is None else key_builder
        self._l1 = l1
        self._l2 = l2
        self._memo: _MemoEntry | None = None
        self._memo_lock = Lock()
        self._metrics = NullMetrics() if metrics is None else metrics
        self._metrics_state = _MetricsState()
        self._policy_state = _PolicyState()
        self._settings = settings

    def _effective_age(self, per_call_age: float | None) -> float | None:
        """Return the smallest explicit call or lexical age."""
        return minimum_policy_age(tuple(self._policy_state._stack), per_call_age)

    def _effective_plan(self, plan: DependencyPlan) -> DependencyPlan:
        """Apply every lexical mode without promoting an existing decision."""
        return apply_policy_frames(plan, tuple(self._policy_state._stack))

    def _fence_failed(
        self,
        layer: _FenceLayer,
        snapshot: _FenceState,
        attempt: tuple[GenerationChange, ...],
    ) -> None:
        """Retain an unacknowledged attempt, merging when the state moved meanwhile."""
        with self._fence_lock:
            current = self._fences[layer]
            retained = (
                attempt
                if current._revision == snapshot._revision
                else merge_changes(current._retained, attempt)
            )
            self._fences[layer] = _FenceState(merge_changes((), retained), current._revision + 1)

    def _fence_snapshot(self, layer: _FenceLayer) -> _FenceState:
        """Return one layer's revisioned fence state."""
        with self._fence_lock:
            return self._fences[layer]

    def _fence_succeeded(self, layer: _FenceLayer, snapshot: _FenceState) -> None:
        """Clear retained changes only if nothing was retained since the snapshot."""
        with self._fence_lock:
            current = self._fences[layer]
            if current._revision == snapshot._revision:
                self._fences[layer] = _FenceState((), current._revision + 1)

    def _fence_unchanged(self, layer: _FenceLayer, snapshot: _FenceState) -> bool:
        """Report whether an unfenced snapshot remains the live empty state."""
        with self._fence_lock:
            current = self._fences[layer]
            return (
                not snapshot._retained
                and current._revision == snapshot._revision
                and not current._retained
            )

    def _finish_flight(self, key: str, flight: _Flight) -> None:
        """Wake followers and remove the record after its last participant."""
        with self._flight_lock:
            flight._done = True
            flight._condition.notify_all()
            flight._participants -= 1
            if flight._participants == 0 and self._flights.get(key) is flight:
                self._flights.pop(key, None)

    def _increment(self, name: str, labels: Mapping[str, str]) -> None:
        """Emit one best-effort counter without affecting cache behavior."""
        correlated = dict(labels)
        digest = self._metrics_state._stack[-1] if self._metrics_state._stack else None
        if digest is not None:
            correlated['exact_call_digest'] = digest
        with suppress(Exception):
            self._metrics.increment(name, correlated)

    def _join_flight(self, key: str) -> tuple[_Flight, bool]:
        """Join an existing same-key flight or become its leader."""
        with self._flight_lock:
            flight = self._flights.get(key)
            if flight is None or flight._done:
                flight = _Flight(Condition(self._flight_lock))
                self._flights[key] = flight
                return flight, True
            flight._participants += 1
            return flight, False

    def _lookup(
        self,
        backend: CacheBackend,
        base_key: str,
        tags: tuple[DependencyTag, ...],
        maximum_age: float,
    ) -> tuple[_LayerMiss, _LayerHit | None]:
        """Resolve and look up one layer while retaining its original vector."""
        layer: _FenceLayer = 'l2' if backend is self._l2 else 'l1'
        fence = self._fence_snapshot(layer)
        if fence._retained:
            raise _CachePathError('unacknowledged_fence')
        try:
            vector = backend.resolve(tags)
            key = self._key_builder.qualified_key(base_key, vector, 'result')
            hit = backend.lookup(key)
        except _BACKEND_ERRORS as error:
            raise _CachePathError(_BACKEND_FAILURE) from error
        except NormalizationError as error:
            raise _CachePathError('normalization_failure') from error
        observed_at = monotonic()
        if hit is None:
            if not self._fence_unchanged(layer, fence):
                raise _CachePathError('unacknowledged_fence')
            return _LayerMiss(key, vector), None
        authoritative_deadline = observed_at + hit.remaining_seconds
        try:
            decoded = decode_hit(hit, maximum_age)
        except StaleHitError:
            candidate = None
        except CodecError:
            raise _CachePathError(_CODEC_FAILURE) from None
        else:
            candidate = _debit_hit_processing_time(
                decoded,
                observed_at,
                authoritative_deadline,
                maximum_age,
            )
        if not self._fence_unchanged(layer, fence):
            raise _CachePathError('unacknowledged_fence')
        return _LayerMiss(key, vector), candidate

    def _observe(
        self,
        name: str,
        value: float,
        labels: Mapping[str, str],
    ) -> None:
        """Emit one best-effort observation without affecting cache behavior."""
        correlated = dict(labels)
        digest = self._metrics_state._stack[-1] if self._metrics_state._stack else None
        if digest is not None:
            correlated['exact_call_digest'] = digest
        with suppress(Exception):
            self._metrics.observe(name, value, correlated)

    def _consume_hit(self, backend: CacheBackend, candidate: _LayerHit) -> DecodedHit:
        """Record one successfully decoded hit at its actual cache layer."""
        hit = candidate._decoded
        if backend is self._l2:
            layer: CacheLayer = 'l2'
        elif self._l2 is None:
            layer = 'local_l1'
        else:
            layer = 'hierarchical_l1'
        self._increment('cache_hit', {'layer': layer})
        self._increment('cache_sg_avoided_calls', {'layer': layer})
        self._observe('cache_sg_avoided_seconds', hit.sg_duration, {'layer': layer})
        return hit

    def _poll_shared_wait(
        self,
        backend: CacheBackend,
        base_key: str,
        plan: DependencyPlan,
        maximum_age: float,
    ) -> tuple[_LayerMiss, _LayerHit | None, FillLease | None]:
        """Poll until a shared hit, takeover lease, or final timeout lookup."""
        deadline = monotonic() + self._settings.limits.lease_wait_seconds
        attempt = 0
        while True:
            remaining = deadline - monotonic()
            if remaining <= 0:
                break
            delay = _wait_delay(attempt, remaining)
            if delay > 0:
                sleep(delay)
            attempt += 1
            miss, candidate = self._lookup(backend, base_key, plan.tags, maximum_age)
            if candidate is not None:
                return miss, candidate, None
            try:
                lease = backend.reserve(miss._key, self._settings.limits.fill_lease_seconds)
            except _BACKEND_ERRORS as error:
                raise _CachePathError(_BACKEND_FAILURE) from error
            if lease is not None:
                return miss, None, lease
        miss, candidate = self._lookup(backend, base_key, plan.tags, maximum_age)
        return miss, candidate, None

    def _record_bypass(self, reason: str) -> None:
        """Emit one fixed safe bypass reason without becoming a dependency."""
        self._increment('cache_bypass', {'reason': reason})

    def _handle_hierarchy_l1_error(self, error: _CachePathError) -> None:
        """Validate one recoverable L1 failure and record malformed data."""
        if error._reason not in {_BACKEND_FAILURE, _CODEC_FAILURE, 'unacknowledged_fence'}:
            raise error
        if error._reason == _CODEC_FAILURE:
            self._record_bypass(_CODEC_FAILURE)

    def _population_deadline(
        self,
        value: object,
        fetched_at: float,
        maximum_age: float,
        population_decision: PopulationCallback | None,
    ) -> float | None:
        """Apply one post-fetch decline or shortening without changing authority."""
        if population_decision is None:
            return fetched_at + maximum_age
        limit = population_limit(value, population_decision)
        if limit.reason is not None:
            self._observe('cache_url_age_cap', limit.seconds or 0.0, {'reason': limit.reason})
        if limit.seconds is None:
            self._record_bypass(limit.bypass_reason or 'population_decision_failure')
            return None
        return fetched_at + min(maximum_age, limit.seconds)

    def _release(self, backend: CacheBackend, lease: FillLease | None) -> None:
        """Best-effort release one held lease after every unsuccessful fill."""
        if lease is None:
            return
        try:
            backend.release(lease)
        except _BACKEND_ERRORS:
            self._record_bypass(_BACKEND_FAILURE)

    def _promote(
        self,
        backend: CacheBackend,
        miss: _LayerMiss,
        outcome: _SharedOutcome,
        deadline: float,
    ) -> bool:
        """Store one isolated L2 value under the original local vector."""
        lease: FillLease | None = None
        committed = False
        try:
            lease = backend.reserve(miss._key, self._settings.limits.fill_lease_seconds)
            if lease is None:
                self._increment('cache_lease_rejection', {'layer': 'hierarchical_l1'})
                return False
            self._increment('cache_fill', {'layer': 'hierarchical_l1'})
            lifetime = deadline - outcome._observed_at
            if lifetime <= 0:
                return False
            payload = encode_result(
                TimedResult(outcome._sg_duration or 0.0, outcome._value),
                outcome._origin_elapsed or 0.0,
                lifetime,
            )
            if len(payload) > self._settings.limits.max_serialized_result_bytes:
                self._record_bypass('result_too_large_bytes')
                return False
            remaining = deadline - monotonic()
            if remaining <= 0:
                return False
            committed = backend.commit(lease, payload, remaining)
        except CodecBoundError as error:
            self._record_bypass(bound_reason(error))
        except CodecError:
            self._record_bypass(_CODEC_FAILURE)
        except _BACKEND_ERRORS:
            self._record_bypass(_BACKEND_FAILURE)
        finally:
            self._release(backend, lease)
        return committed

    def _store_shared_with_lease(
        self,
        backend: CacheBackend,
        lease: FillLease,
        fetch: _Fetch,
        maximum_age: float,
        population_decision: PopulationCallback | None,
    ) -> _SharedOutcome:
        """Fetch and return the duration safe for a later local promotion."""
        layer: CacheLayer = 'local_l1' if backend is self._l1 and self._l2 is None else 'l2'
        self._increment('cache_fill', {'layer': layer})
        fetch_started = monotonic()
        value = fetch()
        fetched_at = monotonic()
        sg_duration = fetched_at - fetch_started
        promotion_deadline = self._population_deadline(
            value,
            fetched_at,
            maximum_age,
            population_decision,
        )
        if promotion_deadline is None:
            return _SharedOutcome(
                None,
                sg_duration,
                value,
                0.0,
                fetched_at,
                _key=lease.key,
            )
        lifetime = promotion_deadline - fetched_at
        try:
            payload = encode_result(TimedResult(sg_duration, value), 0.0, lifetime)
        except CodecBoundError as error:
            self._record_bypass(bound_reason(error))
            return _SharedOutcome(
                None,
                sg_duration,
                value,
                0.0,
                fetched_at,
                _key=lease.key,
            )
        except CodecError:
            self._record_bypass(_CODEC_FAILURE)
            return _SharedOutcome(
                None,
                sg_duration,
                value,
                0.0,
                fetched_at,
                _key=lease.key,
            )
        if len(payload) > self._settings.limits.max_serialized_result_bytes:
            self._record_bypass('result_too_large_bytes')
            return _SharedOutcome(
                None,
                sg_duration,
                value,
                0.0,
                fetched_at,
                _key=lease.key,
            )
        remaining = promotion_deadline - monotonic()
        if remaining <= 0:
            self._record_bypass('maximum_age_elapsed')
            return _SharedOutcome(
                None,
                sg_duration,
                value,
                0.0,
                fetched_at,
                _key=lease.key,
            )
        committed = False
        try:
            committed = backend.commit(lease, payload, remaining)
        except _BACKEND_ERRORS:
            self._record_bypass(_BACKEND_FAILURE)
        return _SharedOutcome(
            promotion_deadline if committed else None,
            sg_duration,
            value,
            0.0,
            fetched_at,
            _key=lease.key,
        )

    def _shared_reserved_outcome(  # noqa: PLR0913, PLR0917 - one held fill
        self,
        backend: CacheBackend,
        lease: FillLease,
        base_key: str,
        plan: DependencyPlan,
        fetch: _Fetch,
        maximum_age: float,
        population_decision: PopulationCallback | None,
    ) -> _SharedOutcome:
        """Recheck one successful shared reservation before fetching."""
        recheck_miss, candidate = self._lookup(
            backend,
            base_key,
            plan.tags,
            maximum_age,
        )
        if candidate is not None:
            hit = self._consume_hit(backend, candidate)
            return _SharedOutcome(
                candidate._deadline,
                hit.sg_duration,
                hit.value,
                hit.origin_elapsed,
                candidate._observed_at,
                _key=recheck_miss._key,
            )
        if recheck_miss._key != lease.key:
            return _SharedOutcome(
                None,
                None,
                fetch(),
                None,
                monotonic(),
                _key=lease.key,
            )
        return self._store_shared_with_lease(
            backend,
            lease,
            fetch,
            maximum_age,
            population_decision,
        )

    def _store_local(
        self,
        backend: CacheBackend,
        miss: _LayerMiss,
        fetch: _Fetch,
        maximum_age: float,
        population_decision: PopulationCallback | None,
    ) -> _SharedOutcome:
        """Fetch and conditionally populate one local-only layer."""
        lease: FillLease | None = None
        try:
            lease = backend.reserve(miss._key, self._settings.limits.fill_lease_seconds)
        except _BACKEND_ERRORS:
            self._record_bypass(_BACKEND_FAILURE)
        if lease is None:
            value = fetch()
            return _SharedOutcome(
                None,
                None,
                value,
                None,
                monotonic(),
                _key=miss._key,
            )
        try:
            return self._store_shared_with_lease(
                backend,
                lease,
                fetch,
                maximum_age,
                population_decision,
            )
        finally:
            self._release(backend, lease)

    def _wait_record(self) -> WaitRecord:
        """Return this thread's innermost read request's wait accounting."""
        return self._metrics_state._waits[-1]

    def _wait_for_flight(self, key: str, flight: _Flight) -> bool:
        """Wait within the configured budget and retire this participation."""
        record = self._wait_record()
        if record.begin('local'):
            self._increment('cache_collapse', {'outcome': 'local'})
        started_at = monotonic()
        outcome: WaitOutcome = 'cancelled'
        try:
            with self._flight_lock:
                try:
                    completed = flight._condition.wait_for(
                        lambda: flight._done,
                        timeout=self._settings.limits.lease_wait_seconds,
                    )
                finally:
                    flight._participants -= 1
                    if flight._participants == 0 and self._flights.get(key) is flight:
                        self._flights.pop(key, None)
            outcome = 'completed' if completed else 'timeout'
            if not completed and record.expire('local'):
                self._increment('cache_lease_timeout', {'outcome': 'local'})
            return completed
        finally:
            record.end('local', monotonic() - started_at, outcome)

    def _wait_for_shared(  # noqa: PLR0913, PLR0917 - preserve one fill decision
        self,
        backend: CacheBackend,
        base_key: str,
        plan: DependencyPlan,
        fetch: _Fetch,
        maximum_age: float,
        population_decision: PopulationCallback | None,
    ) -> _SharedOutcome:
        """Account bounded shared polling before a hit, takeover fill, or direct fetch."""
        record = self._wait_record()
        if record.begin('distributed'):
            self._increment('cache_collapse', {'outcome': 'distributed'})
        started_at = monotonic()
        outcome: WaitOutcome = 'cancelled'
        try:
            miss, candidate, lease = self._poll_shared_wait(backend, base_key, plan, maximum_age)
            outcome = (
                'completed'
                if candidate is not None
                else ('takeover' if lease is not None else 'timeout')
            )
        except _CachePathError:
            outcome = 'failure'
            raise
        finally:
            record.end('distributed', monotonic() - started_at, outcome)
        if candidate is not None:
            hit = self._consume_hit(backend, candidate)
            return _SharedOutcome(
                candidate._deadline,
                hit.sg_duration,
                hit.value,
                hit.origin_elapsed,
                candidate._observed_at,
                _key=miss._key,
            )
        if lease is not None:
            try:
                return self._shared_reserved_outcome(
                    backend,
                    lease,
                    base_key,
                    plan,
                    fetch,
                    maximum_age,
                    population_decision,
                )
            finally:
                self._release(backend, lease)
        if record.expire('distributed'):
            self._increment('cache_lease_timeout', {'outcome': 'distributed'})
        return _SharedOutcome(None, None, fetch(), None, monotonic(), _key=miss._key)

    def _shared_outcome(  # noqa: PLR0913, PLR0917 - preserve one fill decision
        self,
        backend: CacheBackend,
        base_key: str,
        plan: DependencyPlan,
        fetch: _Fetch,
        maximum_age: float,
        population_decision: PopulationCallback | None,
    ) -> _SharedOutcome:
        """Return one L2 hit, held fill, or nonpopulating timeout result."""
        miss, candidate = self._lookup(
            backend,
            base_key,
            plan.tags,
            maximum_age,
        )
        if candidate is not None:
            hit = self._consume_hit(backend, candidate)
            return _SharedOutcome(
                candidate._deadline,
                hit.sg_duration,
                hit.value,
                hit.origin_elapsed,
                candidate._observed_at,
                _key=miss._key,
            )
        try:
            lease = backend.reserve(miss._key, self._settings.limits.fill_lease_seconds)
        except _BACKEND_ERRORS as error:
            raise _CachePathError(_BACKEND_FAILURE) from error
        if lease is not None:
            try:
                return self._shared_reserved_outcome(
                    backend,
                    lease,
                    base_key,
                    plan,
                    fetch,
                    maximum_age,
                    population_decision,
                )
            finally:
                self._release(backend, lease)
        return self._wait_for_shared(
            backend,
            base_key,
            plan,
            fetch,
            maximum_age,
            population_decision,
        )

    @property
    def bypass_active(self) -> bool:
        """Report whether an active lexical scope bypasses the cache."""
        return policy_bypass_active(tuple(self._policy_state._stack))

    @staticmethod
    def build_metrics(
        settings: CacheSettings,
        environ: Mapping[str, object],
        sink: MetricsSink,
    ) -> MetricsSink:
        """Build the supported safe metrics boundary from cache settings.

        Args:
            settings: Loaded cache settings naming an optional secret.
            environ: Environment-style mapping used only during construction.
            sink: Destination metrics implementation.

        Returns:
            A safe sink with optional keyed exact-call correlation.

        Raises:
            ValueError: A configured secret is absent, non-UTF-8, or shorter
                than 32 encoded bytes.
        """
        environment_name = settings.metrics_digest_secret_env
        if environment_name is None:
            return SafeMetrics(sink)
        value = environ.get(environment_name)
        text = value if isinstance(value, str) else ''
        try:
            metrics_key = text.encode('utf-8')
            return SafeMetrics(sink, metrics_key)
        except (UnicodeEncodeError, ValueError):
            raise ValueError(_CORRELATION_ERROR) from None

    def _select_age(
        self,
        call: ExactCall,
        plan: DependencyPlan,
        layer: CacheLayer,
        per_call_age: float | None,
        remaining_l2_seconds: float | None = None,
    ) -> float:
        """Select one layer age through the accepted TTL boundary."""
        context = TTLContext.from_call(call, self._identity, plan, layer=layer)
        if remaining_l2_seconds is not None:
            context = replace(context, remaining_l2_seconds=remaining_l2_seconds)
        maximum_age = select_maximum_age(self._settings, context, per_call_age)
        self._observe(
            'cache_ttl_selection',
            maximum_age,
            {'classification': plan.classification, 'layer': layer},
        )
        return maximum_age

    @contextmanager
    def _correlation_scope(self, digest: str | None) -> Generator[None]:
        """Apply one nest-safe exact-call metric correlation scope."""
        stack = self._metrics_state._stack
        stack.append(digest)
        try:
            yield
        finally:
            popped = stack.pop()
            if popped is not digest:
                raise RuntimeError('query cache metrics stack is corrupted')

    def _memo_value(
        self,
        first_key: str,
        call: ExactCall,
        plan: DependencyPlan,
        base_key: str,
        parse: Parse,
    ) -> object:
        """Return the memoized value when its keys, parser, and deadlines still hold."""
        with self._memo_lock:
            entry = self._memo
        if entry is None or entry._first_key != first_key or entry._parse is not parse:
            return _MISS
        now = monotonic()
        effective_age = self._effective_age(None)
        if effective_age is not None and now - entry._origin_at >= effective_age:
            return _MISS
        if now < entry._first_deadline:
            self._increment('cache_schema_view', {'outcome': 'reuse'})
            return entry._value
        l2 = self._l2
        if (
            self._l1 is None
            or l2 is None
            or now >= entry._deep_deadline
            or entry._shared_key is None
        ):
            return _MISS
        fence = self._fence_snapshot('l2')
        shared_key = None
        if not fence._retained:
            try:
                shared_key = self._key_builder.qualified_key(
                    base_key,
                    l2.resolve(plan.tags),
                    'result',
                )
            except (*_BACKEND_ERRORS, NormalizationError):
                shared_key = None
        if self._fence_unchanged('l2', fence) and shared_key == entry._shared_key:
            refreshed_at = monotonic()
            fresh_enough = refreshed_at < entry._deep_deadline and (
                effective_age is None or refreshed_at - entry._origin_at < effective_age
            )
            if fresh_enough:
                first_age = self._select_age(
                    call,
                    plan,
                    'hierarchical_l1',
                    effective_age,
                    entry._deep_deadline - refreshed_at,
                )
                refreshed = replace(
                    entry,
                    _first_deadline=min(refreshed_at + first_age, entry._deep_deadline),
                )
                with self._memo_lock:
                    if self._memo is entry:
                        self._memo = refreshed
                self._increment('cache_schema_view', {'outcome': 'revalidated'})
                return entry._value
        return _MISS

    def _promotion_deadline(
        self,
        call: ExactCall,
        plan: DependencyPlan,
        per_call_age: float | None,
        outcome: _SharedOutcome,
    ) -> float | None:
        """Anchor the L1 deadline at the moment the shared value was observed."""
        deep_deadline = outcome._promotion_deadline
        if deep_deadline is None:
            return None
        remaining = deep_deadline - outcome._observed_at
        if remaining <= 0:
            return None
        l1_age = self._select_age(
            call,
            plan,
            'hierarchical_l1',
            per_call_age,
            remaining,
        )
        return min(deep_deadline, outcome._observed_at + l1_age)

    def _read_hierarchy(  # noqa: PLR0913, PLR0917 - preserve one fill decision
        self,
        base_key: str,
        call: ExactCall,
        plan: DependencyPlan,
        fetch: _Fetch,
        per_call_age: float | None,
        population_decision: PopulationCallback | None,
    ) -> _ReadOutcome:
        """Read L1 then L2 while preserving vector and duration boundaries."""
        l1 = self._l1
        l2 = self._l2
        if l1 is None or l2 is None:
            raise _CachePathError(_BACKEND_FAILURE)
        l1_age = self._select_age(call, plan, 'hierarchical_l1', per_call_age)
        try:
            l1_miss, candidate = self._lookup(
                l1,
                base_key,
                plan.tags,
                l1_age,
            )
        except _CachePathError as error:
            self._handle_hierarchy_l1_error(error)
            l2_age = self._select_age(call, plan, 'l2', per_call_age)
            outcome = self._shared_outcome(
                l2,
                base_key,
                plan,
                fetch,
                l2_age,
                population_decision,
            )
            deep_deadline = outcome._promotion_deadline
            if deep_deadline is None:
                deep_deadline = outcome._observed_at + l2_age
            return _ReadOutcome(
                outcome._value,
                outcome._key,
                outcome._key,
                _shared_origin_at(outcome),
                deep_deadline,
                deep_deadline,
            )
        if candidate is not None:
            deadline = candidate._deadline
            hit = self._consume_hit(l1, candidate)
            return _ReadOutcome(
                hit.value,
                l1_miss._key,
                None,
                candidate._observed_at - hit.origin_elapsed,
                deadline,
                deadline,
            )
        flight, leader = self._join_flight(l1_miss._key)
        if leader:
            try:
                l2_age = self._select_age(call, plan, 'l2', per_call_age)
                outcome = self._shared_outcome(
                    l2,
                    base_key,
                    plan,
                    fetch,
                    l2_age,
                    population_decision,
                )
                deadline = self._promotion_deadline(
                    call,
                    plan,
                    per_call_age,
                    outcome,
                )
                promoted = deadline is not None and self._promote(
                    l1,
                    l1_miss,
                    outcome,
                    deadline,
                )
                deep_deadline = outcome._promotion_deadline
                if deep_deadline is None:
                    deep_deadline = outcome._observed_at + l2_age
                first_deadline = (
                    deadline
                    if promoted and deadline is not None
                    else min(deep_deadline, outcome._observed_at + l1_age)
                )
                return _ReadOutcome(
                    outcome._value,
                    l1_miss._key,
                    outcome._key,
                    _shared_origin_at(outcome),
                    first_deadline,
                    deep_deadline,
                )
            finally:
                self._finish_flight(l1_miss._key, flight)
        completed = self._wait_for_flight(l1_miss._key, flight)
        first_key = l1_miss._key
        if completed:
            try:
                retry_miss, candidate = self._lookup(
                    l1,
                    base_key,
                    plan.tags,
                    l1_age,
                )
                first_key = retry_miss._key
            except _CachePathError as error:
                self._record_bypass(error._reason)
                candidate = None
            if candidate is not None:
                deadline = candidate._deadline
                hit = self._consume_hit(l1, candidate)
                return _ReadOutcome(
                    hit.value,
                    first_key,
                    None,
                    candidate._observed_at - hit.origin_elapsed,
                    deadline,
                    deadline,
                )
        value = fetch()
        origin_at = monotonic()
        deadline = origin_at + l1_age
        return _ReadOutcome(value, first_key, None, origin_at, deadline, deadline)

    def _read_local(  # noqa: PLR0913, PLR0917 - preserve one fill decision
        self,
        base_key: str,
        call: ExactCall,
        plan: DependencyPlan,
        fetch: _Fetch,
        per_call_age: float | None,
        population_decision: PopulationCallback | None,
    ) -> _ReadOutcome:
        """Read through local-only storage with bounded single-flight."""
        backend = self._l1
        if backend is None:
            raise _CachePathError(_BACKEND_FAILURE)
        maximum_age = self._select_age(call, plan, 'local_l1', per_call_age)
        miss, candidate = self._lookup(
            backend,
            base_key,
            plan.tags,
            maximum_age,
        )
        if candidate is not None:
            deadline = candidate._deadline
            hit = self._consume_hit(backend, candidate)
            return _ReadOutcome(
                hit.value,
                miss._key,
                None,
                candidate._observed_at - hit.origin_elapsed,
                deadline,
                deadline,
            )
        flight, leader = self._join_flight(miss._key)
        if leader:
            try:
                outcome = self._store_local(
                    backend,
                    miss,
                    fetch,
                    maximum_age,
                    population_decision,
                )
                deadline = outcome._promotion_deadline
                if deadline is None:
                    deadline = outcome._observed_at + maximum_age
                return _ReadOutcome(
                    outcome._value,
                    outcome._key,
                    None,
                    _shared_origin_at(outcome),
                    deadline,
                    deadline,
                )
            finally:
                self._finish_flight(miss._key, flight)
        completed = self._wait_for_flight(miss._key, flight)
        first_key = miss._key
        if completed:
            try:
                retry_miss, candidate = self._lookup(
                    backend,
                    base_key,
                    plan.tags,
                    maximum_age,
                )
                first_key = retry_miss._key
            except _CachePathError as error:
                self._record_bypass(error._reason)
                candidate = None
            if candidate is not None:
                deadline = candidate._deadline
                hit = self._consume_hit(backend, candidate)
                return _ReadOutcome(
                    hit.value,
                    first_key,
                    None,
                    candidate._observed_at - hit.origin_elapsed,
                    deadline,
                    deadline,
                )
        value = fetch()
        origin_at = monotonic()
        deadline = origin_at + maximum_age
        return _ReadOutcome(value, first_key, None, origin_at, deadline, deadline)

    def _read_shared(  # noqa: PLR0913, PLR0917 - preserve one fill decision
        self,
        base_key: str,
        call: ExactCall,
        plan: DependencyPlan,
        fetch: _Fetch,
        per_call_age: float | None,
        population_decision: PopulationCallback | None,
    ) -> _ReadOutcome:
        """Read shared-only storage with bounded distributed-lease waiting."""
        backend = self._l2
        if backend is None:
            raise _CachePathError(_BACKEND_FAILURE)
        maximum_age = self._select_age(call, plan, 'l2', per_call_age)
        outcome = self._shared_outcome(
            backend,
            base_key,
            plan,
            fetch,
            maximum_age,
            population_decision,
        )
        deadline = outcome._promotion_deadline
        if deadline is None:
            deadline = outcome._observed_at + maximum_age
        return _ReadOutcome(
            outcome._value,
            outcome._key,
            outcome._key,
            _shared_origin_at(outcome),
            deadline,
            deadline,
        )

    def _read_cacheable(  # noqa: PLR0913, PLR0917 - preserve one fill decision
        self,
        base_key: str,
        call: ExactCall,
        plan: DependencyPlan,
        fetch: _Fetch,
        per_call_age: float | None,
        population_decision: PopulationCallback | None,
    ) -> _ReadOutcome:
        """Dispatch one cacheable call to exactly one supported topology."""
        if self._l1 is not None and self._l2 is None:
            return self._read_local(
                base_key,
                call,
                plan,
                fetch,
                per_call_age,
                population_decision,
            )
        if self._l1 is None and self._l2 is not None:
            return self._read_shared(
                base_key,
                call,
                plan,
                fetch,
                per_call_age,
                population_decision,
            )
        return self._read_hierarchy(
            base_key,
            call,
            plan,
            fetch,
            per_call_age,
            population_decision,
        )

    def _acquire_parsed(
        self,
        base_key: str,
        call: ExactCall,
        plan: DependencyPlan,
        fetch: _Fetch,
        parse: Parse,
    ) -> object:
        """Read, parse, and publish one memo entry, or fall back to a direct parse."""
        with self._correlation_scope(None):
            try:
                outcome = self._read_cacheable(
                    base_key,
                    call,
                    plan,
                    fetch,
                    self._effective_age(None),
                    None,
                )
            except _CachePathError as error:
                self._record_bypass(error._reason)
                return parse(fetch())
        result = 'rebuild'
        try:
            value = parse(outcome._value)
        except (TypeError, ValueError):
            self._increment('cache_schema_view', {'outcome': 'malformed'})
            raw_value = fetch()
            fetched_at = monotonic()
            value = parse(raw_value)
            effective_age = self._effective_age(None)
            if self._l1 is None:
                first_age = self._select_age(call, plan, 'l2', effective_age)
                deep_age = first_age
            elif self._l2 is None:
                first_age = self._select_age(call, plan, 'local_l1', effective_age)
                deep_age = first_age
            else:
                first_age = self._select_age(call, plan, 'hierarchical_l1', effective_age)
                deep_age = self._select_age(call, plan, 'l2', effective_age)
            outcome = replace(
                outcome,
                _origin_at=fetched_at,
                _first_deadline=fetched_at + first_age,
                _deep_deadline=fetched_at + deep_age,
            )
            result = 'refetch'
        with self._memo_lock:
            self._memo = _MemoEntry(
                outcome._first_key,
                outcome._shared_key,
                outcome._origin_at,
                outcome._first_deadline,
                outcome._deep_deadline,
                parse,
                value,
            )
        self._increment('cache_schema_view', {'outcome': result})
        return value

    def _read_parsed_cacheable(
        self,
        base_key: str,
        call: ExactCall,
        plan: DependencyPlan,
        fetch: _Fetch,
        parse: Parse,
    ) -> object:
        """Reuse or acquire one parsed value for a normalized cacheable call."""
        first = self._l1 if self._l1 is not None else self._l2
        first_layer: _FenceLayer = 'l1' if self._l1 is not None else 'l2'
        if first is None:
            raise _CachePathError(_BACKEND_FAILURE)
        while True:
            fence = self._fence_snapshot(first_layer)
            if fence._retained:
                return self._acquire_parsed(base_key, call, plan, fetch, parse)
            try:
                first_key = self._key_builder.qualified_key(
                    base_key,
                    first.resolve(plan.tags),
                    'result',
                )
            except (*_BACKEND_ERRORS, NormalizationError):
                return self._acquire_parsed(base_key, call, plan, fetch, parse)
            if not self._fence_unchanged(first_layer, fence):
                return self._acquire_parsed(base_key, call, plan, fetch, parse)
            reused = self._memo_value(first_key, call, plan, base_key, parse)
            if reused is not _MISS:
                return reused
            flight_key = f'memo:{first_key}'
            flight, leader = self._join_flight(flight_key)
            if leader:
                try:
                    return self._acquire_parsed(base_key, call, plan, fetch, parse)
                finally:
                    self._finish_flight(flight_key, flight)
            if not self._wait_for_flight(flight_key, flight):
                return self._acquire_parsed(base_key, call, plan, fetch, parse)

    @contextmanager
    def _wait_scope(self) -> Generator[None]:
        """Collect and emit one request's waits while retaining its metric correlation."""
        record = WaitRecord()
        stack = self._metrics_state._waits
        stack.append(record)
        try:
            yield
        finally:
            popped = stack.pop()
            if popped is not record:
                raise RuntimeError('query cache wait stack is corrupted')
            for seconds, outcome in record.observations():
                self._observe('cache_wait_seconds', seconds, {'outcome': outcome})

    def close(self) -> None:
        """Close configured backends in L1-then-L2 order.

        Backend close failures propagate immediately, so a failed L1 close
        prevents the L2 close attempt. Repeated calls are delegated to the
        backends. This method does not wait for active cache operations.
        """
        for backend in (self._l1, self._l2):
            if backend is not None:
                backend.close()

    def invalidate(self, changes: tuple[GenerationChange, ...]) -> bool:
        """Apply ordered generation changes to L2 and then L1.

        Args:
            changes: Ordered immutable changes derived by the client wrapper.

        Returns:
            Whether every configured layer acknowledged the changes. A layer
            that did not acknowledge retains them, serves no hit, and replays
            them ahead of the next invalidation sent to it.
        """
        acknowledged = True
        pairs: tuple[tuple[_FenceLayer, CacheBackend | None], ...] = (
            ('l2', self._l2),
            ('l1', self._l1),
        )
        for layer, backend in pairs:
            if backend is None:
                continue
            with self._fence_lock:
                snapshot = self._fences[layer]
            attempt = merge_changes(snapshot._retained, changes)
            try:
                backend.change(attempt)
            except _BACKEND_ERRORS:
                acknowledged = False
                self._record_bypass(_BACKEND_FAILURE)
                self._fence_failed(layer, snapshot, attempt)
            else:
                self._fence_succeeded(layer, snapshot)
        return acknowledged

    @contextmanager
    def policy(
        self,
        mode: PolicyMode | None = None,
        maximum_age: float | None = None,
    ) -> Generator[None]:
        """Apply a nestable thread-local cache policy.

        Nested scopes can only downgrade policy. Scoped and per-call ages
        combine by minimum; bypass dominates every inner scope.

        Args:
            mode: Optional event-aware, TTL-only, or bypass decision.
            maximum_age: Optional positive age in seconds, combined by minimum
                with other scoped and per-call ages. The resulting explicit age
                replaces configured rules and defaults; global safety,
                response, remaining-L2, and hierarchical-L1 caps still apply.

        Yields:
            Control while the validated policy is active.

        Raises:
            TypeError: A supplied maximum age is not a finite non-boolean
                number.
            ValueError: The mode is unsupported or the maximum age is not
                positive.
        """
        frame = policy_frame(mode, maximum_age)
        stack = self._policy_state._stack
        stack.append(frame)
        try:
            yield
        finally:
            popped = stack.pop()
            if popped is not frame:
                raise RuntimeError('query cache policy stack is corrupted')

    @contextmanager
    def bypass(self) -> Generator[None]:
        """Bypass every cache read and population within this thread-local scope."""
        with self.policy(mode='bypass'):
            yield

    def read(
        self,
        call: ExactCall,
        plan: DependencyPlan,
        fetch: _Fetch,
        client_options: Mapping[str, object],
        per_call_age: float | None = None,
        *,
        _population_decision: PopulationCallback | None = None,
    ) -> object:
        """Read one exact call through the configured cache topology.

        Args:
            call: Explicit normalized SG read call.
            plan: Complete conservative pre-fetch dependency plan.
            fetch: Caller-owned synchronous SG operation.
            client_options: Complete response-affecting client options.
            per_call_age: Optional positive maximum age in seconds, combined by
                minimum with every active policy scope. The resulting explicit
                age replaces configured rules before safety caps are applied.
            _population_decision: Optional private post-fetch population cap.

        Returns:
            An isolated cache hit or the direct SG result.

        Raises:
            TypeError: A non-bypassed call supplies a nonfinite, boolean, or
                nonnumeric maximum age.
            ValueError: A non-bypassed call supplies a nonpositive maximum age.
        """
        with self._correlation_scope(None):
            effective_plan = self._effective_plan(plan)
            self._increment(
                'cache_query_classification',
                {
                    'classification': effective_plan.classification,
                    'reason': effective_plan.reason,
                },
            )
            for step in effective_plan.broadening_steps:
                self._increment('cache_dependency_broadening', {'reason': step})
            if effective_plan.classification == 'bypass':
                self._record_bypass(effective_plan.reason)
                return fetch()
            effective_age = self._effective_age(per_call_age)
            try:
                base_key = self._key_builder.base_key(
                    self._settings.site_namespace,
                    self._identity,
                    client_options,
                    call,
                )
            except NormalizationError:
                self._record_bypass('normalization_failure')
                return fetch()
            digest = None
            if isinstance(self._metrics, SafeMetrics):
                with suppress(Exception):
                    digest = self._metrics.correlation_digest(self._key_builder, base_key)
            with self._correlation_scope(digest), self._wait_scope():
                try:
                    return self._read_cacheable(
                        base_key,
                        call,
                        effective_plan,
                        fetch,
                        effective_age,
                        _population_decision,
                    )._value
                except _CachePathError as error:
                    self._record_bypass(error._reason)
                    return fetch()

    def read_parsed(
        self,
        call: ExactCall,
        plan: DependencyPlan,
        fetch: _Fetch,
        client_options: Mapping[str, object],
        parse: Parse,
    ) -> object:
        """Read one exact call and return its parsed value through a bounded memo.

        When rebuilding the parsed memo, a first parse that raises `TypeError`
        or `ValueError` triggers one direct refetch and another parse. Fetch
        failures and a failing second parse propagate. Direct fallback parsing
        is not retried.
        Memo hits may return the same immutable object.

        Args:
            call: Explicit normalized SG read call.
            plan: Complete conservative pre-fetch dependency plan.
            fetch: Caller-owned synchronous SG operation returning a storable
                value.
            client_options: Complete response-affecting client options.
            parse: Pure conversion from the stored value to an immutable parsed
                value. Parser identity is part of memo reuse.

        Returns:
            The parsed value for the current generation-qualified key.
        """
        effective_plan = self._effective_plan(plan)
        if effective_plan.classification == 'bypass':
            self._record_bypass(effective_plan.reason)
            return parse(fetch())
        try:
            base_key = self._key_builder.base_key(
                self._settings.site_namespace,
                self._identity,
                client_options,
                call,
            )
        except NormalizationError:
            self._record_bypass('normalization_failure')
            return parse(fetch())
        with self._wait_scope():
            return self._read_parsed_cacheable(
                base_key,
                call,
                effective_plan,
                fetch,
                parse,
            )


def _debit_hit_processing_time(
    hit: DecodedHit,
    observed_at: float,
    authoritative_deadline: float,
    maximum_age: float,
) -> _LayerHit | None:
    """Debit core processing time before publishing one decoded hit."""
    decided_at = monotonic()
    processing_elapsed = max(0.0, decided_at - observed_at)
    if (
        decided_at >= authoritative_deadline
        or hit.origin_elapsed + processing_elapsed >= maximum_age
    ):
        return None
    return _LayerHit(
        replace(hit, origin_elapsed=hit.origin_elapsed + processing_elapsed),
        authoritative_deadline,
        decided_at,
    )


def _shared_origin_at(outcome: _SharedOutcome) -> float:
    """Recover the monotonic value origin from one shared outcome."""
    return outcome._observed_at - (outcome._origin_elapsed or 0.0)


def _wait_delay(attempt: int, remaining: float) -> float:
    """Return one jittered bounded poll delay for a held distributed lease."""
    base_delay = min(0.01 * 2 ** min(attempt, 4), 0.1)
    return min(remaining, base_delay * uniform(0.75, 1.25))  # noqa: S311 - non-security wait jitter


def client_binding(cache: QueryCache) -> tuple[CacheSettings, Identity]:
    """Return the read-only binding used by the package's SG client adapter.

    Args:
        cache: Query cache being wrapped by the package's SG client adapter.

    Returns:
        The settings and exact effective identity retained by this cache.
    """
    return cache._settings, cache._identity
