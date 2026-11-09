"""Compose native asynchronous reads, parsed memo reuse, fences, and terminal close."""

from asyncio import (
    CancelledError,
    Future,
    Task,
    create_task,
    get_running_loop,
    shield,
    sleep,
    timeout_at,
)
from collections.abc import AsyncGenerator, Awaitable, Callable, Generator, Mapping
from contextlib import asynccontextmanager, contextmanager, suppress
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from functools import partial
from inspect import iscoroutinefunction
from time import monotonic
from types import TracebackType
from typing import Literal, Self, TypeAlias, TypeVar

from sg_query_cache._async_backend import AsyncCacheBackend
from sg_query_cache._async_support import (
    Admission,
    CleanupOutcome,
    DeadlineRecord,
    OperationToken,
    enter_cleanup,
    join_shared_close,
    joined_cleanup,
    publish_deadline,
)
from sg_query_cache._backend import (
    BACKEND_ERRORS,
    BackendUnavailable,
)
from sg_query_cache._cache import Parse, QueryCache
from sg_query_cache._cache_common import (
    PolicyFrame,
    PolicyMode,
    PopulationCallback,
    StaleHitError,
    TimedResult,
    WaitCategory,
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
from sg_query_cache._model import (
    DependencyPlan,
    ExactCall,
    FillLease,
    GenerationChange,
    Identity,
)

_T = TypeVar('_T')

_Fetch: TypeAlias = Callable[[], Awaitable[object]]
_Layer: TypeAlias = Literal['l1', 'l2']

_METHODS = ('change', 'close', 'commit', 'flush_site', 'lookup', 'release', 'reserve', 'resolve')
_MISS = object()


class _CachePathError(Exception):
    """Redirect one failed cache path to caller-owned SG I/O."""

    def __init__(self, reason: str) -> None:
        """Retain one safe metric reason."""
        super().__init__(reason)
        self.reason = reason


@dataclass(slots=True)
class _CloseState:
    """Retain backend obligations and one outcome through shared-task interruption."""

    pending: list[AsyncCacheBackend]
    completion: Future[CleanupOutcome[None]]
    settling: bool = False


@dataclass(frozen=True, slots=True)
class _FenceState:
    """Pair retained work with the revision that owns its acknowledgment."""

    retained: tuple[GenerationChange, ...] = ()
    revision: int = 0


@dataclass(slots=True)
class _LeaseHold:
    """Record whether a successful commit consumed one held lease."""

    committed: bool = False


@dataclass(slots=True)
class _Obligation:
    """Track one fence layer until acknowledged or synchronously retained."""

    layer: _Layer
    backend: AsyncCacheBackend
    changes: tuple[GenerationChange, ...]
    settled: bool = False
    acknowledged: bool = False


@dataclass(frozen=True, slots=True)
class _ReadInput:
    """Keep the immutable inputs shared by one read's asynchronous phases."""

    base_key: str
    call: ExactCall
    plan: DependencyPlan
    fetch: _Fetch
    age: float | None
    population: PopulationCallback | None = None
    waits: WaitRecord = field(default_factory=WaitRecord, compare=False)


@dataclass(frozen=True, slots=True)
class _Resolved:
    """Retain the original qualified key and its local fence snapshot."""

    backend: AsyncCacheBackend
    layer: _Layer
    key: str
    fence: _FenceState
    age: float


@dataclass(frozen=True, slots=True)
class _Outcome:
    """Carry a value with its original age and layer validity for promotion and memo."""

    value: object
    first: _Resolved
    shared: _Resolved | None
    origin_at: float
    observed_at: float
    first_deadline: float
    deep_deadline: float
    sg_duration: float
    stored: bool
    from_cache: bool = False


@dataclass(frozen=True, slots=True)
class _MemoEntry:
    """Retain exactly one immutable parsed result and its acquisition deadlines."""

    outcome: _Outcome
    parse: Parse
    value: object


class AsyncQueryCache:
    """Cache exact SG reads on one event loop with terminal resource ownership.

    Construction is loop-independent; the first operation binds the instance.
    Close joins admitted operations before closing all backends. Start close
    outside their dependency chains; only direct owner self-close is detected.
    The async context manager closes this cache and preserves a body exception
    ahead of ordinary cleanup failure, with caller cancellation taking
    precedence.

    Backend failures on read paths fall back to the caller's fetch, whose
    failures propagate. The cache never owns or closes the fetch callback's SG
    client.
    """

    def __init__(  # noqa: PLR0913 - required public topology interface
        self,
        settings: CacheSettings,
        identity: Identity,
        *,
        key_builder: KeyBuilder | None = None,
        l1: AsyncCacheBackend | None = None,
        l2: AsyncCacheBackend | None = None,
        metrics: MetricsSink | None = None,
    ) -> None:
        """Validate topology and retain the supplied backend resources.

        Args:
            settings: Immutable policy and operational bounds.
            identity: Exact effective SG identity namespace.
            key_builder: Optional standard-compatible exact key builder.
            l1: Local backend when enabled by settings.
            l2: Shared backend when enabled by settings, distinct from L1.
            metrics: Optional synchronous, prompt metrics destination.

        Raises:
            ValueError: Wiring disagrees with settings or a backend method is
                not a coroutine function; the error names that method.
        """
        expected = (settings.l1.enabled, settings.l2.enabled)
        supplied = (l1 is not None, l2 is not None)
        if supplied != expected or (l1 is not None and l1 is l2):
            l1_state = 'enabled' if settings.l1.enabled else 'disabled'
            l2_state = 'enabled' if settings.l2.enabled else 'disabled'
            supplied_description = {
                (False, False): 'no backends were supplied',
                (True, False): 'l1 was supplied',
                (False, True): 'l2 was supplied',
                (True, True): 'l1 and l2 were supplied',
            }[supplied]
            detail = '; l1 and l2 must be distinct backends' if l1 is not None and l1 is l2 else ''
            raise ValueError(
                'backend wiring does not match the settings: '
                f'L1 {l1_state} and L2 {l2_state}, but {supplied_description}{detail}'
            )
        for backend in (l1, l2):
            if backend is not None:
                for method in _METHODS:
                    if not iscoroutinefunction(getattr(backend, method, None)):
                        raise ValueError(
                            f'async backend method {method} must be a coroutine function'
                        )
        self._admission = Admission()
        self._close_task: Task[CleanupOutcome[None]] | None = None
        self._close_state: _CloseState | None = None
        self._fences = {'l1': _FenceState(), 'l2': _FenceState()}
        self._flights: dict[str, Future[None]] = {}
        self.identity = identity
        self._key_builder = KeyBuilder() if key_builder is None else key_builder
        self._l1 = l1
        self._l2 = l2
        self._memo: _MemoEntry | None = None
        self._metrics = NullMetrics() if metrics is None else metrics
        self._correlations: ContextVar[tuple[str | None, ...]] = ContextVar(
            'correlations', default=()
        )
        self._policies: ContextVar[tuple[PolicyFrame, ...]] = ContextVar('policies', default=())
        self.settings = settings

    def _deadline(self) -> float:
        """Anchor one cooperative operation budget on the bound running loop."""
        return get_running_loop().time() + self.settings.limits.operation_timeout_seconds

    def _finish_flight(self, key: str, flight: Future[None]) -> None:
        """Retire the exact leader and publish notification without sharing a value."""
        if self._flights.get(key) is flight:
            del self._flights[key]
        flight.set_result(None)

    def _increment(self, name: str, labels: Mapping[str, str]) -> None:
        """Emit an isolated counter after any associated state decision."""
        correlated = dict(labels)
        stack = self._correlations.get()
        if stack and stack[-1] is not None:
            correlated['exact_call_digest'] = stack[-1]
        with suppress(Exception):
            self._metrics.increment(name, correlated)

    def _begin_wait(self, waits: WaitRecord, category: WaitCategory) -> float:
        """Count the category's first wait and return its loop-clock start."""
        if waits.begin(category):
            self._increment('cache_collapse', {'outcome': category})
        return get_running_loop().time()

    def _bypass(self, reason: str) -> None:
        """Record one safe cache bypass reason."""
        self._increment('cache_bypass', {'reason': reason})

    def _expire_wait(self, waits: WaitRecord, category: WaitCategory) -> None:
        """Count at most one expired budget per request and wait category."""
        if waits.expire(category):
            self._increment('cache_lease_timeout', {'outcome': category})

    def _layer_name(self, layer: _Layer) -> CacheLayer:
        """Select the configured age and metric name for one backend layer."""
        if layer == 'l2':
            return 'l2'
        return 'local_l1' if self._l2 is None else 'hierarchical_l1'

    async def _direct_fetch(self, fetch: _Fetch) -> object:
        """Deliver pending cancellation before dispatching a bypassed SG operation."""
        try:
            await sleep(0)
            return await fetch()
        except CancelledError:
            layer = self._layer_name('l1' if self._l1 is not None else 'l2')
            self._increment('cache_fill', {'layer': layer, 'outcome': 'cancelled'})
            raise

    async def _fetch(self, request: _ReadInput, resolved: _Resolved) -> _Outcome:
        """Record a fill only after SG returns, preserving cancellation as its own outcome."""
        layer = self._layer_name(resolved.layer)
        started_at = monotonic()
        try:
            await sleep(0)
            value = await request.fetch()
        except CancelledError:
            self._increment('cache_fill', {'layer': layer, 'outcome': 'cancelled'})
            raise
        fetched_at = monotonic()
        outcome = _Outcome(
            value,
            resolved,
            resolved if resolved.layer == 'l2' else None,
            fetched_at,
            fetched_at,
            fetched_at + resolved.age,
            fetched_at + resolved.age,
            fetched_at - started_at,
            stored=False,
        )
        self._increment('cache_fill', {'layer': layer})
        return outcome

    def _observe(self, name: str, value: float, labels: Mapping[str, str]) -> None:
        """Emit an isolated observation after any associated state decision."""
        correlated = dict(labels)
        stack = self._correlations.get()
        if stack and stack[-1] is not None:
            correlated['exact_call_digest'] = stack[-1]
        with suppress(Exception):
            self._metrics.observe(name, value, correlated)

    def _age(self, request: _ReadInput, layer: _Layer) -> float:
        """Select a layer's age through the shared policy owner."""
        name = self._layer_name(layer)
        context = TTLContext.from_call(request.call, self.identity, request.plan, layer=name)
        age = select_maximum_age(self.settings, context, request.age)
        self._observe(
            'cache_ttl_selection',
            age,
            {'classification': request.plan.classification, 'layer': name},
        )
        return age

    def _emit_waits(self, waits: WaitRecord) -> None:
        """Emit each category's accumulated duration and latest wait outcome."""
        for seconds, outcome in waits.observations():
            self._observe('cache_wait_seconds', seconds, {'outcome': outcome})

    @staticmethod
    def _owns_deadline(backend: AsyncCacheBackend) -> bool:
        """Detect explicit deadline ownership without extending the backend protocol."""
        return getattr(type(backend), 'owns_published_deadline', False) is True

    async def _await_backend(
        self,
        backend: AsyncCacheBackend,
        action: Callable[[], Awaitable[_T]],
        record: DeadlineRecord,
    ) -> _T:
        """Let the owning backend time its action or retain the custom-backend timer."""
        if self._owns_deadline(backend):
            return await action()
        timer = timeout_at(record.deadline)
        try:
            async with timer:
                return await action()
        finally:
            if timer.expired():
                record.expired = True

    def _population_deadline(self, request: _ReadInput, outcome: _Outcome) -> float | None:
        """Apply the shared callback contract without changing the SG origin sample."""
        if request.population is None:
            return outcome.deep_deadline
        limit = population_limit(outcome.value, request.population)
        if limit.reason is not None:
            self._observe('cache_url_age_cap', limit.seconds or 0.0, {'reason': limit.reason})
        if limit.seconds is None:
            self._bypass(limit.bypass_reason or 'population_decision_failure')
            return None
        return min(outcome.deep_deadline, outcome.origin_at + limit.seconds)

    @contextmanager
    def _published_deadline(self) -> Generator[DeadlineRecord]:
        """Publish one budget and report its recorded expiry even under cancellation."""
        record = DeadlineRecord(self._deadline())
        with publish_deadline(record):
            try:
                yield record
            finally:
                if record.expired:
                    self._increment('backend_failure', {'reason': 'finalization_timeout'})

    async def _backend_call(
        self, backend: AsyncCacheBackend, action: Callable[[], Awaitable[_T]]
    ) -> _T:
        """Await one backend operation under its original cooperative deadline."""
        with self._published_deadline() as record:
            try:
                return await self._await_backend(backend, action, record)
            except TimeoutError as error:
                raise BackendUnavailable('async backend operation deadline expired') from error

    def _cleanup_error(
        self, error: Exception | None, *, expired: bool = False
    ) -> Exception | None:
        """Record cleanup failure and translate cooperative deadline expiration."""
        if error is None:
            return None
        if not expired:
            labels = {'reason': 'backend_failure'}
            if isinstance(error, BackendUnavailable) and error.failure_class is not None:
                labels['failure_class'] = error.failure_class
            self._increment('backend_failure', labels)
        if isinstance(error, TimeoutError):
            return BackendUnavailable('async backend operation deadline expired')
        return error

    async def _close_backend(self, backend: AsyncCacheBackend) -> CleanupOutcome[None]:
        """Join one backend under its deadline, retaining any unstarted obligation."""
        with self._published_deadline() as record:
            outcome = await enter_cleanup(
                backend.close, deadline=None if self._owns_deadline(backend) else record.deadline
            )
            return replace(
                outcome, error=self._cleanup_error(outcome.error, expired=record.expired)
            )

    async def _close_all(self, state: _CloseState) -> CleanupOutcome[None]:
        """Drain active owners, then join every backend cleanup and finish closed."""
        await self._admission.wait_idle()
        first_error: Exception | None = None
        cancellation: CancelledError | None = None
        while state.pending:
            outcome = await self._close_backend(state.pending.pop(0))
            cancellation = cancellation or outcome.cancellation
            error = outcome.error
            if outcome.cancellation is not None and error is None:
                error = self._cleanup_error(BackendUnavailable('cleanup task cancelled'))
            if first_error is None:
                first_error = error
        self._memo = None
        self._flights.clear()
        self._admission.finish_close()
        return CleanupOutcome(error=first_error, cancellation=cancellation)

    @asynccontextmanager
    async def _held_lease(
        self, backend: AsyncCacheBackend, lease: FillLease
    ) -> AsyncGenerator[_LeaseHold]:
        """Own release through repeated cancellation without replacing its first cause."""
        hold = _LeaseHold()
        original: CancelledError | None = None
        try:
            yield hold
        except CancelledError as error:
            original = error
            raise
        finally:
            if not hold.committed:
                with self._published_deadline() as record:
                    outcome = await joined_cleanup(
                        lambda: backend.release(lease),
                        deadline=None if self._owns_deadline(backend) else record.deadline,
                    )
                    self._cleanup_error(outcome.error, expired=record.expired)
                if original is None and outcome.cancellation is not None:
                    raise outcome.cancellation

    @contextmanager
    def _correlation_scope(self, digest: str | None) -> Generator[None]:
        """Restore this instance's immutable correlation snapshot on scope exit."""
        token = self._correlations.set((*self._correlations.get(), digest))
        try:
            yield
        finally:
            self._correlations.reset(token)

    async def _reserve(self, resolved: _Resolved) -> FillLease | None:
        """Reserve a deepest-layer lease through the cache failure boundary."""
        try:
            return await self._backend_call(
                resolved.backend,
                lambda: resolved.backend.reserve(
                    resolved.key, self.settings.limits.fill_lease_seconds
                ),
            )
        except BACKEND_ERRORS as error:
            raise _CachePathError('backend_failure') from error

    def _retain(self, obligation: _Obligation) -> None:
        """Settle unacknowledged work synchronously before another layer can await."""
        current = self._fences[obligation.layer]
        self._fences[obligation.layer] = _FenceState(
            merge_changes(current.retained, obligation.changes), current.revision + 1
        )
        obligation.settled = True

    async def _fence(
        self, obligations: list[_Obligation], record: DeadlineRecord, *, layer_timers: bool
    ) -> None:
        """Attempt L2 before L1, recording failed layers before the next await."""
        for obligation in obligations:
            snapshot = self._fences[obligation.layer]
            obligation.changes = merge_changes(snapshot.retained, obligation.changes)
            try:
                if layer_timers:
                    await self._await_backend(
                        obligation.backend,
                        partial(obligation.backend.change, obligation.changes),
                        record,
                    )
                else:
                    await obligation.backend.change(obligation.changes)
            except BACKEND_ERRORS:
                self._retain(obligation)
                self._bypass('backend_failure')
                if record.expired:
                    raise
            else:
                if record.expired:
                    return
                if self._fences[obligation.layer].revision == snapshot.revision:
                    self._fences[obligation.layer] = _FenceState((), snapshot.revision + 1)
                obligation.acknowledged = True
                obligation.settled = True

    async def _settle_close(
        self, task: Task[CleanupOutcome[None]], state: _CloseState
    ) -> CancelledError | None:
        """Claim interrupted cleanup inline and publish the same outcome to waiters."""
        if state.completion.done() or state.settling:
            return None
        state.settling = True
        cancellation: CancelledError | None = None
        if task.cancelled():
            # Cancellation before entry or during admission leaves every backend
            # pending. This waiter owns settlement without creating another task.
            while True:
                try:
                    await self._admission.wait_idle()
                    break
                except CancelledError as error:
                    cancellation = cancellation or error
            outcome = await self._close_all(state)
            cancellation = cancellation or outcome.cancellation
        else:
            outcome = task.result()
        failure = outcome.error
        if task.cancelled():
            interrupted = self._cleanup_error(BackendUnavailable('cleanup task cancelled'))
            failure = failure or interrupted
        state.completion.set_result(CleanupOutcome(error=failure))
        return cancellation

    def _unchanged(self, resolved: _Resolved) -> bool:
        """Require the empty retained state observed before backend I/O."""
        current = self._fences[resolved.layer]
        return not current.retained and not resolved.fence.retained and current == resolved.fence

    async def _lookup(self, resolved: _Resolved) -> _Outcome | None:
        """Decode one isolated hit while preserving authoritative age and fences."""
        try:
            hit = await self._backend_call(
                resolved.backend, lambda: resolved.backend.lookup(resolved.key)
            )
        except BACKEND_ERRORS as error:
            raise _CachePathError('backend_failure') from error
        observed_at = monotonic()
        if not self._unchanged(resolved):
            raise _CachePathError('unacknowledged_fence')
        if hit is None:
            return None
        try:
            decoded = decode_hit(hit, resolved.age)
        except StaleHitError:
            return None
        except CodecError as error:
            raise _CachePathError('codec_failure') from error
        decided_at = monotonic()
        origin_at = observed_at - decoded.origin_elapsed
        deadline = min(observed_at + hit.remaining_seconds, origin_at + resolved.age)
        if decided_at >= deadline:
            return None
        outcome = _Outcome(
            decoded.value,
            resolved,
            resolved if resolved.layer == 'l2' else None,
            origin_at,
            decided_at,
            deadline,
            deadline,
            decoded.sg_duration,
            stored=True,
            from_cache=True,
        )
        layer = self._layer_name(resolved.layer)
        self._increment('cache_hit', {'layer': layer})
        self._increment('cache_sg_avoided_calls', {'layer': layer})
        self._observe('cache_sg_avoided_seconds', decoded.sg_duration, {'layer': layer})
        return outcome

    async def _resolve(self, request: _ReadInput, layer: _Layer) -> _Resolved:
        """Resolve the requested vector without allowing a concurrent retained fence."""
        backend = self._l1 if layer == 'l1' else self._l2
        if backend is None:
            raise _CachePathError('backend_failure')
        fence = self._fences[layer]
        if fence.retained:
            raise _CachePathError('unacknowledged_fence')
        age = self._age(request, layer)
        try:
            vector = await self._backend_call(backend, lambda: backend.resolve(request.plan.tags))
            key = self._key_builder.qualified_key(request.base_key, vector, 'result')
        except BACKEND_ERRORS as error:
            raise _CachePathError('backend_failure') from error
        except NormalizationError as error:
            raise _CachePathError('normalization_failure') from error
        resolved = _Resolved(backend, layer, key, fence, age)
        if not self._unchanged(resolved):
            raise _CachePathError('unacknowledged_fence')
        return resolved

    async def _memo_value(self, request: _ReadInput, first: _Resolved, parse: Parse) -> object:
        """Reuse or revalidate the single memo at one callback-free decision sample."""
        entry = self._memo
        if entry is None or entry.parse is not parse or entry.outcome.first.key != first.key:
            self._memo = None
            return _MISS
        outcome = entry.outcome
        now = monotonic()
        if request.age is not None and now - outcome.origin_at >= request.age:
            return _MISS
        if now < outcome.first_deadline:
            self._increment('cache_schema_view', {'outcome': 'reuse'})
            return entry.value
        if (
            self._l1 is None
            or self._l2 is None
            or outcome.shared is None
            or now >= outcome.deep_deadline
        ):
            return _MISS
        deep: _Resolved | None = None
        with suppress(_CachePathError):
            deep = await self._resolve(request, 'l2')
        decided_at = monotonic()
        if deep is None or not (
            self._memo is entry
            and self._unchanged(first)
            and self._unchanged(deep)
            and deep.key == outcome.shared.key
            and decided_at < outcome.deep_deadline
            and (request.age is None or decided_at - outcome.origin_at < request.age)
        ):
            return _MISS
        self._memo = replace(
            entry,
            outcome=replace(
                outcome,
                first=first,
                shared=deep,
                first_deadline=min(decided_at + first.age, outcome.deep_deadline),
            ),
        )
        self._increment('cache_schema_view', {'outcome': 'revalidated'})
        return entry.value

    async def _poll_deep(
        self, request: _ReadInput, resolved: _Resolved, wait_deadline: float
    ) -> tuple[_Resolved, FillLease | None, _Outcome | None]:
        """Finish deepest-layer coordination before any takeover or fallback fetch."""
        attempt = 0
        started: float | None = None
        outcome: WaitOutcome = 'cancelled'
        try:
            while True:
                if get_running_loop().time() >= wait_deadline:
                    self._expire_wait(request.waits, 'distributed')
                    outcome = 'timeout'
                    return resolved, None, None
                lease = await self._reserve(resolved)
                if lease is not None:
                    outcome = 'takeover'
                    return resolved, lease, None
                remaining = wait_deadline - get_running_loop().time()
                if remaining <= 0:
                    self._expire_wait(request.waits, 'distributed')
                    outcome = 'timeout'
                    return resolved, None, None
                if started is None:
                    started = self._begin_wait(request.waits, 'distributed')
                await sleep(min(remaining, 0.01 * 2 ** min(attempt, 3)))
                attempt += 1
                resolved = await self._resolve(request, resolved.layer)
                hit = await self._lookup(resolved)
                if hit is not None:
                    outcome = 'completed'
                    return resolved, None, hit
        except _CachePathError:
            outcome = 'failure'
            raise
        finally:
            if started is not None:
                request.waits.end('distributed', get_running_loop().time() - started, outcome)

    def _valid_outcome(self, outcome: _Outcome) -> bool:
        """Recheck every layer used by one acquisition after asynchronous work."""
        return self._unchanged(outcome.first) and (
            outcome.shared is None or self._unchanged(outcome.shared)
        )

    def _checked_outcome(self, outcome: _Outcome) -> _Outcome:
        """Reject a cached source invalidated or expired during promotion or release."""
        if outcome.from_cache:
            now = monotonic()
            if not self._valid_outcome(outcome):
                raise _CachePathError('unacknowledged_fence')
            if now >= outcome.deep_deadline:
                raise _CachePathError('maximum_age_elapsed')
        return outcome

    async def _commit(self, lease: FillLease, outcome: _Outcome, deadline: float) -> bool:
        """Encode and commit one optional value under its exact lease and age."""
        lifetime = deadline - outcome.observed_at
        if lifetime <= 0:
            return False
        try:
            payload = encode_result(
                TimedResult(outcome.sg_duration, outcome.value),
                max(0.0, outcome.observed_at - outcome.origin_at),
                lifetime,
            )
            if len(payload) > self.settings.limits.max_serialized_result_bytes:
                self._bypass('result_too_large_bytes')
                return False
            # Deliver pending cancellation before dispatch, even for an inline backend.
            await sleep(0)
            remaining = deadline - monotonic()
            if remaining <= 0 or not self._valid_outcome(outcome):
                return False
            return await self._backend_call(
                outcome.first.backend,
                lambda: outcome.first.backend.commit(lease, payload, remaining),
            )
        except CodecBoundError as error:
            self._bypass(bound_reason(error))
        except CodecError:
            self._bypass('codec_failure')
        except BACKEND_ERRORS:
            self._bypass('backend_failure')
        return False

    async def _promote(self, first: _Resolved, outcome: _Outcome) -> _Outcome:
        """Promote under the original L1 vector without extending L2 origin age."""
        deadline = min(outcome.deep_deadline, outcome.observed_at + first.age)
        promoted = replace(outcome, first=first, shared=outcome.first, first_deadline=deadline)
        # A value already as old as the L1 age would be stale on its first L1 lookup.
        if (
            not outcome.stored
            or outcome.observed_at - outcome.origin_at >= first.age
            or not self._valid_outcome(promoted)
        ):
            return promoted
        try:
            lease = await self._backend_call(
                first.backend,
                lambda: first.backend.reserve(first.key, self.settings.limits.fill_lease_seconds),
            )
        except BACKEND_ERRORS:
            self._bypass('backend_failure')
            return promoted
        if lease is None:
            self._increment('cache_lease_rejection', {'layer': 'hierarchical_l1'})
            return promoted
        async with self._held_lease(first.backend, lease) as hold:
            if self._valid_outcome(promoted):
                hold.committed = await self._commit(lease, promoted, deadline)
        return promoted

    async def _reserved(
        self, request: _ReadInput, resolved: _Resolved, lease: FillLease
    ) -> _Outcome:
        """Recheck a granted lease and join its release unless a commit consumed it."""
        async with self._held_lease(resolved.backend, lease) as hold:
            rechecked = await self._resolve(request, resolved.layer)
            hit = await self._lookup(rechecked)
            if hit is not None:
                return hit
            outcome = await self._fetch(request, resolved)
            if rechecked.key != lease.key:
                return outcome
            deadline = self._population_deadline(request, outcome)
            if deadline is None:
                return outcome
            hold.committed = await self._commit(lease, outcome, deadline)
            if hold.committed:
                return replace(
                    outcome, first_deadline=deadline, deep_deadline=deadline, stored=True
                )
            return outcome

    async def _deep(
        self, request: _ReadInput, resolved: _Resolved, wait_deadline: float
    ) -> _Outcome:
        """Compete for a deepest-layer lease within the caller's original wait budget."""
        resolved, lease, hit = await self._poll_deep(request, resolved, wait_deadline)
        if hit is not None:
            return hit
        if lease is not None:
            return await self._reserved(request, resolved, lease)
        return await self._fetch(request, resolved)

    async def _wait_flight(self, flight: Future[None], deadline: float, waits: WaitRecord) -> bool:
        """Wait for notification without cancelling the leader or renewing the budget."""
        started = self._begin_wait(waits, 'local')
        outcome: WaitOutcome = 'cancelled'
        try:
            try:
                async with timeout_at(deadline):
                    await shield(flight)
            except TimeoutError:
                outcome = 'timeout'
                self._expire_wait(waits, 'local')
                return False
            outcome = 'completed'
            return True
        finally:
            waits.end('local', get_running_loop().time() - started, outcome)

    async def _read_cacheable(self, request: _ReadInput, wait_deadline: float) -> _Outcome:
        """Read the configured hierarchy with notification-only leader re-election."""
        while True:
            first_layer: _Layer = 'l1' if self._l1 is not None else 'l2'
            try:
                first = await self._resolve(request, first_layer)
                hit = await self._lookup(first)
            except _CachePathError as error:
                if (
                    first_layer != 'l1'
                    or self._l2 is None
                    or error.reason == 'normalization_failure'
                ):
                    raise
                self._bypass(error.reason)
                first = await self._resolve(request, 'l2')
                hit = await self._lookup(first)
            if hit is not None:
                return hit
            flight = self._flights.get(first.key)
            if flight is not None:
                if (
                    await self._wait_flight(flight, wait_deadline, request.waits)
                    and get_running_loop().time() < wait_deadline
                ):
                    continue
                return await self._fetch(request, first)
            flight = get_running_loop().create_future()
            self._flights[first.key] = flight
            try:
                if first.layer == 'l1' and self._l2 is not None:
                    deep = await self._resolve(request, 'l2')
                    outcome = await self._lookup(deep)
                    if outcome is None:
                        outcome = await self._deep(request, deep, wait_deadline)
                    return self._checked_outcome(await self._promote(first, outcome))
                return self._checked_outcome(await self._deep(request, first, wait_deadline))
            finally:
                self._finish_flight(first.key, flight)

    async def _acquire_parsed(self, request: _ReadInput, parse: Parse, deadline: float) -> object:
        """Keep acquisition, parsing, malformed refetch, and memo publication in one flight."""
        outcome = await self._read_cacheable(request, deadline)
        result = 'rebuild'
        try:
            parsed = parse(outcome.value)
        except (TypeError, ValueError):
            self._increment('cache_schema_view', {'outcome': 'malformed'})
            fresh = await self._fetch(request, outcome.first)
            parsed = parse(fresh.value)
            outcome = replace(
                fresh,
                shared=outcome.shared,
                deep_deadline=fresh.origin_at
                + (outcome.shared.age if outcome.shared is not None else outcome.first.age),
            )
            result = 'refetch'
        if self._valid_outcome(outcome):
            self._memo = _MemoEntry(outcome, parse, parsed)
        self._increment('cache_schema_view', {'outcome': result})
        return parsed

    async def _read_parsed(self, request: _ReadInput, parse: Parse) -> object:
        """Coordinate parsed acquisitions, including results no backend can store."""
        deadline = get_running_loop().time() + self.settings.limits.lease_wait_seconds
        while True:
            try:
                first = await self._resolve(request, 'l1' if self._l1 is not None else 'l2')
            except _CachePathError as error:
                # Without an L1 key there is no memo; the hierarchy read still reaches L2.
                if self._l1 is None or self._l2 is None or error.reason == 'normalization_failure':
                    raise
                return await self._acquire_parsed(request, parse, deadline)
            reused = await self._memo_value(request, first, parse)
            if reused is not _MISS:
                return reused
            key = f'memo:{first.key}'
            flight = self._flights.get(key)
            if flight is not None:
                if (
                    await self._wait_flight(flight, deadline, request.waits)
                    and get_running_loop().time() < deadline
                ):
                    continue
                return parse((await self._fetch(request, first)).value)
            flight = get_running_loop().create_future()
            self._flights[key] = flight
            try:
                return await self._acquire_parsed(request, parse, deadline)
            finally:
                self._finish_flight(key, flight)

    @contextmanager
    def admit(self) -> Generator[OperationToken]:
        """Admit one complete wrapper operation, including planning and cleanup.

        Yields:
            An explicit cache-specific token, valid only until context exit.

        Raises:
            RuntimeError: Cache is closing, closed, or bound to another loop.
        """
        with self._admission.operation() as token:
            yield token

    async def __aenter__(self) -> Self:
        """Bind and validate this open cache before entering its owning context.

        Raises:
            RuntimeError: The cache is closing, closed, or bound to another
                loop.
        """
        with self.admit():
            return self

    @staticmethod
    def build_metrics(
        settings: CacheSettings, environ: Mapping[str, object], sink: MetricsSink
    ) -> MetricsSink:
        """Build a validating metrics sink with optional keyed correlation.

        Args:
            settings: Loaded settings naming an optional correlation secret.
            environ: Environment mapping consulted only during construction.
            sink: Destination metrics implementation.

        Returns:
            A validating sink with optional keyed exact-call correlation.

        Raises:
            ValueError: A configured secret is absent, non-UTF-8, or shorter
                than 32 encoded bytes.
        """
        return QueryCache.build_metrics(settings, environ, sink)

    @property
    def bypass_active(self) -> bool:
        """Report whether this task's lexical scopes require bypass."""
        return policy_bypass_active(self._policies.get())

    async def close(self) -> None:
        """Join all admitted work and close every backend exactly once.

        Caller cancellation is deferred until cleanup completes, then
        propagated.
        There is no imposed deadline for active operations. Calls after
        terminal close return quietly on the bound loop.

        Raises:
            RuntimeError: Called by this cache's active owner or on another
                loop.
            Exception: First ordinary backend failure after every close
                attempt.
        """
        if self._admission.begin_close():
            self._close_state = _CloseState(
                [backend for backend in (self._l1, self._l2) if backend is not None],
                get_running_loop().create_future(),
            )
            self._close_task = create_task(self._close_all(self._close_state))
        elif self._admission.closed:
            return
        task, state = self._close_task, self._close_state
        if task is None or state is None:
            raise RuntimeError('close task was not published')
        await join_shared_close(task, state.completion, lambda: self._settle_close(task, state))

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close resources with cancellation, body error, then cleanup precedence.

        Args:
            exc_type: Active body exception type, when present.
            exc: Active body exception whose precedence must be retained.
            traceback: Traceback supplied by the async context manager
                protocol.
        """
        try:
            if exc is None:
                await self.close()
            else:
                with suppress(Exception):
                    await self.close()
        except CancelledError:
            if isinstance(exc, CancelledError):
                raise exc from None
            raise

    async def invalidate_admitted(
        self, changes: tuple[GenerationChange, ...], *, token: OperationToken
    ) -> bool:
        """Attempt and settle an ordered fence within an existing operation.

        Args:
            changes: Ordered conservative generation changes to apply.
            token: Live capability from this cache's `admit` context.

        Returns:
            Whether every configured layer acknowledged this attempt. Failed
            and unvisited layers retain work for replay by the next fence.

        Raises:
            RuntimeError: The token is foreign or expired, or the cache belongs
                to another loop.
        """
        self._admission.validate(token)
        pairs: tuple[tuple[_Layer, AsyncCacheBackend | None], ...] = (
            ('l2', self._l2),
            ('l1', self._l1),
        )
        obligations = [
            _Obligation(layer, backend, changes) for layer, backend in pairs if backend is not None
        ]
        with self._published_deadline() as record:
            layer_timers = any(self._owns_deadline(item.backend) for item in obligations)
            outcome = await joined_cleanup(
                lambda: self._fence(obligations, record, layer_timers=layer_timers),
                deadline=None if layer_timers else record.deadline,
            )
            # The operation owner performs this even if the inner task never ran.
            for obligation in obligations:
                if not obligation.settled:
                    self._retain(obligation)
            self._cleanup_error(outcome.error, expired=record.expired)
        if outcome.cancellation is not None:
            raise outcome.cancellation
        return outcome.error is None and all(item.acknowledged for item in obligations)

    async def invalidate(self, changes: tuple[GenerationChange, ...]) -> bool:
        """Apply L2 then L1 fences and retain every unacknowledged layer.

        Args:
            changes: Ordered conservative generation changes to apply.

        Returns:
            Whether every configured layer acknowledged this attempt.

        Raises:
            RuntimeError: The cache is closing, closed, or bound to another
                loop.
        """
        with self.admit() as token:
            return await self.invalidate_admitted(changes, token=token)

    @contextmanager
    def policy(
        self, mode: PolicyMode | None = None, maximum_age: float | None = None
    ) -> Generator[None]:
        """Apply a lexical policy snapshot inherited independently by child tasks.

        Args:
            mode: Optional conservative policy downgrade.
            maximum_age: Optional positive age in seconds, combined by minimum
                with other scoped and per-call ages. The resulting explicit age
                replaces configured rules and defaults; global safety,
                response, remaining-L2, and hierarchical-L1 caps still apply.

        Yields:
            Control while this cache instance's policy scope is active.

        Raises:
            TypeError: A supplied maximum age is not a finite non-boolean
                number.
            ValueError: The mode is unsupported or the maximum age is not
                positive.
        """
        frame = policy_frame(mode, maximum_age)
        token = self._policies.set((*self._policies.get(), frame))
        try:
            yield
        finally:
            self._policies.reset(token)

    @contextmanager
    def bypass(self) -> Generator[None]:
        """Bypass cache reads and population within this inherited lexical scope."""
        with self.policy(mode='bypass'):
            yield

    async def read_admitted(  # noqa: PLR0913 - preserve read inputs plus explicit admission token
        self,
        call: ExactCall,
        plan: DependencyPlan,
        fetch: _Fetch,
        client_options: Mapping[str, object],
        per_call_age: float | None = None,
        *,
        token: OperationToken,
        _population_decision: PopulationCallback | None = None,
    ) -> object:
        """Read within the explicit lifetime of an already admitted operation.

        Args:
            call: Exact normalized SG call.
            plan: Conservative pre-fetch dependency plan.
            fetch: Caller-owned native async SG operation.
            client_options: Complete response-affecting client options.
            per_call_age: Optional positive maximum age in seconds, combined by
                minimum with active scopes. This explicit age replaces
                configured rules before safety caps are applied.
            token: Live capability from this cache's `admit` context.
            _population_decision: Optional private post-fetch age cap or
                decline.

        Returns:
            An isolated cache hit or the direct SG result.

        Raises:
            RuntimeError: The token is foreign or expired, or the cache belongs
                to another loop.
            TypeError: A non-bypassed call supplies a nonfinite, boolean, or
                nonnumeric maximum age.
            ValueError: A non-bypassed call supplies a nonpositive maximum age.
            Exception: The caller-owned SG operation fails.
            CancelledError: The caller cancels the read.
        """
        self._admission.validate(token)
        frames = self._policies.get()
        effective = apply_policy_frames(plan, frames)
        with self._correlation_scope(None):
            self._increment(
                'cache_query_classification',
                {'classification': effective.classification, 'reason': effective.reason},
            )
            for step in effective.broadening_steps:
                self._increment('cache_dependency_broadening', {'reason': step})
            if effective.classification == 'bypass':
                self._bypass(effective.reason)
                return await self._direct_fetch(fetch)
            age = minimum_policy_age(frames, per_call_age)
            try:
                base = self._key_builder.base_key(
                    self.settings.site_namespace, self.identity, client_options, call
                )
            except NormalizationError:
                self._bypass('normalization_failure')
                return await self._direct_fetch(fetch)
            digest = None
            if isinstance(self._metrics, SafeMetrics):
                with suppress(Exception):
                    digest = self._metrics.correlation_digest(self._key_builder, base)
            request = _ReadInput(base, call, effective, fetch, age, _population_decision)
            with self._correlation_scope(digest):
                try:
                    deadline = get_running_loop().time() + self.settings.limits.lease_wait_seconds
                    return (await self._read_cacheable(request, deadline)).value
                except _CachePathError as error:
                    self._bypass(error.reason)
                    return await self._direct_fetch(fetch)
                finally:
                    self._emit_waits(request.waits)

    async def read(
        self,
        call: ExactCall,
        plan: DependencyPlan,
        fetch: _Fetch,
        client_options: Mapping[str, object],
        per_call_age: float | None = None,
        *,
        _population_decision: PopulationCallback | None = None,
    ) -> object:
        """Read one exact call through native asynchronous cache operations.

        Args:
            call: Exact normalized SG call.
            plan: Conservative pre-fetch dependency plan.
            fetch: Caller-owned native async SG operation.
            client_options: Complete response-affecting client options.
            per_call_age: Optional positive maximum age in seconds, combined by
                minimum with active scopes. This explicit age replaces
                configured rules before safety caps are applied.
            _population_decision: Optional private post-fetch age cap or
                decline.

        Returns:
            An isolated cache hit or the direct SG result.

        Raises:
            RuntimeError: The cache is closing, closed, or bound to another
                loop.
            TypeError: A non-bypassed call supplies a nonfinite, boolean, or
                nonnumeric maximum age.
            ValueError: A non-bypassed call supplies a nonpositive maximum age.
            Exception: The caller-owned SG operation fails.
            CancelledError: The caller cancels the read.
        """
        with self.admit() as token:
            return await self.read_admitted(
                call,
                plan,
                fetch,
                client_options,
                per_call_age,
                token=token,
                _population_decision=_population_decision,
            )

    async def read_parsed_admitted(  # noqa: PLR0913 - preserve read inputs plus explicit admission token
        self,
        call: ExactCall,
        plan: DependencyPlan,
        fetch: _Fetch,
        client_options: Mapping[str, object],
        parse: Parse,
        *,
        token: OperationToken,
    ) -> object:
        """Acquire one immutable parsed view within an existing admission lifetime.

        When rebuilding the parsed memo, a first parse that raises `TypeError`
        or `ValueError` triggers one direct refetch and another parse. Fetch
        failures and a failing second parse propagate. Direct fallback parsing
        is not retried.
        Memo hits may return the same immutable object.

        Args:
            call: Exact normalized call for the reduced view.
            plan: Complete conservative schema dependency plan.
            fetch: Native async acquisition returning a storable value.
            client_options: Complete response-affecting client options.
            parse: Pure conversion to an immutable view; identity controls
                reuse.
            token: Live capability from this cache's `admit` context.

        Returns:
            The parsed view for the currently resolved key and valid age.

        Raises:
            RuntimeError: The token is foreign or expired, or the cache belongs
                to another loop.
        """
        self._admission.validate(token)
        frames = self._policies.get()
        effective = apply_policy_frames(plan, frames)
        with self._correlation_scope(None):
            if effective.classification == 'bypass':
                self._bypass(effective.reason)
                return parse(await self._direct_fetch(fetch))
            try:
                base = self._key_builder.base_key(
                    self.settings.site_namespace, self.identity, client_options, call
                )
                request = _ReadInput(
                    base, call, effective, fetch, minimum_policy_age(frames, None)
                )
                try:
                    return await self._read_parsed(request, parse)
                finally:
                    self._emit_waits(request.waits)
            except NormalizationError:
                self._bypass('normalization_failure')
            except _CachePathError as error:
                self._bypass(error.reason)
            return parse(await self._direct_fetch(fetch))

    async def read_parsed(
        self,
        call: ExactCall,
        plan: DependencyPlan,
        fetch: _Fetch,
        client_options: Mapping[str, object],
        parse: Parse,
    ) -> object:
        """Read an exact call through the cache's single immutable parsed memo.

        When rebuilding the parsed memo, a first parse that raises `TypeError`
        or `ValueError` triggers one direct refetch and another parse. Fetch
        failures and a failing second parse propagate. Direct fallback parsing
        is not retried.
        Memo hits may return the same immutable object.

        Args:
            call: Exact normalized call for the reduced view.
            plan: Complete conservative schema dependency plan.
            fetch: Native async acquisition returning a storable value.
            client_options: Complete response-affecting client options.
            parse: Pure conversion to an immutable view; identity controls
                reuse.

        Returns:
            The parsed view for the currently resolved key and valid age.

        Raises:
            RuntimeError: The cache is closing, closed, or bound to another
                loop.
        """
        with self.admit() as token:
            return await self.read_parsed_admitted(
                call, plan, fetch, client_options, parse, token=token
            )


def client_binding(cache: AsyncQueryCache) -> tuple[CacheSettings, Identity]:
    """Return the supported package-internal settings and exact identity binding.

    Args:
        cache: Async cache wrapped by the package's SG client adapter.

    Returns:
        The immutable settings and identity retained by the cache.
    """
    return cache.settings, cache.identity
