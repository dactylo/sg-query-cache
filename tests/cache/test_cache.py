from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass
from threading import Barrier, Event
from typing import Literal, cast

import pytest

import sg_query_cache._cache as cache_module
import sg_query_cache._cache_common as cache_common_module
import sg_query_cache._local as local_module
from sg_query_cache._backend import BackendUnavailable, CacheBackend
from sg_query_cache._cache import QueryCache
from sg_query_cache._codec import _MAX_DEPTH, _MAX_ITEMS, MAX_BYTES, CodecError, encode_value
from sg_query_cache._config import CacheSettings, L1Settings, L2Settings, LimitSettings
from sg_query_cache._keys import KeyBuilder
from sg_query_cache._local import LocalBackend
from sg_query_cache._model import (
    CacheHit,
    DependencyPlan,
    DependencyTag,
    ExactCall,
    FillLease,
    GenerationChange,
    Identity,
)
from tests._fakes import (
    ResolveFailureBackend,
    _BlockingBackend,
    _DelegatingBackend,
    _FakeClock,
    _FencedBackend,
    _RecordingMetrics,
)

_CLIENT_OPTIONS = {'localized': False, 'convert_datetimes_to_utc': True}
_SITE = 'production'


class _ObservedBackend(_DelegatingBackend):
    def __init__(
        self,
        backend: CacheBackend,
        *,
        gate_reserve_at: int | None = None,
        fail_lookup_at: int | None = None,
        after_first_reserve: Callable[[], None] | None = None,
    ) -> None:
        self._after_first_reserve = after_first_reserve
        self._allow_reserve = Event()
        super().__init__(backend)
        self._commit_calls = 0
        self._fail_lookup_at = fail_lookup_at
        self._gate_reserve_at = gate_reserve_at
        self._lookup_calls = 0
        self._release_calls = 0
        self._reserve_calls = 0
        self._reserve_entered = Event()

    def commit(self, lease: FillLease, payload: bytes, remaining_seconds: float) -> bool:
        self._commit_calls += 1
        return self._backend.commit(lease, payload, remaining_seconds)

    def lookup(self, key: str) -> CacheHit | None:
        self._lookup_calls += 1
        if self._lookup_calls == self._fail_lookup_at:
            raise BackendUnavailable('selected lookup failure')
        return self._backend.lookup(key)

    def release(self, lease: FillLease) -> bool:
        self._release_calls += 1
        return self._backend.release(lease)

    def reserve(self, key: str, lease_seconds: float) -> FillLease | None:
        self._reserve_calls += 1
        if self._reserve_calls == self._gate_reserve_at:
            self._reserve_entered.set()
            assert self._allow_reserve.wait(timeout=2)
        lease = self._backend.reserve(key, lease_seconds)
        if self._reserve_calls == 1 and self._after_first_reserve is not None:
            self._after_first_reserve()
        return lease


class _LegacyHitBackend(_ObservedBackend):
    def lookup(self, key: str) -> CacheHit | None:
        hit = super().lookup(key)
        if hit is None:
            return None
        return CacheHit(hit.payload, hit.remaining_seconds)


@dataclass(frozen=True, slots=True)
class _PopulationDecision:
    remaining_seconds: float | None
    reason: str | None = None


def _cache(
    settings: CacheSettings | None = None,
    *,
    backend: CacheBackend | None = None,
    clock: _FakeClock | None = None,
) -> tuple[QueryCache, _FakeClock, _RecordingMetrics]:
    selected_settings = settings or CacheSettings()
    selected_clock = clock or _FakeClock()
    metrics = _RecordingMetrics()
    selected_backend = backend or LocalBackend(selected_settings, selected_clock, metrics)
    return (
        QueryCache(
            selected_settings,
            Identity('script', 'studio', 'render'),
            KeyBuilder(),
            l1=selected_backend,
            metrics=metrics,
        ),
        selected_clock,
        metrics,
    )


def _call() -> ExactCall:
    return ExactCall('find', (('entity_type', 'Asset'), ('filters', ())))


def _parse(value: object) -> object:
    return ('parsed', value)


def _plan(classification: str = 'event_aware') -> DependencyPlan:
    tags = (
        DependencyTag.site(_SITE),
        DependencyTag.authorization(_SITE),
        DependencyTag.type_membership(_SITE, 'Asset'),
    )
    if classification == 'bypass':
        tags = ()
    return DependencyPlan(classification, tags, classification)


def _read(
    cache: QueryCache,
    fetch: Callable[[], object],
    *,
    plan: DependencyPlan | None = None,
    per_call_age: float | None = None,
    population_decision: Callable[[object], _PopulationDecision] | None = None,
) -> object:
    return cache.read(
        _call(),
        plan or _plan(),
        fetch,
        _CLIENT_OPTIONS,
        per_call_age=per_call_age,
        _population_decision=population_decision,
    )


def _single_layer_cache(
    settings: CacheSettings,
    backend: CacheBackend,
    metrics: _RecordingMetrics,
) -> QueryCache:
    identity = Identity('script', 'studio', 'render')
    if settings.l1.enabled:
        return QueryCache(settings, identity, KeyBuilder(), l1=backend, metrics=metrics)
    return QueryCache(settings, identity, KeyBuilder(), l2=backend, metrics=metrics)


def _single_layer_settings(topology: str) -> CacheSettings:
    if topology == 'local':
        return CacheSettings()
    return CacheSettings(l1=L1Settings(enabled=False), l2=L2Settings(enabled=True))


def _store_payload(
    backend: CacheBackend,
    call: ExactCall,
    plan: DependencyPlan,
    payload: bytes,
    *,
    age: float = 30,
) -> str:
    base_key = KeyBuilder.base_key(
        _SITE,
        Identity('script', 'studio', 'render'),
        _CLIENT_OPTIONS,
        call,
    )
    vector = backend.resolve(plan.tags)
    key = KeyBuilder.qualified_key(base_key, vector, 'result')
    lease = backend.reserve(key, 30)
    assert lease is not None
    assert backend.commit(lease, payload, age)
    return key


def _shared_caches(
    *,
    lease_wait_seconds: float = 0.5,
) -> tuple[QueryCache, QueryCache, LocalBackend, _FakeClock]:
    settings = CacheSettings(
        l1=L1Settings(enabled=False),
        l2=L2Settings(enabled=True),
        limits=LimitSettings(lease_wait_seconds=lease_wait_seconds),
    )
    clock = _FakeClock()
    metrics = _RecordingMetrics()
    backend = LocalBackend(settings, clock, metrics)
    identity = Identity('script', 'studio', 'render')
    first = QueryCache(settings, identity, KeyBuilder(), l2=backend, metrics=metrics)
    second = QueryCache(settings, identity, KeyBuilder(), l2=backend, metrics=metrics)
    return first, second, backend, clock


def test_backend_failure_bypasses_safely_without_catching_fetch_exception() -> None:

    class _FailingBackend:
        def resolve(self, tags: tuple[DependencyTag, ...]) -> object:
            _ = tags
            raise BackendUnavailable('backend-private-detail')

    cache, _clock, metrics = _cache(backend=cast('CacheBackend', _FailingBackend()))
    expected = LookupError('sg-original')

    def fetch() -> object:
        raise expected

    with pytest.raises(LookupError) as captured:
        _read(cache, fetch)

    assert captured.value is expected
    assert all('backend-private-detail' not in str(event) for event in metrics._increments)


@pytest.mark.parametrize(
    ('bound', 'reason'),
    [
        ('bytes', 'result_too_large_bytes'),
        ('depth', 'result_too_deep'),
        ('items', 'result_too_large_items'),
    ],
)
def test_bound_failures_report_their_own_bypass_reason(bound: str, reason: str) -> None:
    cache, _clock, metrics = _cache()
    value: object
    if bound == 'bytes':
        value = 'x' * MAX_BYTES
    elif bound == 'depth':
        value = []
        cursor = cast('list[object]', value)
        for _ in range(_MAX_DEPTH):
            nested: list[object] = []
            cursor.append(nested)
            cursor = nested
    else:
        value = list(range(_MAX_ITEMS))

    assert _read(cache, lambda: value) is value

    assert ('cache_bypass', {'reason': reason}) in metrics._increments
    assert ('cache_bypass', {'reason': 'codec_failure'}) not in metrics._increments


def test_bypass_active_reports_the_lexical_stack() -> None:
    cache, _, _ = _cache()

    assert cache.bypass_active is False
    with cache.policy(mode='ttl_only'):
        assert cache.bypass_active is False
        with cache.bypass():
            assert cache.bypass_active is True
    assert cache.bypass_active is False


@pytest.mark.parametrize('scope', ['explicit', 'context'])
def test_bypass_neither_reads_nor_populates(scope: str) -> None:
    cache, _clock, _metrics = _cache()
    calls = 0

    def fetch() -> object:
        nonlocal calls
        calls += 1
        return [{'call': calls}]

    context = cache.bypass() if scope == 'context' else nullcontext()
    plan = _plan('bypass') if scope == 'explicit' else _plan()
    with context:
        assert _read(cache, fetch, plan=plan) == [{'call': 1}]
        assert _read(cache, fetch, plan=plan) == [{'call': 2}]
    assert _read(cache, fetch) == [{'call': 3}]
    assert _read(cache, fetch) == [{'call': 3}]
    assert calls == 3


def test_cached_none_is_a_hit_instead_of_an_internal_miss() -> None:
    cache, _clock, _metrics = _cache()
    call = ExactCall('find_one', (('entity_type', 'Asset'), ('filters', ())))
    calls = 0

    def fetch() -> object:
        nonlocal calls
        calls += 1
        return None

    assert cache.read(call, _plan(), fetch, _CLIENT_OPTIONS) is None
    assert cache.read(call, _plan(), fetch, _CLIENT_OPTIONS) is None
    assert calls == 1


def test_codec_failure_on_a_hit_falls_through_to_fetch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache, _clock, _metrics = _cache()
    calls = 0

    def fetch() -> object:
        nonlocal calls
        calls += 1
        return [{'id': 7}]

    assert _read(cache, fetch) == [{'id': 7}]
    original_decode = local_module.decode_value

    def fail_once(payload: bytes) -> object:
        monkeypatch.setattr(local_module, 'decode_value', original_decode)
        _ = payload
        raise CodecError('payload-private-detail')

    monkeypatch.setattr(local_module, 'decode_value', fail_once)
    assert _read(cache, fetch) == [{'id': 7}]
    assert calls == 2


def test_complete_local_hit_decodes_once(monkeypatch: pytest.MonkeyPatch) -> None:
    cache, _clock, _metrics = _cache()
    assert _read(cache, lambda: {'items': [1]}) == {'items': [1]}
    original_backend_decode = local_module.decode_value
    original_core_decode = cache_common_module.decode_value
    decode_count = 0

    def count_backend_decode(payload: bytes) -> object:
        nonlocal decode_count
        decode_count += 1
        return original_backend_decode(payload)

    def count_core_decode(payload: bytes) -> object:
        nonlocal decode_count
        decode_count += 1
        return original_core_decode(payload)

    monkeypatch.setattr(local_module, 'decode_value', count_backend_decode)
    monkeypatch.setattr(cache_common_module, 'decode_value', count_core_decode)

    value = cast('dict[str, list[int]]', _read(cache, lambda: {'items': [2]}))

    assert value == {'items': [1]}
    assert decode_count == 1


def test_concurrent_fence_outcomes_cannot_drop_a_retained_change() -> None:
    settings = CacheSettings()
    clock = _FakeClock()
    entered = Event()
    proceed = Event()

    class _InterleavedBackend(_ObservedBackend):
        def __init__(self, backend: CacheBackend) -> None:
            super().__init__(backend)
            self._attempts: list[tuple[GenerationChange, ...]] = []

        def change(self, changes: tuple[GenerationChange, ...]) -> None:
            self._attempts.append(changes)
            if len(self._attempts) == 1:
                entered.set()
                assert proceed.wait(timeout=2)
                self._backend.change(changes)
                return
            if len(self._attempts) == 2:
                raise BackendUnavailable('fence failed')
            self._backend.change(changes)

    backend = _InterleavedBackend(LocalBackend(settings, clock, _RecordingMetrics()))
    cache, _clock, _metrics = _cache(settings, backend=backend, clock=clock)
    asset = GenerationChange(DependencyTag.type_membership(_SITE, 'Asset'))
    shot = GenerationChange(DependencyTag.type_membership(_SITE, 'Shot'))
    calls = 0

    def fetch() -> object:
        nonlocal calls
        calls += 1
        return [{'call': calls}]

    assert _read(cache, fetch) == [{'call': 1}]
    with ThreadPoolExecutor(max_workers=1) as pool:
        stale_snapshot = pool.submit(cache.invalidate, (shot,))
        assert entered.wait(timeout=2)
        assert cache.invalidate((asset,)) is False
        proceed.set()
        assert stale_snapshot.result(timeout=2) is True

    assert _read(cache, fetch) == [{'call': 2}]
    assert _read(cache, fetch) == [{'call': 3}]
    assert cache.invalidate((shot,)) is True
    assert backend._attempts[-1] == (asset, shot)
    assert _read(cache, fetch) == [{'call': 4}]
    assert _read(cache, fetch) == [{'call': 4}]


def test_constructor_rejects_backend_shape_that_disagrees_with_settings() -> None:
    settings = CacheSettings()
    clock = _FakeClock()
    metrics = _RecordingMetrics()
    backend = LocalBackend(settings, clock, metrics)
    identity = Identity('script', 'studio', 'render')

    with pytest.raises(ValueError, match='topology'):
        QueryCache(settings, identity, KeyBuilder(), metrics=metrics)
    with pytest.raises(ValueError, match='topology'):
        QueryCache(settings, identity, KeyBuilder(), l1=backend, l2=backend, metrics=metrics)


def test_container_shaped_metrics_sink_is_not_replaced() -> None:

    class _CountingSink(_RecordingMetrics):
        __slots__ = ()

        def __len__(self) -> int:
            return len(self._increments)

    settings = CacheSettings()
    clock = _FakeClock()
    sink = _CountingSink()
    cache = QueryCache(
        settings,
        Identity('script', 'studio', 'render'),
        KeyBuilder(),
        l1=LocalBackend(settings, clock, sink),
        metrics=sink,
    )

    assert _read(cache, lambda: [{'id': 1}]) == [{'id': 1}]
    assert any(name == 'cache_query_classification' for name, _labels in sink._increments)


def test_custom_backend_hit_expiring_during_decode_is_a_miss(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = CacheSettings()
    clock = _FakeClock()
    stored = LocalBackend(settings, clock, _RecordingMetrics())

    backend = _LegacyHitBackend(stored)
    cache, _clock, _metrics = _cache(settings, backend=backend, clock=clock)
    monkeypatch.setattr(cache_module, 'monotonic', clock)
    assert _read(cache, lambda: {'items': [1]}) == {'items': [1]}
    clock._advance(29.75)
    original_decode = cache_common_module.decode_value

    def delayed_decode(payload: bytes) -> object:
        value = original_decode(payload)
        clock._advance(0.25)
        return value

    monkeypatch.setattr(cache_common_module, 'decode_value', delayed_decode)

    assert _read(cache, lambda: {'items': [2]}) == {'items': [2]}


def test_custom_backend_two_field_hit_uses_legacy_decode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = CacheSettings()
    clock = _FakeClock()
    stored = LocalBackend(settings, clock, _RecordingMetrics())

    backend = _LegacyHitBackend(stored)
    cache, _clock, _metrics = _cache(settings, backend=backend, clock=clock)
    assert _read(cache, lambda: {'items': [1]}) == {'items': [1]}
    original_decode = cache_common_module.decode_value
    decode_count = 0

    def count_decode(payload: bytes) -> object:
        nonlocal decode_count
        decode_count += 1
        return original_decode(payload)

    monkeypatch.setattr(cache_common_module, 'decode_value', count_decode)

    assert _read(cache, lambda: {'items': [2]}) == {'items': [1]}
    assert decode_count == 1


def test_delayed_success_cannot_clear_a_newer_identical_failure() -> None:
    settings = CacheSettings()
    clock = _FakeClock()
    entered = Event()
    proceed = Event()
    outcomes = ['fail', 'block', 'ok', 'fail', 'ok']

    class _ScriptedBackend(_ObservedBackend):
        def __init__(self, backend: CacheBackend) -> None:
            super().__init__(backend)
            self._attempts: list[tuple[GenerationChange, ...]] = []

        def change(self, changes: tuple[GenerationChange, ...]) -> None:
            self._attempts.append(changes)
            outcome = outcomes[len(self._attempts) - 1]
            if outcome == 'block':
                entered.set()
                assert proceed.wait(timeout=2)
            if outcome == 'fail':
                raise BackendUnavailable('fence failed')
            self._backend.change(changes)

    backend = _ScriptedBackend(LocalBackend(settings, clock, _RecordingMetrics()))
    cache, _clock, _metrics = _cache(settings, backend=backend, clock=clock)
    asset = GenerationChange(DependencyTag.type_membership(_SITE, 'Asset'))
    shot = GenerationChange(DependencyTag.type_membership(_SITE, 'Shot'))
    calls = 0

    def fetch() -> object:
        nonlocal calls
        calls += 1
        return [{'call': calls}]

    assert _read(cache, fetch) == [{'call': 1}]
    assert cache.invalidate((asset,)) is False
    with ThreadPoolExecutor(max_workers=1) as pool:
        delayed = pool.submit(cache.invalidate, (shot,))
        assert entered.wait(timeout=2)
        assert cache.invalidate((shot,)) is True
        assert cache.invalidate((asset,)) is False
        proceed.set()
        assert delayed.result(timeout=2) is True

    assert backend._attempts[1] == (asset, shot)
    assert backend._attempts[2] == (asset, shot)
    assert backend._attempts[3] == (asset,)
    assert _read(cache, fetch) == [{'call': 2}]
    assert _read(cache, fetch) == [{'call': 3}]
    assert cache.invalidate((shot,)) is True
    assert backend._attempts[-1] == (asset, shot)
    assert _read(cache, fetch) == [{'call': 4}]
    assert _read(cache, fetch) == [{'call': 4}]


def test_independent_caches_keep_nested_policy_scopes_separate() -> None:
    first, _, _ = _cache()
    second, _, _ = _cache()

    with first.policy(mode='ttl_only'):
        assert first.bypass_active is False
        assert second.bypass_active is False
        with second.bypass():
            assert first.bypass_active is False
            assert second.bypass_active is True
        assert second.bypass_active is False


def test_invalid_policy_input_is_rejected_before_changing_thread_state() -> None:
    cache, _clock, _metrics = _cache()

    with (
        pytest.raises(ValueError, match='mode'),
        cache.policy(mode=cast('Literal["event_aware"]', 'adaptive')),
    ):
        pass
    with pytest.raises(ValueError, match='maximum age'), cache.policy(maximum_age=0):
        pass

    calls = 0

    def fetch() -> object:
        nonlocal calls
        calls += 1
        return [{'id': 7}]

    assert _read(cache, fetch) == [{'id': 7}]
    assert _read(cache, fetch) == [{'id': 7}]
    assert calls == 1


def test_local_age_begins_when_fetch_returns(monkeypatch: pytest.MonkeyPatch) -> None:
    cache, clock, _metrics = _cache()
    monkeypatch.setattr('sg_query_cache._cache.monotonic', clock)
    calls = 0

    def fetch() -> object:
        nonlocal calls
        calls += 1
        clock._advance(10)
        return [{'id': 7}]

    assert _read(cache, fetch, per_call_age=5) == [{'id': 7}]
    clock._advance(4.9)
    assert _read(cache, fetch, per_call_age=5) == [{'id': 7}]
    assert calls == 1
    clock._advance(0.1)
    assert _read(cache, fetch, per_call_age=5) == [{'id': 7}]
    assert _read(cache, fetch, per_call_age=5) == [{'id': 7}]
    assert calls == 2


def test_local_malformed_envelope_fetches_without_population_or_private_metrics() -> None:
    settings = CacheSettings()
    clock = _FakeClock()
    metrics = _RecordingMetrics()
    backend = LocalBackend(settings, clock, metrics)
    cache = QueryCache(
        settings,
        Identity('script', 'studio', 'render'),
        KeyBuilder(),
        l1=backend,
        metrics=metrics,
    )
    private_payload = 'payload-private-detail'
    _store_payload(backend, _call(), _plan(), encode_value((private_payload, {'id': 1})))
    calls = 0

    def fetch() -> object:
        nonlocal calls
        calls += 1
        return [{'response-private-detail': calls}]

    assert _read(cache, fetch) == [{'response-private-detail': 1}]
    assert _read(cache, fetch) == [{'response-private-detail': 2}]
    assert calls == 2
    assert ('cache_bypass', {'reason': 'codec_failure'}) in metrics._increments
    assert private_payload not in str(metrics._increments)
    assert 'response-private-detail' not in str(metrics._increments)


def test_local_miss_populates_an_isolated_hit() -> None:
    cache, _clock, _metrics = _cache()
    calls = 0

    def fetch() -> object:
        nonlocal calls
        calls += 1
        return [{'id': 7, 'links': [{'id': 3}]}]

    first = cast('list[dict[str, object]]', _read(cache, fetch))
    first[0]['links'] = []
    second = cast('list[dict[str, object]]', _read(cache, fetch))
    second[0]['links'] = []
    third = cast('list[dict[str, object]]', _read(cache, fetch))

    assert calls == 1
    assert first is not second
    assert second is not third
    assert third == [{'id': 7, 'links': [{'id': 3}]}]


def test_local_single_flight_collapses_concurrent_fetches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = CacheSettings(limits=LimitSettings(lease_wait_seconds=1))
    cache, _clock, _metrics = _cache(settings)
    fetch_entered = Event()
    follower_waiting = Event()
    release_fetch = Event()
    calls = 0
    original_wait = QueryCache._wait_for_flight

    def observed_wait(
        selected_cache: QueryCache,
        key: str,
        flight: cache_module._Flight,
    ) -> bool:
        follower_waiting.set()
        return original_wait(selected_cache, key, flight)

    monkeypatch.setattr(QueryCache, '_wait_for_flight', observed_wait)

    def fetch() -> object:
        nonlocal calls
        calls += 1
        fetch_entered.set()
        assert release_fetch.wait(timeout=3)
        return [{'id': 7}]

    with ThreadPoolExecutor(max_workers=2) as executor:
        try:
            first = executor.submit(_read, cache, fetch)
            assert fetch_entered.wait(timeout=2)
            second = executor.submit(_read, cache, fetch)
            assert follower_waiting.wait(timeout=2)
        finally:
            release_fetch.set()
        assert first.result(timeout=2) == [{'id': 7}]
        assert second.result(timeout=2) == [{'id': 7}]

    assert calls == 1


def test_local_single_flight_timeout_fetches_without_population() -> None:
    settings = CacheSettings(limits=LimitSettings(lease_wait_seconds=0.02))
    cache, _clock, _metrics = _cache(settings)
    barrier = Barrier(2)
    calls = 0

    def fetch() -> object:
        nonlocal calls
        calls += 1
        current = calls
        if current == 1:
            barrier.wait()
            barrier.wait()
        return [{'call': current}]

    with ThreadPoolExecutor(max_workers=2) as executor:
        leader = executor.submit(_read, cache, fetch)
        barrier.wait()
        follower = executor.submit(_read, cache, fetch)
        assert follower.result() == [{'call': 2}]
        barrier.wait()
        assert leader.result() == [{'call': 1}]

    assert _read(cache, fetch) == [{'call': 1}]
    assert calls == 2


def test_nested_policy_cannot_promote_a_weaker_plan_or_outer_scope() -> None:
    cache, _clock, _metrics = _cache()
    calls = 0

    def fetch() -> object:
        nonlocal calls
        calls += 1
        return [{'call': calls}]

    with cache.policy(mode='ttl_only'), cache.policy(mode='event_aware'):
        assert _read(cache, fetch) == [{'call': 1}]
        assert _read(cache, fetch) == [{'call': 1}]
    with cache.bypass(), cache.policy(mode='event_aware'):
        assert _read(cache, fetch) == [{'call': 2}]
        assert _read(cache, fetch) == [{'call': 3}]
    with cache.policy(mode='event_aware'):
        assert _read(cache, fetch, plan=_plan('bypass')) == [{'call': 4}]
        assert _read(cache, fetch, plan=_plan('bypass')) == [{'call': 5}]


def test_nested_scope_cannot_relax_an_enclosing_age() -> None:
    cache, clock, _metrics = _cache()
    calls = 0

    def fetch() -> object:
        nonlocal calls
        calls += 1
        return [{'call': calls}]

    assert _read(cache, fetch) == [{'call': 1}]
    clock._advance(15)

    with cache.policy(maximum_age=5), cache.policy(maximum_age=600):
        assert _read(cache, fetch) == [{'call': 2}]
    assert calls == 2


def test_oversized_failed_fence_is_canonical_before_backend_and_in_retained_state() -> None:
    settings = CacheSettings()
    clock = _FakeClock()
    fenced = _FencedBackend(LocalBackend(settings, clock, _RecordingMetrics()), failures=2)
    cache, _clock, _metrics = _cache(settings, backend=fenced, clock=clock)
    site_change = GenerationChange(DependencyTag.site(_SITE))
    site = (site_change,)
    oversized = tuple(
        GenerationChange(DependencyTag.entity(_SITE, 'Asset', entity_id))
        for entity_id in range(1, 258)
    )

    assert cache.invalidate(oversized) is False
    assert fenced._seen == [site]
    assert cache._fences['l1']._retained == site

    shot = GenerationChange(DependencyTag.type_membership(_SITE, 'Shot'))
    assert cache.invalidate((shot,)) is False
    assert fenced._seen[-1] == (site_change, shot)
    assert cache._fences['l1']._retained == site

    task = GenerationChange(DependencyTag.type_membership(_SITE, 'Task'))
    assert cache.invalidate((task,)) is True
    assert fenced._seen[-1] == (site_change, task)


def test_oversized_integer_timing_metadata_fails_open_as_codec_failure() -> None:
    settings = CacheSettings()
    clock = _FakeClock()
    metrics = _RecordingMetrics()
    backend = LocalBackend(settings, clock, metrics)
    cache = QueryCache(
        settings,
        Identity('script', 'studio', 'render'),
        KeyBuilder(),
        l1=backend,
        metrics=metrics,
    )
    _store_payload(
        backend,
        _call(),
        _plan(),
        encode_value(('sg-query-cache-result-v1', 10**400, 30.0, 0.0, [{'id': 7}])),
    )
    calls = 0

    def fetch() -> object:
        nonlocal calls
        calls += 1
        return [{'id': 8}]

    assert _read(cache, fetch) == [{'id': 8}]
    assert calls == 1
    assert ('cache_bypass', {'reason': 'codec_failure'}) in metrics._increments


def test_oversized_value_returns_directly_without_evicting_a_useful_hit() -> None:
    settings = CacheSettings(
        l1=L1Settings(result_budget_bytes=10_000),
        limits=LimitSettings(max_serialized_result_bytes=200),
    )
    cache, _clock, _metrics = _cache(settings)
    calls = 0

    def small_fetch() -> object:
        nonlocal calls
        calls += 1
        return [{'id': 7}]

    assert _read(cache, small_fetch) == [{'id': 7}]
    large_call = ExactCall('find_one', (('entity_type', 'Asset'), ('filters', ())))
    large_calls = 0

    def large_fetch() -> object:
        nonlocal large_calls
        large_calls += 1
        return [{'code': 'x' * 100}]

    for _index in range(2):
        assert cache.read(large_call, _plan(), large_fetch, _CLIENT_OPTIONS) == [
            {'code': 'x' * 100}
        ]
    assert _read(cache, small_fetch) == [{'id': 7}]
    assert calls == 1
    assert large_calls == 2


def test_parsed_read_rejects_a_first_key_crossing_a_retained_fence() -> None:
    settings = CacheSettings()
    clock = _FakeClock()
    l1 = _BlockingBackend(LocalBackend(settings, clock, _RecordingMetrics()), failures=1)
    cache = QueryCache(
        settings,
        Identity('script', 'studio', 'render'),
        KeyBuilder(),
        l1=l1,
        metrics=_RecordingMetrics(),
    )
    calls = 0

    def fetch() -> object:
        nonlocal calls
        calls += 1
        return {'call': calls}

    assert cache.read_parsed(_call(), _plan(), fetch, _CLIENT_OPTIONS, _parse) == (
        'parsed',
        {'call': 1},
    )
    l1._arm('resolve')
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(cache.read_parsed, _call(), _plan(), fetch, _CLIENT_OPTIONS, _parse)
        assert l1._entered.wait(timeout=2)
        acknowledged = cache.invalidate(
            (GenerationChange(DependencyTag.type_membership(_SITE, 'Asset')),)
        )
        l1._release_block.set()
        value = future.result(timeout=2)

    assert acknowledged is False
    assert value == ('parsed', {'call': 2})
    assert calls == 2


def test_per_call_age_joins_the_lexical_minimum() -> None:
    cache, clock, _metrics = _cache()
    calls = 0

    def fetch() -> object:
        nonlocal calls
        calls += 1
        return [{'call': calls}]

    assert _read(cache, fetch) == [{'call': 1}]
    clock._advance(15)

    with cache.policy(maximum_age=5):
        assert _read(cache, fetch, per_call_age=600) == [{'call': 2}]
    assert calls == 2


@pytest.mark.parametrize('per_call_age', [float('nan'), float('inf')])
def test_per_call_age_rejects_nonfinite_values_inside_a_lexical_scope(
    per_call_age: float,
) -> None:
    cache, _clock, _metrics = _cache()

    with (
        cache.policy(maximum_age=5),
        pytest.raises(
            TypeError,
            match='per-call maximum-age override must be a finite number',
        ),
    ):
        _read(cache, lambda: [{'id': 1}], per_call_age=per_call_age)


def test_policy_scope_is_thread_local() -> None:
    cache, _clock, _metrics = _cache()
    calls = 0

    def fetch() -> object:
        nonlocal calls
        calls += 1
        return [{'call': calls}]

    def worker() -> tuple[object, object]:
        return _read(cache, fetch), _read(cache, fetch)

    with cache.bypass(), ThreadPoolExecutor(max_workers=1) as executor:
        assert _read(cache, fetch) == [{'call': 1}]
        assert executor.submit(worker).result() == ([{'call': 2}], [{'call': 2}])
        assert _read(cache, fetch) == [{'call': 3}]

    assert calls == 3


def test_post_fetch_decision_cannot_revive_an_expired_original_lease() -> None:
    settings = CacheSettings()
    clock = _FakeClock()
    backend = LocalBackend(settings, clock, _RecordingMetrics())
    cache = QueryCache(
        settings,
        Identity('script', 'studio', 'render'),
        KeyBuilder(),
        l1=backend,
    )
    calls = 0

    def fetch() -> object:
        nonlocal calls
        calls += 1
        return [{'call': calls}]

    def expire_lease(_value: object) -> _PopulationDecision:
        clock._advance(31)
        return _PopulationDecision(5)

    assert _read(cache, fetch, population_decision=expire_lease) == [{'call': 1}]
    assert _read(cache, fetch, population_decision=expire_lease) == [{'call': 2}]
    assert calls == 2


def test_post_fetch_decision_delay_consumes_the_original_fill_age(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache, _clock, _metrics = _cache()
    local_clock = _FakeClock()
    monkeypatch.setattr(cache_module, 'monotonic', local_clock)
    calls = 0

    def fetch() -> object:
        nonlocal calls
        calls += 1
        return [{'call': calls}]

    def delayed_decision(_value: object) -> _PopulationDecision:
        local_clock._advance(6)
        return _PopulationDecision(5)

    assert _read(cache, fetch, population_decision=delayed_decision) == [{'call': 1}]
    assert _read(cache, fetch, population_decision=delayed_decision) == [{'call': 2}]
    assert calls == 2


def test_post_fetch_invalidation_cannot_publish_under_the_new_vector() -> None:
    settings = CacheSettings()
    clock = _FakeClock()
    backend = LocalBackend(settings, clock, _RecordingMetrics())
    cache = QueryCache(
        settings,
        Identity('script', 'studio', 'render'),
        KeyBuilder(),
        l1=backend,
    )
    calls = 0

    def fetch() -> object:
        nonlocal calls
        calls += 1
        return [{'call': calls}]

    def invalidate_after_fetch(_value: object) -> _PopulationDecision:
        cache.invalidate((GenerationChange(DependencyTag.type_membership(_SITE, 'Asset')),))
        return _PopulationDecision(5)

    assert _read(cache, fetch, population_decision=invalidate_after_fetch) == [{'call': 1}]
    assert _read(cache, fetch, population_decision=invalidate_after_fetch) == [{'call': 2}]
    assert calls == 2


def test_query_cache_constructs_default_key_builder() -> None:
    settings = CacheSettings()
    clock = _FakeClock()
    metrics = _RecordingMetrics()
    backend = LocalBackend(settings, clock, metrics)
    cache = QueryCache(
        settings,
        Identity('script', 'studio', 'render'),
        l1=backend,
        metrics=metrics,
    )
    calls = 0

    def fetch() -> object:
        nonlocal calls
        calls += 1
        return {'id': 7}

    assert _read(cache, fetch) == {'id': 7}
    assert _read(cache, fetch) == {'id': 7}
    assert calls == 1


def test_retained_changes_collapse_to_the_site_generation_beyond_the_bound() -> None:
    settings = CacheSettings()
    clock = _FakeClock()
    fenced = _FencedBackend(LocalBackend(settings, clock, _RecordingMetrics()), failures=300)
    cache, _clock, _metrics = _cache(settings, backend=fenced, clock=clock)

    for entity_id in range(1, 301):
        assert (
            cache.invalidate((GenerationChange(DependencyTag.entity(_SITE, 'Asset', entity_id)),))
            is False
        )
    assert (
        cache.invalidate((GenerationChange(DependencyTag.type_membership(_SITE, 'Shot')),)) is True
    )

    assert fenced._seen[-1][0] == GenerationChange(DependencyTag.site(_SITE))
    assert len(fenced._seen[-1]) == 2


def test_scoped_age_is_capped_and_per_call_age_has_stronger_precedence() -> None:
    settings = CacheSettings(global_safety_max_seconds=20)
    cache, clock, _metrics = _cache(settings)
    calls = 0

    def fetch() -> object:
        nonlocal calls
        calls += 1
        return [{'call': calls}]

    with cache.policy(maximum_age=100):
        assert _read(cache, fetch, per_call_age=3) == [{'call': 1}]
        clock._advance(3)
        assert _read(cache, fetch, per_call_age=3) == [{'call': 2}]
    assert calls == 2


def test_scoped_and_per_call_ages_use_the_minimum() -> None:
    cache, _, _ = _cache()

    assert cache._effective_age(None) is None
    with cache.policy(maximum_age=5), cache.policy(maximum_age=600):
        assert cache._effective_age(None) == 5
        assert cache._effective_age(3) == 3


def test_serialization_delay_shortens_the_committed_lifetime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = CacheSettings()
    clock = _FakeClock()
    backend = LocalBackend(settings, clock, _RecordingMetrics())
    cache, _clock, _metrics = _cache(settings, backend=backend, clock=clock)
    monkeypatch.setattr(cache_module, 'monotonic', clock)
    original = cache_common_module.encode_value

    def slow_encode(value: object) -> bytes:
        clock._advance(5)
        return original(value)

    monkeypatch.setattr(cache_common_module, 'encode_value', slow_encode)

    assert _read(cache, lambda: [{'id': 1}]) == [{'id': 1}]

    key = KeyBuilder.qualified_key(
        KeyBuilder.base_key(
            _SITE,
            Identity('script', 'studio', 'render'),
            _CLIENT_OPTIONS,
            _call(),
        ),
        backend.resolve(_plan().tags),
        'result',
    )
    hit = backend.lookup(key)
    assert hit is not None
    assert hit.remaining_seconds == 25
    monkeypatch.setattr(cache_common_module, 'encode_value', original)
    assert _read(cache, lambda: [{'id': 2}], per_call_age=6.0) == [{'id': 1}]
    assert _read(cache, lambda: [{'id': 2}], per_call_age=5.0) == [{'id': 2}]


def test_serialization_time_is_deducted_before_commit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache, clock, _metrics = _cache()
    monkeypatch.setattr('sg_query_cache._cache.monotonic', clock)
    original_encode = cache_common_module.encode_value

    def slow_encode(value: object) -> bytes:
        clock._advance(3)
        return original_encode(value)

    monkeypatch.setattr(cache_common_module, 'encode_value', slow_encode)
    calls = 0

    def fetch() -> object:
        nonlocal calls
        calls += 1
        return [{'id': 7}]

    assert _read(cache, fetch, per_call_age=5) == [{'id': 7}]
    clock._advance(2)
    assert _read(cache, fetch, per_call_age=5) == [{'id': 7}]
    assert calls == 2


def test_shared_late_reservation_rechecks_the_committed_result() -> None:
    first, _unused, backend, _clock = _shared_caches()
    observed = _ObservedBackend(backend, gate_reserve_at=1)
    settings = CacheSettings(
        l1=L1Settings(enabled=False),
        l2=L2Settings(enabled=True),
        limits=LimitSettings(lease_wait_seconds=0.5),
    )
    second = QueryCache(
        settings,
        Identity('script', 'studio', 'render'),
        KeyBuilder(),
        l2=observed,
        metrics=_RecordingMetrics(),
    )
    fetch_started = Event()
    allow_fetch = Event()
    responses: list[dict[str, int]] = []

    def fetch() -> list[dict[str, int]]:
        response = {'call': len(responses) + 1}
        responses.append(response)
        if len(responses) == 1:
            fetch_started.set()
            assert allow_fetch.wait(timeout=2)
        return [response]

    with ThreadPoolExecutor(max_workers=2) as executor:
        holder = executor.submit(_read, first, fetch)
        assert fetch_started.wait(timeout=2)
        late = executor.submit(_read, second, fetch)
        assert observed._reserve_entered.wait(timeout=2)
        allow_fetch.set()
        assert holder.result() == [{'call': 1}]
        observed._allow_reserve.set()

    assert late.result() == [{'call': 1}]
    assert responses == [{'call': 1}]
    assert observed._lookup_calls == 2
    assert observed._reserve_calls == 1
    assert observed._commit_calls == 0
    assert observed._release_calls == 1


def test_shared_lease_timeout_fetches_without_population() -> None:
    first, second, _backend, _clock = _shared_caches(lease_wait_seconds=0.02)
    started = Event()
    release = Event()
    calls = 0

    def fetch() -> object:
        nonlocal calls
        calls += 1
        current = calls
        if current == 1:
            started.set()
            assert release.wait(1)
        return [{'call': current}]

    with ThreadPoolExecutor(max_workers=2) as executor:
        holder = executor.submit(_read, first, fetch)
        assert started.wait(1)
        waiter = executor.submit(_read, second, fetch)
        assert waiter.result() == [{'call': 2}]
        release.set()

    assert holder.result() == [{'call': 1}]
    assert _read(second, fetch) == [{'call': 1}]
    assert calls == 2


def test_shared_malformed_envelope_skips_lease_wait_and_population(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = CacheSettings(
        l1=L1Settings(enabled=False),
        l2=L2Settings(enabled=True),
        limits=LimitSettings(lease_wait_seconds=0.02),
    )
    clock = _FakeClock()
    metrics = _RecordingMetrics()
    backend = LocalBackend(settings, clock, metrics)
    cache = QueryCache(
        settings,
        Identity('script', 'studio', 'render'),
        KeyBuilder(),
        l2=backend,
        metrics=metrics,
    )
    private_payload = 'shared-payload-private-detail'
    key = _store_payload(backend, _call(), _plan(), encode_value((private_payload, {'id': 1})))
    blocker = backend.reserve(key, 30)
    assert blocker is not None

    def forbidden_sleep(seconds: float) -> None:
        _ = seconds
        raise AssertionError('malformed L2 must not wait on a lease')

    monkeypatch.setattr(cache_module, 'sleep', forbidden_sleep)
    response = [{'shared-response-private-detail': 1}]
    assert _read(cache, lambda: response) is response
    assert ('cache_bypass', {'reason': 'codec_failure'}) in metrics._increments
    assert private_payload not in str(metrics._increments)
    assert 'shared-response-private-detail' not in str(metrics._increments)


def test_shared_post_reservation_generation_change_does_not_fill_the_old_key() -> None:
    _first, _second, backend, _clock = _shared_caches()
    change = (GenerationChange(DependencyTag.type_membership(_SITE, 'Asset')),)
    observed = _ObservedBackend(backend, after_first_reserve=lambda: backend.change(change))
    settings = CacheSettings(l1=L1Settings(enabled=False), l2=L2Settings(enabled=True))
    cache = QueryCache(
        settings,
        Identity('script', 'studio', 'render'),
        KeyBuilder(),
        l2=observed,
        metrics=_RecordingMetrics(),
    )
    calls = 0

    def fetch() -> list[dict[str, int]]:
        nonlocal calls
        calls += 1
        return [{'call': calls}]

    assert _read(cache, fetch) == [{'call': 1}]
    assert observed._lookup_calls == 2
    assert observed._reserve_calls == 1
    assert observed._commit_calls == 0
    assert observed._release_calls == 1


def test_shared_post_reservation_recheck_failure_falls_through_safely() -> None:
    _first, _second, backend, _clock = _shared_caches()
    observed = _ObservedBackend(backend, fail_lookup_at=2)
    settings = CacheSettings(l1=L1Settings(enabled=False), l2=L2Settings(enabled=True))
    metrics = _RecordingMetrics()
    cache = QueryCache(
        settings,
        Identity('script', 'studio', 'render'),
        KeyBuilder(),
        l2=observed,
        metrics=metrics,
    )
    response = [{'id': 7}]

    assert _read(cache, lambda: response) is response
    assert observed._lookup_calls == 2
    assert observed._reserve_calls == 1
    assert observed._commit_calls == 0
    assert observed._release_calls == 1
    assert ('cache_bypass', {'reason': 'backend_failure'}) in metrics._increments


def test_shared_post_reservation_recheck_miss_still_fills() -> None:
    _first, _second, backend, _clock = _shared_caches()
    observed = _ObservedBackend(backend)
    settings = CacheSettings(l1=L1Settings(enabled=False), l2=L2Settings(enabled=True))
    cache = QueryCache(
        settings,
        Identity('script', 'studio', 'render'),
        KeyBuilder(),
        l2=observed,
        metrics=_RecordingMetrics(),
    )
    calls = 0

    def fetch() -> list[dict[str, int]]:
        nonlocal calls
        calls += 1
        return [{'call': calls}]

    assert _read(cache, fetch) == [{'call': 1}]
    assert observed._lookup_calls == 2
    assert observed._reserve_calls == 1
    assert observed._commit_calls == 1
    assert observed._release_calls == 1
    assert calls == 1


def test_shared_waiter_late_reservation_rechecks_the_committed_result() -> None:
    first, _unused, backend, _clock = _shared_caches()
    observed = _ObservedBackend(backend, gate_reserve_at=2)
    settings = CacheSettings(
        l1=L1Settings(enabled=False),
        l2=L2Settings(enabled=True),
        limits=LimitSettings(lease_wait_seconds=0.5),
    )
    second = QueryCache(
        settings,
        Identity('script', 'studio', 'render'),
        KeyBuilder(),
        l2=observed,
        metrics=_RecordingMetrics(),
    )
    fetch_started = Event()
    allow_fetch = Event()
    responses: list[dict[str, int]] = []

    def fetch() -> list[dict[str, int]]:
        response = {'call': len(responses) + 1}
        responses.append(response)
        if len(responses) == 1:
            fetch_started.set()
            assert allow_fetch.wait(timeout=2)
        return [response]

    with ThreadPoolExecutor(max_workers=2) as executor:
        holder = executor.submit(_read, first, fetch)
        assert fetch_started.wait(timeout=2)
        late_waiter = executor.submit(_read, second, fetch)
        assert observed._reserve_entered.wait(timeout=2)
        allow_fetch.set()
        assert holder.result() == [{'call': 1}]
        observed._allow_reserve.set()

    assert late_waiter.result() == [{'call': 1}]
    assert responses == [{'call': 1}]
    assert observed._lookup_calls == 3
    assert observed._reserve_calls == 2
    assert observed._commit_calls == 0
    assert observed._release_calls == 1


def test_shared_waiter_reresolves_after_generation_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first, second, backend, _clock = _shared_caches()
    plan = _plan()
    base_key = KeyBuilder.base_key(
        _SITE, Identity('script', 'studio', 'render'), _CLIENT_OPTIONS, _call()
    )
    old_key = KeyBuilder.qualified_key(base_key, backend.resolve(plan.tags), 'result')
    lease = backend.reserve(old_key, 30)
    assert lease is not None
    old_payload = cache_common_module.encode_result(
        cache_common_module.TimedResult(0, [{'version': 'old'}]), 0, 30
    )
    calls = 0
    published = False

    def new_fetch() -> object:
        nonlocal calls
        calls += 1
        return [{'version': 'new'}]

    def publish_old_result(_delay: float) -> None:
        nonlocal published
        if published:
            return
        # The reader is already waiting on the old lease when invalidation arrives.
        backend.change((GenerationChange(DependencyTag.type_membership(_SITE, 'Asset')),))
        assert backend.commit(lease, old_payload, 30)
        published = True

    monkeypatch.setattr(cache_module, 'sleep', publish_old_result)
    assert _read(second, new_fetch) == [{'version': 'new'}]
    assert published
    assert _read(first, new_fetch) == [{'version': 'new'}]
    assert calls == 1


def test_shorter_caller_age_revalidates_without_codec_failure() -> None:
    cache, clock, metrics = _cache()
    calls = 0

    def fetch() -> object:
        nonlocal calls
        calls += 1
        return [{'call': calls}]

    assert _read(cache, fetch) == [{'call': 1}]
    clock._advance(5)
    assert _read(cache, fetch, per_call_age=3) == [{'call': 2}]
    assert _read(cache, fetch, per_call_age=3) == [{'call': 2}]
    assert calls == 2
    assert ('cache_bypass', {'reason': 'codec_failure'}) not in metrics._increments


def test_stale_failure_cannot_expand_a_collapsed_retained_fence() -> None:
    settings = CacheSettings()
    clock = _FakeClock()
    entered = Event()
    proceed = Event()

    class _InterleavedFailureBackend(_ObservedBackend):
        def __init__(self, backend: CacheBackend) -> None:
            super().__init__(backend)
            self._attempts: list[tuple[GenerationChange, ...]] = []

        def change(self, changes: tuple[GenerationChange, ...]) -> None:
            self._attempts.append(changes)
            if len(self._attempts) == 1:
                entered.set()
                assert proceed.wait(timeout=2)
                raise BackendUnavailable('stale fence failed')
            if len(self._attempts) == 2:
                raise BackendUnavailable('site fence failed')
            self._backend.change(changes)

    backend = _InterleavedFailureBackend(LocalBackend(settings, clock, _RecordingMetrics()))
    cache, _clock, _metrics = _cache(settings, backend=backend, clock=clock)
    asset = GenerationChange(DependencyTag.type_membership(_SITE, 'Asset'))
    site_change = GenerationChange(DependencyTag.site(_SITE))
    site = (site_change,)

    with ThreadPoolExecutor(max_workers=1) as pool:
        stale = pool.submit(cache.invalidate, (asset,))
        assert entered.wait(timeout=2)
        assert cache.invalidate(site) is False
        assert cache._fences['l1']._retained == site
        proceed.set()
        assert stale.result(timeout=2) is False

    assert cache._fences['l1']._retained == site
    shot = GenerationChange(DependencyTag.type_membership(_SITE, 'Shot'))
    assert cache.invalidate((shot,)) is True
    assert backend._attempts[-1] == (site_change, shot)


def test_unacknowledged_changes_are_retained_and_replayed_before_reads_resume() -> None:
    settings = CacheSettings()
    clock = _FakeClock()
    fenced = _FencedBackend(LocalBackend(settings, clock, _RecordingMetrics()), failures=1)
    cache, _clock, metrics = _cache(settings, backend=fenced, clock=clock)
    asset = GenerationChange(DependencyTag.type_membership(_SITE, 'Asset'))
    shot = GenerationChange(DependencyTag.type_membership(_SITE, 'Shot'))
    calls = 0

    def fetch() -> object:
        nonlocal calls
        calls += 1
        return [{'call': calls}]

    assert _read(cache, fetch) == [{'call': 1}]
    assert cache.invalidate((asset,)) is False
    assert _read(cache, fetch) == [{'call': 2}]
    assert _read(cache, fetch) == [{'call': 3}]
    assert ('cache_bypass', {'reason': 'unacknowledged_fence'}) in metrics._increments

    assert cache.invalidate((shot,)) is True
    assert fenced._seen[-1] == (asset, shot)
    assert _read(cache, fetch) == [{'call': 4}]
    assert _read(cache, fetch) == [{'call': 4}]


@pytest.mark.parametrize(
    'value',
    [
        object(),
        {'unsupported': {1, 2}},
    ],
)
def test_uncacheable_values_return_directly(value: object) -> None:
    cache, _clock, _metrics = _cache()
    calls = 0

    def fetch() -> object:
        nonlocal calls
        calls += 1
        return value

    assert _read(cache, fetch) is value
    assert _read(cache, fetch) is value
    assert calls == 2


@pytest.mark.parametrize('failure', ['encode', 'fetch'])
def test_shared_holder_releases_lease_in_finally(
    failure: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first, second, _backend, _clock = _shared_caches(lease_wait_seconds=0.02)
    expected = LookupError('sg-original')
    if failure == 'encode':

        def fail_encode(value: object, origin_elapsed: float, lifetime: float) -> bytes:
            _ = value, origin_elapsed, lifetime
            raise CodecError('closed')

        monkeypatch.setattr(cache_module, 'encode_result', fail_encode)

    calls = 0

    def fetch() -> object:
        nonlocal calls
        calls += 1
        if failure == 'fetch' and calls == 1:
            raise expected
        return [{'call': calls}]

    context = pytest.raises(LookupError) if failure == 'fetch' else nullcontext()
    with context:
        _read(first, fetch)
    if failure == 'encode':
        monkeypatch.undo()

    assert _read(second, fetch) == [{'call': 2}]
    assert _read(second, fetch) == [{'call': 2}]
    assert calls == 2


@pytest.mark.parametrize('topology', ['local', 'shared'])
def test_parsed_fill_honors_a_later_shorter_scope(
    topology: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _single_layer_settings(topology)
    clock = _FakeClock()
    monkeypatch.setattr(cache_module, 'monotonic', clock)
    metrics = _RecordingMetrics()
    backend = LocalBackend(settings, clock, metrics)
    cache = _single_layer_cache(settings, backend, metrics)
    calls = 0

    def fetch() -> object:
        nonlocal calls
        calls += 1
        return {'call': calls}

    assert cache.read_parsed(_call(), _plan(), fetch, _CLIENT_OPTIONS, _parse) == (
        'parsed',
        {'call': 1},
    )
    assert cache.read_parsed(_call(), _plan(), fetch, _CLIENT_OPTIONS, _parse) == (
        'parsed',
        {'call': 1},
    )
    clock._advance(2)

    with cache.policy(maximum_age=1):
        assert cache.read_parsed(_call(), _plan(), fetch, _CLIENT_OPTIONS, _parse) == (
            'parsed',
            {'call': 2},
        )

    assert calls == 2


@pytest.mark.parametrize('topology', ['local', 'shared'])
def test_parsed_hit_honors_a_later_shorter_scope(
    topology: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _single_layer_settings(topology)
    clock = _FakeClock()
    monkeypatch.setattr(cache_module, 'monotonic', clock)
    metrics = _RecordingMetrics()
    backend = LocalBackend(settings, clock, metrics)
    _store_payload(
        backend,
        _call(),
        _plan(),
        encode_value(('sg-query-cache-result-v1', 0.0, 30.0, 0.0, {'stored': True})),
    )
    clock._advance(8)
    cache = _single_layer_cache(settings, backend, metrics)
    calls = 0

    def fetch() -> object:
        nonlocal calls
        calls += 1
        return {'call': calls}

    assert cache.read_parsed(_call(), _plan(), fetch, _CLIENT_OPTIONS, _parse) == (
        'parsed',
        {'stored': True},
    )
    with cache.policy(maximum_age=5):
        assert cache.read_parsed(_call(), _plan(), fetch, _CLIENT_OPTIONS, _parse) == (
            'parsed',
            {'call': 1},
        )
    assert calls == 1


def test_unknown_plan_classification_bypasses_instead_of_reaching_ttl_selection() -> None:
    cache, _clock, _metrics = _cache()
    plan = DependencyPlan('future_policy', _plan().tags, 'unknown_policy')
    calls = 0

    def fetch() -> object:
        nonlocal calls
        calls += 1
        return [{'call': calls}]

    assert _read(cache, fetch, plan=plan) == [{'call': 1}]
    assert _read(cache, fetch, plan=plan) == [{'call': 2}]
    assert calls == 2


@pytest.mark.parametrize('topology', ['local', 'shared'])
def test_unstored_parsed_value_honors_a_later_shorter_scope(
    topology: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _single_layer_settings(topology)
    clock = _FakeClock()
    monkeypatch.setattr(cache_module, 'monotonic', clock)
    metrics = _RecordingMetrics()
    backend = LocalBackend(settings, clock, metrics)
    cache = _single_layer_cache(settings, backend, metrics)
    calls = 0

    def fetch() -> object:
        nonlocal calls
        calls += 1
        return bytes([calls]) * (settings.limits.max_serialized_result_bytes + 1)

    first = cache.read_parsed(_call(), _plan(), fetch, _CLIENT_OPTIONS, _parse)
    assert cache.read_parsed(_call(), _plan(), fetch, _CLIENT_OPTIONS, _parse) == first
    clock._advance(2)
    with cache.policy(maximum_age=1):
        second = cache.read_parsed(_call(), _plan(), fetch, _CLIENT_OPTIONS, _parse)

    assert first != second
    assert calls == 2


@pytest.mark.parametrize('topology', ['local', 'shared'])
def test_preliminary_resolve_failure_uses_the_ordinary_single_layer_path(
    topology: str,
) -> None:
    settings = _single_layer_settings(topology)
    clock = _FakeClock()
    metrics = _RecordingMetrics()
    raw_backend = LocalBackend(settings, clock, metrics)
    _store_payload(
        raw_backend,
        _call(),
        _plan(),
        encode_value(('sg-query-cache-result-v1', 0.0, 30.0, 0.0, {'stored': True})),
    )
    backend = ResolveFailureBackend(raw_backend, failures=1)
    cache = _single_layer_cache(settings, backend, metrics)
    calls = 0

    def fetch() -> object:
        nonlocal calls
        calls += 1
        return {'fresh': calls}

    assert cache.read_parsed(_call(), _plan(), fetch, _CLIENT_OPTIONS, _parse) == (
        'parsed',
        {'stored': True},
    )
    assert calls == 0


@pytest.mark.parametrize(
    ('failures', 'acknowledged'),
    [(1, False), (0, True)],
    ids=['retained-fence', 'advanced-revision'],
)
def test_ordinary_read_rejects_a_hit_crossing_a_fence_revision(
    failures: int,
    acknowledged: bool,
) -> None:
    settings = CacheSettings()
    clock = _FakeClock()
    raw_l1 = LocalBackend(settings, clock, _RecordingMetrics())
    _store_payload(
        raw_l1,
        _call(),
        _plan(),
        encode_value(('sg-query-cache-result-v1', 0.0, 5.0, 0.01, [{'id': 7}])),
        age=5,
    )
    l1 = _BlockingBackend(raw_l1, failures)
    cache = QueryCache(
        settings,
        Identity('script', 'studio', 'render'),
        KeyBuilder(),
        l1=l1,
        metrics=_RecordingMetrics(),
    )
    calls = 0

    def fetch() -> object:
        nonlocal calls
        calls += 1
        return [{'id': 8}]

    l1._arm('lookup')
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(_read, cache, fetch)
        assert l1._entered.wait(timeout=2)
        result = cache.invalidate(
            (GenerationChange(DependencyTag.type_membership(_SITE, 'Asset')),)
        )
        l1._release_block.set()
        value = future.result(timeout=2)

    assert result is acknowledged
    assert value == [{'id': 8}]
    assert calls == 1


@pytest.mark.parametrize(
    ('cap', 'reason', 'bypass'),
    [
        pytest.param(None, 'expired', 'population_declined', id='decline-with-reason'),
        pytest.param(None, None, 'population_declined', id='decline-without-reason'),
        pytest.param(0, 'invalid', 'population_decision_failure', id='zero'),
        pytest.param(-1, 'invalid', 'population_decision_failure', id='negative'),
        pytest.param(float('nan'), 'invalid', 'population_decision_failure', id='nan'),
        pytest.param(float('inf'), 'invalid', 'population_decision_failure', id='infinity'),
        pytest.param(
            float('-inf'), 'invalid', 'population_decision_failure', id='negative-infinity'
        ),
        pytest.param(True, 'invalid', 'population_decision_failure', id='bool'),
        pytest.param('2', 'invalid', 'population_decision_failure', id='string'),
        pytest.param('raise', 'invalid', 'population_decision_failure', id='callback-failure'),
    ],
)
def test_population_decision_decline_and_invalid_cap_return_without_populating(
    cap: object,
    reason: str | None,
    bypass: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache, clock, metrics = _cache()
    monkeypatch.setattr(cache_module, 'monotonic', clock)
    calls = 0

    def fetch() -> object:
        nonlocal calls
        calls += 1
        return calls

    def decision(_value: object) -> _PopulationDecision:
        if cap == 'raise':
            raise ValueError('decision failed')
        return _PopulationDecision(cast('float | None', cap), reason)

    assert _read(cache, fetch, population_decision=decision) == 1
    assert _read(cache, fetch, population_decision=decision) == 2
    assert calls == 2
    assert ('cache_bypass', {'reason': bypass}) in metrics._increments
    age_events = [
        (value, labels)
        for name, value, labels in metrics._observations
        if name == 'cache_url_age_cap'
    ]
    assert age_events == ([(0.0, {'reason': reason})] * 2 if cap is None and reason else [])


@pytest.mark.parametrize(
    ('cap', 'reason', 'expected_after_three_seconds'),
    [
        pytest.param(2.0, 'signed_url', 2, id='shortened-with-reason'),
        pytest.param(2.0, None, 2, id='shortened-without-reason'),
        pytest.param(100.0, 'signed_url', 1, id='capped-by-maximum-age'),
    ],
)
def test_population_decision_valid_cap_preserves_cache_age_and_reason_metric(
    cap: float,
    reason: str | None,
    expected_after_three_seconds: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache, clock, metrics = _cache()
    monkeypatch.setattr(cache_module, 'monotonic', clock)
    calls = 0

    def fetch() -> object:
        nonlocal calls
        calls += 1
        return calls

    def read() -> object:
        return _read(
            cache,
            fetch,
            per_call_age=20.0,
            population_decision=lambda _value: _PopulationDecision(cap, reason),
        )

    assert read() == 1
    assert read() == 1
    clock._advance(3)
    assert read() == expected_after_three_seconds
    clock._advance(18)
    assert read() == (3 if cap == 2.0 else 2)
    age_events = [
        (value, labels)
        for name, value, labels in metrics._observations
        if name == 'cache_url_age_cap'
    ]
    assert age_events == ([(cap, {'reason': reason})] * calls if reason else [])


def test_url_capped_commit_keeps_the_true_origin_age(monkeypatch: pytest.MonkeyPatch) -> None:
    cache, clock, _metrics = _cache()
    monkeypatch.setattr(cache_module, 'monotonic', clock)
    calls = 0

    def fetch() -> object:
        nonlocal calls
        calls += 1
        return [{'call': calls}]

    assert _read(
        cache,
        fetch,
        population_decision=lambda _value: _PopulationDecision(10.0),
    ) == [{'call': 1}]
    clock._advance(5)

    assert _read(cache, fetch, per_call_age=20.0) == [{'call': 1}]
    assert calls == 1


def test_wait_delay_is_clamped_for_unbounded_attempts() -> None:
    assert cache_module._wait_delay(2_000, 1.0) <= 0.125
    assert cache_module._wait_delay(0, 0.001) <= 0.001
