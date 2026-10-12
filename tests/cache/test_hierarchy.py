from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from threading import Barrier, Event, Lock
from time import monotonic, sleep
from typing import Literal, cast

import pytest

import sg_query_cache._cache as cache_module
import sg_query_cache._cache_common as cache_common_module
import sg_query_cache._local as local_module
from sg_query_cache._backend import BackendUnavailable, CacheBackend
from sg_query_cache._cache import QueryCache
from sg_query_cache._codec import CodecBoundError, encode_value
from sg_query_cache._config import CacheSettings, L2Settings, LimitSettings
from sg_query_cache._keys import KeyBuilder
from sg_query_cache._local import LocalBackend
from sg_query_cache._model import (
    DependencyPlan,
    DependencyTag,
    ExactCall,
    FillLease,
    GenerationChange,
    GenerationVector,
    Identity,
)
from tests._fakes import (
    ResolveFailureBackend,
    _BlockingBackend,
    _FakeClock,
    _FencedBackend,
    _ObservedBackend,
    _RecordingMetrics,
)

_CLIENT_OPTIONS = {'localized': False, 'convert_datetimes_to_utc': True}
_SITE = 'production'


class _AdvancingResolveBackend(_ObservedBackend):
    __slots__ = ('_advance', '_clock', '_trigger_at')

    def __init__(
        self,
        backend: CacheBackend,
        clock: _FakeClock,
        *,
        trigger_at: int,
        advance: float,
    ) -> None:
        super().__init__(backend)
        self._advance = advance
        self._clock = clock
        self._trigger_at = trigger_at

    def resolve(self, tags: tuple[DependencyTag, ...]) -> GenerationVector:
        vector = super().resolve(tags)
        if len(self._resolved) == self._trigger_at:
            self._clock._advance(self._advance)
        return vector


class _GenerationBarrierBackend(_ObservedBackend):
    __slots__ = ('_barrier', '_change', '_l2', '_lock', '_resolves')

    def __init__(
        self,
        backend: CacheBackend,
        l2: CacheBackend,
        change: GenerationChange,
        parties: int,
    ) -> None:
        super().__init__(backend)
        self._barrier = Barrier(parties)
        self._change = change
        self._l2 = l2
        self._lock = Lock()
        self._resolves = 0

    def resolve(self, tags: tuple[DependencyTag, ...]) -> GenerationVector:
        with self._lock:
            self._resolves += 1
            attempt = self._resolves
        if attempt <= self._barrier.parties:
            vector = self._backend.resolve(tags)
            self._barrier.wait(timeout=2)
            return vector
        if attempt == self._barrier.parties + 1:
            changes = (self._change,)
            self._backend.change(changes)
            self._l2.change(changes)
        return self._backend.resolve(tags)


class _ReleaseTrackingBackend(_ObservedBackend):
    __slots__ = ('_release_attempts',)

    def __init__(
        self,
        backend: CacheBackend,
        *,
        commit_outcome: Literal['fail', 'reject'],
        replacement_payload: bytes,
    ) -> None:
        super().__init__(
            backend,
            commit_outcome=commit_outcome,
            replacement_payload=replacement_payload,
        )
        self._release_attempts: list[FillLease] = []

    def release(self, lease: FillLease) -> bool:
        self._release_attempts.append(lease)
        return super().release(lease)


class _ChangeHookBackend(_ObservedBackend):
    __slots__ = ('_before_change',)

    def __init__(self, backend: CacheBackend, before_change: Callable[[], None]) -> None:
        super().__init__(backend)
        self._before_change = before_change

    def change(self, changes: tuple[GenerationChange, ...]) -> None:
        self._before_change()
        self._backend.change(changes)


@dataclass(frozen=True, slots=True)
class _PopulationDecision:
    remaining_seconds: float | None


class _ArmedMetrics(_RecordingMetrics):
    __slots__ = ('_advance', '_clock', '_event', '_labels')

    def __init__(self, clock: _FakeClock) -> None:
        super().__init__()
        self._advance = 0.0
        self._clock = clock
        self._event: tuple[Literal['increment', 'observe'], str] | None = None
        self._labels: dict[str, str] = {}

    def _advance_on(
        self,
        method: Literal['increment', 'observe'],
        name: str,
        labels: Mapping[str, str],
    ) -> None:
        if self._event != (method, name) or not self._labels.items() <= labels.items():
            return
        self._event = None
        self._clock._advance(self._advance)

    def _arm(
        self,
        method: Literal['increment', 'observe'],
        name: str,
        labels: Mapping[str, str],
        *,
        advance: float,
    ) -> None:
        self._advance = advance
        self._event = (method, name)
        self._labels = dict(labels)

    def increment(self, name: str, labels: Mapping[str, str]) -> None:
        self._advance_on('increment', name, labels)
        super().increment(name, labels)

    def observe(self, name: str, value: float, labels: Mapping[str, str]) -> None:
        self._advance_on('observe', name, labels)
        super().observe(name, value, labels)


class _SlowMetrics(_RecordingMetrics):
    __slots__ = ('_advance', '_clock', '_seen')

    def __init__(self, clock: _FakeClock, advance: float) -> None:
        super().__init__()
        self._advance = advance
        self._clock = clock
        self._seen = 0

    def observe(self, name: str, value: float, labels: Mapping[str, str]) -> None:
        if name == 'cache_ttl_selection' and labels.get('layer') == 'hierarchical_l1':
            self._seen += 1
            if self._seen == 2:
                self._clock._advance(self._advance)
        super().observe(name, value, labels)


def _backend(settings: CacheSettings, clock: _FakeClock) -> LocalBackend:
    return LocalBackend(settings, clock, _RecordingMetrics())


def _cache(
    settings: CacheSettings,
    l1: CacheBackend,
    l2: CacheBackend,
    *,
    metrics: _RecordingMetrics | None = None,
) -> QueryCache:
    return QueryCache(
        settings,
        Identity('script', 'studio', 'render'),
        KeyBuilder(),
        l1=l1,
        l2=l2,
        metrics=metrics or _RecordingMetrics(),
    )


def _call() -> ExactCall:
    return ExactCall('find', (('entity_type', 'Asset'), ('filters', ())))


def _encoded_result() -> bytes:
    return encode_value(('sg-query-cache-result-v1', 0.0, 5.0, 0.01, [{'id': 7}]))


def _identity_parse(value: object) -> object:
    return value


def _parse(value: object) -> object:
    return ('parsed', value)


def _plan() -> DependencyPlan:
    return DependencyPlan(
        'event_aware',
        (
            DependencyTag.site(_SITE),
            DependencyTag.authorization(_SITE),
            DependencyTag.type_membership(_SITE, 'Asset'),
        ),
        'event_aware',
    )


def _parsed_read(cache: QueryCache, fetch: Callable[[], object]) -> object:
    return cache.read_parsed(_call(), _plan(), fetch, _CLIENT_OPTIONS, _parse)


def _read(
    cache: QueryCache,
    fetch: Callable[[], object],
    population_decision: Callable[[object], _PopulationDecision] | None = None,
) -> object:
    return cache.read(
        _call(),
        _plan(),
        fetch,
        _CLIENT_OPTIONS,
        _population_decision=population_decision,
    )


def _settings() -> CacheSettings:
    return CacheSettings(l2=L2Settings(enabled=True))


def _store_payload(backend: CacheBackend, payload: bytes, age: float) -> None:
    base_key = KeyBuilder.base_key(
        _SITE,
        Identity('script', 'studio', 'render'),
        _CLIENT_OPTIONS,
        _call(),
    )
    vector = backend.resolve(_plan().tags)
    key = KeyBuilder.qualified_key(base_key, vector, 'result')
    lease = backend.reserve(key, 30)
    assert lease is not None
    assert backend.commit(lease, payload, age)


def test_completed_old_key_flight_rejoins_the_current_memo_key() -> None:
    settings = _settings()
    clock = _FakeClock()
    raw_l1 = _backend(settings, clock)
    l2 = _backend(settings, clock)
    plan = _plan()
    base_key = KeyBuilder.base_key(
        _SITE,
        Identity('script', 'studio', 'render'),
        _CLIENT_OPTIONS,
        _call(),
    )
    old_key = KeyBuilder.qualified_key(base_key, raw_l1.resolve(plan.tags), 'result')
    change = GenerationChange(DependencyTag.type_membership(_SITE, 'Asset'))
    l1 = _GenerationBarrierBackend(raw_l1, l2, change, parties=4)
    cache = _cache(settings, l1, l2)
    release = Event()
    oversized = b'x' * (settings.limits.max_serialized_result_bytes + 1)
    calls = 0

    def fetch() -> object:
        nonlocal calls
        calls += 1
        assert release.wait(timeout=2)
        return oversized

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(_parsed_read, cache, fetch) for _ in range(4)]
        deadline = monotonic() + 2
        while monotonic() < deadline:
            flight = cache._flights.get(f'memo:{old_key}')
            if flight is not None and flight._participants == 4:
                break
            sleep(0.005)
        else:
            pytest.fail('four callers did not join the old-key memo flight')
        release.set()
        results = [future.result(timeout=5) for future in futures]

    assert results == [('parsed', oversized)] * 4
    assert calls == 1


def test_hierarchy_parsed_fill_honors_a_later_shorter_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings()
    clock = _FakeClock()
    monkeypatch.setattr(cache_module, 'monotonic', clock)
    cache = _cache(settings, _backend(settings, clock), _backend(settings, clock))
    calls = 0

    def fetch() -> object:
        nonlocal calls
        calls += 1
        return {'call': calls}

    assert _parsed_read(cache, fetch) == ('parsed', {'call': 1})
    assert _parsed_read(cache, fetch) == ('parsed', {'call': 1})
    clock._advance(2)
    with cache.policy(maximum_age=1):
        assert _parsed_read(cache, fetch) == ('parsed', {'call': 2})
    assert calls == 2


def test_host_clock_skew_cannot_extend_l2_age() -> None:
    settings = _settings()
    l1_clock = _FakeClock(start=1_000_000_000)
    l2_clock = _FakeClock(start=10)
    l2 = _backend(settings, l2_clock)
    seed = _cache(settings, _backend(settings, _FakeClock()), l2)
    cache = _cache(settings, _backend(settings, l1_clock), l2)
    calls = 0

    def fetch() -> object:
        nonlocal calls
        calls += 1
        return [{'call': calls}]

    assert _read(seed, fetch) == [{'call': 1}]
    l2_clock._advance(595)
    assert _read(cache, fetch) == [{'call': 1}]
    l1_clock._advance(5)
    l2_clock._advance(5)
    assert _read(cache, fetch) == [{'call': 2}]
    assert calls == 2


def test_interleaved_reader_cannot_repromote_a_pre_write_value() -> None:
    settings = _settings()
    clock = _FakeClock()
    value = ['PRE-WRITE']
    reader_cache: list[QueryCache] = []

    def interleave() -> None:
        if reader_cache:
            _read(reader_cache[0], lambda: list(value))

    l1 = _ChangeHookBackend(_backend(settings, clock), interleave)
    l2 = _ChangeHookBackend(_backend(settings, clock), interleave)
    cache = _cache(settings, l1, l2)
    reader_cache.append(cache)

    assert _read(cache, lambda: list(value)) == ['PRE-WRITE']
    value[0] = 'POST-WRITE'
    assert cache.invalidate((GenerationChange(DependencyTag.type_membership(_SITE, 'Asset')),))

    assert _read(cache, lambda: list(value)) == ['POST-WRITE']


def test_l1_failure_continues_to_a_healthy_l2() -> None:

    class _FailingL1:
        def resolve(self, tags: tuple[DependencyTag, ...]) -> GenerationVector:
            _ = tags
            raise BackendUnavailable('local unavailable')

    settings = _settings()
    l2 = _backend(settings, _FakeClock())
    seed = _cache(settings, _backend(settings, _FakeClock()), l2)
    assert _read(seed, lambda: [{'id': 7}]) == [{'id': 7}]
    cache = _cache(settings, cast('CacheBackend', _FailingL1()), l2)
    calls = 0

    def fetch() -> object:
        nonlocal calls
        calls += 1
        return [{'id': 8}]

    assert _read(cache, fetch) == [{'id': 7}]
    assert calls == 0


def test_l1_hit_avoids_l2() -> None:
    settings = _settings()
    l1 = _backend(settings, _FakeClock())
    observed_l2 = _ObservedBackend(_backend(settings, _FakeClock()))
    cache = _cache(settings, l1, observed_l2)
    calls = 0

    def fetch() -> object:
        nonlocal calls
        calls += 1
        return [{'id': 7}]

    assert _read(cache, fetch) == [{'id': 7}]
    lookups_after_fill = observed_l2._lookup_count
    assert _read(cache, fetch) == [{'id': 7}]

    assert calls == 1
    assert observed_l2._lookup_count == lookups_after_fill


def test_l1_promotion_expires_at_hierarchical_cap() -> None:
    settings = _settings()
    l1_clock = _FakeClock()
    l2_clock = _FakeClock()
    l2 = _ObservedBackend(_backend(settings, l2_clock))
    seed = _cache(settings, _backend(settings, _FakeClock()), l2)
    cache = _cache(settings, _backend(settings, l1_clock), l2)
    calls = 0

    def fetch() -> object:
        nonlocal calls
        calls += 1
        return [{'id': 7}]

    assert _read(seed, fetch) == [{'id': 7}]
    assert _read(cache, fetch) == [{'id': 7}]
    promoted_lookups = l2._lookup_count
    l1_clock._advance(10)
    assert _read(cache, fetch) == [{'id': 7}]

    assert calls == 1
    assert l2._lookup_count == promoted_lookups + 1


def test_l1_promotion_expires_at_shorter_l2_remaining_age() -> None:
    settings = _settings()
    l1_clock = _FakeClock()
    l2_clock = _FakeClock()
    l2 = _backend(settings, l2_clock)
    seed = _cache(settings, _backend(settings, _FakeClock()), l2)
    cache = _cache(settings, _backend(settings, l1_clock), l2)
    calls = 0

    def fetch() -> object:
        nonlocal calls
        calls += 1
        return [{'call': calls}]

    assert _read(seed, fetch) == [{'call': 1}]
    l2_clock._advance(595)
    assert _read(cache, fetch) == [{'call': 1}]
    l1_clock._advance(5)
    l2_clock._advance(5)
    assert _read(cache, fetch) == [{'call': 2}]
    assert calls == 2


def test_l2_hit_decodes_once_before_l1_promotion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings()
    l2 = _backend(settings, _FakeClock())
    seed = _cache(settings, _backend(settings, _FakeClock()), l2)
    assert _read(seed, lambda: {'items': [1]}) == {'items': [1]}
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
    cache = _cache(settings, _backend(settings, _FakeClock()), l2)

    promoted = cast('dict[str, list[int]]', _read(cache, lambda: {'items': [2]}))

    assert promoted == {'items': [1]}
    assert decode_count == 1


def test_l2_hit_expiring_during_decode_fetches_without_promotion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings()
    l2_clock = _FakeClock()
    l2 = _backend(settings, l2_clock)
    seed = _cache(settings, _backend(settings, _FakeClock()), l2)
    assert _read(seed, lambda: [{'id': 7}]) == [{'id': 7}]
    l2_clock._advance(595)
    original_decode = local_module.decode_value

    def delayed_decode(payload: bytes) -> object:
        value = original_decode(payload)
        l2_clock._advance(5)
        return value

    monkeypatch.setattr(local_module, 'decode_value', delayed_decode)
    observed_l2 = _ObservedBackend(l2)
    cache = _cache(settings, _backend(settings, _FakeClock()), observed_l2)
    calls = 0

    def fetch() -> object:
        nonlocal calls
        calls += 1
        return [{'id': 8}]

    assert _read(cache, fetch) == [{'id': 8}]
    assert _read(cache, fetch) == [{'id': 8}]
    assert calls == 1
    assert observed_l2._lookup_count == 2


def test_l2_hit_promotes_an_isolated_value_to_l1() -> None:
    settings = _settings()
    l2 = _backend(settings, _FakeClock())
    first = _cache(settings, _backend(settings, _FakeClock()), l2)
    second = _cache(settings, _backend(settings, _FakeClock()), l2)
    calls = 0

    def fetch() -> object:
        nonlocal calls
        calls += 1
        return [{'id': 7, 'links': [{'id': 3}]}]

    assert _read(first, fetch) == [{'id': 7, 'links': [{'id': 3}]}]
    promoted = _read(second, fetch)
    assert isinstance(promoted, list)
    promoted[0]['links'] = []

    assert _read(second, fetch) == [{'id': 7, 'links': [{'id': 3}]}]
    assert calls == 1


def test_l2_invalidation_leaves_l1_usable_only_until_short_cap() -> None:
    settings = _settings()
    l1_clock = _FakeClock()
    l2 = _backend(settings, _FakeClock())
    cache = _cache(settings, _backend(settings, l1_clock), l2)
    calls = 0

    def fetch() -> object:
        nonlocal calls
        calls += 1
        return [{'call': calls}]

    assert _read(cache, fetch) == [{'call': 1}]
    l2.change((GenerationChange(DependencyTag.type_membership(_SITE, 'Asset')),))
    assert _read(cache, fetch) == [{'call': 1}]
    l1_clock._advance(10)
    assert _read(cache, fetch) == [{'call': 2}]
    assert calls == 2


def test_malformed_l1_records_codec_failure_and_continues_to_l2() -> None:
    settings = _settings()
    l1 = _backend(settings, _FakeClock())
    l2 = _backend(settings, _FakeClock())
    private_payload = 'local-payload-private-detail'
    _store_payload(l1, encode_value((private_payload, {'id': 1})), 10)
    _store_payload(
        l2,
        encode_value(('sg-query-cache-result-v1', 0.0, 600.0, 0.01, [{'id': 7}])),
        600,
    )
    metrics = _RecordingMetrics()
    cache = QueryCache(
        settings,
        Identity('script', 'studio', 'render'),
        KeyBuilder(),
        l1=l1,
        l2=l2,
        metrics=metrics,
    )
    calls = 0

    def fetch() -> object:
        nonlocal calls
        calls += 1
        return [{'hierarchy-response-private-detail': 8}]

    assert _read(cache, fetch) == [{'id': 7}]
    assert calls == 0
    assert ('cache_bypass', {'reason': 'codec_failure'}) in metrics._increments
    assert private_payload not in str(metrics._increments)
    assert 'hierarchy-response-private-detail' not in str(metrics._increments)


def test_malformed_l2_fetches_directly_without_l1_or_l2_population() -> None:
    settings = _settings()
    l1 = _backend(settings, _FakeClock())
    l2 = _backend(settings, _FakeClock())
    private_payload = 'hierarchy-shared-payload-private-detail'
    _store_payload(l2, encode_value((private_payload, {'id': 1})), 600)
    metrics = _RecordingMetrics()
    cache = QueryCache(
        settings,
        Identity('script', 'studio', 'render'),
        KeyBuilder(),
        l1=l1,
        l2=l2,
        metrics=metrics,
    )
    calls = 0

    def fetch() -> object:
        nonlocal calls
        calls += 1
        return [{'hierarchy-shared-response-private-detail': calls}]

    assert _read(cache, fetch) == [{'hierarchy-shared-response-private-detail': 1}]
    assert _read(cache, fetch) == [{'hierarchy-shared-response-private-detail': 2}]
    assert calls == 2
    assert ('cache_bypass', {'reason': 'codec_failure'}) in metrics._increments
    assert private_payload not in str(metrics._increments)
    assert 'hierarchy-shared-response-private-detail' not in str(metrics._increments)


def test_memo_is_keyed_by_parser_identity() -> None:
    settings = _settings()
    clock = _FakeClock()
    cache = _cache(settings, _backend(settings, clock), _backend(settings, clock))

    assert cache.read_parsed(
        _call(),
        _plan(),
        lambda: {'v': 1},
        _CLIENT_OPTIONS,
        _parse,
    ) == ('parsed', {'v': 1})
    assert cache.read_parsed(
        _call(),
        _plan(),
        lambda: {'v': 2},
        _CLIENT_OPTIONS,
        _identity_parse,
    ) == {'v': 1}


def test_memo_is_not_used_while_a_layer_has_unacknowledged_changes() -> None:
    settings = _settings()
    clock = _FakeClock()
    l1 = _backend(settings, clock)
    l2 = _FencedBackend(_backend(settings, clock), failures=1)
    cache = _cache(settings, l1, l2)
    calls = 0

    def fetch() -> object:
        nonlocal calls
        calls += 1
        return {'call': calls}

    assert _parsed_read(cache, fetch) == ('parsed', {'call': 1})
    assert (
        cache.invalidate((GenerationChange(DependencyTag.type_membership(_SITE, 'Asset')),))
        is False
    )
    assert _parsed_read(cache, fetch) == ('parsed', {'call': 2})


def test_parse_failure_refetches_once_and_publishes_the_memo() -> None:
    settings = _settings()
    clock = _FakeClock()
    cache = _cache(settings, _backend(settings, clock), _backend(settings, clock))
    calls = 0

    def fetch() -> object:
        nonlocal calls
        calls += 1
        return b'not-a-view' if calls == 1 else b'good'

    def parse(value: object) -> object:
        if value == b'not-a-view':
            raise ValueError('malformed')
        return ('parsed', value)

    assert cache.read_parsed(
        _call(),
        _plan(),
        fetch,
        _CLIENT_OPTIONS,
        parse,
    ) == ('parsed', b'good')
    assert cache.read_parsed(
        _call(),
        _plan(),
        fetch,
        _CLIENT_OPTIONS,
        parse,
    ) == ('parsed', b'good')
    assert calls == 2


def test_parsed_read_rejects_shared_revalidation_crossing_a_retained_fence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings()
    clock = _FakeClock()
    monkeypatch.setattr(cache_module, 'monotonic', clock)
    l2 = _BlockingBackend(_backend(settings, clock), failures=1)
    cache = _cache(settings, _backend(settings, clock), l2)
    calls = 0

    def fetch() -> object:
        nonlocal calls
        calls += 1
        return {'call': calls}

    assert _parsed_read(cache, fetch) == ('parsed', {'call': 1})
    clock._advance(11)
    l2._arm('resolve')
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(_parsed_read, cache, fetch)
        assert l2._entered.wait(timeout=2)
        acknowledged = cache.invalidate(
            (GenerationChange(DependencyTag.type_membership(_SITE, 'Asset')),)
        )
        l2._release_block.set()
        value = future.result(timeout=2)

    assert acknowledged is False
    assert value == ('parsed', {'call': 2})
    assert calls == 2


def test_parsed_read_uses_l2_after_preliminary_l1_resolution_failure() -> None:
    settings = _settings()
    clock = _FakeClock()
    l2 = _backend(settings, clock)
    _store_payload(l2, _encoded_result(), 5)
    l1 = ResolveFailureBackend(_backend(settings, clock), failures=2)
    cache = _cache(settings, l1, l2)
    calls = 0

    def fetch() -> object:
        nonlocal calls
        calls += 1
        return [{'id': 8}]

    assert _parsed_read(cache, fetch) == ('parsed', [{'id': 7}])
    assert calls == 0


def test_parsed_view_is_reused_and_revalidated_without_lookups(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings()
    clock = _FakeClock()
    monkeypatch.setattr(cache_module, 'monotonic', clock)
    l2 = _ObservedBackend(_backend(settings, clock))
    cache = _cache(settings, _backend(settings, clock), l2)
    calls = 0

    def fetch() -> object:
        nonlocal calls
        calls += 1
        return {'call': calls}

    assert _parsed_read(cache, fetch) == ('parsed', {'call': 1})
    lookups = l2._lookup_count
    assert _parsed_read(cache, fetch) == ('parsed', {'call': 1})
    assert l2._lookup_count == lookups
    clock._advance(11)
    assert _parsed_read(cache, fetch) == ('parsed', {'call': 1})
    assert l2._lookup_count == lookups
    assert calls == 1

    l2.change((GenerationChange(DependencyTag.type_membership(_SITE, 'Asset')),))
    clock._advance(11)
    assert _parsed_read(cache, fetch) == ('parsed', {'call': 2})
    assert calls == 2


def test_promoted_hit_keeps_its_origin_age_for_a_shorter_caller_age(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings()
    clock = _FakeClock()
    monkeypatch.setattr(cache_module, 'monotonic', clock)
    l2 = _backend(settings, clock)
    cache = _cache(settings, _backend(settings, clock), l2)
    calls = 0

    def fetch() -> object:
        nonlocal calls
        calls += 1
        return [{'call': calls}]

    assert _read(cache, fetch) == [{'call': 1}]
    clock._advance(11)
    assert _read(cache, fetch) == [{'call': 1}]
    clock._advance(1)

    assert cache.read(_call(), _plan(), fetch, _CLIENT_OPTIONS, per_call_age=2.0) == [{'call': 2}]
    assert calls == 2


def test_promotion_bound_failure_reports_its_own_bypass_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings()
    clock = _FakeClock()
    l2 = _backend(settings, clock)
    l1 = _backend(settings, clock)
    _store_payload(l2, _encoded_result(), 5)
    metrics = _RecordingMetrics()
    cache = QueryCache(
        settings,
        Identity('script', 'studio', 'render'),
        KeyBuilder(),
        l1=l1,
        l2=l2,
        metrics=metrics,
    )

    def reject_bound(_timed: object, _origin_elapsed: float, _lifetime: float) -> bytes:
        raise CodecBoundError('depth')

    monkeypatch.setattr(cache_module, 'encode_result', reject_bound)

    assert _read(cache, lambda: [{'id': 8}]) == [{'id': 7}]
    assert ('cache_bypass', {'reason': 'result_too_deep'}) in metrics._increments
    assert ('cache_bypass', {'reason': 'codec_failure'}) not in metrics._increments


def test_promotion_deadline_is_anchored_before_metrics_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings()
    clock = _FakeClock()
    monkeypatch.setattr(cache_module, 'monotonic', clock)
    l2 = _backend(settings, clock)
    l1 = _backend(settings, clock)
    _store_payload(l2, _encoded_result(), 5)
    metrics = _SlowMetrics(clock, 3)
    cache = QueryCache(
        settings,
        Identity('script', 'studio', 'render'),
        KeyBuilder(),
        l1=l1,
        l2=l2,
        metrics=metrics,
    )

    assert _read(cache, lambda: [{'id': 8}]) == [{'id': 7}]
    clock._advance(4)

    assert _read(cache, lambda: [{'id': 8}]) == [{'id': 8}]


def test_promotion_over_configured_byte_limit_records_size_reason() -> None:
    settings = CacheSettings(
        l2=L2Settings(enabled=True),
        limits=LimitSettings(max_serialized_result_bytes=100),
    )
    clock = _FakeClock()
    l2 = _backend(_settings(), clock)
    l1 = _backend(settings, clock)
    _store_payload(l2, _encoded_result(), 5)
    metrics = _RecordingMetrics()
    cache = QueryCache(
        settings,
        Identity('script', 'studio', 'render'),
        KeyBuilder(),
        l1=l1,
        l2=l2,
        metrics=metrics,
    )

    assert _read(cache, lambda: [{'id': 8}]) == [{'id': 7}]
    assert ('cache_bypass', {'reason': 'result_too_large_bytes'}) in metrics._increments
    assert ('cache_bypass', {'reason': 'codec_failure'}) not in metrics._increments


def test_promotion_uses_original_l1_vector() -> None:
    settings = _settings()
    l2 = _backend(settings, _FakeClock())
    seed = _cache(settings, _backend(settings, _FakeClock()), l2)
    assert _read(seed, lambda: [{'id': 7}]) == [{'id': 7}]

    l1 = _backend(settings, _FakeClock())
    change = (GenerationChange(DependencyTag.type_membership(_SITE, 'Asset')),)
    observed_l2 = _ObservedBackend(l2, on_lookup=lambda: l1.change(change))
    cache = _cache(settings, l1, observed_l2)
    calls = 0

    def fetch() -> object:
        nonlocal calls
        calls += 1
        return [{'id': 8}]

    assert _read(cache, fetch) == [{'id': 7}]
    assert _read(cache, fetch) == [{'id': 7}]

    assert calls == 0
    assert observed_l2._lookup_count == 2


def test_revalidation_anchors_first_deadline_before_metrics_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings()
    clock = _FakeClock()
    monkeypatch.setattr(cache_module, 'monotonic', clock)
    l2 = _ObservedBackend(_backend(settings, clock))
    metrics = _ArmedMetrics(clock)
    cache = QueryCache(
        settings,
        Identity('script', 'studio', 'render'),
        KeyBuilder(),
        l1=_backend(settings, clock),
        l2=l2,
        metrics=metrics,
    )
    calls = 0

    def fetch() -> object:
        nonlocal calls
        calls += 1
        return {'call': calls}

    assert _parsed_read(cache, fetch) == ('parsed', {'call': 1})
    clock._advance(20)
    metrics._arm(
        'observe',
        'cache_ttl_selection',
        {'layer': 'hierarchical_l1'},
        advance=3,
    )
    assert _parsed_read(cache, fetch) == ('parsed', {'call': 1})
    resolves_after_revalidation = len(l2._resolved)

    clock._advance(6.9)
    assert _parsed_read(cache, fetch) == ('parsed', {'call': 1})
    assert len(l2._resolved) == resolves_after_revalidation

    clock._advance(0.2)
    assert _parsed_read(cache, fetch) == ('parsed', {'call': 1})
    assert len(l2._resolved) == resolves_after_revalidation + 1
    assert calls == 1


@pytest.mark.parametrize(
    ('age_before_resolve', 'maximum_age'),
    [(599.9, None), (19.9, 20.0)],
    ids=['deep-deadline', 'lexical-origin-age'],
)
def test_revalidation_rechecks_age_after_shared_resolution(
    age_before_resolve: float,
    maximum_age: float | None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings()
    clock = _FakeClock()
    monkeypatch.setattr(cache_module, 'monotonic', clock)
    l2 = _AdvancingResolveBackend(
        _backend(settings, clock),
        clock,
        trigger_at=3,
        advance=0.2,
    )
    cache = _cache(settings, _backend(settings, clock), l2)
    calls = 0

    def fetch() -> object:
        nonlocal calls
        calls += 1
        return {'call': calls}

    assert _parsed_read(cache, fetch) == ('parsed', {'call': 1})
    clock._advance(age_before_resolve)

    if maximum_age is None:
        result = _parsed_read(cache, fetch)
    else:
        with cache.policy(maximum_age=maximum_age):
            result = _parsed_read(cache, fetch)

    assert result == ('parsed', {'call': 2})
    assert calls == 2


def test_shared_fill_deadline_includes_commit_and_release_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings()
    local_clock = _FakeClock()
    monkeypatch.setattr(cache_module, 'monotonic', local_clock)
    observed_l2 = _ObservedBackend(
        _backend(settings, _FakeClock()),
        on_commit=lambda: local_clock._advance(6),
    )
    cache = _cache(settings, _backend(settings, _FakeClock()), observed_l2)
    calls = 0

    def fetch() -> object:
        nonlocal calls
        calls += 1
        return [{'id': 7}]

    assert cache.read(_call(), _plan(), fetch, _CLIENT_OPTIONS, per_call_age=5) == [{'id': 7}]
    lookups_after_fill = observed_l2._lookup_count
    assert cache.read(_call(), _plan(), fetch, _CLIENT_OPTIONS, per_call_age=5) == [{'id': 7}]
    assert calls == 1
    assert observed_l2._lookup_count == lookups_after_fill + 1


def test_slow_reuse_telemetry_does_not_move_the_first_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings()
    clock = _FakeClock()
    monkeypatch.setattr(cache_module, 'monotonic', clock)
    l2 = _ObservedBackend(_backend(settings, clock))
    metrics = _ArmedMetrics(clock)
    cache = QueryCache(
        settings,
        Identity('script', 'studio', 'render'),
        KeyBuilder(),
        l1=_backend(settings, clock),
        l2=l2,
        metrics=metrics,
    )
    calls = 0

    def fetch() -> object:
        nonlocal calls
        calls += 1
        return {'call': calls}

    assert _parsed_read(cache, fetch) == ('parsed', {'call': 1})
    resolves_after_acquisition = len(l2._resolved)
    clock._advance(9.9)
    metrics._arm(
        'increment',
        'cache_schema_view',
        {'outcome': 'reuse'},
        advance=0.2,
    )

    assert _parsed_read(cache, fetch) == ('parsed', {'call': 1})
    assert len(l2._resolved) == resolves_after_acquisition
    assert _parsed_read(cache, fetch) == ('parsed', {'call': 1})
    assert len(l2._resolved) == resolves_after_acquisition + 1
    assert calls == 1


@pytest.mark.parametrize(
    ('age_before_revalidation', 'maximum_age'),
    [(599.9, None), (19.9, 20.0)],
    ids=['deep-deadline', 'lexical-origin-age'],
)
def test_slow_telemetry_after_a_fresh_decision_changes_only_the_next_call(
    age_before_revalidation: float,
    maximum_age: float | None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings()
    clock = _FakeClock()
    monkeypatch.setattr(cache_module, 'monotonic', clock)
    metrics = _ArmedMetrics(clock)
    cache = QueryCache(
        settings,
        Identity('script', 'studio', 'render'),
        KeyBuilder(),
        l1=_backend(settings, clock),
        l2=_backend(settings, clock),
        metrics=metrics,
    )
    calls = 0

    def fetch() -> object:
        nonlocal calls
        calls += 1
        return {'call': calls}

    assert _parsed_read(cache, fetch) == ('parsed', {'call': 1})
    clock._advance(age_before_revalidation)
    metrics._arm(
        'increment',
        'cache_schema_view',
        {'outcome': 'revalidated'},
        advance=0.2,
    )

    if maximum_age is None:
        first = _parsed_read(cache, fetch)
        second = _parsed_read(cache, fetch)
    else:
        with cache.policy(maximum_age=maximum_age):
            first = _parsed_read(cache, fetch)
            second = _parsed_read(cache, fetch)

    assert first == ('parsed', {'call': 1})
    assert second == ('parsed', {'call': 2})
    assert calls == 2


@pytest.mark.parametrize('commit_outcome', ['fail', 'reject'])
def test_shared_commit_must_succeed_before_l1_promotion(
    commit_outcome: Literal['fail', 'reject'],
) -> None:
    settings = _settings()
    replacement_payload = encode_value(
        ('sg-query-cache-result-v1', 0.0, 600.0, 0.0, [{'version': 'new'}])
    )
    observed_l2 = _ReleaseTrackingBackend(
        _backend(settings, _FakeClock()),
        commit_outcome=commit_outcome,
        replacement_payload=replacement_payload,
    )
    metrics = _RecordingMetrics()
    cache = _cache(settings, _backend(settings, _FakeClock()), observed_l2, metrics=metrics)
    calls = 0

    def fetch() -> object:
        nonlocal calls
        calls += 1
        return [{'version': 'rejected'}]

    decisions = 0

    def decide(_value: object) -> _PopulationDecision:
        nonlocal decisions
        decisions += 1
        return _PopulationDecision(5)

    assert _read(cache, fetch, decide) == [{'version': 'rejected'}]
    if commit_outcome == 'fail':
        assert ('cache_bypass', {'reason': 'backend_failure'}) in metrics._increments
        assert len(observed_l2._release_attempts) == 1
    assert _read(cache, fetch) == [{'version': 'new'}]
    assert calls == 1
    assert decisions == 1


@pytest.mark.parametrize('storable', [True, False])
def test_concurrent_parsed_reads_share_one_fetch(storable: bool) -> None:
    settings = _settings()
    clock = _FakeClock()
    l1 = _backend(settings, clock)
    cache = _cache(settings, l1, _backend(settings, clock))
    fetch_release = Event()
    parse_entered = Event()
    parse_release = Event()
    counter_lock = Lock()
    value = {'call': 1} if storable else b'x' * (settings.limits.max_serialized_result_bytes + 1)
    calls = 0
    parses = 0

    def fetch() -> object:
        nonlocal calls
        with counter_lock:
            calls += 1
        assert fetch_release.wait(timeout=2)
        return value

    def parse(raw: object) -> object:
        nonlocal parses
        with counter_lock:
            parses += 1
        parse_entered.set()
        assert parse_release.wait(timeout=2)
        return ('parsed', raw)

    base_key = KeyBuilder.base_key(
        _SITE,
        Identity('script', 'studio', 'render'),
        _CLIENT_OPTIONS,
        _call(),
    )
    flight_key = 'memo:' + KeyBuilder.qualified_key(
        base_key,
        l1.resolve(_plan().tags),
        'result',
    )

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [
            pool.submit(cache.read_parsed, _call(), _plan(), fetch, _CLIENT_OPTIONS, parse)
            for _ in range(4)
        ]
        try:
            deadline = monotonic() + 2
            while monotonic() < deadline:
                flight = cache._flights.get(flight_key)
                if flight is not None and flight._participants == 4:
                    break
                sleep(0.005)
            else:
                pytest.fail('four callers did not join the memo flight')
            fetch_release.set()
            assert parse_entered.wait(timeout=2)
            assert calls == 1
            assert parses == 1
            assert not any(future.done() for future in futures)
        finally:
            fetch_release.set()
            parse_release.set()
        results = [future.result(timeout=5) for future in futures]

    assert results == [('parsed', value)] * 4
    assert calls == 1
    assert parses == 1


def test_stored_malformed_view_refetches_once_and_records_all_view_outcomes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings()
    clock = _FakeClock()
    monkeypatch.setattr(cache_module, 'monotonic', clock)
    metrics = _RecordingMetrics()
    identity = Identity('script', 'studio', 'render')
    hierarchy = QueryCache(
        settings,
        identity,
        KeyBuilder(),
        l1=_backend(settings, clock),
        l2=_backend(settings, clock),
        metrics=metrics,
    )

    assert _parsed_read(hierarchy, lambda: {'schema': 1}) == ('parsed', {'schema': 1})
    assert _parsed_read(hierarchy, lambda: {'schema': 2}) == ('parsed', {'schema': 1})
    clock._advance(11)
    assert _parsed_read(hierarchy, lambda: {'schema': 3}) == ('parsed', {'schema': 1})

    local_settings = CacheSettings()
    malformed_backend = _backend(local_settings, clock)
    _store_payload(
        malformed_backend,
        encode_value(('sg-query-cache-result-v1', 0.0, 30.0, 0.0, b'not-a-view')),
        30,
    )
    clock._advance(8)
    local = QueryCache(
        local_settings,
        identity,
        KeyBuilder(),
        l1=malformed_backend,
        metrics=metrics,
    )
    calls = 0

    def fetch() -> object:
        nonlocal calls
        calls += 1
        return b'good'

    def parse(value: object) -> object:
        if value == b'not-a-view':
            raise ValueError('malformed')
        return ('parsed', value)

    assert local.read_parsed(_call(), _plan(), fetch, _CLIENT_OPTIONS, parse) == (
        'parsed',
        b'good',
    )
    with local.policy(maximum_age=5):
        assert local.read_parsed(_call(), _plan(), fetch, _CLIENT_OPTIONS, parse) == (
            'parsed',
            b'good',
        )
    assert calls == 1
    assert {
        labels['outcome'] for name, labels in metrics._increments if name == 'cache_schema_view'
    } == {'reuse', 'revalidated', 'rebuild', 'malformed', 'refetch'}


def test_ttl_only_scope_cannot_downgrade_a_schema_plan() -> None:
    settings = _settings()
    clock = _FakeClock()
    l2 = _ObservedBackend(_backend(settings, clock))
    cache = _cache(settings, _backend(settings, clock), l2)
    schema_plan = DependencyPlan(
        'schema',
        (
            DependencyTag.site(_SITE),
            DependencyTag.schema(_SITE),
            DependencyTag.authorization(_SITE),
        ),
        'schema_read',
    )

    with cache.policy(mode='ttl_only'):
        cache.read_parsed(
            _call(),
            schema_plan,
            lambda: {'schema': 1},
            _CLIENT_OPTIONS,
            _identity_parse,
        )

    assert DependencyTag.schema(_SITE) in l2._resolved[-1]
    with cache.bypass():
        assert cache.read_parsed(
            _call(),
            schema_plan,
            lambda: {'schema': 2},
            _CLIENT_OPTIONS,
            _identity_parse,
        ) == {'schema': 2}


def test_uncommitted_parsed_view_is_still_memoized(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings()
    clock = _FakeClock()
    monkeypatch.setattr(cache_module, 'monotonic', clock)
    cache = _cache(settings, _backend(settings, clock), _backend(settings, clock))
    calls = 0
    oversized = b'x' * (settings.limits.max_serialized_result_bytes + 1)

    def fetch() -> object:
        nonlocal calls
        calls += 1
        return oversized

    assert _parsed_read(cache, fetch) == ('parsed', oversized)
    assert _parsed_read(cache, fetch) == ('parsed', oversized)
    clock._advance(11)
    assert _parsed_read(cache, fetch) == ('parsed', oversized)
    clock._advance(1_800)
    assert _parsed_read(cache, fetch) == ('parsed', oversized)
    assert calls == 2


def test_wrapper_invalidation_immediately_redirects_l1() -> None:
    settings = _settings()
    cache = _cache(
        settings,
        _backend(settings, _FakeClock()),
        _backend(settings, _FakeClock()),
    )
    calls = 0

    def fetch() -> object:
        nonlocal calls
        calls += 1
        return [{'call': calls}]

    assert _read(cache, fetch) == [{'call': 1}]
    cache.invalidate((GenerationChange(DependencyTag.type_membership(_SITE, 'Asset')),))
    assert _read(cache, fetch) == [{'call': 2}]
    assert calls == 2
