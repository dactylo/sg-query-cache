import asyncio
from dataclasses import dataclass, replace
from inspect import signature
from time import monotonic

import pytest

import sg_query_cache
import sg_query_cache._async_cache as cache_module
import sg_query_cache._cache_common as common_module
from sg_query_cache._async_cache import AsyncQueryCache
from sg_query_cache._async_local import AsyncLocalBackend
from sg_query_cache._backend import BackendUnavailable
from sg_query_cache._cache import QueryCache
from sg_query_cache._config import CacheSettings, L1Settings, L2Settings
from sg_query_cache._keys import KeyBuilder
from sg_query_cache._model import (
    CacheHit,
    DependencyPlan,
    DependencyTag,
    ExactCall,
    FillLease,
    GenerationChange,
    Identity,
)
from tests._async_fakes import _Phase, _PhaseBackend, _RecordingMetrics
from tests.cache._async_cases import (
    _CALL,
    _IDENTITY,
    _OPTIONS,
    _PLAN,
    _answer,
)
from tests.cache._async_cases import (
    fixture as _fixture,
)
from tests.cache._async_cases import (
    parse as _parse,
)
from tests.cache._async_cases import (
    read as _read,
)
from tests.cache._async_cases import (
    turn as _turn,
)


def _assert_distributed_wait_metrics(metrics: _RecordingMetrics, outcome: str) -> None:
    events = metrics._events
    assert events.count(('cache_collapse', None, {'outcome': 'distributed'})) == 1
    waits = [event for event in events if event[0] == 'cache_wait_seconds']
    assert len(waits) == 1
    assert waits[0][2] == {'outcome': outcome}
    assert waits[0][1] is not None
    assert waits[0][1] >= 0
    if outcome == 'takeover':
        assert waits[0][1] < 0.2
    assert events.count(('cache_lease_timeout', None, {'outcome': 'distributed'})) == (
        outcome == 'timeout'
    )
    if outcome == 'failure':
        assert ('cache_bypass', None, {'reason': 'backend_failure'}) in events


async def _finish_distributed_poll(
    backend: _PhaseBackend, held: FillLease, outcome: str, waiting: asyncio.Event
) -> None:
    if outcome == 'cancelled':
        waiting.set()
        await asyncio.Event().wait()
    elif outcome == 'failure':
        failure = _Phase()
        failure._error = BackendUnavailable('poll failed')
        failure._proceed.set()
        backend._before['lookup'] = failure
    else:
        assert await backend.release(held)
        if outcome == 'completed':
            lease = await backend.reserve(held.key, 30)
            assert lease is not None
            payload = common_module.encode_result(common_module.TimedResult(0, 84), 0, 30)
            assert await backend.commit(lease, payload, 30)


def test_builtin_hit_does_not_decode_twice(monkeypatch: pytest.MonkeyPatch) -> None:
    async def scenario() -> None:
        fixture = _fixture()
        assert await _read(fixture._cache) == 42

        def forbidden(_payload: bytes) -> object:
            raise AssertionError('core decoded an already decoded hit')

        monkeypatch.setattr(common_module, 'decode_value', forbidden)
        assert await _read(fixture._cache) == 42
        await fixture._cache.close()

    asyncio.run(scenario())


def test_child_policy_snapshot_and_independent_instances() -> None:
    async def scenario() -> None:
        first = _fixture()
        second = _fixture()
        gate = asyncio.Event()

        async def child() -> tuple[bool, bool]:
            await gate.wait()
            return first._cache.bypass_active, second._cache.bypass_active

        with first._cache.bypass():
            task = asyncio.create_task(child())
        await _turn()
        assert not first._cache.bypass_active
        gate.set()
        assert await task == (True, False)
        await first._cache.close()
        await second._cache.close()

    asyncio.run(scenario())


def test_constructor_matches_sync_and_rejects_each_synchronous_method() -> None:
    cache_type = sg_query_cache.AsyncQueryCache
    expected = signature(QueryCache).parameters
    actual = signature(cache_type).parameters
    assert [(p.name, p.kind, p.default) for p in actual.values()] == [
        (p.name, p.kind, p.default) for p in expected.values()
    ]
    settings = CacheSettings()
    for name in (
        'change',
        'close',
        'commit',
        'flush_site',
        'lookup',
        'release',
        'reserve',
        'resolve',
    ):
        backend = _PhaseBackend(AsyncLocalBackend(settings, monotonic, _RecordingMetrics()))
        setattr(backend, name, lambda: None)
        with pytest.raises(ValueError, match=name):
            cache_type(settings, _IDENTITY, l1=backend)


def test_custom_two_field_hit_decodes_once(monkeypatch: pytest.MonkeyPatch) -> None:
    async def scenario() -> None:
        fixture = _fixture()
        backend = fixture._l1
        assert backend is not None
        assert await _read(fixture._cache) == 42
        original = backend.lookup

        async def without_carrier(key: str) -> CacheHit | None:
            hit = await original(key)
            return None if hit is None else replace(hit, decoded=None)

        backend.lookup = without_carrier
        decoder = common_module.decode_value
        decodes = 0

        def counted(payload: bytes) -> object:
            nonlocal decodes
            decodes += 1
            return decoder(payload)

        monkeypatch.setattr(common_module, 'decode_value', counted)
        assert await _read(fixture._cache) == 42
        assert decodes == 1
        await fixture._cache.close()

    asyncio.run(scenario())


@pytest.mark.parametrize('topology', ['local', 'shared', 'hierarchy'])
@pytest.mark.parametrize('outcome', ['completed', 'takeover', 'timeout', 'failure', 'cancelled'])
def test_distributed_wait_counts_once_and_records_outcome(
    topology: str, outcome: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def scenario() -> None:
        settings = CacheSettings(
            l1=L1Settings(enabled=topology != 'shared'),
            l2=L2Settings(enabled=topology != 'local'),
        )
        if outcome == 'timeout':
            settings = replace(settings, limits=replace(settings.limits, lease_wait_seconds=0.05))
        fixture = _fixture(topology, settings)
        backend = fixture._l1 if topology == 'local' else fixture._l2
        assert backend is not None
        vector = await backend.resolve(_PLAN.tags)
        base = KeyBuilder.base_key('production', _IDENTITY, _OPTIONS, _CALL)
        key = KeyBuilder.qualified_key(base, vector, 'result')
        held = await backend.reserve(key, 30)
        assert held is not None
        original = asyncio.sleep
        polls = 0
        waiting = asyncio.Event()

        async def poll(delay: float) -> None:
            nonlocal polls
            if delay <= 0 or outcome == 'timeout':
                await original(delay)
                return
            polls += 1
            if polls != 3:
                await original(0)
                return
            await _finish_distributed_poll(backend, held, outcome, waiting)

        async def fetch() -> object:
            if outcome == 'takeover':
                await original(0.2)
            return 42

        monkeypatch.setattr(cache_module, 'sleep', poll)
        task = asyncio.create_task(fixture._cache.read(_CALL, _PLAN, fetch, _OPTIONS))
        if outcome == 'cancelled':
            await asyncio.wait_for(waiting.wait(), timeout=2)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            assert await asyncio.wait_for(task, timeout=2) == (
                84 if outcome == 'completed' else 42
            )
        await fixture._cache.close()
        _assert_distributed_wait_metrics(fixture._metrics, outcome)

    asyncio.run(scenario())


def test_distributed_wait_expiry_does_not_reserve_again_or_populate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        settings = CacheSettings(l1=L1Settings(enabled=False), l2=L2Settings(enabled=True))
        settings = replace(settings, limits=replace(settings.limits, lease_wait_seconds=0.001))
        fixture = _fixture('shared', settings)
        backend = fixture._l2
        assert backend is not None
        vector = await backend.resolve(_PLAN.tags)
        base = KeyBuilder.base_key('production', _IDENTITY, _OPTIONS, _CALL)
        key = KeyBuilder.qualified_key(base, vector, 'result')
        assert await backend.reserve(key, 30) is not None
        original = asyncio.sleep

        async def sleep_until_lease_expires(delay: float) -> None:
            await original(delay)
            if delay > 0:
                fixture._clock._now += 31

        monkeypatch.setattr(cache_module, 'sleep', sleep_until_lease_expires)
        assert await _read(fixture._cache) == 42
        assert backend._calls.count('reserve') <= 2
        assert 'commit' not in backend._calls
        await fixture._cache.close()

    asyncio.run(scenario())


def test_encoding_time_debits_origin_deadline(monkeypatch: pytest.MonkeyPatch) -> None:
    async def scenario() -> None:
        fixture = _fixture()
        monkeypatch.setattr(cache_module, 'monotonic', fixture._clock)
        encoder = cache_module.encode_result

        def slow_encode(
            timed: common_module.TimedResult, origin_elapsed: float, lifetime: float
        ) -> bytes:
            payload = encoder(timed, origin_elapsed, lifetime)
            fixture._clock._now += 3
            return payload

        monkeypatch.setattr(cache_module, 'encode_result', slow_encode)
        assert await fixture._cache.read(_CALL, _PLAN, _answer, _OPTIONS, 2) == 42
        assert fixture._l1 is not None
        assert 'commit' not in fixture._l1._calls
        assert 'release' in fixture._l1._calls
        await fixture._cache.close()

    asyncio.run(scenario())


@pytest.mark.parametrize('topology', ['local', 'shared', 'hierarchy'])
def test_exact_call_identity_and_options_separate_keys(topology: str) -> None:
    async def scenario() -> None:
        fixture = _fixture(topology)
        calls = 0

        async def fetch() -> object:
            nonlocal calls
            calls += 1
            return calls

        cache = fixture._cache
        assert await cache.read(_CALL, _PLAN, fetch, _OPTIONS) == 1
        changed = ExactCall('find', (('entity_type', 'Shot'), ('filters', ())))
        assert await cache.read(changed, _PLAN, fetch, _OPTIONS) == 2
        assert await cache.read(_CALL, _PLAN, fetch, {**_OPTIONS, 'localized': True}) == 3
        assert await cache.read(_CALL, _PLAN, fetch, _OPTIONS) == 1
        backend = fixture._l1 or fixture._l2
        assert backend is not None
        base = KeyBuilder.base_key('production', _IDENTITY, _OPTIONS, _CALL)
        vector = await backend.resolve(_PLAN.tags)
        hit = await backend.lookup(KeyBuilder.qualified_key(base, vector, 'result'))
        assert hit is not None
        assert common_module.decode_hit(hit, 30).value == 1
        other = AsyncQueryCache(
            cache_module.client_binding(cache)[0],
            Identity('script', 'tests', 'other'),
            l1=fixture._l1,
            l2=fixture._l2,
        )
        assert await other.read(_CALL, _PLAN, fetch, _OPTIONS) == 4
        await cache.close()

    asyncio.run(scenario())


@pytest.mark.parametrize('topology', ['local', 'shared', 'hierarchy'])
@pytest.mark.parametrize('operation', ['resolve', 'lookup', 'reserve', 'commit'])
def test_backend_failure_preserves_direct_success(topology: str, operation: str) -> None:
    async def scenario() -> None:
        fixture = _fixture(topology)
        backend = fixture._l2 or fixture._l1
        assert backend is not None
        phase = _Phase()
        phase._proceed.set()
        phase._error = BackendUnavailable('offline')
        backend._before[operation] = phase
        calls = 0

        async def fetch() -> object:
            nonlocal calls
            calls += 1
            return 42

        assert await fixture._cache.read(_CALL, _PLAN, fetch, _OPTIONS) == 42
        assert calls == 1
        assert ('cache_bypass', None, {'reason': 'backend_failure'}) in fixture._metrics._events
        await fixture._cache.close()

    asyncio.run(scenario())


def test_memo_decision_sample_precedes_telemetry(monkeypatch: pytest.MonkeyPatch) -> None:
    async def scenario() -> None:
        fixture = _fixture()
        monkeypatch.setattr(cache_module, 'monotonic', fixture._clock)
        cache = fixture._cache
        calls = 0

        async def fetch() -> object:
            nonlocal calls
            calls += 1
            return calls

        first = await cache.read_parsed(_CALL, _PLAN, fetch, _OPTIONS, _parse)
        fixture._clock._now += 29

        def telemetry() -> None:
            name, _value, labels = fixture._metrics._events[-1]
            if name == 'cache_schema_view' and labels['outcome'] == 'reuse':
                fixture._clock._now += 2

        fixture._metrics._callback = telemetry
        assert await cache.read_parsed(_CALL, _PLAN, fetch, _OPTIONS, _parse) is first
        assert await cache.read_parsed(_CALL, _PLAN, fetch, _OPTIONS, _parse) == ('parsed', 2)
        await cache.close()

    asyncio.run(scenario())


@pytest.mark.parametrize('topology', ['local', 'shared', 'hierarchy'])
@pytest.mark.parametrize('stored', [True, False])
def test_memo_first_and_deep_deadlines(
    topology: str, stored: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def scenario() -> None:
        fixture = _fixture(topology)
        monkeypatch.setattr(cache_module, 'monotonic', fixture._clock)
        if not stored:
            backend = fixture._l2 or fixture._l1
            assert backend is not None
            phase = _Phase()
            phase._error = BackendUnavailable('commit lost')
            phase._proceed.set()
            backend._before['commit'] = phase
        plan = replace(_PLAN, classification='schema')
        calls = 0

        async def fetch() -> object:
            nonlocal calls
            calls += 1
            return calls

        async def parsed() -> object:
            return await fixture._cache.read_parsed(_CALL, plan, fetch, _OPTIONS, _parse)

        first = await parsed()
        assert await parsed() is first
        fixture._clock._now += 11
        if topology == 'hierarchy':
            assert fixture._l2 is not None
            before = fixture._l2._calls.count('resolve')
            assert await parsed() is first
            assert fixture._l2._calls.count('resolve') == before + 1
        else:
            assert await parsed() is first
        fixture._clock._now = 2801
        assert await parsed() != first
        assert calls == 2
        await fixture._cache.close()

    asyncio.run(scenario())


def test_memo_key_change_discards_previous_entry_even_if_new_acquisition_fails() -> None:
    async def scenario() -> None:
        fixture = _fixture()
        parses = 0

        def parse(value: object) -> object:
            nonlocal parses
            parses += 1
            return ('parsed', value)

        async def fail() -> object:
            raise LookupError('schema unavailable')

        cache = fixture._cache
        assert await cache.read_parsed(_CALL, _PLAN, _answer, _OPTIONS, parse) == ('parsed', 42)
        changed = ExactCall('schema_read', ())
        with pytest.raises(LookupError, match='schema unavailable'):
            await cache.read_parsed(changed, _PLAN, fail, _OPTIONS, parse)
        assert await cache.read_parsed(_CALL, _PLAN, _answer, _OPTIONS, parse) == ('parsed', 42)
        assert parses == 2
        await cache.close()

    asyncio.run(scenario())


def test_memo_local_hit_never_acquires_shared_key(monkeypatch: pytest.MonkeyPatch) -> None:
    async def scenario() -> None:
        fixture = _fixture('hierarchy')
        monkeypatch.setattr(cache_module, 'monotonic', fixture._clock)
        cache = fixture._cache
        plan = replace(_PLAN, classification='schema')
        assert await cache.read(_CALL, plan, _answer, _OPTIONS) == 42
        assert fixture._l2 is not None
        before = fixture._l2._calls.count('resolve')
        first = await cache.read_parsed(_CALL, plan, _answer, _OPTIONS, _parse)
        assert fixture._l2._calls.count('resolve') == before
        assert cache._memo is not None
        assert cache._memo._outcome._shared is None
        assert cache._memo._outcome._first_deadline == cache._memo._outcome._deep_deadline
        fixture._clock._now += 11
        assert await cache.read_parsed(_CALL, plan, _answer, _OPTIONS, _parse) == first
        assert fixture._l2._calls.count('lookup') > 1
        await cache.close()

    asyncio.run(scenario())


@pytest.mark.parametrize('topology', ['local', 'shared', 'hierarchy'])
def test_memo_parser_identity_malformed_refetch_and_generation(topology: str) -> None:
    async def scenario() -> None:
        fixture = _fixture(topology)
        values = iter([b'bad', b'good', b'new'])
        calls = 0

        async def fetch() -> object:
            nonlocal calls
            calls += 1
            return next(values)

        def parse(value: object) -> object:
            if value == b'bad':
                raise ValueError('invalid reduced view')
            return ('view', value)

        cache = fixture._cache
        first = await cache.read_parsed(_CALL, _PLAN, fetch, _OPTIONS, parse)
        assert first == ('view', b'good')
        assert await cache.read_parsed(_CALL, _PLAN, fetch, _OPTIONS, parse) is first
        assert calls == 2
        other = await cache.read_parsed(_CALL, _PLAN, fetch, _OPTIONS, _parse)
        assert other == ('parsed', b'bad')
        assert calls == 2
        assert await cache.invalidate((GenerationChange(_PLAN.tags[0]),))
        assert await cache.read_parsed(_CALL, _PLAN, fetch, _OPTIONS, parse) == ('view', b'new')
        assert calls == 3
        await cache.close()

    asyncio.run(scenario())


@pytest.mark.parametrize('topology', ['local', 'shared', 'hierarchy'])
@pytest.mark.parametrize('operation', ['resolve', 'lookup'])
def test_failed_memo_acquisition_never_publishes_an_entry(topology: str, operation: str) -> None:
    async def scenario() -> None:
        fixture = _fixture(topology)
        backend = fixture._l2 or fixture._l1
        assert backend is not None
        failure = _Phase()
        failure._error = BackendUnavailable('memo backend unavailable')
        failure._proceed.set()
        backend._before[operation] = failure
        calls = 0

        async def fetch() -> object:
            nonlocal calls
            calls += 1
            return calls

        cache = fixture._cache
        assert await cache.read_parsed(_CALL, _PLAN, fetch, _OPTIONS, _parse) == ('parsed', 1)
        assert await cache.read_parsed(_CALL, _PLAN, fetch, _OPTIONS, _parse) == ('parsed', 2)
        assert cache._memo is None
        await cache.close()

    asyncio.run(scenario())


def test_memo_reuses_oversized_value_in_every_topology() -> None:
    async def scenario(topology: str) -> None:
        settings = CacheSettings(
            l1=L1Settings(enabled=topology != 'shared'),
            l2=L2Settings(enabled=topology != 'local'),
        )
        settings = replace(
            settings, limits=replace(settings.limits, max_serialized_result_bytes=100)
        )
        metrics = _RecordingMetrics()
        l1 = AsyncLocalBackend(settings, monotonic, metrics) if settings.l1.enabled else None
        l2 = AsyncLocalBackend(settings, monotonic, metrics) if settings.l2.enabled else None
        cache = sg_query_cache.AsyncQueryCache(settings, _IDENTITY, l1=l1, l2=l2)
        calls = 0

        async def fetch() -> object:
            nonlocal calls
            calls += 1
            return b'x' * 1000

        def parse(value: object) -> object:
            return ('parsed', value)

        first = await cache.read_parsed(_CALL, _PLAN, fetch, _OPTIONS, parse)
        assert await cache.read_parsed(_CALL, _PLAN, fetch, _OPTIONS, parse) is first
        assert calls == 1
        await cache.close()

    for topology in ('local', 'shared', 'hierarchy'):
        asyncio.run(scenario(topology))


def test_older_success_cannot_clear_newer_retention_and_unrelated_fence_replays_it() -> None:
    async def scenario() -> None:
        fixture = _fixture()
        cache = fixture._cache
        backend = fixture._l1
        assert backend is not None
        first = GenerationChange(DependencyTag.type_membership('production', 'Asset'))
        second = GenerationChange(DependencyTag.type_membership('production', 'Shot'))
        seen: list[tuple[GenerationChange, ...]] = []
        original = backend.change

        async def recorded(changes: tuple[GenerationChange, ...]) -> None:
            seen.append(changes)
            await original(changes)

        backend.change = recorded
        paused = _Phase()
        backend._after['change'] = paused
        older = asyncio.create_task(cache.invalidate((first,)))
        await paused._entered.wait()
        backend._after.clear()
        failure = _Phase()
        failure._proceed.set()
        failure._error = BackendUnavailable('newer failure')
        backend._before['change'] = failure
        assert not await cache.invalidate((second,))
        paused._proceed.set()
        assert await older
        assert cache._fences['l1']._retained == (second,)
        backend._before.clear()
        assert await cache.invalidate((first,))
        assert seen[-1] == (second, first)
        assert not cache._fences['l1']._retained
        await cache.close()

    asyncio.run(scenario())


def test_retained_changes_collapse_to_one_site_change() -> None:
    async def scenario() -> None:
        fixture = _fixture()
        assert fixture._l1 is not None
        failure = _Phase()
        failure._error = BackendUnavailable('failed')
        failure._proceed.set()
        fixture._l1._before['change'] = failure
        changes = tuple(
            GenerationChange(DependencyTag.type_membership('production', f'Type{number}'))
            for number in range(257)
        )
        assert not await fixture._cache.invalidate(changes)
        assert fixture._cache._fences['l1']._retained == (
            GenerationChange(DependencyTag.site('production')),
        )
        fixture._l1._before.clear()
        assert await fixture._cache.invalidate(())
        assert not fixture._cache._fences['l1']._retained
        await fixture._cache.close()

    asyncio.run(scenario())


@pytest.mark.parametrize('topology', ['local', 'shared', 'hierarchy'])
def test_retained_fence_prevents_memo_reuse_until_replayed(topology: str) -> None:
    async def scenario() -> None:
        fixture = _fixture(topology)
        calls = 0

        async def fetch() -> object:
            nonlocal calls
            calls += 1
            return calls

        cache = fixture._cache
        assert await cache.read_parsed(_CALL, _PLAN, fetch, _OPTIONS, _parse) == ('parsed', 1)
        for backend in (fixture._l1, fixture._l2):
            if backend is not None:
                failure = _Phase()
                failure._error = BackendUnavailable('unacknowledged')
                failure._proceed.set()
                backend._before['change'] = failure
        assert not await cache.invalidate((GenerationChange(_PLAN.tags[0]),))
        assert await cache.read_parsed(_CALL, _PLAN, fetch, _OPTIONS, _parse) == ('parsed', 2)
        for backend in (fixture._l1, fixture._l2):
            if backend is not None:
                backend._before.clear()
        assert await cache.invalidate(())
        assert await cache.read_parsed(_CALL, _PLAN, fetch, _OPTIONS, _parse) == ('parsed', 3)
        await cache.close()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ('cap', 'reason', 'observed_age'),
    [
        pytest.param(None, 'expired', 0.0, id='decline-with-reason'),
        pytest.param(None, None, None, id='decline-without-reason'),
        pytest.param(2.0, 'signed_url', 2.0, id='shortened'),
        pytest.param(-1.0, 'invalid', None, id='invalid'),
        pytest.param('error', 'failed', None, id='callback-failure'),
    ],
)
def test_population_decision_declines_or_shortens_without_changing_return(
    cap: object, reason: str | None, observed_age: float | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    @dataclass(frozen=True)
    class _Decision:
        remaining_seconds: float | None
        reason: str | None

    async def scenario() -> None:
        fixture = _fixture()
        monkeypatch.setattr(cache_module, 'monotonic', fixture._clock)
        calls = 0

        async def fetch() -> object:
            nonlocal calls
            calls += 1
            return calls

        def decision(_value: object) -> _Decision:
            if cap == 'error':
                raise ValueError('URL parsing failed')
            assert cap is None or isinstance(cap, float)
            return _Decision(cap, reason)

        async def read() -> object:
            return await fixture._cache.read(
                _CALL, _PLAN, fetch, _OPTIONS, _population_decision=decision
            )

        assert await read() == 1
        age_events = [
            event for event in fixture._metrics._events if event[0] == 'cache_url_age_cap'
        ]
        expected = (
            []
            if observed_age is None
            else [('cache_url_age_cap', observed_age, {'reason': reason})]
        )
        assert age_events == expected
        assert await read() == (1 if cap == 2.0 else 2)
        fixture._clock._now += 3
        assert await read() == (2 if cap == 2.0 else 3)
        await fixture._cache.close()

    asyncio.run(scenario())


def test_scoped_age_never_relaxes_outer_bound() -> None:
    async def scenario() -> None:
        settings = CacheSettings()
        metrics = _RecordingMetrics()
        backend = AsyncLocalBackend(settings, monotonic, metrics)
        cache = sg_query_cache.AsyncQueryCache(settings, _IDENTITY, l1=backend, metrics=metrics)

        async def fetch() -> object:
            return 42

        with cache.policy(maximum_age=2), cache.policy(maximum_age=10):
            assert await cache.read(_CALL, _PLAN, fetch, _OPTIONS, 20) == 42
        ages = [value for name, value, _labels in metrics._events if name == 'cache_ttl_selection']
        assert ages
        assert set(ages) == {2}
        await cache.close()

    asyncio.run(scenario())


@pytest.mark.parametrize('topology', ['local', 'shared', 'hierarchy'])
def test_shorter_read_age_accounts_for_original_sg_age(
    topology: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def scenario() -> None:
        fixture = _fixture(topology)
        monkeypatch.setattr(cache_module, 'monotonic', fixture._clock)
        calls = 0

        async def fetch() -> object:
            nonlocal calls
            calls += 1
            return calls

        cache = fixture._cache
        assert await cache.read(_CALL, _PLAN, fetch, _OPTIONS) == 1
        fixture._clock._now += 3
        assert await cache.read(_CALL, _PLAN, fetch, _OPTIONS, 2) == 2
        assert calls == 2
        await cache.close()

    asyncio.run(scenario())


@pytest.mark.parametrize('topology', ['local', 'shared', 'hierarchy'])
def test_read_collapses_and_returns_isolated_hits(topology: str) -> None:
    async def scenario() -> None:
        settings = CacheSettings(
            l1=L1Settings(enabled=topology != 'shared'),
            l2=L2Settings(enabled=topology != 'local'),
        )
        metrics = _RecordingMetrics()
        l1 = AsyncLocalBackend(settings, monotonic, metrics) if settings.l1.enabled else None
        l2 = AsyncLocalBackend(settings, monotonic, metrics) if settings.l2.enabled else None
        cache = sg_query_cache.AsyncQueryCache(settings, _IDENTITY, l1=l1, l2=l2)
        phase = _Phase()
        calls = 0

        async def fetch() -> object:
            nonlocal calls
            calls += 1
            await phase._wait()
            return [{'id': calls}]

        leader = asyncio.create_task(cache.read(_CALL, _PLAN, fetch, _OPTIONS))
        await phase._entered.wait()
        followers = [
            asyncio.create_task(cache.read(_CALL, _PLAN, fetch, _OPTIONS)) for _ in range(3)
        ]
        phase._proceed.set()
        results = await asyncio.gather(leader, *followers)
        assert calls == 1
        assert results == [[{'id': 1}]] * 4
        assert len({id(result) for result in results}) == 4
        await cache.close()

    asyncio.run(scenario())


@pytest.mark.parametrize('topology', ['local', 'shared', 'hierarchy'])
@pytest.mark.parametrize('no_store', ['oversized', 'failed'])
def test_concurrent_memo_acquisition_remains_single_flight_without_storage(
    topology: str, no_store: str
) -> None:
    async def scenario() -> None:
        settings = CacheSettings(
            l1=L1Settings(enabled=topology != 'shared'),
            l2=L2Settings(enabled=topology != 'local'),
        )
        if no_store == 'oversized':
            settings = replace(
                settings, limits=replace(settings.limits, max_serialized_result_bytes=100)
            )
        fixture = _fixture(topology, settings)
        backend = fixture._l2 or fixture._l1
        assert backend is not None
        if no_store == 'failed':
            failure = _Phase()
            failure._proceed.set()
            failure._error = BackendUnavailable('no storage')
            backend._before['commit'] = failure
        fetching = _Phase()
        calls = 0

        async def fetch() -> object:
            nonlocal calls
            calls += 1
            await fetching._wait()
            return b'x' * 1000

        async def parsed() -> object:
            return await fixture._cache.read_parsed(_CALL, _PLAN, fetch, _OPTIONS, _parse)

        leader = asyncio.create_task(parsed())
        await fetching._entered.wait()
        followers = [asyncio.create_task(parsed()) for _ in range(3)]
        await _turn()
        fetching._proceed.set()
        results = await asyncio.gather(leader, *followers)
        assert calls == 1
        assert all(result is results[0] for result in results)
        await fixture._cache.close()

    asyncio.run(scenario())


@pytest.mark.parametrize('topology', ['local', 'shared', 'hierarchy'])
@pytest.mark.parametrize('kind', ['site', 'authorization', 'schema', 'identity'])
def test_each_schema_generation_permits_a_new_memo_acquisition(topology: str, kind: str) -> None:
    async def scenario() -> None:
        fixture = _fixture(topology)
        tags = {
            'site': DependencyTag.site('production'),
            'authorization': DependencyTag.authorization('production'),
            'schema': DependencyTag.schema('production'),
            'identity': DependencyTag.identity_visibility('production', 1),
        }
        plan = DependencyPlan('schema', tuple(tags.values()), 'schema')
        calls = 0

        async def fetch() -> object:
            nonlocal calls
            calls += 1
            return calls

        cache = fixture._cache
        first = await cache.read_parsed(_CALL, plan, fetch, _OPTIONS, _parse)
        assert await cache.invalidate((GenerationChange(tags[kind]),))
        assert await cache.read_parsed(_CALL, plan, fetch, _OPTIONS, _parse) != first
        assert calls == 2
        await cache.close()

    asyncio.run(scenario())


@pytest.mark.parametrize('parsed', [False, True])
def test_lookup_and_memo_resolution_recheck_retained_revision(parsed: bool) -> None:
    async def scenario() -> None:
        fixture = _fixture()
        assert fixture._l1 is not None
        cache = fixture._cache
        if parsed:
            assert await cache.read_parsed(_CALL, _PLAN, _answer, _OPTIONS, _parse) == (
                'parsed',
                42,
            )
        else:
            assert await _read(cache) == 42
        paused = _Phase()
        fixture._l1._after['resolve' if parsed else 'lookup'] = paused

        async def fresh() -> object:
            return 43

        async def read() -> object:
            if parsed:
                return await cache.read_parsed(_CALL, _PLAN, fresh, _OPTIONS, _parse)
            return await cache.read(_CALL, _PLAN, fresh, _OPTIONS)

        reader = asyncio.create_task(read())
        await paused._entered.wait()
        failure = _Phase()
        failure._error = BackendUnavailable('fence')
        failure._proceed.set()
        fixture._l1._before['change'] = failure
        assert not await cache.invalidate((GenerationChange(_PLAN.tags[0]),))
        paused._proceed.set()
        assert await reader == (('parsed', 43) if parsed else 43)
        await cache.close()

    asyncio.run(scenario())
