import asyncio
from dataclasses import dataclass, replace
from inspect import Parameter, signature
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
from tests._async_fakes import Phase, PhaseBackend, RecordingMetrics
from tests._fakes import CLIENT_OPTIONS, turn
from tests.cache._async_cases import (
    CALL,
    IDENTITY,
    PLAN,
    answer,
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


def _assert_distributed_wait_metrics(metrics: RecordingMetrics, outcome: str) -> None:
    events = metrics.events
    assert events.count(('cache_collapse', None, {'outcome': 'distributed'})) == 1
    waits = [event for event in events if event[0] == 'cache_wait_seconds']
    assert len(waits) == 1
    assert waits[0][2] == {'outcome': outcome}
    assert waits[0][1] is not None
    assert waits[0][1] >= 0
    if outcome == 'takeover':
        assert waits[0][1] == 0.375
    assert events.count(('cache_lease_timeout', None, {'outcome': 'distributed'})) == (
        outcome == 'timeout'
    )
    if outcome == 'failure':
        assert ('cache_bypass', None, {'reason': 'backend_failure'}) in events


async def _finish_distributed_poll(
    backend: PhaseBackend, held: FillLease, outcome: str, waiting: asyncio.Event
) -> None:
    if outcome == 'cancelled':
        waiting.set()
        await asyncio.Event().wait()
    elif outcome == 'failure':
        failure = Phase()
        failure.error = BackendUnavailable('poll failed')
        failure.proceed.set()
        backend.before['lookup'] = failure
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
        assert await _read(fixture.cache) == 42

        def forbidden(_payload: bytes) -> object:
            raise AssertionError('core decoded an already decoded hit')

        monkeypatch.setattr(common_module, 'decode_value', forbidden)
        assert await _read(fixture.cache) == 42
        await fixture.cache.close()

    asyncio.run(scenario())


def test_cache_constructor_parameters_after_identity_are_keyword_only() -> None:
    parameters = list(signature(AsyncQueryCache).parameters.values())
    assert [(parameter.name, parameter.kind) for parameter in parameters] == [
        ('settings', Parameter.POSITIONAL_OR_KEYWORD),
        ('identity', Parameter.POSITIONAL_OR_KEYWORD),
        ('key_builder', Parameter.KEYWORD_ONLY),
        ('l1', Parameter.KEYWORD_ONLY),
        ('l2', Parameter.KEYWORD_ONLY),
        ('metrics', Parameter.KEYWORD_ONLY),
    ]


def test_child_policy_snapshot_and_independent_instances() -> None:
    async def scenario() -> None:
        first = _fixture()
        second = _fixture()
        gate = asyncio.Event()

        async def child() -> tuple[bool, bool]:
            await gate.wait()
            return first.cache.bypass_active, second.cache.bypass_active

        with first.cache.bypass():
            task = asyncio.create_task(child())
        await turn()
        assert not first.cache.bypass_active
        gate.set()
        assert await task == (True, False)
        await first.cache.close()
        await second.cache.close()

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
        backend = PhaseBackend(AsyncLocalBackend(settings, monotonic, RecordingMetrics()))
        setattr(backend, name, lambda: None)
        with pytest.raises(ValueError, match=name):
            cache_type(settings, IDENTITY, l1=backend)


def test_custom_two_field_hit_decodes_once(monkeypatch: pytest.MonkeyPatch) -> None:
    async def scenario() -> None:
        fixture = _fixture()
        backend = fixture.l1
        assert backend is not None
        assert await _read(fixture.cache) == 42
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
        assert await _read(fixture.cache) == 42
        assert decodes == 1
        await fixture.cache.close()

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
        if outcome == 'takeover':
            monkeypatch.setattr(asyncio.get_running_loop(), 'time', fixture.clock)
        backend = fixture.l1 if topology == 'local' else fixture.l2
        assert backend is not None
        vector = await backend.resolve(PLAN.tags)
        base = KeyBuilder.base_key('production', IDENTITY, CLIENT_OPTIONS, CALL)
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
            fixture.clock.now += 0.125 if outcome == 'takeover' else 0
            if polls != 3:
                await original(0)
                return
            await _finish_distributed_poll(backend, held, outcome, waiting)

        async def fetch() -> object:
            if outcome == 'takeover':
                fixture.clock.now += 0.5
            return 42

        monkeypatch.setattr(cache_module, 'sleep', poll)
        task = asyncio.create_task(fixture.cache.read(CALL, PLAN, fetch, CLIENT_OPTIONS))
        if outcome == 'cancelled':
            await asyncio.wait_for(waiting.wait(), timeout=2)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            assert await asyncio.wait_for(task, timeout=2) == (
                84 if outcome == 'completed' else 42
            )
        await fixture.cache.close()
        _assert_distributed_wait_metrics(fixture.metrics, outcome)

    asyncio.run(scenario())


def test_distributed_wait_expiry_does_not_reserve_again_or_populate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        settings = CacheSettings(l1=L1Settings(enabled=False), l2=L2Settings(enabled=True))
        settings = replace(settings, limits=replace(settings.limits, lease_wait_seconds=0.001))
        fixture = _fixture('shared', settings)
        backend = fixture.l2
        assert backend is not None
        vector = await backend.resolve(PLAN.tags)
        base = KeyBuilder.base_key('production', IDENTITY, CLIENT_OPTIONS, CALL)
        key = KeyBuilder.qualified_key(base, vector, 'result')
        assert await backend.reserve(key, 30) is not None
        original = asyncio.sleep

        async def sleep_until_lease_expires(delay: float) -> None:
            await original(delay)
            if delay > 0:
                fixture.clock.now += 31

        monkeypatch.setattr(cache_module, 'sleep', sleep_until_lease_expires)
        assert await _read(fixture.cache) == 42
        assert backend.calls.count('reserve') <= 2
        assert 'commit' not in backend.calls
        await fixture.cache.close()

    asyncio.run(scenario())


def test_encoding_time_debits_origin_deadline(monkeypatch: pytest.MonkeyPatch) -> None:
    async def scenario() -> None:
        fixture = _fixture()
        monkeypatch.setattr(cache_module, 'monotonic', fixture.clock)
        encoder = cache_module.encode_result

        def slow_encode(
            timed: common_module.TimedResult, origin_elapsed: float, lifetime: float
        ) -> bytes:
            payload = encoder(timed, origin_elapsed, lifetime)
            fixture.clock.now += 3
            return payload

        monkeypatch.setattr(cache_module, 'encode_result', slow_encode)
        assert await fixture.cache.read(CALL, PLAN, answer, CLIENT_OPTIONS, 2) == 42
        assert fixture.l1 is not None
        assert 'commit' not in fixture.l1.calls
        assert 'release' in fixture.l1.calls
        await fixture.cache.close()

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

        cache = fixture.cache
        assert await cache.read(CALL, PLAN, fetch, CLIENT_OPTIONS) == 1
        changed = ExactCall('find', (('entity_type', 'Shot'), ('filters', ())))
        assert await cache.read(changed, PLAN, fetch, CLIENT_OPTIONS) == 2
        assert await cache.read(CALL, PLAN, fetch, {**CLIENT_OPTIONS, 'localized': True}) == 3
        assert await cache.read(CALL, PLAN, fetch, CLIENT_OPTIONS) == 1
        backend = fixture.l1 or fixture.l2
        assert backend is not None
        base = KeyBuilder.base_key('production', IDENTITY, CLIENT_OPTIONS, CALL)
        vector = await backend.resolve(PLAN.tags)
        hit = await backend.lookup(KeyBuilder.qualified_key(base, vector, 'result'))
        assert hit is not None
        assert common_module.decode_hit(hit, 30).value == 1
        other = AsyncQueryCache(
            cache_module.client_binding(cache)[0],
            Identity('script', 'tests', 'other'),
            l1=fixture.l1,
            l2=fixture.l2,
        )
        assert await other.read(CALL, PLAN, fetch, CLIENT_OPTIONS) == 4
        await cache.close()

    asyncio.run(scenario())


@pytest.mark.parametrize('topology', ['local', 'shared', 'hierarchy'])
@pytest.mark.parametrize('operation', ['resolve', 'lookup', 'reserve', 'commit'])
def test_backend_failure_preserves_direct_success(topology: str, operation: str) -> None:
    async def scenario() -> None:
        fixture = _fixture(topology)
        backend = fixture.l2 or fixture.l1
        assert backend is not None
        phase = Phase()
        phase.proceed.set()
        phase.error = BackendUnavailable('offline')
        backend.before[operation] = phase
        calls = 0

        async def fetch() -> object:
            nonlocal calls
            calls += 1
            return 42

        assert await fixture.cache.read(CALL, PLAN, fetch, CLIENT_OPTIONS) == 42
        assert calls == 1
        assert ('cache_bypass', None, {'reason': 'backend_failure'}) in fixture.metrics.events
        await fixture.cache.close()

    asyncio.run(scenario())


def test_memo_decision_sample_precedes_telemetry(monkeypatch: pytest.MonkeyPatch) -> None:
    async def scenario() -> None:
        fixture = _fixture()
        monkeypatch.setattr(cache_module, 'monotonic', fixture.clock)
        cache = fixture.cache
        calls = 0

        async def fetch() -> object:
            nonlocal calls
            calls += 1
            return calls

        first = await cache.read_parsed(CALL, PLAN, fetch, CLIENT_OPTIONS, _parse)
        fixture.clock.now += 29

        def telemetry() -> None:
            name, _value, labels = fixture.metrics.events[-1]
            if name == 'cache_schema_view' and labels['outcome'] == 'reuse':
                fixture.clock.now += 2

        fixture.metrics.callback = telemetry
        assert await cache.read_parsed(CALL, PLAN, fetch, CLIENT_OPTIONS, _parse) is first
        assert await cache.read_parsed(CALL, PLAN, fetch, CLIENT_OPTIONS, _parse) == ('parsed', 2)
        await cache.close()

    asyncio.run(scenario())


@pytest.mark.parametrize('topology', ['local', 'shared', 'hierarchy'])
@pytest.mark.parametrize('stored', [True, False])
def test_memo_first_and_deep_deadlines(
    topology: str, stored: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def scenario() -> None:
        fixture = _fixture(topology)
        monkeypatch.setattr(cache_module, 'monotonic', fixture.clock)
        if not stored:
            backend = fixture.l2 or fixture.l1
            assert backend is not None
            phase = Phase()
            phase.error = BackendUnavailable('commit lost')
            phase.proceed.set()
            backend.before['commit'] = phase
        plan = replace(PLAN, classification='schema')
        calls = 0

        async def fetch() -> object:
            nonlocal calls
            calls += 1
            return calls

        async def parsed() -> object:
            return await fixture.cache.read_parsed(CALL, plan, fetch, CLIENT_OPTIONS, _parse)

        first = await parsed()
        assert await parsed() is first
        fixture.clock.now += 11
        if topology == 'hierarchy':
            assert fixture.l2 is not None
            before = fixture.l2.calls.count('resolve')
            assert await parsed() is first
            assert fixture.l2.calls.count('resolve') == before + 1
        else:
            assert await parsed() is first
        fixture.clock.now = 2801
        assert await parsed() != first
        assert calls == 2
        await fixture.cache.close()

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

        cache = fixture.cache
        assert await cache.read_parsed(CALL, PLAN, answer, CLIENT_OPTIONS, parse) == ('parsed', 42)
        changed = ExactCall('schema_read', ())
        with pytest.raises(LookupError, match='schema unavailable'):
            await cache.read_parsed(changed, PLAN, fail, CLIENT_OPTIONS, parse)
        assert await cache.read_parsed(CALL, PLAN, answer, CLIENT_OPTIONS, parse) == ('parsed', 42)
        assert parses == 2
        await cache.close()

    asyncio.run(scenario())


def test_memo_local_hit_never_acquires_shared_key(monkeypatch: pytest.MonkeyPatch) -> None:
    async def scenario() -> None:
        fixture = _fixture('hierarchy')
        monkeypatch.setattr(cache_module, 'monotonic', fixture.clock)
        cache = fixture.cache
        plan = replace(PLAN, classification='schema')
        assert await cache.read(CALL, plan, answer, CLIENT_OPTIONS) == 42
        assert fixture.l2 is not None
        before = fixture.l2.calls.count('resolve')
        first = await cache.read_parsed(CALL, plan, answer, CLIENT_OPTIONS, _parse)
        assert fixture.l2.calls.count('resolve') == before
        assert cache._memo is not None
        assert cache._memo.outcome.shared is None
        assert cache._memo.outcome.first_deadline == cache._memo.outcome.deep_deadline
        fixture.clock.now += 11
        assert await cache.read_parsed(CALL, plan, answer, CLIENT_OPTIONS, _parse) == first
        assert fixture.l2.calls.count('lookup') > 1
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

        cache = fixture.cache
        first = await cache.read_parsed(CALL, PLAN, fetch, CLIENT_OPTIONS, parse)
        assert first == ('view', b'good')
        assert await cache.read_parsed(CALL, PLAN, fetch, CLIENT_OPTIONS, parse) is first
        assert calls == 2
        other = await cache.read_parsed(CALL, PLAN, fetch, CLIENT_OPTIONS, _parse)
        assert other == ('parsed', b'bad')
        assert calls == 2
        assert await cache.invalidate((GenerationChange(PLAN.tags[0]),))
        assert await cache.read_parsed(CALL, PLAN, fetch, CLIENT_OPTIONS, parse) == (
            'view',
            b'new',
        )
        assert calls == 3
        await cache.close()

    asyncio.run(scenario())


@pytest.mark.parametrize('topology', ['local', 'shared', 'hierarchy'])
@pytest.mark.parametrize('operation', ['resolve', 'lookup'])
def test_failed_memo_acquisition_never_publishes_an_entry(topology: str, operation: str) -> None:
    async def scenario() -> None:
        fixture = _fixture(topology)
        backend = fixture.l2 or fixture.l1
        assert backend is not None
        failure = Phase()
        failure.error = BackendUnavailable('memo backend unavailable')
        failure.proceed.set()
        backend.before[operation] = failure
        calls = 0

        async def fetch() -> object:
            nonlocal calls
            calls += 1
            return calls

        cache = fixture.cache
        assert await cache.read_parsed(CALL, PLAN, fetch, CLIENT_OPTIONS, _parse) == ('parsed', 1)
        assert await cache.read_parsed(CALL, PLAN, fetch, CLIENT_OPTIONS, _parse) == ('parsed', 2)
        assert cache._memo is None
        await cache.close()

    asyncio.run(scenario())


def test_older_success_cannot_clear_newer_retention_and_unrelated_fence_replays_it() -> None:
    async def scenario() -> None:
        fixture = _fixture()
        cache = fixture.cache
        backend = fixture.l1
        assert backend is not None
        first = GenerationChange(DependencyTag.type_membership('production', 'Asset'))
        second = GenerationChange(DependencyTag.type_membership('production', 'Shot'))
        seen: list[tuple[GenerationChange, ...]] = []
        original = backend.change

        async def recorded(changes: tuple[GenerationChange, ...]) -> None:
            seen.append(changes)
            await original(changes)

        backend.change = recorded
        paused = Phase()
        backend.after['change'] = paused
        older = asyncio.create_task(cache.invalidate((first,)))
        await paused.entered.wait()
        backend.after.clear()
        failure = Phase()
        failure.proceed.set()
        failure.error = BackendUnavailable('newer failure')
        backend.before['change'] = failure
        assert not await cache.invalidate((second,))
        paused.proceed.set()
        assert await older
        assert cache._fences['l1'].retained == (second,)
        backend.before.clear()
        assert await cache.invalidate((first,))
        assert seen[-1] == (second, first)
        assert not cache._fences['l1'].retained
        await cache.close()

    asyncio.run(scenario())


def test_retained_changes_collapse_to_one_site_change() -> None:
    async def scenario() -> None:
        fixture = _fixture()
        assert fixture.l1 is not None
        failure = Phase()
        failure.error = BackendUnavailable('failed')
        failure.proceed.set()
        fixture.l1.before['change'] = failure
        changes = tuple(
            GenerationChange(DependencyTag.type_membership('production', f'Type{number}'))
            for number in range(257)
        )
        assert not await fixture.cache.invalidate(changes)
        assert fixture.cache._fences['l1'].retained == (
            GenerationChange(DependencyTag.site('production')),
        )
        fixture.l1.before.clear()
        assert await fixture.cache.invalidate(())
        assert not fixture.cache._fences['l1'].retained
        await fixture.cache.close()

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

        cache = fixture.cache
        assert await cache.read_parsed(CALL, PLAN, fetch, CLIENT_OPTIONS, _parse) == ('parsed', 1)
        for backend in (fixture.l1, fixture.l2):
            if backend is not None:
                failure = Phase()
                failure.error = BackendUnavailable('unacknowledged')
                failure.proceed.set()
                backend.before['change'] = failure
        assert not await cache.invalidate((GenerationChange(PLAN.tags[0]),))
        assert await cache.read_parsed(CALL, PLAN, fetch, CLIENT_OPTIONS, _parse) == ('parsed', 2)
        for backend in (fixture.l1, fixture.l2):
            if backend is not None:
                backend.before.clear()
        assert await cache.invalidate(())
        assert await cache.read_parsed(CALL, PLAN, fetch, CLIENT_OPTIONS, _parse) == ('parsed', 3)
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
        monkeypatch.setattr(cache_module, 'monotonic', fixture.clock)
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
            return await fixture.cache.read(
                CALL, PLAN, fetch, CLIENT_OPTIONS, _population_decision=decision
            )

        assert await read() == 1
        age_events = [event for event in fixture.metrics.events if event[0] == 'cache_url_age_cap']
        expected = (
            []
            if observed_age is None
            else [('cache_url_age_cap', observed_age, {'reason': reason})]
        )
        assert age_events == expected
        assert await read() == (1 if cap == 2.0 else 2)
        fixture.clock.now += 3
        assert await read() == (2 if cap == 2.0 else 3)
        await fixture.cache.close()

    asyncio.run(scenario())


def test_scoped_age_never_relaxes_outer_bound() -> None:
    async def scenario() -> None:
        settings = CacheSettings()
        metrics = RecordingMetrics()
        backend = AsyncLocalBackend(settings, monotonic, metrics)
        cache = sg_query_cache.AsyncQueryCache(settings, IDENTITY, l1=backend, metrics=metrics)

        async def fetch() -> object:
            return 42

        with cache.policy(maximum_age=2), cache.policy(maximum_age=10):
            assert await cache.read(CALL, PLAN, fetch, CLIENT_OPTIONS, 20) == 42
        ages = [value for name, value, _labels in metrics.events if name == 'cache_ttl_selection']
        assert ages
        assert set(ages) == {2}
        await cache.close()

    asyncio.run(scenario())


@pytest.mark.parametrize('topology', ['local', 'shared', 'hierarchy'])
def test_successful_fill_skips_releasing_its_consumed_lease(topology: str) -> None:
    async def scenario() -> None:
        fixture = _fixture(topology)
        assert await _read(fixture.cache) == 42
        for backend in (fixture.l1, fixture.l2):
            if backend is not None:
                assert 'commit' in backend.calls
                assert 'release' not in backend.calls
        await fixture.cache.close()

    asyncio.run(scenario())


@pytest.mark.parametrize('topology', ['local', 'shared', 'hierarchy'])
def test_shorter_read_age_accounts_for_original_sg_age(
    topology: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def scenario() -> None:
        fixture = _fixture(topology)
        monkeypatch.setattr(cache_module, 'monotonic', fixture.clock)
        calls = 0

        async def fetch() -> object:
            nonlocal calls
            calls += 1
            return calls

        cache = fixture.cache
        assert await cache.read(CALL, PLAN, fetch, CLIENT_OPTIONS) == 1
        fixture.clock.now += 3
        assert await cache.read(CALL, PLAN, fetch, CLIENT_OPTIONS, 2) == 2
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
        metrics = RecordingMetrics()
        l1 = AsyncLocalBackend(settings, monotonic, metrics) if settings.l1.enabled else None
        l2 = AsyncLocalBackend(settings, monotonic, metrics) if settings.l2.enabled else None
        cache = sg_query_cache.AsyncQueryCache(settings, IDENTITY, l1=l1, l2=l2)
        phase = Phase()
        calls = 0

        async def fetch() -> object:
            nonlocal calls
            calls += 1
            await phase.wait()
            return [{'id': calls}]

        leader = asyncio.create_task(cache.read(CALL, PLAN, fetch, CLIENT_OPTIONS))
        await phase.entered.wait()
        followers = [
            asyncio.create_task(cache.read(CALL, PLAN, fetch, CLIENT_OPTIONS)) for _ in range(3)
        ]
        phase.proceed.set()
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
        backend = fixture.l2 or fixture.l1
        assert backend is not None
        if no_store == 'failed':
            failure = Phase()
            failure.proceed.set()
            failure.error = BackendUnavailable('no storage')
            backend.before['commit'] = failure
        fetching = Phase()
        calls = 0

        async def fetch() -> object:
            nonlocal calls
            calls += 1
            await fetching.wait()
            return b'x' * 1000

        async def parsed() -> object:
            return await fixture.cache.read_parsed(CALL, PLAN, fetch, CLIENT_OPTIONS, _parse)

        leader = asyncio.create_task(parsed())
        await fetching.entered.wait()
        followers = [asyncio.create_task(parsed()) for _ in range(3)]
        await turn()
        fetching.proceed.set()
        results = await asyncio.gather(leader, *followers)
        assert calls == 1
        assert all(result is results[0] for result in results)
        await fixture.cache.close()

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

        cache = fixture.cache
        first = await cache.read_parsed(CALL, plan, fetch, CLIENT_OPTIONS, _parse)
        assert await cache.invalidate((GenerationChange(tags[kind]),))
        assert await cache.read_parsed(CALL, plan, fetch, CLIENT_OPTIONS, _parse) != first
        assert calls == 2
        await cache.close()

    asyncio.run(scenario())


@pytest.mark.parametrize('parsed', [False, True])
def test_lookup_and_memo_resolution_recheck_retained_revision(parsed: bool) -> None:
    async def scenario() -> None:
        fixture = _fixture()
        assert fixture.l1 is not None
        cache = fixture.cache
        if parsed:
            assert await cache.read_parsed(CALL, PLAN, answer, CLIENT_OPTIONS, _parse) == (
                'parsed',
                42,
            )
        else:
            assert await _read(cache) == 42
        paused = Phase()
        fixture.l1.after['resolve' if parsed else 'lookup'] = paused

        async def fresh() -> object:
            return 43

        async def read() -> object:
            if parsed:
                return await cache.read_parsed(CALL, PLAN, fresh, CLIENT_OPTIONS, _parse)
            return await cache.read(CALL, PLAN, fresh, CLIENT_OPTIONS)

        reader = asyncio.create_task(read())
        await paused.entered.wait()
        failure = Phase()
        failure.error = BackendUnavailable('fence')
        failure.proceed.set()
        fixture.l1.before['change'] = failure
        assert not await cache.invalidate((GenerationChange(PLAN.tags[0]),))
        paused.proceed.set()
        assert await reader == (('parsed', 43) if parsed else 43)
        await cache.close()

    asyncio.run(scenario())


def test_wiring_error_identifies_shared_backend_instance() -> None:
    settings = CacheSettings(l2=L2Settings(enabled=True))
    backend = AsyncLocalBackend(settings, monotonic, RecordingMetrics())
    with pytest.raises(ValueError, match='backend wiring') as error:
        AsyncQueryCache(settings, IDENTITY, l1=backend, l2=backend)
    assert str(error.value) == (
        'backend wiring does not match the settings: '
        'L1 enabled and L2 enabled, but l1 and l2 were supplied; '
        'l1 and l2 must be distinct backends'
    )


@pytest.mark.parametrize(
    ('enabled', 'supplied', 'description'),
    [
        ((True, False), (False, False), 'no backends were supplied'),
        ((True, False), (False, True), 'l2 was supplied'),
        ((True, False), (True, True), 'l1 and l2 were supplied'),
        ((False, True), (False, False), 'no backends were supplied'),
        ((False, True), (True, False), 'l1 was supplied'),
        ((False, True), (True, True), 'l1 and l2 were supplied'),
        ((True, True), (False, False), 'no backends were supplied'),
        ((True, True), (True, False), 'l1 was supplied'),
        ((True, True), (False, True), 'l2 was supplied'),
    ],
)
def test_wiring_error_names_layers_and_backends(
    enabled: tuple[bool, bool], supplied: tuple[bool, bool], description: str
) -> None:
    settings = CacheSettings(l1=L1Settings(enabled=enabled[0]), l2=L2Settings(enabled=enabled[1]))
    l1 = AsyncLocalBackend(settings, monotonic, RecordingMetrics()) if supplied[0] else None
    l2 = AsyncLocalBackend(settings, monotonic, RecordingMetrics()) if supplied[1] else None
    with pytest.raises(ValueError, match='backend wiring') as error:
        AsyncQueryCache(settings, IDENTITY, l1=l1, l2=l2)
    l1_state = 'enabled' if enabled[0] else 'disabled'
    l2_state = 'enabled' if enabled[1] else 'disabled'
    assert str(error.value) == (
        'backend wiring does not match the settings: '
        f'L1 {l1_state} and L2 {l2_state}, but {description}'
    )
