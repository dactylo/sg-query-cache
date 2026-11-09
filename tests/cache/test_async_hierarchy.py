import asyncio
from dataclasses import replace

import pytest

import sg_query_cache._async_cache as cache_module
from sg_query_cache._backend import BackendUnavailable
from sg_query_cache._model import GenerationChange
from tests._async_fakes import Phase
from tests._fakes import CLIENT_OPTIONS
from tests.cache._async_cases import (
    CALL,
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


def test_aged_l2_hit_skips_an_unusable_l1_promotion(monkeypatch: pytest.MonkeyPatch) -> None:
    async def scenario() -> None:
        fixture = _fixture('hierarchy')
        monkeypatch.setattr(cache_module, 'monotonic', fixture.clock)
        assert fixture.l1 is not None
        assert await _read(fixture.cache) == 42
        fixture.clock.now += 11
        fixture.l1.calls.clear()

        async def fresh() -> object:
            return 43

        assert await fixture.cache.read(CALL, PLAN, fresh, CLIENT_OPTIONS) == 42
        assert 'reserve' not in fixture.l1.calls
        await fixture.cache.close()

    asyncio.run(scenario())


def test_cached_source_age_cannot_be_extended_by_promotion_wait(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        fixture = _fixture('hierarchy')
        monkeypatch.setattr(cache_module, 'monotonic', fixture.clock)
        assert fixture.l1 is not None
        assert await _read(fixture.cache) == 42
        await fixture.l1.flush_site('production', 'test')
        phase = Phase()
        fixture.l1.before['reserve'] = phase

        async def fresh() -> object:
            return 43

        reader = asyncio.create_task(fixture.cache.read(CALL, PLAN, fresh, CLIENT_OPTIONS))
        await phase.entered.wait()
        fixture.clock.now += 601
        phase.proceed.set()
        assert await reader == 43
        await fixture.cache.close()

    asyncio.run(scenario())


def test_failed_l2_is_retained_before_l1_generation_change_reply() -> None:
    async def scenario() -> None:
        fixture = _fixture('hierarchy')
        cache = fixture.cache
        assert fixture.l1 is not None
        assert fixture.l2 is not None
        assert await _read(cache) == 42
        failure = Phase()
        failure.error = BackendUnavailable('L2 unavailable')
        failure.proceed.set()
        paused = Phase()
        fixture.l2.before['change'] = failure
        fixture.l1.after['change'] = paused
        fence = asyncio.create_task(cache.invalidate((GenerationChange(PLAN.tags[0]),)))
        await paused.entered.wait()
        assert cache._fences['l2'].retained
        assert not cache._fences['l1'].retained
        before = fixture.l2.calls.count('lookup')

        async def fresh() -> object:
            return 43

        assert await cache.read(CALL, PLAN, fresh, CLIENT_OPTIONS) == 43
        assert fixture.l2.calls.count('lookup') == before
        assert fixture.l1.calls.count('commit') == 1
        paused.proceed.set()
        assert not await fence
        await cache.close()

    asyncio.run(scenario())


def test_generation_change_during_fetch_keeps_original_l1_key() -> None:
    async def scenario() -> None:
        fixture = _fixture('hierarchy')
        assert fixture.l1 is not None
        calls = 0
        local = fixture.l1

        async def fetch() -> object:
            nonlocal calls
            calls += 1
            if calls == 1:
                await local.flush_site('production', 'test')
            return calls

        cache = fixture.cache
        assert await cache.read(CALL, PLAN, fetch, CLIENT_OPTIONS) == 1
        before = fixture.l2.calls.count('lookup') if fixture.l2 is not None else 0
        assert await cache.read(CALL, PLAN, fetch, CLIENT_OPTIONS) == 1
        assert calls == 1
        assert fixture.l2 is not None
        assert fixture.l2.calls.count('lookup') == before + 1
        await cache.close()

    asyncio.run(scenario())


def test_memo_revalidation_failure_continues_through_the_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        fixture = _fixture('hierarchy')
        cache = fixture.cache
        l2 = fixture.l2
        assert l2 is not None
        monkeypatch.setattr(cache_module, 'monotonic', fixture.clock)
        plan = replace(PLAN, classification='schema')
        calls = 0

        async def fetch() -> object:
            nonlocal calls
            calls += 1
            return calls

        assert await cache.read_parsed(CALL, plan, fetch, CLIENT_OPTIONS, _parse) == ('parsed', 1)
        fixture.clock.now += 11

        def fail_once() -> None:
            del l2.before['resolve']

        failure = Phase()
        failure.error = BackendUnavailable('L2 unavailable')
        failure.proceed.set()
        failure.on_enter = fail_once
        l2.before['resolve'] = failure
        assert await cache.read_parsed(CALL, plan, fetch, CLIENT_OPTIONS, _parse) == ('parsed', 1)
        assert calls == 1
        await cache.close()

    asyncio.run(scenario())


def test_memo_revalidation_rechecks_first_layer_after_shared_await(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        fixture = _fixture('hierarchy')
        cache = fixture.cache
        assert fixture.l1 is not None
        assert fixture.l2 is not None
        monkeypatch.setattr(cache_module, 'monotonic', fixture.clock)
        plan = replace(PLAN, classification='schema')
        assert await cache.read_parsed(CALL, plan, answer, CLIENT_OPTIONS, _parse) == (
            'parsed',
            42,
        )
        fixture.clock.now += 11
        paused = Phase()
        fixture.l2.after['resolve'] = paused

        async def fresh() -> object:
            return 43

        reader = asyncio.create_task(cache.read_parsed(CALL, plan, fresh, CLIENT_OPTIONS, _parse))
        await paused.entered.wait()
        failure = Phase()
        failure.error = BackendUnavailable('fence')
        failure.proceed.set()
        fixture.l1.before['change'] = failure
        assert not await cache.invalidate((GenerationChange(PLAN.tags[0]),))
        paused.proceed.set()
        assert await reader == ('parsed', 43)
        await cache.close()

    asyncio.run(scenario())


def test_parsed_read_uses_l2_after_preliminary_l1_resolution_failure() -> None:
    async def scenario() -> None:
        fixture = _fixture('hierarchy')
        cache = fixture.cache
        assert fixture.l1 is not None
        plan = replace(PLAN, classification='schema')
        calls = 0

        async def fetch() -> object:
            nonlocal calls
            calls += 1
            return calls

        assert await cache.read_parsed(CALL, plan, fetch, CLIENT_OPTIONS, _parse) == ('parsed', 1)
        failure = Phase()
        failure.error = BackendUnavailable('L1 unavailable')
        failure.proceed.set()
        fixture.l1.before['resolve'] = failure
        assert await cache.read_parsed(CALL, plan, fetch, CLIENT_OPTIONS, _parse) == ('parsed', 1)
        assert calls == 1
        await cache.close()

    asyncio.run(scenario())


@pytest.mark.parametrize('mode', ['hit', 'promotion', 'memo'])
def test_paused_successful_fence_keeps_eligible_hits_promotions_and_memo(mode: str) -> None:
    async def scenario() -> None:
        fixture = _fixture('hierarchy')
        cache = fixture.cache
        assert fixture.l1 is not None
        assert fixture.l2 is not None
        assert await _read(cache) == 42
        if mode == 'promotion':
            await fixture.l1.flush_site('production', 'test')
        elif mode == 'memo':
            assert await cache.read_parsed(CALL, PLAN, answer, CLIENT_OPTIONS, _parse) == (
                'parsed',
                42,
            )
        phase = Phase()
        fixture.l2.before['change'] = phase
        fence = asyncio.create_task(cache.invalidate((GenerationChange(PLAN.tags[0]),)))
        await phase.entered.wait()
        assert all(not state.retained for state in cache._fences.values())

        async def forbidden() -> object:
            raise AssertionError('successful pending fence triggered SG')

        if mode == 'memo':
            assert await cache.read_parsed(CALL, PLAN, forbidden, CLIENT_OPTIONS, _parse) == (
                'parsed',
                42,
            )
        else:
            assert await cache.read(CALL, PLAN, forbidden, CLIENT_OPTIONS) == 42
        if mode == 'promotion':
            assert fixture.l1.calls.count('commit') == 2
        phase.proceed.set()
        assert await fence
        await cache.close()

    asyncio.run(scenario())


@pytest.mark.parametrize('operation', ['reserve', 'commit'])
def test_promotion_rechecks_source_fence_after_await(operation: str) -> None:
    async def scenario() -> None:
        fixture = _fixture('hierarchy')
        cache = fixture.cache
        assert fixture.l1 is not None
        assert fixture.l2 is not None
        assert await _read(cache) == 42
        await fixture.l1.flush_site('production', 'test')
        paused = Phase()
        fixture.l1.before[operation] = paused

        async def fresh() -> object:
            return 43

        reader = asyncio.create_task(cache.read(CALL, PLAN, fresh, CLIENT_OPTIONS))
        await paused.entered.wait()
        failure = Phase()
        failure.proceed.set()
        failure.error = BackendUnavailable('failed fence')
        fixture.l2.before['change'] = failure
        assert not await cache.invalidate((GenerationChange(PLAN.tags[0]),))
        paused.proceed.set()
        assert await reader == 43
        await cache.close()

    asyncio.run(scenario())
