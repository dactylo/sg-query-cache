import asyncio
from dataclasses import replace

import pytest

import sg_query_cache._async_cache as cache_module
from sg_query_cache._backend import BackendUnavailable
from sg_query_cache._model import GenerationChange
from tests._async_fakes import _Phase
from tests.cache._async_cases import (
    _CALL,
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


def test_cached_source_age_cannot_be_extended_by_promotion_wait(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        fixture = _fixture('hierarchy')
        monkeypatch.setattr(cache_module, 'monotonic', fixture._clock)
        assert fixture._l1 is not None
        assert await _read(fixture._cache) == 42
        await fixture._l1.flush_site('production', 'test')
        phase = _Phase()
        fixture._l1._before['reserve'] = phase

        async def fresh() -> object:
            return 43

        reader = asyncio.create_task(fixture._cache.read(_CALL, _PLAN, fresh, _OPTIONS))
        await phase._entered.wait()
        fixture._clock._now += 601
        phase._proceed.set()
        assert await reader == 43
        await fixture._cache.close()

    asyncio.run(scenario())


def test_failed_l2_is_retained_before_l1_generation_change_reply() -> None:
    async def scenario() -> None:
        fixture = _fixture('hierarchy')
        cache = fixture._cache
        assert fixture._l1 is not None
        assert fixture._l2 is not None
        assert await _read(cache) == 42
        failure = _Phase()
        failure._error = BackendUnavailable('L2 unavailable')
        failure._proceed.set()
        paused = _Phase()
        fixture._l2._before['change'] = failure
        fixture._l1._after['change'] = paused
        fence = asyncio.create_task(cache.invalidate((GenerationChange(_PLAN.tags[0]),)))
        await paused._entered.wait()
        assert cache._fences['l2']._retained
        assert not cache._fences['l1']._retained
        before = fixture._l2._calls.count('lookup')

        async def fresh() -> object:
            return 43

        assert await cache.read(_CALL, _PLAN, fresh, _OPTIONS) == 43
        assert fixture._l2._calls.count('lookup') == before
        assert fixture._l1._calls.count('commit') == 1
        paused._proceed.set()
        assert not await fence
        await cache.close()

    asyncio.run(scenario())


def test_generation_change_during_fetch_keeps_original_l1_key() -> None:
    async def scenario() -> None:
        fixture = _fixture('hierarchy')
        assert fixture._l1 is not None
        calls = 0
        local = fixture._l1

        async def fetch() -> object:
            nonlocal calls
            calls += 1
            if calls == 1:
                await local.flush_site('production', 'test')
            return calls

        cache = fixture._cache
        assert await cache.read(_CALL, _PLAN, fetch, _OPTIONS) == 1
        before = fixture._l2._calls.count('lookup') if fixture._l2 is not None else 0
        assert await cache.read(_CALL, _PLAN, fetch, _OPTIONS) == 1
        assert calls == 1
        assert fixture._l2 is not None
        assert fixture._l2._calls.count('lookup') == before + 1
        await cache.close()

    asyncio.run(scenario())


def test_memo_revalidation_rechecks_first_layer_after_shared_await(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        fixture = _fixture('hierarchy')
        cache = fixture._cache
        assert fixture._l1 is not None
        assert fixture._l2 is not None
        monkeypatch.setattr(cache_module, 'monotonic', fixture._clock)
        plan = replace(_PLAN, classification='schema')
        assert await cache.read_parsed(_CALL, plan, _answer, _OPTIONS, _parse) == ('parsed', 42)
        fixture._clock._now += 11
        paused = _Phase()
        fixture._l2._after['resolve'] = paused

        async def fresh() -> object:
            return 43

        reader = asyncio.create_task(cache.read_parsed(_CALL, plan, fresh, _OPTIONS, _parse))
        await paused._entered.wait()
        failure = _Phase()
        failure._error = BackendUnavailable('fence')
        failure._proceed.set()
        fixture._l1._before['change'] = failure
        assert not await cache.invalidate((GenerationChange(_PLAN.tags[0]),))
        paused._proceed.set()
        assert await reader == ('parsed', 43)
        await cache.close()

    asyncio.run(scenario())


@pytest.mark.parametrize('mode', ['hit', 'promotion', 'memo'])
def test_paused_successful_fence_keeps_eligible_hits_promotions_and_memo(mode: str) -> None:
    async def scenario() -> None:
        fixture = _fixture('hierarchy')
        cache = fixture._cache
        assert fixture._l1 is not None
        assert fixture._l2 is not None
        assert await _read(cache) == 42
        if mode == 'promotion':
            await fixture._l1.flush_site('production', 'test')
        elif mode == 'memo':
            assert await cache.read_parsed(_CALL, _PLAN, _answer, _OPTIONS, _parse) == (
                'parsed',
                42,
            )
        phase = _Phase()
        fixture._l2._before['change'] = phase
        fence = asyncio.create_task(cache.invalidate((GenerationChange(_PLAN.tags[0]),)))
        await phase._entered.wait()
        assert all(not state._retained for state in cache._fences.values())

        async def forbidden() -> object:
            raise AssertionError('successful pending fence triggered SG')

        if mode == 'memo':
            assert await cache.read_parsed(_CALL, _PLAN, forbidden, _OPTIONS, _parse) == (
                'parsed',
                42,
            )
        else:
            assert await cache.read(_CALL, _PLAN, forbidden, _OPTIONS) == 42
        if mode == 'promotion':
            assert fixture._l1._calls.count('commit') == 2
        phase._proceed.set()
        assert await fence
        await cache.close()

    asyncio.run(scenario())


@pytest.mark.parametrize('operation', ['reserve', 'commit'])
def test_promotion_rechecks_source_fence_after_await(operation: str) -> None:
    async def scenario() -> None:
        fixture = _fixture('hierarchy')
        cache = fixture._cache
        assert fixture._l1 is not None
        assert fixture._l2 is not None
        assert await _read(cache) == 42
        await fixture._l1.flush_site('production', 'test')
        paused = _Phase()
        fixture._l1._before[operation] = paused

        async def fresh() -> object:
            return 43

        reader = asyncio.create_task(cache.read(_CALL, _PLAN, fresh, _OPTIONS))
        await paused._entered.wait()
        failure = _Phase()
        failure._proceed.set()
        failure._error = BackendUnavailable('failed fence')
        fixture._l2._before['change'] = failure
        assert not await cache.invalidate((GenerationChange(_PLAN.tags[0]),))
        paused._proceed.set()
        assert await reader == 43
        await cache.close()

    asyncio.run(scenario())
