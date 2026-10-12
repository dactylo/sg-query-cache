import asyncio
from collections.abc import Awaitable, Callable, Coroutine
from typing import Any, TypeVar

import pytest

import sg_query_cache._async_cache as cache_module
import sg_query_cache._async_support as support
from sg_query_cache._async_local import AsyncLocalBackend
from sg_query_cache._backend import BackendUnavailable
from sg_query_cache._cache_common import WaitRecord
from sg_query_cache._model import GenerationChange
from tests._async_fakes import _Phase
from tests.cache._async_cases import (
    _CALL,
    _OPTIONS,
    _PLAN,
    _answer,
    _Fixture,
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

_T = TypeVar('_T')


async def _assert_close_outcomes(
    first: asyncio.Task[None], ordinary: list[asyncio.Task[None]], cancel_caller: bool
) -> None:
    if cancel_caller:
        with pytest.raises(asyncio.CancelledError, match='first caller'):
            await first
        assert first.cancelling() == 2
    else:
        with pytest.raises(BackendUnavailable):
            await first
        assert first.cancelling() == 0
    for task in ordinary:
        with pytest.raises(BackendUnavailable):
            await task
        assert task.cancelling() == 0


async def _assert_closed_backends(fixture: _Fixture) -> None:
    assert fixture._cache._admission.closed
    assert fixture._l1 is not None
    assert fixture._l2 is not None
    for backend in (fixture._l1, fixture._l2):
        assert backend._calls.count('close') == 1
        with pytest.raises(BackendUnavailable):
            await backend._backend.resolve(_PLAN.tags)
    await fixture._cache.close()


def test_cancelled_local_waiter_records_a_cancelled_wait() -> None:
    async def scenario() -> None:
        fixture = _fixture()
        phase = _Phase()

        async def fetch() -> object:
            await phase._wait()
            return 42

        leader = asyncio.create_task(fixture._cache.read(_CALL, _PLAN, fetch, _OPTIONS))
        await asyncio.wait_for(phase._entered.wait(), timeout=2)
        follower = asyncio.create_task(_read(fixture._cache))
        await _turn()
        follower.cancel()
        with pytest.raises(asyncio.CancelledError):
            await follower
        phase._proceed.set()
        assert await leader == 42
        await fixture._cache.close()
        events = fixture._metrics._events
        assert events.count(('cache_collapse', None, {'outcome': 'local'})) == 1
        waits = [event for event in events if event[0] == 'cache_wait_seconds']
        assert len(waits) == 1
        assert waits[0][2] == {'outcome': 'cancelled'}

    asyncio.run(scenario())


def test_close_rejects_direct_owner_and_other_loop_even_after_closed() -> None:
    fixture = _fixture()

    async def scenario() -> None:
        with fixture._cache.admit(), pytest.raises(RuntimeError, match='own admitted'):
            await fixture._cache.close()
        await fixture._cache.close()

    asyncio.run(scenario())
    with pytest.raises(RuntimeError, match='different event loop'):
        asyncio.run(fixture._cache.close())


@pytest.mark.parametrize(('pre_entry', 'blocks'), [(False, True), (True, True), (True, False)])
def test_close_resubmits_only_unstarted_backend_with_original_expired_deadline(
    pre_entry: bool,
    blocks: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        fixture = _fixture('hierarchy')
        assert fixture._l1 is not None
        if blocks:
            fixture._l1._after['close'] = _Phase()
        deadline = asyncio.get_running_loop().time()
        submissions: list[float | None] = []
        budget_calls: list[float] = []
        original_join = support.joined_cleanup

        def budget() -> float:
            budget_calls.append(deadline)
            return deadline

        def cancel_before_entry(coro: Coroutine[Any, Any, _T]) -> asyncio.Task[_T]:
            task = asyncio.create_task(coro)
            if pre_entry:
                task.cancel('unstarted cleanup')
            else:
                asyncio.get_running_loop().call_soon(task.cancel, 'entered cleanup')
            return task

        async def submit(
            action: Callable[[], Awaitable[_T]], *, deadline: float | None = None
        ) -> support.CleanupOutcome[_T]:
            submissions.append(deadline)
            if len(submissions) == 1:
                with monkeypatch.context() as patch:
                    patch.setattr(support, 'create_task', cancel_before_entry)
                    return await original_join(action, deadline=deadline)
            return await original_join(action, deadline=deadline)

        monkeypatch.setattr(fixture._cache, '_deadline', budget)
        monkeypatch.setattr(support, 'joined_cleanup', submit)
        with pytest.raises(BackendUnavailable, match='cleanup task cancelled'):
            await fixture._cache.close()
        assert submissions == [deadline] * (3 if pre_entry else 2)
        assert budget_calls == [deadline, deadline]
        await _assert_closed_backends(fixture)

    asyncio.run(scenario())


@pytest.mark.parametrize('cancel_caller', [False, True])
def test_backend_close_deadlines_attempt_all_layers_and_record_timeouts(
    cancel_caller: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def scenario() -> None:
        fixture = _fixture('hierarchy')
        assert fixture._l1 is not None
        assert fixture._l2 is not None
        first, second = _Phase(), _Phase()
        fixture._l1._after['close'] = first
        fixture._l2._after['close'] = second
        deadlines: list[float] = []

        def expired_deadline() -> float:
            deadline = asyncio.get_running_loop().time()
            deadlines.append(deadline)
            return deadline

        monkeypatch.setattr(fixture._cache, '_deadline', expired_deadline)
        closer = asyncio.create_task(fixture._cache.close())
        await first._entered.wait()
        if cancel_caller:
            closer.cancel('caller')
            with pytest.raises(asyncio.CancelledError, match='caller'):
                await closer
            assert closer.cancelling() == 1
        else:
            with pytest.raises(BackendUnavailable, match='deadline expired'):
                await closer
        assert first._cancelled.is_set()
        assert second._cancelled.is_set()
        assert len(deadlines) == 2
        assert deadlines[1] > deadlines[0]
        assert (
            fixture._metrics._events.count(
                ('backend_failure', None, {'reason': 'finalization_timeout'})
            )
            == 2
        )
        await _assert_closed_backends(fixture)

    asyncio.run(scenario())


@pytest.mark.parametrize('where', ['before', 'after'])
@pytest.mark.parametrize('operation', ['resolve', 'lookup', 'reserve', 'commit', 'release'])
def test_backend_phase_cancellation_never_retries_or_populates_after_cancel(
    operation: str, where: str
) -> None:
    async def scenario() -> None:
        fixture = _fixture()
        backend = fixture._l1
        assert backend is not None
        assert isinstance(backend._backend, AsyncLocalBackend)
        phase = _Phase()
        getattr(backend, f'_{where}')[operation] = phase
        calls = 0

        async def fetch() -> object:
            nonlocal calls
            calls += 1
            return calls

        owner = asyncio.create_task(fixture._cache.read(_CALL, _PLAN, fetch, _OPTIONS))
        await phase._entered.wait()
        owner.cancel('original')
        if operation == 'release':
            await _turn()
            owner.cancel('again')
            assert not phase._cancelled.is_set()
            phase._proceed.set()
        with pytest.raises(asyncio.CancelledError, match='original'):
            await owner
        assert calls == (1 if operation in {'commit', 'release'} else 0)
        assert fixture._cache._flights == {}
        assert not fixture._cache._admission._tokens
        backend._before.clear()
        backend._after.clear()
        if operation == 'reserve' and where == 'after':
            assert backend._backend._store.lease_bytes > 0
            fixture._clock._now += 31
        else:
            assert backend._backend._store.lease_bytes == 0
        result = await fixture._cache.read(_CALL, _PLAN, fetch, _OPTIONS)
        if operation == 'release' or (operation == 'commit' and where == 'after'):
            assert result == 1
            assert calls == 1
        else:
            assert result == calls
        await fixture._cache.close()

    asyncio.run(scenario())


@pytest.mark.parametrize('parsed', [False, True])
def test_cancelled_leader_notifies_three_followers_and_one_replacement_fetch(parsed: bool) -> None:
    async def scenario() -> None:
        fixture = _fixture()
        phase = _Phase()
        phase._cleanup_proceed.clear()
        replacement = _Phase()
        calls = 0

        async def fetch() -> object:
            nonlocal calls
            calls += 1
            if calls == 1:
                await phase._wait()
            else:
                await replacement._wait()
            return calls

        async def read() -> object:
            if parsed:
                return await fixture._cache.read_parsed(_CALL, _PLAN, fetch, _OPTIONS, _parse)
            return await fixture._cache.read(_CALL, _PLAN, fetch, _OPTIONS)

        leader = asyncio.create_task(read())
        await phase._entered.wait()
        followers = [asyncio.create_task(read()) for _ in range(3)]
        await _turn()
        leader.cancel('leader')
        await phase._cancelled.wait()
        await _turn()
        assert calls == 1
        phase._cleanup_proceed.set()
        with pytest.raises(asyncio.CancelledError, match='leader'):
            await leader
        await replacement._entered.wait()
        await _turn()
        assert calls == 2
        replacement._proceed.set()
        results = await asyncio.gather(*followers)
        assert results == ([('parsed', 2)] * 3 if parsed else [2] * 3)
        assert calls == 2
        cancelled = [
            event
            for event in fixture._metrics._events
            if event[0] == 'cache_fill' and event[2].get('outcome') == 'cancelled'
        ]
        assert len(cancelled) == 1
        assert fixture._cache._flights == {}
        await fixture._cache.close()

    asyncio.run(scenario())


@pytest.mark.parametrize('parsed', [False, True])
def test_cancelled_follower_cannot_cancel_leader_notification(parsed: bool) -> None:
    async def scenario() -> None:
        fixture = _fixture()
        phase = _Phase()
        calls = 0

        async def fetch() -> object:
            nonlocal calls
            calls += 1
            await phase._wait()
            return 42

        async def read() -> object:
            if parsed:
                return await fixture._cache.read_parsed(_CALL, _PLAN, fetch, _OPTIONS, _parse)
            return await fixture._cache.read(_CALL, _PLAN, fetch, _OPTIONS)

        leader = asyncio.create_task(read())
        await phase._entered.wait()
        follower = asyncio.create_task(read())
        await _turn()
        follower.cancel('follower')
        with pytest.raises(asyncio.CancelledError, match='follower'):
            await follower
        assert not leader.done()
        assert all(not flight.cancelled() for flight in fixture._cache._flights.values())
        phase._proceed.set()
        assert await leader == (('parsed', 42) if parsed else 42)
        assert calls == 1
        await fixture._cache.close()

    asyncio.run(scenario())


def test_context_cancellation_precedes_later_close_cancellation_and_failure() -> None:
    async def scenario() -> None:
        fixture = _fixture()
        assert fixture._l1 is not None
        close = _Phase()
        close._error = ValueError('close error')
        fixture._l1._before['close'] = close
        body = _Phase()

        async def owner() -> None:
            async with fixture._cache:
                await body._wait()

        task = asyncio.create_task(owner())
        await body._entered.wait()
        task.cancel('body cancellation')
        await close._entered.wait()
        task.cancel('close cancellation')
        close._proceed.set()
        with pytest.raises(asyncio.CancelledError, match='body cancellation'):
            await task
        assert fixture._cache._admission.closed
        assert task.cancelling() == 2

    asyncio.run(scenario())


def test_context_manager_body_error_precedes_close_failure() -> None:
    async def scenario() -> None:
        fixture = _fixture()
        assert fixture._l1 is not None
        phase = _Phase()
        phase._proceed.set()
        phase._error = ValueError('close failed')
        fixture._l1._before['close'] = phase
        with pytest.raises(LookupError, match='body failed'):
            async with fixture._cache:
                raise LookupError('body failed')
        assert fixture._cache._admission.closed
        assert ('backend_failure', None, {'reason': 'backend_failure'}) in fixture._metrics._events

    asyncio.run(scenario())


def test_fence_timeout_retains_unvisited_l1_before_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    async def scenario() -> None:
        fixture = _fixture('hierarchy')
        assert fixture._l2 is not None
        phase = _Phase()
        fixture._l2._before['change'] = phase
        monkeypatch.setattr(fixture._cache, '_deadline', lambda: asyncio.get_running_loop().time())
        assert not await fixture._cache.invalidate((GenerationChange(_PLAN.tags[0]),))
        assert phase._cancelled.is_set()
        assert fixture._l1 is not None
        assert 'change' not in fixture._l1._calls
        assert all(state._retained for state in fixture._cache._fences.values())
        assert (
            'backend_failure',
            None,
            {'reason': 'finalization_timeout'},
        ) in fixture._metrics._events
        await fixture._cache.close()

    asyncio.run(scenario())


def test_inline_close_delivers_pending_caller_request_before_next_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        fixture = _fixture('hierarchy')
        assert fixture._l1 is not None
        caller = asyncio.current_task()
        assert caller is not None
        original_close = fixture._l1.close

        async def pending_request() -> None:
            await original_close()
            caller.cancel('pending caller')
            raise ValueError('first resource failed before delivery')

        def cancel_shared(coro: Coroutine[Any, Any, _T]) -> asyncio.Task[_T]:
            task = asyncio.create_task(coro)
            task.cancel('shared close')
            return task

        monkeypatch.setattr(fixture._l1, 'close', pending_request)
        monkeypatch.setattr(cache_module, 'create_task', cancel_shared)
        with pytest.raises(asyncio.CancelledError, match='pending caller'):
            await fixture._cache.close()
        assert caller.cancelling() == 1
        await _assert_closed_backends(fixture)

    asyncio.run(scenario())


def test_inline_close_owner_preserves_original_cancellation_through_backend_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        fixture = _fixture('hierarchy')
        assert fixture._l1 is not None
        entered, cancelled, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
        received: list[asyncio.CancelledError] = []
        original_close = fixture._l1.close
        original_shield = support.shield

        async def observe_cancellation(future: asyncio.Future[None]) -> None:
            try:
                await original_shield(future)
            except asyncio.CancelledError as error:
                received.append(error)
                cancelled.set()
                raise

        async def cooperative_close() -> None:
            await original_close()
            entered.set()
            await release.wait()

        def cancel_shared(coro: Coroutine[Any, Any, _T]) -> asyncio.Task[_T]:
            task = asyncio.create_task(coro)
            task.cancel('shared close')
            return task

        monkeypatch.setattr(fixture._l1, 'close', cooperative_close)
        monkeypatch.setattr(cache_module, 'create_task', cancel_shared)
        monkeypatch.setattr(support, 'shield', observe_cancellation)
        owner = asyncio.create_task(fixture._cache.close())
        ordinary = [asyncio.create_task(fixture._cache.close()) for _ in range(2)]
        await asyncio.wait_for(entered.wait(), timeout=2)
        owner.cancel('first caller')
        await cancelled.wait()
        owner.cancel('later caller')
        await _turn()
        assert not owner.done()
        assert all(not task.done() for task in ordinary)
        release.set()
        with pytest.raises(asyncio.CancelledError, match='first caller') as raised:
            await owner
        assert raised.value is received[0]
        assert owner.cancelling() == 2
        for task in ordinary:
            with pytest.raises(BackendUnavailable):
                await task
        await _assert_closed_backends(fixture)

    asyncio.run(scenario())


def test_inline_close_timeout_does_not_replace_actual_caller_cancellation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        fixture = _fixture('hierarchy')
        assert fixture._l1 is not None
        expired, finish = asyncio.Event(), asyncio.Event()
        observed: list[asyncio.CancelledError] = []
        original_close = fixture._l1.close
        original_shield = support.shield

        async def observe_cancellation(future: asyncio.Future[None]) -> None:
            try:
                await original_shield(future)
            except asyncio.CancelledError as error:
                if error.args == ('actual caller',):
                    observed.append(error)
                raise

        async def cooperative_close() -> None:
            await original_close()
            try:
                await finish.wait()
            except asyncio.CancelledError:
                expired.set()
                await support.joined_cleanup(finish.wait)
                raise

        def cancel_shared(coro: Coroutine[Any, Any, _T]) -> asyncio.Task[_T]:
            task = asyncio.create_task(coro)
            task.cancel('shared close')
            return task

        monkeypatch.setattr(fixture._l1, 'close', cooperative_close)
        monkeypatch.setattr(fixture._cache, '_deadline', asyncio.get_running_loop().time)
        monkeypatch.setattr(cache_module, 'create_task', cancel_shared)
        monkeypatch.setattr(support, 'shield', observe_cancellation)
        closer = asyncio.create_task(fixture._cache.close())
        ordinary = asyncio.create_task(fixture._cache.close())
        await asyncio.wait_for(expired.wait(), timeout=2)
        closer.cancel('actual caller')
        await _turn()
        assert not closer.done()
        finish.set()
        with pytest.raises(asyncio.CancelledError, match='actual caller') as raised:
            await asyncio.wait_for(closer, timeout=2)
        assert raised.value is observed[0]
        assert closer.cancelling() == 1
        with pytest.raises(BackendUnavailable):
            await ordinary
        await _assert_closed_backends(fixture)

    asyncio.run(scenario())


def test_lease_sleep_cancellation_does_not_fetch_or_invent_ownership(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        fixture = _fixture('shared')
        backend = fixture._l2
        assert backend is not None
        await _read(fixture._cache)
        assert fixture._cache._memo is None
        await backend.flush_site('production', 'test')
        vector = await backend.resolve(_PLAN.tags)
        base = cache_module.KeyBuilder.base_key(
            'production', cache_module.client_binding(fixture._cache)[1], _OPTIONS, _CALL
        )
        key = cache_module.KeyBuilder.qualified_key(base, vector, 'result')
        held = await backend.reserve(key, 30)
        assert held is not None
        sleeping = asyncio.Event()
        real_sleep = asyncio.sleep

        async def observed_sleep(delay: float) -> None:
            if delay > 0:
                sleeping.set()
            await real_sleep(delay)

        monkeypatch.setattr(cache_module, 'sleep', observed_sleep)
        calls = 0

        async def fetch() -> object:
            nonlocal calls
            calls += 1
            return 43

        reader = asyncio.create_task(fixture._cache.read(_CALL, _PLAN, fetch, _OPTIONS))
        await sleeping.wait()
        reader.cancel('lease wait')
        with pytest.raises(asyncio.CancelledError, match='lease wait'):
            await reader
        assert calls == 0
        assert await backend.release(held)
        await fixture._cache.close()

    asyncio.run(scenario())


def test_owning_context_rejects_direct_self_close_without_changing_owner() -> None:
    async def scenario() -> None:
        fixture = _fixture()
        with fixture._cache.admit(), pytest.raises(RuntimeError, match='own admitted'):
            async with fixture._cache:
                assert await _read(fixture._cache) == 42
        await fixture._cache.close()

    asyncio.run(scenario())


def test_precommit_cancellation_skips_population_and_preserves_successful_fill() -> None:
    async def scenario() -> None:
        fixture = _fixture()
        owner = asyncio.current_task()
        assert owner is not None

        def cancel_after_sg() -> None:
            if fixture._metrics._events[-1][0] == 'cache_fill':
                owner.cancel('precommit')

        fixture._metrics._callback = cancel_after_sg
        with pytest.raises(asyncio.CancelledError, match='precommit'):
            await _read(fixture._cache)
        assert fixture._l1 is not None
        assert isinstance(fixture._l1._backend, AsyncLocalBackend)
        assert 'commit' not in fixture._l1._calls
        assert 'release' in fixture._l1._calls
        assert ('cache_fill', None, {'layer': 'local_l1'}) in fixture._metrics._events
        assert not any(
            event[2].get('outcome') == 'cancelled' for event in fixture._metrics._events
        )
        fixture._metrics._callback = None
        await fixture._cache.close()

    asyncio.run(scenario())


@pytest.mark.parametrize('operation', ['reserve', 'commit'])
@pytest.mark.parametrize('where', ['before', 'after'])
def test_promotion_cancellation_releases_known_lease_and_keeps_completed_store(
    operation: str, where: str
) -> None:
    async def scenario() -> None:
        fixture = _fixture('hierarchy')
        assert fixture._l1 is not None
        assert isinstance(fixture._l1._backend, AsyncLocalBackend)
        phase = _Phase()
        getattr(fixture._l1, f'_{where}')[operation] = phase
        owner = asyncio.create_task(_read(fixture._cache))
        await phase._entered.wait()
        owner.cancel('promotion')
        with pytest.raises(asyncio.CancelledError, match='promotion'):
            await owner
        assert fixture._l2 is not None
        assert isinstance(fixture._l2._backend, AsyncLocalBackend)
        assert fixture._l2._backend._store.result_bytes > 0
        if operation == 'reserve' and where == 'after':
            assert fixture._l1._backend._store.lease_bytes > 0
        else:
            assert fixture._l1._backend._store.lease_bytes == 0
        if operation == 'commit' and where == 'after':
            assert fixture._l1._backend._store.result_bytes > 0
        fixture._l1._before.clear()
        fixture._l1._after.clear()
        await fixture._cache.close()

    asyncio.run(scenario())


@pytest.mark.parametrize('parsed', [False, True])
def test_bypass_cancellation_is_delivered_before_sg_dispatch(parsed: bool) -> None:
    async def scenario() -> None:
        fixture = _fixture()
        owner = asyncio.current_task()
        assert owner is not None
        calls = 0

        async def fetch() -> object:
            nonlocal calls
            calls += 1
            return 42

        def cancel() -> None:
            fixture._metrics._callback = None
            owner.cancel('before dispatch')

        fixture._metrics._callback = cancel

        async def read() -> object:
            if parsed:
                return await fixture._cache.read_parsed(_CALL, _PLAN, fetch, _OPTIONS, _parse)
            return await fixture._cache.read(_CALL, _PLAN, fetch, _OPTIONS)

        with (
            fixture._cache.bypass(),
            pytest.raises(asyncio.CancelledError, match='before dispatch'),
        ):
            await read()
        assert calls == 0
        await fixture._cache.close()

    asyncio.run(scenario())


@pytest.mark.parametrize('parsed', [False, True])
def test_re_elected_followers_keep_one_deadline_and_count_once(
    parsed: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def scenario() -> None:
        fixture = _fixture()
        first = _Phase()
        second = _Phase()
        deadlines: dict[str, list[float]] = {}
        original = fixture._cache._wait_flight

        async def observed(
            flight: asyncio.Future[None], deadline: float, waits: WaitRecord
        ) -> bool:
            task = asyncio.current_task()
            assert task is not None
            deadlines.setdefault(task.get_name(), []).append(deadline)
            return await original(flight, deadline, waits)

        monkeypatch.setattr(fixture._cache, '_wait_flight', observed)
        calls = 0

        async def fetch() -> object:
            nonlocal calls
            calls += 1
            await (first if calls == 1 else second)._wait()
            return calls

        async def read() -> object:
            if parsed:
                return await fixture._cache.read_parsed(_CALL, _PLAN, fetch, _OPTIONS, _parse)
            return await fixture._cache.read(_CALL, _PLAN, fetch, _OPTIONS)

        leader = asyncio.create_task(read())
        await first._entered.wait()
        followers = [asyncio.create_task(read(), name=f'follower-{number}') for number in range(3)]
        await _turn()
        leader.cancel('replace')
        with pytest.raises(asyncio.CancelledError, match='replace'):
            await leader
        await asyncio.wait_for(second._entered.wait(), timeout=2)
        await _turn()
        repeated = [values for values in deadlines.values() if len(values) > 1]
        assert len(repeated) == 2
        assert all(len(set(values)) == 1 for values in repeated)
        second._proceed.set()
        await asyncio.gather(*followers)
        assert calls == 2
        await fixture._cache.close()
        events = fixture._metrics._events
        assert events.count(('cache_collapse', None, {'outcome': 'local'})) == 3
        waits = [event for event in events if event[0] == 'cache_wait_seconds']
        assert len(waits) == 3
        assert all(event[2] == {'outcome': 'completed'} for event in waits)

    asyncio.run(scenario())


@pytest.mark.parametrize('second_failure', [None, ValueError('second backend failure')])
def test_independently_cancelled_backend_cleanup_still_closes_other_layer(
    second_failure: Exception | None,
) -> None:
    async def scenario() -> None:
        fixture = _fixture('hierarchy')
        assert fixture._l1 is not None
        assert fixture._l2 is not None
        phase = _Phase()

        def cancel_inner() -> None:
            inner = asyncio.current_task()
            assert inner is not None
            inner.cancel('inner backend close')

        phase._on_enter = cancel_inner
        fixture._l1._after['close'] = phase
        later = _Phase()
        later._error = second_failure
        later._proceed.set()
        fixture._l2._after['close'] = later
        with pytest.raises(BackendUnavailable, match='cleanup task cancelled'):
            await fixture._cache.close()
        await _assert_closed_backends(fixture)

    asyncio.run(scenario())


@pytest.mark.parametrize('prebody', [False, True])
@pytest.mark.parametrize('cancel_caller', [False, True])
def test_interrupted_shared_close_settles_real_backends_and_concurrent_waiters(
    prebody: bool, cancel_caller: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def scenario() -> None:
        fixture = _fixture('hierarchy')
        assert fixture._l1 is not None
        assert fixture._l2 is not None
        operation = _Phase()
        cleanup = _Phase()
        fixture._l1._after['close'] = cleanup
        waiting = asyncio.Event()
        shared_tasks: list[asyncio.Task[object]] = []
        original_idle = fixture._cache._admission.wait_idle

        async def observed_idle() -> None:
            waiting.set()
            await original_idle()

        def create(coro: Coroutine[Any, Any, _T]) -> asyncio.Task[_T]:
            task = asyncio.create_task(coro)
            shared_tasks.append(task)
            if prebody:
                task.cancel('shared close')
            return task

        async def admitted() -> None:
            with fixture._cache.admit():
                await operation._wait()

        monkeypatch.setattr(fixture._cache._admission, 'wait_idle', observed_idle)
        monkeypatch.setattr(cache_module, 'create_task', create)
        active = asyncio.create_task(admitted())
        await operation._entered.wait()
        first = asyncio.create_task(fixture._cache.close())
        ordinary = [asyncio.create_task(fixture._cache.close()) for _ in range(2)]
        # A pre-entry cancellation must transfer the active join to its waiter.
        if not prebody:
            await waiting.wait()
            shared_tasks[0].cancel('shared close')
        await _turn()
        if cancel_caller:
            first.cancel('first caller')
            await _turn()
            first.cancel('later caller')
        assert not cleanup._entered.is_set()
        assert not first.done()
        assert all(not task.done() for task in ordinary)
        with pytest.raises(RuntimeError, match='closing'):
            await _read(fixture._cache)
        operation._proceed.set()
        await active
        await asyncio.wait_for(cleanup._entered.wait(), timeout=2)
        assert not fixture._cache._admission.closed
        cleanup._proceed.set()
        await _assert_close_outcomes(first, ordinary, cancel_caller)
        assert len(shared_tasks) == 1
        await _assert_closed_backends(fixture)

    asyncio.run(scenario())


@pytest.mark.parametrize('stale_request', [False, True])
def test_inline_close_keeps_independent_child_cancellation_as_shared_failure(
    stale_request: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def scenario() -> None:
        fixture = _fixture('hierarchy')
        assert fixture._l1 is not None
        original_close = fixture._l1.close
        caller = asyncio.current_task()
        assert caller is not None
        if stale_request:
            caller.cancel('handled earlier')
            with pytest.raises(asyncio.CancelledError, match='handled earlier'):
                await asyncio.sleep(0)

        async def independent_close() -> None:
            await original_close()
            child = asyncio.create_task(asyncio.sleep(0))
            child.cancel('independent resource')
            await child

        def cancel_shared(coro: Coroutine[Any, Any, _T]) -> asyncio.Task[_T]:
            task = asyncio.create_task(coro)
            task.cancel('shared close')
            return task

        monkeypatch.setattr(fixture._l1, 'close', independent_close)
        monkeypatch.setattr(cache_module, 'create_task', cancel_shared)
        with pytest.raises(BackendUnavailable, match='cleanup task cancelled'):
            await fixture._cache.close()
        assert caller.cancelling() == int(stale_request)
        await _assert_closed_backends(fixture)

    asyncio.run(scenario())


@pytest.mark.parametrize('cancel_caller', [False, True])
def test_fence_inner_prebody_cancel_is_settled_by_outer_owner(
    cancel_caller: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def scenario() -> None:
        fixture = _fixture('hierarchy')
        owner = asyncio.current_task()
        assert owner is not None

        def cancel_before_start(coro: Coroutine[Any, Any, _T]) -> asyncio.Task[_T]:
            inner = asyncio.create_task(coro)
            inner.cancel('inner')
            if cancel_caller:
                owner.cancel('caller')
            return inner

        with monkeypatch.context() as patch:
            patch.setattr(support, 'create_task', cancel_before_start)
            if cancel_caller:
                with pytest.raises(asyncio.CancelledError, match='caller'):
                    await fixture._cache.invalidate((GenerationChange(_PLAN.tags[0]),))
            else:
                assert not await fixture._cache.invalidate((GenerationChange(_PLAN.tags[0]),))
        assert all(state._retained for state in fixture._cache._fences.values())
        assert fixture._l1 is not None
        assert fixture._l2 is not None
        assert 'change' not in fixture._l1._calls
        assert 'change' not in fixture._l2._calls
        assert await _read(fixture._cache) == 42
        assert 'lookup' not in fixture._l1._calls
        assert 'lookup' not in fixture._l2._calls
        assert await fixture._cache.invalidate((GenerationChange(_PLAN.tags[0]),))
        assert all(not state._retained for state in fixture._cache._fences.values())
        await fixture._cache.close()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    'failure', [None, BackendUnavailable('backend close'), ValueError('ordinary close')]
)
def test_close_drains_whole_admission_attempts_all_backends_and_preserves_cancellation(
    failure: Exception | None,
) -> None:
    async def scenario() -> None:
        fixture = _fixture('hierarchy')
        operation = _Phase()
        cleanup = _Phase()
        cleanup._error = failure
        assert fixture._l1 is not None
        assert fixture._l2 is not None
        fixture._l1._before['close'] = cleanup

        async def admitted() -> None:
            with fixture._cache.admit() as token:
                await operation._wait()
                assert (
                    await fixture._cache.read_admitted(
                        _CALL, _PLAN, _answer, _OPTIONS, token=token
                    )
                    == 42
                )

        active = asyncio.create_task(admitted())
        await operation._entered.wait()
        closer = asyncio.create_task(fixture._cache.close())
        ordinary = asyncio.create_task(fixture._cache.close())
        await _turn()
        closer.cancel('first close')
        await _turn()
        closer.cancel('second close')
        assert not cleanup._entered.is_set()
        with pytest.raises(RuntimeError, match='closing'):
            await _read(fixture._cache)
        operation._proceed.set()
        await active
        await cleanup._entered.wait()
        assert not cleanup._cancelled.is_set()
        cleanup._proceed.set()
        with pytest.raises(asyncio.CancelledError, match='first close'):
            await closer
        if failure is None:
            await ordinary
        else:
            with pytest.raises(type(failure), match=str(failure)):
                await ordinary
        assert closer.cancelling() == 2
        assert fixture._l2._calls.count('close') == 1
        assert fixture._cache._admission.closed
        await fixture._cache.close()

    asyncio.run(scenario())


@pytest.mark.parametrize('timeout', [False, True])
def test_sg_cancellation_precedes_repeated_release_cancellation_and_cleanup_failure(
    timeout: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def scenario() -> None:
        fixture = _fixture()
        assert fixture._l1 is not None
        sg = _Phase()
        release = _Phase()
        release._cleanup_proceed.clear()
        release._error = BackendUnavailable('release failure')
        fixture._l1._before['release'] = release

        async def fetch() -> object:
            await sg._wait()
            return 42

        reader = asyncio.create_task(fixture._cache.read(_CALL, _PLAN, fetch, _OPTIONS))
        await sg._entered.wait()
        with monkeypatch.context() as patch:
            if timeout:
                patch.setattr(
                    fixture._cache, '_deadline', lambda: asyncio.get_running_loop().time()
                )
            reader.cancel('original SG cancellation')
            await release._entered.wait()
            if timeout:
                await release._cancelled.wait()
            reader.cancel('later release cancellation')
            await _turn()
            release._cleanup_proceed.set()
            release._proceed.set()
            with pytest.raises(asyncio.CancelledError, match='original SG cancellation'):
                await reader
        assert reader.cancelling() == 2
        assert fixture._cache._flights == {}
        assert (
            'cache_fill',
            None,
            {'layer': 'local_l1', 'outcome': 'cancelled'},
        ) in fixture._metrics._events
        assert (
            'backend_failure',
            None,
            {'reason': 'finalization_timeout' if timeout else 'backend_failure'},
        ) in fixture._metrics._events
        await fixture._cache.close()

    asyncio.run(scenario())
