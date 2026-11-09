import asyncio
from collections.abc import Awaitable, Callable, Coroutine
from dataclasses import replace
from typing import Any, TypeVar

import pytest

import sg_query_cache._async_cache as cache_module
import sg_query_cache._async_support as support
from sg_query_cache._async_local import AsyncLocalBackend
from sg_query_cache._backend import BackendProtocolError, BackendUnavailable
from sg_query_cache._cache_common import WaitRecord
from sg_query_cache._config import CacheSettings, L1Settings, L2Settings, LimitSettings
from sg_query_cache._model import FillLease, GenerationChange
from sg_query_cache.valkey_async import AsyncValkeyBackend, AsyncValkeyMetadataStore
from tests._async_fakes import Phase, PhaseBackend, RecordingMetrics
from tests._fakes import CLIENT_OPTIONS, FakeClock, turn
from tests._valkey_control_cases import RECORD_A
from tests.cache._async_cases import (
    CALL,
    IDENTITY,
    PLAN,
    Fixture,
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
from tests.valkey._async_helpers import BackendClient

_T = TypeVar('_T')


class _DeadlineCache:
    def __init__(self, clock: FakeClock, *, hierarchy: bool = False) -> None:
        self.settings = CacheSettings(
            l1=L1Settings(enabled=hierarchy),
            l2=L2Settings(enabled=True),
            limits=LimitSettings(operation_timeout_seconds=1, circuit_failure_threshold=1),
        )
        self.metrics = RecordingMetrics()
        self.result = BackendClient([])
        self.metadata = BackendClient(
            [
                ('info', {'role': 'master', 'master_replid': 'primary'}),
                ('get', RECORD_A),
                ('exists', 1),
                ('info', {'role': 'master', 'master_replid': 'primary'}),
                ('get', RECORD_A),
                ('exists', 1),
                *[('evalsha', [b'ok', 1]) for _ in range(4)],
            ]
        )
        # A longer backend budget exposes both missing publication and a second timer.
        backend_settings = replace(
            self.settings, limits=replace(self.settings.limits, operation_timeout_seconds=5)
        )
        self.store = AsyncValkeyMetadataStore(self.metadata, backend_settings, self.metrics)
        self.backend = AsyncValkeyBackend(self.result, self.store, backend_settings, self.metrics)
        self.backend._result_breaker._clock = clock
        self.store._breaker._clock = clock
        self.local = (
            PhaseBackend(AsyncLocalBackend(self.settings, clock, self.metrics))
            if hierarchy
            else None
        )
        self.cache = cache_module.AsyncQueryCache(
            self.settings, IDENTITY, l1=self.local, l2=self.backend, metrics=self.metrics
        )

    def failures(self, operation: str) -> list[dict[str, str]]:
        return [
            labels
            for name, _value, labels in self.metrics.events
            if name == 'backend_failure'
            and labels.get('backend_operation') == operation
            and labels.get('outcome') == 'failure'
        ]

    def timeouts(self) -> list[dict[str, str]]:
        return [
            labels
            for name, _value, labels in self.metrics.events
            if name == 'backend_failure' and labels.get('reason') == 'finalization_timeout'
        ]


class _DeadlineOperation:
    def __init__(self, clock: FakeClock, *, cooperative: bool) -> None:
        self.clock = clock
        self.cooperative = cooperative
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.calls = 0
        self.cancellations = 0

    async def __call__(self, _sha: str, _numkeys: int, *_args: object) -> object:
        self.calls += 1
        self.clock.advance(1)
        try:
            await turn()
            # Bound the failure if the cache deadline is ignored in favor of the backend budget.
            self.clock.advance(5)
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancellations += 1
            self.entered.set()
            if self.cooperative:
                try:
                    await self.release.wait()
                except asyncio.CancelledError:
                    self.cancellations += 1
                    raise
            raise


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


async def _assert_closed_backends(fixture: Fixture) -> None:
    assert fixture.cache._admission.closed
    assert fixture.l1 is not None
    assert fixture.l2 is not None
    for backend in (fixture.l1, fixture.l2):
        assert backend.calls.count('close') == 1
        with pytest.raises(BackendUnavailable):
            await backend.backend.resolve(PLAN.tags)
    await fixture.cache.close()


def test_async_cache_cleanup_failure_metric_carries_failure_class() -> None:
    fixture = _fixture()
    error = BackendUnavailable('redacted', failure_class='replica')

    assert fixture.cache._cleanup_error(error) is error
    assert (
        'backend_failure',
        None,
        {'reason': 'backend_failure', 'failure_class': 'replica'},
    ) in fixture.metrics.events


def test_cache_delayed_response_uses_only_backend_timer(monkeypatch: pytest.MonkeyPatch) -> None:
    async def scenario() -> None:
        clock = FakeClock()
        loop = asyncio.get_running_loop()
        monkeypatch.setattr(loop, 'time', clock)
        tested = _DeadlineCache(clock)
        response = loop.create_future()
        loop.call_at(clock() + 0.9, response.set_result, None)
        armed: list[float | None] = []
        original_timeout = cache_module.timeout_at

        def cache_timeout(deadline: float | None) -> asyncio.Timeout:
            armed.append(deadline)
            return original_timeout(deadline)

        async def delayed(_sha: str, _numkeys: int, *_args: object) -> object:
            # Response callback is due first, but expiry cancels its queued consumer.
            clock.advance(6)
            return await response

        monkeypatch.setattr(cache_module, 'timeout_at', cache_timeout)
        monkeypatch.setattr(tested.result, 'evalsha', delayed)
        assert await _read(tested.cache) == 42
        assert not armed
        assert len(tested.failures('lookup')) == 1
        assert len(tested.timeouts()) == 1
        await tested.cache.close()

    asyncio.run(scenario())


def test_cancelled_local_waiter_records_a_cancelled_wait() -> None:
    async def scenario() -> None:
        fixture = _fixture()
        phase = Phase()

        async def fetch() -> object:
            await phase.wait()
            return 42

        leader = asyncio.create_task(fixture.cache.read(CALL, PLAN, fetch, CLIENT_OPTIONS))
        await asyncio.wait_for(phase.entered.wait(), timeout=2)
        follower = asyncio.create_task(_read(fixture.cache))
        await turn()
        follower.cancel()
        with pytest.raises(asyncio.CancelledError):
            await follower
        phase.proceed.set()
        assert await leader == 42
        await fixture.cache.close()
        events = fixture.metrics.events
        assert events.count(('cache_collapse', None, {'outcome': 'local'})) == 1
        waits = [event for event in events if event[0] == 'cache_wait_seconds']
        assert len(waits) == 1
        assert waits[0][2] == {'outcome': 'cancelled'}

    asyncio.run(scenario())


def test_close_rejects_direct_owner_and_other_loop_even_after_closed() -> None:
    fixture = _fixture()

    async def scenario() -> None:
        with fixture.cache.admit(), pytest.raises(RuntimeError, match='own admitted'):
            await fixture.cache.close()
        await fixture.cache.close()

    asyncio.run(scenario())
    with pytest.raises(RuntimeError, match='different event loop'):
        asyncio.run(fixture.cache.close())


@pytest.mark.parametrize(('pre_entry', 'blocks'), [(False, True), (True, True), (True, False)])
def test_close_resubmits_only_unstarted_backend_with_original_expired_deadline(
    pre_entry: bool,
    blocks: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        fixture = _fixture('hierarchy')
        assert fixture.l1 is not None
        if blocks:
            fixture.l1.after['close'] = Phase()
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

        monkeypatch.setattr(fixture.cache, '_deadline', budget)
        monkeypatch.setattr(support, 'joined_cleanup', submit)
        with pytest.raises(BackendUnavailable, match='cleanup task cancelled'):
            await fixture.cache.close()
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
        assert fixture.l1 is not None
        assert fixture.l2 is not None
        first, second = Phase(), Phase()
        fixture.l1.after['close'] = first
        fixture.l2.after['close'] = second
        deadlines: list[float] = []

        def expired_deadline() -> float:
            deadline = asyncio.get_running_loop().time()
            deadlines.append(deadline)
            return deadline

        monkeypatch.setattr(fixture.cache, '_deadline', expired_deadline)
        closer = asyncio.create_task(fixture.cache.close())
        await first.entered.wait()
        if cancel_caller:
            closer.cancel('caller')
            with pytest.raises(asyncio.CancelledError, match='caller'):
                await closer
            assert closer.cancelling() == 1
        else:
            with pytest.raises(BackendUnavailable, match='deadline expired'):
                await closer
        assert first.cancelled.is_set()
        assert second.cancelled.is_set()
        assert len(deadlines) == 2
        assert deadlines[1] > deadlines[0]
        assert (
            fixture.metrics.events.count(
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
        backend = fixture.l1
        assert backend is not None
        assert isinstance(backend.backend, AsyncLocalBackend)
        phase = Phase()
        getattr(backend, where)[operation] = phase
        if operation == 'release':
            # A successful commit consumes its lease, so only an unstored fill releases one.
            rejected = Phase()
            rejected.error = BackendUnavailable('commit failed')
            rejected.proceed.set()
            backend.before['commit'] = rejected
        calls = 0

        async def fetch() -> object:
            nonlocal calls
            calls += 1
            return calls

        owner = asyncio.create_task(fixture.cache.read(CALL, PLAN, fetch, CLIENT_OPTIONS))
        await phase.entered.wait()
        owner.cancel('original')
        if operation == 'release':
            await turn()
            owner.cancel('again')
            assert not phase.cancelled.is_set()
            phase.proceed.set()
        with pytest.raises(asyncio.CancelledError, match='original'):
            await owner
        assert calls == (1 if operation in {'commit', 'release'} else 0)
        assert fixture.cache._flights == {}
        assert not fixture.cache._admission._tokens
        backend.before.clear()
        backend.after.clear()
        if operation == 'reserve' and where == 'after':
            assert (await backend.backend.storage_usage())['lease_bytes'] > 0
            fixture.clock.now += 31
        else:
            assert (await backend.backend.storage_usage())['lease_bytes'] == 0
        result = await fixture.cache.read(CALL, PLAN, fetch, CLIENT_OPTIONS)
        if operation == 'commit' and where == 'after':
            assert result == 1
            assert calls == 1
        else:
            assert result == calls
        await fixture.cache.close()

    asyncio.run(scenario())


@pytest.mark.parametrize('parsed', [False, True])
def test_cancelled_leader_notifies_three_followers_and_one_replacement_fetch(parsed: bool) -> None:
    async def scenario() -> None:
        fixture = _fixture()
        phase = Phase()
        phase.cleanup_proceed.clear()
        replacement = Phase()
        calls = 0

        async def fetch() -> object:
            nonlocal calls
            calls += 1
            if calls == 1:
                await phase.wait()
            else:
                await replacement.wait()
            return calls

        async def read() -> object:
            if parsed:
                return await fixture.cache.read_parsed(CALL, PLAN, fetch, CLIENT_OPTIONS, _parse)
            return await fixture.cache.read(CALL, PLAN, fetch, CLIENT_OPTIONS)

        leader = asyncio.create_task(read())
        await phase.entered.wait()
        followers = [asyncio.create_task(read()) for _ in range(3)]
        await turn()
        leader.cancel('leader')
        await phase.cancelled.wait()
        await turn()
        assert calls == 1
        phase.cleanup_proceed.set()
        with pytest.raises(asyncio.CancelledError, match='leader'):
            await leader
        await replacement.entered.wait()
        await turn()
        assert calls == 2
        replacement.proceed.set()
        results = await asyncio.gather(*followers)
        assert results == ([('parsed', 2)] * 3 if parsed else [2] * 3)
        assert calls == 2
        cancelled = [
            event
            for event in fixture.metrics.events
            if event[0] == 'cache_fill' and event[2].get('outcome') == 'cancelled'
        ]
        assert len(cancelled) == 1
        assert fixture.cache._flights == {}
        await fixture.cache.close()

    asyncio.run(scenario())


@pytest.mark.parametrize('parsed', [False, True])
def test_cancelled_follower_cannot_cancel_leader_notification(parsed: bool) -> None:
    async def scenario() -> None:
        fixture = _fixture()
        phase = Phase()
        calls = 0

        async def fetch() -> object:
            nonlocal calls
            calls += 1
            await phase.wait()
            return 42

        async def read() -> object:
            if parsed:
                return await fixture.cache.read_parsed(CALL, PLAN, fetch, CLIENT_OPTIONS, _parse)
            return await fixture.cache.read(CALL, PLAN, fetch, CLIENT_OPTIONS)

        leader = asyncio.create_task(read())
        await phase.entered.wait()
        follower = asyncio.create_task(read())
        await turn()
        follower.cancel('follower')
        with pytest.raises(asyncio.CancelledError, match='follower'):
            await follower
        assert not leader.done()
        assert all(not flight.cancelled() for flight in fixture.cache._flights.values())
        phase.proceed.set()
        assert await leader == (('parsed', 42) if parsed else 42)
        assert calls == 1
        await fixture.cache.close()

    asyncio.run(scenario())


def test_context_cancellation_precedes_later_close_cancellation_and_failure() -> None:
    async def scenario() -> None:
        fixture = _fixture()
        assert fixture.l1 is not None
        close = Phase()
        close.error = ValueError('close error')
        fixture.l1.before['close'] = close
        body = Phase()

        async def owner() -> None:
            async with fixture.cache:
                await body.wait()

        task = asyncio.create_task(owner())
        await body.entered.wait()
        task.cancel('body cancellation')
        await close.entered.wait()
        task.cancel('close cancellation')
        close.proceed.set()
        with pytest.raises(asyncio.CancelledError, match='body cancellation'):
            await task
        assert fixture.cache._admission.closed
        assert task.cancelling() == 2

    asyncio.run(scenario())


def test_context_manager_body_error_precedes_close_failure() -> None:
    async def scenario() -> None:
        fixture = _fixture()
        assert fixture.l1 is not None
        phase = Phase()
        phase.proceed.set()
        phase.error = ValueError('close failed')
        fixture.l1.before['close'] = phase
        with pytest.raises(LookupError, match='body failed'):
            async with fixture.cache:
                raise LookupError('body failed')
        assert fixture.cache._admission.closed
        assert ('backend_failure', None, {'reason': 'backend_failure'}) in fixture.metrics.events

    asyncio.run(scenario())


def test_custom_backend_keeps_cache_timer_and_single_expiry_metric(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        fixture = _fixture()
        assert fixture.l1 is not None
        stalled = Phase()
        fixture.l1.before['resolve'] = stalled
        monkeypatch.setattr(fixture.cache, '_deadline', asyncio.get_running_loop().time)
        assert await _read(fixture.cache) == 42
        assert stalled.cancelled.is_set()
        assert (
            fixture.metrics.events.count(
                ('backend_failure', None, {'reason': 'finalization_timeout'})
            )
            == 1
        )
        await fixture.cache.close()

    asyncio.run(scenario())


def test_fence_timeout_retains_unvisited_l1_before_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    async def scenario() -> None:
        fixture = _fixture('hierarchy')
        assert fixture.l2 is not None
        phase = Phase()
        fixture.l2.before['change'] = phase
        monkeypatch.setattr(fixture.cache, '_deadline', lambda: asyncio.get_running_loop().time())
        assert not await fixture.cache.invalidate((GenerationChange(PLAN.tags[0]),))
        assert phase.cancelled.is_set()
        assert fixture.l1 is not None
        assert 'change' not in fixture.l1.calls
        assert all(state.retained for state in fixture.cache._fences.values())
        assert (
            'backend_failure',
            None,
            {'reason': 'finalization_timeout'},
        ) in fixture.metrics.events
        await fixture.cache.close()

    asyncio.run(scenario())


def test_inline_close_delivers_pending_caller_request_before_next_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        fixture = _fixture('hierarchy')
        assert fixture.l1 is not None
        caller = asyncio.current_task()
        assert caller is not None
        original_close = fixture.l1.close

        async def pending_request() -> None:
            await original_close()
            caller.cancel('pending caller')
            raise ValueError('first resource failed before delivery')

        def cancel_shared(coro: Coroutine[Any, Any, _T]) -> asyncio.Task[_T]:
            task = asyncio.create_task(coro)
            task.cancel('shared close')
            return task

        monkeypatch.setattr(fixture.l1, 'close', pending_request)
        monkeypatch.setattr(cache_module, 'create_task', cancel_shared)
        with pytest.raises(asyncio.CancelledError, match='pending caller'):
            await fixture.cache.close()
        assert caller.cancelling() == 1
        await _assert_closed_backends(fixture)

    asyncio.run(scenario())


def test_inline_close_owner_preserves_original_cancellation_through_backend_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        fixture = _fixture('hierarchy')
        assert fixture.l1 is not None
        entered, cancelled, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
        received: list[asyncio.CancelledError] = []
        original_close = fixture.l1.close
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

        monkeypatch.setattr(fixture.l1, 'close', cooperative_close)
        monkeypatch.setattr(cache_module, 'create_task', cancel_shared)
        monkeypatch.setattr(support, 'shield', observe_cancellation)
        owner = asyncio.create_task(fixture.cache.close())
        ordinary = [asyncio.create_task(fixture.cache.close()) for _ in range(2)]
        await asyncio.wait_for(entered.wait(), timeout=2)
        owner.cancel('first caller')
        await cancelled.wait()
        owner.cancel('later caller')
        await turn()
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
        assert fixture.l1 is not None
        expired, finish = asyncio.Event(), asyncio.Event()
        observed: list[asyncio.CancelledError] = []
        original_close = fixture.l1.close
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

        monkeypatch.setattr(fixture.l1, 'close', cooperative_close)
        monkeypatch.setattr(fixture.cache, '_deadline', asyncio.get_running_loop().time)
        monkeypatch.setattr(cache_module, 'create_task', cancel_shared)
        monkeypatch.setattr(support, 'shield', observe_cancellation)
        closer = asyncio.create_task(fixture.cache.close())
        ordinary = asyncio.create_task(fixture.cache.close())
        await asyncio.wait_for(expired.wait(), timeout=2)
        closer.cancel('actual caller')
        await turn()
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
        backend = fixture.l2
        assert backend is not None
        await _read(fixture.cache)
        assert fixture.cache._memo is None
        await backend.flush_site('production', 'test')
        vector = await backend.resolve(PLAN.tags)
        base = cache_module.KeyBuilder.base_key(
            'production', cache_module.client_binding(fixture.cache)[1], CLIENT_OPTIONS, CALL
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

        reader = asyncio.create_task(fixture.cache.read(CALL, PLAN, fetch, CLIENT_OPTIONS))
        await sleeping.wait()
        reader.cancel('lease wait')
        with pytest.raises(asyncio.CancelledError, match='lease wait'):
            await reader
        assert calls == 0
        assert await backend.release(held)
        await fixture.cache.close()

    asyncio.run(scenario())


def test_owning_context_rejects_direct_self_close_without_changing_owner() -> None:
    async def scenario() -> None:
        fixture = _fixture()
        with fixture.cache.admit(), pytest.raises(RuntimeError, match='own admitted'):
            async with fixture.cache:
                assert await _read(fixture.cache) == 42
        await fixture.cache.close()

    asyncio.run(scenario())


def test_precommit_cancellation_skips_population_and_preserves_successful_fill() -> None:
    async def scenario() -> None:
        fixture = _fixture()
        owner = asyncio.current_task()
        assert owner is not None

        def cancel_after_sg() -> None:
            if fixture.metrics.events[-1][0] == 'cache_fill':
                owner.cancel('precommit')

        fixture.metrics.callback = cancel_after_sg
        with pytest.raises(asyncio.CancelledError, match='precommit'):
            await _read(fixture.cache)
        assert fixture.l1 is not None
        assert isinstance(fixture.l1.backend, AsyncLocalBackend)
        assert 'commit' not in fixture.l1.calls
        assert 'release' in fixture.l1.calls
        assert ('cache_fill', None, {'layer': 'local_l1'}) in fixture.metrics.events
        assert not any(event[2].get('outcome') == 'cancelled' for event in fixture.metrics.events)
        fixture.metrics.callback = None
        await fixture.cache.close()

    asyncio.run(scenario())


@pytest.mark.parametrize('operation', ['reserve', 'commit'])
@pytest.mark.parametrize('where', ['before', 'after'])
def test_promotion_cancellation_releases_known_lease_and_keeps_completed_store(
    operation: str, where: str
) -> None:
    async def scenario() -> None:
        fixture = _fixture('hierarchy')
        assert fixture.l1 is not None
        assert isinstance(fixture.l1.backend, AsyncLocalBackend)
        phase = Phase()
        getattr(fixture.l1, where)[operation] = phase
        owner = asyncio.create_task(_read(fixture.cache))
        await phase.entered.wait()
        owner.cancel('promotion')
        with pytest.raises(asyncio.CancelledError, match='promotion'):
            await owner
        assert fixture.l2 is not None
        assert isinstance(fixture.l2.backend, AsyncLocalBackend)
        assert (await fixture.l2.backend.storage_usage())['result_bytes'] > 0
        if operation == 'reserve' and where == 'after':
            assert (await fixture.l1.backend.storage_usage())['lease_bytes'] > 0
        else:
            assert (await fixture.l1.backend.storage_usage())['lease_bytes'] == 0
        if operation == 'commit' and where == 'after':
            assert (await fixture.l1.backend.storage_usage())['result_bytes'] > 0
        fixture.l1.before.clear()
        fixture.l1.after.clear()
        await fixture.cache.close()

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
            fixture.metrics.callback = None
            owner.cancel('before dispatch')

        fixture.metrics.callback = cancel

        async def read() -> object:
            if parsed:
                return await fixture.cache.read_parsed(CALL, PLAN, fetch, CLIENT_OPTIONS, _parse)
            return await fixture.cache.read(CALL, PLAN, fetch, CLIENT_OPTIONS)

        with (
            fixture.cache.bypass(),
            pytest.raises(asyncio.CancelledError, match='before dispatch'),
        ):
            await read()
        assert calls == 0
        await fixture.cache.close()

    asyncio.run(scenario())


@pytest.mark.parametrize('parsed', [False, True])
def test_re_elected_followers_keep_one_deadline_and_count_once(
    parsed: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def scenario() -> None:
        fixture = _fixture()
        first = Phase()
        second = Phase()
        deadlines: dict[str, list[float]] = {}
        original = fixture.cache._wait_flight

        async def observed(
            flight: asyncio.Future[None], deadline: float, waits: WaitRecord
        ) -> bool:
            task = asyncio.current_task()
            assert task is not None
            deadlines.setdefault(task.get_name(), []).append(deadline)
            return await original(flight, deadline, waits)

        monkeypatch.setattr(fixture.cache, '_wait_flight', observed)
        calls = 0

        async def fetch() -> object:
            nonlocal calls
            calls += 1
            await (first if calls == 1 else second).wait()
            return calls

        async def read() -> object:
            if parsed:
                return await fixture.cache.read_parsed(CALL, PLAN, fetch, CLIENT_OPTIONS, _parse)
            return await fixture.cache.read(CALL, PLAN, fetch, CLIENT_OPTIONS)

        leader = asyncio.create_task(read())
        await first.entered.wait()
        followers = [asyncio.create_task(read(), name=f'follower-{number}') for number in range(3)]
        await turn()
        leader.cancel('replace')
        with pytest.raises(asyncio.CancelledError, match='replace'):
            await leader
        await asyncio.wait_for(second.entered.wait(), timeout=2)
        await turn()
        repeated = [values for values in deadlines.values() if len(values) > 1]
        assert len(repeated) == 2
        assert all(len(set(values)) == 1 for values in repeated)
        second.proceed.set()
        await asyncio.gather(*followers)
        assert calls == 2
        await fixture.cache.close()
        events = fixture.metrics.events
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
        assert fixture.l1 is not None
        assert fixture.l2 is not None
        phase = Phase()

        def cancel_inner() -> None:
            inner = asyncio.current_task()
            assert inner is not None
            inner.cancel('inner backend close')

        phase.on_enter = cancel_inner
        fixture.l1.after['close'] = phase
        later = Phase()
        later.error = second_failure
        later.proceed.set()
        fixture.l2.after['close'] = later
        with pytest.raises(BackendUnavailable, match='cleanup task cancelled'):
            await fixture.cache.close()
        await _assert_closed_backends(fixture)

    asyncio.run(scenario())


@pytest.mark.parametrize('prebody', [False, True])
@pytest.mark.parametrize('cancel_caller', [False, True])
def test_interrupted_shared_close_settles_real_backends_and_concurrent_waiters(
    prebody: bool, cancel_caller: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def scenario() -> None:
        fixture = _fixture('hierarchy')
        assert fixture.l1 is not None
        assert fixture.l2 is not None
        operation = Phase()
        cleanup = Phase()
        fixture.l1.after['close'] = cleanup
        waiting = asyncio.Event()
        shared_tasks: list[asyncio.Task[object]] = []
        original_idle = fixture.cache._admission.wait_idle

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
            with fixture.cache.admit():
                await operation.wait()

        monkeypatch.setattr(fixture.cache._admission, 'wait_idle', observed_idle)
        monkeypatch.setattr(cache_module, 'create_task', create)
        active = asyncio.create_task(admitted())
        await operation.entered.wait()
        first = asyncio.create_task(fixture.cache.close())
        ordinary = [asyncio.create_task(fixture.cache.close()) for _ in range(2)]
        # A pre-entry cancellation must transfer the active join to its waiter.
        if not prebody:
            await waiting.wait()
            shared_tasks[0].cancel('shared close')
        await turn()
        if cancel_caller:
            first.cancel('first caller')
            await turn()
            first.cancel('later caller')
        assert not cleanup.entered.is_set()
        assert not first.done()
        assert all(not task.done() for task in ordinary)
        with pytest.raises(RuntimeError, match='closing'):
            await _read(fixture.cache)
        operation.proceed.set()
        await active
        await asyncio.wait_for(cleanup.entered.wait(), timeout=2)
        assert not fixture.cache._admission.closed
        cleanup.proceed.set()
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
        assert fixture.l1 is not None
        original_close = fixture.l1.close
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

        monkeypatch.setattr(fixture.l1, 'close', independent_close)
        monkeypatch.setattr(cache_module, 'create_task', cancel_shared)
        with pytest.raises(BackendUnavailable, match='cleanup task cancelled'):
            await fixture.cache.close()
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
                    await fixture.cache.invalidate((GenerationChange(PLAN.tags[0]),))
            else:
                assert not await fixture.cache.invalidate((GenerationChange(PLAN.tags[0]),))
        assert all(state.retained for state in fixture.cache._fences.values())
        assert fixture.l1 is not None
        assert fixture.l2 is not None
        assert 'change' not in fixture.l1.calls
        assert 'change' not in fixture.l2.calls
        assert await _read(fixture.cache) == 42
        assert 'lookup' not in fixture.l1.calls
        assert 'lookup' not in fixture.l2.calls
        assert await fixture.cache.invalidate((GenerationChange(PLAN.tags[0]),))
        assert all(not state.retained for state in fixture.cache._fences.values())
        await fixture.cache.close()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    'failure', [None, BackendUnavailable('backend close'), ValueError('ordinary close')]
)
def test_close_drains_whole_admission_attempts_all_backends_and_preserves_cancellation(
    failure: Exception | None,
) -> None:
    async def scenario() -> None:
        fixture = _fixture('hierarchy')
        operation = Phase()
        cleanup = Phase()
        cleanup.error = failure
        assert fixture.l1 is not None
        assert fixture.l2 is not None
        fixture.l1.before['close'] = cleanup

        async def admitted() -> None:
            with fixture.cache.admit() as token:
                await operation.wait()
                assert (
                    await fixture.cache.read_admitted(
                        CALL, PLAN, answer, CLIENT_OPTIONS, token=token
                    )
                    == 42
                )

        active = asyncio.create_task(admitted())
        await operation.entered.wait()
        closer = asyncio.create_task(fixture.cache.close())
        ordinary = asyncio.create_task(fixture.cache.close())
        await turn()
        closer.cancel('first close')
        await turn()
        closer.cancel('second close')
        assert not cleanup.entered.is_set()
        with pytest.raises(RuntimeError, match='closing'):
            await _read(fixture.cache)
        operation.proceed.set()
        await active
        await cleanup.entered.wait()
        assert not cleanup.cancelled.is_set()
        cleanup.proceed.set()
        with pytest.raises(asyncio.CancelledError, match='first close'):
            await closer
        if failure is None:
            await ordinary
        else:
            with pytest.raises(type(failure), match=str(failure)):
                await ordinary
        assert closer.cancelling() == 2
        assert fixture.l2.calls.count('close') == 1
        assert fixture.cache._admission.closed
        await fixture.cache.close()

    asyncio.run(scenario())


@pytest.mark.parametrize('timeout', [False, True])
def test_sg_cancellation_precedes_repeated_release_cancellation_and_cleanup_failure(
    timeout: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def scenario() -> None:
        fixture = _fixture()
        assert fixture.l1 is not None
        sg = Phase()
        release = Phase()
        release.cleanup_proceed.clear()
        release.error = BackendUnavailable('release failure')
        fixture.l1.before['release'] = release

        async def fetch() -> object:
            await sg.wait()
            return 42

        reader = asyncio.create_task(fixture.cache.read(CALL, PLAN, fetch, CLIENT_OPTIONS))
        await sg.entered.wait()
        with monkeypatch.context() as patch:
            if timeout:
                patch.setattr(
                    fixture.cache, '_deadline', lambda: asyncio.get_running_loop().time()
                )
            reader.cancel('original SG cancellation')
            await release.entered.wait()
            if timeout:
                await release.cancelled.wait()
            reader.cancel('later release cancellation')
            await turn()
            release.cleanup_proceed.set()
            release.proceed.set()
            with pytest.raises(asyncio.CancelledError, match='original SG cancellation'):
                await reader
        assert reader.cancelling() == 2
        assert fixture.cache._flights == {}
        assert (
            'cache_fill',
            None,
            {'layer': 'local_l1', 'outcome': 'cancelled'},
        ) in fixture.metrics.events
        assert (
            'backend_failure',
            None,
            {'reason': 'finalization_timeout' if timeout else 'backend_failure'},
        ) in fixture.metrics.events
        await fixture.cache.close()

    asyncio.run(scenario())


@pytest.mark.parametrize('role', ['result', 'metadata'])
def test_async_stalled_backend_opens_circuit(role: str, monkeypatch: pytest.MonkeyPatch) -> None:
    async def scenario() -> None:
        clock = FakeClock()
        monkeypatch.setattr(asyncio.get_running_loop(), 'time', clock)
        tested = _DeadlineCache(clock)
        stalled = _DeadlineOperation(clock, cooperative=False)
        client = tested.result if role == 'result' else tested.metadata
        monkeypatch.setattr(client, 'evalsha', stalled)
        assert await _read(tested.cache) == 42
        assert len(tested.failures('lookup' if role == 'result' else 'resolve')) == 1
        assert len(tested.timeouts()) == 1
        assert await _read(tested.cache) == 42
        assert stalled.calls == 1
        assert stalled.cancellations == 1
        await tested.cache.close()

    asyncio.run(scenario())


@pytest.mark.parametrize('fence', [False, True])
def test_cache_deadline_owns_finite_late_cleanup(
    fence: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def scenario() -> None:
        clock = FakeClock()
        monkeypatch.setattr(asyncio.get_running_loop(), 'time', clock)
        tested = _DeadlineCache(clock, hierarchy=fence)
        stalled = _DeadlineOperation(clock, cooperative=True)
        monkeypatch.setattr(tested.metadata if fence else tested.result, 'evalsha', stalled)
        operation = asyncio.create_task(
            tested.cache.invalidate((GenerationChange(PLAN.tags[0]),))
            if fence
            else _read(tested.cache)
        )
        await stalled.entered.wait()
        clock.advance(10)
        await turn()
        await turn()
        stalled.release.set()
        assert await operation == (False if fence else 42)
        assert stalled.cancellations == 1
        assert len(tested.failures('change' if fence else 'lookup')) == 1
        assert len(tested.timeouts()) == 1
        if fence:
            assert tested.local is not None
            assert 'change' not in tested.local.calls
            assert all(state.retained for state in tested.cache._fences.values())
        await tested.cache.close()

    asyncio.run(scenario())


@pytest.mark.parametrize('role', ['result', 'metadata'])
@pytest.mark.parametrize('half_open', [False, True])
def test_cache_reports_expiry_when_caller_cancels_late_cleanup(
    role: str, half_open: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def scenario() -> None:
        clock = FakeClock()
        monkeypatch.setattr(asyncio.get_running_loop(), 'time', clock)
        tested = _DeadlineCache(clock)
        breaker = tested.backend._result_breaker if role == 'result' else tested.store._breaker
        if half_open:
            breaker.fail()
            clock.advance(tested.settings.limits.circuit_probe_seconds)
        tested.metrics.events.clear()
        stalled = _DeadlineOperation(clock, cooperative=True)
        monkeypatch.setattr(
            tested.result if role == 'result' else tested.metadata, 'evalsha', stalled
        )
        reader = asyncio.create_task(_read(tested.cache))
        await stalled.entered.wait()
        reader.cancel('caller after expiry')
        with pytest.raises(asyncio.CancelledError, match='caller after expiry'):
            await reader
        assert stalled.cancellations == 2
        assert len(tested.failures('lookup' if role == 'result' else 'resolve')) == 1
        assert len(tested.timeouts()) == 1
        assert await _read(tested.cache) == 42
        assert stalled.calls == 1
        await tested.cache.close()

    asyncio.run(scenario())


@pytest.mark.parametrize('caller_cancel', [False, True])
def test_cache_close_reports_expiry_behind_first_error(
    caller_cancel: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def scenario() -> None:
        clock = FakeClock()
        monkeypatch.setattr(asyncio.get_running_loop(), 'time', clock)
        tested = _DeadlineCache(clock)
        first_error = BackendProtocolError('first resource')
        tested.result.close_error = first_error
        stalled = _DeadlineOperation(clock, cooperative=True)

        async def later_close() -> None:
            tested.metadata.closed += 1
            await stalled('', 0)

        monkeypatch.setattr(tested.metadata, 'close', later_close)
        closer = asyncio.create_task(tested.cache.close())
        ordinary = asyncio.create_task(tested.cache.close())
        await stalled.entered.wait()
        if caller_cancel:
            closer.cancel('caller during close')
        clock.advance(10)
        await turn()
        stalled.release.set()
        if caller_cancel:
            with pytest.raises(asyncio.CancelledError, match='caller during close'):
                await closer
        else:
            with pytest.raises(BackendProtocolError) as raised:
                await closer
            assert raised.value is first_error
        with pytest.raises(BackendProtocolError) as shared:
            await ordinary
        assert shared.value is first_error
        assert tested.result.closed == tested.metadata.closed == 1
        assert stalled.cancellations == 1
        assert len(tested.timeouts()) == 1
        await tested.cache.close()
        assert len(tested.timeouts()) == 1

    asyncio.run(scenario())


@pytest.mark.parametrize('suppressed_expiry', [False, True])
def test_mixed_fence_keeps_one_deadline_and_stops_on_recorded_expiry(
    suppressed_expiry: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def scenario() -> None:
        clock = FakeClock()
        monkeypatch.setattr(asyncio.get_running_loop(), 'time', clock)
        tested = _DeadlineCache(clock, hierarchy=True)
        assert tested.local is not None
        deadline = clock() + 1
        records: list[support.DeadlineRecord | None] = []

        async def shared_change(_sha: str, _numkeys: int, *_args: object) -> object:
            records.append(support.PUBLISHED_DEADLINE.get())
            clock.advance(1 if suppressed_expiry else 0.5)
            if suppressed_expiry:
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    return [b'ok', 1, 0]
            return [b'ok', 1, 0]

        async def local_change(_changes: tuple[GenerationChange, ...]) -> None:
            records.append(support.PUBLISHED_DEADLINE.get())
            clock.advance(0.5)
            await asyncio.Event().wait()

        monkeypatch.setattr(tested.metadata, 'evalsha', shared_change)
        monkeypatch.setattr(tested.local, 'change', local_change)
        assert not await tested.cache.invalidate((GenerationChange(PLAN.tags[0]),))
        assert len(records) == (1 if suppressed_expiry else 2)
        assert all(record is records[0] for record in records)
        assert records[0] is not None
        assert records[0].deadline == deadline
        assert records[0].expired
        assert tested.cache._fences['l1'].retained
        assert bool(tested.cache._fences['l2'].retained) == suppressed_expiry
        assert len(tested.timeouts()) == 1
        await tested.cache.close()

    asyncio.run(scenario())


@pytest.mark.parametrize('caller_cancel', [False, True])
def test_owned_release_reports_one_expiry_after_joined_cleanup(
    caller_cancel: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def scenario() -> None:
        clock = FakeClock()
        monkeypatch.setattr(asyncio.get_running_loop(), 'time', clock)
        tested = _DeadlineCache(clock)
        stalled = _DeadlineOperation(clock, cooperative=True)
        monkeypatch.setattr(tested.result, 'evalsha', stalled)

        async def release() -> None:
            async with tested.cache._held_lease(tested.backend, FillLease('key', 'token')):
                pass

        owner = asyncio.create_task(release())
        await stalled.entered.wait()
        if caller_cancel:
            owner.cancel('caller during release')
        clock.advance(10)
        await turn()
        assert not owner.done()
        stalled.release.set()
        if caller_cancel:
            with pytest.raises(asyncio.CancelledError, match='caller during release'):
                await owner
        else:
            await owner
        assert stalled.cancellations == 1
        assert len(tested.failures('release')) == 1
        assert len(tested.timeouts()) == 1
        assert support.PUBLISHED_DEADLINE.get() is None
        await tested.cache.close()

    asyncio.run(scenario())
