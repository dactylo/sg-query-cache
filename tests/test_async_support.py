import asyncio
from collections.abc import Coroutine
from typing import Any, TypeVar

import pytest

from sg_query_cache import _async_support as support
from sg_query_cache._async_support import LoopAffinity
from sg_query_cache._backend import BackendUnavailable
from tests._async_fakes import _Phase

_T = TypeVar('_T')


async def _check(affinity: LoopAffinity) -> None:
    affinity.check()


async def _turn() -> None:
    ready = asyncio.Event()
    asyncio.get_running_loop().call_soon(ready.set)
    await ready.wait()


def test_active_close_join_outlasts_backend_operation_deadline() -> None:
    async def exercise() -> None:
        admission = support.Admission()
        operation = _Phase()

        async def admitted() -> None:
            with admission.operation():
                await operation._wait()

        owner = asyncio.create_task(admitted())
        await operation._entered.wait()
        admission.begin_close()
        closer = asyncio.create_task(support.joined_cleanup(admission.wait_idle))
        backend = _Phase()
        outcome = await support.joined_cleanup(
            backend._wait,
            deadline=asyncio.get_running_loop().time(),
        )
        assert isinstance(outcome.error, TimeoutError)
        assert not closer.done()
        assert not owner.done()
        operation._proceed.set()
        await owner
        closed = await closer
        assert closed.error is None
        admission.finish_close()

    asyncio.run(exercise())


def test_admission_exit_on_failure_revokes_token_and_unblocks_close() -> None:
    async def exercise() -> None:
        admission = support.Admission()
        tokens: list[support.OperationToken] = []

        def fail() -> None:
            with admission.operation() as token:
                tokens.append(token)
                raise ValueError('operation failure')

        with pytest.raises(ValueError, match='operation failure'):
            fail()
        with pytest.raises(RuntimeError):
            admission.validate(tokens[0])
        assert admission.begin_close()
        await admission.wait_idle()
        admission.finish_close()

    asyncio.run(exercise())


def test_admission_tokens_are_live_cache_specific_capabilities() -> None:
    async def exercise() -> None:
        admission = support.Admission()
        other = support.Admission()
        with admission.operation() as token:
            admission.validate(token)
            with pytest.raises(RuntimeError):
                other.validate(token)
            with pytest.raises(RuntimeError, match='own admitted operation'):
                admission.begin_close()
        with pytest.raises(RuntimeError):
            admission.validate(token)
        with admission.operation() as fresh:
            admission.validate(fresh)

    asyncio.run(exercise())


def test_cleanup_deadline_waits_for_finite_cancellation_cleanup() -> None:
    async def exercise() -> None:
        phase = _Phase()
        phase._cleanup_proceed.clear()
        owner = asyncio.create_task(
            support.joined_cleanup(phase._wait, deadline=asyncio.get_running_loop().time())
        )
        await phase._cancelled.wait()
        assert not owner.done()
        owner.cancel('caller')
        await _turn()
        assert not owner.done()
        phase._cleanup_proceed.set()
        outcome = await owner
        assert isinstance(outcome.error, TimeoutError)
        assert outcome.cancellation is not None
        assert outcome.cancellation.args == ('caller',)
        assert phase._finished.is_set()

    asyncio.run(exercise())


def test_cleanup_preserves_base_exception_instead_of_normalizing_failure() -> None:
    class _FatalSignal(BaseException):
        pass

    async def exercise() -> None:
        failure = _FatalSignal()

        async def action() -> None:
            raise failure

        with pytest.raises(_FatalSignal) as caught:
            await support.joined_cleanup(action)
        assert caught.value is failure

    asyncio.run(exercise())


def test_closed_admission_still_rejects_another_loop() -> None:
    admission = support.Admission()

    async def close() -> None:
        admission.begin_close()
        await admission.wait_idle()
        admission.finish_close()

    asyncio.run(close())
    with pytest.raises(RuntimeError, match='different event loop'):
        asyncio.run(close())


def test_closing_joins_whole_operations_and_allows_explicit_live_token() -> None:
    async def exercise() -> None:
        admission = support.Admission()
        entered = asyncio.Event()
        release = asyncio.Event()

        async def operation() -> None:
            with admission.operation() as token:
                entered.set()
                await release.wait()
                admission.validate(token)

                async def owned_cleanup() -> None:
                    admission.validate(token)
                    with pytest.raises(RuntimeError, match='closing'), admission.operation():
                        pytest.fail('inherited context must not grant admission')

                await asyncio.create_task(owned_cleanup())

        owner = asyncio.create_task(operation())
        await entered.wait()
        assert admission.begin_close()
        assert not admission.begin_close()
        waiter = asyncio.create_task(admission.wait_idle())
        await _turn()
        assert not waiter.done()
        release.set()
        await owner
        await waiter
        admission.finish_close()
        assert admission.closed
        assert not admission.begin_close()
        with pytest.raises(RuntimeError, match='closed'), admission.operation():
            pytest.fail('closed admission must refuse operations')

    asyncio.run(exercise())


def test_enclosing_timeout_receives_deferred_cancellation_after_settlement() -> None:
    async def exercise() -> None:
        phase = _Phase()
        outcomes: list[support.CleanupOutcome[None]] = []
        timeout = asyncio.timeout(None)

        async def caller() -> None:
            async with timeout:
                outcome = await support.joined_cleanup(phase._wait)
                outcomes.append(outcome)
                if outcome.cancellation is not None:
                    raise outcome.cancellation

        owner = asyncio.create_task(caller())
        await phase._entered.wait()
        timeout.reschedule(asyncio.get_running_loop().time())
        await _turn()
        assert not owner.done()
        phase._proceed.set()
        with pytest.raises(TimeoutError):
            await owner
        assert outcomes[0].cancellation is not None
        assert phase._finished.is_set()
        assert owner.cancelling() == 0  # The enclosing timeout owns its own uncancel.

    asyncio.run(exercise())


def test_independent_inner_cancellation_and_caller_cancellation_are_both_recorded() -> None:
    async def exercise() -> None:
        owner = asyncio.current_task()
        assert owner is not None

        async def action() -> None:
            inner = asyncio.current_task()
            assert inner is not None
            inner.cancel('inner')
            owner.cancel('caller')
            await asyncio.Event().wait()

        outcome = await support.joined_cleanup(action)
        assert isinstance(outcome.error, BackendUnavailable)
        assert outcome.cancellation is not None
        assert outcome.cancellation.args == ('caller',)
        assert owner.cancelling() == 1

    asyncio.run(exercise())


def test_independent_inner_cancellation_is_failed_cleanup() -> None:
    async def exercise() -> None:
        async def action() -> None:
            inner = asyncio.current_task()
            assert inner is not None
            inner.cancel('independent')
            await asyncio.Event().wait()

        outcome = await support.joined_cleanup(action)
        assert isinstance(outcome.error, BackendUnavailable)
        assert outcome.cancellation is None

    asyncio.run(exercise())


@pytest.mark.parametrize('failure', [None, ValueError('failed'), BackendUnavailable('backend')])
def test_joined_cleanup_delivers_value_or_failure_without_cancellation(
    failure: Exception | None,
) -> None:
    async def exercise() -> None:
        async def action() -> int:
            if failure is not None:
                raise failure
            return 42

        outcome = await support.joined_cleanup(action)
        assert outcome.value == (42 if failure is None else None)
        assert outcome.error is failure
        assert outcome.cancellation is None

    asyncio.run(exercise())


def test_joined_cleanup_leaves_no_tasks_or_unhandled_errors() -> None:
    async def exercise() -> None:
        loop = asyncio.get_running_loop()
        unhandled: list[dict[str, Any]] = []
        loop.set_exception_handler(lambda _loop, context: unhandled.append(context))
        before = asyncio.all_tasks()

        async def action() -> None:
            owner = asyncio.current_task()
            assert owner is not None
            raise ValueError('captured')

        outcome = await support.joined_cleanup(action)
        assert isinstance(outcome.error, ValueError)
        await _turn()
        assert asyncio.all_tasks() == before
        assert unhandled == []

    asyncio.run(exercise())


@pytest.mark.parametrize('when', ['before', 'during', 'completion'])
@pytest.mark.parametrize('failure', [None, ValueError('failed'), BackendUnavailable('backend')])
def test_joined_cleanup_preserves_first_cancellation_and_settles_failures(
    when: str,
    failure: Exception | None,
) -> None:
    async def exercise() -> None:
        phase = _Phase()
        phase._error = failure
        outcomes: list[support.CleanupOutcome[None]] = []
        settled = asyncio.Event()

        async def caller() -> None:
            if when == 'before':
                current = asyncio.current_task()
                assert current is not None
                current.cancel('first')
            outcome = await support.joined_cleanup(phase._wait)
            outcomes.append(outcome)
            settled.set()
            if outcome.cancellation is not None:
                raise outcome.cancellation

        owner = asyncio.create_task(caller())
        await phase._entered.wait()
        if when == 'during':
            owner.cancel('first')
        if when != 'completion':
            await _turn()
            owner.cancel('second')
            await _turn()
            assert not owner.done()
            assert not phase._cancelled.is_set()
        phase._proceed.set()
        if when == 'completion':
            owner.cancel('first')
        with pytest.raises(asyncio.CancelledError, match='first'):
            await owner
        assert settled.is_set()
        assert phase._finished.is_set()
        assert outcomes[0].error is failure
        assert owner.cancelling() == (1 if when == 'completion' else 2)

    asyncio.run(exercise())


def test_loop_affinity_constructs_outside_a_loop_and_binds_on_first_check() -> None:
    affinity = LoopAffinity()
    first = asyncio.Runner()
    second = asyncio.Runner()
    try:
        first.run(_check(affinity))
        first.run(_check(affinity))

        with pytest.raises(RuntimeError, match='different event loop'):
            second.run(_check(affinity))
    finally:
        first.close()
        second.close()


def test_nested_admission_keeps_owner_active_until_every_context_exits() -> None:
    async def exercise() -> None:
        admission = support.Admission()
        with admission.operation() as outer:
            with admission.operation() as inner:
                assert outer is not inner
                assert inner.owner is asyncio.current_task()
                with pytest.raises(RuntimeError):
                    admission.validate(support.OperationToken(inner.owner))
            admission.validate(outer)
            with pytest.raises(RuntimeError):
                admission.validate(inner)
            with pytest.raises(RuntimeError, match='own admitted operation'):
                admission.begin_close()
        assert admission.begin_close()
        await admission.wait_idle()
        admission.finish_close()

    asyncio.run(exercise())


@pytest.mark.parametrize('cancel_caller', [False, True])
def test_pre_body_inner_cancellation_still_reaches_owner_settlement(
    monkeypatch: pytest.MonkeyPatch,
    cancel_caller: bool,
) -> None:
    async def exercise() -> None:
        started = False
        task_create = asyncio.create_task
        owner = asyncio.current_task()
        assert owner is not None

        def cancel_before_start(coro: Coroutine[Any, Any, _T]) -> asyncio.Task[_T]:
            inner = task_create(coro)
            inner.cancel('inner')
            if cancel_caller:
                owner.cancel('caller')
            return inner

        async def action() -> None:
            nonlocal started
            started = True

        monkeypatch.setattr(support, 'create_task', cancel_before_start)
        outcome = await support.joined_cleanup(action)
        assert not started
        assert isinstance(outcome.error, BackendUnavailable)
        if cancel_caller:
            assert outcome.cancellation is not None
            assert outcome.cancellation.args == ('caller',)
            assert owner.cancelling() == 1
        else:
            assert outcome.cancellation is None

    asyncio.run(exercise())


@pytest.mark.parametrize('failure', [ValueError('failure'), BackendUnavailable('backend')])
def test_cleanup_completion_itself_can_cancel_caller_without_losing_failure(
    failure: Exception,
) -> None:
    async def exercise() -> None:
        owner = asyncio.current_task()
        assert owner is not None

        async def action() -> None:
            owner.cancel('completion')
            raise failure

        outcome = await support.joined_cleanup(action)
        assert outcome.error is failure
        assert outcome.cancellation is not None
        assert outcome.cancellation.args == ('completion',)
        assert owner.cancelling() == 1

    asyncio.run(exercise())


def test_previous_cancellation_count_is_not_misreported_as_new_cancellation() -> None:
    async def exercise() -> None:
        owner = asyncio.current_task()
        assert owner is not None
        owner.cancel('previous')
        with pytest.raises(asyncio.CancelledError):
            await _turn()

        async def action() -> None:
            raise asyncio.CancelledError('independent')

        outcome = await support.joined_cleanup(action)
        assert isinstance(outcome.error, BackendUnavailable)
        assert outcome.cancellation is None
        assert owner.cancelling() == 1

    asyncio.run(exercise())


def test_taskgroup_waits_for_cancelled_member_cleanup() -> None:
    async def exercise() -> None:
        phase = _Phase()
        settled = asyncio.Event()

        async def member() -> None:
            outcome = await support.joined_cleanup(phase._wait)
            settled.set()
            if outcome.cancellation is not None:
                raise outcome.cancellation

        async def fail() -> None:
            await phase._entered.wait()
            raise ValueError('group failure')

        async def release() -> None:
            await phase._entered.wait()
            await _turn()
            await _turn()
            assert not settled.is_set()
            phase._proceed.set()

        members: list[asyncio.Task[None]] = []

        async def run_group() -> None:
            async with asyncio.TaskGroup() as group:
                members.append(group.create_task(member()))
                group.create_task(fail())

        releaser = asyncio.create_task(release())
        with pytest.raises(ExceptionGroup) as caught:
            await run_group()
        await releaser
        assert settled.is_set()
        assert members[0].cancelling() == 1
        assert isinstance(caught.value.exceptions[0], ValueError)

    asyncio.run(exercise())


def test_whole_cleanup_uses_original_deadline_across_nested_work() -> None:
    async def exercise() -> None:
        first = _Phase()
        first._proceed.set()
        second = _Phase()

        async def action() -> None:
            await first._wait()
            await second._wait()

        outcome = await support.joined_cleanup(
            action,
            deadline=asyncio.get_running_loop().time(),
        )
        assert first._finished.is_set()
        assert second._cancelled.is_set()
        assert second._finished.is_set()
        assert isinstance(outcome.error, TimeoutError)
        assert outcome.cancellation is None

    asyncio.run(exercise())
