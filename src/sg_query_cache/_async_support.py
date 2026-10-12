"""Provide loop-owned primitives for asynchronous cache components."""

from asyncio import (
    AbstractEventLoop,
    CancelledError,
    Event,
    Future,
    Task,
    create_task,
    current_task,
    get_running_loop,
    shield,
    timeout_at,
)
from collections.abc import Awaitable, Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import Generic, Literal, TypeVar

from sg_query_cache._backend import BackendUnavailable

_T = TypeVar('_T')


@dataclass(frozen=True)
class CleanupOutcome(Generic[_T]):
    """Keep cleanup results separate from deferred caller cancellation.

    The caller must settle final synchronous obligations before applying its
    error policy or re-raising `cancellation`. Cancellation counts are
    retained.

    Attributes:
        value: Action result on success, including a successful `None` result.
        error: Ordinary action failure, `TimeoutError` for an expired deadline,
            or `BackendUnavailable` for independently cancelled inner work.
        cancellation: First caller cancellation received while joining, if any.
    """

    value: _T | None = None
    error: Exception | None = None
    cancellation: CancelledError | None = None


class LoopAffinity:
    """Bind a component to one event loop on first use."""

    __slots__ = ('_loop',)

    def __init__(self) -> None:
        """Create an unbound loop-affinity guard."""
        self._loop: AbstractEventLoop | None = None

    def check(self) -> None:
        """Bind first use or reject access from a different event loop.

        Raises:
            RuntimeError: No event loop is running, or the guard is bound to
                another loop.
        """
        loop = get_running_loop()
        if self._loop is None:
            self._loop = loop
        elif loop is not self._loop:
            raise RuntimeError('async cache component used from a different event loop')


@dataclass(frozen=True, eq=False)
class OperationToken:
    """Identify one admission lifetime and its original owner task.

    Tokens are issued by `Admission.operation`. An explicitly passed live token
    may be reused by owned cleanup; copied context alone carries no permission.
    Constructing another token does not grant admission.

    Attributes:
        owner: Task whose synchronous operation context owns this token.
    """

    owner: Task[object]


class Admission:
    """Track whole operations and terminal close on one event loop.

    Operation entry and exit never await. Close owners call `begin_close`, join
    `wait_idle`, perform their resource cleanup, then call `finish_close`.
    Internal calls explicitly validate a live token, including during closing.
    """

    def __init__(self) -> None:
        """Create an open admission guard that binds on its first operation."""
        self._affinity = LoopAffinity()
        self._idle = Event()
        self._idle.set()
        self._state: Literal['open', 'closing', 'closed'] = 'open'
        self._tokens: set[OperationToken] = set()

    def begin_close(self) -> bool:
        """Reject self-close and atomically refuse new operations.

        Returns:
            Whether this call changed open to closing. The caller owns creating
            one shared close task without awaiting after a true result.

        Raises:
            RuntimeError: Called by an active operation owner or on another
                loop.
        """
        self._affinity.check()
        if any(token.owner is current_task() for token in self._tokens):
            raise RuntimeError('cannot close from own admitted operation')
        if self._state != 'open':
            return False
        self._state = 'closing'
        return True

    @property
    def closed(self) -> bool:
        """Check loop affinity and report whether terminal close has finished."""
        self._affinity.check()
        return self._state == 'closed'

    def finish_close(self) -> None:
        """Mark a closing guard closed after all admitted operations exit.

        Raises:
            RuntimeError: Close has not begun, operations remain, or loop
                differs.
        """
        self._affinity.check()
        if self._state == 'open' or self._tokens:
            raise RuntimeError('close must begin and join active operations first')
        self._state = 'closed'

    @contextmanager
    def operation(self) -> Generator[OperationToken]:
        """Admit one whole operation and revoke its token on synchronous exit.

        Yields:
            A cache-specific live capability for explicitly admitted internal
            use.

        Raises:
            RuntimeError: Closing, closed, outside a task, or on another loop.
        """
        self._affinity.check()
        if self._state != 'open':
            raise RuntimeError(f'async cache is {self._state}')
        owner = current_task()
        if owner is None:
            raise RuntimeError('admission requires an asyncio task')
        token = OperationToken(owner)
        self._tokens.add(token)
        self._idle.clear()
        try:
            yield token
        finally:
            self._tokens.remove(token)
            if not self._tokens:
                self._idle.set()

    def validate(self, token: OperationToken) -> None:
        """Require a token issued by this guard whose operation remains active.

        Args:
            token: Explicit capability owned by the enclosing operation.

        Raises:
            RuntimeError: Token is foreign or expired, or loop differs.
        """
        self._affinity.check()
        if token not in self._tokens:
            raise RuntimeError('operation token is foreign or no longer active')

    async def wait_idle(self) -> None:
        """Join active operation lifetimes without imposing a close deadline.

        Call after `begin_close` to prevent new admission during the join.
        This does not cancel operation owners or detect indirect dependency
        cycles.
        """
        self._affinity.check()
        await self._idle.wait()


async def _capture_cleanup(
    action: Callable[[], Awaitable[_T]],
    deadline: float | None,
) -> CleanupOutcome[_T]:
    """Capture ordinary failure inside the owned task under one deadline."""
    try:
        async with timeout_at(deadline):
            return CleanupOutcome(value=await action())
    except Exception as error:  # noqa: BLE001 - capture ordinary cleanup failure before joining
        return CleanupOutcome(error=error)


async def _wait_deferring_cancellation(completion: Future[_T]) -> CancelledError | None:
    """Wait until completion is done and return the first caller cancellation."""
    cancellation: CancelledError | None = None
    while not completion.done():
        try:
            await shield(completion)
        except CancelledError as error:
            if cancellation is None:
                cancellation = error
    return cancellation


async def join_shared_close(
    task: Task[CleanupOutcome[None]],
    completion: Future[CleanupOutcome[None]],
    settle: Callable[[], Awaitable[CancelledError | None]],
) -> None:
    """Join shared cleanup, settle its owner, and propagate the final outcome.

    Args:
        task: Shared terminal cleanup task, possibly interrupted before entry.
        completion: Outcome future published by the owner's settlement.
        settle: Claim or join the owner's remaining obligations after the task
            finishes, returning any deferred caller cancellation.

    Raises:
        CancelledError: First caller cancellation from notification,
            settlement, or completion, after all obligations settle.
        Exception: Shared cleanup failure when caller cancellation is absent.
    """
    notification: Future[None] = get_running_loop().create_future()

    def finished(_task: Task[CleanupOutcome[None]]) -> None:
        if not notification.done():
            notification.set_result(None)

    task.add_done_callback(finished)
    cancellation = await _wait_deferring_cancellation(notification)
    settlement_cancellation = await settle()
    later = await _wait_deferring_cancellation(completion)
    outcome = completion.result()
    cancellation = cancellation or settlement_cancellation or later
    if cancellation is not None:
        raise cancellation
    if outcome.error is not None:
        raise outcome.error


async def joined_cleanup(
    action: Callable[[], Awaitable[_T]],
    *,
    deadline: float | None = None,
) -> CleanupOutcome[_T]:
    """Own one cleanup task and join through repeated caller cancellation.

    Args:
        action: Called inside the owned task to create cleanup work. Passing a
            callable avoids leaking a coroutine if the task never starts.
        deadline: Absolute running-loop time shared by all nested work, or
            `None` for an unbounded close join. Cancellation cleanup is
            cooperative and may finish after this deadline.

    Returns:
        Completed result and failure with the first caller cancellation
        recorded separately. Independently cancelled inner work is failed
        cleanup. The caller owns final synchronous settlement and cancellation
        propagation.
    """
    task = create_task(_capture_cleanup(action, deadline))
    completed: Future[None] = get_running_loop().create_future()

    def finish(_task: Task[CleanupOutcome[_T]]) -> None:
        completed.set_result(None)

    task.add_done_callback(finish)
    # This notification never inherits inner cancellation. Every cancellation of
    # its shield comes from the caller, even if the inner task never started.
    cancellation = await _wait_deferring_cancellation(completed)
    try:
        outcome = task.result()
    except CancelledError:
        outcome = CleanupOutcome[_T](error=BackendUnavailable('cleanup task cancelled'))
    return replace(outcome, cancellation=cancellation)


async def enter_cleanup(
    action: Callable[[], Awaitable[None]], *, deadline: float | None = None
) -> CleanupOutcome[None]:
    """Join cleanup, resubmitting only until the action has been entered.

    Args:
        action: Cleanup obligation invoked exactly once, even if an owned task
            is cancelled before entering it.
        deadline: Absolute running-loop time retained across every submission,
            or `None` for an unbounded join. Entered cleanup is cooperative.

    Returns:
        First ordinary failure and first caller cancellation across all
        submissions, retained separately for the owner's error policy.
    """
    entered = False
    failure: Exception | None = None
    cancellation: CancelledError | None = None

    async def close() -> None:
        nonlocal entered
        entered = True
        await action()

    # The deadline belongs to the resource task, not the joining caller.
    # Resubmit only if cancellation prevented entry; never redeliver cleanup.
    while not entered:
        outcome = await joined_cleanup(close, deadline=deadline)
        failure = failure or outcome.error
        cancellation = cancellation or outcome.cancellation
    return CleanupOutcome(error=failure, cancellation=cancellation)
