import asyncio
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import pytest
from valkey.exceptions import DataError, NoScriptError, ResponseError

import sg_query_cache._async_valkey_control as module
from sg_query_cache._backend import (
    BackendCapacity,
    BackendProtocolError,
    BackendUnavailable,
    StaleIncarnation,
)
from sg_query_cache._model import FillLease
from sg_query_cache._valkey_common import Incarnation, ScriptArgument, decode_change_reply
from sg_query_cache._valkey_scripts import (
    COMMIT_SCRIPT_SHA,
    CONTROL_CREATE_SCRIPT_SHA,
    CONTROL_ROTATE_SCRIPT_SHA,
)
from tests._fakes import _FakeClock
from tests._valkey_control_cases import (
    _COUNTER_A,
    _COUNTER_B,
    _RECORD_A,
    _RECORD_B,
    _control_traces,
    _ControlTrace,
    _publication_traces,
    _PublicationTrace,
)
from tests.valkey._async_helpers import BackendClient
from tests.valkey._control_helpers import AtomicControlClient, TraceControlClient


class _AsyncAtomicClient:
    def __init__(self) -> None:
        self._source = AtomicControlClient(replication_id='primary')
        self._source._control = _RECORD_A
        self._source._counters.add(_COUNTER_A)
        self._phase: str | None = None
        self._entered = asyncio.Event()
        self._proceed = asyncio.Event()
        self._calls: list[str] = []
        self._pins = 0
        self._pin_connection: int | None = None
        self._missing_script = False

    async def _step(self, name: str) -> None:
        self._calls.append(name)
        if name == self._phase:
            self._entered.set()
            await self._proceed.wait()

    def epoch(self) -> tuple[int, int, None]:
        return self._source.epoch()

    async def evalsha(self, sha: str, numkeys: int, *args: ScriptArgument) -> object:
        if self._missing_script:
            self._missing_script = False
            raise NoScriptError('load required')
        name = 'create' if sha == CONTROL_CREATE_SCRIPT_SHA else 'rotate'
        assert sha in (CONTROL_CREATE_SCRIPT_SHA, CONTROL_ROTATE_SCRIPT_SHA)
        await self._step(name)
        return self._source.evalsha(sha, numkeys, *args)

    async def exists(self, key: str) -> object:
        await self._step('exists')
        return self._source.exists(key)

    async def get(self, key: str) -> object:
        await self._step('get')
        return self._source.get(key)

    async def info(self, section: str) -> object:
        await self._step('info')
        return self._source.info(section)

    @asynccontextmanager
    async def pin(self) -> AsyncGenerator['_AsyncAtomicClient | _AsyncPinnedConnectionView']:
        await self._step('pin')
        self._pins += 1
        try:
            if self._pin_connection is None:
                yield self
            else:
                yield _AsyncPinnedConnectionView(self, self._pin_connection)
        finally:
            self._pins -= 1
            await self._step('unpin')

    async def refresh(self) -> None:
        await self._step('refresh')

    async def refresh_endpoint(self) -> None:
        await self._step('discovery')

    async def script_load(self, script: bytes | str) -> object:
        await self._step('load')
        return self._source.script_load(script)


class _AsyncPinnedConnectionView:
    def __init__(self, owner: _AsyncAtomicClient, connection: int) -> None:
        self._owner = owner
        self._connection = connection

    def connection_generation(self) -> int:
        return self._connection

    async def evalsha(self, sha: str, numkeys: int, *args: ScriptArgument) -> object:
        return await self._owner.evalsha(sha, numkeys, *args)

    async def exists(self, key: str) -> object:
        return await self._owner.exists(key)

    async def get(self, key: str) -> object:
        return await self._owner.get(key)

    async def info(self, section: str) -> object:
        return await self._owner.info(section)

    async def script_load(self, script: bytes | str) -> object:
        return await self._owner.script_load(script)


def test_control_forced_validation_resets_connection_coverage() -> None:
    async def scenario() -> None:
        client = _AsyncAtomicClient()
        client._pin_connection = 1
        clock = _FakeClock(start=0.0)
        control = module.AsyncValkeyControl(
            client,
            key_prefix='sgqc',
            validation_seconds=30,
            clock=clock,
            jitter_factory=lambda seconds: seconds,
        )
        await control._validate_incarnation()
        client._pin_connection = 2
        await control._validate_incarnation()
        assert control._state.covered == frozenset({1, 2})

        client._pin_connection = 1
        await control._validate_incarnation(force=True)
        assert control._state.covered == frozenset({1})
        assert client._source._info_calls == 6

    asyncio.run(scenario())


def test_control_join_does_not_extend_validation_lifetime() -> None:
    async def scenario() -> None:
        client = _AsyncAtomicClient()
        client._pin_connection = 1
        clock = _FakeClock(start=0.0)
        control = module.AsyncValkeyControl(
            client,
            key_prefix='sgqc',
            validation_seconds=30,
            clock=clock,
            jitter_factory=lambda seconds: seconds,
        )
        await control._validate_incarnation()
        clock._advance(20)
        client._pin_connection = 2
        await control._validate_incarnation()
        assert control._state.validated_at == 0.0
        clock._advance(9)
        client._pin_connection = 1
        await control._validate_incarnation()
        assert client._source._info_calls == 4

        clock._advance(1)
        client._pin_connection = 2
        await control._validate_incarnation()
        assert client._source._info_calls == 6
        assert control._state.covered == frozenset({2})
        assert control._state.validated_at == 30.0
        client._pin_connection = 1
        await control._validate_incarnation()
        assert client._source._info_calls == 8
        assert control._state.covered == frozenset({1, 2})
        assert control._state.validated_at == 30.0

    asyncio.run(scenario())


@pytest.mark.parametrize('trace', _publication_traces(), ids=lambda trace: trace._name)
def test_control_publication_obeys_shared_revision_races(trace: _PublicationTrace) -> None:
    async def scenario() -> None:
        client = _AsyncAtomicClient()
        clock = _FakeClock(start=0.0)
        control = module.AsyncValkeyControl(
            client,
            key_prefix='sgqc',
            validation_seconds=30,
            clock=clock,
            jitter_factory=lambda seconds: seconds,
        )
        if trace._older_joins:
            client._pin_connection = 1
            await control._validate_incarnation()
            client._pin_connection = 2
            clock._advance(10)
        entered, proceed = asyncio.Event(), asyncio.Event()
        effects: list[str] = []

        async def older(_pin: object, incarnation: Incarnation) -> bytes:
            effects.append('older-operation-entered')
            entered.set()
            await proceed.wait()
            return incarnation.record

        async def old_run() -> bytes:
            result = await control.run(older)
            effects.append('older-settled')
            return result

        async def newer(_pin: object, incarnation: Incarnation) -> bytes:
            effects.append('newer-operation-entered')
            if trace._transition == 'clear':
                raise StaleIncarnation('stale')
            return incarnation.record

        old = asyncio.create_task(old_run())
        await entered.wait()
        if trace._older_joins:
            client._pin_connection = 3
            clock._advance(10)
        client._source._control = _RECORD_B
        client._source._counters.add(_COUNTER_B)
        try:
            newer_result: bytes | type[Exception] = await control.run(newer, force=True)
        except StaleIncarnation:
            newer_result = StaleIncarnation
        effects.append('newer-settled')
        proceed.set()
        assert (await old, newer_result) == trace._acknowledgments
        assert tuple(effects) == trace._effects
        state = control._state
        assert (
            None if state.incarnation is None else state.incarnation.record
        ) == trace._final_record
        assert state.revision == trace._final_revision
        if trace._final_covered is not None:
            assert state.covered == trace._final_covered
            assert state.validated_at == (None if trace._transition == 'clear' else 20.0)

    asyncio.run(scenario())


@pytest.mark.parametrize(
    'phase',
    [
        'discovery',
        'pin',
        'info',
        'get',
        'exists',
        'create',
        'rotate',
        'load',
        'operation',
        'settlement',
        'discard',
        'unpin',
    ],
)
def test_control_cancellation_never_acknowledges_or_keeps_validation(phase: str) -> None:
    async def scenario() -> None:
        client = _AsyncAtomicClient()
        control = module.AsyncValkeyControl(client, key_prefix='sgqc', validation_seconds=30)
        await control._validate_incarnation()
        client._phase = phase
        if phase in ('create', 'load'):
            client._source._control = None
        if phase == 'load':
            client._missing_script = True
        if phase == 'rotate':
            client._source._counters.clear()

        async def operation(_pin: object, incarnation: Incarnation) -> bytes:
            await client._step('operation')
            if phase == 'settlement':
                client._phase = 'discovery'
            if phase == 'discard':
                client._source._connection_generation += 1
                client._phase = 'refresh'
            return incarnation.record

        task = asyncio.create_task(control.run(operation, force=True))
        await client._entered.wait()
        task.cancel('caller')
        with pytest.raises(asyncio.CancelledError, match='caller'):
            await task
        assert client._pins == 0
        assert control._state.incarnation is None
        assert client._calls.count('operation') <= 1

    asyncio.run(scenario())


class _AsyncTraceClient:
    def __init__(self, trace: _ControlTrace) -> None:
        self._source = TraceControlClient(trace)

    def epoch(self) -> tuple[int, int, object | None]:
        return self._source.epoch()

    async def evalsha(self, sha: str, numkeys: int, *args: ScriptArgument) -> object:
        return self._source.evalsha(sha, numkeys, *args)

    async def exists(self, key: str) -> object:
        return self._source.exists(key)

    async def get(self, key: str) -> object:
        return self._source.get(key)

    async def info(self, section: str) -> object:
        return self._source.info(section)

    @asynccontextmanager
    async def pin(self) -> AsyncGenerator['_AsyncTraceClient']:
        with self._source.pin():
            yield self

    async def refresh(self) -> None:
        self._source.refresh()

    async def refresh_endpoint(self) -> None:
        self._source.refresh_endpoint()

    async def script_load(self, script: bytes | str) -> object:
        return self._source.script_load(script)


@pytest.mark.parametrize('trace', _control_traces(), ids=lambda trace: trace._name)
def test_control_executes_shared_semantic_traces(trace: _ControlTrace) -> None:
    async def scenario() -> None:
        client = _AsyncTraceClient(trace)
        control = module.AsyncValkeyControl(
            client,
            key_prefix='sgqc',
            validation_seconds=30,
            clock=lambda: 0.0,
            jitter_factory=lambda seconds: seconds,
            nonce_factory=lambda: trace._proposal_nonce,
        )

        async def operation(_client: object, incarnation: Incarnation) -> bytes:
            decode_change_reply(client._source._effect('operation'))
            return incarnation.record

        for expected in trace._acknowledgments:
            if isinstance(expected, bytes):
                assert await control.run(operation) == expected
            else:
                with pytest.raises(expected):
                    await control.run(operation)
        state = control._state
        assert client._source._effects == []
        assert (
            None if state.incarnation is None else state.incarnation.record
        ) == trace._final_record
        assert state.revision == trace._final_revision
        assert state.role_epoch == trace._final_epoch

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ('replies', 'expected', 'commands'),
    [
        (
            [('evalsha', NoScriptError('missing')), ('script_load', b'0' * 40)],
            BackendProtocolError,
            2,
        ),
        (
            [('evalsha', NoScriptError('missing')), ('script_load', b'bad-digest')],
            BackendProtocolError,
            2,
        ),
        (
            [
                ('evalsha', NoScriptError('missing')),
                ('script_load', COMMIT_SCRIPT_SHA),
                ('evalsha', NoScriptError('missing again')),
            ],
            BackendUnavailable,
            3,
        ),
        ([('evalsha', DataError('bad arguments'))], BackendProtocolError, 1),
        ([('evalsha', ValueError('private server reply'))], BackendUnavailable, 1),
        ([('evalsha', ResponseError('OOM capacity'))], BackendCapacity, 1),
        ([('evalsha', [b'unknown'])], BackendProtocolError, 1),
        ([('evalsha', StaleIncarnation('stale'))], StaleIncarnation, 1),
    ],
)
def test_commit_executor_preserves_error_classification_and_retry_bound(
    replies: list[tuple[str, object]],
    expected: type[Exception],
    commands: int,
) -> None:
    async def scenario() -> None:
        client = BackendClient(replies)
        with pytest.raises(expected):
            await module.execute_commit(
                client,
                prefix='results',
                lease=FillLease('key', 'token'),
                payload=b'payload',
                deadline=1.0,
                clock=lambda: 0.0,
            )
        assert len(client._source._calls) == commands

    asyncio.run(scenario())


def test_control_reuses_validation_on_alternating_covered_connections() -> None:
    async def scenario() -> None:
        client = _AsyncAtomicClient()
        client._pin_connection = 1
        clock = _FakeClock(start=0.0)
        control = module.AsyncValkeyControl(
            client,
            key_prefix='sgqc',
            validation_seconds=30,
            clock=clock,
            jitter_factory=lambda seconds: seconds,
        )
        original = await control._validate_incarnation()
        client._pin_connection = 2
        assert await control._validate_incarnation() == original
        assert client._source._info_calls == 4

        for connection in (1, 2, 1, 2):
            client._pin_connection = connection
            assert await control._validate_incarnation() == original
        assert client._source._info_calls == 4

    asyncio.run(scenario())


@pytest.mark.parametrize('phase', ['initial', 'load', 'retry'])
@pytest.mark.parametrize(
    'failure',
    [
        ValueError('private server reply'),
        BackendCapacity('capacity'),
        BackendProtocolError('protocol'),
        BackendUnavailable('unavailable'),
        StaleIncarnation('stale'),
    ],
)
def test_script_executor_translates_client_errors_and_preserves_backend_errors(
    phase: str, failure: Exception
) -> None:
    async def scenario() -> None:
        replies: list[tuple[str, object]] = []
        if phase != 'initial':
            replies.append(('evalsha', NoScriptError('missing')))
        if phase == 'retry':
            replies.append(('script_load', COMMIT_SCRIPT_SHA))
        replies.append(('script_load' if phase == 'load' else 'evalsha', failure))
        client = BackendClient(replies)
        expected = BackendUnavailable if isinstance(failure, ValueError) else type(failure)
        with pytest.raises(expected) as caught:
            await module.execute_script(
                client, script='script', sha=COMMIT_SCRIPT_SHA, keys=(), args=()
            )
        if isinstance(failure, ValueError):
            assert 'private server reply' not in str(caught.value)
            assert caught.value.__cause__ is None
            assert caught.value.__suppress_context__
        else:
            assert caught.value is failure

    asyncio.run(scenario())
