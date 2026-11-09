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
from tests._fakes import FakeClock
from tests._valkey_control_cases import (
    COUNTER_A,
    COUNTER_B,
    RECORD_A,
    RECORD_B,
    ControlTrace,
    PublicationTrace,
    control_traces,
    publication_traces,
)
from tests.valkey._async_helpers import BackendClient
from tests.valkey._control_helpers import AtomicControlClient, TraceControlClient


class _AsyncAtomicClient:
    def __init__(self) -> None:
        self.source = AtomicControlClient(replication_id='primary')
        self.source.control = RECORD_A
        self.source.counters.add(COUNTER_A)
        self.phase: str | None = None
        self.entered = asyncio.Event()
        self._proceed = asyncio.Event()
        self.calls: list[str] = []
        self.pins = 0
        self.pin_connection: int | None = None
        self.missing_script = False

    def epoch(self) -> tuple[int, int, None]:
        return self.source.epoch()

    async def step(self, name: str) -> None:
        self.calls.append(name)
        if name == self.phase:
            self.entered.set()
            await self._proceed.wait()

    async def evalsha(self, sha: str, numkeys: int, *args: ScriptArgument) -> object:
        if self.missing_script:
            self.missing_script = False
            raise NoScriptError('load required')
        name = 'create' if sha == CONTROL_CREATE_SCRIPT_SHA else 'rotate'
        assert sha in (CONTROL_CREATE_SCRIPT_SHA, CONTROL_ROTATE_SCRIPT_SHA)
        await self.step(name)
        return self.source.evalsha(sha, numkeys, *args)

    async def exists(self, key: str) -> object:
        await self.step('exists')
        return self.source.exists(key)

    async def get(self, key: str) -> object:
        await self.step('get')
        return self.source.get(key)

    async def info(self, section: str) -> object:
        await self.step('info')
        return self.source.info(section)

    @asynccontextmanager
    async def pin(self) -> AsyncGenerator['_AsyncAtomicClient | _AsyncPinnedConnectionView']:
        await self.step('pin')
        self.pins += 1
        try:
            if self.pin_connection is None:
                yield self
            else:
                yield _AsyncPinnedConnectionView(self, self.pin_connection)
        finally:
            self.pins -= 1
            await self.step('unpin')

    async def refresh(self) -> None:
        await self.step('refresh')

    async def refresh_endpoint(self) -> None:
        await self.step('discovery')

    async def script_load(self, script: bytes | str) -> object:
        await self.step('load')
        return self.source.script_load(script)


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
        client.pin_connection = 1
        clock = FakeClock(start=0.0)
        control = module.AsyncValkeyControl(
            client,
            key_prefix='sgqc',
            validation_seconds=30,
            clock=clock,
            jitter_factory=lambda seconds: seconds,
        )
        await control._validate_incarnation()
        client.pin_connection = 2
        await control._validate_incarnation()
        assert control._state.covered == frozenset({1, 2})

        client.pin_connection = 1
        await control._validate_incarnation(force=True)
        assert control._state.covered == frozenset({1})
        assert client.source.info_calls == 6

    asyncio.run(scenario())


def test_control_join_does_not_extend_validation_lifetime() -> None:
    async def scenario() -> None:
        client = _AsyncAtomicClient()
        client.pin_connection = 1
        clock = FakeClock(start=0.0)
        control = module.AsyncValkeyControl(
            client,
            key_prefix='sgqc',
            validation_seconds=30,
            clock=clock,
            jitter_factory=lambda seconds: seconds,
        )
        await control._validate_incarnation()
        clock.advance(20)
        client.pin_connection = 2
        await control._validate_incarnation()
        assert control._state.validated_at == 0.0
        clock.advance(9)
        client.pin_connection = 1
        await control._validate_incarnation()
        assert client.source.info_calls == 4

        clock.advance(1)
        client.pin_connection = 2
        await control._validate_incarnation()
        assert client.source.info_calls == 6
        assert control._state.covered == frozenset({2})
        assert control._state.validated_at == 30.0
        client.pin_connection = 1
        await control._validate_incarnation()
        assert client.source.info_calls == 8
        assert control._state.covered == frozenset({1, 2})
        assert control._state.validated_at == 30.0

    asyncio.run(scenario())


@pytest.mark.parametrize('trace', publication_traces(), ids=lambda trace: trace.name)
def test_control_publication_obeys_shared_revision_races(trace: PublicationTrace) -> None:
    async def scenario() -> None:
        client = _AsyncAtomicClient()
        clock = FakeClock(start=0.0)
        control = module.AsyncValkeyControl(
            client,
            key_prefix='sgqc',
            validation_seconds=30,
            clock=clock,
            jitter_factory=lambda seconds: seconds,
        )
        if trace.older_joins:
            client.pin_connection = 1
            await control._validate_incarnation()
            client.pin_connection = 2
            clock.advance(10)
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
            if trace.transition == 'clear':
                raise StaleIncarnation('stale')
            return incarnation.record

        old = asyncio.create_task(old_run())
        await entered.wait()
        if trace.older_joins:
            client.pin_connection = 3
            clock.advance(10)
        client.source.control = RECORD_B
        client.source.counters.add(COUNTER_B)
        try:
            newer_result: bytes | type[Exception] = await control.run(newer, force=True)
        except StaleIncarnation:
            newer_result = StaleIncarnation
        effects.append('newer-settled')
        proceed.set()
        assert (await old, newer_result) == trace.acknowledgments
        assert tuple(effects) == trace.effects
        state = control._state
        assert (
            None if state.incarnation is None else state.incarnation.record
        ) == trace.final_record
        assert state.revision == trace.final_revision
        if trace.final_covered is not None:
            assert state.covered == trace.final_covered
            assert state.validated_at == (None if trace.transition == 'clear' else 20.0)

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
        client.phase = phase
        if phase in ('create', 'load'):
            client.source.control = None
        if phase == 'load':
            client.missing_script = True
        if phase == 'rotate':
            client.source.counters.clear()

        async def operation(_pin: object, incarnation: Incarnation) -> bytes:
            await client.step('operation')
            if phase == 'settlement':
                client.phase = 'discovery'
            if phase == 'discard':
                client.source.connection_generation += 1
                client.phase = 'refresh'
            return incarnation.record

        task = asyncio.create_task(control.run(operation, force=True))
        await client.entered.wait()
        task.cancel('caller')
        with pytest.raises(asyncio.CancelledError, match='caller'):
            await task
        assert client.pins == 0
        assert control._state.incarnation is None
        assert client.calls.count('operation') <= 1

    asyncio.run(scenario())


class _AsyncTraceClient:
    def __init__(self, trace: ControlTrace) -> None:
        self.source = TraceControlClient(trace)

    def epoch(self) -> tuple[int, int, object | None]:
        return self.source.epoch()

    async def evalsha(self, sha: str, numkeys: int, *args: ScriptArgument) -> object:
        return self.source.evalsha(sha, numkeys, *args)

    async def exists(self, key: str) -> object:
        return self.source.exists(key)

    async def get(self, key: str) -> object:
        return self.source.get(key)

    async def info(self, section: str) -> object:
        return self.source.info(section)

    @asynccontextmanager
    async def pin(self) -> AsyncGenerator['_AsyncTraceClient']:
        with self.source.pin():
            yield self

    async def refresh(self) -> None:
        self.source.refresh()

    async def refresh_endpoint(self) -> None:
        self.source.refresh_endpoint()

    async def script_load(self, script: bytes | str) -> object:
        return self.source.script_load(script)


@pytest.mark.parametrize('trace', control_traces(), ids=lambda trace: trace.name)
def test_control_executes_shared_semantic_traces(trace: ControlTrace) -> None:
    async def scenario() -> None:
        client = _AsyncTraceClient(trace)
        control = module.AsyncValkeyControl(
            client,
            key_prefix='sgqc',
            validation_seconds=30,
            clock=lambda: 0.0,
            jitter_factory=lambda seconds: seconds,
            nonce_factory=lambda: trace.proposal_nonce,
        )

        async def operation(_client: object, incarnation: Incarnation) -> bytes:
            decode_change_reply(client.source.effect('operation'))
            return incarnation.record

        for expected in trace.acknowledgments:
            if isinstance(expected, bytes):
                assert await control.run(operation) == expected
            else:
                with pytest.raises(expected):
                    await control.run(operation)
        state = control._state
        assert client.source.effects == []
        assert (
            None if state.incarnation is None else state.incarnation.record
        ) == trace.final_record
        assert state.revision == trace.final_revision
        assert state.role_epoch == trace.final_epoch

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
        assert len(client.source.calls) == commands

    asyncio.run(scenario())


def test_control_replica_validation_reply_discards_pool() -> None:
    class _ReplicaSource(AtomicControlClient):
        def __init__(self) -> None:
            super().__init__(replication_id='primary')
            self.demoted = False

        def info(self, section: str) -> object:
            _ = section
            self.info_calls += 1
            role = 'slave' if self.demoted else 'master'
            return {'role': role, 'master_replid': self.replication_id}

    async def scenario() -> None:
        client = _AsyncAtomicClient()
        source = _ReplicaSource()
        source.control = RECORD_A
        source.counters.add(COUNTER_A)
        client.source = source
        control = module.AsyncValkeyControl(
            client,
            key_prefix='sgqc',
            validation_seconds=30,
            clock=FakeClock(start=0.0),
            jitter_factory=lambda seconds: seconds,
        )
        await control._validate_incarnation()
        source.demoted = True

        with pytest.raises(BackendUnavailable):
            await control._validate_incarnation(force=True)

        assert client.calls.count('refresh') == 1
        assert control._state.incarnation is None

    asyncio.run(scenario())


def test_control_reuses_validation_on_alternating_covered_connections() -> None:
    async def scenario() -> None:
        client = _AsyncAtomicClient()
        client.pin_connection = 1
        clock = FakeClock(start=0.0)
        control = module.AsyncValkeyControl(
            client,
            key_prefix='sgqc',
            validation_seconds=30,
            clock=clock,
            jitter_factory=lambda seconds: seconds,
        )
        original = await control._validate_incarnation()
        client.pin_connection = 2
        assert await control._validate_incarnation() == original
        assert client.source.info_calls == 4

        for connection in (1, 2, 1, 2):
            client.pin_connection = connection
            assert await control._validate_incarnation() == original
        assert client.source.info_calls == 4

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
