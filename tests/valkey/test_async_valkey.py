import asyncio
import socket
import ssl
import subprocess
import sys
from collections import deque
from collections.abc import AsyncGenerator, Awaitable, Callable, Coroutine, Iterable, Mapping
from contextlib import asynccontextmanager
from dataclasses import replace
from traceback import format_exception
from typing import Any, Literal, Protocol, TypeAlias, TypeVar, cast

import pytest
import valkey
from valkey.asyncio import Connection, Sentinel, Valkey
from valkey.asyncio.connection import AbstractConnection, SSLConnection
from valkey.asyncio.retry import Retry
from valkey.backoff import NoBackoff
from valkey.exceptions import ConnectionError as ValkeyConnectionError
from valkey.exceptions import DataError, NoScriptError, ReadOnlyError, ResponseError
from valkey.exceptions import TimeoutError as ValkeyTimeoutError

import sg_query_cache._async_support as support
import sg_query_cache._valkey_common as common_module
import sg_query_cache.valkey as sync_module
import sg_query_cache.valkey_async as module
from sg_query_cache._async_cache import AsyncQueryCache
from sg_query_cache._async_local import AsyncLocalBackend
from sg_query_cache._async_valkey_control import AsyncValkeyControl
from sg_query_cache._backend import (
    BackendCapacity,
    BackendProtocolError,
    BackendUnavailable,
    StaleIncarnation,
)
from sg_query_cache._cache_common import TimedResult, decode_hit, encode_result
from sg_query_cache._codec import encode_value
from sg_query_cache._config import (
    CacheSettings,
    L1Settings,
    L2Settings,
    LimitSettings,
    ValkeyRoleSettings,
    ValkeySettings,
    _ValkeyMode,
    parse_sentinel_address,
)
from sg_query_cache._keys import KeyBuilder
from sg_query_cache._metrics import NullMetrics
from sg_query_cache._model import (
    DependencyPlan,
    DependencyTag,
    ExactCall,
    FillLease,
    GenerationChange,
    Identity,
)
from sg_query_cache._valkey_scripts import (
    COMMIT_SCRIPT_SHA,
    CONTROL_CREATE_SCRIPT_SHA,
    CONTROL_ROTATE_SCRIPT_SHA,
    RESOLVE_SCRIPT,
    RESOLVE_SCRIPT_SHA,
)
from tests._exceptions import exception_text as _exception_text
from tests._fakes import (
    PRIMARY_INFO,
    FakeClock,
    RecordingMetrics,
    valkey_passwords,
    valkey_test_settings,
)
from tests._tls import TLS_CA, perform_tls_handshake
from tests._valkey_control_cases import RECORD_A, RECORD_B
from tests.valkey._async_helpers import BackendClient

# Fake transports answer at once, so only host scheduling delay can spend an operation or
# close budget; a generous one keeps that delay from failing tests about other behavior.
_FAKE_TRANSPORT_LIMITS = LimitSettings(operation_timeout_seconds=5.0)
_T = TypeVar('_T')

_AsyncClientPath: TypeAlias = Literal[
    'managed',
    'sentinel-discovery',
    'sentinel-primary',
    'standalone',
]


class _DeadlineRole:
    def __init__(self, role: str, clock: FakeClock, *, half_open: bool) -> None:
        self.role = role
        self.metrics = RecordingMetrics()
        settings = CacheSettings(
            limits=LimitSettings(operation_timeout_seconds=5, circuit_failure_threshold=1)
        )
        self.client = BackendClient(
            [
                ('info', {'role': 'master', 'master_replid': 'primary'}),
                ('get', RECORD_A),
                ('exists', 1),
                ('info', {'role': 'master', 'master_replid': 'primary'}),
                ('get', RECORD_A),
                ('exists', 1),
            ]
        )
        metadata = module.AsyncValkeyMetadataStore(self.client, settings, self.metrics)
        if role == 'metadata':
            self.owner = metadata
            self.guard = metadata._guard
            self.breaker = metadata._breaker
        else:
            backend = module.AsyncValkeyBackend(self.client, metadata, settings, self.metrics)
            self.owner = backend
            self.guard = backend._result_guard
            self.breaker = backend._result_breaker
        self.breaker._clock = clock
        if half_open:
            self.breaker.fail()
            clock.advance(settings.limits.circuit_probe_seconds)
        self.metrics.increments.clear()

    async def operation(self) -> object:
        if isinstance(self.owner, module.AsyncValkeyBackend):
            return await self.owner.lookup('key')
        return await self.owner.resolve((DependencyTag.site('site'),))

    def transitions(self, outcome: str) -> list[dict[str, str]]:
        return [
            labels
            for name, labels in self.metrics.increments
            if name == 'backend_failure'
            and labels.get('backend_operation') == self.role
            and labels.get('outcome') == outcome
        ]


class _DeadlineStall:
    def __init__(self, clock: FakeClock, settlement: str) -> None:
        self.clock = clock
        self.settlement = settlement
        self.cleanup_entered = asyncio.Event()
        self.cleanup_release = asyncio.Event()
        self.cancellations = 0
        self.calls = 0
        self.expired_at: float | None = None

    async def __call__(self, _sha: str, _numkeys: int, *_args: object) -> object:
        self.calls += 1
        self.clock.advance(1)
        try:
            await _deadline_turn()
            # Advance the fallback timer so an ignored deadline fails without hanging.
            self.clock.advance(5)
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancellations += 1
            self.expired_at = self.clock()
            if self.settlement != 'stalled':
                self.cleanup_entered.set()
                try:
                    await self.cleanup_release.wait()
                except asyncio.CancelledError:
                    self.cancellations += 1
                    raise
            raise


class _PrimaryDiscovery:
    async def discover_master(self, _service_name: str) -> tuple[str, int]:
        return 'primary', 6379


class _TLSContextConnection(Protocol):
    def _connection_arguments(self) -> Mapping[str, object]: ...


class _Transport:
    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.wires: list[_Wire] = []
        self.replies: deque[bytes] = deque()
        self.connect_entered = asyncio.Event()
        self.connect_proceed = asyncio.Event()
        self.connect_proceed.set()

        async def connect(connection: Connection) -> None:
            wire = _Wire(self.replies)
            self.wires.append(wire)
            connection._reader = wire.reader
            connection._writer = cast('asyncio.StreamWriter', wire)
            self.connect_entered.set()
            await self.connect_proceed.wait()

        monkeypatch.setattr(Connection, '_connect', connect)


class _Wire:
    def __init__(self, replies: deque[bytes]) -> None:
        self.reader = asyncio.StreamReader()
        self.sent: list[bytes] = []
        self.closed = False
        self.drain_entered = asyncio.Event()
        self.drain_proceed = asyncio.Event()
        self.drain_proceed.set()
        self.close_error: Exception | None = None
        self._replies = replies

    def close(self) -> None:
        self.closed = True
        if self.close_error is not None:
            raise self.close_error

    async def drain(self) -> None:
        self.drain_entered.set()
        await self.drain_proceed.wait()

    async def wait_closed(self) -> None:
        pytest.fail('owned connections must not wait for graceful shutdown')

    def writelines(self, lines: Iterable[bytes]) -> None:
        self.sent.append(b''.join(lines))
        if self._replies:
            self.reader.feed_data(self._replies.popleft())


def _bulk(payload: bytes) -> bytes:
    return b'$' + str(len(payload)).encode() + b'\r\n' + payload + b'\r\n'


async def _configured_connection(
    path: _AsyncClientPath,
    *,
    host: str = 'cache.test',
    tls: bool,
) -> tuple[module._ConfiguredRole, AbstractConnection]:
    mode = 'sentinel' if path.startswith('sentinel-') else path
    role = ValkeyRoleSettings(
        host,
        'PASSWORD',
        mode=cast('_ValkeyMode', mode),
        tls=tls,
        sentinels=('one:26379', 'two:26379', 'three:26379') if mode == 'sentinel' else (),
        service_name='primary' if mode == 'sentinel' else None,
    )
    configured = await module._create_client(role, CacheSettings(), 'password', NullMetrics())
    raw = configured._discovery[0] if path == 'sentinel-discovery' else configured._client
    assert raw is not None
    return configured, raw.connection_pool.make_connection()


async def _deadline_turn() -> None:
    loop = asyncio.get_running_loop()
    ready = loop.create_future()
    loop.call_soon(ready.set_result, None)
    await ready


def _proving_role(mode: str) -> module._RoleClient:
    tracker = module._ConnectionGeneration(prove_primary=True)

    def factory() -> module._RawClient:
        pool = (
            module._SentinelConnectionPool(
                'service',
                _PrimaryDiscovery(),
                valkey_connect_func=tracker.initialize,
                lib_name=None,
                lib_version=None,
            )
            if mode == 'sentinel'
            else module._ConnectionPool(
                valkey_connect_func=tracker.initialize,
                lib_name=None,
                lib_version=None,
            )
        )
        return cast('module._RawClient', cast('object', Valkey.from_pool(pool)))

    return module._RoleClient(
        factory, tracker, module._sentinel_endpoint if mode == 'sentinel' else None
    )


def test_backend_close_applies_one_budget_after_join_and_attempts_later_roles() -> None:
    async def scenario() -> None:
        result, metadata = BackendClient([]), BackendClient([])
        result.close_proceed.clear()
        settings = CacheSettings(limits=LimitSettings(operation_timeout_seconds=0.001))
        backend = module.AsyncValkeyBackend(
            result, cast('module.AsyncValkeyMetadataStore', metadata), settings, NullMetrics()
        )
        closer = asyncio.create_task(backend.close())
        try:
            # Only a hung close should exhaust this guard; host load can delay the budget's expiry.
            done, _pending = await asyncio.wait((closer,), timeout=5.0)
            assert done == {closer}
            with pytest.raises(BackendUnavailable, match='deadline'):
                await closer
            assert result.closed == metadata.closed == 1
        finally:
            result.close_proceed.set()
            await asyncio.gather(closer, return_exceptions=True)

    asyncio.run(scenario())


def test_backend_close_joins_repeated_caller_cancellation_and_attempts_all_failures() -> None:
    async def scenario() -> None:
        result, metadata = BackendClient([]), BackendClient([])
        result.close_proceed.clear()
        result.close_error = ValueError('first close failed')
        backend = module.AsyncValkeyBackend(
            result,
            cast('module.AsyncValkeyMetadataStore', metadata),
            CacheSettings(l2=L2Settings(enabled=True)),
            NullMetrics(),
        )
        first, second = asyncio.create_task(backend.close()), asyncio.create_task(backend.close())
        await result.close_entered.wait()
        first.cancel('first')
        barrier = asyncio.Event()
        asyncio.get_running_loop().call_soon(barrier.set)
        await barrier.wait()
        first.cancel('second')
        result.close_proceed.set()
        with pytest.raises(asyncio.CancelledError, match='first'):
            await first
        with pytest.raises(ValueError, match='first close failed'):
            await second
        assert first.cancelling() == 2
        assert result.closed == metadata.closed == 1

    asyncio.run(scenario())


def test_backend_close_keeps_first_interruption_before_later_ordinary_failure() -> None:
    async def scenario() -> None:
        class _InterruptedClose(BackendClient):
            async def close(self) -> None:
                await super().close()
                task = asyncio.current_task()
                assert task is not None
                task.cancel('independent first resource')
                await asyncio.Event().wait()

        result, metadata = _InterruptedClose([]), BackendClient([])
        metadata.close_error = ValueError('later resource failure')
        backend = module.AsyncValkeyBackend(
            result,
            cast('module.AsyncValkeyMetadataStore', metadata),
            CacheSettings(),
            NullMetrics(),
        )
        with pytest.raises(BackendUnavailable, match='cleanup task cancelled'):
            await backend.close()
        assert result.closed == metadata.closed == 1

    asyncio.run(scenario())


def test_backend_expired_operation_opens_breaker_without_mutation_retry() -> None:
    async def scenario() -> None:
        class _BlockingClient(BackendClient):
            async def evalsha(self, sha: str, numkeys: int, *args: object) -> object:
                self.source.calls.append(('evalsha', sha, numkeys, *args))
                await asyncio.Event().wait()
                raise AssertionError('blocked command returned')

        result, metadata = _BlockingClient([]), BackendClient([])
        settings = CacheSettings(
            limits=LimitSettings(operation_timeout_seconds=0.001, circuit_failure_threshold=1)
        )
        backend = module.AsyncValkeyBackend(
            result, cast('module.AsyncValkeyMetadataStore', metadata), settings, NullMetrics()
        )
        with pytest.raises(BackendUnavailable, match='deadline'):
            await backend.reserve('key', 5)
        with pytest.raises(BackendUnavailable, match='breaker'):
            await backend.reserve('key', 5)
        assert len(result.source.calls) == 1
        await backend.close()

    asyncio.run(scenario())


def test_backend_metadata_operations_use_one_fenced_script_and_distinct_role() -> None:
    async def scenario() -> None:
        metadata = BackendClient(
            [
                ('info', {'role': 'master', 'master_replid': 'primary'}),
                ('get', RECORD_A),
                ('exists', 1),
                ('info', {'role': 'master', 'master_replid': 'primary'}),
                ('get', RECORD_A),
                ('exists', 1),
                ('evalsha', [b'ok', 3, 4]),
                ('evalsha', [b'ok', 2, 0]),
                ('evalsha', [b'ok', 1, 0]),
            ]
        )
        result = BackendClient([])
        settings, metrics = (
            CacheSettings(l2=L2Settings(enabled=True), limits=_FAKE_TRANSPORT_LIMITS),
            RecordingMetrics(),
        )
        store = module.AsyncValkeyMetadataStore(metadata, settings, metrics)
        backend = module.AsyncValkeyBackend(result, store, settings, metrics)
        tags = (
            DependencyTag.site('production'),
            DependencyTag.type_membership('production', 'Asset'),
        )
        vector = await backend.resolve(tags)
        assert vector.entries == tuple(zip(tags, (3, 4), strict=True))
        await backend.change(tuple(GenerationChange(tag) for tag in tags))
        await backend.flush_site('production', 'operator')
        assert len([call for call in metadata.source.calls if call[0] == 'evalsha']) == 3
        assert result.source.calls == []
        await backend.close()

    asyncio.run(scenario())


def test_backend_result_lease_round_trip_decodes_hit_and_closes_both_roles() -> None:
    async def scenario() -> None:
        backend_type = module.AsyncValkeyBackend
        payload = encode_value({'id': 7})
        result = BackendClient(
            [
                ('evalsha', [b'reserved']),
                ('evalsha', [b'committed']),
                ('evalsha', [b'hit', payload, 5000]),
                ('evalsha', [b'released']),
            ]
        )
        metadata = BackendClient([])
        backend = backend_type(
            result,
            cast('module.AsyncValkeyMetadataStore', metadata),
            CacheSettings(l2=L2Settings(enabled=True), limits=_FAKE_TRANSPORT_LIMITS),
            NullMetrics(),
        )
        lease = await backend.reserve('qualified', 5)
        assert lease is not None
        assert await backend.commit(lease, payload, 5)
        hit = await backend.lookup('qualified')
        assert hit is not None
        assert hit.decoded is not None
        assert hit.decoded.value == {'id': 7}
        assert 0 < hit.remaining_seconds <= 5
        assert await backend.release(lease)
        await backend.close()
        await backend.close()
        assert (result.closed, metadata.closed) == (1, 1)

    asyncio.run(scenario())


def test_cleanup_retranslation_keeps_failure_class(monkeypatch: pytest.MonkeyPatch) -> None:
    async def scenario() -> None:
        configured = await module._create_client(
            ValkeyRoleSettings('cache.test', 'PASSWORD', tls=False),
            CacheSettings(),
            'password',
            NullMetrics(),
        )

        async def close(_client: Valkey, **_kwargs: object) -> None:
            raise ValkeyTimeoutError('credential-canary-cleanup')

        monkeypatch.setattr(Valkey, 'aclose', close)
        with pytest.raises(BackendUnavailable) as caught:
            await configured.close()
        assert caught.value.failure_class == 'timeout'
        assert str(caught.value) == 'Valkey operation is unavailable (timeout)'
        assert caught.value.__cause__ is None
        assert caught.value.__context__ is None

    asyncio.run(scenario())


def test_configured_backend_uses_maintained_native_protocol_for_all_roles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        transport = _Transport(monkeypatch)
        payload = encode_value({'id': 7})
        primary = PRIMARY_INFO
        transport.replies.extend(
            (
                b'+OK\r\n',
                b'+OK\r\n',
                b'+OK\r\n',
                b'*1\r\n' + _bulk(b'reserved'),
                b'*1\r\n' + _bulk(b'committed'),
                b'*3\r\n' + _bulk(b'hit') + _bulk(payload) + b':5000\r\n',
                b'*1\r\n' + _bulk(b'released'),
                b'+OK\r\n',
                b'+OK\r\n',
                b'+OK\r\n',
                _bulk(primary),
                _bulk(RECORD_A),
                b':1\r\n',
                _bulk(primary),
                _bulk(RECORD_A),
                b':1\r\n',
                b'*2\r\n' + _bulk(b'ok') + b':7\r\n',
            )
        )
        settings = valkey_test_settings(limits=_FAKE_TRANSPORT_LIMITS, mode='standalone')
        backend = await module.AsyncValkeyBackend.from_settings(
            settings,
            valkey_passwords(settings),
            NullMetrics(),
        )
        lease = await backend.reserve('key', 5)
        assert lease is not None
        assert await backend.commit(lease, payload, 5)
        hit = await backend.lookup('key')
        assert hit is not None
        assert hit.payload == payload
        assert await backend.release(lease)
        tag = DependencyTag.site('production')
        assert (await backend.resolve((tag,))).entries == ((tag, 7),)
        await backend.close()
        assert len(transport.wires) == 2
        assert all(wire.closed for wire in transport.wires)
        assert not transport.replies
        assert sum(b'EVALSHA' in command for wire in transport.wires for command in wire.sent) == 5

    asyncio.run(scenario())


def test_configured_pool_replacement_redacts_constructor_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        settings = CacheSettings(l2=L2Settings(enabled=True), limits=_FAKE_TRANSPORT_LIMITS)
        backend = await module.AsyncValkeyBackend.from_settings(
            settings,
            valkey_passwords(settings),
            NullMetrics(),
        )

        def fail(_pool: object) -> Valkey:
            raise ValueError('credential-bearing replacement failure')

        monkeypatch.setattr(Valkey, 'from_pool', fail)
        try:
            with pytest.raises(BackendUnavailable) as caught:
                await backend._result_client.refresh()
            assert 'credential' not in str(caught.value)
            assert caught.value.__context__ is None
        finally:
            await backend.close()

    asyncio.run(scenario())


def test_construction_and_factory_cleanup_metrics_carry_failure_class() -> None:
    async def scenario() -> None:
        class _FailingClose:
            async def close(self) -> None:
                raise BackendUnavailable('redacted', failure_class='connection')

        metrics = RecordingMetrics()
        resource = _FailingClose()
        await module._cleanup_construction((resource.close,), ValueError('construction'), metrics)
        assert (
            'backend_failure',
            {
                'backend_operation': 'construction_close',
                'outcome': 'failure',
                'failure_class': 'connection',
            },
        ) in metrics.increments

    asyncio.run(scenario())


def test_expired_commit_keeps_original_sg_result_and_releases_original_lease(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        clock = FakeClock()
        monkeypatch.setattr(module, 'monotonic', clock)
        settings = CacheSettings(
            l1=L1Settings(enabled=False),
            l2=L2Settings(enabled=True),
            limits=_FAKE_TRANSPORT_LIMITS,
        )

        def load(_call: tuple[object, ...]) -> object:
            clock.advance(1.0)
            return COMMIT_SCRIPT_SHA

        result = BackendClient(
            [
                ('evalsha', [b'miss']),
                ('evalsha', [b'reserved']),
                ('evalsha', [b'miss']),
                ('evalsha', NoScriptError('missing')),
                ('script_load', load),
                ('evalsha', [b'released']),
            ]
        )
        backend = module.AsyncValkeyBackend(
            result,
            AsyncLocalBackend(settings, clock, NullMetrics()),
            settings,
            NullMetrics(),
        )
        cache = AsyncQueryCache(
            settings,
            Identity('script', 'studio', 'render'),
            key_builder=KeyBuilder(),
            l2=backend,
        )
        answer = {'items': [1]}

        async def fetch() -> object:
            return answer

        value = await cache.read(
            ExactCall('find', (('entity_type', 'Asset'), ('filters', ()))),
            DependencyPlan('event_aware', (DependencyTag.site('production'),), 'event_aware'),
            fetch,
            {'localized': False, 'convert_datetimes_to_utc': True},
            per_call_age=0.05,
        )
        assert value is answer
        assert result.source.replies == []
        reserve, attempt, release = (
            result.source.calls[1],
            result.source.calls[3],
            result.source.calls[-1],
        )
        assert reserve[-2] == attempt[-3] == release[-1]
        await cache.close()

    asyncio.run(scenario())


def test_factory_owns_native_two_role_pools_and_zero_retry() -> None:
    async def scenario() -> None:
        settings = CacheSettings(l2=L2Settings(enabled=True), limits=_FAKE_TRANSPORT_LIMITS)
        backend = await module.AsyncValkeyBackend.from_settings(
            settings,
            valkey_passwords(settings),
            NullMetrics(),
        )
        metadata = cast(
            'module._RoleClient',
            cast('module.AsyncValkeyMetadataStore', backend._metadata)._client,
        )._active()
        result = cast('module._RoleClient', backend._result_client)._active()
        assert metadata.connection_pool is not result.connection_pool
        for raw in (metadata, result):
            assert isinstance(raw.connection_pool, module._ConnectionPool)
            options: dict[str, object] = cast('Any', raw.connection_pool).connection_kwargs
            retry = options['retry']
            assert isinstance(retry, Retry)
            assert retry._retries == 0
        await backend.close()
        with pytest.raises(RuntimeError, match='closed'):
            await backend.lookup('key')

    asyncio.run(scenario())


def test_factory_partial_construction_closes_already_owned_primary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        original = Valkey.from_pool
        originals: list[Valkey] = []
        closed: list[Valkey] = []
        original_close = Valkey.aclose

        def create(pool: module._ConnectionPool) -> Valkey:
            if originals:
                raise ValueError('credential-bearing constructor failure')
            client = original(pool)
            originals.append(client)
            return client

        async def close(client: Valkey, close_connection_pool: bool | None = None) -> None:
            closed.append(client)
            await original_close(client, close_connection_pool=close_connection_pool)

        monkeypatch.setattr(Valkey, 'from_pool', create)
        monkeypatch.setattr(Valkey, 'aclose', close)
        settings = CacheSettings(l2=L2Settings(enabled=True))
        with pytest.raises(BackendUnavailable) as error:
            await module.AsyncValkeyBackend.from_settings(
                settings,
                {
                    settings.l2.valkey.metadata.password_env: 'metadata',
                    settings.l2.valkey.result.password_env: 'result',
                },
                NullMetrics(),
            )
        assert 'credential' not in str(error.value)
        assert closed == originals

    asyncio.run(scenario())


def test_guard_failure_metrics_carry_failure_class() -> None:
    async def scenario() -> None:
        metrics = RecordingMetrics()
        settings = CacheSettings(limits=_FAKE_TRANSPORT_LIMITS)
        store = module.AsyncValkeyMetadataStore(BackendClient([]), settings, metrics)
        backend = module.AsyncValkeyBackend(BackendClient([]), store, settings, metrics)

        async def fail() -> None:
            raise BackendUnavailable('redacted', failure_class='timeout')

        with pytest.raises(BackendUnavailable):
            await store._guard('metadata', fail)
        with pytest.raises(BackendUnavailable):
            await backend._result_guard('result', fail)

        assert (
            'backend_failure',
            {'backend_operation': 'metadata', 'outcome': 'failure', 'failure_class': 'timeout'},
        ) in metrics.increments
        assert (
            'backend_failure',
            {'backend_operation': 'result', 'outcome': 'failure', 'failure_class': 'timeout'},
        ) in metrics.increments

    asyncio.run(scenario())


def test_inline_backend_timeout_does_not_replace_actual_caller_cancellation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        expired, finish = asyncio.Event(), asyncio.Event()
        observed: list[asyncio.CancelledError] = []
        original_shield = support.shield

        async def observe_cancellation(future: asyncio.Future[None]) -> None:
            try:
                await original_shield(future)
            except asyncio.CancelledError as error:
                if error.args == ('actual caller',):
                    observed.append(error)
                raise

        class _CooperativeClose(BackendClient):
            async def close(self) -> None:
                await super().close()
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

        result, metadata = _CooperativeClose([]), BackendClient([])
        backend = module.AsyncValkeyBackend(
            result,
            cast('module.AsyncValkeyMetadataStore', metadata),
            CacheSettings(limits=LimitSettings(operation_timeout_seconds=0.001)),
            NullMetrics(),
        )
        monkeypatch.setattr(module, 'create_task', cancel_shared)
        monkeypatch.setattr(support, 'shield', observe_cancellation)
        closer = asyncio.create_task(backend.close())
        ordinary = asyncio.create_task(backend.close())
        await asyncio.wait_for(expired.wait(), timeout=2)
        closer.cancel('actual caller')
        barrier = asyncio.Event()
        asyncio.get_running_loop().call_soon(barrier.set)
        await barrier.wait()
        assert not closer.done()
        finish.set()
        with pytest.raises(asyncio.CancelledError, match='actual caller') as raised:
            await asyncio.wait_for(closer, timeout=2)
        assert raised.value is observed[0]
        assert closer.cancelling() == 1
        with pytest.raises(BackendUnavailable):
            await ordinary
        assert result.closed == metadata.closed == 1
        assert backend._lifecycle._admission.closed

    asyncio.run(scenario())


def test_installed_maintained_pool_releases_reuses_and_disconnects_owned_connections(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    disconnected: list[AbstractConnection] = []
    original_disconnect = AbstractConnection.disconnect

    async def connect(_connection: AbstractConnection) -> None:
        return None

    async def readable(_connection: AbstractConnection) -> bool:
        return False

    async def disconnect(connection: AbstractConnection, nowait: bool = False) -> None:
        assert nowait
        disconnected.append(connection)
        await original_disconnect(connection, nowait=nowait)

    monkeypatch.setattr(AbstractConnection, 'connect', connect)
    monkeypatch.setattr(AbstractConnection, 'can_read_destructive', readable)
    monkeypatch.setattr(AbstractConnection, 'disconnect', disconnect)

    async def scenario() -> None:
        pool = module._ConnectionPool()
        owner = Valkey.from_pool(pool)
        first = owner.client()
        second = owner.client()
        await first.initialize()
        await second.initialize()
        first_connection = first.connection
        second_connection = second.connection
        assert first_connection is not None
        assert second_connection is not None
        assert first_connection is not second_connection
        await first.aclose()
        await pool.settle_release(second_connection, pool.borrow_tokens[second_connection])
        second.connection = None
        await second.aclose()
        reused = owner.client()
        await reused.initialize()
        assert reused.connection in (first_connection, second_connection)
        await pool.disconnect(inuse_connections=False)
        assert len(disconnected) == 1
        assert reused.connection not in disconnected
        disconnected.clear()
        await pool.settle_disconnect()
        assert set(disconnected) == {first_connection, second_connection}
        await reused.aclose()
        await owner.aclose()

    asyncio.run(scenario())


def test_maintained_sentinel_discovers_primary_and_uses_owned_pool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        transport = _Transport(monkeypatch)
        tracker = module._ConnectionGeneration(prove_primary=True)
        discovery_pool = module._ConnectionPool(lib_name=None, lib_version=None)
        sentinel = cast(
            '_SentinelAPI',
            Sentinel(
                (parse_sentinel_address('[::1]:26379'),),
                min_other_sentinels=2,
                sentinel_kwargs={'connection_pool': discovery_pool},
                lib_name=None,
                lib_version=None,
                valkey_connect_func=tracker.initialize,
            ),
        )
        fields = [
            b'name',
            b'service',
            b'ip',
            b'primary',
            b'port',
            b'6379',
            b'flags',
            b'master',
            b'num-other-sentinels',
            b'2',
        ]
        reply = b'*1\r\n*10\r\n' + b''.join(
            b'$' + str(len(field)).encode() + b'\r\n' + field + b'\r\n' for field in fields
        )
        transport.replies.append(reply)
        assert await sentinel.discover_master('service') == ('primary', 6379)
        client = sentinel.master_for(
            'service', connection_pool_class=module._SentinelConnectionPool
        )
        pinned = client.client()
        transport.replies.extend((reply, _bulk(PRIMARY_INFO)))
        await pinned.initialize()
        assert tracker.for_connection(pinned.connection) == 1
        transport.replies.append(b'$2\r\nok\r\n')
        assert await pinned.execute_command('GET', 'key') == b'ok'
        await pinned.aclose(close_connection_pool=False)
        await client.aclose(close_connection_pool=True)
        for discovery in sentinel.sentinels:
            await discovery.aclose(close_connection_pool=True)
        assert all(wire.closed for wire in transport.wires)
        assert not any(b'INFO' in command for command in transport.wires[0].sent)
        assert b'INFO' in transport.wires[1].sent[0]
        assert b'GET' in transport.wires[1].sent[1]

    asyncio.run(scenario())


def test_managed_warm_hit_sends_only_two_scripts_across_both_roles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        transport = _Transport(monkeypatch)
        primary = _bulk(PRIMARY_INFO)
        setup = (b'+OK\r\n',) * 3
        validation = (primary, _bulk(RECORD_A), b':1\r\n') * 2
        resolved = b'*2\r\n' + _bulk(b'ok') + b':7\r\n'
        payload = encode_value({'id': 7})
        hit = b'*3\r\n' + _bulk(b'hit') + _bulk(payload) + b':5000\r\n'
        transport.replies.extend((*setup, primary, *validation, resolved, *setup, primary, hit))
        settings = valkey_test_settings(limits=_FAKE_TRANSPORT_LIMITS, mode='managed')
        backend = await module.AsyncValkeyBackend.from_settings(
            settings,
            valkey_passwords(settings),
            NullMetrics(),
        )
        tags = (DependencyTag.site('production'),)
        assert (await backend.resolve(tags)).entries == ((tags[0], 7),)
        assert await backend.lookup('key') is not None
        assert len(transport.wires) == 2
        marks = [len(wire.sent) for wire in transport.wires]
        transport.replies.extend((resolved, hit))
        assert (await backend.resolve(tags)).entries == ((tags[0], 7),)
        assert await backend.lookup('key') is not None
        commands = [
            command
            for wire, mark in zip(transport.wires, marks, strict=True)
            for command in wire.sent[mark:]
        ]
        assert len(commands) == 2
        assert all(b'EVALSHA' in command and b'INFO' not in command for command in commands)
        assert not transport.replies
        await backend.close()

    asyncio.run(scenario())


def test_missing_valkey_async_submodule_is_not_rewritten() -> None:
    result = subprocess.run(
        [
            sys.executable,
            '-c',
            """
import sys
import valkey

sys.modules['valkey.asyncio'] = None
try:
    import sg_query_cache.valkey_async
except ModuleNotFoundError as error:
    assert error.name == 'valkey.asyncio'
    assert error.__cause__ is None
else:
    raise AssertionError('missing valkey.asyncio import succeeded')
""",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_missing_valkey_async_symbol_is_not_rewritten() -> None:
    result = subprocess.run(
        [
            sys.executable,
            '-c',
            """
import valkey.asyncio

del valkey.asyncio.Valkey
try:
    import sg_query_cache.valkey_async
except ImportError as error:
    assert type(error) is ImportError
    assert "cannot import name 'Valkey'" in str(error)
    assert error.__cause__ is None
else:
    raise AssertionError('missing valkey.asyncio.Valkey import succeeded')
""",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_partial_sentinel_discovery_construction_closes_every_allocated_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        original, original_close = Valkey.from_pool, Valkey.aclose
        allocated: list[Valkey] = []
        closed: list[Valkey] = []

        def create(pool: module._ConnectionPool) -> Valkey:
            if len(allocated) == 2:
                raise ValueError('third discovery failed')
            client = original(pool)
            allocated.append(client)
            return client

        async def close(client: Valkey, close_connection_pool: bool | None = None) -> None:
            closed.append(client)
            await original_close(client, close_connection_pool=close_connection_pool)

        monkeypatch.setattr(Valkey, 'from_pool', create)
        monkeypatch.setattr(Valkey, 'aclose', close)
        base = CacheSettings()
        metadata = replace(
            base.l2.valkey.metadata,
            mode='sentinel',
            sentinels=('one:26379', 'two:26379', 'three:26379'),
            service_name='metadata',
        )
        result = replace(
            base.l2.valkey.result,
            mode='sentinel',
            sentinels=metadata.sentinels,
            service_name='result',
        )
        settings = replace(
            base,
            l2=L2Settings(enabled=True, valkey=ValkeySettings(metadata=metadata, result=result)),
        )
        with pytest.raises(BackendUnavailable):
            await module.AsyncValkeyBackend.from_settings(
                settings,
                {metadata.password_env: 'metadata', result.password_env: 'result'},
                NullMetrics(),
            )
        assert len(allocated) == 2
        assert closed == allocated

    asyncio.run(scenario())


def test_project_import_missing_valkey_name_is_not_rewritten() -> None:
    result = subprocess.run(
        [
            sys.executable,
            '-c',
            """
import builtins
import sg_query_cache
import valkey

original_import = builtins.__import__

def fail_project_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name == 'sg_query_cache._async_valkey_control':
        raise ModuleNotFoundError('project import failed', name='valkey')
    return original_import(name, globals, locals, fromlist, level)

builtins.__import__ = fail_project_import
try:
    import sg_query_cache.valkey_async
except ModuleNotFoundError as error:
    assert str(error) == 'project import failed'
    assert error.name == 'valkey'
    assert error.__cause__ is None
else:
    raise AssertionError('project import failure was suppressed')
""",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_rotation_required_rotates_control_before_reporting_stale(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        transport = _Transport(monkeypatch)
        primary = _bulk(PRIMARY_INFO)
        transport.replies.extend(
            (
                *((b'+OK\r\n',) * 3),
                *((primary, _bulk(RECORD_A), b':1\r\n') * 2),
                b'*1\r\n' + _bulk(b'rotation-required'),
                primary,
                b'*2\r\n' + _bulk(b'rotated') + _bulk(RECORD_B),
            )
        )
        settings = valkey_test_settings(limits=_FAKE_TRANSPORT_LIMITS, mode='standalone')
        backend = await module.AsyncValkeyBackend.from_settings(
            settings,
            valkey_passwords(settings),
            RecordingMetrics(),
        )
        try:
            with pytest.raises(StaleIncarnation):
                await backend.resolve((DependencyTag.site('production'),))
            scripts = [
                command
                for wire in transport.wires
                for command in wire.sent
                if b'EVALSHA' in command
            ]
            assert len(scripts) == 2
            assert CONTROL_ROTATE_SCRIPT_SHA.encode() in scripts[1]
        finally:
            await backend.close()

    asyncio.run(scenario())


@pytest.mark.parametrize('operation', ['resolve', 'change', 'control_create', 'commit'])
@pytest.mark.parametrize('refusal', ['readonly', 'noreplicas'])
def test_script_write_refusal_opens_role_breaker_without_replaying(
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
    refusal: str,
) -> None:
    async def scenario() -> None:
        transport = _Transport(monkeypatch)
        metrics = RecordingMetrics()
        primary = _bulk(PRIMARY_INFO)
        error = (
            b"-READONLY You can't write against a read only replica.\r\n"
            if refusal == 'readonly'
            else b'-NOREPLICAS Not enough good replicas to write.\r\n'
        )
        transport.replies.extend((b'+OK\r\n',) * 3)
        if operation in ('resolve', 'change'):
            transport.replies.extend((primary, _bulk(RECORD_A), b':1\r\n') * 2)
        elif operation == 'control_create':
            transport.replies.extend((primary, b'$-1\r\n'))
        else:
            transport.replies.append(b'*1\r\n' + _bulk(b'reserved'))
        transport.replies.append(error)
        settings = valkey_test_settings(
            # Deadline behavior is covered separately; this test exercises write refusal.
            limits=LimitSettings(circuit_failure_threshold=1, operation_timeout_seconds=5.0),
        )
        backend = await module.AsyncValkeyBackend.from_settings(
            settings,
            valkey_passwords(settings),
            metrics,
        )
        role = 'result' if operation == 'commit' else 'metadata'
        owned = cast(
            'module._ConfiguredRole',
            backend._result_client
            if role == 'result'
            else cast('module.AsyncValkeyMetadataStore', backend._metadata)._client,
        )
        epoch = owned.epoch()[0]

        async def attempt() -> object:
            if operation in ('resolve', 'control_create'):
                return await backend.resolve((DependencyTag.site('production'),))
            if operation == 'change':
                return await backend.change((GenerationChange(DependencyTag.site('production')),))
            lease = await backend.reserve('qualified-result', 2.0)
            assert lease is not None
            return await backend.commit(lease, encode_value({'id': 7}), 3.5)

        try:
            with pytest.raises(BackendUnavailable):
                await attempt()

            assert (owned.epoch()[0] > epoch) is (refusal == 'readonly')
            assert (
                'backend_failure',
                {'backend_operation': role, 'outcome': 'open', 'failure_class': 'unknown'},
            ) in (metrics.increments)
            assert not any(
                name == 'cache_metadata_write_rejection' for name, _ in metrics.increments
            )
            if role == 'result':
                assert (
                    'cache_commit',
                    {'layer': 'l2', 'outcome': 'rejected'},
                ) not in metrics.increments
                assert not any(name == 'cache_lease_rejection' for name, _ in metrics.increments)
            expected_scripts = 2 if role == 'result' else 1
            assert (
                sum(b'EVALSHA' in command for wire in transport.wires for command in wire.sent)
                == expected_scripts
            )
            if operation == 'control_create':
                assert CONTROL_CREATE_SCRIPT_SHA.encode() in next(
                    command
                    for wire in transport.wires
                    for command in wire.sent
                    if b'EVALSHA' in command
                )
        finally:
            await backend.close()

    asyncio.run(scenario())


@pytest.mark.parametrize('operation', ['resolve', 'commit'])
def test_script_capacity_keeps_existing_caller_outcome(
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    async def scenario() -> None:
        transport = _Transport(monkeypatch)
        metrics = RecordingMetrics()
        primary = _bulk(PRIMARY_INFO)
        transport.replies.extend((b'+OK\r\n',) * 3)
        if operation == 'resolve':
            transport.replies.extend((primary, _bulk(RECORD_A), b':1\r\n') * 2)
        else:
            transport.replies.append(b'*1\r\n' + _bulk(b'reserved'))
        transport.replies.append(b'*1\r\n' + _bulk(b'capacity'))
        settings = valkey_test_settings(
            limits=LimitSettings(circuit_failure_threshold=1, operation_timeout_seconds=5.0),
        )
        backend = await module.AsyncValkeyBackend.from_settings(
            settings,
            valkey_passwords(settings),
            metrics,
        )
        role = 'metadata' if operation == 'resolve' else 'result'
        owned = cast(
            'module._ConfiguredRole',
            cast('module.AsyncValkeyMetadataStore', backend._metadata)._client
            if role == 'metadata'
            else backend._result_client,
        )
        epoch = owned.epoch()[0]
        try:
            if operation == 'resolve':
                with pytest.raises(BackendCapacity):
                    await backend.resolve((DependencyTag.site('production'),))
                assert ('cache_metadata_write_rejection', {'reason': 'noeviction'}) in (
                    metrics.increments
                )
            else:
                lease = await backend.reserve('qualified-result', 2.0)
                assert lease is not None
                assert await backend.commit(lease, encode_value({'id': 7}), 3.5) is False
            assert not any(
                name == 'backend_failure' and labels['outcome'] == 'open'
                for name, labels in metrics.increments
            )
            assert owned.epoch()[0] == epoch
        finally:
            await backend.close()

    asyncio.run(scenario())


@pytest.mark.parametrize('reader_role', ['result', 'metadata'])
def test_configured_managed_role_rejects_replica_and_discards_before_mutation(
    monkeypatch: pytest.MonkeyPatch,
    reader_role: str,
) -> None:
    async def scenario() -> None:
        transport = _Transport(monkeypatch)
        replica = b'role:slave\r\nmaster_replid:primary\r\n'
        transport.replies.extend((b'+OK\r\n', b'+OK\r\n', b'+OK\r\n', _bulk(replica)))
        base = CacheSettings(limits=_FAKE_TRANSPORT_LIMITS)
        role = replace(base.l2.valkey.result, mode='managed', tls=False)
        settings = replace(
            base,
            l2=L2Settings(
                enabled=True,
                valkey=replace(
                    base.l2.valkey,
                    result=role,
                    metadata=replace(base.l2.valkey.metadata, mode='managed', tls=False),
                ),
            ),
        )
        backend = await module.AsyncValkeyBackend.from_settings(
            settings,
            {role.password_env: 'result', settings.l2.valkey.metadata.password_env: 'metadata'},
            NullMetrics(),
        )
        owned = cast(
            'module._ConfiguredRole',
            backend._result_client
            if reader_role == 'result'
            else cast('module.AsyncValkeyMetadataStore', backend._metadata)._client,
        )
        try:
            operation = (
                backend.reserve('key', 5)
                if reader_role == 'result'
                else backend.resolve((DependencyTag.site('production'),))
            )
            with pytest.raises(BackendUnavailable) as caught:
                await operation
            assert caught.value.__cause__ is None
            assert caught.value.__context__ is None
            assert b'INFO' in transport.wires[0].sent[3]
            assert owned.epoch()[0] == 1
            assert transport.wires[0].closed
            assert not any(
                b'EVALSHA' in command for wire in transport.wires for command in wire.sent
            )
        finally:
            await backend.close()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    'path',
    ['standalone', 'managed', 'sentinel-discovery', 'sentinel-primary'],
)
@pytest.mark.parametrize('tls', [False, True])
def test_role_constructor_checks_hostnames_only_for_tls_connections(
    path: _AsyncClientPath,
    tls: bool,
) -> None:
    async def scenario() -> None:
        configured, connection = await _configured_connection(path, tls=tls)
        try:
            assert isinstance(connection, SSLConnection) is tls
            if tls:
                tls_connection = cast(
                    '_TLSContextConnection',
                    cast('object', connection),
                )
                arguments = tls_connection._connection_arguments()
                context = arguments['ssl']
                assert isinstance(context, ssl.SSLContext)
                assert context.check_hostname is True
            else:
                assert isinstance(connection, Connection)
        finally:
            await connection.disconnect(nowait=True)
            await configured.close()

    asyncio.run(scenario())


@pytest.mark.parametrize(('hostname', 'accepted'), [('cache.test', True), ('other.test', False)])
def test_configured_tls_context_verifies_server_hostname(
    monkeypatch: pytest.MonkeyPatch,
    hostname: str,
    accepted: bool,
) -> None:
    async def scenario() -> None:
        monkeypatch.setenv('SSL_CERT_FILE', str(TLS_CA))
        configured, connection = await _configured_connection(
            'standalone', host=hostname, tls=True
        )
        try:
            tls_connection = cast(
                '_TLSContextConnection',
                cast('object', connection),
            )
            arguments = tls_connection._connection_arguments()
            context = cast('ssl.SSLContext', arguments['ssl'])

            def wrap(sock: socket.socket) -> ssl.SSLSocket:
                return context.wrap_socket(sock, server_hostname=hostname)

            if accepted:
                perform_tls_handshake(wrap)
            else:
                with pytest.raises(ssl.SSLCertVerificationError) as captured:
                    perform_tls_handshake(wrap)
                assert 'hostname mismatch' in captured.value.verify_message.lower()
        finally:
            await connection.disconnect(nowait=True)
            await configured.close()

    asyncio.run(scenario())


@pytest.mark.parametrize('phase', ['before-entry', 'waiting-idle'])
def test_backend_independently_cancelled_close_settles_every_owned_role(
    phase: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def scenario() -> None:
        result, metadata = BackendClient([]), BackendClient([])
        backend = module.AsyncValkeyBackend(
            result,
            cast('module.AsyncValkeyMetadataStore', metadata),
            CacheSettings(l2=L2Settings(enabled=True)),
            NullMetrics(),
        )
        entered, proceed = asyncio.Event(), asyncio.Event()

        async def admitted() -> None:
            with backend._lifecycle.operation():
                entered.set()
                await proceed.wait()

        active = asyncio.create_task(admitted())
        await entered.wait()
        if phase == 'before-entry':
            original = module.create_task

            def cancelled_before_entry(
                coroutine: Coroutine[Any, Any, module.CleanupOutcome[None]],
            ) -> asyncio.Task[module.CleanupOutcome[None]]:
                task = original(coroutine)
                task.cancel('independent before body')
                return task

            monkeypatch.setattr(module, 'create_task', cancelled_before_entry)
        closer = asyncio.create_task(backend.close())
        barrier = asyncio.Event()
        asyncio.get_running_loop().call_soon(barrier.set)
        await barrier.wait()
        close_task = backend._lifecycle._task
        assert close_task is not None
        if phase == 'waiting-idle':
            idle_entered = asyncio.Event()
            asyncio.get_running_loop().call_soon(idle_entered.set)
            await idle_entered.wait()
        if phase == 'waiting-idle':
            close_task.cancel('independent')
        other = asyncio.create_task(backend.close())
        assert result.closed == metadata.closed == 0
        proceed.set()
        await active
        for caller in (closer, other):
            with pytest.raises(BackendUnavailable):
                await caller
        assert result.closed == metadata.closed == 1
        await backend.close()

    asyncio.run(scenario())


@pytest.mark.parametrize('cancel_mode', ['ordinary', 'stale', 'caller'])
def test_inline_backend_close_distinguishes_child_and_caller_cancellation(
    cancel_mode: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def scenario() -> None:
        caller = asyncio.current_task()
        assert caller is not None
        received: list[asyncio.CancelledError] = []
        original_shield = support.shield

        async def observe_cancellation(future: asyncio.Future[None]) -> None:
            try:
                await original_shield(future)
            except asyncio.CancelledError as error:
                received.append(error)
                raise

        if cancel_mode == 'stale':
            caller.cancel('handled earlier')
            with pytest.raises(asyncio.CancelledError, match='handled earlier'):
                await asyncio.sleep(0)

        class _IndependentClose(BackendClient):
            async def close(self) -> None:
                await super().close()
                child = asyncio.create_task(asyncio.sleep(0))
                child.cancel('independent resource')
                await child

        class _LaterClose(BackendClient):
            async def close(self) -> None:
                await super().close()
                if cancel_mode == 'caller':
                    caller.cancel('actual caller')
                    await asyncio.sleep(0)

        def cancel_shared(coro: Coroutine[Any, Any, _T]) -> asyncio.Task[_T]:
            task = asyncio.create_task(coro)
            task.cancel('shared close')
            return task

        result, metadata = _IndependentClose([]), _LaterClose([])
        backend = module.AsyncValkeyBackend(
            result,
            cast('module.AsyncValkeyMetadataStore', metadata),
            CacheSettings(),
            NullMetrics(),
        )
        monkeypatch.setattr(module, 'create_task', cancel_shared)
        monkeypatch.setattr(support, 'shield', observe_cancellation)
        if cancel_mode == 'caller':
            with pytest.raises(asyncio.CancelledError, match='actual caller') as raised:
                await backend.close()
            assert raised.value is received[0]
        else:
            with pytest.raises(BackendUnavailable, match='interrupted'):
                await backend.close()
        assert caller.cancelling() == int(cancel_mode != 'ordinary')
        assert result.closed == metadata.closed == 1
        assert backend._lifecycle._admission.closed

    asyncio.run(scenario())


@pytest.mark.parametrize('cancel_mode', ['ordinary', 'stale', 'pending', 'during'])
def test_construction_cleanup_preserves_cancellation_provenance(cancel_mode: str) -> None:
    async def scenario() -> None:
        caller = asyncio.current_task()
        assert caller is not None
        attempts: list[str] = []
        original = ValueError('original construction')
        metrics = RecordingMetrics()
        if cancel_mode in {'stale', 'pending'}:
            caller.cancel(cancel_mode)
            if cancel_mode == 'stale':
                with pytest.raises(asyncio.CancelledError, match='stale'):
                    await asyncio.sleep(0)

        async def independent_close() -> None:
            attempts.append('independent')
            child = asyncio.create_task(asyncio.sleep(0))
            child.cancel('independent resource')
            await child

        async def later_close() -> None:
            attempts.append('later')
            if cancel_mode == 'during':
                caller.cancel('during')
                await asyncio.sleep(0)

        async def failed_construction() -> None:
            await module._cleanup_construction((independent_close, later_close), original, metrics)
            raise original

        if cancel_mode in {'pending', 'during'}:
            with pytest.raises(asyncio.CancelledError, match=cancel_mode):
                await failed_construction()
        else:
            with pytest.raises(ValueError, match='original construction') as raised:
                await failed_construction()
            assert raised.value is original
        assert attempts == ['independent', 'later']
        assert caller.cancelling() == int(cancel_mode != 'ordinary')
        assert (
            'backend_failure',
            {'backend_operation': 'construction_close', 'outcome': 'failure'},
        ) in metrics.increments

    asyncio.run(scenario())


@pytest.mark.parametrize(('pre_entry', 'blocks'), [(False, True), (True, True), (True, False)])
def test_close_resources_resubmit_only_unstarted_action_with_original_expired_deadline(
    pre_entry: bool,
    blocks: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        first, second = BackendClient([]), BackendClient([])
        if blocks:
            first.close_proceed.clear()
        deadline = asyncio.get_running_loop().time()
        submissions: list[float | None] = []
        original_join = support.joined_cleanup

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

        monkeypatch.setattr(support, 'joined_cleanup', submit)
        record = support.DeadlineRecord(deadline)
        with support.publish_deadline(record):
            outcome = await module._close_resources((first.close, second.close), deadline=deadline)
        assert isinstance(outcome.error, BackendUnavailable)
        assert str(outcome.error) == 'cleanup task cancelled'
        assert record.expired is blocks
        assert outcome.cancellation is None
        assert submissions == [deadline] * (3 if pre_entry else 2)
        assert first.closed == second.closed == 1

    asyncio.run(scenario())


@pytest.mark.parametrize('elapsed', [0.5, 2.0])
def test_backend_lookup_debits_decode_and_guard_telemetry_once(
    elapsed: float, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def scenario() -> None:
        clock = FakeClock()
        monkeypatch.setattr(module, 'monotonic', clock)
        decoded: list[bytes] = []
        original = common_module.decode_value

        def decode(payload: bytes) -> object:
            decoded.append(payload)
            clock.advance(elapsed)
            return original(payload)

        class _Metrics(RecordingMetrics):
            def observe(self, name: str, value: float, labels: Mapping[str, str]) -> None:
                super().observe(name, value, labels)
                if name == 'backend_latency_seconds':
                    clock.advance(0.25)

        monkeypatch.setattr(common_module, 'decode_value', decode)
        payload = encode_value({'id': 7})
        result, metadata = (
            BackendClient([('evalsha', [b'hit', payload, 2000])]),
            BackendClient([]),
        )
        backend = module.AsyncValkeyBackend(
            result, cast('module.AsyncValkeyMetadataStore', metadata), CacheSettings(), _Metrics()
        )
        hit = await backend.lookup('key')
        assert decoded == [payload]
        if elapsed == 2:
            assert hit is None
        else:
            assert hit is not None
            assert hit.remaining_seconds == 1.25
        await backend.close()

    asyncio.run(scenario())


@pytest.mark.parametrize('tls', [False, True])
def test_factory_owns_each_sentinel_discovery_pool_and_primary(tls: bool) -> None:
    async def scenario() -> None:
        # Pool ownership is the contract; host scheduling latency is not.
        base = CacheSettings(limits=LimitSettings(operation_timeout_seconds=5.0))
        role = replace(
            base.l2.valkey.metadata,
            mode='sentinel',
            database=7,
            tls=tls,
            sentinels=('one:26379', 'two:26379', 'three:26379'),
            service_name='primary',
        )
        settings = replace(
            base,
            l2=L2Settings(
                enabled=True,
                valkey=ValkeySettings(
                    metadata=role,
                    result=replace(
                        role,
                        host='result',
                        service_name='result',
                        password_env=base.l2.valkey.result.password_env,
                    ),
                ),
            ),
        )
        backend = await module.AsyncValkeyBackend.from_settings(
            settings,
            {role.password_env: 'metadata', base.l2.valkey.result.password_env: 'result'},
            NullMetrics(),
        )
        roles = (
            cast(
                'module._ConfiguredRole',
                cast('module.AsyncValkeyMetadataStore', backend._metadata)._client,
            ),
            cast('module._ConfiguredRole', backend._result_client),
        )
        pools: list[module._RawClient] = []
        for owned in roles:
            primary = owned._active()
            assert isinstance(primary.connection_pool, module._SentinelConnectionPool)
            assert primary.connection_pool.connection_class is (
                module._SentinelSSLConnection if tls else module._SentinelConnection
            )
            assert primary.connection_pool.make_connection().db == 7
            assert all(raw.connection_pool.make_connection().db == 0 for raw in owned._discovery)
            assert len(owned._discovery) == 3
            pools.extend(owned._discovery)
            pools.append(primary)
        assert len({id(raw.connection_pool) for raw in pools}) == 8
        await backend.close()
        assert all(owned._client is None and not owned._discovery for owned in roles)

    asyncio.run(scenario())


class _SentinelAPI(Protocol):
    sentinels: list[module._RawClient]

    async def discover_master(self, service_name: str) -> tuple[str, int]: ...

    def master_for(self, service_name: str, **kwargs: object) -> module._RawClient: ...


def _role(
    *,
    retry: Retry | None = None,
    retry_on_error: list[type[Exception]] | None = None,
    lib_name: str | None = None,
) -> module._RoleClient:
    connections = module._ConnectionGeneration()

    def factory() -> module._RawClient:
        pool = module._ConnectionPool(
            valkey_connect_func=connections.initialize,
            lib_name=lib_name,
            lib_version=None,
            retry=retry,
            retry_on_error=retry_on_error,
        )
        return cast('module._RawClient', cast('object', Valkey.from_pool(pool)))

    return module._RoleClient(factory, connections)


def test_async_helpers_never_invoke_sync_tracker_or_clients(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*_args: object, **_kwargs: object) -> object:
        pytest.fail('async helper invoked synchronous Valkey I/O or tracker')

    monkeypatch.setattr(sync_module._ConnectionGeneration, 'connected', forbidden)
    monkeypatch.setattr(valkey.Valkey, 'execute_command', forbidden)
    monkeypatch.setattr(valkey.Sentinel, 'discover_master', forbidden)

    async def scenario() -> None:
        transport = _Transport(monkeypatch)
        transport.replies.append(b'$2\r\nok\r\n')
        role = _role()
        assert await role.get('key') == b'ok'
        assert role.epoch() == (0, 1, None)
        await role.close()

    asyncio.run(scenario())


@pytest.mark.parametrize('elapsed', [0.025, 0.100])
def test_backend_lookup_debits_awaited_endpoint_settlement_from_server_ttl(
    elapsed: float, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def scenario() -> None:
        clock = FakeClock()
        monkeypatch.setattr(module, 'monotonic', clock)
        transport = _Transport(monkeypatch)
        payload = encode_value({'id': 7})
        transport.replies.append(b'*3\r\n' + _bulk(b'hit') + _bulk(payload) + b':50\r\n')
        role = _role()
        probes = 0

        async def endpoint(_client: module._RawClient) -> str:
            nonlocal probes
            probes += 1
            if probes == 2:
                clock.advance(elapsed)
                await asyncio.sleep(0)
            return 'primary'

        role._endpoint_probe = endpoint
        backend = module.AsyncValkeyBackend(
            role,
            cast('module.AsyncValkeyMetadataStore', BackendClient([])),
            CacheSettings(limits=_FAKE_TRANSPORT_LIMITS),
            NullMetrics(),
        )
        hit = await backend.lookup('key')
        if elapsed == 0.100:
            assert hit is None
        else:
            assert hit is not None
            assert hit.remaining_seconds == pytest.approx(0.025)
            assert hit.decoded is not None
            assert hit.decoded.value == {'id': 7}
        assert probes == 2
        assert len(transport.wires[0].sent) == 1
        await backend.close()

    asyncio.run(scenario())


def test_cancelled_acquisition_disconnects_before_pool_reuse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        transport = _Transport(monkeypatch)
        transport.connect_proceed.clear()
        role = _role()

        async def borrow() -> None:
            async with role.pin():
                pytest.fail('cancelled acquisition yielded a pin')

        acquisition = asyncio.create_task(borrow())
        await transport.connect_entered.wait()
        acquisition.cancel()
        with pytest.raises(asyncio.CancelledError):
            await acquisition
        assert transport.wires[0].closed
        transport.connect_proceed.set()
        async with role.pin() as pinned:
            assert pinned.connection_generation() == 1
            assert len(transport.wires) == 2
        await role.close()

    asyncio.run(scenario())


def test_cancelled_settlement_propagates_after_pin_release(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        transport = _Transport(monkeypatch)
        role = _role()
        entered = asyncio.Event()
        proceed = asyncio.Event()
        probes = 0

        async def probe(_client: module._RawClient) -> object:
            nonlocal probes
            probes += 1
            if probes == 2:
                entered.set()
                await proceed.wait()
            return 'primary'

        role._endpoint_probe = probe
        transport.replies.append(b'$2\r\nok\r\n')
        task = asyncio.create_task(role.get('key'))
        await entered.wait()
        task.cancel('settlement')
        with pytest.raises(asyncio.CancelledError, match='settlement'):
            await task
        client = role._client
        assert client is not None
        assert not client.connection_pool._in_use_connections
        assert not transport.wires[0].closed
        await role.close()

    asyncio.run(scenario())


def test_close_detaches_before_await_and_joins_repeated_cancellation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        transport = _Transport(monkeypatch)
        role = _role()
        async with role.pin():
            pass
        entered = asyncio.Event()
        proceed = asyncio.Event()
        original_disconnect = module._ConnectionPool.disconnect

        async def delayed_disconnect(
            pool: module._ConnectionPool, inuse_connections: bool = True
        ) -> None:
            entered.set()
            await proceed.wait()
            await original_disconnect(pool, inuse_connections)

        monkeypatch.setattr(module._ConnectionPool, 'disconnect', delayed_disconnect)
        close = asyncio.create_task(role.close())
        await entered.wait()
        with pytest.raises(RuntimeError, match='closed'):
            async with role.pin():
                pytest.fail('closed role admitted a new pin')
        close.cancel('first')
        asyncio.get_running_loop().call_soon(close.cancel, 'second')
        proceed.set()
        with pytest.raises(asyncio.CancelledError, match='first'):
            await close
        assert transport.wires[0].closed
        assert close.cancelling() == 2

    asyncio.run(scenario())


def test_close_settles_when_cleanup_task_is_cancelled_before_entry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        transport = _Transport(monkeypatch)
        role = _role()
        async with role.pin():
            pass
        client = role._client
        assert client is not None
        original_create_task = asyncio.create_task

        def cancel_cleanup(action: Coroutine[Any, Any, _T]) -> asyncio.Task[_T]:
            task = original_create_task(action)
            task.cancel()
            return task

        monkeypatch.setattr(support, 'create_task', cancel_cleanup)
        with pytest.raises(BackendUnavailable):
            await role.close()
        assert transport.wires[0].closed
        assert client.connection_pool._available_connections
        assert role._client is None

    asyncio.run(scenario())


@pytest.mark.parametrize('failure', ['timeout', 'socket'])
def test_command_write_failure_disconnects_without_retry(
    monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    async def scenario() -> None:
        transport = _Transport(monkeypatch)
        pool = module._ConnectionPool(socket_timeout=0.001, lib_name=None, lib_version=None)
        connection = pool.make_connection()
        await connection.connect()
        wire = transport.wires[0]
        if failure == 'timeout':
            wire.drain_proceed.clear()
        else:

            async def fail_write() -> None:
                raise OSError('socket write failed')

            monkeypatch.setattr(wire, 'drain', fail_write)
        expected = ValkeyTimeoutError if failure == 'timeout' else ValkeyConnectionError
        with pytest.raises(expected):
            await connection.send_command('GET', 'key')
        assert wire.closed
        assert len(wire.sent) == 1
        await pool.disconnect()

    asyncio.run(scenario())


@pytest.mark.parametrize('phase', ['send', 'read'])
def test_command_cancellation_disconnects_only_its_connection(
    monkeypatch: pytest.MonkeyPatch, phase: str
) -> None:
    async def scenario() -> None:
        transport = _Transport(monkeypatch)
        role = _role()
        async with role.pin() as first, role.pin() as second:
            first_wire, second_wire = transport.wires
            initial = first.connection_generation()
            epoch = role.epoch()[0]
            if phase == 'send':
                first_wire.drain_proceed.clear()
            command = asyncio.create_task(first.get('key'))
            await first_wire.drain_entered.wait()
            command.cancel('command interrupted')
            with pytest.raises(asyncio.CancelledError, match='command interrupted'):
                await command
            assert first_wire.closed
            assert not second_wire.closed
            assert role.epoch()[0] == epoch
            second_wire.reader.feed_data(b'$2\r\nok\r\n')
            assert await second.get('key') == b'ok'
            reconnect = asyncio.create_task(first.get('key'))
            transport.connect_entered.clear()
            await transport.connect_entered.wait()
            transport.wires[-1].reader.feed_data(b'$3\r\nnew\r\n')
            assert await reconnect == b'new'
            assert first.connection_generation() != initial
        await role.close()

    asyncio.run(scenario())


@pytest.mark.parametrize('phase', ['setup', 'readiness'])
def test_cancelled_acquisition_during_setup_or_readiness_reconnects(
    monkeypatch: pytest.MonkeyPatch, phase: str
) -> None:
    async def scenario() -> None:
        transport = _Transport(monkeypatch)
        role = _role(lib_name='transport-test' if phase == 'setup' else None)
        entered = asyncio.Event()
        proceed = asyncio.Event()

        async def readiness(_connection: AbstractConnection) -> bool:
            entered.set()
            await proceed.wait()
            return False

        if phase == 'readiness':
            monkeypatch.setattr(AbstractConnection, 'can_read_destructive', readiness)

        async def borrow() -> None:
            async with role.pin():
                pytest.fail('cancelled acquisition yielded a pin')

        task = asyncio.create_task(borrow())
        await transport.connect_entered.wait()
        if phase == 'setup':
            await transport.wires[0].drain_entered.wait()
        else:
            await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert transport.wires[0].closed
        proceed.set()
        if phase == 'setup':
            transport.replies.append(b'+OK\r\n')
        async with role.pin() as pin:
            assert pin.connection_generation() == (1 if phase == 'setup' else 2)
        assert len(transport.wires) == 2
        await role.close()

    asyncio.run(scenario())


@pytest.mark.parametrize('phase', ['physical', 'setup'])
def test_cancelled_pinned_reconnect_closes_before_successful_reuse(
    monkeypatch: pytest.MonkeyPatch, phase: str
) -> None:
    async def scenario() -> None:
        transport = _Transport(monkeypatch)
        role = _role(lib_name='transport-test' if phase == 'setup' else None)
        if phase == 'setup':
            transport.replies.extend([b'+OK\r\n', b'+OK\r\n'])
        async with role.pin() as first, role.pin() as other:
            connection = first._client.connection
            assert connection is not None
            initial = first.connection_generation()
            pool_epoch = role.epoch()[0]
            await connection.disconnect(nowait=True)
            transport.connect_entered.clear()
            setup_entered = asyncio.Event()
            setup_proceed = asyncio.Event()
            if phase == 'physical':
                transport.connect_proceed.clear()
            else:
                original_setup = connection.on_connect

                async def setup() -> None:
                    await original_setup()
                    setup_entered.set()
                    await setup_proceed.wait()

                monkeypatch.setattr(connection, 'on_connect', setup)
                transport.replies.append(b'+OK\r\n')
            command = asyncio.create_task(first.get('key'))
            await transport.connect_entered.wait()
            reconnect_wire = transport.wires[-1]
            if phase == 'setup':
                await setup_entered.wait()
            command.cancel('incomplete reconnect')
            with pytest.raises(asyncio.CancelledError, match='incomplete reconnect'):
                await command
            assert reconnect_wire.closed
            assert not connection.is_connected
            assert first.connection_generation() == initial
            assert role.epoch()[0] == pool_epoch
            assert not transport.wires[1].closed
            transport.replies.append(b'$2\r\nok\r\n')
            assert await other.get('key') == b'ok'
            transport.connect_proceed.set()
            setup_proceed.set()
            if phase == 'setup':
                transport.replies.append(b'+OK\r\n')
            transport.replies.append(b'$3\r\nnew\r\n')
            assert await first.get('key') == b'new'
            assert first.connection_generation() == 3
            assert len(transport.wires) == 4
        await role.close()

    asyncio.run(scenario())


def test_commit_cancelled_during_noscript_load_settles_pin_without_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        transport = _Transport(monkeypatch)
        transport.replies.append(b'-NOSCRIPT absent\r\n')
        role = _role()
        load_entered = asyncio.Event()
        original_load = module._PinnedRoleClient.script_load

        async def load(pinned: module._PinnedRoleClient, script: bytes | str) -> object:
            load_entered.set()
            return await original_load(pinned, script)

        monkeypatch.setattr(module._PinnedRoleClient, 'script_load', load)
        backend = module.AsyncValkeyBackend(
            role,
            cast('module.AsyncValkeyMetadataStore', BackendClient([])),
            CacheSettings(l2=L2Settings(enabled=True)),
            NullMetrics(),
        )
        task = asyncio.create_task(backend.commit(FillLease('key', 'token'), b'payload', 1.0))
        await load_entered.wait()
        task.cancel('script loading')
        with pytest.raises(asyncio.CancelledError, match='script loading'):
            await task
        assert len(transport.wires) == 1
        assert len(transport.wires[0].sent) == 2
        assert transport.wires[0].closed
        await backend.close()

    asyncio.run(scenario())


def test_discard_allows_new_borrow_before_old_cleanup_finishes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        transport = _Transport(monkeypatch)
        role = _role()
        entered = asyncio.Event()
        proceed = asyncio.Event()
        original_disconnect = module._ConnectionPool.disconnect

        async with role.pin() as old_pin:
            old_pool = old_pin._client.connection_pool

            async def delayed_disconnect(inuse_connections: bool = True) -> None:
                entered.set()
                await proceed.wait()
                await original_disconnect(old_pool, inuse_connections)

            monkeypatch.setattr(old_pool, 'disconnect', delayed_disconnect)
            refresh = asyncio.create_task(role.refresh())
            await entered.wait()
            async with role.pin() as new_pin:
                assert new_pin._client.connection_pool is not old_pool
                with pytest.raises(BackendUnavailable):
                    await old_pin.get('key')
                refresh.cancel('discard')
                proceed.set()
                with pytest.raises(asyncio.CancelledError, match='discard'):
                    await refresh
                assert transport.wires[0].closed
                assert not transport.wires[1].closed
                transport.replies.append(b'$2\r\nok\r\n')
                assert await new_pin.get('key') == b'ok'
        await role.close()

    asyncio.run(scenario())


def test_discard_detaches_overlapping_pins_and_close_attempts_every_connection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        transport = _Transport(monkeypatch)
        role = _role()
        async with role.pin() as first, role.pin() as second:
            transport.wires[0].close_error = ValueError('close failed')
            with pytest.raises(BackendUnavailable):
                await role.refresh()
            assert all(wire.closed for wire in transport.wires)
            with pytest.raises(BackendUnavailable):
                await first.get('key')
            with pytest.raises(BackendUnavailable):
                await second.get('key')
            async with role.pin():
                assert len(transport.wires) == 3
        await role.close()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    'failure',
    [
        ValkeyConnectionError('discovery failed'),
        ValueError('private reply'),
        BackendCapacity('capacity'),
        BackendProtocolError('protocol'),
        BackendUnavailable('unavailable'),
        StaleIncarnation('stale'),
    ],
)
def test_discovery_preserves_backend_errors_and_discards_client_failures(
    monkeypatch: pytest.MonkeyPatch, failure: Exception
) -> None:
    async def scenario() -> None:
        transport = _Transport(monkeypatch)
        role = _role()

        async def probe(_client: module._RawClient) -> object:
            raise failure

        async with role.pin():
            client = role._client
            role._endpoint_probe = probe
            client_failure = isinstance(failure, ValkeyConnectionError | ValueError)
            expected = BackendUnavailable if client_failure else type(failure)
            with pytest.raises(expected) as caught:
                await role.refresh_endpoint()
            if client_failure:
                assert client is not role._client
                assert transport.wires[0].closed
                assert 'private reply' not in str(caught.value)
                assert caught.value.__suppress_context__
            else:
                assert caught.value is failure
                assert client is role._client
                assert not transport.wires[0].closed
        await role.close()

    asyncio.run(scenario())


def test_endpoint_discovery_cancellation_preserves_other_pins(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        transport = _Transport(monkeypatch)
        role = _role()
        entered = asyncio.Event()
        proceed = asyncio.Event()

        async def probe(_client: module._RawClient) -> object:
            entered.set()
            await proceed.wait()
            return 'primary'

        role._endpoint_probe = probe
        async with role.pin() as pinned:
            task = asyncio.create_task(role.refresh_endpoint())
            await entered.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert not transport.wires[0].closed
            transport.replies.append(b'$2\r\nok\r\n')
            assert await pinned.get('key') == b'ok'
        await role.close()

    asyncio.run(scenario())


def test_epoch_sampling_is_pure_and_tracker_is_weak_and_loop_owned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    role = _role()

    async def scenario() -> None:
        transport = _Transport(monkeypatch)
        async with role.pin() as pinned:
            assert role.epoch() == (0, 1, None)
            assert pinned.connection_generation() == 1
            for _ in range(5):
                assert role.epoch() == (0, 1, None)
            assert not transport.wires[0].sent
        await role.close()
        del pinned
        assert len(role._connections._by_connection) == 0

    asyncio.run(scenario())

    async def wrong_loop() -> None:
        with pytest.raises(RuntimeError, match='different event loop'):
            role.epoch()
        with pytest.raises(RuntimeError, match='different event loop'):
            await role.close()

    asyncio.run(wrong_loop())


def test_metadata_acquisition_translates_connection_refusal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def refused(_connection: Connection) -> None:
        raise OSError('refused')

    monkeypatch.setattr(Connection, '_connect', refused)

    async def scenario() -> None:
        role = _role()
        store = module.AsyncValkeyMetadataStore(
            role, CacheSettings(limits=_FAKE_TRANSPORT_LIMITS), NullMetrics()
        )
        with pytest.raises(BackendUnavailable) as caught:
            await store.resolve((DependencyTag.site('production'),))
        assert caught.value.__cause__ is None
        assert caught.value.__context__ is None
        assert role.epoch()[0] == 1
        await store.close()

    asyncio.run(scenario())


def test_metadata_acquisition_translates_unexpected_client_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def failed_initialize(_client: Valkey) -> None:
        raise KeyError('private acquisition detail')

    monkeypatch.setattr(Valkey, 'initialize', failed_initialize)

    async def scenario() -> None:
        role = _role()
        store = module.AsyncValkeyMetadataStore(
            role, CacheSettings(limits=_FAKE_TRANSPORT_LIMITS), NullMetrics()
        )
        try:
            with pytest.raises(BackendUnavailable) as caught:
                await store.resolve((DependencyTag.site('production'),))
            assert str(caught.value) == 'Valkey operation failed (unknown)'
            assert caught.value.__cause__ is None
            assert caught.value.__context__ is None
        finally:
            await store.close()

    asyncio.run(scenario())


def test_new_async_metadata_socket_receives_full_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        transport = _Transport(monkeypatch)
        primary = PRIMARY_INFO
        replies = (_bulk(primary), _bulk(RECORD_A), b':1\r\n') * 2
        transport.replies.extend(replies)
        role = _role()
        control = AsyncValkeyControl(role, key_prefix='sgqc', validation_seconds=30)

        await control._validate_incarnation()
        transport.replies.extend(replies)
        async with role.pin():
            await control._validate_incarnation()

        assert len(transport.wires) == 2
        assert transport.wires[1].sent.count(b'*2\r\n$4\r\nINFO\r\n$11\r\nreplication\r\n') == 2
        await role.close()

    asyncio.run(scenario())


def test_pin_cannot_be_reused_after_scope_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    async def scenario() -> None:
        transport = _Transport(monkeypatch)
        role = _role()
        async with role.pin() as pinned:
            pass
        transport.replies.append(b'$2\r\nok\r\n')
        with pytest.raises(BackendUnavailable, match='pin'):
            await pinned.get('key')
        await role.close()

    asyncio.run(scenario())


def test_pin_release_failure_exposes_no_active_command_detail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    command_canary = 'credential-canary-command'
    release_canary = 'credential-canary-release'
    command_error = ResponseError(command_canary)
    release_error = ValkeyTimeoutError(release_canary)
    for error, canary in ((command_error, command_canary), (release_error, release_canary)):
        error.add_note(canary)
        error.__dict__['client_detail'] = canary

    async def scenario() -> None:
        _Transport(monkeypatch)
        original_close = Valkey.aclose

        async def command(_client: Valkey, *_args: object, **_kwargs: object) -> object:
            raise command_error

        async def close(client: Valkey, close_connection_pool: bool | None = None) -> None:
            await original_close(client, close_connection_pool=close_connection_pool)
            if close_connection_pool is False:
                raise release_error

        monkeypatch.setattr(Valkey, 'execute_command', command)
        monkeypatch.setattr(Valkey, 'aclose', close)
        role = _role()
        settings = CacheSettings(limits=_FAKE_TRANSPORT_LIMITS)
        metadata = module.AsyncValkeyMetadataStore(role, settings, NullMetrics())
        backend = module.AsyncValkeyBackend(role, metadata, settings, NullMetrics())
        with pytest.raises(BackendUnavailable) as caught:
            await backend.lookup('key')
        await backend.close()

        failure = caught.value
        assert failure.failure_class == 'timeout'
        assert str(failure) == 'Valkey operation is unavailable (timeout)'
        assert failure.__cause__ is None
        assert failure.__context__ is None
        assert not getattr(failure, '__notes__', ())
        for canary in (command_canary, release_canary):
            assert canary not in repr(vars(failure))
            assert canary not in ''.join(format_exception(failure))
        traceback = failure.__traceback__
        while traceback is not None:
            if traceback.tb_frame.f_code.co_name == 'translate_client_error':
                assert all(
                    value is not command_error and value is not release_error
                    for value in traceback.tb_frame.f_locals.values()
                )
            traceback = traceback.tb_next

    asyncio.run(scenario())


@pytest.mark.parametrize('mode', ['standalone', 'sentinel'])
@pytest.mark.parametrize('cleanup', ['success', 'error', 'cancelled'])
def test_pin_release_never_frees_a_connection_reborrowed_during_cleanup(
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    cleanup: str,
) -> None:
    async def scenario() -> None:
        transport = _Transport(monkeypatch)
        transport.replies.extend([_bulk(PRIMARY_INFO)] * 2)
        role = _proving_role(mode)
        released = asyncio.Event()
        finish = asyncio.Event()
        borrowed: asyncio.Future[AbstractConnection | None] = (
            asyncio.get_running_loop().create_future()
        )
        original_release = module._ConnectionPool.release

        async def signalling_release(
            pool: module._ConnectionPool, connection: AbstractConnection
        ) -> None:
            await original_release(pool, connection)
            if released.is_set():
                return
            released.set()
            if cleanup == 'error':
                raise RuntimeError('failure after release')
            if cleanup == 'cancelled':
                raise asyncio.CancelledError

        monkeypatch.setattr(module._ConnectionPool, 'release', signalling_release)

        async def borrow_after_release() -> None:
            await released.wait()
            async with role.pin() as pinned:
                borrowed.set_result(pinned._client.connection)
                await finish.wait()

        pin = role.pin()
        first = await pin.__aenter__()
        reused = first._client.connection
        borrower = asyncio.create_task(borrow_after_release())
        if cleanup == 'success':
            await pin.__aexit__(None, None, None)
        else:
            with pytest.raises(BackendUnavailable):
                await pin.__aexit__(None, None, None)
        # The borrower re-acquired the connection before the first owner settled.
        assert await borrowed is reused
        async with role.pin() as third:
            assert third._client.connection is not reused
        finish.set()
        await borrower
        client = role._client
        assert client is not None
        assert not client.connection_pool._in_use_connections
        assert len(client.connection_pool._available_connections) == 2
        await role.close()

    asyncio.run(scenario())


def test_pin_release_preserves_first_cancellation_while_joining(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        transport = _Transport(monkeypatch)
        role = _role()
        release_entered = asyncio.Event()
        release_proceed = asyncio.Event()
        original_release = module._ConnectionPool.release

        async def delayed_release(
            pool: module._ConnectionPool, connection: AbstractConnection
        ) -> None:
            release_entered.set()
            await release_proceed.wait()
            await original_release(pool, connection)

        monkeypatch.setattr(module._ConnectionPool, 'release', delayed_release)

        async def operation() -> None:
            async with role.pin() as pinned:
                await pinned.get('key')

        task = asyncio.create_task(operation())
        await transport.connect_entered.wait()
        await transport.wires[0].drain_entered.wait()
        task.cancel('first')
        await release_entered.wait()
        task.cancel('second')
        release_proceed.set()
        with pytest.raises(asyncio.CancelledError, match='first'):
            await task
        assert task.cancelling() == 2
        await role.close()

    asyncio.run(scenario())


def test_pool_disables_automatic_mutation_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    async def scenario() -> None:
        transport = _Transport(monkeypatch)
        transport.replies.extend([b'-READONLY demoted\r\n', b':1\r\n'])
        role = _role(retry=Retry(NoBackoff(), 3), retry_on_error=[ReadOnlyError])
        async with role.pin() as pinned:
            with pytest.raises(ReadOnlyError):
                await pinned.evalsha('0' * 40, 1, 'key')
        assert len(transport.replies) == 1
        assert transport.wires[0].closed
        await role.close()

    asyncio.run(scenario())


def test_role_command_ignores_other_pins_reconnects(monkeypatch: pytest.MonkeyPatch) -> None:
    async def scenario() -> None:
        transport = _Transport(monkeypatch)
        role = _role()
        probes = 0

        async def probe(_client: module._RawClient) -> object:
            nonlocal probes
            probes += 1
            if probes == 2:
                async with role.pin() as pin:
                    connection = pin._client.connection
                    assert connection is not None
                    await connection.disconnect(nowait=True)
                    await connection.connect()
            return 'primary'

        role._endpoint_probe = probe
        transport.replies.append(b'$2\r\nok\r\n')
        assert await role.get('key') == b'ok'
        # Aggregate reconnects from another pin do not invalidate this span.
        assert role.epoch()[0] == 0
        await role.close()

    asyncio.run(scenario())


def test_script_load_retries_only_the_supported_noscript_case(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        transport = _Transport(monkeypatch)
        transport.replies.extend(
            [b'-NOSCRIPT absent\r\n', f'${40}\r\n{RESOLVE_SCRIPT_SHA}\r\n'.encode(), b':7\r\n']
        )
        role = _role()
        async with role.pin() as pinned:
            assert (
                await module.execute_script(
                    pinned, script=RESOLVE_SCRIPT, sha=RESOLVE_SCRIPT_SHA, keys=('key',), args=(1,)
                )
                == 7
            )
        assert len(transport.wires[0].sent) == 3
        await role.close()

    asyncio.run(scenario())


@pytest.mark.parametrize('cancel_cleanup', [False, True])
@pytest.mark.parametrize('tls', [False, True])
def test_sentinel_demotion_detaches_before_cleanup_with_overlapping_pins(
    monkeypatch: pytest.MonkeyPatch, cancel_cleanup: bool, tls: bool
) -> None:
    async def scenario() -> None:
        transport = _Transport(monkeypatch)
        tracker = module._ConnectionGeneration()

        def factory() -> module._RawClient:
            pool = module._SentinelConnectionPool(
                'service',
                _PrimaryDiscovery(),
                lib_name=None,
                lib_version=None,
                valkey_connect_func=tracker.initialize,
                ssl=tls,
            )
            return cast('module._RawClient', cast('object', Valkey.from_pool(pool)))

        role = module._RoleClient(factory, tracker)
        entered = asyncio.Event()
        proceed = asyncio.Event()
        async with role.pin() as first, role.pin() as other:
            assert isinstance(first._client.connection, SSLConnection) is tls
            pool = first._client.connection_pool
            original_disconnect = pool.disconnect

            async def cleanup(inuse_connections: bool = True) -> None:
                # This boundary follows the real Sentinel READONLY handler.
                assert role._client is None
                assert role.epoch()[0] == 1
                entered.set()
                await proceed.wait()
                await original_disconnect(inuse_connections)

            monkeypatch.setattr(pool, 'disconnect', cleanup)
            transport.replies.append(b'-READONLY demoted\r\n')
            command = asyncio.create_task(first.evalsha('0' * 40, 1, 'key'))
            # The done callback also wakes the assertion if upstream takes its
            # prohibited graceful-close path, so failure cannot leave the gate blocked.
            command.add_done_callback(lambda _task: entered.set())
            await entered.wait()
            if command.done():
                await command
            assert role._client is None
            with pytest.raises(BackendUnavailable, match='discarded pool'):
                await other.get('key')
            if cancel_cleanup:
                command.cancel('demotion cleanup')
            proceed.set()
            if cancel_cleanup:
                with pytest.raises(asyncio.CancelledError, match='demotion cleanup'):
                    await command
            else:
                with pytest.raises(ValkeyConnectionError, match='previous master'):
                    await command
            assert all(wire.closed for wire in transport.wires)
            assert len(transport.wires[0].sent) == 1
        await role.close()

    asyncio.run(scenario())


@pytest.mark.parametrize('change', ['endpoint', 'connection'])
def test_role_delivery_span_drift_discards_pool(
    monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    async def scenario() -> None:
        transport = _Transport(monkeypatch)
        role = _role()
        probes = 0

        async def probe(client: module._RawClient) -> object:
            nonlocal probes
            probes += 1
            if probes == 2:
                if change == 'endpoint':
                    return 'new-primary'
                connection = next(iter(client.connection_pool._in_use_connections))
                await connection.disconnect(nowait=True)
                await connection.connect()
            return 'old-primary'

        role._endpoint_probe = probe
        transport.replies.append(b'$2\r\nok\r\n')
        with pytest.raises(BackendUnavailable, match='changed during'):
            await role.get('key')
        assert all(wire.closed for wire in transport.wires)
        assert role.epoch()[0] >= 1
        await role.close()

    asyncio.run(scenario())


@pytest.mark.parametrize('reply', [b'-NOSCRIPT still absent\r\n', b'-READONLY demoted\r\n'])
def test_script_failure_never_redelivers_again(
    monkeypatch: pytest.MonkeyPatch, reply: bytes
) -> None:
    async def scenario() -> None:
        transport = _Transport(monkeypatch)
        transport.replies.extend(
            [b'-NOSCRIPT absent\r\n', f'$40\r\n{RESOLVE_SCRIPT_SHA}\r\n'.encode(), reply]
        )
        role = _role()
        async with role.pin() as pinned:
            with pytest.raises(BackendUnavailable):
                await module.execute_script(
                    pinned, script=RESOLVE_SCRIPT, sha=RESOLVE_SCRIPT_SHA, keys=('key',), args=()
                )
        assert sum(len(wire.sent) for wire in transport.wires) == 3
        await role.close()

    asyncio.run(scenario())


@pytest.mark.parametrize('when', ['before_entry', 'during'])
def test_pin_release_settles_independent_cleanup_cancellation(
    monkeypatch: pytest.MonkeyPatch, when: str
) -> None:
    async def scenario() -> None:
        _Transport(monkeypatch)
        role = _role()
        pin = role.pin()
        pinned = await pin.__aenter__()
        pool = pinned._client.connection_pool
        original_create_task = asyncio.create_task

        def cancel_cleanup(action: Coroutine[Any, Any, _T]) -> asyncio.Task[_T]:
            task = original_create_task(action)
            task.cancel()
            return task

        async def cancelled_release(_connection: AbstractConnection) -> None:
            raise asyncio.CancelledError

        with monkeypatch.context() as scoped:
            if when == 'before_entry':
                scoped.setattr(support, 'create_task', cancel_cleanup)
            else:
                scoped.setattr(pool, 'release', cancelled_release)
            with pytest.raises(BackendUnavailable):
                await pin.__aexit__(None, None, None)
        assert not pool._in_use_connections
        assert len(pool._available_connections) == 1
        await role.close()

    asyncio.run(scenario())


@pytest.mark.parametrize('phase', ['refresh', 'discovery', 'pin'])
@pytest.mark.parametrize(('remaining', 'delay'), [(0.05, 0.1), (0.1, 0.04)])
def test_commit_debits_prepared_role_delay(
    monkeypatch: pytest.MonkeyPatch,
    phase: str,
    remaining: float,
    delay: float,
) -> None:
    async def scenario() -> None:
        clock = FakeClock()
        monkeypatch.setattr(module, 'monotonic', clock)
        transport = _Transport(monkeypatch)
        transport.replies.append(b'*1\r\n' + _bulk(b'committed'))
        role = _role()
        backend = module.AsyncValkeyBackend(
            role,
            cast('module.AsyncValkeyMetadataStore', BackendClient([])),
            CacheSettings(l2=L2Settings(enabled=True)),
            NullMetrics(),
        )
        original_pin = role.pin
        original_refresh = role.refresh
        original_discovery = role.refresh_endpoint

        @asynccontextmanager
        async def delayed_pin() -> AsyncGenerator[module._PinnedRoleClient]:
            async with original_pin() as pinned:
                clock.advance(delay if phase == 'pin' else 0.0)
                yield pinned

        async def delayed_refresh() -> None:
            await original_refresh()
            if phase == 'refresh':
                clock.advance(delay)

        discoveries = 0

        async def delayed_discovery() -> None:
            nonlocal discoveries
            await original_discovery()
            discoveries += 1
            if phase == 'discovery' and discoveries == 1:
                clock.advance(delay)

        monkeypatch.setattr(role, 'pin', delayed_pin)
        monkeypatch.setattr(role, 'refresh', delayed_refresh)
        monkeypatch.setattr(role, 'refresh_endpoint', delayed_discovery)

        def recover(_breaker: module._CircuitBreaker) -> bool:
            return True

        if phase == 'refresh':
            monkeypatch.setattr(module._CircuitBreaker, 'admit', recover)
        payload = encode_result(TimedResult(0.0, {'id': 7}), 0.0, remaining)
        lease = FillLease('result-key', 'original-token')
        assert await backend.commit(lease, payload, remaining) is (delay < remaining)
        sent = [command for wire in transport.wires for command in wire.sent]
        if delay >= remaining:
            assert sent == []
            assert lease == FillLease('result-key', 'original-token')
        else:
            assert len(sent) == 1
            ttl = int(sent[0].split(b'\r\n')[-2])
            assert 0 < ttl <= 60
            assert payload in sent[0]
            assert decode_hit(module.CacheHit(payload, ttl / 1000), 1.0).origin_elapsed >= 0.04
        await backend.close()

    asyncio.run(scenario())


@pytest.mark.parametrize('load_delay', [0.04, 0.0995, 0.1])
def test_commit_noscript_recovery_spends_original_deadline(
    monkeypatch: pytest.MonkeyPatch,
    load_delay: float,
) -> None:
    async def scenario() -> None:
        clock = FakeClock()
        monkeypatch.setattr(module, 'monotonic', clock)

        def load(_call: tuple[object, ...]) -> object:
            clock.advance(load_delay)
            return COMMIT_SCRIPT_SHA

        result = BackendClient(
            [
                ('evalsha', NoScriptError('missing')),
                ('script_load', load),
                ('evalsha', [b'committed']),
            ]
        )
        backend = module.AsyncValkeyBackend(
            result,
            cast('module.AsyncValkeyMetadataStore', BackendClient([])),
            CacheSettings(l2=L2Settings(enabled=True)),
            NullMetrics(),
        )
        admitted = await backend.commit(FillLease('key', 'token'), encode_value(7), 0.1)
        assert admitted is (load_delay < 0.099)
        calls = [call for call in result.source.calls if call[0] == 'evalsha']
        assert len(calls) == (2 if load_delay < 0.099 else 1)
        if len(calls) == 2:
            assert calls[0][-1] == 100
            assert 0 < cast('int', calls[1][-1]) <= 60
        await backend.close()

    asyncio.run(scenario())


@pytest.mark.parametrize('reconnect', [False, True])
def test_commit_noscript_uses_one_pin_and_rejects_reconnect(
    monkeypatch: pytest.MonkeyPatch,
    reconnect: bool,
) -> None:
    async def scenario() -> None:
        transport = _Transport(monkeypatch)
        transport.replies.extend(
            [
                b'-NOSCRIPT absent\r\n',
                _bulk(COMMIT_SCRIPT_SHA.encode()),
                b'*1\r\n' + _bulk(b'committed'),
            ]
        )
        role = _role()
        original_load = module._PinnedRoleClient.script_load
        pins: list[module._PinnedRoleClient] = []

        async def load(pinned: module._PinnedRoleClient, script: bytes | str) -> object:
            pins.append(pinned)
            reply = await original_load(pinned, script)
            if reconnect:
                connection = pinned._client.connection
                assert connection is not None
                await connection.disconnect(nowait=True)
                await connection.connect()
            return reply

        monkeypatch.setattr(module._PinnedRoleClient, 'script_load', load)
        backend = module.AsyncValkeyBackend(
            role,
            cast('module.AsyncValkeyMetadataStore', BackendClient([])),
            CacheSettings(l2=L2Settings(enabled=True), limits=_FAKE_TRANSPORT_LIMITS),
            NullMetrics(),
        )
        if reconnect:
            with pytest.raises(BackendUnavailable, match='changed during operation'):
                await backend.commit(FillLease('key', 'token'), b'payload', 1.0)
            assert role.epoch()[0] == 1
        else:
            assert await backend.commit(FillLease('key', 'token'), b'payload', 1.0)
            assert role.epoch()[0] == 0
        assert len(pins) == 1
        assert len(transport.wires) == (2 if reconnect else 1)
        assert sum(len(wire.sent) for wire in transport.wires) == 3
        await backend.close()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ('stage', 'error_type', 'expected'),
    [
        ('initial', DataError, BackendProtocolError),
        ('load', DataError, BackendProtocolError),
        ('retry', DataError, BackendProtocolError),
        ('retry', NoScriptError, BackendUnavailable),
    ],
)
def test_commit_redacts_client_exception_context(
    stage: str,
    error_type: type[Exception],
    expected: type[Exception],
) -> None:
    async def scenario() -> None:
        replies: list[tuple[str, object]] = []
        if stage != 'initial':
            replies.append(('evalsha', NoScriptError('missing')))
        if stage == 'retry':
            replies.append(('script_load', COMMIT_SCRIPT_SHA))
        replies.append(
            (
                'script_load' if stage == 'load' else 'evalsha',
                error_type('credential-bearing-client-canary'),
            )
        )
        result = BackendClient(replies)
        backend = module.AsyncValkeyBackend(
            result,
            cast('module.AsyncValkeyMetadataStore', BackendClient([])),
            CacheSettings(l2=L2Settings(enabled=True), limits=_FAKE_TRANSPORT_LIMITS),
            NullMetrics(),
        )
        with pytest.raises(expected) as captured:
            await backend.commit(FillLease('key', 'token'), b'payload', 1.0)
        assert captured.value.__cause__ is None
        assert captured.value.__context__ is None
        assert 'credential-bearing-client-canary' not in _exception_text(captured.value)
        await backend.close()

    asyncio.run(scenario())


@pytest.mark.parametrize('mode', ['standalone', 'managed', 'sentinel'])
def test_configured_role_enables_primary_proof_only_for_ha(mode: _ValkeyMode) -> None:
    async def scenario() -> None:
        settings = ValkeyRoleSettings(
            'cache',
            'PASSWORD',
            mode=mode,
            tls=False,
            sentinels=('one:26379', 'two:26379', 'three:26379') if mode == 'sentinel' else (),
            service_name='service' if mode == 'sentinel' else None,
        )
        role = await module._create_client(
            settings, CacheSettings(limits=_FAKE_TRANSPORT_LIMITS), 'password', NullMetrics()
        )
        assert role._connections._prove_primary == (mode != 'standalone')
        await role.close()

    asyncio.run(scenario())


@pytest.mark.parametrize('mode', ['managed', 'sentinel'])
def test_ha_proof_runs_once_per_new_socket_and_not_per_operation(
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
) -> None:
    async def scenario() -> None:
        transport = _Transport(monkeypatch)
        role = _proving_role(mode)
        primary = _bulk(PRIMARY_INFO)
        transport.replies.extend((primary, _bulk(b'ok'), _bulk(b'ok')))
        for _ in range(2):
            assert await role.run_operation(lambda pin: pin.get('key')) == b'ok'
        assert len(transport.wires) == 1
        async with role.pin() as held:
            transport.replies.extend((primary, _bulk(b'ok')))
            assert await role.run_operation(lambda pin: pin.get('key')) == b'ok'
            assert held._client.connection is not None
            connection = held._client.connection
        await connection.disconnect(nowait=True)
        transport.replies.extend((primary, _bulk(b'ok')))
        assert await role.run_operation(lambda pin: pin.get('key')) == b'ok'
        assert len(transport.wires) == 3
        for wire in transport.wires:
            assert b'INFO' in wire.sent[0]
            assert not any(b'INFO' in command for command in wire.sent[1:])
        assert not transport.replies
        await role.close()

    asyncio.run(scenario())


@pytest.mark.parametrize('mode', ['managed', 'sentinel'])
@pytest.mark.parametrize(
    ('reply', 'failure'),
    [
        (b'role:master\r\n', BackendProtocolError),
        (b'role:slave\r\nmaster_replid:private-value\r\n', BackendUnavailable),
    ],
)
def test_ha_failed_proof_discards_pool_and_new_socket_is_proven(
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    reply: bytes,
    failure: type[Exception],
) -> None:
    async def scenario() -> None:
        transport = _Transport(monkeypatch)
        role = _proving_role(mode)
        pool = role._active()
        transport.replies.append(_bulk(reply))
        with pytest.raises(failure) as caught:
            async with role.pin():
                pytest.fail('rejected proof acquired a pin')
        assert caught.value.__cause__ is None
        assert caught.value.__context__ is None
        assert 'private-value' not in str(caught.value)
        assert role.epoch()[0] == 1
        assert transport.wires[0].closed
        transport.replies.extend((_bulk(PRIMARY_INFO), _bulk(b'ok')))
        assert await role.run_operation(lambda pin: pin.get('key')) == b'ok'
        assert role._client is not pool
        assert len(transport.wires) == 2
        assert b'INFO' in transport.wires[1].sent[0]
        await role.close()

    asyncio.run(scenario())


@pytest.mark.parametrize('mode', ['managed', 'sentinel'])
def test_ha_proof_cancellation_closes_socket_and_next_pin_proves_again(
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
) -> None:
    async def scenario() -> None:
        transport = _Transport(monkeypatch)
        role = _proving_role(mode)

        async def acquire() -> None:
            async with role.pin():
                pytest.fail('proof without a reply acquired a pin')

        task = asyncio.create_task(acquire())
        async with asyncio.timeout(1):
            while not transport.wires or not transport.wires[0].sent:
                if task.done():
                    await task
                await asyncio.sleep(0)
        assert b'INFO' in transport.wires[0].sent[0]
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert transport.wires[0].closed
        transport.replies.append(_bulk(PRIMARY_INFO))
        async with role.pin():
            assert len(transport.wires) == 2
            assert b'INFO' in transport.wires[1].sent[0]
        await role.close()

    asyncio.run(scenario())


@pytest.mark.parametrize('mode', ['managed', 'sentinel'])
def test_ha_reconnect_before_fenced_command_is_proven_and_never_replayed(
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
) -> None:
    async def scenario() -> None:
        transport = _Transport(monkeypatch)
        role = _proving_role(mode)
        control = AsyncValkeyControl(role, key_prefix='sgqc', validation_seconds=30)
        primary = _bulk(PRIMARY_INFO)
        validation = (primary, _bulk(RECORD_A), b':1\r\n') * 2
        transport.replies.extend((primary, *validation))
        await control._validate_incarnation()
        before = role.epoch()[0]
        transport.replies.extend((primary, b':1\r\n'))

        async def operation(
            pin: module.PinnedControlClient, _incarnation: common_module.Incarnation
        ) -> object:
            assert isinstance(pin, module._PinnedRoleClient)
            assert pin._client.connection is not None
            await pin._client.connection.disconnect(nowait=True)
            return await pin.evalsha('0' * 40, 1, 'key')

        with pytest.raises(BackendUnavailable, match='changed during operation'):
            await control.run(operation)
        assert role.epoch()[0] > before
        assert b'INFO' in transport.wires[1].sent[0]
        assert b'EVALSHA' in transport.wires[1].sent[1]
        assert sum(b'EVALSHA' in command for wire in transport.wires for command in wire.sent) == 1
        transport.replies.extend((primary, *validation))
        await control._validate_incarnation()
        assert len(transport.wires) == 3
        assert sum(b'INFO' in command for command in transport.wires[2].sent) == 3
        assert not transport.replies
        await role.close()

    asyncio.run(scenario())


@pytest.mark.parametrize('role', ['result', 'metadata'])
@pytest.mark.parametrize('password', [None, '', 917])
def test_password_errors_name_role_and_variable(
    monkeypatch: pytest.MonkeyPatch, role: str, password: object
) -> None:
    settings = CacheSettings(l2=L2Settings(enabled=True))
    environ: dict[str, object] = {
        settings.l2.valkey.result.password_env: 'sensitive-result',
        settings.l2.valkey.metadata.password_env: 'sensitive-metadata',
    }
    variable = getattr(settings.l2.valkey, role).password_env
    if password is None:
        del environ[variable]
    else:
        environ[variable] = password

    def unexpected_client(*_args: object, **_kwargs: object) -> None:
        pytest.fail('password validation must precede client allocation')

    monkeypatch.setattr(module, '_create_client', unexpected_client)
    message = f'cache.l2.valkey.{role}.password_env variable {variable} is unset or empty'

    async def exercise() -> None:
        with pytest.raises(ValueError, match='password_env') as captured:
            await module.AsyncValkeyBackend.from_settings(settings, environ, NullMetrics())
        assert str(captured.value) == message
        if role == 'metadata':
            with pytest.raises(ValueError, match='password_env') as captured:
                await module.AsyncValkeyMetadataStore.from_settings(
                    settings, environ, NullMetrics()
                )
            assert str(captured.value) == message

    asyncio.run(exercise())


@pytest.mark.parametrize('role_name', ['result', 'metadata'])
def test_role_settlement_keeps_failure_class(
    monkeypatch: pytest.MonkeyPatch, role_name: str
) -> None:
    async def scenario() -> None:
        _Transport(monkeypatch)
        raw_error = ValkeyTimeoutError('credential-canary-settlement')
        calls: list[str] = []

        async def command(_client: Valkey, name: str, *_args: object) -> object:
            calls.append(name)
            if name == 'INFO':
                return {'role': 'master', 'master_replid': 'primary'}
            if name == 'GET':
                return RECORD_A
            if name == 'EXISTS':
                return 1
            assert name == 'EVALSHA'
            raise raw_error

        monkeypatch.setattr(Valkey, 'execute_command', command)
        role = _role()
        settings = CacheSettings(limits=_FAKE_TRANSPORT_LIMITS)
        metadata = module.AsyncValkeyMetadataStore(role, settings, NullMetrics())
        backend = module.AsyncValkeyBackend(role, metadata, settings, NullMetrics())
        operation = (
            backend.lookup('key')
            if role_name == 'result'
            else metadata.resolve((DependencyTag.site('site'),))
        )
        with pytest.raises(BackendUnavailable, match='role changed during operation') as caught:
            await operation
        assert caught.value.failure_class == 'timeout'
        assert caught.value.__cause__ is None
        assert caught.value.__context__ is None
        assert calls.count('EVALSHA') == 1
        traceback = caught.value.__traceback__
        while traceback is not None:
            if traceback.tb_frame.f_code.co_name in {'run_operation', '_settle'}:
                assert all(
                    value is not raw_error for value in traceback.tb_frame.f_locals.values()
                )
            traceback = traceback.tb_next
        await backend.close()

    asyncio.run(scenario())


@pytest.mark.parametrize('boundary', ['connect', 'validation'])
def test_replica_is_classified_at_connect_and_validation(
    monkeypatch: pytest.MonkeyPatch, boundary: str
) -> None:
    async def scenario() -> None:
        transport = _Transport(monkeypatch)
        transport.replies.append(_bulk(b'role:slave\r\nmaster_replid:primary\r\n'))
        role = _proving_role('managed') if boundary == 'connect' else _role()
        metadata = module.AsyncValkeyMetadataStore(
            role, CacheSettings(limits=_FAKE_TRANSPORT_LIMITS), NullMetrics()
        )
        with pytest.raises(BackendUnavailable) as caught:
            await metadata.resolve((DependencyTag.site('site'),))
        assert caught.value.failure_class == 'replica'
        assert caught.value.__cause__ is None
        assert caught.value.__context__ is None
        await metadata.close()

    asyncio.run(scenario())


@pytest.mark.parametrize('role', ['result', 'metadata'])
@pytest.mark.parametrize('half_open', [False, True])
@pytest.mark.parametrize('settlement', ['stalled', 'finite_cleanup', 'caller_cancel'])
def test_published_deadline_owns_stalled_backend_and_cleanup(
    monkeypatch: pytest.MonkeyPatch, role: str, half_open: bool, settlement: str
) -> None:
    async def scenario() -> None:
        clock = FakeClock()
        monkeypatch.setattr(asyncio.get_running_loop(), 'time', clock)
        tested = _DeadlineRole(role, clock, half_open=half_open)
        stalled = _DeadlineStall(clock, settlement)
        monkeypatch.setattr(tested.client, 'evalsha', stalled)
        record = support.DeadlineRecord(clock() + 1)
        with support.publish_deadline(record):
            operation = asyncio.create_task(tested.operation())
            if settlement != 'stalled':
                await stalled.cleanup_entered.wait()
                clock.advance(10)
                await _deadline_turn()
                assert not operation.done()
                if settlement == 'caller_cancel':
                    operation.cancel('caller after expiry')
                else:
                    stalled.cleanup_release.set()
            if settlement == 'caller_cancel':
                with pytest.raises(asyncio.CancelledError, match='caller after expiry'):
                    await operation
            else:
                with pytest.raises(BackendUnavailable, match='deadline') as caught:
                    await operation
                assert caught.value.failure_class == 'timeout'
        assert record.expired
        assert stalled.expired_at == record.deadline
        assert stalled.cancellations == (2 if settlement == 'caller_cancel' else 1)
        transitions = tested.transitions('reopened' if half_open else 'open')
        assert len(transitions) == 1
        assert transitions[0]['failure_class'] == 'timeout'
        with pytest.raises(BackendUnavailable, match='breaker'):
            await tested.operation()
        assert stalled.calls == 1
        await tested.owner.close()

    asyncio.run(scenario())


@pytest.mark.parametrize('role', ['result', 'metadata'])
@pytest.mark.parametrize('half_open', [False, True])
@pytest.mark.parametrize('record_expired', [False, True])
def test_caller_cancellation_uses_own_timer_and_abandons_only_admitted_probe(
    monkeypatch: pytest.MonkeyPatch, role: str, half_open: bool, record_expired: bool
) -> None:
    async def scenario() -> None:
        clock = FakeClock()
        monkeypatch.setattr(asyncio.get_running_loop(), 'time', clock)
        tested = _DeadlineRole(role, clock, half_open=half_open)
        entered = asyncio.Event()

        async def pending() -> None:
            entered.set()
            await asyncio.Event().wait()

        with support.publish_deadline(support.DeadlineRecord(clock() + 1, record_expired)):
            operation = asyncio.create_task(tested.guard('test', pending))
            await entered.wait()
            if half_open:
                with pytest.raises(BackendUnavailable, match='breaker'):
                    await tested.guard('test', pending)
                assert tested.breaker._state == 'half_open'
                assert not tested.transitions('probe_abandoned')
            operation.cancel('caller')
            with pytest.raises(asyncio.CancelledError, match='caller'):
                await operation
        assert len(tested.transitions('probe_abandoned')) == int(half_open)
        assert not tested.transitions('open')
        assert not tested.transitions('reopened')
        assert tested.breaker.admit() is half_open
        tested.breaker.succeed()
        await tested.owner.close()

    asyncio.run(scenario())


@pytest.mark.parametrize('role', ['result', 'metadata'])
@pytest.mark.parametrize('half_open', [False, True])
@pytest.mark.parametrize('failure', [BackendCapacity, StaleIncarnation])
def test_capacity_and_stale_preserve_breaker_settlement(
    role: str, half_open: bool, failure: type[Exception]
) -> None:
    async def scenario() -> None:
        tested = _DeadlineRole(role, FakeClock(), half_open=half_open)

        async def rejected() -> None:
            raise failure('rejected')

        with pytest.raises(failure):
            await tested.guard('test', rejected)
        assert tested.breaker._state == 'closed'
        assert tested.breaker._failures == 0
        assert len(tested.transitions('closed')) == int(half_open)
        rejection = ('cache_metadata_write_rejection', {'reason': 'noeviction'})
        assert (rejection in tested.metrics.increments) is (
            role == 'metadata' and failure is BackendCapacity
        )
        await tested.owner.close()

    asyncio.run(scenario())


@pytest.mark.parametrize('role', ['result', 'metadata'])
@pytest.mark.parametrize('half_open', [False, True])
@pytest.mark.parametrize('failure', [BackendCapacity, StaleIncarnation, ValueError, None])
def test_own_deadline_counts_when_cleanup_replaces_its_outcome(
    monkeypatch: pytest.MonkeyPatch,
    role: str,
    half_open: bool,
    failure: type[Exception] | None,
) -> None:
    async def scenario() -> None:
        clock = FakeClock()
        monkeypatch.setattr(asyncio.get_running_loop(), 'time', clock)
        tested = _DeadlineRole(role, clock, half_open=half_open)
        record = support.DeadlineRecord(clock() + 1)

        async def replaced() -> None:
            clock.advance(1)
            try:
                await _deadline_turn()
                # Advance the fallback timer so an ignored deadline fails without hanging.
                clock.advance(5)
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                if failure is not None:
                    raise failure('replacement') from None

        with support.publish_deadline(record):
            if failure is None:
                await tested.guard('test', replaced)
            else:
                with pytest.raises(failure, match='replacement'):
                    await tested.guard('test', replaced)
        assert record.expired
        assert len(tested.transitions('reopened' if half_open else 'open')) == 1
        assert not tested.transitions('closed')
        assert not tested.transitions('probe_abandoned')
        assert tested.breaker._state == 'open'
        await tested.owner.close()

    asyncio.run(scenario())


@pytest.mark.parametrize('role', ['result', 'metadata'])
def test_delayed_loop_decides_response_with_the_single_backend_timer(
    monkeypatch: pytest.MonkeyPatch, role: str
) -> None:
    async def scenario() -> None:
        clock = FakeClock()
        loop = asyncio.get_running_loop()
        monkeypatch.setattr(loop, 'time', clock)
        tested = _DeadlineRole(role, clock, half_open=False)
        record = support.DeadlineRecord(clock() + 1)
        response = loop.create_future()
        loop.call_at(record.deadline - 0.1, response.set_result, 'ready')

        async def delayed() -> object:
            # Both callbacks become due in one delayed iteration. The response
            # callback queues the task, then the owning timer cancels that task.
            clock.advance(6)
            return await response

        with (
            support.publish_deadline(record),
            pytest.raises(BackendUnavailable, match='deadline'),
        ):
            await tested.guard('test', delayed)
        assert record.expired
        assert response.result() == 'ready'
        assert len(tested.transitions('open')) == 1
        await tested.owner.close()

    asyncio.run(scenario())


def test_span_drift_does_not_discard_a_replacement_pool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        transport = _Transport(monkeypatch)
        role = _role()
        probes = 0
        entered = asyncio.Event()
        proceed = asyncio.Event()

        async def probe(client: module._RawClient) -> object:
            nonlocal probes
            probes += 1
            if probes == 1:
                original_disconnect = client.connection_pool.disconnect

                async def delayed_disconnect(inuse_connections: bool = True) -> None:
                    entered.set()
                    await proceed.wait()
                    await original_disconnect(inuse_connections)

                monkeypatch.setattr(client.connection_pool, 'disconnect', delayed_disconnect)
            return 'primary' if probes == 1 else 'replacement'

        role._endpoint_probe = probe
        transport.replies.append(b'$2\r\nok\r\n')
        command = asyncio.create_task(role.get('key'))
        await entered.wait()
        async with role.pin() as replacement:
            proceed.set()
            with pytest.raises(BackendUnavailable):
                await command
            assert role._client is not None
            assert role._client.connection_pool is replacement._client.connection_pool
            assert not transport.wires[-1].closed
        await role.close()

    asyncio.run(scenario())


@pytest.mark.parametrize('caller_cancel', [False, True])
def test_terminal_close_keeps_first_error_and_records_later_resource_expiry(
    monkeypatch: pytest.MonkeyPatch, caller_cancel: bool
) -> None:
    async def scenario() -> None:
        clock = FakeClock()
        monkeypatch.setattr(asyncio.get_running_loop(), 'time', clock)
        first, second = BackendClient([]), BackendClient([])
        first.close_error = BackendProtocolError('first resource')
        entered = asyncio.Event()
        release = asyncio.Event()

        async def later_close() -> None:
            second.closed += 1
            clock.advance(1)
            try:
                await _deadline_turn()
                # Advance the fallback timer so an ignored deadline fails without hanging.
                clock.advance(5)
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                entered.set()
                await release.wait()
                raise

        monkeypatch.setattr(second, 'close', later_close)
        backend = module.AsyncValkeyBackend(
            first,
            cast('module.AsyncValkeyMetadataStore', second),
            CacheSettings(limits=_FAKE_TRANSPORT_LIMITS),
            NullMetrics(),
        )
        record = support.DeadlineRecord(clock() + 1)
        with support.publish_deadline(record):
            close = asyncio.create_task(backend.close())
            await entered.wait()
            assert clock() == record.deadline
            if caller_cancel:
                close.cancel('caller during close')
            release.set()
            if caller_cancel:
                with pytest.raises(asyncio.CancelledError, match='caller during close'):
                    await close
            else:
                with pytest.raises(BackendProtocolError, match='first resource'):
                    await close
        assert first.closed == second.closed == 1
        assert record.expired

    asyncio.run(scenario())


@pytest.mark.parametrize(
    'backend_type', [module.AsyncValkeyBackend, module.AsyncValkeyMetadataStore]
)
def test_shipped_backends_declare_deadline_ownership(backend_type: type[object]) -> None:
    assert getattr(backend_type, 'owns_published_deadline', False) is True


def test_translated_failure_exposes_no_client_detail(monkeypatch: pytest.MonkeyPatch) -> None:
    canary = 'credential-canary-construction'
    raw_error = ValkeyTimeoutError(canary)
    raw_error.add_note(canary)
    raw_error.__dict__['client_detail'] = canary

    def fail(*_args: object, **_kwargs: object) -> object:
        raise raw_error

    async def scenario() -> None:
        monkeypatch.setattr(Valkey, 'from_pool', fail)
        settings = CacheSettings(l2=L2Settings(enabled=True))
        with pytest.raises(BackendUnavailable) as caught:
            await module.AsyncValkeyBackend.from_settings(
                settings,
                {
                    settings.l2.valkey.result.password_env: 'result',
                    settings.l2.valkey.metadata.password_env: 'metadata',
                },
                NullMetrics(),
            )
        failure = caught.value
        assert failure.failure_class == 'timeout'
        assert canary not in str(failure)
        assert canary not in repr(vars(failure))
        assert not getattr(failure, '__notes__', ())
        assert failure.__cause__ is None
        assert failure.__context__ is None
        assert canary not in ''.join(format_exception(failure))
        traceback = failure.__traceback__
        while traceback is not None:
            if traceback.tb_frame.f_code.co_name == 'translate_client_error':
                assert all(
                    value is not raw_error for value in traceback.tb_frame.f_locals.values()
                )
            traceback = traceback.tb_next

    asyncio.run(scenario())


def test_valkey_async_module_names_missing_extra() -> None:
    result = subprocess.run(
        [
            sys.executable,
            '-c',
            """
import sys

sys.modules['valkey'] = None
try:
    import sg_query_cache.valkey_async
except ImportError as error:
    assert str(error) == (
        "sg_query_cache.valkey_async requires the 'valkey' extra: "
        "pip install 'sg-query-cache[valkey]'"
    )
    assert isinstance(error.__cause__, ModuleNotFoundError)
    assert error.__cause__.name == 'valkey'
else:
    raise AssertionError('missing valkey import succeeded')
""",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
