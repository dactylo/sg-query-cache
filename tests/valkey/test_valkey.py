import gc
import socket
import ssl
from collections.abc import Callable, Generator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, replace
from threading import Barrier, Lock, Thread
from types import SimpleNamespace
from typing import Literal, Protocol, TypeAlias, cast
from unittest.mock import ANY
from weakref import WeakMethod, WeakSet, ref

import pytest
from valkey import Sentinel, Valkey
from valkey.connection import Connection, SSLConnection
from valkey.exceptions import ConnectionError as ValkeyConnectionError
from valkey.exceptions import DataError, NoScriptError, ReadOnlyError, ResponseError
from valkey.exceptions import TimeoutError as ValkeyTimeoutError
from valkey.sentinel import SentinelConnectionPoolProxy

import sg_query_cache._cache_common as cache_common_module
import sg_query_cache._config as config
import sg_query_cache._valkey_common as common_module
import sg_query_cache.valkey as valkey_module
from sg_query_cache._backend import (
    BackendCapacity,
    BackendProtocolError,
    BackendUnavailable,
    CacheBackend,
    StaleIncarnation,
)
from sg_query_cache._cache import QueryCache
from sg_query_cache._codec import encode_value
from sg_query_cache._config import (
    CacheSettings,
    L1Settings,
    L2Settings,
    LimitSettings,
    _ValkeyMode,
)
from sg_query_cache._keys import KeyBuilder
from sg_query_cache._local import LocalBackend
from sg_query_cache._model import (
    DependencyPlan,
    DependencyTag,
    ExactCall,
    FillLease,
    GenerationChange,
    Identity,
)
from sg_query_cache._valkey_common import Incarnation
from sg_query_cache._valkey_control import ValkeyControl
from sg_query_cache._valkey_proof import PrimaryProofError
from sg_query_cache._valkey_scripts import COMMIT_SCRIPT_SHA, CONTROL_CREATE_SCRIPT_SHA
from sg_query_cache.valkey import (
    ValkeyBackend,
    ValkeyMetadataStore,
    _CircuitBreaker,
    _ConnectionGeneration,
    _create_client,
    _RoleClient,
)
from tests._exceptions import exception_text as _exception_text
from tests._fakes import _FakeClock, _RecordingMetrics
from tests._tls import TLS_CA, perform_tls_handshake
from tests._valkey_control_cases import _ACTIVE_CONTROL, _OrderedClient
from tests.valkey._control_helpers import (
    AtomicControlClient,
    ConnectedAtomicControlClient,
    PooledResultClient,
    ReconnectObservable,
)
from tests.valkey._control_helpers import enabled_settings as _role_settings

_ClientPath: TypeAlias = Literal[
    'managed',
    'sentinel-discovery',
    'sentinel-primary',
    'standalone',
]


class _CapturingClient:
    def close(self) -> None:
        return None


class _ConnectObserver:
    def __init__(self, tracker: _ConnectionGeneration, observations: list[int]) -> None:
        self._tracker = tracker
        self._observations = observations

    def connected(self, _connection: object) -> None:
        self._observations.append(self._tracker._generation)


class _FailingSetupConnection:
    def on_connect(self) -> None:
        raise ResponseError('setup failed')


class _SentinelPrimaryPool(Protocol):
    sentinel_manager: Sentinel


class _TLSWrappingConnection(Protocol):
    def _wrap_socket_with_ssl(self, sock: socket.socket) -> ssl.SSLSocket: ...


@dataclass
class _ScriptedRecording:
    metrics: _RecordingMetrics
    created: list[_OrderedClient] | None = None


def _created_reply(call: tuple[object, ...]) -> list[object]:
    return [b'created', call[-1]]


def _delay_lookup_telemetry(
    monkeypatch: pytest.MonkeyPatch,
    clock: _FakeClock,
    seconds: float,
) -> None:
    original_observe = _RecordingMetrics.observe

    def delayed_observe(
        metrics: _RecordingMetrics,
        name: str,
        value: float,
        labels: Mapping[str, str],
    ) -> None:
        original_observe(metrics, name, value, labels)
        if name == 'backend_latency_seconds' and labels == {'backend_operation': 'lookup'}:
            clock._advance(seconds)

    monkeypatch.setattr(_RecordingMetrics, 'observe', delayed_observe)


def _role_for_client_path(
    path: _ClientPath,
    *,
    host: str = 'cache.test',
    tls: bool,
) -> config.ValkeyRoleSettings:
    mode = 'sentinel' if path.startswith('sentinel-') else path
    return config.ValkeyRoleSettings(
        host,
        'PASSWORD',
        mode=cast('_ValkeyMode', mode),
        tls=tls,
        sentinels=('one:26379', 'two:26379', 'three:26379') if mode == 'sentinel' else (),
        service_name='primary' if mode == 'sentinel' else None,
    )


def _validation_replies(replication_id: str) -> list[tuple[str, object]]:
    return [
        ('info', {'role': 'master', 'master_replid': replication_id}),
        ('get', None),
        ('evalsha', _created_reply),
        ('info', {'role': 'master', 'master_replid': replication_id}),
        ('get', _ACTIVE_CONTROL),
        ('exists', 1),
    ]


def enabled_settings() -> CacheSettings:
    return CacheSettings(l1=L1Settings(enabled=False), l2=L2Settings(enabled=True))


def _scripted_backend(
    monkeypatch: pytest.MonkeyPatch,
    *,
    metadata_replies: list[tuple[str, object]] | None = None,
    result_replies: list[tuple[str, object]] | None = None,
    settings: CacheSettings | None = None,
    recording: _ScriptedRecording | None = None,
) -> tuple[CacheBackend, _OrderedClient, _OrderedClient]:
    metadata = _OrderedClient(metadata_replies or [])
    result = _OrderedClient(result_replies or [])
    clients = iter((metadata, result))

    def create_client(**_kwargs: object) -> _OrderedClient:
        client = next(clients, None)
        if client is None:
            client = _OrderedClient([])
        if recording is not None and recording.created is not None:
            recording.created.append(client)
        return client

    monkeypatch.setattr(valkey_module, 'Valkey', create_client)
    backend = ValkeyBackend.from_settings(
        settings or enabled_settings(),
        {
            'SG_QUERY_CACHE_METADATA_PASSWORD': 'metadata-secret',
            'SG_QUERY_CACHE_RESULT_PASSWORD': 'result-secret',
        },
        recording.metrics if recording is not None else _RecordingMetrics(),
    )
    return backend, metadata, result


def test_backend_factory_closes_metadata_after_result_construction_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:

    class _TrackingClient:
        def __init__(self) -> None:
            self._close_calls = 0

        def close(self) -> None:
            self._close_calls += 1

    metadata = _TrackingClient()
    constructor_calls = 0

    def create_client(**_kwargs: object) -> _TrackingClient:
        nonlocal constructor_calls
        constructor_calls += 1
        if constructor_calls == 1:
            return metadata
        raise ValkeyConnectionError('credential-bearing-result-construction')

    monkeypatch.setattr(valkey_module, 'Valkey', create_client)

    with pytest.raises(BackendUnavailable) as captured:
        ValkeyBackend.from_settings(
            enabled_settings(),
            {
                'SG_QUERY_CACHE_METADATA_PASSWORD': 'metadata-secret',
                'SG_QUERY_CACHE_RESULT_PASSWORD': 'result-secret',
            },
            _RecordingMetrics(),
        )

    assert metadata._close_calls == 1
    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None
    assert 'credential-bearing' not in _exception_text(captured.value)


def test_backend_factory_passes_each_role_field_as_a_client_argument(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []

    def create_client(**kwargs: object) -> _CapturingClient:
        calls.append(kwargs)
        return _CapturingClient()

    monkeypatch.setattr(valkey_module, 'Valkey', create_client)
    result_role = config.ValkeyRoleSettings(
        'results.example',
        'RESULT_PASSWORD',
        port=6380,
        database=2,
        tls=True,
        username='result-user',
        key_prefix='result-prefix',
    )
    metadata_role = config.ValkeyRoleSettings(
        'metadata.example',
        'METADATA_PASSWORD',
        port=6379,
        database=3,
        tls=False,
        username='metadata-user',
        key_prefix='metadata-prefix',
    )
    settings = CacheSettings(
        l1=L1Settings(enabled=False),
        l2=L2Settings(
            enabled=True,
            valkey=config.ValkeySettings(result=result_role, metadata=metadata_role),
        ),
    )

    backend = ValkeyBackend.from_settings(
        settings,
        {'RESULT_PASSWORD': 'result-secret', 'METADATA_PASSWORD': 'metadata-secret'},
        _RecordingMetrics(),
    )

    assert backend is not None
    assert calls == [
        {
            'host': 'metadata.example',
            'port': 6379,
            'db': 3,
            'ssl': False,
            'ssl_check_hostname': False,
            'username': 'metadata-user',
            'password': 'metadata-secret',
            'socket_connect_timeout': 0.1,
            'socket_timeout': 0.25,
            'decode_responses': False,
            'retry': ANY,
            'retry_on_timeout': False,
            'valkey_connect_func': ANY,
        },
        {
            'host': 'results.example',
            'port': 6380,
            'db': 2,
            'ssl': True,
            'ssl_check_hostname': True,
            'username': 'result-user',
            'password': 'result-secret',
            'socket_connect_timeout': 0.1,
            'socket_timeout': 0.25,
            'decode_responses': False,
            'retry': ANY,
            'retry_on_timeout': False,
            'valkey_connect_func': ANY,
        },
    ]


def test_backend_factory_validates_both_passwords_before_allocating(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []

    def create_client(**kwargs: object) -> _CapturingClient:
        calls.append(kwargs)
        return _CapturingClient()

    monkeypatch.setattr(valkey_module, 'Valkey', create_client)

    with pytest.raises(ValueError, match='password'):
        ValkeyBackend.from_settings(
            enabled_settings(),
            {'SG_QUERY_CACHE_METADATA_PASSWORD': 'metadata-secret'},
            _RecordingMetrics(),
        )

    assert calls == []


def test_breaker_allows_only_one_half_open_probe_and_recovers() -> None:
    clock = _FakeClock()
    breaker = _CircuitBreaker(
        'metadata',
        threshold=5,
        probe_seconds=10,
        metrics=_RecordingMetrics(),
        clock=clock,
    )
    for _ in range(4):
        assert not breaker.admit()
        breaker.fail()
    assert not breaker.admit()
    breaker.fail()

    with pytest.raises(BackendUnavailable):
        breaker.admit()
    clock._advance(10)
    assert breaker.admit()
    with pytest.raises(BackendUnavailable):
        breaker.admit()

    breaker.succeed()
    assert not breaker.admit()


def test_breakers_reopen_after_a_failed_probe_and_are_role_independent() -> None:
    clock = _FakeClock()
    metrics = _RecordingMetrics()
    metadata = _CircuitBreaker(
        'metadata', threshold=1, probe_seconds=10, metrics=metrics, clock=clock
    )
    result = _CircuitBreaker('result', threshold=1, probe_seconds=10, metrics=metrics, clock=clock)
    metadata.fail()

    assert not result.admit()
    clock._advance(10)
    assert metadata.admit()
    metadata.fail()
    with pytest.raises(BackendUnavailable):
        metadata.admit()
    assert not result.admit()
    assert metrics._increments == [
        ('backend_failure', {'backend_operation': 'metadata', 'outcome': 'open'}),
        ('backend_failure', {'backend_operation': 'metadata', 'outcome': 'half_open'}),
        ('backend_failure', {'backend_operation': 'metadata', 'outcome': 'reopened'}),
    ]


def test_change_and_site_flush_each_use_one_atomic_metadata_script(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replid = 'a' * 40
    backend, metadata, result = _scripted_backend(
        monkeypatch,
        metadata_replies=[
            *_validation_replies(replid),
            ('evalsha', [b'ok', 1, 1]),
            ('evalsha', [b'ok', 1, 0]),
        ],
    )
    changes = (
        GenerationChange(DependencyTag.site('production')),
        GenerationChange(DependencyTag.entity('production', 'Asset', 7)),
    )

    assert backend.change(changes) is None
    assert backend.flush_site('production', 'operator') is None

    assert len([call for call in metadata._calls if call[0] == 'evalsha']) == 3
    assert result._calls == []


def test_close_attempts_both_failing_roles_once_and_redacts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:

    class _FailingCloseClient:
        def __init__(self, detail: str) -> None:
            self._close_calls = 0
            self._detail = detail

        def close(self) -> None:
            self._close_calls += 1
            raise ValkeyConnectionError(self._detail)

    metadata = _FailingCloseClient('credential-bearing-metadata-close')
    result = _FailingCloseClient('credential-bearing-result-close')
    clients = iter((metadata, result))

    def create_client(**_kwargs: object) -> _FailingCloseClient:
        return next(clients)

    monkeypatch.setattr(valkey_module, 'Valkey', create_client)
    backend = ValkeyBackend.from_settings(
        enabled_settings(),
        {
            'SG_QUERY_CACHE_METADATA_PASSWORD': 'metadata-secret',
            'SG_QUERY_CACHE_RESULT_PASSWORD': 'result-secret',
        },
        _RecordingMetrics(),
    )

    with pytest.raises(BackendUnavailable) as captured:
        backend.close()
    backend.close()

    assert result._close_calls == 1
    assert metadata._close_calls == 1
    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None
    assert 'credential-bearing' not in _exception_text(captured.value)


def test_close_releases_both_role_clients_once(monkeypatch: pytest.MonkeyPatch) -> None:
    backend, metadata, result = _scripted_backend(monkeypatch)

    backend.close()
    backend.close()

    assert metadata._calls == [('close',)]
    assert result._calls == [('close',)]


def test_complete_valkey_hit_decodes_once(monkeypatch: pytest.MonkeyPatch) -> None:
    replid = 'a' * 40
    payload = encode_value(('sg-query-cache-result-v1', 0.0, 30.0, 0.01, {'items': [1]}))
    backend, _metadata, _result = _scripted_backend(
        monkeypatch,
        metadata_replies=[
            *_validation_replies(replid),
            ('evalsha', [b'ok', 1]),
        ],
        result_replies=[('evalsha', [b'hit', payload, 30_000])],
    )
    settings = enabled_settings()
    cache = QueryCache(
        settings,
        Identity('script', 'studio', 'render'),
        KeyBuilder(),
        l2=backend,
    )
    call = ExactCall('find', (('entity_type', 'Asset'), ('filters', ())))
    plan = DependencyPlan(
        'event_aware',
        (DependencyTag.site('production'),),
        'event_aware',
    )
    decodes = {'backend': 0, 'core': 0}
    original_backend_decode = common_module.decode_value
    original_core_decode = cache_common_module.decode_value

    def count_backend_decode(encoded: bytes) -> object:
        decodes['backend'] += 1
        return original_backend_decode(encoded)

    def count_core_decode(encoded: bytes) -> object:
        decodes['core'] += 1
        return original_core_decode(encoded)

    monkeypatch.setattr(common_module, 'decode_value', count_backend_decode)
    monkeypatch.setattr(cache_common_module, 'decode_value', count_core_decode)

    value = cache.read(
        call,
        plan,
        lambda: pytest.fail('complete Valkey hit must not fetch'),
        {'localized': False, 'convert_datetimes_to_utc': True},
    )

    assert value == {'items': [1]}
    assert decodes == {'backend': 1, 'core': 0}


def test_connection_generation_callback_owner_supports_weak_reference() -> None:
    generation = _ConnectionGeneration()

    callback = WeakMethod(generation._connected)

    assert callback() is not None


def test_connection_tracking_releases_dead_connection_objects() -> None:
    generation = _ConnectionGeneration()
    connection = ReconnectObservable()
    borrowed = SimpleNamespace(connection=connection)
    weak_connection = ref(connection)

    generation._borrowed(borrowed)
    connection._reconnect()
    assert generation._generation == 1
    del borrowed
    del connection
    gc.collect()

    registered = object.__getattribute__(generation, '_registered')
    assert weak_connection() is None
    assert isinstance(registered, WeakSet)
    assert len(cast('WeakSet[object]', registered)) == 0

    replacement = ReconnectObservable()
    generation._borrowed(SimpleNamespace(connection=replacement))
    replacement._reconnect()
    assert generation._generation == 2


def test_expired_commit_keeps_original_sg_result_and_releases_original_lease(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = _FakeClock()
    monkeypatch.setattr(valkey_module, 'monotonic', clock)
    settings = enabled_settings()

    def load(_call: tuple[object, ...]) -> object:
        clock._advance(1.0)
        return COMMIT_SCRIPT_SHA

    result = _OrderedClient(
        [
            ('evalsha', [b'miss']),
            ('evalsha', [b'reserved']),
            ('evalsha', [b'miss']),
            ('evalsha', NoScriptError('missing')),
            ('script_load', load),
            ('evalsha', [b'released']),
        ]
    )
    role = valkey_module._RoleClient(
        lambda: cast('valkey_module._RawValkeyClient', result),
        valkey_module._ConnectionGeneration(),
    )
    backend = ValkeyBackend(
        role,
        LocalBackend(settings, clock, _RecordingMetrics()),
        settings,
        _RecordingMetrics(),
    )
    cache = QueryCache(settings, Identity('script', 'studio', 'render'), KeyBuilder(), l2=backend)
    answer = {'items': [1]}
    value = cache.read(
        ExactCall('find', (('entity_type', 'Asset'), ('filters', ()))),
        DependencyPlan('event_aware', (DependencyTag.site('production'),), 'event_aware'),
        lambda: answer,
        {'localized': False, 'convert_datetimes_to_utc': True},
        per_call_age=0.05,
    )
    assert value is answer
    assert result._replies == []
    reserve, attempt, release = result._calls[1], result._calls[3], result._calls[-1]
    assert reserve[-2] == attempt[-3] == release[-1]
    cache.close()


def test_failed_production_connection_setup_does_not_advance_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    callbacks: list[Callable[[object], None]] = []

    def create_client(**kwargs: object) -> AtomicControlClient:
        callbacks.append(cast('Callable[[object], None]', kwargs['valkey_connect_func']))
        return AtomicControlClient()

    monkeypatch.setattr(valkey_module, 'Valkey', create_client)
    settings = _role_settings()
    role = _create_client(settings.l2.valkey.metadata, settings, 'secret')

    with pytest.raises(ResponseError, match='setup failed'):
        callbacks[0](_FailingSetupConnection())

    assert role.epoch() == (0, 0, None)


def test_lease_reserve_commit_and_release_are_token_checked_scripts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = encode_value({'id': 7})
    backend, metadata, result = _scripted_backend(
        monkeypatch,
        result_replies=[
            ('evalsha', [b'reserved']),
            ('evalsha', [b'committed']),
            ('evalsha', [b'reserved']),
            ('evalsha', [b'released']),
        ],
    )

    committed_lease = backend.reserve('qualified-result', 2.0)
    assert committed_lease is not None
    assert backend.commit(committed_lease, payload, 3.5)
    released_lease = backend.reserve('qualified-result', 2.0)
    assert released_lease is not None
    assert backend.release(released_lease)

    assert committed_lease.key == released_lease.key == 'qualified-result'
    assert committed_lease.token != released_lease.token
    assert all(
        len(lease.token) == 64
        and lease.token.lower() == lease.token
        and all(character in '0123456789abcdef' for character in lease.token)
        for lease in (committed_lease, released_lease)
    )
    assert metadata._calls == []
    assert [call[0] for call in result._calls] == ['evalsha'] * 4


def test_local_input_rejections_do_not_open_the_result_breaker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:

    class _ResultClient(AtomicControlClient):
        def __init__(self) -> None:
            super().__init__()
            self._script_calls = 0

        def evalsha(self, sha: str, numkeys: int, *keys_and_args: object) -> object:
            _ = sha, numkeys, keys_and_args
            self._script_calls += 1
            return [b'miss']

    result = _ResultClient()
    clients = iter((AtomicControlClient(), result))

    def create_client(**kwargs: object) -> AtomicControlClient:
        _ = kwargs
        return next(clients)

    monkeypatch.setattr(valkey_module, 'Valkey', create_client)
    settings = _role_settings()
    settings = replace(
        settings,
        limits=replace(settings.limits, circuit_failure_threshold=1),
    )
    backend = ValkeyBackend.from_settings(
        settings,
        {'METADATA_PASSWORD': 'secret', 'RESULT_PASSWORD': 'secret'},
        _RecordingMetrics(),
    )

    for _ in range(5):
        with pytest.raises(BackendProtocolError):
            backend.lookup('')
    assert backend.lookup('qualified-key') is None
    assert result._script_calls == 1


def test_lookup_debits_guard_telemetry_from_remaining_age(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = _FakeClock()
    payload = encode_value({'id': 7})
    backend, _metadata, _result = _scripted_backend(
        monkeypatch,
        result_replies=[('evalsha', [b'hit', payload, 1_250])],
    )
    monkeypatch.setattr(valkey_module, 'monotonic', clock)
    _delay_lookup_telemetry(monkeypatch, clock, 0.5)

    hit = backend.lookup('qualified-result')

    assert hit is not None
    assert hit.remaining_seconds == 0.75


def test_lookup_debits_local_decode_time_from_server_ttl(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = _FakeClock()
    payload = encode_value({'id': 7})
    backend, _metadata, _result = _scripted_backend(
        monkeypatch,
        result_replies=[('evalsha', [b'hit', payload, 1_250])],
    )
    original_decode = common_module.decode_value
    decode_count = 0

    def delayed_decode(encoded: bytes) -> object:
        nonlocal decode_count
        decode_count += 1
        value = original_decode(encoded)
        clock._advance(0.25)
        return value

    monkeypatch.setattr(common_module, 'decode_value', delayed_decode)
    monkeypatch.setattr(valkey_module, 'monotonic', clock)

    hit = backend.lookup('qualified-result')

    assert hit is not None
    assert hit.decoded is not None
    assert hit.decoded.value == {'id': 7}
    assert hit.remaining_seconds == 1.0
    assert decode_count == 1


def test_lookup_expiring_during_guard_telemetry_is_a_miss(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = _FakeClock()
    payload = encode_value({'id': 7})
    backend, _metadata, _result = _scripted_backend(
        monkeypatch,
        result_replies=[('evalsha', [b'hit', payload, 250])],
    )
    monkeypatch.setattr(valkey_module, 'monotonic', clock)
    _delay_lookup_telemetry(monkeypatch, clock, 0.5)

    assert backend.lookup('qualified-result') is None


def test_lookup_expiring_during_local_decode_is_a_miss(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = _FakeClock()
    payload = encode_value({'id': 7})
    backend, _metadata, _result = _scripted_backend(
        monkeypatch,
        result_replies=[('evalsha', [b'hit', payload, 250])],
    )
    original_decode = common_module.decode_value

    def delayed_decode(encoded: bytes) -> object:
        value = original_decode(encoded)
        clock._advance(0.25)
        return value

    monkeypatch.setattr(common_module, 'decode_value', delayed_decode)
    monkeypatch.setattr(valkey_module, 'monotonic', clock)

    assert backend.lookup('qualified-result') is None


def test_lookup_uses_one_result_script_and_server_ttl(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = encode_value({'id': 7})
    backend, metadata, result = _scripted_backend(
        monkeypatch,
        result_replies=[('evalsha', [b'hit', payload, 1_250])],
    )

    hit = backend.lookup('qualified-result')

    assert hit is not None
    assert hit.payload == payload
    assert hit.remaining_seconds == pytest.approx(1.25, abs=0.01)
    assert metadata._calls == []
    assert len(result._calls) == 1
    assert result._calls[0][0:3] == ('evalsha', result._calls[0][1], 1)


def test_managed_mode_uses_the_direct_writable_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []

    def create_client(**kwargs: object) -> AtomicControlClient:
        calls.append(kwargs)
        return AtomicControlClient()

    monkeypatch.setattr(valkey_module, 'Valkey', create_client)
    backend = ValkeyBackend.from_settings(
        _role_settings('managed'),
        {'METADATA_PASSWORD': 'metadata-secret', 'RESULT_PASSWORD': 'result-secret'},
        _RecordingMetrics(),
    )
    backend.close()

    assert [call['host'] for call in calls] == ['metadata.example', 'result.example']
    assert all('service_name' not in call for call in calls)
    for call in calls:
        attempts = 0

        def fail() -> None:
            nonlocal attempts
            attempts += 1
            raise OSError('simulated command failure')

        retry = cast('_RetryPolicy', call['retry'])
        with pytest.raises(OSError, match='simulated command failure'):
            retry.call_with_retry(fail, lambda _error: None)
        assert attempts == 1


def test_metadata_oom_is_redacted_capacity(monkeypatch: pytest.MonkeyPatch) -> None:
    replid = 'a' * 40
    backend, _metadata, _result = _scripted_backend(
        monkeypatch,
        metadata_replies=[
            *_validation_replies(replid),
            ('evalsha', ResponseError('OOM credential-bearing-capacity-detail')),
        ],
    )

    with pytest.raises(BackendCapacity) as captured:
        backend.resolve((DependencyTag.site('production'),))

    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None
    assert 'credential-bearing' not in _exception_text(captured.value)


def test_metadata_rotation_publishes_changed_nonce_and_control_metric() -> None:
    class _RotationClient(_OrderedClient):
        def __init__(self) -> None:
            super().__init__([])
            self._record = f'{"a" * 40}|{"1" * 32}|sgqc:v1:counter:{"1" * 32}'.encode()
            self._rotations = 0

        def evalsha(self, sha: str, numkeys: int, *keys_and_args: object) -> object:
            self._calls.append(('evalsha', sha, numkeys, *keys_and_args))
            self._rotations += 1
            self._record = cast('bytes', keys_and_args[-1])
            return [b'rotated', self._record]

        def exists(self, key: str) -> object:
            return int(key.encode() == self._record.split(b'|')[2])

        def get(self, key: str) -> object:
            assert key == 'sgqc:v1:control'
            return self._record

        def info(self, section: str) -> object:
            assert section == 'replication'
            return {'role': 'master', 'master_replid': 'a' * 40}

    client = _RotationClient()
    metrics = _RecordingMetrics()
    store = valkey_module.ValkeyMetadataStore(client, CacheSettings(), metrics)
    observed = store._control.validate()
    store.rotate_incarnation()
    replacement = store._control.validate()
    assert replacement.nonce != observed.nonce
    assert client._rotations == 1
    assert metrics._increments.count(('cache_incarnation', {'outcome': 'rotated'})) == 1


def test_metadata_rotation_redacts_client_failure_and_opens_breaker() -> None:
    client = _OrderedClient([('info', ValkeyConnectionError('credential-secret'))])
    metrics = _RecordingMetrics()
    settings = CacheSettings(limits=LimitSettings(circuit_failure_threshold=1))
    store = valkey_module.ValkeyMetadataStore(client, settings, metrics)
    with pytest.raises(BackendUnavailable) as captured:
        store.rotate_incarnation()
    assert 'credential-secret' not in _exception_text(captured.value)
    calls = list(client._calls)
    with pytest.raises(BackendUnavailable):
        store.rotate_incarnation()
    assert client._calls == calls
    assert (
        'backend_failure',
        {'backend_operation': 'metadata', 'outcome': 'failure'},
    ) in metrics._increments


def test_metadata_rotation_rejects_closed_store_before_guard() -> None:
    client = _OrderedClient([])
    metrics = _RecordingMetrics()
    store = valkey_module.ValkeyMetadataStore(client, CacheSettings(), metrics)
    store.close()
    increments, observations = list(metrics._increments), list(metrics._observations)
    with pytest.raises(BackendUnavailable, match='closed'):
        store.rotate_incarnation()
    assert client._calls == [('close',)]
    assert metrics._increments == increments
    assert metrics._observations == observations


def test_oversized_result_is_not_sent_to_commit_script(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend, _metadata, result = _scripted_backend(
        monkeypatch,
        result_replies=[('evalsha', [b'reserved']), ('evalsha', [b'released'])],
    )
    lease = backend.reserve('qualified-result', 2.0)
    assert lease is not None
    payload = b'x' * 1_048_577

    assert not backend.commit(lease, payload, 3.0)

    assert len(result._calls) == 2
    assert all(payload not in call for call in result._calls)


def test_resolve_accepts_the_v1_valkey_tag_ceiling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replid = 'a' * 40
    tokens = list(range(1, 33))
    backend, metadata, _result = _scripted_backend(
        monkeypatch,
        metadata_replies=[
            *_validation_replies(replid),
            ('evalsha', [b'ok', *tokens]),
        ],
        settings=CacheSettings(
            l1=L1Settings(enabled=False),
            l2=L2Settings(enabled=True),
            limits=LimitSettings(max_dependency_vector_bytes=10_000),
        ),
    )
    tags = tuple(
        DependencyTag.entity('production', 'Asset', entity_id) for entity_id in range(1, 33)
    )

    vector = backend.resolve(tags)

    assert vector.entries == tuple(zip(tags, tokens, strict=True))
    assert len([call for call in metadata._calls if call[0] == 'evalsha']) == 2


def test_resolve_rejects_more_than_v1_ceiling_before_evalsha(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend, metadata, result = _scripted_backend(monkeypatch)
    tags = tuple(
        DependencyTag.entity('production', 'Asset', entity_id) for entity_id in range(1, 34)
    )

    with pytest.raises(BackendProtocolError, match='dependency vector exceeds the tag bound'):
        backend.resolve(tags)

    assert metadata._calls == []
    assert result._calls == []


def test_resolve_uses_one_batched_metadata_script_after_control_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replid = 'a' * 40
    metadata = _OrderedClient(
        [
            *_validation_replies(replid),
            ('evalsha', [b'ok', 1, 2]),
            ('evalsha', [b'ok', b'1', b'2']),
        ]
    )
    result = _OrderedClient([])
    clients = iter((metadata, result))

    def create_client(**_kwargs: object) -> _OrderedClient:
        return next(clients)

    monkeypatch.setattr(valkey_module, 'Valkey', create_client)
    settings = enabled_settings()
    backend = ValkeyBackend.from_settings(
        settings,
        {
            'SG_QUERY_CACHE_METADATA_PASSWORD': 'metadata-secret',
            'SG_QUERY_CACHE_RESULT_PASSWORD': 'result-secret',
        },
        _RecordingMetrics(),
    )
    tags = (
        DependencyTag.site('production'),
        DependencyTag.entity('production', 'Asset', 7),
    )

    first = backend.resolve(tags)
    second = backend.resolve(tags)

    assert first.entries == ((tags[0], 1), (tags[1], 2))
    assert second.entries == first.entries
    assert len([call for call in metadata._calls if call[0] == 'evalsha']) == 3
    assert result._calls == []


def test_result_pinned_reconnect_discards_only_after_own_connection_drift() -> None:
    result = PooledResultClient(reconnect_indexes={0})
    role = _RoleClient(lambda: result, _ConnectionGeneration())

    with pytest.raises(BackendUnavailable):
        role.evalsha('sha', 0)

    assert len(result._children) == 1
    assert result._children[0].connection._callback_observations == 2
    assert result._close_calls == 1
    assert role.epoch()[1] >= 2


def test_result_pool_expansion_does_not_discard_overlapping_pins() -> None:
    result = PooledResultClient(barrier=Barrier(2))
    role = _RoleClient(lambda: result, _ConnectionGeneration())
    outcomes: list[object] = []
    failures: list[BackendUnavailable] = []
    outcome_lock = Lock()

    def invoke() -> None:
        try:
            outcome = role.evalsha('sha', 0)
            with outcome_lock:
                outcomes.append(outcome)
        except BackendUnavailable as error:
            with outcome_lock:
                failures.append(error)

    threads = [Thread(target=invoke) for _index in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert failures == []
    assert sorted(cast('list[list[object]]', outcomes), key=lambda item: cast('int', item[1])) == [
        [b'child', 0],
        [b'child', 1],
    ]
    assert len({id(child.connection) for child in result._children}) == 2
    assert result._close_calls == 0
    assert role.epoch() == (0, 2, None)


def test_result_sentinel_endpoint_drift_is_driven_by_backend_operation() -> None:
    endpoint = ['old']

    class _ResultEndpointClient(AtomicControlClient):
        def __init__(self, name: str) -> None:
            super().__init__()
            self._close_calls = 0
            self._name = name
            self._script_calls = 0

        def close(self) -> None:
            self._close_calls += 1

        def evalsha(self, sha: str, numkeys: int, *keys_and_args: object) -> object:
            _ = sha, numkeys, keys_and_args
            self._script_calls += 1
            return [b'miss']

    clients = {name: _ResultEndpointClient(name) for name in ('old', 'new')}
    settings = _role_settings('sentinel')
    role = _RoleClient(
        lambda: clients[endpoint[0]],
        _ConnectionGeneration(),
        lambda _client: endpoint[0],
    )
    metadata = ValkeyMetadataStore(AtomicControlClient(), settings, _RecordingMetrics())
    backend = ValkeyBackend(role, metadata, settings, _RecordingMetrics())
    endpoint[0] = 'new'

    with pytest.raises(BackendUnavailable):
        backend.lookup('qualified-key')

    assert clients['old']._script_calls == 0
    assert clients['old']._close_calls == 1
    assert role.epoch()[0] == 1


def test_role_epoch_reads_are_pure_and_keep_components_separate() -> None:
    probes = 0
    client = ConnectedAtomicControlClient()

    def endpoint_probe(_client: object) -> str:
        nonlocal probes
        probes += 1
        return 'primary-a'

    role = _RoleClient(lambda: client, _ConnectionGeneration(), endpoint_probe)
    role.refresh_endpoint()
    before = role.epoch()
    after = role.epoch()

    assert before == after == (0, 0, 'primary-a')
    assert probes == 2


@pytest.mark.parametrize('operation', ['resolve', 'change', 'control_create', 'commit'])
@pytest.mark.parametrize('refusal', ['readonly', 'noreplicas'])
def test_script_write_refusal_opens_role_breaker_without_replaying(
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
    refusal: str,
) -> None:
    metrics = _RecordingMetrics()
    created: list[_OrderedClient] = []
    error = (
        ReadOnlyError("You can't write against a read only replica.")
        if refusal == 'readonly'
        else ResponseError('NOREPLICAS Not enough good replicas to write.')
    )
    metadata_replies: list[tuple[str, object]] = []
    result_replies: list[tuple[str, object]] = []
    if operation in ('resolve', 'change'):
        metadata_replies.extend(_validation_replies('a' * 40))
    elif operation == 'control_create':
        metadata_replies.extend(
            [('info', {'role': 'master', 'master_replid': 'a' * 40}), ('get', None)]
        )
    else:
        result_replies.append(('evalsha', [b'reserved']))
    role = 'result' if operation == 'commit' else 'metadata'
    (result_replies if role == 'result' else metadata_replies).append(('evalsha', error))
    backend, metadata, result = _scripted_backend(
        monkeypatch,
        metadata_replies=metadata_replies,
        result_replies=result_replies,
        settings=CacheSettings(
            l1=L1Settings(enabled=False),
            l2=L2Settings(enabled=True),
            limits=LimitSettings(circuit_failure_threshold=1),
        ),
        recording=_ScriptedRecording(metrics, created),
    )

    def attempt() -> object:
        if operation in ('resolve', 'control_create'):
            return backend.resolve((DependencyTag.site('production'),))
        if operation == 'change':
            return backend.change((GenerationChange(DependencyTag.site('production')),))
        lease = backend.reserve('qualified-result', 2.0)
        assert lease is not None
        return backend.commit(lease, encode_value({'id': 7}), 3.5)

    with pytest.raises(BackendUnavailable):
        attempt()

    original = result if role == 'result' else metadata
    assert (('close',) in original._calls) is (refusal == 'readonly')
    assert ('backend_failure', {'backend_operation': role, 'outcome': 'open'}) in (
        metrics._increments
    )
    assert not any(name == 'cache_metadata_write_rejection' for name, _ in metrics._increments)
    if role == 'result':
        assert ('cache_commit', {'layer': 'l2', 'outcome': 'rejected'}) not in metrics._increments
        assert not any(name == 'cache_lease_rejection' for name, _ in metrics._increments)
    expected_scripts = 1 if operation == 'control_create' else 2
    assert sum(call[0] == 'evalsha' for client in created for call in client._calls) == (
        expected_scripts
    )
    if operation == 'control_create':
        assert next(call for call in metadata._calls if call[0] == 'evalsha')[1] == (
            CONTROL_CREATE_SCRIPT_SHA
        )
    backend.close()


@pytest.mark.parametrize('operation', ['resolve', 'commit'])
def test_script_capacity_keeps_existing_caller_outcome(
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    metrics = _RecordingMetrics()
    backend, metadata, result = _scripted_backend(
        monkeypatch,
        metadata_replies=[*_validation_replies('a' * 40), ('evalsha', [b'capacity'])]
        if operation == 'resolve'
        else None,
        result_replies=[('evalsha', [b'reserved']), ('evalsha', [b'capacity'])]
        if operation == 'commit'
        else None,
        settings=CacheSettings(
            l1=L1Settings(enabled=False),
            l2=L2Settings(enabled=True),
            limits=LimitSettings(circuit_failure_threshold=1),
        ),
        recording=_ScriptedRecording(metrics),
    )

    if operation == 'resolve':
        with pytest.raises(BackendCapacity):
            backend.resolve((DependencyTag.site('production'),))
        assert ('cache_metadata_write_rejection', {'reason': 'noeviction'}) in (
            metrics._increments
        )
    else:
        lease = backend.reserve('qualified-result', 2.0)
        assert lease is not None
        assert backend.commit(lease, encode_value({'id': 7}), 3.5) is False
    assert not any(
        name == 'backend_failure' and labels['outcome'] == 'open'
        for name, labels in metrics._increments
    )
    assert ('close',) not in (metadata if operation == 'resolve' else result)._calls
    backend.close()


@pytest.mark.parametrize('unchanged', [False, True])
def test_metadata_rotation_forces_observation_and_rotates_once(
    monkeypatch: pytest.MonkeyPatch,
    unchanged: bool,
) -> None:
    metrics = _RecordingMetrics()
    client = _OrderedClient([])
    store = valkey_module.ValkeyMetadataStore(client, CacheSettings(), metrics)
    observed = Incarnation(
        'a' * 40, '1' * 32, 'counter-one', f'{"a" * 40}|{"1" * 32}|counter-one'.encode()
    )
    replacement = (
        observed
        if unchanged
        else Incarnation(
            'a' * 40, '2' * 32, 'counter-two', f'{"a" * 40}|{"2" * 32}|counter-two'.encode()
        )
    )
    calls: list[object] = []

    def validate(control: ValkeyControl, force: bool = False) -> Incarnation:
        assert control is store._control
        calls.append(('validate', force))
        return observed

    def rotate(control: ValkeyControl, active: Incarnation) -> Incarnation:
        assert control is store._control
        calls.append(('rotate', active))
        return replacement

    monkeypatch.setattr(ValkeyControl, 'validate', validate)
    monkeypatch.setattr(ValkeyControl, 'rotate', rotate)
    if unchanged:
        with pytest.raises(BackendUnavailable, match='did not rotate'):
            store.rotate_incarnation()
        assert (
            'backend_failure',
            {'backend_operation': 'metadata', 'outcome': 'failure'},
        ) in metrics._increments
    else:
        assert store.rotate_incarnation() is None
        assert any(
            name == 'backend_latency_seconds' and labels == {'backend_operation': 'metadata'}
            for name, _, labels in metrics._observations
        )
    assert calls == [('validate', True), ('rotate', observed)]


@pytest.mark.parametrize(
    'path',
    ['standalone', 'managed', 'sentinel-discovery', 'sentinel-primary'],
)
@pytest.mark.parametrize('tls', [False, True])
def test_role_constructor_checks_hostnames_only_for_tls_connections(
    path: _ClientPath,
    tls: bool,
) -> None:
    settings = CacheSettings()
    role = _role_for_client_path(path, tls=tls)
    client = valkey_module._create_raw_client(
        role,
        settings,
        'password',
        valkey_module._ConnectionGeneration(),
    )
    discovery_clients: list[Valkey] = []
    try:
        raw = cast('Valkey', cast('object', client))
        if path.startswith('sentinel-'):
            pool = cast(
                '_SentinelPrimaryPool',
                cast('object', raw.connection_pool),
            )
            discovery_clients = pool.sentinel_manager.sentinels
        if path == 'sentinel-discovery':
            assert discovery_clients
            raw = discovery_clients[0]
        connection = raw.connection_pool.make_connection()
        assert isinstance(connection, SSLConnection) is tls
        if tls:
            assert cast('SSLConnection', connection).check_hostname is True
        else:
            assert isinstance(connection, Connection)
    finally:
        client.close()
        for discovery in discovery_clients:
            discovery.close()


@pytest.mark.parametrize(('hostname', 'accepted'), [('cache.test', True), ('other.test', False)])
def test_configured_tls_connection_verifies_server_hostname(
    monkeypatch: pytest.MonkeyPatch,
    hostname: str,
    accepted: bool,
) -> None:
    monkeypatch.setenv('SSL_CERT_FILE', str(TLS_CA))
    role = _role_for_client_path('standalone', host=hostname, tls=True)
    client = valkey_module._create_raw_client(
        role,
        CacheSettings(),
        'password',
        valkey_module._ConnectionGeneration(),
    )
    try:
        raw = cast('Valkey', cast('object', client))
        connection = cast(
            '_TLSWrappingConnection',
            cast('object', raw.connection_pool.make_connection()),
        )

        def wrap(sock: socket.socket) -> ssl.SSLSocket:
            return connection._wrap_socket_with_ssl(sock)

        if accepted:
            perform_tls_handshake(wrap)
        else:
            with pytest.raises(ssl.SSLCertVerificationError) as captured:
                perform_tls_handshake(wrap)
            assert 'hostname mismatch' in captured.value.verify_message.lower()
    finally:
        client.close()


@pytest.mark.parametrize('elapsed', [0.025, 0.100])
def test_lookup_debits_endpoint_settlement_from_server_ttl(
    elapsed: float, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = _FakeClock()
    monkeypatch.setattr(valkey_module, 'monotonic', clock)
    payload = encode_value({'id': 7})
    result = _OrderedClient([('evalsha', [b'hit', payload, 50])])
    role = valkey_module._RoleClient(
        lambda: cast('valkey_module._RawValkeyClient', result),
        valkey_module._ConnectionGeneration(),
    )
    probes = 0

    def endpoint(_client: valkey_module._RawValkeyClient) -> str:
        nonlocal probes
        probes += 1
        if probes == 2:
            clock._advance(elapsed)
        return 'primary'

    role._endpoint = 'primary'
    role._endpoint_probe = endpoint
    backend = ValkeyBackend(
        role,
        cast('valkey_module.ValkeyMetadataStore', _OrderedClient([])),
        CacheSettings(),
        _RecordingMetrics(),
    )
    hit = backend.lookup('key')
    if elapsed == 0.100:
        assert hit is None
    else:
        assert hit is not None
        assert hit.remaining_seconds == pytest.approx(0.025)
        assert hit.decoded is not None
        assert hit.decoded.value == {'id': 7}
    assert probes == 2
    assert len(result._calls) == 1
    backend.close()


@pytest.mark.parametrize(
    'failure',
    [
        ValkeyConnectionError('credential-bearing-connection'),
        ValkeyTimeoutError('credential-bearing-timeout'),
    ],
)
def test_client_transport_failures_are_redacted_backend_unavailable(
    monkeypatch: pytest.MonkeyPatch,
    failure: Exception,
) -> None:
    backend, _metadata, _result = _scripted_backend(
        monkeypatch,
        result_replies=[('evalsha', failure)],
    )
    sensitive_key = 'credential-' + 'bearing-key'

    with pytest.raises(BackendUnavailable) as captured:
        backend.lookup(sensitive_key)

    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None
    assert 'credential-bearing' not in _exception_text(captured.value)


@pytest.mark.parametrize('reply', [[b'stale'], [b'missing-counter']])
def test_metadata_script_fences_stale_or_partial_incarnation(
    monkeypatch: pytest.MonkeyPatch,
    reply: object,
) -> None:
    replid = 'a' * 40
    backend, _metadata, _result = _scripted_backend(
        monkeypatch,
        metadata_replies=[
            *_validation_replies(replid),
            ('evalsha', reply),
        ],
    )

    with pytest.raises(StaleIncarnation):
        backend.resolve((DependencyTag.site('production'),))


@pytest.mark.parametrize('phase', ['refresh', 'discovery', 'pin'])
@pytest.mark.parametrize(('remaining', 'delay'), [(0.05, 0.1), (0.1, 0.04)])
def test_commit_debits_prepared_role_delay(
    monkeypatch: pytest.MonkeyPatch,
    phase: str,
    remaining: float,
    delay: float,
) -> None:
    clock = _FakeClock()
    monkeypatch.setattr(valkey_module, 'monotonic', clock)
    backend, _metadata, result = _scripted_backend(
        monkeypatch,
        result_replies=[('evalsha', [b'committed'])],
    )

    def replacement_client(**_kwargs: object) -> _OrderedClient:
        return result

    monkeypatch.setattr(valkey_module, 'Valkey', replacement_client)
    role_type = valkey_module._RoleClient
    original_pin = role_type.pin
    original_refresh = role_type.refresh
    original_discovery = role_type.refresh_endpoint

    @contextmanager
    def delayed_pin(role: valkey_module._RoleClient) -> Generator[valkey_module._PinnedRoleClient]:
        with original_pin(role) as pinned:
            clock._advance(delay if phase == 'pin' else 0.0)
            yield pinned

    def delayed_refresh(role: valkey_module._RoleClient) -> None:
        original_refresh(role)
        if phase == 'refresh':
            clock._advance(delay)

    discoveries = 0

    def delayed_discovery(role: valkey_module._RoleClient) -> None:
        nonlocal discoveries
        original_discovery(role)
        discoveries += 1
        if phase == 'discovery' and discoveries == 1:
            clock._advance(delay)

    monkeypatch.setattr(role_type, 'pin', delayed_pin)
    monkeypatch.setattr(role_type, 'refresh', delayed_refresh)
    monkeypatch.setattr(role_type, 'refresh_endpoint', delayed_discovery)

    def recover(_breaker: valkey_module._CircuitBreaker) -> bool:
        return True

    if phase == 'refresh':
        monkeypatch.setattr(valkey_module._CircuitBreaker, 'admit', recover)
    payload = cache_common_module.encode_result(
        cache_common_module.TimedResult(0.0, {'id': 7}),
        0.0,
        remaining,
    )
    lease = FillLease('result-key', 'original-token')
    assert backend.commit(lease, payload, remaining) is (delay < remaining)
    commands = [call for call in result._calls if call[0] == 'evalsha']
    if delay >= remaining:
        assert commands == []
        assert lease == FillLease('result-key', 'original-token')
    else:
        assert len(commands) == 1
        assert commands[0][-2] == payload
        ttl = commands[0][-1]
        assert isinstance(ttl, int)
        assert 0 < ttl <= 60
        hit = valkey_module.CacheHit(payload, ttl / 1000)
        assert cache_common_module.decode_hit(hit, 1.0).origin_elapsed >= 0.04


@pytest.mark.parametrize('load_delay', [0.04, 0.0995, 0.1])
def test_commit_noscript_recovery_spends_original_deadline(
    monkeypatch: pytest.MonkeyPatch,
    load_delay: float,
) -> None:
    clock = _FakeClock()
    monkeypatch.setattr(valkey_module, 'monotonic', clock)

    def load(_call: tuple[object, ...]) -> object:
        clock._advance(load_delay)
        return COMMIT_SCRIPT_SHA

    backend, _metadata, result = _scripted_backend(
        monkeypatch,
        result_replies=[
            ('evalsha', NoScriptError('missing')),
            ('script_load', load),
            ('evalsha', [b'committed']),
        ],
    )
    assert backend.commit(FillLease('key', 'token'), encode_value(7), 0.1) is (load_delay < 0.099)
    calls = [call for call in result._calls if call[0] == 'evalsha']
    assert len(calls) == (2 if load_delay < 0.099 else 1)
    if len(calls) == 2:
        assert calls[0][-1] == 100
        assert 0 < cast('int', calls[1][-1]) <= 60


@pytest.mark.parametrize(
    ('stage', 'error_type', 'expected'),
    [
        ('initial', DataError, BackendProtocolError),
        ('load', DataError, BackendProtocolError),
        ('retry', DataError, BackendProtocolError),
        ('retry', NoScriptError, BackendUnavailable),
        ('initial', ValueError, BackendUnavailable),
    ],
)
def test_commit_redacts_client_exception_context(
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
    error_type: type[Exception],
    expected: type[Exception],
) -> None:
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
    backend, _metadata, _result = _scripted_backend(monkeypatch, result_replies=replies)
    with pytest.raises(expected) as captured:
        backend.commit(FillLease('key', 'token'), b'payload', 1.0)
    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None
    assert 'credential-bearing-client-canary' not in _exception_text(captured.value)
    backend.close()


class _RetryPolicy(Protocol):
    def call_with_retry(
        self,
        operation: Callable[[], object],
        failure: Callable[[Exception], object],
    ) -> object: ...


class _NativePool(Protocol):
    connection_kwargs: dict[str, object]


class _NativeConnection(Protocol):
    _sock: socket.socket | None

    def register_connect_callback(self, callback: Callable[[object], None]) -> None: ...


@contextmanager
def _script_native_sockets(
    monkeypatch: pytest.MonkeyPatch,
    payload: bytes,
) -> Generator[list[socket.socket]]:
    sockets: list[socket.socket] = []
    peers: list[socket.socket] = []

    def connect(_connection: Connection) -> socket.socket:
        connection, peer = socket.socketpair()
        connection.settimeout(1)
        sockets.extend((connection, peer))
        peers.append(peer)
        peer.settimeout(1)
        peer.sendall(
            b'+OK\r\n' * 3 + b'$' + str(len(payload)).encode() + b'\r\n' + payload + b'\r\n'
        )
        return connection

    def primary_address(_pool: object) -> tuple[str, int]:
        return 'primary', 6379

    monkeypatch.setattr(Connection, '_connect', connect)
    monkeypatch.setattr(SentinelConnectionPoolProxy, 'get_master_address', primary_address)
    try:
        yield peers
    finally:
        for item in sockets:
            item.close()


@pytest.mark.parametrize('mode', ['managed', 'sentinel', 'standalone'])
@pytest.mark.parametrize('primary', [True, False])
def test_native_connect_proves_each_ha_socket(
    monkeypatch: pytest.MonkeyPatch,
    mode: config._ValkeyMode,
    primary: bool,
) -> None:
    payload = f'role:{"master" if primary else "slave"}\r\nmaster_replid:{"a" * 40}\r\n'.encode()
    proof_command = b'*2\r\n$4\r\nINFO\r\n$11\r\nreplication\r\n'
    observations: list[int] = []
    tracker = _ConnectionGeneration()
    observer = _ConnectObserver(tracker, observations)
    role = replace(_role_settings(mode).l2.valkey.metadata, tls=False)
    raw = cast(
        'Valkey', valkey_module._create_raw_client(role, CacheSettings(), 'password', tracker)
    )
    connection = raw.connection_pool.make_connection()
    observed = cast('_NativeConnection', connection)
    observed.register_connect_callback(observer.connected)
    try:
        with _script_native_sockets(monkeypatch, payload) as peers:
            if not primary and mode != 'standalone':
                with pytest.raises(PrimaryProofError):
                    connection.connect()
                assert observed._sock is None
                assert observations == []
                assert tracker._generation == 0
                assert peers[-1].recv(8192).endswith(proof_command)
            else:
                for generation in (1, 2):
                    connection.connect()
                    sent = peers[-1].recv(8192)
                    assert sent.count(b'AUTH') == 1
                    assert sent.count(b'SETINFO') == 2
                    assert (
                        sent.endswith(proof_command)
                        if mode != 'standalone'
                        else proof_command not in sent
                    )
                    assert observations == list(range(1, generation + 1))
                    connection.disconnect()
                second = raw.connection_pool.make_connection()
                try:
                    second.connect()
                    sent = peers[-1].recv(8192)
                    assert (
                        sent.endswith(proof_command)
                        if mode != 'standalone'
                        else proof_command not in sent
                    )
                    assert tracker._generation == 3
                finally:
                    second.disconnect()
            if mode == 'sentinel':
                pool = cast('_SentinelPrimaryPool', raw.connection_pool)
                for discovery in pool.sentinel_manager.sentinels:
                    discovery_pool = cast('_NativePool', discovery.connection_pool)
                    assert discovery_pool.connection_kwargs.get('valkey_connect_func') is None
    finally:
        connection.disconnect()
        raw.close()


@pytest.mark.parametrize('mode', ['managed', 'sentinel'])
@pytest.mark.parametrize(
    ('payload', 'error'),
    [
        (b'role:slave\r\nmaster_replid:' + b'a' * 40 + b'\r\n', BackendUnavailable),
        (b'role:master\r\n', BackendProtocolError),
    ],
)
def test_native_connect_proof_is_redacted_at_backend_boundary(
    monkeypatch: pytest.MonkeyPatch,
    mode: config._ValkeyMode,
    payload: bytes,
    error: type[Exception],
) -> None:
    settings = _role_settings(mode)
    role = replace(settings.l2.valkey.metadata, tls=False)
    with _script_native_sockets(monkeypatch, payload):
        with pytest.raises(error) as captured:
            _create_client(role, settings, 'credential-bearing-password')
        assert captured.value.__cause__ is None
        assert captured.value.__context__ is None
        assert 'credential-bearing' not in str(captured.value)


@pytest.mark.parametrize('mode', ['managed', 'sentinel'])
@pytest.mark.parametrize(
    ('payload', 'error', 'message'),
    [
        (
            b'role:slave\r\nmaster_replid:credential-bearing-reply\r\n',
            BackendUnavailable,
            'Valkey metadata role is not writable',
        ),
        (
            b'role:master\r\ncredential-bearing-reply:value\r\n',
            BackendProtocolError,
            'Valkey replication information is malformed',
        ),
    ],
)
def test_native_refresh_preserves_redacted_proof_failure(
    monkeypatch: pytest.MonkeyPatch,
    mode: config._ValkeyMode,
    payload: bytes,
    error: type[Exception],
    message: str,
) -> None:
    settings = _role_settings(mode)
    configured = replace(settings.l2.valkey.metadata, tls=False)
    with _script_native_sockets(monkeypatch, b'role:master\r\nmaster_replid:primary\r\n'):
        role = _create_client(configured, settings, 'credential-bearing-password')
        try:
            with _script_native_sockets(monkeypatch, payload):
                with pytest.raises(error) as captured:
                    role.refresh()
                assert str(captured.value) == message
                assert captured.value.__cause__ is None
                assert captured.value.__context__ is None
                assert 'credential-bearing' not in _exception_text(captured.value)
        finally:
            role.close()


@pytest.mark.parametrize(
    ('error', 'expected'),
    [(ValueError, BackendUnavailable), (DataError, BackendProtocolError)],
)
def test_refresh_redacts_raw_factory_errors(
    monkeypatch: pytest.MonkeyPatch,
    error: type[Exception],
    expected: type[Exception],
) -> None:
    client = ConnectedAtomicControlClient()
    role = _RoleClient(lambda: client, _ConnectionGeneration())

    def failing_factory() -> ConnectedAtomicControlClient:
        raise error('credential-bearing-factory-error')

    monkeypatch.setattr(role, '_factory', failing_factory)
    try:
        with pytest.raises(expected) as captured:
            role.refresh()
        assert captured.value.__cause__ is None
        assert captured.value.__context__ is None
        assert 'credential-bearing' not in _exception_text(captured.value)
    finally:
        role.close()


def test_sentinel_mode_uses_discovery_for_each_role(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel_calls: list[tuple[tuple[tuple[str, int], ...], dict[str, object]]] = []
    services: list[str] = []

    class _FakePool:
        def get_master_address(self) -> tuple[str, int]:
            return 'primary.example', 6379

    class _SentinelClient(AtomicControlClient):
        connection_pool = _FakePool()

    class _FakeSentinel:
        def __init__(self, sentinels: tuple[tuple[str, int], ...], **kwargs: object) -> None:
            sentinel_calls.append((sentinels, kwargs))

        def master_for(self, service_name: str) -> _SentinelClient:
            services.append(service_name)
            return _SentinelClient()

    monkeypatch.setattr(valkey_module, 'Sentinel', _FakeSentinel)
    settings = _role_settings('sentinel')
    settings = replace(
        settings,
        l2=replace(
            settings.l2,
            valkey=replace(
                settings.l2.valkey,
                metadata=replace(settings.l2.valkey.metadata, database=7),
                result=replace(settings.l2.valkey.result, database=9),
            ),
        ),
    )
    backend = ValkeyBackend.from_settings(
        settings,
        {'METADATA_PASSWORD': 'metadata-secret', 'RESULT_PASSWORD': 'result-secret'},
        _RecordingMetrics(),
    )
    backend.close()

    assert services == ['metadata-primary', 'result-primary']
    assert [call[1]['db'] for call in sentinel_calls] == [7, 9]
    assert all(
        'db' not in cast('dict[str, object]', call[1]['sentinel_kwargs'])
        for call in sentinel_calls
    )
    assert sentinel_calls[0][0] == (
        ('sentinel-a', 26379),
        ('sentinel-b', 26379),
        ('sentinel-c', 26379),
    )
    assert all(call[1]['min_other_sentinels'] == 2 for call in sentinel_calls)
    assert all('retry' in call[1] for call in sentinel_calls)
    assert all('valkey_connect_func' in call[1] for call in sentinel_calls)
    assert all(
        'retry' in cast('dict[str, object]', call[1]['sentinel_kwargs']) for call in sentinel_calls
    )
    assert all(
        'valkey_connect_func' not in cast('dict[str, object]', call[1]['sentinel_kwargs'])
        for call in sentinel_calls
    )
