from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from typing import cast

import pytest
from valkey.exceptions import ConnectionError as ValkeyConnectionError
from valkey.exceptions import TimeoutError as ValkeyTimeoutError

import sg_query_cache._cache as cache_module
import sg_query_cache.valkey as valkey_module
from sg_query_cache import daemon_plugin
from sg_query_cache._backend import (
    BackendUnavailable,
    CacheBackend,
)
from sg_query_cache._cache import QueryCache
from sg_query_cache._client import CachedShotgun, ClientContext
from sg_query_cache._codec import CodecError
from sg_query_cache._config import CacheSettings, L1Settings, L2Settings, LimitSettings
from sg_query_cache._keys import KeyBuilder
from sg_query_cache._local import LocalBackend
from sg_query_cache._model import (
    CacheHit,
    DependencyPlan,
    DependencyTag,
    ExactCall,
    FillLease,
    GenerationChange,
    GenerationVector,
    Identity,
)
from sg_query_cache._valkey_scripts import (
    COMMIT_SCRIPT_SHA,
    CONTROL_CREATE_SCRIPT_SHA,
    LOOKUP_SCRIPT_SHA,
    RESOLVE_SCRIPT_SHA,
)
from sg_query_cache.valkey import (
    ValkeyBackend,
    ValkeyMetadataStore,
    _CircuitBreaker,
    _ConnectionGeneration,
    _RawValkeyClient,
    _RoleClient,
    _ValkeyClient,
)
from tests._fakes import CLIENT_OPTIONS, FakeClock, RecordingMetrics, asset_call, asset_plan
from tests.valkey._control_helpers import AtomicControlClient, ProvingClient, enabled_settings

_SITE = 'production'


@dataclass(frozen=True, slots=True)
class _EventContext:
    event: Mapping[str, object]


class _CapacityControlClient(AtomicControlClient):
    def evalsha(self, sha: str, numkeys: int, *keys_and_args: object) -> object:
        if sha == RESOLVE_SCRIPT_SHA:
            return [b'capacity']
        assert sha == CONTROL_CREATE_SCRIPT_SHA
        return super().evalsha(sha, numkeys, *keys_and_args)


class _FailingResultClient:
    def __init__(self, failures: list[Exception]) -> None:
        self.evalsha_calls = 0
        self._failures = failures
        self.refresh_calls = 0

    def close(self) -> None:
        return None

    def evalsha(self, sha: str, numkeys: int, *keys_and_args: object) -> object:
        assert sha == LOOKUP_SCRIPT_SHA
        assert numkeys == 1
        _ = keys_and_args
        self.evalsha_calls += 1
        raise self._failures.pop(0)

    def refresh(self) -> None:
        self.refresh_calls += 1


class _MalformedPayloadBackend:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def change(self, changes: tuple[GenerationChange, ...]) -> None:
        _ = changes

    def close(self) -> None:
        return None

    def commit(self, lease: FillLease, payload: bytes, remaining_seconds: float) -> bool:
        _ = lease, payload, remaining_seconds
        return False

    def flush_site(self, site_namespace: str, reason: str) -> None:
        _ = site_namespace, reason

    def lookup(self, key: str) -> CacheHit | None:
        _ = key
        return CacheHit(self._payload, 30)

    def release(self, lease: FillLease) -> bool:
        _ = lease
        return False

    def reserve(self, key: str, lease_seconds: float) -> FillLease | None:
        _ = key, lease_seconds
        return None

    def resolve(self, tags: tuple[DependencyTag, ...]) -> GenerationVector:
        return GenerationVector('1' * 32, tuple((tag, 1) for tag in tags))


class _SchemaUnavailableClient:
    def __init__(self) -> None:
        self._find_calls = 0

    def find(
        self,
        entity_type: str,
        *_args: object,
        **_kwargs: object,
    ) -> list[dict[str, object]]:
        self._find_calls += 1
        return [{'id': self._find_calls, 'type': entity_type}]

    def schema_read(self, project_entity: Mapping[str, object] | None = None) -> object:
        _ = project_entity
        raise BackendUnavailable('schema unavailable')


class _StaticMetadata:
    def change(self, changes: tuple[GenerationChange, ...]) -> None:
        _ = changes

    def close(self) -> None:
        return None

    def flush_site(self, site_namespace: str, reason: str) -> None:
        _ = site_namespace, reason

    def resolve(self, tags: tuple[DependencyTag, ...]) -> GenerationVector:
        return GenerationVector('1' * 32, tuple((tag, 1) for tag in tags))


class _TimeoutRawClient:
    def __init__(self, command_failure: Exception | None = None) -> None:
        self.close_calls = 0
        self._command_failure = command_failure
        self.evalsha_calls = 0

    def client(self) -> '_TimeoutRawClient':
        return self

    def close(self) -> None:
        self.close_calls += 1

    def evalsha(self, sha: str, numkeys: int, *keys_and_args: object) -> object:
        assert sha == LOOKUP_SCRIPT_SHA
        assert numkeys == 1
        _ = keys_and_args
        self.evalsha_calls += 1
        if self._command_failure is not None:
            raise self._command_failure
        return [b'miss']

    def exists(self, key: str) -> object:
        raise AssertionError(key)

    def get(self, key: str) -> object:
        raise AssertionError(key)

    def info(self, section: str) -> object:
        raise AssertionError(section)

    def script_load(self, script: bytes | str) -> object:
        raise AssertionError(script)


def _cache(backend: CacheBackend, settings: CacheSettings | None = None) -> QueryCache:
    selected = settings or CacheSettings()
    return QueryCache(
        selected,
        Identity('script', 'studio', 'render'),
        key_builder=KeyBuilder(),
        l1=backend,
    )


def _read(cache: QueryCache, fetch: Callable[[], object], call: ExactCall | None = None) -> object:
    return cache.read(call or asset_call(), asset_plan(), fetch, CLIENT_OPTIONS)


def _assert_single_direct_sg_result(backend: CacheBackend, settings: CacheSettings) -> None:
    cache = QueryCache(
        settings,
        Identity('script', 'studio', 'render'),
        key_builder=KeyBuilder(),
        l2=backend,
    )
    calls = 0

    def fetch() -> list[dict[str, int]]:
        nonlocal calls
        calls += 1
        return [{'call': calls}]

    assert _read(cache, fetch) == [{'call': 1}]
    assert calls == 1


def _shared_settings(*, circuit_failure_threshold: int = 5) -> CacheSettings:
    return CacheSettings(
        l1=L1Settings(enabled=False),
        l2=L2Settings(enabled=True, acknowledge_mutually_trusted_clients=True),
        limits=LimitSettings(circuit_failure_threshold=circuit_failure_threshold),
    )


def test_codec_refusal_preserves_direct_sg_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = LocalBackend(CacheSettings(), FakeClock(), RecordingMetrics())
    cache = _cache(backend)
    calls = 0

    def refuse(_value: object, _origin_elapsed: float, _lifetime: float) -> bytes:
        raise CodecError('refused')

    def fetch() -> object:
        nonlocal calls
        calls += 1
        return [{'call': calls}]

    monkeypatch.setattr(cache_module, 'encode_result', refuse)
    assert _read(cache, fetch) == [{'call': 1}]
    assert _read(cache, fetch) == [{'call': 2}]


@pytest.mark.parametrize('failure_mode', ['closed-backend', 'stopped-daemon'])
def test_failure_mode_cannot_extend_cached_value_past_maximum_age(
    failure_mode: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = FakeClock()
    backend = LocalBackend(CacheSettings(), clock, RecordingMetrics())
    cache = _cache(backend)
    calls = 0

    def fetch() -> object:
        nonlocal calls
        calls += 1
        return [{'call': calls}]

    monkeypatch.setattr(daemon_plugin._State, 'backend', backend)
    monkeypatch.setattr(daemon_plugin._State, 'site_namespace', _SITE)
    assert cache.read(asset_call(), asset_plan(), fetch, CLIENT_OPTIONS, per_call_age=1) == [
        {'call': 1}
    ]
    if failure_mode == 'closed-backend':
        backend.close()
    else:
        monkeypatch.setattr(daemon_plugin._State, 'backend', None)
        monkeypatch.setattr(daemon_plugin._State, 'site_namespace', None)
    clock.advance(2)
    assert cache.read(asset_call(), asset_plan(), fetch, CLIENT_OPTIONS, per_call_age=1) == [
        {'call': 2}
    ]


def test_malformed_cache_data_falls_back_to_direct_sg_without_population() -> None:
    cache = _cache(_MalformedPayloadBackend(b'not-a-closed-codec-value'))
    calls = 0

    def fetch() -> object:
        nonlocal calls
        calls += 1
        return [{'call': calls}]

    assert _read(cache, fetch) == [{'call': 1}]
    assert _read(cache, fetch) == [{'call': 2}]


def test_real_local_metadata_capacity_preserves_direct_sg_access() -> None:
    probe_metrics = RecordingMetrics()
    LocalBackend(CacheSettings(), FakeClock(), probe_metrics)
    permanent_bytes = next(
        int(value)
        for name, value, labels in probe_metrics.observations
        if name == 'generation_metadata_bytes' and labels == {'tag_kind': 'global'}
    )
    settings = CacheSettings(l1=L1Settings(metadata_budget_bytes=permanent_bytes))
    backend = LocalBackend(settings, FakeClock(), RecordingMetrics())
    cache = _cache(backend, settings)
    scoped = DependencyPlan(
        'event_aware',
        (DependencyTag.entity(_SITE, 'Asset', 7),),
        'event_aware',
    )

    assert cache.read(
        asset_call(),
        scoped,
        lambda: [{'source': 'sg'}],
        CLIENT_OPTIONS,
    ) == [{'source': 'sg'}]


def test_real_open_breaker_preserves_direct_sg_access_without_more_valkey_io() -> None:
    settings = _shared_settings(circuit_failure_threshold=1)
    client = _FailingResultClient([ValkeyConnectionError('connection failed')])
    backend = ValkeyBackend(
        cast('_ValkeyClient', client),
        _StaticMetadata(),
        settings,
        RecordingMetrics(),
    )
    cache = QueryCache(
        settings,
        Identity('script', 'studio', 'render'),
        key_builder=KeyBuilder(),
        l2=backend,
    )
    calls = 0

    def fetch() -> list[dict[str, int]]:
        nonlocal calls
        calls += 1
        return [{'call': calls}]

    assert _read(cache, fetch) == [{'call': 1}]
    assert _read(cache, fetch) == [{'call': 2}]
    assert client.evalsha_calls == 1


def test_real_valkey_metadata_capacity_preserves_direct_sg_access() -> None:
    settings = _shared_settings()
    control_client = _CapacityControlClient()
    metadata = ValkeyMetadataStore(control_client, settings, RecordingMetrics())
    result_client = _FailingResultClient([])
    backend = ValkeyBackend(
        cast('_ValkeyClient', result_client),
        metadata,
        settings,
        RecordingMetrics(),
    )
    cache = QueryCache(
        settings,
        Identity('script', 'studio', 'render'),
        key_builder=KeyBuilder(),
        l2=backend,
    )

    assert _read(cache, lambda: [{'source': 'sg'}]) == [{'source': 'sg'}]
    assert result_client.evalsha_calls == 0


def test_result_eviction_never_serves_an_evicted_value() -> None:
    settings = CacheSettings(l1=L1Settings(result_budget_bytes=350))
    backend = LocalBackend(settings, FakeClock(), RecordingMetrics())
    cache = _cache(backend, settings)
    calls = {'Asset': 0, 'Shot': 0}

    def fetch(entity_type: str) -> list[dict[str, object]]:
        calls[entity_type] += 1
        return [{'type': entity_type, 'call': calls[entity_type]}]

    assert _read(cache, lambda: fetch('Asset'), asset_call('Asset')) == [
        {'type': 'Asset', 'call': 1}
    ]
    assert _read(cache, lambda: fetch('Shot'), asset_call('Shot')) == [{'type': 'Shot', 'call': 1}]
    assert _read(cache, lambda: fetch('Asset'), asset_call('Asset')) == [
        {'type': 'Asset', 'call': 2}
    ]


def test_skipped_callback_without_setup_flush_is_not_reported_as_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = LocalBackend(CacheSettings(), FakeClock(), RecordingMetrics())
    cache = _cache(backend)
    calls = 0

    def fetch() -> list[dict[str, int]]:
        nonlocal calls
        calls += 1
        return [{'call': calls}]

    assert _read(cache, fetch) == [{'call': 1}]
    monkeypatch.setattr(daemon_plugin._State, 'backend', None)
    monkeypatch.setattr(daemon_plugin._State, 'site_namespace', None)

    with pytest.raises(RuntimeError, match='setup is incomplete'):
        daemon_plugin._callback(_EventContext({}))
    assert _read(cache, fetch) == [{'call': 1}]

    monkeypatch.setattr(daemon_plugin._State, 'backend', backend)
    monkeypatch.setattr(daemon_plugin._State, 'site_namespace', _SITE)
    backend.flush_site(_SITE, 'daemon_recovery')
    assert _read(cache, fetch) == [{'call': 2}]


def test_sync_backend_recovers_after_construction_during_outage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = enabled_settings('managed')
    settings = replace(
        settings,
        l2=replace(settings.l2, acknowledge_mutually_trusted_clients=True),
        limits=replace(settings.limits, circuit_failure_threshold=1),
    )
    clock = FakeClock()
    outage = True
    factory_calls: list[str] = []
    clients: dict[str, ProvingClient] = {}
    payloads: list[bytes] = []

    class _ResultClient(ProvingClient):
        def evalsha(self, sha: str, numkeys: int, *args: object) -> object:
            if sha == COMMIT_SCRIPT_SHA:
                payloads.append(next(value for value in args if isinstance(value, bytes)))
            return super().evalsha(sha, numkeys, *args)

    def create_client(**kwargs: object) -> ProvingClient:
        host = cast('str', kwargs['host'])
        factory_calls.append(host)
        if outage:
            raise ValkeyConnectionError('outage')
        client_type = ProvingClient if host == 'metadata.example' else _ResultClient
        client = client_type(cast('Callable[[object], None]', kwargs['valkey_connect_func']))
        if host == 'metadata.example':
            client.metadata_replies.extend([[b'ok', 1, 1, 1]] * 6)
        else:
            client.metadata_replies.extend(([b'miss'], [b'reserved'], [b'miss'], [b'committed']))
        clients[host] = client
        return client

    monkeypatch.setattr(valkey_module, 'Valkey', create_client)
    backend = ValkeyBackend.from_settings(
        settings,
        {'METADATA_PASSWORD': 'secret', 'RESULT_PASSWORD': 'secret'},
        RecordingMetrics(),
    )
    metadata = cast('ValkeyMetadataStore', backend._metadata)
    metadata._breaker = _CircuitBreaker(
        'metadata', threshold=1, probe_seconds=10, metrics=RecordingMetrics(), clock=clock
    )
    assert factory_calls == []
    cache = QueryCache(
        settings, Identity('script', 'studio', 'render'), key_builder=KeyBuilder(), l2=backend
    )
    calls = 0

    def fetch() -> list[dict[str, int]]:
        nonlocal calls
        calls += 1
        return [{'call': calls}]

    assert _read(cache, fetch) == [{'call': 1}]
    assert factory_calls == ['metadata.example']
    assert _read(cache, fetch) == [{'call': 2}]
    assert factory_calls == ['metadata.example']
    outage = False
    clock.advance(10)
    assert _read(cache, fetch) == [{'call': 3}]
    result = clients['result.example']
    # Reuse the actual committed envelope as the result server's next hit.
    assert result.metadata_replies == []
    result.metadata_replies.append([b'hit', payloads[0], 30000])
    assert _read(cache, fetch) == [{'call': 3}]
    assert calls == 3
    assert factory_calls == ['metadata.example', 'metadata.example', 'result.example']
    cache.close()


def test_unavailable_schema_bypasses_cache_and_keeps_find_available() -> None:
    settings = CacheSettings(site_namespace='studio')
    backend = LocalBackend(settings, FakeClock(), RecordingMetrics())
    cache = QueryCache(
        settings,
        Identity('script', 'studio', 'render'),
        key_builder=KeyBuilder(),
        l1=backend,
    )
    client = _SchemaUnavailableClient()
    wrapper = CachedShotgun(
        client,
        cache,
        ClientContext(
            site_url='https://example.shotgrid.autodesk.com',
            site_namespace='studio',
            localized=False,
            convert_datetimes_to_utc=True,
            records_per_page=500,
        ),
    )

    assert wrapper.find('Asset', [], ['id']) == [{'id': 1, 'type': 'Asset'}]
    assert wrapper.find('Asset', [], ['id']) == [{'id': 2, 'type': 'Asset'}]


def test_valkey_command_operation_timeout_preserves_direct_sg_without_retry() -> None:
    settings = _shared_settings()
    client = _FailingResultClient([ValkeyTimeoutError('command operation timed out')])
    backend = ValkeyBackend(
        cast('_ValkeyClient', client),
        _StaticMetadata(),
        settings,
        RecordingMetrics(),
    )

    _assert_single_direct_sg_result(backend, settings)

    assert client.refresh_calls == 0
    assert client.evalsha_calls == 1


def test_valkey_connection_establishment_timeout_preserves_direct_sg_without_retry() -> None:
    settings = _shared_settings()
    raw = _TimeoutRawClient()
    factory_calls = 0

    def factory() -> _RawValkeyClient:
        nonlocal factory_calls
        factory_calls += 1
        if factory_calls == 1:
            return raw
        raise ValkeyTimeoutError('connection establishment timed out')

    client = _RoleClient(factory, _ConnectionGeneration())
    client.refresh()
    backend = ValkeyBackend(
        client,
        _StaticMetadata(),
        settings,
        RecordingMetrics(),
    )
    clock = FakeClock()
    breaker = _CircuitBreaker(
        'result',
        threshold=1,
        probe_seconds=1,
        metrics=RecordingMetrics(),
        clock=clock,
    )
    breaker.fail()
    clock.advance(1)
    backend._result_breaker = breaker

    _assert_single_direct_sg_result(backend, settings)

    assert factory_calls == 2
    assert raw.evalsha_calls == 0
    assert raw.close_calls == 1
