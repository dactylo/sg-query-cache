from collections.abc import Callable, Generator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TypeVar, cast

import pytest
from valkey.exceptions import ConnectionError as ValkeyConnectionError
from valkey.exceptions import TimeoutError as ValkeyTimeoutError

import sg_query_cache._cache as cache_module
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
    CONTROL_CREATE_SCRIPT_SHA,
    LOOKUP_SCRIPT_SHA,
    RESOLVE_SCRIPT_SHA,
    ScriptClient,
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
from tests._fakes import _FakeClock, _RecordingMetrics

_T = TypeVar('_T')

_CLIENT_OPTIONS = {'localized': False, 'convert_datetimes_to_utc': True}
_SITE = 'production'


@dataclass(frozen=True, slots=True)
class _EventContext:
    event: Mapping[str, object]


class _CapacityControlClient:
    def __init__(self) -> None:
        self._control: bytes | None = None
        self._counter_key: str | None = None

    def close(self) -> None:
        return None

    def epoch(self) -> tuple[int, int, None]:
        return 0, 0, None

    def evalsha(self, sha: str, numkeys: int, *keys_and_args: object) -> object:
        if sha == RESOLVE_SCRIPT_SHA:
            return [b'capacity']
        assert sha == CONTROL_CREATE_SCRIPT_SHA
        assert numkeys == 2
        _control_key, counter_key, record = cast('tuple[str, str, bytes]', keys_and_args)
        self._counter_key = counter_key
        if self._control is None:
            self._control = record
            return [b'created', record]
        return [b'winner', self._control]

    def exists(self, key: str) -> object:
        return int(key == self._counter_key)

    def get(self, key: str) -> object:
        _ = key
        return self._control

    def info(self, section: str) -> object:
        assert section == 'replication'
        return {'role': 'master', 'master_replid': 'a' * 40}

    @contextmanager
    def pin(self) -> Generator['_CapacityControlClient']:
        yield self

    def refresh(self) -> None:
        return None

    def refresh_endpoint(self) -> None:
        return None

    def run_operation(self, operation: Callable[[ScriptClient], _T]) -> _T:
        return operation(self)

    def script_load(self, script: bytes | str) -> object:
        raise AssertionError(script)


class _FailingResultClient:
    def __init__(self, failures: list[Exception]) -> None:
        self._evalsha_calls = 0
        self._failures = failures
        self._refresh_calls = 0

    def close(self) -> None:
        return None

    def evalsha(self, sha: str, numkeys: int, *keys_and_args: object) -> object:
        assert sha == LOOKUP_SCRIPT_SHA
        assert numkeys == 1
        _ = keys_and_args
        self._evalsha_calls += 1
        raise self._failures.pop(0)

    def refresh(self) -> None:
        self._refresh_calls += 1


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

    def find(  # noqa: PLR0913, PLR0917 - official-compatible test boundary
        self,
        entity_type: str,
        filters: object,
        fields: list[str] | None = None,
        order: list[dict[str, str]] | None = None,
        filter_operator: str | None = None,
        limit: int = 0,
        retired_only: bool = False,
        page: int = 0,
        include_archived_projects: bool = True,
        additional_filter_presets: object = None,
    ) -> list[dict[str, object]]:
        _ = (
            filters,
            fields,
            order,
            filter_operator,
            limit,
            retired_only,
            page,
            include_archived_projects,
            additional_filter_presets,
        )
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
        self._close_calls = 0
        self._command_failure = command_failure
        self._evalsha_calls = 0

    def client(self) -> '_TimeoutRawClient':
        return self

    def close(self) -> None:
        self._close_calls += 1

    def evalsha(self, sha: str, numkeys: int, *keys_and_args: object) -> object:
        assert sha == LOOKUP_SCRIPT_SHA
        assert numkeys == 1
        _ = keys_and_args
        self._evalsha_calls += 1
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


def _call(entity_type: str = 'Asset') -> ExactCall:
    return ExactCall('find', (('entity_type', entity_type), ('filters', ())))


def _cache(backend: CacheBackend, settings: CacheSettings | None = None) -> QueryCache:
    selected = settings or CacheSettings()
    return QueryCache(
        selected,
        Identity('script', 'studio', 'render'),
        KeyBuilder(),
        l1=backend,
    )


def _plan() -> DependencyPlan:
    return DependencyPlan(
        'event_aware',
        (
            DependencyTag.site(_SITE),
            DependencyTag.authorization(_SITE),
            DependencyTag.type_membership(_SITE, 'Asset'),
        ),
        'event_aware',
    )


def _read(cache: QueryCache, fetch: Callable[[], object], call: ExactCall | None = None) -> object:
    return cache.read(call or _call(), _plan(), fetch, _CLIENT_OPTIONS)


def _assert_single_direct_sg_result(backend: CacheBackend, settings: CacheSettings) -> None:
    cache = QueryCache(
        settings,
        Identity('script', 'studio', 'render'),
        KeyBuilder(),
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
    backend = LocalBackend(CacheSettings(), _FakeClock(), _RecordingMetrics())
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
    clock = _FakeClock()
    backend = LocalBackend(CacheSettings(), clock, _RecordingMetrics())
    cache = _cache(backend)
    calls = 0

    def fetch() -> object:
        nonlocal calls
        calls += 1
        return [{'call': calls}]

    monkeypatch.setattr(daemon_plugin._State, '_backend', backend)
    monkeypatch.setattr(daemon_plugin._State, '_site_namespace', _SITE)
    assert cache.read(_call(), _plan(), fetch, _CLIENT_OPTIONS, per_call_age=1) == [{'call': 1}]
    if failure_mode == 'closed-backend':
        backend.close()
    else:
        monkeypatch.setattr(daemon_plugin._State, '_backend', None)
        monkeypatch.setattr(daemon_plugin._State, '_site_namespace', None)
    clock._advance(2)
    assert cache.read(_call(), _plan(), fetch, _CLIENT_OPTIONS, per_call_age=1) == [{'call': 2}]


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
    probe_metrics = _RecordingMetrics()
    LocalBackend(CacheSettings(), _FakeClock(), probe_metrics)
    permanent_bytes = next(
        int(value)
        for name, value, labels in probe_metrics._observations
        if name == 'generation_metadata_bytes' and labels == {'tag_kind': 'global'}
    )
    settings = CacheSettings(l1=L1Settings(metadata_budget_bytes=permanent_bytes))
    backend = LocalBackend(settings, _FakeClock(), _RecordingMetrics())
    cache = _cache(backend, settings)
    scoped = DependencyPlan(
        'event_aware',
        (DependencyTag.entity(_SITE, 'Asset', 7),),
        'event_aware',
    )

    assert cache.read(
        _call(),
        scoped,
        lambda: [{'source': 'sg'}],
        _CLIENT_OPTIONS,
    ) == [{'source': 'sg'}]


def test_real_open_breaker_preserves_direct_sg_access_without_more_valkey_io() -> None:
    settings = _shared_settings(circuit_failure_threshold=1)
    client = _FailingResultClient([ValkeyConnectionError('connection failed')])
    backend = ValkeyBackend(
        cast('_ValkeyClient', client),
        _StaticMetadata(),
        settings,
        _RecordingMetrics(),
    )
    cache = QueryCache(
        settings,
        Identity('script', 'studio', 'render'),
        KeyBuilder(),
        l2=backend,
    )
    calls = 0

    def fetch() -> list[dict[str, int]]:
        nonlocal calls
        calls += 1
        return [{'call': calls}]

    assert _read(cache, fetch) == [{'call': 1}]
    assert _read(cache, fetch) == [{'call': 2}]
    assert client._evalsha_calls == 1


def test_real_valkey_metadata_capacity_preserves_direct_sg_access() -> None:
    settings = _shared_settings()
    control_client = _CapacityControlClient()
    metadata = ValkeyMetadataStore(control_client, settings, _RecordingMetrics())
    result_client = _FailingResultClient([])
    backend = ValkeyBackend(
        cast('_ValkeyClient', result_client),
        metadata,
        settings,
        _RecordingMetrics(),
    )
    cache = QueryCache(
        settings,
        Identity('script', 'studio', 'render'),
        KeyBuilder(),
        l2=backend,
    )

    assert _read(cache, lambda: [{'source': 'sg'}]) == [{'source': 'sg'}]
    assert result_client._evalsha_calls == 0


def test_result_eviction_never_serves_an_evicted_value() -> None:
    settings = CacheSettings(l1=L1Settings(result_budget_bytes=350))
    backend = LocalBackend(settings, _FakeClock(), _RecordingMetrics())
    cache = _cache(backend, settings)
    calls = {'Asset': 0, 'Shot': 0}

    def fetch(entity_type: str) -> list[dict[str, object]]:
        calls[entity_type] += 1
        return [{'type': entity_type, 'call': calls[entity_type]}]

    assert _read(cache, lambda: fetch('Asset'), _call('Asset')) == [{'type': 'Asset', 'call': 1}]
    assert _read(cache, lambda: fetch('Shot'), _call('Shot')) == [{'type': 'Shot', 'call': 1}]
    assert _read(cache, lambda: fetch('Asset'), _call('Asset')) == [{'type': 'Asset', 'call': 2}]


def test_skipped_callback_without_setup_flush_is_not_reported_as_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = LocalBackend(CacheSettings(), _FakeClock(), _RecordingMetrics())
    cache = _cache(backend)
    calls = 0

    def fetch() -> list[dict[str, int]]:
        nonlocal calls
        calls += 1
        return [{'call': calls}]

    assert _read(cache, fetch) == [{'call': 1}]
    monkeypatch.setattr(daemon_plugin._State, '_backend', None)
    monkeypatch.setattr(daemon_plugin._State, '_site_namespace', None)

    with pytest.raises(RuntimeError, match='setup is incomplete'):
        daemon_plugin.callback(_EventContext({}))
    assert _read(cache, fetch) == [{'call': 1}]

    monkeypatch.setattr(daemon_plugin._State, '_backend', backend)
    monkeypatch.setattr(daemon_plugin._State, '_site_namespace', _SITE)
    backend.flush_site(_SITE, 'daemon_recovery')
    assert _read(cache, fetch) == [{'call': 2}]


def test_unavailable_schema_bypasses_cache_and_keeps_find_available() -> None:
    settings = CacheSettings(site_namespace='studio')
    backend = LocalBackend(settings, _FakeClock(), _RecordingMetrics())
    cache = QueryCache(
        settings,
        Identity('script', 'studio', 'render'),
        KeyBuilder(),
        l1=backend,
    )
    client = _SchemaUnavailableClient()
    wrapper = CachedShotgun(
        client,
        cache,
        ClientContext(
            'https://example.shotgrid.autodesk.com',
            'studio',
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
        _RecordingMetrics(),
    )

    _assert_single_direct_sg_result(backend, settings)

    assert client._refresh_calls == 0
    assert client._evalsha_calls == 1


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
    backend = ValkeyBackend(
        client,
        _StaticMetadata(),
        settings,
        _RecordingMetrics(),
    )
    clock = _FakeClock()
    breaker = _CircuitBreaker(
        'result',
        threshold=1,
        probe_seconds=1,
        metrics=_RecordingMetrics(),
        clock=clock,
    )
    breaker.fail()
    clock._advance(1)
    backend._result_breaker = breaker

    _assert_single_direct_sg_result(backend, settings)

    assert factory_calls == 2
    assert raw._evalsha_calls == 0
    assert raw._close_calls == 1
