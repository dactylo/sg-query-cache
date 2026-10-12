import importlib
import json
import logging
import os
import sys
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import ClassVar, Never, Protocol, cast

import pytest

from sg_query_cache import daemon_plugin
from sg_query_cache._backend import BackendUnavailable, StaleIncarnation
from sg_query_cache._model import DependencyTag, GenerationChange

_FIXTURE_PATH = Path(__file__).parent / 'fixtures' / 'events.json'
_METADATA_SECRET_ENV = 'SG_QUERY_CACHE_METADATA_' + 'PASSWORD'
_PLUGIN_PATH = Path(daemon_plugin.__file__)
_SITE = 'testing'


class _CallbackResult(Protocol):
    delivery_attempt: int
    failure: object | None


class _DriveResult(Protocol):
    callbacks: tuple[_CallbackResult, ...]
    ok: bool


class _HarnessSettingsFactory(Protocol):
    def __call__(
        self,
        *,
        site: str,
        config: Mapping[str, object],
        secrets: Mapping[str, str],
    ) -> object: ...


class _LoadedPlugin(Protocol):
    callbacks: tuple[str, ...]


class _DaemonTesting(Protocol):
    HarnessSettings: _HarnessSettingsFactory

    def drive_sync_plugin(
        self,
        plugin: object,
        events: object,
        *,
        delivery_attempt: int = 1,
        is_operator_replay: bool = False,
    ) -> _DriveResult: ...

    def load_plugin(
        self,
        path: str | Path,
        *,
        plugin_id: str,
        execution: str = 'sync',
        settings: object | None = None,
    ) -> _LoadedPlugin: ...


class _RecordingRegistrar:
    def __init__(self, config_path: str) -> None:
        self._calls: list[tuple[str, object]] = []
        self.config = _TrackingConfig(config_path)

    def add_callback(self, **values: object) -> None:
        self._calls.append(('callback', values))

    def set_setup(self, setup: object) -> None:
        self._calls.append(('setup', setup))


class _RecordingStore:
    _instances: ClassVar[list['_RecordingStore']] = []
    _next_failure: ClassVar[BaseException | None] = None
    _timeline: ClassVar[list[tuple[int, str, int | None]]] = []

    def __init__(self) -> None:
        self._identifier = len(type(self)._instances)
        self._operations: list[tuple[str, object]] = []
        self._failure = type(self)._next_failure
        type(self)._next_failure = None
        type(self)._instances.append(self)

    def _record(self, operation: str, payload: object) -> None:
        self._operations.append((operation, payload))
        active = daemon_plugin._State._backend
        active_identifier = active._identifier if isinstance(active, _RecordingStore) else None
        type(self)._timeline.append((self._identifier, operation, active_identifier))

    def change(self, changes: tuple[GenerationChange, ...]) -> None:
        self._record('change', changes)
        if self._failure is not None:
            raise self._failure

    def close(self) -> None:
        self._record('close', None)

    def flush_site(self, site_namespace: str, reason: str) -> None:
        self._record('flush_site', (site_namespace, reason))

    @classmethod
    def from_settings(
        cls,
        settings: object,
        environ: Mapping[str, str],
        metrics: object,
    ) -> '_RecordingStore':
        _ = settings, metrics
        assert environ == {_METADATA_SECRET_ENV: 'metadata-secret'}
        return cls()

    def resolve(self, tags: tuple[DependencyTag, ...]) -> object:
        self._record('resolve', tags)
        return object()


class _RevealedSecret:
    def __init__(self, value: str, reveals: list[str], name: str) -> None:
        self._name = name
        self._reveals = reveals
        self._value = value

    def reveal(self) -> str:
        self._reveals.append(self._name)
        return self._value


class _TrackingConfig(Mapping[str, object]):
    def __init__(self, config_path: str) -> None:
        self._accessed: list[str] = []
        self._values = {'config_path': config_path, 'unused': 'must-not-be-read'}

    def __getitem__(self, key: str) -> object:
        self._accessed.append(key)
        return self._values[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)


@dataclass
class _PluginContext:
    event: Mapping[str, object]
    _reveals: list[str]
    secrets: Mapping[str, _RevealedSecret]
    site: str


def _fake_harness_call(*_args: object, **_values: object) -> object:
    return object()


def _fake_harness_settings(**_values: object) -> object:
    return object()


def _install_recording_valkey(monkeypatch: pytest.MonkeyPatch) -> None:
    module = ModuleType('sg_query_cache.valkey')
    vars(module)['ValkeyMetadataStore'] = _RecordingStore
    monkeypatch.setitem(sys.modules, 'sg_query_cache.valkey', module)


def _integration_unavailable(required: bool, message: str) -> Never:
    if required:
        pytest.fail(message)
    pytest.skip(message)


def _load_daemon_testing() -> _DaemonTesting:
    source = os.environ.get('SG_EVENT_DAEMON_SOURCE')
    required = os.environ.get('SG_QUERY_CACHE_REQUIRE_INTEGRATION') == '1'
    if not source:
        _integration_unavailable(required, 'integration requires SG_EVENT_DAEMON_SOURCE')
    try:
        source_root = Path(source).resolve(strict=True)
    except OSError:
        _integration_unavailable(required, 'explicit daemon source root is invalid')
    if not source_root.is_dir():
        _integration_unavailable(required, 'explicit daemon source root is invalid')
    try:
        module = importlib.import_module('sg_event_daemon.testing')
    except ImportError:
        _integration_unavailable(
            required,
            'sg_event_daemon.testing is unavailable from the explicit source',
        )
    origin_value = vars(module).get('__file__')
    if not isinstance(origin_value, str):
        _integration_unavailable(required, 'explicit daemon testing module has no origin')
    try:
        origin = Path(origin_value).resolve(strict=True)
    except OSError:
        _integration_unavailable(required, 'explicit daemon testing module has an invalid origin')
    if not origin.is_relative_to(source_root):
        _integration_unavailable(
            required,
            'daemon testing module is outside the explicit source root',
        )
    interface = ('HarnessSettings', 'drive_sync_plugin', 'load_plugin')
    if not all(callable(vars(module).get(name)) for name in interface):
        _integration_unavailable(required, 'explicit daemon testing module is incompatible')
    return cast('_DaemonTesting', module)


def _testing_module(origin: Path, *, compatible: bool = True) -> ModuleType:
    module = ModuleType('sg_event_daemon.testing')
    module.__file__ = str(origin)
    if compatible:
        vars(module).update(
            {
                'HarnessSettings': _fake_harness_settings,
                'drive_sync_plugin': _fake_harness_call,
                'load_plugin': _fake_harness_call,
            }
        )
    return module


@pytest.fixture(autouse=True)
def reset_plugin_state(monkeypatch: pytest.MonkeyPatch) -> None:
    _RecordingStore._instances.clear()
    _RecordingStore._next_failure = None
    _RecordingStore._timeline.clear()
    monkeypatch.setattr(daemon_plugin._State, '_backend', None)
    monkeypatch.setattr(daemon_plugin._State, '_config_path', None)
    monkeypatch.setattr(daemon_plugin._State, '_site_namespace', None)


@pytest.fixture
def cache_config(tmp_path: Path) -> Path:
    path = tmp_path / 'cache.toml'
    path.write_text(
        'schema_version = 1\n'
        '[cache]\n'
        "site_namespace = 'testing'\n"
        '[cache.l2]\n'
        'enabled = true\n'
        '[cache.l2.valkey.result]\n'
        "host = 'cache-results'\n"
        "password_env = 'SG_QUERY_CACHE_RESULT_PASSWORD'\n"
        '[cache.l2.valkey.metadata]\n'
        "host = 'cache-metadata'\n"
        "password_env = 'SG_QUERY_CACHE_METADATA_PASSWORD'\n",
        encoding='utf-8',
    )
    return path


def _load_with_harness(cache_config: Path) -> tuple[_DaemonTesting, _LoadedPlugin]:
    testing = _load_daemon_testing()
    settings = testing.HarnessSettings(
        site=_SITE,
        config={'config_path': str(cache_config)},
        secrets={_METADATA_SECRET_ENV: _METADATA_SECRET_ENV},
    )
    loaded = testing.load_plugin(
        _PLUGIN_PATH,
        plugin_id='sg-query-cache',
        execution='sync',
        settings=settings,
    )
    return testing, loaded


@pytest.fixture
def event() -> Mapping[str, object]:
    document = cast('object', json.loads(_FIXTURE_PATH.read_text(encoding='utf-8')))
    assert isinstance(document, dict)
    return cast('Mapping[str, Mapping[str, object]]', document)['asset_change']


def _context(
    event: Mapping[str, object] | None = None,
    *,
    extra_secret: bool = False,
) -> _PluginContext:
    reveals: list[str] = []
    secrets = {
        _METADATA_SECRET_ENV: _RevealedSecret('metadata-secret', reveals, _METADATA_SECRET_ENV)
    }
    if extra_secret:
        secrets['UNUSED_SECRET'] = _RevealedSecret('unused-secret', reveals, 'UNUSED_SECRET')
    return _PluginContext(event or {}, reveals, secrets, _SITE)


@pytest.mark.parametrize(
    'failure',
    [
        StaleIncarnation('stale incarnation'),
        BackendUnavailable('Valkey role changed during operation'),
        RuntimeError('backend failure'),
    ],
)
def test_callback_propagates_backend_failure_for_daemon_retry(
    event: Mapping[str, object], failure: BaseException, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _RecordingStore()
    store._failure = failure
    monkeypatch.setattr(daemon_plugin._State, '_backend', store)
    daemon_plugin._State._site_namespace = _SITE

    with pytest.raises(type(failure)):
        daemon_plugin.callback(_context(event))


def test_callback_returns_after_successful_change(
    event: Mapping[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _RecordingStore()
    monkeypatch.setattr(daemon_plugin._State, '_backend', store)
    daemon_plugin._State._site_namespace = _SITE

    result = daemon_plugin.callback(_context(event))

    assert result is None
    assert store._operations[0][0] == 'change'


def test_daemon_testing_accepts_any_explicit_compatible_source_root(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / 'reviewed-daemon'
    origin = source / 'src' / 'sg_event_daemon' / 'testing.py'
    origin.parent.mkdir(parents=True)
    origin.touch()
    module = _testing_module(origin)
    monkeypatch.setenv('SG_EVENT_DAEMON_SOURCE', str(source))
    monkeypatch.setenv('SG_QUERY_CACHE_REQUIRE_INTEGRATION', '1')

    def import_module(_name: str) -> ModuleType:
        return module

    monkeypatch.setattr(importlib, 'import_module', import_module)

    assert _load_daemon_testing() is module


def test_daemon_testing_rejects_incompatible_explicit_module(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / 'reviewed-daemon'
    origin = source / 'src' / 'sg_event_daemon' / 'testing.py'
    origin.parent.mkdir(parents=True)
    origin.touch()
    module = _testing_module(origin, compatible=False)
    monkeypatch.setenv('SG_EVENT_DAEMON_SOURCE', str(source))
    monkeypatch.setenv('SG_QUERY_CACHE_REQUIRE_INTEGRATION', '1')

    def import_module(_name: str) -> ModuleType:
        return module

    monkeypatch.setattr(importlib, 'import_module', import_module)

    with pytest.raises(pytest.fail.Exception, match='incompatible'):
        _load_daemon_testing()


def test_daemon_testing_rejects_invalid_explicit_source_root(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv('SG_EVENT_DAEMON_SOURCE', str(tmp_path / 'missing-daemon'))
    monkeypatch.setenv('SG_QUERY_CACHE_REQUIRE_INTEGRATION', '1')

    with pytest.raises(pytest.fail.Exception, match='source root is invalid'):
        _load_daemon_testing()


def test_daemon_testing_rejects_module_outside_explicit_source_root(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / 'reviewed-daemon'
    source.mkdir()
    origin = tmp_path / 'unrelated-install' / 'sg_event_daemon' / 'testing.py'
    origin.parent.mkdir(parents=True)
    origin.touch()
    module = _testing_module(origin)
    monkeypatch.setenv('SG_EVENT_DAEMON_SOURCE', str(source))
    monkeypatch.setenv('SG_QUERY_CACHE_REQUIRE_INTEGRATION', '1')

    def import_module(_name: str) -> ModuleType:
        return module

    monkeypatch.setattr(importlib, 'import_module', import_module)

    with pytest.raises(pytest.fail.Exception, match='outside the explicit source root'):
        _load_daemon_testing()


def test_daemon_testing_rejects_module_without_origin(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / 'reviewed-daemon'
    source.mkdir()
    module = ModuleType('sg_event_daemon.testing')
    vars(module).update(
        {
            'HarnessSettings': _fake_harness_settings,
            'drive_sync_plugin': _fake_harness_call,
            'load_plugin': _fake_harness_call,
        }
    )
    monkeypatch.setenv('SG_EVENT_DAEMON_SOURCE', str(source))
    monkeypatch.setenv('SG_QUERY_CACHE_REQUIRE_INTEGRATION', '1')

    def import_module(_name: str) -> ModuleType:
        return module

    monkeypatch.setattr(importlib, 'import_module', import_module)

    with pytest.raises(pytest.fail.Exception, match='has no origin'):
        _load_daemon_testing()


def test_daemon_testing_requires_explicit_source_in_required_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv('SG_EVENT_DAEMON_SOURCE', raising=False)
    monkeypatch.setenv('SG_QUERY_CACHE_REQUIRE_INTEGRATION', '1')

    with pytest.raises(pytest.fail.Exception, match='requires SG_EVENT_DAEMON_SOURCE'):
        _load_daemon_testing()


@pytest.mark.integration
def test_public_harness_loads_path_and_drives_setup_before_callback(
    monkeypatch: pytest.MonkeyPatch,
    cache_config: Path,
    event: Mapping[str, object],
) -> None:
    _install_recording_valkey(monkeypatch)
    monkeypatch.setenv(_METADATA_SECRET_ENV, 'metadata-secret')
    testing, loaded = _load_with_harness(cache_config)

    result = testing.drive_sync_plugin(loaded, (event,))

    assert result.ok
    assert loaded.callbacks == ('invalidate-cache',)
    operations = _RecordingStore._instances[0]._operations
    assert operations[0:2] == [
        ('resolve', (DependencyTag.site(_SITE),)),
        ('flush_site', (_SITE, 'daemon_setup')),
    ]
    assert operations[2] == (
        'change',
        (
            GenerationChange(DependencyTag.type_value(_SITE, 'Asset')),
            GenerationChange(DependencyTag.type_field(_SITE, 'Asset', 'code')),
            GenerationChange(DependencyTag.entity(_SITE, 'Asset', 7)),
        ),
    )


@pytest.mark.integration
@pytest.mark.parametrize(
    'failure',
    [
        StaleIncarnation('stale incarnation'),
        BackendUnavailable('Valkey role changed during operation'),
    ],
)
def test_public_harness_keeps_uncertain_mutation_failed_until_retry_succeeds(
    monkeypatch: pytest.MonkeyPatch,
    cache_config: Path,
    event: Mapping[str, object],
    failure: BaseException,
) -> None:
    _install_recording_valkey(monkeypatch)
    monkeypatch.setenv(_METADATA_SECRET_ENV, 'metadata-secret')
    testing, loaded = _load_with_harness(cache_config)
    _RecordingStore._next_failure = failure

    failed = testing.drive_sync_plugin(loaded, (event,), delivery_attempt=1)
    retried = testing.drive_sync_plugin(loaded, (event,), delivery_attempt=2)

    assert not failed.ok
    assert failed.callbacks[0].failure is not None
    assert retried.ok
    assert retried.callbacks[0].delivery_attempt == 2


@pytest.mark.integration
def test_duplicate_and_operator_replayed_callbacks_are_safe_and_log_no_payload(
    monkeypatch: pytest.MonkeyPatch,
    cache_config: Path,
    event: Mapping[str, object],
    caplog: pytest.LogCaptureFixture,
) -> None:
    _install_recording_valkey(monkeypatch)
    monkeypatch.setenv(_METADATA_SECRET_ENV, 'metadata-secret')
    testing, loaded = _load_with_harness(cache_config)
    sensitive_event = dict(event)
    sensitive_event['meta'] = {
        **cast('Mapping[str, object]', event['meta']),
        'new_value': 'https://private.invalid/filter?secret=payload',
    }

    with caplog.at_level(logging.DEBUG):
        duplicate = testing.drive_sync_plugin(loaded, (sensitive_event, sensitive_event))
        replay = testing.drive_sync_plugin(
            loaded,
            (sensitive_event,),
            delivery_attempt=3,
            is_operator_replay=True,
        )

    assert duplicate.ok
    assert replay.ok
    assert 'private.invalid' not in caplog.text
    assert 'metadata-secret' not in caplog.text


def test_register_reads_only_config_path_and_declares_setup_before_callback(
    cache_config: Path,
) -> None:
    registrar = _RecordingRegistrar(str(cache_config))

    daemon_plugin.register(registrar)

    assert registrar.config._accessed == ['config_path']
    assert registrar._calls == [
        ('setup', daemon_plugin.setup),
        (
            'callback',
            {'id': 'invalidate-cache', 'callback': daemon_plugin.callback},
        ),
    ]


def test_second_setup_publishes_fresh_flushed_store_before_closing_previous(
    monkeypatch: pytest.MonkeyPatch,
    cache_config: Path,
) -> None:
    _install_recording_valkey(monkeypatch)
    daemon_plugin.register(_RecordingRegistrar(str(cache_config)))
    context = _context()

    daemon_plugin.setup(context)
    daemon_plugin.setup(context)

    first, second = _RecordingStore._instances
    assert _RecordingStore._timeline == [
        (0, 'resolve', None),
        (0, 'flush_site', None),
        (1, 'resolve', 0),
        (1, 'flush_site', 0),
        (0, 'close', 1),
    ]
    assert daemon_plugin._State._backend is second
    assert first._operations[-1] == ('close', None)
    assert second._operations == [
        ('resolve', (DependencyTag.site(_SITE),)),
        ('flush_site', (_SITE, 'daemon_setup')),
    ]


def test_setup_rejects_site_mismatch_before_backend_creation(
    monkeypatch: pytest.MonkeyPatch,
    cache_config: Path,
) -> None:
    _install_recording_valkey(monkeypatch)
    daemon_plugin.register(_RecordingRegistrar(str(cache_config)))
    context = _context()
    context.site = 'other-site'

    with pytest.raises(RuntimeError, match='site namespace'):
        daemon_plugin.setup(context)

    assert _RecordingStore._instances == []
    assert context._reveals == []


def test_setup_reveals_only_metadata_secret_validates_and_flushes(
    monkeypatch: pytest.MonkeyPatch,
    cache_config: Path,
) -> None:
    _install_recording_valkey(monkeypatch)
    daemon_plugin.register(_RecordingRegistrar(str(cache_config)))
    context = _context(extra_secret=True)

    daemon_plugin.setup(context)

    assert context._reveals == [_METADATA_SECRET_ENV]
    assert _RecordingStore._instances[0]._operations == [
        ('resolve', (DependencyTag.site(_SITE),)),
        ('flush_site', (_SITE, 'daemon_setup')),
    ]
