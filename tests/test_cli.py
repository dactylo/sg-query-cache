import argparse
import importlib.metadata
import sys
from collections.abc import Callable, Iterator, Mapping, Sequence
from importlib import import_module
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import TypeVar, cast

import pytest

from sg_query_cache import _cli
from sg_query_cache._backend import BackendUnavailable
from sg_query_cache._config import (
    CacheSettings,
    L2Settings,
    ValkeyRoleSettings,
    ValkeySettings,
)
from sg_query_cache._metrics import NullMetrics
from tests._cli_helpers import invoke_cli

_T = TypeVar('_T')

_VALID_COMMAND_ARGUMENTS = ['daemon-plugin-path']

_METADATA_CREDENTIAL = 'metadata-credential'
_PREFIX = 'metadata-prefix'
_RESULT_CREDENTIAL = 'result-credential'
_SITE = 'studio'
_SITE_DIGEST = 'da0daaba4b156961c049ab4b85b6d0bcba2872200b7dedb4aa77f9208c601444'


class _FakeMetadataStore:
    _created_with: tuple[CacheSettings, dict[str, str], object] | None = None

    def __init__(self) -> None:
        self._rotations = 0
        self._closed = False
        self._flushed: tuple[str, str] | None = None

    def close(self) -> None:
        self._closed = True

    def flush_site(self, site_namespace: str, reason: str) -> None:
        self._flushed = (site_namespace, reason)

    @classmethod
    def from_settings(
        cls,
        settings: CacheSettings,
        environ: Mapping[str, str],
        metrics: object,
    ) -> '_FakeMetadataStore':
        cls._created_with = settings, dict(environ), metrics
        return cls()

    def rotate_incarnation(self) -> None:
        self._rotations += 1


def _fail_on_io(*_args: object, **_kwargs: object) -> None:
    raise AssertionError('CLI help and parser errors must not perform I/O')


def _fail_plugin_import(_name: str) -> ModuleType:
    details = (_METADATA_CREDENTIAL, (_SITE, 1, 1, _PREFIX))
    raise SyntaxError(*details)


def _fake_valkey_import(name: str) -> ModuleType | SimpleNamespace:
    if name == 'sg_query_cache.valkey':
        return SimpleNamespace(ValkeyMetadataStore=_FakeMetadataStore)
    return import_module(name)


def _fixed_version(_name: str) -> str:
    return '9.8.7'


def _help_entries(
    parser: argparse.ArgumentParser, prefix: str
) -> Iterator[tuple[str, str | None]]:
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            subparsers = cast('argparse._SubParsersAction[argparse.ArgumentParser]', action)
            visible = [choice.dest for choice in subparsers._choices_actions]
            for choice in subparsers._choices_actions:
                yield f'{prefix} {choice.dest}', choice.help
            for name in visible:
                yield from _help_entries(subparsers.choices[name], f'{prefix} {name}')
        elif action.help is not argparse.SUPPRESS:
            label = '/'.join(action.option_strings) or action.dest
            yield f'{prefix} {label}', action.help


def _module_at(path: Path) -> Callable[[str], SimpleNamespace]:
    def load_module(_name: str) -> SimpleNamespace:
        return SimpleNamespace(__file__=str(path))

    return load_module


def _open_store_raising(error: Exception) -> Callable[[CacheSettings], _FakeMetadataStore]:
    def open_store(_settings: CacheSettings) -> _FakeMetadataStore:
        raise error

    return open_store


def _raise_store_construction_failure(_settings: CacheSettings) -> _FakeMetadataStore:
    raise BackendUnavailable(f'/private/install/{_SITE}/{_PREFIX}/{_METADATA_CREDENTIAL}')


def _return_store(store: _T) -> Callable[[CacheSettings], _T]:
    def open_store(_settings: CacheSettings) -> _T:
        return store

    return open_store


def _settings() -> CacheSettings:
    result = ValkeyRoleSettings('result.internal', 'RESULT_PASSWORD', key_prefix='result-prefix')
    metadata = ValkeyRoleSettings(
        'metadata.internal',
        'METADATA_PASSWORD',
        key_prefix=_PREFIX,
    )
    return CacheSettings(
        l2=L2Settings(enabled=True, valkey=ValkeySettings(result=result, metadata=metadata))
    )


def _load_test_settings(_path: str) -> CacheSettings:
    return _settings()


def _starts_in_lowercase_style(text: str) -> bool:
    first = text.split(maxsplit=1)[0]
    is_acronym = first.upper() == first and any(character.isalpha() for character in first)
    return text[0].islower() or is_acronym


def test_daemon_plugin_path_failure_is_redacted(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    missing = tmp_path / 'private-install' / 'daemon_plugin.py'
    monkeypatch.setattr(
        _cli,
        'import_module',
        _module_at(missing),
    )

    assert _cli.main(('daemon-plugin-path',)) == 1

    stdout, stderr = capsys.readouterr()
    assert stdout == ''
    assert 'sg_query_cache' in stderr
    assert str(tmp_path) not in stderr


def test_daemon_plugin_path_prints_only_existing_python_file(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_cli, '_load_settings', _fail_on_io)
    monkeypatch.setattr(_cli, '_open_store', _fail_on_io)
    monkeypatch.delitem(sys.modules, 'sg_query_cache.valkey', raising=False)

    assert _cli.main(('daemon-plugin-path',)) == 0

    stdout, stderr = capsys.readouterr()
    path = Path(stdout.strip())
    assert stderr == ''
    assert path == Path(_cli.__file__).with_name('daemon_plugin.py').resolve()
    assert path.is_file()
    assert 'sg_query_cache.valkey' not in sys.modules


def test_daemon_plugin_path_redacts_ordinary_import_failure(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_cli, 'import_module', _fail_plugin_import)

    assert _cli.main(('daemon-plugin-path',)) == 1

    stdout, stderr = capsys.readouterr()
    assert stdout == ''
    assert stderr == 'error: sg_query_cache daemon plugin is unavailable\n'
    assert _SITE not in stderr
    assert _PREFIX not in stderr
    assert _METADATA_CREDENTIAL not in stderr


def test_every_visible_option_and_command_has_lowercase_help() -> None:
    problems = [
        f'{name}: {text!r}'
        for name, text in _help_entries(_cli._build_parser(), 'sg-query-cache')
        if not text or not _starts_in_lowercase_style(text) or text.endswith('.')
    ]
    assert problems == []


def test_flush_acknowledges_opaque_site_digest_and_closes_store(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _FakeMetadataStore()
    monkeypatch.setattr(_cli, '_load_settings', _load_test_settings)
    monkeypatch.setattr(_cli, '_open_store', _return_store(store))
    monkeypatch.setenv('METADATA_PASSWORD', _METADATA_CREDENTIAL)

    status = _cli.main(
        ('flush', '--config', 'cache.toml', '--site', _SITE, '--reason', 'operator')
    )

    stdout, stderr = capsys.readouterr()
    assert status == 0
    assert stdout == f'flush acknowledged site={_SITE_DIGEST}\n'
    assert stderr == ''
    assert store._flushed == (_SITE, 'operator')
    assert store._closed
    assert _SITE not in stdout
    assert _METADATA_CREDENTIAL not in stdout


def test_flush_rejects_an_invalid_reason_as_a_usage_error() -> None:
    status, stdout, stderr = invoke_cli(
        ['flush', '--config', 'cache.toml', '--site', _SITE, '--reason', '']
    )
    assert (status, stdout) == (2, '')
    assert stderr.endswith(
        'sg-query-cache flush: error: argument --reason: must be 1 to 64 ASCII characters\n'
    )


def test_flush_rejects_an_invalid_site_as_a_usage_error() -> None:
    status, stdout, stderr = invoke_cli(
        ['flush', '--config', 'cache.toml', '--site', ' site ', '--reason', 'operator']
    )
    assert (status, stdout) == (2, '')
    assert stderr.endswith(
        'sg-query-cache flush: error: argument --site: must be nonempty ASCII without '
        'leading or trailing spaces\n'
    )


@pytest.mark.parametrize(
    ('arguments', 'tokens'),
    [
        pytest.param(
            ['--help'],
            ('--version', 'flush', 'rotate-incarnation', 'daemon-plugin-path'),
            id='root',
        ),
        pytest.param(
            ['flush', '--help'],
            ('flush', '--config FILE', '--site SITE', '--reason REASON'),
            id='flush',
        ),
        pytest.param(
            ['rotate-incarnation', '--help'],
            ('rotate-incarnation', '--config FILE', '--confirm-prefix PREFIX'),
            id='rotate-incarnation',
        ),
        pytest.param(
            ['daemon-plugin-path', '--help'],
            ('daemon-plugin-path',),
            id='daemon-plugin-path',
        ),
    ],
)
def test_help_lists_commands_and_options_without_backend_io(
    arguments: list[str], tokens: tuple[str, ...], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(_cli, '_load_settings', _fail_on_io)
    monkeypatch.setattr(_cli, '_open_store', _fail_on_io)
    status, stdout, stderr = invoke_cli(arguments)
    assert (status, stderr) == (0, '')
    assert stdout.startswith('usage: sg-query-cache')
    normalized = ' '.join(stdout.split())
    for token in tokens:
        assert token in normalized
    if arguments == ['--help']:
        assert stdout.index('options:') < stdout.index('commands:')


def test_interrupted_command_exits_quietly(monkeypatch: pytest.MonkeyPatch) -> None:
    def interrupt(_path: str) -> CacheSettings:
        raise KeyboardInterrupt

    monkeypatch.setattr(_cli, '_load_settings', interrupt)

    assert invoke_cli(
        ['flush', '--config', 'cache.toml', '--site', _SITE, '--reason', 'operator']
    ) == (130, '', '')


def test_missing_command_is_a_usage_error() -> None:
    status, stdout, stderr = invoke_cli([])
    assert (status, stdout) == (2, '')
    assert stderr.endswith(
        'sg-query-cache: error: the following arguments are required: COMMAND\n'
    )


def test_missing_metadata_is_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing(name: str) -> str:
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib.metadata, 'version', missing)
    status, stdout, stderr = invoke_cli(['--version'])
    assert (status, stdout) == (2, '')
    assert stderr == 'error: installed distribution metadata for sg-query-cache is unavailable\n'


def test_normal_runs_do_not_read_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(name: str) -> str:
        raise AssertionError(name)

    monkeypatch.setattr(importlib.metadata, 'version', forbidden)
    assert invoke_cli(['--help'])[0] == 0


def test_open_store_uses_only_metadata_role_and_secret(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _FakeMetadataStore._created_with = None
    monkeypatch.setattr(
        _cli,
        'import_module',
        _fake_valkey_import,
    )
    monkeypatch.setenv('METADATA_PASSWORD', _METADATA_CREDENTIAL)
    monkeypatch.setenv('RESULT_PASSWORD', _RESULT_CREDENTIAL)
    settings = _settings()

    store = _cli._open_store(settings)

    assert isinstance(store, _FakeMetadataStore)
    assert _FakeMetadataStore._created_with is not None
    assert _FakeMetadataStore._created_with[:2] == (
        settings,
        {'METADATA_PASSWORD': _METADATA_CREDENTIAL},
    )
    assert isinstance(_FakeMetadataStore._created_with[2], NullMetrics)
    created_settings = _FakeMetadataStore._created_with[0]
    assert created_settings.l2.valkey.metadata.host == 'metadata.internal'
    assert created_settings.l2.valkey.result.host == 'result.internal'
    assert _RESULT_CREDENTIAL not in _FakeMetadataStore._created_with[1].values()


@pytest.mark.parametrize(
    'arguments',
    [
        ('--unknown',),
        ('flush',),
        ('flush', '--config', 'cache.toml', '--site', 'production'),
        ('rotate-incarnation', '--config', 'cache.toml'),
        ('daemon-plugin-path', '--unknown'),
    ],
)
def test_parser_errors_exit_two_without_io(
    arguments: Sequence[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_cli, '_load_settings', _fail_on_io)
    monkeypatch.setattr(_cli, '_open_store', _fail_on_io)

    with pytest.raises(SystemExit) as raised:
        _cli.main(arguments)

    assert raised.value.code == 2


def test_rotation_calls_owned_operation_once_and_closes_store(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _FakeMetadataStore()
    monkeypatch.setattr(_cli, '_load_settings', _load_test_settings)
    monkeypatch.setattr(_cli, '_open_store', _return_store(store))

    status = _cli.main(
        ('rotate-incarnation', '--config', 'cache.toml', '--confirm-prefix', _PREFIX)
    )

    stdout, stderr = capsys.readouterr()
    assert status == 0
    assert stdout == 'incarnation rotation acknowledged for all sites\n'
    assert stderr == ''
    assert store._rotations == 1
    assert store._closed
    assert _PREFIX not in stdout


def test_rotation_redacts_owned_operation_failure_and_closes_store(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _FakeMetadataStore()

    def fail_rotation() -> None:
        raise BackendUnavailable(_METADATA_CREDENTIAL)

    monkeypatch.setattr(store, 'rotate_incarnation', fail_rotation)
    monkeypatch.setattr(_cli, '_load_settings', _load_test_settings)
    monkeypatch.setattr(_cli, '_open_store', _return_store(store))
    assert (
        _cli.main(('rotate-incarnation', '--config', 'cache.toml', '--confirm-prefix', _PREFIX))
        == 1
    )
    stdout, stderr = capsys.readouterr()
    assert stdout == ''
    assert stderr == 'error: incarnation rotation was not acknowledged\n'
    assert store._closed


def test_rotation_requires_exact_prefix_before_opening_store(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_cli, '_load_settings', _load_test_settings)
    monkeypatch.setattr(_cli, '_open_store', _fail_on_io)

    status = _cli.main(
        ('rotate-incarnation', '--config', 'cache.toml', '--confirm-prefix', 'wrong-prefix')
    )

    stdout, stderr = capsys.readouterr()
    assert status == 2
    assert stdout == ''
    assert stderr == 'error: prefix confirmation does not match configuration\n'
    assert _PREFIX not in stderr


@pytest.mark.parametrize(
    ('arguments', 'message'),
    [
        (
            ('flush', '--config', 'cache.toml', '--site', _SITE, '--reason', 'operator'),
            'error: flush operation was not acknowledged\n',
        ),
        (
            ('rotate-incarnation', '--config', 'cache.toml', '--confirm-prefix', _PREFIX),
            'error: incarnation rotation was not acknowledged\n',
        ),
    ],
)
def test_store_construction_runtime_failure_is_redacted_operational_status(
    arguments: Sequence[str],
    message: str,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_cli, '_load_settings', _load_test_settings)
    monkeypatch.setattr(_cli, '_open_store', _raise_store_construction_failure)

    assert _cli.main(arguments) == 1

    stdout, stderr = capsys.readouterr()
    assert stdout == ''
    assert stderr == message
    assert _SITE not in stderr
    assert _PREFIX not in stderr
    assert _METADATA_CREDENTIAL not in stderr


@pytest.mark.parametrize(
    'error',
    [
        pytest.param(
            ModuleNotFoundError(f"No module named 'valkey' {_METADATA_CREDENTIAL}", name='valkey'),
            id='missing-valkey-extra',
        ),
        pytest.param(ValueError(_METADATA_CREDENTIAL), id='missing-password-or-disabled-l2'),
        pytest.param(KeyError(_METADATA_CREDENTIAL), id='missing-setting'),
    ],
)
@pytest.mark.parametrize(
    'arguments',
    [
        pytest.param(
            ['flush', '--config', 'cache.toml', '--site', _SITE, '--reason', 'operator'],
            id='flush',
        ),
        pytest.param(
            ['rotate-incarnation', '--config', 'cache.toml', '--confirm-prefix', _PREFIX],
            id='rotate-incarnation',
        ),
    ],
)
def test_store_setup_failure_is_a_redacted_configuration_error(
    arguments: list[str],
    error: Exception,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_cli, '_load_settings', _load_test_settings)
    monkeypatch.setattr(_cli, '_open_store', _open_store_raising(error))

    assert invoke_cli(arguments) == (2, '', 'error: configuration is invalid or unavailable\n')


def test_unavailable_stdout_is_an_error(capsys: pytest.CaptureFixture[str]) -> None:
    # Restore stdout before capsys finalizes, so its capture stream is not reinstated.
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(sys, 'stdout', None)
        status = _cli.main(['--help'])

    assert (status, capsys.readouterr().err) == (2, 'error: standard output is unavailable\n')


@pytest.mark.parametrize(
    'command',
    [
        pytest.param(['flush', '--site', _SITE, '--reason', 'operator'], id='flush'),
        pytest.param(['rotate-incarnation', '--confirm-prefix', _PREFIX], id='rotate-incarnation'),
    ],
)
def test_unreadable_configuration_is_a_redacted_setup_error(
    command: list[str],
    tmp_path: Path,
) -> None:
    missing = tmp_path / _METADATA_CREDENTIAL / 'cache.toml'

    status, stdout, stderr = invoke_cli([*command, '--config', str(missing)])

    assert (status, stdout) == (2, '')
    assert stderr == 'error: configuration is invalid or unavailable\n'


def test_version_ignores_other_valid_arguments(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(importlib.metadata, 'version', _fixed_version)
    assert invoke_cli(['--version', *_VALID_COMMAND_ARGUMENTS]) == (
        0,
        'sg-query-cache 9.8.7\n',
        '',
    )


def test_version_prints_the_installed_version(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(importlib.metadata, 'version', _fixed_version)
    assert invoke_cli(['--version']) == (0, 'sg-query-cache 9.8.7\n', '')
