import os
import subprocess
import sys
import textwrap
import tomllib
from pathlib import Path
from time import monotonic

import sg_query_cache

_PROJECT_ROOT = Path(__file__).parents[1]
_PYPROJECT = _PROJECT_ROOT / 'pyproject.toml'
_SOURCE_ROOT = _PROJECT_ROOT / 'src'
_SUPPORTED_EXPORTS = [
    'AsyncCacheBackend',
    'AsyncCachedShotgun',
    'AsyncLocalBackend',
    'AsyncQueryCache',
    'CacheSettings',
    'CachedShotgun',
    'ClientContext',
    'Identity',
    'LocalBackend',
    'MetricsSink',
    'NullMetrics',
    'QueryCache',
]


class _HumanPolicyClient:
    _find_calls: int
    _schema_reads: int

    def __init__(self) -> None:
        self._find_calls = 0
        self._schema_reads = 0

    def find(self, *_args: object, **_kwargs: object) -> list[dict[str, int]]:
        self._find_calls += 1
        return [{'id': 1}]

    def schema_read(self, _project_entity: object) -> dict[str, object]:
        self._schema_reads += 1
        return {}


def _run_python(*arguments: str) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment['PYTHONPATH'] = str(_SOURCE_ROOT)
    return subprocess.run(
        [sys.executable, *arguments],
        check=False,
        capture_output=True,
        env=environment,
        text=True,
    )


def test_cached_shotgun_reads_identity_and_settings_through_its_cache() -> None:
    client = _HumanPolicyClient()
    settings = sg_query_cache.CacheSettings()
    metrics = sg_query_cache.NullMetrics()
    backend = sg_query_cache.LocalBackend(settings, monotonic, metrics)
    cache = sg_query_cache.QueryCache(
        settings,
        sg_query_cache.Identity(
            kind='human',
            issuer='tests',
            namespace='human:1',
            entity_id=1,
        ),
        l1=backend,
        metrics=metrics,
    )
    wrapped = sg_query_cache.CachedShotgun(
        client,
        cache,
        sg_query_cache.ClientContext(
            site_url='https://example.invalid',
            site_namespace=settings.site_namespace,
            localized=False,
            convert_datetimes_to_utc=True,
            records_per_page=500,
        ),
    )

    assert wrapped.find('Asset', [], ['id']) == [{'id': 1}]
    assert client._find_calls == 1
    assert client._schema_reads == 0
    cache.close()


def test_console_script_matches_the_module_help_without_installation() -> None:
    with _PYPROJECT.open('rb') as stream:
        project = tomllib.load(stream)
    target = project['project']['scripts']['sg-query-cache']
    assert target == 'sg_query_cache._cli:main'

    console = _run_python(
        '-c',
        textwrap.dedent(
            """
            import importlib
            import sys

            module_name, attribute = sys.argv[1].split(':', 1)
            command = getattr(importlib.import_module(module_name), attribute)
            raise SystemExit(command(['--help']))
            """
        ),
        target,
    )
    module = _run_python('-m', 'sg_query_cache', '--help')

    assert console.returncode == 0, console.stderr
    assert module.returncode == 0, module.stderr
    assert module.stdout.startswith('usage: sg-query-cache [-h] [--version] COMMAND ...\n')
    assert console.stdout == module.stdout


def test_package_root_and_local_mode_do_not_import_valkey() -> None:
    completed = _run_python(
        '-c',
        textwrap.dedent(
            """
            import importlib.abc
            import sys
            from time import monotonic


            class _BlockValkey(importlib.abc.MetaPathFinder):
                def find_spec(self, fullname, path=None, target=None):
                    if fullname == 'valkey' or fullname.startswith('valkey.'):
                        raise ModuleNotFoundError('blocked optional dependency')
                    return None


            sys.meta_path.insert(0, _BlockValkey())

            from sg_query_cache import (
                AsyncCacheBackend,
                AsyncCachedShotgun,
                AsyncLocalBackend,
                AsyncQueryCache,
                CacheSettings,
                CachedShotgun,
                ClientContext,
                Identity,
                LocalBackend,
                NullMetrics,
                QueryCache,
            )


            class _FakeClient:
                def find(self, *args, **kwargs):
                    return [{'id': 1}]


            settings = CacheSettings()
            metrics = NullMetrics()
            backend = LocalBackend(settings, monotonic, metrics)
            async_backend: AsyncCacheBackend = AsyncLocalBackend(settings, monotonic, metrics)
            assert AsyncCachedShotgun.__name__ == 'AsyncCachedShotgun'
            cache = QueryCache(
                settings,
                Identity(kind='script', issuer='tests', namespace='public-api'),
                l1=backend,
                metrics=metrics,
            )
            client = CachedShotgun(
                _FakeClient(),
                cache,
                ClientContext(
                    site_url='https://example.invalid',
                    site_namespace=settings.site_namespace,
                    localized=False,
                    convert_datetimes_to_utc=True,
                    records_per_page=500,
                ),
            )
            assert client.find('EventLogEntry', [], ['id']) == [{'id': 1}]
            cache.close()

            import asyncio
            asyncio.run(async_backend.close())
            """
        ),
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == ''


def test_package_root_exports_exactly_the_supported_public_api() -> None:
    assert sg_query_cache.__all__ == _SUPPORTED_EXPORTS


def test_valkey_backend_imports_when_the_optional_dependency_is_installed() -> None:
    completed = _run_python(
        '-c',
        'from sg_query_cache.valkey import ValkeyBackend; '
        'assert ValkeyBackend.__name__ == "ValkeyBackend"',
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == ''
