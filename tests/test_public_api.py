import textwrap
from pathlib import Path

import sg_query_cache
from tests._cli_helpers import run_python

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


def test_package_root_and_local_mode_do_not_import_valkey() -> None:
    completed = run_python(
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
    assert set(sg_query_cache.__all__) == set(_SUPPORTED_EXPORTS)
    assert len(sg_query_cache.__all__) == len(_SUPPORTED_EXPORTS)


def test_valkey_backend_imports_when_the_optional_dependency_is_installed() -> None:
    completed = run_python(
        '-c',
        'from sg_query_cache.valkey import ValkeyBackend; '
        'assert ValkeyBackend.__name__ == "ValkeyBackend"',
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == ''
