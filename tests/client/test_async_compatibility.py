import inspect
import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import pytest

from sg_query_cache import AsyncCachedShotgun, CachedShotgun
from tests.client._compatibility import EXPLICIT_METHODS


def _parameter_contract(
    method: Callable[..., object],
) -> tuple[tuple[str, inspect._ParameterKind, object], ...]:
    return tuple(
        (parameter.name, parameter.kind, parameter.default)
        for parameter in inspect.signature(method).parameters.values()
    )


def test_complete_explicit_async_surface_matches_sync_parameter_contracts() -> None:
    actual = tuple(
        name
        for name, value in vars(AsyncCachedShotgun).items()
        if name in EXPLICIT_METHODS and callable(value)
    )

    assert set(actual) == set(EXPLICIT_METHODS)
    for name in EXPLICIT_METHODS:
        async_method = getattr(AsyncCachedShotgun, name)
        assert inspect.iscoroutinefunction(async_method), name
        assert _parameter_contract(async_method) == _parameter_contract(
            getattr(CachedShotgun, name)
        ), name


def test_constructor_keeps_configuration_keyword_only() -> None:
    parameters = tuple(inspect.signature(AsyncCachedShotgun).parameters.values())

    assert tuple(parameter.name for parameter in parameters) == (
        'client',
        'cache',
        'context',
        'indeterminate_write_exceptions',
    )
    assert all(
        parameter.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD for parameter in parameters[:3]
    )
    assert parameters[3].kind is inspect.Parameter.KEYWORD_ONLY
    assert parameters[3].default == ()


def test_open_matches_the_prescribed_local_factory_signature() -> None:
    parameters = tuple(inspect.signature(AsyncCachedShotgun.open).parameters.values())

    assert tuple(parameter.name for parameter in parameters) == (
        'client',
        'settings',
        'identity',
        'context',
        'metrics',
        'environ',
        'indeterminate_write_exceptions',
    )
    assert all(
        parameter.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD for parameter in parameters[:4]
    )
    assert all(parameter.kind is inspect.Parameter.KEYWORD_ONLY for parameter in parameters[4:])
    assert tuple(parameter.default for parameter in parameters[4:]) == (None, None, ())


@pytest.mark.integration
def test_selected_native_async_client_signatures_match() -> None:
    source_value = os.environ.get('SG_QUERY_CACHE_ALT_API_SOURCE')
    required = os.environ.get('SG_QUERY_CACHE_REQUIRE_INTEGRATION') == '1'
    if source_value is None:
        if required:
            pytest.fail('SG_QUERY_CACHE_ALT_API_SOURCE is required for integration')
        pytest.skip('native async SG Python API source was not selected')

    source = Path(source_value)
    source_dir = source / 'src' if (source / 'src' / 'shotgun_api3').is_dir() else source
    if not (source_dir / 'shotgun_api3').is_dir():
        if required:
            pytest.fail('selected sibling source has no shotgun_api3 source package')
        pytest.skip('selected native async source layout is unavailable')

    script = f"""
import inspect
from shotgun_api3 import AsyncShotgun
from sg_query_cache import AsyncCachedShotgun

methods = {EXPLICIT_METHODS!r}
def contract(method):
    return tuple(
        (value.name, value.kind, value.default)
        for value in inspect.signature(method).parameters.values()
    )
for name in methods:
    assert contract(getattr(AsyncCachedShotgun, name)) == contract(
        getattr(AsyncShotgun, name)
    ), name
"""
    environment = os.environ.copy()
    environment['PYTHONPATH'] = os.pathsep.join((str(source_dir), str(Path.cwd() / 'src')))
    result = subprocess.run(
        [sys.executable, '-c', script],
        check=False,
        capture_output=True,
        env=environment,
        text=True,
    )

    assert result.returncode == 0, result.stderr
