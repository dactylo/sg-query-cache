import asyncio
import os
from collections.abc import Iterator
from typing import TYPE_CHECKING

import pytest

from sg_query_cache._config import CacheSettings
from sg_query_cache.valkey_async import AsyncValkeyBackend
from tests._backend_contract import (
    SyncDriver,
    _BackendContractTests,
    _BackendFactory,
    _BackendHarness,
)
from tests._fakes import _RecordingMetrics
from tests.valkey._environment import settings as _integration_settings
from tests.valkey._integration_helpers import ServerClock, flush_roles

pytestmark = pytest.mark.integration

if TYPE_CHECKING:
    from sg_query_cache._backend import CacheBackend


@pytest.fixture
def backend_factory() -> Iterator[_BackendFactory]:
    if os.environ.get('SG_QUERY_CACHE_REQUIRE_INTEGRATION') != '1':
        pytest.skip('requires explicitly configured disposable Valkey roles')
    backends: list[CacheBackend] = []
    with asyncio.Runner() as runner:

        def create(settings: CacheSettings | None = None) -> _BackendHarness:
            limits = None if settings is None else settings.limits
            resolved = _integration_settings(os.environ, limits)
            flush_roles(resolved)
            metrics = _RecordingMetrics()
            backend = SyncDriver(
                runner.run(AsyncValkeyBackend.from_settings(resolved, os.environ, metrics)),
                runner,
            )
            backends.append(backend)
            return _BackendHarness(backend, ServerClock(), metrics)

        yield create
        for backend in backends:
            backend.close()


class TestAsyncValkeyBackend(_BackendContractTests):
    pass
