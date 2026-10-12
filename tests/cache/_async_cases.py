import asyncio
from dataclasses import dataclass

from sg_query_cache._async_cache import AsyncQueryCache
from sg_query_cache._async_local import AsyncLocalBackend
from sg_query_cache._config import CacheSettings, L1Settings, L2Settings
from sg_query_cache._model import DependencyPlan, DependencyTag, ExactCall, Identity
from tests._async_fakes import _PhaseBackend, _RecordingMetrics

_CALL = ExactCall('find', (('entity_type', 'Asset'), ('filters', ())))
_IDENTITY = Identity('script', 'tests', 'async-core')
_OPTIONS = {'localized': False, 'convert_datetimes_to_utc': True}
_PLAN = DependencyPlan('event_aware', (DependencyTag.site('production'),), 'supported')


class _Clock:
    def __init__(self) -> None:
        self._now = 1000.0

    def __call__(self) -> float:
        return self._now


@dataclass
class _Fixture:
    _cache: AsyncQueryCache
    _l1: _PhaseBackend | None
    _l2: _PhaseBackend | None
    _clock: _Clock
    _metrics: _RecordingMetrics


async def _answer() -> object:
    return 42


def fixture(topology: str = 'local', settings: CacheSettings | None = None) -> _Fixture:
    selected = settings or CacheSettings(
        l1=L1Settings(enabled=topology != 'shared'),
        l2=L2Settings(enabled=topology != 'local'),
    )
    metrics = _RecordingMetrics()
    clock = _Clock()
    l1 = (
        _PhaseBackend(AsyncLocalBackend(selected, clock, metrics)) if selected.l1.enabled else None
    )
    l2 = (
        _PhaseBackend(AsyncLocalBackend(selected, clock, metrics)) if selected.l2.enabled else None
    )
    cache = AsyncQueryCache(selected, _IDENTITY, l1=l1, l2=l2, metrics=metrics)
    return _Fixture(cache, l1, l2, clock, metrics)


def parse(value: object) -> object:
    return ('parsed', value)


async def read(cache: AsyncQueryCache) -> object:
    return await cache.read(_CALL, _PLAN, _answer, _OPTIONS)


async def turn() -> None:
    ready = asyncio.Event()
    asyncio.get_running_loop().call_soon(ready.set)
    await ready.wait()
