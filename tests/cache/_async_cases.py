from dataclasses import dataclass

from sg_query_cache._async_cache import AsyncQueryCache
from sg_query_cache._async_local import AsyncLocalBackend
from sg_query_cache._config import CacheSettings, L1Settings, L2Settings
from sg_query_cache._model import DependencyPlan, DependencyTag, Identity
from tests._async_fakes import PhaseBackend, RecordingMetrics
from tests._fakes import CLIENT_OPTIONS, asset_call

CALL = asset_call()
IDENTITY = Identity('script', 'tests', 'async-core')
PLAN = DependencyPlan('event_aware', (DependencyTag.site('production'),), 'supported')


class _Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


@dataclass
class Fixture:
    cache: AsyncQueryCache
    l1: PhaseBackend | None
    l2: PhaseBackend | None
    clock: _Clock
    metrics: RecordingMetrics


async def answer() -> object:
    return 42


def fixture(topology: str = 'local', settings: CacheSettings | None = None) -> Fixture:
    selected = settings or CacheSettings(
        l1=L1Settings(enabled=topology != 'shared'),
        l2=L2Settings(enabled=topology != 'local'),
    )
    metrics = RecordingMetrics()
    clock = _Clock()
    l1 = PhaseBackend(AsyncLocalBackend(selected, clock, metrics)) if selected.l1.enabled else None
    l2 = PhaseBackend(AsyncLocalBackend(selected, clock, metrics)) if selected.l2.enabled else None
    cache = AsyncQueryCache(selected, IDENTITY, l1=l1, l2=l2, metrics=metrics)
    return Fixture(cache, l1, l2, clock, metrics)


def parse(value: object) -> object:
    return ('parsed', value)


async def read(cache: AsyncQueryCache) -> object:
    return await cache.read(CALL, PLAN, answer, CLIENT_OPTIONS)
