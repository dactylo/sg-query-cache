import asyncio
from collections.abc import Awaitable, Callable, Mapping
from typing import TypeVar

from sg_query_cache._async_backend import AsyncCacheBackend
from sg_query_cache._model import (
    CacheHit,
    DependencyTag,
    FillLease,
    GenerationChange,
    GenerationVector,
)

_T = TypeVar('_T')


class Phase:
    def __init__(self) -> None:
        self.entered = asyncio.Event()
        self.proceed = asyncio.Event()
        self.cancelled = asyncio.Event()
        self.cleanup_proceed = asyncio.Event()
        self.cleanup_proceed.set()
        self.finished = asyncio.Event()
        self.error: Exception | None = None
        self.on_enter: Callable[[], None] | None = None

    async def wait(self) -> None:
        self.entered.set()
        try:
            if self.on_enter is not None:
                self.on_enter()
            await self.proceed.wait()
            if self.error is not None:
                raise self.error
        except asyncio.CancelledError:
            self.cancelled.set()
            await self.cleanup_proceed.wait()
            raise
        finally:
            self.finished.set()


class PhaseBackend:
    def __init__(self, backend: AsyncCacheBackend) -> None:
        self.backend = backend
        self.before: dict[str, Phase] = {}
        self.after: dict[str, Phase] = {}
        self.calls: list[str] = []

    async def _call(self, name: str, action: Callable[[], Awaitable[_T]]) -> _T:
        self.calls.append(name)
        if phase := self.before.get(name):
            await phase.wait()
        value = await action()
        if phase := self.after.get(name):
            await phase.wait()
        return value

    async def change(self, changes: tuple[GenerationChange, ...]) -> None:
        await self._call('change', lambda: self.backend.change(changes))

    async def close(self) -> None:
        await self._call('close', self.backend.close)

    async def commit(self, lease: FillLease, payload: bytes, remaining_seconds: float) -> bool:
        return await self._call(
            'commit', lambda: self.backend.commit(lease, payload, remaining_seconds)
        )

    async def flush_site(self, site_namespace: str, reason: str) -> None:
        await self._call('flush_site', lambda: self.backend.flush_site(site_namespace, reason))

    async def lookup(self, key: str) -> CacheHit | None:
        return await self._call('lookup', lambda: self.backend.lookup(key))

    async def release(self, lease: FillLease) -> bool:
        return await self._call('release', lambda: self.backend.release(lease))

    async def reserve(self, key: str, lease_seconds: float) -> FillLease | None:
        return await self._call('reserve', lambda: self.backend.reserve(key, lease_seconds))

    async def resolve(self, tags: tuple[DependencyTag, ...]) -> GenerationVector:
        return await self._call('resolve', lambda: self.backend.resolve(tags))


class RecordingMetrics:
    def __init__(self) -> None:
        self.events: list[tuple[str, float | None, dict[str, str]]] = []
        self.callback: Callable[[], None] | None = None

    def increment(self, name: str, labels: Mapping[str, str]) -> None:
        self.events.append((name, None, dict(labels)))
        if self.callback is not None:
            self.callback()

    def observe(self, name: str, value: float, labels: Mapping[str, str]) -> None:
        self.events.append((name, value, dict(labels)))
        if self.callback is not None:
            self.callback()
