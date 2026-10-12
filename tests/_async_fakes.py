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


class _Phase:
    def __init__(self) -> None:
        self._entered = asyncio.Event()
        self._proceed = asyncio.Event()
        self._cancelled = asyncio.Event()
        self._cleanup_proceed = asyncio.Event()
        self._cleanup_proceed.set()
        self._finished = asyncio.Event()
        self._error: Exception | None = None
        self._on_enter: Callable[[], None] | None = None

    async def _wait(self) -> None:
        self._entered.set()
        try:
            if self._on_enter is not None:
                self._on_enter()
            await self._proceed.wait()
            if self._error is not None:
                raise self._error
        except asyncio.CancelledError:
            self._cancelled.set()
            await self._cleanup_proceed.wait()
            raise
        finally:
            self._finished.set()


class _PhaseBackend:  # pyright: ignore[reportUnusedClass]
    def __init__(self, backend: AsyncCacheBackend) -> None:
        self._backend = backend
        self._before: dict[str, _Phase] = {}
        self._after: dict[str, _Phase] = {}
        self._calls: list[str] = []

    async def _call(self, name: str, action: Callable[[], Awaitable[_T]]) -> _T:
        self._calls.append(name)
        if phase := self._before.get(name):
            await phase._wait()
        value = await action()
        if phase := self._after.get(name):
            await phase._wait()
        return value

    async def change(self, changes: tuple[GenerationChange, ...]) -> None:
        await self._call('change', lambda: self._backend.change(changes))

    async def close(self) -> None:
        await self._call('close', self._backend.close)

    async def commit(self, lease: FillLease, payload: bytes, remaining_seconds: float) -> bool:
        return await self._call(
            'commit', lambda: self._backend.commit(lease, payload, remaining_seconds)
        )

    async def flush_site(self, site_namespace: str, reason: str) -> None:
        await self._call('flush_site', lambda: self._backend.flush_site(site_namespace, reason))

    async def lookup(self, key: str) -> CacheHit | None:
        return await self._call('lookup', lambda: self._backend.lookup(key))

    async def release(self, lease: FillLease) -> bool:
        return await self._call('release', lambda: self._backend.release(lease))

    async def reserve(self, key: str, lease_seconds: float) -> FillLease | None:
        return await self._call('reserve', lambda: self._backend.reserve(key, lease_seconds))

    async def resolve(self, tags: tuple[DependencyTag, ...]) -> GenerationVector:
        return await self._call('resolve', lambda: self._backend.resolve(tags))


class _RecordingMetrics:  # pyright: ignore[reportUnusedClass]
    def __init__(self) -> None:
        self._events: list[tuple[str, float | None, dict[str, str]]] = []
        self._callback: Callable[[], None] | None = None

    def increment(self, name: str, labels: Mapping[str, str]) -> None:
        self._events.append((name, None, dict(labels)))
        if self._callback is not None:
            self._callback()

    def observe(self, name: str, value: float, labels: Mapping[str, str]) -> None:
        self._events.append((name, value, dict(labels)))
        if self._callback is not None:
            self._callback()
