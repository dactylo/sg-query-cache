import asyncio
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import TypeVar

from sg_query_cache._async_valkey_control import PinnedControlClient
from tests._valkey_control_cases import _OrderedClient

_T = TypeVar('_T')


class BackendClient:
    def __init__(self, replies: list[tuple[str, object]]) -> None:
        self._source = _OrderedClient(replies)
        self._closed = 0
        self._close_error: Exception | None = None
        self._close_entered = asyncio.Event()
        self._close_proceed = asyncio.Event()
        self._close_proceed.set()

    async def close(self) -> None:
        self._closed += 1
        self._close_entered.set()
        await self._close_proceed.wait()
        if self._close_error is not None:
            raise self._close_error

    def epoch(self) -> tuple[int, int, None]:
        return 0, 0, None

    async def evalsha(self, sha: str, numkeys: int, *args: object) -> object:
        return self._source.evalsha(sha, numkeys, *args)

    async def exists(self, key: str) -> object:
        return self._source.exists(key)

    async def get(self, key: str) -> object:
        return self._source.get(key)

    async def info(self, section: str) -> object:
        return self._source.info(section)

    @asynccontextmanager
    async def pin(self) -> AsyncGenerator['BackendClient']:
        yield self

    async def refresh(self) -> None:
        return None

    async def refresh_endpoint(self) -> None:
        return None

    async def run_operation(
        self,
        operation: Callable[[PinnedControlClient], Awaitable[_T]],
    ) -> _T:
        return await operation(self)

    async def script_load(self, script: bytes | str) -> object:
        return self._source.script_load(script)
