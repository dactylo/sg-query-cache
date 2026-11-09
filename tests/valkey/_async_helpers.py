import asyncio
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import TypeVar

from sg_query_cache._async_valkey_control import PinnedControlClient
from tests._valkey_control_cases import OrderedClient

_T = TypeVar('_T')


class BackendClient:
    def __init__(self, replies: list[tuple[str, object]]) -> None:
        self.source = OrderedClient(replies)
        self.closed = 0
        self.close_error: Exception | None = None
        self.close_entered = asyncio.Event()
        self.close_proceed = asyncio.Event()
        self.close_proceed.set()

    async def close(self) -> None:
        self.closed += 1
        self.close_entered.set()
        await self.close_proceed.wait()
        if self.close_error is not None:
            raise self.close_error

    def epoch(self) -> tuple[int, int, None]:
        return 0, 0, None

    async def evalsha(self, sha: str, numkeys: int, *args: object) -> object:
        return self.source.evalsha(sha, numkeys, *args)

    async def exists(self, key: str) -> object:
        return self.source.exists(key)

    async def get(self, key: str) -> object:
        return self.source.get(key)

    async def info(self, section: str) -> object:
        return self.source.info(section)

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
        return self.source.script_load(script)
