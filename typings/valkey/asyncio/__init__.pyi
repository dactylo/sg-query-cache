from typing import Self

from valkey.asyncio.connection import AbstractConnection, ConnectionPool
from valkey.asyncio.connection import Connection as Connection

class Valkey:
    connection: AbstractConnection | None
    connection_pool: ConnectionPool

    @classmethod
    def from_pool(cls, connection_pool: ConnectionPool) -> Self: ...
    def client(self) -> Self: ...
    async def aclose(self, close_connection_pool: bool | None = None) -> None: ...
    async def initialize(self) -> Self: ...

class Sentinel:
    sentinels: list[Valkey]

    def __init__(
        self,
        sentinels: tuple[tuple[str, int], ...],
        min_other_sentinels: int = 0,
        **connection_kwargs: object,
    ) -> None: ...
    async def discover_master(self, service_name: str) -> tuple[str, int]: ...
    def master_for(self, service_name: str, **kwargs: object) -> Valkey: ...
