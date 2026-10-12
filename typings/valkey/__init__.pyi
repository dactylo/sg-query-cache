from collections.abc import Callable, Mapping
from typing import Self

from valkey.connection import ConnectionPool
from valkey.retry import Retry

class Valkey:
    connection_pool: ConnectionPool
    connection: object | None

    def __init__(
        self,
        *,
        host: str = 'localhost',
        port: int = 6379,
        db: int = 0,
        password: str | None = None,
        socket_timeout: float | None = None,
        socket_connect_timeout: float | None = None,
        decode_responses: bool = False,
        retry_on_timeout: bool = False,
        ssl: bool = False,
        ssl_check_hostname: bool = False,
        username: str | None = None,
        retry: Retry | None = None,
        valkey_connect_func: Callable[[object], None] | None = None,
    ) -> None: ...
    def client(self) -> Self: ...
    def close(self) -> None: ...
    def delete(self, *names: bytes | str) -> int: ...
    def evalsha(self, sha: str, numkeys: int, *keys_and_args: bytes | str | int) -> object: ...
    def execute_command(self, *args: bytes | str | int) -> object: ...
    def exists(self, key: bytes | str) -> int: ...
    def get(self, key: bytes | str) -> bytes | str | None: ...
    def info(self, section: str | None = None) -> Mapping[str | bytes, object]: ...
    def script_load(self, script: bytes | str) -> bytes | str: ...
    def set(self, name: bytes | str, value: bytes | str | float) -> bool: ...

class Sentinel:
    sentinels: list[Valkey]

    def __init__(
        self,
        sentinels: tuple[tuple[str, int], ...],
        min_other_sentinels: int = 0,
        sentinel_kwargs: dict[str, object] | None = None,
        **connection_kwargs: object,
    ) -> None: ...
    def discover_master(self, service_name: str) -> tuple[str, int]: ...
    def master_for(self, service_name: str, **kwargs: object) -> Valkey: ...
