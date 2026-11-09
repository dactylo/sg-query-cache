from collections.abc import Callable, Generator, Mapping
from contextlib import contextmanager
from threading import Barrier, Event, Lock
from typing import TypeVar, cast

import pytest

import sg_query_cache._config as config
import sg_query_cache.valkey as valkey_module
from sg_query_cache._config import (
    CacheSettings,
    L1Settings,
    L2Settings,
    ValkeySettings,
    _ValkeyMode,
)
from sg_query_cache._valkey_scripts import (
    CONTROL_CREATE_SCRIPT_SHA,
    CONTROL_ROTATE_SCRIPT_SHA,
    ScriptClient,
)
from tests._valkey_control_cases import (
    COUNTER_A,
    COUNTER_B,
    ControlTrace,
    control_reply,
)

_T = TypeVar('_T')

_PREFIX = 'sgqc'
_REPLICATION_A = 'a' * 40
_REPLICATION_B = 'b' * 40


class AtomicControlClient:
    def __init__(
        self,
        *,
        replication_id: str = _REPLICATION_A,
    ) -> None:
        self.connection_generation = 0
        self.control: bytes | None = None
        self.counters: set[str] = set()
        self.info_calls = 0
        self.lock = Lock()
        self.metadata_replies: list[object] = []
        self.pre_rotation_control: bytes | None = None
        self.replication_id = replication_id

    def client(self) -> '_BorrowedAtomicControlClient':
        return _BorrowedAtomicControlClient(self)

    def close(self) -> None:
        return None

    def epoch(self) -> tuple[int, int, None]:
        return 0, self.connection_generation, None

    def evalsha(self, sha: str, numkeys: int, *keys_and_args: object) -> object:
        if sha not in (CONTROL_CREATE_SCRIPT_SHA, CONTROL_ROTATE_SCRIPT_SHA):
            if self.metadata_replies:
                self.pre_rotation_control = self.control
                return self.metadata_replies.pop(0)
            return [b'ok', 1, 0]
        with self.lock:
            self.control, reply = control_reply(
                sha, numkeys, keys_and_args, self.control, self.counters
            )
            return reply

    def exists(self, key: str) -> object:
        with self.lock:
            return int(key in self.counters)

    def get(self, key: str) -> object:
        _ = key
        with self.lock:
            return self.control

    def info(self, section: str) -> object:
        assert section == 'replication'
        self.info_calls += 1
        return {'role': 'master', 'master_replid': self.replication_id}

    @contextmanager
    def pin(self) -> Generator['AtomicControlClient']:
        yield self

    def refresh(self) -> None:
        return None

    def refresh_endpoint(self) -> None:
        return None

    def run_operation(self, operation: Callable[[ScriptClient], _T]) -> _T:
        return operation(self)

    def script_load(self, script: bytes | str) -> object:
        raise AssertionError(script)


class EndpointControlClient:
    def __init__(
        self,
        name: str,
        state: AtomicControlClient,
        endpoint: list[str],
        fenced_script_endpoints: list[str],
    ) -> None:
        self.close_calls = 0
        self._endpoint = endpoint
        self._fenced_script_endpoints = fenced_script_endpoints
        self._name = name
        self._state = state

    def client(self) -> 'EndpointControlClient':
        return self

    def close(self) -> None:
        self.close_calls += 1

    def evalsha(self, sha: str, numkeys: int, *keys_and_args: object) -> object:
        if sha in (CONTROL_CREATE_SCRIPT_SHA, CONTROL_ROTATE_SCRIPT_SHA):
            return self._state.evalsha(sha, numkeys, *keys_and_args)
        self._fenced_script_endpoints.append(self._name)
        return [b'ok', 1]

    def exists(self, key: str) -> object:
        result = self._state.exists(key)
        if self._endpoint[0] == 'old':
            self._endpoint[0] = 'new'
        return result

    def get(self, key: str) -> object:
        return self._state.get(key)

    def info(self, section: str) -> object:
        return self._state.info(section)

    def script_load(self, script: bytes | str) -> object:
        raise AssertionError(script)


class ReconnectObservable:
    def __init__(self) -> None:
        self._callbacks: list[Callable[[object], None]] = []

    def reconnect(self) -> None:
        for callback in self._callbacks:
            callback(self)

    def register_connect_callback(self, callback: Callable[[object], None]) -> None:
        if callback not in self._callbacks:
            self._callbacks.append(callback)


class _BorrowedAtomicControlClient:
    def __init__(self, owner: AtomicControlClient) -> None:
        self.owner = owner

    def close(self) -> None:
        return None

    def evalsha(self, sha: str, numkeys: int, *keys_and_args: object) -> object:
        return self.owner.evalsha(sha, numkeys, *keys_and_args)

    def exists(self, key: str) -> object:
        return self.owner.exists(key)

    def get(self, key: str) -> object:
        return self.owner.get(key)

    def info(self, section: str) -> object:
        return self.owner.info(section)

    def script_load(self, script: bytes | str) -> object:
        return self.owner.script_load(script)


class _FixedPrimaryPool:
    def get_master_address(self) -> tuple[str, int]:
        return 'primary.example', 6379


class _PinnedConnectionView(_BorrowedAtomicControlClient):
    def __init__(self, owner: AtomicControlClient, connection: int) -> None:
        super().__init__(owner)
        self._connection = connection

    def connection_generation(self) -> int:
        return self._connection


class PinnedConnectionClient(_BorrowedAtomicControlClient):
    def __init__(self, *, replication_id: str = _REPLICATION_A) -> None:
        super().__init__(AtomicControlClient(replication_id=replication_id))
        self.pin_connection = 1

    def epoch(self) -> tuple[int, int, None]:
        return self.owner.epoch()

    @contextmanager
    def pin(self) -> Generator[_PinnedConnectionView]:
        yield _PinnedConnectionView(self.owner, self.pin_connection)

    def refresh(self) -> None:
        self.owner.refresh()

    def refresh_endpoint(self) -> None:
        self.owner.refresh_endpoint()


class TraceControlClient:
    def __init__(self, trace: ControlTrace) -> None:
        self.effects = list(trace.effects)
        self._proposal_nonce = trace.proposal_nonce
        self._role_epoch: tuple[int, int, object | None] = (0, 0, None)

    def effect(self, name: str) -> object:
        effect = self.effects.pop(0)
        assert effect.name == name
        if effect.epoch_after is not None:
            self._role_epoch = effect.epoch_after
        return effect.reply

    def epoch(self) -> tuple[int, int, object | None]:
        return self._role_epoch

    def evalsha(self, sha: str, numkeys: int, *keys_and_args: object) -> object:
        assert numkeys == 2
        assert sha in (CONTROL_CREATE_SCRIPT_SHA, CONTROL_ROTATE_SCRIPT_SHA)
        counter = f'sgqc:v1:counter:{self._proposal_nonce}'
        proposal = f'primary|{self._proposal_nonce}|{counter}'.encode()
        expected = ('sgqc:v1:control', counter)
        if sha == CONTROL_ROTATE_SCRIPT_SHA:
            expected += (
                b'primary|00000000000000000000000000000001|sgqc:v1:counter:00000000000000000000000000000001',
            )
        assert keys_and_args == (*expected, proposal)
        return self.effect('create' if sha == CONTROL_CREATE_SCRIPT_SHA else 'rotate')

    def exists(self, key: str) -> object:
        assert key in (COUNTER_A, COUNTER_B)
        return self.effect('exists')

    def get(self, key: str) -> object:
        assert key == 'sgqc:v1:control'
        return self.effect('get')

    def info(self, section: str) -> object:
        assert section == 'replication'
        return self.effect('info')

    @contextmanager
    def pin(self) -> Generator['TraceControlClient']:
        self.effect('pin')
        try:
            yield self
        finally:
            self.effect('unpin')

    def refresh(self) -> None:
        self.effect('refresh')

    def refresh_endpoint(self) -> None:
        self.effect('refresh_endpoint')

    def script_load(self, script: bytes | str) -> object:
        raise AssertionError(script)


class AlreadyConnectedObservable(ReconnectObservable):
    def __init__(self) -> None:
        super().__init__()
        self.callback_observations = 0
        self._connected = False
        self.disconnect_calls = 0
        self.successful_connections = 0

    def connect(self) -> None:
        if self._connected:
            return
        self._connected = True
        self.successful_connections += 1
        for callback in self._callbacks:
            self.callback_observations += 1
            callback(self)

    def disconnect(self) -> None:
        self.disconnect_calls += 1
        self._connected = False


class ReconnectingBorrowedClient(_BorrowedAtomicControlClient):
    def __init__(self, owner: AtomicControlClient, connection: object) -> None:
        super().__init__(owner)
        self.connection = connection


class PreObservedClient(AtomicControlClient):
    def __init__(self, callback: Callable[[object], None]) -> None:
        super().__init__()
        self.connection = _PreObservedConnection(callback)

    def client(self) -> ReconnectingBorrowedClient:
        self.connection.connect()
        return ReconnectingBorrowedClient(self, self.connection)


class ConnectedAtomicControlClient(AtomicControlClient):
    def __init__(self) -> None:
        super().__init__()
        self.close_calls = 0
        self.connection = ReconnectObservable()
        self.metadata_script_calls = 0

    def client(self) -> ReconnectingBorrowedClient:
        return ReconnectingBorrowedClient(self, self.connection)

    def close(self) -> None:
        self.close_calls += 1


class AlreadyConnectedAtomicClient(ConnectedAtomicControlClient):
    def __init__(self) -> None:
        super().__init__()
        self.connection = AlreadyConnectedObservable()

    def client(self) -> ReconnectingBorrowedClient:
        self.connection.connect()
        return ReconnectingBorrowedClient(self, self.connection)


class PooledResultChild(_BorrowedAtomicControlClient):
    def __init__(self, owner: 'PooledResultClient', index: int) -> None:
        super().__init__(owner)
        self.connection = AlreadyConnectedObservable()
        self._index = index

    def evalsha(self, sha: str, numkeys: int, *keys_and_args: object) -> object:
        _ = sha, numkeys, keys_and_args
        owner = cast('PooledResultClient', self.owner)
        if owner.barrier is not None:
            owner.barrier.wait()
        if self._index in owner.reconnect_indexes:
            self.connection.disconnect()
            self.connection.connect()
        return [b'child', self._index]


class PooledResultClient(AtomicControlClient):
    def __init__(
        self,
        *,
        barrier: Barrier | None = None,
        reconnect_indexes: set[int] | None = None,
    ) -> None:
        super().__init__()
        self.barrier = barrier
        self.children: list[PooledResultChild] = []
        self.close_calls = 0
        self.reconnect_indexes: set[int] = (
            set() if reconnect_indexes is None else reconnect_indexes
        )

    def client(self) -> PooledResultChild:
        with self.lock:
            child = PooledResultChild(self, len(self.children))
            self.children.append(child)
        child.connection.connect()
        return child

    def close(self) -> None:
        self.close_calls += 1


class PooledMetadataChild(_BorrowedAtomicControlClient):
    def __init__(self, owner: 'PooledMetadataClient', index: int) -> None:
        super().__init__(owner)
        self.connection = AlreadyConnectedObservable()
        self._index = index

    def evalsha(self, sha: str, numkeys: int, *keys_and_args: object) -> object:
        owner = cast('PooledMetadataClient', self.owner)
        if sha in (CONTROL_CREATE_SCRIPT_SHA, CONTROL_ROTATE_SCRIPT_SHA):
            return owner.evalsha(sha, numkeys, *keys_and_args)
        if self._index == 0:
            owner.first_operation_entered.set()
            assert owner.release_first_operation.wait(timeout=5)
        return [b'ok', 1]


class PooledMetadataClient(AtomicControlClient):
    def __init__(self) -> None:
        super().__init__()
        self.children: list[PooledMetadataChild] = []
        self.close_calls = 0
        self.first_operation_entered = Event()
        self.release_first_operation = Event()

    def client(self) -> PooledMetadataChild:
        with self.lock:
            child = PooledMetadataChild(self, len(self.children))
            self.children.append(child)
        child.connection.connect()
        return child

    def close(self) -> None:
        self.close_calls += 1


class ReconnectingClient(AtomicControlClient):
    def __init__(self) -> None:
        super().__init__()
        self.connection = ReconnectObservable()
        self._connect_pending = True

    def _connect(self) -> None:
        if not self._connect_pending:
            return
        self._connect_pending = False
        self.connection.reconnect()

    def client(self) -> _BorrowedAtomicControlClient:
        self._connect()
        return ReconnectingBorrowedClient(self, self.connection)

    def evalsha(self, sha: str, numkeys: int, *keys_and_args: object) -> object:
        self._connect()
        return super().evalsha(sha, numkeys, *keys_and_args)

    def info(self, section: str) -> object:
        self._connect()
        return super().info(section)

    def reconnect_to(self, replication_id: str) -> None:
        self.replication_id = replication_id
        self._connect_pending = True


class _PreObservedConnection:
    def __init__(self, callback: Callable[[object], None]) -> None:
        self._callback = callback
        self.connect_calls = 0
        self._connected = False
        self.disconnect_calls = 0
        self._setup_complete = False

    def __hash__(self) -> int:
        assert self._setup_complete
        return object.__hash__(self)

    def connect(self) -> None:
        if self._connected:
            return
        self.connect_calls += 1
        self._connected = True
        self._callback(self)

    def disconnect(self) -> None:
        self.disconnect_calls += 1
        self._connected = False

    def on_connect(self) -> None:
        self._setup_complete = True

    def register_connect_callback(self, _callback: Callable[[object], None]) -> None:
        raise AssertionError('pre-observed production connection used fallback callback')


class _ProvingConnection:
    def __init__(self, owner: AtomicControlClient, hook: Callable[[object], None]) -> None:
        self.connected = False
        self.disconnect_calls = 0
        self._hook = hook
        self._owner = owner
        self.sent: list[tuple[object, ...]] = []

    def connect(self) -> None:
        if self.connected:
            return
        self.connected = True
        self._hook(self)

    def disconnect(self) -> None:
        self.disconnect_calls += 1
        self.connected = False

    def on_connect(self) -> None:
        return None

    def read_response(self) -> object:
        raw = self._owner.info('replication')
        return raw if isinstance(raw, bytes) else _info_reply(cast('Mapping[str, object]', raw))

    def send_command(self, *args: object) -> None:
        self.sent.append(args)


def _info_reply(fields: Mapping[str, object]) -> bytes:
    return b''.join(f'{name}:{value}\r\n'.encode() for name, value in fields.items())


def enabled_settings(mode: config._ValkeyMode = 'standalone') -> CacheSettings:
    result = config.ValkeyRoleSettings(
        'result.example',
        'RESULT_PASSWORD',
        mode=mode,
        sentinels=('sentinel-a:26379', 'sentinel-b:26379', 'sentinel-c:26379')
        if mode == 'sentinel'
        else (),
        service_name='result-primary' if mode == 'sentinel' else None,
    )
    metadata = config.ValkeyRoleSettings(
        'metadata.example',
        'METADATA_PASSWORD',
        mode=mode,
        sentinels=('sentinel-a:26379', 'sentinel-b:26379', 'sentinel-c:26379')
        if mode == 'sentinel'
        else (),
        service_name='metadata-primary' if mode == 'sentinel' else None,
    )
    return CacheSettings(
        l1=L1Settings(enabled=False),
        l2=L2Settings(enabled=True, valkey=ValkeySettings(result=result, metadata=metadata)),
    )


class ProvingClient(AtomicControlClient):
    def __init__(
        self, hook: Callable[[object], None], *, replication_id: str = _REPLICATION_A
    ) -> None:
        super().__init__(replication_id=replication_id)
        self.close_calls = 0
        self.connection = _ProvingConnection(self, hook)
        self.connection_pool = _FixedPrimaryPool()

    def client(self) -> ReconnectingBorrowedClient:
        self.connection.connect()
        return ReconnectingBorrowedClient(self, self.connection)

    def close(self) -> None:
        self.close_calls += 1


def install_proving_clients(
    monkeypatch: pytest.MonkeyPatch,
    mode: _ValkeyMode,
    factories: list[type[ProvingClient]],
) -> list[ProvingClient]:
    clients: list[ProvingClient] = []
    pending = iter(factories)

    def create_client(**kwargs: object) -> ProvingClient:
        client = next(pending)(cast('Callable[[object], None]', kwargs['valkey_connect_func']))
        clients.append(client)
        return client

    class _Sentinel:
        def __init__(self, _sentinels: object, **kwargs: object) -> None:
            self._kwargs = kwargs

        def master_for(self, _name: str) -> ProvingClient:
            return create_client(**self._kwargs)

    monkeypatch.setattr(
        valkey_module,
        'Sentinel' if mode == 'sentinel' else 'Valkey',
        _Sentinel if mode == 'sentinel' else create_client,
    )
    return clients


def nonce(index: int) -> str:
    return f'{index:032x}'
