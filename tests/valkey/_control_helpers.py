from collections.abc import Callable, Generator, Mapping
from contextlib import contextmanager
from threading import Barrier, Event, Lock
from typing import TypeVar, cast

import sg_query_cache._config as config
from sg_query_cache._config import CacheSettings, L1Settings, L2Settings, ValkeySettings
from sg_query_cache._valkey_scripts import (
    CONTROL_CREATE_SCRIPT_SHA,
    CONTROL_ROTATE_SCRIPT_SHA,
    ScriptClient,
)
from tests._valkey_control_cases import (
    _COUNTER_A,
    _COUNTER_B,
    _ControlTrace,
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
        self._connection_generation = 0
        self._control: bytes | None = None
        self._counters: set[str] = set()
        self._info_calls = 0
        self._lock = Lock()
        self._metadata_replies: list[object] = []
        self._pre_rotation_control: bytes | None = None
        self._replication_id = replication_id

    def client(self) -> 'BorrowedAtomicControlClient':
        return BorrowedAtomicControlClient(self)

    def close(self) -> None:
        return None

    def epoch(self) -> tuple[int, int, None]:
        return 0, self._connection_generation, None

    def evalsha(self, sha: str, numkeys: int, *keys_and_args: object) -> object:
        if sha not in (CONTROL_CREATE_SCRIPT_SHA, CONTROL_ROTATE_SCRIPT_SHA):
            if self._metadata_replies:
                self._pre_rotation_control = self._control
                return self._metadata_replies.pop(0)
            return [b'ok', 1, 0]
        with self._lock:
            self._control, reply = control_reply(
                sha, numkeys, keys_and_args, self._control, self._counters
            )
            return reply

    def exists(self, key: str) -> object:
        with self._lock:
            return int(key in self._counters)

    def get(self, key: str) -> object:
        _ = key
        with self._lock:
            return self._control

    def info(self, section: str) -> object:
        assert section == 'replication'
        self._info_calls += 1
        return {'role': 'master', 'master_replid': self._replication_id}

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


class BorrowedAtomicControlClient:
    def __init__(self, owner: AtomicControlClient) -> None:
        self._owner = owner

    def close(self) -> None:
        return None

    def evalsha(self, sha: str, numkeys: int, *keys_and_args: object) -> object:
        return self._owner.evalsha(sha, numkeys, *keys_and_args)

    def exists(self, key: str) -> object:
        return self._owner.exists(key)

    def get(self, key: str) -> object:
        return self._owner.get(key)

    def info(self, section: str) -> object:
        return self._owner.info(section)

    def script_load(self, script: bytes | str) -> object:
        return self._owner.script_load(script)


class EndpointControlClient:
    def __init__(
        self,
        name: str,
        state: AtomicControlClient,
        endpoint: list[str],
        fenced_script_endpoints: list[str],
    ) -> None:
        self._close_calls = 0
        self._endpoint = endpoint
        self._fenced_script_endpoints = fenced_script_endpoints
        self._name = name
        self._state = state

    def client(self) -> 'EndpointControlClient':
        return self

    def close(self) -> None:
        self._close_calls += 1

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


class FixedPrimaryPool:
    def get_master_address(self) -> tuple[str, int]:
        return 'primary.example', 6379


class PinnedConnectionView(BorrowedAtomicControlClient):
    def __init__(self, owner: AtomicControlClient, connection: int) -> None:
        super().__init__(owner)
        self._connection = connection

    def connection_generation(self) -> int:
        return self._connection


class PinnedConnectionClient(BorrowedAtomicControlClient):
    def __init__(self, *, replication_id: str = _REPLICATION_A) -> None:
        super().__init__(AtomicControlClient(replication_id=replication_id))
        self._pin_connection = 1

    def epoch(self) -> tuple[int, int, None]:
        return self._owner.epoch()

    @contextmanager
    def pin(self) -> Generator[PinnedConnectionView]:
        yield PinnedConnectionView(self._owner, self._pin_connection)

    def refresh(self) -> None:
        self._owner.refresh()

    def refresh_endpoint(self) -> None:
        self._owner.refresh_endpoint()


class PreObservedConnection:
    def __init__(self, callback: Callable[[object], None]) -> None:
        self._callback = callback
        self._connect_calls = 0
        self._connected = False
        self._disconnect_calls = 0
        self._setup_complete = False

    def __hash__(self) -> int:
        assert self._setup_complete
        return object.__hash__(self)

    def connect(self) -> None:
        if self._connected:
            return
        self._connect_calls += 1
        self._connected = True
        self._callback(self)

    def disconnect(self) -> None:
        self._disconnect_calls += 1
        self._connected = False

    def on_connect(self) -> None:
        self._setup_complete = True

    def register_connect_callback(self, _callback: Callable[[object], None]) -> None:
        raise AssertionError('pre-observed production connection used fallback callback')


class ProvingConnection:
    def __init__(self, owner: AtomicControlClient, hook: Callable[[object], None]) -> None:
        self._connected = False
        self._disconnect_calls = 0
        self._hook = hook
        self._owner = owner
        self._sent: list[tuple[object, ...]] = []

    def connect(self) -> None:
        if self._connected:
            return
        self._connected = True
        self._hook(self)

    def disconnect(self) -> None:
        self._disconnect_calls += 1
        self._connected = False

    def on_connect(self) -> None:
        return None

    def read_response(self) -> object:
        raw = self._owner.info('replication')
        return raw if isinstance(raw, bytes) else info_reply(cast('Mapping[str, object]', raw))

    def send_command(self, *args: object) -> None:
        self._sent.append(args)


class ReconnectObservable:
    def __init__(self) -> None:
        self._callbacks: list[Callable[[object], None]] = []

    def _reconnect(self) -> None:
        for callback in self._callbacks:
            callback(self)

    def register_connect_callback(self, callback: Callable[[object], None]) -> None:
        if callback not in self._callbacks:
            self._callbacks.append(callback)


class TraceControlClient:
    def __init__(self, trace: _ControlTrace) -> None:
        self._effects = list(trace._effects)
        self._proposal_nonce = trace._proposal_nonce
        self._role_epoch: tuple[int, int, object | None] = (0, 0, None)

    def _effect(self, name: str) -> object:
        effect = self._effects.pop(0)
        assert effect._name == name
        if effect._epoch_after is not None:
            self._role_epoch = effect._epoch_after
        return effect._reply

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
        return self._effect('create' if sha == CONTROL_CREATE_SCRIPT_SHA else 'rotate')

    def exists(self, key: str) -> object:
        assert key in (_COUNTER_A, _COUNTER_B)
        return self._effect('exists')

    def get(self, key: str) -> object:
        assert key == 'sgqc:v1:control'
        return self._effect('get')

    def info(self, section: str) -> object:
        assert section == 'replication'
        return self._effect('info')

    @contextmanager
    def pin(self) -> Generator['TraceControlClient']:
        self._effect('pin')
        try:
            yield self
        finally:
            self._effect('unpin')

    def refresh(self) -> None:
        self._effect('refresh')

    def refresh_endpoint(self) -> None:
        self._effect('refresh_endpoint')

    def script_load(self, script: bytes | str) -> object:
        raise AssertionError(script)


class AlreadyConnectedObservable(ReconnectObservable):
    def __init__(self) -> None:
        super().__init__()
        self._callback_observations = 0
        self._connected = False
        self._disconnect_calls = 0
        self._successful_connections = 0

    def connect(self) -> None:
        if self._connected:
            return
        self._connected = True
        self._successful_connections += 1
        for callback in self._callbacks:
            self._callback_observations += 1
            callback(self)

    def disconnect(self) -> None:
        self._disconnect_calls += 1
        self._connected = False


class ReconnectingBorrowedClient(BorrowedAtomicControlClient):
    def __init__(self, owner: AtomicControlClient, connection: object) -> None:
        super().__init__(owner)
        self.connection = connection


class PreObservedClient(AtomicControlClient):
    def __init__(self, callback: Callable[[object], None]) -> None:
        super().__init__()
        self.connection = PreObservedConnection(callback)

    def client(self) -> ReconnectingBorrowedClient:
        self.connection.connect()
        return ReconnectingBorrowedClient(self, self.connection)


class ConnectedAtomicControlClient(AtomicControlClient):
    def __init__(self) -> None:
        super().__init__()
        self._close_calls = 0
        self.connection = ReconnectObservable()
        self._metadata_script_calls = 0

    def client(self) -> ReconnectingBorrowedClient:
        return ReconnectingBorrowedClient(self, self.connection)

    def close(self) -> None:
        self._close_calls += 1


class AlreadyConnectedAtomicClient(ConnectedAtomicControlClient):
    def __init__(self) -> None:
        super().__init__()
        self.connection = AlreadyConnectedObservable()

    def client(self) -> ReconnectingBorrowedClient:
        self.connection.connect()
        return ReconnectingBorrowedClient(self, self.connection)


class PooledResultChild(BorrowedAtomicControlClient):
    def __init__(self, owner: 'PooledResultClient', index: int) -> None:
        super().__init__(owner)
        self.connection = AlreadyConnectedObservable()
        self._index = index

    def evalsha(self, sha: str, numkeys: int, *keys_and_args: object) -> object:
        _ = sha, numkeys, keys_and_args
        owner = cast('PooledResultClient', self._owner)
        if owner._barrier is not None:
            owner._barrier.wait()
        if self._index in owner._reconnect_indexes:
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
        self._barrier = barrier
        self._children: list[PooledResultChild] = []
        self._close_calls = 0
        self._reconnect_indexes: set[int] = (
            set() if reconnect_indexes is None else reconnect_indexes
        )

    def client(self) -> PooledResultChild:
        with self._lock:
            child = PooledResultChild(self, len(self._children))
            self._children.append(child)
        child.connection.connect()
        return child

    def close(self) -> None:
        self._close_calls += 1


class PooledMetadataChild(BorrowedAtomicControlClient):
    def __init__(self, owner: 'PooledMetadataClient', index: int) -> None:
        super().__init__(owner)
        self.connection = AlreadyConnectedObservable()
        self._index = index

    def evalsha(self, sha: str, numkeys: int, *keys_and_args: object) -> object:
        owner = cast('PooledMetadataClient', self._owner)
        if sha in (CONTROL_CREATE_SCRIPT_SHA, CONTROL_ROTATE_SCRIPT_SHA):
            return owner.evalsha(sha, numkeys, *keys_and_args)
        if self._index == 0:
            owner._first_operation_entered.set()
            assert owner._release_first_operation.wait(timeout=5)
        return [b'ok', 1]


class PooledMetadataClient(AtomicControlClient):
    def __init__(self) -> None:
        super().__init__()
        self._children: list[PooledMetadataChild] = []
        self._close_calls = 0
        self._first_operation_entered = Event()
        self._release_first_operation = Event()

    def client(self) -> PooledMetadataChild:
        with self._lock:
            child = PooledMetadataChild(self, len(self._children))
            self._children.append(child)
        child.connection.connect()
        return child

    def close(self) -> None:
        self._close_calls += 1


class ReconnectingClient(AtomicControlClient):
    def __init__(self) -> None:
        super().__init__()
        self.connection = ReconnectObservable()
        self._connect_pending = True

    def _connect(self) -> None:
        if not self._connect_pending:
            return
        self._connect_pending = False
        self.connection._reconnect()

    def _reconnect_to(self, replication_id: str) -> None:
        self._replication_id = replication_id
        self._connect_pending = True

    def client(self) -> BorrowedAtomicControlClient:
        self._connect()
        return ReconnectingBorrowedClient(self, self.connection)

    def evalsha(self, sha: str, numkeys: int, *keys_and_args: object) -> object:
        self._connect()
        return super().evalsha(sha, numkeys, *keys_and_args)

    def info(self, section: str) -> object:
        self._connect()
        return super().info(section)


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
        self._close_calls = 0
        self.connection = ProvingConnection(self, hook)
        self.connection_pool = FixedPrimaryPool()

    def client(self) -> ReconnectingBorrowedClient:
        self.connection.connect()
        return ReconnectingBorrowedClient(self, self.connection)

    def close(self) -> None:
        self._close_calls += 1


def info_reply(fields: Mapping[str, object]) -> bytes:
    return b''.join(f'{name}:{value}\r\n'.encode() for name, value in fields.items())


def nonce(index: int) -> str:
    return f'{index:032x}'
