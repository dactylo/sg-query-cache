from collections.abc import Callable, Generator, Mapping
from contextlib import contextmanager
from threading import Barrier, Lock, Thread
from typing import cast

import pytest
from valkey.exceptions import ReadOnlyError

import sg_query_cache.valkey as valkey_module
from sg_query_cache._backend import (
    BackendCapacity,
    BackendProtocolError,
    BackendUnavailable,
    StaleIncarnation,
)
from sg_query_cache._codec import encode_value
from sg_query_cache._config import _ValkeyMode
from sg_query_cache._model import DependencyTag, GenerationChange
from sg_query_cache._valkey_common import Incarnation, decode_change_reply
from sg_query_cache._valkey_control import ValkeyControl
from sg_query_cache._valkey_scripts import (
    CONTROL_CREATE_SCRIPT_SHA,
    CONTROL_ROTATE_SCRIPT_SHA,
)
from sg_query_cache.valkey import (
    ValkeyBackend,
    ValkeyMetadataStore,
    _ConnectionGeneration,
    _create_client,
    _RoleClient,
)
from tests._exceptions import exception_text as _exception_text
from tests._fakes import FakeClock, RecordingMetrics
from tests._valkey_control_cases import (
    ACTIVE_CONTROL,
    COUNTER_A,
    COUNTER_B,
    ControlTrace,
    OrderedClient,
    PublicationTrace,
    control_traces,
    publication_traces,
)
from tests.valkey._control_helpers import (
    AlreadyConnectedAtomicClient,
    AlreadyConnectedObservable,
    AtomicControlClient,
    ConnectedAtomicControlClient,
    EndpointControlClient,
    PinnedConnectionClient,
    PooledMetadataChild,
    PooledMetadataClient,
    PreObservedClient,
    ProvingClient,
    ReconnectingClient,
    TraceControlClient,
    enabled_settings,
    nonce,
)
from tests.valkey._control_helpers import install_proving_clients as _install_proving_clients

_PREFIX = 'sgqc'
_REPLICATION_A = 'a' * 40
_REPLICATION_B = 'b' * 40


def _publication_control(
    trace: PublicationTrace,
) -> tuple[PinnedConnectionClient, ValkeyControl, FakeClock]:
    client = PinnedConnectionClient(replication_id='primary')
    client.owner.control = trace.initial_record
    client.owner.counters.add(COUNTER_A)
    clock = FakeClock(start=0.0)
    control = ValkeyControl(
        client,
        key_prefix='sgqc',
        validation_seconds=30,
        clock=clock,
        jitter_factory=lambda seconds: seconds,
    )
    if trace.older_joins:
        control.validate()
        client.pin_connection = 2
        clock.advance(10)
    return client, control, clock


def test_backend_revalidates_before_reusing_a_transparently_reconnected_pool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:

    metadata = ReconnectingClient()
    metadata.metadata_replies.extend(([b'ok', 1], [b'ok', 1]))
    result = ReconnectingClient()
    clients = iter((metadata, result))

    def create_client(**kwargs: object) -> ReconnectingClient:
        _ = kwargs
        return next(clients)

    monkeypatch.setattr(valkey_module, 'Valkey', create_client)
    backend = ValkeyBackend.from_settings(
        enabled_settings('managed'),
        {'METADATA_PASSWORD': 'metadata-secret', 'RESULT_PASSWORD': 'result-secret'},
        RecordingMetrics(),
    )
    tag = DependencyTag.site('site')
    original = backend.resolve((tag,))

    metadata.reconnect_to(_REPLICATION_B)
    replacement = backend.resolve((tag,))

    assert replacement.incarnation != original.incarnation
    assert metadata.control is not None
    assert metadata.control.startswith(f'{_REPLICATION_B}|'.encode())
    assert metadata.info_calls == 4


def test_control_clients_converge_on_one_atomic_create_winner() -> None:
    barrier = Barrier(50)
    client = AtomicControlClient()
    incarnations: list[Incarnation] = []
    failures: list[BaseException] = []
    result_lock = Lock()

    def validate(index: int) -> None:
        try:
            barrier.wait()
            result = ValkeyControl(
                client,
                key_prefix=_PREFIX,
                validation_seconds=30,
                nonce_factory=lambda: nonce(index + 1),
            ).validate()
        except (BackendProtocolError, BackendUnavailable) as error:
            with result_lock:
                failures.append(error)
        else:
            with result_lock:
                incarnations.append(result)

    threads = [Thread(target=validate, args=(index,)) for index in range(50)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert failures == []
    assert len(incarnations) == 50
    assert len({incarnation.nonce for incarnation in incarnations}) == 1


def test_control_clients_converge_on_one_atomic_replacement_winner() -> None:
    client = AtomicControlClient()
    original = ValkeyControl(
        client,
        key_prefix=_PREFIX,
        validation_seconds=30,
        nonce_factory=lambda: nonce(1),
    ).validate()
    client.replication_id = _REPLICATION_B
    barrier = Barrier(50)
    incarnations: list[Incarnation] = []
    failures: list[BaseException] = []
    result_lock = Lock()

    def validate(index: int) -> None:
        try:
            barrier.wait()
            result = ValkeyControl(
                client,
                key_prefix=_PREFIX,
                validation_seconds=30,
                nonce_factory=lambda: nonce(index + 2),
            ).validate()
        except (BackendProtocolError, BackendUnavailable) as error:
            with result_lock:
                failures.append(error)
        else:
            with result_lock:
                incarnations.append(result)

    threads = [Thread(target=validate, args=(index,)) for index in range(50)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert failures == []
    assert len(incarnations) == 50
    assert len({incarnation.nonce for incarnation in incarnations}) == 1
    assert incarnations[0].nonce != original.nonce


@pytest.mark.parametrize('trace', control_traces(), ids=lambda trace: trace.name)
def test_control_executes_shared_semantic_traces(trace: ControlTrace) -> None:
    client = TraceControlClient(trace)
    control = ValkeyControl(
        client,
        key_prefix='sgqc',
        validation_seconds=30,
        clock=lambda: 0.0,
        jitter_factory=lambda seconds: seconds,
        nonce_factory=lambda: trace.proposal_nonce,
    )

    def operation(_pinned: object, incarnation: Incarnation) -> bytes:
        decode_change_reply(client.effect('operation'))
        return incarnation.record

    for expected in trace.acknowledgments:
        if isinstance(expected, bytes):
            assert control.run(operation) == expected
        else:
            with pytest.raises(expected):
                control.run(operation)

    state = control._snapshot()
    assert client.effects == []
    assert (None if state.incarnation is None else state.incarnation.record) == trace.final_record
    assert state.revision == trace.final_revision
    assert state.role_epoch == trace.final_epoch


@pytest.mark.parametrize('trace', publication_traces(), ids=lambda trace: trace.name)
def test_control_executes_shared_publication_traces_with_barriers(
    trace: PublicationTrace,
) -> None:
    client, control, clock = _publication_control(trace)
    entered = Barrier(2, timeout=5)
    settled = Barrier(2, timeout=5)
    older: list[bytes | type[Exception]] = []
    effects: list[str] = []

    def old_operation(_pinned: object, incarnation: Incarnation) -> bytes:
        effects.append('older-operation-entered')
        entered.wait()
        settled.wait()
        return incarnation.record

    def old_run() -> None:
        try:
            older.append(control.run(old_operation))
        except (AssertionError, BackendUnavailable, StaleIncarnation) as error:
            older.append(type(error))
        finally:
            effects.append('older-settled')

    def new_operation(_pinned: object, incarnation: Incarnation) -> bytes:
        effects.append('newer-operation-entered')
        if trace.transition == 'clear':
            decode_change_reply([b'stale'])
        return incarnation.record

    thread = Thread(target=old_run)
    thread.start()
    entered.wait()
    if trace.older_joins:
        client.pin_connection = 3
        clock.advance(10)
    with client.owner.lock:
        client.owner.control = trace.replacement_record
        client.owner.counters.add(COUNTER_B)
    try:
        newer: bytes | type[Exception] = control.run(new_operation, force=True)
    except StaleIncarnation:
        newer = StaleIncarnation
    finally:
        effects.append('newer-settled')
        settled.wait()
        thread.join(timeout=5)

    assert not thread.is_alive()
    assert (*older, newer) == trace.acknowledgments
    assert tuple(effects) == trace.effects
    state = control._snapshot()
    assert (None if state.incarnation is None else state.incarnation.record) == trace.final_record
    assert state.revision == trace.final_revision
    if trace.final_covered is not None:
        assert state.covered == trace.final_covered
        assert state.validated_at == (None if trace.transition == 'clear' else 20.0)


def test_control_forced_validation_resets_connection_coverage() -> None:
    client = PinnedConnectionClient()
    clock = FakeClock(start=0.0)
    control = ValkeyControl(
        client,
        key_prefix=_PREFIX,
        validation_seconds=30,
        clock=clock,
        jitter_factory=lambda seconds: seconds,
    )
    control.validate()
    client.pin_connection = 2
    control.validate()
    assert control._snapshot().covered == frozenset({1, 2})

    client.pin_connection = 1
    control.validate(force=True)
    assert control._snapshot().covered == frozenset({1})
    assert client.owner.info_calls == 6


def test_control_join_does_not_extend_validation_lifetime() -> None:
    client = PinnedConnectionClient()
    clock = FakeClock(start=0.0)
    control = ValkeyControl(
        client,
        key_prefix=_PREFIX,
        validation_seconds=30,
        clock=clock,
        jitter_factory=lambda seconds: seconds,
    )
    control.validate()
    clock.advance(20)
    client.pin_connection = 2
    control.validate()
    assert control._snapshot().validated_at == 0.0
    clock.advance(9)
    client.pin_connection = 1
    control.validate()
    assert client.owner.info_calls == 4

    clock.advance(1)
    client.pin_connection = 2
    control.validate()
    assert client.owner.info_calls == 6
    assert control._snapshot().covered == frozenset({2})
    assert control._snapshot().validated_at == 30.0
    client.pin_connection = 1
    control.validate()
    assert client.owner.info_calls == 8
    assert control._snapshot().covered == frozenset({1, 2})
    assert control._snapshot().validated_at == 30.0


def test_control_keeps_a_container_shaped_metrics_sink() -> None:

    class _CountingSink(RecordingMetrics):
        __slots__ = ()

        def __len__(self) -> int:
            return len(self.increments)

    sink = _CountingSink()
    control = ValkeyControl(
        AtomicControlClient(),
        key_prefix=_PREFIX,
        validation_seconds=30,
        metrics=sink,
    )

    control.validate()

    assert ('cache_incarnation', {'outcome': 'validated'}) in sink.increments


def test_control_reuses_validation_on_alternating_covered_connections() -> None:
    client = PinnedConnectionClient()
    clock = FakeClock(start=0.0)
    control = ValkeyControl(
        client,
        key_prefix=_PREFIX,
        validation_seconds=30,
        clock=clock,
        jitter_factory=lambda seconds: seconds,
    )
    original = control.validate()
    client.pin_connection = 2
    assert control.validate() == original
    assert client.owner.info_calls == 4

    for connection in (1, 2, 1, 2):
        client.pin_connection = connection
        assert control.validate() == original
    assert client.owner.info_calls == 4


def test_control_rotates_for_absence_mismatch_and_missing_counter() -> None:
    client = AtomicControlClient()
    nonces = iter((nonce(1), nonce(2), nonce(3), nonce(4)))
    control = ValkeyControl(
        client,
        key_prefix=_PREFIX,
        validation_seconds=30,
        nonce_factory=lambda: next(nonces),
    )

    first = control.validate()
    client.replication_id = _REPLICATION_B
    second = control.validate(force=True)
    client.counters.remove(second.counter_key)
    third = control.validate(force=True)
    client.control = None
    fourth = control.validate(force=True)

    assert [value.nonce for value in (first, second, third, fourth)] == [
        nonce(1),
        nonce(2),
        nonce(3),
        nonce(4),
    ]


def test_control_validation_uses_one_pinned_physical_connection() -> None:
    class _PinnedClient(AtomicControlClient):
        def __init__(self) -> None:
            super().__init__()
            self._active_connection: int | None = None
            self.command_connections: list[int] = []
            self._next_connection = 0
            self.pin_calls = 0

        def _record_connection(self) -> None:
            if self._active_connection is None:
                self._next_connection += 1
                self.command_connections.append(self._next_connection)
            else:
                self.command_connections.append(self._active_connection)

        @contextmanager
        def pin(self) -> Generator[AtomicControlClient]:
            self.pin_calls += 1
            self._next_connection += 1
            self._active_connection = self._next_connection
            try:
                yield self
            finally:
                self._active_connection = None

        def evalsha(self, sha: str, numkeys: int, *keys_and_args: object) -> object:
            self._record_connection()
            return super().evalsha(sha, numkeys, *keys_and_args)

        def exists(self, key: str) -> object:
            self._record_connection()
            return super().exists(key)

        def get(self, key: str) -> object:
            self._record_connection()
            return super().get(key)

        def info(self, section: str) -> object:
            self._record_connection()
            return super().info(section)

    client = _PinnedClient()

    ValkeyControl(
        client,
        key_prefix=_PREFIX,
        validation_seconds=30,
        nonce_factory=lambda: nonce(1),
    ).validate()

    assert client.pin_calls == 1
    assert len(set(client.command_connections)) == 1


def test_counter_exhaustion_rotates_before_reporting_stale(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata = AtomicControlClient()
    metadata.metadata_replies.append([b'rotation-required'])
    clients = iter((metadata, AtomicControlClient()))

    def create_client(**kwargs: object) -> AtomicControlClient:
        _ = kwargs
        return next(clients)

    monkeypatch.setattr(valkey_module, 'Valkey', create_client)
    backend = ValkeyBackend.from_settings(
        enabled_settings(),
        {'METADATA_PASSWORD': 'secret', 'RESULT_PASSWORD': 'secret'},
        RecordingMetrics(),
    )

    with pytest.raises(StaleIncarnation):
        backend.resolve((DependencyTag.site('site'),))

    assert metadata.pre_rotation_control is not None
    assert metadata.control != metadata.pre_rotation_control


def test_demotion_discards_pool_and_revalidates_before_next_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:

    class _DemotedClient(AtomicControlClient):
        def __init__(self) -> None:
            super().__init__()
            self.close_calls = 0
            self._initial = True

        def close(self) -> None:
            self.close_calls += 1

        def info(self, section: str) -> object:
            _ = section
            if self._initial:
                self._initial = False
                return {'role': 'master', 'master_replid': self.replication_id}
            raise ReadOnlyError('credential-bearing-demotion')

    demoted = _DemotedClient()
    result = AtomicControlClient()
    promoted = AtomicControlClient(replication_id=_REPLICATION_B)
    clients = iter((demoted, promoted))

    def create_client(**kwargs: object) -> AtomicControlClient:
        return result if kwargs['host'] == 'result.example' else next(clients)

    monkeypatch.setattr(valkey_module, 'Valkey', create_client)
    backend = ValkeyBackend.from_settings(
        enabled_settings('managed'),
        {'METADATA_PASSWORD': 'metadata-secret', 'RESULT_PASSWORD': 'result-secret'},
        RecordingMetrics(),
    )
    change = (GenerationChange(DependencyTag.site('site')),)

    with pytest.raises(BackendUnavailable):
        backend.change(change)
    assert demoted.close_calls == 1
    assert promoted.info_calls == 0

    backend.change(change)

    assert promoted.info_calls == 2


def test_drift_discarded_mutation_is_unacknowledged_and_not_retried() -> None:
    class _MutationReconnectClient(ConnectedAtomicControlClient):
        def evalsha(self, sha: str, numkeys: int, *keys_and_args: object) -> object:
            reply = super().evalsha(sha, numkeys, *keys_and_args)
            if sha not in (CONTROL_CREATE_SCRIPT_SHA, CONTROL_ROTATE_SCRIPT_SHA):
                self.metadata_script_calls += 1
                self.connection.reconnect()
            return reply

    client = _MutationReconnectClient()
    role = _RoleClient(lambda: client, _ConnectionGeneration())
    store = ValkeyMetadataStore(role, enabled_settings('managed'), RecordingMetrics())

    with pytest.raises(BackendUnavailable, match='changed during operation'):
        store.change((GenerationChange(DependencyTag.site('site')),))

    assert client.metadata_script_calls == 1
    assert client.close_calls == 1


def test_drift_discards_semantic_exception_and_pool_before_reraise() -> None:
    class _CapacityReconnectClient(ConnectedAtomicControlClient):
        def evalsha(self, sha: str, numkeys: int, *keys_and_args: object) -> object:
            if sha in (CONTROL_CREATE_SCRIPT_SHA, CONTROL_ROTATE_SCRIPT_SHA):
                return super().evalsha(sha, numkeys, *keys_and_args)
            self.metadata_script_calls += 1
            self.connection.reconnect()
            return [b'capacity']

    client = _CapacityReconnectClient()
    role = _RoleClient(lambda: client, _ConnectionGeneration())
    store = ValkeyMetadataStore(role, enabled_settings('managed'), RecordingMetrics())

    with pytest.raises(BackendUnavailable, match='changed during operation'):
        store.resolve((DependencyTag.site('site'),))

    assert client.metadata_script_calls == 1
    assert client.close_calls == 1


def test_initial_epoch_observes_connection_established_inside_client_checkout() -> None:
    class _ObservedClient(AlreadyConnectedAtomicClient):
        def info(self, section: str) -> object:
            assert self.connection.callback_observations == 1
            return super().info(section)

    client = _ObservedClient()
    role = _RoleClient(lambda: client, _ConnectionGeneration())
    control = ValkeyControl(
        role,
        key_prefix=_PREFIX,
        validation_seconds=30,
        nonce_factory=lambda: nonce(1),
    )

    assert control.validate().nonce == nonce(1)
    assert client.connection.disconnect_calls == 1
    assert client.connection.successful_connections == 2
    assert role.epoch() == (0, 1, None)


@pytest.mark.parametrize(
    'replication_info',
    [
        {'role': 'slave', 'master_replid': _REPLICATION_A},
        {'role': 'master'},
    ],
)
def test_managed_primary_identity_fails_closed(
    replication_info: Mapping[str, object],
) -> None:
    class _ReplicationClient(AtomicControlClient):
        def info(self, section: str) -> object:
            _ = section
            return replication_info

    client = _ReplicationClient()
    with pytest.raises((BackendProtocolError, BackendUnavailable)):
        ValkeyControl(
            client,
            key_prefix=_PREFIX,
            validation_seconds=30,
            nonce_factory=lambda: nonce(1),
        ).validate()


def test_metadata_pool_expansion_does_not_discard_a_stable_overlapping_pin() -> None:
    metadata = PooledMetadataClient()
    role = _RoleClient(lambda: metadata, _ConnectionGeneration())
    store = ValkeyMetadataStore(role, enabled_settings('managed'), RecordingMetrics())
    tag = DependencyTag.site('site')
    first_outcomes: list[object] = []
    first_failures: list[Exception] = []

    def invoke_first() -> None:
        try:
            first_outcomes.append(store.resolve((tag,)))
        except Exception as error:  # noqa: BLE001 - thread outcome is asserted below
            first_failures.append(error)

    thread = Thread(target=invoke_first)
    thread.start()
    assert metadata.first_operation_entered.wait(timeout=5)
    second_outcome = store.resolve((tag,))
    metadata.release_first_operation.set()
    thread.join(timeout=5)

    assert not thread.is_alive()
    assert first_failures == []
    assert len(first_outcomes) == 1
    assert second_outcome.entries[0][0] == tag
    assert len({id(child.connection) for child in metadata.children}) == 2
    assert metadata.close_calls == 0
    assert role.epoch() == (0, 2, None)


def test_new_metadata_socket_receives_full_validation() -> None:
    metadata = PooledMetadataClient()
    metadata.release_first_operation.set()
    role = _RoleClient(lambda: metadata, _ConnectionGeneration())
    store = ValkeyMetadataStore(role, enabled_settings('managed'), RecordingMetrics())
    tag = DependencyTag.site('site')

    store.resolve((tag,))
    info_calls = metadata.info_calls
    store.resolve((tag,))

    assert len({id(child.connection) for child in metadata.children}) >= 2
    assert metadata.info_calls == info_calls + 2


def test_new_metadata_socket_rotates_after_primary_promotion() -> None:
    metadata = PooledMetadataClient()
    metadata.release_first_operation.set()
    role = _RoleClient(lambda: metadata, _ConnectionGeneration())
    store = ValkeyMetadataStore(role, enabled_settings('managed'), RecordingMetrics())
    tag = DependencyTag.site('site')

    original = store.resolve((tag,))
    metadata.replication_id = _REPLICATION_B
    replacement = store.resolve((tag,))

    assert replacement.incarnation != original.incarnation
    assert metadata.control is not None
    assert metadata.control.startswith(f'{_REPLICATION_B}|'.encode())


def test_periodic_jitter_and_reconnect_force_validation() -> None:
    client = AtomicControlClient()
    clock = FakeClock()
    control = ValkeyControl(
        client,
        key_prefix=_PREFIX,
        validation_seconds=30,
        clock=clock,
        nonce_factory=lambda: nonce(1),
        jitter_factory=lambda _seconds: 27,
    )

    control.validate()
    assert client.info_calls == 2
    clock.advance(26)
    control.validate()
    assert client.info_calls == 2
    clock.advance(1)
    control.validate()
    assert client.info_calls == 4
    client.connection_generation += 1
    control.validate()
    assert client.info_calls == 6


def test_production_connection_observes_setup_without_first_pin_reconnect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clients: list[PreObservedClient] = []

    def create_client(**kwargs: object) -> PreObservedClient:
        callback = cast('Callable[[object], None]', kwargs['valkey_connect_func'])
        client = PreObservedClient(callback)
        clients.append(client)
        return client

    monkeypatch.setattr(valkey_module, 'Valkey', create_client)
    settings = enabled_settings()
    role = _create_client(settings.l2.valkey.metadata, settings, 'secret')
    control = ValkeyControl(
        role,
        key_prefix=_PREFIX,
        validation_seconds=30,
        nonce_factory=lambda: nonce(1),
    )

    assert control.validate().nonce == nonce(1)
    connection = clients[0].connection
    assert connection.connect_calls == 1
    assert connection.disconnect_calls == 0
    assert role.epoch() == (0, 1, None)

    connection.disconnect()
    connection.connect()

    assert role.epoch() == (0, 2, None)


def test_reconnect_after_response_before_settlement_discards_read_outcome() -> None:
    class _ResponseReconnectClient(ConnectedAtomicControlClient):
        def evalsha(self, sha: str, numkeys: int, *keys_and_args: object) -> object:
            if sha in (CONTROL_CREATE_SCRIPT_SHA, CONTROL_ROTATE_SCRIPT_SHA):
                return super().evalsha(sha, numkeys, *keys_and_args)
            self.metadata_script_calls += 1
            reply = [b'ok', 1]
            self.connection.reconnect()
            return reply

    client = _ResponseReconnectClient()
    role = _RoleClient(lambda: client, _ConnectionGeneration())
    store = ValkeyMetadataStore(role, enabled_settings('managed'), RecordingMetrics())

    with pytest.raises(BackendUnavailable, match='changed during operation'):
        store.resolve((DependencyTag.site('site'),))

    assert client.metadata_script_calls == 1
    assert role.epoch()[1] == 1
    assert client.close_calls == 1


def test_reconnected_metadata_socket_receives_full_validation() -> None:
    class _ReusedMetadataClient(PooledMetadataClient):
        def client(self) -> PooledMetadataChild:
            if not self.children:
                return super().client()
            child = self.children[0]
            child.connection.connect()
            return child

    metadata = _ReusedMetadataClient()
    metadata.release_first_operation.set()
    role = _RoleClient(lambda: metadata, _ConnectionGeneration())
    store = ValkeyMetadataStore(role, enabled_settings('managed'), RecordingMetrics())
    tag = DependencyTag.site('site')

    store.resolve((tag,))
    info_calls = metadata.info_calls
    connection: AlreadyConnectedObservable = metadata.children[0].connection
    connection.disconnect()
    connection.connect()
    store.resolve((tag,))

    assert metadata.info_calls == info_calls + 2


def test_replica_validation_reply_discards_pool_before_next_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:

    class _ReplicaClient(AtomicControlClient):
        def __init__(self) -> None:
            super().__init__()
            self.close_calls = 0

        def close(self) -> None:
            self.close_calls += 1

        def info(self, section: str) -> object:
            _ = section
            self.info_calls += 1
            role = 'master' if self.info_calls == 1 else 'slave'
            return {'role': role, 'master_replid': self.replication_id}

    demoted = _ReplicaClient()
    result = AtomicControlClient()
    promoted = AtomicControlClient(replication_id=_REPLICATION_B)
    clients = iter((demoted, promoted))

    def create_client(**kwargs: object) -> AtomicControlClient:
        return result if kwargs['host'] == 'result.example' else next(clients)

    monkeypatch.setattr(valkey_module, 'Valkey', create_client)
    backend = ValkeyBackend.from_settings(
        enabled_settings('managed'),
        {'METADATA_PASSWORD': 'metadata-secret', 'RESULT_PASSWORD': 'result-secret'},
        RecordingMetrics(),
    )
    change = (GenerationChange(DependencyTag.site('site')),)

    with pytest.raises(BackendUnavailable):
        backend.change(change)
    assert demoted.close_calls == 1

    backend.change(change)

    assert demoted.info_calls == 2
    assert promoted.info_calls >= 2


def test_replication_change_during_cas_restarts_before_adoption() -> None:
    class _ChangingReplicationClient(AtomicControlClient):
        def __init__(self) -> None:
            super().__init__()
            self._replication_ids = iter(
                (_REPLICATION_A, _REPLICATION_B, _REPLICATION_B, _REPLICATION_B)
            )

        def info(self, section: str) -> object:
            assert section == 'replication'
            return {'role': 'master', 'master_replid': next(self._replication_ids)}

    client = _ChangingReplicationClient()
    nonces = iter((nonce(1), nonce(2)))
    control = ValkeyControl(
        client,
        key_prefix=_PREFIX,
        validation_seconds=30,
        nonce_factory=lambda: next(nonces),
    )

    incarnation = control.validate()

    assert incarnation.replication_id == _REPLICATION_B
    assert incarnation.nonce == nonce(2)


def test_script_reported_stale_incarnation_clears_cached_validation() -> None:
    client = AtomicControlClient()
    control = ValkeyControl(client, key_prefix=_PREFIX, validation_seconds=30)
    first = control.validate()
    client.control = b'|'.join(
        (_REPLICATION_A.encode(), nonce(9).encode(), f'{_PREFIX}:v1:counter:{nonce(9)}'.encode())
    )
    client.counters.add(f'{_PREFIX}:v1:counter:{nonce(9)}')
    client.metadata_replies.append([b'stale'])

    with pytest.raises(StaleIncarnation):
        control.run(lambda pinned, _incarnation: decode_change_reply(pinned.evalsha('x' * 40, 0)))

    second = control.validate()
    assert second.record == client.control
    assert second.nonce != first.nonce


def test_sentinel_drift_at_fenced_operation_entry_discards_read_outcome_and_pool() -> None:
    state = AtomicControlClient()
    endpoint = ['old']
    fenced_script_endpoints: list[str] = []

    class _EntryDriftClient(EndpointControlClient):
        def evalsha(self, sha: str, numkeys: int, *keys_and_args: object) -> object:
            if sha not in (CONTROL_CREATE_SCRIPT_SHA, CONTROL_ROTATE_SCRIPT_SHA):
                endpoint[0] = 'new'
            return super().evalsha(sha, numkeys, *keys_and_args)

    clients = {
        name: _EntryDriftClient(name, state, endpoint, fenced_script_endpoints)
        for name in ('old', 'new')
    }
    role_client = _RoleClient(
        lambda: clients[endpoint[0]],
        _ConnectionGeneration(),
        lambda _client: endpoint[0],
    )
    store = ValkeyMetadataStore(role_client, enabled_settings('sentinel'), RecordingMetrics())

    with pytest.raises(BackendUnavailable, match='changed during operation'):
        store.resolve((DependencyTag.site('site'),))

    assert clients['old'].close_calls == 1
    assert fenced_script_endpoints == ['old']


def test_stable_semantic_exception_is_reraised_after_endpoint_settlement() -> None:
    probes = 0
    client = AtomicControlClient()

    def probe(_client: object) -> str:
        nonlocal probes
        probes += 1
        return 'primary'

    role = _RoleClient(lambda: client, _ConnectionGeneration(), probe)
    control = ValkeyControl(
        role,
        key_prefix=_PREFIX,
        validation_seconds=30,
        nonce_factory=lambda: nonce(1),
    )
    semantic = BackendCapacity('stable capacity outcome')

    def operation(_client: object, _incarnation: Incarnation) -> None:
        raise semantic

    with pytest.raises(BackendCapacity) as captured:
        control.run(operation)

    assert captured.value is semantic
    assert probes == 2


def test_standalone_control_adopts_a_concurrent_rotation_winner() -> None:
    replid = 'a' * 40
    loser_nonce = '1' * 32
    winner_nonce = '2' * 32
    loser_counter = f'sgqc:v1:counter:{loser_nonce}'
    winner_counter = f'sgqc:v1:counter:{winner_nonce}'
    loser = f'{replid}|{loser_nonce}|{loser_counter}'.encode()
    winner = f'{replid}|{winner_nonce}|{winner_counter}'.encode()

    class _ConcurrentRotationClient(OrderedClient):
        def __init__(self) -> None:
            super().__init__([])
            self.control_reads = 0

        def evalsha(self, sha: str, numkeys: int, *keys_and_args: object) -> object:
            raise AssertionError((sha, numkeys, keys_and_args))

        def exists(self, key: str) -> object:
            assert key in (loser_counter, winner_counter)
            return 1

        def get(self, key: str) -> object:
            assert key == 'sgqc:v1:control'
            self.control_reads += 1
            return loser if self.control_reads == 1 else winner

        def info(self, section: str) -> object:
            assert section == 'replication'
            return {'role': 'master', 'master_replid': replid}

        def script_load(self, script: bytes | str) -> object:
            raise AssertionError(script)

    client = _ConcurrentRotationClient()
    control = ValkeyControl(
        client,
        key_prefix='sgqc',
        validation_seconds=30,
    )

    incarnation = control.validate()

    assert incarnation == Incarnation(replid, winner_nonce, winner_counter, winner)
    assert client.control_reads == 4


def test_standalone_control_atomically_creates_and_caches_initial_state() -> None:
    nonce = '1' * 32
    replid = 'a' * 40
    counter_key = f'sgqc:v1:counter:{nonce}'
    record = f'{replid}|{nonce}|{counter_key}'.encode()
    client = OrderedClient(
        [
            ('info', {'role': 'master', 'master_replid': replid}),
            ('get', None),
            ('evalsha', [b'created', record]),
            ('info', {'role': 'master', 'master_replid': replid}),
            ('get', ACTIVE_CONTROL),
            ('exists', 1),
        ]
    )
    clock = FakeClock()
    control = ValkeyControl(
        client,
        key_prefix='sgqc',
        validation_seconds=30,
        clock=clock,
        nonce_factory=lambda: nonce,
    )

    first = control.validate()
    second = control.validate()

    assert first == second == Incarnation(replid, nonce, counter_key, record)
    assert client.replies == []
    assert [call[0] for call in client.calls] == [
        'info',
        'get',
        'evalsha',
        'info',
        'get',
        'exists',
    ]


@pytest.mark.parametrize(
    'replication_info',
    [
        {'role': b'\xffmaster', 'master_replid': b'a' * 40},
        {'role': b'master', 'master_replid': b'\xff' + b'a' * 40},
    ],
)
def test_standalone_control_rejects_non_ascii_replication_bytes(
    replication_info: dict[str, object],
) -> None:
    client = OrderedClient([('info', replication_info)])
    control = ValkeyControl(
        client,
        key_prefix='sgqc',
        validation_seconds=30,
    )

    with pytest.raises(BackendProtocolError) as captured:
        control.validate()

    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None
    assert 'master' not in _exception_text(captured.value)


def test_standalone_control_rotates_a_record_with_an_unrelated_counter() -> None:
    replid = 'a' * 40
    corrupt_nonce = '1' * 32
    corrupt_counter = f'unrelated:v1:counter:{corrupt_nonce}'
    replacement_nonce = '2' * 32
    replacement_counter = f'sgqc:v1:counter:{replacement_nonce}'

    class _CorruptRecordClient(OrderedClient):
        def __init__(self) -> None:
            super().__init__([])
            self._record = f'{replid}|{corrupt_nonce}|{corrupt_counter}'.encode()

        def evalsha(self, sha: str, numkeys: int, *keys_and_args: object) -> object:
            self.calls.append(('evalsha', sha, numkeys, *keys_and_args))
            replacement = cast('bytes', keys_and_args[-1])
            self._record = replacement
            return [b'rotated', replacement]

        def exists(self, key: str) -> object:
            self.calls.append(('exists', key))
            return 1

        def get(self, key: str) -> object:
            self.calls.append(('get', key))
            return self._record

        def info(self, section: str) -> object:
            self.calls.append(('info', section))
            return {'role': 'master', 'master_replid': replid}

        def script_load(self, script: bytes | str) -> object:
            raise AssertionError(script)

    client = _CorruptRecordClient()
    control = ValkeyControl(
        client,
        key_prefix='sgqc',
        validation_seconds=30,
        nonce_factory=lambda: replacement_nonce,
    )

    incarnation = control.validate()

    assert incarnation.counter_key == replacement_counter
    assert ('exists', corrupt_counter) not in client.calls


@pytest.mark.parametrize('mode', ['managed', 'sentinel'])
def test_warmed_hit_reuses_proven_connections_without_info(
    monkeypatch: pytest.MonkeyPatch,
    mode: _ValkeyMode,
) -> None:
    class _MetadataClient(ProvingClient):
        def __init__(self, hook: Callable[[object], None]) -> None:
            super().__init__(hook)
            self.metadata_replies.extend(([b'ok', 1], [b'ok', 1]))

    class _ResultClient(ProvingClient):
        def __init__(self, hook: Callable[[object], None]) -> None:
            super().__init__(hook)
            self.metadata_replies.extend(
                ([b'hit', encode_value({'id': 7}), 1000], [b'hit', encode_value({'id': 7}), 1000])
            )

    clients = _install_proving_clients(monkeypatch, mode, [_MetadataClient, _ResultClient])
    backend = ValkeyBackend.from_settings(
        enabled_settings(mode),
        {'METADATA_PASSWORD': 'secret', 'RESULT_PASSWORD': 'secret'},
        RecordingMetrics(),
    )
    tags = (DependencyTag.site('site'),)
    first = backend.resolve(tags)
    assert backend.lookup('key') is not None
    counts = [(len(client.connection.sent), client.info_calls) for client in clients]
    assert backend.resolve(tags) == first
    assert backend.lookup('key') is not None
    assert [(len(client.connection.sent), client.info_calls) for client in clients] == counts
    assert all(client.connection.sent == [('INFO', 'replication')] for client in clients)


@pytest.mark.parametrize('operation', ['resolve', 'change'])
def test_connect_proof_during_fenced_command_rejects_outcome_without_replay(
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    class _ReconnectingClient(ProvingClient):
        reconnect = True
        script_calls = 0

        def evalsha(self, sha: str, numkeys: int, *args: object) -> object:
            if sha not in (CONTROL_CREATE_SCRIPT_SHA, CONTROL_ROTATE_SCRIPT_SHA):
                self.script_calls += 1
                if self.reconnect:
                    self.connection.disconnect()
                    self.connection.connect()
            return super().evalsha(sha, numkeys, *args)

    clients = _install_proving_clients(
        monkeypatch, 'managed', [_ReconnectingClient, ProvingClient]
    )
    settings = enabled_settings('managed')
    role = _create_client(settings.l2.valkey.metadata, settings, 'secret')
    with role.pin():
        pass
    client = cast('_ReconnectingClient', clients[0])
    client.metadata_replies.append([b'ok', 1] if operation == 'resolve' else [b'ok', 1, 0])
    store = ValkeyMetadataStore(role, settings, RecordingMetrics())

    def invoke() -> object:
        if operation == 'resolve':
            return store.resolve((DependencyTag.site('site'),))
        return store.change((GenerationChange(DependencyTag.site('site')),))

    with pytest.raises(BackendUnavailable, match='changed during operation'):
        invoke()
    assert client.connection.sent == [('INFO', 'replication')] * 2
    assert client.script_calls == 1
    assert client.close_calls == 1
    client.reconnect = False
    # The discarded pool is replaced and its control is fully validated.
    if operation == 'resolve':
        replacement = clients[1]
        replacement.metadata_replies.append([b'ok', 1])
        assert store.resolve((DependencyTag.site('site'),)).incarnation
        assert replacement.info_calls == 3


@pytest.mark.parametrize(
    'failure_type', [BackendCapacity, BackendProtocolError, BackendUnavailable, StaleIncarnation]
)
def test_role_settlement_does_not_classify_local_backend_failures(
    failure_type: type[Exception],
) -> None:
    endpoint = ['first']
    client = AtomicControlClient()
    role = _RoleClient(lambda: client, _ConnectionGeneration(), lambda _client: endpoint[0])
    control = ValkeyControl(
        role,
        key_prefix=_PREFIX,
        validation_seconds=30,
        nonce_factory=lambda: nonce(1),
    )
    local_failure = failure_type('local backend failure')

    def operation(_client: object, _incarnation: Incarnation) -> None:
        endpoint[0] = 'second'
        raise local_failure

    with pytest.raises(BackendUnavailable, match='role changed during operation') as caught:
        control.run(operation)
    assert caught.value is not local_failure
    assert caught.value.failure_class is None
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None
    role.close()
