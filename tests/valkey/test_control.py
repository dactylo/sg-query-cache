from collections.abc import Callable, Generator, Mapping
from contextlib import contextmanager
from dataclasses import replace
from threading import Barrier, Lock, Thread
from types import SimpleNamespace
from typing import cast

import pytest
from valkey.exceptions import NoScriptError, ReadOnlyError, ResponseError

import sg_query_cache.valkey as valkey_module
import tests.valkey._integration_helpers as integration_helpers
from sg_query_cache._backend import (
    BackendCapacity,
    BackendProtocolError,
    BackendUnavailable,
    StaleIncarnation,
)
from sg_query_cache._codec import encode_value
from sg_query_cache._config import _ValkeyMode
from sg_query_cache._model import DependencyTag, FillLease, GenerationChange
from sg_query_cache._valkey_common import Incarnation, decode_change_reply
from sg_query_cache._valkey_control import ValkeyControl
from sg_query_cache._valkey_scripts import (
    COMMIT_SCRIPT_SHA,
    CONTROL_CREATE_SCRIPT_SHA,
    CONTROL_ROTATE_SCRIPT_SHA,
)
from sg_query_cache.valkey import (
    ValkeyBackend,
    ValkeyMetadataStore,
    _CircuitBreaker,
    _ConnectionGeneration,
    _create_client,
    _RoleClient,
)
from tests._exceptions import exception_text as _exception_text
from tests._fakes import _FakeClock, _RecordingMetrics
from tests._valkey_control_cases import (
    _ACTIVE_CONTROL,
    _COUNTER_A,
    _COUNTER_B,
    _control_traces,
    _ControlTrace,
    _OrderedClient,
    _publication_traces,
    _PublicationTrace,
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
    PooledResultChild,
    PooledResultClient,
    PreObservedClient,
    ProvingClient,
    ReconnectingClient,
    TraceControlClient,
    enabled_settings,
    nonce,
)

_PREFIX = 'sgqc'
_REPLICATION_A = 'a' * 40
_REPLICATION_B = 'b' * 40


def _breaker_clock(owner: ValkeyMetadataStore | ValkeyBackend, role: str) -> _FakeClock:
    clock = _FakeClock()
    object.__setattr__(
        owner,
        '_breaker' if role == 'metadata' else '_result_breaker',
        _CircuitBreaker(
            role, threshold=1, probe_seconds=10, metrics=_RecordingMetrics(), clock=clock
        ),
    )
    return clock


def _install_proving_clients(
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


def _publication_control(
    trace: _PublicationTrace,
) -> tuple[PinnedConnectionClient, ValkeyControl, _FakeClock]:
    client = PinnedConnectionClient(replication_id='primary')
    client._owner._control = trace._initial_record
    client._owner._counters.add(_COUNTER_A)
    clock = _FakeClock(start=0.0)
    control = ValkeyControl(
        client,
        key_prefix='sgqc',
        validation_seconds=30,
        clock=clock,
        jitter_factory=lambda seconds: seconds,
    )
    if trace._older_joins:
        control.validate()
        client._pin_connection = 2
        clock._advance(10)
    return client, control, clock


def test_backend_revalidates_before_reusing_a_transparently_reconnected_pool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:

    metadata = ReconnectingClient()
    metadata._metadata_replies.extend(([b'ok', 1], [b'ok', 1]))
    result = ReconnectingClient()
    clients = iter((metadata, result))

    def create_client(**kwargs: object) -> ReconnectingClient:
        _ = kwargs
        return next(clients)

    monkeypatch.setattr(valkey_module, 'Valkey', create_client)
    backend = ValkeyBackend.from_settings(
        enabled_settings('managed'),
        {'METADATA_PASSWORD': 'metadata-secret', 'RESULT_PASSWORD': 'result-secret'},
        _RecordingMetrics(),
    )
    tag = DependencyTag.site('site')
    original = backend.resolve((tag,))

    metadata._reconnect_to(_REPLICATION_B)
    replacement = backend.resolve((tag,))

    assert replacement.incarnation != original.incarnation
    assert metadata._control is not None
    assert metadata._control.startswith(f'{_REPLICATION_B}|'.encode())
    assert metadata._info_calls == 4


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
    client._replication_id = _REPLICATION_B
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


@pytest.mark.parametrize('trace', _control_traces(), ids=lambda trace: trace._name)
def test_control_executes_shared_semantic_traces(trace: _ControlTrace) -> None:
    client = TraceControlClient(trace)
    control = ValkeyControl(
        client,
        key_prefix='sgqc',
        validation_seconds=30,
        clock=lambda: 0.0,
        jitter_factory=lambda seconds: seconds,
        nonce_factory=lambda: trace._proposal_nonce,
    )

    def operation(_pinned: object, incarnation: Incarnation) -> bytes:
        decode_change_reply(client._effect('operation'))
        return incarnation.record

    for expected in trace._acknowledgments:
        if isinstance(expected, bytes):
            assert control.run(operation) == expected
        else:
            with pytest.raises(expected):
                control.run(operation)

    state = control._snapshot()
    assert client._effects == []
    assert (None if state.incarnation is None else state.incarnation.record) == trace._final_record
    assert state.revision == trace._final_revision
    assert state.role_epoch == trace._final_epoch


@pytest.mark.parametrize('trace', _publication_traces(), ids=lambda trace: trace._name)
def test_control_executes_shared_publication_traces_with_barriers(
    trace: _PublicationTrace,
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
        if trace._transition == 'clear':
            decode_change_reply([b'stale'])
        return incarnation.record

    thread = Thread(target=old_run)
    thread.start()
    entered.wait()
    if trace._older_joins:
        client._pin_connection = 3
        clock._advance(10)
    with client._owner._lock:
        client._owner._control = trace._replacement_record
        client._owner._counters.add(_COUNTER_B)
    try:
        newer: bytes | type[Exception] = control.run(new_operation, force=True)
    except StaleIncarnation:
        newer = StaleIncarnation
    finally:
        effects.append('newer-settled')
        settled.wait()
        thread.join(timeout=5)

    assert not thread.is_alive()
    assert (*older, newer) == trace._acknowledgments
    assert tuple(effects) == trace._effects
    state = control._snapshot()
    assert (None if state.incarnation is None else state.incarnation.record) == trace._final_record
    assert state.revision == trace._final_revision
    if trace._final_covered is not None:
        assert state.covered == trace._final_covered
        assert state.validated_at == (None if trace._transition == 'clear' else 20.0)


def test_control_forced_validation_resets_connection_coverage() -> None:
    client = PinnedConnectionClient()
    clock = _FakeClock(start=0.0)
    control = ValkeyControl(
        client,
        key_prefix=_PREFIX,
        validation_seconds=30,
        clock=clock,
        jitter_factory=lambda seconds: seconds,
    )
    control.validate()
    client._pin_connection = 2
    control.validate()
    assert control._snapshot().covered == frozenset({1, 2})

    client._pin_connection = 1
    control.validate(force=True)
    assert control._snapshot().covered == frozenset({1})
    assert client._owner._info_calls == 6


def test_control_join_does_not_extend_validation_lifetime() -> None:
    client = PinnedConnectionClient()
    clock = _FakeClock(start=0.0)
    control = ValkeyControl(
        client,
        key_prefix=_PREFIX,
        validation_seconds=30,
        clock=clock,
        jitter_factory=lambda seconds: seconds,
    )
    control.validate()
    clock._advance(20)
    client._pin_connection = 2
    control.validate()
    assert control._snapshot().validated_at == 0.0
    clock._advance(9)
    client._pin_connection = 1
    control.validate()
    assert client._owner._info_calls == 4

    clock._advance(1)
    client._pin_connection = 2
    control.validate()
    assert client._owner._info_calls == 6
    assert control._snapshot().covered == frozenset({2})
    assert control._snapshot().validated_at == 30.0
    client._pin_connection = 1
    control.validate()
    assert client._owner._info_calls == 8
    assert control._snapshot().covered == frozenset({1, 2})
    assert control._snapshot().validated_at == 30.0


def test_control_keeps_a_container_shaped_metrics_sink() -> None:

    class _CountingSink(_RecordingMetrics):
        __slots__ = ()

        def __len__(self) -> int:
            return len(self._increments)

    sink = _CountingSink()
    control = ValkeyControl(
        AtomicControlClient(),
        key_prefix=_PREFIX,
        validation_seconds=30,
        metrics=sink,
    )

    control.validate()

    assert ('cache_incarnation', {'outcome': 'validated'}) in sink._increments


def test_control_reuses_validation_on_alternating_covered_connections() -> None:
    client = PinnedConnectionClient()
    clock = _FakeClock(start=0.0)
    control = ValkeyControl(
        client,
        key_prefix=_PREFIX,
        validation_seconds=30,
        clock=clock,
        jitter_factory=lambda seconds: seconds,
    )
    original = control.validate()
    client._pin_connection = 2
    assert control.validate() == original
    assert client._owner._info_calls == 4

    for connection in (1, 2, 1, 2):
        client._pin_connection = connection
        assert control.validate() == original
    assert client._owner._info_calls == 4


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
    client._replication_id = _REPLICATION_B
    second = control.validate(force=True)
    client._counters.remove(second.counter_key)
    third = control.validate(force=True)
    client._control = None
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
            self._command_connections: list[int] = []
            self._next_connection = 0
            self._pin_calls = 0

        def _record_connection(self) -> None:
            if self._active_connection is None:
                self._next_connection += 1
                self._command_connections.append(self._next_connection)
            else:
                self._command_connections.append(self._active_connection)

        @contextmanager
        def pin(self) -> Generator[AtomicControlClient]:
            self._pin_calls += 1
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

    assert client._pin_calls == 1
    assert len(set(client._command_connections)) == 1


def test_counter_exhaustion_rotates_before_reporting_stale(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata = AtomicControlClient()
    metadata._metadata_replies.append([b'rotation-required'])
    clients = iter((metadata, AtomicControlClient()))

    def create_client(**kwargs: object) -> AtomicControlClient:
        _ = kwargs
        return next(clients)

    monkeypatch.setattr(valkey_module, 'Valkey', create_client)
    backend = ValkeyBackend.from_settings(
        enabled_settings(),
        {'METADATA_PASSWORD': 'secret', 'RESULT_PASSWORD': 'secret'},
        _RecordingMetrics(),
    )

    with pytest.raises(StaleIncarnation):
        backend.resolve((DependencyTag.site('site'),))

    assert metadata._pre_rotation_control is not None
    assert metadata._control != metadata._pre_rotation_control


def test_demotion_discards_pool_and_revalidates_before_next_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:

    class _DemotedClient(AtomicControlClient):
        def __init__(self) -> None:
            super().__init__()
            self._close_calls = 0
            self._initial = True

        def close(self) -> None:
            self._close_calls += 1

        def info(self, section: str) -> object:
            _ = section
            if self._initial:
                self._initial = False
                return {'role': 'master', 'master_replid': self._replication_id}
            raise ReadOnlyError('credential-bearing-demotion')

    demoted = _DemotedClient()
    result = AtomicControlClient()
    promoted = AtomicControlClient(replication_id=_REPLICATION_B)
    clients = iter((demoted, result, promoted))

    def create_client(**kwargs: object) -> AtomicControlClient:
        _ = kwargs
        return next(clients)

    monkeypatch.setattr(valkey_module, 'Valkey', create_client)
    backend = ValkeyBackend.from_settings(
        enabled_settings('managed'),
        {'METADATA_PASSWORD': 'metadata-secret', 'RESULT_PASSWORD': 'result-secret'},
        _RecordingMetrics(),
    )
    change = (GenerationChange(DependencyTag.site('site')),)

    with pytest.raises(BackendUnavailable):
        backend.change(change)
    assert demoted._close_calls == 1
    assert promoted._info_calls == 0

    backend.change(change)

    assert promoted._info_calls == 2


def test_drift_discarded_mutation_is_unacknowledged_and_not_retried() -> None:
    class _MutationReconnectClient(ConnectedAtomicControlClient):
        def evalsha(self, sha: str, numkeys: int, *keys_and_args: object) -> object:
            reply = super().evalsha(sha, numkeys, *keys_and_args)
            if sha not in (CONTROL_CREATE_SCRIPT_SHA, CONTROL_ROTATE_SCRIPT_SHA):
                self._metadata_script_calls += 1
                self.connection._reconnect()
            return reply

    client = _MutationReconnectClient()
    role = _RoleClient(lambda: client, _ConnectionGeneration())
    store = ValkeyMetadataStore(role, enabled_settings('managed'), _RecordingMetrics())

    with pytest.raises(BackendUnavailable, match='changed during operation'):
        store.change((GenerationChange(DependencyTag.site('site')),))

    assert client._metadata_script_calls == 1
    assert client._close_calls == 1


def test_drift_discards_semantic_exception_and_pool_before_reraise() -> None:
    class _CapacityReconnectClient(ConnectedAtomicControlClient):
        def evalsha(self, sha: str, numkeys: int, *keys_and_args: object) -> object:
            if sha in (CONTROL_CREATE_SCRIPT_SHA, CONTROL_ROTATE_SCRIPT_SHA):
                return super().evalsha(sha, numkeys, *keys_and_args)
            self._metadata_script_calls += 1
            self.connection._reconnect()
            return [b'capacity']

    client = _CapacityReconnectClient()
    role = _RoleClient(lambda: client, _ConnectionGeneration())
    store = ValkeyMetadataStore(role, enabled_settings('managed'), _RecordingMetrics())

    with pytest.raises(BackendUnavailable, match='changed during operation'):
        store.resolve((DependencyTag.site('site'),))

    assert client._metadata_script_calls == 1
    assert client._close_calls == 1


def test_elasticache_policy_uses_read_only_control_plane_descriptions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv('SG_QUERY_CACHE_TEST_ELASTICACHE_REGION', 'us-west-2')
    monkeypatch.setenv(
        'SG_QUERY_CACHE_TEST_ELASTICACHE_METADATA_REPLICATION_GROUP',
        'metadata-group',
    )
    calls: list[list[str]] = []
    outputs = iter(('metadata-member-1\n', 'metadata-parameters\n', 'noeviction\n'))

    def run(command: list[str], **kwargs: object) -> object:
        _ = kwargs
        calls.append(command)
        return SimpleNamespace(returncode=0, stdout=next(outputs))

    def find_aws(_name: str) -> str:
        return '/usr/bin/aws'

    monkeypatch.setattr(integration_helpers.shutil, 'which', find_aws)
    monkeypatch.setattr(integration_helpers.subprocess, 'run', run)

    assert integration_helpers.eviction_policy('metadata') == 'noeviction'
    assert [command[2] for command in calls] == [
        'describe-replication-groups',
        'describe-cache-clusters',
        'describe-cache-parameters',
    ]
    assert 'metadata-group' in calls[0]
    assert 'metadata-member-1' in calls[1]
    assert 'metadata-parameters' in calls[2]


def test_failed_reconnect_proof_discards_socket_and_rebuilds_pool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clients = _install_proving_clients(monkeypatch, 'managed', [ProvingClient])
    settings = enabled_settings('managed')
    role = _create_client(settings.l2.valkey.metadata, settings, 'secret')
    client = clients[0]

    def shared_client(**_kwargs: object) -> ProvingClient:
        return client

    def replication_info(_section: str) -> object:
        return {'role': current_role, 'master_replid': _REPLICATION_A}

    current_role = 'slave'
    monkeypatch.setattr(valkey_module, 'Valkey', shared_client)
    client.connection.disconnect()
    monkeypatch.setattr(client, 'info', replication_info)
    before = role.epoch()
    with pytest.raises(BackendUnavailable, match='not writable') as captured, role.pin():
        pytest.fail('reader pin was yielded')
    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None
    assert role.epoch()[0] == before[0] + 1
    assert not client.connection._connected
    current_role = 'master'
    with role.pin():
        assert client.connection._connected
    assert client.connection._sent == [('INFO', 'replication')] * 3


def test_initial_epoch_observes_connection_established_inside_client_checkout() -> None:
    class _ObservedClient(AlreadyConnectedAtomicClient):
        def info(self, section: str) -> object:
            assert self.connection._callback_observations == 1
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
    assert client.connection._disconnect_calls == 1
    assert client.connection._successful_connections == 2
    assert role.epoch() == (0, 1, None)


def test_metadata_breaker_open_prevents_event_acknowledgment_and_io(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    class _FailingClient:
        def client(self) -> '_FailingClient':
            return self

        def close(self) -> None:
            return None

        def evalsha(self, *_args: object) -> object:
            nonlocal calls
            calls += 1
            raise OSError('unavailable')

        def exists(self, _key: str) -> object:
            raise OSError('unavailable')

        def get(self, _key: str) -> object:
            raise OSError('unavailable')

        def info(self, _section: str) -> object:
            nonlocal calls
            calls += 1
            raise OSError('unavailable')

        def script_load(self, _script: bytes | str) -> object:
            raise OSError('unavailable')

    clients = iter((_FailingClient(), _FailingClient()))

    def create_client(**kwargs: object) -> _FailingClient:
        _ = kwargs
        return next(clients)

    monkeypatch.setattr(valkey_module, 'Valkey', create_client)
    settings = enabled_settings()
    settings = replace(
        settings,
        limits=replace(settings.limits, circuit_failure_threshold=1),
    )
    backend = ValkeyBackend.from_settings(
        settings,
        {'METADATA_PASSWORD': 'secret', 'RESULT_PASSWORD': 'secret'},
        _RecordingMetrics(),
    )

    with pytest.raises(BackendUnavailable):
        backend.change((GenerationChange(DependencyTag.site('site')),))
    before = calls
    with pytest.raises(BackendUnavailable):
        backend.change((GenerationChange(DependencyTag.site('site')),))

    assert calls == before


def test_metadata_guard_reopens_when_half_open_refresh_fails() -> None:
    class _RefreshClient(AtomicControlClient):
        def __init__(self) -> None:
            super().__init__()
            self._fail_info = True
            self._fail_refresh = True
            self._refresh_calls = 0

        def info(self, section: str) -> object:
            if self._fail_info:
                raise OSError('metadata unavailable')
            return super().info(section)

        def refresh(self) -> None:
            self._refresh_calls += 1
            if self._fail_refresh:
                raise BackendUnavailable('fresh metadata client is unavailable')

    client = _RefreshClient()
    settings = replace(
        enabled_settings(),
        limits=replace(
            enabled_settings().limits,
            circuit_failure_threshold=1,
            circuit_probe_seconds=10,
        ),
    )
    store = ValkeyMetadataStore(client, settings, _RecordingMetrics())
    clock = _breaker_clock(store, 'metadata')

    with pytest.raises(BackendUnavailable):
        store.resolve((DependencyTag.site('site'),))
    client._fail_info = False
    clock._advance(10)
    with pytest.raises(BackendUnavailable):
        store.resolve((DependencyTag.site('site'),))

    client._fail_refresh = False
    client._metadata_replies.append([b'ok', 1])
    clock._advance(10)
    assert store.resolve((DependencyTag.site('site'),)).incarnation
    assert client._refresh_calls == 2


def test_metadata_pool_expansion_does_not_discard_a_stable_overlapping_pin() -> None:
    metadata = PooledMetadataClient()
    role = _RoleClient(lambda: metadata, _ConnectionGeneration())
    store = ValkeyMetadataStore(role, enabled_settings('managed'), _RecordingMetrics())
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
    assert metadata._first_operation_entered.wait(timeout=5)
    second_outcome = store.resolve((tag,))
    metadata._release_first_operation.set()
    thread.join(timeout=5)

    assert not thread.is_alive()
    assert first_failures == []
    assert len(first_outcomes) == 1
    assert second_outcome.entries[0][0] == tag
    assert len({id(child.connection) for child in metadata._children}) == 2
    assert metadata._close_calls == 0
    assert role.epoch() == (0, 2, None)


def test_new_metadata_socket_receives_full_validation() -> None:
    metadata = PooledMetadataClient()
    metadata._release_first_operation.set()
    role = _RoleClient(lambda: metadata, _ConnectionGeneration())
    store = ValkeyMetadataStore(role, enabled_settings('managed'), _RecordingMetrics())
    tag = DependencyTag.site('site')

    store.resolve((tag,))
    info_calls = metadata._info_calls
    store.resolve((tag,))

    assert len({id(child.connection) for child in metadata._children}) >= 2
    assert metadata._info_calls == info_calls + 2


def test_new_metadata_socket_rotates_after_primary_promotion() -> None:
    metadata = PooledMetadataClient()
    metadata._release_first_operation.set()
    role = _RoleClient(lambda: metadata, _ConnectionGeneration())
    store = ValkeyMetadataStore(role, enabled_settings('managed'), _RecordingMetrics())
    tag = DependencyTag.site('site')

    original = store.resolve((tag,))
    metadata._replication_id = _REPLICATION_B
    replacement = store.resolve((tag,))

    assert replacement.incarnation != original.incarnation
    assert metadata._control is not None
    assert metadata._control.startswith(f'{_REPLICATION_B}|'.encode())


def test_older_validation_cannot_overwrite_a_newer_one() -> None:
    client = AtomicControlClient()
    control = ValkeyControl(client, key_prefix=_PREFIX, validation_seconds=30)
    newer = b'|'.join(
        (_REPLICATION_A.encode(), nonce(6).encode(), f'{_PREFIX}:v1:counter:{nonce(6)}'.encode())
    )

    def operation(_pinned: object, incarnation: Incarnation) -> bytes:
        # A rotation lands and another caller validates and publishes the
        # newer state while this run, which validated the older state, is
        # still in flight. The older state must not be published afterwards.
        with client._lock:
            client._control = newer
            client._counters.add(f'{_PREFIX}:v1:counter:{nonce(6)}')
        assert control.validate(force=True).record == newer
        return incarnation.record

    older = control.run(operation, force=True)
    info_calls = client._info_calls

    assert older != newer
    assert control.validate().record == newer
    assert client._info_calls == info_calls


def test_periodic_jitter_and_reconnect_force_validation() -> None:
    client = AtomicControlClient()
    clock = _FakeClock()
    control = ValkeyControl(
        client,
        key_prefix=_PREFIX,
        validation_seconds=30,
        clock=clock,
        nonce_factory=lambda: nonce(1),
        jitter_factory=lambda _seconds: 27,
    )

    control.validate()
    assert client._info_calls == 2
    clock._advance(26)
    control.validate()
    assert client._info_calls == 2
    clock._advance(1)
    control.validate()
    assert client._info_calls == 4
    client._connection_generation += 1
    control.validate()
    assert client._info_calls == 6


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
    assert connection._connect_calls == 1
    assert connection._disconnect_calls == 0
    assert role.epoch() == (0, 1, None)

    connection.disconnect()
    connection.connect()

    assert role.epoch() == (0, 2, None)


def test_publication_is_skipped_when_validation_was_cleared_during_the_operation() -> None:
    client = AtomicControlClient()
    control = ValkeyControl(client, key_prefix=_PREFIX, validation_seconds=30)
    replacement = b'|'.join(
        (_REPLICATION_A.encode(), nonce(5).encode(), f'{_PREFIX}:v1:counter:{nonce(5)}'.encode())
    )

    def operation(_pinned: object, incarnation: Incarnation) -> bytes:
        # Another thread observes a rotation and clears the cached validation
        # while this operation is in flight; the fresh validation this run
        # performed must not be republished afterwards.
        with client._lock:
            client._control = replacement
            client._counters.add(f'{_PREFIX}:v1:counter:{nonce(5)}')
        control._clear_validation()
        return incarnation.record

    stale_record = control.run(operation, force=True)
    info_calls = client._info_calls

    fresh = control.validate()

    assert stale_record != replacement
    assert fresh.record == replacement
    assert client._info_calls > info_calls


@pytest.mark.parametrize('mode', ['managed', 'sentinel'])
@pytest.mark.parametrize('reader_role', ['metadata', 'result'])
def test_reader_fails_closed_before_cache_operations(
    monkeypatch: pytest.MonkeyPatch,
    mode: _ValkeyMode,
    reader_role: str,
) -> None:
    class _ReaderClient(ProvingClient):
        def info(self, section: str) -> object:
            _ = section
            return {'role': 'slave', 'master_replid': _REPLICATION_A}

        def evalsha(self, sha: str, numkeys: int, *args: object) -> object:
            _ = sha, numkeys, args
            raise AssertionError('reader executed a cache script')

    clients = _install_proving_clients(
        monkeypatch,
        mode,
        [_ReaderClient, ProvingClient]
        if reader_role == 'metadata'
        else [ProvingClient, _ReaderClient],
    )
    with pytest.raises(BackendUnavailable, match='not writable') as captured:
        ValkeyBackend.from_settings(
            enabled_settings(mode),
            {'METADATA_PASSWORD': 'secret', 'RESULT_PASSWORD': 'secret'},
            _RecordingMetrics(),
        )
    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None
    reader = clients[-1]
    assert reader.connection._disconnect_calls == 1
    assert reader.connection._sent == [('INFO', 'replication')]
    assert not reader.connection._connected
    if reader_role == 'result':
        assert clients[0]._close_calls == 1


@pytest.mark.parametrize('probe_outcome', ['capacity', 'stale'])
def test_metadata_guard_closes_after_noncounting_half_open_outcome(
    probe_outcome: str,
) -> None:
    class _ProbeClient(AtomicControlClient):
        def __init__(self) -> None:
            super().__init__()
            self._fail_info = True
            self._capacity_failure = False

        def evalsha(self, sha: str, numkeys: int, *keys_and_args: object) -> object:
            if self._capacity_failure:
                self._capacity_failure = False
                raise ResponseError('OOM command not allowed when used memory exceeds maxmemory')
            return super().evalsha(sha, numkeys, *keys_and_args)

        def info(self, section: str) -> object:
            if self._fail_info:
                raise OSError('metadata unavailable')
            return super().info(section)

    client = _ProbeClient()
    settings = replace(
        enabled_settings(),
        limits=replace(
            enabled_settings().limits,
            circuit_failure_threshold=1,
            circuit_probe_seconds=10,
        ),
    )
    store = ValkeyMetadataStore(client, settings, _RecordingMetrics())
    clock = _breaker_clock(store, 'metadata')

    with pytest.raises(BackendUnavailable):
        store.resolve((DependencyTag.site('site'),))
    client._fail_info = False
    if probe_outcome == 'capacity':
        client._capacity_failure = True
        expected_error = BackendCapacity
    else:
        client._metadata_replies.append([b'rotation-required'])
        expected_error = StaleIncarnation
    clock._advance(10)
    with pytest.raises(expected_error):
        store.resolve((DependencyTag.site('site'),))

    client._metadata_replies.append([b'ok', 1])
    assert store.resolve((DependencyTag.site('site'),)).incarnation


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


@pytest.mark.parametrize('reconnect', [False, True])
def test_commit_noscript_uses_one_pin_and_rejects_reconnect(
    monkeypatch: pytest.MonkeyPatch,
    reconnect: bool,
) -> None:
    result = PooledResultClient()
    role = _RoleClient(lambda: result, _ConnectionGeneration())
    calls: list[str] = []

    def evaluate(
        _child: PooledResultChild,
        sha: str,
        numkeys: int,
        *args: object,
    ) -> object:
        assert sha == COMMIT_SCRIPT_SHA
        assert numkeys == 2
        assert args
        calls.append('evalsha')
        if len(calls) == 1:
            raise NoScriptError('missing')
        return [b'committed']

    def load(child: PooledResultChild, _script: bytes | str) -> object:
        calls.append('load')
        if reconnect:
            child.connection.disconnect()
            child.connection.connect()
        return COMMIT_SCRIPT_SHA

    monkeypatch.setattr(PooledResultChild, 'evalsha', evaluate)
    monkeypatch.setattr(PooledResultChild, 'script_load', load)
    settings = enabled_settings('managed')
    backend = ValkeyBackend(
        role,
        ValkeyMetadataStore(AtomicControlClient(), settings, _RecordingMetrics()),
        settings,
        _RecordingMetrics(),
    )
    if reconnect:
        with pytest.raises(BackendUnavailable, match='changed during operation'):
            backend.commit(FillLease('key', 'token'), b'payload', 1.0)
        assert result._close_calls == 1
    else:
        assert backend.commit(FillLease('key', 'token'), b'payload', 1.0)
        assert result._close_calls == 0
    assert calls == ['evalsha', 'load', 'evalsha']
    assert len(result._children) == 1
    backend.close()


def test_reconnect_after_response_before_settlement_discards_read_outcome() -> None:
    class _ResponseReconnectClient(ConnectedAtomicControlClient):
        def evalsha(self, sha: str, numkeys: int, *keys_and_args: object) -> object:
            if sha in (CONTROL_CREATE_SCRIPT_SHA, CONTROL_ROTATE_SCRIPT_SHA):
                return super().evalsha(sha, numkeys, *keys_and_args)
            self._metadata_script_calls += 1
            reply = [b'ok', 1]
            self.connection._reconnect()
            return reply

    client = _ResponseReconnectClient()
    role = _RoleClient(lambda: client, _ConnectionGeneration())
    store = ValkeyMetadataStore(role, enabled_settings('managed'), _RecordingMetrics())

    with pytest.raises(BackendUnavailable, match='changed during operation'):
        store.resolve((DependencyTag.site('site'),))

    assert client._metadata_script_calls == 1
    assert role.epoch()[1] == 1
    assert client._close_calls == 1


def test_reconnect_at_fenced_operation_entry_discards_read_outcome_and_pool() -> None:
    class _EntryReconnectClient(ConnectedAtomicControlClient):
        def evalsha(self, sha: str, numkeys: int, *keys_and_args: object) -> object:
            if sha in (CONTROL_CREATE_SCRIPT_SHA, CONTROL_ROTATE_SCRIPT_SHA):
                return super().evalsha(sha, numkeys, *keys_and_args)
            self._metadata_script_calls += 1
            self._replication_id = _REPLICATION_B
            self.connection._reconnect()
            return [b'ok', 1]

    client = _EntryReconnectClient()
    role = _RoleClient(lambda: client, _ConnectionGeneration())
    store = ValkeyMetadataStore(role, enabled_settings('managed'), _RecordingMetrics())

    with pytest.raises(BackendUnavailable, match='changed during operation'):
        store.resolve((DependencyTag.site('site'),))

    assert client._metadata_script_calls == 1
    assert client._close_calls == 1


def test_reconnected_metadata_socket_receives_full_validation() -> None:
    class _ReusedMetadataClient(PooledMetadataClient):
        def client(self) -> PooledMetadataChild:
            if not self._children:
                return super().client()
            child = self._children[0]
            child.connection.connect()
            return child

    metadata = _ReusedMetadataClient()
    metadata._release_first_operation.set()
    role = _RoleClient(lambda: metadata, _ConnectionGeneration())
    store = ValkeyMetadataStore(role, enabled_settings('managed'), _RecordingMetrics())
    tag = DependencyTag.site('site')

    store.resolve((tag,))
    info_calls = metadata._info_calls
    connection: AlreadyConnectedObservable = metadata._children[0].connection
    connection.disconnect()
    connection.connect()
    store.resolve((tag,))

    assert metadata._info_calls == info_calls + 2


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


def test_result_guard_closes_after_noncounting_half_open_capacity() -> None:
    class _ProbeClient(AtomicControlClient):
        def __init__(self) -> None:
            super().__init__()
            self._outcome = 'unavailable'

        def evalsha(self, sha: str, numkeys: int, *keys_and_args: object) -> object:
            _ = sha, numkeys, keys_and_args
            if self._outcome == 'unavailable':
                raise OSError('result unavailable')
            if self._outcome == 'capacity':
                raise ResponseError('OOM command not allowed when used memory exceeds maxmemory')
            return [b'miss']

    settings = replace(
        enabled_settings(),
        limits=replace(
            enabled_settings().limits,
            circuit_failure_threshold=1,
            circuit_probe_seconds=10,
        ),
    )
    result = _ProbeClient()
    metadata = ValkeyMetadataStore(AtomicControlClient(), settings, _RecordingMetrics())
    backend = ValkeyBackend(result, metadata, settings, _RecordingMetrics())
    clock = _breaker_clock(backend, 'result')

    with pytest.raises(BackendUnavailable):
        backend.lookup('qualified-key')
    result._outcome = 'capacity'
    clock._advance(10)
    with pytest.raises(BackendCapacity):
        backend.lookup('qualified-key')

    result._outcome = 'miss'
    assert backend.lookup('qualified-key') is None


def test_result_guard_reopens_when_half_open_refresh_fails() -> None:
    class _RefreshClient(AtomicControlClient):
        def __init__(self) -> None:
            super().__init__()
            self._fail_refresh = True
            self._fail_script = True
            self._refresh_calls = 0

        def evalsha(self, sha: str, numkeys: int, *keys_and_args: object) -> object:
            _ = sha, numkeys, keys_and_args
            if self._fail_script:
                raise OSError('result unavailable')
            return [b'miss']

        def refresh(self) -> None:
            self._refresh_calls += 1
            if self._fail_refresh:
                raise BackendUnavailable('fresh result client is unavailable')

    settings = replace(
        enabled_settings(),
        limits=replace(
            enabled_settings().limits,
            circuit_failure_threshold=1,
            circuit_probe_seconds=10,
        ),
    )
    result = _RefreshClient()
    metadata = ValkeyMetadataStore(AtomicControlClient(), settings, _RecordingMetrics())
    backend = ValkeyBackend(result, metadata, settings, _RecordingMetrics())
    clock = _breaker_clock(backend, 'result')

    with pytest.raises(BackendUnavailable):
        backend.lookup('qualified-key')
    result._fail_script = False
    clock._advance(10)
    with pytest.raises(BackendUnavailable):
        backend.lookup('qualified-key')

    result._fail_refresh = False
    clock._advance(10)
    assert backend.lookup('qualified-key') is None
    assert result._refresh_calls == 2


def test_script_reported_stale_incarnation_clears_cached_validation() -> None:
    client = AtomicControlClient()
    control = ValkeyControl(client, key_prefix=_PREFIX, validation_seconds=30)
    first = control.validate()
    client._control = b'|'.join(
        (_REPLICATION_A.encode(), nonce(9).encode(), f'{_PREFIX}:v1:counter:{nonce(9)}'.encode())
    )
    client._counters.add(f'{_PREFIX}:v1:counter:{nonce(9)}')
    client._metadata_replies.append([b'stale'])

    with pytest.raises(StaleIncarnation):
        control.run(lambda pinned, _incarnation: decode_change_reply(pinned.evalsha('x' * 40, 0)))

    second = control.validate()
    assert second.record == client._control
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
    store = ValkeyMetadataStore(role_client, enabled_settings('sentinel'), _RecordingMetrics())

    with pytest.raises(BackendUnavailable, match='changed during operation'):
        store.resolve((DependencyTag.site('site'),))

    assert clients['old']._close_calls == 1
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
    assert probes == 3


def test_standalone_control_adopts_a_concurrent_rotation_winner() -> None:
    replid = 'a' * 40
    loser_nonce = '1' * 32
    winner_nonce = '2' * 32
    loser_counter = f'sgqc:v1:counter:{loser_nonce}'
    winner_counter = f'sgqc:v1:counter:{winner_nonce}'
    loser = f'{replid}|{loser_nonce}|{loser_counter}'.encode()
    winner = f'{replid}|{winner_nonce}|{winner_counter}'.encode()

    class _ConcurrentRotationClient(_OrderedClient):
        def __init__(self) -> None:
            super().__init__([])
            self._control_reads = 0

        def evalsha(self, sha: str, numkeys: int, *keys_and_args: object) -> object:
            raise AssertionError((sha, numkeys, keys_and_args))

        def exists(self, key: str) -> object:
            assert key in (loser_counter, winner_counter)
            return 1

        def get(self, key: str) -> object:
            assert key == 'sgqc:v1:control'
            self._control_reads += 1
            return loser if self._control_reads == 1 else winner

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
    assert client._control_reads == 4


def test_standalone_control_atomically_creates_and_caches_initial_state() -> None:
    nonce = '1' * 32
    replid = 'a' * 40
    counter_key = f'sgqc:v1:counter:{nonce}'
    record = f'{replid}|{nonce}|{counter_key}'.encode()
    client = _OrderedClient(
        [
            ('info', {'role': 'master', 'master_replid': replid}),
            ('get', None),
            ('evalsha', [b'created', record]),
            ('info', {'role': 'master', 'master_replid': replid}),
            ('get', _ACTIVE_CONTROL),
            ('exists', 1),
        ]
    )
    clock = _FakeClock()
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
    assert client._replies == []
    assert [call[0] for call in client._calls] == [
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
    client = _OrderedClient([('info', replication_info)])
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


@pytest.mark.parametrize('mode', ['managed', 'sentinel'])
def test_malformed_connect_proof_closes_connection(
    monkeypatch: pytest.MonkeyPatch,
    mode: _ValkeyMode,
) -> None:
    class _MalformedClient(ProvingClient):
        def info(self, section: str) -> object:
            _ = section
            return {'role': 'master'}

    clients = _install_proving_clients(monkeypatch, mode, [_MalformedClient])
    with pytest.raises(BackendProtocolError) as captured:
        ValkeyBackend.from_settings(
            enabled_settings(mode),
            {'METADATA_PASSWORD': 'secret', 'RESULT_PASSWORD': 'secret'},
            _RecordingMetrics(),
        )
    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None
    assert clients[0].connection._disconnect_calls == 1
    assert not clients[0].connection._connected


def test_standalone_control_rotates_a_record_with_an_unrelated_counter() -> None:
    replid = 'a' * 40
    corrupt_nonce = '1' * 32
    corrupt_counter = f'unrelated:v1:counter:{corrupt_nonce}'
    replacement_nonce = '2' * 32
    replacement_counter = f'sgqc:v1:counter:{replacement_nonce}'

    class _CorruptRecordClient(_OrderedClient):
        def __init__(self) -> None:
            super().__init__([])
            self._record = f'{replid}|{corrupt_nonce}|{corrupt_counter}'.encode()

        def evalsha(self, sha: str, numkeys: int, *keys_and_args: object) -> object:
            self._calls.append(('evalsha', sha, numkeys, *keys_and_args))
            replacement = cast('bytes', keys_and_args[-1])
            self._record = replacement
            return [b'rotated', replacement]

        def exists(self, key: str) -> object:
            self._calls.append(('exists', key))
            return 1

        def get(self, key: str) -> object:
            self._calls.append(('get', key))
            return self._record

        def info(self, section: str) -> object:
            self._calls.append(('info', section))
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
    assert ('exists', corrupt_counter) not in client._calls


@pytest.mark.parametrize('mode', ['managed', 'sentinel'])
def test_warmed_hit_reuses_proven_connections_without_info(
    monkeypatch: pytest.MonkeyPatch,
    mode: _ValkeyMode,
) -> None:
    clients = _install_proving_clients(monkeypatch, mode, [ProvingClient, ProvingClient])
    backend = ValkeyBackend.from_settings(
        enabled_settings(mode),
        {'METADATA_PASSWORD': 'secret', 'RESULT_PASSWORD': 'secret'},
        _RecordingMetrics(),
    )
    metadata, result = clients
    metadata._metadata_replies.extend(([b'ok', 1], [b'ok', 1]))
    result._metadata_replies.extend(
        ([b'hit', encode_value({'id': 7}), 1000], [b'hit', encode_value({'id': 7}), 1000])
    )
    tags = (DependencyTag.site('site'),)
    first = backend.resolve(tags)
    assert backend.lookup('key') is not None
    counts = [(len(client.connection._sent), client._info_calls) for client in clients]
    assert backend.resolve(tags) == first
    assert backend.lookup('key') is not None
    assert [(len(client.connection._sent), client._info_calls) for client in clients] == counts
    assert all(client.connection._sent == [('INFO', 'replication')] for client in clients)


@pytest.mark.parametrize('operation', ['resolve', 'change'])
def test_connect_proof_during_fenced_command_rejects_outcome_without_replay(
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    class _ReconnectingClient(ProvingClient):
        _reconnect = True
        _script_calls = 0

        def evalsha(self, sha: str, numkeys: int, *args: object) -> object:
            if sha not in (CONTROL_CREATE_SCRIPT_SHA, CONTROL_ROTATE_SCRIPT_SHA):
                self._script_calls += 1
                if self._reconnect:
                    self.connection.disconnect()
                    self.connection.connect()
            return super().evalsha(sha, numkeys, *args)

    clients = _install_proving_clients(
        monkeypatch, 'managed', [_ReconnectingClient, ProvingClient]
    )
    settings = enabled_settings('managed')
    role = _create_client(settings.l2.valkey.metadata, settings, 'secret')
    client = cast('_ReconnectingClient', clients[0])
    client._metadata_replies.append([b'ok', 1] if operation == 'resolve' else [b'ok', 1, 0])
    store = ValkeyMetadataStore(role, settings, _RecordingMetrics())

    def invoke() -> object:
        if operation == 'resolve':
            return store.resolve((DependencyTag.site('site'),))
        return store.change((GenerationChange(DependencyTag.site('site')),))

    with pytest.raises(BackendUnavailable, match='changed during operation'):
        invoke()
    assert client.connection._sent == [('INFO', 'replication')] * 2
    assert client._script_calls == 1
    assert client._close_calls == 1
    client._reconnect = False
    # The discarded pool is replaced and its control is fully validated.
    if operation == 'resolve':
        replacement = clients[1]
        replacement._metadata_replies.append([b'ok', 1])
        assert store.resolve((DependencyTag.site('site'),)).incarnation
        assert replacement._info_calls == 3
