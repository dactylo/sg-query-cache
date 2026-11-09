import os
import shutil
import subprocess
import time
from collections.abc import Generator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Literal, Protocol, TypeAlias, cast

import pytest
from valkey import Sentinel, Valkey

from sg_query_cache._backend import BackendUnavailable, StaleIncarnation
from sg_query_cache._config import CacheSettings, ValkeyRoleSettings, parse_sentinel_address
from sg_query_cache._model import DependencyTag, FillLease, GenerationVector
from sg_query_cache._valkey_scripts import RESOLVE_SCRIPT_SHA
from sg_query_cache.valkey import ValkeyBackend, _PinnedRoleClient, _RoleClient
from tests._fakes import FakeClock, RecordingMetrics
from tests.valkey._environment import RETENTION_SECONDS, topology_settings
from tests.valkey._environment import required as _environment_required

Profile: TypeAlias = Literal['managed', 'sentinel']
Role: TypeAlias = Literal['metadata', 'result']

BASE_KEY = f'sgqc:v1:base:{"a" * 64}'
_CONVERGENCE_CLIENTS = 50
_CONVERGENCE_SECONDS = 30.0


class AdminClient(Protocol):
    def close(self) -> None: ...

    def evalsha(self, sha: str, numkeys: int, *keys_and_args: object) -> object: ...

    def get(self, key: str) -> object: ...

    def info(self, section: str) -> Mapping[str, object]: ...

    def pttl(self, key: str) -> int: ...

    def set(self, key: str, value: bytes) -> object: ...


class _RoleClientObserver(Protocol):
    def epoch(self) -> tuple[int, int, object | None]: ...


class _FlushClient(Protocol):
    def close(self) -> None: ...

    def flushdb(self) -> object: ...


class ServerClock(FakeClock):
    # The server times TTLs itself, so report the real time that it also observes.
    def __call__(self) -> float:
        return time.monotonic()

    def advance(self, seconds: float) -> None:
        if seconds > 7_000:
            time.sleep(RETENTION_SECONDS + 0.05)
        else:
            time.sleep(seconds + 0.02)


@dataclass(frozen=True, slots=True)
class Topology:
    profile: Profile
    settings: CacheSettings


@dataclass(frozen=True, slots=True)
class RoleObservation:
    pool_epoch: int
    connection_generation: int
    endpoint: object | None
    pool_identity: int | None
    served_generation: int | None = None


def _backend_role_client(backend: ValkeyBackend, role: Role) -> object:
    if role == 'metadata':
        metadata = cast('object', object.__getattribute__(backend, '_metadata'))
        return cast('object', object.__getattribute__(metadata, '_client'))
    return cast('object', object.__getattribute__(backend, '_result_client'))


def _resolve_replacement(
    backend: ValkeyBackend,
    tag: DependencyTag,
    previous: str,
    deadline: float,
    interval_seconds: float,
) -> tuple[GenerationVector | None, bool]:
    vector = None
    neutral_failure = False
    while time.monotonic() < deadline:
        try:
            vector = backend.resolve((tag,))
        except (BackendUnavailable, StaleIncarnation):
            neutral_failure = True
            time.sleep(interval_seconds)
            continue
        if vector.incarnation == previous:
            pytest.fail('a successful resolve after rotation delivered the old incarnation')
        break
    return vector, neutral_failure


def assert_role_replaced(
    retained: RoleObservation,
    before_first_attempt: RoleObservation,
    settled: RoleObservation,
    *,
    neutral_failure: bool,
    profile: Profile,
) -> None:
    proactive = before_first_attempt.pool_epoch > retained.pool_epoch or (
        profile == 'sentinel' and settled.endpoint != retained.endpoint
    )
    # The first successful operation ran on a socket that connected, and proved a writable
    # primary, after the retained observation; another socket's reconnect cannot satisfy this.
    assert settled.served_generation is not None
    assert settled.served_generation > retained.connection_generation
    if profile == 'sentinel':
        assert settled.endpoint != retained.endpoint
    else:
        assert settled.endpoint is retained.endpoint is None
    if neutral_failure or proactive:
        assert settled.pool_epoch > retained.pool_epoch
        assert settled.pool_identity != retained.pool_identity
        return
    # A stable managed endpoint can reconnect at connection borrow without a client-visible signal.
    assert profile == 'managed'
    assert settled.pool_epoch == retained.pool_epoch
    assert settled.pool_identity == retained.pool_identity


def assert_stale_metadata_control(
    promoted: AdminClient,
    control_key: str,
    old_control: bytes,
) -> None:
    promoted_control = promoted.get(control_key)
    assert isinstance(promoted_control, bytes)
    assert promoted_control != old_control
    old_counter = old_control.split(b'|')[2].decode('ascii')
    stale_reply = promoted.evalsha(
        RESOLVE_SCRIPT_SHA,
        2,
        control_key,
        old_counter,
        old_control,
        7_200_000,
    )
    assert stale_reply in ([b'stale'], [b'missing-counter'])


def backend_role_observation(
    backend: ValkeyBackend,
    role: Role,
    *,
    served: Sequence[int] = (),
) -> RoleObservation:
    client = cast('_RoleClientObserver', _backend_role_client(backend, role))
    pool_epoch, connection_generation, endpoint = client.epoch()
    assert isinstance(pool_epoch, int)
    assert not isinstance(pool_epoch, bool)
    assert isinstance(connection_generation, int)
    assert not isinstance(connection_generation, bool)
    pool = object.__getattribute__(client, '_client')
    return RoleObservation(
        pool_epoch,
        connection_generation,
        endpoint,
        None if pool is None else id(pool),
        served[-1] if served else None,
    )


def concurrent_replacement_incarnation(
    settings: CacheSettings,
    tag: DependencyTag,
    previous: str,
) -> str:
    deadline = time.monotonic() + _CONVERGENCE_SECONDS

    # Each simulated client retries a neutral failure, as a caller does after a cache bypass.
    def replace_incarnation(_index: int) -> tuple[str, int]:
        failures = 0
        while True:
            try:
                concurrent = ValkeyBackend.from_settings(settings, os.environ, RecordingMetrics())
                try:
                    return concurrent.resolve((tag,)).incarnation, failures
                finally:
                    concurrent.close()
            except (BackendUnavailable, StaleIncarnation):
                failures += 1
                if time.monotonic() >= deadline:
                    raise
                time.sleep(0.1)

    with ThreadPoolExecutor(max_workers=_CONVERGENCE_CLIENTS) as executor:
        outcomes = tuple(executor.map(replace_incarnation, range(_CONVERGENCE_CLIENTS)))
    retried = sum(1 for _, failures in outcomes if failures)
    print(f'{retried} of {_CONVERGENCE_CLIENTS} replacement clients retried a neutral failure')
    incarnations = {incarnation for incarnation, _ in outcomes}
    assert len(incarnations) == 1
    (winner,) = incarnations
    assert winner != previous
    return winner


def replid(client: AdminClient) -> str:
    info = client.info('replication')
    assert info.get('role') in ('master', b'master')
    value = info.get('master_replid')
    if isinstance(value, bytes):
        value = value.decode('ascii')
    assert isinstance(value, str)
    assert value
    return value


def required(name: str) -> str:
    return _environment_required(name, os.environ)


def _aws_text(operation: str, arguments: tuple[str, ...], query: str) -> str:
    if shutil.which('aws') is None:
        pytest.fail('the required AWS CLI is unavailable; it was not installed')
    command = [
        'aws',
        'elasticache',
        operation,
        '--region',
        required('SG_QUERY_CACHE_TEST_ELASTICACHE_REGION'),
        *arguments,
        '--query',
        query,
        '--output',
        'text',
    ]
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    value = completed.stdout.strip()
    if completed.returncode != 0 or not value or value == 'None':
        pytest.fail(f'AWS CLI {operation} failed')
    return value


def eviction_policy(role: Role) -> str:
    prefix = f'SG_QUERY_CACHE_TEST_ELASTICACHE_{role.upper()}'
    member = _aws_text(
        'describe-replication-groups',
        ('--replication-group-id', required(f'{prefix}_REPLICATION_GROUP')),
        'ReplicationGroups[0].MemberClusters[0]',
    )
    parameter_group = _aws_text(
        'describe-cache-clusters',
        ('--cache-cluster-id', member),
        'CacheClusters[0].CacheParameterGroup.CacheParameterGroupName',
    )
    return _aws_text(
        'describe-cache-parameters',
        ('--cache-parameter-group-name', parameter_group),
        "Parameters[?ParameterName=='maxmemory-policy'].ParameterValue | [0]",
    )


def integration_client(
    role: ValkeyRoleSettings,
    settings: CacheSettings,
    *,
    credentials: tuple[str, str] | None = None,
) -> Valkey:
    return Valkey(
        host=role.host,
        port=role.port,
        db=role.database,
        ssl=role.tls,
        ssl_check_hostname=role.tls,
        username=role.username if credentials is None else credentials[0],
        password=required(role.password_env) if credentials is None else credentials[1],
        socket_connect_timeout=settings.limits.connect_timeout_seconds,
        socket_timeout=settings.limits.operation_timeout_seconds,
        decode_responses=False,
        retry_on_timeout=False,
    )


def flush_roles(settings: CacheSettings) -> None:
    result = cast('_FlushClient', integration_client(settings.l2.valkey.result, settings))
    metadata = cast('_FlushClient', integration_client(settings.l2.valkey.metadata, settings))
    try:
        result.flushdb()
        metadata.flushdb()
    finally:
        result.close()
        metadata.close()


def reserve_until(
    backend: ValkeyBackend,
    key: str,
    deadline: float,
    interval_seconds: float,
) -> tuple[FillLease | None, bool]:
    neutral_failure = False
    while time.monotonic() < deadline:
        try:
            return backend.reserve(key, 30), neutral_failure
        except BackendUnavailable:
            neutral_failure = True
            time.sleep(interval_seconds)
    pytest.fail('result-role reserve did not recover before the deadline')


@contextmanager
def served_pin_generations(backend: ValkeyBackend, role: Role) -> Generator[list[int]]:
    role_client = _backend_role_client(backend, role)
    original = _RoleClient.pin
    served: list[int] = []

    @contextmanager
    def pin(self: _RoleClient) -> Generator[_PinnedRoleClient]:
        with original(self) as pinned:
            yield pinned
            generation = pinned.connection_generation()
            if self is role_client and generation is not None:
                served.append(generation)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(_RoleClient, 'pin', pin)
        yield served


def observed_replacement(
    backend: ValkeyBackend,
    tag: DependencyTag,
    previous: str,
    deadline: float,
    interval_seconds: float,
) -> tuple[GenerationVector, bool, RoleObservation]:
    metrics = cast('RecordingMetrics', object.__getattribute__(backend, '_metrics'))
    recorded = len(metrics.increments)
    with served_pin_generations(backend, 'metadata') as served:
        vector, neutral_failure = _resolve_replacement(
            backend, tag, previous, deadline, interval_seconds
        )
    settled = backend_role_observation(backend, 'metadata', served=served)
    assert vector is not None
    # The first successful resolve fully validated the control state before delivering it.
    assert ('cache_incarnation', {'outcome': 'validated'}) in metrics.increments[recorded:]
    return vector, neutral_failure, settled


def observed_write(
    backend: ValkeyBackend,
    key: str,
    deadline: float,
    interval_seconds: float,
) -> tuple[FillLease | None, bool, RoleObservation]:
    # Reads may still be served by a socket attached to a demoted node, a residual the design
    # accepts; a write cannot be, so the first write identifies the replacement socket.
    with served_pin_generations(backend, 'result') as served:
        lease, neutral_failure = reserve_until(backend, key, deadline, interval_seconds)
    return lease, neutral_failure, backend_role_observation(backend, 'result', served=served)


def topology(profile: Profile) -> Topology:
    return Topology(profile, topology_settings(profile, os.environ))


def primary(topology: Topology, role: Role) -> AdminClient:
    settings = (
        topology.settings.l2.valkey.metadata
        if role == 'metadata'
        else topology.settings.l2.valkey.result
    )
    kwargs: dict[str, object] = {
        'db': settings.database,
        'ssl': settings.tls,
        'username': settings.username,
        'password': required(settings.password_env),
        'socket_connect_timeout': topology.settings.limits.connect_timeout_seconds,
        'socket_timeout': topology.settings.limits.operation_timeout_seconds,
        'decode_responses': False,
        'retry_on_timeout': False,
    }
    if settings.tls:
        kwargs['ssl_check_hostname'] = True
    if topology.profile == 'sentinel':
        discovery = Sentinel(
            tuple(parse_sentinel_address(value) for value in settings.sentinels),
            min_other_sentinels=2,
            sentinel_kwargs={key: value for key, value in kwargs.items() if key != 'db'},
            **kwargs,
        )
        assert settings.service_name is not None
        return cast('AdminClient', discovery.master_for(settings.service_name))
    return cast(
        'AdminClient',
        Valkey(
            host=settings.host,
            port=settings.port,
            db=settings.database,
            ssl=settings.tls,
            ssl_check_hostname=settings.tls,
            username=settings.username,
            password=required(settings.password_env),
            socket_connect_timeout=topology.settings.limits.connect_timeout_seconds,
            socket_timeout=topology.settings.limits.operation_timeout_seconds,
            decode_responses=False,
            retry_on_timeout=False,
        ),
    )


def wait_for_new_primary(
    topology: Topology,
    role: Role,
    previous_replid: str,
    *,
    timeout_seconds: float,
    interval_seconds: float,
) -> AdminClient:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        client = primary(topology, role)
        try:
            if replid(client) != previous_replid:
                return client
        except Exception as error:  # noqa: BLE001 - bounded external acceptance polling
            _ = error
        client.close()
        time.sleep(interval_seconds)
    pytest.fail(f'{topology.profile} {role} promotion did not expose a new replication identity')
