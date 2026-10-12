import os
import shutil
import subprocess
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, Protocol, TypeAlias, cast

import pytest
from valkey import Sentinel, Valkey

from sg_query_cache._config import CacheSettings, ValkeyRoleSettings, parse_sentinel_address
from sg_query_cache.valkey import ValkeyBackend
from tests._fakes import _FakeClock
from tests.valkey._environment import required as _environment_required
from tests.valkey._environment import topology_settings

Profile: TypeAlias = Literal['managed', 'sentinel']
Role: TypeAlias = Literal['metadata', 'result']

BASE_KEY = f'sgqc:v1:base:{"a" * 64}'
RETENTION_SECONDS = 0.1


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


class ServerClock(_FakeClock):
    def _advance(self, seconds: float) -> None:
        if seconds > 7_000:
            time.sleep(RETENTION_SECONDS + 0.05)
        else:
            time.sleep(seconds + 0.02)


@dataclass(frozen=True, slots=True)
class Topology:
    _profile: Profile
    settings: CacheSettings


@dataclass(frozen=True, slots=True)
class RoleObservation:
    _pool_epoch: int
    _connection_generation: int
    _endpoint: object | None
    _pool_identity: int | None


def _backend_role_client(backend: ValkeyBackend, role: Role) -> object:
    if role == 'metadata':
        metadata = cast('object', object.__getattribute__(backend, '_metadata'))
        return cast('object', object.__getattribute__(metadata, '_client'))
    return cast('object', object.__getattribute__(backend, '_result_client'))


def assert_role_replaced(
    retained: RoleObservation,
    before_first_attempt: RoleObservation,
    settled: RoleObservation,
    *,
    neutral_failure: bool,
    profile: Profile,
) -> None:
    proactive = before_first_attempt._pool_epoch > retained._pool_epoch or (
        profile == 'sentinel' and settled._endpoint != retained._endpoint
    )
    assert neutral_failure or proactive
    assert settled._pool_epoch > retained._pool_epoch
    assert settled._connection_generation > retained._connection_generation
    assert settled._pool_identity != retained._pool_identity
    if profile == 'sentinel':
        assert settled._endpoint != retained._endpoint
    else:
        assert settled._endpoint is retained._endpoint is None


def backend_role_observation(backend: ValkeyBackend, role: Role) -> RoleObservation:
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
    )


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
    if topology._profile == 'sentinel':
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
