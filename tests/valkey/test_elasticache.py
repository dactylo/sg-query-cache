import asyncio
import os
import shutil
import subprocess
import time
from contextlib import AsyncExitStack
from dataclasses import dataclass, replace
from typing import cast

import pytest
from valkey.exceptions import ConnectionError as ValkeyConnectionError
from valkey.exceptions import ReadOnlyError
from valkey.exceptions import TimeoutError as ValkeyTimeoutError

from sg_query_cache._backend import BackendUnavailable
from sg_query_cache._codec import encode_value
from sg_query_cache._config import LimitSettings
from sg_query_cache._keys import KeyBuilder
from sg_query_cache._model import DependencyTag, FillLease, GenerationChange, GenerationVector
from sg_query_cache.valkey import ValkeyBackend
from sg_query_cache.valkey_async import AsyncValkeyBackend
from tests._fakes import RecordingMetrics
from tests.valkey import _async_integration_helpers as async_helpers
from tests.valkey._integration_helpers import (
    BASE_KEY,
    AdminClient,
    Role,
    RoleObservation,
    Topology,
    assert_role_replaced,
    assert_stale_metadata_control,
    backend_role_observation,
    concurrent_replacement_incarnation,
    eviction_policy,
    integration_client,
    observed_replacement,
    observed_write,
    primary,
    replid,
    required,
    reserve_until,
    wait_for_new_primary,
)
from tests.valkey._integration_helpers import (
    topology as _make_topology,
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.elasticache,
    pytest.mark.skipif(
        os.environ.get('SG_QUERY_CACHE_REQUIRE_INTEGRATION') != '1',
        reason='requires explicitly configured disposable services',
    ),
    pytest.mark.skipif(
        os.environ.get('SG_QUERY_CACHE_REQUIRE_ELASTICACHE') != '1',
        reason='requires separately approved ElastiCache test-failover',
    ),
]

_FAILOVER_ATTEMPT_LIMIT = 15
_FAILOVER_WINDOW_HOURS = 24
# The metadata tests leave leases under BASE_KEY on the shared provider groups; the result tests
# resolve the same tag, so they need their own base key to start without those leases.
_RESULT_BASE_KEY = f'sgqc:v1:base:{"d" * 64}'


@dataclass(frozen=True, slots=True)
class _Failover:
    role: Role
    backend: ValkeyBackend
    old_primary: AdminClient
    promoted: AdminClient
    retained: RoleObservation
    before_first_attempt: RoleObservation


def _assert_replaced_leases(
    backend: ValkeyBackend,
    old_result_key: str,
    new_result_key: str,
    old_leases: tuple[FillLease, FillLease],
    deadline: float,
) -> None:
    assert new_result_key != old_result_key
    assert backend.lookup(new_result_key) is None
    replacement_result_lease = reserve_until(backend, new_result_key, deadline, 1)[0]
    assert replacement_result_lease is not None
    assert replacement_result_lease.token != old_leases[0].token
    replacement_fill_lease = reserve_until(backend, f'{new_result_key}-lease-loss', deadline, 1)[0]
    assert replacement_fill_lease is not None
    assert replacement_fill_lease.token != old_leases[1].token


def _aws_failover(role: Role) -> None:
    if shutil.which('aws') is None:
        pytest.fail('the required AWS CLI is unavailable; it was not installed')
    prefix = f'SG_QUERY_CACHE_TEST_ELASTICACHE_{role.upper()}'
    command = [
        'aws',
        'elasticache',
        'test-failover',
        '--region',
        required('SG_QUERY_CACHE_TEST_ELASTICACHE_REGION'),
        '--replication-group-id',
        required(f'{prefix}_REPLICATION_GROUP'),
        '--node-group-id',
        required(f'{prefix}_NODE_GROUP'),
        '--output',
        'json',
    ]
    completed = subprocess.run(command, check=False, capture_output=True, timeout=30)
    if completed.returncode != 0:
        pytest.fail('AWS CLI test-failover failed; do not retry blindly')


def _finish(failover: _Failover, neutral_failure: bool, settled: RoleObservation) -> None:
    assert_role_replaced(
        failover.retained,
        failover.before_first_attempt,
        settled,
        neutral_failure=neutral_failure,
        profile='managed',
    )
    failover.promoted.close()
    failover.old_primary.close()
    failover.backend.close()


def _node_client(topology: Topology, role: Role, endpoint: tuple[str, int]) -> AdminClient:
    settings = (
        topology.settings.l2.valkey.metadata
        if role == 'metadata'
        else topology.settings.l2.valkey.result
    )
    host, port = endpoint
    node = replace(settings, host=host, port=port)
    return cast('AdminClient', integration_client(node, topology.settings))


def _primary_node_endpoint(role: Role) -> tuple[str, int]:
    if shutil.which('aws') is None:
        pytest.fail('the required AWS CLI is unavailable; it was not installed')
    prefix = f'SG_QUERY_CACHE_TEST_ELASTICACHE_{role.upper()}'
    node_group = required(f'{prefix}_NODE_GROUP')
    command = [
        'aws',
        'elasticache',
        'describe-replication-groups',
        '--region',
        required('SG_QUERY_CACHE_TEST_ELASTICACHE_REGION'),
        '--replication-group-id',
        required(f'{prefix}_REPLICATION_GROUP'),
        '--query',
        (
            f"ReplicationGroups[0].NodeGroups[?NodeGroupId=='{node_group}'] | [0]"
            ".NodeGroupMembers[?CurrentRole=='primary'] | [0].ReadEndpoint.[Address, Port]"
        ),
        '--output',
        'text',
    ]
    completed = subprocess.run(command, check=False, capture_output=True, text=True, timeout=30)
    fields = completed.stdout.split()
    if completed.returncode != 0 or len(fields) != 2:
        pytest.fail('AWS CLI describe-replication-groups did not report one primary node')
    return fields[0], int(fields[1])


async def _promote_async(
    topology: Topology,
    role: Role,
    backend: AsyncValkeyBackend,
    old_primary: AdminClient,
    cleanup: AsyncExitStack,
) -> tuple[AdminClient, RoleObservation, RoleObservation]:
    previous_replid = await asyncio.to_thread(replid, old_primary)
    endpoint = await asyncio.to_thread(_primary_node_endpoint, role)
    former_primary = _node_client(topology, role, endpoint)
    cleanup.callback(former_primary.close)
    assert await asyncio.to_thread(replid, former_primary) == previous_replid
    retained = async_helpers.backend_role_observation(backend, role)
    await asyncio.to_thread(_aws_failover, role)
    promoted = await asyncio.to_thread(
        wait_for_new_primary,
        topology,
        role,
        previous_replid,
        timeout_seconds=900,
        interval_seconds=1,
    )
    cleanup.callback(promoted.close)
    with pytest.raises((ReadOnlyError, ValkeyConnectionError, ValkeyTimeoutError)):
        await asyncio.to_thread(former_primary.set, 'sgqc-elasticache-demotion-probe', b'1')
    before_first_attempt = async_helpers.backend_role_observation(backend, role)
    return promoted, retained, before_first_attempt


def _provider_topology() -> Topology:
    # Provider acceptance measures the production budgets; only local harnesses get test slack.
    topology = _make_topology('managed')
    return replace(topology, settings=replace(topology.settings, limits=LimitSettings()))


def _begin_failover(
    role: Role,
    expected_policy: str,
) -> tuple[Topology, ValkeyBackend, AdminClient, GenerationVector]:
    print(
        'ElastiCache test-failover quota: '
        f'{_FAILOVER_ATTEMPT_LIMIT} shard failovers per rolling {_FAILOVER_WINDOW_HOURS} hours'
    )
    topology = _provider_topology()
    backend = ValkeyBackend.from_settings(topology.settings, os.environ, RecordingMetrics())
    old_primary = primary(topology, role)
    assert eviction_policy(role) == expected_policy
    vector = backend.resolve((DependencyTag.site('failover'),))
    return topology, backend, old_primary, vector


async def _metadata_failover_async(cleanup: AsyncExitStack) -> None:
    topology = _provider_topology()
    backend = await AsyncValkeyBackend.from_settings(
        topology.settings, os.environ, RecordingMetrics()
    )
    cleanup.push_async_callback(backend.close)
    old_primary = primary(topology, 'metadata')
    cleanup.callback(old_primary.close)
    assert await asyncio.to_thread(eviction_policy, 'metadata') == 'noeviction'
    tag = DependencyTag.site('failover')
    vector = await backend.resolve((tag,))
    result_key = KeyBuilder.qualified_key(BASE_KEY, vector, 'result')
    control_key = f'{topology.settings.l2.valkey.metadata.key_prefix}:v1:control'
    old_control = old_primary.get(control_key)
    assert isinstance(old_control, bytes)
    old_tag_key = (
        f'{topology.settings.l2.valkey.metadata.key_prefix}:v1:metadata:'
        f'{vector.incarnation}:{KeyBuilder.tag_key(tag)}'
    )
    old_result_lease = await backend.reserve(result_key, 30)
    assert old_result_lease is not None
    assert await backend.commit(old_result_lease, encode_value({'id': 'old'}), 600)
    old_fill_lease = await backend.reserve(f'{result_key}-lease-loss', 30)
    assert old_fill_lease is not None
    await backend.change((GenerationChange(tag),))
    acknowledged_tag = old_primary.get(old_tag_key)
    assert acknowledged_tag is not None

    promoted, retained, before_first_attempt = await _promote_async(
        topology, 'metadata', backend, old_primary, cleanup
    )
    promoted_tag = promoted.get(old_tag_key)
    bump_outcome = ('absent', 'retained')[int(promoted_tag == acknowledged_tag)]
    print(f'ElastiCache acknowledged metadata bump was {bump_outcome} after failover')
    winning_incarnation = await async_helpers.concurrent_replacement_incarnation(
        topology.settings, tag, vector.incarnation
    )
    deadline = time.monotonic() + 120
    replacement, neutral_failure, settled = await async_helpers.observed_replacement(
        backend, tag, vector.incarnation, deadline, 1
    )
    assert replacement.incarnation == winning_incarnation
    assert_stale_metadata_control(promoted, control_key, old_control)

    new_result_key = KeyBuilder.qualified_key(BASE_KEY, replacement, 'result')
    assert new_result_key != result_key
    assert await backend.lookup(new_result_key) is None
    replacement_result_lease, _ = await async_helpers.reserve_until(
        backend, new_result_key, deadline, 1
    )
    assert replacement_result_lease is not None
    assert replacement_result_lease.token != old_result_lease.token
    replacement_fill_lease, _ = await async_helpers.reserve_until(
        backend, f'{new_result_key}-lease-loss', deadline, 1
    )
    assert replacement_fill_lease is not None
    assert replacement_fill_lease.token != old_fill_lease.token
    assert_role_replaced(
        retained, before_first_attempt, settled, neutral_failure=neutral_failure, profile='managed'
    )


def _promote(
    role: Role,
    backend: ValkeyBackend,
    old_primary: AdminClient,
) -> _Failover:
    topology = _provider_topology()
    previous_replid = replid(old_primary)
    # The primary endpoint name follows promotion, so probe the former primary by its node name.
    former_primary = _node_client(topology, role, _primary_node_endpoint(role))
    assert replid(former_primary) == previous_replid
    retained = backend_role_observation(backend, role)
    _aws_failover(role)
    promoted = wait_for_new_primary(
        topology,
        role,
        previous_replid,
        timeout_seconds=900,
        interval_seconds=1,
    )
    with pytest.raises((ReadOnlyError, ValkeyConnectionError, ValkeyTimeoutError)):
        former_primary.set('sgqc-elasticache-demotion-probe', b'1')
    former_primary.close()
    before_first_attempt = backend_role_observation(backend, role)
    return _Failover(role, backend, old_primary, promoted, retained, before_first_attempt)


async def _result_failover_async(cleanup: AsyncExitStack) -> None:
    topology = _provider_topology()
    backend = await AsyncValkeyBackend.from_settings(
        topology.settings, os.environ, RecordingMetrics()
    )
    cleanup.push_async_callback(backend.close)
    old_primary = primary(topology, 'result')
    cleanup.callback(old_primary.close)
    assert await asyncio.to_thread(eviction_policy, 'result') == 'allkeys-lru'
    vector = await backend.resolve((DependencyTag.site('failover'),))
    result_key = KeyBuilder.qualified_key(_RESULT_BASE_KEY, vector, 'result')
    lease = await backend.reserve(result_key, 30)
    assert lease is not None
    assert await backend.commit(lease, encode_value({'id': 1}), 600)
    # A commit cannot expire later than its completion plus the supplied lifetime.
    expiry_upper = time.monotonic() + 600
    before = await backend.lookup(result_key)
    assert before is not None
    lost_lease = await backend.reserve(f'{result_key}-lease-loss', 30)
    assert lost_lease is not None
    released = await backend.reserve(f'{result_key}-release', 30)
    assert released is not None
    assert await backend.release(released)

    _, retained, before_first_attempt = await _promote_async(
        topology, 'result', backend, old_primary, cleanup
    )
    deadline = time.monotonic() + 120
    neutral_failure = False
    while time.monotonic() < deadline:
        try:
            after_started = time.monotonic()
            after = await backend.lookup(result_key)
            break
        except BackendUnavailable:
            neutral_failure = True
            await asyncio.sleep(1)
    else:
        pytest.fail('result lookup did not recover before the failover deadline')
    first_write, write_failed, settled = await async_helpers.observed_write(
        backend, f'{result_key}-first-write', deadline, 1
    )
    neutral_failure |= write_failed
    assert first_write is not None
    if after is not None:
        assert after_started + after.remaining_seconds <= expiry_upper + 0.01
    else:
        replacement, _ = await async_helpers.reserve_until(backend, result_key, deadline, 1)
        assert replacement is not None
        assert replacement.token != lease.token
    replacement_lease, _ = await async_helpers.reserve_until(
        backend, f'{result_key}-lease-loss', deadline, 1
    )
    assert replacement_lease is None or replacement_lease.token != lost_lease.token
    assert_role_replaced(
        retained, before_first_attempt, settled, neutral_failure=neutral_failure, profile='managed'
    )


def test_async_metadata_failover_rotates_control_and_rejects_stale_keys() -> None:
    async def scenario() -> None:
        async with AsyncExitStack() as cleanup:
            await _metadata_failover_async(cleanup)

    asyncio.run(scenario())


def test_async_result_failover_preserves_age_or_allows_only_safe_miss() -> None:
    async def scenario() -> None:
        async with AsyncExitStack() as cleanup:
            await _result_failover_async(cleanup)

    asyncio.run(scenario())


def test_metadata_failover_rotates_control_and_rejects_stale_keys() -> None:
    topology, backend, old_primary, vector = _begin_failover('metadata', 'noeviction')
    tag = DependencyTag.site('failover')
    result_key = KeyBuilder.qualified_key(BASE_KEY, vector, 'result')
    control_key = f'{topology.settings.l2.valkey.metadata.key_prefix}:v1:control'
    old_control = old_primary.get(control_key)
    assert isinstance(old_control, bytes)
    old_tag_key = (
        f'{topology.settings.l2.valkey.metadata.key_prefix}:v1:metadata:'
        f'{vector.incarnation}:{KeyBuilder.tag_key(tag)}'
    )
    old_result_lease = backend.reserve(result_key, 30)
    assert old_result_lease is not None
    assert backend.commit(old_result_lease, encode_value({'id': 'old'}), 600)
    old_fill_lease = backend.reserve(f'{result_key}-lease-loss', 30)
    assert old_fill_lease is not None
    backend.change((GenerationChange(tag),))
    acknowledged_tag = old_primary.get(old_tag_key)
    assert acknowledged_tag is not None

    failover = _promote('metadata', backend, old_primary)
    promoted = failover.promoted
    promoted_tag = promoted.get(old_tag_key)
    bump_outcome = ('absent', 'retained')[int(promoted_tag == acknowledged_tag)]
    print(f'ElastiCache acknowledged metadata bump was {bump_outcome} after failover')

    winning_incarnation = concurrent_replacement_incarnation(
        topology.settings, tag, vector.incarnation
    )
    deadline = time.monotonic() + 120
    replacement, neutral_failure, settled = observed_replacement(
        backend, tag, vector.incarnation, deadline, 1
    )
    assert replacement.incarnation == winning_incarnation
    assert_stale_metadata_control(promoted, control_key, old_control)

    new_result_key = KeyBuilder.qualified_key(BASE_KEY, replacement, 'result')
    _assert_replaced_leases(
        backend, result_key, new_result_key, (old_result_lease, old_fill_lease), deadline
    )
    _finish(failover, neutral_failure, settled)


def test_result_failover_preserves_age_or_allows_only_safe_miss() -> None:
    _, backend, old_primary, vector = _begin_failover('result', 'allkeys-lru')
    result_key = KeyBuilder.qualified_key(_RESULT_BASE_KEY, vector, 'result')
    lease = backend.reserve(result_key, 30)
    assert lease is not None
    assert backend.commit(lease, encode_value({'id': 1}), 600)
    # A commit cannot expire later than its completion plus the supplied lifetime.
    expiry_upper = time.monotonic() + 600
    before = backend.lookup(result_key)
    assert before is not None
    lost_lease = backend.reserve(f'{result_key}-lease-loss', 30)
    assert lost_lease is not None
    released = backend.reserve(f'{result_key}-release', 30)
    assert released is not None
    assert backend.release(released)

    failover = _promote('result', backend, old_primary)
    deadline = time.monotonic() + 120
    neutral_failure = False
    while time.monotonic() < deadline:
        try:
            after_started = time.monotonic()
            after = backend.lookup(result_key)
            break
        except BackendUnavailable:
            neutral_failure = True
            time.sleep(1)
    else:
        pytest.fail('result lookup did not recover before the failover deadline')
    first_write, write_failed, settled = observed_write(
        backend, f'{result_key}-first-write', deadline, 1
    )
    neutral_failure |= write_failed
    assert first_write is not None
    if after is not None:
        assert after_started + after.remaining_seconds <= expiry_upper + 0.01
    else:
        replacement = reserve_until(backend, result_key, deadline, 1)[0]
        assert replacement is not None
        assert replacement.token != lease.token
    replacement_lease = reserve_until(backend, f'{result_key}-lease-loss', deadline, 1)[0]
    assert replacement_lease is None or replacement_lease.token != lost_lease.token
    _finish(failover, neutral_failure, settled)
