import os
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import pytest
from valkey.exceptions import ConnectionError as ValkeyConnectionError
from valkey.exceptions import ReadOnlyError

from sg_query_cache._backend import BackendUnavailable, StaleIncarnation
from sg_query_cache._codec import encode_value
from sg_query_cache._keys import KeyBuilder
from sg_query_cache._model import DependencyTag, FillLease, GenerationChange, GenerationVector
from sg_query_cache._valkey_scripts import RESOLVE_SCRIPT_SHA
from sg_query_cache.valkey import ValkeyBackend
from tests._fakes import _RecordingMetrics
from tests.valkey._integration_helpers import (
    BASE_KEY,
    AdminClient,
    Role,
    RoleObservation,
    Topology,
    assert_role_replaced,
    backend_role_observation,
    eviction_policy,
    primary,
    replid,
    required,
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


@dataclass(frozen=True, slots=True)
class _Failover:
    role: Role
    backend: ValkeyBackend
    old_primary: AdminClient
    promoted: AdminClient
    retained: RoleObservation
    before_first_attempt: RoleObservation


def _assert_stale_metadata_control(
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


def _begin_failover(
    role: Role,
    expected_policy: str,
) -> tuple[Topology, ValkeyBackend, AdminClient, GenerationVector]:
    print(
        'ElastiCache test-failover quota: '
        f'{_FAILOVER_ATTEMPT_LIMIT} shard failovers per rolling {_FAILOVER_WINDOW_HOURS} hours'
    )
    topology = _make_topology('managed')
    backend = ValkeyBackend.from_settings(topology.settings, os.environ, _RecordingMetrics())
    old_primary = primary(topology, role)
    assert eviction_policy(role) == expected_policy
    vector = backend.resolve((DependencyTag.site('failover'),))
    return topology, backend, old_primary, vector


def _finish(failover: _Failover, neutral_failure: bool) -> None:
    assert neutral_failure
    assert_role_replaced(
        failover.retained,
        failover.before_first_attempt,
        backend_role_observation(failover.backend, failover.role),
        neutral_failure=neutral_failure,
        profile='managed',
    )
    failover.promoted.close()
    failover.old_primary.close()
    failover.backend.close()


def _reserve_until(backend: ValkeyBackend, key: str, deadline: float) -> FillLease | None:
    while time.monotonic() < deadline:
        try:
            return backend.reserve(key, 30)
        except BackendUnavailable:
            time.sleep(1)
    pytest.fail('ElastiCache result-role reserve did not recover before the deadline')


def _assert_replaced_leases(
    backend: ValkeyBackend,
    old_result_key: str,
    new_result_key: str,
    old_leases: tuple[FillLease, FillLease],
    deadline: float,
) -> None:
    assert new_result_key != old_result_key
    assert backend.lookup(new_result_key) is None
    replacement_result_lease = _reserve_until(backend, new_result_key, deadline)
    assert replacement_result_lease is not None
    assert replacement_result_lease.token != old_leases[0].token
    replacement_fill_lease = _reserve_until(backend, f'{new_result_key}-lease-loss', deadline)
    assert replacement_fill_lease is not None
    assert replacement_fill_lease.token != old_leases[1].token


def _wait_changed(role: Role, previous: str) -> AdminClient:
    topology = _make_topology('managed')
    deadline = time.monotonic() + 900
    while time.monotonic() < deadline:
        client = primary(topology, role)
        try:
            if replid(client) != previous:
                return client
        except Exception as error:  # noqa: BLE001 - bounded external acceptance polling
            _ = error
        client.close()
        time.sleep(1)
    pytest.fail(f'ElastiCache {role} failover did not expose a new replication identity')


def _promote(
    role: Role,
    backend: ValkeyBackend,
    old_primary: AdminClient,
) -> _Failover:
    previous_replid = replid(old_primary)
    retained = backend_role_observation(backend, role)
    _aws_failover(role)
    promoted = _wait_changed(role, previous_replid)
    with pytest.raises((ReadOnlyError, ValkeyConnectionError)):
        old_primary.set('sgqc-elasticache-demotion-probe', b'1')
    before_first_attempt = backend_role_observation(backend, role)
    return _Failover(role, backend, old_primary, promoted, retained, before_first_attempt)


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

    def replace_incarnation(_index: int) -> str:
        concurrent = ValkeyBackend.from_settings(
            topology.settings,
            os.environ,
            _RecordingMetrics(),
        )
        try:
            return concurrent.resolve((tag,)).incarnation
        finally:
            concurrent.close()

    with ThreadPoolExecutor(max_workers=50) as executor:
        incarnations = tuple(executor.map(replace_incarnation, range(50)))
    assert len(set(incarnations)) == 1
    winning_incarnation = incarnations[0]
    assert winning_incarnation != vector.incarnation

    deadline = time.monotonic() + 120
    replacement = None
    neutral_failure = False
    while time.monotonic() < deadline:
        try:
            replacement = backend.resolve((tag,))
        except (BackendUnavailable, StaleIncarnation):
            neutral_failure = True
            time.sleep(1)
            continue
        if replacement.incarnation != vector.incarnation:
            break
    assert replacement is not None
    assert replacement.incarnation == winning_incarnation
    _assert_stale_metadata_control(promoted, control_key, old_control)

    new_result_key = KeyBuilder.qualified_key(BASE_KEY, replacement, 'result')
    _assert_replaced_leases(
        backend, result_key, new_result_key, (old_result_lease, old_fill_lease), deadline
    )
    _finish(failover, neutral_failure)


def test_result_failover_preserves_age_or_allows_only_safe_miss() -> None:
    _, backend, old_primary, vector = _begin_failover('result', 'allkeys-lru')
    result_key = KeyBuilder.qualified_key(BASE_KEY, vector, 'result')
    lease = backend.reserve(result_key, 30)
    assert lease is not None
    assert backend.commit(lease, encode_value({'id': 1}), 600)
    before = backend.lookup(result_key)
    assert before is not None
    lost_lease = backend.reserve(f'{result_key}-lease-loss', 30)
    assert lost_lease is not None
    released = backend.reserve(f'{result_key}-release', 30)
    assert released is not None
    assert backend.release(released)

    failover = _promote('result', backend, old_primary)
    deadline = time.monotonic() + 120
    after = None
    neutral_failure = False
    while time.monotonic() < deadline:
        try:
            after = backend.lookup(result_key)
            break
        except BackendUnavailable:
            neutral_failure = True
            time.sleep(1)
    if after is not None:
        assert after.remaining_seconds <= before.remaining_seconds
    else:
        replacement = _reserve_until(backend, result_key, deadline)
        assert replacement is not None
        assert replacement.token != lease.token
    replacement_lease = _reserve_until(backend, f'{result_key}-lease-loss', deadline)
    assert replacement_lease is None or replacement_lease.token != lost_lease.token
    _finish(failover, neutral_failure)
