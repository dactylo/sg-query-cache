import os
import socket
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
from valkey.exceptions import ConnectionError as ValkeyConnectionError
from valkey.exceptions import ReadOnlyError

from sg_query_cache._backend import BackendUnavailable, StaleIncarnation
from sg_query_cache._codec import encode_value
from sg_query_cache._keys import KeyBuilder
from sg_query_cache._model import DependencyTag, FillLease, GenerationChange
from sg_query_cache._valkey_scripts import RESOLVE_SCRIPT_SHA
from sg_query_cache.valkey import ValkeyBackend
from tests._fakes import _RecordingMetrics
from tests.valkey._environment import integer
from tests.valkey._integration_helpers import (
    AdminClient,
    Profile,
    Role,
    Topology,
    assert_role_replaced,
    backend_role_observation,
    primary,
    replid,
    required,
)
from tests.valkey._integration_helpers import (
    topology as _make_topology,
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get('SG_QUERY_CACHE_REQUIRE_INTEGRATION') != '1',
        reason='requires the controller-owned failover topology',
    ),
]

_DEADLINE_SECONDS = 30.0
_METADATA_FAILOVER_BASE_KEY = f'sgqc:v1:base:{"b" * 64}'
_RESULT_FAILOVER_BASE_KEY = f'sgqc:v1:base:{"c" * 64}'


def _control(command: str) -> None:
    host = required('SG_QUERY_CACHE_TEST_FAILOVER_CONTROL_HOST')
    port = integer('SG_QUERY_CACHE_TEST_FAILOVER_CONTROL_PORT', os.environ)
    with socket.create_connection((host, port), timeout=5) as connection:
        connection.sendall(f'{command}\n'.encode('ascii'))
        reply = connection.makefile('rb').readline(128)
    assert reply == b'OK\n'


def _reserve_after_failover(
    backend: ValkeyBackend,
    key: str,
) -> tuple[FillLease | None, bool]:
    deadline = time.monotonic() + _DEADLINE_SECONDS
    neutral_failure = False
    while time.monotonic() < deadline:
        try:
            return backend.reserve(key, 30), neutral_failure
        except BackendUnavailable:
            neutral_failure = True
            time.sleep(0.1)
    pytest.fail('result-role reserve did not recover before the failover deadline')


def _wait_for_new_primary(
    topology: Topology,
    role: Role,
    previous_replid: str,
) -> AdminClient:
    deadline = time.monotonic() + _DEADLINE_SECONDS
    while time.monotonic() < deadline:
        client = primary(topology, role)
        try:
            if replid(client) != previous_replid:
                return client
        except Exception as error:  # noqa: BLE001 - bounded external acceptance polling
            _ = error
        client.close()
        time.sleep(0.1)
    pytest.fail(f'{topology._profile} {role} promotion did not expose a new replication identity')


@pytest.mark.parametrize('profile', ['sentinel', 'managed'])
def test_metadata_failover_rotates_before_reuse_and_converges(  # noqa: PLR0915 - one failover retains its complete invariant sequence
    profile: Profile,
) -> None:
    topology = _make_topology(profile)
    backend = ValkeyBackend.from_settings(topology.settings, os.environ, _RecordingMetrics())
    old_primary = primary(topology, 'metadata')
    tag = DependencyTag.site('failover')
    old_vector = backend.resolve((tag,))
    old_result_key = KeyBuilder.qualified_key(_METADATA_FAILOVER_BASE_KEY, old_vector, 'result')
    old_result_lease = backend.reserve(old_result_key, 30)
    assert old_result_lease is not None
    assert backend.commit(old_result_lease, encode_value({'id': 'old'}), 60)
    old_lease_key = f'{old_result_key}-lease'
    old_fill_lease = backend.reserve(old_lease_key, 30)
    assert old_fill_lease is not None
    retained = backend_role_observation(backend, 'metadata')
    old_control = old_primary.get(f'{topology.settings.l2.valkey.metadata.key_prefix}:v1:control')
    assert isinstance(old_control, bytes)
    old_tag_key = (
        f'{topology.settings.l2.valkey.metadata.key_prefix}:v1:metadata:'
        f'{old_vector.incarnation}:{KeyBuilder.tag_key(tag)}'
    )
    _control(f'PREPARE {profile} metadata')
    backend.change((GenerationChange(tag),))
    changed_only_on_old_primary = old_primary.get(old_tag_key)
    assert changed_only_on_old_primary is not None
    old_replid = replid(old_primary)
    started = time.monotonic()
    _control(f'PROMOTE {profile} metadata')
    promoted = _wait_for_new_primary(topology, 'metadata', old_replid)
    print(f'pre-rotation interval started at monotonic {started:.6f}')
    with pytest.raises((ReadOnlyError, ValkeyConnectionError)):
        old_primary.set('sgqc-failover-demotion-probe', b'1')
    assert promoted.get(old_tag_key) != changed_only_on_old_primary
    before_first_attempt = backend_role_observation(backend, 'metadata')

    def replace_incarnation(_index: int) -> str:
        concurrent = ValkeyBackend.from_settings(
            topology.settings, os.environ, _RecordingMetrics()
        )
        try:
            return concurrent.resolve((tag,)).incarnation
        finally:
            concurrent.close()

    with ThreadPoolExecutor(max_workers=50) as executor:
        replacement_incarnations = tuple(executor.map(replace_incarnation, range(50)))
    assert len(set(replacement_incarnations)) == 1
    replacement_incarnation = replacement_incarnations[0]
    assert replacement_incarnation != old_vector.incarnation

    deadline = time.monotonic() + _DEADLINE_SECONDS
    new_vector = None
    neutral_failure = False
    while time.monotonic() < deadline:
        try:
            new_vector = backend.resolve((tag,))
        except (BackendUnavailable, StaleIncarnation):
            neutral_failure = True
            time.sleep(0.1)
            continue
        if new_vector.incarnation != old_vector.incarnation:
            break
        time.sleep(0.1)
    assert new_vector is not None
    assert new_vector.incarnation == replacement_incarnation
    settled = backend_role_observation(backend, 'metadata')
    assert_role_replaced(
        retained,
        before_first_attempt,
        settled,
        neutral_failure=neutral_failure,
        profile=profile,
    )
    assert (
        promoted.get(f'{topology.settings.l2.valkey.metadata.key_prefix}:v1:control')
        != old_control
    )
    old_counter = old_control.split(b'|')[2].decode('ascii')
    stale_reply = promoted.evalsha(
        RESOLVE_SCRIPT_SHA,
        2,
        f'{topology.settings.l2.valkey.metadata.key_prefix}:v1:control',
        old_counter,
        old_control,
        7_200_000,
    )
    assert stale_reply in ([b'stale'], [b'missing-counter'])

    new_result_key = KeyBuilder.qualified_key(_METADATA_FAILOVER_BASE_KEY, new_vector, 'result')
    assert new_result_key != old_result_key
    assert backend.lookup(new_result_key) is None
    replacement_result_lease = backend.reserve(new_result_key, 30)
    assert replacement_result_lease is not None
    assert replacement_result_lease.token != old_result_lease.token
    new_lease_key = f'{new_result_key}-lease'
    replacement_fill_lease = backend.reserve(new_lease_key, 30)
    assert replacement_fill_lease is not None
    assert replacement_fill_lease.token != old_fill_lease.token
    promoted.close()
    old_primary.close()
    backend.close()


@pytest.mark.parametrize('profile', ['sentinel', 'managed'])
def test_result_failover_preserves_age_or_causes_only_miss_and_duplicate_fill(
    profile: Profile,
) -> None:
    topology = _make_topology(profile)
    backend = ValkeyBackend.from_settings(topology.settings, os.environ, _RecordingMetrics())
    result_primary = primary(topology, 'result')
    vector = backend.resolve((DependencyTag.site('failover'),))
    result_key = KeyBuilder.qualified_key(_RESULT_FAILOVER_BASE_KEY, vector, 'result')
    lease = backend.reserve(result_key, 30)
    assert lease is not None
    assert backend.commit(lease, encode_value({'id': 1}), 60)
    before = backend.lookup(result_key)
    assert before is not None
    lost_lease_key = f'{result_key}-lease-loss'
    lost_lease = backend.reserve(lost_lease_key, 30)
    assert lost_lease is not None
    retained = backend_role_observation(backend, 'result')
    _control(f'PREPARE {profile} result')
    old_replid = replid(result_primary)
    _control(f'PROMOTE {profile} result')
    promoted = _wait_for_new_primary(topology, 'result', old_replid)
    with pytest.raises((ReadOnlyError, ValkeyConnectionError)):
        result_primary.set('sgqc-failover-demotion-probe', b'1')
    before_first_attempt = backend_role_observation(backend, 'result')

    deadline = time.monotonic() + _DEADLINE_SECONDS
    after = None
    neutral_failure = False
    while time.monotonic() < deadline:
        try:
            after = backend.lookup(result_key)
            break
        except BackendUnavailable:
            neutral_failure = True
            time.sleep(0.1)
    if after is not None:
        assert after.remaining_seconds <= before.remaining_seconds
    else:
        replacement, reserve_failed = _reserve_after_failover(backend, result_key)
        neutral_failure |= reserve_failed
        assert replacement is not None
        assert replacement.token != lease.token
    replacement_lease, reserve_failed = _reserve_after_failover(backend, lost_lease_key)
    neutral_failure |= reserve_failed
    assert replacement_lease is None or replacement_lease.token != lost_lease.token
    settled = backend_role_observation(backend, 'result')
    assert_role_replaced(
        retained,
        before_first_attempt,
        settled,
        neutral_failure=neutral_failure,
        profile=profile,
    )
    promoted.close()
    result_primary.close()
    backend.close()
