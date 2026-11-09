import os
import socket
import time

import pytest
from valkey.exceptions import ConnectionError as ValkeyConnectionError
from valkey.exceptions import ReadOnlyError

from sg_query_cache._backend import BackendUnavailable
from sg_query_cache._codec import encode_value
from sg_query_cache._keys import KeyBuilder
from sg_query_cache._model import DependencyTag, GenerationChange
from sg_query_cache.valkey import ValkeyBackend
from tests._fakes import RecordingMetrics
from tests.valkey._environment import integer
from tests.valkey._integration_helpers import (
    Profile,
    assert_role_replaced,
    assert_stale_metadata_control,
    backend_role_observation,
    concurrent_replacement_incarnation,
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


@pytest.mark.parametrize('profile', ['sentinel', 'managed'])
def test_metadata_failover_rotates_before_reuse_and_converges(
    profile: Profile,
) -> None:
    topology = _make_topology(profile)
    backend = ValkeyBackend.from_settings(topology.settings, os.environ, RecordingMetrics())
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
    promoted = wait_for_new_primary(
        topology,
        'metadata',
        old_replid,
        timeout_seconds=_DEADLINE_SECONDS,
        interval_seconds=0.1,
    )
    print(f'pre-rotation interval started at monotonic {started:.6f}')
    with pytest.raises((ReadOnlyError, ValkeyConnectionError)):
        old_primary.set('sgqc-failover-demotion-probe', b'1')
    assert promoted.get(old_tag_key) != changed_only_on_old_primary
    before_first_attempt = backend_role_observation(backend, 'metadata')
    replacement_incarnation = concurrent_replacement_incarnation(
        topology.settings, tag, old_vector.incarnation
    )
    new_vector, neutral_failure, settled = observed_replacement(
        backend, tag, old_vector.incarnation, time.monotonic() + _DEADLINE_SECONDS, 0.1
    )
    assert new_vector.incarnation == replacement_incarnation
    assert_role_replaced(
        retained,
        before_first_attempt,
        settled,
        neutral_failure=neutral_failure,
        profile=profile,
    )
    assert_stale_metadata_control(
        promoted,
        f'{topology.settings.l2.valkey.metadata.key_prefix}:v1:control',
        old_control,
    )

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
    backend = ValkeyBackend.from_settings(topology.settings, os.environ, RecordingMetrics())
    result_primary = primary(topology, 'result')
    vector = backend.resolve((DependencyTag.site('failover'),))
    result_key = KeyBuilder.qualified_key(_RESULT_FAILOVER_BASE_KEY, vector, 'result')
    lease = backend.reserve(result_key, 30)
    assert lease is not None
    assert backend.commit(lease, encode_value({'id': 1}), 60)
    # A commit cannot expire later than its completion plus the supplied lifetime.
    expiry_upper = time.monotonic() + 60
    before = backend.lookup(result_key)
    assert before is not None
    lost_lease_key = f'{result_key}-lease-loss'
    lost_lease = backend.reserve(lost_lease_key, 30)
    assert lost_lease is not None
    retained = backend_role_observation(backend, 'result')
    _control(f'PREPARE {profile} result')
    old_replid = replid(result_primary)
    _control(f'PROMOTE {profile} result')
    promoted = wait_for_new_primary(
        topology,
        'result',
        old_replid,
        timeout_seconds=_DEADLINE_SECONDS,
        interval_seconds=0.1,
    )
    with pytest.raises((ReadOnlyError, ValkeyConnectionError)):
        result_primary.set('sgqc-failover-demotion-probe', b'1')
    before_first_attempt = backend_role_observation(backend, 'result')

    deadline = time.monotonic() + _DEADLINE_SECONDS
    neutral_failure = False
    while time.monotonic() < deadline:
        try:
            after_started = time.monotonic()
            after = backend.lookup(result_key)
            break
        except BackendUnavailable:
            neutral_failure = True
            time.sleep(0.1)
    else:
        pytest.fail('result lookup did not recover before the failover deadline')
    first_write, write_failed, settled = observed_write(
        backend, f'{result_key}-first-write', time.monotonic() + _DEADLINE_SECONDS, 0.1
    )
    neutral_failure |= write_failed
    assert first_write is not None
    if after is not None:
        assert after_started + after.remaining_seconds <= expiry_upper + 0.01
    else:
        replacement, reserve_failed = reserve_until(
            backend, result_key, time.monotonic() + _DEADLINE_SECONDS, 0.1
        )
        neutral_failure |= reserve_failed
        assert replacement is not None
        assert replacement.token != lease.token
    replacement_lease, reserve_failed = reserve_until(
        backend, lost_lease_key, time.monotonic() + _DEADLINE_SECONDS, 0.1
    )
    neutral_failure |= reserve_failed
    assert replacement_lease is None or replacement_lease.token != lost_lease.token
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
