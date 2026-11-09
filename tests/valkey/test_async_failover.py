import asyncio
import os
import time
from contextlib import AsyncExitStack

import pytest
from valkey.exceptions import ConnectionError as ValkeyConnectionError
from valkey.exceptions import ReadOnlyError

from sg_query_cache._backend import BackendUnavailable
from sg_query_cache._codec import encode_value
from sg_query_cache._keys import KeyBuilder
from sg_query_cache._model import DependencyTag, GenerationChange
from sg_query_cache.valkey_async import AsyncValkeyBackend
from tests._fakes import RecordingMetrics
from tests.valkey._async_integration_helpers import (
    backend_role_observation,
    concurrent_replacement_incarnation,
    observed_replacement,
    observed_write,
    reserve_until,
)
from tests.valkey._environment import integer
from tests.valkey._integration_helpers import (
    Profile,
    assert_role_replaced,
    assert_stale_metadata_control,
    primary,
    replid,
    required,
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


async def _control(command: str) -> None:
    host = required('SG_QUERY_CACHE_TEST_FAILOVER_CONTROL_HOST')
    port = integer('SG_QUERY_CACHE_TEST_FAILOVER_CONTROL_PORT', os.environ)
    async with asyncio.timeout(5):
        reader, writer = await asyncio.open_connection(host, port)
        try:
            writer.write(f'{command}\n'.encode('ascii'))
            await writer.drain()
            reply = await reader.readline()
        finally:
            writer.close()
            await writer.wait_closed()
    assert reply == b'OK\n'


async def _metadata_failover(profile: Profile, cleanup: AsyncExitStack) -> None:
    topology = _make_topology(profile)
    backend = await AsyncValkeyBackend.from_settings(
        topology.settings, os.environ, RecordingMetrics()
    )
    cleanup.push_async_callback(backend.close)
    old_primary = primary(topology, 'metadata')
    cleanup.callback(old_primary.close)
    tag = DependencyTag.site('failover')
    old_vector = await backend.resolve((tag,))
    old_result_key = KeyBuilder.qualified_key(_METADATA_FAILOVER_BASE_KEY, old_vector, 'result')
    old_result_lease = await backend.reserve(old_result_key, 30)
    assert old_result_lease is not None
    assert await backend.commit(old_result_lease, encode_value({'id': 'old'}), 60)
    old_lease_key = f'{old_result_key}-lease'
    old_fill_lease = await backend.reserve(old_lease_key, 30)
    assert old_fill_lease is not None
    retained = backend_role_observation(backend, 'metadata')
    old_control = old_primary.get(f'{topology.settings.l2.valkey.metadata.key_prefix}:v1:control')
    assert isinstance(old_control, bytes)
    old_tag_key = (
        f'{topology.settings.l2.valkey.metadata.key_prefix}:v1:metadata:'
        f'{old_vector.incarnation}:{KeyBuilder.tag_key(tag)}'
    )
    await _control(f'PREPARE {profile} metadata')
    await backend.change((GenerationChange(tag),))
    changed_only_on_old_primary = old_primary.get(old_tag_key)
    assert changed_only_on_old_primary is not None
    old_replid = replid(old_primary)
    started = time.monotonic()
    await _control(f'PROMOTE {profile} metadata')
    promoted = await asyncio.to_thread(
        wait_for_new_primary,
        topology,
        'metadata',
        old_replid,
        timeout_seconds=_DEADLINE_SECONDS,
        interval_seconds=0.1,
    )
    cleanup.callback(promoted.close)
    print(f'pre-rotation interval started at monotonic {started:.6f}')
    with pytest.raises((ReadOnlyError, ValkeyConnectionError)):
        old_primary.set('sgqc-failover-demotion-probe', b'1')
    assert promoted.get(old_tag_key) != changed_only_on_old_primary
    before_first_attempt = backend_role_observation(backend, 'metadata')
    replacement_incarnation = await concurrent_replacement_incarnation(
        topology.settings, tag, old_vector.incarnation
    )
    new_vector, neutral_failure, settled = await observed_replacement(
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
    assert await backend.lookup(new_result_key) is None
    replacement_result_lease = await backend.reserve(new_result_key, 30)
    assert replacement_result_lease is not None
    assert replacement_result_lease.token != old_result_lease.token
    new_lease_key = f'{new_result_key}-lease'
    replacement_fill_lease = await backend.reserve(new_lease_key, 30)
    assert replacement_fill_lease is not None
    assert replacement_fill_lease.token != old_fill_lease.token


async def _result_failover(profile: Profile, cleanup: AsyncExitStack) -> None:
    topology = _make_topology(profile)
    backend = await AsyncValkeyBackend.from_settings(
        topology.settings, os.environ, RecordingMetrics()
    )
    cleanup.push_async_callback(backend.close)
    result_primary = primary(topology, 'result')
    cleanup.callback(result_primary.close)
    vector = await backend.resolve((DependencyTag.site('failover'),))
    result_key = KeyBuilder.qualified_key(_RESULT_FAILOVER_BASE_KEY, vector, 'result')
    lease = await backend.reserve(result_key, 30)
    assert lease is not None
    assert await backend.commit(lease, encode_value({'id': 1}), 60)
    # A commit cannot expire later than its completion plus the supplied lifetime.
    expiry_upper = time.monotonic() + 60
    before = await backend.lookup(result_key)
    assert before is not None
    lost_lease_key = f'{result_key}-lease-loss'
    lost_lease = await backend.reserve(lost_lease_key, 30)
    assert lost_lease is not None
    retained = backend_role_observation(backend, 'result')
    await _control(f'PREPARE {profile} result')
    old_replid = replid(result_primary)
    await _control(f'PROMOTE {profile} result')
    promoted = await asyncio.to_thread(
        wait_for_new_primary,
        topology,
        'result',
        old_replid,
        timeout_seconds=_DEADLINE_SECONDS,
        interval_seconds=0.1,
    )
    cleanup.callback(promoted.close)
    with pytest.raises((ReadOnlyError, ValkeyConnectionError)):
        result_primary.set('sgqc-failover-demotion-probe', b'1')
    before_first_attempt = backend_role_observation(backend, 'result')

    deadline = time.monotonic() + _DEADLINE_SECONDS
    neutral_failure = False
    while time.monotonic() < deadline:
        try:
            after_started = time.monotonic()
            after = await backend.lookup(result_key)
            break
        except BackendUnavailable:
            neutral_failure = True
            await asyncio.sleep(0.1)
    else:
        pytest.fail('result lookup did not recover before the failover deadline')
    first_write, write_failed, settled = await observed_write(
        backend, f'{result_key}-first-write', time.monotonic() + _DEADLINE_SECONDS, 0.1
    )
    neutral_failure |= write_failed
    assert first_write is not None
    if after is not None:
        assert after_started + after.remaining_seconds <= expiry_upper + 0.01
    else:
        replacement, reserve_failed = await reserve_until(
            backend, result_key, time.monotonic() + _DEADLINE_SECONDS, 0.1
        )
        neutral_failure |= reserve_failed
        assert replacement is not None
        assert replacement.token != lease.token
    replacement_lease, reserve_failed = await reserve_until(
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


@pytest.mark.parametrize('profile', ['sentinel', 'managed'])
def test_metadata_failover_rotates_before_reuse_and_converges(profile: Profile) -> None:
    async def scenario() -> None:
        async with AsyncExitStack() as cleanup:
            await _metadata_failover(profile, cleanup)

    asyncio.run(scenario())


@pytest.mark.parametrize('profile', ['sentinel', 'managed'])
def test_result_failover_preserves_age_or_causes_only_miss_and_duplicate_fill(
    profile: Profile,
) -> None:
    async def scenario() -> None:
        async with AsyncExitStack() as cleanup:
            await _result_failover(profile, cleanup)

    asyncio.run(scenario())
