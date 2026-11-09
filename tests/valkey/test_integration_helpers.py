from types import SimpleNamespace
from typing import cast

import pytest

import sg_query_cache.valkey as valkey_module
import tests.valkey._integration_helpers as integration_helpers
from sg_query_cache.valkey import (
    ValkeyBackend,
    _RoleClient,
)
from tests._fakes import RecordingMetrics
from tests.valkey._control_helpers import (
    AlreadyConnectedObservable,
    AtomicControlClient,
    ReconnectingBorrowedClient,
    enabled_settings,
)


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


@pytest.mark.parametrize(('serving', 'accepted'), [(0, False), (1, True)])
def test_served_pin_generation_identifies_the_socket_that_served(
    monkeypatch: pytest.MonkeyPatch,
    serving: int,
    accepted: bool,
) -> None:

    class _TwoSocketClient(AtomicControlClient):
        def __init__(self) -> None:
            super().__init__()
            self.serving = 0
            self.sockets = (AlreadyConnectedObservable(), AlreadyConnectedObservable())

        def client(self) -> ReconnectingBorrowedClient:
            socket = self.sockets[self.serving]
            socket.connect()
            return ReconnectingBorrowedClient(self, socket)

    pooled = _TwoSocketClient()

    def create_client(**kwargs: object) -> AtomicControlClient:
        return pooled if kwargs['host'] == 'result.example' else AtomicControlClient()

    monkeypatch.setattr(valkey_module, 'Valkey', create_client)
    backend = ValkeyBackend.from_settings(
        enabled_settings('managed'),
        {'METADATA_PASSWORD': 'metadata-secret', 'RESULT_PASSWORD': 'result-secret'},
        RecordingMetrics(),
    )
    role = cast('_RoleClient', object.__getattribute__(backend, '_result_client'))
    for socket in (0, 1):
        pooled.serving = socket
        role.get('warm')
    retained = integration_helpers.backend_role_observation(backend, 'result')
    pooled.sockets[1].disconnect()
    pooled.serving = 1
    role.get('reconnect')
    pooled.serving = serving

    with integration_helpers.served_pin_generations(backend, 'result') as served:
        role.get('served')

    settled = integration_helpers.backend_role_observation(backend, 'result', served=served)
    assert settled.connection_generation > retained.connection_generation
    if accepted:
        integration_helpers.assert_role_replaced(
            retained, retained, settled, neutral_failure=False, profile='managed'
        )
    else:
        with pytest.raises(AssertionError):
            integration_helpers.assert_role_replaced(
                retained, retained, settled, neutral_failure=False, profile='managed'
            )


@pytest.mark.parametrize(
    ('retained', 'settled', 'neutral_failure', 'profile'),
    [
        pytest.param(
            integration_helpers.RoleObservation(2, 7, None, 41),
            integration_helpers.RoleObservation(2, 7, None, 41, 7),
            False,
            'managed',
            id='managed-without-reconnect',
        ),
        pytest.param(
            integration_helpers.RoleObservation(2, 7, None, 41),
            integration_helpers.RoleObservation(2, 9, None, 41, 6),
            False,
            'managed',
            id='other-socket-reconnected',
        ),
        pytest.param(
            integration_helpers.RoleObservation(2, 7, None, 41),
            integration_helpers.RoleObservation(2, 9, None, 41),
            False,
            'managed',
            id='served-socket-unrecorded',
        ),
        pytest.param(
            integration_helpers.RoleObservation(2, 7, None, 41),
            integration_helpers.RoleObservation(2, 9, None, 42, 9),
            False,
            'managed',
            id='unsignaled-pool-replacement',
        ),
        pytest.param(
            integration_helpers.RoleObservation(2, 7, None, 41),
            integration_helpers.RoleObservation(2, 9, None, 41, 9),
            True,
            'managed',
            id='signal-without-pool-replacement',
        ),
        pytest.param(
            integration_helpers.RoleObservation(2, 7, ('primary-a', 6379), 41),
            integration_helpers.RoleObservation(2, 9, ('primary-a', 6379), 41, 9),
            False,
            'sentinel',
            id='sentinel-without-signal',
        ),
    ],
)
def test_role_replacement_rejects_unproven_transitions(
    retained: integration_helpers.RoleObservation,
    settled: integration_helpers.RoleObservation,
    neutral_failure: bool,
    profile: integration_helpers.Profile,
) -> None:
    with pytest.raises(AssertionError):
        integration_helpers.assert_role_replaced(
            retained, retained, settled, neutral_failure=neutral_failure, profile=profile
        )


@pytest.mark.parametrize(
    ('settled', 'neutral_failure'),
    [
        pytest.param(
            integration_helpers.RoleObservation(3, 9, None, 42, 9), True, id='signaled-replacement'
        ),
        pytest.param(
            integration_helpers.RoleObservation(2, 9, None, 41, 9), False, id='managed-reconnect'
        ),
    ],
)
def test_role_replacement_accepts_signaled_replacement_or_managed_reconnect(
    settled: integration_helpers.RoleObservation,
    neutral_failure: bool,
) -> None:
    retained = integration_helpers.RoleObservation(2, 7, None, 41)

    integration_helpers.assert_role_replaced(
        retained, retained, settled, neutral_failure=neutral_failure, profile='managed'
    )
