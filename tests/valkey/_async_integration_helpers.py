import asyncio
import os
import time
from collections.abc import AsyncGenerator, Generator, Sequence
from contextlib import asynccontextmanager, contextmanager
from typing import cast

import pytest

from sg_query_cache._backend import BackendUnavailable, StaleIncarnation
from sg_query_cache._config import CacheSettings
from sg_query_cache._model import DependencyTag, FillLease, GenerationVector
from sg_query_cache.valkey_async import AsyncValkeyBackend, _PinnedRoleClient, _RoleClient
from tests._fakes import RecordingMetrics
from tests.valkey._integration_helpers import Role, RoleObservation

_CONVERGENCE_CLIENTS = 50
_CONVERGENCE_SECONDS = 30.0


def _backend_role_client(backend: AsyncValkeyBackend, role: Role) -> _RoleClient:
    if role == 'metadata':
        metadata = object.__getattribute__(backend, '_metadata')
        return cast('_RoleClient', object.__getattribute__(metadata, '_client'))
    return cast('_RoleClient', object.__getattribute__(backend, '_result_client'))


def backend_role_observation(
    backend: AsyncValkeyBackend,
    role: Role,
    *,
    served: Sequence[int] = (),
) -> RoleObservation:
    client = _backend_role_client(backend, role)
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


async def concurrent_replacement_incarnation(
    settings: CacheSettings,
    tag: DependencyTag,
    previous: str,
) -> str:
    deadline = time.monotonic() + _CONVERGENCE_SECONDS

    async def replace_incarnation(_index: int) -> tuple[str, int]:
        failures = 0
        while True:
            try:
                concurrent = await AsyncValkeyBackend.from_settings(
                    settings, os.environ, RecordingMetrics()
                )
                try:
                    return (await concurrent.resolve((tag,))).incarnation, failures
                finally:
                    await concurrent.close()
            except (BackendUnavailable, StaleIncarnation):
                failures += 1
                if time.monotonic() >= deadline:
                    raise
                await asyncio.sleep(0.1)

    tasks = tuple(
        asyncio.create_task(replace_incarnation(index)) for index in range(_CONVERGENCE_CLIENTS)
    )
    try:
        outcomes = await asyncio.gather(*tasks)
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        _ = await asyncio.gather(*tasks, return_exceptions=True)
    retried = sum(1 for _, failures in outcomes if failures)
    print(f'{retried} of {_CONVERGENCE_CLIENTS} replacement clients retried a neutral failure')
    incarnations = {incarnation for incarnation, _ in outcomes}
    assert len(incarnations) == 1
    (winner,) = incarnations
    assert winner != previous
    return winner


async def reserve_until(
    backend: AsyncValkeyBackend,
    key: str,
    deadline: float,
    interval_seconds: float,
) -> tuple[FillLease | None, bool]:
    neutral_failure = False
    while time.monotonic() < deadline:
        try:
            return await backend.reserve(key, 30), neutral_failure
        except BackendUnavailable:
            neutral_failure = True
            await asyncio.sleep(interval_seconds)
    pytest.fail('result-role reserve did not recover before the deadline')


async def resolve_replacement(
    backend: AsyncValkeyBackend,
    tag: DependencyTag,
    previous: str,
    deadline: float,
    interval_seconds: float,
) -> tuple[GenerationVector | None, bool]:
    vector = None
    neutral_failure = False
    while time.monotonic() < deadline:
        try:
            vector = await backend.resolve((tag,))
        except (BackendUnavailable, StaleIncarnation):
            neutral_failure = True
            await asyncio.sleep(interval_seconds)
            continue
        if vector.incarnation == previous:
            pytest.fail('a successful resolve after rotation delivered the old incarnation')
        break
    return vector, neutral_failure


@contextmanager
def served_pin_generations(
    backend: AsyncValkeyBackend,
    role: Role,
) -> Generator[list[int]]:
    role_client = _backend_role_client(backend, role)
    original = _RoleClient.pin
    served: list[int] = []

    @asynccontextmanager
    async def pin(self: _RoleClient) -> AsyncGenerator[_PinnedRoleClient]:
        async with original(self) as pinned:
            yield pinned
            generation = pinned.connection_generation()
            if self is role_client and generation is not None:
                served.append(generation)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(_RoleClient, 'pin', pin)
        yield served


async def observed_replacement(
    backend: AsyncValkeyBackend,
    tag: DependencyTag,
    previous: str,
    deadline: float,
    interval_seconds: float,
) -> tuple[GenerationVector, bool, RoleObservation]:
    metrics = cast('RecordingMetrics', object.__getattribute__(backend, '_metrics'))
    recorded = len(metrics.increments)
    with served_pin_generations(backend, 'metadata') as served:
        vector, neutral_failure = await resolve_replacement(
            backend, tag, previous, deadline, interval_seconds
        )
    settled = backend_role_observation(backend, 'metadata', served=served)
    assert vector is not None
    # The first successful resolve fully validated control state before delivery.
    assert ('cache_incarnation', {'outcome': 'validated'}) in metrics.increments[recorded:]
    return vector, neutral_failure, settled


async def observed_write(
    backend: AsyncValkeyBackend,
    key: str,
    deadline: float,
    interval_seconds: float,
) -> tuple[FillLease | None, bool, RoleObservation]:
    # A read may reach a demoted node; the first write proves a replacement primary.
    with served_pin_generations(backend, 'result') as served:
        lease, neutral_failure = await reserve_until(backend, key, deadline, interval_seconds)
    return lease, neutral_failure, backend_role_observation(backend, 'result', served=served)
