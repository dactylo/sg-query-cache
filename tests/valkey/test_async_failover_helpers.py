import asyncio
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import pytest

from sg_query_cache._backend import BackendUnavailable, StaleIncarnation
from sg_query_cache._model import DependencyTag, FillLease, GenerationVector
from sg_query_cache.valkey_async import AsyncValkeyBackend, _PinnedRoleClient, _RoleClient
from tests._fakes import RecordingMetrics
from tests.valkey import _async_integration_helpers as helpers
from tests.valkey._integration_helpers import assert_role_replaced

if TYPE_CHECKING:
    from sg_query_cache._config import CacheSettings


class _FakeBackend:
    def __init__(self, role: _RoleClient) -> None:
        self._metadata = SimpleNamespace(_client=role)
        self._result_client = role
        self._metrics = RecordingMetrics()
        self.resolutions: list[GenerationVector | Exception] = []
        self.reservations: list[FillLease | Exception | None] = []
        self.resolved = 0
        self.reserved = 0

    async def reserve(self, key: str, lease_seconds: float) -> FillLease | None:
        assert key == 'key'
        assert lease_seconds == 30
        self.reserved += 1
        async with self._result_client.pin():
            outcome = self.reservations.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

    async def resolve(self, tags: tuple[DependencyTag, ...]) -> GenerationVector:
        assert tags == (DependencyTag.site('site'),)
        self.resolved += 1
        async with self._result_client.pin():
            outcome = self.resolutions.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome


class _FakePin:
    def __init__(self, generation: int) -> None:
        self._generation = generation

    def connection_generation(self) -> int:
        return self._generation


def _backend(monkeypatch: pytest.MonkeyPatch) -> tuple[_FakeBackend, _RoleClient]:
    role = object.__new__(_RoleClient)
    object.__setattr__(role, '_affinity', SimpleNamespace(check=lambda: None))
    object.__setattr__(role, '_client', object())
    object.__setattr__(role, '_connections', SimpleNamespace(generation=7))
    object.__setattr__(role, '_endpoint', None)
    object.__setattr__(role, '_pool_epoch', 2)
    object.__setattr__(role, '_test_sockets', [7, 7])
    object.__setattr__(role, '_test_serving', 0)

    @asynccontextmanager
    async def pin(self: _RoleClient) -> AsyncGenerator[_PinnedRoleClient]:
        sockets = object.__getattribute__(self, '_test_sockets')
        serving = object.__getattribute__(self, '_test_serving')
        yield cast('_PinnedRoleClient', _FakePin(sockets[serving]))

    monkeypatch.setattr(_RoleClient, 'pin', pin)
    return _FakeBackend(role), role


def test_convergence_failure_cancels_and_closes_other_clients(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Client:
        def __init__(self, index: int) -> None:
            self._index = index
            self.closed = False
            self.cancelled = False

        async def close(self) -> None:
            self.closed = True

        async def resolve(self, tags: tuple[DependencyTag, ...]) -> GenerationVector:
            assert tags == (DependencyTag.site('site'),)
            if self._index == 0:
                await asyncio.sleep(0)
                raise BackendUnavailable('exhausted')
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                self.cancelled = True
                raise
            raise AssertionError('unreachable')

    async def scenario() -> None:
        clients: list[_Client] = []

        async def create(*args: object) -> AsyncValkeyBackend:
            assert len(args) == 3
            client = _Client(len(clients))
            clients.append(client)
            return cast('AsyncValkeyBackend', client)

        monkeypatch.setattr(AsyncValkeyBackend, 'from_settings', create)
        monkeypatch.setattr(helpers, '_CONVERGENCE_SECONDS', 0.0)
        with pytest.raises(BackendUnavailable, match='exhausted'):
            await helpers.concurrent_replacement_incarnation(
                cast('CacheSettings', object()), DependencyTag.site('site'), 'old'
            )
        assert len(clients) == 50
        assert all(client.closed for client in clients)
        assert all(client.cancelled for client in clients[1:])

    asyncio.run(scenario())


def test_observed_replacement_rejects_missing_deadline_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        fake, _ = _backend(monkeypatch)
        with pytest.raises(AssertionError):
            await helpers.observed_replacement(
                cast('AsyncValkeyBackend', fake), DependencyTag.site('site'), 'old', 0, 0
            )
        assert fake.resolved == 0

    asyncio.run(scenario())


def test_observed_replacement_requires_new_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        fake, role = _backend(monkeypatch)
        backend = cast('AsyncValkeyBackend', fake)
        tag = DependencyTag.site('site')
        fresh = GenerationVector('fresh', ((tag, 1),))
        fake._metrics.increment('cache_incarnation', {'outcome': 'validated'})
        fake.resolutions = [fresh]
        with pytest.raises(AssertionError):
            await helpers.observed_replacement(backend, tag, 'old', time.monotonic() + 1, 0)
        fake.resolutions = [fresh]
        object.__setattr__(role, '_test_sockets', [8, 8])
        object.__getattribute__(role, '_connections').generation = 8
        original = fake.resolve

        async def validated(tags: tuple[DependencyTag, ...]) -> GenerationVector:
            result = await original(tags)
            fake._metrics.increment('cache_incarnation', {'outcome': 'validated'})
            return result

        monkeypatch.setattr(fake, 'resolve', validated)
        vector, failed, settled = await helpers.observed_replacement(
            backend, tag, 'old', time.monotonic() + 1, 0
        )
        assert vector == fresh
        assert not failed
        assert settled.served_generation == 8

    asyncio.run(scenario())


def test_observed_write_captures_first_success_after_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        fake, role = _backend(monkeypatch)
        backend = cast('AsyncValkeyBackend', fake)
        lease = FillLease('key', 'token')
        fake.reservations = [BackendUnavailable('retry'), lease]
        original = fake.reserve

        async def reconnect(key: str, seconds: float) -> FillLease | None:
            if fake.reserved:
                object.__setattr__(role, '_test_sockets', [8, 8])
                object.__getattribute__(role, '_connections').generation = 8
            return await original(key, seconds)

        monkeypatch.setattr(fake, 'reserve', reconnect)
        first, failed, settled = await helpers.observed_write(
            backend, 'key', time.monotonic() + 1, 0
        )
        assert first == lease
        assert failed
        assert fake.reserved == 2
        assert settled.served_generation == 8

    asyncio.run(scenario())


def test_reserve_deadline_reports_failed_recovery(monkeypatch: pytest.MonkeyPatch) -> None:
    async def scenario() -> None:
        fake, _ = _backend(monkeypatch)
        with pytest.raises(pytest.fail.Exception, match='did not recover'):
            await helpers.reserve_until(cast('AsyncValkeyBackend', fake), 'key', 0, 0)

    asyncio.run(scenario())


def test_resolve_retries_and_rejects_old_success(monkeypatch: pytest.MonkeyPatch) -> None:
    async def scenario() -> None:
        fake, _ = _backend(monkeypatch)
        backend = cast('AsyncValkeyBackend', fake)
        tag = DependencyTag.site('site')
        fresh = GenerationVector('fresh', ((tag, 1),))
        fake.resolutions = [BackendUnavailable('retry'), StaleIncarnation('retry'), fresh]
        vector, failed = await helpers.resolve_replacement(
            backend, tag, 'old', time.monotonic() + 1, 0
        )
        assert vector == fresh
        assert failed
        assert fake.resolved == 3
        fake.resolutions = [GenerationVector('old', ((tag, 1),))]
        with pytest.raises(pytest.fail.Exception, match='old incarnation'):
            await helpers.resolve_replacement(backend, tag, 'old', time.monotonic() + 1, 0)

    asyncio.run(scenario())


@pytest.mark.parametrize(('serving', 'accepted'), [(0, False), (1, True)])
def test_served_pin_generation_identifies_the_serving_socket(
    monkeypatch: pytest.MonkeyPatch,
    serving: int,
    accepted: bool,
) -> None:
    async def scenario() -> None:
        fake, role = _backend(monkeypatch)
        backend = cast('AsyncValkeyBackend', fake)
        retained = helpers.backend_role_observation(backend, 'result')
        object.__getattribute__(role, '_test_sockets')[1] = 8
        object.__getattribute__(role, '_connections').generation = 8
        object.__setattr__(role, '_test_serving', serving)
        with helpers.served_pin_generations(backend, 'result') as served:
            async with role.pin():
                pass
        settled = helpers.backend_role_observation(backend, 'result', served=served)
        assert settled.connection_generation > retained.connection_generation
        assert settled.served_generation == (7 if serving == 0 else 8)
        if accepted:
            assert_role_replaced(
                retained, retained, settled, neutral_failure=False, profile='managed'
            )
        else:
            with pytest.raises(AssertionError):
                assert_role_replaced(
                    retained, retained, settled, neutral_failure=False, profile='managed'
                )

    asyncio.run(scenario())


@pytest.mark.parametrize('mode', ['retry', 'split', 'old'])
def test_concurrent_replacement_requires_one_new_winner(
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
) -> None:
    class _Client:
        def __init__(self, index: int) -> None:
            self._index = index
            self.closed = False

        async def close(self) -> None:
            self.closed = True

        async def resolve(self, tags: tuple[DependencyTag, ...]) -> GenerationVector:
            assert tags == (DependencyTag.site('site'),)
            if mode == 'retry' and self._index == 0:
                raise BackendUnavailable('retry')
            incarnation = 'old' if mode == 'old' else 'fresh'
            if mode == 'split' and self._index == 1:
                incarnation = 'other'
            return GenerationVector(incarnation, ((tags[0], 1),))

    async def scenario() -> None:
        clients: list[_Client] = []

        async def create(*args: object) -> AsyncValkeyBackend:
            assert len(args) == 3
            client = _Client(len(clients))
            clients.append(client)
            return cast('AsyncValkeyBackend', client)

        monkeypatch.setattr(AsyncValkeyBackend, 'from_settings', create)
        action = helpers.concurrent_replacement_incarnation(
            cast('CacheSettings', object()), DependencyTag.site('site'), 'old'
        )
        if mode == 'retry':
            assert await action == 'fresh'
            assert len(clients) == 51
        else:
            with pytest.raises(AssertionError):
                await action
            assert len(clients) == 50
        assert all(client.closed for client in clients)

    asyncio.run(scenario())
