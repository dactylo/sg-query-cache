import asyncio
from collections.abc import Callable, Mapping
from dataclasses import replace
from threading import Event, Lock
from typing import Literal

from sg_query_cache._backend import BackendUnavailable, CacheBackend
from sg_query_cache._config import CacheSettings, L2Settings, LimitSettings, _ValkeyMode
from sg_query_cache._keys import KeyBuilder
from sg_query_cache._model import (
    CacheHit,
    DependencyPlan,
    DependencyTag,
    ExactCall,
    FillLease,
    GenerationChange,
    GenerationVector,
    Identity,
)

CLIENT_OPTIONS = {'localized': False, 'convert_datetimes_to_utc': True}
PRIMARY_INFO = b'role:master\r\nmaster_replid:primary\r\n'


class DelegatingBackend:
    __slots__ = ('backend',)

    def __init__(self, backend: CacheBackend) -> None:
        self.backend = backend

    def change(self, changes: tuple[GenerationChange, ...]) -> None:
        self.backend.change(changes)

    def close(self) -> None:
        self.backend.close()

    def commit(self, lease: FillLease, payload: bytes, remaining_seconds: float) -> bool:
        return self.backend.commit(lease, payload, remaining_seconds)

    def flush_site(self, site_namespace: str, reason: str) -> None:
        self.backend.flush_site(site_namespace, reason)

    def lookup(self, key: str) -> CacheHit | None:
        return self.backend.lookup(key)

    def release(self, lease: FillLease) -> bool:
        return self.backend.release(lease)

    def reserve(self, key: str, lease_seconds: float) -> FillLease | None:
        return self.backend.reserve(key, lease_seconds)

    def resolve(self, tags: tuple[DependencyTag, ...]) -> GenerationVector:
        return self.backend.resolve(tags)


class FakeClock:
    __slots__ = ('_lock', '_value')

    def __init__(self, start: float = 1_000.0) -> None:
        self._lock = Lock()
        self._value = start

    def __call__(self) -> float:
        with self._lock:
            return self._value

    def advance(self, seconds: float) -> None:
        with self._lock:
            self._value += seconds


class ObservedBackend(DelegatingBackend):
    __slots__ = (
        '_commit_outcome',
        '_on_commit',
        '_replacement_payload',
        'lookup_count',
        'on_lookup',
        'resolved',
    )

    def __init__(
        self,
        backend: CacheBackend,
        commit_outcome: Literal['delegate', 'fail', 'reject'] = 'delegate',
        on_commit: Callable[[], None] | None = None,
        on_lookup: Callable[[], None] | None = None,
        replacement_payload: bytes | None = None,
    ) -> None:
        super().__init__(backend)
        self._commit_outcome = commit_outcome
        self.lookup_count = 0
        self._on_commit = on_commit
        self.on_lookup = on_lookup
        self._replacement_payload = replacement_payload
        self.resolved: list[tuple[DependencyTag, ...]] = []

    def commit(self, lease: FillLease, payload: bytes, remaining_seconds: float) -> bool:
        if self._commit_outcome == 'delegate':
            committed = self.backend.commit(lease, payload, remaining_seconds)
        else:
            replacement_payload = self._replacement_payload
            assert replacement_payload is not None
            assert self.backend.release(lease)
            replacement = self.backend.reserve(lease.key, 30)
            assert replacement is not None
            assert self.backend.commit(replacement, replacement_payload, remaining_seconds)
            if self._commit_outcome == 'fail':
                raise BackendUnavailable('shared commit failed')
            committed = False
        if self._on_commit is not None:
            callback = self._on_commit
            self._on_commit = None
            callback()
        return committed

    def lookup(self, key: str) -> CacheHit | None:
        self.lookup_count += 1
        if self.on_lookup is not None:
            callback = self.on_lookup
            self.on_lookup = None
            callback()
        return self.backend.lookup(key)

    def resolve(self, tags: tuple[DependencyTag, ...]) -> GenerationVector:
        self.resolved.append(tags)
        return self.backend.resolve(tags)


class FencedBackend(ObservedBackend):
    __slots__ = ('_failures', 'seen')

    def __init__(self, backend: CacheBackend, failures: int) -> None:
        super().__init__(backend)
        self._failures = failures
        self.seen: list[tuple[GenerationChange, ...]] = []

    def change(self, changes: tuple[GenerationChange, ...]) -> None:
        self.seen.append(changes)
        if self._failures:
            self._failures -= 1
            raise BackendUnavailable('fence failed')
        self.backend.change(changes)


class RecordingMetrics:
    __slots__ = ('increments', 'observations')

    def __init__(self) -> None:
        self.increments: list[tuple[str, dict[str, str]]] = []
        self.observations: list[tuple[str, float, dict[str, str]]] = []

    def increment(self, name: str, labels: Mapping[str, str]) -> None:
        self.increments.append((name, dict(labels)))

    def observe(self, name: str, value: float, labels: Mapping[str, str]) -> None:
        self.observations.append((name, value, dict(labels)))


class ResolveFailureBackend(DelegatingBackend):
    __slots__ = ('_failures',)

    def __init__(self, backend: CacheBackend, failures: int) -> None:
        super().__init__(backend)
        self._failures = failures

    def resolve(self, tags: tuple[DependencyTag, ...]) -> GenerationVector:
        if self._failures:
            self._failures -= 1
            raise BackendUnavailable('selected resolve failure')
        return super().resolve(tags)


class BlockingBackend(FencedBackend):
    __slots__ = ('_block_operation', 'entered', 'release_block')

    def __init__(self, backend: CacheBackend, failures: int = 0) -> None:
        super().__init__(backend, failures)
        self._block_operation: Literal['lookup', 'resolve'] | None = None
        self.entered = Event()
        self.release_block = Event()

    def _block(self, operation: Literal['lookup', 'resolve']) -> None:
        if self._block_operation != operation:
            return
        self._block_operation = None
        self.entered.set()
        assert self.release_block.wait(timeout=2)

    def arm(self, operation: Literal['lookup', 'resolve']) -> None:
        self._block_operation = operation
        self.entered.clear()
        self.release_block.clear()

    def lookup(self, key: str) -> CacheHit | None:
        hit = super().lookup(key)
        self._block('lookup')
        return hit

    def resolve(self, tags: tuple[DependencyTag, ...]) -> GenerationVector:
        vector = super().resolve(tags)
        self._block('resolve')
        return vector


class ThrowingMetrics:
    def increment(self, name: str, labels: Mapping[str, str]) -> None:
        _ = name, labels
        raise RuntimeError('metrics unavailable')

    def observe(self, name: str, value: float, labels: Mapping[str, str]) -> None:
        _ = name, value, labels
        raise RuntimeError('metrics unavailable')


def asset_call(entity_type: str = 'Asset') -> ExactCall:
    return ExactCall('find', (('entity_type', entity_type), ('filters', ())))


def asset_plan(classification: str = 'event_aware', *, broadening: bool = False) -> DependencyPlan:
    tags = (
        DependencyTag.site('production'),
        DependencyTag.authorization('production'),
        DependencyTag.type_membership('production', 'Asset'),
    )
    if classification == 'bypass':
        tags = ()
    return DependencyPlan(
        classification, tags, classification, ('field_to_type',) if broadening else ()
    )


def store_payload(
    backend: CacheBackend,
    call: ExactCall,
    plan: DependencyPlan,
    payload: bytes,
    *,
    age: float = 30,
) -> str:
    base_key = KeyBuilder.base_key(
        'production',
        Identity('script', 'studio', 'render'),
        CLIENT_OPTIONS,
        call,
    )
    vector = backend.resolve(plan.tags)
    key = KeyBuilder.qualified_key(base_key, vector, 'result')
    lease = backend.reserve(key, 30)
    assert lease is not None
    assert backend.commit(lease, payload, age)
    return key


async def turn() -> None:
    ready = asyncio.Event()
    asyncio.get_running_loop().call_soon(ready.set)
    await ready.wait()


def valkey_passwords(settings: CacheSettings) -> dict[str, str]:
    return {
        settings.l2.valkey.metadata.password_env: 'metadata',
        settings.l2.valkey.result.password_env: 'result',
    }


def valkey_test_settings(
    *, limits: LimitSettings, mode: _ValkeyMode = 'standalone'
) -> CacheSettings:
    base = CacheSettings(limits=limits)
    return replace(
        base,
        l2=L2Settings(
            enabled=True,
            valkey=replace(
                base.l2.valkey,
                metadata=replace(base.l2.valkey.metadata, mode=mode, tls=False),
                result=replace(base.l2.valkey.result, mode=mode, tls=False),
            ),
        ),
    )
