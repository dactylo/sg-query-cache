from collections.abc import Callable, Mapping
from threading import Event, Lock
from typing import Literal

from sg_query_cache._backend import BackendUnavailable, CacheBackend
from sg_query_cache._model import (
    CacheHit,
    DependencyTag,
    FillLease,
    GenerationChange,
    GenerationVector,
)


class _DelegatingBackend:
    __slots__ = ('_backend',)

    def __init__(self, backend: CacheBackend) -> None:
        self._backend = backend

    def change(self, changes: tuple[GenerationChange, ...]) -> None:
        self._backend.change(changes)

    def close(self) -> None:
        self._backend.close()

    def commit(self, lease: FillLease, payload: bytes, remaining_seconds: float) -> bool:
        return self._backend.commit(lease, payload, remaining_seconds)

    def flush_site(self, site_namespace: str, reason: str) -> None:
        self._backend.flush_site(site_namespace, reason)

    def lookup(self, key: str) -> CacheHit | None:
        return self._backend.lookup(key)

    def release(self, lease: FillLease) -> bool:
        return self._backend.release(lease)

    def reserve(self, key: str, lease_seconds: float) -> FillLease | None:
        return self._backend.reserve(key, lease_seconds)

    def resolve(self, tags: tuple[DependencyTag, ...]) -> GenerationVector:
        return self._backend.resolve(tags)


class _FakeClock:  # pyright: ignore[reportUnusedClass]
    __slots__ = ('_lock', '_value')

    def __init__(self, start: float = 1_000.0) -> None:
        self._lock = Lock()
        self._value = start

    def __call__(self) -> float:
        with self._lock:
            return self._value

    def _advance(self, seconds: float) -> None:
        with self._lock:
            self._value += seconds


class _ObservedBackend(_DelegatingBackend):
    __slots__ = (
        '_commit_outcome',
        '_lookup_count',
        '_on_commit',
        '_on_lookup',
        '_replacement_payload',
        '_resolved',
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
        self._lookup_count = 0
        self._on_commit = on_commit
        self._on_lookup = on_lookup
        self._replacement_payload = replacement_payload
        self._resolved: list[tuple[DependencyTag, ...]] = []

    def commit(self, lease: FillLease, payload: bytes, remaining_seconds: float) -> bool:
        if self._commit_outcome == 'delegate':
            committed = self._backend.commit(lease, payload, remaining_seconds)
        else:
            replacement_payload = self._replacement_payload
            assert replacement_payload is not None
            assert self._backend.release(lease)
            replacement = self._backend.reserve(lease.key, 30)
            assert replacement is not None
            assert self._backend.commit(replacement, replacement_payload, remaining_seconds)
            if self._commit_outcome == 'fail':
                raise BackendUnavailable('shared commit failed')
            committed = False
        if self._on_commit is not None:
            callback = self._on_commit
            self._on_commit = None
            callback()
        return committed

    def lookup(self, key: str) -> CacheHit | None:
        self._lookup_count += 1
        if self._on_lookup is not None:
            callback = self._on_lookup
            self._on_lookup = None
            callback()
        return self._backend.lookup(key)

    def resolve(self, tags: tuple[DependencyTag, ...]) -> GenerationVector:
        self._resolved.append(tags)
        return self._backend.resolve(tags)


class _FencedBackend(_ObservedBackend):
    __slots__ = ('_failures', '_seen')

    def __init__(self, backend: CacheBackend, failures: int) -> None:
        super().__init__(backend)
        self._failures = failures
        self._seen: list[tuple[GenerationChange, ...]] = []

    def change(self, changes: tuple[GenerationChange, ...]) -> None:
        self._seen.append(changes)
        if self._failures:
            self._failures -= 1
            raise BackendUnavailable('fence failed')
        self._backend.change(changes)


class ResolveFailureBackend(_DelegatingBackend):
    __slots__ = ('_failures',)

    def __init__(self, backend: CacheBackend, failures: int) -> None:
        super().__init__(backend)
        self._failures = failures

    def resolve(self, tags: tuple[DependencyTag, ...]) -> GenerationVector:
        if self._failures:
            self._failures -= 1
            raise BackendUnavailable('selected resolve failure')
        return super().resolve(tags)


class _BlockingBackend(_FencedBackend):  # pyright: ignore[reportUnusedClass]
    __slots__ = ('_block_operation', '_entered', '_release_block')

    def __init__(self, backend: CacheBackend, failures: int = 0) -> None:
        super().__init__(backend, failures)
        self._block_operation: Literal['lookup', 'resolve'] | None = None
        self._entered = Event()
        self._release_block = Event()

    def _arm(self, operation: Literal['lookup', 'resolve']) -> None:
        self._block_operation = operation
        self._entered.clear()
        self._release_block.clear()

    def _block(self, operation: Literal['lookup', 'resolve']) -> None:
        if self._block_operation != operation:
            return
        self._block_operation = None
        self._entered.set()
        assert self._release_block.wait(timeout=2)

    def lookup(self, key: str) -> CacheHit | None:
        hit = super().lookup(key)
        self._block('lookup')
        return hit

    def resolve(self, tags: tuple[DependencyTag, ...]) -> GenerationVector:
        vector = super().resolve(tags)
        self._block('resolve')
        return vector


class _RecordingMetrics:  # pyright: ignore[reportUnusedClass]
    __slots__ = ('_increments', '_observations')

    def __init__(self) -> None:
        self._increments: list[tuple[str, dict[str, str]]] = []
        self._observations: list[tuple[str, float, dict[str, str]]] = []

    def increment(self, name: str, labels: Mapping[str, str]) -> None:
        self._increments.append((name, dict(labels)))

    def observe(self, name: str, value: float, labels: Mapping[str, str]) -> None:
        self._observations.append((name, value, dict(labels)))
