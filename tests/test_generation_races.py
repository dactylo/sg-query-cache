from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from threading import Event
from typing import Literal, cast

import pytest

from sg_query_cache._backend import CacheBackend, StaleIncarnation
from sg_query_cache._cache import QueryCache
from sg_query_cache._codec import encode_value
from sg_query_cache._config import CacheSettings, L1Settings, L2Settings, LimitSettings
from sg_query_cache._keys import KeyBuilder
from sg_query_cache._local import LocalBackend
from sg_query_cache._model import (
    CacheHit,
    DependencyPlan,
    DependencyTag,
    FillLease,
    GenerationChange,
    GenerationVector,
    Identity,
)
from sg_query_cache._valkey_control import ValkeyControl
from sg_query_cache._valkey_scripts import (
    CHANGE_SCRIPT_SHA,
    COMMIT_SCRIPT_SHA,
    LOOKUP_SCRIPT_SHA,
    RESERVE_SCRIPT_SHA,
    RESOLVE_SCRIPT_SHA,
)
from sg_query_cache.valkey import ValkeyBackend, ValkeyMetadataStore
from tests._fakes import (
    CLIENT_OPTIONS,
    DelegatingBackend,
    FakeClock,
    RecordingMetrics,
    asset_call,
    asset_plan,
)
from tests.valkey._control_helpers import AtomicControlClient

_SITE = 'production'


@dataclass(frozen=True, slots=True)
class _Decision:
    remaining_seconds: float | None


class _IncarnationClient(AtomicControlClient):
    def __init__(self) -> None:
        super().__init__()
        self._leases: dict[str, str] = {}
        self.operation_entered = Event()
        self.operation_release = Event()
        self._pause_sha: str | None = None
        self._results: dict[str, bytes] = {}

    def _gate(self, sha: str) -> None:
        if sha != self._pause_sha:
            return
        self.operation_entered.set()
        assert self.operation_release.wait(timeout=2)
        self._pause_sha = None

    def _metadata_reply(
        self,
        sha: str,
        numkeys: int,
        keys_and_args: tuple[object, ...],
    ) -> object:
        record = cast('bytes', keys_and_args[numkeys])
        if record != self.control:
            return [b'stale']
        if sha == CHANGE_SCRIPT_SHA:
            return [b'ok', numkeys - 2, 0]
        assert sha == RESOLVE_SCRIPT_SHA
        return [b'ok', *(1 for _index in range(numkeys - 2))]

    def _result_reply(
        self,
        sha: str,
        numkeys: int,
        keys_and_args: tuple[object, ...],
    ) -> object:
        if sha == RESERVE_SCRIPT_SHA:
            lease_key, token, _ttl = cast('tuple[str, str, int]', keys_and_args)
            assert numkeys == 1
            if lease_key in self._leases:
                return [b'busy']
            self._leases[lease_key] = token
            return [b'reserved']
        if sha == COMMIT_SCRIPT_SHA:
            lease_key, result_key, token, payload, _ttl = cast(
                'tuple[str, str, str, bytes, int]', keys_and_args
            )
            assert numkeys == 2
            if self._leases.get(lease_key) != token:
                return [b'rejected']
            self._leases.pop(lease_key)
            self._results[result_key] = payload
            return [b'committed']
        if sha == LOOKUP_SCRIPT_SHA:
            result_key = cast('str', keys_and_args[0])
            assert numkeys == 1
            payload = self._results.get(result_key)
            return [b'miss'] if payload is None else [b'hit', payload, 30_000]
        raise AssertionError(sha)

    def arm(self, sha: str) -> None:
        self.operation_entered.clear()
        self.operation_release.clear()
        self._pause_sha = sha

    def evalsha(self, sha: str, numkeys: int, *keys_and_args: object) -> object:
        self._gate(sha)
        if sha in {COMMIT_SCRIPT_SHA, LOOKUP_SCRIPT_SHA, RESERVE_SCRIPT_SHA}:
            return self._result_reply(sha, numkeys, keys_and_args)
        if sha in {CHANGE_SCRIPT_SHA, RESOLVE_SCRIPT_SHA}:
            return self._metadata_reply(sha, numkeys, keys_and_args)
        return super().evalsha(sha, numkeys, *keys_and_args)


class _ObservedBackend(DelegatingBackend):
    def __init__(
        self,
        backend: CacheBackend,
        *,
        on_lookup: Callable[[], None] | None = None,
        on_reserve: Callable[[], None] | None = None,
        after_change: Callable[[], None] | None = None,
    ) -> None:
        super().__init__(backend)
        self._on_lookup = on_lookup
        self._on_reserve = on_reserve
        self._after_change = after_change

    def change(self, changes: tuple[GenerationChange, ...]) -> None:
        self.backend.change(changes)
        if self._after_change is not None:
            self._after_change()

    def lookup(self, key: str) -> CacheHit | None:
        if self._on_lookup is not None:
            self._on_lookup()
        return self.backend.lookup(key)

    def reserve(self, key: str, lease_seconds: float) -> FillLease | None:
        if self._on_reserve is not None:
            self._on_reserve()
        return self.backend.reserve(key, lease_seconds)


def _cache(
    backend: CacheBackend,
    settings: CacheSettings | None = None,
    *,
    shared: bool = False,
) -> QueryCache:
    selected = settings or CacheSettings()
    return QueryCache(
        selected,
        Identity('script', 'studio', 'render'),
        key_builder=KeyBuilder(),
        l1=None if shared else backend,
        l2=backend if shared else None,
    )


def _read(
    cache: QueryCache,
    fetch: Callable[[], object],
    decision: Callable[[object], _Decision] | None = None,
    plan: DependencyPlan | None = None,
) -> object:
    return cache.read(
        asset_call(),
        plan or asset_plan(),
        fetch,
        CLIENT_OPTIONS,
        _population_decision=decision,
    )


def _scoped_plan() -> DependencyPlan:
    return DependencyPlan(
        'event_aware',
        (DependencyTag.entity(_SITE, 'Asset', 7),),
        'event_aware',
    )


@pytest.mark.parametrize('event_phase', ['before_fetch', 'after_fetch'])
def test_event_during_fill_never_publishes_old_value_under_new_vector(
    event_phase: Literal['before_fetch', 'after_fetch'],
) -> None:
    clock = FakeClock()
    backend = LocalBackend(CacheSettings(), clock, RecordingMetrics())
    change = (GenerationChange(DependencyTag.type_membership(_SITE, 'Asset')),)
    observed = _ObservedBackend(
        backend,
        on_reserve=(lambda: backend.change(change)) if event_phase == 'before_fetch' else None,
    )
    cache = _cache(observed)
    calls = 0

    def fetch() -> object:
        nonlocal calls
        calls += 1
        return [{'version': calls}]

    def after_fetch(_value: object) -> _Decision:
        backend.change(change)
        return _Decision(30)

    decision = after_fetch if event_phase == 'after_fetch' else None
    assert _read(cache, fetch, decision) == [{'version': 1}]
    assert _read(cache, fetch) == [{'version': 2}]


def test_expired_first_lease_cannot_overwrite_second_holder() -> None:
    clock = FakeClock()
    backend = LocalBackend(CacheSettings(), clock, RecordingMetrics())
    first = backend.reserve('qualified-key', 1)
    assert first is not None
    allow_late_commit = Event()
    late_commit_entered = Event()

    def late_commit() -> bool:
        late_commit_entered.set()
        assert allow_late_commit.wait(timeout=2)
        return backend.commit(first, encode_value({'version': 1}), 30)

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(late_commit)
        assert late_commit_entered.wait(timeout=2)
        clock.advance(2)
        second = backend.reserve('qualified-key', 1)
        assert second is not None
        assert backend.commit(second, encode_value({'version': 2}), 30)
        allow_late_commit.set()
        assert not future.result(timeout=2)
    hit = backend.lookup('qualified-key')
    assert hit is not None
    assert hit.payload == encode_value({'version': 2})


@pytest.mark.parametrize(
    ('operation', 'script_sha'),
    [('change', CHANGE_SCRIPT_SHA), ('resolve', RESOLVE_SCRIPT_SHA)],
)
def test_incarnation_rotation_rejects_inflight_stale_metadata_operation(
    operation: str,
    script_sha: str,
) -> None:
    client = _IncarnationClient()
    settings = CacheSettings(
        l1=L1Settings(enabled=False),
        l2=L2Settings(enabled=True, acknowledge_mutually_trusted_clients=True),
    )
    store = ValkeyMetadataStore(client, settings, RecordingMetrics())
    tag = DependencyTag.type_membership(_SITE, 'Asset')
    old_vector = store.resolve((tag,))
    active_control = ValkeyControl(
        client,
        key_prefix='sgqc',
        validation_seconds=30,
        jitter_factory=lambda seconds: seconds,
        nonce_factory=lambda: 'f' * 32,
    )
    old_incarnation = active_control.validate()
    assert old_incarnation.nonce == old_vector.incarnation
    client.arm(script_sha)

    def stale_operation() -> object:
        if operation == 'change':
            return store.change((GenerationChange(tag),))
        return store.resolve((tag,))

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(stale_operation)
        assert client.operation_entered.wait(timeout=2)
        new_incarnation = active_control.rotate(old_incarnation)
        client.operation_release.set()
        with pytest.raises(StaleIncarnation):
            future.result(timeout=2)

    assert new_incarnation != old_incarnation


@pytest.mark.parametrize(
    ('operation', 'script_sha'),
    [('commit', COMMIT_SCRIPT_SHA), ('reserve', RESERVE_SCRIPT_SHA)],
)
def test_incarnation_rotation_isolates_inflight_stale_result_operation(
    operation: str,
    script_sha: str,
) -> None:
    client = _IncarnationClient()
    control = ValkeyControl(
        client,
        key_prefix='sgqc',
        validation_seconds=30,
        jitter_factory=lambda seconds: seconds,
        nonce_factory=iter(('1' * 32, '2' * 32)).__next__,
    )
    old_incarnation = control.validate()
    tag = DependencyTag.type_membership(_SITE, 'Asset')
    old = GenerationVector(old_incarnation.nonce, ((tag, 1),))
    base = KeyBuilder.base_key(
        _SITE,
        Identity('script', 'studio', 'render'),
        CLIENT_OPTIONS,
        asset_call(),
    )
    old_key = KeyBuilder.qualified_key(base, old, 'result')
    settings = CacheSettings(
        l1=L1Settings(enabled=False),
        l2=L2Settings(enabled=True, acknowledge_mutually_trusted_clients=True),
    )
    metadata = ValkeyMetadataStore(client, settings, RecordingMetrics())
    backend = ValkeyBackend(client, metadata, settings, RecordingMetrics())
    lease = backend.reserve(old_key, 30) if operation == 'commit' else None
    assert operation != 'commit' or lease is not None
    client.arm(script_sha)

    def stale_operation() -> FillLease | bool | None:
        if operation == 'commit':
            return backend.commit(cast('FillLease', lease), encode_value({'version': 1}), 30)
        return backend.reserve(old_key, 30)

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(stale_operation)
        assert client.operation_entered.wait(timeout=2)
        new_incarnation = control.rotate(old_incarnation)
        new = GenerationVector(new_incarnation.nonce, ((tag, 1),))
        new_key = KeyBuilder.qualified_key(base, new, 'result')
        client.operation_release.set()
        result = future.result(timeout=2)

    assert result is not None
    assert old_key != new_key
    assert backend.lookup(new_key) is None
    if operation == 'commit':
        assert result is True
        assert backend.lookup(old_key) is not None


def test_multi_tag_change_exposes_deterministic_cross_layer_intermediate_vector() -> None:
    settings = CacheSettings(
        l2=L2Settings(enabled=True, acknowledge_mutually_trusted_clients=True),
    )
    l1 = LocalBackend(settings, FakeClock(), RecordingMetrics())
    l2 = LocalBackend(settings, FakeClock(), RecordingMetrics())
    tags = (
        DependencyTag.type_membership(_SITE, 'Asset'),
        DependencyTag.type_value(_SITE, 'Asset'),
        DependencyTag.type_field(_SITE, 'Asset', 'code'),
    )
    before_l1 = l1.resolve(tags)
    before_l2 = l2.resolve(tags)
    l2_changed = Event()
    allow_l1 = Event()

    def pause_after_l2() -> None:
        l2_changed.set()
        assert allow_l1.wait(timeout=2)

    cache = QueryCache(
        settings,
        Identity('script', 'studio', 'render'),
        key_builder=KeyBuilder(),
        l1=l1,
        l2=_ObservedBackend(l2, after_change=pause_after_l2),
    )
    changes = tuple(GenerationChange(tag) for tag in tags)

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(cache.invalidate, changes)
        assert l2_changed.wait(timeout=2)
        intermediate_l1 = l1.resolve(tags)
        intermediate_l2 = l2.resolve(tags)
        allow_l1.set()
        future.result(timeout=2)
    after_l1 = l1.resolve(tags)
    after_l2 = l2.resolve(tags)

    assert intermediate_l1.entries == before_l1.entries
    assert intermediate_l2.entries == after_l2.entries
    assert before_l1.entries != after_l1.entries
    assert before_l2.entries != after_l2.entries


def test_pre_and_post_write_invalidations_strand_intervening_fill() -> None:
    backend = LocalBackend(CacheSettings(), FakeClock(), RecordingMetrics())
    cache = _cache(backend)
    change = (GenerationChange(DependencyTag.type_membership(_SITE, 'Asset')),)
    fetch_entered = Event()
    release_fetch = Event()
    cache.invalidate(change)

    def fetch() -> list[dict[str, int]]:
        fetch_entered.set()
        assert release_fetch.wait(timeout=2)
        return [{'version': 1}]

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_read, cache, fetch, lambda _value: _Decision(30))
        assert fetch_entered.wait(timeout=2)
        cache.invalidate(change)
        release_fetch.set()
        assert future.result(timeout=2) == [{'version': 1}]
    cache.invalidate(change)
    assert _read(cache, lambda: [{'version': 2}]) == [{'version': 2}]
    assert _read(cache, lambda: [{'version': 3}]) == [{'version': 2}]


def test_scoped_expiry_during_fill_strands_old_vector() -> None:
    settings = CacheSettings(
        limits=LimitSettings(scoped_generation_retention_seconds=1),
    )
    clock = FakeClock()
    backend = LocalBackend(settings, clock, RecordingMetrics())
    cache = _cache(backend, settings)
    calls = 0

    def fetch() -> object:
        nonlocal calls
        calls += 1
        return [{'version': calls}]

    assert _read(
        cache,
        fetch,
        lambda _value: (clock.advance(2), _Decision(30))[1],
        _scoped_plan(),
    ) == [{'version': 1}]
    assert _read(cache, fetch, plan=_scoped_plan()) == [{'version': 2}]
    assert _read(cache, fetch, plan=_scoped_plan()) == [{'version': 2}]


def test_scoped_expiry_during_lookup_makes_old_vector_unreachable() -> None:
    settings = CacheSettings(
        limits=LimitSettings(scoped_generation_retention_seconds=1),
    )
    clock = FakeClock()
    backend = LocalBackend(settings, clock, RecordingMetrics())
    cache = _cache(backend, settings)
    assert _read(cache, lambda: [{'version': 1}], plan=_scoped_plan()) == [{'version': 1}]
    clock.advance(2)

    assert _read(cache, lambda: [{'version': 2}], plan=_scoped_plan()) == [{'version': 2}]
    assert _read(cache, lambda: [{'version': 3}], plan=_scoped_plan()) == [{'version': 2}]


def test_sg_write_before_event_cannot_make_inflight_old_result_reachable() -> None:
    backend = LocalBackend(CacheSettings(), FakeClock(), RecordingMetrics())
    cache = _cache(backend)
    fetch_started = Event()
    event_done = Event()
    source = {'version': 1}

    def fetch() -> object:
        observed = source['version']
        fetch_started.set()
        assert event_done.wait(timeout=2)
        return [{'version': observed}]

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_read, cache, fetch)
        assert fetch_started.wait(timeout=2)
        source['version'] = 2
        cache.invalidate((GenerationChange(DependencyTag.type_membership(_SITE, 'Asset')),))
        event_done.set()
        assert future.result(timeout=2) == [{'version': 1}]

    assert _read(cache, lambda: [{'version': source['version']}]) == [{'version': 2}]
