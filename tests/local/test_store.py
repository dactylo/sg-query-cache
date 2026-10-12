from collections import OrderedDict
from math import ceil
from typing import Never, TypeVar

import pytest

import sg_query_cache._local_store as local_store_module
from sg_query_cache._backend import BackendCapacity, BackendProtocolError
from sg_query_cache._codec import encode_value
from sg_query_cache._config import CacheSettings, L1Settings
from sg_query_cache._keys import NormalizationError
from sg_query_cache._local_store import LocalStore, StoreOperation
from sg_query_cache._model import DependencyTag, FillLease, GenerationChange
from tests.local._store_helpers import CountedQueue

_K = TypeVar('_K')
_V = TypeVar('_V')


class _NoScanMixin:
    def __iter__(self) -> Never:
        raise AssertionError('whole-table iteration')

    def copy(self) -> Never:
        raise AssertionError('whole-table copy')

    def items(self) -> Never:
        raise AssertionError('whole-table items scan')

    def keys(self) -> Never:
        raise AssertionError('whole-table copy')

    def values(self) -> Never:
        raise AssertionError('whole-table accounting scan')


class _NoScanDict(_NoScanMixin, dict[_K, _V]):
    pass


class _NoScanResults(_NoScanMixin, OrderedDict[_K, _V]):
    pass


def _assert_indexes(store: LocalStore) -> None:
    assert len(store._result_queue) == len(store._results)
    assert len(store._lease_queue) == len(store._leases)
    assert len(store._generation_queue) == len(store._generations) - 4


def _fill(store: LocalStore, key: str, now: float = 0.0, duration: float = 60.0) -> None:
    lease = store.reserve(key, 10.0, 'b' * 32, StoreOperation(now))
    assert lease is not None
    assert store.commit(lease, encode_value({'id': 7}), duration, StoreOperation(now))


def _invoke(store: LocalStore, operation: str, now: float) -> None:
    op = StoreOperation(now)
    tag = DependencyTag.site('production')
    match operation:
        case 'lookup':
            store.lookup('result-193', op)
        case 'resolve':
            store.resolve((tag, DependencyTag.entity('production', 'Asset', 193)), op)
        case 'reserve':
            store.reserve('new', 10.0, 'b' * 32, op)
        case 'commit':
            store.commit(
                FillLease('lease-193', store._leases['lease-193']._token), b'[]', 10.0, op
            )
        case 'release':
            store.release(FillLease('lease-193', store._leases['lease-193']._token), op)
        case 'change':
            store.change((GenerationChange(tag),), op)
        case 'flush':
            store.flush_site(tag, 'manual', op)
        case _:
            raise AssertionError(operation)


def _store() -> LocalStore:
    return LocalStore(CacheSettings(), 'a' * 32, 0.0)


def _populated_store(count: int = 193) -> LocalStore:
    store = _store()
    for index in range(1, count + 1):
        _fill(store, f'result-{index}')
        assert store.reserve(f'lease-{index}', 60.0, 'b' * 32, StoreOperation(0.0)) is not None
        store.resolve((DependencyTag.entity('production', 'Asset', index),), StoreOperation(0.0))
    return store


def test_capacity_rejection_preserves_addressed_vector_and_token_counter() -> None:
    store = _store()
    existing = DependencyTag.entity('production', 'Asset', 1)
    missing = DependencyTag.entity('production', 'Asset', 2)
    store.resolve((existing,), StoreOperation(0.0))
    before = store._generations[existing]
    counter = store._generation_counter
    store._metadata_budget_bytes = store.generation_bytes

    with pytest.raises(BackendCapacity):
        store.resolve((existing, missing), StoreOperation(1.0))

    assert store._generations[existing] is before
    assert missing not in store._generations
    assert store._generation_counter == counter
    _assert_indexes(store)


def test_change_does_not_materialize_expired_or_absent_scoped_tags() -> None:
    store = _populated_store()
    expired = next(reversed(store._generation_queue))
    absent = DependencyTag.entity('production', 'Asset', 10_000)

    store.change((GenerationChange(expired), GenerationChange(absent)), StoreOperation(7_200.0))

    assert expired not in store._generations
    assert absent not in store._generations
    _assert_indexes(store)


@pytest.mark.parametrize(
    'operation', ['lookup', 'resolve', 'reserve', 'commit', 'release', 'change', 'flush']
)
@pytest.mark.parametrize('now', [0.0, 8_000.0])
def test_every_operation_bounds_all_queue_visits_without_table_scans(
    operation: str, now: float
) -> None:
    store = _populated_store()
    results = CountedQueue(store._result_queue)
    leases = CountedQueue(store._lease_queue)
    generations = CountedQueue(store._generation_queue)
    store._result_queue, store._lease_queue, store._generation_queue = results, leases, generations
    store._generations = _NoScanDict(store._generations)
    store._leases = _NoScanDict(store._leases)
    store._results = _NoScanResults(store._results)

    _invoke(store, operation, now)

    assert results._visits == leases._visits == generations._visits == 64
    _assert_indexes(store)


def test_expired_fixed_cohorts_drain_within_population_traversal_bound() -> None:
    store = _populated_store()
    for _ in range(ceil(193 / 64)):
        store.lookup('absent', StoreOperation(7_200.0))

    assert not store._results
    assert not store._leases
    assert len(store._generations) == 4
    assert store.result_bytes == store.lease_bytes == 0
    _assert_indexes(store)


def test_expired_scoped_tail_resolves_fresh_at_equal_deadlines() -> None:
    store = _populated_store()
    tag = next(reversed(store._generation_queue))
    old = store._generations[tag]

    vector = store.resolve((tag,), StoreOperation(7_200.0))

    assert vector.entries[0][1] > old._token
    assert store._generations[tag]._retirement_deadline == 14_400.0
    _assert_indexes(store)


def test_expired_tail_lease_rejects_commit_without_replacement_owner() -> None:
    store = _populated_store()
    key = next(reversed(store._lease_queue))
    old = store._leases[key]

    assert not store.commit(FillLease(key, old._token), b'[]', 60.0, StoreOperation(61.0))
    assert key not in store._leases
    assert store.lookup(key, StoreOperation(61.0)) is None


def test_fixed_clock_lease_churn_and_result_recreation_keep_indexes_bounded() -> None:
    store = LocalStore(CacheSettings(l1=L1Settings(result_budget_bytes=300)), 'a' * 32, 0.0)
    generation_bytes = store.generation_bytes
    maximum_result_bytes = 0
    for index in range(4):
        lease = store.reserve('reused-lease', 60.0, 'b' * 32, StoreOperation(0.0))
        assert lease is not None
        assert store.release(lease, StoreOperation(0.0))
        _fill(store, f'result-{index % 2}')
        maximum_result_bytes = max(maximum_result_bytes, store.result_bytes)
        _assert_indexes(store)
        assert len(store._result_queue) == 1
        assert store.lease_bytes == 0
    assert store.generation_bytes == generation_bytes
    assert maximum_result_bytes <= 300


def test_isolated_addressed_expiry_is_immediate_for_lookup_and_reserve() -> None:
    store = _populated_store()
    target = next(reversed(store._result_queue))
    assert store.lookup(target, StoreOperation(60.0)) is None
    key = next(reversed(store._lease_queue))
    old = store._leases[key]._token

    replacement = store.reserve(key, 10.0, 'b' * 32, StoreOperation(60.0))

    assert replacement is not None
    assert replacement.token != old


def test_new_generation_key_encoding_failure_stays_a_protocol_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _store()

    def reject_key(_tag: DependencyTag) -> str:
        raise NormalizationError('sensitive key input')

    monkeypatch.setattr(local_store_module.KeyBuilder, 'tag_key', reject_key)
    store.resolve((DependencyTag.site('production'),), StoreOperation(0.0))

    with pytest.raises(BackendProtocolError, match='dependency tag cannot be encoded safely'):
        store.resolve((DependencyTag.entity('production', 'Asset', 7),), StoreOperation(0.0))


def test_old_cohort_progress_survives_new_tail_entries_and_lru_hits() -> None:
    store = _populated_store()
    cohort = set(store._results)
    _fill(store, 'hot', duration=10_000.0)
    for index in range(ceil(193 / 64)):
        store.lookup('hot', StoreOperation(61.0))
        _fill(store, f'new-{index}', now=61.0)
    assert cohort.isdisjoint(store._results)
    assert store.lookup('hot', StoreOperation(61.0)) is not None
    _assert_indexes(store)


def test_refresh_does_not_reorder_or_duplicate_maintenance_entries() -> None:
    store = _populated_store(193)
    target = next(reversed(store._result_queue))
    order = list(store._result_queue)
    for _ in range(20):
        _fill(store, target)
        assert len(store._result_queue) == 193
    # Every operation rotates 64 entries, while refresh preserves the ring.
    current = list(store._result_queue)
    offset = order.index(current[0])
    assert current == order[offset:] + order[:offset]


def test_resolve_encodes_each_generation_key_only_when_record_is_created(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _store()
    tag = DependencyTag.entity('production', 'Asset', 7)
    calls: list[DependencyTag] = []
    original = local_store_module.KeyBuilder.tag_key

    def counted_tag_key(value: DependencyTag) -> str:
        calls.append(value)
        return original(value)

    monkeypatch.setattr(local_store_module.KeyBuilder, 'tag_key', counted_tag_key)
    store.resolve((tag,), StoreOperation(0.0))
    store.resolve((tag,), StoreOperation(1.0))
    store.change((GenerationChange(tag),), StoreOperation(2.0))

    assert calls == [tag]


def test_shortened_expiry_is_reclaimed_within_population_traversal_bound() -> None:
    store = _populated_store()
    target = next(reversed(store._result_queue))
    _fill(store, target, duration=5.0)
    for _ in range(ceil(len(store._result_queue) / 64)):
        store.lookup('absent', StoreOperation(5.0))

    assert target not in store._results
    assert len(store._results) == 192
    _assert_indexes(store)


def test_small_live_queues_visit_each_record_only_once() -> None:
    store = _populated_store(3)
    results = CountedQueue(store._result_queue)
    leases = CountedQueue(store._lease_queue)
    generations = CountedQueue(store._generation_queue)
    store._result_queue, store._lease_queue, store._generation_queue = results, leases, generations

    store.lookup('absent', StoreOperation(0.0))

    assert results._visits == leases._visits == generations._visits == 3
