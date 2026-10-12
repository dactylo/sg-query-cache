from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Event
from typing import cast

import pytest

import sg_query_cache._local as local_module
from sg_query_cache._backend import (
    BackendCapacity,
    BackendProtocolError,
    BackendUnavailable,
    CacheBackend,
)
from sg_query_cache._cache_common import TimedResult, decode_hit, encode_result
from sg_query_cache._codec import encode_value
from sg_query_cache._config import CacheSettings, L1Settings, LimitSettings
from sg_query_cache._keys import KeyBuilder
from sg_query_cache._local import LocalBackend
from sg_query_cache._local_store import LocalStore, StoreOperation
from sg_query_cache._model import DependencyTag, FillLease, GenerationChange
from tests._backend_contract import _BackendContractTests, _BackendFactory, _BackendHarness
from tests._exceptions import exception_text as _exposed_exception_text
from tests._fakes import _FakeClock, _RecordingMetrics


class _ExceptionalFloat(float):
    def __float__(self) -> float:
        raise RuntimeError('credential-bearing-numeric-conversion-error')


def _assert_closed_exception(
    error: BaseException,
    expected_type: type[BaseException],
    sensitive_input: str,
) -> None:
    assert isinstance(error, expected_type)
    assert error.__cause__ is None
    assert error.__context__ is None
    assert sensitive_input not in _exposed_exception_text(error)


def _commit(backend: CacheBackend, key: str, payload: bytes) -> bool:
    lease = backend.reserve(key, 10.0)
    assert lease is not None
    return backend.commit(lease, payload, 30.0)


def _generation_record_size(
    tag: DependencyTag,
    token: int,
    deadline: float | None,
) -> int:
    maintenance = 0 if deadline is None else 64
    return (
        len(KeyBuilder.tag_key(tag).encode('ascii'))
        + len(encode_value((token, deadline)))
        + 96
        + maintenance
    )


@pytest.fixture
def backend_factory() -> _BackendFactory:
    def create(settings: CacheSettings | None = None) -> _BackendHarness:
        clock = _FakeClock()
        metrics = _RecordingMetrics()
        backend: CacheBackend = LocalBackend(settings or CacheSettings(), clock, metrics)
        return _BackendHarness(backend, clock, metrics)

    return create


class TestLocalBackend(_BackendContractTests):
    pass


def _settings(
    *,
    result_budget_bytes: int = 134_217_728,
    metadata_budget_bytes: int = 16_777_216,
    max_serialized_result_bytes: int = 1_048_576,
) -> CacheSettings:
    return CacheSettings(
        l1=L1Settings(
            result_budget_bytes=result_budget_bytes,
            metadata_budget_bytes=metadata_budget_bytes,
        ),
        limits=LimitSettings(max_serialized_result_bytes=max_serialized_result_bytes),
    )


def test_backend_exception_chain_does_not_expose_invalid_opaque_key(
    backend_factory: _BackendFactory,
) -> None:
    sensitive_key = 'credential-bearing-key' + '\ud800'
    backend = backend_factory()._backend

    with pytest.raises(BackendProtocolError) as captured:
        backend.lookup(sensitive_key)

    _assert_closed_exception(captured.value, BackendProtocolError, 'credential-bearing-key')


def test_backend_exceptions_do_not_expose_opaque_input(
    backend_factory: _BackendFactory,
) -> None:
    sensitive_input = 'credential=' + 'query&url=https://private.example/payload'
    backend = backend_factory()._backend
    lease = backend.reserve('qualified-key', 1.0)
    assert lease is not None

    with pytest.raises(BackendProtocolError) as captured:
        backend.commit(lease, cast('bytes', sensitive_input), 1.0)

    _assert_closed_exception(captured.value, BackendProtocolError, sensitive_input)


def test_backward_clock_is_unavailable_instead_of_renewing_age(
    backend_factory: _BackendFactory,
) -> None:
    harness = backend_factory()
    assert _commit(harness._backend, 'qualified-key', encode_value({'id': 7}))
    harness._clock._advance(-1.0)

    with pytest.raises(BackendUnavailable, match='clock moved backward'):
        harness._backend.lookup('qualified-key')


def test_change_capacity_failure_acknowledges_no_partial_update(
    backend_factory: _BackendFactory,
) -> None:
    globals_ = (
        DependencyTag.site('production'),
        DependencyTag.schema('production'),
        DependencyTag.authorization('production'),
        DependencyTag.human_visibility('production'),
    )
    scoped = DependencyTag.entity('production', 'Asset', 7)
    metadata_budget = sum(
        _generation_record_size(tag, token, None) for token, tag in enumerate(globals_, start=1)
    ) + _generation_record_size(scoped, 5, 8_200.0)
    backend = backend_factory(_settings(metadata_budget_bytes=metadata_budget))._backend
    before = backend.resolve((scoped,)).entries[0][1]

    with pytest.raises(BackendCapacity, match='metadata capacity'):
        backend.change(tuple(GenerationChange(scoped) for _index in range(5)))

    assert backend.resolve((scoped,)).entries[0][1] == before


def test_clock_samples_remain_serialized_with_store_transitions() -> None:
    backend = LocalBackend(CacheSettings(), _FakeClock(), _RecordingMetrics())
    sampled = Event()
    resume = Event()

    def paused_clock() -> float:
        sampled.set()
        assert resume.wait(timeout=2.0)
        return 1_000.0

    backend._clock = paused_clock
    with ThreadPoolExecutor(max_workers=1) as executor:
        lookup = executor.submit(backend.lookup, 'absent')
        assert sampled.wait(timeout=2.0)
        acquired = backend._lock.acquire(blocking=False)
        if acquired:
            backend._lock.release()
        resume.set()
        assert lookup.result(timeout=2.0) is None

    assert not acquired


def test_commit_debits_preparation_time_from_ttl_and_source_age(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = _FakeClock()
    backend = LocalBackend(CacheSettings(), clock, _RecordingMetrics())
    lease = backend.reserve('result-key', 10.0)
    assert lease is not None
    entry_captured = Event()
    original_positive_seconds = local_module._positive_seconds

    def signal_after_entry(value: object, owner: str) -> float:
        result = original_positive_seconds(value, owner)
        entry_captured.set()
        return result

    monkeypatch.setattr(local_module, '_positive_seconds', signal_after_entry)
    payload = encode_result(TimedResult(0.0, {'id': 7}), 0.0, 0.1)
    executor = ThreadPoolExecutor(max_workers=1)
    lock_held = True
    backend._lock.acquire()
    try:
        commit = executor.submit(backend.commit, lease, payload, 0.1)
        assert entry_captured.wait(timeout=2.0)
        clock._advance(0.04)
        backend._lock.release()
        lock_held = False
        assert commit.result(timeout=2.0)
    finally:
        if lock_held:
            backend._lock.release()
        executor.shutdown(wait=True)

    hit = backend.lookup('result-key')

    assert hit is not None
    assert hit.remaining_seconds <= 0.06 + 1e-12
    assert decode_hit(hit, 1.0).origin_elapsed == pytest.approx(0.04)


def test_commit_huge_duration_is_a_closed_protocol_failure(
    backend_factory: _BackendFactory,
) -> None:
    backend = backend_factory()._backend
    lease = backend.reserve('qualified-key', 1.0)
    assert lease is not None
    huge_duration = 10**10_000

    with pytest.raises(BackendProtocolError) as captured:
        backend.commit(lease, encode_value({'id': 7}), huge_duration)

    _assert_closed_exception(captured.value, BackendProtocolError, 'credential')


def test_commit_normalizes_exceptional_numeric_subclass(
    backend_factory: _BackendFactory,
) -> None:
    backend = backend_factory()._backend
    lease = backend.reserve('qualified-key', 1.0)
    assert lease is not None

    with pytest.raises(BackendProtocolError) as captured:
        backend.commit(lease, encode_value({'id': 7}), _ExceptionalFloat(1.0))

    _assert_closed_exception(
        captured.value,
        BackendProtocolError,
        'credential-bearing-numeric-conversion-error',
    )


def test_commit_rejects_lifetime_expired_waiting_for_transition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = _FakeClock()
    backend = LocalBackend(CacheSettings(), clock, _RecordingMetrics())
    lease = backend.reserve('result-key', 10.0)
    assert lease is not None
    entry_captured = Event()
    original_positive_seconds = local_module._positive_seconds

    def signal_after_entry(value: object, owner: str) -> float:
        result = original_positive_seconds(value, owner)
        entry_captured.set()
        return result

    monkeypatch.setattr(local_module, '_positive_seconds', signal_after_entry)
    executor = ThreadPoolExecutor(max_workers=1)
    lock_held = True
    backend._lock.acquire()
    try:
        commit = executor.submit(backend.commit, lease, encode_value({'id': 7}), 0.05)
        assert entry_captured.wait(timeout=2.0)
        clock._advance(0.1)
        backend._lock.release()
        lock_held = False
        assert commit.result(timeout=2.0) is False
    finally:
        if lock_held:
            backend._lock.release()
        executor.shutdown(wait=True)

    assert backend.lookup('result-key') is None


def test_constructor_normalizes_exceptional_clock_subclass() -> None:
    def exceptional_clock() -> float:
        return _ExceptionalFloat(1.0)

    with pytest.raises(BackendUnavailable) as captured:
        LocalBackend(CacheSettings(), exceptional_clock, _RecordingMetrics())

    _assert_closed_exception(
        captured.value,
        BackendUnavailable,
        'credential-bearing-numeric-conversion-error',
    )


def test_constructor_normalizes_injected_clock_failure() -> None:
    sensitive_input = 'credential-bearing-clock-error'

    def failing_clock() -> float:
        raise RuntimeError(sensitive_input)

    with pytest.raises(BackendUnavailable) as captured:
        LocalBackend(CacheSettings(), failing_clock, _RecordingMetrics())

    _assert_closed_exception(captured.value, BackendUnavailable, sensitive_input)


def test_constructor_normalizes_injected_entropy_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sensitive_input = 'credential-bearing-constructor-entropy-error'

    def failing_entropy(_byte_count: int) -> str:
        raise OSError(sensitive_input)

    monkeypatch.setattr('sg_query_cache._local.token_hex', failing_entropy)

    with pytest.raises(BackendUnavailable) as captured:
        LocalBackend(CacheSettings(), _FakeClock(), _RecordingMetrics())

    _assert_closed_exception(captured.value, BackendUnavailable, sensitive_input)


def test_corrupt_snapshot_decode_allows_replacement_without_deleting_it(
    backend_factory: _BackendFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = backend_factory()._backend
    assert _commit(backend, 'result-key', b'corrupt')
    decoding = Event()
    replaced = Event()
    original_decode = local_module.decode_value

    def paused_decode(encoded: bytes) -> object:
        decoding.set()
        assert replaced.wait(timeout=2.0)
        return original_decode(encoded)

    monkeypatch.setattr(local_module, 'decode_value', paused_decode)
    payload = encode_value({'id': 9})
    with ThreadPoolExecutor(max_workers=2) as executor:
        lookup = executor.submit(backend.lookup, 'result-key')
        assert decoding.wait(timeout=2.0)
        replacement = executor.submit(_commit, backend, 'result-key', payload)
        try:
            assert replacement.result(timeout=1.0)
        finally:
            replaced.set()
        assert lookup.result(timeout=2.0) is None

    hit = backend.lookup('result-key')
    assert hit is not None
    assert hit.payload == payload


def test_entry_above_one_mib_is_declined_without_evicting_useful_result(
    backend_factory: _BackendFactory,
) -> None:
    backend = backend_factory(_settings(result_budget_bytes=2_097_152))._backend
    useful = encode_value({'id': 7})
    assert _commit(backend, 'useful', useful)

    assert not _commit(backend, 'oversized', b'x' * 1_048_577)

    hit = backend.lookup('useful')
    assert hit is not None
    assert hit.payload == useful


def test_expired_lease_frees_metadata_before_generation_resolution(
    backend_factory: _BackendFactory,
) -> None:
    harness = backend_factory(_settings(metadata_budget_bytes=1_084))
    assert harness._backend.reserve('k', 1.0) is not None
    harness._clock._advance(1.0)

    vector = harness._backend.resolve((DependencyTag.entity('production', 'Asset', 7),))

    assert vector.entries[0][1] > 0


def test_expired_lease_token_is_never_reused(backend_factory: _BackendFactory) -> None:
    harness = backend_factory()
    first = harness._backend.reserve('qualified-key', 1.0)
    assert first is not None
    harness._clock._advance(1.0)

    second = harness._backend.reserve('qualified-key', 1.0)

    assert second is not None
    assert second.token != first.token


def test_fifty_concurrent_reservations_produce_one_current_lease(
    backend_factory: _BackendFactory,
) -> None:
    backend = backend_factory()._backend
    start = Barrier(51)

    def reserve() -> FillLease | None:
        start.wait(timeout=5.0)
        return backend.reserve('hot-qualified-key', 30.0)

    with ThreadPoolExecutor(max_workers=50) as executor:
        futures = [executor.submit(reserve) for _index in range(50)]
        start.wait(timeout=5.0)
        results = [future.result(timeout=5.0) for future in futures]

    assert len(results) == 50
    assert sum(lease is not None for lease in results) == 1


def test_lookup_debits_decode_time_from_remaining_age(
    backend_factory: _BackendFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = backend_factory()
    payload = encode_value({'id': 7})
    assert _commit(harness._backend, 'result-key', payload)
    original_decode = local_module.decode_value

    def delayed_decode(encoded: bytes) -> object:
        value = original_decode(encoded)
        harness._clock._advance(0.25)
        return value

    monkeypatch.setattr(local_module, 'decode_value', delayed_decode)

    hit = harness._backend.lookup('result-key')

    assert hit is not None
    assert hit.remaining_seconds == 29.75


def test_lookup_debits_slow_maintenance_metrics_before_return(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = _FakeClock()
    metrics = _RecordingMetrics()
    backend = LocalBackend(CacheSettings(), clock, metrics)
    lease = backend.reserve('expired', 10.0)
    assert lease is not None
    assert backend.commit(lease, encode_value({'id': 1}), 1.0)
    assert _commit(backend, 'result-key', encode_value({'id': 7}))
    clock._advance(1.0)
    original_observe = _RecordingMetrics.observe

    def slow_observe(
        sink: _RecordingMetrics,
        name: str,
        value: float,
        labels: Mapping[str, str],
    ) -> None:
        original_observe(sink, name, value, labels)
        if name == 'cache_serialized_bytes' and labels.get('outcome') == 'expired':
            clock._advance(5.0)

    monkeypatch.setattr(_RecordingMetrics, 'observe', slow_observe)

    hit = backend.lookup('result-key')

    assert hit is not None
    assert hit.remaining_seconds == 24.0


def test_lookup_expiring_during_decode_is_a_miss(
    backend_factory: _BackendFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = backend_factory()
    payload = encode_value({'id': 7})
    assert _commit(harness._backend, 'result-key', payload)
    harness._clock._advance(29.5)
    original_decode = local_module.decode_value

    def delayed_decode(encoded: bytes) -> object:
        value = original_decode(encoded)
        harness._clock._advance(0.5)
        return value

    monkeypatch.setattr(local_module, 'decode_value', delayed_decode)

    assert harness._backend.lookup('result-key') is None


def test_lookup_returns_an_isolated_bytes_value(backend_factory: _BackendFactory) -> None:
    backend = backend_factory()._backend
    payload = encode_value({'id': 7})
    assert _commit(backend, 'result-key', payload)

    first = backend.lookup('result-key')
    second = backend.lookup('result-key')

    assert first is not None
    assert second is not None
    assert first.payload == second.payload == payload


def test_metadata_pressure_raises_backend_capacity(
    backend_factory: _BackendFactory,
) -> None:
    backend = backend_factory(_settings(metadata_budget_bytes=1_000))._backend

    with pytest.raises(BackendCapacity, match='metadata capacity'):
        backend.reserve('k' * 300, 10.0)


def test_reads_update_recency_before_result_pressure_evicts_lru(
    backend_factory: _BackendFactory,
) -> None:
    payload = encode_value({'id': 7})
    entry_size = len(b'result-a') + len(payload) + 128 + 64
    backend = backend_factory(_settings(result_budget_bytes=entry_size * 2))._backend
    assert _commit(backend, 'result-a', payload)
    assert _commit(backend, 'result-b', payload)
    assert backend.lookup('result-a') is not None

    assert _commit(backend, 'result-c', payload)

    assert backend.lookup('result-a') is not None
    assert backend.lookup('result-b') is None
    assert backend.lookup('result-c') is not None


def test_reserve_huge_duration_is_a_closed_protocol_failure(
    backend_factory: _BackendFactory,
) -> None:
    backend = backend_factory()._backend
    huge_duration = 10**10_000

    with pytest.raises(BackendProtocolError) as captured:
        backend.reserve('qualified-key', huge_duration)

    _assert_closed_exception(captured.value, BackendProtocolError, 'credential')


def test_reserve_normalizes_exceptional_numeric_subclass(
    backend_factory: _BackendFactory,
) -> None:
    backend = backend_factory()._backend

    with pytest.raises(BackendProtocolError) as captured:
        backend.reserve('qualified-key', _ExceptionalFloat(1.0))

    _assert_closed_exception(
        captured.value,
        BackendProtocolError,
        'credential-bearing-numeric-conversion-error',
    )


def test_reserve_normalizes_injected_entropy_failure(
    backend_factory: _BackendFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sensitive_input = 'credential-bearing-reservation-entropy-error'
    backend = backend_factory()._backend

    def failing_entropy(_byte_count: int) -> str:
        raise OSError(sensitive_input)

    monkeypatch.setattr('sg_query_cache._local.token_hex', failing_entropy)

    with pytest.raises(BackendUnavailable) as captured:
        backend.reserve('qualified-key', 1.0)

    _assert_closed_exception(captured.value, BackendUnavailable, sensitive_input)


def test_resolve_suppresses_sensitive_vector_encoding_failure(
    backend_factory: _BackendFactory,
) -> None:
    sensitive_input = 'credential-bearing-tag' + '\ud800'
    tag = DependencyTag.entity('production', sensitive_input, 7)

    with pytest.raises(BackendProtocolError) as captured:
        backend_factory()._backend.resolve((tag,))

    _assert_closed_exception(captured.value, BackendProtocolError, 'credential-bearing-tag')


def test_result_accounting_includes_key_payload_and_fixed_overhead(
    backend_factory: _BackendFactory,
) -> None:
    key = 'result-key'
    payload = encode_value({'id': 7})
    exact_size = len(key.encode('utf-8')) + len(payload) + 128 + 64
    too_small = backend_factory(_settings(result_budget_bytes=exact_size - 1))._backend
    exact = backend_factory(_settings(result_budget_bytes=exact_size))._backend

    assert not _commit(too_small, key, payload)
    assert _commit(exact, key, payload)


def test_result_eviction_never_removes_generations_or_leases(
    backend_factory: _BackendFactory,
) -> None:
    payload = encode_value({'id': 7})
    entry_size = len(b'result-a') + len(payload) + 128 + 64
    harness = backend_factory(_settings(result_budget_bytes=entry_size))
    backend = harness._backend
    tag = DependencyTag.entity('production', 'Asset', 7)
    token = backend.resolve((tag,)).entries[0][1]
    held = backend.reserve('held-lease', 10.0)
    assert held is not None
    assert _commit(backend, 'result-a', payload)

    assert _commit(backend, 'result-b', payload)

    assert backend.lookup('result-a') is None
    assert backend.lookup('result-b') is not None
    assert backend.resolve((tag,)).entries[0][1] == token
    assert backend.reserve('held-lease', 10.0) is None


def test_slow_reentrant_metrics_run_outside_lock_and_preserve_commit_age(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = _FakeClock()
    metrics = _RecordingMetrics()
    backend = LocalBackend(CacheSettings(), clock, metrics)
    observed: list[float] = []
    original_observe = _RecordingMetrics.observe
    with ThreadPoolExecutor(max_workers=1) as executor:

        def reentrant_observe(
            sink: _RecordingMetrics,
            name: str,
            value: float,
            labels: Mapping[str, str],
        ) -> None:
            original_observe(sink, name, value, labels)
            if name == 'cache_serialized_bytes' and labels.get('outcome') == 'admitted':
                clock._advance(5.0)
                hit = executor.submit(backend.lookup, 'result-key').result(timeout=1.0)
                assert hit is not None
                observed.append(hit.remaining_seconds)

        monkeypatch.setattr(_RecordingMetrics, 'observe', reentrant_observe)
        assert _commit(backend, 'result-key', encode_value({'id': 7}))

    assert observed == [25.0]
    hit = backend.lookup('result-key')
    assert hit is not None
    assert hit.remaining_seconds == 25.0


def test_storage_usage_is_fresh_and_does_not_observe_time_or_maintain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = _FakeClock()
    metrics = _RecordingMetrics()
    backend = LocalBackend(CacheSettings(), clock, metrics)
    empty = backend.storage_usage()
    assert empty['result_bytes'] == empty['lease_bytes'] == 0
    tag = DependencyTag.entity('production', 'Asset', 1)
    backend.resolve((tag,))
    assert _commit(backend, 'result', b'[]')
    lease = backend.reserve('pending', 10)
    assert lease is not None
    usage = backend.storage_usage()
    assert set(usage) == {'result_bytes', 'generation_bytes', 'lease_bytes'}
    assert usage['result_bytes'] == len('result') + len(b'[]') + 128 + 64
    assert usage['generation_bytes'] > empty['generation_bytes'] > 0
    assert (
        usage['lease_bytes'] == len('pending') + len(encode_value((lease.token, 1010.0))) + 96 + 64
    )
    increments, observations = list(metrics._increments), list(metrics._observations)
    clock._advance(8000)

    def forbidden_clock() -> float:
        raise AssertionError('snapshot must not sample time')

    def forbidden_maintenance(_store: LocalStore, _operation: StoreOperation) -> None:
        raise AssertionError('snapshot must not advance maintenance')

    monkeypatch.setattr(backend, '_clock', forbidden_clock)
    monkeypatch.setattr(LocalStore, '_maintain', forbidden_maintenance)
    assert backend.storage_usage() == usage
    assert metrics._increments == increments
    assert metrics._observations == observations
    isolated = backend.storage_usage()
    isolated['result_bytes'] = -1
    isolated.clear()
    assert backend.storage_usage() == usage
    backend.close()
    assert backend.storage_usage() == dict.fromkeys(usage, 0)


def test_storage_usage_serializes_with_store_close() -> None:
    backend = LocalBackend(CacheSettings(), _FakeClock(), _RecordingMetrics())
    entered = Event()

    def snapshot() -> dict[str, int]:
        entered.set()
        return backend.storage_usage()

    with ThreadPoolExecutor(max_workers=1) as executor:
        with backend._lock:
            pending = executor.submit(snapshot)
            assert entered.wait(timeout=5)
            with pytest.raises(TimeoutError):
                pending.result(timeout=0.05)
            backend.close()
        assert pending.result(timeout=5) == {
            'result_bytes': 0,
            'generation_bytes': 0,
            'lease_bytes': 0,
        }
