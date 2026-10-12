import asyncio
import inspect
from collections.abc import Awaitable, Callable, Iterator
from typing import cast

import pytest

import sg_query_cache._async_local as async_local_module
from sg_query_cache._async_backend import AsyncCacheBackend
from sg_query_cache._async_local import AsyncLocalBackend
from sg_query_cache._backend import (
    BackendCapacity,
    BackendProtocolError,
    BackendUnavailable,
)
from sg_query_cache._cache_common import TimedResult, decode_hit, encode_result
from sg_query_cache._codec import encode_value
from sg_query_cache._config import CacheSettings, L1Settings, LimitSettings
from sg_query_cache._local_store import LocalStore, StoreOperation
from sg_query_cache._model import (
    CacheHit,
    DependencyTag,
    FillLease,
    GenerationChange,
)
from tests._backend_contract import (
    SyncDriver,
    _BackendContractTests,
    _BackendFactory,
    _BackendHarness,
)
from tests._fakes import _FakeClock, _RecordingMetrics
from tests.local._store_helpers import CountedQueue

_ASYNC_BACKEND_METHODS = (
    'change',
    'close',
    'commit',
    'flush_site',
    'lookup',
    'release',
    'reserve',
    'resolve',
)


async def _invoke(backend: AsyncLocalBackend, operation: str) -> object:
    result: object = None
    match operation:
        case 'change':
            result = await backend.change(())
        case 'close':
            result = await backend.close()
        case 'commit':
            result = await backend.commit(FillLease('key', 'token'), encode_value(1), 1.0)
        case 'flush_site':
            result = await backend.flush_site('production', 'operator')
        case 'lookup':
            result = await backend.lookup('key')
        case 'release':
            result = await backend.release(FillLease('key', 'token'))
        case 'reserve':
            result = await backend.reserve('key', 1.0)
        case 'resolve':
            result = await backend.resolve((DependencyTag.site('production'),))
        case _:
            raise AssertionError(operation)
    return result


async def _invoke_invalid(backend: AsyncLocalBackend, operation: str) -> object:
    calls: dict[str, Callable[[], Awaitable[object]]] = {
        'change': lambda: backend.change(cast('tuple[GenerationChange, ...]', [])),
        'commit_lease': lambda: backend.commit(cast('FillLease', object()), b'[]', 1.0),
        'commit_payload': lambda: backend.commit(
            FillLease('key', 'token'), cast('bytes', 'value'), 1.0
        ),
        'commit_seconds': lambda: backend.commit(FillLease('key', 'token'), b'[]', 0.0),
        'flush_reason': lambda: backend.flush_site('production', ''),
        'flush_site': lambda: backend.flush_site('', 'operator'),
        'lookup': lambda: backend.lookup(''),
        'release': lambda: backend.release(cast('FillLease', object())),
        'reserve_key': lambda: backend.reserve('', 1.0),
        'reserve_seconds': lambda: backend.reserve('key', 0.0),
        'resolve': lambda: backend.resolve(cast('tuple[DependencyTag, ...]', [])),
    }
    if operation not in calls:
        raise AssertionError(operation)
    return await calls[operation]()


async def _invoke_populated_operation(backend: AsyncLocalBackend, operation: str) -> None:
    tag = DependencyTag.site('production')
    match operation:
        case 'change':
            await backend.change((GenerationChange(tag),))
        case 'commit':
            record = backend._store._leases['lease-193']
            await backend.commit(FillLease('lease-193', record._token), b'[]', 10.0)
        case 'flush_site':
            await backend.flush_site('production', 'operator')
        case 'lookup':
            await backend.lookup('result-193')
        case 'release':
            record = backend._store._leases['lease-193']
            await backend.release(FillLease('lease-193', record._token))
        case 'reserve':
            await backend.reserve('new', 10.0)
        case 'resolve':
            await backend.resolve((tag, DependencyTag.entity('production', 'Asset', 193)))
        case _:
            raise AssertionError(operation)


@pytest.fixture
def backend_factory() -> Iterator[_BackendFactory]:
    runner = asyncio.Runner()
    backends: list[AsyncLocalBackend] = []

    def create(settings: CacheSettings | None = None) -> _BackendHarness:
        clock = _FakeClock()
        metrics = _RecordingMetrics()
        backend = AsyncLocalBackend(settings or CacheSettings(), clock, metrics)
        backends.append(backend)
        driver = SyncDriver(backend, runner)
        return _BackendHarness(driver, clock, metrics)

    yield create

    for backend in reversed(backends):
        runner.run(backend.close())
    runner.close()


class TestAsyncLocalBackend(_BackendContractTests):
    pass


def _populate_store(store: LocalStore, count: int = 193) -> None:
    for index in range(1, count + 1):
        result_key = f'result-{index}'
        result_lease = store.reserve(result_key, 10.0, 'b' * 32, StoreOperation(1_000.0))
        assert result_lease is not None
        assert store.commit(
            result_lease,
            encode_value({'id': 7}),
            60.0,
            StoreOperation(1_000.0),
        )
        assert store.reserve(f'lease-{index}', 60.0, 'b' * 32, StoreOperation(1_000.0)) is not None
        store.resolve(
            (DependencyTag.entity('production', 'Asset', index),),
            StoreOperation(1_000.0),
        )


def test_async_backend_protocol_declares_exactly_eight_coroutine_methods() -> None:
    declared = {
        name
        for name, value in vars(AsyncCacheBackend).items()
        if inspect.iscoroutinefunction(value)
    }

    assert declared == set(_ASYNC_BACKEND_METHODS)


def test_close_binds_outside_constructed_backend_and_is_idempotent_on_its_loop() -> None:
    backend = AsyncLocalBackend(CacheSettings(), _FakeClock(), _RecordingMetrics())
    runner = asyncio.Runner()
    try:
        runner.run(backend.close())
        runner.run(backend.close())
    finally:
        runner.close()


def test_commit_debits_preparation_time_from_ttl_and_source_age(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = _FakeClock()
    backend = AsyncLocalBackend(CacheSettings(), clock, _RecordingMetrics())
    original_fill_lease = async_local_module.fill_lease

    def delayed_fill_lease(value: object) -> FillLease:
        result = original_fill_lease(value)
        clock._advance(0.04)
        return result

    monkeypatch.setattr(async_local_module, 'fill_lease', delayed_fill_lease)
    payload = encode_result(TimedResult(0.0, {'id': 7}), 0.0, 0.1)

    async def commit_read_and_close() -> CacheHit | None:
        lease = await backend.reserve('result-key', 10.0)
        assert lease is not None
        assert await backend.commit(lease, payload, 0.1)
        hit = await backend.lookup('result-key')
        await backend.close()
        return hit

    hit = asyncio.run(commit_read_and_close())

    assert hit is not None
    assert hit.remaining_seconds <= 0.06 + 1e-12
    assert decode_hit(hit, 1.0).origin_elapsed == pytest.approx(0.04)


def test_commit_rejects_lifetime_expired_during_preparation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = _FakeClock()
    backend = AsyncLocalBackend(CacheSettings(), clock, _RecordingMetrics())
    original_fill_lease = async_local_module.fill_lease

    def delayed_fill_lease(value: object) -> FillLease:
        result = original_fill_lease(value)
        clock._advance(0.1)
        return result

    monkeypatch.setattr(async_local_module, 'fill_lease', delayed_fill_lease)

    async def commit_and_close() -> bool:
        lease = await backend.reserve('result-key', 10.0)
        assert lease is not None
        result = await backend.commit(lease, encode_value({'id': 7}), 0.05)
        assert await backend.lookup('result-key') is None
        await backend.close()
        return result

    assert asyncio.run(commit_and_close()) is False


def test_constructor_reports_clock_and_metadata_capacity_failures() -> None:
    def failing_clock() -> float:
        raise RuntimeError('sensitive injected failure')

    with pytest.raises(BackendUnavailable, match='monotonic clock is unavailable') as captured:
        AsyncLocalBackend(CacheSettings(), failing_clock, _RecordingMetrics())
    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None

    settings = CacheSettings(l1=L1Settings(metadata_budget_bytes=1))
    with pytest.raises(BackendCapacity, match='permanent generations'):
        AsyncLocalBackend(settings, _FakeClock(), _RecordingMetrics())


@pytest.mark.parametrize('operation', _ASYNC_BACKEND_METHODS)
def test_every_closed_backend_call_rejects_a_different_loop_before_state(
    operation: str,
) -> None:
    backend = AsyncLocalBackend(CacheSettings(), _FakeClock(), _RecordingMetrics())
    owner = asyncio.Runner()
    other = asyncio.Runner()
    try:
        owner.run(backend.close())

        with pytest.raises(RuntimeError, match='different event loop'):
            other.run(_invoke(backend, operation))
    finally:
        owner.close()
        other.close()


@pytest.mark.parametrize(
    ('operation', 'message'),
    [
        ('change', 'generation changes must be an immutable tuple'),
        ('commit_lease', 'operation requires a fill lease'),
        ('commit_payload', 'commit payload must be encoded bytes'),
        ('commit_seconds', 'remaining lifetime must be positive'),
        ('flush_reason', 'site flush reason must be short nonempty ASCII'),
        ('flush_site', 'site namespace is invalid'),
        ('lookup', 'backend key must be nonempty text'),
        ('release', 'operation requires a fill lease'),
        ('reserve_key', 'backend key must be nonempty text'),
        ('reserve_seconds', 'lease duration must be positive'),
        ('resolve', 'dependency vector must be an immutable tag tuple'),
    ],
)
def test_backend_protocol_validation_matches_the_synchronous_boundary(
    operation: str,
    message: str,
) -> None:
    backend = AsyncLocalBackend(CacheSettings(), _FakeClock(), _RecordingMetrics())
    runner = asyncio.Runner()
    try:
        with pytest.raises(BackendProtocolError, match=message):
            runner.run(_invoke_invalid(backend, operation))
        runner.run(backend.close())
    finally:
        runner.close()


@pytest.mark.parametrize(
    'operation',
    ['change', 'commit', 'flush_site', 'lookup', 'release', 'reserve', 'resolve'],
)
def test_each_store_operation_preserves_the_per_queue_visit_bound(operation: str) -> None:
    backend = AsyncLocalBackend(CacheSettings(), _FakeClock(), _RecordingMetrics())
    _populate_store(backend._store)
    results = CountedQueue(backend._store._result_queue)
    leases = CountedQueue(backend._store._lease_queue)
    generations = CountedQueue(backend._store._generation_queue)
    backend._store._result_queue = results
    backend._store._lease_queue = leases
    backend._store._generation_queue = generations

    async def invoke_and_close() -> None:
        await _invoke_populated_operation(backend, operation)
        await backend.close()

    asyncio.run(invoke_and_close())

    assert results._visits == leases._visits == generations._visits == 64


def test_lookup_decodes_once_and_debits_decode_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = _FakeClock()
    backend = AsyncLocalBackend(CacheSettings(), clock, _RecordingMetrics())
    payload = encode_value({'id': 7})
    original_decode = async_local_module.decode_value
    decode_calls = 0

    def delayed_decode(encoded: bytes) -> object:
        nonlocal decode_calls
        decode_calls += 1
        value = original_decode(encoded)
        clock._advance(0.25)
        return value

    monkeypatch.setattr(async_local_module, 'decode_value', delayed_decode)

    async def store_and_read() -> CacheHit | None:
        lease = await backend.reserve('result-key', 10.0)
        assert lease is not None
        assert await backend.commit(lease, payload, 30.0)
        hit = await backend.lookup('result-key')
        await backend.close()
        return hit

    hit = asyncio.run(store_and_read())

    assert hit is not None
    assert hit.decoded is not None
    assert hit.decoded.value == {'id': 7}
    assert hit.remaining_seconds == 29.75
    assert decode_calls == 1


def test_lookup_reports_a_backward_clock_as_unavailable() -> None:
    clock = _FakeClock()
    backend = AsyncLocalBackend(CacheSettings(), clock, _RecordingMetrics())
    runner = asyncio.Runner()
    try:
        assert runner.run(backend.reserve('key', 1.0)) is not None
        clock._advance(-1.0)

        with pytest.raises(BackendUnavailable, match='clock moved backward'):
            runner.run(backend.lookup('key'))
        runner.run(backend.close())
    finally:
        runner.close()


def test_reserve_normalizes_entropy_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = AsyncLocalBackend(CacheSettings(), _FakeClock(), _RecordingMetrics())

    def failing_entropy(_byte_count: int) -> str:
        raise OSError('sensitive injected failure')

    monkeypatch.setattr(async_local_module, 'token_hex', failing_entropy)
    runner = asyncio.Runner()
    try:
        with pytest.raises(BackendUnavailable, match='entropy source is unavailable') as captured:
            runner.run(backend.reserve('key', 1.0))
        assert captured.value.__cause__ is None
        assert captured.value.__context__ is None
        runner.run(backend.close())
    finally:
        runner.close()


def test_resolve_enforces_configured_tag_and_byte_bounds() -> None:
    limits = LimitSettings(max_dependency_tags=1, max_dependency_vector_bytes=50)
    backend = AsyncLocalBackend(
        CacheSettings(limits=limits),
        _FakeClock(),
        _RecordingMetrics(),
    )
    runner = asyncio.Runner()
    try:
        with pytest.raises(BackendProtocolError, match='tag bound'):
            runner.run(
                backend.resolve(
                    (
                        DependencyTag.site('production'),
                        DependencyTag.schema('production'),
                    )
                )
            )
        with pytest.raises(BackendProtocolError, match='byte bound'):
            runner.run(backend.resolve((DependencyTag.entity('production', 'Asset', 7),)))
        runner.run(backend.close())
    finally:
        runner.close()


def test_same_loop_concurrent_reservations_produce_one_current_lease() -> None:
    backend = AsyncLocalBackend(CacheSettings(), _FakeClock(), _RecordingMetrics())

    async def reserve_all() -> list[FillLease | None]:
        results = await asyncio.gather(
            *(backend.reserve('hot-qualified-key', 30.0) for _index in range(50))
        )
        await backend.close()
        return results

    results = asyncio.run(reserve_all())

    assert len(results) == 50
    assert sum(lease is not None for lease in results) == 1


def test_same_loop_operations_reject_calls_after_close() -> None:
    backend = AsyncLocalBackend(CacheSettings(), _FakeClock(), _RecordingMetrics())
    runner = asyncio.Runner()
    try:
        runner.run(backend.close())

        with pytest.raises(BackendUnavailable, match='backend is closed'):
            runner.run(backend.lookup('key'))
    finally:
        runner.close()


def test_storage_usage_is_fresh_and_does_not_observe_time_or_maintain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = _FakeClock()
    metrics = _RecordingMetrics()
    backend = AsyncLocalBackend(CacheSettings(), clock, metrics)

    async def scenario() -> None:
        empty = await backend.storage_usage()
        assert empty['result_bytes'] == empty['lease_bytes'] == 0
        await backend.resolve((DependencyTag.entity('production', 'Asset', 1),))
        result_lease = await backend.reserve('result', 10)
        assert result_lease is not None
        assert await backend.commit(result_lease, b'[]', 30)
        lease = await backend.reserve('pending', 10)
        assert lease is not None
        usage = await backend.storage_usage()
        assert set(usage) == {'result_bytes', 'generation_bytes', 'lease_bytes'}
        assert usage['result_bytes'] == len('result') + len(b'[]') + 128 + 64
        assert usage['generation_bytes'] > empty['generation_bytes'] > 0
        assert (
            usage['lease_bytes']
            == len('pending') + len(encode_value((lease.token, 1010.0))) + 96 + 64
        )
        increments, observations = list(metrics._increments), list(metrics._observations)
        clock._advance(8000)

        def forbidden_clock() -> float:
            raise AssertionError('snapshot must not sample time')

        def forbidden_maintenance(_store: LocalStore, _operation: StoreOperation) -> None:
            raise AssertionError('snapshot must not advance maintenance')

        monkeypatch.setattr(backend, '_clock', forbidden_clock)
        monkeypatch.setattr(LocalStore, '_maintain', forbidden_maintenance)
        assert await backend.storage_usage() == usage
        assert metrics._increments == increments
        assert metrics._observations == observations
        isolated = await backend.storage_usage()
        isolated['result_bytes'] = -1
        isolated.clear()
        assert await backend.storage_usage() == usage
        await backend.close()
        assert await backend.storage_usage() == dict.fromkeys(usage, 0)

    asyncio.run(scenario())
    with pytest.raises(RuntimeError, match='loop'):
        asyncio.run(backend.storage_usage())
