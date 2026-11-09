import asyncio
import hmac
import json
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import ClassVar

import pytest
from typing_extensions import override

import sg_query_cache._async_cache as async_core
import sg_query_cache._cache as sync_core
import sg_query_cache._local as sync_local
from sg_query_cache import (
    AsyncCachedShotgun,
    AsyncLocalBackend,
    AsyncQueryCache,
    CachedShotgun,
    CacheSettings,
    ClientContext,
    Identity,
    LocalBackend,
    QueryCache,
)
from sg_query_cache._keys import KeyBuilder, _KeyPurpose
from sg_query_cache._metrics import SafeMetrics
from sg_query_cache._model import (
    DependencyPlan,
    ExactCall,
    FillLease,
    GenerationChange,
    GenerationVector,
)
from sg_query_cache._valkey_common import lookup_operands, reserve_operands
from tests._fakes import FakeClock, RecordingMetrics

# Hand-written typed JSON, independent of the production codec and key normalizers.
_BASE_BYTES = (
    b'["tuple",[["str","base-key"],["tuple",[["str","production"],'
    b'["tuple",[["str","script"],["str","tests"],["str","wire"],["null"],["null"]]],'
    b'["tuple",[["tuple",[["str","localized"],["bool",false]]],'
    b'["tuple",[["str","convert_datetimes_to_utc"],["bool",true]]]]],'
    b'["tuple",[["str","schema_read"],["tuple",[["tuple",[["str","project_entity"],'
    b'["null"]]],["tuple",[["str","site_url"],'
    b'["str","https://example.invalid"]]]]]]]]]]]'
)
_CONTEXT = ClientContext(
    site_url='https://example.invalid',
    site_namespace='production',
    localized=False,
    convert_datetimes_to_utc=True,
    records_per_page=500,
)
_IDENTITY = Identity('script', 'tests', 'wire')
_INCARNATION = 'a' * 32
_PARITY_OPERATIONS: tuple[tuple[str, tuple[object, ...], dict[str, object]], ...] = (
    ('find', ('Asset', [['id', 'is', 7]], ['id', 'code']), {}),
    (
        'find',
        ('Asset', [['code', 'is', 'hero']], ['id']),
        {'order': [{'field_name': 'code', 'direction': 'asc'}]},
    ),
    ('find_one', ('Asset', [['id', 'in', [7, 8]]], ['code']), {}),
    ('schema_read', (), {}),
    ('schema_read', ({'type': 'Project', 'id': 7},), {}),
    ('schema_entity_read', (), {}),
    ('schema_field_read', ('Asset',), {}),
    ('schema_field_read', ('Asset', 'code', {'type': 'Project', 'id': 7}), {}),
    (
        'work_schedule_read',
        ('2026-01-01', '2026-01-31'),
        {'user': {'type': 'HumanUser', 'id': 3}},
    ),
    ('create', ('Asset', {'code': 'oak'}), {}),
    ('update', ('Asset', 7, {'code': 'elm'}), {}),
    ('delete', ('Asset', 7), {}),
    ('revive', ('Asset', 7), {}),
    (
        'batch',
        (
            [
                {'request_type': 'create', 'entity_type': 'Asset', 'data': {'code': 'ash'}},
                {
                    'request_type': 'update',
                    'entity_type': 'Asset',
                    'entity_id': 7,
                    'data': {'code': 'fir'},
                },
                {'request_type': 'delete', 'entity_type': 'Shot', 'entity_id': 9},
            ],
        ),
        {},
    ),
    ('follow', ({'type': 'HumanUser', 'id': 3}, {'type': 'Asset', 'id': 7}), {}),
    ('schema_field_create', ('Asset', 'text', 'Caption'), {}),
    ('schema_field_delete', ('Asset', 'caption'), {}),
    ('schema_field_update', ('Asset', 'caption', {'description': 'New caption'}), {}),
    ('share_thumbnail', ([{'type': 'Asset', 'id': 7}],), {}),
    ('unfollow', ({'type': 'HumanUser', 'id': 3}, {'type': 'Asset', 'id': 7}), {}),
    ('update_project_last_accessed', ({'type': 'Project', 'id': 7},), {}),
    ('upload', ('Asset', 7, '/example/asset.mov'), {}),
    ('upload_filmstrip_thumbnail', ('Asset', 7, '/example/filmstrip.jpg'), {}),
    ('upload_thumbnail', ('Asset', 7, '/example/thumbnail.jpg'), {}),
    ('user_subscriptions_create', ([{'type': 'HumanUser', 'id': 3}],), {}),
    ('work_schedule_update', ('2026-01-01', True), {}),
)
_PARITY_RESULTS: dict[str, object] = {
    'batch': [
        {'type': 'Asset', 'id': 18, 'code': 'ash'},
        {'type': 'Asset', 'id': 7, 'code': 'fir'},
        True,
    ],
    'create': {'type': 'Asset', 'id': 17, 'code': 'oak'},
    'delete': True,
    'find': [{'type': 'Asset', 'id': 7}],
    'find_one': {'type': 'Asset', 'id': 7},
    'follow': {'type': 'Follow', 'id': 19},
    'revive': True,
    'schema_field_create': 'caption',
    'schema_field_delete': True,
    'schema_field_update': True,
    'share_thumbnail': 20,
    'unfollow': {'type': 'Follow', 'id': 19},
    'update': {'type': 'Asset', 'id': 7, 'code': 'elm'},
    'update_project_last_accessed': None,
    'upload': 21,
    'upload_filmstrip_thumbnail': 22,
    'upload_thumbnail': 23,
    'user_subscriptions_create': True,
    'work_schedule_update': {'date': '2026-01-01', 'working': True},
}
_PARITY_SETTINGS = CacheSettings(
    site_namespace='production',
    enabled_ttl_only_methods=('work_schedule_read',),
)
_PAYLOAD = (
    b'["tuple",[["str","sg-query-cache-result-v1"],["float",0.0],'
    b'["float",30.0],["float",0.125],["map",[]]]]'
)
_SCHEMA_RESPONSE: object = json.loads(
    (Path(__file__).parent / 'fixtures' / 'schema.json').read_text(encoding='utf-8')
)
_SECRET = b'0123456789abcdef0123456789abcdef'


class _AsyncParityRaw:
    def __getattr__(self, name: str) -> Callable[..., Awaitable[object]]:
        async def invoke(*_args: object, **_kwargs: object) -> object:
            return _parity_result(name)

        return invoke


class _AsyncRaw:
    def __init__(self, clock: FakeClock) -> None:
        self._raw = _Raw(clock)

    async def schema_read(self, project_entity: object = None) -> object:
        return self._raw.schema_read(project_entity)

    async def share_thumbnail(self, *args: object, **kwargs: object) -> int:
        return self._raw.share_thumbnail(*args, **kwargs)


class _Keys(KeyBuilder):
    bases: ClassVar[list[str]] = []
    qualified: ClassVar[list[tuple[str, GenerationVector, str]]] = []

    def __init__(self) -> None:
        _Keys.bases = []
        _Keys.qualified = []

    @staticmethod
    @override
    def base_key(
        site: str, identity: Identity, client_options: Mapping[str, object], call: ExactCall
    ) -> str:
        key = KeyBuilder.base_key(site, identity, client_options, call)
        _Keys.bases.append(key)
        return key

    @staticmethod
    @override
    def qualified_key(base_key: str, vector: GenerationVector, purpose: _KeyPurpose) -> str:
        key = KeyBuilder.qualified_key(base_key, vector, purpose)
        _Keys.qualified.append((purpose, vector, key))
        return key


class _ParityRaw:
    def __getattr__(self, name: str) -> Callable[..., object]:
        def invoke(*_args: object, **_kwargs: object) -> object:
            return _parity_result(name)

        return invoke


class _Raw:
    def __init__(self, clock: FakeClock) -> None:
        self._clock = clock

    def schema_read(self, project_entity: object = None) -> object:
        _ = project_entity
        self._clock.advance(0.125)
        return {}

    def share_thumbnail(self, *args: object, **kwargs: object) -> int:
        _ = args, kwargs
        return 1


class _SyncBackend(LocalBackend):
    def __init__(self, clock: FakeClock) -> None:
        super().__init__(CacheSettings(), clock, RecordingMetrics())
        self.changes: list[tuple[GenerationChange, ...]] = []
        self.leases: list[FillLease] = []

    @override
    def change(self, changes: tuple[GenerationChange, ...]) -> None:
        self.changes.append(changes)
        super().change(changes)

    @override
    def reserve(self, key: str, lease_seconds: float) -> FillLease | None:
        lease = super().reserve(key, lease_seconds)
        if lease is not None:
            self.leases.append(lease)
        return lease


class _AsyncBackend(AsyncLocalBackend):
    def __init__(self, clock: FakeClock) -> None:
        super().__init__(CacheSettings(), clock, RecordingMetrics())
        self.changes: list[tuple[GenerationChange, ...]] = []
        self.leases: list[FillLease] = []

    @override
    async def change(self, changes: tuple[GenerationChange, ...]) -> None:
        self.changes.append(changes)
        await super().change(changes)

    @override
    async def reserve(self, key: str, lease_seconds: float) -> FillLease | None:
        lease = await super().reserve(key, lease_seconds)
        if lease is not None:
            self.leases.append(lease)
        return lease


@dataclass
class _Trace:
    bases: list[str]
    qualified: list[tuple[str, GenerationVector, str]]
    payloads: list[bytes]
    digests: set[str]
    changes: list[tuple[GenerationChange, ...]]
    leases: list[FillLease]


async def _async_parity_trace(monkeypatch: pytest.MonkeyPatch) -> list[tuple[object, ...]]:
    trace: list[tuple[object, ...]] = []

    async def read_admitted(
        _cache: AsyncQueryCache,
        call: ExactCall,
        plan: DependencyPlan,
        fetch: Callable[[], Awaitable[object]],
        client_options: Mapping[str, object],
        *_args: object,
        **_kwargs: object,
    ) -> object:
        trace.append(('read', call, plan, dict(client_options)))
        return await fetch()

    async def read_parsed_admitted(
        _cache: AsyncQueryCache,
        call: ExactCall,
        plan: DependencyPlan,
        fetch: Callable[[], Awaitable[object]],
        client_options: Mapping[str, object],
        parse: Callable[[object], object],
        **_kwargs: object,
    ) -> object:
        trace.append(('read', call, plan, dict(client_options)))
        return parse(await fetch())

    async def invalidate_admitted(
        _cache: AsyncQueryCache,
        changes: tuple[GenerationChange, ...],
        **_kwargs: object,
    ) -> bool:
        trace.append(('invalidate', changes))
        return True

    monkeypatch.setattr(AsyncQueryCache, 'read_admitted', read_admitted)
    monkeypatch.setattr(AsyncQueryCache, 'read_parsed_admitted', read_parsed_admitted)
    monkeypatch.setattr(AsyncQueryCache, 'invalidate_admitted', invalidate_admitted)
    metrics = RecordingMetrics()
    cache = AsyncQueryCache(
        _PARITY_SETTINGS,
        _IDENTITY,
        l1=AsyncLocalBackend(_PARITY_SETTINGS, FakeClock(), metrics),
        metrics=metrics,
    )
    wrapper = AsyncCachedShotgun(_AsyncParityRaw(), cache, _CONTEXT)
    for name, args, kwargs in _PARITY_OPERATIONS:
        await getattr(wrapper, name)(*args, **kwargs)
    await cache.close()
    return trace


def _digests(metrics: RecordingMetrics) -> set[str]:
    return {
        labels['exact_call_digest']
        for _, labels in metrics.increments
        if 'exact_call_digest' in labels
    }


async def _async_trace(
    monkeypatch: pytest.MonkeyPatch, project: dict[str, object] | None
) -> _Trace:
    clock = FakeClock()
    tokens = iter((_INCARNATION, 'b' * 32, 'c' * 32))

    def token(_size: int) -> str:
        return next(tokens)

    monkeypatch.setattr(async_core, 'monotonic', clock)
    monkeypatch.setattr(sync_local, 'token_hex', token)
    backend = _AsyncBackend(clock)
    keys = _Keys()
    metrics = RecordingMetrics()
    cache = AsyncQueryCache(
        CacheSettings(),
        _IDENTITY,
        key_builder=keys,
        l1=backend,
        metrics=SafeMetrics(metrics, _SECRET),
    )
    client = AsyncCachedShotgun(_AsyncRaw(clock), cache, _CONTEXT)
    assert await client.schema_read(project) == {}
    payloads = [record.payload for record in backend._backend._store._results.values()]
    assert await client.share_thumbnail([], source_entity={'type': 'Asset', 'id': 7}) == 1
    await cache.close()
    return _Trace(
        keys.bases,
        list(dict.fromkeys(keys.qualified)),
        payloads,
        _digests(metrics),
        backend.changes,
        backend.leases,
    )


def _parity_result(name: str) -> object:
    if name == 'schema_read':
        return _SCHEMA_RESPONSE
    return _PARITY_RESULTS.get(name)


def _sync_parity_trace(monkeypatch: pytest.MonkeyPatch) -> list[tuple[object, ...]]:
    trace: list[tuple[object, ...]] = []

    def read(
        _cache: QueryCache,
        call: ExactCall,
        plan: DependencyPlan,
        fetch: Callable[[], object],
        client_options: Mapping[str, object],
        *_args: object,
        **_kwargs: object,
    ) -> object:
        trace.append(('read', call, plan, dict(client_options)))
        return fetch()

    def read_parsed(
        _cache: QueryCache,
        call: ExactCall,
        plan: DependencyPlan,
        fetch: Callable[[], object],
        client_options: Mapping[str, object],
        parse: Callable[[object], object],
    ) -> object:
        trace.append(('read', call, plan, dict(client_options)))
        return parse(fetch())

    def invalidate(_cache: QueryCache, changes: tuple[GenerationChange, ...]) -> bool:
        trace.append(('invalidate', changes))
        return True

    monkeypatch.setattr(QueryCache, 'read', read)
    monkeypatch.setattr(QueryCache, 'read_parsed', read_parsed)
    monkeypatch.setattr(QueryCache, 'invalidate', invalidate)
    metrics = RecordingMetrics()
    cache = QueryCache(
        _PARITY_SETTINGS,
        _IDENTITY,
        key_builder=KeyBuilder(),
        l1=LocalBackend(_PARITY_SETTINGS, FakeClock(), metrics),
        metrics=metrics,
    )
    wrapper = CachedShotgun(_ParityRaw(), cache, _CONTEXT)
    for name, args, kwargs in _PARITY_OPERATIONS:
        getattr(wrapper, name)(*args, **kwargs)
    cache.close()
    return trace


def _sync_trace(monkeypatch: pytest.MonkeyPatch, project: dict[str, object] | None) -> _Trace:
    clock = FakeClock()
    tokens = iter((_INCARNATION, 'b' * 32, 'c' * 32))

    def token(_size: int) -> str:
        return next(tokens)

    monkeypatch.setattr(sync_core, 'monotonic', clock)
    monkeypatch.setattr(sync_local, 'token_hex', token)
    backend = _SyncBackend(clock)
    keys = _Keys()
    metrics = RecordingMetrics()
    cache = QueryCache(
        CacheSettings(),
        _IDENTITY,
        key_builder=keys,
        l1=backend,
        metrics=SafeMetrics(metrics, _SECRET),
    )
    client = CachedShotgun(_Raw(clock), cache, _CONTEXT)
    assert client.schema_read(project) == {}
    payloads = [record.payload for record in backend._store._results.values()]
    assert client.share_thumbnail([], source_entity={'type': 'Asset', 'id': 7}) == 1
    cache.close()
    return _Trace(
        keys.bases,
        list(dict.fromkeys(keys.qualified)),
        payloads,
        _digests(metrics),
        backend.changes,
        backend.leases,
    )


def test_public_wrappers_share_independent_golden_wire_and_invalidation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sync = _sync_trace(monkeypatch, None)
    asynchronous = asyncio.run(_async_trace(monkeypatch, None))
    assert asynchronous == sync
    base = 'sgqc:v1:base:' + sha256(_BASE_BYTES).hexdigest()
    assert sync.bases == [base]
    assert sync.payloads == [_PAYLOAD]
    assert [lease.key for lease in sync.leases] == [sync.qualified[0][2]]
    assert [lease.token for lease in sync.leases] == ['0000000000000001' + 'b' * 32]
    assert sync.digests == {hmac.digest(_SECRET, base.encode(), 'sha256').hex()}
    assert {purpose for purpose, _, _ in sync.qualified} == {'result'}
    vector = sync.qualified[0][1]
    assert vector.incarnation == _INCARNATION
    assert [(tag.kind.value, tag.components, token) for tag, token in vector.entries] == [
        ('site', ('production',), 1),
        ('schema', ('production',), 2),
        ('authorization', ('production',), 3),
    ]
    assert [
        [(change.tag.kind.value, change.tag.components) for change in changes]
        for changes in sync.changes
    ] == [[('site', ('production',))]] * 2
    lease = KeyBuilder.qualified_key(base, vector, 'lease')
    assert lease != sync.qualified[0][2]
    assert lease.startswith(f'sgqc:v1:lease:{_INCARNATION}:')
    # Both qualified domains use independent literal typed-vector operands.
    for purpose in ('lease', 'result'):
        operand = (
            b'["tuple",[["str","qualified-key"],["tuple",[["str","'
            + base.encode()
            + b'"],["str","'
            + purpose.encode()
            + b'"],["tuple",[["str","'
            + _INCARNATION.encode()
            + b'"],["tuple",'
            b'[["tuple",[["str","site"],["tuple",[["str","production"]]],["int",1]]],'
            b'["tuple",[["str","schema"],["tuple",[["str","production"]]],["int",2]]],'
            b'["tuple",[["str","authorization"],["tuple",[["str","production"]]],'
            b'["int",3]]]]]]]]]]]'
        )
        expected = f'sgqc:v1:{purpose}:{_INCARNATION}:' + sha256(operand).hexdigest()
        assert (lease if purpose == 'lease' else sync.qualified[0][2]) == expected
    result_key = sync.qualified[0][2]
    assert lookup_operands('fixture', result_key).keys == (f'fixture:v1:result:{result_key}',)
    reservation = reserve_operands('fixture', result_key, 30000, 'b' * 32)
    assert reservation.keys == (f'fixture:v1:lease:{result_key}',)
    assert reservation.args == ('b' * 32, 30000)


def test_semantically_different_project_inputs_separate_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    unscoped = _sync_trace(monkeypatch, None)
    project: dict[str, object] = {'type': 'Project', 'id': 7}
    sync = _sync_trace(monkeypatch, project)
    asynchronous = asyncio.run(_async_trace(monkeypatch, project))
    assert sync == asynchronous
    assert unscoped.bases
    assert sync.bases
    assert sync.bases != unscoped.bases
    assert sync.qualified != unscoped.qualified
    assert sync.digests != unscoped.digests


def test_wrappers_send_identical_reads_and_fences_to_their_caches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sync = _sync_parity_trace(monkeypatch)
    asynchronous = asyncio.run(_async_parity_trace(monkeypatch))

    assert asynchronous == sync
    read_methods = {
        entry[1].method for entry in sync if entry[0] == 'read' and isinstance(entry[1], ExactCall)
    }
    assert read_methods == {
        'find',
        'find_one',
        'schema_entity_read',
        'schema_field_read',
        'schema_read',
        'schema_view',
        'work_schedule_read',
    }
    assert sum(entry[0] == 'invalidate' for entry in sync) >= len(
        [name for name, _args, _kwargs in _PARITY_OPERATIONS if name in _PARITY_RESULTS]
    )
