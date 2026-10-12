import asyncio
import hmac
from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from typing import ClassVar

import pytest
from typing_extensions import override

import sg_query_cache._async_cache as async_core
import sg_query_cache._async_local as async_local
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
from sg_query_cache._model import ExactCall, FillLease, GenerationChange, GenerationVector
from sg_query_cache._valkey_common import lookup_operands, reserve_operands
from tests._fakes import _FakeClock, _RecordingMetrics

_CONTEXT = ClientContext(
    'https://example.invalid',
    'production',
    localized=False,
    convert_datetimes_to_utc=True,
    records_per_page=500,
)
_IDENTITY = Identity('script', 'tests', 'wire')
_INCARNATION = 'a' * 32
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
_SECRET = b'0123456789abcdef0123456789abcdef'
_PAYLOAD = (
    b'["tuple",[["str","sg-query-cache-result-v1"],["float",0.0],'
    b'["float",30.0],["float",0.125],["map",[]]]]'
)


class _AsyncRaw:
    def __init__(self, clock: _FakeClock) -> None:
        self._raw = _Raw(clock)

    async def schema_read(self, project_entity: object = None) -> object:
        return self._raw.schema_read(project_entity)

    async def share_thumbnail(self, *args: object, **kwargs: object) -> int:
        return self._raw.share_thumbnail(*args, **kwargs)


class _Keys(KeyBuilder):
    _bases: ClassVar[list[str]] = []
    _qualified: ClassVar[list[tuple[str, GenerationVector, str]]] = []

    def __init__(self) -> None:
        _Keys._bases = []
        _Keys._qualified = []

    @staticmethod
    @override
    def base_key(
        site: str, identity: Identity, client_options: Mapping[str, object], call: ExactCall
    ) -> str:
        key = KeyBuilder.base_key(site, identity, client_options, call)
        _Keys._bases.append(key)
        return key

    @staticmethod
    @override
    def qualified_key(base_key: str, vector: GenerationVector, purpose: _KeyPurpose) -> str:
        key = KeyBuilder.qualified_key(base_key, vector, purpose)
        _Keys._qualified.append((purpose, vector, key))
        return key


class _Raw:
    def __init__(self, clock: _FakeClock) -> None:
        self._clock = clock

    def schema_read(self, project_entity: object = None) -> object:
        _ = project_entity
        self._clock._advance(0.125)
        return {}

    def share_thumbnail(self, *args: object, **kwargs: object) -> int:
        _ = args, kwargs
        return 1


class _SyncBackend(LocalBackend):
    def __init__(self, clock: _FakeClock) -> None:
        super().__init__(CacheSettings(), clock, _RecordingMetrics())
        self._changes: list[tuple[GenerationChange, ...]] = []
        self._leases: list[FillLease] = []

    @override
    def change(self, changes: tuple[GenerationChange, ...]) -> None:
        self._changes.append(changes)
        super().change(changes)

    @override
    def reserve(self, key: str, lease_seconds: float) -> FillLease | None:
        lease = super().reserve(key, lease_seconds)
        if lease is not None:
            self._leases.append(lease)
        return lease


class _AsyncBackend(AsyncLocalBackend):
    def __init__(self, clock: _FakeClock) -> None:
        super().__init__(CacheSettings(), clock, _RecordingMetrics())
        self._changes: list[tuple[GenerationChange, ...]] = []
        self._leases: list[FillLease] = []

    @override
    async def change(self, changes: tuple[GenerationChange, ...]) -> None:
        self._changes.append(changes)
        await super().change(changes)

    @override
    async def reserve(self, key: str, lease_seconds: float) -> FillLease | None:
        lease = await super().reserve(key, lease_seconds)
        if lease is not None:
            self._leases.append(lease)
        return lease


@dataclass
class _Trace:
    _bases: list[str]
    _qualified: list[tuple[str, GenerationVector, str]]
    _payloads: list[bytes]
    _digests: set[str]
    _changes: list[tuple[GenerationChange, ...]]
    _leases: list[FillLease]


def _digests(metrics: _RecordingMetrics) -> set[str]:
    return {
        labels['exact_call_digest']
        for _, labels in metrics._increments
        if 'exact_call_digest' in labels
    }


async def _async_trace(
    monkeypatch: pytest.MonkeyPatch, project: dict[str, object] | None
) -> _Trace:
    clock = _FakeClock()
    tokens = iter((_INCARNATION, 'b' * 32, 'c' * 32))

    def token(_size: int) -> str:
        return next(tokens)

    monkeypatch.setattr(async_core, 'monotonic', clock)
    monkeypatch.setattr(async_local, 'token_hex', token)
    backend = _AsyncBackend(clock)
    keys = _Keys()
    metrics = _RecordingMetrics()
    cache = AsyncQueryCache(
        CacheSettings(), _IDENTITY, keys, l1=backend, metrics=SafeMetrics(metrics, _SECRET)
    )
    client = AsyncCachedShotgun(_AsyncRaw(clock), cache, _CONTEXT)
    assert await client.schema_read(project) == {}
    payloads = [record.payload for record in backend._store._results.values()]
    assert await client.share_thumbnail([], source_entity={'type': 'Asset', 'id': 7}) == 1
    await cache.close()
    return _Trace(
        keys._bases,
        list(dict.fromkeys(keys._qualified)),
        payloads,
        _digests(metrics),
        backend._changes,
        backend._leases,
    )


def _sync_trace(monkeypatch: pytest.MonkeyPatch, project: dict[str, object] | None) -> _Trace:
    clock = _FakeClock()
    tokens = iter((_INCARNATION, 'b' * 32, 'c' * 32))

    def token(_size: int) -> str:
        return next(tokens)

    monkeypatch.setattr(sync_core, 'monotonic', clock)
    monkeypatch.setattr(sync_local, 'token_hex', token)
    backend = _SyncBackend(clock)
    keys = _Keys()
    metrics = _RecordingMetrics()
    cache = QueryCache(
        CacheSettings(), _IDENTITY, keys, l1=backend, metrics=SafeMetrics(metrics, _SECRET)
    )
    client = CachedShotgun(_Raw(clock), cache, _CONTEXT)
    assert client.schema_read(project) == {}
    payloads = [record.payload for record in backend._store._results.values()]
    assert client.share_thumbnail([], source_entity={'type': 'Asset', 'id': 7}) == 1
    cache.close()
    return _Trace(
        keys._bases,
        list(dict.fromkeys(keys._qualified)),
        payloads,
        _digests(metrics),
        backend._changes,
        backend._leases,
    )


def test_public_wrappers_share_independent_golden_wire_and_invalidation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sync = _sync_trace(monkeypatch, None)
    asynchronous = asyncio.run(_async_trace(monkeypatch, None))
    assert asynchronous == sync
    base = 'sgqc:v1:base:' + sha256(_BASE_BYTES).hexdigest()
    assert sync._bases == [base]
    assert sync._payloads == [_PAYLOAD]
    assert [lease.key for lease in sync._leases] == [sync._qualified[0][2]]
    assert [lease.token for lease in sync._leases] == ['0000000000000001' + 'b' * 32]
    assert sync._digests == {hmac.digest(_SECRET, base.encode(), 'sha256').hex()}
    assert {purpose for purpose, _, _ in sync._qualified} == {'result'}
    vector = sync._qualified[0][1]
    assert vector.incarnation == _INCARNATION
    assert [(tag.kind.value, tag.components, token) for tag, token in vector.entries] == [
        ('site', ('production',), 1),
        ('schema', ('production',), 2),
        ('authorization', ('production',), 3),
    ]
    assert [
        [(change.tag.kind.value, change.tag.components) for change in changes]
        for changes in sync._changes
    ] == [[('site', ('production',))]] * 2
    lease = KeyBuilder.qualified_key(base, vector, 'lease')
    assert lease != sync._qualified[0][2]
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
        assert (lease if purpose == 'lease' else sync._qualified[0][2]) == expected
    result_key = sync._qualified[0][2]
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
    assert unscoped._bases
    assert sync._bases
    assert sync._bases != unscoped._bases
    assert sync._qualified != unscoped._qualified
    assert sync._digests != unscoped._digests
