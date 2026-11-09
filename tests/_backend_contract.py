import asyncio
from dataclasses import dataclass
from typing import Protocol

import pytest

from sg_query_cache._async_backend import AsyncCacheBackend
from sg_query_cache._backend import CacheBackend
from sg_query_cache._codec import encode_value
from sg_query_cache._config import CacheSettings
from sg_query_cache._keys import KeyBuilder
from sg_query_cache._model import (
    CacheHit,
    DependencyTag,
    FillLease,
    GenerationChange,
    GenerationVector,
)
from tests._fakes import FakeClock, RecordingMetrics

_BASE_KEY = f'sgqc:v1:base:{"a" * 64}'
_SITE = 'production'


@dataclass(frozen=True, slots=True)
class BackendHarness:
    backend: CacheBackend
    clock: FakeClock
    metrics: RecordingMetrics


class BackendFactory(Protocol):
    def __call__(self, settings: CacheSettings | None = None) -> BackendHarness: ...


class SyncDriver:
    __slots__ = ('_backend', '_runner')

    def __init__(self, backend: AsyncCacheBackend, runner: asyncio.Runner) -> None:
        self._backend = backend
        self._runner = runner

    def change(self, changes: tuple[GenerationChange, ...]) -> None:
        self._runner.run(self._backend.change(changes))

    def close(self) -> None:
        self._runner.run(self._backend.close())

    def commit(self, lease: FillLease, payload: bytes, remaining_seconds: float) -> bool:
        return self._runner.run(self._backend.commit(lease, payload, remaining_seconds))

    def flush_site(self, site_namespace: str, reason: str) -> None:
        self._runner.run(self._backend.flush_site(site_namespace, reason))

    def lookup(self, key: str) -> CacheHit | None:
        return self._runner.run(self._backend.lookup(key))

    def release(self, lease: FillLease) -> bool:
        return self._runner.run(self._backend.release(lease))

    def reserve(self, key: str, lease_seconds: float) -> FillLease | None:
        return self._runner.run(self._backend.reserve(key, lease_seconds))

    def resolve(self, tags: tuple[DependencyTag, ...]) -> GenerationVector:
        return self._runner.run(self._backend.resolve(tags))


def _qualified_key(vector: GenerationVector) -> str:
    return KeyBuilder.qualified_key(_BASE_KEY, vector, 'result')


def _store(backend: CacheBackend, key: str, payload: bytes, age: float = 30.0) -> None:
    lease = backend.reserve(key, 10.0)
    assert lease is not None
    assert backend.commit(lease, payload, age)


class BackendContractTests:
    def test_absent_scoped_tag_gets_a_fresh_token(
        self,
        backend_factory: BackendFactory,
    ) -> None:
        harness = backend_factory()
        tag = DependencyTag.entity(_SITE, 'Asset', 7)

        first = harness.backend.resolve((tag,))
        harness.clock.advance(7_201.0)
        second = harness.backend.resolve((tag,))

        assert second.entries[0][1] > first.entries[0][1]

    def test_change_replaces_materialized_scoped_tag_without_creating_absent_tag(
        self,
        backend_factory: BackendFactory,
    ) -> None:
        # This checks materialized tags, so use normal retention rather than an expiry fixture.
        harness = backend_factory(CacheSettings())
        materialized = DependencyTag.entity(_SITE, 'Asset', 7)
        absent = DependencyTag.entity(_SITE, 'Asset', 8)
        first_token = harness.backend.resolve((materialized,)).entries[0][1]

        harness.backend.change((GenerationChange(materialized),))
        harness.clock.advance(0.15)
        changed_token = harness.backend.resolve((materialized,)).entries[0][1]
        harness.backend.change((GenerationChange(absent), GenerationChange(absent)))
        absent_token = harness.backend.resolve((absent,)).entries[0][1]

        assert changed_token == first_token + 1
        assert absent_token == changed_token + 1

    def test_duplicate_and_reordered_changes_only_redirect_to_misses(
        self,
        backend_factory: BackendFactory,
    ) -> None:
        harness = backend_factory()
        first_tag = DependencyTag.type_membership(_SITE, 'Asset')
        second_tag = DependencyTag.type_field(_SITE, 'Asset', 'code')
        old_vector = harness.backend.resolve((first_tag, second_tag))
        _store(harness.backend, _qualified_key(old_vector), encode_value([{'id': 7}]))

        harness.backend.change(
            (
                GenerationChange(second_tag),
                GenerationChange(first_tag),
                GenerationChange(second_tag),
            )
        )
        new_vector = harness.backend.resolve((first_tag, second_tag))

        assert all(
            new > old
            for (_old_tag, old), (_new_tag, new) in zip(
                old_vector.entries,
                new_vector.entries,
                strict=True,
            )
        )
        assert harness.backend.lookup(_qualified_key(new_vector)) is None

    def test_expired_entry_is_absent(self, backend_factory: BackendFactory) -> None:
        harness = backend_factory()
        key = 'qualified-result'
        _store(harness.backend, key, encode_value({'id': 7}), age=2.0)

        harness.clock.advance(2.0)

        assert harness.backend.lookup(key) is None

    def test_expired_lease_holder_cannot_overwrite_later_holder(
        self,
        backend_factory: BackendFactory,
    ) -> None:
        harness = backend_factory()
        key = 'qualified-result'
        first = harness.backend.reserve(key, 2.0)
        assert first is not None
        harness.clock.advance(2.0)
        second = harness.backend.reserve(key, 2.0)
        assert second is not None

        assert not harness.backend.commit(first, encode_value('stale'), 10.0)
        assert harness.backend.commit(second, encode_value('fresh'), 10.0)
        hit = harness.backend.lookup(key)
        assert hit is not None
        assert hit.payload == encode_value('fresh')

    def test_flush_changes_the_site_generation(self, backend_factory: BackendFactory) -> None:
        harness = backend_factory()
        site = DependencyTag.site(_SITE)
        before = harness.backend.resolve((site,)).entries[0][1]

        assert harness.backend.flush_site(_SITE, 'operator') is None

        after = harness.backend.resolve((site,)).entries[0][1]
        assert after > before

    def test_generation_change_redirects_later_lookup_to_a_miss(
        self,
        backend_factory: BackendFactory,
    ) -> None:
        harness = backend_factory()
        tag = DependencyTag.type_membership(_SITE, 'Asset')
        old_vector = harness.backend.resolve((tag,))
        _store(harness.backend, _qualified_key(old_vector), encode_value([{'id': 7}]))

        harness.backend.change((GenerationChange(tag),))
        new_vector = harness.backend.resolve((tag,))

        assert new_vector != old_vector
        assert harness.backend.lookup(_qualified_key(new_vector)) is None

    @pytest.mark.parametrize(
        'tag',
        [
            DependencyTag.site(_SITE),
            DependencyTag.schema(_SITE),
            DependencyTag.authorization(_SITE),
            DependencyTag.human_visibility(_SITE),
        ],
    )
    def test_global_tags_are_permanent_and_changed(
        self,
        backend_factory: BackendFactory,
        tag: DependencyTag,
    ) -> None:
        harness = backend_factory()
        first = harness.backend.resolve((tag,)).entries[0][1]
        harness.clock.advance(7_201.0)

        retained = harness.backend.resolve((tag,)).entries[0][1]
        harness.backend.change((GenerationChange(tag),))
        changed = harness.backend.resolve((tag,)).entries[0][1]

        assert retained == first
        assert changed > retained

    def test_lookup_reports_decreasing_backend_authoritative_age(
        self,
        backend_factory: BackendFactory,
    ) -> None:
        harness = backend_factory()
        clock = harness.clock
        key = 'qualified-result'
        store_started = clock()
        _store(harness.backend, key, encode_value({'id': 7}), age=10.0)
        store_finished = clock()

        first_started = clock()
        first = harness.backend.lookup(key)
        first_finished = clock()
        clock.advance(2.5)
        second_started = clock()
        second = harness.backend.lookup(key)
        second_finished = clock()

        assert first is not None
        assert second is not None
        # A lookup reports the backend's remaining age less its own duration, so a real server's
        # value lies within bounds measured around each call; a fake clock makes them exact.
        for hit, started, finished in (
            (first, first_started, first_finished),
            (second, second_started, second_finished),
        ):
            latest = 10.0 - (started - store_finished)
            earliest = 10.0 - (finished - store_started) - (finished - started)
            assert earliest - 0.01 <= hit.remaining_seconds <= latest + 0.01

    def test_lookup_returns_a_fresh_decoded_value(
        self,
        backend_factory: BackendFactory,
    ) -> None:
        harness = backend_factory()
        key = 'qualified-result'
        _store(harness.backend, key, encode_value({'links': [{'id': 7}]}))

        first = harness.backend.lookup(key)
        assert first is not None
        assert first.decoded is not None
        first_value = first.decoded.value
        assert isinstance(first_value, dict)
        first_value['links'] = []

        second = harness.backend.lookup(key)

        assert second is not None
        assert second.decoded is not None
        assert second.decoded.value == {'links': [{'id': 7}]}
        assert second.decoded.value is not first_value

    def test_malformed_payload_is_a_removed_miss(
        self,
        backend_factory: BackendFactory,
    ) -> None:
        harness = backend_factory()
        key = 'qualified-result'
        _store(harness.backend, key, b'not-typed-json')

        assert harness.backend.lookup(key) is None
        assert harness.backend.lookup(key) is None

        _store(harness.backend, key, encode_value('replacement'))
        hit = harness.backend.lookup(key)
        assert hit is not None
        assert hit.payload == encode_value('replacement')

    def test_release_is_lease_token_checked(self, backend_factory: BackendFactory) -> None:
        harness = backend_factory()
        key = 'qualified-result'
        first = harness.backend.reserve(key, 2.0)
        assert first is not None
        harness.clock.advance(2.0)
        second = harness.backend.reserve(key, 2.0)
        assert second is not None

        assert not harness.backend.release(first)
        assert harness.backend.reserve(key, 2.0) is None
        assert harness.backend.release(second)
        assert harness.backend.reserve(key, 2.0) is not None

    def test_resolve_returns_every_tag_in_request_order(
        self,
        backend_factory: BackendFactory,
    ) -> None:
        harness = backend_factory()
        tags = (
            DependencyTag.type_field(_SITE, 'Asset', 'code'),
            DependencyTag.site(_SITE),
            DependencyTag.entity(_SITE, 'Asset', 7),
        )

        vector = harness.backend.resolve(tags)

        assert tuple(tag for tag, _token in vector.entries) == tags
        assert all(token > 0 for _tag, token in vector.entries)

    def test_retired_scoped_token_is_never_reused(
        self,
        backend_factory: BackendFactory,
    ) -> None:
        harness = backend_factory()
        retired = DependencyTag.type_value(_SITE, 'Asset')
        intervening = DependencyTag.type_value(_SITE, 'Shot')
        first = harness.backend.resolve((retired,)).entries[0][1]
        harness.clock.advance(7_201.0)
        middle = harness.backend.resolve((intervening,)).entries[0][1]

        fresh = harness.backend.resolve((retired,)).entries[0][1]

        assert first < middle < fresh
