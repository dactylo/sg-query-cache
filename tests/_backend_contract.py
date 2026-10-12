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
from tests._fakes import _FakeClock, _RecordingMetrics

_BASE_KEY = f'sgqc:v1:base:{"a" * 64}'
_SITE = 'production'


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


@dataclass(frozen=True, slots=True)
class _BackendHarness:
    _backend: CacheBackend
    _clock: _FakeClock
    _metrics: _RecordingMetrics


class _BackendFactory(Protocol):
    def __call__(self, settings: CacheSettings | None = None) -> _BackendHarness: ...


def _qualified_key(vector: GenerationVector) -> str:
    return KeyBuilder.qualified_key(_BASE_KEY, vector, 'result')


def _store(backend: CacheBackend, key: str, payload: bytes, age: float = 30.0) -> None:
    lease = backend.reserve(key, 10.0)
    assert lease is not None
    assert backend.commit(lease, payload, age)


class _BackendContractTests:  # pyright: ignore[reportUnusedClass]
    def test_absent_scoped_tag_gets_a_fresh_token(
        self,
        backend_factory: _BackendFactory,
    ) -> None:
        harness = backend_factory()
        tag = DependencyTag.entity(_SITE, 'Asset', 7)

        first = harness._backend.resolve((tag,))
        harness._clock._advance(7_201.0)
        second = harness._backend.resolve((tag,))

        assert second.entries[0][1] > first.entries[0][1]

    def test_change_replaces_materialized_scoped_tag_without_creating_absent_tag(
        self,
        backend_factory: _BackendFactory,
    ) -> None:
        # This checks materialized tags, so use normal retention rather than an expiry fixture.
        harness = backend_factory(CacheSettings())
        materialized = DependencyTag.entity(_SITE, 'Asset', 7)
        absent = DependencyTag.entity(_SITE, 'Asset', 8)
        first_token = harness._backend.resolve((materialized,)).entries[0][1]

        harness._backend.change((GenerationChange(materialized),))
        harness._clock._advance(0.15)
        changed_token = harness._backend.resolve((materialized,)).entries[0][1]
        harness._backend.change((GenerationChange(absent), GenerationChange(absent)))
        absent_token = harness._backend.resolve((absent,)).entries[0][1]

        assert changed_token == first_token + 1
        assert absent_token == changed_token + 1

    def test_duplicate_and_reordered_changes_only_redirect_to_misses(
        self,
        backend_factory: _BackendFactory,
    ) -> None:
        harness = backend_factory()
        first_tag = DependencyTag.type_membership(_SITE, 'Asset')
        second_tag = DependencyTag.type_field(_SITE, 'Asset', 'code')
        old_vector = harness._backend.resolve((first_tag, second_tag))
        _store(harness._backend, _qualified_key(old_vector), encode_value([{'id': 7}]))

        harness._backend.change(
            (
                GenerationChange(second_tag),
                GenerationChange(first_tag),
                GenerationChange(second_tag),
            )
        )
        new_vector = harness._backend.resolve((first_tag, second_tag))

        assert all(
            new > old
            for (_old_tag, old), (_new_tag, new) in zip(
                old_vector.entries,
                new_vector.entries,
                strict=True,
            )
        )
        assert harness._backend.lookup(_qualified_key(new_vector)) is None

    def test_expired_entry_is_absent(self, backend_factory: _BackendFactory) -> None:
        harness = backend_factory()
        key = 'qualified-result'
        _store(harness._backend, key, encode_value({'id': 7}), age=2.0)

        harness._clock._advance(2.0)

        assert harness._backend.lookup(key) is None

    def test_expired_lease_holder_cannot_overwrite_later_holder(
        self,
        backend_factory: _BackendFactory,
    ) -> None:
        harness = backend_factory()
        key = 'qualified-result'
        first = harness._backend.reserve(key, 2.0)
        assert first is not None
        harness._clock._advance(2.0)
        second = harness._backend.reserve(key, 2.0)
        assert second is not None

        assert not harness._backend.commit(first, encode_value('stale'), 10.0)
        assert harness._backend.commit(second, encode_value('fresh'), 10.0)
        hit = harness._backend.lookup(key)
        assert hit is not None
        assert hit.payload == encode_value('fresh')

    def test_flush_changes_the_site_generation(self, backend_factory: _BackendFactory) -> None:
        harness = backend_factory()
        site = DependencyTag.site(_SITE)
        before = harness._backend.resolve((site,)).entries[0][1]

        assert harness._backend.flush_site(_SITE, 'operator') is None

        after = harness._backend.resolve((site,)).entries[0][1]
        assert after > before

    def test_generation_change_redirects_later_lookup_to_a_miss(
        self,
        backend_factory: _BackendFactory,
    ) -> None:
        harness = backend_factory()
        tag = DependencyTag.type_membership(_SITE, 'Asset')
        old_vector = harness._backend.resolve((tag,))
        _store(harness._backend, _qualified_key(old_vector), encode_value([{'id': 7}]))

        harness._backend.change((GenerationChange(tag),))
        new_vector = harness._backend.resolve((tag,))

        assert new_vector != old_vector
        assert harness._backend.lookup(_qualified_key(new_vector)) is None

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
        backend_factory: _BackendFactory,
        tag: DependencyTag,
    ) -> None:
        harness = backend_factory()
        first = harness._backend.resolve((tag,)).entries[0][1]
        harness._clock._advance(7_201.0)

        retained = harness._backend.resolve((tag,)).entries[0][1]
        harness._backend.change((GenerationChange(tag),))
        changed = harness._backend.resolve((tag,)).entries[0][1]

        assert retained == first
        assert changed > retained

    def test_lookup_reports_decreasing_backend_authoritative_age(
        self,
        backend_factory: _BackendFactory,
    ) -> None:
        harness = backend_factory()
        key = 'qualified-result'
        _store(harness._backend, key, encode_value({'id': 7}), age=10.0)

        first = harness._backend.lookup(key)
        harness._clock._advance(2.5)
        second = harness._backend.lookup(key)

        assert first is not None
        assert second is not None
        assert first.remaining_seconds == pytest.approx(10.0, abs=0.05)
        assert second.remaining_seconds == pytest.approx(7.5, abs=0.05)

    def test_lookup_returns_a_fresh_decoded_value(
        self,
        backend_factory: _BackendFactory,
    ) -> None:
        harness = backend_factory()
        key = 'qualified-result'
        _store(harness._backend, key, encode_value({'links': [{'id': 7}]}))

        first = harness._backend.lookup(key)
        assert first is not None
        assert first.decoded is not None
        first_value = first.decoded.value
        assert isinstance(first_value, dict)
        first_value['links'] = []

        second = harness._backend.lookup(key)

        assert second is not None
        assert second.decoded is not None
        assert second.decoded.value == {'links': [{'id': 7}]}
        assert second.decoded.value is not first_value

    def test_malformed_payload_is_a_removed_miss(
        self,
        backend_factory: _BackendFactory,
    ) -> None:
        harness = backend_factory()
        key = 'qualified-result'
        _store(harness._backend, key, b'not-typed-json')

        assert harness._backend.lookup(key) is None
        assert harness._backend.lookup(key) is None

        _store(harness._backend, key, encode_value('replacement'))
        hit = harness._backend.lookup(key)
        assert hit is not None
        assert hit.payload == encode_value('replacement')

    def test_release_is_lease_token_checked(self, backend_factory: _BackendFactory) -> None:
        harness = backend_factory()
        key = 'qualified-result'
        first = harness._backend.reserve(key, 2.0)
        assert first is not None
        harness._clock._advance(2.0)
        second = harness._backend.reserve(key, 2.0)
        assert second is not None

        assert not harness._backend.release(first)
        assert harness._backend.reserve(key, 2.0) is None
        assert harness._backend.release(second)
        assert harness._backend.reserve(key, 2.0) is not None

    def test_resolve_returns_every_tag_in_request_order(
        self,
        backend_factory: _BackendFactory,
    ) -> None:
        harness = backend_factory()
        tags = (
            DependencyTag.type_field(_SITE, 'Asset', 'code'),
            DependencyTag.site(_SITE),
            DependencyTag.entity(_SITE, 'Asset', 7),
        )

        vector = harness._backend.resolve(tags)

        assert tuple(tag for tag, _token in vector.entries) == tags
        assert all(token > 0 for _tag, token in vector.entries)

    def test_retired_scoped_token_is_never_reused(
        self,
        backend_factory: _BackendFactory,
    ) -> None:
        harness = backend_factory()
        retired = DependencyTag.type_value(_SITE, 'Asset')
        intervening = DependencyTag.type_value(_SITE, 'Shot')
        first = harness._backend.resolve((retired,)).entries[0][1]
        harness._clock._advance(7_201.0)
        middle = harness._backend.resolve((intervening,)).entries[0][1]

        fresh = harness._backend.resolve((retired,)).entries[0][1]

        assert first < middle < fresh
