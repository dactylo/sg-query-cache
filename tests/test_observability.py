from collections.abc import Callable, Generator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from threading import Barrier, Event, Lock
from time import monotonic, sleep
from typing import TypeVar, cast

import pytest

import sg_query_cache._cache as cache_module
from sg_query_cache._backend import BackendCapacity, BackendUnavailable
from sg_query_cache._cache import QueryCache
from sg_query_cache._cache_common import TimedResult, encode_result
from sg_query_cache._codec import encode_value
from sg_query_cache._config import CacheSettings, L1Settings, L2Settings, LimitSettings
from sg_query_cache._keys import KeyBuilder
from sg_query_cache._local import LocalBackend
from sg_query_cache._metrics import SafeMetrics
from sg_query_cache._model import (
    CacheHit,
    DependencyPlan,
    DependencyTag,
    ExactCall,
    FillLease,
    GenerationChange,
    Identity,
)
from sg_query_cache._valkey_control import ValkeyControl
from sg_query_cache._valkey_scripts import (
    CHANGE_SCRIPT_SHA,
    LOOKUP_SCRIPT_SHA,
    RESOLVE_SCRIPT_SHA,
    ScriptClient,
)
from sg_query_cache.valkey import ValkeyBackend, ValkeyMetadataStore, _CircuitBreaker
from tests._fakes import _DelegatingBackend, _FakeClock, _RecordingMetrics
from tests._valkey_control_cases import control_reply

_T = TypeVar('_T')

_CLIENT_OPTIONS = {'localized': False, 'convert_datetimes_to_utc': True}
_REPLICATION = 'a' * 40
_SENSITIVE_CANARIES = (
    'filter-private',
    'field-private',
    'identity-private',
    'https://private.example',
    'credential-private',
    'result-private',
    'signed-private',
    'raw-base-key-private',
    'generation-token-private',
    'flush-reason-private',
)
_SENSITIVE_IDENTITY = Identity('script', 'identity-private', 'credential-private')
_SENSITIVE_KEY = ':'.join(_SENSITIVE_CANARIES)
_SITE = 'production'


@dataclass(frozen=True, slots=True)
class _AgeDecision:
    remaining_seconds: float | None
    reason: str


class _ControlClient:
    def __init__(self) -> None:
        self._control: bytes | None = None
        self._counters: set[str] = set()
        self._fail_metadata = False
        self._metadata_replies: list[object] = []
        self._result_replies: list[object] = []

    def _control_reply(
        self,
        sha: str,
        numkeys: int,
        keys_and_args: tuple[object, ...],
    ) -> object:
        self._control, reply = control_reply(
            sha, numkeys, keys_and_args, self._control, self._counters
        )
        return reply

    def _lookup_reply(self, numkeys: int) -> object:
        assert numkeys == 1
        return self._result_replies.pop(0)

    def _metadata_reply(self, sha: str, numkeys: int) -> object:
        if self._fail_metadata:
            raise OSError('metadata unavailable')
        if self._metadata_replies:
            return self._metadata_replies.pop(0)
        if sha == CHANGE_SCRIPT_SHA:
            return [b'ok', numkeys - 2, 0]
        return [b'ok', *(1 for _index in range(numkeys - 2))]

    def close(self) -> None:
        return None

    def epoch(self) -> tuple[int, int, None]:
        return 0, 0, None

    def evalsha(self, sha: str, numkeys: int, *keys_and_args: object) -> object:
        if sha == LOOKUP_SCRIPT_SHA:
            return self._lookup_reply(numkeys)
        if sha in {CHANGE_SCRIPT_SHA, RESOLVE_SCRIPT_SHA}:
            return self._metadata_reply(sha, numkeys)
        return self._control_reply(sha, numkeys, keys_and_args)

    def exists(self, key: str) -> object:
        return int(key in self._counters)

    def get(self, key: str) -> object:
        _ = key
        return self._control

    def info(self, section: str) -> object:
        assert section == 'replication'
        return {'role': 'master', 'master_replid': _REPLICATION}

    @contextmanager
    def pin(self) -> Generator['_ControlClient']:
        yield self

    def refresh(self) -> None:
        return None

    def refresh_endpoint(self) -> None:
        return None

    def run_operation(self, operation: Callable[[ScriptClient], _T]) -> _T:
        return operation(self)

    def script_load(self, script: bytes | str) -> object:
        raise AssertionError(script)


class _FailingWaitBackend(_DelegatingBackend):
    _fail_lookup = False

    def lookup(self, key: str) -> CacheHit | None:
        if self._fail_lookup:
            raise BackendUnavailable('lookup failed during wait')
        return super().lookup(key)


class _Interrupted(BaseException):
    pass


class _ReserveBarrierBackend(_DelegatingBackend):
    def __init__(self, backend: LocalBackend) -> None:
        super().__init__(backend)
        self._reserve_barrier = Barrier(2)
        self._reserve_lock = Lock()
        self._reserves_completed = Event()
        self._reserve_attempts = 0
        self._reserve_count = 0

    def reserve(self, key: str, lease_seconds: float) -> FillLease | None:
        with self._reserve_lock:
            attempt = self._reserve_attempts
            self._reserve_attempts += 1
        if attempt < 2:
            self._reserve_barrier.wait(timeout=2)
        lease = self._backend.reserve(key, lease_seconds)
        with self._reserve_lock:
            self._reserve_count += 1
            if self._reserve_count == 2:
                self._reserves_completed.set()
        return lease


class _ThrowingMetrics:
    def increment(self, name: str, labels: object) -> None:
        raise RuntimeError(f'counter failed: {name}: {labels!r}')

    def observe(self, name: str, value: float, labels: object) -> None:
        raise RuntimeError(f'observation failed: {name}: {value}: {labels!r}')


def _advance_distributed_wait(
    backend: _FailingWaitBackend,
    lease: FillLease,
    outcome: str,
) -> None:
    if outcome == 'cancelled':
        raise _Interrupted
    if outcome == 'failure':
        backend._fail_lookup = True
        return
    assert backend.release(lease)
    if outcome == 'completed':
        replacement = backend.reserve(lease.key, 30)
        assert replacement is not None
        payload = encode_result(TimedResult(0.0, 1), 0.0, 30)
        assert backend.commit(replacement, payload, 30)


def _assert_metrics_private(*recordings: _RecordingMetrics) -> None:
    for recording in recordings:
        for name, labels in recording._increments:
            fields: tuple[object, ...] = (
                name,
                *labels.keys(),
                *labels.values(),
            )
            for field in fields:
                rendered = repr(field)
                assert all(canary not in rendered for canary in _SENSITIVE_CANARIES)
        for name, value, labels in recording._observations:
            fields = (
                name,
                value,
                *labels.keys(),
                *labels.values(),
            )
            for field in fields:
                rendered = repr(field)
                assert all(canary not in rendered for canary in _SENSITIVE_CANARIES)


def _await_follower(cache: QueryCache, *, prefix: str = '') -> None:
    deadline = monotonic() + 2
    while monotonic() < deadline:
        with cache._flight_lock:
            if any(
                key.startswith(prefix) and flight._participants == 2
                for key, flight in cache._flights.items()
            ):
                return
        Event().wait(0.001)
    raise AssertionError('local follower did not join the active fill')


def _call() -> ExactCall:
    return ExactCall(
        'find',
        (
            ('entity_type', 'Asset'),
            ('filters', (('code', 'is', 'filter-private'),)),
            ('fields', ('field-private',)),
            ('site_url', 'https://private.example'),
        ),
    )


def _metric_labels(recording: _RecordingMetrics, name: str) -> list[dict[str, str]]:
    return [
        {key: value for key, value in labels.items() if key != 'exact_call_digest'}
        for metric, labels in recording._increments
        if metric == name
    ]


def _recorded_names(metrics: _RecordingMetrics) -> set[str]:
    return {
        *(name for name, _labels in metrics._increments),
        *(name for name, _value, _labels in metrics._observations),
    }


def _wait_observations(recording: _RecordingMetrics) -> list[tuple[float, dict[str, str]]]:
    return [
        (value, {key: label for key, label in labels.items() if key != 'exact_call_digest'})
        for name, value, labels in recording._observations
        if name == 'cache_wait_seconds'
    ]


def _wait_settings(topology: str, *, wait_seconds: float = 1.0) -> CacheSettings:
    return CacheSettings(
        l1=L1Settings(enabled=topology != 'shared'),
        l2=L2Settings(enabled=topology != 'local', acknowledge_mutually_trusted_clients=True),
        limits=LimitSettings(lease_wait_seconds=wait_seconds),
    )


def _plan(*, broadening: bool = False) -> DependencyPlan:
    return DependencyPlan(
        'event_aware',
        (
            DependencyTag.site(_SITE),
            DependencyTag.authorization(_SITE),
            DependencyTag.type_membership(_SITE, 'Asset'),
        ),
        'event_aware',
        ('field_to_type',) if broadening else (),
    )


def _read(
    cache: QueryCache,
    fetch: Callable[[], object],
    *,
    plan: DependencyPlan | None = None,
    decision: Callable[[object], _AgeDecision] | None = None,
) -> object:
    return cache.read(
        _call(),
        plan or _plan(),
        fetch,
        _CLIENT_OPTIONS,
        _population_decision=decision,
    )


def test_cache_and_local_backend_emit_the_complete_observability_contract() -> None:
    clock = _FakeClock()
    recording = _RecordingMetrics()
    metrics = SafeMetrics(recording)
    settings = CacheSettings(
        limits=LimitSettings(scoped_generation_retention_seconds=1),
    )
    backend = LocalBackend(settings, clock, metrics)
    cache = QueryCache(
        settings,
        _SENSITIVE_IDENTITY,
        KeyBuilder(),
        l1=backend,
        metrics=metrics,
    )

    assert _read(
        cache,
        lambda: [{'id': 1, 'value': 'result-private', 'image': 'signed-private'}],
        plan=_plan(broadening=True),
        decision=lambda _value: _AgeDecision(5, 'signed_url'),
    ) == [{'id': 1, 'value': 'result-private', 'image': 'signed-private'}]
    assert _read(cache, lambda: [{'id': 2}]) == [
        {'id': 1, 'value': 'result-private', 'image': 'signed-private'}
    ]
    assert _read(
        cache,
        lambda: [{'id': 3}],
        plan=DependencyPlan('bypass', (), 'unknown_operator'),
    ) == [{'id': 3}]

    scoped = DependencyTag.entity(_SITE, 'Asset', 7)
    backend.resolve((scoped,))
    clock._advance(2)
    backend.resolve((scoped,))
    backend.change((GenerationChange(DependencyTag.entity(_SITE, 'Asset', 8)),))
    backend.change((GenerationChange(scoped),))
    backend.flush_site(_SITE, 'operator')

    first = backend.reserve(_SENSITIVE_KEY, 1)
    assert first is not None
    clock._advance(2)
    second = backend.reserve(_SENSITIVE_KEY, 1)
    assert second is not None
    assert backend.commit(second, b'\x80', 1)
    assert not backend.commit(first, b'\x80', 1)

    required = {
        'cache_bypass',
        'cache_commit',
        'cache_dependency_broadening',
        'cache_dependency_vector_bytes',
        'cache_dependency_vector_tags',
        'cache_fill',
        'cache_hit',
        'cache_invalidation',
        'cache_lease_rejection',
        'cache_miss',
        'cache_query_classification',
        'cache_scoped_generation',
        'cache_serialized_bytes',
        'cache_sg_avoided_calls',
        'cache_sg_avoided_seconds',
        'cache_site_flush',
        'cache_ttl_selection',
        'cache_url_age_cap',
        'cache_vector_resolution',
        'generation_metadata_bytes',
    }
    assert required <= _recorded_names(recording), required - _recorded_names(recording)
    _assert_metrics_private(recording)


def test_circuit_breaker_transitions_ignore_metrics_sink_failures() -> None:
    clock = _FakeClock()
    breaker = _CircuitBreaker(
        'result',
        threshold=1,
        probe_seconds=5,
        metrics=_ThrowingMetrics(),
        clock=clock,
    )

    breaker.fail()
    with pytest.raises(BackendUnavailable, match='circuit breaker is open'):
        breaker.admit()

    clock._advance(5)
    assert breaker.admit()
    breaker.fail()
    with pytest.raises(BackendUnavailable, match='circuit breaker is open'):
        breaker.admit()

    clock._advance(5)
    assert breaker.admit()
    breaker.succeed()
    assert not breaker.admit()

    recording = _RecordingMetrics()
    recorded = _CircuitBreaker(
        'result',
        threshold=1,
        probe_seconds=5,
        metrics=SafeMetrics(recording),
        clock=clock,
    )
    recorded.fail()
    clock._advance(5)
    assert recorded.admit()
    recorded.fail()
    clock._advance(5)
    assert recorded.admit()
    recorded.succeed()
    _assert_metrics_private(recording)


def test_flush_metrics_use_only_fixed_internal_reason_labels() -> None:
    for owner in ('local', 'valkey'):
        recording = _RecordingMetrics()
        metrics = SafeMetrics(recording)
        settings = CacheSettings(
            l1=L1Settings(enabled=owner == 'local'),
            l2=L2Settings(
                enabled=owner == 'valkey',
                acknowledge_mutually_trusted_clients=owner == 'valkey',
            ),
        )
        backend = (
            LocalBackend(settings, _FakeClock(), metrics)
            if owner == 'local'
            else ValkeyMetadataStore(_ControlClient(), settings, metrics)
        )

        backend.flush_site(_SITE, 'customer-ticket-private')
        backend.flush_site(_SITE, 'daemon_setup')

        assert (
            'cache_site_flush',
            {'reason': 'operator'},
        ) in recording._increments
        assert (
            'cache_site_flush',
            {'reason': 'daemon_setup'},
        ) in recording._increments
        assert 'customer-ticket-private' not in repr(recording._increments)
        _assert_metrics_private(recording)


@pytest.mark.parametrize('topology', ['local', 'hierarchy', 'shared'])
def test_followers_count_one_collapse_and_wait(topology: str) -> None:
    settings = _wait_settings(topology)
    recording = _RecordingMetrics()
    metrics = SafeMetrics(recording)
    backend = LocalBackend(settings, _FakeClock(), metrics)
    selected_backend = _ReserveBarrierBackend(backend) if topology == 'shared' else backend
    cache = QueryCache(
        settings,
        _SENSITIVE_IDENTITY,
        KeyBuilder(),
        l1=LocalBackend(settings, _FakeClock(), SafeMetrics(recording))
        if topology == 'hierarchy'
        else (backend if topology == 'local' else None),
        l2=selected_backend if topology != 'local' else None,
        metrics=metrics,
    )
    rendezvous = Barrier(3)
    fetch_entered = Event()
    release_fetch = Event()

    def fetch() -> list[dict[str, int]]:
        fetch_entered.set()
        assert release_fetch.wait(timeout=2)
        return [{'id': 1}]

    def read() -> object:
        rendezvous.wait(timeout=2)
        return _read(cache, fetch)

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(read) for _ in range(2)]
        try:
            rendezvous.wait(timeout=2)
            assert fetch_entered.wait(timeout=2)
            if topology == 'shared':
                assert cast('_ReserveBarrierBackend', selected_backend)._reserves_completed.wait(
                    timeout=2
                )
            else:
                _await_follower(cache)
        finally:
            release_fetch.set()
        assert [future.result(timeout=3) for future in futures] == [[{'id': 1}]] * 2

    category = 'distributed' if topology == 'shared' else 'local'
    assert _metric_labels(recording, 'cache_collapse') == [{'outcome': category}]
    assert _wait_observations(recording)[0][1] == {'outcome': 'completed'}
    assert len(_wait_observations(recording)) == 1
    _assert_metrics_private(recording)


@pytest.mark.parametrize('topology', ['shared', 'hierarchy'])
@pytest.mark.parametrize('outcome', ['completed', 'takeover', 'timeout', 'failure', 'cancelled'])
def test_distributed_wait_records_one_outcome(
    monkeypatch: pytest.MonkeyPatch,
    topology: str,
    outcome: str,
) -> None:
    settings = _wait_settings(topology, wait_seconds=0.02)
    recording = _RecordingMetrics()
    backend = _FailingWaitBackend(LocalBackend(settings, _FakeClock(), SafeMetrics(recording)))
    cache = QueryCache(
        settings,
        _SENSITIVE_IDENTITY,
        l1=LocalBackend(settings, _FakeClock(), SafeMetrics(recording))
        if topology == 'hierarchy'
        else None,
        l2=backend,
        metrics=SafeMetrics(recording),
    )
    base_key = KeyBuilder.base_key(
        settings.site_namespace, _SENSITIVE_IDENTITY, _CLIENT_OPTIONS, _call()
    )
    key = KeyBuilder.qualified_key(base_key, backend.resolve(_plan().tags), 'result')
    lease = backend.reserve(key, 30)
    assert lease is not None

    def advance(_seconds: float) -> None:
        _advance_distributed_wait(backend, lease, outcome)

    def fetch() -> object:
        if outcome == 'takeover':
            sleep(0.2)
        return 2

    if outcome != 'timeout':
        monkeypatch.setattr(cache_module, 'sleep', advance)
    try:
        if outcome == 'cancelled':
            with pytest.raises(_Interrupted):
                _read(cache, fetch)
        else:
            assert _read(cache, fetch) == (1 if outcome == 'completed' else 2)
    finally:
        backend.release(lease)
    assert _metric_labels(recording, 'cache_collapse') == [{'outcome': 'distributed'}]
    observations = _wait_observations(recording)
    assert len(observations) == 1
    assert observations[0][1] == {'outcome': outcome}
    assert _metric_labels(recording, 'cache_lease_timeout') == (
        [{'outcome': 'distributed'}] if outcome == 'timeout' else []
    )
    if outcome == 'completed':
        assert {'layer': 'l2'} not in _metric_labels(recording, 'cache_fill')
    if outcome == 'takeover':
        assert _metric_labels(recording, 'cache_fill').count({'layer': 'l2'}) == 1
        assert observations[0][0] < 0.2
    if outcome == 'failure':
        assert {'reason': 'backend_failure'} in _metric_labels(recording, 'cache_bypass')


def test_hierarchy_promotion_preserves_observed_sg_duration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = _FakeClock()
    monkeypatch.setattr(cache_module, 'monotonic', clock)
    recording = _RecordingMetrics()
    metrics = SafeMetrics(recording)
    shared_settings = CacheSettings(
        l1=L1Settings(enabled=False),
        l2=L2Settings(enabled=True, acknowledge_mutually_trusted_clients=True),
    )
    l2 = LocalBackend(shared_settings, clock, metrics)
    shared = QueryCache(
        shared_settings,
        _SENSITIVE_IDENTITY,
        KeyBuilder(),
        l2=l2,
        metrics=metrics,
    )

    def fetch() -> str:
        clock._advance(3.25)
        return 'shared'

    assert _read(shared, fetch) == 'shared'

    hierarchy_settings = CacheSettings(
        l2=L2Settings(enabled=True, acknowledge_mutually_trusted_clients=True),
    )
    l1 = LocalBackend(hierarchy_settings, clock, metrics)
    hierarchy = QueryCache(
        hierarchy_settings,
        _SENSITIVE_IDENTITY,
        KeyBuilder(),
        l1=l1,
        l2=l2,
        metrics=metrics,
    )
    assert _read(hierarchy, lambda: 'unexpected') == 'shared'
    assert _read(hierarchy, lambda: 'unexpected') == 'shared'

    assert (
        'cache_sg_avoided_seconds',
        3.25,
        {'layer': 'l2'},
    ) in recording._observations
    assert (
        'cache_sg_avoided_seconds',
        3.25,
        {'layer': 'hierarchical_l1'},
    ) in recording._observations
    _assert_metrics_private(recording)


def test_incarnation_validation_and_rotation_emit_stable_outcomes() -> None:
    recording = _RecordingMetrics()
    metrics = SafeMetrics(recording)
    client = _ControlClient()
    control = ValkeyControl(
        client,
        key_prefix='sgqc',
        validation_seconds=30,
        jitter_factory=lambda seconds: seconds,
        nonce_factory=iter(('1' * 32, '2' * 32)).__next__,
        metrics=metrics,
    )

    first = control.validate()
    second = control.rotate(first)

    assert first != second
    assert ('cache_incarnation', {'outcome': 'validated'}) in recording._increments
    assert ('cache_incarnation', {'outcome': 'rotated'}) in recording._increments
    _assert_metrics_private(recording)


def test_local_owner_reports_exact_storage_transitions() -> None:
    clock = _FakeClock()
    recording = _RecordingMetrics()
    metrics = SafeMetrics(recording)
    settings = CacheSettings(
        l1=L1Settings(result_budget_bytes=500),
        limits=LimitSettings(scoped_generation_retention_seconds=1),
    )
    backend = LocalBackend(settings, clock, metrics)
    scoped = DependencyTag.entity(_SITE, 'Asset', 99)

    backend.resolve((scoped,))
    first = backend.reserve(f'{_SENSITIVE_KEY}:first', 5)
    second = backend.reserve(f'{_SENSITIVE_KEY}:second', 5)
    assert first is not None
    assert second is not None
    assert backend.commit(first, encode_value(('first',)), 30)
    assert backend.commit(second, encode_value(('second',)), 30)
    clock._advance(2)
    backend.resolve((scoped,))

    assert (
        'cache_scoped_generation',
        {'outcome': 'created', 'tag_kind': 'entity'},
    ) in recording._increments
    assert (
        'cache_scoped_generation',
        {'outcome': 'expired', 'tag_kind': 'entity'},
    ) in recording._increments
    assert any(
        name == 'cache_serialized_bytes'
        and labels
        == {
            'layer': 'local_l1',
            'outcome': 'evicted',
        }
        for name, _value, labels in recording._observations
    )
    assert {
        labels['tag_kind']
        for name, _value, labels in recording._observations
        if name == 'generation_metadata_bytes'
    } == {'global', 'scoped'}
    _assert_metrics_private(recording)


def test_metric_payloads_never_disclose_sensitive_runtime_values() -> None:
    recording = _RecordingMetrics()
    metrics = SafeMetrics(recording)
    settings = CacheSettings()
    backend = LocalBackend(settings, _FakeClock(), metrics)
    cache = QueryCache(
        settings,
        Identity('script', 'identity-private', 'credential-private'),
        KeyBuilder(),
        l1=backend,
        metrics=metrics,
    )
    call = ExactCall(
        'find',
        (
            ('entity_type', 'Asset'),
            ('filters', (('code', 'is', 'filter-private'),)),
            ('fields', ('field-private',)),
            ('site_url', 'https://private.example'),
        ),
    )

    cache.read(
        call,
        _plan(),
        lambda: {'value': 'result-private', 'url': 'https://signed-private.example'},
        _CLIENT_OPTIONS,
    )
    lease = backend.reserve(_SENSITIVE_KEY, 5)
    assert lease is not None
    backend.release(lease)
    backend.flush_site(_SITE, 'flush-reason-private')

    _assert_metrics_private(recording)
    for _name, labels in recording._increments:
        assert set(labels) <= {
            'backend_operation',
            'classification',
            'exact_call_digest',
            'layer',
            'method',
            'outcome',
            'reason',
            'tag_kind',
        }
    for _name, value, labels in recording._observations:
        assert value >= 0
        assert set(labels) <= {
            'backend_operation',
            'classification',
            'exact_call_digest',
            'layer',
            'method',
            'outcome',
            'reason',
            'tag_kind',
        }


def test_parsed_follower_counts_one_local_collapse() -> None:
    recording = _RecordingMetrics()
    settings = CacheSettings()
    cache = QueryCache(
        settings,
        _SENSITIVE_IDENTITY,
        l1=LocalBackend(settings, _FakeClock(), SafeMetrics(recording)),
        metrics=SafeMetrics(recording),
    )
    entered = Event()
    release = Event()

    def fetch() -> object:
        entered.set()
        assert release.wait(timeout=2)
        return 1

    def parse(value: object) -> object:
        return value

    def read() -> object:
        return cache.read_parsed(_call(), _plan(), fetch, _CLIENT_OPTIONS, parse)

    with ThreadPoolExecutor(max_workers=2) as executor:
        leader = executor.submit(read)
        try:
            assert entered.wait(timeout=2)
            follower = executor.submit(read)
            _await_follower(cache, prefix='memo:')
        finally:
            release.set()
        assert leader.result(timeout=3) == follower.result(timeout=3) == 1
    assert _metric_labels(recording, 'cache_collapse') == [{'outcome': 'local'}]
    assert len(_wait_observations(recording)) == 1
    assert _wait_observations(recording)[0][1] == {'outcome': 'completed'}


def test_parsed_wait_timeout_can_reuse_cached_result() -> None:
    recording = _RecordingMetrics()
    settings = CacheSettings(limits=LimitSettings(lease_wait_seconds=0.02))
    cache = QueryCache(
        settings,
        _SENSITIVE_IDENTITY,
        l1=LocalBackend(settings, _FakeClock(), SafeMetrics(recording)),
        metrics=SafeMetrics(recording),
    )
    entered = Event()
    release = Event()
    fetch_calls = 0
    parse_calls = 0

    def fetch() -> object:
        nonlocal fetch_calls
        fetch_calls += 1
        return 1

    def parse(value: object) -> object:
        nonlocal parse_calls
        parse_calls += 1
        if parse_calls == 1:
            entered.set()
            assert release.wait(timeout=2)
        return value

    def read() -> object:
        return cache.read_parsed(_call(), _plan(), fetch, _CLIENT_OPTIONS, parse)

    with ThreadPoolExecutor(max_workers=2) as executor:
        leader = executor.submit(read)
        try:
            assert entered.wait(timeout=2)
            follower = executor.submit(read)
            assert follower.result(timeout=1) == 1
        finally:
            release.set()
        assert leader.result(timeout=3) == 1
    assert fetch_calls == 1
    assert _metric_labels(recording, 'cache_lease_timeout') == [{'outcome': 'local'}]
    assert _metric_labels(recording, 'cache_hit') == [{'layer': 'local_l1'}]


def test_raw_backend_hit_does_not_claim_an_avoided_sg_call() -> None:
    clock = _FakeClock()
    recording = _RecordingMetrics()
    metrics = SafeMetrics(recording)
    settings = CacheSettings()
    backend = LocalBackend(settings, clock, metrics)
    lease = backend.reserve(_SENSITIVE_KEY, 5)
    assert lease is not None
    assert backend.commit(lease, encode_value(('raw',)), 30)
    recording._increments.clear()
    recording._observations.clear()

    assert backend.lookup(_SENSITIVE_KEY) is not None
    assert not any(
        name in {'cache_hit', 'cache_sg_avoided_calls'} for name, _labels in recording._increments
    )
    assert not any(
        name == 'cache_sg_avoided_seconds' for name, _value, _labels in recording._observations
    )
    _assert_metrics_private(recording)


def test_raw_valkey_hit_does_not_claim_an_avoided_sg_call() -> None:
    recording = _RecordingMetrics()
    metrics = SafeMetrics(recording)
    settings = CacheSettings(
        l1=L1Settings(enabled=False),
        l2=L2Settings(enabled=True, acknowledge_mutually_trusted_clients=True),
    )
    client = _ControlClient()
    store = ValkeyMetadataStore(client, settings, metrics)
    backend = ValkeyBackend(client, store, settings, metrics)
    client._result_replies.append([b'hit', encode_value(('raw',)), 1_000])

    assert backend.lookup(_SENSITIVE_KEY) is not None
    assert not any(
        name in {'cache_hit', 'cache_sg_avoided_calls'} for name, _labels in recording._increments
    )
    assert not any(
        name == 'cache_sg_avoided_seconds' for name, _value, _labels in recording._observations
    )
    _assert_metrics_private(recording)


def test_request_counts_local_waits_once_across_memo_and_result_flights() -> None:
    recording = _RecordingMetrics()
    settings = CacheSettings(limits=LimitSettings(lease_wait_seconds=0.05))
    cache = QueryCache(
        settings,
        _SENSITIVE_IDENTITY,
        l1=LocalBackend(settings, _FakeClock(), SafeMetrics(recording)),
        metrics=SafeMetrics(recording),
    )
    entered = Event()
    release = Event()
    calls = 0

    def fetch() -> object:
        nonlocal calls
        calls += 1
        if calls == 1:
            entered.set()
            assert release.wait(timeout=2)
        return 1

    def parse(value: object) -> object:
        return value

    def read() -> object:
        return cache.read_parsed(_call(), _plan(), fetch, _CLIENT_OPTIONS, parse)

    with ThreadPoolExecutor(max_workers=2) as executor:
        leader = executor.submit(read)
        try:
            assert entered.wait(timeout=2)
            follower = executor.submit(read)
            assert follower.result(timeout=1) == 1
        finally:
            release.set()
        assert leader.result(timeout=3) == 1
    assert _metric_labels(recording, 'cache_collapse') == [{'outcome': 'local'}]
    assert _metric_labels(recording, 'cache_lease_timeout') == [{'outcome': 'local'}]
    observations = _wait_observations(recording)
    assert len(observations) == 1
    assert observations[0][1] == {'outcome': 'timeout'}
    assert observations[0][0] >= 0.08


def test_successful_hit_reports_exact_observed_sg_duration_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = _FakeClock()
    monkeypatch.setattr(cache_module, 'monotonic', clock)
    recording = _RecordingMetrics()
    metrics = SafeMetrics(recording)
    settings = CacheSettings()
    backend = LocalBackend(settings, clock, metrics)
    cache = QueryCache(
        settings,
        _SENSITIVE_IDENTITY,
        KeyBuilder(),
        l1=backend,
        metrics=metrics,
    )

    def fetch() -> str:
        clock._advance(2.5)
        return 'fresh'

    assert _read(cache, fetch) == 'fresh'
    assert _read(cache, lambda: 'unexpected') == 'fresh'

    assert recording._increments.count(('cache_sg_avoided_calls', {'layer': 'local_l1'})) == 1
    assert (
        recording._observations.count(('cache_sg_avoided_seconds', 2.5, {'layer': 'local_l1'}))
        == 1
    )
    _assert_metrics_private(recording)


@pytest.mark.parametrize(
    'envelope',
    [
        ('sg-query-cache-result-v1', 300.0, 'malformed'),
        ('sg-query-cache-result-v1', 0.0, 300.0, -1.0, 'malformed'),
    ],
    ids=['short-envelope', 'malformed-duration'],
)
def test_unsafe_envelope_falls_back_without_reporting_an_avoided_call(
    envelope: tuple[object, ...],
) -> None:
    clock = _FakeClock()
    recording = _RecordingMetrics()
    metrics = SafeMetrics(recording)
    settings = CacheSettings()
    backend = LocalBackend(settings, clock, metrics)
    builder = KeyBuilder()
    identity = _SENSITIVE_IDENTITY
    cache = QueryCache(settings, identity, builder, l1=backend, metrics=metrics)
    vector = backend.resolve(_plan().tags)
    base_key = builder.base_key(_SITE, identity, _CLIENT_OPTIONS, _call())
    key = builder.qualified_key(base_key, vector, 'result')
    lease = backend.reserve(key, 5)
    assert lease is not None
    assert backend.commit(
        lease,
        encode_value(envelope),
        300,
    )

    assert _read(cache, lambda: 'fresh') == 'fresh'
    assert not any(name.startswith('cache_sg_avoided') for name in _recorded_names(recording))
    _assert_metrics_private(recording)


def test_valkey_backend_phases_noeviction_and_recovery_are_observable() -> None:
    recording = _RecordingMetrics()
    metrics = SafeMetrics(recording)
    settings = CacheSettings(
        l1=L1Settings(enabled=False),
        l2=L2Settings(enabled=True, acknowledge_mutually_trusted_clients=True),
    )
    client = _ControlClient()
    client._metadata_replies.extend(([b'capacity'], [b'ok', 1]))
    store = ValkeyMetadataStore(client, settings, metrics)
    tag = DependencyTag.type_membership(_SITE, 'Asset')

    with pytest.raises(BackendCapacity):
        store.resolve((tag,))
    assert store.resolve((tag,)).entries[0][0] == tag
    client._fail_metadata = True
    with pytest.raises(BackendUnavailable):
        store.resolve((tag,))

    names = _recorded_names(recording)
    assert {
        'backend_failure',
        'backend_latency_seconds',
        'cache_metadata_recovery_seconds',
        'cache_metadata_write_rejection',
    } <= names
    _assert_metrics_private(recording)


def test_valkey_lookup_reports_observable_miss_reasons_without_guessing() -> None:
    recording = _RecordingMetrics()
    metrics = SafeMetrics(recording)
    settings = CacheSettings(
        l1=L1Settings(enabled=False),
        l2=L2Settings(enabled=True, acknowledge_mutually_trusted_clients=True),
        limits=LimitSettings(max_serialized_result_bytes=8),
    )
    client = _ControlClient()
    store = ValkeyMetadataStore(client, settings, metrics)
    backend = ValkeyBackend(client, store, settings, metrics)
    client._result_replies.extend(
        (
            [b'miss'],
            [b'hit', b'\xc1', 1_000],
            [b'hit', encode_value(('oversized',)), 1_000],
        )
    )

    assert backend.lookup(_SENSITIVE_KEY) is None
    assert backend.lookup(_SENSITIVE_KEY) is None
    assert backend.lookup(_SENSITIVE_KEY) is None

    reasons = {labels['reason'] for name, labels in recording._increments if name == 'cache_miss'}
    assert reasons == {'absence', 'malformed', 'oversized'}
    _assert_metrics_private(recording)
