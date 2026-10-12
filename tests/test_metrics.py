from collections.abc import Callable, Mapping
from dataclasses import dataclass

import pytest

from sg_query_cache import MetricsSink, QueryCache
from sg_query_cache._config import CacheSettings
from sg_query_cache._keys import KeyBuilder
from sg_query_cache._local import LocalBackend
from sg_query_cache._metrics import NullMetrics, SafeMetrics
from sg_query_cache._model import DependencyPlan, DependencyTag, ExactCall, Identity
from tests._fakes import _FakeClock, _RecordingMetrics

_CLIENT_OPTIONS = {'localized': False, 'convert_datetimes_to_utc': True}
_CORRELATION_ERROR = 'configured metrics correlation secret is unavailable or invalid'
_ENV_NAME = 'SGQC_TEST_METRICS_SECRET'
_IDENTITY = Identity('script', 'studio', 'render')
_METRICS_KEY_TEXT = 'm' * 32
_PLAN = DependencyPlan(
    'event_aware',
    (
        DependencyTag.site('production'),
        DependencyTag.authorization('production'),
        DependencyTag.type_membership('production', 'Asset'),
    ),
    'entity_query',
)


@dataclass(frozen=True, slots=True)
class _DeclinePopulation:
    remaining_seconds: None = None
    reason: str = 'test_decline'


class _ThrowingMetrics:
    def increment(self, name: str, labels: Mapping[str, str]) -> None:
        _ = name, labels
        raise RuntimeError('metrics unavailable')

    def observe(self, name: str, value: float, labels: Mapping[str, str]) -> None:
        _ = name, value, labels
        raise RuntimeError('metrics unavailable')


def _correlated_cache(
    settings: CacheSettings,
    environ: Mapping[str, object],
    sink: MetricsSink,
    *,
    backend: LocalBackend | None = None,
) -> tuple[QueryCache, LocalBackend]:
    metrics = QueryCache.build_metrics(settings, environ, sink)
    selected_backend = backend or LocalBackend(settings, _FakeClock(), NullMetrics())
    return (
        QueryCache(settings, _IDENTITY, l1=selected_backend, metrics=metrics),
        selected_backend,
    )


def _read(cache: QueryCache, call: ExactCall, fetch: Callable[[], object]) -> object:
    return cache.read(call, _PLAN, fetch, _CLIENT_OPTIONS)


def test_backend_metrics_outside_query_cache_have_no_exact_call_digest() -> None:
    settings = CacheSettings(metrics_digest_secret_env=_ENV_NAME)
    recording = _RecordingMetrics()
    metrics = QueryCache.build_metrics(
        settings,
        {_ENV_NAME: _METRICS_KEY_TEXT},
        recording,
    )
    backend = LocalBackend(settings, _FakeClock(), metrics)

    backend.resolve(_PLAN.tags)

    labels = [event_labels for _name, event_labels in recording._increments] + [
        event_labels for _name, _value, event_labels in recording._observations
    ]
    assert labels
    assert all('exact_call_digest' not in event_labels for event_labels in labels)


def test_configured_metrics_correlate_one_exact_miss_and_hit() -> None:
    settings = CacheSettings(metrics_digest_secret_env=_ENV_NAME)
    recording = _RecordingMetrics()
    cache, _backend = _correlated_cache(
        settings,
        {_ENV_NAME: _METRICS_KEY_TEXT},
        recording,
    )
    call = ExactCall('find', (('entity_type', 'Asset'), ('filters', ('oak-tree',))))
    fetches = 0

    def fetch() -> object:
        nonlocal fetches
        fetches += 1
        return [{'id': 7}]

    assert _read(cache, call, fetch) == [{'id': 7}]
    assert _read(cache, call, fetch) == [{'id': 7}]

    base_key = KeyBuilder.base_key('production', _IDENTITY, _CLIENT_OPTIONS, call)
    expected = KeyBuilder.metrics_digest(base_key, _METRICS_KEY_TEXT.encode())
    exact_events = [
        (name, labels)
        for name, labels in recording._increments
        if name in {'cache_fill', 'cache_hit'}
    ]
    assert fetches == 1
    assert {name for name, _labels in exact_events} == {'cache_fill', 'cache_hit'}
    assert all(labels['exact_call_digest'] == expected for _name, labels in exact_events)
    emitted = repr((recording._increments, recording._observations))
    assert _METRICS_KEY_TEXT not in emitted
    assert _ENV_NAME not in emitted
    assert 'oak-tree' not in emitted


def test_correlated_throwing_metrics_remain_best_effort() -> None:
    settings = CacheSettings(metrics_digest_secret_env=_ENV_NAME)
    cache, _backend = _correlated_cache(
        settings,
        {_ENV_NAME: _METRICS_KEY_TEXT},
        _ThrowingMetrics(),
    )
    call = ExactCall('find', (('entity_type', 'Asset'), ('filters', ())))

    assert _read(cache, call, lambda: [{'id': 7}]) == [{'id': 7}]


def test_disabled_metrics_correlation_omits_the_exact_call_label() -> None:
    settings = CacheSettings()
    recording = _RecordingMetrics()
    cache, _backend = _correlated_cache(settings, {_ENV_NAME: 'unused-secret'}, recording)
    call = ExactCall('find', (('entity_type', 'Asset'), ('filters', ())))

    _read(cache, call, lambda: [{'id': 7}])
    _read(cache, call, lambda: pytest.fail('cache hit refetched'))

    events = [
        labels for name, labels in recording._increments if name in {'cache_fill', 'cache_hit'}
    ]
    assert events
    assert all('exact_call_digest' not in labels for labels in events)


@pytest.mark.parametrize(
    'environ',
    [{}, {_ENV_NAME: 'too-short'}, {_ENV_NAME: 42}],
)
def test_metrics_construction_rejects_unusable_secrets_without_disclosure(
    environ: Mapping[str, object],
) -> None:
    settings = CacheSettings(metrics_digest_secret_env=_ENV_NAME)

    with pytest.raises(ValueError, match=_CORRELATION_ERROR) as captured:
        QueryCache.build_metrics(settings, environ, _RecordingMetrics())

    message = str(captured.value)
    assert message == _CORRELATION_ERROR
    assert _ENV_NAME not in message
    assert 'too-short' not in message


def test_metrics_keys_do_not_change_backend_keys_or_split_results() -> None:
    settings = CacheSettings(metrics_digest_secret_env=_ENV_NAME)
    backend = LocalBackend(settings, _FakeClock(), NullMetrics())
    first, _backend = _correlated_cache(
        settings,
        {_ENV_NAME: 'a' * 32},
        _RecordingMetrics(),
        backend=backend,
    )
    second, _backend = _correlated_cache(
        settings,
        {_ENV_NAME: 'b' * 32},
        _RecordingMetrics(),
        backend=backend,
    )
    call = ExactCall('find', (('entity_type', 'Asset'), ('filters', ())))

    assert _read(first, call, lambda: [{'id': 7}]) == [{'id': 7}]
    keys_after_fill = tuple(backend._store._results)
    assert _read(second, call, lambda: pytest.fail('metrics key split cache')) == [{'id': 7}]

    assert tuple(backend._store._results) == keys_after_fill


def test_nested_reads_restore_the_enclosing_exact_call_digest() -> None:
    settings = CacheSettings(metrics_digest_secret_env=_ENV_NAME)
    recording = _RecordingMetrics()
    cache, _backend = _correlated_cache(
        settings,
        {_ENV_NAME: _METRICS_KEY_TEXT},
        recording,
    )
    outer = ExactCall('find', (('entity_type', 'Asset'), ('filters', ('outer',))))
    inner = ExactCall('find', (('entity_type', 'Asset'), ('filters', ('inner',))))

    def fetch_outer() -> object:
        assert _read(cache, inner, lambda: [{'id': 8}]) == [{'id': 8}]
        return [{'id': 7}]

    assert cache.read(
        outer,
        _PLAN,
        fetch_outer,
        _CLIENT_OPTIONS,
        _population_decision=lambda _value: _DeclinePopulation(),
    ) == [{'id': 7}]

    outer_base = KeyBuilder.base_key('production', _IDENTITY, _CLIENT_OPTIONS, outer)
    inner_base = KeyBuilder.base_key('production', _IDENTITY, _CLIENT_OPTIONS, inner)
    outer_digest = KeyBuilder.metrics_digest(outer_base, _METRICS_KEY_TEXT.encode())
    inner_digest = KeyBuilder.metrics_digest(inner_base, _METRICS_KEY_TEXT.encode())
    fill_digests = [
        labels['exact_call_digest']
        for name, labels in recording._increments
        if name == 'cache_fill'
    ]
    declined = [
        labels['exact_call_digest']
        for name, labels in recording._increments
        if name == 'cache_bypass' and labels.get('reason') == 'population_declined'
    ]
    assert fill_digests == [outer_digest, inner_digest]
    assert declined == [outer_digest]


def test_null_metrics_discards_valid_events() -> None:
    metrics = NullMetrics()

    assert metrics.increment('cache_hit', {'layer': 'l1'}) is None
    assert metrics.observe('backend_latency_seconds', 0.1, {'backend_operation': 'lookup'}) is None


def test_safe_metrics_forwards_documented_names_and_labels() -> None:
    recording = _RecordingMetrics()
    metrics = SafeMetrics(recording)

    metrics.increment('cache_hit', {'layer': 'l2', 'method': 'find'})
    metrics.observe(
        'backend_latency_seconds',
        0.25,
        {'backend_operation': 'lookup', 'outcome': 'success'},
    )

    assert recording._increments == [('cache_hit', {'layer': 'l2', 'method': 'find'})]
    assert recording._observations == [
        (
            'backend_latency_seconds',
            0.25,
            {'backend_operation': 'lookup', 'outcome': 'success'},
        )
    ]


def test_safe_metrics_forwards_only_a_keyed_exact_call_digest() -> None:
    recording = _RecordingMetrics()
    metrics = SafeMetrics(recording, metrics_key=b'k' * 32)

    metrics.increment(
        'cache_hit',
        {'layer': 'l1', 'exact_call_digest': 'a' * 64},
    )

    assert recording._increments == [('cache_hit', {'layer': 'l1', 'exact_call_digest': 'a' * 64})]


def test_safe_metrics_omits_exact_call_digest_without_metrics_key() -> None:
    recording = _RecordingMetrics()
    metrics = SafeMetrics(recording)

    metrics.increment(
        'cache_hit',
        {'layer': 'l1', 'exact_call_digest': 'a' * 64},
    )

    assert recording._increments == [('cache_hit', {'layer': 'l1'})]


def test_safe_metrics_rejects_short_metrics_keys_without_revealing_them() -> None:
    secret = b'too-short'

    with pytest.raises(ValueError, match='at least 32 bytes') as error:
        SafeMetrics(_RecordingMetrics(), metrics_key=secret)

    assert secret.decode() not in str(error.value)


@pytest.mark.parametrize('name', ['custom_metric', '', 'cache-hit'])
def test_safe_metrics_rejects_undocumented_metric_names(name: str) -> None:
    with pytest.raises(ValueError, match='metric name'):
        SafeMetrics(_RecordingMetrics()).increment(name, {'layer': 'l1'})


@pytest.mark.parametrize(
    'label', ['filters', 'fields', 'query', 'url', 'credential', 'payload', 'custom']
)
def test_safe_metrics_rejects_unsafe_or_unknown_labels(label: str) -> None:
    with pytest.raises(ValueError, match='metric label'):
        SafeMetrics(_RecordingMetrics()).increment('cache_hit', {label: 'secret-value'})


@pytest.mark.parametrize('digest', ['raw-query', 'A' * 64, 'a' * 63])
def test_safe_metrics_rejects_non_hmac_correlation_values(digest: str) -> None:
    metrics = SafeMetrics(_RecordingMetrics(), metrics_key=b'k' * 32)

    with pytest.raises(ValueError, match='exact-call digest'):
        metrics.increment('cache_hit', {'exact_call_digest': digest})
