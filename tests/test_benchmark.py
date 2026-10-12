from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event
from time import perf_counter
from typing import TYPE_CHECKING, Protocol, cast

import pytest

import sg_query_cache._async_cache as async_cache_module
import sg_query_cache._cache as cache_module
from sg_query_cache import AsyncQueryCache
from tests._benchmark_loader import load_benchmark

_BENCHMARK = Path('benchmarks/query_cache.py')
if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Mapping

    import benchmarks.query_cache as _typed_module

    _MODULE = _typed_module
else:
    _MODULE = load_benchmark(_BENCHMARK, 'sg_query_cache_benchmark')
_REQUIRED_FIELDS = {
    'cache_operations',
    'complete_hit_latency_ms',
    'correctness_violations',
    'lease_rejections',
    'memory_estimates_bytes',
    'normal_hit_backend_phases',
    'operations',
    'payload_bytes',
    'profile',
    'scenario',
    'seed',
    'sg_calls_avoided',
    'sg_latency_model',
    'tag_count',
    'throughput_ops_per_second',
    'vector_resolution_latency_ms',
    'workers',
}
_SCENARIOS = _MODULE._SCENARIOS
_build_workload = _MODULE._build_workload
_parse_args = _MODULE._parse_args
_settings = _MODULE._settings
_BenchmarkOptions = _MODULE._BenchmarkOptions
_main = _MODULE.main
_run_benchmark = _MODULE._run_benchmark


class _AsyncRawClient(Protocol):
    async def update(self, *args: object, **kwargs: object) -> object: ...


class _AsyncBenchmarkModule(Protocol):
    _RawClient: type[_AsyncRawClient]
    _measure: Callable[[Callable[[], Awaitable[object]], int], Awaitable[object]]
    _mixed: Callable[[int], Awaitable[list[dict[str, object]]]]


def _load_async_benchmark() -> _AsyncBenchmarkModule:
    module = load_benchmark(Path('benchmarks/async_query_cache.py'), 'async_benchmark')
    return cast('_AsyncBenchmarkModule', module)


def _run(*arguments: str) -> subprocess.CompletedProcess[str]:
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith('SG_QUERY_CACHE_REQUIRE_')
        and not key.startswith('SG_QUERY_CACHE_TEST_')
    }
    return subprocess.run(
        [sys.executable, str(_BENCHMARK), *arguments],
        check=False,
        capture_output=True,
        env=environment,
        text=True,
    )


def _shared_environ() -> dict[str, str]:
    return {
        'SG_QUERY_CACHE_REQUIRE_INTEGRATION': '1',
        'SG_QUERY_CACHE_TEST_RESULT_HOST': 'result.invalid',
        'SG_QUERY_CACHE_TEST_RESULT_PORT': '6379',
        'SG_QUERY_CACHE_TEST_RESULT_DATABASE': '0',
        'SG_QUERY_CACHE_TEST_RESULT_TLS': 'false',
        'SG_QUERY_CACHE_TEST_RESULT_PASSWORD': 'result-secret',
        'SG_QUERY_CACHE_TEST_METADATA_HOST': 'metadata.invalid',
        'SG_QUERY_CACHE_TEST_METADATA_PORT': '6379',
        'SG_QUERY_CACHE_TEST_METADATA_DATABASE': '0',
        'SG_QUERY_CACHE_TEST_METADATA_TLS': 'false',
        'SG_QUERY_CACHE_TEST_METADATA_PASSWORD': 'metadata-secret',
    }


def test_argument_parser_selects_one_named_scenario() -> None:
    arguments = _parse_args(
        ['--profile', 'local', '--scenario', 'event-storm', '--json', '--seed', '17']
    )

    assert arguments.profile == 'local'
    assert arguments.scenario == 'event-storm'
    assert arguments.json is True
    assert arguments.seed == 17


@pytest.mark.parametrize(
    'scenario', ['local-maintenance', 'decode', 'wrapper-hit', 'mixed-read-write']
)
def test_async_benchmark_reports_measured_scenario(scenario: str) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            'benchmarks/async_query_cache.py',
            '--scenario',
            scenario,
            '--population',
            '1',
            '--iterations',
            '10',
            '--json',
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stderr
    report = json.loads(completed.stdout)
    assert report['scenario'] == scenario
    assert report['iterations'] == 10
    assert report['runtime']['python']
    cases = report['cases']
    assert cases
    for case in cases:
        assert case['operations'] > 0
        assert case['latency_ms']['p99'] >= 0
        assert case['loop_lag_ms']['samples'] > 0
    if scenario == 'local-maintenance':
        assert [case['name'] for case in cases] == ['steady-state', 'expiry-backlog', 'churn']
        for case in cases:
            assert set(case) == {
                'name',
                'operations',
                'latency_ms',
                'loop_lag_ms',
                'throughput_ops_per_second',
                'population_per_queue',
                'accounting_bytes',
            }
            assert case['population_per_queue'] == 1
            assert case['operations'] == 10
            assert set(case['accounting_bytes']) == {
                'result_bytes',
                'generation_bytes',
                'lease_bytes',
            }
        steady, expired, churn = (case['accounting_bytes'] for case in cases)
        assert steady['result_bytes'] > 0
        assert steady['lease_bytes'] > 0
        assert expired['result_bytes'] == expired['lease_bytes'] == 0
        assert 0 < expired['generation_bytes'] < steady['generation_bytes']
        assert churn['result_bytes'] > steady['result_bytes']
        assert churn['generation_bytes'] > steady['generation_bytes']
    elif scenario == 'decode':
        assert cases[-1]['payload_bytes'] == 1_048_576
        assert all(case['encode_ms']['p99'] > 0 for case in cases)
    elif scenario == 'wrapper-hit':
        assert cases[0]['schema_reads'] == 1
        assert cases[0]['sg_calls_avoided'] == 10
        assert cases[1]['sg_calls'] == 1
        assert cases[1]['followers'] == 9
    else:
        success, failed = cases
        assert success['read_hits'] == 10
        assert success['read_bypasses'] == 0
        assert success['fence_outcome'] == 'acknowledged'
        assert failed['read_hits'] == 0
        assert failed['read_bypasses'] == 10
        assert failed['fence_outcome'] == 'failed-retained'
        assert all(case['window_duration_ms'] > 0 for case in cases)


def test_async_local_benchmark_does_not_require_optional_valkey_distribution() -> None:
    script = """
import importlib.metadata
import runpy
import sys
from pathlib import Path

def missing(name):
    raise importlib.metadata.PackageNotFoundError(name)

importlib.metadata.version = missing
sys.argv = [str(Path('benchmarks/async_query_cache.py').resolve()),
            '--scenario', 'local-maintenance', '--population', '1', '--iterations', '1', '--json']
runpy.run_path(sys.argv[0], run_name='__main__')
"""
    result = subprocess.run(
        [sys.executable, '-c', script], check=False, capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)['runtime']['valkey'] == 'not installed'


def test_async_mixed_benchmark_rejects_expired_fence_after_sg_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_async_benchmark()
    measure = module._measure
    update = module._RawClient.update
    close = AsyncQueryCache.close
    closed: list[AsyncQueryCache] = []

    async def observed_close(cache: AsyncQueryCache) -> None:
        await close(cache)
        closed.append(cache)

    async def scenario() -> None:
        loop = asyncio.get_running_loop()
        original_time = loop.time
        offset = 0.0
        dispatched = asyncio.Event()

        def now() -> float:
            return original_time() + offset

        async def observed_update(raw: _AsyncRawClient, *args: object, **kwargs: object) -> object:
            value = await update(raw, *args, **kwargs)
            dispatched.set()
            return value

        async def expire_fence(
            operation: Callable[[], Awaitable[object]], iterations: int
        ) -> object:
            nonlocal offset
            offset = 121.0
            await dispatched.wait()
            return await measure(operation, iterations)

        monkeypatch.setattr(loop, 'time', now)
        monkeypatch.setattr(module._RawClient, 'update', observed_update)
        monkeypatch.setattr(module, '_measure', expire_fence)
        monkeypatch.setattr(AsyncQueryCache, 'close', observed_close)
        with pytest.raises(RuntimeError, match='fence outcomes'):
            await module._mixed(1)
        assert dispatched.is_set()
        assert len(closed) == 1
        assert not [task for task in asyncio.all_tasks() if task is not asyncio.current_task()]

    asyncio.run(scenario())


def test_async_mixed_benchmark_rejects_expired_hit_without_reporting_bypass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_async_benchmark()
    measure = module._measure
    close = AsyncQueryCache.close
    closed: list[AsyncQueryCache] = []
    offset = 0.0

    def now() -> float:
        return perf_counter() + offset

    async def expire_hit(operation: Callable[[], Awaitable[object]], iterations: int) -> object:
        nonlocal offset
        offset = 31.0
        return await measure(operation, iterations)

    async def observed_close(cache: AsyncQueryCache) -> None:
        await close(cache)
        closed.append(cache)

    async def scenario() -> None:
        with pytest.raises(RuntimeError, match='warmed cache'):
            await module._mixed(1)
        assert len(closed) == 1
        assert not [task for task in asyncio.all_tasks() if task is not asyncio.current_task()]

    monkeypatch.setattr(module, 'perf_counter', now)
    monkeypatch.setattr(async_cache_module, 'monotonic', now)
    monkeypatch.setattr(module, '_measure', expire_hit)
    monkeypatch.setattr(AsyncQueryCache, 'close', observed_close)
    asyncio.run(scenario())


def test_dependency_scenario_reports_production_analyzer_transitions() -> None:
    report = _run_benchmark(
        'local',
        'dependency-broadening',
        options=_BenchmarkOptions(_operations=2, _workers=1),
    )
    operations = cast('dict[str, int]', report['cache_operations'])

    assert report['operations'] == 2
    assert operations['broadening:limit:max_dependency_tags'] > 0
    assert operations['broadening:broaden:entity_to_type'] > 0
    assert operations['broadening:broaden:field_to_type_value'] > 0
    assert operations['broadening:limit:max_dependency_vector_bytes'] > 0


def test_event_storm_operation_override_remains_fixed() -> None:
    accepted = _build_workload('event-storm', seed=23, operations=10_000, workers=1)

    assert accepted._operations == 10_000
    with pytest.raises(ValueError, match='event-storm requires exactly 10,000 operations'):
        _build_workload('event-storm', seed=23, operations=9_999, workers=4)


def test_fixed_seed_produces_the_same_bounded_workload() -> None:
    first = _build_workload('mixed-values', seed=23, operations=48, workers=4)
    _settings('shared', 'mixed-values', _shared_environ())
    second = _build_workload('mixed-values', seed=23, operations=48, workers=4)
    different = _build_workload('mixed-values', seed=29, operations=48, workers=4)

    assert first == second
    assert first != different
    assert first._operations == 48
    assert first._workers == 4


def test_hierarchical_fanout_minimum_uses_two_independent_l1s() -> None:
    report = _run_benchmark(
        'local',
        'hierarchical-fanout',
        options=_BenchmarkOptions(_operations=2, _workers=2),
    )
    operations = cast('dict[str, int]', report['cache_operations'])

    assert operations['fanout_participating_l1s'] == 2
    assert operations['fanout_shared_l2_instances'] == 1
    assert operations['fanout_shared_l2_operations'] > 0


def test_hot_key_minimum_retains_fan_in_evidence(monkeypatch: pytest.MonkeyPatch) -> None:
    fetch_entered = Event()
    follower_waiting = Event()
    release_fetch = Event()
    original_fetch = _MODULE._ReadState._add_sg_call
    original_wait = cache_module.QueryCache._wait_for_flight

    def observed_fetch(state: _typed_module._ReadState) -> None:
        original_fetch(state)
        fetch_entered.set()
        assert release_fetch.wait(timeout=3)

    def observed_wait(
        cache: cache_module.QueryCache,
        key: str,
        flight: cache_module._Flight,
    ) -> bool:
        follower_waiting.set()
        return original_wait(cache, key, flight)

    monkeypatch.setattr(_MODULE._ReadState, '_add_sg_call', observed_fetch)
    monkeypatch.setattr(cache_module.QueryCache, '_wait_for_flight', observed_wait)
    with ThreadPoolExecutor(max_workers=1) as executor:
        try:
            future = executor.submit(
                _run_benchmark,
                'local',
                'hot-key-burst',
                options=_BenchmarkOptions(_operations=2, _workers=2),
            )
            assert fetch_entered.wait(timeout=2)
            assert follower_waiting.wait(timeout=2)
        finally:
            release_fetch.set()
        report = future.result(timeout=3)
    operations = cast('dict[str, int]', report['cache_operations'])

    assert operations['fills'] == 1
    assert operations['hits'] == 1
    assert report['correctness_violations'] == 0
    assert report['operations'] == 2
    assert report['sg_calls_avoided'] == 1
    assert report['normal_hit_backend_phases'] == 2


def test_link_fanout_scenario_reports_invalidations_by_tag_kind() -> None:
    completed = _run(
        '--profile',
        'local',
        '--scenario',
        'link-fanout-events',
        '--json',
        '--operations',
        '64',
        '--workers',
        '4',
    )

    assert completed.returncode == 0, completed.stderr
    report = json.loads(completed.stdout)
    operations = report['cache_operations']
    assert report['scenario'] == 'link-fanout-events'
    assert report['tag_count'] == 21
    assert operations['analyzer_max_output_tags'] == 21
    assert operations['max_dependency_vector_bytes'] == 4_096
    assert operations['type_value_invalidations'] > 0
    assert operations['link_fanout_reads'] == 64
    assert 0 <= operations['link_fanout_hits'] <= 64
    assert report['correctness_violations'] == 0


def test_local_json_report_has_the_complete_schema() -> None:
    completed = _run(
        '--profile',
        'local',
        '--scenario',
        'hot-key-burst',
        '--json',
        '--operations',
        '24',
        '--workers',
        '4',
    )

    assert completed.returncode == 0, completed.stderr
    report = json.loads(completed.stdout)
    assert set(report) == _REQUIRED_FIELDS
    assert report['profile'] == 'local'
    assert report['scenario'] == 'hot-key-burst'
    assert report['operations'] == 24
    assert report['normal_hit_backend_phases'] == 2
    assert report['workers'] == 4
    assert report['correctness_violations'] == 0
    assert set(report['complete_hit_latency_ms']) == {'p50', 'p95', 'p99'}
    assert set(report['vector_resolution_latency_ms']) == {'p50', 'p95', 'p99'}


def test_metadata_churn_requires_its_single_deterministic_worker() -> None:
    accepted = _build_workload('metadata-churn', seed=23, operations=160, workers=1)

    assert accepted._workers == 1
    with pytest.raises(ValueError, match='operations and workers must be positive'):
        _build_workload('metadata-churn', seed=23, operations=160, workers=0)
    with pytest.raises(ValueError, match='metadata-churn requires exactly 1 worker'):
        _build_workload('metadata-churn', seed=23, operations=160, workers=2)


def test_metadata_minimum_forces_capacity_and_recreation_path() -> None:
    report = _run_benchmark(
        'local',
        'metadata-churn',
        options=_BenchmarkOptions(_operations=160, _workers=1),
    )
    operations = cast('dict[str, int]', report['cache_operations'])

    assert operations['metadata_capacity_rejections'] > 0
    assert report['correctness_violations'] == 0


def test_mixed_value_minimum_commits_and_hits_both_encoded_classes() -> None:
    report = _run_benchmark(
        'local',
        'mixed-values',
        options=_BenchmarkOptions(_operations=4, _workers=1),
    )
    operations = cast('dict[str, int]', report['cache_operations'])
    memory = cast('dict[str, int]', report['memory_estimates_bytes'])

    assert operations['mixed_near_cap_commits'] == 1
    assert operations['mixed_near_cap_hits'] == 1
    assert 1_000_000 <= operations['mixed_near_cap_encoded_bytes'] <= 1_048_576
    assert operations['mixed_small_commits'] == 1
    assert operations['mixed_small_hits'] == 1
    assert 0 < operations['mixed_small_encoded_bytes'] < 4_096
    assert memory['result_current'] >= operations['mixed_near_cap_encoded_bytes']
    assert memory['result_peak'] >= operations['mixed_near_cap_encoded_bytes']


def test_namespace_entropy_failure_is_a_redacted_setup_error(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:

    def fail_entropy(_byte_count: int) -> str:
        raise OSError('sensitive entropy failure detail')

    for name, value in _shared_environ().items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(_MODULE, 'token_hex', fail_entropy, raising=False)

    status = _main(
        [
            '--profile',
            'shared',
            '--scenario',
            'hot-key-burst',
            '--json',
            '--operations',
            '2',
            '--workers',
            '2',
        ]
    )
    captured = capsys.readouterr()

    assert status == 2
    assert captured.out == ''
    assert captured.err == 'query-cache benchmark: shared namespace generation failed\n'
    assert 'sensitive' not in captured.err


def test_normal_hit_backend_phase_measurement_is_deterministic() -> None:
    first = _run_benchmark(
        'local',
        'hot-key-burst',
        options=_BenchmarkOptions(_operations=2, _workers=2),
    )
    second = _run_benchmark(
        'local',
        'mixed-values',
        options=_BenchmarkOptions(_operations=4, _workers=1),
    )

    assert first['normal_hit_backend_phases'] == 2
    assert second['normal_hit_backend_phases'] == 2


def test_normal_hit_phase_backend_error_is_a_redacted_setup_failure(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    original_topology = _MODULE._topology
    calls = 0

    def fail_second_topology(
        profile: str,
        scenario: str,
        metrics: _typed_module._Metrics,
        environ: Mapping[str, str],
    ) -> object:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise _MODULE.BackendUnavailable('private probe failure')
        return original_topology(profile, scenario, metrics, environ)

    monkeypatch.setattr(_MODULE, '_topology', fail_second_topology)
    status = _main(
        [
            '--profile',
            'local',
            '--scenario',
            'hot-key-burst',
            '--json',
            '--operations',
            '2',
            '--workers',
            '2',
        ]
    )
    captured = capsys.readouterr()

    assert status == 2
    assert captured.out == ''
    assert captured.err == 'query-cache benchmark: normal-hit backend phase measurement failed\n'
    assert 'private' not in captured.err


def test_normal_hit_phase_topology_is_independent_and_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    topology_type = _MODULE._Topology
    original_close = topology_type._close
    original_topology = _MODULE._topology
    owned: list[_typed_module._Topology] = []
    closed: list[_typed_module._Topology] = []
    scenarios: list[str] = []

    def observed_topology(
        profile: str,
        scenario: str,
        metrics: _typed_module._Metrics,
        environ: Mapping[str, str],
    ) -> _typed_module._Topology:
        topology = original_topology(profile, scenario, metrics, environ)
        owned.append(topology)
        scenarios.append(scenario)
        return topology

    def observed_close(topology: _typed_module._Topology) -> None:
        closed.append(topology)
        original_close(topology)

    monkeypatch.setattr(topology_type, '_close', observed_close)
    monkeypatch.setattr(_MODULE, '_topology', observed_topology)
    report = _run_benchmark(
        'local',
        'hot-key-burst',
        options=_BenchmarkOptions(_operations=2, _workers=2),
    )
    assert report['normal_hit_backend_phases'] == 2
    assert scenarios == ['hot-key-burst', 'normal-hit']
    assert len(owned) == 2
    assert owned[0] is not owned[1]
    assert [id(topology) for topology in closed] == [id(topology) for topology in reversed(owned)]


def test_one_hit_minimum_forces_selected_key_eviction() -> None:
    report = _run_benchmark(
        'local',
        'one-hit-scan',
        options=_BenchmarkOptions(_operations=96, _workers=1),
    )
    operations = cast('dict[str, int]', report['cache_operations'])

    assert operations['evictions'] > 0
    assert operations['hot_key_refetches'] == 1


def test_scenario_selection_is_closed_to_the_acceptance_matrix() -> None:
    assert _SCENARIOS == (
        'hot-key-burst',
        'identity-cardinality',
        'one-hit-scan',
        'dependency-broadening',
        'event-storm',
        'metadata-churn',
        'mixed-values',
        'failure-recovery',
        'hierarchical-fanout',
        'link-fanout-events',
        'wrapper-schema',
    )

    with pytest.raises(SystemExit, match='2'):
        _parse_args(['--profile', 'local', '--scenario', 'unknown', '--json'])


def test_shared_namespaces_are_unique_per_invocation_and_scenario() -> None:
    environ = _shared_environ()

    first = _settings('shared', 'hot-key-burst', environ)
    second = _settings('shared', 'hot-key-burst', environ)
    other = _settings('shared', 'mixed-values', environ)
    prefixes = {
        first.l2.valkey.result.key_prefix,
        second.l2.valkey.result.key_prefix,
        other.l2.valkey.result.key_prefix,
    }

    assert len(prefixes) == 3
    assert first.l2.valkey.result.key_prefix == first.l2.valkey.metadata.key_prefix
    assert 'hot-key-burst' in first.l2.valkey.result.key_prefix
    assert 'mixed-values' in other.l2.valkey.result.key_prefix


@pytest.mark.parametrize(
    'scenario',
    [
        'dependency-broadening',
        'event-storm',
        'hierarchical-fanout',
        'hot-key-burst',
        'mixed-values',
    ],
)
def test_supported_shared_scenario_names_fit_the_key_prefix(scenario: str) -> None:
    settings = _settings('shared', scenario, _shared_environ())
    prefix = settings.l2.valkey.result.key_prefix

    assert scenario in prefix
    assert len(prefix) <= 64


@pytest.mark.parametrize(
    ('profile', 'scenario'),
    [
        ('shared', 'failure-recovery'),
        ('hierarchical', 'failure-recovery'),
        ('shared', 'metadata-churn'),
        ('hierarchical', 'metadata-churn'),
        ('shared', 'one-hit-scan'),
        ('hierarchical', 'one-hit-scan'),
        ('shared', 'identity-cardinality'),
        ('hierarchical', 'identity-cardinality'),
    ],
)
def test_unsafe_profile_scenarios_are_rejected_explicitly(profile: str, scenario: str) -> None:
    with pytest.raises(ValueError, match=f'{scenario} does not support the {profile} profile'):
        _run_benchmark(profile, scenario, environ={})


@pytest.mark.parametrize('profile', ['shared', 'hierarchical'])
def test_shared_profiles_refuse_implicit_or_nondisposable_endpoints(profile: str) -> None:
    completed = _run(
        '--profile',
        profile,
        '--scenario',
        'hot-key-burst',
        '--json',
        '--operations',
        '8',
        '--workers',
        '8',
    )

    assert completed.returncode == 2
    assert completed.stdout == ''
    assert 'explicitly configured disposable Valkey roles' in completed.stderr


@pytest.mark.parametrize(
    ('scenario', 'minimum_operations', 'workers'),
    [
        ('hot-key-burst', 2, 2),
        ('identity-cardinality', 1_001, 1),
        ('one-hit-scan', 96, 1),
        ('dependency-broadening', 2, 1),
        ('metadata-churn', 160, 1),
        ('mixed-values', 4, 1),
        ('failure-recovery', 8, 1),
        ('hierarchical-fanout', 2, 2),
    ],
)
def test_scenario_operation_minimums_preserve_named_invariants(
    scenario: str,
    minimum_operations: int,
    workers: int,
) -> None:
    accepted = _build_workload(
        scenario,
        seed=23,
        operations=minimum_operations,
        workers=workers,
    )

    assert accepted._operations == minimum_operations
    with pytest.raises(
        ValueError,
        match=rf'{scenario} requires at least {minimum_operations:,} operations',
    ):
        _build_workload(
            scenario,
            seed=23,
            operations=minimum_operations - 1,
            workers=workers,
        )


@pytest.mark.parametrize('scenario', ['hot-key-burst', 'hierarchical-fanout'])
def test_concurrent_scenarios_require_at_least_two_workers(scenario: str) -> None:
    accepted = _build_workload(scenario, seed=23, operations=2, workers=2)

    assert accepted._workers == 2
    with pytest.raises(ValueError, match=rf'{scenario} requires at least 2 workers'):
        _build_workload(scenario, seed=23, operations=2, workers=1)


@pytest.mark.parametrize('seed', [0, 23, 4_294_967_295])
def test_mixed_value_minimum_contains_small_and_near_cap_payloads(seed: int) -> None:
    workload = _build_workload('mixed-values', seed=seed, operations=4, workers=1)

    assert workload._keys == (0, 0, 1, 1)
    assert workload._payload_sizes[:2] == (780_000, 780_000)
    assert workload._payload_sizes[2:] == (64, 64)


@pytest.mark.parametrize(
    ('scenario', 'operations', 'accepted_workers', 'rejected_workers'),
    [
        ('hot-key-burst', 2, 2, 3),
        ('identity-cardinality', 1_001, 64, 65),
        ('one-hit-scan', 96, 64, 65),
        ('dependency-broadening', 2, 2, 3),
        ('event-storm', 10_000, 64, 65),
        ('metadata-churn', 160, 1, 2),
        ('mixed-values', 4, 1, 2),
        ('mixed-values', 24, 20, 21),
        ('failure-recovery', 8, 1, 2),
        ('failure-recovery', 48, 40, 41),
        ('hierarchical-fanout', 2, 2, 3),
    ],
)
def test_worker_contracts_reject_the_first_worker_without_meaningful_work(
    scenario: str,
    operations: int,
    accepted_workers: int,
    rejected_workers: int,
) -> None:
    accepted = _build_workload(
        scenario,
        seed=23,
        operations=operations,
        workers=accepted_workers,
    )

    assert accepted._workers == accepted_workers
    with pytest.raises(
        ValueError,
        match=r'(workers must not exceed|requires exactly 1 worker)',
    ):
        _build_workload(
            scenario,
            seed=23,
            operations=operations,
            workers=rejected_workers,
        )


def test_workload_overrides_accept_global_limits_and_reject_unbounded_values() -> None:
    maximum = _build_workload('hot-key-burst', seed=23, operations=10_000, workers=64)

    assert maximum._operations == 10_000
    assert maximum._workers == 64
    with pytest.raises(ValueError, match='operations must not exceed 10,000'):
        _build_workload('hot-key-burst', seed=23, operations=10_001, workers=4)
    with pytest.raises(ValueError, match='workers must not exceed 64'):
        _build_workload('hot-key-burst', seed=23, operations=8, workers=65)


def test_wrapper_schema_scenario_counts_one_schema_read() -> None:
    completed = _run(
        '--profile',
        'local',
        '--scenario',
        'wrapper-schema',
        '--json',
        '--operations',
        '32',
        '--workers',
        '4',
    )

    assert completed.returncode == 0, completed.stderr
    report = json.loads(completed.stdout)
    operations = report['cache_operations']
    assert report['scenario'] == 'wrapper-schema'
    assert operations['schema_reads'] == 1
    assert operations['complete_hit_samples'] == operations['hits'] == report['operations'] == 32
    assert report['correctness_violations'] == 0
