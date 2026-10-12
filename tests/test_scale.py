import os
from pathlib import Path
from time import monotonic
from typing import TYPE_CHECKING, cast

import pytest

from tests._benchmark_loader import load_benchmark

_BENCHMARK = Path('benchmarks/query_cache.py')
if TYPE_CHECKING:
    import benchmarks.query_cache as _typed_module

    _MODULE = _typed_module
else:
    _MODULE = load_benchmark(_BENCHMARK, 'sg_query_cache_scale_benchmark')
_SCENARIOS = _MODULE._SCENARIOS
_run_benchmark = _MODULE._run_benchmark

pytestmark = pytest.mark.slow


def _integer(report: dict[str, object], field: str) -> int:
    value = report[field]
    assert isinstance(value, int)
    return value


def _mapping(report: dict[str, object], field: str) -> dict[str, int]:
    value = report[field]
    assert isinstance(value, dict)
    return cast('dict[str, int]', value)


@pytest.mark.integration
@pytest.mark.parametrize(
    ('profile', 'expected_fills'),
    [('shared', 1), ('hierarchical', 2)],
)
def test_disposable_valkey_profiles_preserve_scale_invariants(
    profile: str,
    expected_fills: int,
) -> None:
    if os.environ.get('SG_QUERY_CACHE_REQUIRE_INTEGRATION') != '1':
        pytest.skip('requires explicitly configured disposable Valkey roles')

    report = _run_benchmark(profile, 'hot-key-burst', environ=os.environ)

    assert report['correctness_violations'] == 0
    assert _mapping(report, 'cache_operations')['fills'] == expected_fills
    assert _integer(report, 'normal_hit_backend_phases') == 2


def test_local_scale_matrix_is_correct_and_bounded() -> None:
    # One collected test keeps xdist from repeating the matrix on different workers.
    started_at = monotonic()
    reports = {scenario: _run_benchmark('local', scenario) for scenario in _SCENARIOS}
    elapsed = monotonic() - started_at

    assert elapsed < 30
    assert all(report['correctness_violations'] == 0 for report in reports.values())
    assert all(_integer(report, 'normal_hit_backend_phases') == 2 for report in reports.values())
    assert reports['event-storm']['operations'] == 10_000

    operations = _mapping(reports['failure-recovery'], 'cache_operations')
    assert operations['failure_timeouts'] == 5
    assert operations['breaker_open_bypasses'] == 1
    assert operations['backend_restarts'] == 1
    assert operations['recovery_hits'] == 1
    assert _mapping(reports['hot-key-burst'], 'cache_operations')['fills'] == 1
    assert _integer(reports['hot-key-burst'], 'sg_calls_avoided') > 0
    report = reports['identity-cardinality']
    assert _integer(report, 'operations') > 1_000
    assert _mapping(report, 'cache_operations')['backend_instances'] == 1
    assert _mapping(report, 'cache_operations')['identity_count'] == 1_001
    assert 0 < _mapping(report, 'memory_estimates_bytes')['result_peak'] <= 2_097_152
    assert _mapping(report, 'memory_estimates_bytes')['measured_backends'] == 1
    report = reports['one-hit-scan']
    assert _mapping(report, 'cache_operations')['evictions'] > 0
    assert _mapping(report, 'cache_operations')['hot_key_refetches'] == 1
    assert _mapping(report, 'cache_operations')['maximum_age_refetches'] == 1
    assert _mapping(report, 'memory_estimates_bytes')['result_peak'] <= 2_097_152
    assert _mapping(report, 'memory_estimates_bytes')['measured_backends'] == 1
