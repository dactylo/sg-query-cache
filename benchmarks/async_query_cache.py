# Same-file collaborators use private implementation members.
# pyright: reportPrivateUsage=false

# ruff: noqa: INP001
"""Measure native async cache maintenance, codecs, wrapper hits, and fence windows.

Run with `--scenario` and optional `--population`, `--iterations`, and
`--json`.
Local and controlled-double measurements require no services. Loop lag uses
continuous ready-task scheduling gaps; setup is excluded from timings.

Importing this script prepends the checkout's `src` directory to `sys.path`.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import platform
import sys
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from time import perf_counter
from typing import TYPE_CHECKING

from typing_extensions import override

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Mapping

    from sg_query_cache._async_support import OperationToken

sys.path.insert(0, str(__file__.rsplit('/benchmarks/', 1)[0] + '/src'))

from sg_query_cache import (
    AsyncCachedShotgun,
    AsyncLocalBackend,
    AsyncQueryCache,
    CacheSettings,
    ClientContext,
    Identity,
)
from sg_query_cache._backend import BackendUnavailable
from sg_query_cache._cache_common import TimedResult, encode_result
from sg_query_cache._codec import MAX_BYTES, decode_value
from sg_query_cache._config import L1Settings, LimitSettings
from sg_query_cache._model import DependencyTag, GenerationChange

_CONTEXT = ClientContext(
    'https://example.shotgrid.autodesk.com',
    'production',
    localized=False,
    convert_datetimes_to_utc=True,
    records_per_page=500,
)
_IDENTITY = Identity('script', 'benchmark', 'async')
_SCENARIOS = ('local-maintenance', 'decode', 'wrapper-hit', 'mixed-read-write')


class _Clock:
    """Provide an explicitly advanced clock for maintenance workloads."""

    def __init__(self) -> None:
        """Start with all populated records live."""
        self._now = 1000.0

    def __call__(self) -> float:
        """Return the controlled backend time."""
        return self._now


class _Metrics:
    """Count cache outcomes without retaining per-operation observations."""

    def __init__(self) -> None:
        """Initialize named counters."""
        self._counts: dict[str, int] = {}

    def increment(self, name: str, labels: Mapping[str, str]) -> None:
        """Count one event; labels are not retained."""
        _ = labels
        self._counts[name] = self._counts.get(name, 0) + 1

    def observe(self, name: str, value: float, labels: Mapping[str, str]) -> None:
        """Discard histogram events to bound instrumentation memory."""
        _ = name, value, labels


class _FenceBackend(AsyncLocalBackend):
    """Delay or fail real local generation changes under an explicit gate."""

    def __init__(self, metrics: _Metrics) -> None:
        """Create the native local backend and its controlled fence events."""
        super().__init__(CacheSettings(), perf_counter, metrics)
        self._entered = asyncio.Event()
        self._proceed = asyncio.Event()
        self._fail = False

    @override
    async def change(self, changes: tuple[GenerationChange, ...]) -> None:
        self._entered.set()
        await self._proceed.wait()
        if self._fail:
            raise BackendUnavailable('controlled fence failure')
        await super().change(changes)


class _FenceCache(AsyncQueryCache):
    """Record the actual fence acknowledgments for one mixed benchmark window."""

    def __init__(self, settings: CacheSettings, backend: _FenceBackend, metrics: _Metrics) -> None:
        """Create the measured cache and retain its two write-fence outcomes."""
        super().__init__(settings, _IDENTITY, l1=backend, metrics=metrics)
        self._outcomes: list[bool] = []

    @override
    async def invalidate_admitted(
        self, changes: tuple[GenerationChange, ...], *, token: OperationToken
    ) -> bool:
        acknowledged = await super().invalidate_admitted(changes, token=token)
        self._outcomes.append(acknowledged)
        return acknowledged


class _RawClient:
    """Supply a complete generated SG schema and controlled native async calls."""

    def __init__(self) -> None:
        """Generate the existing benchmark's 40-type, 80-field schema shape."""
        self._calls = 0
        self._schema_reads = 0
        self._writes = 0
        self._gate: asyncio.Event | None = None
        self._schema = {
            f'CustomEntity{index:02d}': {
                'id': {'data_type': {'value': 'number'}, 'properties': {}},
                **{
                    f'sg_field_{field:03d}': {
                        'data_type': {'value': 'text'},
                        'properties': {},
                    }
                    for field in range(80)
                },
            }
            for index in range(40)
        }

    async def find(self, *args: object, **kwargs: object) -> object:
        """Return one result after the optional leader gate opens."""
        _ = args, kwargs
        self._calls += 1
        if self._gate is not None:
            await self._gate.wait()
        return [{'id': 1, 'type': 'CustomEntity01'}]

    async def schema_read(self, project_entity: object = None) -> object:
        """Count schema acquisition independently from parsed memo reuse."""
        _ = project_entity
        self._schema_reads += 1
        return self._schema

    async def update(self, *args: object, **kwargs: object) -> object:
        """Acknowledge the controlled write used to invalidate a warmed query."""
        _ = args, kwargs
        self._writes += 1
        return {'id': 1, 'type': 'CustomEntity01'}


@dataclass
class _Samples:
    """Retain latency and ready-callback lag samples in seconds."""

    _latency: list[float]
    _lag: list[float]


async def _measure(operation: Callable[[], Awaitable[object]], iterations: int) -> _Samples:
    """Sample consecutive ready-task scheduling gaps throughout measured operations."""
    latency: list[float] = []
    lag: list[float] = []
    running = True

    async def sample() -> None:
        previous = perf_counter()
        while running:
            await asyncio.sleep(0)
            now = perf_counter()
            lag.append(now - previous)
            previous = now

    sampler = asyncio.create_task(sample())
    await asyncio.sleep(0)
    for _ in range(iterations):
        start = perf_counter()
        await operation()
        latency.append(perf_counter() - start)
        await asyncio.sleep(0)
    running = False
    await sampler
    return _Samples(latency, lag)


def _percentiles(values: list[float]) -> dict[str, float | int]:
    """Report nearest-rank latency percentiles in milliseconds."""
    ordered = sorted(values)
    return {
        **{
            f'p{percentile}': ordered[
                min(len(ordered) - 1, max(0, (len(ordered) * percentile + 99) // 100 - 1))
            ]
            * 1000
            for percentile in (50, 95, 99)
        },
        'max': ordered[-1] * 1000,
        'samples': len(ordered),
    }


def _report(name: str, samples: _Samples) -> dict[str, object]:
    """Summarize measured work, excluding population and scheduler setup."""
    return {
        'name': name,
        'operations': len(samples._latency),
        'latency_ms': _percentiles(samples._latency),
        'loop_lag_ms': _percentiles(samples._lag),
        'throughput_ops_per_second': len(samples._latency) / sum(samples._latency),
    }


async def _decode(iterations: int) -> list[dict[str, object]]:
    """Measure representative and exactly byte-ceiling admitted envelopes."""
    reports: list[dict[str, object]] = []
    overhead = len(encode_result(TimedResult(0.01, ''), 0.0, 30.0))
    for name, value in (
        (
            'representative',
            [{'id': index, 'type': 'Asset', 'code': 'x' * 64} for index in range(100)],
        ),
        ('node-ceiling', [None] * 99_994),
        ('byte-ceiling', 'x' * (MAX_BYTES - overhead)),
    ):
        payload = encode_result(TimedResult(0.01, value), 0.0, 30.0)

        async def decode(payload: bytes = payload) -> object:
            return decode_value(payload)

        async def encode(value: object = value) -> object:
            return encode_result(TimedResult(0.01, value), 0.0, 30.0)

        async def worker_decode(payload: bytes = payload) -> object:
            return await asyncio.to_thread(decode_value, payload)

        samples = await _measure(decode, iterations)
        encoded = await _measure(encode, iterations)
        worker = await _measure(worker_decode, iterations)
        report = _report(name, samples)
        report.update(
            payload_bytes=len(payload),
            codec_nodes=100_000
            if name == 'node-ceiling'
            else (6 if name == 'byte-ceiling' else 706),
            encode_ms=_percentiles(encoded._latency),
            encode_loop_lag_ms=_percentiles(encoded._lag),
            worker_decode_ms=_percentiles(worker._latency),
            worker_loop_lag_ms=_percentiles(worker._lag),
        )
        reports.append(report)
    return reports


async def _maintenance(population: int, iterations: int) -> list[dict[str, object]]:
    """Populate all three maintenance queues and measure steady, expired, and churn work."""
    reports: list[dict[str, object]] = []
    for name in ('steady-state', 'expiry-backlog', 'churn'):
        clock = _Clock()
        settings = CacheSettings(l1=L1Settings(metadata_budget_bytes=134_217_728))
        backend = AsyncLocalBackend(settings, clock, _Metrics())
        payload = encode_result(TimedResult(0.0, 42), 0.0, 30.0)
        for index in range(population):
            lease = await backend.reserve(f'result-{index}', 30)
            if lease is None or not await backend.commit(lease, payload, 30):
                raise RuntimeError('maintenance population did not fit')
            await backend.reserve(f'lease-{index}', 30)
            await backend.resolve((DependencyTag.entity('production', 'Asset', index + 1),))
        if name == 'expiry-backlog':
            clock._now += 8000
        counter = 0

        async def lookup(
            backend: AsyncLocalBackend = backend,
            name: str = name,
            payload: bytes = payload,
        ) -> object:
            nonlocal counter
            result = await backend.lookup('missing')
            if name == 'churn':
                key = f'churn-{counter}'
                lease = await backend.reserve(key, 30)
                if lease is not None:
                    await backend.commit(lease, payload, 30)
                await backend.resolve((DependencyTag.entity('production', 'Shot', counter + 1),))
                counter += 1
            return result

        samples = await _measure(lookup, iterations)
        report = _report(name, samples)
        report.update(
            population_per_queue=population,
            accounting_bytes=await backend.storage_usage(),
        )
        reports.append(report)
        await backend.close()
    return reports


async def _mixed(iterations: int) -> list[dict[str, object]]:
    """Measure reads inside one delayed successful fence and one retained failure window."""
    reports: list[dict[str, object]] = []
    for failed in (False, True):
        metrics = _Metrics()
        backend = _FenceBackend(metrics)
        settings = CacheSettings(limits=LimitSettings(operation_timeout_seconds=120))
        cache = _FenceCache(settings, backend, metrics)
        raw = _RawClient()
        client = AsyncCachedShotgun(raw, cache=cache, context=_CONTEXT)

        async def read(client: AsyncCachedShotgun = client) -> object:
            return await client.find('CustomEntity01', [], ['id'])

        async with cache:
            await read()
            backend._fail = failed
            started = perf_counter()
            writer = asyncio.create_task(
                client.update('CustomEntity01', 1, {'sg_field_000': 'new'})
            )
            try:
                await backend._entered.wait()
                if failed:
                    backend._proceed.set()
                    await writer
                before = raw._calls
                schema_before = raw._schema_reads
                samples = await _measure(read, iterations)
                await asyncio.sleep(0.005)
                duration = perf_counter() - started
            finally:
                backend._proceed.set()
                await writer
            if cache._outcomes != [not failed, not failed]:
                raise RuntimeError('mixed benchmark left its intended fence outcomes')
            calls = raw._calls - before
            if not failed and calls:
                raise RuntimeError('mixed benchmark left its warmed cache hit window')
            report = _report('failed-retained' if failed else 'successful-in-flight', samples)
            report.update(
                read_hits=iterations - calls,
                read_bypasses=calls,
                sg_calls=calls + raw._schema_reads - schema_before,
                sg_find_calls=calls,
                sg_schema_calls=raw._schema_reads - schema_before,
                sg_calls_avoided=iterations - calls,
                writes=1,
                reads=iterations,
                mix=f'{iterations}:1 reads:SG-write',
                sg_writes=raw._writes,
                cooperative_fence_budget_seconds=120,
                minimum_fence_delay_ms=5,
                window_duration_ms=duration * 1000,
                fence_outcome='failed-retained' if failed else 'acknowledged',
            )
            reports.append(report)
    return reports


def _valkey_version() -> str:
    """Report optional distribution availability without requiring installation."""
    try:
        return version('valkey')
    except PackageNotFoundError:
        return 'not installed'


async def _wrapper(iterations: int) -> list[dict[str, object]]:
    """Measure complete warm-memo hits and one concurrent cold hot-key flight."""
    raw = _RawClient()
    reports: list[dict[str, object]] = []
    async with AsyncCachedShotgun.open(raw, CacheSettings(), _IDENTITY, _CONTEXT) as client:

        async def read() -> object:
            return await client.find('CustomEntity01', [], ['id'])

        await read()
        before = raw._calls
        samples = await _measure(read, iterations)
        report = _report('complete-wrapper-hit', samples)
        report.update(
            sg_calls=raw._calls - before,
            sg_calls_avoided=iterations - raw._calls + before,
            schema_reads=raw._schema_reads,
            schema_entity_types=40,
            fields_per_type=81,
        )
        reports.append(report)
        await client.update('CustomEntity01', 1, {'sg_field_000': 'updated'})
        raw._gate = asyncio.Event()
        before = raw._calls

        async def burst() -> object:
            tasks = [asyncio.create_task(read()) for _ in range(iterations)]
            await asyncio.sleep(0)
            if raw._gate is None:
                raise RuntimeError('missing leader gate')
            raw._gate.set()
            return await asyncio.gather(*tasks)

        samples = await _measure(burst, 1)
        report = _report('hot-key-followers', samples)
        report.update(
            sg_calls=raw._calls - before,
            followers=iterations - 1,
            sg_calls_avoided=iterations - raw._calls + before,
            requests=iterations,
        )
        reports.append(report)
    return reports


async def _run(scenario: str, population: int, iterations: int) -> dict[str, object]:
    """Dispatch one bounded scenario and attach non-identifying runtime metadata."""
    if scenario == 'local-maintenance':
        cases = await _maintenance(population, iterations)
    elif scenario == 'decode':
        cases = await _decode(iterations)
    elif scenario == 'wrapper-hit':
        cases = await _wrapper(iterations)
    else:
        cases = await _mixed(iterations)
    return {
        'scenario': scenario,
        'iterations': iterations,
        'population': population,
        'topology': 'local',
        'service_evidence': 'none; native local or controlled async double',
        'runtime': {
            'python': platform.python_version(),
            'platform': platform.system(),
            'machine': platform.machine(),
            'valkey': _valkey_version(),
        },
        'loop_lag_model': 'continuous sleep(0) gaps; yield after each operation',
        'cases': cases,
    }


def main() -> None:
    """Run the selected benchmark and emit its measured report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scenario', choices=_SCENARIOS, required=True)
    parser.add_argument('--population', type=int, default=1)
    parser.add_argument('--iterations', type=int, default=1000)
    parser.add_argument('--json', action='store_true')
    options = parser.parse_args()
    if options.population < 1 or options.iterations < 1:
        parser.error('population and iterations must be positive')
    report = asyncio.run(_run(options.scenario, options.population, options.iterations))
    print(json.dumps(report, indent=None if options.json else 2))


if __name__ == '__main__':
    main()
