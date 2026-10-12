# Same-file collaborators use private implementation members.
# pyright: reportPrivateUsage=false

# ruff: noqa: INP001
"""Run deterministic bounded synthetic workloads against supported cache topologies.

Run with `--profile`, `--scenario`, and `--json`; optional sizing and seed
arguments select a bounded workload. Shared profiles require the explicit
disposable-service integration environment described in `docs/benchmarks.md`.
The local failure-recovery scenario also requires the optional Valkey package.

Importing this script prepends the checkout's `src` directory to `sys.path`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from importlib import import_module
from math import ceil
from secrets import token_hex
from threading import Lock
from time import monotonic, sleep
from typing import TYPE_CHECKING, Protocol, TypeVar, cast

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from sg_query_cache.valkey import ValkeyBackend

sys.path.insert(0, str(__file__.rsplit('/benchmarks/', 1)[0] + '/src'))

from sg_query_cache import CacheSettings, Identity, LocalBackend, QueryCache
from sg_query_cache._backend import (
    BackendCapacity,
    BackendProtocolError,
    BackendUnavailable,
    CacheBackend,
    StaleIncarnation,
)
from sg_query_cache._client import CachedShotgun, ClientContext
from sg_query_cache._config import (
    L1Settings,
    L2Settings,
    LimitSettings,
    ValkeyRoleSettings,
    ValkeySettings,
)
from sg_query_cache._dependencies import DependencyAnalyzer, SchemaView
from sg_query_cache._keys import KeyBuilder
from sg_query_cache._model import (
    CacheHit,
    DependencyPlan,
    DependencyTag,
    ExactCall,
    ExactValue,
    FillLease,
    GenerationChange,
    GenerationVector,
)

_T = TypeVar('_T')

_BREAKER_FAILURES = 5
_CLIENT_OPTIONS = {'localized': False, 'convert_datetimes_to_utc': True}
_COMMIT_ARGUMENTS = 5
_COMMIT_KEYS = 2
_DEFAULT_OPERATIONS: Mapping[str, int] = {
    'hot-key-burst': 96,
    'identity-cardinality': 1_001,
    'one-hit-scan': 256,
    'dependency-broadening': 96,
    'event-storm': 10_000,
    'metadata-churn': 192,
    'mixed-values': 24,
    'failure-recovery': 48,
    'hierarchical-fanout': 96,
    'link-fanout-events': 256,
    'wrapper-schema': 64,
}
_DEFAULT_WORKERS: Mapping[str, int] = {
    'hot-key-burst': 16,
    'identity-cardinality': 16,
    'one-hit-scan': 8,
    'dependency-broadening': 8,
    'event-storm': 16,
    'metadata-churn': 1,
    'mixed-values': 8,
    'failure-recovery': 8,
    'hierarchical-fanout': 16,
    'link-fanout-events': 8,
    'wrapper-schema': 4,
}
_FANOUT_EVENT_INTERVAL = 4
_FANOUT_TARGETS = (
    'Shot',
    'Version',
    'Task',
    'Note',
    'Sequence',
    'Playlist',
    'PublishedFile',
    'Project',
)
_MAX_LOCAL_METADATA_BYTES = 1_048_576
_MAX_LOCAL_RESULT_BYTES = 2_097_152
_MAX_OPERATIONS = 10_000
_MAX_SMALL_ENCODED_BYTES = 4_096
_MAX_WORKERS = 64
_MIN_NEAR_CAP_ENCODED_BYTES = 1_000_000
_NAMESPACE_TOKEN_BYTES = 16
_NEAR_CAP_PAYLOAD_BYTES = 780_000
_RELEASE_ARGUMENTS = 2
_RESERVE_ARGUMENTS = 3
_SCENARIOS = (
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
# Values are minimum operations, fixed operations, minimum workers, fixed workers,
# and operations completed before any trailing parallel work.
_SCENARIO_CONTRACTS: Mapping[str, tuple[int, int | None, int, int | None, int | None]] = {
    'hot-key-burst': (2, None, 2, None, 0),
    'identity-cardinality': (1_001, None, 1, None, 0),
    'one-hit-scan': (96, None, 1, None, 0),
    'dependency-broadening': (2, None, 1, None, 0),
    'event-storm': (10_000, 10_000, 1, None, 0),
    'metadata-churn': (160, None, 1, 1, None),
    'mixed-values': (4, None, 1, None, 4),
    'failure-recovery': (_BREAKER_FAILURES + 3, None, 1, None, _BREAKER_FAILURES + 3),
    'hierarchical-fanout': (2, None, 2, None, 0),
    'link-fanout-events': (8, None, 1, None, 0),
    'wrapper-schema': (4, None, 1, None, 0),
}
_SHARED_NORMAL_HIT_PHASES = 2
_SITE = 'benchmark'
_UNSUPPORTED_SHARED_SCENARIOS = frozenset(
    {
        'failure-recovery',
        'identity-cardinality',
        'metadata-churn',
        'one-hit-scan',
        'wrapper-schema',
    }
)


class _Arguments(argparse.Namespace):
    """Hold validated command-line benchmark arguments."""

    json: bool
    operations: int | None
    profile: str
    scenario: str
    seed: int
    workers: int | None


class _Clock:
    """Provide a thread-safe controllable monotonic clock for local metadata churn."""

    __slots__ = ('_lock', '_value')

    def __init__(self) -> None:
        """Start a synchronized clock at the current monotonic time."""
        self._lock = Lock()
        self._value = monotonic()

    def __call__(self) -> float:
        """Return the current synthetic monotonic value."""
        with self._lock:
            return self._value

    def _advance(self, seconds: float) -> None:
        """Advance the synthetic clock by a positive duration."""
        with self._lock:
            self._value += seconds


class _Closable(Protocol):
    """Expose one close operation for benchmark-owned resources."""

    def close(self) -> None:
        """Release the resource."""
        ...


class _Deterministic:
    """Provide a tiny reproducible non-security integer sequence."""

    __slots__ = ('_state',)

    def __init__(self, seed: int) -> None:
        """Seed one repeatable integer generator."""
        self._state = seed & 0xFFFFFFFF

    def _range(self, start: int, stop: int) -> int:
        """Return one deterministic integer in `[start, stop)`."""
        if stop <= start:
            raise ValueError('deterministic range requires an increasing bound')
        self._state = (1_664_525 * self._state + 1_013_904_223) & 0xFFFFFFFF
        return start + self._state % (stop - start)


class _MeasuredBackend:
    """Measure calls while delegating every cache operation to a real backend."""

    __slots__ = (
        '_backend',
        '_committed_sizes',
        '_lock',
        '_memory_current',
        '_memory_peak',
        '_memory_probe',
        '_operations',
        '_vector_latencies_ms',
    )

    def _committed_payload_sizes(self) -> list[int]:
        """Return complete encoded byte sizes accepted by the delegated owner."""
        with self._lock:
            return list(self._committed_sizes)

    def _memory_estimates(self) -> dict[str, int]:
        """Return actual current and observed peak `LocalBackend` owner bytes."""
        with self._lock:
            result_current, metadata_current = self._memory_current
            result_peak, metadata_peak = self._memory_peak
        return {
            'measured': int(self._memory_probe is not None),
            'metadata_current': metadata_current,
            'metadata_peak': metadata_peak,
            'result_current': result_current,
            'result_peak': result_peak,
        }

    def _operation_counts(self) -> dict[str, int]:
        """Return a detached operation counter mapping."""
        with self._lock:
            return dict(self._operations)

    def _record(self, name: str) -> None:
        """Increment one backend-operation counter."""
        with self._lock:
            self._operations[name] += 1

    def _sample_memory(self) -> None:
        """Record one actual backend-owned current and peak byte snapshot."""
        if self._memory_probe is None:
            return
        current = self._memory_probe()
        with self._lock:
            self._memory_current = current
            self._memory_peak = (
                max(self._memory_peak[0], current[0]),
                max(self._memory_peak[1], current[1]),
            )

    def __init__(
        self,
        backend: CacheBackend,
        *,
        memory_probe: Callable[[], tuple[int, int]] | None = None,
    ) -> None:
        """Retain one backend and initialize bounded measurement state."""
        self._backend = backend
        self._committed_sizes: list[int] = []
        self._lock = Lock()
        self._memory_current = (0, 0)
        self._memory_peak = (0, 0)
        self._memory_probe = memory_probe
        self._operations = {
            'change': 0,
            'commit': 0,
            'lookup': 0,
            'release': 0,
            'reserve': 0,
            'resolve': 0,
        }
        self._vector_latencies_ms: list[float] = []
        self._sample_memory()

    def _vector_latencies(self) -> list[float]:
        """Return detached vector-resolution timings in milliseconds."""
        with self._lock:
            return list(self._vector_latencies_ms)

    def change(self, changes: tuple[GenerationChange, ...]) -> None:
        """Delegate one ordered generation change."""
        self._record('change')
        try:
            self._backend.change(changes)
        finally:
            self._sample_memory()

    def close(self) -> None:
        """Close the delegated backend."""
        self._backend.close()

    def commit(self, lease: FillLease, payload: bytes, remaining_seconds: float) -> bool:
        """Delegate a token-checked commit and sample actual backend bytes."""
        self._record('commit')
        try:
            committed = self._backend.commit(lease, payload, remaining_seconds)
            if committed:
                with self._lock:
                    self._committed_sizes.append(len(payload))
            return committed
        finally:
            self._sample_memory()

    def flush_site(self, site_namespace: str, reason: str) -> None:
        """Delegate one explicit site-generation flush."""
        try:
            self._backend.flush_site(site_namespace, reason)
        finally:
            self._sample_memory()

    def lookup(self, key: str) -> CacheHit | None:
        """Delegate one result lookup."""
        self._record('lookup')
        try:
            return self._backend.lookup(key)
        finally:
            self._sample_memory()

    def release(self, lease: FillLease) -> bool:
        """Delegate one token-checked lease release."""
        self._record('release')
        try:
            return self._backend.release(lease)
        finally:
            self._sample_memory()

    def reserve(self, key: str, lease_seconds: float) -> FillLease | None:
        """Delegate one fill-lease reservation."""
        self._record('reserve')
        try:
            return self._backend.reserve(key, lease_seconds)
        finally:
            self._sample_memory()

    def resolve(self, tags: tuple[DependencyTag, ...]) -> GenerationVector:
        """Time one complete vector resolution through the delegated backend."""
        started_at = monotonic()
        try:
            return self._backend.resolve(tags)
        finally:
            elapsed_ms = (monotonic() - started_at) * 1_000
            with self._lock:
                self._operations['resolve'] += 1
                self._vector_latencies_ms.append(elapsed_ms)
            self._sample_memory()


class _Metrics:
    """Collect fixed cache metrics safely across benchmark worker threads."""

    __slots__ = ('_increments', '_lock', '_observations')

    def __init__(self) -> None:
        """Initialize synchronized counters and sampled observations."""
        self._increments: dict[tuple[str, tuple[tuple[str, str], ...]], int] = {}
        self._lock = Lock()
        self._observations: dict[tuple[str, tuple[tuple[str, str], ...]], list[float]] = {}

    def _count(self, name: str, labels: Mapping[str, str] | None = None) -> int:
        """Return the number of matching counter events."""
        selected = None if labels is None else tuple(labels.items())
        with self._lock:
            return sum(
                count
                for (metric_name, metric_labels), count in self._increments.items()
                if metric_name == name and (selected is None or metric_labels == selected)
            )

    def _observation_count(
        self,
        name: str,
        labels: Mapping[str, str] | None = None,
    ) -> int:
        """Return the number of matching observations."""
        selected = None if labels is None else tuple(labels.items())
        with self._lock:
            return sum(
                len(values)
                for (metric_name, metric_labels), values in self._observations.items()
                if metric_name == name and (selected is None or metric_labels == selected)
            )

    def increment(self, name: str, labels: Mapping[str, str]) -> None:
        """Record one counter event.

        Args:
            name: Metric name.
            labels: Low-cardinality metric labels.
        """
        key = name, tuple(labels.items())
        with self._lock:
            self._increments[key] = self._increments.get(key, 0) + 1

    def observe(self, name: str, value: float, labels: Mapping[str, str]) -> None:
        """Record one numeric observation.

        Args:
            name: Metric name.
            value: Observed value.
            labels: Low-cardinality metric labels.
        """
        key = name, tuple(labels.items())
        with self._lock:
            self._observations.setdefault(key, []).append(value)


class _ResultProtocolClient:
    """Drive actual `ValkeyBackend` scripts through a bounded result protocol seam."""

    __slots__ = ('_closed', '_failure', '_lookup_attempts', '_stored')

    def __init__(self, failure: BaseException | None = None) -> None:
        """Initialize the controlled script replies and optional lookup failure."""
        self._closed = False
        self._failure = failure
        self._lookup_attempts = 0
        self._stored: bytes | None = None

    def close(self) -> None:
        """Record production backend resource closure."""
        self._closed = True

    def evalsha(self, sha: str, numkeys: int, *keys_and_args: object) -> object:
        """Return server-shaped lookup/reserve/commit/release replies."""
        _ = sha
        if numkeys == 1 and len(keys_and_args) == 1:
            self._lookup_attempts += 1
            if self._failure is not None:
                raise self._failure
            if self._stored is None:
                return [b'miss']
            return [b'hit', self._stored, 30_000]
        if numkeys == 1 and len(keys_and_args) == _RESERVE_ARGUMENTS:
            return [b'reserved']
        if numkeys == _COMMIT_KEYS and len(keys_and_args) == _COMMIT_ARGUMENTS:
            payload = keys_and_args[3]
            if not isinstance(payload, bytes):
                raise TypeError('commit payload must be bytes')
            self._stored = payload
            return [b'committed']
        if numkeys == 1 and len(keys_and_args) == _RELEASE_ARGUMENTS:
            return [b'released']
        raise RuntimeError('unexpected result protocol command')

    def refresh(self) -> None:
        """Retain the controlled protocol client."""

    def run_operation(self, operation: Callable[[_ResultProtocolClient], _T]) -> _T:
        """Run result work against the prepared deterministic client."""
        return operation(self)

    def script_load(self, script: bytes | str) -> object:
        """Reject unexpected script loading in the preloaded benchmark client.

        Raises:
            RuntimeError: Always, because benchmark scripts must already be
                loaded.
        """
        _ = script
        raise RuntimeError('benchmark scripts must already be loaded')


class _SchemaClient:
    """Answer schema and find calls for the wrapper-schema scenario."""

    __slots__ = ('_lock', '_schema', '_schema_reads')

    def __init__(self, entity_types: int, fields_per_type: int) -> None:
        """Generate the requested schema and initialize its read counter."""
        self._lock = Lock()
        self._schema = {
            f'CustomEntity{index:02d}': {
                'id': {'data_type': {'value': 'number'}, 'properties': {}},
                **{
                    f'sg_field_{field:03d}': {
                        'data_type': {'value': 'text'},
                        'properties': {},
                    }
                    for field in range(fields_per_type)
                },
            }
            for index in range(entity_types)
        }
        self._schema_reads = 0

    def find(self, *args: object, **kwargs: object) -> list[dict[str, object]]:
        """Return one deterministic entity result."""
        _ = args, kwargs
        return [{'id': 1, 'type': 'CustomEntity01'}]

    def schema_read(self, project_entity: object = None) -> object:
        """Return the generated schema and record its acquisition."""
        _ = project_entity
        with self._lock:
            self._schema_reads += 1
        return self._schema


class _StaticMetadata:
    """Provide deterministic metadata responses for the result-circuit scenario."""

    __slots__ = ()

    def change(self, changes: tuple[GenerationChange, ...]) -> None:
        """Accept an unused deterministic invalidation."""
        _ = changes

    def close(self) -> None:
        """Release no resources."""

    def flush_site(self, site_namespace: str, reason: str) -> None:
        """Accept an unused deterministic flush."""
        _ = site_namespace, reason

    def resolve(self, tags: tuple[DependencyTag, ...]) -> GenerationVector:
        """Return one stable complete vector."""
        return GenerationVector('f' * 32, tuple((tag, 1) for tag in tags))


class _ValkeyBackendConstructor(Protocol):
    """Construct the maintained backend around controlled protocol seams."""

    def __call__(
        self,
        result_client: object,
        metadata: object,
        settings: CacheSettings,
        metrics: _Metrics,
    ) -> CacheBackend:
        """Construct one production breaker owner."""
        ...


@dataclass(slots=True)
class _ReadState:
    """Collect synchronized SG calls, hit timings, and correctness violations."""

    _complete_hit_latencies_ms: list[float]
    _correctness_violations: int
    _evidence: dict[str, int]
    _lock: Lock
    _sg_calls: int

    def _add_hit(self, milliseconds: float) -> None:
        """Record one complete cache hit."""
        with self._lock:
            self._complete_hit_latencies_ms.append(milliseconds)

    def _add_sg_call(self) -> None:
        """Record one modeled SG call."""
        with self._lock:
            self._sg_calls += 1

    @classmethod
    def _create(cls) -> _ReadState:
        """Create empty synchronized read measurements."""
        return cls([], 0, {}, Lock(), 0)

    def _record(self, name: str, value: int) -> None:
        """Record one scenario-owned integer observation."""
        with self._lock:
            self._evidence[name] = value

    def _sg_call_count(self) -> int:
        """Return the synchronized number of direct SG fetches."""
        with self._lock:
            return self._sg_calls

    def _violation(self) -> None:
        """Record one stale, malformed, or over-age result."""
        with self._lock:
            self._correctness_violations += 1


@dataclass(slots=True)
class _Topology:
    """Own one benchmark topology and every closeable backend resource."""

    _cache: QueryCache
    _measured: list[_MeasuredBackend]
    _metrics: _Metrics
    _resources: list[_Closable]
    _settings: CacheSettings

    def _close(self) -> None:
        """Close each backend exactly once in reverse construction order."""
        for resource in reversed(self._resources):
            resource.close()


@dataclass(frozen=True, slots=True)
class _Workload:
    """Describe one deterministic bounded synthetic workload."""

    _keys: tuple[int, ...]
    _operations: int
    _payload_sizes: tuple[int, ...]
    _scenario: str
    _seed: int
    _workers: int


def _aggregate_memory(backends: Sequence[_MeasuredBackend]) -> dict[str, int]:
    """Aggregate actual `LocalBackend` current/peak bytes and disclose coverage."""
    result = {
        'measured_backends': 0,
        'unmeasured_backends': 0,
        'metadata_current': 0,
        'metadata_peak': 0,
        'result_current': 0,
        'result_peak': 0,
    }
    for backend in backends:
        estimate = backend._memory_estimates()
        if estimate['measured']:
            result['measured_backends'] += 1
        else:
            result['unmeasured_backends'] += 1
        for name in ('metadata_current', 'metadata_peak', 'result_current', 'result_peak'):
            result[name] += estimate[name]
    return result


def _aggregate_operations(
    backends: Sequence[_MeasuredBackend],
    metrics: _Metrics,
    capacity_rejections: int,
    evidence: Mapping[str, int],
) -> dict[str, int]:
    """Aggregate backend calls and cache outcomes without reconstructing private state."""
    aggregate = {
        'change': 0,
        'commit': 0,
        'evictions': metrics._observation_count(
            'cache_serialized_bytes',
            {'layer': 'local_l1', 'outcome': 'evicted'},
        ),
        'fills': metrics._count('cache_fill'),
        'hits': metrics._count('cache_hit'),
        'lookup': 0,
        'metadata_capacity_rejections': capacity_rejections
        + metrics._count('cache_metadata_write_rejection'),
        'release': 0,
        'reserve': 0,
        'resolve': 0,
    }
    for backend in backends:
        for name, count in backend._operation_counts().items():
            aggregate[name] += count
    aggregate.update(evidence)
    return aggregate


def _analyzer_call(
    fields: tuple[ExactValue, ...],
    filters: tuple[ExactValue, ...],
) -> ExactCall:
    """Build one complete immutable call for the production dependency analyzer."""
    return ExactCall(
        'find',
        (
            ('entity_type', 'Asset'),
            ('filters', filters),
            ('fields', fields),
            ('order', ()),
            ('filter_operator', 'all'),
            ('limit', 0),
            ('retired_only', False),
            ('page', 0),
            ('include_archived_projects', True),
            ('additional_filter_presets', None),
        ),
    )


def _broadening_plans(limits: LimitSettings) -> tuple[DependencyPlan, DependencyPlan]:
    """Drive both production dependency bounds and every accepted transition."""
    field_names = tuple(f'field_{index}' for index in range(40))
    schema = SchemaView.from_api_response(
        {
            'Asset': {
                'id': {'data_type': {'value': 'number'}, 'properties': {}},
                'project': {
                    'data_type': {'value': 'entity'},
                    'properties': {'valid_types': {'value': ['Project']}},
                },
                **{
                    name: {'data_type': {'value': 'text'}, 'properties': {}}
                    for name in field_names
                },
            }
        }
    )
    projects = tuple((('type', 'Project'), ('id', index + 1)) for index in range(40))
    tag_plan = DependencyAnalyzer().plan(
        _analyzer_call(
            ('id', *field_names),
            (
                ('id', 'in', tuple(range(1, 41))),
                ('project', 'in', projects),
            ),
        ),
        Identity('script', 'benchmark', 'broadening'),
        _SITE,
        schema,
        limits,
    )

    long_field = 'field_' + ('x' * 2_200)
    byte_schema = SchemaView.from_api_response(
        {
            'Asset': {
                'id': {'data_type': {'value': 'number'}, 'properties': {}},
                long_field: {'data_type': {'value': 'text'}, 'properties': {}},
            }
        }
    )
    byte_plan = DependencyAnalyzer().plan(
        _analyzer_call(('id', long_field), ()),
        Identity('script', 'benchmark', 'broadening'),
        _SITE,
        byte_schema,
        limits,
    )
    return tag_plan, byte_plan


def _cache_for_identity(topology: _Topology, identity: Identity) -> QueryCache:
    """Bind one exact identity to the already measured topology backends."""
    settings = topology._settings
    if settings.l2.enabled and settings.l1.enabled:
        return QueryCache(
            settings,
            identity,
            KeyBuilder(),
            l1=topology._measured[0],
            l2=topology._measured[-1],
            metrics=topology._metrics,
        )
    if settings.l2.enabled:
        return QueryCache(
            settings,
            identity,
            KeyBuilder(),
            l2=topology._measured[-1],
            metrics=topology._metrics,
        )
    return QueryCache(
        settings,
        identity,
        KeyBuilder(),
        l1=topology._measured[0],
        metrics=topology._metrics,
    )


def _call(key: int) -> ExactCall:
    """Build one exact synthetic SG call."""
    return ExactCall('find', (('entity_type', 'Asset'), ('filters', ()), ('page', key)))


def _fanout_plan(limits: LimitSettings) -> DependencyPlan:
    """Build one event-aware plan whose returned link field spans eight target types."""
    schema = SchemaView.from_api_response(
        {
            'Asset': {
                'id': {'data_type': {'value': 'number'}, 'properties': {}},
                'links': {
                    'data_type': {'value': 'multi_entity'},
                    'properties': {'valid_types': {'value': list(_FANOUT_TARGETS)}},
                },
            }
        }
    )
    return DependencyAnalyzer().plan(
        _analyzer_call(('id', 'links'), ()),
        Identity('script', 'benchmark', 'fanout'),
        _SITE,
        schema,
        limits,
    )


def _latency_model(scenario: str) -> tuple[float, dict[str, object]]:
    """Return the fixed synthetic SG latency and its honest report form."""
    seconds = 0.005 if scenario == 'hot-key-burst' else 0.0
    return seconds, {'kind': 'fixed', 'seconds': seconds}


def _local_memory(backend: LocalBackend) -> tuple[int, int]:
    """Read actual `LocalBackend` owner totals without expanding production telemetry."""
    usage = backend.storage_usage()
    return usage['result_bytes'], usage['generation_bytes'] + usage['lease_bytes']


def _next_namespace(scenario: str) -> str:
    """Return one run-unique and scenario-specific disposable prefix."""
    try:
        token = token_hex(_NAMESPACE_TOKEN_BYTES)
    except (OSError, RuntimeError):
        raise ValueError('shared namespace generation failed') from None
    if len(token) != _NAMESPACE_TOKEN_BYTES * 2 or any(
        character not in '0123456789abcdef' for character in token
    ):
        raise ValueError('shared namespace generation failed')
    return f'sgqc-bm-{scenario}-{token}'


def _percentiles(values: Sequence[float]) -> dict[str, float]:
    """Return nearest-rank milliseconds rounded to six decimal places."""
    if not values:
        return {'p50': 0.0, 'p95': 0.0, 'p99': 0.0}
    ordered = sorted(values)

    def percentile(fraction: float) -> float:
        index = max(0, ceil(len(ordered) * fraction) - 1)
        return round(ordered[index], 6)

    return {'p50': percentile(0.50), 'p95': percentile(0.95), 'p99': percentile(0.99)}


def _plan() -> DependencyPlan:
    """Build one ordinary synthetic dependency plan."""
    tags = (
        DependencyTag.site(_SITE),
        DependencyTag.authorization(_SITE),
        DependencyTag.type_membership(_SITE, 'Asset'),
    )
    return DependencyPlan('event_aware', tags, 'synthetic')


def _default_plan(_index: int) -> DependencyPlan:
    """Return the ordinary three-tag synthetic dependency plan."""
    return _plan()


def _required(name: str, environ: Mapping[str, str]) -> str:
    """Return one explicit disposable-service setting or reject the profile."""
    value = environ.get(name)
    if not value:
        raise ValueError('shared profiles require explicitly configured disposable Valkey roles')
    return value


def _boolean(name: str, environ: Mapping[str, str]) -> bool:
    """Parse one required lowercase-compatible boolean environment value."""
    value = _required(name, environ).lower()
    if value == 'true':
        return True
    if value == 'false':
        return False
    raise ValueError('shared profiles require explicitly configured disposable Valkey roles')


def _integer(name: str, environ: Mapping[str, str]) -> int:
    """Parse one required integer environment value."""
    try:
        return int(_required(name, environ))
    except ValueError:
        raise ValueError(
            'shared profiles require explicitly configured disposable Valkey roles'
        ) from None


def _role(role: str, environ: Mapping[str, str], namespace: str) -> ValkeyRoleSettings:
    """Build one standalone role from the disposable integration environment."""
    prefix = f'SG_QUERY_CACHE_TEST_{role.upper()}'
    return ValkeyRoleSettings(
        _required(f'{prefix}_HOST', environ),
        f'{prefix}_PASSWORD',
        port=_integer(f'{prefix}_PORT', environ),
        database=_integer(f'{prefix}_DATABASE', environ),
        tls=_boolean(f'{prefix}_TLS', environ),
        username=environ.get(f'{prefix}_USERNAME') or None,
        key_prefix=namespace,
    )


def _run_metadata_churn(
    profile: str,
    metrics: _Metrics,
    workload: _Workload,
    state: _ReadState,
    environ: Mapping[str, str],
) -> tuple[_Topology, int]:
    """Exercise scoped expiration, recreation, pressure, and recovery."""
    if profile != 'local':
        raise ValueError(f'metadata-churn does not support the {profile} profile')
    _ = environ

    clock = _Clock()
    settings = CacheSettings(
        site_namespace=_SITE,
        l1=L1Settings(
            result_budget_bytes=_MAX_LOCAL_RESULT_BYTES,
            metadata_budget_bytes=32_768,
        ),
        limits=LimitSettings(scoped_generation_retention_seconds=0.1),
    )
    backend = LocalBackend(settings, clock, metrics)
    measured = _MeasuredBackend(backend, memory_probe=lambda: _local_memory(backend))
    cache = QueryCache(
        settings,
        Identity('script', 'benchmark', 'synthetic'),
        KeyBuilder(),
        l1=measured,
        metrics=metrics,
    )
    topology = _Topology(cache, [measured], metrics, [measured], settings)
    tags = tuple(
        DependencyTag.entity(_SITE, 'Asset', index + 1) for index in range(workload._operations)
    )
    first = measured.resolve((tags[0],)).entries[0][1]
    capacity_rejections = 0
    for tag in tags[1:]:
        try:
            measured.resolve((tag,))
        except BackendCapacity:
            capacity_rejections += 1
            break
    clock._advance(0.2)
    recreated = measured.resolve((tags[0],)).entries[0][1]
    if recreated == first:
        state._violation()
    return topology, capacity_rejections


def _run_wrapper_schema(topology: _Topology, workload: _Workload, state: _ReadState) -> None:
    """Drive entity reads through the wrapper over a production-shaped schema."""
    client = _SchemaClient(120, 100)
    wrapper = CachedShotgun(
        client,
        topology._cache,
        ClientContext(
            'https://benchmark.example',
            _SITE,
            localized=False,
            convert_datetimes_to_utc=True,
            records_per_page=500,
        ),
    )

    for index in range(4):
        wrapper.find(
            'CustomEntity01',
            [['sg_field_001', 'is', f'x{index}']],
            ['sg_field_002'],
        )

    def read(index: int) -> None:
        started = monotonic()
        wrapper.find(
            'CustomEntity01',
            [['sg_field_001', 'is', f'x{index % 4}']],
            ['sg_field_002'],
        )
        state._add_hit((monotonic() - started) * 1000)

    with ThreadPoolExecutor(max_workers=workload._workers) as executor:
        list(executor.map(read, range(workload._operations)))
    state._record('complete_hit_samples', len(state._complete_hit_latencies_ms))
    state._record('schema_reads', client._schema_reads)


def _settings(profile: str, scenario: str, environ: Mapping[str, str]) -> CacheSettings:
    """Build bounded settings for one local or explicitly gated Valkey profile."""
    vector_bytes = 4_096 if scenario == 'link-fanout-events' else 2_048
    limits = LimitSettings(
        max_dependency_vector_bytes=vector_bytes,
        scoped_generation_retention_seconds=0.1,
    )
    l1 = L1Settings(
        enabled=profile != 'shared',
        result_budget_bytes=_MAX_LOCAL_RESULT_BYTES,
        metadata_budget_bytes=_MAX_LOCAL_METADATA_BYTES,
    )
    if profile == 'local':
        return CacheSettings(site_namespace=_SITE, l1=l1, limits=limits)
    if environ.get('SG_QUERY_CACHE_REQUIRE_INTEGRATION') != '1':
        raise ValueError('shared profiles require explicitly configured disposable Valkey roles')
    namespace = _next_namespace(scenario)
    return CacheSettings(
        site_namespace=_SITE,
        l1=l1,
        l2=L2Settings(
            enabled=True,
            valkey=ValkeySettings(
                result=_role('result', environ, namespace),
                metadata=_role('metadata', environ, namespace),
            ),
        ),
        limits=limits,
    )


def _topology(
    profile: str,
    scenario: str,
    metrics: _Metrics,
    environ: Mapping[str, str],
) -> _Topology:
    """Construct one supported topology over actual completed backend APIs."""
    settings = _settings(profile, scenario, environ)
    identity = Identity('script', 'benchmark', 'synthetic')
    if profile == 'local':
        backend = LocalBackend(settings, monotonic, metrics)
        measured = _MeasuredBackend(backend, memory_probe=lambda: _local_memory(backend))
        cache = QueryCache(settings, identity, KeyBuilder(), l1=measured, metrics=metrics)
        return _Topology(cache, [measured], metrics, [measured], settings)

    valkey_module = import_module('sg_query_cache.valkey')
    valkey_backend = cast('type[ValkeyBackend]', vars(valkey_module)['ValkeyBackend'])

    shared = _MeasuredBackend(valkey_backend.from_settings(settings, environ, metrics))
    if profile == 'shared':
        cache = QueryCache(settings, identity, KeyBuilder(), l2=shared, metrics=metrics)
        return _Topology(cache, [shared], metrics, [shared], settings)
    local_backend = LocalBackend(settings, monotonic, metrics)
    local = _MeasuredBackend(
        local_backend,
        memory_probe=lambda: _local_memory(local_backend),
    )
    cache = QueryCache(settings, identity, KeyBuilder(), l1=local, l2=shared, metrics=metrics)
    return _Topology(cache, [local, shared], metrics, [local, shared], settings)


def _validate_workload_size(scenario: str, operations: int, workers: int) -> None:
    """Reject unsafe or semantically incomplete caller-controlled sizes."""
    if operations <= 0 or workers <= 0:
        raise ValueError('benchmark operations and workers must be positive')
    if operations > _MAX_OPERATIONS:
        raise ValueError('benchmark operations must not exceed 10,000')
    if workers > _MAX_WORKERS:
        raise ValueError('benchmark workers must not exceed 64')
    (
        minimum_operations,
        fixed_operations,
        minimum_workers,
        fixed_workers,
        parallel_offset,
    ) = _SCENARIO_CONTRACTS[scenario]
    if fixed_operations is not None and operations != fixed_operations:
        raise ValueError(f'{scenario} requires exactly {fixed_operations:,} operations')
    if operations < minimum_operations:
        raise ValueError(f'{scenario} requires at least {minimum_operations:,} operations')
    if fixed_workers is not None and workers != fixed_workers:
        raise ValueError(f'{scenario} requires exactly {fixed_workers:,} worker')
    if workers < minimum_workers:
        raise ValueError(f'{scenario} requires at least {minimum_workers:,} workers')
    if parallel_offset is not None:
        meaningful_workers = max(1, operations - parallel_offset)
        if workers > meaningful_workers:
            raise ValueError(
                f'{scenario} workers must not exceed {meaningful_workers:,} '
                f'for {operations:,} operations'
            )


def _build_workload(
    scenario: str,
    *,
    seed: int,
    operations: int | None = None,
    workers: int | None = None,
) -> _Workload:
    """Build one deterministic scenario plan from a local random generator."""
    if scenario not in _SCENARIOS:
        raise ValueError('unknown benchmark scenario')
    operation_count = _DEFAULT_OPERATIONS[scenario] if operations is None else operations
    worker_count = _DEFAULT_WORKERS[scenario] if workers is None else workers
    _validate_workload_size(scenario, operation_count, worker_count)
    generator = _Deterministic(seed)
    if scenario == 'hot-key-burst':
        keys = (0,) * operation_count
        payload_sizes = (512,) * operation_count
    elif scenario == 'one-hit-scan':
        keys = tuple(range(operation_count))
        payload_sizes = (16_384,) * operation_count
    elif scenario == 'mixed-values':
        trailing_operations = operation_count - 4
        keys = (0, 0, 1, 1, *(generator._range(2, 8) for _ in range(trailing_operations)))
        trailing_sizes = tuple(
            _NEAR_CAP_PAYLOAD_BYTES if generator._range(0, 4) == 0 else generator._range(32, 513)
            for _ in range(trailing_operations)
        )
        payload_sizes = (
            _NEAR_CAP_PAYLOAD_BYTES,
            _NEAR_CAP_PAYLOAD_BYTES,
            64,
            64,
            *trailing_sizes,
        )
    elif scenario == 'hierarchical-fanout':
        keys = tuple(
            generator._range(0, max(2, worker_count // 2)) for _ in range(operation_count)
        )
        payload_sizes = tuple(generator._range(128, 1_025) for _ in range(operation_count))
    else:
        keys = tuple(range(operation_count))
        payload_sizes = tuple(generator._range(64, 513) for _ in range(operation_count))
    return _Workload(keys, operation_count, payload_sizes, scenario, seed, worker_count)


def _parse_args(argv: Sequence[str] | None = None) -> _Arguments:
    """Parse one closed benchmark command without opening a backend."""
    parser = argparse.ArgumentParser(prog='query_cache.py')
    parser.add_argument('--profile', choices=('local', 'shared', 'hierarchical'), required=True)
    parser.add_argument('--scenario', choices=_SCENARIOS, required=True)
    parser.add_argument('--json', action='store_true', required=True)
    parser.add_argument('--seed', type=int, default=1729)
    parser.add_argument('--operations', type=int, help='positive count up to 10,000')
    parser.add_argument('--workers', type=int, help='positive count up to 64')
    return parser.parse_args(argv, namespace=_Arguments())


@dataclass(frozen=True, slots=True)
class _ReadOptions:
    """Configure one synthetic SG fetch and correctness check."""

    _payload_size: int = 128
    _per_call_age: float | None = None
    _sg_latency_seconds: float = 0.0
    _version: int = 1


def _read_once(
    cache: QueryCache,
    call: ExactCall,
    plan: DependencyPlan,
    state: _ReadState,
    options: _ReadOptions,
) -> None:
    """Run one cache read and validate its expected scripted version."""
    fetched = False

    def fetch() -> dict[str, object]:
        nonlocal fetched
        fetched = True
        state._add_sg_call()
        if options._sg_latency_seconds > 0:
            sleep(options._sg_latency_seconds)
        return {
            'payload': bytes(options._payload_size),
            'version': options._version,
        }

    started_at = monotonic()
    value = cache.read(
        call,
        plan,
        fetch,
        _CLIENT_OPTIONS,
        per_call_age=options._per_call_age,
    )
    elapsed_ms = (monotonic() - started_at) * 1_000
    if not fetched:
        state._add_hit(elapsed_ms)
    if not isinstance(value, dict):
        state._violation()
        return
    result = cast('dict[str, object]', value)
    observed_version = result.get('version')
    if observed_version != options._version:
        state._violation()


def _normal_hit_backend_phases(profile: str, environ: Mapping[str, str]) -> int:
    """Measure one warmed hit on an independent backend topology."""
    metrics = _Metrics()
    state = _ReadState._create()
    topology: _Topology | None = None
    try:
        topology = _topology(profile, 'normal-hit', metrics, environ)
        call = _call(0)
        plan = _default_plan(0)
        options = _ReadOptions()
        _read_once(topology._cache, call, plan, state, options)
        before = tuple(backend._operation_counts() for backend in topology._measured)
        fetched_before_hit = state._sg_call_count()
        _read_once(topology._cache, call, plan, state, options)
        phases = sum(
            count - owner_before[name]
            for backend, owner_before in zip(topology._measured, before, strict=True)
            for name, count in backend._operation_counts().items()
        )
        if (
            fetched_before_hit != 1
            or state._sg_call_count() != fetched_before_hit
            or len(state._complete_hit_latencies_ms) != 1
            or state._correctness_violations != 0
            or phases <= 0
            or (profile == 'shared' and phases != _SHARED_NORMAL_HIT_PHASES)
        ):
            raise ValueError('normal-hit backend phase measurement failed')
    except (BackendCapacity, BackendProtocolError, BackendUnavailable, StaleIncarnation):
        raise ValueError('normal-hit backend phase measurement failed') from None
    else:
        return phases
    finally:
        if topology is not None:
            try:
                topology._close()
            except (BackendCapacity, BackendProtocolError, BackendUnavailable, StaleIncarnation):
                raise ValueError('normal-hit backend phase measurement failed') from None


def _parallel_reads(
    cache_for_index: Callable[[int], QueryCache],
    workload: _Workload,
    state: _ReadState,
    options: _ReadOptions | None = None,
    plan_for_index: Callable[[int], DependencyPlan] | None = None,
) -> None:
    """Run one deterministic read sequence through a bounded worker pool."""
    planner: Callable[[int], DependencyPlan] = plan_for_index or _default_plan
    selected_options = options or _ReadOptions()
    with ThreadPoolExecutor(max_workers=workload._workers) as executor:
        futures = [
            executor.submit(
                _read_once,
                cache_for_index(index),
                _call(workload._keys[index]),
                planner(index),
                state,
                replace(selected_options, _payload_size=workload._payload_sizes[index]),
            )
            for index in range(workload._operations)
        ]
        for future in futures:
            future.result()


@dataclass(frozen=True, slots=True)
class _BenchmarkOptions:
    """Hold optional benchmark sizing and deterministic seed inputs."""

    _operations: int | None = None
    _seed: int = 1729
    _workers: int | None = None


def _run_dependency_broadening(
    topology: _Topology,
    workload: _Workload,
    state: _ReadState,
) -> None:
    """Analyze real over-bound calls and exercise their bounded production plans."""
    plans = _broadening_plans(topology._settings.limits)
    for plan in plans:
        for step in plan.broadening_steps:
            state._record(f'broadening:{step}', state._evidence.get(f'broadening:{step}', 0) + 1)
    state._record('analyzer_max_output_tags', max(len(plan.tags) for plan in plans))
    _parallel_reads(
        lambda _index: topology._cache,
        workload,
        state,
        plan_for_index=lambda index: plans[index % len(plans)],
    )


def _run_event_storm(topology: _Topology, workload: _Workload, state: _ReadState) -> None:
    """Apply 10,000 mostly unmaterialized scoped changes through the real backend."""
    backend = topology._measured[-1]

    def change(index: int) -> None:
        backend.change(
            (GenerationChange(DependencyTag.entity(_SITE, 'Asset', workload._keys[index] + 1)),)
        )

    with ThreadPoolExecutor(max_workers=workload._workers) as executor:
        list(executor.map(change, range(workload._operations)))

    _read_once(
        topology._cache,
        _call(0),
        _plan(),
        state,
        _ReadOptions(),
    )
    topology._cache.invalidate((GenerationChange(DependencyTag.type_membership(_SITE, 'Asset')),))
    _read_once(
        topology._cache,
        _call(0),
        _plan(),
        state,
        _ReadOptions(_version=2),
    )


def _run_failure_recovery(
    topology: _Topology,
    workload: _Workload,
    state: _ReadState,
) -> None:
    """Drive the maintained Valkey result circuit through timeout, open, and restart."""
    _ = workload
    valkey_module = import_module('sg_query_cache.valkey')
    exception_module = import_module('valkey.exceptions')
    constructor = cast('_ValkeyBackendConstructor', vars(valkey_module)['ValkeyBackend'])
    timeout_type = vars(exception_module)['TimeoutError']
    if not isinstance(timeout_type, type) or not issubclass(timeout_type, BaseException):
        raise TypeError('Valkey timeout exception is unavailable')

    settings = CacheSettings(
        site_namespace=_SITE,
        l1=L1Settings(enabled=False),
        l2=L2Settings(enabled=True),
        limits=LimitSettings(circuit_failure_threshold=_BREAKER_FAILURES),
    )
    failing_client = _ResultProtocolClient(timeout_type('controlled timeout'))
    failing_backend = constructor(
        failing_client,
        _StaticMetadata(),
        settings,
        topology._metrics,
    )
    failing_cache = QueryCache(
        settings,
        Identity('script', 'benchmark', 'failure-recovery'),
        l2=failing_backend,
        metrics=topology._metrics,
    )
    failure_key = 91_001
    call = _call(failure_key)
    options = _ReadOptions(_payload_size=128)
    for _index in range(_BREAKER_FAILURES):
        _read_once(failing_cache, call, _plan(), state, options)
    before_open = state._sg_call_count()
    attempts_before_open = failing_client._lookup_attempts
    _read_once(failing_cache, call, _plan(), state, options)
    breaker_bypasses = int(
        state._sg_call_count() - before_open == 1
        and failing_client._lookup_attempts == attempts_before_open
    )
    failing_cache.close()

    healthy_client = _ResultProtocolClient()
    healthy_backend = constructor(
        healthy_client,
        _StaticMetadata(),
        settings,
        topology._metrics,
    )
    healthy_cache = QueryCache(
        settings,
        Identity('script', 'benchmark', 'failure-recovery'),
        l2=healthy_backend,
        metrics=topology._metrics,
    )
    before_recovery = state._sg_call_count()
    try:
        _read_once(healthy_cache, call, _plan(), state, options)
        _read_once(healthy_cache, call, _plan(), state, options)
        recovery_hits = int(state._sg_call_count() - before_recovery == 1)
        remaining = workload._operations - (_BREAKER_FAILURES + 3)
        if remaining > 0:
            recovered_workload = replace(
                workload,
                _operations=remaining,
                _keys=(failure_key,) * remaining,
                _payload_sizes=(options._payload_size,) * remaining,
            )
            _parallel_reads(lambda _index: healthy_cache, recovered_workload, state, options)
    finally:
        healthy_cache.close()

    state._record('failure_timeouts', failing_client._lookup_attempts)
    state._record('breaker_open_bypasses', breaker_bypasses)
    state._record('backend_restarts', int(failing_client._closed))
    state._record('recovery_hits', recovery_hits)
    if (
        failing_client._lookup_attempts != _BREAKER_FAILURES
        or breaker_bypasses != 1
        or not failing_client._closed
        or recovery_hits != 1
    ):
        state._violation()


def _run_hierarchical_fanout(
    topology: _Topology,
    workload: _Workload,
    state: _ReadState,
    metrics: _Metrics,
) -> None:
    """Run many independent L1s against one actual shared L2 backend."""
    if topology._settings.l2.enabled:
        shared = topology._measured[-1]
        settings = CacheSettings(
            site_namespace=_SITE,
            l2=topology._settings.l2,
            limits=topology._settings.limits,
        )
    else:
        settings = CacheSettings(site_namespace=_SITE, l2=L2Settings(enabled=True))
        raw_shared = LocalBackend(settings, monotonic, metrics)
        shared = _MeasuredBackend(
            raw_shared,
            memory_probe=lambda: _local_memory(raw_shared),
        )
        topology._measured.append(shared)
        topology._resources.append(shared)
    caches: list[QueryCache] = []
    local_owners: list[_MeasuredBackend] = []
    for _index in range(workload._workers):
        local_backend = LocalBackend(settings, monotonic, metrics)
        local = _MeasuredBackend(
            local_backend,
            memory_probe=lambda backend=local_backend: _local_memory(backend),
        )
        topology._measured.append(local)
        topology._resources.append(local)
        local_owners.append(local)
        caches.append(
            QueryCache(
                settings,
                Identity('script', 'benchmark', 'fanout'),
                KeyBuilder(),
                l1=local,
                l2=shared,
                metrics=metrics,
            )
        )
    _parallel_reads(lambda index: caches[index % len(caches)], workload, state)
    participating_l1s = sum(owner._operation_counts()['resolve'] > 0 for owner in local_owners)
    shared_operations = sum(shared._operation_counts().values())
    state._record('fanout_participating_l1s', participating_l1s)
    state._record('fanout_shared_l2_instances', 1)
    state._record('fanout_shared_l2_operations', shared_operations)
    if participating_l1s != len(local_owners) or shared_operations <= 0:
        state._violation()


def _run_identity_cardinality(
    topology: _Topology,
    workload: _Workload,
    state: _ReadState,
) -> None:
    """Run more than 1,000 exact identities through one shared byte budget."""
    caches = [
        _cache_for_identity(
            topology,
            Identity('script', 'benchmark', f'identity-{index}'),
        )
        for index in range(workload._operations)
    ]
    state._record('backend_instances', len(topology._measured))
    state._record('identity_count', len(caches))
    _parallel_reads(caches.__getitem__, workload, state)


def _run_link_fanout_events(
    topology: _Topology,
    workload: _Workload,
    state: _ReadState,
    metrics: _Metrics,
) -> None:
    """Read one link-returning plan while every target type churns in turn."""
    plan = _fanout_plan(topology._settings.limits)
    state._record('analyzer_max_output_tags', len(plan.tags))
    state._record(
        'max_dependency_vector_bytes',
        topology._settings.limits.max_dependency_vector_bytes,
    )
    bumps = 0
    hits = 0

    for index in range(workload._operations):
        before = metrics._count('cache_hit')
        _read_once(topology._cache, _call(0), plan, state, _ReadOptions(_version=bumps + 1))
        hits += metrics._count('cache_hit') - before
        if (index + 1) % _FANOUT_EVENT_INTERVAL == 0:
            target = _FANOUT_TARGETS[bumps % len(_FANOUT_TARGETS)]
            topology._cache.invalidate(
                (GenerationChange(DependencyTag.type_value(_SITE, target)),)
            )
            bumps += 1

    state._record('link_fanout_reads', workload._operations)
    state._record('link_fanout_hits', hits)
    state._record(
        'type_value_invalidations',
        metrics._count('cache_invalidation', {'tag_kind': 'type_value'}),
    )


def _run_maximum_age_check(state: _ReadState, metrics: _Metrics) -> None:
    """Prove the selected short per-call age expires on the actual local owner clock."""
    clock = _Clock()
    settings = CacheSettings(site_namespace=_SITE)
    backend = LocalBackend(settings, clock, metrics)
    cache = QueryCache(
        settings,
        Identity('script', 'benchmark', 'maximum-age'),
        l1=backend,
        metrics=metrics,
    )
    options = _ReadOptions(_payload_size=64, _per_call_age=0.05)
    before = state._sg_call_count()
    try:
        _read_once(cache, _call(90_001), _plan(), state, options)
        _read_once(cache, _call(90_001), _plan(), state, options)
        clock._advance(0.051)
        _read_once(cache, _call(90_001), _plan(), state, replace(options, _version=2))
    finally:
        cache.close()
    refetches = state._sg_call_count() - before - 1
    state._record('maximum_age_milliseconds', 50)
    state._record('maximum_age_refetches', refetches)
    if refetches != 1:
        state._violation()


def _run_mixed_values(topology: _Topology, workload: _Workload, state: _ReadState) -> None:
    """Commit and hit independently observed small and near-cap encoded results."""

    def exercise_pair(label: str, start: int) -> None:
        starting_commits = [
            len(backend._committed_payload_sizes()) for backend in topology._measured
        ]
        starting_hits = topology._metrics._count('cache_hit')
        starting_sg_calls = state._sg_call_count()
        for index in range(start, start + 2):
            _read_once(
                topology._cache,
                _call(workload._keys[index]),
                _plan(),
                state,
                _ReadOptions(_payload_size=workload._payload_sizes[index]),
            )
        committed_sizes = [
            size
            for backend, offset in zip(topology._measured, starting_commits, strict=True)
            for size in backend._committed_payload_sizes()[offset:]
        ]
        commits = len(committed_sizes)
        hits = topology._metrics._count('cache_hit') - starting_hits
        sg_calls = state._sg_call_count() - starting_sg_calls
        encoded_bytes = max(committed_sizes, default=0)
        state._record(f'mixed_{label}_commits', commits)
        state._record(f'mixed_{label}_encoded_bytes', encoded_bytes)
        state._record(f'mixed_{label}_hits', hits)
        if commits < 1 or hits != 1 or sg_calls != 1:
            state._violation()

    exercise_pair('near_cap', 0)
    exercise_pair('small', 2)
    near_cap_bytes = state._evidence['mixed_near_cap_encoded_bytes']
    small_bytes = state._evidence['mixed_small_encoded_bytes']
    if (
        near_cap_bytes < _MIN_NEAR_CAP_ENCODED_BYTES
        or near_cap_bytes > topology._settings.limits.max_serialized_result_bytes
        or small_bytes <= 0
        or small_bytes >= _MAX_SMALL_ENCODED_BYTES
    ):
        state._violation()

    remaining = workload._operations - 4
    if remaining > 0:
        trailing = replace(
            workload,
            _keys=workload._keys[4:],
            _operations=remaining,
            _payload_sizes=workload._payload_sizes[4:],
        )
        _parallel_reads(lambda _index: topology._cache, trailing, state)


def _run_one_hit_scan(topology: _Topology, workload: _Workload, state: _ReadState) -> None:
    """Warm one key, scan unique keys, then prove eviction causes a safe refetch."""
    for _index in range(2):
        _read_once(topology._cache, _call(0), _plan(), state, _ReadOptions(_payload_size=512))
    _parallel_reads(lambda _index: topology._cache, workload, state)
    before = state._sg_call_count()
    _read_once(
        topology._cache,
        _call(0),
        _plan(),
        state,
        _ReadOptions(_payload_size=512),
    )
    refetches = state._sg_call_count() - before
    state._record('hot_key_refetches', refetches)
    if refetches != 1:
        state._violation()
    _run_maximum_age_check(state, topology._metrics)


def _run_identity_or_scan(
    topology: _Topology,
    workload: _Workload,
    state: _ReadState,
) -> None:
    """Dispatch one existing cardinality or scan scenario."""
    if workload._scenario == 'identity-cardinality':
        _run_identity_cardinality(topology, workload, state)
    else:
        _run_one_hit_scan(topology, workload, state)


def _execute_scenario(
    profile: str,
    workload: _Workload,
    metrics: _Metrics,
    state: _ReadState,
    environ: Mapping[str, str],
) -> tuple[_Topology, int]:
    """Dispatch one closed scenario to its actual cache/backend workload."""
    if workload._scenario == 'metadata-churn':
        return _run_metadata_churn(profile, metrics, workload, state, environ)
    topology = _topology(profile, workload._scenario, metrics, environ)
    scenario = workload._scenario
    if scenario in {'identity-cardinality', 'one-hit-scan'}:
        _run_identity_or_scan(topology, workload, state)
    elif scenario == 'dependency-broadening':
        _run_dependency_broadening(topology, workload, state)
    elif scenario == 'event-storm':
        _run_event_storm(topology, workload, state)
    elif scenario == 'failure-recovery':
        _run_failure_recovery(topology, workload, state)
    elif scenario == 'hierarchical-fanout':
        _run_hierarchical_fanout(topology, workload, state, metrics)
    elif scenario == 'link-fanout-events':
        _run_link_fanout_events(topology, workload, state, metrics)
    elif scenario == 'mixed-values':
        _run_mixed_values(topology, workload, state)
    elif scenario == 'wrapper-schema':
        _run_wrapper_schema(topology, workload, state)
    else:
        sg_latency_seconds, _model = _latency_model(scenario)
        _parallel_reads(
            lambda _index: topology._cache,
            workload,
            state,
            _ReadOptions(_sg_latency_seconds=sg_latency_seconds),
        )
    return topology, 0


def _run_benchmark(
    profile: str,
    scenario: str,
    *,
    options: _BenchmarkOptions | None = None,
    environ: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Run one scenario and return its complete JSON-compatible report."""
    if profile not in {'hierarchical', 'local', 'shared'}:
        raise ValueError('unknown benchmark profile')
    if profile != 'local' and scenario in _UNSUPPORTED_SHARED_SCENARIOS:
        raise ValueError(f'{scenario} does not support the {profile} profile')
    selected_options = options or _BenchmarkOptions()
    workload = _build_workload(
        scenario,
        seed=selected_options._seed,
        operations=selected_options._operations,
        workers=selected_options._workers,
    )
    selected_environ = os.environ if environ is None else environ
    metrics = _Metrics()
    state = _ReadState._create()
    started_at = monotonic()
    topology: _Topology | None = None
    _sg_latency_seconds, latency_model = _latency_model(scenario)
    try:
        topology, capacity_rejections = _execute_scenario(
            profile,
            workload,
            metrics,
            state,
            selected_environ,
        )
        elapsed = monotonic() - started_at
        avoided = metrics._count('cache_sg_avoided_calls')
        return {
            'profile': profile,
            'scenario': scenario,
            'seed': selected_options._seed,
            'workers': workload._workers,
            'operations': workload._operations,
            'payload_bytes': sum(workload._payload_sizes),
            'tag_count': state._evidence.get('analyzer_max_output_tags', 3),
            'sg_latency_model': latency_model,
            'cache_operations': _aggregate_operations(
                topology._measured,
                metrics,
                capacity_rejections,
                state._evidence,
            ),
            'sg_calls_avoided': avoided,
            'throughput_ops_per_second': round(workload._operations / max(elapsed, 1e-9), 6),
            'complete_hit_latency_ms': _percentiles(state._complete_hit_latencies_ms),
            'vector_resolution_latency_ms': _percentiles(
                [
                    latency
                    for backend in topology._measured
                    for latency in backend._vector_latencies()
                ]
            ),
            'memory_estimates_bytes': _aggregate_memory(topology._measured),
            'normal_hit_backend_phases': _normal_hit_backend_phases(profile, selected_environ),
            'lease_rejections': metrics._count('cache_lease_rejection'),
            'correctness_violations': state._correctness_violations,
        }
    finally:
        if topology is not None:
            topology._close()


def main(argv: Sequence[str] | None = None) -> int:
    """Run the selected benchmark and return a correctness-sensitive status.

    Args:
        argv: Arguments excluding the executable, or `None` for process
            arguments.

    Returns:
        Zero for a correct run, one for correctness violations, or two for
        invalid setup.
    """
    arguments = _parse_args(argv)
    try:
        report = _run_benchmark(
            arguments.profile,
            arguments.scenario,
            options=_BenchmarkOptions(
                _operations=arguments.operations,
                _seed=arguments.seed,
                _workers=arguments.workers,
            ),
        )
    except (ImportError, OSError, TypeError, ValueError) as error:
        print(f'query-cache benchmark: {error}', file=sys.stderr)
        return 2
    print(json.dumps(report, separators=(',', ':')))
    violations = report['correctness_violations']
    return 1 if isinstance(violations, int) and violations else 0


if __name__ == '__main__':
    raise SystemExit(main())
