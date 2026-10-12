"""Provide a closed, low-cardinality metrics boundary."""

from collections.abc import Mapping
from math import isfinite
from typing import Protocol

from sg_query_cache._keys import METRICS_KEY_MIN_BYTES, KeyBuilder

_ALLOWED_LABELS = frozenset(
    {
        'backend_operation',
        'classification',
        'layer',
        'method',
        'outcome',
        'reason',
        'tag_kind',
    }
)
_ALLOWED_METRIC_NAMES = frozenset(
    {
        'backend_failure',
        'backend_latency_seconds',
        'cache_bypass',
        'cache_collapse',
        'cache_commit',
        'cache_dependency_broadening',
        'cache_dependency_vector_bytes',
        'cache_dependency_vector_tags',
        'cache_fill',
        'cache_hit',
        'cache_incarnation',
        'cache_invalidation',
        'cache_lease_rejection',
        'cache_lease_timeout',
        'cache_metadata_recovery_seconds',
        'cache_metadata_write_rejection',
        'cache_miss',
        'cache_query_classification',
        'cache_schema_view',
        'cache_scoped_generation',
        'cache_serialized_bytes',
        'cache_sg_avoided_calls',
        'cache_sg_avoided_seconds',
        'cache_site_flush',
        'cache_ttl_selection',
        'cache_url_age_cap',
        'cache_vector_resolution',
        'cache_wait_seconds',
        'generation_metadata_bytes',
    }
)
_CORRELATION_LABEL = 'exact_call_digest'
_DIGEST_LENGTH = 64
_FORBIDDEN_LABELS = frozenset({'credential', 'fields', 'filters', 'payload', 'query', 'url'})
_MAX_LABEL_VALUE_LENGTH = 64


class MetricsSink(Protocol):
    """Receive validated counter and observation events from cache components."""

    def increment(self, name: str, labels: Mapping[str, str]) -> None:
        """Increment one named metric with low-cardinality labels.

        Args:
            name: Documented metric name.
            labels: Validated low-cardinality label mapping.
        """
        ...

    def observe(self, name: str, value: float, labels: Mapping[str, str]) -> None:
        """Record one finite value for a named metric.

        Args:
            name: Documented metric name.
            value: Finite measurement.
            labels: Validated low-cardinality label mapping.
        """
        ...


class NullMetrics:
    """Discard metrics without allocating storage or invoking another component."""

    __slots__ = ()

    def increment(self, name: str, labels: Mapping[str, str]) -> None:
        """Discard one counter event.

        Args:
            name: Metric name, intentionally ignored.
            labels: Metric labels, intentionally ignored.
        """
        _ = name, labels

    def observe(self, name: str, value: float, labels: Mapping[str, str]) -> None:
        """Discard one observation event.

        Args:
            name: Metric name, intentionally ignored.
            value: Measurement, intentionally ignored.
            labels: Metric labels, intentionally ignored.
        """
        _ = name, value, labels


class SafeMetrics:
    """Validate metric names and labels before forwarding events to a sink.

    The optional metrics key is retained only as immutable bytes used for
    package-internal exact-call correlation. It never participates in cache
    keys, exceptions, representations, or labels.
    """

    __slots__ = ('_metrics_key', '_sink')

    def __init__(self, sink: MetricsSink, metrics_key: bytes | None = None) -> None:
        """Retain the destination sink and an immutable optional correlation key.

        Args:
            sink: Destination metrics implementation.
            metrics_key: Optional key of at least 32 bytes enabling exact-call
                digest labels.

        Raises:
            ValueError: The optional metrics key is shorter than 32 bytes.
        """
        if metrics_key is not None and len(metrics_key) < METRICS_KEY_MIN_BYTES:
            raise ValueError('metrics correlation key must contain at least 32 bytes')
        self._sink = sink
        self._metrics_key = None if metrics_key is None else bytes(metrics_key)

    def _validated_labels(self, labels: Mapping[str, str]) -> dict[str, str]:
        """Return a safe copy of candidate metric labels."""
        safe: dict[str, str] = {}
        for name, value in labels.items():
            if name in _FORBIDDEN_LABELS or name not in _ALLOWED_LABELS | {_CORRELATION_LABEL}:
                raise ValueError('metric label is not in the safe allowlist')
            if name == _CORRELATION_LABEL:
                if self._metrics_key is None:
                    continue
                if len(value) != _DIGEST_LENGTH or any(
                    character not in '0123456789abcdef' for character in value
                ):
                    raise ValueError(
                        'exact-call digest must be lowercase HMAC-SHA-256 hexadecimal'
                    )
            elif not value or len(value) > _MAX_LABEL_VALUE_LENGTH or not value.isascii():
                raise ValueError('metric label value must be short, nonempty ASCII')
            safe[name] = value
        return safe

    def correlation_digest(self, key_builder: KeyBuilder, base_key: str) -> str | None:
        """Return the configured keyed digest for one opaque base key.

        Args:
            key_builder: Exact-call key builder that owns digest construction.
            base_key: Already validated opaque exact-call base key.

        Returns:
            A keyed lowercase digest, or `None` when correlation is disabled.
        """
        if self._metrics_key is None:
            return None
        return key_builder.metrics_digest(base_key, self._metrics_key)

    def increment(self, name: str, labels: Mapping[str, str]) -> None:
        """Validate and forward one counter event.

        Args:
            name: Documented metric name.
            labels: Candidate low-cardinality labels.

        Raises:
            ValueError: The name or any label is outside the safe vocabulary.
        """
        self._sink.increment(_validated_name(name), self._validated_labels(labels))

    def observe(self, name: str, value: float, labels: Mapping[str, str]) -> None:
        """Validate and forward one numeric observation.

        Args:
            name: Documented metric name.
            value: Finite measurement.
            labels: Candidate low-cardinality labels.

        Raises:
            ValueError: The name, value, or any label is outside the safe
                vocabulary.
        """
        if not isfinite(value):
            raise ValueError('metric observation must be finite')
        self._sink.observe(_validated_name(name), value, self._validated_labels(labels))


def _validated_name(name: str) -> str:
    """Return one documented metric name or reject it."""
    if name not in _ALLOWED_METRIC_NAMES:
        raise ValueError('metric name is not documented')
    return name
