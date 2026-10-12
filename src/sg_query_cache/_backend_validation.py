"""Validate values shared by synchronous cache backend boundaries."""

from contextlib import suppress
from typing import cast

from sg_query_cache._backend import BackendProtocolError
from sg_query_cache._keys import KeyBuilder, NormalizationError
from sg_query_cache._model import DependencyTag, FillLease, GenerationChange

_MAX_FLUSH_REASON_LENGTH = 64
_METRIC_FLUSH_REASONS = frozenset({'daemon_recovery', 'daemon_setup'})


def _dependency_tags(value: object) -> tuple[DependencyTag, ...]:
    """Require a tuple of dependency tags, raising `BackendProtocolError` otherwise."""
    if not isinstance(value, tuple) or not all(
        isinstance(tag, DependencyTag) for tag in cast('tuple[object, ...]', value)
    ):
        raise BackendProtocolError('dependency vector must be an immutable tag tuple')
    return cast('tuple[DependencyTag, ...]', value)


def encoded_payload(value: object) -> bytes:
    """Require immutable bytes, raising `BackendProtocolError` for other values."""
    if not isinstance(value, bytes):
        raise BackendProtocolError('commit payload must be encoded bytes')
    return value


def fill_lease(value: object) -> FillLease:
    """Require a `FillLease`, raising `BackendProtocolError` for other values."""
    if not isinstance(value, FillLease):
        raise BackendProtocolError('operation requires a fill lease')
    return value


def flush_reason(value: object) -> None:
    """Require 1-64 ASCII characters, raising `BackendProtocolError` otherwise."""
    if (
        not isinstance(value, str)
        or not value
        or not value.isascii()
        or len(value) > _MAX_FLUSH_REASON_LENGTH
    ):
        raise BackendProtocolError('site flush reason must be short nonempty ASCII')


def generation_changes(value: object) -> tuple[GenerationChange, ...]:
    """Require a tuple of generation changes or raise `BackendProtocolError`."""
    if not isinstance(value, tuple) or not all(
        isinstance(change, GenerationChange) for change in cast('tuple[object, ...]', value)
    ):
        raise BackendProtocolError('generation changes must be an immutable tuple')
    return cast('tuple[GenerationChange, ...]', value)


def metric_flush_reason(value: str) -> str:
    """Map operator text to one fixed low-cardinality metric label."""
    return value if value in _METRIC_FLUSH_REASONS else 'operator'


def opaque_key(value: object) -> str:
    """Require nonempty UTF-8 text, raising `BackendProtocolError` otherwise."""
    if not isinstance(value, str) or not value:
        raise BackendProtocolError('backend key must be nonempty text')
    encodable = True
    try:
        value.encode('utf-8')
    except UnicodeEncodeError:
        encodable = False
    if not encodable:
        raise BackendProtocolError('backend key must be valid text')
    return value


def validate_vector_tags(tags: object, maximum: int) -> tuple[DependencyTag, ...]:
    """Validate an ordered immutable dependency tuple.

    Args:
        tags: Tuple containing only dependency tags, without duplicates.
        maximum: Maximum permitted tag count.

    Returns:
        The original validated tuple.

    Raises:
        BackendProtocolError: The tuple is malformed, duplicated, or too large.
    """
    validated = _dependency_tags(tags)
    if len(validated) != len(set(validated)):
        raise BackendProtocolError('dependency vector contains duplicate tags')
    if len(validated) > maximum:
        raise BackendProtocolError('dependency vector exceeds the tag bound')
    return validated


def vector_bytes(tags: tuple[DependencyTag, ...]) -> int:
    """Measure worst-case encoded vector bytes or raise `BackendProtocolError`."""
    size: int | None = None
    with suppress(NormalizationError):
        size = KeyBuilder.worst_case_vector_bytes(tags)
    if size is None:
        raise BackendProtocolError('dependency vector cannot be encoded safely')
    return size


def require_vector_byte_bound(vector_bytes: int, maximum: int) -> None:
    """Raise `BackendProtocolError` when encoded vector bytes exceed the maximum."""
    if vector_bytes > maximum:
        raise BackendProtocolError('dependency vector exceeds the byte bound')
