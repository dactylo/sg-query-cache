"""Build opaque deterministic keys for normalized exact SG calls."""

import hmac
from collections.abc import Mapping, Sequence
from hashlib import sha256
from typing import Literal, TypeAlias, cast

from sg_query_cache._codec import CodecError, encode_value, encoded_size
from sg_query_cache._model import DependencyTag, ExactCall, GenerationVector, Identity
from sg_query_cache._query import FILTER_GROUP_KEYS, ORDER_ITEM_KEYS

_KeyPurpose: TypeAlias = Literal['lease', 'result']

_CACHE_FORMAT = 'v1'
_CLIENT_OPTION_NAMES = ('localized', 'convert_datetimes_to_utc')
_CREDENTIAL_NAMES = frozenset({'api_key', 'session_token'})
_DIGEST_LENGTH = 64
_INCARNATION_LENGTH = 32
_MAX_GENERATION = 2**63 - 1
METRICS_KEY_MIN_BYTES = 32
_OPTIONAL_CLIENT_OPTION_NAMES = ('records_per_page',)
_PAIR_LENGTH = 2


class NormalizationError(ValueError):
    """Report context that cannot safely produce one canonical cache key."""


class KeyBuilder:
    """Build opaque cache-format-v1 backend names and metrics correlation values."""

    @staticmethod
    def base_key(
        site: str,
        identity: Identity,
        client_options: Mapping[str, object],
        call: ExactCall,
    ) -> str:
        """Build the deterministic base key for one complete exact-call context.

        Args:
            site: Normalized SG site identifier.
            identity: Stable effective identity namespace.
            client_options: Complete response-affecting client options.
            call: Explicitly bound immutable SG read call.

        Returns:
            A cache-format-v1 opaque base key.

        Raises:
            NormalizationError: The context is incomplete, secret-bearing, or
                noncanonical.
        """
        normalized_site = _require_site(site)
        context = (
            normalized_site,
            _normalized_identity(identity),
            _normalized_client_options(client_options),
            _normalized_call(call),
        )
        return f'sgqc:{_CACHE_FORMAT}:base:{_typed_digest("base-key", context)}'

    @staticmethod
    def metrics_digest(base_key: str, metrics_key: bytes) -> str:
        """Build an optional keyed exact-call correlation value.

        Args:
            base_key: Valid opaque base key to correlate.
            metrics_key: Deployment-shared HMAC key of at least 32 bytes.

        Returns:
            A lowercase HMAC-SHA-256 digest for metrics only.

        Raises:
            NormalizationError: The base key or HMAC key is invalid.
        """
        normalized_base_key = _require_base_key(base_key)
        normalized_metrics_key = _require_metrics_key(metrics_key)
        return hmac.digest(
            normalized_metrics_key, normalized_base_key.encode('ascii'), 'sha256'
        ).hex()

    @staticmethod
    def qualified_key(
        base_key: str,
        vector: GenerationVector,
        purpose: _KeyPurpose,
    ) -> str:
        """Build a result or lease key from an ordered generation vector.

        Args:
            base_key: Valid opaque base key.
            vector: Ordered generations under one backend incarnation.
            purpose: Either `result` or `lease`.

        Returns:
            An opaque incarnation-qualified cache key.

        Raises:
            NormalizationError: Any key, purpose, incarnation, or generation is
                invalid.
        """
        normalized_base_key = _require_base_key(base_key)
        if purpose not in ('result', 'lease'):
            raise NormalizationError('qualified key purpose must be result or lease')
        normalized_vector = _normalized_vector(vector)
        digest = _typed_digest(
            'qualified-key',
            (normalized_base_key, purpose, normalized_vector),
        )
        return f'sgqc:{_CACHE_FORMAT}:{purpose}:{vector.incarnation}:{digest}'

    @staticmethod
    def tag_key(tag: DependencyTag) -> str:
        """Build a domain-separated deterministic key for one dependency tag.

        Args:
            tag: Validated site-partitioned dependency tag.

        Returns:
            A cache-format-v1 opaque tag key.

        Raises:
            NormalizationError: The tag cannot be encoded within the codec
                vocabulary or bounds.
        """
        return f'sgqc:{_CACHE_FORMAT}:tag:{_typed_digest("tag-key", _normalized_tag(tag))}'

    @staticmethod
    def worst_case_vector_bytes(tags: Sequence[DependencyTag]) -> int:
        """Measure a tag set using maximum incarnation and generation lengths.

        Args:
            tags: Ordered dependency tags to measure before backend resolution.

        Returns:
            The exact typed encoding size with maximum-width generation values.

        Raises:
            NormalizationError: The vector is duplicate, malformed, or too
                large for the codec.
        """
        ordered = tuple(tags)
        if len(ordered) != len(set(ordered)):
            raise NormalizationError('worst-case vector contains a duplicate dependency tag')
        context = (
            'f' * _INCARNATION_LENGTH,
            tuple((*_normalized_tag(tag), _MAX_GENERATION) for tag in ordered),
        )
        try:
            return encoded_size(context)
        except CodecError as error:
            raise NormalizationError('worst-case vector exceeds the codec bounds') from error


def _canonical_mapping(value: tuple[object, ...]) -> tuple[tuple[str, object], ...] | None:
    """Recognize and validate a tuple representation of a mapping."""
    if not value:
        return None
    entries: list[tuple[str, object]] = []
    for item in value:
        if not isinstance(item, tuple):
            return None
        parts = cast('tuple[object, ...]', item)
        if len(parts) != _PAIR_LENGTH or not isinstance(parts[0], str):
            return None
        entries.append((parts[0], parts[1]))
    names = [name for name, _item in entries]
    if len(names) != len(set(names)):
        raise NormalizationError('normalized mapping contains a duplicate key')
    keys = frozenset(names)
    if keys not in (FILTER_GROUP_KEYS, ORDER_ITEM_KEYS):
        raise NormalizationError('normalized value uses an unsupported mapping shape')
    return tuple(sorted(entries, key=lambda entry: entry[0]))


def _canonical_value(value: object) -> object:
    """Canonicalize documented order-insensitive immutable mapping shapes."""
    if not isinstance(value, tuple):
        return value
    values = cast('tuple[object, ...]', value)
    mapping = _canonical_mapping(values)
    if mapping is not None:
        return tuple((name, _canonical_value(item)) for name, item in mapping)
    return tuple(_canonical_value(item) for item in values)


def _normalized_call(call: ExactCall) -> tuple[str, tuple[tuple[str, object], ...]]:
    """Validate and canonicalize one immutable exact call."""
    arguments: list[tuple[str, object]] = []
    for name, value in call.arguments:
        if name in _CREDENTIAL_NAMES:
            raise NormalizationError('credentials cannot be included in an exact-call key')
        if call.method in ('find', 'find_one') and name == 'fields' and value in (None, ()):
            normalized = ('id',)
        else:
            normalized = _canonical_value(value)
        arguments.append((name, normalized))
    return call.method, tuple(arguments)


def _normalized_client_options(
    options: Mapping[str, object],
) -> tuple[tuple[str, bool | int], ...]:
    """Validate the fixed response-affecting client option vocabulary."""
    names = set(options)
    if names & _CREDENTIAL_NAMES:
        raise NormalizationError('credentials cannot be included in the client key context')
    required = set(_CLIENT_OPTION_NAMES)
    if not required <= names or not names <= required | set(_OPTIONAL_CLIENT_OPTION_NAMES):
        raise NormalizationError('key context contains an unknown or missing client option')
    normalized: list[tuple[str, bool | int]] = []
    for name in _CLIENT_OPTION_NAMES:
        value = options[name]
        if not isinstance(value, bool):
            raise NormalizationError('response-affecting client options must be booleans')
        normalized.append((name, value))
    for name in _OPTIONAL_CLIENT_OPTION_NAMES:
        if name not in options:
            continue
        value = options[name]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise NormalizationError('effective page size must be a positive integer')
        normalized.append((name, value))
    return tuple(normalized)


def _normalized_identity(identity: Identity) -> tuple[object, ...]:
    """Convert one validated identity to its complete stable key context."""
    return (
        identity.kind,
        identity.issuer,
        identity.namespace,
        identity.entity_id,
        identity.epoch,
    )


def _normalized_tag(tag: DependencyTag) -> tuple[object, ...]:
    """Convert one validated dependency tag to its typed digest context."""
    return tag.kind.value, tag.components


def _normalized_vector(vector: GenerationVector) -> tuple[object, ...]:
    """Validate and convert an ordered generation vector to typed key context."""
    if (
        len(vector.incarnation) != _INCARNATION_LENGTH
        or vector.incarnation.lower() != vector.incarnation
        or any(character not in '0123456789abcdef' for character in vector.incarnation)
    ):
        raise NormalizationError(
            'generation-vector incarnation must be 32 lowercase hex characters'
        )
    entries: list[tuple[object, ...]] = []
    for tag, generation in vector.entries:
        if generation > _MAX_GENERATION:
            raise NormalizationError('generation counter exceeds the signed 63-bit bound')
        entries.append((*_normalized_tag(tag), generation))
    return vector.incarnation, tuple(entries)


def _require_base_key(base_key: object) -> str:
    """Require the exact opaque base-key syntax produced by this cache format."""
    if not isinstance(base_key, str):
        raise NormalizationError('base key must be a string')
    prefix = f'sgqc:{_CACHE_FORMAT}:base:'
    digest = base_key.removeprefix(prefix)
    if (
        not base_key.startswith(prefix)
        or len(digest) != _DIGEST_LENGTH
        or digest.lower() != digest
        or any(character not in '0123456789abcdef' for character in digest)
    ):
        raise NormalizationError('base key has an invalid cache-format syntax')
    return base_key


def _require_metrics_key(metrics_key: object) -> bytes:
    """Require an immutable HMAC key of the configured minimum length."""
    if not isinstance(metrics_key, bytes) or len(metrics_key) < METRICS_KEY_MIN_BYTES:
        raise NormalizationError('metrics HMAC key must contain at least 32 bytes')
    return metrics_key


def _require_site(site: object) -> str:
    """Require a nonempty normalized site supplied by the wrapper context."""
    if not isinstance(site, str) or not site:
        raise NormalizationError('normalized site must be a nonempty string')
    return site


def _typed_digest(domain: str, value: object) -> str:
    """Hash one domain-separated typed value without a shared naming secret."""
    try:
        payload = encode_value((domain, value))
    except CodecError as error:
        raise NormalizationError('key context is outside the closed value vocabulary') from error
    return sha256(payload).hexdigest()
