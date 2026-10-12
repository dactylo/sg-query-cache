"""Encode and decode the cache's closed typed-JSON value vocabulary."""

import json
from base64 import b64decode, b64encode
from binascii import Error as Base64Error
from collections.abc import Mapping
from datetime import date, datetime
from math import isfinite
from typing import Literal, TypeAlias, cast

_JsonScalar: TypeAlias = bool | float | int | str | None
_JsonValue: TypeAlias = _JsonScalar | list['_JsonValue'] | dict[str, '_JsonValue']

MAX_BYTES = 1_048_576
_MAX_DEPTH = 64
_MAX_ITEMS = 100_000
_CodecBound: TypeAlias = Literal['bytes', 'depth', 'items']

_NODE_LENGTH = 2
_NO_PAYLOAD = object()
_UNKNOWN = object()


class CodecError(ValueError):
    """Report a value outside the closed codec or malformed encoded data."""


class CodecBoundError(CodecError):
    """Report a value that exceeds one fixed structural or byte bound.

    Attributes:
        bound: Which bound was exceeded: `bytes`, `depth`, or `items`.
    """

    __slots__ = ('bound',)

    def __init__(self, bound: _CodecBound) -> None:
        """Record the exceeded codec bound and its stable diagnostic message."""
        super().__init__(f'encoded value exceeds the {bound} bound')
        self.bound = bound


def _consume_item(budget: list[int]) -> None:
    """Consume one node from the fixed item budget."""
    budget[0] += 1
    if budget[0] > _MAX_ITEMS:
        raise CodecBoundError('items')


def _decode_bytes(payload: object) -> bytes:
    """Decode one canonical base64 payload."""
    if not isinstance(payload, str):
        raise CodecError('bytes payload must be base64 text')
    try:
        encoded = payload.encode('ascii')
        value = b64decode(encoded, validate=True)
    except (Base64Error, UnicodeEncodeError) as error:
        raise CodecError('bytes payload must be canonical base64') from error
    if b64encode(value) != encoded:
        raise CodecError('bytes payload must be canonical base64')
    return value


def _decode_date(payload: object) -> date:
    """Decode one canonical ISO date payload."""
    if not isinstance(payload, str):
        raise CodecError('date payload must be canonical ISO text')
    try:
        value = date.fromisoformat(payload)
    except ValueError as error:
        raise CodecError('date payload must be canonical ISO text') from error
    if value.isoformat() != payload:
        raise CodecError('date payload must be canonical ISO text')
    return value


def _decode_datetime(payload: object) -> datetime:
    """Decode one canonical ISO datetime payload."""
    if not isinstance(payload, str):
        raise CodecError('datetime payload must be canonical ISO text')
    try:
        value = datetime.fromisoformat(payload)
    except ValueError as error:
        raise CodecError('datetime payload must be canonical ISO text') from error
    if value.isoformat() != payload:
        raise CodecError('datetime payload must be canonical ISO text')
    return value


def _decode_float(payload: object) -> float:
    """Decode one finite JSON float without accepting integers or booleans."""
    if not isinstance(payload, float) or not isfinite(payload):
        raise CodecError('float payload must be a finite JSON float')
    return payload


def _decode_int(payload: object) -> int:
    """Decode one JSON integer without accepting booleans."""
    if isinstance(payload, bool) or not isinstance(payload, int):
        raise CodecError('integer payload must be a JSON integer')
    return payload


def _decode_items(payload: object, tag: str, depth: int, budget: list[int]) -> list[object]:
    """Decode a sequence payload shared by list and tuple nodes."""
    if not isinstance(payload, list):
        raise CodecError(f'{tag} payload must be a JSON array')
    return [_decode_node(item, depth + 1, budget) for item in cast('list[object]', payload)]


def _decode_map(payload: object, depth: int, budget: list[int]) -> dict[str, object]:
    """Decode an ordered map payload while rejecting duplicate decoded keys."""
    if not isinstance(payload, list):
        raise CodecError('map payload must be a JSON array')
    result: dict[str, object] = {}
    for raw_pair in cast('list[object]', payload):
        if not isinstance(raw_pair, list):
            raise CodecError('each map entry must be a two-item JSON array')
        pair = cast('list[object]', raw_pair)
        if len(pair) != _NODE_LENGTH:
            raise CodecError('each map entry must be a two-item JSON array')
        key = _decode_node(pair[0], depth + 1, budget)
        if not isinstance(key, str):
            raise CodecError('decoded mapping keys must be strings')
        if key in result:
            raise CodecError('decoded map contains a duplicate key')
        result[key] = _decode_node(pair[1], depth + 1, budget)
    return result


def _decode_scalar(raw_tag: str, payload: object) -> object:
    """Decode a scalar payload or return the private unknown-tag marker."""
    result: object = _UNKNOWN
    match raw_tag:
        case 'bool':
            if not isinstance(payload, bool):
                raise CodecError('boolean payload must be a JSON boolean')
            result = payload
        case 'bytes':
            result = _decode_bytes(payload)
        case 'date':
            result = _decode_date(payload)
        case 'datetime':
            result = _decode_datetime(payload)
        case 'float':
            result = _decode_float(payload)
        case 'int':
            result = _decode_int(payload)
        case 'str':
            if not isinstance(payload, str):
                raise CodecError('string payload must be a JSON string')
            result = payload
        case _:
            pass
    return result


def _encode_scalar(value: object) -> _JsonValue | None:
    """Encode recognized scalar types, returning `None` for unrecognized types."""
    node: _JsonValue | None = None
    if value is None:
        node = ['null']
    elif isinstance(value, bool):
        node = ['bool', value]
    elif isinstance(value, int):
        node = ['int', value]
    elif isinstance(value, float):
        if not isfinite(value):
            raise CodecError('float values must be finite')
        node = ['float', value]
    elif isinstance(value, str):
        node = ['str', value]
    elif isinstance(value, bytes):
        node = ['bytes', b64encode(value).decode('ascii')]
    elif isinstance(value, datetime):
        node = ['datetime', value.isoformat()]
    elif isinstance(value, date):
        node = ['date', value.isoformat()]
    return node


def _reject_nonfinite_json(value: str) -> object:
    """Reject nonfinite JSON constants before typed-node validation."""
    raise ValueError(f'nonfinite JSON constant: {value}')


def _require_bytes(data: object) -> bytes:
    """Require immutable encoded bytes at the untrusted codec boundary."""
    if not isinstance(data, bytes):
        raise CodecError('encoded data must be bytes')
    return data


def _require_depth(depth: int) -> None:
    """Reject a node beyond the fixed nesting bound."""
    if depth > _MAX_DEPTH:
        raise CodecBoundError('depth')


def _encode_node(value: object, depth: int, budget: list[int]) -> _JsonValue:
    """Convert one validated Python value to its typed JSON node."""
    _require_depth(depth)
    _consume_item(budget)
    scalar = _encode_scalar(value)
    if scalar is not None:
        return scalar
    if isinstance(value, list):
        values = cast('list[object]', value)
        return ['list', [_encode_node(item, depth + 1, budget) for item in values]]
    if isinstance(value, tuple):
        values = cast('tuple[object, ...]', value)
        return ['tuple', [_encode_node(item, depth + 1, budget) for item in values]]
    if isinstance(value, Mapping):
        items: list[_JsonValue] = []
        for key, item in cast('Mapping[object, object]', value).items():
            if not isinstance(key, str):
                raise CodecError('mapping keys must be strings')
            items.append(
                [
                    _encode_node(key, depth + 1, budget),
                    _encode_node(item, depth + 1, budget),
                ]
            )
        return ['map', items]
    raise CodecError('value uses an unsupported class')


def _split_node(node: object) -> tuple[str, object]:
    """Validate one node's tag and payload shape."""
    if not isinstance(node, list):
        raise CodecError('each encoded node must be a one- or two-item JSON array')
    parts = cast('list[object]', node)
    if not parts or len(parts) > _NODE_LENGTH:
        raise CodecError('each encoded node must be a one- or two-item JSON array')
    raw_tag = parts[0]
    if not isinstance(raw_tag, str):
        raise CodecError('encoded node type tag must be a string')
    if raw_tag == 'null':
        if len(parts) != 1:
            raise CodecError('null node must not contain a payload')
        return raw_tag, _NO_PAYLOAD
    if len(parts) != _NODE_LENGTH:
        raise CodecError(f'{raw_tag} node requires a payload')
    return raw_tag, parts[1]


def _decode_node(node: object, depth: int, budget: list[int]) -> object:
    """Validate and construct one typed JSON node."""
    _require_depth(depth)
    _consume_item(budget)
    raw_tag, payload = _split_node(node)
    if payload is _NO_PAYLOAD:
        return None
    scalar = _decode_scalar(raw_tag, payload)
    if scalar is not _UNKNOWN:
        return scalar
    match raw_tag:
        case 'list':
            return _decode_items(payload, raw_tag, depth, budget)
        case 'map':
            return _decode_map(payload, depth, budget)
        case 'tuple':
            return tuple(_decode_items(payload, raw_tag, depth, budget))
        case _:
            raise CodecError('encoded node uses an unknown type tag')


def decode_value(data: bytes) -> object:
    """Decode one closed typed-JSON value from bounded bytes.

    Args:
        data: Encoded typed-JSON bytes.

    Returns:
        A newly constructed value using the vocabulary documented by
        `encode_value`. Mappings become dictionaries. Aware datetimes use a
        fixed-offset timezone preserving the encoded offset, not the original
        `tzinfo` class.

    Raises:
        CodecError: The bytes are malformed, noncanonical, unsupported, or
            exceed a fixed bound.
    """
    encoded = _require_bytes(data)
    if len(encoded) > MAX_BYTES:
        raise CodecBoundError('bytes')
    try:
        text = encoded.decode('utf-8')
    except UnicodeDecodeError as error:
        raise CodecError('encoded data must be valid UTF-8') from error
    try:
        node, end = json.JSONDecoder(parse_constant=_reject_nonfinite_json).raw_decode(text)
    except (json.JSONDecodeError, RecursionError, ValueError) as error:
        raise CodecError('encoded data must be valid finite JSON') from error
    if end != len(text):
        raise CodecError('encoded data contains trailing bytes')
    return _decode_node(node, 1, [0])


def encode_value(value: object) -> bytes:
    """Encode one value using the closed typed-JSON vocabulary.

    Args:
        value: `None`, booleans, integers, finite floats, strings, bytes,
            dates, datetimes, lists, tuples, or string-keyed mappings.
            Containers may recursively contain those values.

    Returns:
        Compact deterministic UTF-8 JSON bytes preserving list-versus-tuple
        distinctions and mapping iteration order. The fixed limits are 1 MiB,
        nesting depth 64, and 100,000 typed nodes.

    Raises:
        CodecError: The value is unsupported, nonfinite, or exceeds a fixed
            bound.
    """
    node = _encode_node(value, 1, [0])
    try:
        data = json.dumps(
            node,
            allow_nan=False,
            ensure_ascii=False,
            separators=(',', ':'),
            sort_keys=False,
        ).encode('utf-8')
    except (RecursionError, ValueError) as error:
        raise CodecError('value cannot be encoded as finite typed JSON') from error
    if len(data) > MAX_BYTES:
        raise CodecBoundError('bytes')
    return data


def encoded_size(value: object) -> int:
    """Return the byte length of a value's closed typed-JSON encoding.

    Args:
        value: Supported value to measure.

    Returns:
        The exact encoded byte length.

    Raises:
        CodecError: The value is unsupported or exceeds a fixed bound.
    """
    return len(encode_value(value))
