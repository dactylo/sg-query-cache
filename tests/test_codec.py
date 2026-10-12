from datetime import date, datetime, timedelta, timezone
from typing import cast

import pytest

from sg_query_cache._codec import (
    _MAX_DEPTH,
    _MAX_ITEMS,
    MAX_BYTES,
    CodecBoundError,
    CodecError,
    decode_value,
    encode_value,
    encoded_size,
)


def test_bound_failures_name_their_bound() -> None:
    too_many = list(range(_MAX_ITEMS))
    too_deep: list[object] = []
    cursor = too_deep
    for _ in range(_MAX_DEPTH):
        nested: list[object] = []
        cursor.append(nested)
        cursor = nested
    too_wide = 'x' * MAX_BYTES

    with pytest.raises(CodecBoundError) as items:
        encode_value(too_many)
    with pytest.raises(CodecBoundError) as depth:
        encode_value(too_deep)
    with pytest.raises(CodecBoundError) as size:
        encode_value(too_wide)

    assert items.value.bound == 'items'
    assert depth.value.bound == 'depth'
    assert size.value.bound == 'bytes'
    assert issubclass(CodecBoundError, CodecError)


@pytest.mark.parametrize(
    'value',
    [
        None,
        False,
        True,
        -17,
        0,
        42,
        -1.25,
        0.0,
        3.5,
        '',
        'caf\N{LATIN SMALL LETTER E WITH ACUTE}',
        b'\x00\xff',
        date(2026, 8, 30),
        datetime.fromisoformat('2026-08-30T12:34:56.000789'),
        datetime(2026, 8, 30, 12, 34, tzinfo=timezone(timedelta(hours=-7))),
        [1, 'two', None],
        (1, 'two', None),
        {'second': 2, 'first': 1},
    ],
)
def test_closed_values_round_trip_with_exact_types(value: object) -> None:
    decoded = decode_value(encode_value(value))

    assert decoded == value
    assert type(decoded) is type(value)


def test_codec_rejects_data_beyond_the_fixed_byte_bound() -> None:
    value = 'x' * 1_048_576

    with pytest.raises(CodecError, match='byte'):
        encode_value(value)

    with pytest.raises(CodecError, match='byte'):
        decode_value(b' ' * 1_048_577)


def test_decoding_rejects_a_node_beyond_the_depth_bound() -> None:
    data = b'["list",[' * 65 + b'["null"]' + b']]' * 65

    with pytest.raises(CodecError, match='depth'):
        decode_value(data)


def test_decoding_rejects_a_node_beyond_the_item_bound() -> None:
    data = b'["list",[' + b','.join([b'["null"]'] * 100_001) + b']]'

    with pytest.raises(CodecError, match='item'):
        decode_value(data)


def test_encoding_is_compact_typed_json() -> None:
    assert encode_value(None) == b'["null"]'
    assert encode_value([True, 2]) == b'["list",[["bool",true],["int",2]]]'
    assert encoded_size(('x', b'y')) == len(encode_value(('x', b'y')))


@pytest.mark.parametrize(
    'value',
    [
        object(),
        {1: 'not a string key'},
        float('nan'),
        float('inf'),
        float('-inf'),
    ],
)
def test_encoding_rejects_values_outside_the_closed_vocabulary(value: object) -> None:
    with pytest.raises(CodecError):
        encode_value(value)


@pytest.mark.parametrize(
    ('data', 'match'),
    [
        (b'["future",1]', 'type tag'),
        (
            b'["map",[[["str","same"],["int",1]],[["str","same"],["int",2]]]]',
            'duplicate',
        ),
        (b'\xff', 'UTF-8'),
        (b'["float",NaN]', 'JSON'),
        (b'["float",Infinity]', 'JSON'),
        (b'["int",1] trailing', 'trailing'),
        (b'["date","2026-8-3"]', 'canonical'),
        (b'["datetime","2026-08-30T12:00:00Z"]', 'canonical'),
        (b'["bytes","YQ"]', 'base64'),
    ],
)
def test_decoding_rejects_malformed_or_noncanonical_data(data: bytes, match: str) -> None:
    with pytest.raises(CodecError, match=match):
        decode_value(data)


def test_mapping_insertion_order_is_preserved() -> None:
    original = {'second': 2, 'first': 1}

    decoded = decode_value(encode_value(original))

    assert isinstance(decoded, dict)
    assert list(cast('dict[str, object]', decoded)) == ['second', 'first']


def test_round_trip_does_not_share_mutable_state() -> None:
    original = [{'id': 7, 'tags': ['a']}]

    decoded = decode_value(encode_value(original))

    assert decoded == original
    assert isinstance(decoded, list)
    assert decoded is not original
    assert decoded[0] is not original[0]
    assert decoded[0]['tags'] is not original[0]['tags']
