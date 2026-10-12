import pytest

from sg_query_cache._backend import BackendCapacity, BackendProtocolError, StaleIncarnation
from sg_query_cache._valkey_common import (
    Incarnation,
    RoleEpoch,
    ValidationState,
    _decode_lookup_reply,
    cached_incarnation,
    decode_commit_reply,
    decode_release_reply,
    decode_reserve_reply,
    decode_resolve_reply,
    next_validation_state,
    parse_record,
    validation_is_current,
)
from tests._valkey_control_cases import _RECORD_A, _RECORD_B


def test_cleared_validation_is_never_current() -> None:
    state = ValidationState(None, None, None, 2)

    assert not validation_is_current(state, (0, 1, None), 0.0, 30)
    assert cached_incarnation(state, (0, 1, None), 0.0, 30) is None


def test_resolve_decoder_classifies_metadata_capacity() -> None:
    with pytest.raises(BackendCapacity):
        decode_resolve_reply([b'capacity'], expected_count=1)


@pytest.mark.parametrize(
    'reply',
    [
        None,
        [],
        [b'ok'],
        [b'ok', 0],
        [b'ok', True],
        [b'ok', b'not-an-integer'],
        [b'ok', b'\xff1'],
        [b'ok', 2**63],
        [b'unknown', 1],
        [b'ok', 1, b'credential-bearing-token'],
    ],
)
def test_resolve_decoder_rejects_malformed_complete_vectors(reply: object) -> None:
    with pytest.raises(BackendProtocolError) as captured:
        decode_resolve_reply(reply, expected_count=1)

    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None
    assert 'credential-bearing-token' not in repr(captured.value)


@pytest.mark.parametrize('status', [b'stale', b'missing-counter'])
def test_resolve_decoder_rejects_stale_or_partial_control_state(status: bytes) -> None:
    with pytest.raises(StaleIncarnation):
        decode_resolve_reply([status], expected_count=1)


@pytest.mark.parametrize(
    ('epoch', 'now', 'current', 'hit'),
    [
        ((0, 1, 'primary'), 29.0, True, True),
        ((0, 2, 'primary'), 29.0, True, True),
        ((0, 3, 'primary'), 29.0, True, False),
        ((1, 1, 'primary'), 29.0, False, False),
        ((0, 1, 'replacement'), 29.0, False, False),
        ((0, 1, 'primary'), 30.0, False, False),
        ((0, 1, 'primary'), -1.0, False, False),
    ],
)
def test_cached_validation_requires_current_role_and_covered_connection(
    epoch: RoleEpoch, now: float, current: bool, hit: bool
) -> None:
    incarnation = parse_record(_RECORD_A, 'sgqc')
    state = ValidationState(incarnation, (0, 1, 'primary'), 0.0, 1, frozenset({1, 2}))

    assert validation_is_current(state, epoch, now, 30) is current
    assert cached_incarnation(state, epoch, now, 30) == (incarnation if hit else None)


@pytest.mark.parametrize('validated_at', [0.0, 20.0, 29.0])
def test_joined_validation_preserves_original_time_and_epoch(validated_at: float) -> None:
    incarnation = parse_record(_RECORD_A, 'sgqc')
    state = ValidationState(incarnation, (0, 1, None), 0.0, 1, frozenset({1}))

    published = next_validation_state(state, incarnation, (0, 2, None), validated_at, join=True)

    assert published == ValidationState(incarnation, (0, 1, None), 0.0, 2, frozenset({1, 2}))


@pytest.mark.parametrize('reason', ['forced-or-expired', 'changed-incarnation', 'cleared'])
def test_new_validation_replaces_coverage_time_and_epoch(reason: str) -> None:
    incarnation = parse_record(_RECORD_A, 'sgqc')
    state = ValidationState(incarnation, (0, 1, None), 0.0, 1, frozenset({1, 3}))
    candidate: Incarnation = incarnation
    if reason == 'changed-incarnation':
        candidate = parse_record(_RECORD_B, 'sgqc')
    elif reason == 'cleared':
        state = ValidationState(None, None, None, 1)

    published = next_validation_state(
        state, candidate, (0, 2, None), 20.0, join=reason != 'forced-or-expired'
    )

    assert published == ValidationState(candidate, (0, 2, None), 20.0, 2, frozenset({2}))


def test_result_decoders_validate_hit_and_lease_replies() -> None:
    assert _decode_lookup_reply([b'hit', b'payload', 1_250]) == (b'payload', 1_250)
    assert _decode_lookup_reply([b'miss']) is None
    assert decode_commit_reply([b'committed'])
    assert not decode_commit_reply([b'rejected'])
    assert not decode_commit_reply([b'capacity'])
    assert decode_release_reply([b'released'])
    assert not decode_release_reply([b'retained'])
    assert decode_reserve_reply([b'reserved'])
    assert not decode_reserve_reply([b'busy'])
