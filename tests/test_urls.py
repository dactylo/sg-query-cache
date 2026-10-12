from dataclasses import FrozenInstanceError
from datetime import UTC, datetime, timedelta

import pytest

from sg_query_cache._config import CacheSettings, LimitSettings
from sg_query_cache._urls import AgeDecision, cap_response_age

_ABSOLUTE_URL = 'https://files.example.invalid/download?Expires=1788091320&token=secret'
_AWS_URL = (
    'https://files.example.invalid/download?X-Amz-Date=20260830T120000Z'
    '&X-Amz-Expires=120&X-Amz-Signature=secret'
)
_NOW = datetime(2026, 8, 30, 12, tzinfo=UTC)
_SIGNED_SETTINGS = CacheSettings(limits=LimitSettings(clock_error_bound_seconds=0))


def _decision(
    value: object,
    returned_fields: dict[str, str],
    *,
    settings: CacheSettings = _SIGNED_SETTINGS,
) -> AgeDecision:
    return cap_response_age(value, returned_fields, _NOW, settings)


def test_absolute_epoch_expiry_produces_a_margin_adjusted_cap() -> None:
    decision = _decision({'image': _ABSOLUTE_URL}, {'image': 'image'})

    assert decision.remaining_seconds == pytest.approx(90)
    assert decision.reason == 'signed_url'


def test_age_decision_is_frozen_and_slotted() -> None:
    decision = _decision({'image': _ABSOLUTE_URL}, {'image': 'image'})

    assert not hasattr(decision, '__dict__')
    with pytest.raises(FrozenInstanceError):
        decision.reason = 'changed'  # type: ignore[misc]


def test_aws_v4_date_and_duration_produce_a_margin_adjusted_cap() -> None:
    decision = _decision({'image': _AWS_URL}, {'image': 'image'})

    assert decision.remaining_seconds == pytest.approx(90)
    assert decision.reason == 'signed_url'


def test_case_insensitive_parameters_use_the_two_recognized_forms() -> None:
    value = {
        'image': (
            'https://files.example.invalid/download?x-amz-date=20260830T120000Z'
            '&X-AMZ-EXPIRES=120&signature=secret'
        )
    }

    assert _decision(value, {'image': 'image'}).remaining_seconds == pytest.approx(90)


def test_clock_error_bound_preserves_the_full_real_time_margin() -> None:
    clock_error_seconds = 20
    margin_seconds = 30
    settings = CacheSettings(
        limits=LimitSettings(
            clock_error_bound_seconds=clock_error_seconds,
            signed_url_safety_margin_seconds=margin_seconds,
        )
    )
    true_response_time = _NOW + timedelta(seconds=clock_error_seconds)
    embedded_expiry = _NOW + timedelta(seconds=120)

    decision = _decision({'image': _AWS_URL}, {'image': 'image'}, settings=settings)

    assert decision.remaining_seconds == pytest.approx(70)
    assert decision.remaining_seconds is not None
    true_cache_expiry = true_response_time + timedelta(seconds=decision.remaining_seconds)
    assert embedded_expiry - true_cache_expiry == timedelta(seconds=margin_seconds)


def test_configured_margin_is_subtracted_from_the_embedded_expiry() -> None:
    settings = CacheSettings(
        limits=LimitSettings(
            clock_error_bound_seconds=0,
            signed_url_safety_margin_seconds=45,
        )
    )

    decision = _decision({'image': _AWS_URL}, {'image': 'image'}, settings=settings)

    assert decision.remaining_seconds == pytest.approx(75)


def test_earliest_expiry_across_rows_and_fields_wins() -> None:
    value = [
        {
            'image': (
                'https://files.example.invalid/one?X-Amz-Date=20260830T120000Z&X-Amz-Expires=300'
            )
        },
        {'download': _ABSOLUTE_URL},
    ]

    decision = _decision(value, {'image': 'image', 'download': 'image'})

    assert decision.remaining_seconds == pytest.approx(90)
    assert decision.reason == 'signed_url'


@pytest.mark.parametrize(
    'stamp',
    [
        '2026830T120000Z',
        '20260830T12000Z',
        '20260830t120000z',
    ],
)
def test_image_declines_noncanonical_aws_v4_timestamp(stamp: str) -> None:
    url = f'https://files.example.invalid/download?X-Amz-Date={stamp}&X-Amz-Expires=120'

    decision = _decision({'image': url}, {'image': 'image'})

    assert decision.remaining_seconds is None
    assert decision.reason == 'unrecognized_image_url'


def test_nonpositive_remaining_lifetime_declines_population() -> None:
    value = {
        'image': (
            'https://files.example.invalid/download?X-Amz-Date=20260830T120000Z&X-Amz-Expires=20'
        )
    }

    decision = _decision(value, {'image': 'image'})

    assert decision.remaining_seconds is None
    assert decision.reason == 'signed_url_expired'


def test_reasons_and_logs_never_contain_a_url_or_query_string(
    caplog: pytest.LogCaptureFixture,
) -> None:
    unrecognized_url = 'https://files.example.invalid/file?credential=do-not-leak'

    decision = _decision({'image': unrecognized_url}, {'image': 'image'})

    assert unrecognized_url not in caplog.text
    assert '?' not in decision.reason
    assert 'http' not in decision.reason
    assert decision.reason == 'unrecognized_image_url'


def test_response_walking_ignores_arbitrary_strings_and_unreturned_fields() -> None:
    value = {
        'description': _AWS_URL,
        'unreturned_image': 'https://files.example.invalid/file?token=secret',
        'image': None,
    }

    decision = _decision(
        value,
        {'description': 'text', 'image': 'image'},
    )

    assert decision.remaining_seconds == pytest.approx(3600)
    assert decision.reason == 'response_age_unchanged'


def test_transient_image_path_uses_the_five_second_cap() -> None:
    value = {
        'image': 'https://site.example.invalid/images/status/transient/thumbnail.jpg?opaque=secret'
    }

    decision = _decision(value, {'image': 'image'}, settings=CacheSettings())

    assert decision.remaining_seconds == pytest.approx(5)
    assert decision.reason == 'transient_image'


def test_unbounded_clock_error_declines_signed_url_population() -> None:
    decision = _decision(
        {'image': _AWS_URL},
        {'image': 'image'},
        settings=CacheSettings(),
    )

    assert decision.remaining_seconds is None
    assert decision.reason == 'clock_error_unbounded'


def test_upload_attachment_requires_a_recognized_expiry() -> None:
    value = {'file': {'link_type': 'upload', 'url': _AWS_URL, 'name': 'plate.exr'}}

    decision = _decision(value, {'file': 'url'})

    assert decision.remaining_seconds == pytest.approx(90)
    assert decision.reason == 'signed_url'


@pytest.mark.parametrize(
    'stamp',
    [
        '2026830T120000Z',
        '20260830T12000Z',
        '20260830t120000z',
    ],
)
def test_upload_declines_noncanonical_aws_v4_timestamp(stamp: str) -> None:
    url = f'https://files.example.invalid/download?X-Amz-Date={stamp}&X-Amz-Expires=120'

    decision = _decision(
        {'file': {'link_type': 'upload', 'url': url}},
        {'file': 'url'},
    )

    assert decision.remaining_seconds is None
    assert decision.reason == 'unrecognized_upload_url'


@pytest.mark.parametrize(
    ('expires_in_seconds', 'remaining_seconds', 'reason'),
    [
        (20, None, 'signed_url_expired'),
        (32, 2, 'signed_url'),
    ],
)
def test_transient_image_also_obeys_its_signed_expiry(
    expires_in_seconds: int,
    remaining_seconds: int | None,
    reason: str,
) -> None:
    value = {
        'image': (
            'https://site.example.invalid/images/status/transient/thumbnail.jpg'
            '?X-Amz-Date=20260830T120000Z&X-Amz-Expires='
            f'{expires_in_seconds}'
        )
    }

    decision = _decision(value, {'image': 'image'})

    if remaining_seconds is None:
        assert decision.remaining_seconds is None
    else:
        assert decision.remaining_seconds == pytest.approx(remaining_seconds)
    assert decision.reason == reason


@pytest.mark.parametrize(
    'url',
    [
        'https://files.example.invalid/download?token=secret',
        'https://[invalid/download?Expires=1788091320',
        'https://files.example.invalid/download?Expires=not-an-epoch',
        (
            'https://files.example.invalid/download?X-Amz-Date=20260830T120000Z'
            '&X-Amz-Expires=120&x-amz-expires=240'
        ),
    ],
)
def test_unrecognized_nonempty_image_url_declines_population(url: str) -> None:
    decision = _decision({'image': url}, {'image': 'image'})

    assert decision.remaining_seconds is None
    assert decision.reason == 'unrecognized_image_url'


@pytest.mark.parametrize(
    'attachment',
    [
        {'link_type': 'upload', 'url': 'https://files.example.invalid/file?token=secret'},
        {'link_type': 'upload', 'url': ''},
        {'link_type': 'upload'},
    ],
)
def test_unrecognized_upload_attachment_declines_population(
    attachment: dict[str, str],
) -> None:
    decision = _decision({'file': attachment}, {'file': 'url'})

    assert decision.remaining_seconds is None
    assert decision.reason == 'unrecognized_upload_url'


@pytest.mark.parametrize('link_type', ['local', 'url'])
def test_local_and_web_attachment_links_retain_the_selected_age(link_type: str) -> None:
    value = {
        'file': {
            'link_type': link_type,
            'url': _AWS_URL,
            'name': 'reference',
        }
    }

    decision = _decision(value, {'file': 'url'})

    assert decision.remaining_seconds == pytest.approx(3600)
    assert decision.reason == 'response_age_unchanged'
