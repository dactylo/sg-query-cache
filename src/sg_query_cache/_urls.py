"""Classify URL-bearing SG responses and derive safe maximum-age caps."""

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from math import isfinite
from typing import cast
from urllib.parse import parse_qsl, urlsplit

from sg_query_cache._config import CacheSettings

_AWS_DATE_FORMAT = '%Y%m%dT%H%M%SZ'
_AWS_DATE_LENGTH = 16
_TRANSIENT_IMAGE_PATH = '/images/status/transient/'


@dataclass(frozen=True, slots=True)
class AgeDecision:
    """Describe a response-derived age cap or population decline.

    Attributes:
        remaining_seconds: Positive cap in seconds, or `None` when population
            must be declined.
        reason: Stable low-cardinality decision reason with no response data.
    """

    remaining_seconds: float | None
    reason: str

    def __post_init__(self) -> None:
        """Validate the optional positive lifetime cap and stable reason.

        Raises:
            TypeError: The cap is not a finite number or the reason is not a
                string.
            ValueError: The cap is not positive or the reason is empty.
        """
        _require_remaining_seconds(self.remaining_seconds)
        _require_reason(self.reason)


def _casefolded_parameters(url: str) -> dict[str, tuple[str, ...]] | None:
    """Parse HTTP query parameters by case-insensitive name while retaining duplicates."""
    try:
        parsed = urlsplit(url)
        if parsed.scheme.lower() not in {'http', 'https'} or not parsed.netloc:
            return None
        pairs = parse_qsl(
            parsed.query,
            keep_blank_values=True,
            strict_parsing=True,
        )
    except ValueError:
        return None
    grouped: dict[str, list[str]] = {}
    for name, value in pairs:
        grouped.setdefault(name.casefold(), []).append(value)
    return {name: tuple(values) for name, values in grouped.items()}


def _is_canonical_aws_date(value: str) -> bool:
    """Require the exact ASCII AWS v4 `YYYYMMDDTHHMMSSZ` timestamp form."""
    return (
        len(value) == _AWS_DATE_LENGTH
        and value.isascii()
        and value[8] == 'T'
        and value[15] == 'Z'
        and value[:8].isdigit()
        and value[9:15].isdigit()
    )


def _require_aware_datetime(value: object) -> datetime:
    """Require a timezone-aware response timestamp."""
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise TypeError('response time must be a timezone-aware datetime')
    return value


def _require_reason(value: object) -> None:
    """Require a stable nonempty decision reason."""
    if not isinstance(value, str):
        raise TypeError('response-age reason must be a string')
    if not value:
        raise ValueError('response-age reason must be nonempty')


def _require_remaining_seconds(value: object) -> None:
    """Require a positive finite cap or a population-decline marker."""
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, int | float) or not isfinite(value):
        raise TypeError('response-age remaining lifetime must be a finite number')
    if value <= 0:
        raise ValueError('response-age remaining lifetime must be positive')


def _require_returned_fields(value: object) -> Mapping[str, str]:
    """Require returned field names mapped to schema data types."""
    if not isinstance(value, Mapping):
        raise TypeError('returned fields must map names to schema types')
    untyped = cast('Mapping[object, object]', value)
    if any(
        not isinstance(name, str) or not name or not isinstance(field_type, str) or not field_type
        for name, field_type in untyped.items()
    ):
        raise TypeError('returned fields must map nonempty strings to schema types')
    return cast('Mapping[str, str]', untyped)


def _response_rows(value: object) -> tuple[Mapping[object, object], ...] | None:
    """Normalize one SG entity or entity list without walking nested strings."""
    if value is None:
        return ()
    if isinstance(value, Mapping):
        return (cast('Mapping[object, object]', value),)
    if not isinstance(value, list | tuple):
        return None
    rows: list[Mapping[object, object]] = []
    for item in cast('list[object] | tuple[object, ...]', value):
        if not isinstance(item, Mapping):
            return None
        rows.append(cast('Mapping[object, object]', item))
    return tuple(rows)


def _signed_remaining(
    expiry: datetime,
    now: datetime,
    settings: CacheSettings,
) -> AgeDecision:
    """Convert one expiry to a margin-adjusted cap or safe decline."""
    margin = settings.limits.signed_url_safety_margin_seconds
    clock_error = settings.limits.clock_error_bound_seconds
    if clock_error is None:
        return AgeDecision(None, 'clock_error_unbounded')
    if clock_error >= margin:
        return AgeDecision(None, 'clock_error_exceeds_margin')
    remaining = (expiry - now).total_seconds() - margin - clock_error
    if remaining <= 0:
        return AgeDecision(None, 'signed_url_expired')
    return AgeDecision(remaining, 'signed_url')


def _single_parameter(
    parameters: Mapping[str, tuple[str, ...]],
    name: str,
) -> str | None:
    """Return one query parameter only when it occurs exactly once."""
    values = parameters.get(name)
    if values is None or len(values) != 1:
        return None
    return values[0]


def _absolute_expiry(parameters: Mapping[str, tuple[str, ...]]) -> datetime | None:
    """Parse the recognized absolute integer epoch expiry."""
    if 'expires' not in parameters:
        return None
    value = _single_parameter(parameters, 'expires')
    if value is None or not value.isascii() or not value.isdecimal():
        raise ValueError('invalid absolute signed-URL expiry')
    try:
        return datetime.fromtimestamp(int(value), tz=UTC)
    except (OSError, OverflowError, ValueError):
        raise ValueError('invalid absolute signed-URL expiry') from None


def _aws_expiry(parameters: Mapping[str, tuple[str, ...]]) -> datetime | None:
    """Parse the recognized AWS Signature Version 4 expiry pair."""
    has_date = 'x-amz-date' in parameters
    has_duration = 'x-amz-expires' in parameters
    if not has_date and not has_duration:
        return None
    date_value = _single_parameter(parameters, 'x-amz-date')
    duration_value = _single_parameter(parameters, 'x-amz-expires')
    if (
        date_value is None
        or not _is_canonical_aws_date(date_value)
        or duration_value is None
        or not duration_value.isascii()
        or not duration_value.isdecimal()
    ):
        raise ValueError('invalid AWS signed-URL expiry')
    try:
        signed_at = datetime.strptime(date_value, _AWS_DATE_FORMAT).replace(tzinfo=UTC)
        return signed_at + timedelta(seconds=int(duration_value))
    except (OverflowError, ValueError):
        raise ValueError('invalid AWS signed-URL expiry') from None


def _signed_expiry(url: str) -> datetime | None:
    """Return the earliest expiry from either explicitly recognized form."""
    parameters = _casefolded_parameters(url)
    if parameters is None:
        return None
    try:
        expiries = tuple(
            expiry
            for expiry in (_aws_expiry(parameters), _absolute_expiry(parameters))
            if expiry is not None
        )
    except ValueError:
        return None
    if not expiries:
        return None
    return min(expiries)


def _image_url_decision(
    url: str,
    now: datetime,
    settings: CacheSettings,
) -> AgeDecision:
    """Combine independent transient and signed-image caps."""
    try:
        parsed = urlsplit(url)
    except ValueError:
        return AgeDecision(None, 'unrecognized_image_url')
    is_transient = _TRANSIENT_IMAGE_PATH in parsed.path
    expiry = _signed_expiry(url)
    if expiry is None:
        if is_transient:
            return AgeDecision(
                settings.limits.transient_image_max_age_seconds,
                'transient_image',
            )
        return AgeDecision(None, 'unrecognized_image_url')
    signed_decision = _signed_remaining(expiry, now, settings)
    if (
        signed_decision.remaining_seconds is None
        or not is_transient
        or signed_decision.remaining_seconds <= settings.limits.transient_image_max_age_seconds
    ):
        return signed_decision
    return AgeDecision(
        settings.limits.transient_image_max_age_seconds,
        'transient_image',
    )


def _inspect_image(
    value: object,
    now: datetime,
    settings: CacheSettings,
) -> AgeDecision | None:
    """Classify one returned image value without exposing its contents."""
    if value is None or value == '':
        return None
    if not isinstance(value, str):
        return AgeDecision(None, 'unrecognized_image_url')
    return _image_url_decision(value, now, settings)


def _inspect_url(
    value: object,
    now: datetime,
    settings: CacheSettings,
) -> AgeDecision | None:
    """Classify one returned Attachment-style URL value."""
    if value is None:
        return None
    if not isinstance(value, Mapping):
        return AgeDecision(None, 'unrecognized_url_value')
    attachment = cast('Mapping[object, object]', value)
    link_type = attachment.get('link_type')
    if link_type in ('local', 'url'):
        return None
    if link_type != 'upload':
        return AgeDecision(None, 'unrecognized_url_value')
    url = attachment.get('url')
    expiry = _signed_expiry(url) if isinstance(url, str) and url else None
    if expiry is None:
        return AgeDecision(None, 'unrecognized_upload_url')
    return _signed_remaining(expiry, now, settings)


def cap_response_age(
    value: object,
    returned_fields: Mapping[str, str],
    now: datetime,
    settings: CacheSettings,
) -> AgeDecision:
    """Inspect declared URL-bearing fields and derive a safe response-age cap.

    Args:
        value: Direct decoded SG response.
        returned_fields: Returned field names mapped to schema data types.
        now: A timezone-aware time captured when the SG response arrived.
        settings: Immutable loaded cache settings.

    Returns:
        A positive cap safe to combine with the selected age using `min`, or a
        stable decline decision whose remaining lifetime is `None`.

    Raises:
        TypeError: `now` is not a timezone-aware datetime or returned-field
            metadata is malformed.
    """
    response_time = _require_aware_datetime(now)
    fields = _require_returned_fields(returned_fields)
    rows = _response_rows(value)
    if rows is None:
        return AgeDecision(None, 'unrecognized_response_shape')

    utc_now = response_time.astimezone(UTC)
    remaining = settings.global_safety_max_seconds
    reason = 'response_age_unchanged'
    for row in rows:
        for field_name, field_type in fields.items():
            if field_type == 'image':
                decision = _inspect_image(row.get(field_name), utc_now, settings)
            elif field_type == 'url':
                decision = _inspect_url(row.get(field_name), utc_now, settings)
            else:
                continue
            if decision is None:
                continue
            if decision.remaining_seconds is None:
                return decision
            if decision.remaining_seconds < remaining:
                remaining = decision.remaining_seconds
                reason = decision.reason
    return AgeDecision(remaining, reason)
