"""Validate input shapes shared by package components."""

import math


def require_bool(value: object, owner: str) -> bool:
    """Return a strict boolean after validating its type.

    Args:
        value: Candidate boolean value.
        owner: Name included in a validation error.

    Raises:
        TypeError: The value is not a boolean.
    """
    if not isinstance(value, bool):
        raise TypeError(f'{owner} must be a boolean')
    return value


def require_nonempty_string(value: object, owner: str) -> str:
    """Return a nonempty string without normalization.

    Args:
        value: Candidate string value.
        owner: Name included in a validation error.

    Raises:
        TypeError: The value is not a string.
        ValueError: The string is empty.
    """
    if not isinstance(value, str):
        raise TypeError(f'{owner} must be a string')
    if not value:
        raise ValueError(f'{owner} must be nonempty')
    return value


def require_positive_int(value: object, owner: str) -> None:
    """Require a positive non-boolean integer.

    Args:
        value: Candidate integer value.
        owner: Name included in a validation error.

    Raises:
        TypeError: The value is not an integer or is a boolean.
        ValueError: The integer is not positive.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f'{owner} must be an integer')
    if value <= 0:
        raise ValueError(f'{owner} must be positive')


def require_positive_number(value: object, owner: str) -> None:
    """Require a finite positive non-boolean number.

    Args:
        value: Candidate integer or float value.
        owner: Name included in a validation error.

    Raises:
        TypeError: The value is a boolean, nonnumeric, or nonfinite.
        ValueError: The number is not positive.
    """
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise TypeError(f'{owner} must be a finite number')
    if value <= 0:
        raise ValueError(f'{owner} must be positive')
