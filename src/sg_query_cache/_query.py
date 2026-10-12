"""Share immutable SG query argument and mapping grammar."""

from typing import cast

from sg_query_cache._model import ExactCall

FILTER_GROUP_KEYS = frozenset({'filter_operator', 'filters'})
ORDER_ITEM_KEYS = frozenset({'direction', 'field_name'})
_PAIR_LENGTH = 2


def argument(call: ExactCall, name: str) -> object:
    """Return the first named argument, or `None` when absent."""
    for argument_name, value in call.arguments:
        if argument_name == name:
            return value
    return None


def pair_mapping(value: object) -> dict[str, object] | None:
    """Parse unique string-keyed tuple pairs, returning `None` for other shapes."""
    if not isinstance(value, tuple):
        return None
    mapping: dict[str, object] = {}
    for entry in cast('tuple[object, ...]', value):
        if not isinstance(entry, tuple):
            return None
        parts = cast('tuple[object, ...]', entry)
        if len(parts) != _PAIR_LENGTH or not isinstance(parts[0], str):
            return None
        if parts[0] in mapping:
            return None
        mapping[parts[0]] = parts[1]
    return mapping


def filter_group(value: tuple[object, ...]) -> tuple[bool, tuple[object, ...] | None]:
    """Distinguish simple conditions, valid groups, and malformed groups.

    Returns:
        Whether the exact filter-group keys are present, and the child tuple
        when the group's operator and child container are valid. A malformed
        group returns `(True, None)`; other shapes return `(False, None)`.
    """
    if len(value) != _PAIR_LENGTH:
        return False, None
    mapping = pair_mapping(value)
    if mapping is None or frozenset(mapping) != FILTER_GROUP_KEYS:
        return False, None
    filters = mapping['filters']
    if mapping['filter_operator'] not in ('all', 'any') or not isinstance(filters, tuple):
        return True, None
    return True, cast('tuple[object, ...]', filters)
