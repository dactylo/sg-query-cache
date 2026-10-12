"""Classify SG query filters through a closed documented operator grammar."""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal, Protocol, TypeAlias, cast

from sg_query_cache._model import ExactCall
from sg_query_cache._query import argument, filter_group

_QueryClassification: TypeAlias = Literal['bypass', 'event_aware', 'ttl_only']

URL_PROVEN_TTL_ONLY_METHODS = frozenset({'work_schedule_read'})
_ABSOLUTE_DATE_OPERATORS = frozenset(
    {'between', 'greater_than', 'in', 'is', 'is_not', 'less_than', 'not_in'}
)
_ADDRESSING_OPERATORS = frozenset(
    {
        'contains',
        'in',
        'is',
        'is_not',
        'name_contains',
        'name_ends_with',
        'name_not_contains',
        'name_starts_with',
        'not_contains',
        'type_is',
        'type_is_not',
    }
)
_CONDITION_LENGTH = 3
_DOCUMENTED_OPERATORS: Mapping[str, frozenset[str]] = MappingProxyType(
    {
        'addressing': _ADDRESSING_OPERATORS,
        'checkbox': frozenset({'is', 'is_not'}),
        'currency': frozenset(
            {
                'between',
                'greater_than',
                'in',
                'is',
                'is_not',
                'less_than',
                'not_between',
                'not_in',
            }
        ),
        'date': _ABSOLUTE_DATE_OPERATORS
        | frozenset(
            {
                'in_calendar_day',
                'in_calendar_month',
                'in_calendar_week',
                'in_calendar_year',
                'in_last',
                'in_next',
                'not_in_last',
                'not_in_next',
            }
        ),
        'date_time': _ABSOLUTE_DATE_OPERATORS
        | frozenset(
            {
                'in_calendar_day',
                'in_calendar_month',
                'in_calendar_week',
                'in_calendar_year',
                'in_last',
                'in_next',
                'not_in_last',
                'not_in_next',
            }
        ),
        'duration': frozenset(
            {'between', 'greater_than', 'in', 'is', 'is_not', 'less_than', 'not_in'}
        ),
        'entity': frozenset(
            {
                'in',
                'is',
                'is_not',
                'name_contains',
                'name_is',
                'name_not_contains',
                'not_in',
                'type_is',
                'type_is_not',
            }
        ),
        'float': frozenset(
            {'between', 'greater_than', 'in', 'is', 'is_not', 'less_than', 'not_in'}
        ),
        'image': frozenset({'is', 'is_not'}),
        'list': frozenset({'in', 'is', 'is_not', 'not_in'}),
        'multi_entity': frozenset(
            {
                'in',
                'is',
                'is_not',
                'name_contains',
                'name_not_contains',
                'not_in',
                'type_is',
                'type_is_not',
            }
        ),
        'number': frozenset(
            {
                'between',
                'greater_than',
                'in',
                'is',
                'is_not',
                'less_than',
                'not_between',
                'not_in',
            }
        ),
        'percent': frozenset(
            {'between', 'greater_than', 'in', 'is', 'is_not', 'less_than', 'not_in'}
        ),
        'status_list': frozenset({'in', 'is', 'is_not', 'not_in'}),
        'tag_list': frozenset(
            {
                'in',
                'is',
                'is_not',
                'name_contains',
                'name_is',
                'name_not_contains',
                'not_in',
                'type_is',
                'type_is_not',
            }
        ),
        'text': frozenset(
            {
                'contains',
                'ends_with',
                'in',
                'is',
                'is_not',
                'not_contains',
                'not_in',
                'starts_with',
            }
        ),
        'timecode': frozenset(
            {'between', 'greater_than', 'in', 'is', 'is_not', 'less_than', 'not_in'}
        ),
    }
)
_EVENT_AWARE_OPERATORS: Mapping[str, frozenset[str]] = MappingProxyType(
    {
        **_DOCUMENTED_OPERATORS,
        'addressing': frozenset(),
        'date': _ABSOLUTE_DATE_OPERATORS,
        'date_time': _ABSOLUTE_DATE_OPERATORS,
    }
)
_UNFILTERABLE_TYPES = frozenset({'password', 'serializable', 'summary', 'url'})


@dataclass(frozen=True, slots=True)
class FieldSchema:
    """Describe the schema facts needed to classify one SG field.

    Attributes:
        data_type: Documented SG field data type.
        valid_types: Schema-ordered linked entity types, when applicable.
    """

    data_type: str
    valid_types: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validate the field type and schema-ordered linked entity names.

        Raises:
            TypeError: The data type or linked-type tuple has the wrong type.
            ValueError: The data type is empty or a linked entity name is empty
                or duplicated.
        """
        _validate_field_schema(self.data_type, self.valid_types)


class _SchemaLookup(Protocol):
    """Supply conservative field lookup to the query classifier."""

    def resolve_field(self, entity_type: str, field_path: str) -> FieldSchema | None:
        """Return the final field schema only when every path hop is known."""
        ...


@dataclass(frozen=True, slots=True)
class QueryPolicy:
    """Report one cache classification and stable reason code.

    Attributes:
        classification: `event_aware`, `ttl_only`, or `bypass`.
        reason: Stable low-cardinality reason for the decision.
    """

    classification: _QueryClassification
    reason: str

    def __post_init__(self) -> None:
        """Validate the cache classification and stable decision reason.

        Raises:
            TypeError: The reason is not a string.
            ValueError: The classification is unsupported or the reason is
                empty.
        """
        if self.classification not in ('bypass', 'event_aware', 'ttl_only'):
            raise ValueError('query classification is invalid')
        _require_reason(self.reason)


def _classify_conditions(
    entity_type: str,
    conditions: tuple[object, ...],
    schema: _SchemaLookup,
) -> QueryPolicy:
    """Walk all conditions while giving bypass and TTL-only decisions priority."""
    fallback = QueryPolicy('event_aware', 'event_aware')
    for condition in conditions:
        policy = _classify_condition(entity_type, condition, schema)
        if policy.classification == 'bypass':
            return policy
        if (policy.classification == 'ttl_only' and fallback.classification == 'event_aware') or (
            policy.reason == 'tag_list_contains' and fallback.reason == 'event_aware'
        ):
            fallback = policy
    return fallback


def _classify_documented_field(
    field_type: str,
    operator: str,
    value: object,
) -> QueryPolicy:
    """Apply event-aware special cases to one documented type/operator pair."""
    if field_type == 'image' and value is not None:
        return QueryPolicy('bypass', 'unsupported_image_value')
    if operator not in _EVENT_AWARE_OPERATORS[field_type]:
        reason = 'addressing' if field_type == 'addressing' else 'relative_date_operator'
        return QueryPolicy('ttl_only', reason)
    if field_type == 'tag_list' and operator == 'is':
        return QueryPolicy('event_aware', 'tag_list_contains')
    return QueryPolicy('event_aware', 'event_aware')


def _classify_field(field: FieldSchema, operator: str, value: object) -> QueryPolicy:
    """Classify one schema-resolved condition through the closed grammar."""
    documented = _DOCUMENTED_OPERATORS.get(field.data_type)
    if documented is None:
        reason = (
            'unfilterable_field_type'
            if field.data_type in _UNFILTERABLE_TYPES
            else 'unknown_field_type'
        )
        return QueryPolicy('bypass', reason)
    if operator not in documented:
        return QueryPolicy('bypass', 'unknown_operator')
    return _classify_documented_field(field.data_type, operator, value)


def _classify_condition(
    entity_type: str,
    condition: object,
    schema: _SchemaLookup,
) -> QueryPolicy:
    """Classify one simple condition or nested filter group."""
    if not isinstance(condition, tuple):
        return QueryPolicy('bypass', 'malformed_filters')
    values = cast('tuple[object, ...]', condition)
    is_group, group = filter_group(values)
    if is_group and group is None:
        return QueryPolicy('bypass', 'unknown_filter_group')
    if group is not None:
        return _classify_conditions(entity_type, group, schema)
    if (
        len(values) != _CONDITION_LENGTH
        or not isinstance(values[0], str)
        or not isinstance(values[1], str)
    ):
        return QueryPolicy('bypass', 'malformed_filters')
    field = schema.resolve_field(entity_type, values[0])
    if field is None:
        return QueryPolicy('bypass', 'unknown_field')
    return _classify_field(field, values[1], values[2])


def _require_reason(value: object) -> None:
    """Require a stable nonempty policy reason string."""
    if not isinstance(value, str):
        raise TypeError('query policy reason must be a string')
    if not value:
        raise ValueError('query policy reason must be nonempty')


def _string_argument(call: ExactCall, name: str) -> str | None:
    """Return one nonempty string argument or report absence."""
    value = argument(call, name)
    if not isinstance(value, str) or not value:
        return None
    return value


def _query_boundary(call: ExactCall) -> tuple[str | None, QueryPolicy | None]:
    """Classify method-wide semantics before walking individual conditions."""
    if call.method not in ('find', 'find_one'):
        return None, QueryPolicy('bypass', 'unsupported_method')
    entity_type = _string_argument(call, 'entity_type')
    if entity_type is None:
        return None, QueryPolicy('bypass', 'missing_entity_type')
    if entity_type == 'EventLogEntry':
        return None, QueryPolicy('bypass', 'event_log_entry')
    if argument(call, 'filter_operator') not in (None, 'all', 'any'):
        return None, QueryPolicy('bypass', 'unknown_filter_operator')
    if argument(call, 'additional_filter_presets') not in (None, ()):
        return entity_type, QueryPolicy('ttl_only', 'additional_filter_presets')
    return entity_type, None


def _validate_field_schema(data_type: object, valid_types: object) -> None:
    """Validate immutable field facts even across untyped construction boundaries."""
    if not isinstance(data_type, str):
        raise TypeError('field schema data type must be a string')
    if not data_type:
        raise ValueError('field schema data type must be nonempty')
    if not isinstance(valid_types, tuple):
        raise TypeError('field schema valid types must be a tuple')
    values = cast('tuple[object, ...]', valid_types)
    if any(not isinstance(item, str) for item in values):
        raise TypeError('field schema valid types must contain strings')
    typed_values = cast('tuple[str, ...]', values)
    if any(not item for item in typed_values):
        raise ValueError('field schema valid types must be nonempty')
    if len(typed_values) != len(set(typed_values)):
        raise ValueError('field schema valid types must be unique')


def classify_query(call: ExactCall, schema: _SchemaLookup) -> QueryPolicy:
    """Classify a normalized query without evaluating or validating it for SG.

    Args:
        call: Immutable explicitly bound SG call.
        schema: Conservative schema lookup for field paths.

    Returns:
        The strongest safe first-release query classification.
    """
    entity_type, boundary_policy = _query_boundary(call)
    if boundary_policy is not None:
        return boundary_policy
    if entity_type is None:
        return QueryPolicy('bypass', 'missing_entity_type')
    filters = argument(call, 'filters')
    if filters is None:
        return QueryPolicy('event_aware', 'event_aware')
    if not isinstance(filters, tuple):
        return QueryPolicy('bypass', 'malformed_filters')
    return _classify_conditions(entity_type, cast('tuple[object, ...]', filters), schema)


def is_known_field_type(field_type: str) -> bool:
    """Report whether the dependency policy recognizes an SG field type.

    Args:
        field_type: SG schema field data type.

    Returns:
        `True` only for documented filter types or supported return-only types.
    """
    return field_type in _DOCUMENTED_OPERATORS or field_type in _UNFILTERABLE_TYPES
