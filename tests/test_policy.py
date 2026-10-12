from collections.abc import Mapping
from typing import cast

import pytest

from sg_query_cache._model import ExactCall, ExactValue
from sg_query_cache._policy import (
    _EVENT_AWARE_OPERATORS,
    FieldSchema,
    QueryPolicy,
    classify_query,
)

_DOCUMENTED_OPERATORS = {
    'addressing': (
        'is',
        'is_not',
        'contains',
        'not_contains',
        'in',
        'type_is',
        'type_is_not',
        'name_contains',
        'name_not_contains',
        'name_starts_with',
        'name_ends_with',
    ),
    'checkbox': ('is', 'is_not'),
    'currency': (
        'is',
        'is_not',
        'less_than',
        'greater_than',
        'between',
        'not_between',
        'in',
        'not_in',
    ),
    'number': (
        'is',
        'is_not',
        'less_than',
        'greater_than',
        'between',
        'not_between',
        'in',
        'not_in',
    ),
    'date': (
        'is',
        'is_not',
        'greater_than',
        'less_than',
        'in_last',
        'not_in_last',
        'in_next',
        'not_in_next',
        'in_calendar_day',
        'in_calendar_week',
        'in_calendar_month',
        'in_calendar_year',
        'between',
        'in',
        'not_in',
    ),
    'date_time': (
        'is',
        'is_not',
        'greater_than',
        'less_than',
        'in_last',
        'not_in_last',
        'in_next',
        'not_in_next',
        'in_calendar_day',
        'in_calendar_week',
        'in_calendar_month',
        'in_calendar_year',
        'between',
        'in',
        'not_in',
    ),
    'duration': (
        'is',
        'is_not',
        'greater_than',
        'less_than',
        'between',
        'in',
        'not_in',
    ),
    'float': (
        'is',
        'is_not',
        'greater_than',
        'less_than',
        'between',
        'in',
        'not_in',
    ),
    'percent': (
        'is',
        'is_not',
        'greater_than',
        'less_than',
        'between',
        'in',
        'not_in',
    ),
    'timecode': (
        'is',
        'is_not',
        'greater_than',
        'less_than',
        'between',
        'in',
        'not_in',
    ),
    'entity': (
        'is',
        'is_not',
        'type_is',
        'type_is_not',
        'name_contains',
        'name_not_contains',
        'name_is',
        'in',
        'not_in',
    ),
    'image': ('is', 'is_not'),
    'list': ('is', 'is_not', 'in', 'not_in'),
    'status_list': ('is', 'is_not', 'in', 'not_in'),
    'multi_entity': (
        'is',
        'is_not',
        'type_is',
        'type_is_not',
        'name_contains',
        'name_not_contains',
        'in',
        'not_in',
    ),
    'tag_list': (
        'is',
        'is_not',
        'type_is',
        'type_is_not',
        'name_contains',
        'name_not_contains',
        'name_is',
        'in',
        'not_in',
    ),
    'text': (
        'is',
        'is_not',
        'contains',
        'not_contains',
        'starts_with',
        'ends_with',
        'in',
        'not_in',
    ),
}
_EXPECTED_EVENT_AWARE_OPERATORS = {
    'addressing': (),
    'checkbox': ('is', 'is_not'),
    'currency': (
        'is',
        'is_not',
        'less_than',
        'greater_than',
        'between',
        'not_between',
        'in',
        'not_in',
    ),
    'number': (
        'is',
        'is_not',
        'less_than',
        'greater_than',
        'between',
        'not_between',
        'in',
        'not_in',
    ),
    'date': ('is', 'is_not', 'greater_than', 'less_than', 'between', 'in', 'not_in'),
    'date_time': (
        'is',
        'is_not',
        'greater_than',
        'less_than',
        'between',
        'in',
        'not_in',
    ),
    'duration': (
        'is',
        'is_not',
        'greater_than',
        'less_than',
        'between',
        'in',
        'not_in',
    ),
    'float': (
        'is',
        'is_not',
        'greater_than',
        'less_than',
        'between',
        'in',
        'not_in',
    ),
    'percent': (
        'is',
        'is_not',
        'greater_than',
        'less_than',
        'between',
        'in',
        'not_in',
    ),
    'timecode': (
        'is',
        'is_not',
        'greater_than',
        'less_than',
        'between',
        'in',
        'not_in',
    ),
    'entity': (
        'is',
        'is_not',
        'type_is',
        'type_is_not',
        'name_contains',
        'name_not_contains',
        'name_is',
        'in',
        'not_in',
    ),
    'image': ('is', 'is_not'),
    'list': ('is', 'is_not', 'in', 'not_in'),
    'status_list': ('is', 'is_not', 'in', 'not_in'),
    'multi_entity': (
        'is',
        'is_not',
        'type_is',
        'type_is_not',
        'name_contains',
        'name_not_contains',
        'in',
        'not_in',
    ),
    'tag_list': (
        'is',
        'is_not',
        'type_is',
        'type_is_not',
        'name_contains',
        'name_not_contains',
        'name_is',
        'in',
        'not_in',
    ),
    'text': (
        'is',
        'is_not',
        'contains',
        'not_contains',
        'starts_with',
        'ends_with',
        'in',
        'not_in',
    ),
}
_RELATIVE_DATE_OPERATORS = (
    'in_last',
    'not_in_last',
    'in_next',
    'not_in_next',
    'in_calendar_day',
    'in_calendar_week',
    'in_calendar_month',
    'in_calendar_year',
)
_UNFILTERABLE_TYPES = ('password', 'serializable', 'summary', 'url')


class _Schema:
    def __init__(self, fields: Mapping[str, FieldSchema]) -> None:
        self._fields = fields

    def resolve_field(self, entity_type: str, field_path: str) -> FieldSchema | None:
        if entity_type != 'Asset':
            return None
        return self._fields.get(field_path)


def _call(field_name: str, operator: str, value: ExactValue = 'value') -> ExactCall:
    return ExactCall(
        'find',
        (
            ('entity_type', 'Asset'),
            ('filters', ((field_name, operator, value),)),
            ('additional_filter_presets', None),
        ),
    )


def _schema() -> _Schema:
    return _Schema(
        {
            field_type: FieldSchema(field_type)
            for field_type in (*_DOCUMENTED_OPERATORS, *_UNFILTERABLE_TYPES)
        }
    )


def test_event_aware_operator_table_is_exact_and_immutable() -> None:
    expected = {
        field_type: frozenset(operators)
        for field_type, operators in _EXPECTED_EVENT_AWARE_OPERATORS.items()
    }

    assert expected == _EVENT_AWARE_OPERATORS
    with pytest.raises(TypeError):
        cast('dict[str, frozenset[str]]', _EVENT_AWARE_OPERATORS)['text'] = frozenset()


def test_event_log_entry_reads_always_bypass() -> None:
    call = ExactCall(
        'find',
        (
            ('entity_type', 'EventLogEntry'),
            ('filters', ()),
            ('fields', ('id',)),
        ),
    )

    assert classify_query(call, _schema()) == QueryPolicy(
        'bypass',
        'event_log_entry',
    )


@pytest.mark.parametrize(
    ('field_type', 'operator', 'expected_classification'),
    [
        (
            field_type,
            operator,
            (
                'event_aware'
                if operator in _EXPECTED_EVENT_AWARE_OPERATORS[field_type]
                else 'ttl_only'
            ),
        )
        for field_type, operators in _DOCUMENTED_OPERATORS.items()
        for operator in operators
    ],
)
def test_every_documented_pair_has_exact_event_awareness(
    field_type: str,
    operator: str,
    expected_classification: str,
) -> None:
    value = None if field_type == 'image' else 'value'

    policy = classify_query(_call(field_type, operator, value), _schema())

    assert policy.classification == expected_classification


@pytest.mark.parametrize('operator', _DOCUMENTED_OPERATORS['addressing'])
def test_addressing_operators_are_ttl_only(operator: str) -> None:
    assert classify_query(_call('addressing', operator), _schema()) == QueryPolicy(
        'ttl_only',
        'addressing',
    )


def test_nested_filter_groups_are_walked_completely() -> None:
    call = ExactCall(
        'find',
        (
            ('entity_type', 'Asset'),
            (
                'filters',
                (
                    (
                        ('filter_operator', 'any'),
                        (
                            'filters',
                            (
                                ('text', 'contains', 'needle'),
                                ('date', 'in_last', 7),
                            ),
                        ),
                    ),
                ),
            ),
        ),
    )

    assert classify_query(call, _schema()) == QueryPolicy(
        'ttl_only',
        'relative_date_operator',
    )


@pytest.mark.parametrize('field_type', ['date', 'date_time'])
@pytest.mark.parametrize('operator', _RELATIVE_DATE_OPERATORS)
def test_relative_date_operators_are_ttl_only(field_type: str, operator: str) -> None:
    assert classify_query(_call(field_type, operator), _schema()) == QueryPolicy(
        'ttl_only',
        'relative_date_operator',
    )


@pytest.mark.parametrize('operator', ['is', 'is_not'])
def test_image_filters_require_none(operator: str) -> None:
    assert classify_query(_call('image', operator, 'https://example.invalid'), _schema()) == (
        QueryPolicy('bypass', 'unsupported_image_value')
    )
    assert classify_query(_call('image', operator, None), _schema()) == QueryPolicy(
        'event_aware',
        'event_aware',
    )


def test_tag_list_is_uses_the_supported_contains_policy() -> None:
    value = (('type', 'Tag'), ('id', 7))

    assert classify_query(_call('tag_list', 'is', value), _schema()) == QueryPolicy(
        'event_aware',
        'tag_list_contains',
    )


@pytest.mark.parametrize('field_type', _UNFILTERABLE_TYPES)
def test_unfilterable_field_types_bypass(field_type: str) -> None:
    assert classify_query(_call(field_type, 'is'), _schema()) == QueryPolicy(
        'bypass',
        'unfilterable_field_type',
    )


def test_unknown_field_bypasses_conservatively() -> None:
    assert classify_query(_call('missing', 'is'), _schema()) == QueryPolicy(
        'bypass',
        'unknown_field',
    )


def test_unknown_nested_filter_operator_bypasses() -> None:
    call = ExactCall(
        'find',
        (
            ('entity_type', 'Asset'),
            (
                'filters',
                (
                    (
                        ('filter_operator', 'future_operator'),
                        ('filters', (('text', 'contains', 'needle'),)),
                    ),
                ),
            ),
        ),
    )

    assert classify_query(call, _schema()) == QueryPolicy(
        'bypass',
        'unknown_filter_group',
    )


def test_unknown_operator_bypasses_without_raising() -> None:
    assert classify_query(_call('text', 'future_operator'), _schema()) == QueryPolicy(
        'bypass',
        'unknown_operator',
    )


def test_unknown_top_level_filter_operator_bypasses() -> None:
    call = ExactCall(
        'find',
        (
            ('entity_type', 'Asset'),
            ('filters', (('text', 'contains', 'needle'),)),
            ('filter_operator', 'future_operator'),
        ),
    )

    assert classify_query(call, _schema()) == QueryPolicy(
        'bypass',
        'unknown_filter_operator',
    )
