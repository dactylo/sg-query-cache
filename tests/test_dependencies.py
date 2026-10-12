import json
import zlib
from pathlib import Path

import pytest

from sg_query_cache._codec import MAX_BYTES, encode_value, encoded_size
from sg_query_cache._config import LimitSettings
from sg_query_cache._dependencies import DependencyAnalyzer, SchemaView
from sg_query_cache._keys import KeyBuilder
from sg_query_cache._model import (
    DependencyPlan,
    DependencyTag,
    ExactCall,
    ExactValue,
    Identity,
    TagKind,
)
from sg_query_cache._policy import FieldSchema

_FIXTURE = Path(__file__).parent / 'fixtures' / 'schema.json'
_LIMITS = LimitSettings()
_SITE = 'production'


def _call(
    *,
    filters: tuple[ExactValue, ...] = (),
    fields: tuple[ExactValue, ...] | None = ('id',),
    order: tuple[ExactValue, ...] = (),
    include_archived_projects: bool = True,
    additional_filter_presets: tuple[ExactValue, ...] | None = None,
) -> ExactCall:
    arguments: tuple[tuple[str, ExactValue], ...] = (
        ('entity_type', 'Asset'),
        ('filters', filters),
        ('fields', fields),
        ('order', order),
        ('filter_operator', 'all'),
        ('limit', 0),
        ('retired_only', False),
        ('page', 0),
        ('include_archived_projects', include_archived_projects),
        ('additional_filter_presets', additional_filter_presets),
    )
    return ExactCall(
        'find',
        arguments,
    )


def _human() -> Identity:
    return Identity('human', 'studio-sso', 'human:42', entity_id=42)


def _replace_argument(call: ExactCall, name: str, value: ExactValue) -> ExactCall:
    return ExactCall(
        call.method,
        tuple(
            (argument_name, value if argument_name == name else argument_value)
            for argument_name, argument_value in call.arguments
        ),
    )


def _schema() -> SchemaView:
    value = json.loads(_FIXTURE.read_text(encoding='utf-8'))
    return SchemaView.from_api_response(value)


def _plan(
    call: ExactCall,
    *,
    analyzer: DependencyAnalyzer | None = None,
    identity: Identity | None = None,
    limits: LimitSettings = _LIMITS,
    schema: SchemaView | None = None,
) -> DependencyPlan:
    return (analyzer or DependencyAnalyzer()).plan(
        call,
        identity or Identity('script', 'studio-sso', 'render-farm'),
        _SITE,
        schema or _schema(),
        limits,
    )


def test_additional_filter_presets_are_disabled_by_default() -> None:
    call = _call(additional_filter_presets=(('preset_name', 'my_tasks'),))

    disabled = _plan(call)
    enabled = _plan(
        call,
        analyzer=DependencyAnalyzer(enable_additional_filter_presets=True),
    )

    assert disabled.classification == 'bypass'
    assert disabled.reason == 'additional_filter_presets_disabled'
    assert disabled.tags == ()
    assert enabled.classification == 'ttl_only'
    assert enabled.reason == 'additional_filter_presets'
    assert enabled.tags == (
        DependencyTag.site(_SITE),
        DependencyTag.authorization(_SITE),
    )


def test_archived_project_exclusion_uses_broad_project_state() -> None:
    plan = _plan(_call(include_archived_projects=False))

    assert plan.tags[-2:] == (
        DependencyTag.type_membership(_SITE, 'Project'),
        DependencyTag.type_value(_SITE, 'Project'),
    )


def test_compact_schema_view_enforces_the_expanded_size_boundary() -> None:
    # This otherwise-valid document is exactly the documented 16 MiB limit.
    entity_type = 'A' * (16 * 1024 * 1024 - len(b'{"":{}}'))
    document = b'{"' + entity_type.encode() + b'":{}}'

    assert SchemaView.from_compact(zlib.compress(document)) == SchemaView(((entity_type, ()),))
    with pytest.raises(ValueError, match='compact schema view is malformed'):
        SchemaView.from_compact(zlib.compress(b'{"A' + document[2:]))


def test_compact_schema_view_round_trips_and_is_one_codec_node() -> None:
    view = _schema()
    compact = view.to_compact()

    assert SchemaView.from_compact(compact) == view
    assert isinstance(compact, bytes)
    assert encoded_size(compact) < 4 * len(compact)


def test_compact_schema_view_stays_within_the_codec_bounds_for_large_sites() -> None:
    response = {
        f'CustomEntity{index:02d}': {
            f'sg_field_{field:03d}': {
                'data_type': {'value': 'entity' if field % 3 == 0 else 'text'},
                'properties': (
                    {'valid_types': {'value': ['Shot', 'Asset']}} if field % 3 == 0 else {}
                ),
            }
            for field in range(100)
        }
        for index in range(120)
    }
    compact = SchemaView.from_api_response(response).to_compact()

    assert len(encode_value(compact)) < MAX_BYTES


@pytest.mark.parametrize('fields', [None, ()])
def test_default_or_empty_fields_add_no_value_dependency(
    fields: tuple[ExactValue, ...] | None,
) -> None:
    plan = _plan(_call(fields=fields))

    assert plan.tags == (
        DependencyTag.site(_SITE),
        DependencyTag.schema(_SITE),
        DependencyTag.authorization(_SITE),
        DependencyTag.type_membership(_SITE, 'Asset'),
    )


def test_disjunctive_id_filters_retain_base_membership() -> None:
    call = _call(filters=(('id', 'is', 7), ('status', 'is', 'ip')))

    plan = _plan(_replace_argument(call, 'filter_operator', 'any'))

    assert DependencyTag.type_membership(_SITE, 'Asset') in plan.tags
    assert DependencyTag.entity(_SITE, 'Asset', 7) not in plan.tags


def test_dotted_path_covers_every_hop_and_final_field() -> None:
    path = 'task.Task.step.Step.code'

    plan = _plan(_call(filters=((path, 'contains', 'anim'),)))

    assert plan.tags == (
        DependencyTag.site(_SITE),
        DependencyTag.schema(_SITE),
        DependencyTag.authorization(_SITE),
        DependencyTag.type_membership(_SITE, 'Asset'),
        DependencyTag.type_field(_SITE, 'Asset', 'task'),
        DependencyTag.type_membership(_SITE, 'Task'),
        DependencyTag.type_field(_SITE, 'Task', 'step'),
        DependencyTag.type_membership(_SITE, 'Step'),
        DependencyTag.type_field(_SITE, 'Step', 'code'),
    )


def test_dotted_returned_link_covers_final_display_name_targets() -> None:
    path = 'task.Task.step.Step.owner'

    plan = _plan(_call(fields=('id', path)))

    assert plan.tags[-7:] == (
        DependencyTag.type_field(_SITE, 'Asset', 'task'),
        DependencyTag.type_membership(_SITE, 'Task'),
        DependencyTag.type_field(_SITE, 'Task', 'step'),
        DependencyTag.type_membership(_SITE, 'Step'),
        DependencyTag.type_field(_SITE, 'Step', 'owner'),
        DependencyTag.type_membership(_SITE, 'HumanUser'),
        DependencyTag.type_value(_SITE, 'HumanUser'),
    )


def test_general_query_dependencies_follow_authoritative_order() -> None:
    plan = _plan(
        _call(
            filters=(('status', 'is', 'ip'),),
            fields=('id', 'code'),
            order=((('field_name', 'created_at'), ('direction', 'desc')),),
        )
    )

    assert plan.classification == 'event_aware'
    assert plan.reason == 'event_aware'
    assert plan.tags == (
        DependencyTag.site(_SITE),
        DependencyTag.schema(_SITE),
        DependencyTag.authorization(_SITE),
        DependencyTag.type_membership(_SITE, 'Asset'),
        DependencyTag.type_field(_SITE, 'Asset', 'status'),
        DependencyTag.type_field(_SITE, 'Asset', 'code'),
        DependencyTag.type_field(_SITE, 'Asset', 'created_at'),
    )


def test_human_plans_include_global_and_exact_visibility() -> None:
    plan = _plan(_call(), identity=_human())

    assert plan.tags[:5] == (
        DependencyTag.site(_SITE),
        DependencyTag.schema(_SITE),
        DependencyTag.authorization(_SITE),
        DependencyTag.human_visibility(_SITE),
        DependencyTag.identity_visibility(_SITE, 42),
    )


def test_large_id_prefix_does_not_hide_a_late_malformed_condition() -> None:
    valid_prefix = tuple(('id', 'is', entity_id) for entity_id in range(1, 1001))

    plan = _plan(_call(filters=(*valid_prefix, ('id', 'is'))))

    assert plan == DependencyPlan('bypass', (), 'malformed_filters')


def test_malformed_or_permission_hidden_schema_facts_are_unknown() -> None:
    schema = SchemaView.from_api_response(
        {
            'Asset': {
                'missing_type': {'data_type': {'editable': False}},
                'bad_targets': {
                    'data_type': {'value': 'entity'},
                    'properties': {'valid_types': {'value': ['Shot', 7]}},
                },
            },
            'Broken': [],
        }
    )

    assert schema.resolve_field('Asset', 'missing_type') is None
    assert schema.resolve_field('Asset', 'bad_targets') is None
    assert schema.resolve_field('Broken', 'id') is None


def test_project_filter_adds_field_and_target_membership() -> None:
    projects = (
        (('type', 'Project'), ('id', 12)),
        (('type', 'Project'), ('id', 14)),
    )

    plan = _plan(_call(filters=(('project', 'in', projects),)))

    assert plan.tags[-2:] == (
        DependencyTag.type_field(_SITE, 'Asset', 'project'),
        DependencyTag.type_membership(_SITE, 'Project'),
    )


def test_project_filters_add_no_project_tag() -> None:
    plan = _plan(_call(filters=(('project', 'is', (('type', 'Project'), ('id', 5))),)))

    assert all(tag.kind is not TagKind.PROJECT for tag in plan.tags)
    assert DependencyTag.type_field(_SITE, 'Asset', 'project') in plan.tags
    assert 'broaden:project_to_type' not in plan.broadening_steps


@pytest.mark.parametrize(
    ('field_name', 'target_tags'),
    [
        (
            'linked',
            (
                DependencyTag.type_membership(_SITE, 'Shot'),
                DependencyTag.type_value(_SITE, 'Shot'),
                DependencyTag.type_value(_SITE, 'Asset'),
            ),
        ),
        (
            'related',
            (
                DependencyTag.type_membership(_SITE, 'Shot'),
                DependencyTag.type_value(_SITE, 'Shot'),
                DependencyTag.type_value(_SITE, 'Asset'),
            ),
        ),
        (
            'tags',
            (
                DependencyTag.type_membership(_SITE, 'Tag'),
                DependencyTag.type_value(_SITE, 'Tag'),
            ),
        ),
    ],
)
def test_returned_links_cover_display_name_target_dependencies(
    field_name: str,
    target_tags: tuple[DependencyTag, ...],
) -> None:
    plan = _plan(_call(fields=('id', field_name)))

    expected = (
        DependencyTag.site(_SITE),
        DependencyTag.schema(_SITE),
        DependencyTag.authorization(_SITE),
        DependencyTag.type_membership(_SITE, 'Asset'),
        DependencyTag.type_field(_SITE, 'Asset', field_name),
        *target_tags,
    )
    assert plan.tags == expected


@pytest.mark.parametrize(
    'operator',
    ['name_contains', 'name_not_contains', 'name_is'],
)
def test_name_filters_cover_all_link_target_values(operator: str) -> None:
    plan = _plan(_call(filters=(('linked', operator, 'hero'),)))

    assert plan.tags == (
        DependencyTag.site(_SITE),
        DependencyTag.schema(_SITE),
        DependencyTag.authorization(_SITE),
        DependencyTag.type_membership(_SITE, 'Asset'),
        DependencyTag.type_field(_SITE, 'Asset', 'linked'),
        DependencyTag.type_membership(_SITE, 'Shot'),
        DependencyTag.type_value(_SITE, 'Shot'),
        DependencyTag.type_value(_SITE, 'Asset'),
    )


def test_schema_lookup_does_not_change_value_identity_or_compact_form() -> None:
    view = _schema()
    equal_view = SchemaView.from_compact(view.to_compact())
    original_repr = repr(view)
    original_compact = view.to_compact()
    original_hash = hash(view)

    assert view.resolve_field('Asset', 'code') == FieldSchema('text')
    assert view == equal_view
    assert hash(view) == original_hash == hash(equal_view)
    assert repr(view) == original_repr == repr(equal_view)
    assert view.to_compact() == original_compact == equal_view.to_compact()
    assert '_field_index' not in repr(view)


def test_schema_lookup_preserves_first_entity_and_first_field() -> None:
    first = FieldSchema('text')
    later = FieldSchema('number')
    view = SchemaView(
        (
            ('Asset', (('code', first), ('code', later))),
            ('Asset', (('hidden', later),)),
        )
    )

    assert view.resolve_field('Asset', 'code') == first
    assert view.resolve_field('Asset', 'hidden') is None
    assert view.resolve_field('Missing', 'code') is None


def test_schema_view_parses_only_required_field_facts() -> None:
    schema = _schema()

    assert schema.resolve_field('Asset', 'code') == FieldSchema('text')
    assert schema.resolve_field('Asset', 'linked') == FieldSchema(
        'entity',
        ('Shot', 'Asset'),
    )
    assert schema.resolve_field('Asset', 'missing') is None
    assert schema.resolve_field('Missing', 'code') is None


def test_tag_limit_uses_each_broadening_class_in_order() -> None:
    fields = tuple(f'field_{index}' for index in range(10))
    schema = SchemaView.from_api_response(
        {
            'Asset': {
                **{
                    field_name: {
                        'data_type': {'value': 'text'},
                        'properties': {},
                    }
                    for field_name in fields
                },
                'id': {'data_type': {'value': 'number'}, 'properties': {}},
                'project': {
                    'data_type': {'value': 'entity'},
                    'properties': {'valid_types': {'value': ['Project']}},
                },
            }
        }
    )
    projects = tuple((('type', 'Project'), ('id', project_id)) for project_id in range(101, 111))
    call = _call(
        filters=(
            ('id', 'in', tuple(range(1, 11))),
            ('project', 'in', projects),
        ),
        fields=('id', *fields),
    )

    plan = _plan(
        call,
        limits=LimitSettings(max_dependency_tags=7),
        schema=schema,
    )

    assert plan.classification == 'event_aware'
    assert plan.broadening_steps == (
        'limit:max_dependency_tags',
        'broaden:entity_to_type',
        'broaden:field_to_type_value',
    )
    assert plan.tags == (
        DependencyTag.site(_SITE),
        DependencyTag.schema(_SITE),
        DependencyTag.authorization(_SITE),
        DependencyTag.type_membership(_SITE, 'Asset'),
        DependencyTag.type_value(_SITE, 'Asset'),
        DependencyTag.type_membership(_SITE, 'Project'),
    )


def test_ten_thousand_flat_ids_broaden_to_the_expected_five_tags() -> None:
    plan = _plan(_call(filters=(('id', 'in', tuple(range(1, 10_001))),)))

    assert plan.classification == 'event_aware'
    assert plan.broadening_steps == (
        'limit:max_dependency_tags',
        'broaden:entity_to_type',
    )
    assert plan.tags == (
        DependencyTag.site(_SITE),
        DependencyTag.schema(_SITE),
        DependencyTag.authorization(_SITE),
        DependencyTag.type_membership(_SITE, 'Asset'),
        DependencyTag.type_value(_SITE, 'Asset'),
    )


@pytest.mark.parametrize(
    ('entity_ids', 'expected_ids'),
    [
        ((9, 7, 9, 5, 7), (9, 7, 5)),
        ((5, 9, 7, 5, 9), (5, 9, 7)),
    ],
)
def test_top_level_ids_preserve_first_occurrence_order(
    entity_ids: tuple[int, ...],
    expected_ids: tuple[int, ...],
) -> None:
    plan = _plan(_call(filters=(('id', 'in', entity_ids),)))

    assert plan.tags == (
        DependencyTag.site(_SITE),
        DependencyTag.schema(_SITE),
        DependencyTag.authorization(_SITE),
        *(DependencyTag.entity(_SITE, 'Asset', entity_id) for entity_id in expected_ids),
    )


@pytest.mark.parametrize(
    ('operator', 'value', 'entity_ids'),
    [('is', 7, (7,)), ('in', (7, 9), (7, 9))],
)
def test_top_level_conjunctive_id_lookup_carries_every_candidate(
    operator: str,
    value: ExactValue,
    entity_ids: tuple[int, ...],
) -> None:
    plan = _plan(_call(filters=(('id', operator, value),)))

    assert DependencyTag.type_membership(_SITE, 'Asset') not in plan.tags
    assert plan.tags[-len(entity_ids) :] == tuple(
        DependencyTag.entity(_SITE, 'Asset', entity_id) for entity_id in entity_ids
    )


@pytest.mark.parametrize(
    ('operator', 'value', 'expected'),
    [
        ('is', (('type', 'Asset'), ('id', 3)), ('Asset',)),
        ('in', ((('type', 'Shot'), ('id', 3)), (('type', 'Asset'), ('id', 4))), ('Asset', 'Shot')),
        ('is', None, ('Asset', 'Shot')),
        ('is', 'not-a-reference', ('Asset', 'Shot')),
        ('is_not', (('type', 'Asset'), ('id', 3)), ('Asset', 'Shot')),
        ('not_in', ((('type', 'Asset'), ('id', 3)),), ('Asset', 'Shot')),
        ('type_is', 'Asset', ('Asset',)),
        ('type_is_not', 'Asset', ('Asset', 'Shot')),
    ],
)
def test_link_filters_carry_target_membership_for_existence(
    operator: str,
    value: ExactValue,
    expected: tuple[str, ...],
) -> None:
    plan = _plan(_call(filters=(('linked', operator, value),)))

    memberships = tuple(
        tag.components[1] for tag in plan.tags if tag.kind is TagKind.TYPE_MEMBERSHIP
    )
    assert plan.classification == 'event_aware'
    assert memberships == expected


@pytest.mark.parametrize(
    'data',
    [
        b'',
        b'not-zlib',
        zlib.compress(b'[]'),
        zlib.compress(b'{"Asset": 1}'),
        zlib.compress(b'{}') + b'trailing',
    ],
)
def test_compact_schema_view_rejects_malformed_bytes(data: bytes) -> None:
    with pytest.raises(ValueError, match='compact schema view is malformed'):
        SchemaView.from_compact(data)


def test_type_filter_covers_link_field_and_target_membership() -> None:
    plan = _plan(_call(filters=(('linked', 'type_is', 'Shot'),)))

    assert plan.tags == (
        DependencyTag.site(_SITE),
        DependencyTag.schema(_SITE),
        DependencyTag.authorization(_SITE),
        DependencyTag.type_membership(_SITE, 'Asset'),
        DependencyTag.type_field(_SITE, 'Asset', 'linked'),
        DependencyTag.type_membership(_SITE, 'Shot'),
    )


def test_unbounded_plan_bypasses_when_ttl_only_is_not_allowed() -> None:
    plan = _plan(
        _call(fields=('id', 'code')),
        analyzer=DependencyAnalyzer(allow_ttl_only=False),
        limits=LimitSettings(max_dependency_tags=2),
    )

    assert plan.classification == 'bypass'
    assert plan.reason == 'dependency_bounds_exceeded'
    assert plan.broadening_steps == (
        'limit:max_dependency_tags',
        'broaden:field_to_type_value',
        'fallback:bypass',
    )
    assert plan.tags == ()


def test_unbounded_plan_falls_back_to_bounded_ttl_only() -> None:
    plan = _plan(
        _call(fields=('id', 'code')),
        limits=LimitSettings(max_dependency_tags=2),
    )

    assert plan.classification == 'ttl_only'
    assert plan.reason == 'dependency_bounds_exceeded'
    assert plan.broadening_steps == (
        'limit:max_dependency_tags',
        'broaden:field_to_type_value',
        'fallback:ttl_only',
    )
    assert plan.tags == (
        DependencyTag.site(_SITE),
        DependencyTag.authorization(_SITE),
    )


def test_unknown_order_field_type_falls_back_to_ttl_only() -> None:
    schema = SchemaView.from_api_response(
        {
            'Asset': {
                'id': {'data_type': {'value': 'number'}, 'properties': {}},
                'future_value': {
                    'data_type': {'value': 'future_compound'},
                    'properties': {},
                },
            }
        }
    )
    order = ((('field_name', 'future_value'), ('direction', 'asc')),)

    plan = _plan(_call(order=order), schema=schema)

    assert plan.classification == 'ttl_only'
    assert plan.reason == 'unknown_field_type'
    assert plan.tags == (
        DependencyTag.site(_SITE),
        DependencyTag.authorization(_SITE),
    )


def test_unknown_returned_field_type_falls_back_to_ttl_only() -> None:
    schema = SchemaView.from_api_response(
        {
            'Asset': {
                'id': {'data_type': {'value': 'number'}, 'properties': {}},
                'future_value': {
                    'data_type': {'value': 'future_compound'},
                    'properties': {},
                },
            }
        }
    )

    plan = _plan(_call(fields=('id', 'future_value')), schema=schema)

    assert plan.classification == 'ttl_only'
    assert plan.reason == 'unknown_field_type'
    assert plan.tags == (
        DependencyTag.site(_SITE),
        DependencyTag.authorization(_SITE),
    )


def test_unknown_schema_falls_back_to_ttl_only_or_bypass() -> None:
    call = _call(filters=(('permission_hidden', 'is', 7),))

    ttl_only = _plan(call)
    bypass = _plan(call, analyzer=DependencyAnalyzer(allow_ttl_only=False))

    assert ttl_only.classification == 'ttl_only'
    assert ttl_only.reason == 'unknown_field'
    assert ttl_only.tags == (
        DependencyTag.site(_SITE),
        DependencyTag.authorization(_SITE),
    )
    assert bypass.classification == 'bypass'
    assert bypass.reason == 'unknown_field'
    assert bypass.tags == ()


def test_vector_byte_limit_broadens_without_a_tag_count_overflow() -> None:
    long_field = 'field_' + ('x' * 2200)
    schema = SchemaView.from_api_response(
        {
            'Asset': {
                'id': {'data_type': {'value': 'number'}, 'properties': {}},
                long_field: {'data_type': {'value': 'text'}, 'properties': {}},
            }
        }
    )

    plan = _plan(_call(fields=('id', long_field)), schema=schema)

    assert plan.broadening_steps == (
        'limit:max_dependency_vector_bytes',
        'broaden:field_to_type_value',
    )
    assert len(plan.tags) <= _LIMITS.max_dependency_tags
    assert KeyBuilder.worst_case_vector_bytes(plan.tags) <= (_LIMITS.max_dependency_vector_bytes)
