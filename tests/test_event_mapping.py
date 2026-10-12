import json
from collections.abc import Mapping
from pathlib import Path
from typing import cast

import pytest

from sg_query_cache._event_mapping import changes_for_event
from sg_query_cache._model import DependencyTag, GenerationChange, TagKind

_FIXTURE_PATH = Path(__file__).parent / 'fixtures' / 'events.json'
_SITE = 'production'


@pytest.fixture(scope='module')
def events() -> Mapping[str, Mapping[str, object]]:
    document = cast('object', json.loads(_FIXTURE_PATH.read_text(encoding='utf-8')))
    assert isinstance(document, dict)
    return cast('Mapping[str, Mapping[str, object]]', document)


def _change_event(
    template: Mapping[str, object],
    entity_type: str,
    attribute_name: str,
    *,
    entity_id: int = 41,
) -> dict[str, object]:
    event = dict(template)
    event['attribute_name'] = attribute_name
    event['entity'] = {'id': entity_id, 'type': entity_type}
    event['event_type'] = f'Shotgun_{entity_type}_Change'
    event['meta'] = {
        'attribute_name': attribute_name,
        'entity_id': entity_id,
        'entity_type': entity_type,
        'type': 'attribute_change',
    }
    event['project'] = None
    return event


def test_api_user_lifecycle_changes_global_authorization_before_data_tags(
    events: Mapping[str, Mapping[str, object]],
) -> None:
    assert changes_for_event(_SITE, events['api_user_retirement']) == (
        GenerationChange(DependencyTag.authorization(_SITE)),
        GenerationChange(DependencyTag.type_membership(_SITE, 'ApiUser')),
        GenerationChange(DependencyTag.entity(_SITE, 'ApiUser', 13)),
    )


def test_bounded_affected_humans_change_each_exact_identity(
    events: Mapping[str, Mapping[str, object]],
) -> None:
    changes = changes_for_event(_SITE, events['affected_users'])

    assert changes[:2] == (
        GenerationChange(DependencyTag.identity_visibility(_SITE, 17)),
        GenerationChange(DependencyTag.identity_visibility(_SITE, 23)),
    )
    assert GenerationChange(DependencyTag.human_visibility(_SITE)) not in changes


def test_combined_human_identity_bound_allows_exactly_32_unique_ids(
    events: Mapping[str, Mapping[str, object]],
) -> None:
    event = _change_event(events['human_user_change'], 'HumanUser', 'name', entity_id=99)
    event['meta'] = {
        **cast('Mapping[str, object]', event['meta']),
        'affected_human_user_ids': list(range(1, 32)),
    }

    changes = changes_for_event(_SITE, event)

    assert tuple(change.tag for change in changes[:32]) == (
        *(DependencyTag.identity_visibility(_SITE, human_id) for human_id in range(1, 32)),
        DependencyTag.identity_visibility(_SITE, 99),
    )
    assert GenerationChange(DependencyTag.human_visibility(_SITE)) not in changes


def test_combined_human_identity_bound_deduplicates_the_event_entity(
    events: Mapping[str, Mapping[str, object]],
) -> None:
    event = _change_event(events['human_user_change'], 'HumanUser', 'name', entity_id=99)
    event['meta'] = {
        **cast('Mapping[str, object]', event['meta']),
        'affected_human_user_ids': [*range(1, 32), 99],
    }

    changes = changes_for_event(_SITE, event)

    identity_tags = tuple(
        change.tag for change in changes if change.tag.kind.value == 'identity_visibility'
    )
    assert len(identity_tags) == 32
    assert identity_tags[-1] == DependencyTag.identity_visibility(_SITE, 99)
    assert GenerationChange(DependencyTag.human_visibility(_SITE)) not in changes


def test_combined_human_identity_overflow_uses_only_global_visibility(
    events: Mapping[str, Mapping[str, object]],
) -> None:
    event = _change_event(events['human_user_change'], 'HumanUser', 'name', entity_id=99)
    event['meta'] = {
        **cast('Mapping[str, object]', event['meta']),
        'affected_human_user_ids': list(range(1, 33)),
    }

    changes = changes_for_event(_SITE, event)

    assert changes[0] == GenerationChange(DependencyTag.human_visibility(_SITE))
    assert not any(change.tag.kind.value == 'identity_visibility' for change in changes)


def test_field_definition_change_invalidates_schema_and_affected_type_state(
    events: Mapping[str, Mapping[str, object]],
) -> None:
    assert changes_for_event(_SITE, events['schema_field_change']) == (
        GenerationChange(DependencyTag.schema(_SITE)),
        GenerationChange(DependencyTag.type_membership(_SITE, 'Asset')),
        GenerationChange(DependencyTag.type_value(_SITE, 'Asset')),
        GenerationChange(DependencyTag.type_field(_SITE, 'Asset', 'code')),
    )


@pytest.mark.parametrize('metadata_field', ['description', 'bad field', None, 7])
def test_field_definition_rejects_contradictory_or_malformed_metadata_field(
    events: Mapping[str, Mapping[str, object]], metadata_field: object
) -> None:
    event = dict(events['schema_field_change'])
    event['meta'] = {
        **cast('Mapping[str, object]', event['meta']),
        'field_name': metadata_field,
    }

    assert changes_for_event(_SITE, event) == (GenerationChange(DependencyTag.site(_SITE)),)


def test_field_definition_uses_top_level_attribute_when_metadata_omits_field(
    events: Mapping[str, Mapping[str, object]],
) -> None:
    event = dict(events['schema_field_change'])
    meta = dict(cast('Mapping[str, object]', event['meta']))
    meta.pop('field_name')
    event['meta'] = meta

    changes = changes_for_event(_SITE, event)

    assert changes[-1] == GenerationChange(DependencyTag.type_field(_SITE, 'Asset', 'code'))


def test_human_permission_change_uses_global_authorization(
    events: Mapping[str, Mapping[str, object]],
) -> None:
    event = dict(events['human_user_change'])
    event['attribute_name'] = 'permission_rule_set'
    event['meta'] = {
        'attribute_name': 'permission_rule_set',
        'entity_id': 17,
        'entity_type': 'HumanUser',
        'type': 'attribute_change',
    }

    changes = changes_for_event(_SITE, event)

    assert changes[0] == GenerationChange(DependencyTag.authorization(_SITE))
    assert GenerationChange(DependencyTag.identity_visibility(_SITE, 17)) in changes


def test_human_user_change_targets_exact_identity(
    events: Mapping[str, Mapping[str, object]],
) -> None:
    changes = changes_for_event(_SITE, events['human_user_change'])

    assert changes[0] == GenerationChange(DependencyTag.identity_visibility(_SITE, 17))
    assert GenerationChange(DependencyTag.human_visibility(_SITE)) not in changes


def test_known_field_change_uses_ordered_value_field_and_entity_tags(
    events: Mapping[str, Mapping[str, object]],
) -> None:
    assert changes_for_event(_SITE, events['asset_change']) == (
        GenerationChange(DependencyTag.type_value(_SITE, 'Asset')),
        GenerationChange(DependencyTag.type_field(_SITE, 'Asset', 'code')),
        GenerationChange(DependencyTag.entity(_SITE, 'Asset', 7)),
    )


@pytest.mark.parametrize('action', ['New', 'Retirement', 'Revival'])
def test_lifecycle_changes_membership_and_entity(
    events: Mapping[str, Mapping[str, object]], action: str
) -> None:
    event = dict(events['asset_new'])
    event['event_type'] = f'Shotgun_Asset_{action}'

    assert changes_for_event(_SITE, event) == (
        GenerationChange(DependencyTag.type_membership(_SITE, 'Asset')),
        GenerationChange(DependencyTag.entity(_SITE, 'Asset', 7)),
    )


def test_nonfield_schema_event_changes_schema_and_affected_type_state(
    events: Mapping[str, Mapping[str, object]],
) -> None:
    event = dict(events['schema_field_change'])
    event['attribute_name'] = None
    event['event_type'] = 'Shotgun_EntityType_Change'
    event['meta'] = {'entity_type': 'Asset', 'type': 'schema_change'}

    assert changes_for_event(_SITE, event) == (
        GenerationChange(DependencyTag.schema(_SITE)),
        GenerationChange(DependencyTag.type_membership(_SITE, 'Asset')),
        GenerationChange(DependencyTag.type_value(_SITE, 'Asset')),
    )


def test_ordinary_events_change_no_project_tag(
    events: Mapping[str, Mapping[str, object]],
) -> None:
    event = _change_event(next(iter(events.values())), 'Version', 'sg_status_list')
    event['project'] = {'type': 'Project', 'id': 5}

    changes = changes_for_event(_SITE, event)

    assert all(change.tag.kind is not TagKind.PROJECT for change in changes)
    assert (
        GenerationChange(DependencyTag.type_field(_SITE, 'Version', 'sg_status_list')) in changes
    )


@pytest.mark.parametrize(
    'event_type',
    [
        'Shotgun_PermissionRuleSet_ChangeRule',
        'Shotgun_PermissionRuleSet_DeleteRule',
        'Shotgun_PermissionRuleSet_NewRule',
        'Shotgun_PermissionRole_Change',
    ],
)
def test_permission_events_change_global_authorization(
    events: Mapping[str, Mapping[str, object]], event_type: str
) -> None:
    event = dict(events['permission_rule'])
    event['event_type'] = event_type
    if event_type == 'Shotgun_PermissionRole_Change':
        event['entity'] = {'id': 3, 'type': 'PermissionRole'}
        event['attribute_name'] = 'name'
        event['meta'] = {
            'attribute_name': 'name',
            'entity_id': 3,
            'entity_type': 'PermissionRole',
            'type': 'attribute_change',
        }

    changes = changes_for_event(_SITE, event)

    assert changes[0] == GenerationChange(DependencyTag.authorization(_SITE))


@pytest.mark.parametrize(
    'event',
    [
        {},
        {'event_type': 'Toolkit_Asset_Change'},
        {'event_type': 'Shotgun__Change'},
    ],
)
def test_malformed_event_changes_site(event: Mapping[str, object]) -> None:
    assert changes_for_event(_SITE, event) == (GenerationChange(DependencyTag.site(_SITE)),)


@pytest.mark.parametrize(
    ('field', 'value'),
    [
        ('id', True),
        ('type', 'Asset'),
        ('attribute_name', 'bad field'),
        ('meta', []),
        ('entity', {'id': 0, 'type': 'Asset'}),
        ('user', {'id': 0, 'type': 'HumanUser'}),
        ('project', {'id': 5, 'type': 'Asset'}),
        ('session_uuid', ''),
        ('created_at', ''),
    ],
)
def test_malformed_public_event_field_changes_site(
    events: Mapping[str, Mapping[str, object]], field: str, value: object
) -> None:
    event = dict(events['asset_change'])
    event[field] = value

    assert changes_for_event(_SITE, event) == (GenerationChange(DependencyTag.site(_SITE)),)


@pytest.mark.parametrize(
    ('field', 'value'),
    [('entity_id', 8), ('entity_type', 'Shot'), ('attribute_name', 'description')],
)
def test_contradictory_event_metadata_changes_site(
    events: Mapping[str, Mapping[str, object]], field: str, value: object
) -> None:
    event = dict(events['asset_change'])
    event['meta'] = {**cast('Mapping[str, object]', event['meta']), field: value}

    assert changes_for_event(_SITE, event) == (GenerationChange(DependencyTag.site(_SITE)),)


def test_unbounded_affected_humans_change_global_visibility(
    events: Mapping[str, Mapping[str, object]],
) -> None:
    event = dict(events['affected_users'])
    event['meta'] = {'affected_human_user_ids': list(range(1, 34))}

    changes = changes_for_event(_SITE, event)

    assert changes[0] == GenerationChange(DependencyTag.human_visibility(_SITE))
    assert not any(change.tag.kind.value == 'identity_visibility' for change in changes)


@pytest.mark.parametrize(
    ('category', 'entity_type', 'attribute_name'),
    [
        ('group', 'Group', 'users'),
        ('project_membership', 'ProjectUserConnection', 'project'),
        ('assignment', 'Task', 'task_assignees'),
        ('ownership', 'Asset', 'owner'),
        ('recipient', 'Note', 'addressings_to'),
        ('uncertain_user_state', 'HumanUser', 'status'),
    ],
)
def test_uncertain_visibility_categories_change_global_human_visibility(
    events: Mapping[str, Mapping[str, object]],
    category: str,
    entity_type: str,
    attribute_name: str,
) -> None:
    event = _change_event(events['asset_change'], entity_type, attribute_name)
    if category == 'uncertain_user_state':
        event['meta'] = {
            **cast('Mapping[str, object]', event['meta']),
            'affected_human_user_ids': 'unbounded',
        }

    changes = changes_for_event(_SITE, event)

    assert changes[0] == GenerationChange(DependencyTag.human_visibility(_SITE))


def test_uncertain_visibility_event_changes_global_human_visibility(
    events: Mapping[str, Mapping[str, object]],
) -> None:
    changes = changes_for_event(_SITE, events['group_change'])

    assert changes[0] == GenerationChange(DependencyTag.human_visibility(_SITE))


def test_unknown_event_type_changes_site(
    events: Mapping[str, Mapping[str, object]],
) -> None:
    event = dict(events['asset_change'])
    event['event_type'] = 'Shotgun_Asset_Unknown'

    assert changes_for_event(_SITE, event) == (GenerationChange(DependencyTag.site(_SITE)),)
