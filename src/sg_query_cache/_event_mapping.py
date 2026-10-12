# Same-file collaborators use private implementation members.
# pyright: reportPrivateUsage=false

"""Map closed SG event shapes to conservative generation changes."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from re import compile as compile_pattern
from typing import cast

from sg_query_cache._client_common import (
    HUMAN_VISIBILITY_ENTITY_TYPES,
    HUMAN_VISIBILITY_FIELDS,
    PERMISSION_FIELDS,
    PERMISSION_TYPES,
)
from sg_query_cache._model import DependencyTag, GenerationChange

_AFFECTED_HUMANS_KEY = 'affected_human_user_ids'
_ENTITY_TYPE_PATTERN = compile_pattern(r'[A-Za-z][A-Za-z0-9_]{0,63}')
_FIELD_NAME_PATTERN = compile_pattern(r'[A-Za-z][A-Za-z0-9_]{0,127}')
_LIFECYCLE_ACTIONS = frozenset({'New', 'Retirement', 'Revival'})
_MAX_SCOPED_TAGS = 32
_PERMISSION_RULE_EVENTS = frozenset(
    {
        'Shotgun_PermissionRuleSet_ChangeRule',
        'Shotgun_PermissionRuleSet_DeleteRule',
        'Shotgun_PermissionRuleSet_NewRule',
    }
)
_REQUIRED_FIELDS = frozenset(
    {
        'attribute_name',
        'created_at',
        'entity',
        'event_type',
        'id',
        'meta',
        'project',
        'session_uuid',
        'type',
        'user',
    }
)
_SCHEMA_ENTITY_TYPES = frozenset({'EntityType', 'Field', 'Schema', 'SchemaField'})


class _InvalidEventError(ValueError):
    """Mark an event that requires site-wide fallback."""


@dataclass(frozen=True, slots=True)
class _ParsedEvent:
    """Hold validated event fields used by the closed mapper."""

    _attribute_name: str | None
    _entity: tuple[str, int] | None
    _event_type: str
    _meta: Mapping[str, object]
    _project: tuple[str, int] | None


def _affected_ids(meta: Mapping[str, object], key: str) -> tuple[int, ...] | None:
    """Return one validated, bounded, duplicate-free affected-ID list."""
    if key not in meta:
        return ()
    value = meta[key]
    if isinstance(value, str | bytes) or not isinstance(value, Sequence):
        return None
    values = cast('Sequence[object]', value)
    identifiers: list[int] = []
    for candidate in values:
        if type(candidate) is not int or candidate <= 0:
            return None
        if candidate not in identifiers:
            identifiers.append(candidate)
        if len(identifiers) > _MAX_SCOPED_TAGS:
            return None
    return tuple(identifiers)


def _authorization_tags(
    site_namespace: str,
    entity_type: str,
    attribute_name: str | None,
) -> tuple[DependencyTag, ...]:
    """Return global authorization tags for permission-bearing events."""
    if entity_type in PERMISSION_TYPES or (
        entity_type == 'HumanUser' and attribute_name in PERMISSION_FIELDS
    ):
        return (DependencyTag.authorization(site_namespace),)
    return ()


def _data_tags(
    site_namespace: str,
    ordinary: tuple[str, str, int],
    attribute_name: str | None,
    meta: Mapping[str, object],
) -> tuple[tuple[DependencyTag, ...], ...]:
    """Return ordered membership, value, field, and entity tag groups."""
    action, entity_type, entity_id = ordinary
    if action in _LIFECYCLE_ACTIONS:
        membership = (DependencyTag.type_membership(site_namespace, entity_type),)
        type_value: tuple[DependencyTag, ...] = ()
        type_field: tuple[DependencyTag, ...] = ()
    else:
        if attribute_name is None:
            raise _InvalidEventError
        meta_attribute = meta.get('attribute_name')
        if meta_attribute is not None and meta_attribute != attribute_name:
            raise _InvalidEventError
        membership = ()
        type_value = (DependencyTag.type_value(site_namespace, entity_type),)
        type_field = (DependencyTag.type_field(site_namespace, entity_type, attribute_name),)
    entity = (DependencyTag.entity(site_namespace, entity_type, entity_id),)
    return membership, type_value, type_field, entity


def _event_mapping(value: object) -> Mapping[str, object] | None:
    """Return a string-keyed event mapping with all public fields."""
    if not isinstance(value, Mapping):
        return None
    untyped = cast('Mapping[object, object]', value)
    if not all(isinstance(key, str) for key in untyped):
        return None
    event = cast('Mapping[str, object]', untyped)
    if not _REQUIRED_FIELDS.issubset(event):
        return None
    return event


def _field_name(value: object) -> str | None:
    """Return one closed SG field name."""
    if not isinstance(value, str) or _FIELD_NAME_PATTERN.fullmatch(value) is None:
        return None
    return value


def _link(value: object) -> tuple[str, int] | None:
    """Return one validated SG entity link."""
    if not isinstance(value, Mapping):
        return None
    mapping = cast('Mapping[object, object]', value)
    entity_type = mapping.get('type')
    entity_id = mapping.get('id')
    if (
        not isinstance(entity_type, str)
        or _ENTITY_TYPE_PATTERN.fullmatch(entity_type) is None
        or type(entity_id) is not int
        or entity_id <= 0
    ):
        return None
    return entity_type, entity_id


def _mapping(value: object) -> Mapping[str, object] | None:
    """Return one string-keyed mapping."""
    if not isinstance(value, Mapping):
        return None
    untyped = cast('Mapping[object, object]', value)
    if not all(isinstance(key, str) for key in untyped):
        return None
    return cast('Mapping[str, object]', untyped)


def _match_entity_metadata(
    meta: Mapping[str, object],
    entity_type: str,
    entity_id: int,
) -> None:
    """Reject metadata that contradicts the authoritative event link."""
    if meta.get('entity_type', entity_type) != entity_type:
        raise _InvalidEventError
    if meta.get('entity_id', entity_id) != entity_id:
        raise _InvalidEventError


def _optional_field_name(value: object) -> str | None:
    """Return a valid optional field name or reject the event."""
    if value is None:
        return None
    field_name = _field_name(value)
    if field_name is None:
        raise _InvalidEventError
    return field_name


def _optional_link(value: object) -> tuple[str, int] | None:
    """Return a valid optional link or reject the event."""
    if value is None:
        return None
    link = _link(value)
    if link is None:
        raise _InvalidEventError
    return link


def _ordered_changes(
    groups: Sequence[Sequence[DependencyTag]],
) -> tuple[GenerationChange, ...]:
    """Deduplicate tags while preserving authoritative group order."""
    tags: list[DependencyTag] = []
    for group in groups:
        for tag in group:
            if tag not in tags:
                tags.append(tag)
    if not tags:
        raise _InvalidEventError
    return tuple(GenerationChange(tag) for tag in tags)


def _ordinary_event(
    event_type: str,
    entity: tuple[str, int] | None,
) -> tuple[str, str, int] | None:
    """Extract one standard SG entity event."""
    if not event_type.startswith('Shotgun_'):
        return None
    remainder = event_type.removeprefix('Shotgun_')
    try:
        event_entity_type, action = remainder.rsplit('_', 1)
    except ValueError:
        return None
    if (
        _ENTITY_TYPE_PATTERN.fullmatch(event_entity_type) is None
        or action not in _LIFECYCLE_ACTIONS | {'Change'}
        or event_entity_type in _SCHEMA_ENTITY_TYPES
        or entity is None
        or entity[0] != event_entity_type
    ):
        return None
    return action, event_entity_type, entity[1]


def _positive_id(value: object) -> int:
    """Return a positive integer or reject the event."""
    if type(value) is not int or value <= 0:
        raise _InvalidEventError
    return value


def _required_mapping(value: object) -> Mapping[str, object]:
    """Return a string-keyed mapping or reject the event."""
    mapping = _mapping(value)
    if mapping is None:
        raise _InvalidEventError
    return mapping


def _required_text(value: object) -> str:
    """Return a nonempty string or reject the event."""
    if not isinstance(value, str) or not value:
        raise _InvalidEventError
    return value


def _schema_tags(
    site_namespace: str,
    event_type: str,
    attribute_name: str | None,
    meta: Mapping[str, object],
) -> tuple[DependencyTag, ...] | None:
    """Return tags for one recognized schema or field-definition event."""
    if not event_type.startswith('Shotgun_'):
        return None
    remainder = event_type.removeprefix('Shotgun_')
    try:
        schema_type, action = remainder.rsplit('_', 1)
    except ValueError:
        return None
    if schema_type not in _SCHEMA_ENTITY_TYPES or action not in _LIFECYCLE_ACTIONS | {'Change'}:
        return None
    affected_type = meta.get('entity_type')
    if not isinstance(affected_type, str) or _ENTITY_TYPE_PATTERN.fullmatch(affected_type) is None:
        raise _InvalidEventError
    tags = [
        DependencyTag.schema(site_namespace),
        DependencyTag.type_membership(site_namespace, affected_type),
        DependencyTag.type_value(site_namespace, affected_type),
    ]
    if schema_type in {'Field', 'SchemaField'}:
        if attribute_name is None:
            raise _InvalidEventError
        if 'field_name' in meta:
            metadata_field = _field_name(meta['field_name'])
            if metadata_field is None or metadata_field != attribute_name:
                raise _InvalidEventError
        tags.append(DependencyTag.type_field(site_namespace, affected_type, attribute_name))
    return tuple(tags)


def _site_change(site_namespace: str) -> tuple[GenerationChange, ...]:
    """Return the conservative site-wide fallback."""
    return (GenerationChange(DependencyTag.site(site_namespace)),)


def _validated_event(value: object) -> _ParsedEvent:
    """Validate the complete public daemon event boundary."""
    event = _event_mapping(value)
    if event is None:
        raise _InvalidEventError
    _positive_id(event['id'])
    if event['type'] != 'EventLogEntry':
        raise _InvalidEventError
    event_type = _required_text(event['event_type'])
    attribute_name = _optional_field_name(event['attribute_name'])
    meta = _required_mapping(event['meta'])
    entity = _optional_link(event['entity'])
    _optional_link(event['user'])
    project = _optional_link(event['project'])
    if project is not None and project[0] != 'Project':
        raise _InvalidEventError
    session_uuid = event['session_uuid']
    if session_uuid is not None:
        _required_text(session_uuid)
    created_at = event['created_at']
    if isinstance(created_at, str):
        _required_text(created_at)
    elif not isinstance(created_at, date | datetime):
        raise _InvalidEventError
    return _ParsedEvent(attribute_name, entity, event_type, meta, project)


def _visibility_tags(
    site_namespace: str,
    entity_type: str,
    entity_id: int,
    attribute_name: str | None,
    meta: Mapping[str, object],
) -> tuple[DependencyTag, ...]:
    """Return exact or global human-visibility tags."""
    affected_humans = _affected_ids(meta, _AFFECTED_HUMANS_KEY)
    if affected_humans is None:
        return (DependencyTag.human_visibility(site_namespace),)
    uncertain = (
        entity_type in HUMAN_VISIBILITY_ENTITY_TYPES or attribute_name in HUMAN_VISIBILITY_FIELDS
    )
    identifiers = list(affected_humans)
    if entity_type == 'HumanUser' and entity_id not in identifiers:
        identifiers.append(entity_id)
    if len(identifiers) > _MAX_SCOPED_TAGS or (not identifiers and uncertain):
        return (DependencyTag.human_visibility(site_namespace),)
    return tuple(
        DependencyTag.identity_visibility(site_namespace, human_id) for human_id in identifiers
    )


def _map_event(site_namespace: str, event: object) -> tuple[GenerationChange, ...]:
    """Map one validated event or reject uncertain coverage."""
    parsed = _validated_event(event)
    schema = _schema_tags(
        site_namespace,
        parsed._event_type,
        parsed._attribute_name,
        parsed._meta,
    )
    if schema is not None:
        return _ordered_changes((schema,))
    if parsed._event_type in _PERMISSION_RULE_EVENTS:
        return _ordered_changes(((DependencyTag.authorization(site_namespace),),))
    ordinary = _ordinary_event(parsed._event_type, parsed._entity)
    if ordinary is None:
        raise _InvalidEventError
    _, entity_type, entity_id = ordinary
    _match_entity_metadata(parsed._meta, entity_type, entity_id)
    membership, type_value, type_field, entity = _data_tags(
        site_namespace,
        ordinary,
        parsed._attribute_name,
        parsed._meta,
    )
    return _ordered_changes(
        (
            _authorization_tags(site_namespace, entity_type, parsed._attribute_name),
            _visibility_tags(
                site_namespace,
                entity_type,
                entity_id,
                parsed._attribute_name,
                parsed._meta,
            ),
            membership,
            type_value,
            type_field,
            entity,
        )
    )


def changes_for_event(
    site_namespace: str,
    event: Mapping[str, object],
) -> tuple[GenerationChange, ...]:
    """Map one immutable SG EventLogEntry to ordered generation changes.

    Args:
        site_namespace: Stable namespace shared by clients for one SG site.
        event: Public daemon EventLogEntry mapping.

    Returns:
        Ordered unique changes, falling back to the site generation whenever
        the event cannot be classified completely.
    """
    try:
        return _map_event(site_namespace, event)
    except _InvalidEventError:
        return _site_change(site_namespace)
