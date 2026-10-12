from collections.abc import Callable
from dataclasses import FrozenInstanceError

import pytest

from sg_query_cache._model import (
    CacheHit,
    DecodedPayload,
    DependencyPlan,
    DependencyTag,
    ExactCall,
    FillLease,
    GenerationChange,
    GenerationVector,
    Identity,
    TagKind,
)


def test_cache_hit_carrier_is_optional_and_not_value_semantics() -> None:
    absent = CacheHit(b'payload', 1)
    present = CacheHit(b'payload', 1, decoded=DecodedPayload(None))

    assert absent.decoded is None
    assert present.decoded is not None
    assert present.decoded.value is None
    assert absent == present
    assert repr(absent) == repr(present)


@pytest.mark.parametrize('remaining_seconds', [True, float('nan'), float('inf')])
def test_cache_hit_rejects_nonfinite_or_boolean_remaining_lifetime(
    remaining_seconds: object,
) -> None:
    with pytest.raises((TypeError, ValueError), match='remaining'):
        CacheHit(b'payload', remaining_seconds)  # type: ignore[arg-type]


def test_dependency_tag_constructors_include_site_namespace_first() -> None:
    tags = (
        DependencyTag.site('production'),
        DependencyTag.schema('production'),
        DependencyTag.authorization('production'),
        DependencyTag.human_visibility('production'),
        DependencyTag.identity_visibility('production', 42),
        DependencyTag.type_membership('production', 'Asset'),
        DependencyTag.type_value('production', 'Asset'),
        DependencyTag.type_field('production', 'Asset', 'code'),
        DependencyTag.project('production', 17),
        DependencyTag.entity('production', 'Asset', 7),
    )

    assert [tag.kind for tag in tags] == [
        TagKind.SITE,
        TagKind.SCHEMA,
        TagKind.AUTHORIZATION,
        TagKind.HUMAN_VISIBILITY,
        TagKind.IDENTITY_VISIBILITY,
        TagKind.TYPE_MEMBERSHIP,
        TagKind.TYPE_VALUE,
        TagKind.TYPE_FIELD,
        TagKind.PROJECT,
        TagKind.ENTITY,
    ]
    assert all(tag.components[0] == 'production' for tag in tags)
    assert TagKind.LINK_PATH.value == 'link_path'


@pytest.mark.parametrize(
    ('arguments', 'match'),
    [
        ({'kind': 'robot', 'issuer': 'issuer', 'namespace': 'namespace'}, 'kind'),
        ({'kind': 'script', 'issuer': '', 'namespace': 'namespace'}, 'issuer'),
        ({'kind': 'script', 'issuer': 'issuer', 'namespace': ''}, 'namespace'),
        ({'kind': 'human', 'issuer': 'issuer', 'namespace': 'namespace'}, 'entity ID'),
        (
            {
                'kind': 'human',
                'issuer': 'issuer',
                'namespace': 'namespace',
                'entity_id': 0,
            },
            'entity ID',
        ),
        (
            {
                'kind': 'script',
                'issuer': 'issuer',
                'namespace': 'namespace',
                'entity_id': -1,
            },
            'entity ID',
        ),
    ],
)
def test_identity_rejects_unstable_or_invalid_namespaces(
    arguments: dict[str, object], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        Identity(**arguments)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    'build',
    [
        lambda: DependencyTag.site(''),
        lambda: DependencyTag.type_field('production', 'Asset', ''),
        lambda: DependencyTag.entity('production', 'Asset', 0),
    ],
)
def test_dependency_tags_reject_invalid_components(build: Callable[[], object]) -> None:
    with pytest.raises(ValueError, match='dependency tag'):
        build()


@pytest.mark.parametrize(
    ('kind', 'components'),
    [
        (TagKind.SITE, (1,)),
        (TagKind.SCHEMA, ('production', 'extra')),
        (TagKind.AUTHORIZATION, ('production', 'extra')),
        (TagKind.HUMAN_VISIBILITY, ('production', 'extra')),
        (TagKind.IDENTITY_VISIBILITY, ('production', '42')),
        (TagKind.TYPE_MEMBERSHIP, ('production', 'Asset', 'extra')),
        (TagKind.TYPE_VALUE, ('production', 'Asset', 'extra')),
        (TagKind.TYPE_FIELD, ('production', 'Asset')),
        (TagKind.LINK_PATH, ('production', 'Asset')),
        (TagKind.PROJECT, ('production', '17')),
        (TagKind.ENTITY, ('production', 'Asset', '7')),
    ],
)
def test_dependency_tag_raw_construction_rejects_kind_specific_shape(
    kind: TagKind, components: object
) -> None:
    with pytest.raises((TypeError, ValueError), match='dependency tag'):
        DependencyTag(kind, components)  # type: ignore[arg-type]


@pytest.mark.parametrize('site_namespace', [' production ', 'prod-☃'])
def test_dependency_tag_rejects_noncanonical_site_namespace(site_namespace: str) -> None:
    with pytest.raises(ValueError, match='site namespace'):
        DependencyTag.site(site_namespace)


def test_identity_requires_a_stable_positive_sg_identity() -> None:
    identity = Identity(
        kind='human',
        issuer='studio-sso',
        namespace='human:42',
        entity_id=42,
    )

    assert identity.kind == 'human'
    assert identity.entity_id == 42


def test_model_values_are_frozen_and_slotted() -> None:
    tag = DependencyTag.entity('production', 'Asset', 7)
    values = (
        ExactCall('find', (('entity_type', 'Asset'),)),
        tag,
        DependencyPlan('event_aware', (tag,), 'supported'),
        GenerationVector('0123456789abcdef0123456789abcdef', ((tag, 1),)),
        CacheHit(b'payload', 1.5),
        FillLease('qualified-key', '0123456789abcdef0123456789abcdef'),
        GenerationChange(tag),
    )

    for value in values:
        assert not hasattr(value, '__dict__')

    with pytest.raises(FrozenInstanceError):
        values[0].method = 'find_one'  # type: ignore[misc]


@pytest.mark.parametrize(
    'build',
    [
        lambda: DependencyTag(TagKind.SITE, ['production']),  # type: ignore[arg-type]
        lambda: ExactCall(
            'find',
            (('filters', [{'field': 'code'}]),),  # type: ignore[arg-type]
        ),
        lambda: ExactCall('find', [['entity_type', 'Asset']]),  # type: ignore[arg-type]
        lambda: DependencyPlan(
            'event_aware',
            [DependencyTag.site('production')],  # type: ignore[arg-type]
            'supported',
        ),
        lambda: DependencyPlan(
            'event_aware',
            (DependencyTag.site('production'),),
            'supported',
            broadening_steps=['field_to_type'],  # type: ignore[arg-type]
        ),
        lambda: GenerationVector(
            '0123456789abcdef0123456789abcdef',
            ([DependencyTag.site('production'), 1],),  # type: ignore[arg-type]
        ),
    ],
)
def test_model_values_reject_mutable_nested_containers(build: Callable[[], object]) -> None:
    with pytest.raises(TypeError, match=r'tuple|immutable'):
        build()


def test_ordered_model_collections_reject_duplicate_tags() -> None:
    tag = DependencyTag.site('production')

    with pytest.raises(ValueError, match='duplicate'):
        DependencyPlan('event_aware', (tag, tag), 'supported')
    with pytest.raises(ValueError, match='duplicate'):
        GenerationVector('0123456789abcdef0123456789abcdef', ((tag, 1), (tag, 2)))


def test_script_identity_uses_stable_scope_without_entity_id() -> None:
    identity = Identity(kind='script', issuer='service-config', namespace='render-farm')

    assert identity.kind == 'script'
    assert identity.issuer == 'service-config'
    assert identity.namespace == 'render-farm'
    assert identity.entity_id is None
