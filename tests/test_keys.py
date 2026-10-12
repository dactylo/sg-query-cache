import hmac
from collections.abc import Mapping
from datetime import date, datetime
from hashlib import sha256
from typing import TypeAlias, cast

import pytest

from sg_query_cache._codec import encoded_size
from sg_query_cache._keys import KeyBuilder, NormalizationError, _KeyPurpose
from sg_query_cache._model import (
    DependencyTag,
    ExactCall,
    GenerationVector,
    Identity,
    IdentityKind,
)

_ExactArgument: TypeAlias = (
    bool | int | float | str | bytes | date | datetime | tuple['_ExactArgument', ...] | None
)

_CLIENT_OPTIONS = {'localized': False, 'convert_datetimes_to_utc': True}
_INCARNATION = '0123456789abcdef0123456789abcdef'
_MAX_GENERATION = 2**63 - 1


def _call(**overrides: _ExactArgument) -> ExactCall:
    arguments: dict[str, _ExactArgument] = {
        'entity_type': 'Asset',
        'filters': (
            (
                ('filter_operator', 'all'),
                ('filters', (('code', 'is', 'private-filter'),)),
            ),
        ),
        'fields': ('id', 'code'),
        'order': (
            (('field_name', 'code'), ('direction', 'asc')),
            (('field_name', 'id'), ('direction', 'desc')),
        ),
        'filter_operator': 'all',
        'limit': 50,
        'retired_only': False,
        'page': 1,
        'include_archived_projects': True,
        'additional_filter_presets': None,
    }
    arguments.update(overrides)
    return ExactCall('find', tuple(arguments.items()))


def _identity(
    *,
    kind: IdentityKind = 'script',
    issuer: str = 'studio-sso',
    namespace: str = 'render-farm',
    entity_id: int | None = 7,
    epoch: str | int | None = 'permissions-v2',
) -> Identity:
    return Identity(
        kind=kind,
        issuer=issuer,
        namespace=namespace,
        entity_id=entity_id,
        epoch=epoch,
    )


def _key(
    call: ExactCall | None = None,
    *,
    site: str = 'https://example.invalid',
    identity: Identity | None = None,
    client_options: Mapping[str, object] = _CLIENT_OPTIONS,
) -> str:
    return KeyBuilder.base_key(site, identity or _identity(), client_options, call or _call())


def test_arbitrary_mapping_shapes_are_rejected_instead_of_reordered() -> None:
    arbitrary_mapping = (('left', 1), ('right', 2))

    with pytest.raises(NormalizationError, match='mapping shape'):
        _key(_call(filters=(arbitrary_mapping,)))


def test_backend_keys_are_versioned_opaque_and_purpose_separated() -> None:
    tag = DependencyTag.type_field('production', 'Asset', 'private_field')
    vector = GenerationVector(_INCARNATION, ((tag, 7),))
    base_key = _key()
    tag_key = KeyBuilder.tag_key(tag)
    result_key = KeyBuilder.qualified_key(base_key, vector, 'result')
    lease_key = KeyBuilder.qualified_key(base_key, vector, 'lease')

    assert base_key.startswith('sgqc:v1:base:')
    assert tag_key.startswith('sgqc:v1:tag:')
    assert result_key.startswith(f'sgqc:v1:result:{_INCARNATION}:')
    assert lease_key.startswith(f'sgqc:v1:lease:{_INCARNATION}:')
    assert result_key != lease_key
    for key in (base_key, tag_key, result_key, lease_key):
        assert 'private-filter' not in key
        assert 'Asset' not in key
        assert 'private_field' not in key
        assert 'render-farm' not in key


def test_documented_filter_group_and_order_item_key_order_is_insignificant() -> None:
    filters_left = (
        (
            ('filter_operator', 'all'),
            ('filters', (('code', 'is', 'private-filter'),)),
        ),
    )
    filters_right = (
        (
            ('filters', (('code', 'is', 'private-filter'),)),
            ('filter_operator', 'all'),
        ),
    )
    order_left = ((('field_name', 'code'), ('direction', 'asc')),)
    order_right = ((('direction', 'asc'), ('field_name', 'code')),)

    assert _key(_call(filters=filters_left)) == _key(_call(filters=filters_right))
    assert _key(_call(order=order_left)) == _key(_call(order=order_right))


@pytest.mark.parametrize(
    ('left', 'right'),
    [
        (('id', 'code'), ('code', 'id')),
        (
            (
                (('field_name', 'code'), ('direction', 'asc')),
                (('field_name', 'id'), ('direction', 'desc')),
            ),
            (
                (('field_name', 'id'), ('direction', 'desc')),
                (('field_name', 'code'), ('direction', 'asc')),
            ),
        ),
    ],
)
def test_field_and_order_list_order_remains_significant(
    left: tuple['_ExactArgument', ...], right: tuple['_ExactArgument', ...]
) -> None:
    argument = 'fields' if left[0] == 'id' else 'order'

    assert _key(_call(**{argument: left})) != _key(_call(**{argument: right}))


@pytest.mark.parametrize(
    ('name', 'changed'),
    [
        ('entity_type', 'Shot'),
        ('filters', (('id', 'is', 8),)),
        ('fields', ('id', 'description')),
        ('order', ((('field_name', 'id'), ('direction', 'asc')),)),
        ('filter_operator', 'any'),
        ('limit', 51),
        ('retired_only', True),
        ('page', 2),
        ('include_archived_projects', False),
        ('additional_filter_presets', ('preset',)),
    ],
)
def test_every_method_argument_changes_the_base_key(name: str, changed: _ExactArgument) -> None:
    assert _key() != _key(_call(**{name: changed}))


@pytest.mark.parametrize(
    ('changed_identity', 'site', 'client_options'),
    [
        (None, 'https://other.invalid', _CLIENT_OPTIONS),
        (_identity(issuer='other-sso'), 'https://example.invalid', _CLIENT_OPTIONS),
        (_identity(namespace='other-scope'), 'https://example.invalid', _CLIENT_OPTIONS),
        (_identity(entity_id=8), 'https://example.invalid', _CLIENT_OPTIONS),
        (
            Identity(kind='human', issuer='studio-sso', namespace='render-farm', entity_id=7),
            'https://example.invalid',
            _CLIENT_OPTIONS,
        ),
        (_identity(epoch='permissions-v3'), 'https://example.invalid', _CLIENT_OPTIONS),
        (
            None,
            'https://example.invalid',
            {'localized': True, 'convert_datetimes_to_utc': True},
        ),
        (
            None,
            'https://example.invalid',
            {'localized': False, 'convert_datetimes_to_utc': False},
        ),
    ],
)
def test_every_site_identity_and_client_context_dimension_changes_the_base_key(
    changed_identity: Identity | None,
    site: str,
    client_options: Mapping[str, object],
) -> None:
    assert _key() != _key(site=site, identity=changed_identity, client_options=client_options)


@pytest.mark.parametrize('credential_name', ['session_token', 'api_key'])
def test_credentials_cannot_be_passed_in_the_key_context(credential_name: str) -> None:
    options = {**_CLIENT_OPTIONS, credential_name: 'top-secret'}
    call = ExactCall('find', (*_call().arguments, (credential_name, 'top-secret')))

    with pytest.raises(NormalizationError, match='credential'):
        _key(client_options=options)
    with pytest.raises(NormalizationError, match='credential'):
        _key(call)


def test_find_and_find_one_use_distinct_base_keys() -> None:
    find = _call()
    find_one = ExactCall('find_one', find.arguments)

    assert _key(find) != _key(find_one)


def test_find_default_and_empty_fields_normalize_to_id() -> None:
    assert _key(_call(fields=None)) == _key(_call(fields=())) == _key(_call(fields=('id',)))


def test_metrics_digest_is_stable_keyed_and_backend_independent() -> None:
    base_key = _key()
    first_key = b'a' * 32
    second_key = b'b' * 32
    before = KeyBuilder.qualified_key(
        base_key,
        GenerationVector(_INCARNATION, ((DependencyTag.site('production'), 1),)),
        'result',
    )

    first = KeyBuilder.metrics_digest(base_key, first_key)
    repeated = KeyBuilder.metrics_digest(base_key, first_key)
    second = KeyBuilder.metrics_digest(base_key, second_key)
    after = KeyBuilder.qualified_key(
        base_key,
        GenerationVector(_INCARNATION, ((DependencyTag.site('production'), 1),)),
        'result',
    )

    assert first == hmac.new(first_key, base_key.encode(), sha256).hexdigest()
    assert repeated == first
    assert second != first
    assert before == after


def test_metrics_digest_rejects_short_keys() -> None:
    with pytest.raises(NormalizationError, match='at least 32 bytes'):
        KeyBuilder.metrics_digest(_key(), b'short')


def test_nested_filter_list_order_remains_significant() -> None:
    first = (
        (
            ('filter_operator', 'all'),
            ('filters', (('code', 'is', 'a'), ('id', 'is', 7))),
        ),
    )
    reversed_filters = (
        (
            ('filter_operator', 'all'),
            ('filters', (('id', 'is', 7), ('code', 'is', 'a'))),
        ),
    )

    assert _key(_call(filters=first)) != _key(_call(filters=reversed_filters))


def test_qualified_keys_validate_purpose_incarnation_and_generation_bounds() -> None:
    tag = DependencyTag.site('production')
    base_key = _key()

    with pytest.raises(NormalizationError, match='purpose'):
        KeyBuilder.qualified_key(
            base_key,
            GenerationVector(_INCARNATION, ((tag, 1),)),
            cast('_KeyPurpose', 'metadata'),
        )
    with pytest.raises(NormalizationError, match='incarnation'):
        KeyBuilder.qualified_key(base_key, GenerationVector('NOT-A-NONCE', ((tag, 1),)), 'result')
    with pytest.raises(NormalizationError, match='generation'):
        KeyBuilder.qualified_key(
            base_key,
            GenerationVector(_INCARNATION, ((tag, _MAX_GENERATION + 1),)),
            'result',
        )


def test_records_per_page_is_an_optional_positive_integer_option() -> None:
    identity = Identity('script', 'studio', 'render')
    without = KeyBuilder.base_key('production', identity, _CLIENT_OPTIONS, _call())
    with_page = KeyBuilder.base_key(
        'production', identity, {**_CLIENT_OPTIONS, 'records_per_page': 500}, _call()
    )
    other_page = KeyBuilder.base_key(
        'production', identity, {**_CLIENT_OPTIONS, 'records_per_page': 100}, _call()
    )

    assert len({without, with_page, other_page}) == 3
    for bad in (0, -1, True, '500'):
        with pytest.raises(NormalizationError):
            KeyBuilder.base_key(
                'production',
                identity,
                {**_CLIENT_OPTIONS, 'records_per_page': bad},
                _call(),
            )


def test_worst_case_vector_bytes_uses_maximum_token_lengths() -> None:
    tags = (DependencyTag.site('production'), DependencyTag.entity('production', 'Asset', 7))
    expected = encoded_size(
        (
            _INCARNATION,
            tuple((tag.kind.value, tag.components, _MAX_GENERATION) for tag in tags),
        )
    )

    assert KeyBuilder.worst_case_vector_bytes(tags) == expected
