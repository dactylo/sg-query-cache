from typing import TYPE_CHECKING, cast

import pytest

from sg_query_cache._client_common import (
    ClientContext,
    EntityPlanContext,
    EntityRead,
    _batch_changes,
    _batch_post_changes,
    _freeze,
    _key_freeze,
    entity_calls,
    entity_read_plan,
    population_callback,
)
from sg_query_cache._config import CacheSettings
from sg_query_cache._dependencies import DependencyAnalyzer, SchemaView
from sg_query_cache._model import DependencyPlan, DependencyTag, GenerationChange, Identity

if TYPE_CHECKING:
    from collections.abc import Callable


def _calculated_read_plan(
    fields: list[str], order: list[dict[str, str]] | None = None
) -> tuple[DependencyPlan, dict[str, str] | None]:
    schema = SchemaView.from_api_response(
        {
            'Task': {
                'id': {'data_type': {'value': 'number'}, 'properties': {}},
                'content': {'data_type': {'value': 'text'}, 'properties': {}},
                'workload': {'data_type': {'value': 'calculated'}, 'properties': {}},
            }
        }
    )
    read = EntityRead(
        'find',
        'Task',
        fields,
        ('Task', [['content', 'is', 'Animation']], fields, order, 'all', 0, False, 0, True, None),
    )
    call, _ = entity_calls(read)
    assert call is not None
    context = EntityPlanContext(
        DependencyAnalyzer(),
        Identity('script', 'studio-sso', 'render-farm'),
        CacheSettings(),
        'production',
    )
    return entity_read_plan(call, schema, 'Task', fields, context)


def test_analysis_and_key_freezing_keep_their_distinct_mapping_contracts() -> None:
    value = {'z': 2, 'a': 1}

    assert _freeze(value) == (('z', 2), ('a', 1))
    assert _key_freeze(value) == b'["map",[[["str","z"],["int",2]],[["str","a"],["int",1]]]]'
    assert _key_freeze({'filter_operator': 'all', 'filters': [['id', 'is', 7]]}) == (
        ('filter_operator', 'all'),
        ('filters', (('id', 'is', 7),)),
    )


def test_batch_created_entities_preserve_first_occurrence_and_collapse_at_bound() -> None:
    requests: list[dict[str, object]] = [
        {'request_type': 'create', 'entity_type': 'Asset', 'data': {}} for _ in range(259)
    ]
    responses = [{'type': 'Asset', 'id': entity_id} for entity_id in range(1, 260)]
    changes = _batch_changes('studio', requests[:2])

    assert _batch_post_changes('studio', requests[:2], changes, responses[:2]) == (
        *changes,
        GenerationChange(DependencyTag.entity('studio', 'Asset', 1)),
        GenerationChange(DependencyTag.entity('studio', 'Asset', 2)),
    )
    assert _batch_post_changes('studio', requests, changes, responses) == (
        GenerationChange(DependencyTag.site('studio')),
    )


def test_calculated_field_is_not_url_bearing() -> None:
    _, returned_fields = _calculated_read_plan(['workload'])

    assert returned_fields == {'workload': 'calculated'}
    assert population_callback(returned_fields, CacheSettings()) is None


@pytest.mark.parametrize(
    ('fields', 'order'),
    [
        (['id', 'workload'], None),
        (['id'], [{'field_name': 'workload', 'direction': 'asc'}]),
    ],
    ids=('selected', 'ordered'),
)
def test_calculated_field_reads_are_ttl_only(
    fields: list[str], order: list[dict[str, str]] | None
) -> None:
    plan, returned_fields = _calculated_read_plan(fields, order)

    assert plan.classification == 'ttl_only'
    assert plan.reason == 'uncertain_field_dependencies'
    assert returned_fields == {
        field: 'number' if field == 'id' else 'calculated' for field in fields
    }


@pytest.mark.parametrize('bad_key', ['', 7])
@pytest.mark.parametrize('request_type', ['create', 'update'])
def test_batch_with_invalid_data_key_uses_site_fence(request_type: str, bad_key: object) -> None:
    request: dict[str, object] = {
        'request_type': request_type,
        'entity_type': 'Asset',
        'data': {bad_key: 'value'},
    }
    if request_type == 'update':
        request['entity_id'] = 7

    assert _batch_changes('studio', [request]) == (GenerationChange(DependencyTag.site('studio')),)


def test_client_context_rejects_positional_arguments() -> None:
    constructor = cast('Callable[..., ClientContext]', ClientContext)
    with pytest.raises(TypeError, match='positional argument'):
        constructor(
            'https://example.shotgrid.autodesk.com',
            site_namespace='studio',
            localized=False,
            convert_datetimes_to_utc=True,
            records_per_page=500,
        )


def test_context_requires_a_positive_integer_page_size() -> None:
    for bad in (0, True, '500'):
        with pytest.raises((TypeError, ValueError)):
            ClientContext(
                site_url='https://example.shotgrid.autodesk.com',
                site_namespace='studio',
                localized=False,
                convert_datetimes_to_utc=True,
                records_per_page=cast('int', bad),
            )


def test_large_batch_collapses_to_one_site_fence() -> None:
    requests: list[dict[str, object]] = [
        {'request_type': 'delete', 'entity_type': 'Asset', 'entity_id': entity_id}
        for entity_id in range(1, 260)
    ]

    assert _batch_changes('studio', requests) == (GenerationChange(DependencyTag.site('studio')),)
