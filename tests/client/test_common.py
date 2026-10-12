from typing import cast

import pytest

from sg_query_cache._client_common import (
    ClientContext,
    batch_changes,
    batch_post_changes,
    freeze,
    key_freeze,
)
from sg_query_cache._model import DependencyTag, GenerationChange


def test_analysis_and_key_freezing_keep_their_distinct_mapping_contracts() -> None:
    value = {'z': 2, 'a': 1}

    assert freeze(value) == (('z', 2), ('a', 1))
    assert key_freeze(value) == b'["map",[[["str","z"],["int",2]],[["str","a"],["int",1]]]]'
    assert key_freeze({'filter_operator': 'all', 'filters': [['id', 'is', 7]]}) == (
        ('filter_operator', 'all'),
        ('filters', (('id', 'is', 7),)),
    )


def test_batch_created_entities_preserve_first_occurrence_and_collapse_at_bound() -> None:
    requests: list[dict[str, object]] = [
        {'request_type': 'create', 'entity_type': 'Asset', 'data': {}} for _ in range(259)
    ]
    responses = [{'type': 'Asset', 'id': entity_id} for entity_id in range(1, 260)]
    changes = batch_changes('studio', requests[:2])

    assert batch_post_changes('studio', requests[:2], changes, responses[:2]) == (
        *changes,
        GenerationChange(DependencyTag.entity('studio', 'Asset', 1)),
        GenerationChange(DependencyTag.entity('studio', 'Asset', 2)),
    )
    assert batch_post_changes('studio', requests, changes, responses) == (
        GenerationChange(DependencyTag.site('studio')),
    )


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

    assert batch_changes('studio', [request]) == (GenerationChange(DependencyTag.site('studio')),)


def test_context_requires_a_positive_integer_page_size() -> None:
    for bad in (0, True, '500'):
        with pytest.raises((TypeError, ValueError)):
            ClientContext(
                'https://example.shotgrid.autodesk.com',
                'studio',
                localized=False,
                convert_datetimes_to_utc=True,
                records_per_page=cast('int', bad),
            )


def test_large_batch_collapses_to_one_site_fence() -> None:
    requests: list[dict[str, object]] = [
        {'request_type': 'delete', 'entity_type': 'Asset', 'entity_id': entity_id}
        for entity_id in range(1, 260)
    ]

    assert batch_changes('studio', requests) == (GenerationChange(DependencyTag.site('studio')),)
