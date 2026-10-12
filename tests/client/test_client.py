from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, date, datetime
from threading import Lock
from typing import cast

import pytest

from sg_query_cache._cache import QueryCache
from sg_query_cache._client import CachedShotgun, ClientContext
from sg_query_cache._config import CacheSettings, L1Settings, L2Settings, LimitSettings
from sg_query_cache._keys import KeyBuilder
from sg_query_cache._local import LocalBackend
from sg_query_cache._model import (
    DependencyPlan,
    DependencyTag,
    ExactCall,
    GenerationChange,
    Identity,
)
from tests._fakes import _FakeClock, _RecordingMetrics

_SCHEMA: dict[str, dict[str, dict[str, object]]] = {
    'Asset': {
        'code': {'data_type': {'value': 'text'}, 'properties': {}},
        'id': {'data_type': {'value': 'number'}, 'properties': {}},
        'project': {
            'data_type': {'value': 'entity'},
            'properties': {'valid_types': {'value': ['Project']}},
        },
        'thumbnail': {'data_type': {'value': 'image'}, 'properties': {}},
    },
    'Project': {
        'id': {'data_type': {'value': 'number'}, 'properties': {}},
    },
    'Shot': {
        'id': {'data_type': {'value': 'number'}, 'properties': {}},
    },
}


class _FakeShotgun:
    __slots__ = ('_calls', '_lock', '_schema')

    def __init__(self, schema: object = _SCHEMA) -> None:
        self._calls: list[tuple[str, tuple[object, ...], dict[str, object]]] = []
        self._schema = schema
        self._lock = Lock()

    def find(  # noqa: PLR0913, PLR0917 - official-compatible fake
        self,
        entity_type: str,
        filters: object,
        fields: list[str] | None = None,
        order: list[dict[str, str]] | None = None,
        filter_operator: str | None = None,
        limit: int = 0,
        retired_only: bool = False,
        page: int = 0,
        include_archived_projects: bool = True,
        additional_filter_presets: object = None,
    ) -> list[dict[str, object]]:
        with self._lock:
            call_number = sum(name == 'find' for name, _args, _kwargs in self._calls) + 1
            self._calls.append(
                (
                    'find',
                    (entity_type, filters),
                    {
                        'fields': fields,
                        'order': order,
                        'filter_operator': filter_operator,
                        'limit': limit,
                        'retired_only': retired_only,
                        'page': page,
                        'include_archived_projects': include_archived_projects,
                        'additional_filter_presets': additional_filter_presets,
                    },
                )
            )
        return [{'id': call_number, 'type': entity_type}]

    def find_one(  # noqa: PLR0913, PLR0917 - official-compatible fake
        self,
        entity_type: str,
        filters: object,
        fields: list[str] | None = None,
        order: list[dict[str, str]] | None = None,
        filter_operator: str | None = None,
        retired_only: bool = False,
        include_archived_projects: bool = True,
        additional_filter_presets: object = None,
    ) -> dict[str, object] | None:
        self._calls.append(
            (
                'find_one',
                (entity_type, filters),
                {
                    'fields': fields,
                    'order': order,
                    'filter_operator': filter_operator,
                    'retired_only': retired_only,
                    'include_archived_projects': include_archived_projects,
                    'additional_filter_presets': additional_filter_presets,
                },
            )
        )
        return {'id': 99, 'type': entity_type}

    def schema_entity_read(
        self, project_entity: Mapping[str, object] | None = None
    ) -> dict[str, object]:
        self._calls.append(('schema_entity_read', (project_entity,), {}))
        return {'Asset': {'name': {'value': 'Asset'}}}

    def schema_field_read(
        self,
        entity_type: str,
        field_name: str | None = None,
        project_entity: Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        self._calls.append(('schema_field_read', (entity_type, field_name, project_entity), {}))
        schema = self._schema
        if not isinstance(schema, dict):
            return {}
        value = cast('dict[str, object]', schema).get(entity_type, {})
        return cast('dict[str, object]', value) if isinstance(value, dict) else {}

    def schema_read(self, project_entity: Mapping[str, object] | None = None) -> object:
        self._calls.append(('schema_read', (project_entity,), {}))
        return self._schema

    def summarize(  # noqa: PLR0913, PLR0917 - official-compatible fake
        self,
        entity_type: str,
        filters: object,
        summary_fields: list[dict[str, str]],
        filter_operator: str | None = None,
        grouping: list[dict[str, str]] | None = None,
        include_archived_projects: bool = True,
    ) -> dict[str, object]:
        self._calls.append(
            (
                'summarize',
                (entity_type, filters, summary_fields),
                {
                    'filter_operator': filter_operator,
                    'grouping': grouping,
                    'include_archived_projects': include_archived_projects,
                },
            )
        )
        return {'summaries': len(self._calls)}


class _CreateClient(_FakeShotgun):
    __slots__ = ('_created',)

    def __init__(self) -> None:
        super().__init__()
        self._created = False

    def batch(self, requests: list[dict[str, object]]) -> list[dict[str, object]]:
        self._calls.append(('batch', (requests,), {}))
        self._created = True
        return [{'id': 17, 'type': 'Asset', 'code': 'oak'}]

    def create(
        self,
        entity_type: str,
        data: dict[str, object],
        return_fields: list[str] | None = None,
    ) -> dict[str, object]:
        self._calls.append(('create', (entity_type, data, return_fields), {}))
        self._created = True
        return {'id': 17, 'type': entity_type, **data}

    def find(  # noqa: PLR0913, PLR0917 - official-compatible fake
        self,
        entity_type: str,
        filters: object,
        fields: list[str] | None = None,
        order: list[dict[str, str]] | None = None,
        filter_operator: str | None = None,
        limit: int = 0,
        retired_only: bool = False,
        page: int = 0,
        include_archived_projects: bool = True,
        additional_filter_presets: object = None,
    ) -> list[dict[str, object]]:
        self._calls.append(('find', (entity_type, filters), {'fields': fields}))
        _ = (
            order,
            filter_operator,
            limit,
            retired_only,
            page,
            include_archived_projects,
            additional_filter_presets,
        )
        return [{'id': 17, 'type': entity_type}] if self._created else []


class _MutatingClient:
    __slots__ = ('_calls', '_failure')

    def __init__(self, failure: Exception | None = None) -> None:
        self._calls: list[tuple[str, tuple[object, ...], dict[str, object]]] = []
        self._failure = failure

    def __getattr__(self, name: str) -> object:
        def invoke(*args: object, **kwargs: object) -> object:
            self._calls.append((name, args, kwargs))
            if self._failure is not None:
                raise self._failure
            if name == 'create':
                return {'id': 17, 'type': args[0]}
            if name == 'batch':
                requests = cast('list[dict[str, object]]', args[0])
                return [
                    {'id': index + 17, 'type': request['entity_type']}
                    if request.get('request_type') == 'create'
                    else True
                    if request.get('request_type') == 'delete'
                    else {'ok': True}
                    for index, request in enumerate(requests)
                ]
            if name in {'delete', 'revive', 'schema_field_delete', 'schema_field_update'}:
                return True
            if name in {'upload', 'upload_filmstrip_thumbnail', 'upload_thumbnail'}:
                return 23
            return {'ok': True}

        return invoke


class _RecordingCache:
    __slots__ = ('_identity', '_invalidations', '_reads', '_settings')

    def __init__(self) -> None:
        self._identity = Identity('script', 'studio', 'render')
        self._settings = CacheSettings(site_namespace='studio')
        self._invalidations: list[tuple[GenerationChange, ...]] = []
        self._reads: list[tuple[ExactCall, DependencyPlan, dict[str, object]]] = []

    def invalidate(self, changes: tuple[GenerationChange, ...]) -> bool:
        self._invalidations.append(changes)
        return True

    def read(
        self,
        call: ExactCall,
        plan: DependencyPlan,
        fetch: Callable[[], object],
        client_options: Mapping[str, object],
        per_call_age: float | None = None,
        *,
        _population_decision: object = None,
    ) -> object:
        _ = (per_call_age, _population_decision)
        self._reads.append((call, plan, dict(client_options)))
        return fetch()


class _SecurityWriteClient(_FakeShotgun):
    __slots__ = ('_during_write',)

    def __init__(self) -> None:
        super().__init__()
        self._during_write: Callable[[], None] = lambda: None

    def batch(self, requests: list[dict[str, object]]) -> list[dict[str, object]]:
        self._calls.append(('batch', (requests,), {}))
        self._during_write()
        return [
            {'id': 17 + index, 'type': request['entity_type']}
            if request.get('request_type') == 'create'
            else {'ok': True}
            for index, request in enumerate(requests)
        ]

    def create(
        self,
        entity_type: str,
        data: dict[str, object],
        return_fields: list[str] | None = None,
    ) -> dict[str, object]:
        self._calls.append(('create', (entity_type, data, return_fields), {}))
        self._during_write()
        return {'id': 17, 'type': entity_type, **data}

    def delete(self, entity_type: str, entity_id: int) -> bool:
        self._calls.append(('delete', (entity_type, entity_id), {}))
        self._during_write()
        return True

    def update(
        self,
        entity_type: str,
        entity_id: int,
        data: dict[str, object],
        multi_entity_update_modes: dict[str, object] | None = None,
    ) -> dict[str, object]:
        self._calls.append(
            (
                'update',
                (entity_type, entity_id, data, multi_entity_update_modes),
                {},
            )
        )
        self._during_write()
        return {'id': entity_id, 'type': entity_type, **data}


class _OfficialCompositeClient(_MutatingClient):
    __slots__ = ('_entity_rpc_committed',)

    def __init__(self, failure: Exception | None = None) -> None:
        super().__init__(failure)
        self._entity_rpc_committed = False

    def create(
        self,
        entity_type: str,
        data: dict[str, object],
        return_fields: list[str] | None = None,
    ) -> dict[str, object]:
        owned_data = data.copy()
        upload_image = owned_data.pop('image', None)
        upload_filmstrip = owned_data.pop('filmstrip_image', None)
        self._calls.append(('create_rpc', (entity_type, owned_data, return_fields), {}))
        self._entity_rpc_committed = True
        if (upload_image or upload_filmstrip) and self._failure is not None:
            raise self._failure
        return {'id': 17, 'type': entity_type, **owned_data}

    def update(
        self,
        entity_type: str,
        entity_id: int,
        data: dict[str, object],
        multi_entity_update_modes: dict[str, object] | None = None,
    ) -> dict[str, object]:
        owned_data = data.copy()
        upload_image = owned_data.pop('image', None)
        upload_filmstrip = owned_data.pop('filmstrip_image', None)
        self._calls.append(
            (
                'update_rpc',
                (entity_type, entity_id, owned_data, multi_entity_update_modes),
                {},
            )
        )
        self._entity_rpc_committed = True
        if (upload_image or upload_filmstrip) and self._failure is not None:
            raise self._failure
        return {'id': entity_id, 'type': entity_type, **owned_data}


def _large_schema(
    fields_per_type: int = 100,
    types: int = 120,
) -> dict[str, dict[str, dict[str, object]]]:
    return {
        f'CustomEntity{index:02d}': {
            **{
                f'sg_field_{field:03d}': {
                    'data_type': {'value': 'text'},
                    'properties': {},
                }
                for field in range(fields_per_type)
            },
            'id': {'data_type': {'value': 'number'}, 'properties': {}},
        }
        for index in range(types)
    }


def _paged_wrapper(
    records_per_page: int,
    cache: QueryCache,
    client: _FakeShotgun,
) -> CachedShotgun:
    return CachedShotgun(
        client,
        cache,
        ClientContext(
            'https://example.shotgrid.autodesk.com',
            'studio',
            localized=False,
            convert_datetimes_to_utc=True,
            records_per_page=records_per_page,
        ),
    )


def _wrapper(
    client: _FakeShotgun | None = None,
    *,
    identity: Identity | None = None,
    settings: CacheSettings | None = None,
) -> tuple[CachedShotgun, _FakeShotgun, LocalBackend]:
    selected_client = client or _FakeShotgun()
    selected_settings = settings or CacheSettings(site_namespace='studio')
    backend = LocalBackend(selected_settings, _FakeClock(), _RecordingMetrics())
    cache = QueryCache(
        selected_settings,
        identity or Identity('script', 'studio', 'render'),
        KeyBuilder(),
        l1=backend,
    )
    context = ClientContext(
        'HTTPS://EXAMPLE.SHOTGRID.AUTODESK.COM/',
        'studio',
        localized=False,
        convert_datetimes_to_utc=True,
        records_per_page=500,
    )
    return CachedShotgun(selected_client, cache, context), selected_client, backend


class _DefiniteWriteError(Exception):
    pass


class _IndeterminateWriteError(Exception):
    pass


def _write_wrapper(
    client: _MutatingClient | None = None,
    *,
    indeterminate_write_exceptions: tuple[type[Exception], ...] = (),
) -> tuple[CachedShotgun, _MutatingClient, _RecordingCache]:
    selected_client = client or _MutatingClient()
    cache = _RecordingCache()
    wrapper = CachedShotgun(
        selected_client,
        cast('QueryCache', cache),
        ClientContext(
            'https://example.shotgrid.autodesk.com',
            'studio',
            localized=False,
            convert_datetimes_to_utc=True,
            records_per_page=500,
        ),
        indeterminate_write_exceptions=indeterminate_write_exceptions,
    )
    return wrapper, selected_client, cache


def test_batch_unions_known_changes_and_passes_the_original_ordered_list() -> None:
    wrapper, client, cache = _write_wrapper()
    requests: list[dict[str, object]] = [
        {'request_type': 'create', 'entity_type': 'Shot', 'data': {'code': 'one'}},
        {
            'request_type': 'update',
            'entity_type': 'Asset',
            'entity_id': 7,
            'data': {'status': 'ip'},
        },
        {'request_type': 'delete', 'entity_type': 'Task', 'entity_id': 9},
    ]

    responses: list[dict[str, object] | bool] = wrapper.batch(requests)

    assert responses == [
        {'id': 17, 'type': 'Shot'},
        {'ok': True},
        True,
    ]

    assert client._calls == [('batch', (requests,), {})]
    assert cache._invalidations[1][:-1] == cache._invalidations[0]
    assert cache._invalidations[1][-1] == GenerationChange(
        DependencyTag.entity('studio', 'Shot', 17)
    )
    assert DependencyTag.site('studio') not in {change.tag for change in cache._invalidations[0]}


def test_batch_with_implicit_upload_fields_uses_and_repeats_the_site_fence() -> None:
    wrapper, _client, cache = _write_wrapper(_MutatingClient(RuntimeError('boom')))
    request: dict[str, object] = {
        'request_type': 'update',
        'entity_type': 'Shot',
        'entity_id': 5,
        'data': {'image': 'thumb.png'},
    }

    with pytest.raises(RuntimeError, match='boom'):
        wrapper.batch([request])

    site = (GenerationChange(DependencyTag.site('studio')),)
    assert cache._invalidations == [site, site]


def test_bypass_scope_performs_no_schema_acquisition() -> None:
    wrapper, client, _backend = _wrapper()

    with wrapper.bypass():
        for _ in range(3):
            wrapper.find('Asset', [], ['id'])

    names = [name for name, _args, _kwargs in client._calls]
    assert names.count('schema_read') == 0
    assert names.count('find') == 3


def test_cold_identity_namespaces_do_not_reserve_per_user_capacity() -> None:
    settings = CacheSettings(site_namespace='studio', identity_policy='scripts_and_humans')
    backend = LocalBackend(settings, _FakeClock(), _RecordingMetrics())
    before = backend.storage_usage()

    wrappers = [
        CachedShotgun(
            _FakeShotgun(),
            QueryCache(
                settings,
                Identity('human', 'studio', f'user-{entity_id}', entity_id=entity_id),
                KeyBuilder(),
                l1=backend,
            ),
            ClientContext(
                'https://example.shotgrid.autodesk.com',
                'studio',
                localized=False,
                convert_datetimes_to_utc=True,
                records_per_page=500,
            ),
        )
        for entity_id in range(1, 1_002)
    ]

    assert len(wrappers) == 1_001
    assert backend.storage_usage() == before


def test_concurrent_calls_keep_exact_keys_and_results_isolated() -> None:
    wrapper, client, _backend = _wrapper()

    def read(entity_type: str) -> tuple[str, int]:
        value = wrapper.find(entity_type, [], ['id'])
        entity_id = value[0]['id']
        assert isinstance(entity_id, int)
        return entity_type, entity_id

    with ThreadPoolExecutor(max_workers=16) as executor:
        results = list(executor.map(read, ('Asset', 'Shot') * 20))

    by_type = {
        entity_type: {
            entity_id for result_type, entity_id in results if result_type == entity_type
        }
        for entity_type in ('Asset', 'Shot')
    }
    assert all(len(entity_ids) == 1 for entity_ids in by_type.values())
    assert by_type['Asset'] != by_type['Shot']
    assert [name for name, _args, _kwargs in client._calls].count('find') == 2


def test_context_normalizes_site_and_find_uses_identity_scoped_schema_once() -> None:
    wrapper, client, _backend = _wrapper()

    assert wrapper.context == ClientContext(
        'https://example.shotgrid.autodesk.com',
        'studio',
        localized=False,
        convert_datetimes_to_utc=True,
        records_per_page=500,
    )
    assert wrapper.find('Asset', [['code', 'is', 'tree']], ['code']) == [
        {'id': 1, 'type': 'Asset'}
    ]
    assert wrapper.find('Asset', [['code', 'is', 'tree']], ['code']) == [
        {'id': 1, 'type': 'Asset'}
    ]
    assert [name for name, _args, _kwargs in client._calls].count('schema_read') == 1
    assert [name for name, _args, _kwargs in client._calls].count('find') == 1


def test_context_replacement_invalidates_old_namespace_before_new_context_read() -> None:
    settings = CacheSettings(site_namespace='studio')
    client = _FakeShotgun()
    backend = LocalBackend(settings, _FakeClock(), _RecordingMetrics())
    cache = QueryCache(
        settings,
        Identity('script', 'studio', 'render'),
        KeyBuilder(),
        l1=backend,
    )
    old_context = ClientContext(
        'https://example.shotgrid.autodesk.com',
        'studio',
        localized=False,
        convert_datetimes_to_utc=True,
        records_per_page=500,
    )
    wrapper = CachedShotgun(client, cache, old_context)
    old_reader = CachedShotgun(client, cache, old_context)

    assert wrapper.find('Asset', [], ['id']) == [{'id': 1, 'type': 'Asset'}]
    assert old_reader.find('Asset', [], ['id']) == [{'id': 1, 'type': 'Asset'}]
    wrapper.replace_context(
        ClientContext(
            'https://example.shotgrid.autodesk.com',
            'studio',
            localized=True,
            convert_datetimes_to_utc=True,
            records_per_page=500,
        )
    )
    assert old_reader.find('Asset', [], ['id']) == [{'id': 2, 'type': 'Asset'}]
    assert old_reader.find('Asset', [], ['id']) == [{'id': 2, 'type': 'Asset'}]
    assert wrapper.find('Asset', [], ['id']) == [{'id': 3, 'type': 'Asset'}]
    assert [name for name, _args, _kwargs in client._calls].count('find') == 3


@pytest.mark.parametrize('method_name', ['create', 'batch'])
def test_create_post_fence_invalidates_exact_id_negative_cache(method_name: str) -> None:
    client = _CreateClient()
    wrapper, _client, _backend = _wrapper(client)
    filters = [['id', 'is', 17]]

    assert wrapper.find('Asset', filters, ['id']) == []
    if method_name == 'create':
        wrapper.create('Asset', {'code': 'oak'})
    else:
        wrapper.batch(
            [{'request_type': 'create', 'entity_type': 'Asset', 'data': {'code': 'oak'}}]
        )
    assert wrapper.find('Asset', filters, ['id']) == [{'id': 17, 'type': 'Asset'}]


@pytest.mark.parametrize(
    ('method_name', 'args'),
    [
        ('create', ('PermissionRole', {'name': 'Artist'})),
        ('update', ('ApiUser', 7, {'firstname': 'Render'})),
        ('delete', ('PermissionRuleSet', 9)),
    ],
)
def test_authorization_writes_fence_unrelated_cached_reads(
    method_name: str,
    args: tuple[object, ...],
) -> None:
    client = _SecurityWriteClient()
    wrapper, _client, _backend = _wrapper(client)

    assert wrapper.find('Asset', [], ['id']) == [{'id': 1, 'type': 'Asset'}]

    def read_during_write() -> None:
        wrapper.find('Asset', [], ['id'])

    client._during_write = read_during_write

    getattr(wrapper, method_name)(*args)

    assert wrapper.find('Asset', [], ['id']) == [{'id': 3, 'type': 'Asset'}]
    assert [name for name, _args, _kwargs in client._calls].count('find') == 3


class _ScheduleClient(_FakeShotgun):
    __slots__ = ()

    def work_schedule_read(
        self,
        start_date: str,
        end_date: str,
        project: dict[str, object] | None = None,
        user: dict[str, object] | None = None,
    ) -> dict[str, object]:
        self._calls.append(
            ('work_schedule_read', (start_date, end_date), {'project': project, 'user': user})
        )
        return {'2026-09-03': {'working': True, 'description': None, 'reason': None}}


def test_documented_temporal_filter_values_reach_the_client() -> None:
    wrapper, client, _backend = _wrapper()

    assert wrapper.find('Asset', [['created_at', 'is', date(2026, 8, 30)]], ['id']) == [
        {'id': 1, 'type': 'Asset'}
    ]
    assert [name for name, _args, _kwargs in client._calls].count('find') == 1


def test_enabled_work_schedule_read_uses_exact_cache() -> None:
    settings = CacheSettings(
        site_namespace='studio', enabled_ttl_only_methods=('work_schedule_read',)
    )
    wrapper, client, _backend = _wrapper(_ScheduleClient(), settings=settings)

    first = wrapper.work_schedule_read('2026-09-01', '2026-09-07')
    second = wrapper.work_schedule_read('2026-09-01', '2026-09-07')

    assert first == second
    assert [name for name, _args, _kwargs in client._calls].count('work_schedule_read') == 1


def test_entity_reference_filters_have_cacheable_exact_keys() -> None:
    wrapper, client, _backend = _wrapper()
    filters = [['project', 'is', {'type': 'Project', 'id': 2}]]

    assert wrapper.find('Asset', filters, ['id']) == [{'id': 1, 'type': 'Asset'}]
    assert wrapper.find('Asset', filters, ['id']) == [{'id': 1, 'type': 'Asset'}]
    assert [name for name, _args, _kwargs in client._calls].count('find') == 1


def test_event_log_entity_reads_always_bypass() -> None:
    wrapper, client, _backend = _wrapper()

    first = wrapper.find('EventLogEntry', [], ['id'])
    second = wrapper.find('EventLogEntry', [], ['id'])

    assert first != second
    assert [name for name, _args, _kwargs in client._calls].count('find') == 2


def test_find_and_find_one_use_distinct_exact_call_keys() -> None:
    wrapper, client, _backend = _wrapper()

    assert wrapper.find('Asset', [], ['id']) == [{'id': 1, 'type': 'Asset'}]
    assert wrapper.find_one('Asset', [], ['id']) == {'id': 99, 'type': 'Asset'}
    assert wrapper.find('Asset', [], ['id']) == [{'id': 1, 'type': 'Asset'}]
    assert wrapper.find_one('Asset', [], ['id']) == {'id': 99, 'type': 'Asset'}
    names = [name for name, _args, _kwargs in client._calls]
    assert names.count('find') == 1
    assert names.count('find_one') == 1


def test_find_positional_and_keyword_forms_share_one_exact_call_key() -> None:
    wrapper, client, _backend = _wrapper()

    positional = wrapper.find('Asset', [['id', 'is', 7]], ['id'], [])
    keyword = wrapper.find(
        entity_type='Asset',
        filters=[['id', 'is', 7]],
        fields=['id'],
        order=[],
    )

    assert positional == keyword == [{'id': 1, 'type': 'Asset'}]
    assert [name for name, _args, _kwargs in client._calls].count('find') == 1


def test_find_variants_forward_and_plan_independent_exact_operands() -> None:
    wrapper, client, cache = _write_wrapper()
    filters = [['id', 'is', 7]]
    fields = ['code', 'id']
    order = [{'field_name': 'code', 'direction': 'desc'}]
    presets = [{'preset_name': 'LATEST', 'latest_by': 'ENTITIES_CREATED_AT'}]

    wrapper.find(
        'EventLogEntry',
        filters,
        fields,
        order,
        filter_operator='all',
        limit=25,
        retired_only=True,
        page=2,
        include_archived_projects=False,
        additional_filter_presets=presets,
    )
    wrapper.find_one(
        'EventLogEntry',
        filters,
        fields,
        order,
        filter_operator='all',
        retired_only=True,
        include_archived_projects=False,
        additional_filter_presets=presets,
    )

    expected_common = (
        ('entity_type', 'EventLogEntry'),
        ('filters', (('id', 'is', 7),)),
        ('fields', ('code', 'id')),
        ('order', ((('field_name', 'code'), ('direction', 'desc')),)),
        ('filter_operator', 'all'),
    )
    assert cache._reads == [
        (
            ExactCall(
                'find',
                (
                    *expected_common,
                    ('limit', 25),
                    ('retired_only', True),
                    ('page', 2),
                    ('include_archived_projects', False),
                    (
                        'additional_filter_presets',
                        (
                            b'["map",[[["str","preset_name"],["str","LATEST"]],[["str","latest_by"],["str","ENTITIES_CREATED_AT"]]]]',
                        ),
                    ),
                    ('site_url', 'https://example.shotgrid.autodesk.com'),
                ),
            ),
            DependencyPlan('bypass', (), 'event_log_entry'),
            {
                'localized': False,
                'convert_datetimes_to_utc': True,
                'records_per_page': 500,
            },
        ),
        (
            ExactCall(
                'find_one',
                (
                    *expected_common,
                    ('retired_only', True),
                    ('include_archived_projects', False),
                    (
                        'additional_filter_presets',
                        (
                            b'["map",[[["str","preset_name"],["str","LATEST"]],[["str","latest_by"],["str","ENTITIES_CREATED_AT"]]]]',
                        ),
                    ),
                    ('site_url', 'https://example.shotgrid.autodesk.com'),
                ),
            ),
            DependencyPlan('bypass', (), 'event_log_entry'),
            {'localized': False, 'convert_datetimes_to_utc': True},
        ),
    ]
    assert client._calls == [
        (
            'find',
            ('EventLogEntry', filters, fields, order, 'all', 25, True, 2, False, presets),
            {},
        ),
        (
            'find_one',
            ('EventLogEntry', filters, fields, order, 'all', True, False, presets),
            {},
        ),
    ]


def test_human_user_lifecycle_fences_the_matching_cached_identity() -> None:
    client = _SecurityWriteClient()
    settings = CacheSettings(
        site_namespace='studio',
        identity_policy='scripts_and_humans',
    )
    wrapper, _client, _backend = _wrapper(
        client,
        identity=Identity('human', 'studio', 'user-7', entity_id=7),
        settings=settings,
    )

    assert wrapper.find('Asset', [], ['id']) == [{'id': 1, 'type': 'Asset'}]

    def read_during_write() -> None:
        wrapper.find('Asset', [], ['id'])

    client._during_write = read_during_write

    wrapper.delete('HumanUser', 7)

    assert wrapper.find('Asset', [], ['id']) == [{'id': 3, 'type': 'Asset'}]


def test_human_user_security_fences_use_exact_and_global_visibility_tags() -> None:
    wrapper, _client, cache = _write_wrapper()

    wrapper.update('HumanUser', 7, {'permission_role_id': 3})
    update = cache._invalidations[0]
    wrapper.delete('HumanUser', 7)
    lifecycle = cache._invalidations[2]
    wrapper.create('HumanUser', {'permission_group': {'id': 5}})
    create = cache._invalidations[4]

    assert tuple(change.tag for change in update[:2]) == (
        DependencyTag.authorization('studio'),
        DependencyTag.identity_visibility('studio', 7),
    )
    assert update == cache._invalidations[1]
    assert lifecycle[0].tag == DependencyTag.identity_visibility('studio', 7)
    assert DependencyTag.authorization('studio') not in {change.tag for change in lifecycle}
    assert lifecycle == cache._invalidations[3]
    assert tuple(change.tag for change in create[:2]) == (
        DependencyTag.authorization('studio'),
        DependencyTag.human_visibility('studio'),
    )
    assert create == cache._invalidations[5][:-1]


@pytest.mark.parametrize(
    ('method_name', 'args'),
    [
        ('create', ('HumanUser', {'login': 'worker'})),
        ('update', ('HumanUser', 7, {'permission_role_id': 3})),
        ('create', ('Group', {'users': list[object]()})),
        ('update', ('ProjectUserConnection', 7, {'user': {'id': 42}})),
        ('delete', ('TaskAssigneeConnection', 7)),
        ('update', ('Note', 7, {'owner': {'id': 42}})),
        (
            'batch',
            (
                [
                    {
                        'request_type': 'update',
                        'entity_type': 'ApiUser',
                        'entity_id': 7,
                        'data': {'firstname': 'Render'},
                    },
                    {
                        'request_type': 'delete',
                        'entity_type': 'Group',
                        'entity_id': 9,
                    },
                ],
            ),
        ),
    ],
)
def test_human_visibility_writes_fence_unrelated_cached_reads(
    method_name: str,
    args: tuple[object, ...],
) -> None:
    client = _SecurityWriteClient()
    settings = CacheSettings(
        site_namespace='studio',
        identity_policy='scripts_and_humans',
    )
    wrapper, _client, _backend = _wrapper(
        client,
        identity=Identity('human', 'studio', 'user-42', entity_id=42),
        settings=settings,
    )

    assert wrapper.find('Asset', [], ['id']) == [{'id': 1, 'type': 'Asset'}]

    def read_during_write() -> None:
        wrapper.find('Asset', [], ['id'])

    client._during_write = read_during_write

    getattr(wrapper, method_name)(*args)

    assert wrapper.find('Asset', [], ['id']) == [{'id': 3, 'type': 'Asset'}]
    assert [name for name, _args, _kwargs in client._calls].count('find') == 3


def test_known_update_uses_conservative_pre_and_post_fences() -> None:
    wrapper, _client, cache = _write_wrapper()

    assert wrapper.update('Asset', 7, {'code': 'oak', 'status': 'ip'}) == {'ok': True}

    expected = tuple(
        GenerationChange(tag)
        for tag in (
            DependencyTag.entity('studio', 'Asset', 7),
            DependencyTag.type_value('studio', 'Asset'),
            DependencyTag.type_field('studio', 'Asset', 'code'),
            DependencyTag.type_field('studio', 'Asset', 'status'),
            DependencyTag.type_membership('studio', 'Asset'),
        )
    )
    assert cache._invalidations == [expected, expected]


def test_malformed_schema_falls_back_without_guessing_dependencies() -> None:
    wrapper, client, _backend = _wrapper(_FakeShotgun(schema='malformed'))

    assert wrapper.find('Asset', [['code', 'is', 'tree']], ['code']) == [
        {'id': 1, 'type': 'Asset'}
    ]
    assert wrapper.find('Asset', [['code', 'is', 'tree']], ['code']) == [
        {'id': 2, 'type': 'Asset'}
    ]
    assert [name for name, _args, _kwargs in client._calls].count('find') == 2


def test_mapping_key_failures_bypass_and_reach_the_client() -> None:
    wrapper, client, _backend = _wrapper()

    wrapper.schema_field_read('Asset', None, cast('dict[str, object]', {'': 1}))
    wrapper.find('Asset', [['id', 'is', {7: 'not-a-string-key'}]], ['id'])

    names = [name for name, _args, _kwargs in client._calls]
    assert names.count('schema_field_read') == 1
    assert names.count('find') == 1


def test_non_string_returned_field_bypasses_instead_of_raising() -> None:
    wrapper, client, _backend = _wrapper()

    wrapper.find('Asset', [['id', 'is', 7]], cast('list[str]', [123]))

    assert client._calls[-1][2]['fields'] == [123]


def test_normalized_site_url_is_part_of_every_exact_key() -> None:
    settings = CacheSettings(site_namespace='studio')
    backend = LocalBackend(settings, _FakeClock(), _RecordingMetrics())
    cache = QueryCache(
        settings,
        Identity('script', 'studio', 'render'),
        KeyBuilder(),
        l1=backend,
    )
    first_client = _FakeShotgun()
    second_client = _FakeShotgun()
    first = CachedShotgun(
        first_client,
        cache,
        ClientContext(
            'https://first.example',
            'studio',
            localized=False,
            convert_datetimes_to_utc=True,
            records_per_page=500,
        ),
    )
    second = CachedShotgun(
        second_client,
        cache,
        ClientContext(
            'https://second.example',
            'studio',
            localized=False,
            convert_datetimes_to_utc=True,
            records_per_page=500,
        ),
    )

    assert first.find('Asset', [], ['id']) == [{'id': 1, 'type': 'Asset'}]
    assert second.find('Asset', [], ['id']) == [{'id': 1, 'type': 'Asset'}]
    assert [name for name, _args, _kwargs in first_client._calls].count('find') == 1
    assert [name for name, _args, _kwargs in second_client._calls].count('find') == 1


@pytest.mark.parametrize('method_name', ['create', 'update'])
@pytest.mark.parametrize('field_name', ['image', 'filmstrip_image'])
def test_official_style_composite_image_write_uses_site_fences(
    method_name: str,
    field_name: str,
) -> None:
    wrapper, client, cache = _write_wrapper(_OfficialCompositeClient())
    data: dict[str, object] = {'code': 'oak', field_name: 'upload.png'}

    if method_name == 'create':
        wrapper.create('Asset', data)
    else:
        wrapper.update('Asset', 7, data)

    expected = (GenerationChange(DependencyTag.site('studio')),)
    assert cache._invalidations == [expected, expected]
    assert client._entity_rpc_committed is True


@pytest.mark.parametrize('method_name', ['create', 'update'])
@pytest.mark.parametrize('field_name', ['image', 'filmstrip_image'])
def test_official_style_partial_image_write_repeats_site_fence_and_exception(
    method_name: str,
    field_name: str,
) -> None:
    failure = _DefiniteWriteError('upload failed after entity RPC')
    wrapper, client, cache = _write_wrapper(_OfficialCompositeClient(failure))
    data: dict[str, object] = {'code': 'oak', field_name: 'upload.png'}

    def write() -> object:
        if method_name == 'create':
            return wrapper.create('Asset', data)
        return wrapper.update('Asset', 7, data)

    with pytest.raises(_DefiniteWriteError) as captured:
        write()

    expected = (GenerationChange(DependencyTag.site('studio')),)
    assert captured.value is failure
    assert client._entity_rpc_committed is True
    assert cache._invalidations == [expected, expected]


@pytest.mark.parametrize('method_name', ['create', 'batch'])
def test_malformed_create_success_uses_site_post_fence(method_name: str) -> None:
    class _MalformedCreateClient(_MutatingClient):
        def batch(self, requests: list[dict[str, object]]) -> list[dict[str, object]]:
            self._calls.append(('batch', (requests,), {}))
            return [{'id': 'unknown'}]

        def create(
            self,
            entity_type: str,
            data: dict[str, object],
            return_fields: list[str] | None = None,
        ) -> dict[str, object]:
            self._calls.append(('create', (entity_type, data, return_fields), {}))
            return {'id': 'unknown'}

    wrapper, _client, cache = _write_wrapper(_MalformedCreateClient())

    if method_name == 'create':
        wrapper.create('Asset', {'code': 'oak'})
    else:
        wrapper.batch(
            [{'request_type': 'create', 'entity_type': 'Asset', 'data': {'code': 'oak'}}]
        )

    assert cache._invalidations[1] == (GenerationChange(DependencyTag.site('studio')),)


@pytest.mark.parametrize('method_name', ['create', 'batch'])
def test_indeterminate_create_uses_site_post_fence(method_name: str) -> None:
    failure = _IndeterminateWriteError('timeout')
    wrapper, _client, cache = _write_wrapper(
        _MutatingClient(failure),
        indeterminate_write_exceptions=(_IndeterminateWriteError,),
    )

    def write() -> object:
        if method_name == 'create':
            return wrapper.create('Asset', {'code': 'oak'})
        return wrapper.batch(
            [
                {
                    'request_type': 'create',
                    'entity_type': 'Asset',
                    'data': {'code': 'oak'},
                }
            ]
        )

    with pytest.raises(_IndeterminateWriteError):
        write()

    assert cache._invalidations[1] == (GenerationChange(DependencyTag.site('studio')),)


def test_paged_finds_split_keys_by_effective_page_size() -> None:
    settings = CacheSettings(site_namespace='studio')
    client = _FakeShotgun()
    cache = QueryCache(
        settings,
        Identity('script', 'studio', 'render'),
        KeyBuilder(),
        l1=LocalBackend(settings, _FakeClock(), _RecordingMetrics()),
    )
    wide = _paged_wrapper(500, cache, client)
    narrow = _paged_wrapper(100, cache, client)

    wide.find('Asset', [], ['id'], page=2)
    narrow.find('Asset', [], ['id'], page=2)
    wide.find('Asset', [], ['id'])
    narrow.find('Asset', [], ['id'])

    assert [name for name, _args, _kwargs in client._calls].count('find') == 3

    wide.find('Asset', [], ['id'], page=True)
    narrow.find('Asset', [], ['id'], page=True)

    assert [name for name, _args, _kwargs in client._calls].count('find') == 5


def test_production_shaped_schema_is_acquired_once_per_generation() -> None:
    wrapper, client, backend = _wrapper(_FakeShotgun(_large_schema()))

    for _ in range(5):
        wrapper.find('CustomEntity07', [['sg_field_001', 'is', 'x']], ['sg_field_002'])
    names = [name for name, _args, _kwargs in client._calls]
    assert names.count('schema_read') == 1

    backend.change((GenerationChange(DependencyTag.schema('studio')),))
    wrapper.find('CustomEntity07', [['sg_field_001', 'is', 'x']], ['sg_field_002'])
    names = [name for name, _args, _kwargs in client._calls]
    assert names.count('schema_read') == 2


def test_project_scoped_schema_and_user_scoped_schedule_are_cached() -> None:
    settings = CacheSettings(
        site_namespace='studio', enabled_ttl_only_methods=('work_schedule_read',)
    )
    wrapper, client, _backend = _wrapper(_ScheduleClient(), settings=settings)
    project: dict[str, object] = {'type': 'Project', 'id': 7}
    user: dict[str, object] = {'type': 'HumanUser', 'id': 8}

    assert wrapper.schema_read(project) == wrapper.schema_read(project)
    assert wrapper.work_schedule_read('2026-09-01', '2026-09-07', project, user) == (
        wrapper.work_schedule_read('2026-09-01', '2026-09-07', project, user)
    )

    names = [name for name, _args, _kwargs in client._calls]
    assert names.count('schema_read') == 1
    assert names.count('work_schedule_read') == 1


def test_python_argument_binding_fails_before_any_cache_or_client_call() -> None:
    wrapper, client, backend = _wrapper()

    with pytest.raises(TypeError):
        wrapper.find('Asset')  # type: ignore[call-arg]

    assert client._calls == []
    assert backend.resolve(()) is not None


def test_requested_entity_exception_survives_schema_acquisition_failure() -> None:
    schema_error = LookupError('schema unavailable')
    requested_error = RuntimeError('requested find failed')

    class _DoublyFailingClient(_FakeShotgun):
        def schema_read(self, project_entity: Mapping[str, object] | None = None) -> object:
            self._calls.append(('schema_read', (project_entity,), {}))
            raise schema_error

        def find(  # noqa: PLR0913, PLR0917 - official-compatible fake
            self,
            entity_type: str,
            filters: object,
            fields: list[str] | None = None,
            order: list[dict[str, str]] | None = None,
            filter_operator: str | None = None,
            limit: int = 0,
            retired_only: bool = False,
            page: int = 0,
            include_archived_projects: bool = True,
            additional_filter_presets: object = None,
        ) -> list[dict[str, object]]:
            _ = (
                entity_type,
                filters,
                fields,
                order,
                filter_operator,
                limit,
                retired_only,
                page,
                include_archived_projects,
                additional_filter_presets,
            )
            raise requested_error

    wrapper, _client, _backend = _wrapper(_DoublyFailingClient())

    with pytest.raises(RuntimeError) as captured:
        wrapper.find('Asset', [], ['id'])

    assert captured.value is requested_error


@pytest.mark.parametrize('entity_type', ['Asset', 'EventLogEntry'])
def test_schema_failure_never_prevents_the_requested_entity_read(entity_type: str) -> None:
    expected = LookupError('schema unavailable')

    class _FailingSchemaClient(_FakeShotgun):
        def schema_read(self, project_entity: Mapping[str, object] | None = None) -> object:
            self._calls.append(('schema_read', (project_entity,), {}))
            raise expected

    wrapper, client, _backend = _wrapper(_FailingSchemaClient())

    assert wrapper.find(entity_type, [], ['id']) == [{'id': 1, 'type': entity_type}]
    names = [name for name, _args, _kwargs in client._calls]
    assert names.count('find') == 1
    assert names.count('schema_read') == (0 if entity_type == 'EventLogEntry' else 1)


def test_schema_methods_use_distinct_cached_calls() -> None:
    wrapper, client, _backend = _wrapper()

    assert wrapper.schema_read() == _SCHEMA
    assert wrapper.schema_read() == _SCHEMA
    assert wrapper.schema_field_read('Asset') == _SCHEMA['Asset']
    assert wrapper.schema_field_read('Asset') == _SCHEMA['Asset']
    assert wrapper.schema_entity_read() == {'Asset': {'name': {'value': 'Asset'}}}
    assert wrapper.schema_entity_read() == {'Asset': {'name': {'value': 'Asset'}}}
    names = [name for name, _args, _kwargs in client._calls]
    assert names.count('schema_read') == 1
    assert names.count('schema_field_read') == 1
    assert names.count('schema_entity_read') == 1


def test_scripts_only_bypasses_every_human_read() -> None:
    wrapper, client, _backend = _wrapper(
        identity=Identity('human', 'studio', 'user-7', entity_id=7)
    )

    assert wrapper.find('Asset', [], ['id']) == [{'id': 1, 'type': 'Asset'}]
    assert wrapper.find('Asset', [], ['id']) == [{'id': 2, 'type': 'Asset'}]
    names = [name for name, _args, _kwargs in client._calls]
    assert names.count('schema_read') == 0
    assert names.count('find') == 2


def test_scripts_only_human_bypass_does_not_acquire_schema() -> None:
    class _FailingSchemaClient(_FakeShotgun):
        def schema_read(self, project_entity: Mapping[str, object] | None = None) -> object:
            self._calls.append(('schema_read', (project_entity,), {}))
            raise LookupError('schema unavailable')

    wrapper, client, _backend = _wrapper(
        _FailingSchemaClient(),
        identity=Identity('human', 'studio', 'user-7', entity_id=7),
    )

    assert wrapper.find('Asset', [], ['id']) == [{'id': 1, 'type': 'Asset'}]
    names = [name for name, _args, _kwargs in client._calls]
    assert names.count('schema_read') == 0
    assert names.count('find') == 1


def test_shared_context_requires_the_configured_site_namespace() -> None:
    settings = CacheSettings(
        site_namespace='shared-studio',
        l1=L1Settings(enabled=False),
        l2=L2Settings(enabled=True),
    )
    backend = LocalBackend(settings, _FakeClock(), _RecordingMetrics())
    cache = QueryCache(
        settings,
        Identity('script', 'studio', 'render'),
        KeyBuilder(),
        l2=backend,
    )

    with pytest.raises(ValueError, match='configured site namespace'):
        CachedShotgun(
            _FakeShotgun(),
            cache,
            ClientContext(
                'https://example.shotgrid.autodesk.com',
                'wrong-site',
                localized=False,
                convert_datetimes_to_utc=True,
                records_per_page=500,
            ),
        )


def test_transient_image_age_is_applied_at_the_wrapper_boundary() -> None:
    class _ImageClient(_FakeShotgun):
        def find(  # noqa: PLR0913, PLR0917 - official-compatible fake
            self,
            entity_type: str,
            filters: object,
            fields: list[str] | None = None,
            order: list[dict[str, str]] | None = None,
            filter_operator: str | None = None,
            limit: int = 0,
            retired_only: bool = False,
            page: int = 0,
            include_archived_projects: bool = True,
            additional_filter_presets: object = None,
        ) -> list[dict[str, object]]:
            self._calls.append(
                (
                    'find',
                    (entity_type, filters),
                    {
                        'fields': fields,
                        'order': order,
                        'filter_operator': filter_operator,
                        'limit': limit,
                        'retired_only': retired_only,
                        'page': page,
                        'include_archived_projects': include_archived_projects,
                        'additional_filter_presets': additional_filter_presets,
                    },
                )
            )
            return [
                {
                    'id': 7,
                    'thumbnail': 'https://cdn.example/images/status/transient/pending',
                }
            ]

    wrapper, _client, backend = _wrapper(_ImageClient())

    wrapper.find('Asset', [], ['thumbnail'])

    deadlines = [record.expiry for record in backend._store._results.values()]
    assert any(1_000 < deadline <= 1_005 for deadline in deadlines)


def test_ttl_only_methods_are_direct_until_explicitly_enabled() -> None:
    wrapper, client, _backend = _wrapper()

    first = wrapper.summarize('Asset', [], [{'field': 'id', 'type': 'count'}])
    second = wrapper.summarize('Asset', [], [{'field': 'id', 'type': 'count'}])

    assert first != second
    assert [name for name, _args, _kwargs in client._calls].count('summarize') == 2


@pytest.mark.parametrize('bad_argument', [{'a'}, 1 + 2j, object()])
def test_unfreezable_arguments_bypass_and_reach_the_client(bad_argument: object) -> None:
    wrapper, client, _backend = _wrapper()

    wrapper.find('Asset', [['id', 'in', bad_argument]], ['id'])
    wrapper.find('Asset', [['id', 'in', bad_argument]], ['id'])
    wrapper.find_one('Asset', [['id', 'in', bad_argument]], ['id'])
    wrapper.schema_field_read('Asset', None, {'type': 'Project', 'id': bad_argument})
    wrapper.schema_field_read('Asset', None, {'type': 'Project', 'id': bad_argument})
    wrapper.summarize('Asset', [['id', 'in', bad_argument]], [{'field': 'id', 'type': 'count'}])

    names = [name for name, _args, _kwargs in client._calls]
    assert names.count('find') == 2
    assert names.count('find_one') == 1
    assert names.count('schema_field_read') == 2
    assert names.count('summarize') == 1


def test_unknown_batch_member_uses_one_site_fence_for_the_whole_batch() -> None:
    wrapper, client, cache = _write_wrapper()
    requests: list[dict[str, object]] = [
        {'request_type': 'revive', 'entity_type': 'Asset', 'entity_id': 7}
    ]

    assert wrapper.batch(requests) == [{'ok': True}]

    expected = (GenerationChange(DependencyTag.site('studio')),)
    assert cache._invalidations == [expected, expected]
    assert client._calls == [('batch', (requests,), {})]


def test_unknown_returned_field_type_with_signed_value_bypasses_population() -> None:
    schema = {
        'Asset': {
            'future_link': {
                'data_type': {'value': 'future_url'},
                'properties': {},
            }
        }
    }
    expires = int(datetime.now(tz=UTC).timestamp()) + 31
    signed_url = f'https://cdn.example/file?Expires={expires}'

    class _FutureUrlClient(_FakeShotgun):
        def find(  # noqa: PLR0913, PLR0917 - official-compatible fake
            self,
            entity_type: str,
            filters: object,
            fields: list[str] | None = None,
            order: list[dict[str, str]] | None = None,
            filter_operator: str | None = None,
            limit: int = 0,
            retired_only: bool = False,
            page: int = 0,
            include_archived_projects: bool = True,
            additional_filter_presets: object = None,
        ) -> list[dict[str, object]]:
            _ = (
                order,
                filter_operator,
                limit,
                retired_only,
                page,
                include_archived_projects,
                additional_filter_presets,
            )
            call_number = sum(name == 'find' for name, _args, _kwargs in self._calls) + 1
            self._calls.append(('find', (entity_type, filters), {'fields': fields}))
            return [{'id': call_number, 'future_link': signed_url}]

    settings = CacheSettings(
        site_namespace='studio',
        limits=LimitSettings(clock_error_bound_seconds=0),
    )
    wrapper, client, _backend = _wrapper(_FutureUrlClient(schema), settings=settings)

    first = wrapper.find('Asset', [], ['future_link'])
    second = wrapper.find('Asset', [], ['future_link'])

    assert first != second
    assert [name for name, _args, _kwargs in client._calls].count('find') == 2


def test_unproven_composite_methods_bypass_with_their_own_reason() -> None:
    settings = CacheSettings(site_namespace='studio')
    client = _FakeShotgun()
    metrics = _RecordingMetrics()
    cache = QueryCache(
        settings,
        Identity('script', 'studio', 'render'),
        KeyBuilder(),
        l1=LocalBackend(settings, _FakeClock(), metrics),
        metrics=metrics,
    )
    wrapper = CachedShotgun(
        client,
        cache,
        ClientContext(
            'https://example.shotgrid.autodesk.com',
            'studio',
            localized=False,
            convert_datetimes_to_utc=True,
            records_per_page=500,
        ),
    )

    wrapper.summarize(
        'EventLogEntry', [['id', 'greater_than', 5000]], [{'field': 'id', 'type': 'maximum'}]
    )
    wrapper.summarize(
        'EventLogEntry', [['id', 'greater_than', 5000]], [{'field': 'id', 'type': 'maximum'}]
    )

    assert [name for name, _args, _kwargs in client._calls].count('summarize') == 2
    assert ('cache_bypass', {'reason': 'composite_url_unproven'}) in metrics._increments
    assert ('cache_bypass', {'reason': 'ttl_only_method_disabled'}) not in metrics._increments


@pytest.mark.parametrize('field_name', ['thumbnail', 'sg_url', 'hidden_field'])
def test_unproven_returned_field_schema_declines_population(field_name: str) -> None:
    wrapper, client, _backend = _wrapper(_FakeShotgun(schema={}))

    wrapper.find('Asset', [], [field_name])
    wrapper.find('Asset', [], [field_name])

    assert [name for name, _args, _kwargs in client._calls].count('find') == 2


@pytest.mark.parametrize(
    'exception_types',
    [[], (_IndeterminateWriteError('instance'),), (KeyboardInterrupt,)],
)
def test_indeterminate_write_exception_configuration_is_validated(
    exception_types: object,
) -> None:
    client = _MutatingClient()
    cache = _RecordingCache()

    with pytest.raises(TypeError, match='indeterminate write exceptions'):
        CachedShotgun(
            client,
            cast('QueryCache', cache),
            ClientContext(
                'https://example.shotgrid.autodesk.com',
                'studio',
                localized=False,
                convert_datetimes_to_utc=True,
                records_per_page=500,
            ),
            indeterminate_write_exceptions=exception_types,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    'method_name',
    ['upload', 'upload_filmstrip_thumbnail', 'upload_thumbnail'],
)
def test_upload_methods_use_site_wide_attachment_fences(method_name: str) -> None:
    wrapper, _client, cache = _write_wrapper()

    getattr(wrapper, method_name)('Asset', 7, 'upload.bin')

    expected = (GenerationChange(DependencyTag.site('studio')),)
    assert cache._invalidations == [expected, expected]


def test_visibility_sensitive_types_and_fields_use_one_global_visibility_tag() -> None:
    wrapper, _client, cache = _write_wrapper()

    wrapper.update('Group', 7, {'users': [], 'owner': {'id': 42}})

    tags = tuple(change.tag for change in cache._invalidations[0])
    assert tags.count(DependencyTag.human_visibility('studio')) == 1
    assert tags[0] == DependencyTag.human_visibility('studio')
    assert cache._invalidations[0] == cache._invalidations[1]


@pytest.mark.parametrize(
    ('method', 'arguments', 'failure', 'exception_types', 'expected_fences'),
    [
        ('delete', ('Asset', 7), _DefiniteWriteError('definite'), (), 1),
        (
            'delete',
            ('Asset', 7),
            _IndeterminateWriteError('timeout'),
            (_IndeterminateWriteError,),
            2,
        ),
        (
            'update',
            ('Asset', 7, {'code': 'oak'}),
            TimeoutError('timeout'),
            (TimeoutError, ConnectionError),
            2,
        ),
        (
            'update',
            ('Asset', 7, {'code': 'oak'}),
            ConnectionError('connection'),
            (TimeoutError, ConnectionError),
            2,
        ),
        (
            'update',
            ('Asset', 7, {'code': 'oak'}),
            _DefiniteWriteError('definite'),
            (TimeoutError, ConnectionError),
            1,
        ),
    ],
    ids=[
        'delete-definite',
        'delete-indeterminate',
        'update-timeout',
        'update-connection',
        'update-definite',
    ],
)
def test_write_failures_preserve_exception_and_apply_only_required_fences(
    method: str,
    arguments: tuple[object, ...],
    failure: Exception,
    exception_types: tuple[type[Exception], ...],
    expected_fences: int,
) -> None:
    wrapper, _client, cache = _write_wrapper(
        _MutatingClient(failure),
        indeterminate_write_exceptions=exception_types,
    )
    operation = cast('Callable[..., object]', getattr(wrapper, method))

    with pytest.raises(type(failure)) as captured:
        operation(*arguments)

    assert captured.value is failure
    assert len(cache._invalidations) == expected_fences


@pytest.mark.parametrize(
    ('method_name', 'args', 'kwargs'),
    [
        ('create', ('Asset', {'code': 'oak'}, ['id', 'code']), {}),
        ('delete', ('Asset', 7), {}),
        ('revive', ('Asset', 7), {}),
        ('follow', ({'type': 'HumanUser', 'id': 3}, {'type': 'Asset', 'id': 7}), {}),
        ('unfollow', ({'type': 'HumanUser', 'id': 3}, {'type': 'Asset', 'id': 7}), {}),
        ('schema_field_create', ('Asset', 'text', 'Label', {'description': 'Asset label'}), {}),
        ('schema_field_delete', ('Asset', 'sg_label'), {}),
        (
            'schema_field_update',
            ('Asset', 'sg_label', {'name': 'Label'}, {'type': 'Project', 'id': 2}),
            {},
        ),
        (
            'share_thumbnail',
            (
                [{'type': 'Asset', 'id': 7}],
                'thumb.png',
                {'type': 'Asset', 'id': 8},
                True,
            ),
            {'dry_run': True},
        ),
        (
            'update',
            ('Asset', 7, {'tags': [{'type': 'Tag', 'id': 4}]}, {'tags': 'add'}),
            {},
        ),
        (
            'update_project_last_accessed',
            ({'type': 'Project', 'id': 2}, {'type': 'HumanUser', 'id': 3}),
            {},
        ),
        (
            'upload',
            (
                'Asset',
                7,
                'image.png',
                'sg_uploaded_movie',
                'Daily',
                'review',
                datetime(2026, 8, 30, tzinfo=UTC),
            ),
            {},
        ),
        ('upload_filmstrip_thumbnail', ('Asset', 7, 'filmstrip.png'), {'alt': 'strip'}),
        ('upload_thumbnail', ('Asset', 7, 'thumb.png'), {'alt': 'thumbnail'}),
        ('user_subscriptions_create', ([{'id': 3}],), {}),
        (
            'work_schedule_update',
            (
                '2026-08-30',
                True,
                'On set',
                {'type': 'Project', 'id': 2},
                {'type': 'HumanUser', 'id': 3},
                'day',
            ),
            {},
        ),
    ],
)
def test_every_planned_mutator_has_pre_and_post_fences(
    method_name: str,
    args: tuple[object, ...],
    kwargs: dict[str, object],
) -> None:
    wrapper, client, cache = _write_wrapper()

    getattr(wrapper, method_name)(*args, **kwargs)

    assert len(cache._invalidations) == 2
    if method_name == 'create':
        assert cache._invalidations[1][:-1] == cache._invalidations[0]
    else:
        assert cache._invalidations[0] == cache._invalidations[1]
    assert client._calls == [(method_name, args, kwargs)]


@pytest.mark.parametrize(
    'namespace', [' padded', 'padded ', 'non-ascii-\N{LATIN SMALL LETTER E WITH ACUTE}']
)
def test_context_rejects_invalid_namespace_before_any_read(namespace: str) -> None:
    with pytest.raises(ValueError, match='ASCII without outer whitespace'):
        ClientContext(
            'https://example.shotgrid.autodesk.com',
            namespace,
            localized=False,
            convert_datetimes_to_utc=True,
            records_per_page=500,
        )
