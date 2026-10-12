import asyncio
from collections.abc import Awaitable, Callable, Coroutine, Generator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from time import monotonic
from typing import Any, cast

import pytest

import sg_query_cache._async_cache as cache_module
import sg_query_cache._async_client as client_module
import sg_query_cache._async_support as support_module
from sg_query_cache import (
    AsyncCachedShotgun,
    AsyncLocalBackend,
    AsyncQueryCache,
    CacheSettings,
    ClientContext,
    Identity,
)
from sg_query_cache._async_backend import AsyncCacheBackend
from sg_query_cache._backend import BackendUnavailable
from sg_query_cache._cache_common import PopulationCallback
from sg_query_cache._config import L1Settings, L2Settings
from sg_query_cache._model import (
    CacheHit,
    DependencyPlan,
    DependencyTag,
    ExactCall,
    FillLease,
    GenerationChange,
    GenerationVector,
)
from sg_query_cache.valkey_async import AsyncValkeyBackend
from tests._async_fakes import _RecordingMetrics

_BATCH_REQUESTS: list[dict[str, object]] = [
    {'request_type': 'create', 'entity_type': 'Asset', 'data': {}},
    {'request_type': 'delete', 'entity_type': 'Asset', 'entity_id': 7},
]
_BATCH_RESULT: list[dict[str, object] | bool] = [{'id': 8, 'type': 'Asset'}, True]
_CONTEXT = ClientContext(
    'https://example.shotgrid.autodesk.com',
    'production',
    localized=False,
    convert_datetimes_to_utc=True,
    records_per_page=500,
)
_CREATED_AT = datetime(2026, 1, 2, tzinfo=UTC)
_ENTITY = {'type': 'Asset', 'id': 7}
_IDENTITY = Identity('script', 'tests', 'async-wrapper')
_PROJECT = {'type': 'Project', 'id': 2}
_SITE_CHANGE = (GenerationChange(DependencyTag.site('production')),)
_USER = {'type': 'HumanUser', 'id': 3}


class _GateBackend:
    def __init__(self, backend: AsyncCacheBackend) -> None:
        self._backend = backend
        self._block_call: int | None = None
        self._change_calls: list[tuple[GenerationChange, ...]] = []
        self._entered = asyncio.Event()
        self._failure: Exception | None = None
        self._proceed = asyncio.Event()

    async def change(self, changes: tuple[GenerationChange, ...]) -> None:
        self._change_calls.append(changes)
        if len(self._change_calls) == self._block_call:
            self._entered.set()
            await self._proceed.wait()
        if self._failure is not None:
            raise self._failure
        await self._backend.change(changes)

    async def close(self) -> None:
        await self._backend.close()

    async def commit(self, lease: FillLease, payload: bytes, remaining_seconds: float) -> bool:
        return await self._backend.commit(lease, payload, remaining_seconds)

    async def flush_site(self, site_namespace: str, reason: str) -> None:
        await self._backend.flush_site(site_namespace, reason)

    async def lookup(self, key: str) -> CacheHit | None:
        return await self._backend.lookup(key)

    async def release(self, lease: FillLease) -> bool:
        return await self._backend.release(lease)

    async def reserve(self, key: str, lease_seconds: float) -> FillLease | None:
        return await self._backend.reserve(key, lease_seconds)

    async def resolve(self, tags: tuple[DependencyTag, ...]) -> GenerationVector:
        return await self._backend.resolve(tags)


class _RawClient:
    def __init__(self, results: Mapping[str, object] | None = None) -> None:
        self._cancellations: list[asyncio.CancelledError] = []
        self._calls: list[tuple[str, tuple[object, ...], dict[str, object]]] = []
        self._gates: dict[str, tuple[asyncio.Event, asyncio.Event]] = {}
        self._results = dict(results or {})

    def __getattr__(self, name: str) -> object:
        async def invoke(*args: object, **kwargs: object) -> object:
            self._calls.append((name, args, kwargs))
            if gate := self._gates.get(name):
                gate[0].set()
                try:
                    await gate[1].wait()
                except asyncio.CancelledError as error:
                    self._cancellations.append(error)
                    raise
            result = self._results.get(name)
            if isinstance(result, BaseException):
                raise result
            return result

        return invoke


class _RecordingCache:
    def __init__(self, *, bypass: bool = False) -> None:
        self._identity = _IDENTITY
        self._settings = CacheSettings()
        self._admissions = 0
        self.bypass_active = bypass
        self._closed = 0
        self._invalidations: list[tuple[GenerationChange, ...]] = []
        self._reads: list[tuple[ExactCall, DependencyPlan, Mapping[str, object], object]] = []

    @contextmanager
    def admit(self) -> Generator[object]:
        self._admissions += 1
        yield object()

    @contextmanager
    def bypass(self) -> Generator[None]:
        previous = self.bypass_active
        self.bypass_active = True
        try:
            yield
        finally:
            self.bypass_active = previous

    async def close(self) -> None:
        self._closed += 1

    async def invalidate_admitted(
        self, changes: tuple[GenerationChange, ...], *, token: object
    ) -> bool:
        _ = token
        self._invalidations.append(changes)
        return True

    async def read_admitted(
        self,
        call: ExactCall,
        plan: DependencyPlan,
        fetch: Callable[[], Awaitable[object]],
        client_options: Mapping[str, object],
        _per_call_age: float | None = None,
        *,
        token: object,
        _population_decision: PopulationCallback | None = None,
    ) -> object:
        _ = _population_decision
        self._reads.append((call, plan, client_options, token))
        return await fetch()

    async def read_parsed_admitted(
        self,
        *arguments: *tuple[
            ExactCall,
            DependencyPlan,
            Callable[[], Awaitable[object]],
            Mapping[str, object],
            Callable[[object], object],
        ],
        token: object,
    ) -> object:
        call, plan, fetch, client_options, parse = arguments
        self._reads.append((call, plan, client_options, token))
        return parse(await fetch())


@dataclass(frozen=True)
class _ForwardCase:
    _name: str
    _args: tuple[object, ...]
    _kwargs: Mapping[str, object]
    _forwarded_args: tuple[object, ...]
    _forwarded_kwargs: Mapping[str, object]
    _result: object


def _forward_cases() -> tuple[_ForwardCase, ...]:
    return (
        _ForwardCase(
            'activity_stream_read',
            ('Asset', 7),
            {'limit': 4},
            ('Asset', 7, None, None, None, 4),
            {},
            {},
        ),
        _ForwardCase(
            'batch',
            (_BATCH_REQUESTS,),
            {},
            (_BATCH_REQUESTS,),
            {},
            _BATCH_RESULT,
        ),
        _ForwardCase(
            'create',
            ('Asset', {'code': 'oak'}),
            {},
            ('Asset', {'code': 'oak'}, None),
            {},
            {'type': 'Asset', 'id': 8},
        ),
        _ForwardCase('delete', ('Asset', 7), {}, ('Asset', 7), {}, _result=True),
        _ForwardCase(
            'find',
            ('EventLogEntry', []),
            {'page': 2},
            ('EventLogEntry', [], None, None, None, 0, False, 2, True, None),
            {},
            [{'id': 7}],
        ),
        _ForwardCase(
            'find_one',
            ('EventLogEntry', []),
            {'retired_only': True},
            ('EventLogEntry', [], None, None, None, True, True, None),
            {},
            {'id': 7},
        ),
        _ForwardCase('follow', (_USER, _ENTITY), {}, (_USER, _ENTITY), {}, {}),
        _ForwardCase('note_thread_read', (7,), {}, (7, None), {}, []),
        _ForwardCase('preferences_read', (), {}, (None,), {}, {}),
        _ForwardCase('revive', ('Asset', 7), {}, ('Asset', 7), {}, _result=True),
        _ForwardCase('schema_entity_read', (), {}, (None,), {}, {}),
        _ForwardCase(
            'schema_field_create',
            ('Asset', 'text', 'Label'),
            {},
            ('Asset', 'text', 'Label', None),
            {},
            'sg_label',
        ),
        _ForwardCase(
            'schema_field_delete',
            ('Asset', 'sg_label'),
            {},
            ('Asset', 'sg_label'),
            {},
            _result=True,
        ),
        _ForwardCase('schema_field_read', ('Asset',), {}, ('Asset', None, None), {}, {}),
        _ForwardCase(
            'schema_field_update',
            ('Asset', 'sg_label', {'name': 'Label'}),
            {},
            ('Asset', 'sg_label', {'name': 'Label'}, None),
            {},
            _result=True,
        ),
        _ForwardCase('schema_read', (), {}, (None,), {}, {}),
        _ForwardCase(
            'share_thumbnail',
            ([_ENTITY],),
            {'source_entity': _ENTITY, 'filmstrip_thumbnail': True, 'foo': 'bar'},
            ([_ENTITY], None, _ENTITY, True),
            {'foo': 'bar'},
            9,
        ),
        _ForwardCase(
            'summarize',
            ('Asset', [], [{'field': 'id', 'type': 'count'}]),
            {},
            ('Asset', [], [{'field': 'id', 'type': 'count'}], None, None, True),
            {},
            {},
        ),
        _ForwardCase(
            'text_search',
            ('oak', {'Asset': []}),
            {},
            ('oak', {'Asset': []}, None, None),
            {},
            {},
        ),
        _ForwardCase('unfollow', (_USER, _ENTITY), {}, (_USER, _ENTITY), {}, {}),
        _ForwardCase(
            'update',
            ('Asset', 7, {'code': 'oak'}),
            {},
            ('Asset', 7, {'code': 'oak'}, None),
            {},
            {'type': 'Asset', 'id': 7},
        ),
        _ForwardCase('update_project_last_accessed', (_PROJECT,), {}, (_PROJECT, None), {}, None),
        _ForwardCase(
            'upload',
            ('Asset', 7, '/fixtures/file'),
            {'created_at': _CREATED_AT},
            ('Asset', 7, '/fixtures/file', None, None, None, _CREATED_AT),
            {},
            9,
        ),
        _ForwardCase(
            'upload_filmstrip_thumbnail',
            ('Asset', 7, '/fixtures/file'),
            {'foo': 'bar'},
            ('Asset', 7, '/fixtures/file'),
            {'foo': 'bar'},
            9,
        ),
        _ForwardCase(
            'upload_thumbnail',
            ('Asset', 7, '/fixtures/file'),
            {'foo': 'bar'},
            ('Asset', 7, '/fixtures/file'),
            {'foo': 'bar'},
            9,
        ),
        _ForwardCase('user_subscriptions_create', ([_USER],), {}, ([_USER],), {}, _result=True),
        _ForwardCase(
            'work_schedule_read',
            ('2026-01-01', '2026-01-02'),
            {},
            ('2026-01-01', '2026-01-02', None, None),
            {},
            {},
        ),
        _ForwardCase(
            'work_schedule_update',
            ('2026-01-01', True),
            {},
            ('2026-01-01', True, None, None, None, None),
            {},
            {},
        ),
    )


def _recording_wrapper(
    raw: _RawClient, cache: _RecordingCache | None = None
) -> tuple[AsyncCachedShotgun, _RecordingCache]:
    selected = cache or _RecordingCache()
    return AsyncCachedShotgun(raw, cast('AsyncQueryCache', selected), _CONTEXT), selected


def _write_failure(method_name: str, scenario_name: str) -> BaseException:
    if scenario_name.endswith('raw_cancellation'):
        return asyncio.CancelledError('raw dispatch cancelled')
    if method_name == 'update':
        return RuntimeError('composite definite failure')
    return OSError('indeterminate')


def test_configured_open_settles_backend_close_cancelled_before_entry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        settings = CacheSettings(l1=L1Settings(enabled=False), l2=L2Settings(enabled=True))
        manager = AsyncCachedShotgun.open(
            _RawClient(),
            settings,
            _IDENTITY,
            _CONTEXT,
            environ={
                settings.l2.valkey.metadata.password_env: 'metadata',
                settings.l2.valkey.result.password_env: 'result',
            },
        )
        wrapper = await manager.__aenter__()
        shared = cast('AsyncValkeyBackend', wrapper._cache._l2)
        original = cache_module.create_task
        interrupted: list[asyncio.Task[object]] = []

        def create(
            coroutine: Coroutine[Any, Any, support_module.CleanupOutcome[object]],
        ) -> asyncio.Task[support_module.CleanupOutcome[object]]:
            frame = cast('Any', coroutine).cr_frame
            owner = frame.f_locals.get('self') if frame is not None else None
            task = original(coroutine)
            if owner is wrapper._cache:
                task.cancel('before core close body')
                interrupted.append(cast('asyncio.Task[object]', task))
            return task

        monkeypatch.setattr(cache_module, 'create_task', create)
        try:
            with pytest.raises(BackendUnavailable):
                await manager.__aexit__(None, None, None)
            assert len(interrupted) == 1
            assert shared._lifecycle._admission.closed
        finally:
            monkeypatch.setattr(cache_module, 'create_task', original)
            await shared.close()

    asyncio.run(scenario())


@pytest.mark.parametrize('case', _forward_cases(), ids=lambda case: case._name)
def test_explicit_methods_forward_raw_arguments_exactly(case: _ForwardCase) -> None:
    async def scenario() -> None:
        raw = _RawClient({case._name: case._result})
        wrapper, cache = _recording_wrapper(raw)

        result = await getattr(wrapper, case._name)(*case._args, **case._kwargs)

        assert result == case._result
        assert raw._calls == [
            (case._name, case._forwarded_args, dict(case._forwarded_kwargs)),
        ]
        assert cache._admissions == 1

    asyncio.run(scenario())


@dataclass(frozen=True)
class _WriteCase:
    _name: str
    _args: tuple[object, ...]
    _result: object


def _expected_dispatch(case: _WriteCase) -> tuple[object, ...]:
    if case._name == 'share_thumbnail':
        return (*case._args, None, None, False)
    omitted_count = {
        'create': 1,
        'schema_field_create': 1,
        'schema_field_update': 1,
        'update': 1,
        'update_project_last_accessed': 1,
        'upload': 4,
        'work_schedule_update': 4,
    }.get(case._name, 0)
    return (*case._args, *((None,) * omitted_count))


def _expected_pre_changes(case: _WriteCase) -> tuple[GenerationChange, ...]:
    entity = GenerationChange(DependencyTag.entity('production', 'Asset', 7))
    membership = GenerationChange(DependencyTag.type_membership('production', 'Asset'))
    type_value = GenerationChange(DependencyTag.type_value('production', 'Asset'))
    field = GenerationChange(DependencyTag.type_field('production', 'Asset', 'code'))
    if case._name == 'batch':
        return (membership, type_value, entity)
    if case._name == 'create':
        return (membership, type_value, field)
    if case._name in {'delete', 'revive'}:
        return (entity, membership, type_value)
    if case._name.startswith('schema_field_'):
        return (GenerationChange(DependencyTag.schema('production')),)
    if case._name == 'update':
        return (entity, type_value, field, membership)
    return _SITE_CHANGE


def _expected_last_changes(case: _WriteCase, phase: str) -> tuple[GenerationChange, ...]:
    changes = _expected_pre_changes(case)
    if case._name not in {'batch', 'create'} or phase == 'pre':
        return changes
    if phase == 'dispatch':
        return _SITE_CHANGE
    created = GenerationChange(DependencyTag.entity('production', 'Asset', 8))
    return (*changes, created)


def _write_cases() -> tuple[_WriteCase, ...]:
    return tuple(
        _WriteCase(case._name, case._args, case._result)
        for case in _forward_cases()
        if case._name
        in {
            'batch',
            'create',
            'delete',
            'follow',
            'revive',
            'schema_field_create',
            'schema_field_delete',
            'schema_field_update',
            'share_thumbnail',
            'unfollow',
            'update',
            'update_project_last_accessed',
            'upload',
            'upload_filmstrip_thumbnail',
            'upload_thumbnail',
            'user_subscriptions_create',
            'work_schedule_update',
        }
    )


def _write_fixture(
    case: _WriteCase,
) -> tuple[
    AsyncCachedShotgun,
    _RawClient,
    AsyncQueryCache,
    _GateBackend,
    _GateBackend,
]:
    settings = CacheSettings(l1=L1Settings(), l2=L2Settings(enabled=True))
    metrics = _RecordingMetrics()
    l1 = _GateBackend(AsyncLocalBackend(settings, monotonic, metrics))
    l2 = _GateBackend(AsyncLocalBackend(settings, monotonic, metrics))
    cache = AsyncQueryCache(settings, _IDENTITY, l1=l1, l2=l2, metrics=metrics)
    raw = _RawClient({case._name: case._result})
    wrapper = AsyncCachedShotgun(raw, cache, _CONTEXT)
    return wrapper, raw, cache, l1, l2


@pytest.mark.parametrize('case', _write_cases(), ids=lambda case: case._name)
@pytest.mark.parametrize('phase', ['pre', 'dispatch', 'post'])
def test_every_write_cancellation_settles_both_layers_without_retry(
    case: _WriteCase, phase: str
) -> None:
    async def scenario() -> None:
        wrapper, raw, cache, l1, l2 = _write_fixture(case)
        raw_entered = asyncio.Event()
        raw_proceed = asyncio.Event()
        if phase == 'pre':
            l2._block_call = 1
        elif phase == 'dispatch':
            raw._gates[case._name] = (raw_entered, raw_proceed)
        else:
            l2._block_call = 2

        task = asyncio.create_task(getattr(wrapper, case._name)(*case._args))
        if phase == 'dispatch':
            await raw_entered.wait()
        else:
            await l2._entered.wait()
        task.cancel()
        if phase == 'dispatch':
            raw_proceed.set()
        else:
            l2._proceed.set()
        with pytest.raises(asyncio.CancelledError) as captured:
            await task

        expected_dispatches = 0 if phase == 'pre' else 1
        expected_fences = 1 if phase == 'pre' else 2
        assert [name for name, _args, _kwargs in raw._calls].count(
            case._name
        ) == expected_dispatches
        if expected_dispatches:
            assert raw._calls == [(case._name, _expected_dispatch(case), {})]
        if phase == 'dispatch':
            assert captured.value is raw._cancellations[0]
        assert len(l2._change_calls) == expected_fences
        assert len(l1._change_calls) == expected_fences
        assert l2._change_calls == l1._change_calls
        assert l2._change_calls[0] == _expected_pre_changes(case)
        assert l2._change_calls[-1] == _expected_last_changes(case, phase)
        assert cache._fences['l2']._retained == ()
        assert cache._fences['l1']._retained == ()
        await cache.close()

    asyncio.run(scenario())


def test_failed_l2_fence_is_retained_before_acknowledged_l1_dispatch() -> None:
    async def scenario() -> None:
        case = next(case for case in _write_cases() if case._name == 'delete')
        wrapper, raw, cache, l1, l2 = _write_fixture(case)
        l2._failure = BackendUnavailable('shared cache unavailable')

        assert await wrapper.delete('Asset', 7)

        expected = _expected_pre_changes(case)
        assert raw._calls == [('delete', ('Asset', 7), {})]
        assert l2._change_calls == [expected, expected]
        assert l1._change_calls == [expected, expected]
        assert cache._fences['l2']._retained == expected
        assert cache._fences['l1']._retained == ()
        await cache.close()

    asyncio.run(scenario())


def test_find_forwards_every_omitted_default_and_keeps_paged_key_options() -> None:
    async def scenario() -> None:
        raw = _RawClient({'find': [{'id': 7}]})
        wrapper, cache = _recording_wrapper(raw)

        assert await wrapper.find('EventLogEntry', [], page=2) == [{'id': 7}]

        call = cache._reads[-1][0]
        assert call.arguments[-2:] == (
            ('additional_filter_presets', None),
            ('site_url', 'https://example.shotgrid.autodesk.com'),
        )
        assert cache._reads[-1][2] == {
            'localized': False,
            'convert_datetimes_to_utc': True,
            'records_per_page': 500,
        }
        assert raw._calls == [
            ('find', ('EventLogEntry', [], None, None, None, 0, False, 2, True, None), {}),
        ]

    asyncio.run(scenario())


def test_injected_cache_and_raw_client_remain_caller_owned() -> None:
    async def scenario() -> None:
        raw = _RawClient()
        wrapper, cache = _recording_wrapper(raw)

        async with wrapper as entered:
            assert entered is wrapper

        assert wrapper.raw_client is raw
        assert cache._closed == 0

    asyncio.run(scenario())


def test_local_open_owns_only_its_constructed_cache() -> None:
    async def scenario() -> None:
        raw = _RawClient({'preferences_read': {}})

        async with AsyncCachedShotgun.open(
            raw,
            CacheSettings(),
            _IDENTITY,
            _CONTEXT,
        ) as wrapper:
            cache = wrapper._cache
            assert await wrapper.preferences_read() == {}

        assert wrapper.raw_client is raw
        with pytest.raises(RuntimeError, match='closed'):
            await cache.read(
                cast('Any', object()),
                cast('Any', object()),
                cast('Any', object()),
                {},
            )

    asyncio.run(scenario())


def test_open_cleanup_cancellation_precedes_body_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        close_entered = asyncio.Event()
        close_proceed = asyncio.Event()

        class _BlockingCache(_RecordingCache):
            def __init__(self, *_args: object, **_kwargs: object) -> None:
                super().__init__()

            async def close(self) -> None:
                close_entered.set()
                await close_proceed.wait()

        monkeypatch.setattr(client_module, 'AsyncQueryCache', _BlockingCache)

        async def use_factory() -> None:
            async with AsyncCachedShotgun.open(_RawClient(), CacheSettings(), _IDENTITY, _CONTEXT):
                raise RuntimeError('body failed')

        task = asyncio.create_task(use_factory())
        await close_entered.wait()
        task.cancel()
        close_proceed.set()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())


def test_open_constructs_configured_l2_and_owns_its_close(monkeypatch: pytest.MonkeyPatch) -> None:
    async def scenario() -> None:
        settings = CacheSettings(l1=L1Settings(), l2=L2Settings(enabled=True))
        shared = AsyncLocalBackend(settings, monotonic, _RecordingMetrics())
        constructed: list[object] = []

        async def create(supplied: object, environ: object, metrics: object) -> AsyncLocalBackend:
            constructed.extend((supplied, environ, metrics))
            return shared

        monkeypatch.setattr(AsyncValkeyBackend, 'from_settings', create)
        manager = AsyncCachedShotgun.open(_RawClient(), settings, _IDENTITY, _CONTEXT, environ={})

        async with manager:
            assert constructed[0] is settings
        with pytest.raises(RuntimeError, match='closed'):
            await shared.lookup('key')

    asyncio.run(scenario())


def test_open_exceptional_exit_preserves_direct_admission_owner_guard() -> None:
    async def scenario() -> None:
        metrics = _RecordingMetrics()
        manager = AsyncCachedShotgun.open(
            _RawClient(), CacheSettings(), _IDENTITY, _CONTEXT, metrics=metrics
        )
        wrapper = await manager.__aenter__()
        original = ValueError('body failed while admitted')
        try:
            with wrapper._cache.admit():
                assert not await manager.__aexit__(ValueError, original, None)
                assert not wrapper._cache._admission.closed
                # The direct guard rejected close before changing admission.
                with wrapper._cache.admit():
                    pass
            assert (
                'backend_failure',
                None,
                {'backend_operation': 'factory_close', 'outcome': 'failure'},
            ) in metrics._events
        finally:
            await wrapper._cache.close()
        assert wrapper._cache._admission.closed

    asyncio.run(scenario())


def test_open_forwards_indeterminate_exceptions_and_exposes_close_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        close_failure = ValueError('close failed')

        class _FailingCache(_RecordingCache):
            def __init__(self, *_args: object, **_kwargs: object) -> None:
                super().__init__()

            async def close(self) -> None:
                raise close_failure

        monkeypatch.setattr(client_module, 'AsyncQueryCache', _FailingCache)
        manager = AsyncCachedShotgun.open(
            _RawClient(),
            CacheSettings(),
            _IDENTITY,
            _CONTEXT,
            indeterminate_write_exceptions=(OSError,),
        )
        with pytest.raises(ValueError, match='close failed') as captured:
            async with manager as wrapper:
                assert wrapper._indeterminate_write_exceptions == (OSError,)
        assert captured.value is close_failure

    asyncio.run(scenario())


def test_open_preserves_body_exception_over_cache_close_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        body_failure = RuntimeError('body failed')
        close_failure = ValueError('close failed')

        class _FailingCache(_RecordingCache):
            def __init__(self, *_args: object, **_kwargs: object) -> None:
                super().__init__()

            async def close(self) -> None:
                raise close_failure

        monkeypatch.setattr(client_module, 'AsyncQueryCache', _FailingCache)
        manager = AsyncCachedShotgun.open(_RawClient(), CacheSettings(), _IDENTITY, _CONTEXT)
        with pytest.raises(RuntimeError, match='body failed') as captured:
            async with manager:
                raise body_failure
        assert captured.value is body_failure

    asyncio.run(scenario())


def test_partial_factory_construction_joins_backend_close_and_preserves_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        construction_failure = RuntimeError('cache construction failed')

        class _PartialBackend:
            def __init__(self, *_args: object, **_kwargs: object) -> None:
                self._closed = False

            async def close(self) -> None:
                self._closed = True
                raise ValueError('lower-priority close failure')

        backend = _PartialBackend()

        def construct_backend(*_args: object, **_kwargs: object) -> _PartialBackend:
            return backend

        def fail_cache(*_args: object, **_kwargs: object) -> None:
            raise construction_failure

        monkeypatch.setattr(client_module, 'AsyncLocalBackend', construct_backend)
        monkeypatch.setattr(client_module, 'AsyncQueryCache', fail_cache)
        manager = AsyncCachedShotgun.open(_RawClient(), CacheSettings(), _IDENTITY, _CONTEXT)

        with pytest.raises(RuntimeError, match='cache construction failed') as captured:
            async with manager:
                raise AssertionError('partial factory entered')

        assert captured.value is construction_failure
        assert backend._closed

    asyncio.run(scenario())


@pytest.mark.parametrize(
    'scenario_name',
    [
        'prestart_raw_cancellation',
        'prestart_indeterminate',
        'prestart_repeat',
    ],
)
def test_prestart_cleanup_failure_retains_both_fences(
    monkeypatch: pytest.MonkeyPatch,
    scenario_name: str,
) -> None:
    def cancel_next_cleanup_task() -> None:
        original_create_task = support_module.create_task

        def cancel_created(coroutine: Coroutine[object, object, object]) -> asyncio.Task[object]:
            monkeypatch.setattr(support_module, 'create_task', original_create_task)
            task = asyncio.create_task(coroutine)
            task.cancel()
            return task

        monkeypatch.setattr(support_module, 'create_task', cancel_created)

    async def scenario() -> None:
        method_name = 'update' if scenario_name.endswith('repeat') else 'delete'
        case = next(case for case in _write_cases() if case._name == method_name)
        wrapper, _raw, cache, l1, l2 = _write_fixture(case)
        arguments: tuple[object, ...] = (
            ('Asset', 7, {'image': '/fixtures/image.jpg'})
            if method_name == 'update'
            else ('Asset', 7)
        )
        failure = _write_failure(method_name, scenario_name)

        class _Raw:
            _calls = 0

            def __getattr__(self, _name: str) -> object:
                async def invoke(*_args: object, **_kwargs: object) -> object:
                    self._calls += 1
                    cancel_next_cleanup_task()
                    raise failure

                return invoke

        prestart_raw = _Raw()
        wrapper = AsyncCachedShotgun(
            prestart_raw,
            cache,
            _CONTEXT,
            indeterminate_write_exceptions=(OSError,) if isinstance(failure, OSError) else (),
        )
        with pytest.raises(type(failure)) as captured:
            await getattr(wrapper, method_name)(*arguments)

        expected = _SITE_CHANGE if method_name == 'update' else _expected_pre_changes(case)
        assert captured.value is failure
        assert prestart_raw._calls == 1
        assert l2._change_calls == [expected]
        assert l1._change_calls == [expected]
        assert cache._fences['l2']._retained == expected
        assert cache._fences['l1']._retained == expected
        await cache.close()

    asyncio.run(scenario())


@pytest.mark.parametrize('cancel_mode', ['ordinary', 'stale', 'pending', 'during'])
def test_partial_cleanup_preserves_cancellation_provenance(cancel_mode: str) -> None:
    async def scenario() -> None:
        caller = asyncio.current_task()
        assert caller is not None
        attempts: list[str] = []
        original = ValueError('original body')
        metrics = _RecordingMetrics()
        if cancel_mode in {'stale', 'pending'}:
            caller.cancel(cancel_mode)
            if cancel_mode == 'stale':
                with pytest.raises(asyncio.CancelledError, match='stale'):
                    await asyncio.sleep(0)

        class _IndependentClose(_RecordingCache):
            async def close(self) -> None:
                attempts.append('independent')
                child = asyncio.create_task(asyncio.sleep(0))
                child.cancel('independent resource')
                await child

        class _LaterClose(_RecordingCache):
            async def close(self) -> None:
                attempts.append('later')
                if cancel_mode == 'during':
                    caller.cancel('during')
                    await asyncio.sleep(0)

        async def failed_body() -> None:
            await client_module._close_partial(
                (
                    cast('AsyncQueryCache', _IndependentClose()),
                    cast('AsyncQueryCache', _LaterClose()),
                ),
                original,
                metrics,
            )
            raise original

        if cancel_mode in {'pending', 'during'}:
            with pytest.raises(asyncio.CancelledError, match=cancel_mode):
                await failed_body()
        else:
            with pytest.raises(ValueError, match='original body') as raised:
                await failed_body()
            assert raised.value is original
        assert attempts == ['independent', 'later']
        assert caller.cancelling() == int(cancel_mode != 'ordinary')
        assert (
            'backend_failure',
            None,
            {'backend_operation': 'factory_close', 'outcome': 'failure'},
        ) in metrics._events

    asyncio.run(scenario())


def test_read_captures_context_before_schema_acquisition() -> None:
    async def scenario() -> None:
        entered = asyncio.Event()
        proceed = asyncio.Event()
        raw = _RawClient({'schema_read': {}, 'find': [{'id': 7}]})
        raw._gates['schema_read'] = (entered, proceed)
        wrapper, cache = _recording_wrapper(raw)
        replacement = replace(_CONTEXT, site_url='https://replacement.invalid')

        read = asyncio.create_task(wrapper.find('Asset', [], ['id']))
        await entered.wait()
        await wrapper.replace_context(replacement)
        proceed.set()
        assert await read == [{'id': 7}]

        assert wrapper.context == replacement
        entity_call = cache._reads[-1][0]
        assert entity_call.arguments[-1] == (
            'site_url',
            'https://example.shotgrid.autodesk.com',
        )
        assert cache._invalidations == [_SITE_CHANGE]

    asyncio.run(scenario())


def test_schema_failure_is_a_read_bypass_but_raw_read_failure_is_preserved() -> None:
    async def scenario() -> None:
        failure = RuntimeError('raw read failed')
        raw = _RawClient({'schema_read': ValueError('schema unavailable'), 'find': failure})
        wrapper, _cache = _recording_wrapper(raw)

        with pytest.raises(RuntimeError, match='raw read failed') as captured:
            await wrapper.find('Asset', [], ['id'])

        assert captured.value is failure
        assert [name for name, _args, _kwargs in raw._calls] == ['schema_read', 'find']

    asyncio.run(scenario())


def test_scoped_and_unnormalizable_bypass_avoid_schema_acquisition() -> None:
    async def scenario() -> None:
        raw = _RawClient({'find': [{'id': 7}]})
        wrapper, cache = _recording_wrapper(raw)

        with wrapper.bypass():
            assert await wrapper.find('Asset', [], ['id']) == [{'id': 7}]
        assert await wrapper.find('Asset', object(), ['id']) == [{'id': 7}]

        assert [name for name, _args, _kwargs in raw._calls] == ['find', 'find']
        assert [item[1].reason for item in cache._reads] == [
            'scoped_bypass',
            'unsupported_argument',
        ]

    asyncio.run(scenario())


def test_scoped_schema_and_schedule_hit_local_cache() -> None:
    async def scenario() -> None:
        settings = CacheSettings(enabled_ttl_only_methods=('work_schedule_read',))
        metrics = _RecordingMetrics()
        backend = AsyncLocalBackend(settings, monotonic, metrics)
        cache = AsyncQueryCache(settings, _IDENTITY, l1=backend, metrics=metrics)
        raw = _RawClient({'schema_read': {}, 'work_schedule_read': {'working': True}})
        wrapper = AsyncCachedShotgun(raw, cache, _CONTEXT)
        project: dict[str, object] = {'type': 'Project', 'id': 7}
        user: dict[str, object] = {'type': 'HumanUser', 'id': 8}
        try:
            assert await wrapper.schema_read(project) == await wrapper.schema_read(project)
            assert await wrapper.work_schedule_read(
                '2026-09-01', '2026-09-07', project, user
            ) == await wrapper.work_schedule_read('2026-09-01', '2026-09-07', project, user)
            names = [name for name, _args, _kwargs in raw._calls]
            assert names.count('schema_read') == 1
            assert names.count('work_schedule_read') == 1
        finally:
            await cache.close()

    asyncio.run(scenario())


@pytest.mark.parametrize('indeterminate', [False, True])
def test_write_exception_classification_preserves_original_failure(indeterminate: bool) -> None:
    async def scenario() -> None:
        failure: BaseException = (
            OSError('indeterminate') if indeterminate else RuntimeError('definite')
        )
        raw = _RawClient({'update': failure})
        recording_cache = _RecordingCache()
        wrapper = AsyncCachedShotgun(
            raw,
            cast('AsyncQueryCache', recording_cache),
            _CONTEXT,
            indeterminate_write_exceptions=(OSError,),
        )

        with pytest.raises(type(failure)) as captured:
            await wrapper.update('Asset', 7, {'code': 'oak'})

        assert captured.value is failure
        assert len(recording_cache._invalidations) == (2 if indeterminate else 1)
        assert len(raw._calls) == 1

    asyncio.run(scenario())


@pytest.mark.parametrize(
    'scenario_name',
    ['later_raw_cancellation', 'later_indeterminate', 'later_repeat'],
)
def test_later_cleanup_cancellation_preserves_dispatched_write(scenario_name: str) -> None:
    async def scenario() -> None:
        method_name = 'update' if scenario_name.endswith('repeat') else 'delete'
        arguments: tuple[object, ...] = (
            ('Asset', 7, {'image': '/fixtures/image.jpg'})
            if method_name == 'update'
            else ('Asset', 7)
        )
        failure = _write_failure(method_name, scenario_name)
        indeterminate_exceptions = (OSError,) if isinstance(failure, OSError) else ()
        case = next(item for item in _write_cases() if item._name == method_name)
        _unused, raw, cache, l1, l2 = _write_fixture(case)
        raw._results[method_name] = failure
        l2._block_call = 2
        wrapper = AsyncCachedShotgun(
            raw,
            cache,
            _CONTEXT,
            indeterminate_write_exceptions=indeterminate_exceptions,
        )

        task = asyncio.create_task(getattr(wrapper, method_name)(*arguments))
        await l2._entered.wait()
        task.cancel()
        l2._proceed.set()
        with pytest.raises(asyncio.CancelledError) as captured:
            await task

        if isinstance(failure, asyncio.CancelledError):
            assert captured.value is failure
        assert len(raw._calls) == 1
        assert len(l2._change_calls) == 2
        assert len(l1._change_calls) == 2
        assert cache._fences['l2']._retained == ()
        assert cache._fences['l1']._retained == ()
        await cache.close()

    asyncio.run(scenario())


@pytest.mark.parametrize('operation_name', ['direct', 'schema', 'write'])
def test_close_waits_for_each_whole_admitted_wrapper_operation(operation_name: str) -> None:
    async def scenario() -> None:
        settings = CacheSettings()
        metrics = _RecordingMetrics()
        backend = AsyncLocalBackend(settings, monotonic, metrics)
        cache = AsyncQueryCache(settings, _IDENTITY, l1=backend, metrics=metrics)
        results: dict[str, object] = {
            'find': [{'id': 7}],
            'preferences_read': {},
            'schema_read': {},
            'update': {'type': 'Asset', 'id': 7},
        }
        raw = _RawClient(results)
        entered = asyncio.Event()
        proceed = asyncio.Event()
        gated_method = {'direct': 'preferences_read', 'schema': 'schema_read', 'write': 'update'}[
            operation_name
        ]
        raw._gates[gated_method] = (entered, proceed)
        wrapper = AsyncCachedShotgun(raw, cache, _CONTEXT)

        if operation_name == 'direct':
            operation = asyncio.create_task(wrapper.preferences_read())
            expected: object = {}
        elif operation_name == 'schema':
            operation = asyncio.create_task(wrapper.find('Asset', [], ['id']))
            expected = [{'id': 7}]
        else:
            operation = asyncio.create_task(wrapper.update('Asset', 7, {'code': 'oak'}))
            expected = {'type': 'Asset', 'id': 7}
        await entered.wait()
        closing = asyncio.create_task(cache.close())
        await asyncio.sleep(0)
        assert not closing.done()
        proceed.set()
        assert await operation == expected
        await closing

    asyncio.run(scenario())


@pytest.mark.parametrize('humans_enabled', [False, True])
def test_human_read_plans_preserve_visibility_tags_and_identity_policy(
    humans_enabled: bool,
) -> None:
    async def scenario() -> None:
        schema = {'Asset': {'id': {'data_type': {'value': 'number'}, 'properties': {}}}}
        raw = _RawClient({'schema_read': schema, 'find': [], 'work_schedule_read': {}})
        cache = _RecordingCache()
        cache._identity = Identity('human', 'tests', 'human-42', entity_id=42)
        cache._settings = CacheSettings(
            identity_policy='scripts_and_humans' if humans_enabled else 'scripts_only',
            enabled_ttl_only_methods=('work_schedule_read',),
        )
        wrapper, _cache = _recording_wrapper(raw, cache)

        await wrapper.schema_read()
        await wrapper.find('Asset', [], ['id'])
        await wrapper.find('Asset', [['missing', 'is', 'value']], ['id'])
        await wrapper.work_schedule_read('2026-01-01', '2026-01-02')

        universal = (
            DependencyTag.site('production'),
            DependencyTag.authorization('production'),
            DependencyTag.human_visibility('production'),
            DependencyTag.identity_visibility('production', 42),
        )
        schema_tags = (universal[0], DependencyTag.schema('production'), *universal[1:])
        if humans_enabled:
            assert [
                (call.method, plan.classification, plan.tags) for call, plan, _, _ in cache._reads
            ] == [
                ('schema_read', 'schema', schema_tags),
                ('schema_view', 'schema', schema_tags),
                (
                    'find',
                    'event_aware',
                    (*schema_tags, DependencyTag.type_membership('production', 'Asset')),
                ),
                ('schema_view', 'schema', schema_tags),
                ('find', 'ttl_only', universal),
                ('work_schedule_read', 'ttl_only', universal),
            ]
        else:
            assert [
                (call.method, plan.reason, plan.tags) for call, plan, _, _ in cache._reads
            ] == [
                ('schema_read', 'human_caching_disabled', ()),
                ('find', 'human_caching_disabled', ()),
                ('find', 'human_caching_disabled', ()),
                ('work_schedule_read', 'human_caching_disabled', ()),
            ]
            assert [name for name, _, _ in raw._calls] == [
                'schema_read',
                'find',
                'find',
                'work_schedule_read',
            ]

    asyncio.run(scenario())
