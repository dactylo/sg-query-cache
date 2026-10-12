import asyncio
import os
import time
from collections.abc import Generator, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import replace
from hashlib import sha1
from secrets import token_hex
from types import SimpleNamespace
from typing import Protocol, cast

import pytest
from valkey.exceptions import ReadOnlyError, ResponseError

from sg_query_cache import (
    AsyncCachedShotgun,
    AsyncLocalBackend,
    AsyncQueryCache,
    CachedShotgun,
    ClientContext,
    Identity,
    LocalBackend,
    QueryCache,
)
from sg_query_cache import _valkey_scripts as scripts
from sg_query_cache._backend import (
    BackendCapacity,
    BackendUnavailable,
    CacheBackend,
    StaleIncarnation,
)
from sg_query_cache._codec import encode_value
from sg_query_cache._config import (
    CacheSettings,
    L1Settings,
    LimitSettings,
    ValkeyRoleSettings,
)
from sg_query_cache._keys import KeyBuilder
from sg_query_cache._model import DependencyTag, GenerationChange, GenerationVector
from sg_query_cache._valkey_common import (
    Incarnation,
    ScriptArgument,
    ScriptOperands,
    commit_operands,
    control_create_operands,
    control_rotate_operands,
    decode_resolve_reply,
    metadata_operands,
)
from sg_query_cache.valkey import ValkeyBackend
from sg_query_cache.valkey_async import AsyncValkeyBackend
from tests._backend_contract import _BackendContractTests, _BackendFactory, _BackendHarness
from tests._fakes import _FakeClock, _RecordingMetrics
from tests.valkey._environment import settings as _integration_settings
from tests.valkey._integration_helpers import ServerClock, flush_roles, integration_client

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get('SG_QUERY_CACHE_REQUIRE_INTEGRATION') != '1',
        reason='requires explicitly configured disposable Valkey roles',
    ),
]


class _AdminClient(scripts.ScriptClient, Protocol):
    def close(self) -> None: ...

    def config_get(self, pattern: str) -> Mapping[bytes | str, bytes | str]: ...

    def config_set(self, name: str, value: bytes | str | int) -> object: ...

    def delete(self, *keys: str) -> object: ...

    def execute_command(self, *args: object) -> object: ...

    def flushdb(self) -> object: ...

    def get(self, key: str) -> object: ...

    def info(self, section: str) -> Mapping[str, object]: ...

    def persist(self, key: str) -> object: ...

    def pttl(self, key: str) -> int: ...

    def rpush(self, key: str, value: str) -> object: ...

    def set(self, key: str, value: bytes | str | int) -> object: ...


def _client(role: ValkeyRoleSettings, settings: CacheSettings) -> _AdminClient:
    return cast('_AdminClient', integration_client(role, settings))


def _config_value(values: Mapping[bytes | str, bytes | str], name: str) -> bytes | str:
    value = values.get(name, values.get(name.encode()))
    if value is None:
        pytest.fail(f'Valkey CONFIG GET omitted {name}')
    return value


@contextmanager
def _demoted(client: _AdminClient) -> Generator[None]:
    try:
        client.execute_command('REPLICAOF', '192.0.2.1', 6379)
        yield
    finally:
        client.execute_command('REPLICAOF', 'NO', 'ONE')


@contextmanager
def _max_memory(client: _AdminClient) -> Generator[None]:
    policy = _config_value(client.config_get('maxmemory-policy'), 'maxmemory-policy')
    limit = _config_value(client.config_get('maxmemory'), 'maxmemory')
    try:
        client.config_set('maxmemory-policy', 'noeviction')
        client.config_set('maxmemory', 1)
        yield
    finally:
        try:
            client.config_set('maxmemory', limit)
        finally:
            client.config_set('maxmemory-policy', policy)


@contextmanager
def _min_replicas(client: _AdminClient) -> Generator[None]:
    original = _config_value(client.config_get('min-replicas-to-write'), 'min-replicas-to-write')
    try:
        client.config_set('min-replicas-to-write', 1)
        yield
    finally:
        client.config_set('min-replicas-to-write', original)


def _new_incarnation(prefix: str) -> Incarnation:
    nonce = token_hex(16)
    counter_key = f'{prefix}:v1:counter:{nonce}'
    record = f'primary|{nonce}|{counter_key}'.encode('ascii')
    return Incarnation('primary', nonce, counter_key, record)


def _metadata_error_case(
    client: _AdminClient,
    prefix: str,
    incarnation: Incarnation,
    case: str,
) -> tuple[str, str, ScriptOperands, bytes | None, str | None]:
    control = f'{prefix}:v1:control'
    if case in ('resolve', 'change', 'resolve-existing'):
        tag = DependencyTag.entity('integration', 'Asset', 901)
        operands = metadata_operands(prefix, incarnation, (tag,), 10)
        if case != 'resolve':
            assert client.set(operands.keys[2], '1')
        script, digest = (
            (scripts.CHANGE_SCRIPT, scripts.CHANGE_SCRIPT_SHA)
            if case == 'change'
            else (scripts.RESOLVE_SCRIPT, scripts.RESOLVE_SCRIPT_SHA)
        )
        return script, digest, operands, incarnation.record, None
    replacement = _new_incarnation(prefix)
    if case == 'create':
        assert client.delete(control) == 1
        return (
            scripts.CONTROL_CREATE_SCRIPT,
            scripts.CONTROL_CREATE_SCRIPT_SHA,
            control_create_operands(control, replacement),
            None,
            replacement.counter_key,
        )
    assert case == 'rotate'
    return (
        scripts.CONTROL_ROTATE_SCRIPT,
        scripts.CONTROL_ROTATE_SCRIPT_SHA,
        control_rotate_operands(control, incarnation.record, replacement),
        incarnation.record,
        replacement.counter_key,
    )


def _raw_script(
    client: _AdminClient,
    script: str,
    digest: str,
    keys: tuple[str, ...],
    args: tuple[ScriptArgument, ...],
) -> object:
    assert sha1(script.encode(), usedforsecurity=False).hexdigest() == digest
    return client.evalsha(digest, len(keys), *keys, *args)


@contextmanager
def _restricted_user(
    admin: _AdminClient,
    role: ValkeyRoleSettings,
    settings: CacheSettings,
    prefix: str,
) -> Generator[_AdminClient]:
    username = f'sgqc-oi02-{token_hex(4)}'
    password = token_hex(16)
    selector = (
        '(+get +set +del +exists +incr +persist +pexpire +pttl '
        f'%RW~{prefix}:v1:counter:* %R~{prefix}:v1:metadata:* %R~{prefix}:v1:control)'
    )
    try:
        admin.execute_command(
            'ACL',
            'SETUSER',
            username,
            'on',
            f'>{password}',
            'resetkeys',
            'resetchannels',
            '-@all',
            '+evalsha',
            '+script|load',
            '+select',
            f'~{prefix}:*',
            selector,
        )
        client = cast(
            '_AdminClient', integration_client(role, settings, credentials=(username, password))
        )
        try:
            yield client
        finally:
            client.close()
    finally:
        admin.execute_command('ACL', 'DELUSER', username)


def _scoped_tag_key(settings: CacheSettings, incarnation: str, tag: DependencyTag) -> str:
    prefix = settings.l2.valkey.metadata.key_prefix
    return f'{prefix}:v1:metadata:{incarnation}:{KeyBuilder.tag_key(tag)}'


def _script_call(
    client: _AdminClient,
    script: str,
    digest: str,
    keys: tuple[str, ...],
    args: tuple[ScriptArgument, ...],
) -> object:
    return scripts.execute_script(client, script=script, sha=digest, keys=keys, args=args)


def _used_memory(client: _AdminClient) -> int:
    value = client.info('memory').get('used_memory')
    if isinstance(value, bool) or not isinstance(value, int | str | bytes):
        pytest.fail('Valkey INFO memory omitted used_memory')
    return int(value)


@pytest.fixture
def script_metadata() -> Iterator[tuple[_AdminClient, CacheSettings, Incarnation]]:
    settings = _integration_settings(os.environ)
    flush_roles(settings)
    client = _client(settings.l2.valkey.metadata, settings)
    prefix = settings.l2.valkey.metadata.key_prefix
    nonce = token_hex(16)
    counter_key = f'{prefix}:v1:counter:{nonce}'
    record = f'primary|{nonce}|{counter_key}'.encode('ascii')
    incarnation = Incarnation('primary', nonce, counter_key, record)
    operands = control_create_operands(f'{prefix}:v1:control', incarnation)
    try:
        assert _script_call(
            client,
            scripts.CONTROL_CREATE_SCRIPT,
            scripts.CONTROL_CREATE_SCRIPT_SHA,
            operands.keys,
            operands.args,
        ) == [b'created', record]
        yield client, settings, incarnation
    finally:
        client.close()


@pytest.fixture
def backend_factory() -> Iterator[_BackendFactory]:
    backends: list[CacheBackend] = []

    def create(settings: CacheSettings | None = None) -> _BackendHarness:
        limits = None if settings is None else settings.limits
        resolved = _integration_settings(os.environ, limits)
        flush_roles(resolved)
        backend: CacheBackend = ValkeyBackend.from_settings(
            resolved,
            os.environ,
            _RecordingMetrics(),
        )
        backends.append(backend)
        return _BackendHarness(backend, ServerClock(), _RecordingMetrics())

    yield create
    for backend in backends:
        backend.close()


class TestValkeyBackend(_BackendContractTests):
    pass


def test_change_script_keeps_final_signed_token_and_reports_overflow(
    script_metadata: tuple[_AdminClient, CacheSettings, Incarnation],
) -> None:
    client, settings, incarnation = script_metadata
    prefix = settings.l2.valkey.metadata.key_prefix
    tag = DependencyTag.entity('integration', 'Asset', 203)
    operands = metadata_operands(prefix, incarnation, (tag,), 10)
    assert client.set(operands.keys[2], '1')
    assert client.set(incarnation.counter_key, '9223372036854775806')
    assert _script_call(
        client, scripts.CHANGE_SCRIPT, scripts.CHANGE_SCRIPT_SHA, operands.keys, operands.args
    ) == [b'ok', 1, 0]
    assert client.get(operands.keys[2]) == b'9223372036854775807'
    assert _script_call(
        client, scripts.CHANGE_SCRIPT, scripts.CHANGE_SCRIPT_SHA, operands.keys, operands.args
    ) == [b'rotation-required']
    assert client.get(operands.keys[2]) == b'9223372036854775807'
    assert client.get(incarnation.counter_key) == b'9223372036854775807'


def test_commit_passes_through_no_replicas() -> None:
    settings = _integration_settings(os.environ)
    flush_roles(settings)
    backend = ValkeyBackend.from_settings(settings, os.environ, _RecordingMetrics())
    client = _client(settings.l2.valkey.result, settings)
    try:
        lease = backend.reserve('script-no-replicas', 30)
        assert lease is not None
        payload = encode_value('no-replicas')
        operands = commit_operands(settings.l2.valkey.result.key_prefix, lease, payload, 10_000)
        client.script_load(scripts.COMMIT_SCRIPT)
        with _min_replicas(client):
            with pytest.raises(ResponseError, match=r'^NOREPLICAS'):
                _raw_script(
                    client,
                    scripts.COMMIT_SCRIPT,
                    scripts.COMMIT_SCRIPT_SHA,
                    operands.keys,
                    operands.args,
                )
            with pytest.raises(BackendUnavailable):
                backend.commit(lease, payload, 10)
    finally:
        client.close()
        backend.close()


def test_commit_passes_through_read_only() -> None:
    settings = _integration_settings(os.environ)
    flush_roles(settings)
    backend = ValkeyBackend.from_settings(settings, os.environ, _RecordingMetrics())
    client = _client(settings.l2.valkey.result, settings)
    try:
        lease = backend.reserve('script-read-only', 30)
        assert lease is not None
        payload = encode_value('read-only')
        operands = commit_operands(settings.l2.valkey.result.key_prefix, lease, payload, 10_000)
        client.script_load(scripts.COMMIT_SCRIPT)
        with _demoted(client), pytest.raises(ReadOnlyError):
            _raw_script(
                client,
                scripts.COMMIT_SCRIPT,
                scripts.COMMIT_SCRIPT_SHA,
                operands.keys,
                operands.args,
            )
    finally:
        try:
            flush_roles(settings)
        finally:
            client.close()
            backend.close()


def test_commit_reports_capacity_and_backend_rejects() -> None:
    settings = _integration_settings(os.environ)
    flush_roles(settings)
    backend = ValkeyBackend.from_settings(settings, os.environ, _RecordingMetrics())
    client = _client(settings.l2.valkey.result, settings)
    try:
        lease = backend.reserve('script-capacity', 30)
        assert lease is not None
        payload = encode_value('capacity')
        operands = commit_operands(settings.l2.valkey.result.key_prefix, lease, payload, 10_000)
        client.script_load(scripts.COMMIT_SCRIPT)
        with _max_memory(client):
            assert _raw_script(
                client,
                scripts.COMMIT_SCRIPT,
                scripts.COMMIT_SCRIPT_SHA,
                operands.keys,
                operands.args,
            ) == [b'capacity']
            assert backend.commit(lease, payload, 10) is False
    finally:
        client.close()
        backend.close()


def test_disposable_roles_have_required_eviction_and_persistence_policies() -> None:
    settings = _integration_settings(os.environ)
    result = _client(settings.l2.valkey.result, settings)
    metadata = _client(settings.l2.valkey.metadata, settings)
    try:
        result_policy = result.config_get('maxmemory-policy')
        result_appendonly = result.config_get('appendonly')
        result_save = result.config_get('save')
        metadata_policy = metadata.config_get('maxmemory-policy')
        metadata_appendonly = metadata.config_get('appendonly')
        metadata_appendfsync = metadata.config_get('appendfsync')
    finally:
        result.close()
        metadata.close()

    assert _config_value(result_policy, 'maxmemory-policy') in (b'allkeys-lru', 'allkeys-lru')
    assert _config_value(result_appendonly, 'appendonly') in (b'no', 'no')
    assert _config_value(result_save, 'save') in (b'', '')
    assert _config_value(metadata_policy, 'maxmemory-policy') in (b'noeviction', 'noeviction')
    assert _config_value(metadata_appendonly, 'appendonly') in (b'yes', 'yes')
    assert _config_value(metadata_appendfsync, 'appendfsync') in (b'everysec', 'everysec')


def test_metadata_backend_passes_through_no_replicas() -> None:
    settings = _integration_settings(os.environ)
    flush_roles(settings)
    backend = ValkeyBackend.from_settings(settings, os.environ, _RecordingMetrics())
    client = _client(settings.l2.valkey.metadata, settings)
    tag = DependencyTag.entity('integration', 'Asset', 902)
    try:
        backend.resolve((tag,))
        with _min_replicas(client):
            with pytest.raises(BackendUnavailable):
                backend.resolve((tag,))
            with pytest.raises(BackendUnavailable):
                backend.change((GenerationChange(tag),))
    finally:
        client.close()
        backend.close()


def test_metadata_exhaustion_bypasses_then_recovers_after_native_expiry() -> None:
    limits = LimitSettings(scoped_generation_retention_seconds=0.5)
    settings = _integration_settings(os.environ, limits)
    flush_roles(settings)
    backend = ValkeyBackend.from_settings(settings, os.environ, _RecordingMetrics())
    metadata = _client(settings.l2.valkey.metadata, settings)
    original_maxmemory = _config_value(metadata.config_get('maxmemory'), 'maxmemory')
    used_memory = _used_memory(metadata)
    metadata.config_set('maxmemory', used_memory + 65_536)
    exhausted = False
    try:
        for batch in range(512):
            tags = tuple(
                DependencyTag.entity('integration', 'Asset', batch * 16 + offset + 1)
                for offset in range(16)
            )
            try:
                backend.resolve(tags)
            except (BackendCapacity, StaleIncarnation):
                exhausted = True
                break
        assert exhausted
        time.sleep(0.6)
        recovered = backend.resolve((DependencyTag.entity('integration', 'Asset', 1_000_000),))
        assert recovered.entries[0][1] > 0
    finally:
        metadata.config_set('maxmemory', original_maxmemory)
        metadata.close()
        backend.close()


@pytest.mark.parametrize('case', ['resolve', 'change', 'create', 'rotate'])
def test_metadata_scripts_report_capacity_at_first_write(
    script_metadata: tuple[_AdminClient, CacheSettings, Incarnation],
    case: str,
) -> None:
    client, settings, incarnation = script_metadata
    prefix = settings.l2.valkey.metadata.key_prefix
    script, digest, operands, control_before, _ = _metadata_error_case(
        client, prefix, incarnation, case
    )
    client.script_load(script)
    with _max_memory(client):
        assert _raw_script(client, script, digest, operands.keys, operands.args) == [b'capacity']
    assert client.get(f'{prefix}:v1:control') == control_before


@pytest.mark.parametrize('counter_kind', ['text', 'list'])
@pytest.mark.parametrize('case', ['resolve', 'change'])
def test_invalid_counter_requires_rotation(
    script_metadata: tuple[_AdminClient, CacheSettings, Incarnation],
    counter_kind: str,
    case: str,
) -> None:
    client, settings, incarnation = script_metadata
    prefix = settings.l2.valkey.metadata.key_prefix
    script, digest, operands, _, _ = _metadata_error_case(client, prefix, incarnation, case)
    if counter_kind == 'text':
        assert client.set(incarnation.counter_key, 'x')
    else:
        assert client.delete(incarnation.counter_key) == 1
        assert client.rpush(incarnation.counter_key, 'x') == 1
    client.script_load(script)
    assert _raw_script(client, script, digest, operands.keys, operands.args) == [
        b'rotation-required'
    ]


@pytest.mark.parametrize('case', ['resolve', 'change', 'create', 'rotate'])
def test_metadata_scripts_pass_through_no_replicas(
    script_metadata: tuple[_AdminClient, CacheSettings, Incarnation],
    case: str,
) -> None:
    client, settings, incarnation = script_metadata
    script, digest, operands, _, _ = _metadata_error_case(
        client, settings.l2.valkey.metadata.key_prefix, incarnation, case
    )
    client.script_load(script)
    with _min_replicas(client), pytest.raises(ResponseError, match=r'^NOREPLICAS'):
        _raw_script(client, script, digest, operands.keys, operands.args)


@pytest.mark.parametrize('case', ['resolve-existing', 'resolve', 'change', 'create', 'rotate'])
def test_metadata_scripts_pass_through_read_only(
    script_metadata: tuple[_AdminClient, CacheSettings, Incarnation],
    case: str,
) -> None:
    client, settings, incarnation = script_metadata
    script, digest, operands, _, _ = _metadata_error_case(
        client, settings.l2.valkey.metadata.key_prefix, incarnation, case
    )
    client.script_load(script)
    try:
        with _demoted(client), pytest.raises(ReadOnlyError):
            _raw_script(client, script, digest, operands.keys, operands.args)
    finally:
        flush_roles(settings)


@pytest.mark.parametrize('case', ['resolve', 'change', 'create', 'rotate'])
def test_metadata_scripts_preserve_post_write_acl_errors(
    script_metadata: tuple[_AdminClient, CacheSettings, Incarnation],
    case: str,
) -> None:
    admin, settings, incarnation = script_metadata
    prefix = settings.l2.valkey.metadata.key_prefix
    script, digest, operands, control_before, new_counter = _metadata_error_case(
        admin, prefix, incarnation, case
    )
    counter_before = admin.get(incarnation.counter_key)
    tag_before = admin.get(operands.keys[2]) if case in ('resolve', 'change') else None
    with _restricted_user(admin, settings.l2.valkey.metadata, settings, prefix) as restricted:
        restricted.script_load(script)
        with pytest.raises(
            ResponseError, match=r'^ACL failure in script: No permissions to access a key'
        ):
            _raw_script(restricted, script, digest, operands.keys, operands.args)
    assert admin.get(f'{prefix}:v1:control') == control_before
    if case in ('resolve', 'change'):
        assert isinstance(counter_before, bytes | str)
        assert admin.get(incarnation.counter_key) == str(int(counter_before) + 1).encode()
        assert admin.get(operands.keys[2]) == tag_before
    else:
        assert new_counter is not None
        assert admin.get(new_counter) is None


def test_resolve_script_control_ceiling_and_lifetimes(
    script_metadata: tuple[_AdminClient, CacheSettings, Incarnation],
) -> None:
    client, settings, incarnation = script_metadata
    prefix = settings.l2.valkey.metadata.key_prefix
    global_tag = DependencyTag.site('integration')
    scoped_tag = DependencyTag.entity('integration', 'Asset', 301)
    operands = metadata_operands(prefix, incarnation, (global_tag, scoped_tag), 0.5)
    assert _script_call(
        client, scripts.RESOLVE_SCRIPT, scripts.RESOLVE_SCRIPT_SHA, operands.keys, operands.args
    ) == [b'ok', b'1', b'2']
    assert client.get(operands.keys[2]) == b'1'
    assert client.pttl(operands.keys[2]) == -1
    assert client.get(operands.keys[3]) == b'2'
    assert 0 < client.pttl(operands.keys[3]) <= 500

    assert client.set(operands.keys[2], '1')
    assert _script_call(
        client, scripts.CHANGE_SCRIPT, scripts.CHANGE_SCRIPT_SHA, operands.keys, operands.args
    ) == [b'ok', 2, 0]
    assert client.get(operands.keys[2]) == b'3'
    assert client.pttl(operands.keys[2]) == -1
    assert client.get(operands.keys[3]) == b'4'
    assert 0 < client.pttl(operands.keys[3]) <= 500

    stale = (b'other-control', *operands.args[1:])
    assert _script_call(
        client, scripts.RESOLVE_SCRIPT, scripts.RESOLVE_SCRIPT_SHA, operands.keys, stale
    ) == [b'stale']
    assert _script_call(
        client, scripts.CHANGE_SCRIPT, scripts.CHANGE_SCRIPT_SHA, operands.keys, stale
    ) == [b'stale']

    over_limit = metadata_operands(
        prefix,
        incarnation,
        tuple(DependencyTag.entity('integration', 'Asset', index) for index in range(1, 34)),
        0.5,
    )
    counter_before = client.get(incarnation.counter_key)
    assert _script_call(
        client,
        scripts.RESOLVE_SCRIPT,
        scripts.RESOLVE_SCRIPT_SHA,
        over_limit.keys,
        over_limit.args,
    ) == [b'too-many-tags']
    assert client.get(incarnation.counter_key) == counter_before
    assert client.get(over_limit.keys[-1]) is None

    assert client.delete(incarnation.counter_key) == 1
    assert _script_call(
        client, scripts.RESOLVE_SCRIPT, scripts.RESOLVE_SCRIPT_SHA, operands.keys, operands.args
    ) == [b'missing-counter']
    assert _script_call(
        client, scripts.CHANGE_SCRIPT, scripts.CHANGE_SCRIPT_SHA, operands.keys, operands.args
    ) == [b'missing-counter']


def test_result_eviction_does_not_change_metadata_generation() -> None:
    settings = _integration_settings(os.environ)
    flush_roles(settings)
    backend = ValkeyBackend.from_settings(settings, os.environ, _RecordingMetrics())
    result = _client(settings.l2.valkey.result, settings)
    original_maxmemory = _config_value(result.config_get('maxmemory'), 'maxmemory')
    used_memory = _used_memory(result)
    result.config_set('maxmemory', used_memory + 262_144)
    tag = DependencyTag.type_value('integration', 'Asset')
    before = backend.resolve((tag,))
    seed_key = 'eviction-seed'
    payload = encode_value('x' * 65_536)
    try:
        seed_lease = backend.reserve(seed_key, 5)
        assert seed_lease is not None
        assert backend.commit(seed_lease, payload, 30)
        for index in range(128):
            lease = backend.reserve(f'pressure-{index}', 5)
            if lease is not None:
                backend.commit(lease, payload, 30)
            if backend.lookup(seed_key) is None:
                break
        assert backend.lookup(seed_key) is None
        assert backend.resolve((tag,)) == before
    finally:
        result.config_set('maxmemory', original_maxmemory)
        result.close()
        backend.close()


def test_scoped_change_uses_materialization_and_preserves_bounded_lifetimes() -> None:
    limits = LimitSettings(scoped_generation_retention_seconds=0.5)
    settings = _integration_settings(os.environ, limits)
    flush_roles(settings)
    backend = ValkeyBackend.from_settings(settings, os.environ, _RecordingMetrics())
    metadata = _client(settings.l2.valkey.metadata, settings)
    seed = DependencyTag.entity('integration', 'Asset', 1)
    expiring = DependencyTag.entity('integration', 'Asset', 2)
    persistent = DependencyTag.entity('integration', 'Asset', 3)
    absent = DependencyTag.entity('integration', 'Asset', 4)
    try:
        initial = backend.resolve((seed, expiring, persistent))
        expiring_key = _scoped_tag_key(settings, initial.incarnation, expiring)
        persistent_key = _scoped_tag_key(settings, initial.incarnation, persistent)
        absent_key = _scoped_tag_key(settings, initial.incarnation, absent)

        assert metadata.get(absent_key) is None
        backend.change((GenerationChange(absent),))
        assert metadata.get(absent_key) is None
        resolved_absent = backend.resolve((absent,))
        assert resolved_absent.entries == ((absent, 4),)

        expiring_before = metadata.get(expiring_key)
        expiring_ttl_before = metadata.pttl(expiring_key)
        persistent_before = metadata.get(persistent_key)
        assert metadata.persist(persistent_key)
        assert metadata.pttl(persistent_key) == -1

        backend.change((GenerationChange(expiring), GenerationChange(persistent)))

        expiring_ttl_after = metadata.pttl(expiring_key)
        assert metadata.get(expiring_key) != expiring_before
        assert 0 < expiring_ttl_after <= expiring_ttl_before
        persistent_ttl_after = metadata.pttl(persistent_key)
        assert metadata.get(persistent_key) != persistent_before
        assert 0 < persistent_ttl_after <= 500
        time.sleep(0.6)
        assert metadata.get(persistent_key) is None
    finally:
        metadata.close()
        backend.close()


def test_scripts_keep_exact_tokens_across_lua_precision_boundary(
    script_metadata: tuple[_AdminClient, CacheSettings, Incarnation],
) -> None:
    client, settings, incarnation = script_metadata
    prefix = settings.l2.valkey.metadata.key_prefix
    first = DependencyTag.entity('integration', 'Asset', 101)
    second = DependencyTag.entity('integration', 'Asset', 102)
    assert client.set(incarnation.counter_key, '9007199254740991')
    operands = metadata_operands(prefix, incarnation, (first, second), 10)

    first_reply = _script_call(
        client, scripts.RESOLVE_SCRIPT, scripts.RESOLVE_SCRIPT_SHA, operands.keys, operands.args
    )
    assert first_reply == [b'ok', b'9007199254740992', b'9007199254740993']
    first_tokens = decode_resolve_reply(first_reply, expected_count=2)
    old_vector = GenerationVector(
        incarnation.nonce,
        tuple(zip((first, second), first_tokens, strict=True)),
    )
    base_key = 'sgqc:v1:base:' + 'a' * 64
    old_result_key = KeyBuilder.qualified_key(base_key, old_vector, 'result')
    assert client.get(operands.keys[2]) == b'9007199254740992'
    assert client.get(operands.keys[3]) == b'9007199254740993'

    assert _script_call(
        client, scripts.CHANGE_SCRIPT, scripts.CHANGE_SCRIPT_SHA, operands.keys, operands.args
    ) == [b'ok', 2, 0]
    assert client.get(operands.keys[2]) == b'9007199254740994'
    assert client.get(operands.keys[3]) == b'9007199254740995'
    updated_reply = _script_call(
        client, scripts.RESOLVE_SCRIPT, scripts.RESOLVE_SCRIPT_SHA, operands.keys, operands.args
    )
    assert updated_reply == [b'ok', b'9007199254740994', b'9007199254740995']
    updated_tokens = decode_resolve_reply(updated_reply, expected_count=2)
    updated_vector = GenerationVector(
        incarnation.nonce,
        tuple(zip((first, second), updated_tokens, strict=True)),
    )
    assert KeyBuilder.qualified_key(base_key, updated_vector, 'result') != old_result_key


def test_scripts_rotate_at_signed_counter_overflow_without_reusing_old_keys(
    script_metadata: tuple[_AdminClient, CacheSettings, Incarnation],
) -> None:
    client, settings, incarnation = script_metadata
    prefix = settings.l2.valkey.metadata.key_prefix
    tag = DependencyTag.entity('integration', 'Asset', 201)
    assert client.set(incarnation.counter_key, '9223372036854775806')
    operands = metadata_operands(prefix, incarnation, (tag,), 10)
    assert _script_call(
        client, scripts.RESOLVE_SCRIPT, scripts.RESOLVE_SCRIPT_SHA, operands.keys, operands.args
    ) == [b'ok', b'9223372036854775807']
    assert client.get(operands.keys[2]) == b'9223372036854775807'

    next_tag = DependencyTag.entity('integration', 'Asset', 202)
    next_operands = metadata_operands(prefix, incarnation, (next_tag,), 10)
    assert _script_call(
        client,
        scripts.RESOLVE_SCRIPT,
        scripts.RESOLVE_SCRIPT_SHA,
        next_operands.keys,
        next_operands.args,
    ) == [b'rotation-required']
    assert client.get(incarnation.counter_key) == b'9223372036854775807'
    replacement_nonce = token_hex(16)
    replacement_counter = f'{prefix}:v1:counter:{replacement_nonce}'
    replacement = Incarnation(
        'primary',
        replacement_nonce,
        replacement_counter,
        f'primary|{replacement_nonce}|{replacement_counter}'.encode('ascii'),
    )
    rotation = control_rotate_operands(f'{prefix}:v1:control', incarnation.record, replacement)
    assert _script_call(
        client,
        scripts.CONTROL_ROTATE_SCRIPT,
        scripts.CONTROL_ROTATE_SCRIPT_SHA,
        rotation.keys,
        rotation.args,
    ) == [b'rotated', replacement.record]
    assert _script_call(
        client, scripts.RESOLVE_SCRIPT, scripts.RESOLVE_SCRIPT_SHA, operands.keys, operands.args
    ) == [b'stale']
    new_operands = metadata_operands(prefix, replacement, (tag,), 10)
    assert new_operands.keys[2] != operands.keys[2]
    assert _script_call(
        client,
        scripts.RESOLVE_SCRIPT,
        scripts.RESOLVE_SCRIPT_SHA,
        new_operands.keys,
        new_operands.args,
    ) == [b'ok', b'1']
    assert client.get(operands.keys[2]) == b'9223372036854775807'
    assert client.get(new_operands.keys[2]) == b'1'


@pytest.mark.parametrize('hierarchical', [False, True])
def test_sync_async_wrappers_share_fills_and_observe_both_invalidations(
    hierarchical: bool,
) -> None:
    async def scenario() -> None:
        settings = _integration_settings(os.environ, LimitSettings())
        prefix = 'sgqc-cross-' + token_hex(8)
        settings = replace(
            settings,
            l1=L1Settings(enabled=hierarchical),
            l2=replace(
                settings.l2,
                valkey=replace(
                    settings.l2.valkey,
                    result=replace(settings.l2.valkey.result, key_prefix=prefix),
                    metadata=replace(settings.l2.valkey.metadata, key_prefix=prefix),
                ),
            ),
        )
        clock = _FakeClock()
        metrics = _RecordingMetrics()
        identity = Identity('script', 'integration', 'cross-wrapper')
        context = ClientContext(
            'https://example.invalid',
            'integration',
            localized=False,
            convert_datetimes_to_utc=True,
            records_per_page=500,
        )
        sync_l2 = ValkeyBackend.from_settings(settings, os.environ, metrics)
        async_l2 = await AsyncValkeyBackend.from_settings(settings, os.environ, metrics)
        sync_cache = QueryCache(
            settings,
            identity,
            l1=LocalBackend(settings, clock, metrics) if hierarchical else None,
            l2=sync_l2,
        )
        async_cache = AsyncQueryCache(
            settings,
            identity,
            l1=AsyncLocalBackend(settings, clock, metrics) if hierarchical else None,
            l2=async_l2,
        )
        calls = [0, 0]

        def sync_read(project_entity: object = None) -> object:
            _ = project_entity
            calls[0] += 1
            return {'version': sum(calls)}

        async def async_read(project_entity: object = None) -> object:
            _ = project_entity
            calls[1] += 1
            return {'version': sum(calls)}

        def sync_write(*args: object, **kwargs: object) -> int:
            _ = args, kwargs
            return 1

        async def async_write(*args: object, **kwargs: object) -> int:
            return sync_write(*args, **kwargs)

        sync = CachedShotgun(
            SimpleNamespace(schema_read=sync_read, share_thumbnail=sync_write), sync_cache, context
        )
        asynchronous = AsyncCachedShotgun(
            SimpleNamespace(schema_read=async_read, share_thumbnail=async_write),
            async_cache,
            context,
        )
        try:
            assert sync.schema_read() == await asynchronous.schema_read() == {'version': 1}
            assert calls == [1, 0]
            sync.share_thumbnail([], source_entity={'type': 'Asset', 'id': 7})
            # Separate L1s observe remote changes when their admitted L1 age expires.
            clock._advance(11)
            assert await asynchronous.schema_read() == sync.schema_read() == {'version': 2}
            assert calls == [1, 1]
            await asynchronous.share_thumbnail([], source_entity={'type': 'Asset', 'id': 7})
            clock._advance(11)
            assert sync.schema_read() == await asynchronous.schema_read() == {'version': 3}
            assert calls == [2, 1]
        finally:
            sync_cache.close()
            await async_cache.close()

    asyncio.run(scenario())
