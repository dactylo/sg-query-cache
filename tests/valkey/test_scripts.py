from collections.abc import Callable
from hashlib import sha1
from typing import TypeVar, cast

import pytest
from valkey.exceptions import (
    DataError,
    NoPermissionError,
    NoScriptError,
    OutOfMemoryError,
    ResponseError,
)

import sg_query_cache._valkey_scripts as scripts_module
from sg_query_cache._backend import (
    BackendCapacity,
    BackendProtocolError,
    BackendUnavailable,
    StaleIncarnation,
)
from sg_query_cache._model import DependencyTag, FillLease
from sg_query_cache._valkey_common import (
    ScriptArgument,
    ScriptOperands,
    commit_operands,
    control_create_operands,
    control_rotate_operands,
    lookup_operands,
    metadata_operands,
    parse_record,
    release_operands,
    reserve_operands,
)
from sg_query_cache._valkey_proof import PrimaryProofError
from tests._valkey_control_cases import _RECORD_A, _RECORD_B, control_reply

_T = TypeVar('_T')


class _ScriptedClient:
    def __init__(self, replies: list[object]) -> None:
        self._calls: list[tuple[object, ...]] = []
        self._replies = replies

    def evalsha(self, sha: str, numkeys: int, *keys_and_args: object) -> object:
        self._calls.append(('evalsha', sha, numkeys, *keys_and_args))
        reply = self._replies.pop(0)
        if isinstance(reply, BaseException):
            raise reply
        return reply

    def run_operation(self, operation: Callable[[scripts_module.ScriptClient], _T]) -> _T:
        return operation(self)

    def script_load(self, script: bytes | str) -> object:
        self._calls.append(('script_load', script))
        reply = self._replies.pop(0)
        if isinstance(reply, BaseException):
            raise reply
        return reply


def test_each_valkey_script_has_its_exact_load_digest() -> None:
    names = (
        'CHANGE',
        'COMMIT',
        'CONTROL_CREATE',
        'CONTROL_ROTATE',
        'LOOKUP',
        'RELEASE',
        'RESERVE',
        'RESOLVE',
    )

    for name in names:
        source = getattr(scripts_module, f'{name}_SCRIPT')
        digest = getattr(scripts_module, f'{name}_SCRIPT_SHA')
        assert digest == sha1(source.encode(), usedforsecurity=False).hexdigest()


@pytest.mark.parametrize('failure_type', [BackendUnavailable, BackendProtocolError])
def test_proof_failure_translates_to_fresh_backend_error(
    failure_type: type[BackendUnavailable] | type[BackendProtocolError],
) -> None:
    failure = failure_type('fixed proof failure')

    with pytest.raises(failure_type) as caught:
        scripts_module.translate_client_error(PrimaryProofError(failure))

    assert caught.value is not failure
    assert str(caught.value) == str(failure)
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None


def test_script_cache_miss_does_not_retry_a_second_noscript() -> None:
    client = _ScriptedClient(
        [
            NoScriptError('missing'),
            scripts_module.LOOKUP_SCRIPT_SHA,
            NoScriptError('missing again'),
        ]
    )

    with pytest.raises(BackendUnavailable):
        scripts_module.execute_script(
            client,
            script=scripts_module.LOOKUP_SCRIPT,
            sha=scripts_module.LOOKUP_SCRIPT_SHA,
            keys=('result-key',),
            args=(),
        )

    assert [call[0] for call in client._calls] == ['evalsha', 'script_load', 'evalsha']


def test_script_cache_miss_loads_once_and_retries_once() -> None:
    client = _ScriptedClient(
        [NoScriptError('missing'), scripts_module.LOOKUP_SCRIPT_SHA.encode(), [b'miss']]
    )

    reply = scripts_module.execute_script(
        client,
        script=scripts_module.LOOKUP_SCRIPT,
        sha=scripts_module.LOOKUP_SCRIPT_SHA,
        keys=('result-key',),
        args=(),
    )

    assert reply == [b'miss']
    assert client._calls == [
        ('evalsha', scripts_module.LOOKUP_SCRIPT_SHA, 1, 'result-key'),
        ('script_load', scripts_module.LOOKUP_SCRIPT),
        ('evalsha', scripts_module.LOOKUP_SCRIPT_SHA, 1, 'result-key'),
    ]


def test_shared_control_reply_rotates_absent_record_from_empty_observed() -> None:
    counters: set[str] = set()
    control, reply = control_reply(
        scripts_module.CONTROL_ROTATE_SCRIPT_SHA,
        2,
        ('control', 'counter', b'', _RECORD_B),
        None,
        counters,
    )
    assert control == _RECORD_B
    assert reply == [b'rotated', _RECORD_B]
    assert counters == {'counter'}


@pytest.mark.parametrize(
    ('error', 'expected'),
    [
        (OutOfMemoryError("command not allowed when used memory > 'maxmemory'."), BackendCapacity),
        (
            ResponseError("OOM command not allowed when used memory > 'maxmemory'."),
            BackendCapacity,
        ),
        (
            NoPermissionError("User cache-room has no permissions to run the 'set' command"),
            BackendUnavailable,
        ),
        (
            ResponseError('ACL failure in script: User cache-room has no permissions'),
            BackendUnavailable,
        ),
        (ResponseError('maxmemory policy rejected a module command'), BackendUnavailable),
        (ResponseError('NOREPLICAS Not enough good replicas to write.'), BackendUnavailable),
    ],
)
def test_translation_reports_capacity_only_for_the_oom_code(
    error: Exception, expected: type[Exception]
) -> None:
    with pytest.raises(expected) as captured:
        scripts_module.translate_client_error(error)

    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None
    assert 'cache-room' not in str(captured.value)


@pytest.mark.parametrize(
    ('name', 'build', 'expected'),
    [
        (
            'CONTROL_CREATE',
            lambda: control_create_operands('sgqc:v1:control', parse_record(_RECORD_A, 'sgqc')),
            ('sgqc:v1:control', 'sgqc:v1:counter:00000000000000000000000000000001', _RECORD_A),
        ),
        (
            'CONTROL_ROTATE',
            lambda: control_rotate_operands(
                'sgqc:v1:control', _RECORD_A, parse_record(_RECORD_B, 'sgqc')
            ),
            (
                'sgqc:v1:control',
                'sgqc:v1:counter:00000000000000000000000000000002',
                _RECORD_A,
                _RECORD_B,
            ),
        ),
        (
            'CONTROL_ROTATE',
            lambda: control_rotate_operands(
                'sgqc:v1:control', None, parse_record(_RECORD_B, 'sgqc')
            ),
            (
                'sgqc:v1:control',
                'sgqc:v1:counter:00000000000000000000000000000002',
                b'',
                _RECORD_B,
            ),
        ),
        ('LOOKUP', lambda: lookup_operands('results', 'key'), ('results:v1:result:key',)),
        (
            'RESERVE',
            lambda: reserve_operands('results', 'key', 1250, 'token'),
            ('results:v1:lease:key', 'token', 1250),
        ),
        (
            'RELEASE',
            lambda: release_operands('results', FillLease('key', 'token')),
            ('results:v1:lease:key', 'token'),
        ),
        (
            'COMMIT',
            lambda: commit_operands('results', FillLease('key', 'token'), b'payload', 1250),
            ('results:v1:lease:key', 'results:v1:result:key', 'token', b'payload', 1250),
        ),
    ],
)
def test_script_operands_keep_golden_wire_order_across_noscript(
    name: str,
    build: Callable[[], ScriptOperands],
    expected: tuple[ScriptArgument, ...],
) -> None:
    script = cast('str', getattr(scripts_module, f'{name}_SCRIPT'))
    digest = cast('str', getattr(scripts_module, f'{name}_SCRIPT_SHA'))
    client = _ScriptedClient([NoScriptError('missing'), digest.encode(), [b'ok']])
    operands = build()

    assert scripts_module.execute_script(
        client,
        script=script,
        sha=digest,
        keys=operands.keys,
        args=operands.args,
    ) == [b'ok']

    key_count = 2 if name in ('COMMIT', 'CONTROL_CREATE', 'CONTROL_ROTATE') else 1
    assert client._calls == [
        ('evalsha', digest, key_count, *expected),
        ('script_load', script),
        ('evalsha', digest, key_count, *expected),
    ]


@pytest.mark.parametrize('name', ['CHANGE', 'RESOLVE'])
def test_metadata_operands_keep_golden_tag_order_and_retention_flags(name: str) -> None:
    operands = metadata_operands(
        'sgqc',
        parse_record(_RECORD_A, 'sgqc'),
        (DependencyTag.site('site'), DependencyTag.entity('site', 'Asset', 7)),
        1.2349,
    )
    script = cast('str', getattr(scripts_module, f'{name}_SCRIPT'))
    digest = cast('str', getattr(scripts_module, f'{name}_SCRIPT_SHA'))
    client = _ScriptedClient([[b'ok']])

    scripts_module.execute_script(
        client,
        script=script,
        sha=digest,
        keys=operands.keys,
        args=operands.args,
    )

    # Digests were independently computed from literal typed-JSON tag contexts.
    assert client._calls == [
        (
            'evalsha',
            digest,
            4,
            'sgqc:v1:control',
            'sgqc:v1:counter:00000000000000000000000000000001',
            (
                'sgqc:v1:metadata:00000000000000000000000000000001:sgqc:v1:tag:'
                '962c92e30c43a1e101fa970f6565566fe5b296352619ee137feee5df11b49253'
            ),
            (
                'sgqc:v1:metadata:00000000000000000000000000000001:sgqc:v1:tag:'
                'dd3b567bdef699e237289c1fcca93eb296e101762d7e0ffdd21f5b66bdce5069'
            ),
            _RECORD_A,
            1234,
            '1',
            '0',
        )
    ]


@pytest.mark.parametrize('loaded', [b'0' * 40, b'not-a-digest'])
def test_script_cache_miss_rejects_invalid_loaded_digest_without_execution(loaded: bytes) -> None:
    client = _ScriptedClient([NoScriptError('missing'), loaded])

    with pytest.raises(BackendProtocolError):
        scripts_module.execute_script(
            client,
            script=scripts_module.LOOKUP_SCRIPT,
            sha=scripts_module.LOOKUP_SCRIPT_SHA,
            keys=('result-key',),
            args=(),
        )

    assert [call[0] for call in client._calls] == ['evalsha', 'script_load']


@pytest.mark.parametrize(
    ('replies', 'expected', 'commands'),
    [
        ([NoScriptError('missing'), b'0' * 40], BackendProtocolError, 2),
        ([NoScriptError('missing'), b'bad-digest'], BackendProtocolError, 2),
        (
            [
                NoScriptError('missing'),
                scripts_module.COMMIT_SCRIPT_SHA,
                NoScriptError('missing again'),
            ],
            BackendUnavailable,
            3,
        ),
        ([DataError('bad arguments')], BackendProtocolError, 1),
        ([ResponseError('OOM capacity')], BackendCapacity, 1),
        ([ValueError('unexpected client failure')], BackendUnavailable, 1),
        ([[b'unknown']], BackendProtocolError, 1),
        ([StaleIncarnation('stale')], StaleIncarnation, 1),
    ],
)
def test_commit_executor_preserves_error_classification_and_retry_bound(
    replies: list[object],
    expected: type[Exception],
    commands: int,
) -> None:
    client = _ScriptedClient(replies)
    with pytest.raises(expected):
        scripts_module.execute_commit(
            client,
            prefix='results',
            lease=FillLease('key', 'token'),
            payload=b'payload',
            deadline=1.0,
            clock=lambda: 0.0,
        )
    assert len(client._calls) == commands


@pytest.mark.parametrize(
    'error_type', [BackendCapacity, BackendProtocolError, BackendUnavailable, StaleIncarnation]
)
@pytest.mark.parametrize('cache_miss', [False, True])
def test_script_preserves_normalized_backend_errors(
    error_type: type[Exception], cache_miss: bool
) -> None:
    error = error_type('already normalized')
    replies: list[object] = (
        [NoScriptError('missing'), scripts_module.LOOKUP_SCRIPT_SHA] if cache_miss else []
    )
    client = _ScriptedClient([*replies, error])
    with pytest.raises(error_type) as caught:
        scripts_module.execute_script(
            client,
            script=scripts_module.LOOKUP_SCRIPT,
            sha=scripts_module.LOOKUP_SCRIPT_SHA,
            keys=('result-key',),
            args=(),
        )
    assert caught.value is error
