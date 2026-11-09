from collections.abc import Callable
from hashlib import sha1
from ssl import SSLError
from traceback import format_exception
from typing import TypeVar, cast

import pytest
from valkey.exceptions import (
    AuthenticationError,
    AuthenticationWrongNumberOfArgsError,
    AuthorizationError,
    DataError,
    NoPermissionError,
    NoScriptError,
    OutOfMemoryError,
    ResponseError,
)
from valkey.exceptions import ConnectionError as ValkeyConnectionError
from valkey.exceptions import TimeoutError as ValkeyTimeoutError

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
    _ScriptOperands,
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
from tests._valkey_control_cases import RECORD_A, RECORD_B

_T = TypeVar('_T')


class _ScriptedClient:
    def __init__(self, replies: list[object]) -> None:
        self.calls: list[tuple[object, ...]] = []
        self._replies = replies

    def evalsha(self, sha: str, numkeys: int, *keys_and_args: object) -> object:
        self.calls.append(('evalsha', sha, numkeys, *keys_and_args))
        reply = self._replies.pop(0)
        if isinstance(reply, BaseException):
            raise reply
        return reply

    def run_operation(self, operation: Callable[[scripts_module.ScriptClient], _T]) -> _T:
        return operation(self)

    def script_load(self, script: bytes | str) -> object:
        self.calls.append(('script_load', script))
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
    expected = (
        'Valkey metadata role is not writable (replica)'
        if isinstance(failure, BackendUnavailable)
        else str(failure)
    )
    assert str(caught.value) == expected
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

    assert [call[0] for call in client.calls] == ['evalsha', 'script_load', 'evalsha']


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
    assert client.calls == [
        ('evalsha', scripts_module.LOOKUP_SCRIPT_SHA, 1, 'result-key'),
        ('script_load', scripts_module.LOOKUP_SCRIPT),
        ('evalsha', scripts_module.LOOKUP_SCRIPT_SHA, 1, 'result-key'),
    ]


def test_translated_failure_exposes_no_client_detail() -> None:
    canary = 'credential-canary-9281'
    error = ValkeyTimeoutError(canary)
    error.add_note(canary)
    error.__dict__['client_detail'] = canary
    client = _ScriptedClient([error])
    with pytest.raises(BackendUnavailable) as caught:
        scripts_module.execute_script(
            client,
            script=scripts_module.LOOKUP_SCRIPT,
            sha=scripts_module.LOOKUP_SCRIPT_SHA,
            keys=('key',),
            args=(),
        )
    failure = caught.value
    assert failure.failure_class == 'timeout'
    assert canary not in str(failure)
    assert canary not in repr(vars(failure))
    assert not getattr(failure, '__notes__', ())
    assert failure.__cause__ is None
    assert failure.__context__ is None
    assert canary not in ''.join(format_exception(failure))
    traceback = failure.__traceback__
    while traceback is not None:
        if traceback.tb_frame.f_code.co_name == 'translate_client_error':
            assert all(value is not error for value in traceback.tb_frame.f_locals.values())
        traceback = traceback.tb_next


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
            lambda: control_create_operands('sgqc:v1:control', parse_record(RECORD_A, 'sgqc')),
            ('sgqc:v1:control', 'sgqc:v1:counter:00000000000000000000000000000001', RECORD_A),
        ),
        (
            'CONTROL_ROTATE',
            lambda: control_rotate_operands(
                'sgqc:v1:control', RECORD_A, parse_record(RECORD_B, 'sgqc')
            ),
            (
                'sgqc:v1:control',
                'sgqc:v1:counter:00000000000000000000000000000002',
                RECORD_A,
                RECORD_B,
            ),
        ),
        (
            'CONTROL_ROTATE',
            lambda: control_rotate_operands(
                'sgqc:v1:control', None, parse_record(RECORD_B, 'sgqc')
            ),
            (
                'sgqc:v1:control',
                'sgqc:v1:counter:00000000000000000000000000000002',
                b'',
                RECORD_B,
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
    build: Callable[[], _ScriptOperands],
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
    assert client.calls == [
        ('evalsha', digest, key_count, *expected),
        ('script_load', script),
        ('evalsha', digest, key_count, *expected),
    ]


@pytest.mark.parametrize('name', ['CHANGE', 'RESOLVE'])
def test_metadata_operands_keep_golden_tag_order_and_retention_flags(name: str) -> None:
    operands = metadata_operands(
        'sgqc',
        parse_record(RECORD_A, 'sgqc'),
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
    assert client.calls == [
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
            RECORD_A,
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

    assert [call[0] for call in client.calls] == ['evalsha', 'script_load']


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
    assert len(client.calls) == commands


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


@pytest.mark.parametrize(
    ('error', 'failure_class'),
    [
        (AuthenticationError('secret'), 'authentication'),
        (AuthenticationWrongNumberOfArgsError('secret'), 'authentication'),
        (AuthorizationError('secret'), 'authentication'),
        (NoPermissionError('secret'), 'authentication'),
        (ResponseError('WRONGPASS secret'), 'authentication'),
        (ResponseError('NOAUTH secret'), 'authentication'),
        (ResponseError('NOPERM secret'), 'authentication'),
        (ResponseError('ERR mentions WRONGPASS'), 'unknown'),
        (ResponseError('ERR mentions NOPERM'), 'unknown'),
        (ResponseError('WRONGPASS_SUFFIX secret'), 'unknown'),
        (ValkeyTimeoutError('secret'), 'timeout'),
        (ValkeyConnectionError('secret'), 'connection'),
        (PrimaryProofError(BackendUnavailable('Valkey metadata role is not writable')), 'replica'),
        (ValueError('secret'), 'unknown'),
    ],
)
def test_translate_client_error_classifies_failures(error: Exception, failure_class: str) -> None:
    with pytest.raises(BackendUnavailable) as caught:
        scripts_module.translate_client_error(error)
    assert caught.value.failure_class == failure_class
    if failure_class == 'replica':
        message = 'Valkey metadata role is not writable'
    elif failure_class == 'unknown':
        message = 'Valkey operation failed'
    else:
        message = 'Valkey operation is unavailable'
    assert str(caught.value) == f'{message} ({failure_class})'
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None


@pytest.mark.parametrize('chain', ['cause', 'context'])
def test_translate_client_error_classifies_tls_chain(chain: str) -> None:
    error = ValkeyConnectionError('secret')
    tls = SSLError('certificate secret')
    if chain == 'cause':
        error.__cause__ = tls
    else:
        error.__context__ = tls
    with pytest.raises(BackendUnavailable) as caught:
        scripts_module.translate_client_error(error)
    assert caught.value.failure_class == 'tls'
    assert str(caught.value) == 'Valkey operation is unavailable (tls)'
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None
