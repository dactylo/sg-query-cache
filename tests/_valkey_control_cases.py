from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Literal, TypeVar, cast

from sg_query_cache._backend import BackendUnavailable, StaleIncarnation
from sg_query_cache._valkey_scripts import (
    CONTROL_CREATE_SCRIPT_SHA,
    CONTROL_ROTATE_SCRIPT_SHA,
    ScriptClient,
)

_T = TypeVar('_T')

ACTIVE_CONTROL = object()

COUNTER_A = 'sgqc:v1:counter:00000000000000000000000000000001'
COUNTER_B = 'sgqc:v1:counter:00000000000000000000000000000002'
RECORD_A = (
    b'primary|00000000000000000000000000000001|sgqc:v1:counter:00000000000000000000000000000001'
)
RECORD_B = (
    b'primary|00000000000000000000000000000002|sgqc:v1:counter:00000000000000000000000000000002'
)


@dataclass(frozen=True)
class _ControlEffect:
    name: str
    reply: object = None
    epoch_after: tuple[int, int, object | None] | None = None


@dataclass(frozen=True)
class ControlTrace:
    name: str
    effects: tuple[_ControlEffect, ...]
    acknowledgments: tuple[bytes | type[Exception], ...]
    final_record: bytes | None
    final_revision: int
    final_epoch: tuple[int, int, object | None] | None = (0, 0, None)
    proposal_nonce: str = '00000000000000000000000000000001'


@dataclass(frozen=True)
class PublicationTrace:
    name: str
    transition: Literal['clear', 'publish']
    acknowledgments: tuple[bytes | type[Exception], bytes | type[Exception]]
    final_record: bytes | None
    final_revision: int
    initial_record: bytes = RECORD_A
    replacement_record: bytes = RECORD_B
    effects: tuple[str, ...] = (
        'older-operation-entered',
        'newer-operation-entered',
        'newer-settled',
        'older-settled',
    )

    older_joins: bool = False
    final_covered: frozenset[int] | None = None


def _validation(record: bytes = RECORD_A) -> tuple[_ControlEffect, ...]:
    return (
        _ControlEffect('info', {'role': 'master', 'master_replid': 'primary'}),
        _ControlEffect('get', record),
        _ControlEffect('exists', 1),
        _ControlEffect('info', {'role': 'master', 'master_replid': 'primary'}),
        _ControlEffect('get', record),
        _ControlEffect('exists', 1),
    )


class OrderedClient:
    def __init__(self, replies: list[tuple[str, object]]) -> None:
        self.calls: list[tuple[object, ...]] = []
        self._control_record: bytes | None = None
        self.replies = replies

    def _reply(self, method: str, call: tuple[object, ...]) -> object:
        self.calls.append(call)
        expected, reply = self.replies.pop(0)
        assert expected == method
        if reply is ACTIVE_CONTROL:
            assert self._control_record is not None
            reply = self._control_record
        if callable(reply):
            reply = cast('Callable[[tuple[object, ...]], object]', reply)(call)
        if isinstance(reply, BaseException):
            raise reply
        if method == 'evalsha' and isinstance(reply, list | tuple):
            items = tuple(cast('list[object] | tuple[object, ...]', reply))
            if len(items) == 2:
                status, record = items
                if status in (b'created', b'rotated', 'created', 'rotated') and isinstance(
                    record, bytes
                ):
                    self._control_record = record
        return cast('object', reply)

    def client(self) -> 'OrderedClient':
        return self

    def close(self) -> None:
        self.calls.append(('close',))

    def epoch(self) -> tuple[int, int, None]:
        return 0, 0, None

    def evalsha(self, sha: str, numkeys: int, *keys_and_args: object) -> object:
        return self._reply('evalsha', ('evalsha', sha, numkeys, *keys_and_args))

    def exists(self, key: str) -> object:
        return self._reply('exists', ('exists', key))

    def get(self, key: str) -> object:
        return self._reply('get', ('get', key))

    def info(self, section: str) -> object:
        return self._reply('info', ('info', section))

    @contextmanager
    def pin(self) -> Generator['OrderedClient']:
        yield self

    def refresh(self) -> None:
        return None

    def refresh_endpoint(self) -> None:
        return None

    def run_operation(self, operation: Callable[[ScriptClient], _T]) -> _T:
        return operation(self)

    def script_load(self, script: bytes | str) -> object:
        return self._reply('script_load', ('script_load', script))


def control_reply(
    sha: str,
    numkeys: int,
    keys_and_args: tuple[object, ...],
    control: bytes | None,
    counters: set[str],
) -> tuple[bytes | None, object]:
    assert numkeys == 2
    _control_key, counter_key, *args = keys_and_args
    assert isinstance(counter_key, str)
    if sha == CONTROL_CREATE_SCRIPT_SHA:
        record = cast('bytes', args[0])
        if control is not None:
            return control, [b'winner', control]
        counters.add(counter_key)
        return record, [b'created', record]
    assert sha == CONTROL_ROTATE_SCRIPT_SHA
    observed, replacement = cast('tuple[bytes, bytes]', tuple(args))
    matches = control is None if observed == b'' else control == observed
    if not matches:
        return control, [b'winner', control or b'']
    counters.add(counter_key)
    return replacement, [b'rotated', replacement]


def control_traces() -> tuple[ControlTrace, ...]:
    begin = (_ControlEffect('refresh_endpoint'), _ControlEffect('pin'))
    operation = (_ControlEffect('operation', [b'ok', 1, 0]),)
    end = (_ControlEffect('refresh_endpoint'), _ControlEffect('unpin'))
    stable = begin + _validation() + operation + end
    missing = (
        _ControlEffect('info', {'role': 'master', 'master_replid': 'primary'}),
        _ControlEffect('get', None),
        _ControlEffect('create', [b'created', RECORD_A]),
        *_validation()[3:],
    )
    counter = (
        *_validation()[:2],
        _ControlEffect('exists', 0),
        _ControlEffect('rotate', [b'rotated', RECORD_B]),
        *_validation(RECORD_B)[3:],
    )
    loser = (
        *_validation()[:2],
        _ControlEffect('exists', 0),
        _ControlEffect('rotate', [b'winner', RECORD_B]),
        *_validation(RECORD_B),
    )
    traces = [
        ControlTrace(
            'cached-validation',
            stable + begin + operation + end,
            (RECORD_A, RECORD_A),
            RECORD_A,
            1,
        ),
        ControlTrace(
            'missing-record', begin + missing + operation + end, (RECORD_A,), RECORD_A, 1
        ),
        ControlTrace(
            'missing-counter',
            begin + counter + operation + end,
            (RECORD_B,),
            RECORD_B,
            1,
            proposal_nonce='00000000000000000000000000000002',
        ),
        ControlTrace(
            'rotation-race-loser', begin + loser + operation + end, (RECORD_B,), RECORD_B, 1
        ),
        ControlTrace(
            'stale-script-status',
            begin + _validation() + (_ControlEffect('operation', [b'stale']),) + end,
            (StaleIncarnation,),
            None,
            1,
            None,
        ),
    ]
    for name, epoch in (
        ('reconnect', (0, 1, None)),
        ('pool', (1, 0, None)),
        ('endpoint', (0, 0, ('new-primary', 6379))),
    ):
        traces.extend(
            (
                ControlTrace(
                    f'pre-operation-{name}-drift',
                    begin
                    + _validation()[:-1]
                    + (
                        _ControlEffect('exists', 1, epoch),
                        _ControlEffect('unpin'),
                        _ControlEffect('refresh'),
                    )
                    + stable,
                    (RECORD_A,),
                    RECORD_A,
                    2,
                    epoch,
                ),
                ControlTrace(
                    f'post-operation-{name}-drift-no-mutation-retry',
                    begin
                    + _validation()
                    + (
                        _ControlEffect('operation', [b'ok', 1, 0], epoch),
                        _ControlEffect('refresh_endpoint'),
                        _ControlEffect('refresh'),
                        _ControlEffect('unpin'),
                    ),
                    (BackendUnavailable,),
                    None,
                    1,
                    None,
                ),
            )
        )
    return tuple(traces)


def publication_traces() -> tuple[PublicationTrace, ...]:
    return (
        PublicationTrace(
            'old-validation-after-newer-clear', 'clear', (RECORD_A, StaleIncarnation), None, 1
        ),
        PublicationTrace('overlapping-publications', 'publish', (RECORD_A, RECORD_B), RECORD_B, 1),
        PublicationTrace(
            'late-coverage-after-newer-clear',
            'clear',
            (RECORD_A, StaleIncarnation),
            None,
            2,
            older_joins=True,
            final_covered=frozenset(),
        ),
        PublicationTrace(
            'late-coverage-after-newer-publish',
            'publish',
            (RECORD_A, RECORD_B),
            RECORD_B,
            2,
            older_joins=True,
            final_covered=frozenset({3}),
        ),
    )
