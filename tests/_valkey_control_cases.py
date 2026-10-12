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

_ACTIVE_CONTROL = object()

_COUNTER_A = 'sgqc:v1:counter:00000000000000000000000000000001'
_COUNTER_B = 'sgqc:v1:counter:00000000000000000000000000000002'
_RECORD_A = (
    b'primary|00000000000000000000000000000001|sgqc:v1:counter:00000000000000000000000000000001'
)
_RECORD_B = (
    b'primary|00000000000000000000000000000002|sgqc:v1:counter:00000000000000000000000000000002'
)


@dataclass(frozen=True)
class _ControlEffect:
    _name: str
    _reply: object = None
    _epoch_after: tuple[int, int, object | None] | None = None


@dataclass(frozen=True)
class _ControlTrace:
    _name: str
    _effects: tuple[_ControlEffect, ...]
    _acknowledgments: tuple[bytes | type[Exception], ...]
    _final_record: bytes | None
    _final_revision: int
    _final_epoch: tuple[int, int, object | None] | None = (0, 0, None)
    _proposal_nonce: str = '00000000000000000000000000000001'


@dataclass(frozen=True)
class _PublicationTrace:
    _name: str
    _transition: Literal['clear', 'publish']
    _acknowledgments: tuple[bytes | type[Exception], bytes | type[Exception]]
    _final_record: bytes | None
    _final_revision: int
    _initial_record: bytes = _RECORD_A
    _replacement_record: bytes = _RECORD_B
    _effects: tuple[str, ...] = (
        'older-operation-entered',
        'newer-operation-entered',
        'newer-settled',
        'older-settled',
    )

    _older_joins: bool = False
    _final_covered: frozenset[int] | None = None


def _publication_traces() -> tuple[_PublicationTrace, ...]:  # pyright: ignore[reportUnusedFunction]
    return (
        _PublicationTrace(
            'old-validation-after-newer-clear', 'clear', (_RECORD_A, StaleIncarnation), None, 1
        ),
        _PublicationTrace(
            'overlapping-publications', 'publish', (_RECORD_A, _RECORD_B), _RECORD_B, 1
        ),
        _PublicationTrace(
            'late-coverage-after-newer-clear',
            'clear',
            (_RECORD_A, StaleIncarnation),
            None,
            2,
            _older_joins=True,
            _final_covered=frozenset(),
        ),
        _PublicationTrace(
            'late-coverage-after-newer-publish',
            'publish',
            (_RECORD_A, _RECORD_B),
            _RECORD_B,
            2,
            _older_joins=True,
            _final_covered=frozenset({3}),
        ),
    )


def _validation(record: bytes = _RECORD_A) -> tuple[_ControlEffect, ...]:
    return (
        _ControlEffect('info', {'role': 'master', 'master_replid': 'primary'}),
        _ControlEffect('get', record),
        _ControlEffect('exists', 1),
        _ControlEffect('info', {'role': 'master', 'master_replid': 'primary'}),
        _ControlEffect('get', record),
        _ControlEffect('exists', 1),
    )


class _OrderedClient:
    def __init__(self, replies: list[tuple[str, object]]) -> None:
        self._calls: list[tuple[object, ...]] = []
        self._control_record: bytes | None = None
        self._replies = replies

    def _reply(self, method: str, call: tuple[object, ...]) -> object:
        self._calls.append(call)
        expected, reply = self._replies.pop(0)
        assert expected == method
        if reply is _ACTIVE_CONTROL:
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

    def client(self) -> '_OrderedClient':
        return self

    def close(self) -> None:
        self._calls.append(('close',))

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
    def pin(self) -> Generator['_OrderedClient']:
        yield self

    def refresh(self) -> None:
        return None

    def refresh_endpoint(self) -> None:
        return None

    def run_operation(self, operation: Callable[[ScriptClient], _T]) -> _T:
        return operation(self)

    def script_load(self, script: bytes | str) -> object:
        return self._reply('script_load', ('script_load', script))


def _control_traces() -> tuple[_ControlTrace, ...]:  # pyright: ignore[reportUnusedFunction]
    begin = (_ControlEffect('refresh_endpoint'), _ControlEffect('pin'))
    operation = (_ControlEffect('operation', [b'ok', 1, 0]),)
    end = (_ControlEffect('refresh_endpoint'), _ControlEffect('unpin'))
    stable = begin + _validation() + operation + end
    missing = (
        _ControlEffect('info', {'role': 'master', 'master_replid': 'primary'}),
        _ControlEffect('get', None),
        _ControlEffect('create', [b'created', _RECORD_A]),
        *_validation()[3:],
    )
    counter = (
        *_validation()[:2],
        _ControlEffect('exists', 0),
        _ControlEffect('rotate', [b'rotated', _RECORD_B]),
        *_validation(_RECORD_B)[3:],
    )
    loser = (
        *_validation()[:2],
        _ControlEffect('exists', 0),
        _ControlEffect('rotate', [b'winner', _RECORD_B]),
        *_validation(_RECORD_B),
    )
    traces = [
        _ControlTrace(
            'cached-validation',
            stable + begin + operation + end,
            (_RECORD_A, _RECORD_A),
            _RECORD_A,
            1,
        ),
        _ControlTrace(
            'missing-record', begin + missing + operation + end, (_RECORD_A,), _RECORD_A, 1
        ),
        _ControlTrace(
            'missing-counter',
            begin + counter + operation + end,
            (_RECORD_B,),
            _RECORD_B,
            1,
            _proposal_nonce='00000000000000000000000000000002',
        ),
        _ControlTrace(
            'rotation-race-loser', begin + loser + operation + end, (_RECORD_B,), _RECORD_B, 1
        ),
        _ControlTrace(
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
                _ControlTrace(
                    f'pre-operation-{name}-drift',
                    begin
                    + _validation()[:-1]
                    + (
                        _ControlEffect('exists', 1, epoch),
                        _ControlEffect('unpin'),
                        _ControlEffect('refresh'),
                    )
                    + stable,
                    (_RECORD_A,),
                    _RECORD_A,
                    2,
                    epoch,
                ),
                _ControlTrace(
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
