import errno
import io
import os
import sys
from collections.abc import Iterator

import pytest

from sg_query_cache import _downstream
from sg_query_cache._downstream import (
    _ClosableTextOutput,
    _is_downstream_closure,
    _silence_descriptor,
    closable_stdout,
)


class _FailingStream(io.StringIO):
    def __init__(self, error: OSError) -> None:
        super().__init__()
        self.error = error
        self.attempts = 0

    def flush(self) -> None:
        raise self.error

    def write(self, _text: str) -> int:
        self.attempts += 1
        raise self.error


@pytest.mark.parametrize('windows', [False, True])
def test_broken_pipe_is_closure_everywhere(windows: bool) -> None:
    assert _is_downstream_closure(BrokenPipeError(), windows=windows)


def test_closable_stdout_flushes_when_leaving_by_system_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    silenced: list[object] = []

    class _BufferedClosedStream(io.StringIO):
        def flush(self) -> None:
            raise BrokenPipeError

    stream = _BufferedClosedStream()

    def record(closed: object) -> bool:
        silenced.append(closed)
        return True

    monkeypatch.setattr(_downstream, '_silence_descriptor', record)
    monkeypatch.setattr(sys, 'stdout', stream)

    def exit_after_help() -> None:
        print('usage: buffered help text')
        raise SystemExit(0)

    with pytest.raises(SystemExit), closable_stdout():
        exit_after_help()
    assert silenced == [stream]


def test_closable_stdout_replaces_a_closed_stream_without_a_descriptor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _ClosedNoDescriptor(io.StringIO):
        flushes = 0

        def flush(self) -> None:
            self.flushes += 1
            raise BrokenPipeError

    stream = _ClosedNoDescriptor()
    monkeypatch.setattr(sys, 'stdout', stream)

    with closable_stdout():
        print('help text')

    replacement = sys.stdout
    assert replacement is not stream
    assert stream.getvalue() == 'help text\n'
    assert stream.flushes == 1
    replacement.write('discarded')
    replacement.flush()
    replacement.close()


def test_closable_stdout_restores_the_original_stream() -> None:
    original = sys.stdout
    with closable_stdout() as output:
        assert sys.stdout is output
        assert not output._closed_by_downstream
    assert sys.stdout is original


def test_closable_stdout_silences_a_closed_descriptor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    silenced: list[object] = []

    def record(stream: object) -> bool:
        silenced.append(stream)
        return True

    monkeypatch.setattr(_downstream, '_silence_descriptor', record)
    monkeypatch.setattr(sys, 'stdout', _FailingStream(BrokenPipeError()))
    original = sys.stdout

    with closable_stdout() as output:
        print('discarded')
        assert output._closed_by_downstream
    assert silenced == [original]


def test_einval_is_closure_only_on_windows() -> None:
    error = OSError(errno.EINVAL, 'Invalid argument')
    assert _is_downstream_closure(error, windows=True)
    assert not _is_downstream_closure(error, windows=False)


def test_flush_latches_windows_einval() -> None:
    output = _ClosableTextOutput(_FailingStream(OSError(errno.EINVAL, 'closed')), windows=True)

    output.flush()
    output.flush()
    assert output._closed_by_downstream


def test_flush_propagates_unrelated_errno() -> None:
    output = _ClosableTextOutput(_FailingStream(OSError(errno.EBADF, 'bad')), windows=True)

    with pytest.raises(OSError, match='bad'):
        output.flush()


def test_other_attributes_come_from_the_wrapped_stream() -> None:
    stream = io.StringIO()
    output = _ClosableTextOutput(stream, windows=False)

    assert output.getvalue is not None
    output.write('kept')
    assert stream.getvalue() == 'kept'


def test_silence_descriptor_reports_streams_without_descriptors() -> None:
    assert not _silence_descriptor(io.StringIO())


def test_unrelated_errno_is_not_closure() -> None:
    assert not _is_downstream_closure(OSError(errno.EBADF, 'bad'), windows=True)


@pytest.mark.parametrize(
    ('error', 'windows'),
    [(BrokenPipeError(), False), (OSError(errno.EINVAL, 'closed'), True)],
)
def test_write_latches_closure_and_discards_later_text(error: OSError, windows: bool) -> None:
    stream = _FailingStream(error)
    output = _ClosableTextOutput(stream, windows=windows)

    assert output.write('first') == len('first')
    assert output.write('second') == len('second')
    assert output._closed_by_downstream
    assert stream.attempts == 1


@pytest.mark.parametrize(
    ('error', 'windows'),
    [(OSError(errno.EINVAL, 'closed'), False), (OSError(errno.EBADF, 'bad'), True)],
)
def test_write_propagates_errors_that_are_not_closure(error: OSError, windows: bool) -> None:
    output = _ClosableTextOutput(_FailingStream(error), windows=windows)

    with pytest.raises(OSError, match=r'closed|bad') as raised:
        output.write('text')
    assert raised.value is error
    assert not output._closed_by_downstream


class _FakeOs:
    O_WRONLY = os.O_WRONLY
    devnull = os.devnull

    def __init__(self, *, redirect_error: OSError | None) -> None:
        self.events: list[tuple[str, int]] = []
        self.redirect_error = redirect_error

    def close(self, fd: int) -> None:
        self.events.append(('close', fd))

    def dup2(self, source: int, _target: int) -> None:
        self.events.append(('dup2', source))
        if self.redirect_error is not None:
            raise self.redirect_error

    def open(self, _path: str, _flags: int) -> int:
        return 37


class _DescriptorStream(io.StringIO):
    def fileno(self) -> int:
        return 1


def test_silence_descriptor_closes_devnull_and_propagates_redirect_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeOs(redirect_error=OSError('injected redirect failure'))
    monkeypatch.setattr(_downstream, 'os', fake)

    with pytest.raises(OSError, match='injected redirect failure'):
        _silence_descriptor(_DescriptorStream())
    assert fake.events == [('dup2', 37), ('close', 37)]


def test_silence_descriptor_redirects_a_descriptor(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeOs(redirect_error=None)
    monkeypatch.setattr(_downstream, 'os', fake)

    assert _silence_descriptor(_DescriptorStream())
    assert fake.events == [('dup2', 37), ('close', 37)]


def test_writelines_stops_consuming_after_closure() -> None:
    consumed: list[int] = []

    def lines() -> Iterator[str]:
        for index in range(5):
            consumed.append(index)
            yield f'{index}\n'

    output = _ClosableTextOutput(_FailingStream(BrokenPipeError()), windows=False)
    output.writelines(lines())
    assert consumed == [0]

    consumed.clear()
    output.writelines(lines())
    assert consumed == []
