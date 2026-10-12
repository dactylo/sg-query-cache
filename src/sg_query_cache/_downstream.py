"""Absorb closure of a downstream reader of standard output."""

import contextlib
import errno
import os
import sys
from collections.abc import Generator, Iterable
from typing import TextIO, cast

_WINDOWS = sys.platform == 'win32'


def _is_downstream_closure(error: OSError, *, windows: bool | None = None) -> bool:
    """Recognize closed-reader errors, including Windows pipe `EINVAL`."""
    on_windows = _WINDOWS if windows is None else windows
    return isinstance(error, BrokenPipeError) or (on_windows and error.errno == errno.EINVAL)


class _ClosableTextOutput:
    """Discard writes after a downstream reader closes."""

    def __init__(self, stream: TextIO, *, windows: bool | None = None) -> None:
        """Wrap a text stream and select closed-pipe error handling."""
        self._closed_downstream = False
        self._stream = stream
        self._windows = windows

    def __getattr__(self, name: str) -> object:
        """Return an attribute of the wrapped stream."""
        return getattr(self._stream, name)

    @property
    def _closed_by_downstream(self) -> bool:
        """Whether the downstream reader has closed the stream."""
        return self._closed_downstream

    def flush(self) -> None:
        """Flush the wrapped stream unless its reader has closed.

        Raises:
            OSError: If flushing fails for a reason other than closure.
        """
        if self._closed_downstream:
            return
        try:
            self._stream.flush()
        except OSError as error:
            if not _is_downstream_closure(error, windows=self._windows):
                raise
            self._closed_downstream = True

    def write(self, text: str) -> int:
        """Write text, or discard it after the reader has closed.

        Args:
            text: Text to write.

        Returns:
            The number of characters accepted.

        Raises:
            OSError: If writing fails for a reason other than closure.
        """
        if self._closed_downstream:
            return len(text)
        try:
            return self._stream.write(text)
        except OSError as error:
            if not _is_downstream_closure(error, windows=self._windows):
                raise
            self._closed_downstream = True
            return len(text)

    def writelines(self, lines: Iterable[str]) -> None:
        """Write lines until the reader closes, then stop consuming them.

        Args:
            lines: Text lines in output order.

        Raises:
            OSError: If writing fails for a reason other than closure.
        """
        if self._closed_downstream:
            return
        for line in lines:
            self.write(line)
            if self._closed_downstream:
                return


def _null_text_stream() -> TextIO:
    """Open a text stream on the null device."""
    sink = os.open(os.devnull, os.O_WRONLY)
    try:
        return os.fdopen(sink, 'w', encoding='utf-8')
    except BaseException:
        os.close(sink)
        raise


def _silence_descriptor(stream: TextIO) -> bool:
    """Point an available stream descriptor at the null device."""
    try:
        target = stream.fileno()
    except (OSError, ValueError):
        return False
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, target)
    finally:
        os.close(devnull)
    return True


@contextlib.contextmanager
def closable_stdout() -> Generator[_ClosableTextOutput, None, None]:
    """Route standard output through a closure-absorbing wrapper.

    On every exit, including `SystemExit` from argument parsing, pending output
    is flushed through the wrapper and the original stream is restored. If the
    reader closed, the original descriptor is pointed at the null device, or a
    stream without a descriptor is replaced by a null-device stream, so the
    interpreter's final flush cannot fail again.

    Yields:
        The installed wrapper.
    """
    original = sys.stdout
    output = _ClosableTextOutput(original)
    sys.stdout = cast('TextIO', output)
    try:
        yield output
    finally:
        try:
            output.flush()
        finally:
            sys.stdout = original
            if output._closed_by_downstream and not _silence_descriptor(original):  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
                # The interpreter still flushes a stream without a descriptor
                # at shutdown, so replace it with one that cannot fail.
                sys.stdout = _null_text_stream()
