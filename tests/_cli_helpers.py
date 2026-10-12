import contextlib
import io
import os
import re
import signal
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Protocol, TypeVar
from unittest import mock

from sg_query_cache._cli import main

# Hang guard, not a performance assertion. Allow for scheduling on shared CI hosts.
CHILD_TIMEOUT_SECONDS = 300
_COLLECT_AFTER_KILL_SECONDS = 10
_CONTROL_CHARACTERS = re.compile(r'[\x00-\x08\x0b-\x1f\x7f-\x9f]')
_ENV = {**os.environ, 'PYTHON_COLORS': '0'}
_Output = TypeVar('_Output', str, bytes)
PROCESS_GROUP_FLAGS = subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == 'win32' else 0
_TRANSCRIPT_FIELD_CHARACTERS = 2048
_TRANSCRIPT_HEAD_BYTES = 2048
_TRANSCRIPT_TAIL_BYTES = 6144


class _Outcome(Protocol):
    @property
    def args(self) -> object: ...

    @property
    def returncode(self) -> int | None: ...

    @property
    def stdout(self) -> str | bytes | None: ...

    @property
    def stderr(self) -> str | bytes | None: ...


def _describe_status(returncode: int | None) -> str:
    if returncode is None:
        return 'not collected'
    if returncode < 0:
        try:
            return f'{returncode} ({signal.Signals(-returncode).name})'
        except ValueError:
            return str(returncode)
    return str(returncode)


def _escape_transcript_line(line: str) -> str:
    line = _CONTROL_CHARACTERS.sub(lambda match: f'\\x{ord(match.group()):02x}', line)
    line = line.replace('##[', '#\\#[')
    body = line.lstrip()
    if body.startswith('::'):
        line = line[: len(line) - len(body)] + '\\' + body
    return line


def _exit_status(error: SystemExit) -> int:
    code = error.code
    if code is None:
        return 0
    return code if isinstance(code, int) else 1


def _terminate_tree(process: subprocess.Popen[_Output]) -> None:
    if sys.platform == 'win32':
        with contextlib.suppress(OSError, subprocess.TimeoutExpired):
            subprocess.run(
                ['taskkill', '/F', '/T', '/PID', str(process.pid)],
                capture_output=True,
                timeout=_COLLECT_AFTER_KILL_SECONDS,
                check=False,
            )
    else:
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(process.pid, signal.SIGKILL)
    with contextlib.suppress(OSError):
        process.kill()


def _transcript_field(label: str, value: object) -> str:
    field = repr(value)
    if len(field) > _TRANSCRIPT_FIELD_CHARACTERS:
        omitted = len(field) - _TRANSCRIPT_FIELD_CHARACTERS
        field = f'{field[:_TRANSCRIPT_FIELD_CHARACTERS]} [... {omitted} characters omitted ...]'
    return f'{label}: {_escape_transcript_line(field)}'


def _transcript_stream(label: str, data: str | bytes | None) -> list[str]:
    if data is None:
        return [f'{label}: <not collected>']
    raw = data if isinstance(data, bytes) else data.encode('utf-8', 'backslashreplace')
    if not raw:
        return [f'{label}: <empty>']
    if len(raw) > _TRANSCRIPT_HEAD_BYTES + _TRANSCRIPT_TAIL_BYTES:
        omitted = len(raw) - _TRANSCRIPT_HEAD_BYTES - _TRANSCRIPT_TAIL_BYTES
        field = (
            raw[:_TRANSCRIPT_HEAD_BYTES].decode('utf-8', 'backslashreplace')
            + f'\n[... {omitted} bytes omitted ...]\n'
            + raw[-_TRANSCRIPT_TAIL_BYTES:].decode('utf-8', 'backslashreplace')
        )
    else:
        field = raw.decode('utf-8', 'backslashreplace')
    body = field.replace('\r\n', '\n').removesuffix('\n')
    return [f'{label}:', *(f'  | {_escape_transcript_line(line)}' for line in body.split('\n'))]


def invoke_cli(
    argv: Sequence[str],
) -> tuple[int, str, str]:
    stdout_buffer = io.BytesIO()
    stderr_buffer = io.BytesIO()
    stdout = io.TextIOWrapper(stdout_buffer, encoding='utf-8', newline='')
    stderr = io.TextIOWrapper(
        stderr_buffer, encoding='utf-8', errors='backslashreplace', newline=''
    )
    with contextlib.ExitStack() as stack:
        stack.enter_context(mock.patch.dict(os.environ, {'COLUMNS': '80', 'PYTHON_COLORS': '0'}))
        stack.enter_context(contextlib.redirect_stdout(stdout))
        stack.enter_context(contextlib.redirect_stderr(stderr))
        try:
            status = main(argv)
        except SystemExit as error:
            status = _exit_status(error)
        stdout.flush()
        stderr.flush()
    return (
        status,
        stdout_buffer.getvalue().decode('utf-8'),
        stderr_buffer.getvalue().decode('utf-8'),
    )


def process_transcript(
    result: _Outcome,
    *,
    cwd: Path | str | None = None,
    note: str | None = None,
) -> str:
    lines = [_transcript_field('command', result.args)]
    if cwd is not None:
        lines.append(_transcript_field('cwd', str(cwd)))
    if note is not None:
        lines.append(f'note: {_escape_transcript_line(note)}')
    lines.append(f'status: {_describe_status(result.returncode)}')
    lines.extend(_transcript_stream('stdout', result.stdout))
    lines.extend(_transcript_stream('stderr', result.stderr))
    return '\n'.join(lines)


def _kill_and_describe(
    process: subprocess.Popen[_Output],
    *,
    cwd: Path | str | None,
    reason: str,
) -> str:
    _terminate_tree(process)
    try:
        stdout, stderr = process.communicate(timeout=_COLLECT_AFTER_KILL_SECONDS)
    except subprocess.TimeoutExpired:
        stdout = stderr = None
        reason = f'{reason}; a descendant outside the process group still holds the output pipes'
        with contextlib.suppress(subprocess.TimeoutExpired):
            process.wait(timeout=_COLLECT_AFTER_KILL_SECONDS)
    killed = subprocess.CompletedProcess(process.args, process.returncode, stdout, stderr)
    return process_transcript(killed, cwd=cwd, note=reason)


def finish_process(
    process: subprocess.Popen[_Output],
    *,
    stdin: _Output | None = None,
    timeout: float = CHILD_TIMEOUT_SECONDS,
    cwd: Path | str | None = None,
) -> subprocess.CompletedProcess[_Output]:
    try:
        stdout, stderr = process.communicate(stdin, timeout=timeout)
    except subprocess.TimeoutExpired:
        raise AssertionError(
            _kill_and_describe(process, cwd=cwd, reason=f'timed out after {timeout:g} s')
        ) from None
    return subprocess.CompletedProcess(process.args, process.returncode, stdout, stderr)


def run_cli(
    argv: Sequence[str],
    *,
    expect: int | None = None,
    cwd: Path | str | None = None,
    env: Mapping[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    process = subprocess.Popen(
        [sys.executable, '-m', 'sg_query_cache', *argv],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding='utf-8',
        env=dict(_ENV if env is None else env),
        cwd=cwd,
        start_new_session=True,
        creationflags=PROCESS_GROUP_FLAGS,
    )
    result = finish_process(process, timeout=CHILD_TIMEOUT_SECONDS, cwd=cwd)
    if expect is not None and result.returncode != expect:
        raise AssertionError(f'expected status {expect}\n{process_transcript(result, cwd=cwd)}')
    return result


def run_setup(
    command: Sequence[str],
    *,
    cwd: Path | str | None = None,
) -> subprocess.CompletedProcess[bytes]:
    process = subprocess.Popen(
        list(command),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=cwd,
        start_new_session=True,
        creationflags=PROCESS_GROUP_FLAGS,
    )
    result = finish_process(process, timeout=CHILD_TIMEOUT_SECONDS, cwd=cwd)
    if result.returncode != 0:
        raise AssertionError(f'setup command failed\n{process_transcript(result, cwd=cwd)}')
    return result


def run_with_closed_output(
    argv: Sequence[str],
) -> subprocess.CompletedProcess[bytes]:
    read_fd, write_fd = os.pipe()
    os.close(read_fd)  # every write to the closed stream happens after its reader closed
    try:
        process = subprocess.Popen(
            [sys.executable, '-m', 'sg_query_cache', *argv],
            stdin=subprocess.DEVNULL,
            stdout=write_fd,
            stderr=subprocess.PIPE,
            env=_ENV,
            start_new_session=True,
            creationflags=PROCESS_GROUP_FLAGS,
        )
    finally:
        os.close(write_fd)
    return finish_process(process, timeout=CHILD_TIMEOUT_SECONDS)
