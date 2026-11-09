import textwrap
import tomllib
from pathlib import Path

from tests._cli_helpers import process_transcript, run_python, run_with_closed_output


def test_closed_stdout_before_help_is_quiet() -> None:
    result = run_with_closed_output(['--help'])
    assert (result.returncode, result.stderr) == (0, b''), process_transcript(result)


def test_closed_stdout_during_plugin_path_keeps_success() -> None:
    result = run_with_closed_output(['daemon-plugin-path'])
    assert (result.returncode, result.stderr) == (0, b''), process_transcript(result)


def test_console_script_matches_the_module_help_without_installation() -> None:
    with (Path(__file__).parents[1] / 'pyproject.toml').open('rb') as stream:
        project = tomllib.load(stream)
    target = project['project']['scripts']['sg-query-cache']
    assert target == 'sg_query_cache._cli:main'

    console = run_python(
        '-c',
        textwrap.dedent(
            """
            import importlib
            import sys

            module_name, attribute = sys.argv[1].split(':', 1)
            command = getattr(importlib.import_module(module_name), attribute)
            raise SystemExit(command(['--help']))
            """
        ),
        target,
    )
    module = run_python('-m', 'sg_query_cache', '--help')

    assert console.returncode == 0, console.stderr
    assert module.returncode == 0, module.stderr
    assert module.stdout.startswith('usage: sg-query-cache [-h] [--version] COMMAND ...\n')
    assert console.stdout == module.stdout
