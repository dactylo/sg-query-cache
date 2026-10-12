from tests._cli_helpers import process_transcript, run_with_closed_output


def test_closed_stdout_before_help_is_quiet() -> None:
    result = run_with_closed_output(['--help'])
    assert (result.returncode, result.stderr) == (0, b''), process_transcript(result)


def test_closed_stdout_during_plugin_path_keeps_success() -> None:
    result = run_with_closed_output(['daemon-plugin-path'])
    assert (result.returncode, result.stderr) == (0, b''), process_transcript(result)
