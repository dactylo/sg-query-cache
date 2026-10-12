"""Provide redacted operator and recovery commands."""

from __future__ import annotations

import argparse
import os
import re
import sys
from hashlib import sha256
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING, cast

from sg_query_cache._backend import BackendProtocolError
from sg_query_cache._backend_validation import flush_reason
from sg_query_cache._config import CacheSettings
from sg_query_cache._downstream import closable_stdout
from sg_query_cache._metrics import NullMetrics
from sg_query_cache._model import DependencyTag

if TYPE_CHECKING:
    from collections.abc import Sequence

    from sg_query_cache.valkey import ValkeyMetadataStore

_COMMAND_GROUPS: tuple[tuple[str, tuple[tuple[str, str], ...]], ...] = (
    (
        'commands',
        (
            ('flush', 'invalidate cached results for one SG site'),
            ('rotate-incarnation', 'invalidate cached results for all sites'),
            ('daemon-plugin-path', 'print the installed SG event-daemon plugin path'),
        ),
    ),
)
_COMMAND_HELP = {name: text for _title, commands in _COMMAND_GROUPS for name, text in commands}
_CONTROL_CHARACTERS = re.compile(r'[\x00-\x1f\x7f-\x9f]')


class _RootParser(argparse.ArgumentParser):
    """Render options first and top-level commands under titled groups."""

    def format_help(self, formatter: argparse.HelpFormatter | None = None) -> str:
        """Return help with options first and commands in titled groups."""
        if formatter is None:
            formatter = self._get_formatter()
        formatter.add_usage(self.usage, self._actions, self._mutually_exclusive_groups)
        formatter.add_text(self.description)
        formatter.start_section('options')
        formatter.add_arguments(self._get_optional_actions())
        formatter.end_section()
        for title, commands in _COMMAND_GROUPS:
            formatter.start_section(title)
            formatter.add_arguments(
                [argparse.Action([], name, help=text) for name, text in commands]
            )
            formatter.end_section()
        formatter.add_text(self.epilog)
        return formatter.format_help()


def _close_store(store: ValkeyMetadataStore) -> bool:
    """Close one store and report a redacted acknowledgement."""
    try:
        store.close()
    except (OSError, RuntimeError, TypeError, ValueError):
        return False
    return True


def _load_settings(path: str) -> CacheSettings:
    """Load operational settings after command selection."""
    return CacheSettings.from_toml(path)


def _open_store(settings: CacheSettings) -> ValkeyMetadataStore:
    """Open the metadata-only operator store after command selection."""
    module = import_module('sg_query_cache.valkey')
    factory = cast('type[ValkeyMetadataStore]', vars(module)['ValkeyMetadataStore'])
    password_env = settings.l2.valkey.metadata.password_env
    password = os.environ.get(password_env)
    environ = {} if password is None else {password_env: password}
    return factory.from_settings(settings, environ, NullMetrics())


def _print_version() -> int:
    """Print the installed distribution version without other work."""
    import importlib.metadata  # noqa: PLC0415 - defer metadata import to the --version path

    try:
        version = importlib.metadata.version('sg-query-cache')
    except importlib.metadata.PackageNotFoundError:
        print(
            'error: installed distribution metadata for sg-query-cache is unavailable',
            file=sys.stderr,
        )
        return 2
    print(f'sg-query-cache {version}')
    return 0


def _reason_argument(value: str) -> str:
    """Accept a flush reason of 1 to 64 ASCII characters."""
    try:
        flush_reason(value)
    except BackendProtocolError:
        raise argparse.ArgumentTypeError('must be 1 to 64 ASCII characters') from None
    return value


def _report_error(message: str) -> None:
    """Print one terminal-safe error line to standard error."""
    visible = _CONTROL_CHARACTERS.sub(lambda match: f'\\x{ord(match.group()):02x}', message)
    print(f'error: {visible}', file=sys.stderr)


def _configuration_failure() -> int:
    """Report one redacted configuration failure."""
    _report_error('configuration is invalid or unavailable')
    return 2


def _flush(config_path: str, site: str, reason: str) -> int:
    """Run one metadata-only site flush."""
    try:
        settings = _load_settings(config_path)
        store = _open_store(settings)
    except (ImportError, KeyError, OSError, TypeError, ValueError):
        return _configuration_failure()
    except Exception:  # noqa: BLE001 - closed redacted store-construction boundary
        _report_error('flush operation was not acknowledged')
        return 1

    completed = False
    try:
        store.flush_site(site, reason)
        completed = True
    except (OSError, RuntimeError, TypeError, ValueError):
        completed = False
    finally:
        closed = _close_store(store)
    if not completed or not closed:
        _report_error('flush operation was not acknowledged')
        return 1
    digest = sha256(site.encode()).hexdigest()
    print(f'flush acknowledged site={digest}')
    return 0


def _resolved_plugin_path(module: object) -> Path:
    """Return an existing absolute Python source path for one module."""
    raw_path = getattr(module, '__file__', None)
    if not isinstance(raw_path, str):
        raise TypeError
    path = Path(raw_path).resolve(strict=True)
    if path.suffix != '.py' or not path.is_file():
        raise ValueError
    return path


def _daemon_plugin_path() -> int:
    """Print the absolute installed daemon plugin path."""
    try:
        module = import_module('sg_query_cache.daemon_plugin')
        path = _resolved_plugin_path(module)
    except Exception:  # noqa: BLE001 - closed redacted plugin import/path boundary
        _report_error('sg_query_cache daemon plugin is unavailable')
        return 1
    print(path)
    return 0


def _rotate(config_path: str, confirm_prefix: str) -> int:
    """Run one confirmed all-sites incarnation rotation."""
    try:
        settings = _load_settings(config_path)
    except (OSError, TypeError, ValueError):
        return _configuration_failure()
    if confirm_prefix != settings.l2.valkey.metadata.key_prefix:
        _report_error('prefix confirmation does not match configuration')
        return 2
    try:
        store = _open_store(settings)
    except (ImportError, KeyError, OSError, TypeError, ValueError):
        return _configuration_failure()
    except Exception:  # noqa: BLE001 - closed redacted store-construction boundary
        _report_error('incarnation rotation was not acknowledged')
        return 1

    acknowledged = False
    try:
        store.rotate_incarnation()
        acknowledged = True
    except (OSError, RuntimeError, TypeError, ValueError):
        acknowledged = False
    finally:
        closed = _close_store(store)
    if not acknowledged or not closed:
        _report_error('incarnation rotation was not acknowledged')
        return 1
    print('incarnation rotation acknowledged for all sites')
    return 0


def _site_argument(value: str) -> str:
    """Accept a site namespace that a site flush can target."""
    try:
        DependencyTag.site(value)
    except (TypeError, ValueError):
        message = 'must be nonempty ASCII without leading or trailing spaces'
        raise argparse.ArgumentTypeError(message) from None
    return value


def _build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser without performing I/O."""
    parser = _RootParser(
        prog='sg-query-cache',
        description='Invalidate shared cached SG results and print the event-daemon plugin path.',
    )
    parser.add_argument('--version', action='store_true', help='show the version and exit')
    commands = parser.add_subparsers(
        dest='command', metavar='COMMAND', parser_class=argparse.ArgumentParser
    )

    flush = commands.add_parser(
        'flush',
        help=_COMMAND_HELP['flush'],
        description='Invalidate every cached result for one SG site by changing its generation.',
    )
    flush.add_argument(
        '--config', required=True, metavar='FILE', help='shared cache settings TOML file'
    )
    flush.add_argument(
        '--site',
        required=True,
        type=_site_argument,
        metavar='SITE',
        help='site namespace to flush',
    )
    flush.add_argument(
        '--reason',
        required=True,
        type=_reason_argument,
        metavar='REASON',
        help='reason for the flush, 1 to 64 ASCII characters',
    )

    rotation = commands.add_parser(
        'rotate-incarnation',
        help=_COMMAND_HELP['rotate-incarnation'],
        description=(
            'Invalidate cached results for every site under the configured key prefix by '
            'rotating its incarnation.'
        ),
    )
    rotation.add_argument(
        '--config', required=True, metavar='FILE', help='shared cache settings TOML file'
    )
    rotation.add_argument(
        '--confirm-prefix',
        required=True,
        metavar='PREFIX',
        help="repeat the metadata role's key_prefix to confirm",
    )

    commands.add_parser(
        'daemon-plugin-path',
        help=_COMMAND_HELP['daemon-plugin-path'],
        description=(
            'Print the absolute path of the SG event-daemon plugin installed with this package.'
        ),
    )
    return parser


def _run(arguments: list[str]) -> int:
    """Parse arguments and run the selected operator command."""
    parser = _build_parser()
    args = parser.parse_args(arguments)
    if args.version:
        return _print_version()
    if args.command is None:
        parser.error('the following arguments are required: COMMAND')
    if args.command == 'daemon-plugin-path':
        return _daemon_plugin_path()
    if args.command == 'flush':
        return _flush(args.config, args.site, args.reason)
    return _rotate(args.config, args.confirm_prefix)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the sg-query-cache command-line interface.

    If a reader closes standard output early, the remaining output is discarded
    and the exit status reflects the work completed before the closure.

    Args:
        argv: Arguments to parse, or `None` to use `sys.argv`.

    Returns:
        The process exit status.
    """
    if sys.stdout is None:
        # Started with standard output closed: nothing can be delivered, and the
        # closed-stdout wrapper cannot wrap a missing stream.
        _report_error('standard output is unavailable')
        return 2
    with closable_stdout():
        try:
            return _run(sys.argv[1:] if argv is None else list(argv))
        except KeyboardInterrupt:
            return 130
