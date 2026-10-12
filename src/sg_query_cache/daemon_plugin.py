# pyright: reportPrivateUsage=false
"""Provide the synchronous path-loadable SG event-daemon plugin.

Importing this module defines registration and lifecycle functions only. The
optional Valkey backend is imported and connected during `setup()`.
"""

from __future__ import annotations

from contextlib import suppress
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Protocol, cast

from sg_query_cache._config import CacheSettings
from sg_query_cache._event_mapping import changes_for_event
from sg_query_cache._metrics import NullMetrics
from sg_query_cache._model import DependencyTag

if TYPE_CHECKING:
    from collections.abc import Mapping

    from sg_query_cache.valkey import ValkeyMetadataStore


class _EventContext(Protocol):
    """Describe the daemon callback context fields the plugin consumes."""

    @property
    def event(self) -> Mapping[str, object]:
        """Return the immutable event mapping."""
        ...


class _Registrar(Protocol):
    """Describe the daemon registration boundary the plugin consumes."""

    def add_callback(self, **values: object) -> None:
        """Register one callback declaration."""
        ...

    @property
    def config(self) -> Mapping[str, object]:
        """Return immutable declaration configuration."""
        ...

    def set_setup(self, setup: object) -> None:
        """Register one setup function."""
        ...


class _Secret(Protocol):
    """Reveal one daemon-managed secret only when setup needs it."""

    def reveal(self) -> str:
        """Return the secret value."""
        ...


class _SetupContext(Protocol):
    """Describe the daemon setup context fields the plugin consumes."""

    @property
    def secrets(self) -> Mapping[str, _Secret]:
        """Return daemon-managed declared secrets."""
        ...

    @property
    def site(self) -> str:
        """Return the daemon site name."""
        ...


class _State:
    """Hold process-local plugin lifecycle state."""

    _backend: ClassVar[ValkeyMetadataStore | None] = None
    _config_path: ClassVar[Path | None] = None
    _site_namespace: ClassVar[str | None] = None


def callback(context: _EventContext) -> None:
    """Apply one event's changes through the metadata boundary.

    Args:
        context: Immutable daemon callback context.

    Raises:
        RuntimeError: Setup is incomplete. Backend failures propagate so the
            daemon retries.
    """
    backend = _State._backend
    site_namespace = _State._site_namespace
    if backend is None or site_namespace is None:
        raise RuntimeError('cache invalidation setup is incomplete')
    backend.change(changes_for_event(site_namespace, context.event))


def setup(context: _SetupContext) -> None:
    """Create metadata state and flush the configured site before callbacks.

    Args:
        context: Daemon setup context containing the site and declared secrets.

    Raises:
        RuntimeError: Registration is incomplete or the daemon site differs
            from the shared cache configuration. Backend failures propagate.
    """
    config_path = _State._config_path
    if config_path is None:
        raise RuntimeError('cache invalidation plugin is not registered')
    settings = CacheSettings.from_toml(config_path)
    if context.site != settings.site_namespace:
        raise RuntimeError('daemon site does not match the cache site namespace')

    password_env = settings.l2.valkey.metadata.password_env
    password = context.secrets[password_env].reveal()
    environ = {password_env: password}

    valkey_module = import_module('sg_query_cache.valkey')
    store_factory = cast('type[ValkeyMetadataStore]', vars(valkey_module)['ValkeyMetadataStore'])
    store = store_factory.from_settings(settings, environ, NullMetrics())
    try:
        store.resolve((DependencyTag.site(settings.site_namespace),))
        store.flush_site(settings.site_namespace, 'daemon_setup')
    except RuntimeError:
        with suppress(RuntimeError):
            store.close()
        raise

    previous = _State._backend
    _State._backend = store
    _State._site_namespace = settings.site_namespace
    if previous is not None:
        previous.close()


def register(registrar: _Registrar) -> None:
    """Register setup and invalidation without performing backend or SG I/O.

    Args:
        registrar: Daemon declaration registrar whose immutable configuration
            includes a nonempty string `config_path` naming the shared TOML
            file.
            Relative paths are interpreted from the process working directory.

    Raises:
        KeyError: The required `config_path` setting is absent.
        TypeError: `config_path` is not a string.
        ValueError: `config_path` is empty.
    """
    config_path = registrar.config['config_path']
    if not isinstance(config_path, str):
        raise TypeError('config_path must be a string')
    if not config_path:
        raise ValueError('config_path must not be empty')
    _State._config_path = Path(config_path)
    registrar.set_setup(setup)
    registrar.add_callback(id='invalidate-cache', callback=callback)
