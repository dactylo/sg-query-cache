"""Implement a loop-owned wrapper around bounded local storage."""

from sg_query_cache._async_support import LoopAffinity
from sg_query_cache._backend import MonotonicClock
from sg_query_cache._config import CacheSettings
from sg_query_cache._local import LocalBackend
from sg_query_cache._metrics import MetricsSink
from sg_query_cache._model import (
    CacheHit,
    DependencyTag,
    FillLease,
    GenerationChange,
    GenerationVector,
)


class AsyncLocalBackend:
    """Provide one loop-owned byte-bounded in-process cache backend.

    Each coroutine checks event-loop affinity before inspecting backend state,
    then delegates to a private `LocalBackend`. No coroutine awaits, so each
    store transition finishes before another task runs; the delegate's lock is
    never contended. Immutable snapshots decode after their transition, and
    telemetry remains exception-isolated.
    """

    __slots__ = ('_affinity', '_backend')

    def __init__(
        self, settings: CacheSettings, clock: MonotonicClock, metrics: MetricsSink
    ) -> None:
        """Initialize an unbound backend and its independently owned store.

        Args:
            settings: Validated cache settings supplying L1 and operational
                bounds.
            clock: Backend-authoritative monotonic clock.
            metrics: Safe metrics boundary retained for owning state
                transitions.

        Raises:
            BackendCapacity: The metadata budget cannot retain permanent
                globals.
            BackendUnavailable: The initial clock or entropy source is invalid.
        """
        self._affinity = LoopAffinity()
        self._backend = LocalBackend(settings, clock, metrics)

    async def change(self, changes: tuple[GenerationChange, ...]) -> None:
        """Replace all applicable generations in caller order.

        Args:
            changes: Ordered generation replacements to acknowledge together.

        Raises:
            BackendCapacity: Required global metadata cannot be retained.
            BackendProtocolError: The ordered change tuple is malformed.
            BackendUnavailable: The backend or clock is unavailable.
            RuntimeError: The backend belongs to a different event loop.
        """
        self._affinity.check()
        self._backend.change(changes)

    async def close(self) -> None:
        """Release resources, rejecting even repeated calls from another loop.

        Raises:
            RuntimeError: The backend belongs to a different event loop.
        """
        self._affinity.check()
        self._backend.close()

    async def commit(self, lease: FillLease, payload: bytes, remaining_seconds: float) -> bool:
        """Store encoded bytes only while the exact lease remains current.

        Args:
            lease: Exact key and token returned by `reserve`.
            payload: Closed-codec bytes to retain.
            remaining_seconds: Positive authoritative lifetime remaining.

        Returns:
            Whether the lease remained current and the result was admitted.

        Raises:
            BackendProtocolError: An argument is outside the backend contract.
            BackendUnavailable: The backend or clock is unavailable.
            RuntimeError: The backend belongs to a different event loop.
        """
        self._affinity.check()
        return self._backend.commit(lease, payload, remaining_seconds)

    async def flush_site(self, site_namespace: str, reason: str) -> None:
        """Replace one site's permanent site generation.

        Args:
            site_namespace: Stable configured site namespace.
            reason: Short operator-facing reason code.

        Raises:
            BackendCapacity: Required global metadata cannot be retained.
            BackendProtocolError: The site namespace or reason is malformed.
            BackendUnavailable: The backend or clock is unavailable.
            RuntimeError: The backend belongs to a different event loop.
        """
        self._affinity.check()
        self._backend.flush_site(site_namespace, reason)

    async def lookup(self, key: str) -> CacheHit | None:
        """Read fresh decoded bytes with backend-authoritative remaining age.

        Args:
            key: Opaque generation-qualified result key.

        Returns:
            An isolated hit carrying a fresh decoded value, or `None` for
            absence, expiry, or corruption.

        Raises:
            BackendProtocolError: The opaque key is malformed.
            BackendUnavailable: The backend or clock is unavailable.
            RuntimeError: The backend belongs to a different event loop.
        """
        self._affinity.check()
        return self._backend.lookup(key)

    async def release(self, lease: FillLease) -> bool:
        """Remove a fill lease only when its exact token remains current.

        Args:
            lease: Exact key and token returned by `reserve`.

        Returns:
            Whether the current lease was removed.

        Raises:
            BackendProtocolError: The lease is malformed.
            BackendUnavailable: The backend or clock is unavailable.
            RuntimeError: The backend belongs to a different event loop.
        """
        self._affinity.check()
        return self._backend.release(lease)

    async def reserve(self, key: str, lease_seconds: float) -> FillLease | None:
        """Acquire temporary fill ownership for one opaque key.

        Args:
            key: Opaque generation-qualified result key.
            lease_seconds: Positive ownership duration.

        Returns:
            A fresh lease, or `None` while another lease remains current.

        Raises:
            BackendCapacity: Metadata capacity cannot retain a new lease.
            BackendProtocolError: An argument is outside the backend contract.
            BackendUnavailable: The backend or clock is unavailable.
            RuntimeError: The backend belongs to a different event loop.
        """
        self._affinity.check()
        return self._backend.reserve(key, lease_seconds)

    async def resolve(self, tags: tuple[DependencyTag, ...]) -> GenerationVector:
        """Resolve one complete bounded dependency vector in request order.

        Args:
            tags: Ordered unique dependency tags.

        Returns:
            Every requested tag and current token under one incarnation.

        Raises:
            BackendCapacity: Metadata capacity cannot retain the vector.
            BackendProtocolError: The vector is malformed or exceeds its
                bounds.
            BackendUnavailable: The backend or clock is unavailable.
            RuntimeError: The backend belongs to a different event loop.
        """
        self._affinity.check()
        return self._backend.resolve(tags)

    async def storage_usage(self) -> dict[str, int]:
        """Return a fresh snapshot of accounted capacity bytes without maintenance.

        Returns:
            Result, generation, and lease bytes under `result_bytes`,
            `generation_bytes`, and `lease_bytes`; all zero after close.
            These are capacity accounting totals, not Python heap sizes.

        Raises:
            RuntimeError: The backend belongs to a different event loop.
        """
        self._affinity.check()
        return self._backend.storage_usage()
