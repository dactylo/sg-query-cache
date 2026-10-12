"""Define the backend-neutral asynchronous cache storage contract."""

from typing import Protocol

from sg_query_cache._model import (
    CacheHit,
    DependencyTag,
    FillLease,
    GenerationChange,
    GenerationVector,
)


class AsyncCacheBackend(Protocol):
    """Store opaque encoded results through loop-native coroutine methods.

    Each method preserves the corresponding `CacheBackend` arguments, return
    values, and backend errors while adding coroutine cancellation cooperation.
    This protocol types the shipped backends; third-party implementations are
    not yet a supported extension point.
    """

    async def change(self, changes: tuple[GenerationChange, ...]) -> None:
        """Replace all applicable generations in caller order.

        Args:
            changes: Ordered generation replacements to acknowledge together.
        """
        ...

    async def close(self) -> None:
        """Release backend resources; repeated same-loop calls have no effect."""
        ...

    async def commit(self, lease: FillLease, payload: bytes, remaining_seconds: float) -> bool:
        """Store encoded bytes only while the exact lease remains current.

        Args:
            lease: Exact key and token returned by `reserve`.
            payload: Closed-codec bytes to retain.
            remaining_seconds: Positive authoritative lifetime remaining.

        Returns:
            Whether the lease remained current and the result was admitted.
        """
        ...

    async def flush_site(self, site_namespace: str, reason: str) -> None:
        """Replace one site's permanent site generation.

        Args:
            site_namespace: Stable configured site namespace.
            reason: Short operator-facing reason code.
        """
        ...

    async def lookup(self, key: str) -> CacheHit | None:
        """Read valid encoded bytes with backend-authoritative remaining age.

        Args:
            key: Opaque generation-qualified result key.

        Returns:
            An isolated hit, optionally carrying the backend's fresh decoded
            value, or `None` for absence, expiry, or corruption.
        """
        ...

    async def release(self, lease: FillLease) -> bool:
        """Remove a fill lease only when its exact token remains current.

        Args:
            lease: Exact key and token returned by `reserve`.

        Returns:
            Whether the current lease was removed.
        """
        ...

    async def reserve(self, key: str, lease_seconds: float) -> FillLease | None:
        """Acquire temporary fill ownership for one opaque key.

        Args:
            key: Opaque generation-qualified result key.
            lease_seconds: Positive ownership duration.

        Returns:
            A fresh lease, or `None` while another lease remains current.
        """
        ...

    async def resolve(self, tags: tuple[DependencyTag, ...]) -> GenerationVector:
        """Resolve one complete bounded dependency vector in request order.

        Args:
            tags: Ordered unique dependency tags.

        Returns:
            Every requested tag and current token under one incarnation.
        """
        ...
