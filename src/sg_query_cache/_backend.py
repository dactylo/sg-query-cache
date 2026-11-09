"""Define the backend-neutral synchronous cache storage contract."""

from typing import Literal, Protocol, TypeAlias

from sg_query_cache._model import (
    CacheHit,
    DependencyTag,
    FillLease,
    GenerationChange,
    GenerationVector,
    TagKind,
)

FailureClass: TypeAlias = Literal[
    'authentication', 'tls', 'connection', 'timeout', 'replica', 'unknown'
]

GLOBAL_TAG_KINDS = frozenset(
    {TagKind.AUTHORIZATION, TagKind.HUMAN_VISIBILITY, TagKind.SCHEMA, TagKind.SITE}
)


class BackendCapacity(RuntimeError):  # noqa: N818 - required backend exception name
    """Report that a bounded backend cannot admit required metadata."""


class BackendProtocolError(RuntimeError):
    """Report invalid input or state at the backend protocol boundary."""


class BackendUnavailable(RuntimeError):  # noqa: N818 - required backend exception name
    """Report an unavailable operation with an optional redacted failure class.

    Attributes:
        failure_class: Client failure category, or `None` for a local failure.
    """

    def __init__(self, *args: object, failure_class: FailureClass | None = None) -> None:
        """Append the client failure category when one is available."""
        self._failure_class: FailureClass | None = failure_class
        super().__init__(*args)
        if failure_class is not None:
            self.args = (f'{self} ({failure_class})',)

    @property
    def failure_class(self) -> FailureClass | None:
        """Return the redacted client failure category, if any."""
        return self._failure_class


class StaleIncarnation(RuntimeError):  # noqa: N818 - required backend exception name
    """Report an operation fenced by a newer backend incarnation."""


BACKEND_ERRORS = (BackendCapacity, BackendProtocolError, BackendUnavailable, StaleIncarnation)


class MonotonicClock(Protocol):
    """Return backend-authoritative monotonic seconds."""

    def __call__(self) -> float:
        """Read the current monotonic time.

        Returns:
            Current monotonic seconds.
        """
        ...


class CacheBackend(Protocol):
    """Store opaque encoded results under generation-qualified keys."""

    def change(self, changes: tuple[GenerationChange, ...]) -> None:
        """Replace all applicable generations in caller order.

        Args:
            changes: Ordered generation replacements to acknowledge together.

        Raises:
            BackendCapacity: Required global metadata cannot be retained.
            BackendProtocolError: The ordered change tuple is malformed.
            BackendUnavailable: The change cannot complete reliably.
            StaleIncarnation: The backend incarnation is no longer current.
        """
        ...

    def close(self) -> None:
        """Release backend resources; repeated calls have no effect."""
        ...

    def commit(self, lease: FillLease, payload: bytes, remaining_seconds: float) -> bool:
        """Store encoded bytes only while the exact lease remains current.

        Args:
            lease: Exact key and token returned by `reserve`.
            payload: Closed-codec bytes to retain.
            remaining_seconds: Positive authoritative lifetime remaining.

        Returns:
            Whether the lease remained current and the result was admitted.

        Raises:
            BackendProtocolError: An argument is outside the backend contract.
            BackendUnavailable: The commit cannot complete reliably.
            StaleIncarnation: The backend incarnation is no longer current.
        """
        ...

    def flush_site(self, site_namespace: str, reason: str) -> None:
        """Replace one site's permanent site generation.

        Args:
            site_namespace: Stable configured site namespace.
            reason: Short operator-facing reason code.

        Raises:
            BackendCapacity: Required global metadata cannot be retained.
            BackendProtocolError: The site namespace or reason is malformed.
            BackendUnavailable: The flush cannot complete reliably.
            StaleIncarnation: The backend incarnation is no longer current.
        """
        ...

    def lookup(self, key: str) -> CacheHit | None:
        """Read valid encoded bytes with backend-authoritative remaining age.

        Args:
            key: Opaque generation-qualified result key.

        Returns:
            An isolated hit, optionally carrying the backend's fresh decoded
            value, or `None` for absence, expiry, or corruption.

        Raises:
            BackendCapacity: The result role rejects required work for
                capacity.
            BackendProtocolError: The opaque key is malformed.
            BackendUnavailable: The lookup cannot complete reliably.
            StaleIncarnation: The backend incarnation is no longer current.
        """
        ...

    def release(self, lease: FillLease) -> bool:
        """Remove a fill lease only when its exact token remains current.

        Args:
            lease: Exact key and token returned by `reserve`.

        Returns:
            Whether the current lease was removed.

        Raises:
            BackendProtocolError: The lease is malformed.
            BackendUnavailable: The release cannot complete reliably.
            StaleIncarnation: The backend incarnation is no longer current.
        """
        ...

    def reserve(self, key: str, lease_seconds: float) -> FillLease | None:
        """Acquire temporary fill ownership for one opaque key.

        Args:
            key: Opaque generation-qualified result key.
            lease_seconds: Positive ownership duration.

        Returns:
            A fresh lease, or `None` while another lease remains current.

        Raises:
            BackendCapacity: Metadata capacity cannot retain a new lease.
            BackendProtocolError: An argument is outside the backend contract.
            BackendUnavailable: The reservation cannot complete reliably.
            StaleIncarnation: The backend incarnation is no longer current.
        """
        ...

    def resolve(self, tags: tuple[DependencyTag, ...]) -> GenerationVector:
        """Resolve one complete bounded dependency vector in request order.

        Args:
            tags: Ordered unique dependency tags.

        Returns:
            Every requested tag and current token under one incarnation.

        Raises:
            BackendCapacity: Metadata capacity cannot retain the vector.
            BackendProtocolError: The vector is malformed or exceeds its
                bounds.
            BackendUnavailable: Resolution cannot complete reliably.
            StaleIncarnation: The backend incarnation is no longer current.
        """
        ...
