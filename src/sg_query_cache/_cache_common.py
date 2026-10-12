"""Share result, policy, fence, and wait accounting across cache implementations."""

from collections.abc import Callable
from dataclasses import dataclass, field
from math import isfinite
from typing import Literal, Protocol, TypeAlias, cast

from sg_query_cache._codec import CodecBoundError, CodecError, decode_value, encode_value
from sg_query_cache._model import (
    CacheHit,
    DependencyPlan,
    DependencyTag,
    GenerationChange,
    TagKind,
)

PolicyMode: TypeAlias = Literal['bypass', 'event_aware', 'ttl_only']
PopulationCallback: TypeAlias = Callable[[object], 'PopulationDecision']
WaitCategory: TypeAlias = Literal['local', 'distributed']
WaitOutcome: TypeAlias = Literal['completed', 'takeover', 'timeout', 'failure', 'cancelled']

_MAX_PENDING_CHANGES = 256
_POLICY_RANK = {'event_aware': 0, 'ttl_only': 1, 'bypass': 2}
_STORED_VALUE_LENGTH = 5
_STORED_VALUE_TAG = 'sg-query-cache-result-v1'
_UNIVERSAL_TAG_KINDS = frozenset(
    {
        TagKind.AUTHORIZATION,
        TagKind.HUMAN_VISIBILITY,
        TagKind.IDENTITY_VISIBILITY,
        TagKind.SITE,
    }
)


@dataclass(frozen=True, slots=True)
class DecodedHit:
    """Keep one user value paired with its SG duration and age at decode time.

    Attributes:
        sg_duration: Original SG fetch duration in seconds.
        value: Isolated decoded user value.
        origin_elapsed: Age in seconds since the original SG response arrived.
    """

    sg_duration: float
    value: object
    origin_elapsed: float


@dataclass(frozen=True, slots=True)
class PolicyFrame:
    """Hold one validated lexical policy override.

    Use `policy_frame` to validate inputs before constructing a scope.

    Attributes:
        maximum_age: Positive age in seconds, or `None` to leave age selection
            unchanged.
        mode: Conservative policy downgrade, or `None` to leave mode unchanged.
    """

    maximum_age: float | None
    mode: PolicyMode | None


class PopulationDecision(Protocol):
    """Expose only the optional safe lifetime selected after a fetch."""

    @property
    def remaining_seconds(self) -> float | None:
        """Return a positive lifetime cap or decline population with `None`."""
        ...


@dataclass(frozen=True, slots=True)
class PopulationLimit:
    """Carry a validated lifetime cap and optional metric reasons.

    `seconds` is absent when population must decline. `reason` is the
    callback's optional observation label; `bypass_reason` distinguishes
    refusal from invalid callback output.
    """

    seconds: float | None
    reason: str | None = None
    bypass_reason: str | None = None


class StaleHitError(Exception):
    """Report an ordinary hit older than this caller permits."""


@dataclass(frozen=True, slots=True)
class TimedResult:
    """Supply SG timing and a user value to the result encoder.

    Attributes:
        sg_duration: Original SG fetch duration in seconds.
        value: User result to include in the stored envelope.
    """

    sg_duration: float
    value: object


@dataclass(slots=True)
class WaitRecord:
    """Account one read request's waits so each category is counted and timed once.

    Pair each `begin` with `end`, even after interruption. Repeated waits in a
    category accumulate elapsed seconds and retain the latest outcome.
    """

    _seconds: dict[WaitCategory, float] = field(default_factory=dict[WaitCategory, float])
    _outcomes: dict[WaitCategory, WaitOutcome] = field(
        default_factory=dict[WaitCategory, WaitOutcome]
    )
    _expired: set[WaitCategory] = field(default_factory=set[WaitCategory])

    def begin(self, category: WaitCategory) -> bool:
        """Start a wait; return True only for the request's first wait in `category`."""
        if category in self._seconds:
            return False
        self._seconds[category] = 0.0
        return True

    def end(self, category: WaitCategory, seconds: float, outcome: WaitOutcome) -> None:
        """Add one finished wait interval and keep `outcome` as the category's latest."""
        self._seconds[category] += seconds
        self._outcomes[category] = outcome

    def expire(self, category: WaitCategory) -> bool:
        """Return True only the first time the request's budget runs out in `category`."""
        if category in self._expired:
            return False
        self._expired.add(category)
        return True

    def observations(self) -> tuple[tuple[float, WaitOutcome], ...]:
        """Return total seconds and latest outcome per category that began, local first."""
        categories: tuple[WaitCategory, ...] = ('local', 'distributed')
        return tuple(
            (self._seconds[category], self._outcomes[category])
            for category in categories
            if category in self._seconds
        )


def _finite_nonnegative(value: object, owner: str) -> float:
    """Require one finite non-boolean number that is not negative."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise CodecError(f'{owner} is malformed')
    try:
        result = float(value)
    except OverflowError:
        raise CodecError(f'{owner} is malformed') from None
    if not isfinite(result) or result < 0:
        raise CodecError(f'{owner} is malformed')
    return result


def _policy_age(value: object, owner: str) -> float:
    """Validate one explicit scoped or per-call maximum age."""
    if isinstance(value, bool) or not isinstance(value, int | float) or not isfinite(value):
        raise TypeError(f'{owner} must be a finite number')
    result = float(value)
    if result <= 0:
        raise ValueError(f'{owner} must be positive')
    return result


def _policy_mode(value: object) -> PolicyMode:
    """Validate one explicit scoped policy mode."""
    if value not in _POLICY_RANK:
        raise ValueError('policy mode must be event_aware, ttl_only, or bypass')
    return cast('PolicyMode', value)


def bound_reason(error: CodecBoundError) -> str:
    """Map a codec bound to its stable cache-bypass reason."""
    return {
        'bytes': 'result_too_large_bytes',
        'depth': 'result_too_deep',
        'items': 'result_too_large_items',
    }[error.bound]


def decode_hit(hit: CacheHit, maximum_age: float) -> DecodedHit:
    """Decode an isolated value that remains young enough for this caller.

    Args:
        hit: Backend hit containing the stored result envelope and
            authoritative remaining lifetime.
        maximum_age: Maximum age in seconds since the original SG response.

    Returns:
        The decoded user value, original SG duration, and elapsed origin age.

    Raises:
        CodecError: The payload or stored timing envelope is malformed.
        StaleHitError: The elapsed origin age reaches the caller's maximum age.
    """
    decoded = decode_value(hit.payload) if hit.decoded is None else hit.decoded.value
    if not isinstance(decoded, tuple):
        raise CodecError('cached result envelope is malformed')
    parts = cast('tuple[object, ...]', decoded)
    if len(parts) != _STORED_VALUE_LENGTH:
        raise CodecError('cached result envelope is malformed')
    tag, origin_elapsed, lifetime_seconds, sg_duration, value = parts
    if tag != _STORED_VALUE_TAG:
        raise CodecError('cached result envelope is malformed')
    origin = _finite_nonnegative(origin_elapsed, 'cached result origin age')
    lifetime = _finite_nonnegative(lifetime_seconds, 'cached result lifetime')
    if lifetime <= 0 or hit.remaining_seconds > lifetime:
        raise CodecError('cached result lifetime is outside its committed bound')
    elapsed = origin + lifetime - hit.remaining_seconds
    if elapsed >= maximum_age:
        raise StaleHitError
    duration = _finite_nonnegative(sg_duration, 'cached result SG duration')
    return DecodedHit(duration, value, elapsed)


def encode_result(timed: TimedResult, origin_elapsed: float, lifetime: float) -> bytes:
    """Encode a result with the timing facts needed to enforce later shorter reads.

    Args:
        timed: Original SG duration in seconds and the user value.
        origin_elapsed: Age in seconds since the original SG response at the
            current storage or promotion observation.
        lifetime: Remaining lifetime in seconds at that same observation.

    Returns:
        Closed-codec bytes containing the result and its timing envelope.

    Raises:
        CodecError: The value cannot be represented within the closed codec
            vocabulary or bounds.
    """
    return encode_value(
        (_STORED_VALUE_TAG, origin_elapsed, lifetime, timed.sg_duration, timed.value)
    )


def merge_changes(
    retained: tuple[GenerationChange, ...],
    changes: tuple[GenerationChange, ...],
) -> tuple[GenerationChange, ...]:
    """Prepend retained changes, deduplicate, and collapse beyond the bound.

    Both inputs must belong to the same site namespace.

    Args:
        retained: Unacknowledged changes to replay before new work.
        changes: New changes in caller order.

    Returns:
        Deduplicated changes in order of first occurrence, unless a retained
        site tag is present: that tag remains first, followed by new non-site
        changes without deduplicating them. A new site-wide change or more than
        256 distinct merged changes collapses to the site tag. Empty inputs
        return an empty tuple.
    """
    if not retained and not changes:
        return ()
    source = changes[0] if changes else retained[0]
    site = GenerationChange(DependencyTag.site(cast('str', source.tag.components[0])))
    if site in changes:
        return (site,)
    merged: dict[GenerationChange, None] = {}
    for change in (*retained, *changes):
        merged[change] = None
        if len(merged) > _MAX_PENDING_CHANGES:
            return (site,)
    if site in retained:
        return (site, *(change for change in changes if change != site))
    return tuple(merged)


def minimum_policy_age(
    frames: tuple[PolicyFrame, ...],
    per_call_age: float | None,
) -> float | None:
    """Return the smallest explicit call or lexical age.

    Args:
        frames: Already validated lexical policy frames.
        per_call_age: Optional positive maximum age in seconds.

    Returns:
        The smallest supplied age, or `None` when neither source overrides age.

    Raises:
        TypeError: The per-call age is not a finite non-boolean number.
        ValueError: The per-call age is not positive.
    """
    ages = [frame.maximum_age for frame in frames if frame.maximum_age is not None]
    if per_call_age is not None:
        ages.append(_policy_age(per_call_age, 'per-call maximum-age override'))
    return min(ages) if ages else None


def policy_bypass_active(frames: tuple[PolicyFrame, ...]) -> bool:
    """Report whether immutable lexical policy frames require bypass."""
    return any(frame.mode == 'bypass' for frame in frames)


def apply_policy_frames(
    plan: DependencyPlan,
    frames: tuple[PolicyFrame, ...],
) -> DependencyPlan:
    """Apply immutable lexical policy frames without promoting an existing decision."""
    classification = plan.classification
    if classification not in {'bypass', 'event_aware', 'schema', 'ttl_only'}:
        return DependencyPlan('bypass', (), 'unknown_policy')
    if classification == 'bypass':
        return plan
    if classification == 'schema':
        return (
            DependencyPlan('bypass', (), 'scoped_bypass', plan.broadening_steps)
            if policy_bypass_active(frames)
            else plan
        )
    rank = _POLICY_RANK.get(classification, 0)
    reason = plan.reason
    for frame in frames:
        if frame.mode is not None and _POLICY_RANK[frame.mode] > rank:
            rank = _POLICY_RANK[frame.mode]
            classification = frame.mode
            reason = f'scoped_{frame.mode}'
    if classification == plan.classification:
        return plan
    if classification == 'bypass':
        return DependencyPlan('bypass', (), reason, plan.broadening_steps)
    tags = tuple(tag for tag in plan.tags if tag.kind in _UNIVERSAL_TAG_KINDS)
    return DependencyPlan('ttl_only', tags, reason, plan.broadening_steps)


def policy_frame(
    mode: PolicyMode | None,
    maximum_age: float | None,
) -> PolicyFrame:
    """Build one validated immutable lexical policy frame.

    Args:
        mode: `event_aware`, `ttl_only`, or `bypass`; `None` leaves mode
            unchanged.
        maximum_age: Positive finite seconds; `None` leaves age unchanged.

    Returns:
        The validated frame.

    Raises:
        TypeError: A supplied age is not a finite non-boolean number.
        ValueError: The mode is unsupported or the age is not positive.
    """
    validated_mode = None if mode is None else _policy_mode(mode)
    validated_age = None if maximum_age is None else _policy_age(maximum_age, 'policy maximum age')
    return PolicyFrame(validated_age, validated_mode)


def population_limit(value: object, callback: PopulationCallback) -> PopulationLimit:
    """Validate one callback result without choosing a layer deadline.

    Callback exceptions and invalid lifetimes decline population. A callback
    need not expose a reason; only string reasons accompany observations.
    """
    try:
        decision = callback(value)
        remaining = decision.remaining_seconds
    except Exception:  # noqa: BLE001 - callback failure must decline population
        return PopulationLimit(None, bypass_reason='population_decision_failure')
    if remaining is None:
        reason = getattr(decision, 'reason', None)
        return PopulationLimit(
            None, reason if isinstance(reason, str) else None, 'population_declined'
        )
    seconds = cast('object', remaining)
    if (
        isinstance(seconds, bool)
        or not isinstance(seconds, int | float)
        or not isfinite(seconds)
        or seconds <= 0
    ):
        return PopulationLimit(None, bypass_reason='population_decision_failure')
    reason = getattr(decision, 'reason', None)
    return PopulationLimit(float(seconds), reason if isinstance(reason, str) else None)
