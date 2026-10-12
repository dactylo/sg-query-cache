"""Define immutable values shared by cache components."""

from dataclasses import dataclass, field
from datetime import date, datetime
from enum import StrEnum
from typing import Literal, TypeAlias, cast

from sg_query_cache._validation import (
    require_nonempty_string,
    require_positive_number,
)

ExactValue: TypeAlias = (
    bool | int | float | str | bytes | date | datetime | tuple['ExactValue', ...] | None
)
IdentityKind: TypeAlias = Literal['script', 'human']
_TagComponent: TypeAlias = str | int

_PAIR_LENGTH = 2


class TagKind(StrEnum):
    """Identify the SG state represented by a dependency tag."""

    AUTHORIZATION = 'authorization'
    ENTITY = 'entity'
    HUMAN_VISIBILITY = 'human_visibility'
    IDENTITY_VISIBILITY = 'identity_visibility'
    LINK_PATH = 'link_path'
    PROJECT = 'project'
    SCHEMA = 'schema'
    SITE = 'site'
    TYPE_FIELD = 'type_field'
    TYPE_MEMBERSHIP = 'type_membership'
    TYPE_VALUE = 'type_value'


@dataclass(frozen=True, slots=True)
class DecodedPayload:
    """Carry one freshly decoded backend value for the cache core."""

    value: object


@dataclass(frozen=True, slots=True)
class CacheHit:
    """Hold an encoded cache result and its authoritative remaining lifetime.

    Attributes:
        payload: Closed-codec bytes returned by the backend.
        remaining_seconds: Positive lifetime remaining at lookup time.
        decoded: Optional freshly decoded value for package-internal handoff.
    """

    payload: bytes
    remaining_seconds: float
    decoded: DecodedPayload | None = field(default=None, compare=False, repr=False, kw_only=True)

    def __post_init__(self) -> None:
        """Validate the encoded payload and positive authoritative lifetime.

        Raises:
            TypeError: The payload, lifetime, or optional decoded handoff has
                the wrong type.
            ValueError: The remaining lifetime is not positive.
        """
        _require_bytes(self.payload, 'cache-hit payload')
        require_positive_number(self.remaining_seconds, 'cache-hit remaining lifetime')
        decoded = cast('object', self.decoded)
        if decoded is not None and not isinstance(decoded, DecodedPayload):
            raise TypeError('cache-hit decoded value must be a DecodedPayload')


@dataclass(frozen=True, slots=True)
class DependencyTag:
    """Identify one ordered, site-partitioned generation.

    The first component must be nonempty ASCII without leading or trailing
    whitespace. Site, schema, authorization, and human-visibility tags contain
    only that namespace. Project and identity-visibility tags add a positive
    ID;
    entity tags add an entity type and positive ID. Type-membership and
    type-value tags add an entity type. Type-field and link-path tags add two
    nonempty strings.

    Attributes:
        kind: Generation vocabulary member.
        components: Ordered strings or positive integers, beginning with the
            site namespace.
    """

    kind: TagKind
    components: tuple[_TagComponent, ...]

    def __post_init__(self) -> None:
        """Validate the ordered components required by this generation kind.

        Raises:
            TypeError: The kind or a component has the wrong type.
            ValueError: The component count, site namespace, or component value
                is invalid.
        """
        _require_dependency_tag_shape(self.kind, self.components)

    @classmethod
    def authorization(cls, site_namespace: str) -> 'DependencyTag':
        """Build a global authorization tag.

        Args:
            site_namespace: Stable namespace shared by clients for one SG site.

        Returns:
            The site-partitioned authorization tag.
        """
        return cls(TagKind.AUTHORIZATION, (site_namespace,))

    @classmethod
    def entity(cls, site_namespace: str, entity_type: str, entity_id: int) -> 'DependencyTag':
        """Build an exact entity tag.

        Args:
            site_namespace: Stable namespace shared by clients for one SG site.
            entity_type: SG entity type.
            entity_id: Positive SG entity ID.

        Returns:
            The site-partitioned entity tag.
        """
        return cls(TagKind.ENTITY, (site_namespace, entity_type, entity_id))

    @classmethod
    def human_visibility(cls, site_namespace: str) -> 'DependencyTag':
        """Build the global human-visibility tag.

        Args:
            site_namespace: Stable namespace shared by clients for one SG site.

        Returns:
            The site-partitioned human-visibility tag.
        """
        return cls(TagKind.HUMAN_VISIBILITY, (site_namespace,))

    @classmethod
    def identity_visibility(cls, site_namespace: str, entity_id: int) -> 'DependencyTag':
        """Build an exact-human visibility tag.

        Args:
            site_namespace: Stable namespace shared by clients for one SG site.
            entity_id: Positive SG `HumanUser` ID.

        Returns:
            The site-partitioned identity-visibility tag.
        """
        return cls(TagKind.IDENTITY_VISIBILITY, (site_namespace, entity_id))

    @classmethod
    def project(cls, site_namespace: str, project_id: int) -> 'DependencyTag':
        """Build an exact project tag.

        Args:
            site_namespace: Stable namespace shared by clients for one SG site.
            project_id: Positive SG `Project` ID.

        Returns:
            The site-partitioned project tag.
        """
        return cls(TagKind.PROJECT, (site_namespace, project_id))

    @classmethod
    def schema(cls, site_namespace: str) -> 'DependencyTag':
        """Build a schema tag.

        Args:
            site_namespace: Stable namespace shared by clients for one SG site.

        Returns:
            The site-partitioned schema tag.
        """
        return cls(TagKind.SCHEMA, (site_namespace,))

    @classmethod
    def site(cls, site_namespace: str) -> 'DependencyTag':
        """Build a site-wide tag.

        Args:
            site_namespace: Stable namespace shared by clients for one SG site.

        Returns:
            The site tag.
        """
        return cls(TagKind.SITE, (site_namespace,))

    @classmethod
    def type_field(cls, site_namespace: str, entity_type: str, field_name: str) -> 'DependencyTag':
        """Build an entity-type field tag.

        Args:
            site_namespace: Stable namespace shared by clients for one SG site.
            entity_type: SG entity type.
            field_name: SG field name.

        Returns:
            The site-partitioned type-field tag.
        """
        return cls(TagKind.TYPE_FIELD, (site_namespace, entity_type, field_name))

    @classmethod
    def type_membership(cls, site_namespace: str, entity_type: str) -> 'DependencyTag':
        """Build an entity-type membership tag.

        Args:
            site_namespace: Stable namespace shared by clients for one SG site.
            entity_type: SG entity type.

        Returns:
            The site-partitioned type-membership tag.
        """
        return cls(TagKind.TYPE_MEMBERSHIP, (site_namespace, entity_type))

    @classmethod
    def type_value(cls, site_namespace: str, entity_type: str) -> 'DependencyTag':
        """Build an entity-type value tag.

        Args:
            site_namespace: Stable namespace shared by clients for one SG site.
            entity_type: SG entity type.

        Returns:
            The site-partitioned type-value tag.
        """
        return cls(TagKind.TYPE_VALUE, (site_namespace, entity_type))


@dataclass(frozen=True, slots=True)
class DependencyPlan:
    """Describe a complete ordered pre-fetch dependency decision.

    Attributes:
        classification: Query policy classification.
        tags: Ordered unique dependency tags.
        reason: Stable policy reason code.
        broadening_steps: Ordered transformations used to satisfy vector
            bounds.
    """

    classification: str
    tags: tuple[DependencyTag, ...]
    reason: str
    broadening_steps: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validate the decision and ordered unique dependency tags.

        Raises:
            TypeError: A decision field, tag tuple, or broadening step has the
                wrong type.
            ValueError: A required string is empty or a dependency tag is
                duplicated.
        """
        require_nonempty_string(self.classification, 'dependency-plan classification')
        require_nonempty_string(self.reason, 'dependency-plan reason')
        tags = _require_dependency_tags(self.tags, 'dependency-plan tags')
        _reject_duplicate_tags(tags, 'dependency plan')
        _require_nonempty_string_tuple(
            self.broadening_steps,
            'dependency-plan broadening steps',
        )


@dataclass(frozen=True, slots=True)
class ExactCall:
    """Hold one normalized SG read method and its ordered bound arguments.

    Attributes:
        method: Normalized SG read method.
        arguments: Ordered method-argument names and closed values.
    """

    method: str
    arguments: tuple[tuple[str, ExactValue], ...]

    def __post_init__(self) -> None:
        """Validate unique ordered argument names and immutable closed values.

        Raises:
            TypeError: The method, argument tuple, or a closed value has the
                wrong form.
            ValueError: The method or an argument name is empty, or an argument
                name is duplicated.
        """
        require_nonempty_string(self.method, 'exact-call method')
        names: set[str] = set()
        arguments = _require_tuple(self.arguments, 'exact-call arguments')
        for argument in arguments:
            if not isinstance(argument, tuple):
                raise TypeError('each exact-call argument must be a two-item tuple')
            parts = cast('tuple[object, ...]', argument)
            if len(parts) != _PAIR_LENGTH:
                raise TypeError('each exact-call argument must be a two-item tuple')
            raw_name, value = parts
            name = require_nonempty_string(raw_name, 'exact-call argument name')
            _require_exact_value(value)
            if name in names:
                raise ValueError('exact call contains a duplicate argument')
            names.add(name)


@dataclass(frozen=True, slots=True)
class FillLease:
    """Identify temporary permission to populate one qualified key.

    Attributes:
        key: Opaque generation-qualified cache key.
        token: Opaque lease token.
    """

    key: str
    token: str

    def __post_init__(self) -> None:
        """Validate the nonempty qualified key and opaque lease token.

        Raises:
            TypeError: The key or token is not a string.
            ValueError: The key or token is empty.
        """
        require_nonempty_string(self.key, 'fill-lease key')
        require_nonempty_string(self.token, 'fill-lease token')


@dataclass(frozen=True, slots=True)
class GenerationChange:
    """Request replacement of one dependency generation.

    Attributes:
        tag: Dependency generation to replace when the backend contract permits
            it.
    """

    tag: DependencyTag

    def __post_init__(self) -> None:
        """Validate the dependency tag whose generation will change.

        Raises:
            TypeError: The tag is not a `DependencyTag` value.
        """
        _require_dependency_tag(self.tag, 'generation-change tag')


@dataclass(frozen=True, slots=True)
class GenerationVector:
    """Bind an ordered dependency set to one backend incarnation.

    Attributes:
        incarnation: Opaque backend incarnation nonce.
        entries: Ordered unique dependency tags and their positive generation
            tokens.
    """

    incarnation: str
    entries: tuple[tuple[DependencyTag, int], ...]

    def __post_init__(self) -> None:
        """Validate the incarnation and ordered unique positive generation entries.

        Raises:
            TypeError: The incarnation, entry pair, dependency tag, or token
                has the wrong type.
            ValueError: The incarnation is empty, a token is not positive, or a
                tag is duplicated.
        """
        require_nonempty_string(self.incarnation, 'generation-vector incarnation')
        entries = _require_tuple(self.entries, 'generation-vector entries')
        tags: list[DependencyTag] = []
        for entry in entries:
            if not isinstance(entry, tuple):
                raise TypeError('each generation-vector entry must be a two-item tuple')
            parts = cast('tuple[object, ...]', entry)
            if len(parts) != _PAIR_LENGTH:
                raise TypeError('each generation-vector entry must be a two-item tuple')
            tag, token = parts
            tags.append(_require_dependency_tag(tag, 'generation-vector tag'))
            _require_positive_integer(token, 'generation token')
        _reject_duplicate_tags(tuple(tags), 'generation vector')


@dataclass(frozen=True, slots=True)
class Identity:
    """Partition cache results by one stable effective SG identity.

    Script scopes may omit an SG `ApiUser` ID. Human identities require their
    positive SG `HumanUser` ID. `issuer`, `namespace`, and `epoch` must be
    stable configuration values, never rotating credentials.

    Attributes:
        kind: Either `script` or `human`.
        issuer: Stable identity issuer or deployment context.
        namespace: Stable configured identity namespace.
        entity_id: Optional positive `ApiUser` ID, or required positive
            `HumanUser` ID.
        epoch: Optional stable permission or configuration epoch.
    """

    kind: IdentityKind
    issuer: str
    namespace: str
    entity_id: int | None = None
    epoch: str | int | None = None

    def __post_init__(self) -> None:
        """Validate the stable partition and the required human identity ID.

        Raises:
            TypeError: An identity field has the wrong type.
            ValueError: The kind is unsupported, a required string is empty, or
                the entity ID is invalid or missing for a human.
        """
        if self.kind not in ('script', 'human'):
            raise ValueError('identity kind must be script or human')
        require_nonempty_string(self.issuer, 'identity issuer')
        require_nonempty_string(self.namespace, 'identity namespace')
        if self.entity_id is not None:
            _require_positive_integer(self.entity_id, 'identity entity ID')
        if self.kind == 'human' and self.entity_id is None:
            raise ValueError('human identity requires a positive SG entity ID')
        if self.epoch is not None:
            _require_epoch(self.epoch)


def _reject_duplicate_tags(tags: tuple[DependencyTag, ...], owner: str) -> None:
    """Reject duplicate tags without changing their authoritative order."""
    if len(tags) != len(set(tags)):
        raise ValueError(f'{owner} contains a duplicate dependency tag')


def _require_bytes(value: object, owner: str) -> None:
    """Require an immutable bytes value."""
    if not isinstance(value, bytes):
        raise TypeError(f'{owner} must be bytes')


def _require_dependency_tag(value: object, owner: str) -> DependencyTag:
    """Require a validated immutable dependency tag."""
    if not isinstance(value, DependencyTag):
        raise TypeError(f'{owner} must be a DependencyTag')
    return value


def _require_epoch(value: object) -> None:
    """Require a nonempty string or integer epoch."""
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise TypeError('identity epoch must be a string or integer')
    if isinstance(value, str) and not value:
        raise ValueError('identity epoch must be nonempty when present')


def _require_exact_value(value: object) -> None:
    """Require a recursively immutable value from the exact-call vocabulary."""
    if value is None or isinstance(value, (bool, int, float, str, bytes, date, datetime)):
        return
    if isinstance(value, tuple):
        for item in cast('tuple[object, ...]', value):
            _require_exact_value(item)
        return
    raise TypeError('exact-call values must use immutable closed value forms')


def _require_positive_integer(value: object, owner: str) -> None:
    """Require a positive non-boolean integer."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f'{owner} must be a positive integer')
    if value <= 0:
        raise ValueError(f'{owner} must be a positive integer')


def _require_tag_kind(value: object) -> TagKind:
    """Require a member of the closed tag-kind vocabulary."""
    if not isinstance(value, TagKind):
        raise TypeError('dependency-tag kind must be a TagKind')
    return value


def _require_tuple(value: object, owner: str) -> tuple[object, ...]:
    """Require an immutable tuple container."""
    if not isinstance(value, tuple):
        raise TypeError(f'{owner} must be a tuple')
    return cast('tuple[object, ...]', value)


def _require_dependency_tags(value: object, owner: str) -> tuple[DependencyTag, ...]:
    """Require an immutable tuple containing only dependency tags."""
    values = _require_tuple(value, owner)
    return tuple(_require_dependency_tag(item, owner) for item in values)


def _require_nonempty_string_tuple(value: object, owner: str) -> tuple[str, ...]:
    """Require an immutable tuple containing only nonempty strings."""
    values = _require_tuple(value, owner)
    for item in values:
        require_nonempty_string(item, owner)
    return cast('tuple[str, ...]', values)


def require_site_namespace(value: object, owner: str = 'site namespace') -> None:
    """Require a nonempty ASCII site namespace without outer whitespace.

    Args:
        value: Candidate namespace shared by clients for one SG site.
        owner: Name used to identify the invalid caller input.

    Raises:
        TypeError: The namespace is not a string.
        ValueError: The namespace is empty, non-ASCII, or has outer whitespace.
    """
    namespace = require_nonempty_string(value, owner)
    if not namespace.isascii() or not namespace.strip() or namespace != namespace.strip():
        raise ValueError(f'{owner} must be nonempty ASCII without outer whitespace')


def _require_dependency_tag_shape(kind: object, components: object) -> None:
    """Require the exact immutable component shape for one dependency-tag kind."""
    tag_kind = _require_tag_kind(kind)
    if not isinstance(components, tuple):
        raise TypeError('dependency tag components must be a tuple')
    values = cast('tuple[object, ...]', components)
    expected_types: tuple[type[str | int], ...]
    match tag_kind:
        case TagKind.AUTHORIZATION | TagKind.HUMAN_VISIBILITY | TagKind.SCHEMA | TagKind.SITE:
            expected_types = (str,)
        case TagKind.IDENTITY_VISIBILITY | TagKind.PROJECT:
            expected_types = (str, int)
        case TagKind.TYPE_MEMBERSHIP | TagKind.TYPE_VALUE:
            expected_types = (str, str)
        case TagKind.ENTITY:
            expected_types = (str, str, int)
        case TagKind.LINK_PATH | TagKind.TYPE_FIELD:
            expected_types = (str, str, str)
    if len(values) != len(expected_types):
        raise ValueError('dependency tag has the wrong component count for its kind')
    require_site_namespace(values[0], 'dependency tag site namespace')
    for value, expected_type in zip(values[1:], expected_types[1:], strict=True):
        if expected_type is str:
            require_nonempty_string(value, 'dependency tag component')
        else:
            _require_positive_integer(value, 'dependency tag component')
