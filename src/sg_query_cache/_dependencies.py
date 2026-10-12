# Same-file collaborators use private implementation members.
# pyright: reportPrivateUsage=false

"""Parse SG schema facts and derive conservative pre-fetch dependencies."""

import json
import zlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Never, Protocol, Self, cast

from sg_query_cache._keys import KeyBuilder
from sg_query_cache._model import (
    DependencyPlan,
    DependencyTag,
    ExactCall,
    Identity,
    TagKind,
)
from sg_query_cache._policy import (
    FieldSchema,
    QueryPolicy,
    classify_query,
    is_known_field_type,
)
from sg_query_cache._query import ORDER_ITEM_KEYS, argument, filter_group, pair_mapping
from sg_query_cache._validation import require_bool

_CONDITION_LENGTH = 3
_LINK_TYPES = frozenset({'addressing', 'entity', 'multi_entity', 'tag_list'})
_MAX_COMPACT_BYTES = 16 * 1024 * 1024
_NAME_OPERATORS = frozenset(
    {
        'name_contains',
        'name_ends_with',
        'name_is',
        'name_not_contains',
        'name_starts_with',
    }
)
_NEGATIVE_LINK_OPERATORS = frozenset({'is_not', 'not_in', 'type_is_not'})
_PAIR_LENGTH = 2
_POSITIVE_REFERENCE_OPERATORS = frozenset({'in', 'is'})


class _DependencyLimits(Protocol):
    """Provide the two pre-backend vector bounds used by the analyzer."""

    @property
    def max_dependency_tags(self) -> int:
        """Return the maximum complete dependency tag count."""
        ...

    @property
    def max_dependency_vector_bytes(self) -> int:
        """Return the maximum worst-case encoded vector size."""
        ...


@dataclass(frozen=True, slots=True)
class _PathHop:
    """Describe one schema-proven traversal hop."""

    _source_type: str
    _field_name: str
    _target_type: str


@dataclass(frozen=True, slots=True)
class _ResolvedPath:
    """Hold every traversal hop and the final field fact."""

    _final_field: FieldSchema
    _final_field_name: str
    _final_type: str
    _hops: tuple[_PathHop, ...]


@dataclass(frozen=True, slots=True)
class SchemaView:
    """Hold immutable conservative facts parsed from an SG schema response."""

    _entities: tuple[tuple[str, tuple[tuple[str, FieldSchema], ...]], ...]

    _field_index: Mapping[str, Mapping[str, FieldSchema]] = field(
        init=False, repr=False, compare=False, hash=False
    )

    def __post_init__(self) -> None:
        """Index the first entity and field occurrences without changing stored facts."""
        entities: dict[str, Mapping[str, FieldSchema]] = {}
        for entity_type, fields in self._entities:
            if entity_type in entities:
                continue
            indexed: dict[str, FieldSchema] = {}
            for name, schema in fields:
                indexed.setdefault(name, schema)
            entities[entity_type] = MappingProxyType(indexed)
        object.__setattr__(self, '_field_index', MappingProxyType(entities))

    def _field(self, entity_type: str, field_name: str) -> FieldSchema | None:
        """Look up one direct field without exposing mutable schema mappings."""
        fields = self._field_index.get(entity_type)
        return None if fields is None else fields.get(field_name)

    def _resolve_path(self, entity_type: str, field_path: str) -> _ResolvedPath | None:
        """Resolve alternating field and target-type segments conservatively."""
        if not field_path:
            return None
        parts = field_path.split('.')
        if len(parts) % 2 == 0:
            return None
        current_type = entity_type
        hops: list[_PathHop] = []
        for index in range(0, len(parts) - 1, 2):
            field_name = parts[index]
            target_type = parts[index + 1]
            field = self._field(current_type, field_name)
            if (
                field is None
                or field.data_type not in _LINK_TYPES
                or target_type not in field.valid_types
            ):
                return None
            hops.append(_PathHop(current_type, field_name, target_type))
            current_type = target_type
        final_field_name = parts[-1]
        final_field = self._field(current_type, final_field_name)
        if final_field is None:
            return None
        return _ResolvedPath(final_field, final_field_name, current_type, tuple(hops))

    @classmethod
    def from_api_response(cls, value: object) -> Self:
        """Parse only field existence, data type, and valid linked types.

        Args:
            value: Decoded SG schema response.

        Returns:
            An immutable view that treats malformed or hidden facts as absent.
        """
        if not isinstance(value, Mapping):
            return cls(())
        entities: list[tuple[str, tuple[tuple[str, FieldSchema], ...]]] = []
        for entity_type, raw_fields in cast('Mapping[object, object]', value).items():
            if not isinstance(entity_type, str) or not entity_type:
                continue
            fields = _parse_fields(raw_fields)
            if fields is not None:
                entities.append((entity_type, fields))
        return cls(tuple(entities))

    @classmethod
    def from_compact(cls, data: bytes) -> Self:
        """Rebuild a view from its compact stored form.

        Args:
            data: One complete compressed JSON document produced by
                `to_compact`, expanding to at most 16 MiB.

        Returns:
            An immutable view reconstructed from the stored schema facts.

        Raises:
            ValueError: The bytes are malformed, contain trailing compressed
                data, or exceed the expanded-size limit.
        """
        return cls(_compact_entities(_compact_document(data)))

    def resolve_field(self, entity_type: str, field_path: str) -> FieldSchema | None:
        """Resolve a final field only when every dotted traversal hop is known.

        Args:
            entity_type: Base SG entity type.
            field_path: Direct or dotted SG field path.

        Returns:
            The final field schema, or `None` when any fact is unknown.
        """
        resolved = self._resolve_path(entity_type, field_path)
        if resolved is None:
            return None
        return resolved._final_field

    def to_compact(self) -> bytes:
        """Serialize the view as one compact opaque byte value.

        Returns:
            Compressed JSON containing the schema facts. Serialization does not
            enforce the 16 MiB expanded-size limit used by `from_compact`.
        """
        document = {
            entity_type: {
                name: [field.data_type, list(field.valid_types)] for name, field in fields
            }
            for entity_type, fields in self._entities
        }
        return zlib.compress(json.dumps(document, separators=(',', ':')).encode('utf-8'))


@dataclass(frozen=True, slots=True)
class DependencyAnalyzer:
    """Derive complete ordered dependencies before an SG fetch begins.

    Attributes:
        allow_ttl_only: Permit safe universal-tag fallback for unanalyzable
            calls.
        enable_additional_filter_presets: Permit configured TTL-only preset
            calls.
    """

    allow_ttl_only: bool = True
    enable_additional_filter_presets: bool = False

    def __post_init__(self) -> None:
        """Validate the explicit dependency fallback and preset flags.

        Raises:
            TypeError: A policy flag is not a boolean.
        """
        require_bool(self.allow_ttl_only, 'TTL-only fallback flag')
        require_bool(
            self.enable_additional_filter_presets,
            'additional-filter-preset flag',
        )

    def _bounded_event_plan(
        self,
        tags: tuple[DependencyTag, ...],
        policy: QueryPolicy,
        identity: Identity,
        site_namespace: str,
        limits: _DependencyLimits,
    ) -> DependencyPlan:
        """Broaden complete dependency classes until both bounds are satisfied."""
        limit_reason = _limit_reason(tags, limits)
        if limit_reason is None:
            return DependencyPlan('event_aware', tags, policy.reason)
        steps = [f'limit:{limit_reason}']
        current = tags
        transformations = (
            (TagKind.ENTITY, 'broaden:entity_to_type', _broaden_entities),
            (TagKind.TYPE_FIELD, 'broaden:field_to_type_value', _broaden_fields),
        )
        for tag_kind, step, transform in transformations:
            if any(tag.kind is tag_kind for tag in current):
                current = transform(current, site_namespace)
                steps.append(step)
                if _limit_reason(current, limits) is None:
                    return DependencyPlan(
                        'event_aware',
                        current,
                        policy.reason,
                        tuple(steps),
                    )
        ttl_tags = universal_tags(identity, site_namespace, include_schema=False)
        if self.allow_ttl_only and _limit_reason(ttl_tags, limits) is None:
            return DependencyPlan(
                'ttl_only',
                ttl_tags,
                'dependency_bounds_exceeded',
                (*steps, 'fallback:ttl_only'),
            )
        return DependencyPlan(
            'bypass',
            (),
            'dependency_bounds_exceeded',
            (*steps, 'fallback:bypass'),
        )

    def _fallback(
        self,
        reason: str,
        identity: Identity,
        site_namespace: str,
        limits: _DependencyLimits,
    ) -> DependencyPlan:
        """Return universal TTL-only tags or a policy-preserving bypass."""
        if not self.allow_ttl_only:
            return DependencyPlan('bypass', (), reason)
        tags = universal_tags(identity, site_namespace, include_schema=False)
        limit_reason = _limit_reason(tags, limits)
        if limit_reason is not None:
            return DependencyPlan(
                'bypass',
                (),
                reason,
                (f'limit:{limit_reason}', 'fallback:bypass'),
            )
        return DependencyPlan(
            'ttl_only',
            tags,
            reason,
        )

    def plan(
        self,
        call: ExactCall,
        identity: Identity,
        site_namespace: str,
        schema: SchemaView,
        limits: _DependencyLimits,
    ) -> DependencyPlan:
        """Build one complete ordered bounded pre-fetch dependency plan.

        Args:
            call: Normalized SG read call.
            identity: Stable effective SG identity.
            site_namespace: Shared invalidation namespace for one SG site.
            schema: Conservative immutable schema view.
            limits: Maximum dependency tag and encoded-vector bounds.

        Returns:
            An event-aware, TTL-only, or bypass dependency plan.
        """
        presets = argument(call, 'additional_filter_presets')
        if presets not in (None, ()) and not self.enable_additional_filter_presets:
            return DependencyPlan('bypass', (), 'additional_filter_presets_disabled')
        policy = classify_query(call, schema)
        if policy.classification == 'ttl_only':
            return self._fallback(policy.reason, identity, site_namespace, limits)
        if policy.classification == 'bypass':
            if policy.reason in {'unknown_field', 'unknown_field_type'}:
                return self._fallback(policy.reason, identity, site_namespace, limits)
            return DependencyPlan('bypass', (), policy.reason)
        tags, failure = _derive_event_tags(call, identity, site_namespace, schema)
        if failure is not None:
            return self._fallback(failure, identity, site_namespace, limits)
        return self._bounded_event_plan(
            tags,
            policy,
            identity,
            site_namespace,
            limits,
        )


def _append_unique(tags: dict[DependencyTag, None], tag: DependencyTag) -> None:
    """Add a dependency only at its first authoritative occurrence."""
    tags[tag] = None


def _add_path_tags(
    tags: dict[DependencyTag, None],
    site_namespace: str,
    schema: SchemaView,
    entity_type: str,
    field_path: str,
) -> _ResolvedPath | None:
    """Add traversal-hop and final-field dependencies in path order."""
    resolved = schema._resolve_path(entity_type, field_path)
    if resolved is None:
        return None
    for hop in resolved._hops:
        _append_unique(
            tags,
            DependencyTag.type_field(site_namespace, hop._source_type, hop._field_name),
        )
        _append_unique(
            tags,
            DependencyTag.type_membership(site_namespace, hop._target_type),
        )
    if resolved._final_field_name != 'id':
        _append_unique(
            tags,
            DependencyTag.type_field(
                site_namespace,
                resolved._final_type,
                resolved._final_field_name,
            ),
        )
    return resolved


def _broaden_entities(
    tags: tuple[DependencyTag, ...],
    site_namespace: str,
) -> tuple[DependencyTag, ...]:
    """Replace the complete exact-entity class with type-wide dependencies."""
    broadened: dict[DependencyTag, None] = {}
    for tag in tags:
        if tag.kind is not TagKind.ENTITY:
            _append_unique(broadened, tag)
            continue
        entity_type = cast('str', tag.components[1])
        _append_unique(
            broadened,
            DependencyTag.type_membership(site_namespace, entity_type),
        )
        _append_unique(broadened, DependencyTag.type_value(site_namespace, entity_type))
    return tuple(broadened)


def _broaden_fields(
    tags: tuple[DependencyTag, ...],
    site_namespace: str,
) -> tuple[DependencyTag, ...]:
    """Replace the complete field and traversal class with type-value tags."""
    broadened: dict[DependencyTag, None] = {}
    for tag in tags:
        if tag.kind not in (TagKind.LINK_PATH, TagKind.TYPE_FIELD):
            _append_unique(broadened, tag)
            continue
        entity_type = cast('str', tag.components[1])
        _append_unique(broadened, DependencyTag.type_value(site_namespace, entity_type))
    return tuple(broadened)


def _ids_from_condition(values: tuple[object, ...]) -> tuple[int, ...] | None:
    """Return finite IDs, no IDs for another field, or `None` for malformed IDs."""
    if len(values) != _CONDITION_LENGTH:
        return None
    if values[0] != 'id' or values[1] not in ('is', 'in'):
        return ()
    candidates = (values[2],) if values[1] == 'is' else values[2]
    if not isinstance(candidates, tuple):
        return None
    entity_ids: list[int] = []
    for entity_id in cast('tuple[object, ...]', candidates):
        if isinstance(entity_id, bool) or not isinstance(entity_id, int) or entity_id <= 0:
            return None
        entity_ids.append(entity_id)
    return tuple(entity_ids)


def _finite_entity_ids(
    call: ExactCall,
    conditions: tuple[object, ...],
) -> tuple[int, ...] | None:
    """Recognize positive finite IDs in a top-level conjunctive tree."""
    if argument(call, 'filter_operator') not in (None, 'all'):
        return None
    entity_ids: dict[int, None] = {}
    for condition in conditions:
        if not isinstance(condition, tuple):
            return None
        values = cast('tuple[object, ...]', condition)
        if filter_group(values)[1] is not None:
            return None
        parsed = _ids_from_condition(values)
        if parsed is None:
            return None
        if not parsed:
            continue
        for entity_id in parsed:
            entity_ids[entity_id] = None
    return tuple(entity_ids) or None


def _limit_reason(
    tags: tuple[DependencyTag, ...],
    limits: _DependencyLimits,
) -> str | None:
    """Report the first authoritative vector bound exceeded by a complete plan."""
    if len(tags) > limits.max_dependency_tags:
        return 'max_dependency_tags'
    if KeyBuilder.worst_case_vector_bytes(tags) > limits.max_dependency_vector_bytes:
        return 'max_dependency_vector_bytes'
    return None


def _object_mapping(value: object) -> Mapping[object, object] | None:
    """Narrow an untyped decoded value to a generic mapping."""
    if not isinstance(value, Mapping):
        return None
    return cast('Mapping[object, object]', value)


def _nested_string_value(value: object) -> str | None:
    """Read one nonempty string from a schema `{value: ...}` wrapper."""
    mapping = _object_mapping(value)
    if mapping is None:
        return None
    nested = mapping.get('value')
    if not isinstance(nested, str) or not nested:
        return None
    return nested


def _order_field(value: object) -> str | None:
    """Extract one field from an exact documented order-item mapping."""
    mapping = pair_mapping(value)
    if mapping is None or frozenset(mapping) != ORDER_ITEM_KEYS:
        return None
    field_name = mapping['field_name']
    if not isinstance(field_name, str) or not field_name:
        return None
    return field_name


def _order_fields(call: ExactCall) -> tuple[str, ...] | None:
    """Extract order field names without changing caller order."""
    raw_order = argument(call, 'order')
    if raw_order in (None, ()):
        return ()
    if not isinstance(raw_order, tuple):
        return None
    fields: list[str] = []
    for item in cast('tuple[object, ...]', raw_order):
        field_name = _order_field(item)
        if field_name is None:
            return None
        fields.append(field_name)
    return tuple(fields)


def _add_order_tags(
    tags: dict[DependencyTag, None],
    call: ExactCall,
    site_namespace: str,
    schema: SchemaView,
    entity_type: str,
) -> str | None:
    """Add order-field dependencies in caller order or report uncertainty."""
    order_fields = _order_fields(call)
    if order_fields is None:
        return 'malformed_order'
    for field_path in order_fields:
        resolved = _add_path_tags(tags, site_namespace, schema, entity_type, field_path)
        if resolved is None:
            return 'unknown_field'
        if not is_known_field_type(resolved._final_field.data_type):
            return 'unknown_field_type'
    return None


def _parse_valid_types(value: object) -> tuple[str, ...] | None:
    """Parse optional schema-ordered valid linked entity types."""
    if value is None:
        return ()
    properties = _object_mapping(value)
    if properties is None:
        return None
    raw_valid_types = properties.get('valid_types')
    if raw_valid_types is None:
        return ()
    valid_types = _object_mapping(raw_valid_types)
    if valid_types is None:
        return None
    raw_values = valid_types.get('value')
    if not isinstance(raw_values, list):
        return None
    values = cast('list[object]', raw_values)
    return (
        None
        if any(not isinstance(item, str) or not item for item in values)
        else tuple(cast('list[str]', values))
    )


def _parse_field(value: object) -> FieldSchema | None:
    """Parse one field only when all required nested facts are well formed."""
    field = _object_mapping(value)
    if field is None:
        return None
    data_type = _nested_string_value(field.get('data_type'))
    if data_type is None:
        return None
    valid_types = _parse_valid_types(field.get('properties'))
    if valid_types is None:
        return None
    try:
        return FieldSchema(data_type, valid_types)
    except (TypeError, ValueError):
        return None


def _parse_fields(value: object) -> tuple[tuple[str, FieldSchema], ...] | None:
    """Parse valid fields for one entity while retaining schema declaration order."""
    if not isinstance(value, Mapping):
        return None
    fields: list[tuple[str, FieldSchema]] = []
    for field_name, raw_field in cast('Mapping[object, object]', value).items():
        if not isinstance(field_name, str) or not field_name:
            continue
        field = _parse_field(raw_field)
        if field is not None:
            fields.append((field_name, field))
    return tuple(fields)


def _raise_compact_malformed(cause: Exception | None = None) -> Never:
    """Raise the compact view's single supported failure."""
    if cause is None:
        raise ValueError('compact schema view is malformed')
    raise ValueError('compact schema view is malformed') from cause


def _add_link_targets(
    tags: dict[DependencyTag, None],
    site_namespace: str,
    field: FieldSchema,
    *,
    include_values: bool,
    selected_types: tuple[str, ...] | None = None,
) -> bool:
    """Add schema-ordered linked target dependencies when targets are known."""
    target_types = selected_types if selected_types is not None else field.valid_types
    if not target_types:
        return False
    for target_type in target_types:
        _append_unique(tags, DependencyTag.type_membership(site_namespace, target_type))
        if include_values:
            _append_unique(tags, DependencyTag.type_value(site_namespace, target_type))
    return True


def _compact_document(data: bytes) -> object:
    """Bound and decode one compact JSON document."""
    try:
        decompressor = zlib.decompressobj()
        expanded = decompressor.decompress(data, _MAX_COMPACT_BYTES)
    except (OSError, zlib.error) as error:
        _raise_compact_malformed(error)
    if decompressor.unconsumed_tail or not decompressor.eof or decompressor.unused_data:
        _raise_compact_malformed()
    try:
        return cast('object', json.loads(expanded))
    except (UnicodeDecodeError, ValueError) as error:
        _raise_compact_malformed(error)


def _compact_entities(
    document: object,
) -> tuple[tuple[str, tuple[tuple[str, FieldSchema], ...]], ...]:
    """Validate one compact document and rebuild immutable field facts."""
    if not isinstance(document, dict):
        _raise_compact_malformed()
    entities: list[tuple[str, tuple[tuple[str, FieldSchema], ...]]] = []
    for entity_type, raw_fields in cast('dict[object, object]', document).items():
        if not isinstance(entity_type, str) or not entity_type or not isinstance(raw_fields, dict):
            _raise_compact_malformed()
        fields: list[tuple[str, FieldSchema]] = []
        for field_name, raw_field in cast('dict[object, object]', raw_fields).items():
            if (
                not isinstance(field_name, str)
                or not field_name
                or not isinstance(raw_field, list)
            ):
                _raise_compact_malformed()
            parts = cast('list[object]', raw_field)
            if (
                len(parts) != _PAIR_LENGTH
                or not isinstance(parts[0], str)
                or not isinstance(parts[1], list)
            ):
                _raise_compact_malformed()
            valid_types = cast('list[object]', parts[1])
            if any(not isinstance(item, str) for item in valid_types):
                _raise_compact_malformed()
            try:
                field = FieldSchema(parts[0], tuple(cast('list[str]', valid_types)))
            except (TypeError, ValueError) as error:
                _raise_compact_malformed(error)
            fields.append((field_name, field))
        entities.append((entity_type, tuple(fields)))
    return tuple(entities)


def _reference_types(
    value: object,
    operator: str,
    valid_types: tuple[str, ...],
) -> tuple[str, ...] | None:
    """Return the schema-ordered target types a proven reference value names."""
    references = (value,) if operator == 'is' else value
    if not isinstance(references, tuple):
        return None
    named: list[str] = []
    for reference in cast('tuple[object, ...]', references):
        mapping = pair_mapping(reference)
        if mapping is None:
            return None
        target_type = mapping.get('type')
        if not isinstance(target_type, str) or target_type not in valid_types:
            return None
        if target_type not in named:
            named.append(target_type)
    if not named:
        return None
    return tuple(target_type for target_type in valid_types if target_type in named)


def _returned_fields(call: ExactCall) -> tuple[str, ...] | None:
    """Normalize omitted and empty returned fields to the implicit `id` field."""
    raw_fields = argument(call, 'fields')
    if raw_fields in (None, ()):
        return ('id',)
    if not isinstance(raw_fields, tuple) or any(
        not isinstance(field, str) or not field for field in cast('tuple[object, ...]', raw_fields)
    ):
        return None
    return cast('tuple[str, ...]', raw_fields)


def _add_returned_tags(
    tags: dict[DependencyTag, None],
    call: ExactCall,
    site_namespace: str,
    schema: SchemaView,
    entity_type: str,
) -> str | None:
    """Add returned-field dependencies in caller order or report uncertainty."""
    fields = _returned_fields(call)
    if fields is None:
        return 'malformed_fields'
    for field_path in fields:
        if field_path == 'id':
            continue
        resolved = _add_path_tags(tags, site_namespace, schema, entity_type, field_path)
        if resolved is None:
            return 'unknown_field'
        if not is_known_field_type(resolved._final_field.data_type):
            return 'unknown_field_type'
        if resolved._final_field.data_type in _LINK_TYPES and not _add_link_targets(
            tags,
            site_namespace,
            resolved._final_field,
            include_values=True,
        ):
            return 'unknown_link_targets'
    return None


def _selected_types(value: object, valid_types: tuple[str, ...]) -> tuple[str, ...]:
    """Select only schema-valid types named by a type operator."""
    if isinstance(value, str):
        requested = (value,)
    elif isinstance(value, tuple):
        values = cast('tuple[object, ...]', value)
        if not all(isinstance(item, str) for item in values):
            return ()
        requested = cast('tuple[str, ...]', values)
    else:
        return ()
    return tuple(target_type for target_type in valid_types if target_type in requested)


def _add_direct_link_filter_tags(
    tags: dict[DependencyTag, None],
    site_namespace: str,
    resolved: _ResolvedPath,
    operator: str,
    value: object,
) -> str | None:
    """Add target existence and value refinements for one direct linked-field filter."""
    field = resolved._final_field
    if operator in _NAME_OPERATORS:
        added = _add_link_targets(tags, site_namespace, field, include_values=True)
    elif operator == 'type_is':
        added = _add_link_targets(
            tags,
            site_namespace,
            field,
            include_values=False,
            selected_types=_selected_types(value, field.valid_types),
        )
    elif operator in _POSITIVE_REFERENCE_OPERATORS:
        added = _add_link_targets(
            tags,
            site_namespace,
            field,
            include_values=False,
            selected_types=_reference_types(value, operator, field.valid_types),
        )
    elif operator in _NEGATIVE_LINK_OPERATORS:
        added = _add_link_targets(tags, site_namespace, field, include_values=False)
    else:
        return None
    return None if added else 'unknown_link_targets'


def _top_level_conditions(call: ExactCall) -> tuple[object, ...] | None:
    """Return the immutable top-level condition sequence."""
    filters = argument(call, 'filters')
    if filters is None:
        return ()
    if not isinstance(filters, tuple):
        return None
    return cast('tuple[object, ...]', filters)


def _walk_conditions(
    conditions: tuple[object, ...],
) -> tuple[tuple[str, str, object], ...]:
    """Flatten nested conditions in caller order after policy validation."""
    flattened: list[tuple[str, str, object]] = []
    for condition in conditions:
        if not isinstance(condition, tuple):
            continue
        values = cast('tuple[object, ...]', condition)
        _is_group, group = filter_group(values)
        if group is not None:
            flattened.extend(_walk_conditions(group))
        elif (
            len(values) == _CONDITION_LENGTH
            and isinstance(values[0], str)
            and isinstance(values[1], str)
        ):
            flattened.append((values[0], values[1], values[2]))
    return tuple(flattened)


def _add_filter_tags(
    tags: dict[DependencyTag, None],
    site_namespace: str,
    schema: SchemaView,
    entity_type: str,
    conditions: tuple[object, ...],
) -> str | None:
    """Add filter dependencies in caller order or report one uncertainty."""
    for field_path, operator, value in _walk_conditions(conditions):
        if field_path == 'id':
            continue
        resolved = _add_path_tags(tags, site_namespace, schema, entity_type, field_path)
        if resolved is None:
            return 'unknown_field'
        if resolved._final_field.data_type in _LINK_TYPES:
            failure = _add_direct_link_filter_tags(
                tags,
                site_namespace,
                resolved,
                operator,
                value,
            )
            if failure is not None:
                return failure
    return None


def universal_tags(
    identity: Identity,
    site_namespace: str,
    *,
    include_schema: bool,
) -> tuple[DependencyTag, ...]:
    """Build universal and exact-human tags in authoritative order.

    Returns:
        Site, optional schema, and authorization tags, followed for a human
        identity by global human-visibility and exact identity-visibility tags.
    """
    tags = [DependencyTag.site(site_namespace)]
    if include_schema:
        tags.append(DependencyTag.schema(site_namespace))
    tags.append(DependencyTag.authorization(site_namespace))
    if identity.kind == 'human':
        tags.append(DependencyTag.human_visibility(site_namespace))
        entity_id = identity.entity_id
        if entity_id is None:
            raise ValueError('human identity requires a positive SG entity ID')
        tags.append(DependencyTag.identity_visibility(site_namespace, entity_id))
    return tuple(tags)


def _derive_event_tags(
    call: ExactCall,
    identity: Identity,
    site_namespace: str,
    schema: SchemaView,
) -> tuple[tuple[DependencyTag, ...], str | None]:
    """Derive a complete narrow tag sequence or one conservative failure reason."""
    entity_type = argument(call, 'entity_type')
    if not isinstance(entity_type, str) or not entity_type:
        return (), 'missing_entity_type'
    conditions = _top_level_conditions(call)
    if conditions is None:
        return (), 'malformed_filters'
    finite_ids = _finite_entity_ids(call, conditions)
    tags = dict.fromkeys(universal_tags(identity, site_namespace, include_schema=True))
    if finite_ids is None:
        _append_unique(tags, DependencyTag.type_membership(site_namespace, entity_type))
    else:
        for entity_id in finite_ids:
            _append_unique(tags, DependencyTag.entity(site_namespace, entity_type, entity_id))
    failure = _add_filter_tags(tags, site_namespace, schema, entity_type, conditions)
    if failure is None:
        failure = _add_returned_tags(tags, call, site_namespace, schema, entity_type)
    if failure is None:
        failure = _add_order_tags(tags, call, site_namespace, schema, entity_type)
    if failure is not None:
        return (), failure
    if argument(call, 'include_archived_projects') is False:
        _append_unique(tags, DependencyTag.type_membership(site_namespace, 'Project'))
        _append_unique(tags, DependencyTag.type_value(site_namespace, 'Project'))
    return tuple(tags), None
