"""Share pure SG wrapper call planning and write-change semantics."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from typing import TYPE_CHECKING, TypeAlias, cast
from urllib.parse import urlsplit, urlunsplit

from sg_query_cache._codec import CodecError, encode_value
from sg_query_cache._dependencies import DependencyAnalyzer, SchemaView, universal_tags
from sg_query_cache._model import (
    DependencyPlan,
    DependencyTag,
    ExactCall,
    ExactValue,
    GenerationChange,
    Identity,
    require_site_namespace,
)
from sg_query_cache._policy import URL_PROVEN_TTL_ONLY_METHODS, is_known_field_type
from sg_query_cache._query import FILTER_GROUP_KEYS, ORDER_ITEM_KEYS
from sg_query_cache._validation import (
    require_bool,
    require_nonempty_string,
    require_positive_int,
)

if TYPE_CHECKING:
    from sg_query_cache._config import CacheSettings

_CallBuilder: TypeAlias = Callable[[], ExactCall]
_MAX_BATCH_CHANGES = 256

HUMAN_VISIBILITY_ENTITY_TYPES = frozenset(
    {
        'Addressing',
        'Group',
        'GroupUserConnection',
        'ProjectUserConnection',
        'TaskAssigneeConnection',
    }
)
HUMAN_VISIBILITY_FIELDS = frozenset(
    {
        'addressings_cc',
        'addressings_to',
        'assigned_to',
        'assignees',
        'group',
        'groups',
        'members',
        'owner',
        'owners',
        'project_members',
        'recipient',
        'recipients',
        'task_assignees',
        'users',
    }
)
IMPLICIT_UPLOAD_FIELDS = frozenset({'filmstrip_image', 'image'})
PERMISSION_FIELDS = frozenset(
    {
        'permission_group',
        'permission_role',
        'permission_role_id',
        'permission_rule_set',
    }
)
PERMISSION_TYPES = frozenset({'ApiUser', 'PermissionRole', 'PermissionRuleSet'})


@dataclass(frozen=True, slots=True)
class ClientContext:
    """Describe immutable response-affecting state for one wrapped SG client.

    Callers must replace this context through the wrapper's `replace_context`
    method when response-affecting client configuration changes. Replacing this
    value does not itself reconfigure the raw client; the two must remain
    consistent.

    Attributes:
        site_url: Normalized SG origin URL.
        site_namespace: Stable invalidation namespace for this SG site.
        localized: Whether the wrapped client localizes response values.
        convert_datetimes_to_utc: Whether the client converts datetimes to UTC.
        records_per_page: Effective server page size of the wrapped client.
    """

    site_url: str
    site_namespace: str
    localized: bool
    convert_datetimes_to_utc: bool
    records_per_page: int

    def __post_init__(self) -> None:
        """Validate and normalize the complete response-affecting context.

        The site URL must be an absolute HTTP or HTTPS origin without
        credentials, query, fragment, or a path other than `/`. Its scheme and
        host are lowercased and its trailing slash is removed; an explicit port
        is retained.

        Raises:
            TypeError: A URL, namespace, boolean option, or page size has the
                wrong type.
            ValueError: The URL is not an accepted origin, the namespace is not
                trimmed nonempty ASCII, or the page size is not positive.
        """
        object.__setattr__(self, 'site_url', _normalize_site_url(self.site_url))
        require_site_namespace(self.site_namespace)
        require_bool(self.localized, 'localized option')
        require_bool(self.convert_datetimes_to_utc, 'datetime-conversion option')
        require_positive_int(self.records_per_page, 'effective page size')


@dataclass(frozen=True, slots=True)
class EntityPlanContext:
    """Bind an entity analyzer to its identity, settings, and site.

    Attributes:
        analyzer: Conservative query dependency analyzer.
        identity: Exact caller identity for permission-aware tags.
        settings: Cache policy and analysis limits.
        site_namespace: Namespace shared by this site's cache keys and tags.
    """

    analyzer: DependencyAnalyzer
    identity: Identity
    settings: CacheSettings
    site_namespace: str


def _identity_bypass_plan(identity: Identity, settings: CacheSettings) -> DependencyPlan | None:
    """Return a bypass plan when settings disable caching for this human identity."""
    if identity.kind == 'human' and settings.identity_policy == 'scripts_only':
        return DependencyPlan('bypass', (), 'human_caching_disabled')
    return None


def _normalize_site_url(value: object) -> str:
    """Normalize one absolute HTTP SG origin without accepting URL secrets."""
    if not isinstance(value, str):
        raise TypeError('site URL must be a string')
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except ValueError:
        raise ValueError('site URL must be a valid HTTP origin') from None
    if (
        parsed.scheme.lower() not in {'http', 'https'}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or parsed.path not in ('', '/')
    ):
        raise ValueError('site URL must be an HTTP origin without credentials or path')
    host = parsed.hostname.lower()
    if ':' in host:
        host = f'[{host}]'
    netloc = host if port is None else f'{host}:{port}'
    return urlunsplit((parsed.scheme.lower(), netloc, '', '', ''))


def _positive_id(value: object) -> bool:
    """Return whether one value is a positive nonboolean integer."""
    return not isinstance(value, bool) and isinstance(value, int) and value > 0


def _returned_field_types(
    fields: list[str] | None,
    entity_type: str,
    schema: SchemaView,
) -> dict[str, str] | None:
    """Return all selected response-field types, or decline unknown field coverage."""
    selected = cast('Sequence[object]', fields or ['id'])
    result: dict[str, str] = {}
    for field_name in selected:
        if not isinstance(field_name, str) or not field_name:
            return None
        field = schema.resolve_field(entity_type, field_name)
        if field is None or not is_known_field_type(field.data_type):
            return None
        result[field_name] = field.data_type
    return result


def _security_visibility_tags(
    site_namespace: str,
    entity_type: str,
    entity_id: int | None,
    fields: Mapping[str, object] | Sequence[str],
) -> tuple[DependencyTag, ...]:
    """Classify conservative authorization and human-visibility write effects."""
    field_names = frozenset(fields)
    tags: list[DependencyTag] = []
    if entity_type in PERMISSION_TYPES or (
        entity_type == 'HumanUser' and not field_names.isdisjoint(PERMISSION_FIELDS)
    ):
        tags.append(DependencyTag.authorization(site_namespace))
    if entity_type == 'HumanUser':
        if entity_id is None:
            tags.append(DependencyTag.human_visibility(site_namespace))
        else:
            tags.append(DependencyTag.identity_visibility(site_namespace, entity_id))
    if entity_type in HUMAN_VISIBILITY_ENTITY_TYPES or not field_names.isdisjoint(
        HUMAN_VISIBILITY_FIELDS
    ):
        global_visibility = DependencyTag.human_visibility(site_namespace)
        if global_visibility not in tags:
            tags.append(global_visibility)
    return tuple(tags)


def batch_has_create(requests: list[dict[str, object]]) -> bool:
    """Return whether a batch may have created an entity with a new ID."""
    return any(request.get('request_type') == 'create' for request in requests)


def bound_call(build: _CallBuilder) -> ExactCall | None:
    """Build one exact call or report an argument outside the closed vocabulary."""
    try:
        return build()
    except (CodecError, TypeError, ValueError):
        return None


def client_options(context: ClientContext, call: ExactCall) -> dict[str, object]:
    """Build the exact-key option vocabulary for one call.

    Returns:
        A new mapping containing localization and datetime-conversion options.
        The effective page size is included only for `find` with a positive
        explicit `page`.
    """
    options: dict[str, object] = {
        'localized': context.localized,
        'convert_datetimes_to_utc': context.convert_datetimes_to_utc,
    }
    if call.method == 'find' and any(
        name == 'page' and isinstance(value, int) and value > 0 for name, value in call.arguments
    ):
        options['records_per_page'] = context.records_per_page
    return options


def create_changes(
    site_namespace: str,
    entity_type: str,
    data: Mapping[str, object],
) -> tuple[GenerationChange, ...]:
    """Build conservative type changes for a create without a known ID."""
    tags = [
        *_security_visibility_tags(site_namespace, entity_type, None, data),
        DependencyTag.type_membership(site_namespace, entity_type),
        DependencyTag.type_value(site_namespace, entity_type),
    ]
    tags.extend(DependencyTag.type_field(site_namespace, entity_type, field) for field in data)
    return tuple(GenerationChange(tag) for tag in tags)


def entity_read_plan(
    call: ExactCall,
    schema: SchemaView,
    entity_type: str,
    fields: list[str] | None,
    context: EntityPlanContext,
) -> tuple[DependencyPlan, dict[str, str] | None]:
    """Analyze one schema-backed entity read and its selected result fields."""
    plan = context.analyzer.plan(
        call,
        context.identity,
        context.site_namespace,
        schema,
        context.settings.limits,
    )
    returned_fields = _returned_field_types(fields, entity_type, schema)
    if returned_fields is None:
        plan = DependencyPlan('bypass', (), 'unknown_returned_field_schema')
    return plan, returned_fields


def exception_types(value: object) -> tuple[type[Exception], ...]:
    """Validate caller-configured transport exception classes.

    Returns:
        The original tuple of `Exception` subclasses.

    Raises:
        TypeError: The value is not a tuple or contains anything other than an
            `Exception` subclass.
    """
    if not isinstance(value, tuple):
        raise TypeError('indeterminate write exceptions must be a tuple')
    values = cast('tuple[object, ...]', value)
    if any(not isinstance(item, type) or not issubclass(item, Exception) for item in values):
        raise TypeError('indeterminate write exceptions must contain exception classes')
    return cast('tuple[type[Exception], ...]', values)


def freeze(value: object) -> ExactValue:
    """Detach supported SG arguments into the exact-call closed vocabulary.

    Lists become tuples; string-keyed mappings become ordered tuples of pairs.
    Supported scalars and temporal values are retained.

    Raises:
        TypeError: A value is unsupported or a mapping key is not a string.
        ValueError: A mapping key is empty.
    """
    if value is None or isinstance(value, bool | int | float | str | bytes | date | datetime):
        return value
    if isinstance(value, Mapping):
        mapping = cast('Mapping[object, object]', value)
        return tuple(
            (require_nonempty_string(name, 'mapping key'), freeze(item))
            for name, item in mapping.items()
        )
    if isinstance(value, list | tuple):
        return tuple(freeze(item) for item in cast('Sequence[object]', value))
    raise TypeError('SG call arguments must use supported immutable value types')


def lifecycle_changes(
    site_namespace: str,
    entity_type: str,
    entity_id: int,
) -> tuple[GenerationChange, ...]:
    """Build exact entity, membership, and value lifecycle changes."""
    tags = (
        *_security_visibility_tags(site_namespace, entity_type, entity_id, ()),
        DependencyTag.entity(site_namespace, entity_type, entity_id),
        DependencyTag.type_membership(site_namespace, entity_type),
        DependencyTag.type_value(site_namespace, entity_type),
    )
    return tuple(GenerationChange(tag) for tag in tags)


def opaque(value: object) -> bytes:
    """Encode one structured SG argument as deterministic closed-codec bytes.

    Raises:
        CodecError: The value is unsupported, nonfinite, or exceeds a codec
            bound.
    """
    return encode_value(value)


def key_freeze(value: object) -> ExactValue:
    """Detach arguments while preserving only recognized mappings as ordered pairs.

    Filter-group and order-item mappings remain tuples of string-keyed pairs.
    Other mappings become opaque closed-codec bytes; lists and tuples are
    recursively detached.

    Raises:
        CodecError: An opaque mapping is unsupported or exceeds a codec bound.
        TypeError: A non-container value has an unsupported type.
    """
    if isinstance(value, Mapping):
        mapping = cast('Mapping[object, object]', value)
        keys = frozenset(mapping)
        if keys in (FILTER_GROUP_KEYS, ORDER_ITEM_KEYS):
            return tuple(
                (require_nonempty_string(name, 'mapping key'), key_freeze(item))
                for name, item in mapping.items()
            )
        return opaque(mapping)
    if isinstance(value, list | tuple):
        return tuple(key_freeze(item) for item in cast('Sequence[object]', value))
    return freeze(value)


def entity_calls(
    method: str, arguments: tuple[tuple[str, object], ...]
) -> tuple[ExactCall | None, ExactCall | None]:
    """Build analysis and exact-key calls from one ordered query vocabulary.

    Args:
        method: The `find` or `find_one` method name.
        arguments: Bound query arguments in the official method's order.

    Returns:
        Both normalized calls, or `None` for a call whose arguments cannot be
        represented by its closed vocabulary.
    """
    analysis = bound_call(
        lambda: ExactCall(method, tuple((name, freeze(value)) for name, value in arguments))
    )
    key = bound_call(
        lambda: ExactCall(method, tuple((name, key_freeze(value)) for name, value in arguments))
    )
    return analysis, key


def parse_schema_view(value: object) -> SchemaView:
    """Rebuild the analyzer view from its compact stored bytes.

    Raises:
        TypeError: The stored value is not bytes.
        ValueError: The compact schema bytes are malformed or exceed the
            expanded-size limit.
    """
    if not isinstance(value, bytes):
        raise TypeError('compact schema view must be bytes')
    return SchemaView.from_compact(value)


def schema_changes(site_namespace: str) -> tuple[GenerationChange, ...]:
    """Build the schema generation change for understood schema writes."""
    return (GenerationChange(DependencyTag.schema(site_namespace)),)


def schema_plan(
    identity: Identity, site_namespace: str, settings: CacheSettings
) -> DependencyPlan:
    """Build the fixed schema plan under the configured identity policy."""
    bypass = _identity_bypass_plan(identity, settings)
    if bypass is not None:
        return bypass
    return DependencyPlan(
        'schema', universal_tags(identity, site_namespace, include_schema=True), 'schema_read'
    )


def site_changes(site_namespace: str) -> tuple[GenerationChange, ...]:
    """Build the broad safe fallback for composite or unknown writes."""
    return (GenerationChange(DependencyTag.site(site_namespace)),)


def create_post_changes(
    site_namespace: str,
    entity_type: str,
    changes: tuple[GenerationChange, ...],
    value: object,
) -> tuple[GenerationChange, ...]:
    """Add a validated created-entity tag or fall back to the whole site."""
    try:
        if not isinstance(value, Mapping):
            return site_changes(site_namespace)
        result = cast('Mapping[object, object]', value)
        entity_id = result.get('id')
        if result.get('type') != entity_type or not _positive_id(entity_id):
            return site_changes(site_namespace)
    except Exception:  # noqa: BLE001 - malformed success requires a site fence
        return site_changes(site_namespace)
    entity_change = GenerationChange(
        DependencyTag.entity(site_namespace, entity_type, cast('int', entity_id))
    )
    return changes if entity_change in changes else (*changes, entity_change)


def batch_post_changes(
    site_namespace: str,
    requests: list[dict[str, object]],
    changes: tuple[GenerationChange, ...],
    value: object,
) -> tuple[GenerationChange, ...]:
    """Add created-entity tags or broaden an uncertain batch result to the site.

    Args:
        site_namespace: Shared invalidation namespace for the batch.
        requests: Original batch requests in dispatch order.
        changes: Conservative changes already derived before dispatch.
        value: Raw batch response, with one result per request.

    Returns:
        The original changes when already site-wide or when no request creates
        an entity. Otherwise, append unique created-entity tags in request
        order, or return only the site change if a required result is
        malformed.
    """
    site = site_changes(site_namespace)
    if changes == site or not batch_has_create(requests):
        return changes
    results = cast('list[object]', value) if isinstance(value, list) else None
    if results is None or len(results) != len(requests):
        return site
    post_changes = dict.fromkeys(changes)
    for request, result in zip(requests, results, strict=True):
        if request.get('request_type') != 'create':
            continue
        entity_type = request.get('entity_type')
        if not isinstance(entity_type, str):
            return site
        created = create_post_changes(
            site_namespace,
            entity_type,
            (),
            result,
        )
        if created == site:
            return site
        post_changes.update(dict.fromkeys(created))
        if len(post_changes) > _MAX_BATCH_CHANGES:
            return site
    return tuple(post_changes)


def ttl_only_plan(
    method: str, identity: Identity, site_namespace: str, settings: CacheSettings
) -> DependencyPlan:
    """Build a universal plan only for an explicitly enabled URL-free method."""
    if method not in URL_PROVEN_TTL_ONLY_METHODS:
        return DependencyPlan('bypass', (), 'composite_url_unproven')
    if method not in settings.enabled_ttl_only_methods:
        return DependencyPlan('bypass', (), 'ttl_only_method_disabled')
    bypass = _identity_bypass_plan(identity, settings)
    if bypass is not None:
        return bypass
    return DependencyPlan(
        'ttl_only',
        universal_tags(identity, site_namespace, include_schema=False),
        'ttl_only_method',
    )


def unconditional_entity_plan(
    identity: Identity,
    settings: CacheSettings,
    entity_type: str,
    bypass_active: Callable[[], bool],
) -> DependencyPlan | None:
    """Apply identity, event-log, and lexical bypasses before schema access."""
    bypass = _identity_bypass_plan(identity, settings)
    if bypass is not None:
        return bypass
    if entity_type == 'EventLogEntry':
        return DependencyPlan('bypass', (), 'event_log_entry')
    if bypass_active():
        return DependencyPlan('bypass', (), 'scoped_bypass')
    return None


def update_changes(
    site_namespace: str,
    entity_type: str,
    entity_id: int,
    data: Mapping[str, object],
) -> tuple[GenerationChange, ...]:
    """Build exact entity, value, changed-field, and membership changes."""
    tags = [
        *_security_visibility_tags(site_namespace, entity_type, entity_id, data),
        DependencyTag.entity(site_namespace, entity_type, entity_id),
        DependencyTag.type_value(site_namespace, entity_type),
    ]
    tags.extend(DependencyTag.type_field(site_namespace, entity_type, field) for field in data)
    tags.append(DependencyTag.type_membership(site_namespace, entity_type))
    return tuple(GenerationChange(tag) for tag in tags)


def batch_changes(
    site_namespace: str,
    requests: list[dict[str, object]],
) -> tuple[GenerationChange, ...]:
    """Union understood batch effects or broaden the whole batch to site."""
    changes: dict[GenerationChange, None] = {}
    for request in requests:
        request_type = request.get('request_type')
        entity_type = request.get('entity_type')
        if not isinstance(entity_type, str) or not entity_type:
            return site_changes(site_namespace)
        data = request.get('data')
        if isinstance(data, Mapping):
            fields = cast('Mapping[object, object]', data)
            if any(not isinstance(field, str) or not field for field in fields):
                return site_changes(site_namespace)
        if request_type == 'create' and isinstance(request.get('data'), Mapping):
            current = create_changes(
                site_namespace,
                entity_type,
                cast('Mapping[str, object]', request['data']),
            )
        elif (
            request_type == 'update'
            and _positive_id(request.get('entity_id'))
            and isinstance(request.get('data'), Mapping)
        ):
            current = update_changes(
                site_namespace,
                entity_type,
                cast('int', request['entity_id']),
                cast('Mapping[str, object]', request['data']),
            )
        elif request_type == 'delete' and _positive_id(request.get('entity_id')):
            current = lifecycle_changes(
                site_namespace,
                entity_type,
                cast('int', request['entity_id']),
            )
        else:
            return site_changes(site_namespace)
        changes.update(dict.fromkeys(current))
        if len(changes) > _MAX_BATCH_CHANGES:
            return site_changes(site_namespace)
    return tuple(changes) if changes else site_changes(site_namespace)
