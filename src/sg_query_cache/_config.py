"""Load and validate the shared versioned cache configuration."""

from __future__ import annotations

import math
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from ipaddress import ip_address
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Self, TypeAlias, cast, get_args

from sg_query_cache._codec import MAX_BYTES
from sg_query_cache._model import require_site_namespace
from sg_query_cache._policy import URL_PROVEN_TTL_ONLY_METHODS
from sg_query_cache._query import argument
from sg_query_cache._validation import (
    require_bool,
    require_nonempty_string,
    require_positive_int,
    require_positive_number,
)

if TYPE_CHECKING:
    from sg_query_cache._model import DependencyPlan, ExactCall, Identity, IdentityKind

CacheLayer: TypeAlias = Literal['hierarchical_l1', 'l2', 'local_l1']
_IdentityPolicy: TypeAlias = Literal['scripts_only', 'scripts_and_humans']
_ValkeyMode: TypeAlias = Literal['managed', 'sentinel', 'standalone']

VALKEY_MAX_DEPENDENCY_TAGS_V1 = 32
_CACHE_KEYS = frozenset(
    {
        'enable_additional_filter_presets',
        'enabled_ttl_only_methods',
        'global_safety_max_seconds',
        'identity_policy',
        'l1',
        'l2',
        'limits',
        'metrics_digest_secret_env',
        'site_namespace',
        'ttl_rule',
    }
)
_GLOBAL_MAXIMUM_SECONDS = 3600
_L1_KEYS = frozenset(
    {
        'enabled',
        'hierarchical_max_age_seconds',
        'local_only_max_age_seconds',
        'metadata_budget_bytes',
        'result_budget_bytes',
    }
)
_L2_KEYS = frozenset(
    {
        'acknowledge_mutually_trusted_clients',
        'backend',
        'enabled',
        'event_aware_max_age_seconds',
        'metadata_budget_bytes',
        'result_budget_bytes',
        'schema_max_age_seconds',
        'ttl_only_max_age_seconds',
        'valkey',
    }
)
_LIMIT_KEYS = frozenset(
    {
        'circuit_failure_threshold',
        'circuit_probe_seconds',
        'clock_error_bound_seconds',
        'connect_timeout_seconds',
        'fill_lease_seconds',
        'incarnation_validation_seconds',
        'lease_wait_seconds',
        'max_dependency_tags',
        'max_dependency_vector_bytes',
        'max_serialized_result_bytes',
        'operation_timeout_seconds',
        'scoped_generation_retention_seconds',
        'signed_url_safety_margin_seconds',
        'transient_image_max_age_seconds',
    }
)
_MAX_DNS_LABEL_LENGTH = 63
_MAX_DNS_NAME_LENGTH = 253
_MAX_KEY_PREFIX_LENGTH = 64
_MAX_TCP_PORT = 65_535
_MIN_SENTINELS = 3
_QUERY_CLASSES = frozenset({'event_aware', 'schema', 'ttl_only'})
_ROOT_KEYS = frozenset({'cache', 'schema_version'})
_RULE_PRECEDENCE = {
    (True, True, False, False): 5,
    (True, False, False, False): 4,
    (False, True, False, False): 3,
    (False, False, True, False): 2,
    (False, False, False, True): 1,
}
_TTL_RULE_KEYS = frozenset(
    {'entity_type', 'identity_kind', 'maximum_age_seconds', 'method', 'query_class'}
)
_VALKEY_KEYS = frozenset({'metadata', 'result'})
_VALKEY_ROLE_KEYS = frozenset(
    {
        'database',
        'host',
        'key_prefix',
        'mode',
        'password_env',
        'port',
        'sentinels',
        'service_name',
        'tls',
        'username',
    }
)


@dataclass(frozen=True, slots=True)
class L1Settings:
    """Configure the optional byte-bounded in-process cache layer.

    A local-only cache uses `local_only_max_age_seconds`; an L1 above L2 uses
    `hierarchical_max_age_seconds` and cannot outlive the authoritative L2 hit.
    Result and metadata byte budgets are enforced separately per local store.
    Disabling this layer requires L2 to be enabled in `CacheSettings`.
    """

    enabled: bool = True
    hierarchical_max_age_seconds: float = 10
    local_only_max_age_seconds: float = 30
    result_budget_bytes: int = 134_217_728
    metadata_budget_bytes: int = 16_777_216

    def __post_init__(self) -> None:
        """Validate local age limits and positive storage budgets.

        Raises:
            TypeError: The enabled flag, an age, or a byte budget has the wrong
                type.
            ValueError: An age or byte budget is not positive.
        """
        require_bool(self.enabled, 'L1 enabled')
        require_positive_number(self.hierarchical_max_age_seconds, 'L1 hierarchical maximum age')
        require_positive_number(self.local_only_max_age_seconds, 'L1 local-only maximum age')
        require_positive_int(self.result_budget_bytes, 'L1 result budget')
        require_positive_int(self.metadata_budget_bytes, 'L1 metadata budget')


@dataclass(frozen=True, slots=True)
class ValkeyRoleSettings:
    """Configure one standalone, Sentinel, or managed Valkey service role.

    Password values are resolved only while constructing a client and are never
    retained by this immutable configuration value.

    `host`, `port`, and `database` identify the service in standalone and
    managed modes. Sentinel mode discovers the primary through at least three
    distinct `sentinels` addresses and requires `service_name`; its `host`
    field remains required but is not used as the discovered endpoint.
    `password_env` names an environment variable, not a password. Clients
    sharing cached data must use the same `key_prefix` for each role.
    """

    host: str
    password_env: str
    mode: _ValkeyMode = 'standalone'
    port: int = 6379
    database: int = 0
    tls: bool = True
    username: str | None = None
    key_prefix: str = 'sgqc'
    sentinels: tuple[str, ...] = ()
    service_name: str | None = None

    def __post_init__(self) -> None:
        """Validate one role endpoint, namespace, and deployment mode.

        Raises:
            TypeError: A connection setting has the wrong type.
            ValueError: An endpoint or namespace is malformed, or Sentinel
                settings do not match the selected mode.
        """
        _require_valkey_host(self.host)
        _require_environment_name(self.password_env, 'Valkey password environment name')
        _valkey_mode(self.mode)
        _require_valkey_port(self.port)
        _require_nonnegative_int(self.database, 'Valkey database')
        require_bool(self.tls, 'Valkey TLS')
        if self.username is not None:
            _require_safe_ascii(self.username, 'Valkey username')
        _require_key_prefix(self.key_prefix)
        sentinels = _require_string_tuple(self.sentinels, 'Valkey sentinels')
        if self.mode == 'sentinel':
            parsed = tuple(parse_sentinel_address(value) for value in sentinels)
            distinct = {_endpoint_identity(host, port) for host, port in parsed}
            if len(parsed) < _MIN_SENTINELS or len(distinct) < _MIN_SENTINELS:
                raise ValueError('Valkey Sentinel mode requires three distinct sentinel addresses')
            if self.service_name is None:
                raise ValueError('Valkey Sentinel mode requires a service name')
            _require_safe_ascii(self.service_name, 'Valkey Sentinel service name')
        elif sentinels or self.service_name is not None:
            raise ValueError('Valkey sentinels and service name require sentinel mode')


@dataclass(frozen=True, slots=True)
class ValkeySettings:
    """Configure the separate result and metadata Valkey service roles.

    Both roles must use the same deployment mode and distinct endpoints.
    `allow_same_endpoint_for_testing` permits a shared endpoint only for direct
    construction in disposable tests; the TOML loader does not expose it.
    """

    result: ValkeyRoleSettings = field(
        default_factory=lambda: ValkeyRoleSettings(
            'cache-results',
            'SG_QUERY_CACHE_RESULT_PASSWORD',
        )
    )
    metadata: ValkeyRoleSettings = field(
        default_factory=lambda: ValkeyRoleSettings(
            'cache-metadata',
            'SG_QUERY_CACHE_METADATA_PASSWORD',
        )
    )
    allow_same_endpoint_for_testing: bool = field(default=False, repr=False)

    def __post_init__(self) -> None:
        """Validate role types, matching modes, and separate endpoints.

        Raises:
            TypeError: A role is not a `ValkeyRoleSettings` value or the test
                flag is not boolean.
            ValueError: Role modes differ or endpoints coincide without the
                explicit test allowance.
        """
        result_value = cast('object', self.result)
        metadata_value = cast('object', self.metadata)
        if not isinstance(result_value, ValkeyRoleSettings) or not isinstance(
            metadata_value, ValkeyRoleSettings
        ):
            raise TypeError('Valkey roles must be immutable ValkeyRoleSettings values')
        require_bool(
            self.allow_same_endpoint_for_testing,
            'same-endpoint test flag',
        )
        if self.result.mode != self.metadata.mode:
            raise ValueError('Valkey roles must use the same deployment mode')
        if self.result.mode == 'sentinel':
            result_endpoint = (self.result.service_name, self.result.database)
            metadata_endpoint = (self.metadata.service_name, self.metadata.database)
        else:
            result_endpoint = (
                *_endpoint_identity(self.result.host, self.result.port),
                self.result.database,
            )
            metadata_endpoint = (
                *_endpoint_identity(self.metadata.host, self.metadata.port),
                self.metadata.database,
            )
        if not self.allow_same_endpoint_for_testing and result_endpoint == metadata_endpoint:
            raise ValueError('Valkey result and metadata roles must use separate endpoints')


@dataclass(frozen=True, slots=True)
class L2Settings:
    """Configure the optional shared cache layer and its trust acknowledgement.

    The event-aware, TTL-only, and schema age fields select separate shared
    cache defaults. Byte budgets describe deployment sizing targets; this
    client does not configure or enforce Valkey server memory limits.
    `acknowledge_mutually_trusted_clients` is required when shared caching uses
    human identities and acknowledges that clients can read and modify each
    other's cache data.
    """

    enabled: bool = False
    backend: str = 'valkey'
    valkey: ValkeySettings = field(default_factory=ValkeySettings)
    event_aware_max_age_seconds: float = 600
    ttl_only_max_age_seconds: float = 60
    schema_max_age_seconds: float = 1800
    result_budget_bytes: int = 8_589_934_592
    metadata_budget_bytes: int = 1_073_741_824
    acknowledge_mutually_trusted_clients: bool = False

    def __post_init__(self) -> None:
        """Validate shared-layer policy, budgets, and Valkey configuration.

        Raises:
            TypeError: A policy flag, age, byte budget, or Valkey settings
                value has the wrong type.
            ValueError: The backend is unsupported or an age or budget is not
                positive.
        """
        require_bool(self.enabled, 'L2 enabled')
        if self.backend != 'valkey':
            raise ValueError('L2 backend must be valkey')
        if not isinstance(cast('object', self.valkey), ValkeySettings):
            raise TypeError('L2 Valkey settings must be an immutable ValkeySettings value')
        require_positive_number(self.event_aware_max_age_seconds, 'L2 event-aware maximum age')
        require_positive_number(self.ttl_only_max_age_seconds, 'L2 TTL-only maximum age')
        require_positive_number(self.schema_max_age_seconds, 'L2 schema maximum age')
        require_positive_int(self.result_budget_bytes, 'L2 result budget')
        require_positive_int(self.metadata_budget_bytes, 'L2 metadata budget')
        require_bool(
            self.acknowledge_mutually_trusted_clients,
            'L2 mutually-trusted-client acknowledgement',
        )


@dataclass(frozen=True, slots=True)
class LimitSettings:
    """Configure bounded storage, dependency, lease, timeout, and URL limits.

    All duration fields use seconds. The serialized-result limit cannot exceed
    1 MiB. Scoped-generation retention controls idle metadata expiry; site and
    global generations do not expire through this setting.

    The synchronous operation timeout is a maintained-client socket timeout.
    The asynchronous timeout is one cooperative budget for an admitted backend
    operation, including connection preparation and cleanup. Cancellation and
    close still join owned work and can exceed that budget. Incarnation
    validation uses a jittered interval, not a strict maximum.
    """

    fill_lease_seconds: float = 30
    lease_wait_seconds: float = 2
    max_dependency_tags: int = 32
    max_dependency_vector_bytes: int = 2048
    max_serialized_result_bytes: int = 1_048_576
    scoped_generation_retention_seconds: float = 7200
    connect_timeout_seconds: float = 0.1
    operation_timeout_seconds: float = 0.25
    circuit_failure_threshold: int = 5
    circuit_probe_seconds: float = 10
    incarnation_validation_seconds: float = 30
    signed_url_safety_margin_seconds: float = 30
    transient_image_max_age_seconds: float = 5
    clock_error_bound_seconds: float | None = None

    def __post_init__(self) -> None:
        """Validate operational bounds and their codec and clock relationships.

        Raises:
            TypeError: A bound has the wrong numeric type or is not finite.
            ValueError: A bound is outside its range, the result limit exceeds
                1 MiB, or the clock error reaches the URL safety margin.
        """
        require_positive_number(self.fill_lease_seconds, 'fill-lease duration')
        require_positive_number(self.lease_wait_seconds, 'lease-wait budget')
        require_positive_int(self.max_dependency_tags, 'maximum dependency tags')
        require_positive_int(self.max_dependency_vector_bytes, 'maximum dependency-vector bytes')
        require_positive_int(self.max_serialized_result_bytes, 'maximum serialized-result bytes')
        if self.max_serialized_result_bytes > MAX_BYTES:
            raise ValueError(
                'maximum serialized-result bytes cannot exceed the 1 MiB codec ceiling'
            )
        require_positive_number(
            self.scoped_generation_retention_seconds,
            'scoped-generation retention',
        )
        require_positive_number(self.connect_timeout_seconds, 'backend connect timeout')
        require_positive_number(self.operation_timeout_seconds, 'backend operation timeout')
        require_positive_int(self.circuit_failure_threshold, 'circuit failure threshold')
        require_positive_number(self.circuit_probe_seconds, 'circuit probe interval')
        require_positive_number(
            self.incarnation_validation_seconds,
            'incarnation validation interval',
        )
        require_positive_number(
            self.signed_url_safety_margin_seconds,
            'signed-URL safety margin',
        )
        require_positive_number(
            self.transient_image_max_age_seconds,
            'transient-image maximum age',
        )
        if self.clock_error_bound_seconds is not None:
            _require_nonnegative_number(self.clock_error_bound_seconds, 'clock error bound')
            if self.clock_error_bound_seconds >= self.signed_url_safety_margin_seconds:
                raise ValueError(
                    'clock error bound must be smaller than the signed-URL safety margin'
                )


@dataclass(frozen=True, slots=True)
class TTLRule:
    """Override a layer default for one supported policy match.

    Exactly one match shape is allowed, in descending precedence: method plus
    entity type, method alone, entity type alone, query class alone, or
    identity kind alone. Rule order does not override this precedence.

    Attributes:
        maximum_age_seconds: Positive age no greater than 3,600 seconds.
        method: Optional SG read method match.
        entity_type: Optional SG entity type match.
        query_class: Optional query-policy classification match.
        identity_kind: Optional `script` or `human` match.
    """

    maximum_age_seconds: float
    method: str | None = None
    entity_type: str | None = None
    query_class: str | None = None
    identity_kind: str | None = None

    @property
    def _match_key(self) -> tuple[bool, bool, bool, bool]:
        """Return the match-field shape used for validation and precedence."""
        return (
            self.method is not None,
            self.entity_type is not None,
            self.query_class is not None,
            self.identity_kind is not None,
        )

    def __post_init__(self) -> None:
        """Validate the age cap and one supported unambiguous match shape.

        Raises:
            TypeError: The age or a supplied match value has the wrong type.
            ValueError: The age is outside (0, 3600], a match value is invalid,
                or the match combination is unsupported.
        """
        require_positive_number(self.maximum_age_seconds, 'TTL-rule maximum age')
        if self.maximum_age_seconds > _GLOBAL_MAXIMUM_SECONDS:
            raise ValueError('TTL-rule maximum age cannot exceed 3,600 seconds')
        for value in (self.method, self.entity_type, self.query_class, self.identity_kind):
            if value is not None:
                require_nonempty_string(value, 'TTL-rule match value')
        if self.identity_kind is not None and self.identity_kind not in ('script', 'human'):
            raise ValueError('TTL-rule identity kind must be script or human')
        if self.query_class is not None and self.query_class not in _QUERY_CLASSES:
            raise ValueError('TTL-rule query class must be event_aware, ttl_only, or schema')
        if not any((self.method, self.entity_type, self.query_class, self.identity_kind)):
            raise ValueError('TTL rule requires at least one match field')
        if self._match_key not in _RULE_PRECEDENCE:
            raise ValueError('unsupported or ambiguous TTL-rule match combination')

    @property
    def _match_tuple(self) -> tuple[str | None, str | None, str | None, str | None]:
        """Return the complete match tuple used for duplicate detection."""
        return (self.method, self.entity_type, self.query_class, self.identity_kind)


@dataclass(frozen=True, slots=True)
class TTLContext:
    """Describe every explicit fact used to select one maximum age.

    Attributes:
        method: Normalized SG read method.
        entity_type: Optional SG entity type.
        query_class: Dependency-plan classification or schema-read class.
        identity_kind: Effective identity class.
        layer: Active cache layer and deployment position.
        scoped_bypass: Whether the caller requires a direct SG read.
        response_deadline_seconds: Optional response-derived remaining
            lifetime.
        remaining_l2_seconds: Optional authoritative L2 lifetime during
            promotion.
    """

    method: str
    entity_type: str | None
    query_class: str
    identity_kind: IdentityKind
    layer: CacheLayer
    scoped_bypass: bool = False
    response_deadline_seconds: float | None = None
    remaining_l2_seconds: float | None = None

    def __post_init__(self) -> None:
        """Validate the lookup identity, layer, and optional lifetime caps.

        Raises:
            TypeError: A context value has the wrong type or a lifetime cap is
                not finite.
            ValueError: A required string is empty, the identity or layer is
                unsupported, or a lifetime cap is negative.
        """
        require_nonempty_string(self.method, 'TTL-context method')
        if self.entity_type is not None:
            require_nonempty_string(self.entity_type, 'TTL-context entity type')
        require_nonempty_string(self.query_class, 'TTL-context query class')
        if self.identity_kind not in ('script', 'human'):
            raise ValueError('TTL-context identity kind must be script or human')
        if self.layer not in ('hierarchical_l1', 'l2', 'local_l1'):
            raise ValueError('TTL-context layer is unsupported')
        require_bool(self.scoped_bypass, 'TTL-context scoped-bypass flag')
        for cap, owner in (
            (self.response_deadline_seconds, 'response-derived deadline'),
            (self.remaining_l2_seconds, 'remaining L2 age'),
        ):
            if cap is not None:
                _require_nonnegative_number(cap, owner)

    @classmethod
    def from_call(
        cls,
        call: ExactCall,
        identity: Identity,
        plan: DependencyPlan,
        *,
        layer: CacheLayer,
    ) -> Self:
        """Build a TTL lookup context from the exact call, identity, and dependency plan.

        Args:
            call: Normalized exact SG call.
            identity: Stable effective SG identity.
            plan: Complete pre-fetch dependency decision.
            layer: Active cache layer and deployment position.

        Returns:
            An immutable explicit TTL lookup context.
        """
        value = argument(call, 'entity_type')
        entity_type = value if isinstance(value, str) and value else None
        return cls(
            method=call.method,
            entity_type=entity_type,
            query_class=plan.classification,
            identity_kind=identity.kind,
            layer=layer,
        )


@dataclass(frozen=True, slots=True)
class CacheSettings:
    """Hold validated version-one settings shared by every cache process.

    At least one cache layer must be enabled. Processes sharing a namespace
    must agree on identity policy, limits, and role settings. The global safety
    maximum cannot exceed 3,600 seconds and caps configured and per-call ages.

    `enabled_ttl_only_methods` can enable `work_schedule_read`.
    `enable_additional_filter_presets` permits supported preset-bearing reads
    under TTL-only policy. `metrics_digest_secret_env` names a secret resolved
    when telemetry is constructed; loading settings does not read it.
    """

    schema_version: int = 1
    identity_policy: _IdentityPolicy = 'scripts_only'
    global_safety_max_seconds: float = 3600
    metrics_digest_secret_env: str | None = field(default=None, repr=False)
    site_namespace: str = 'production'
    enabled_ttl_only_methods: tuple[str, ...] = ()
    enable_additional_filter_presets: bool = False
    l1: L1Settings = field(default_factory=L1Settings)
    l2: L2Settings = field(default_factory=L2Settings)
    limits: LimitSettings = field(default_factory=LimitSettings)
    ttl_rules: tuple[TTLRule, ...] = ()

    def __post_init__(self) -> None:
        """Validate the complete topology and its shared policy constraints.

        Raises:
            TypeError: A component, policy flag, rule collection, or numeric
                bound has the wrong type.
            ValueError: The configuration has unsupported policy values,
                conflicting rules, unsafe shared settings, or no enabled layer.
        """
        _require_cache_components(self.l1, self.l2, self.limits)
        if self.l2.enabled and self.limits.max_dependency_tags > VALKEY_MAX_DEPENDENCY_TAGS_V1:
            raise ValueError('shared Valkey dependency tag limit cannot exceed the v1 ceiling')
        if self.schema_version != 1:
            raise ValueError('unsupported cache schema version')
        if self.identity_policy not in ('scripts_only', 'scripts_and_humans'):
            raise ValueError('identity policy must be scripts_only or scripts_and_humans')
        require_positive_number(self.global_safety_max_seconds, 'global safety maximum')
        if self.global_safety_max_seconds > _GLOBAL_MAXIMUM_SECONDS:
            raise ValueError('global safety maximum cannot exceed 3,600 seconds')
        if self.metrics_digest_secret_env is not None:
            require_nonempty_string(
                self.metrics_digest_secret_env, 'metrics secret environment name'
            )
        require_site_namespace(self.site_namespace)
        require_bool(
            self.enable_additional_filter_presets,
            'additional-filter-preset enablement',
        )
        ttl_only_methods = _require_string_tuple(
            self.enabled_ttl_only_methods,
            'enabled TTL-only methods',
        )
        ttl_rules = _require_ttl_rule_tuple(self.ttl_rules)
        _validate_ttl_only_methods(ttl_only_methods)
        _reject_duplicate_rules(ttl_rules)
        if (
            self.identity_policy == 'scripts_and_humans'
            and self.l2.enabled
            and not self.l2.acknowledge_mutually_trusted_clients
        ):
            raise ValueError(
                'shared human caching requires acknowledging mutually trusted clients'
            )
        if not self.l1.enabled and not self.l2.enabled:
            raise ValueError('at least one cache layer must be enabled')

    @classmethod
    def from_toml(cls, path: str | Path) -> Self:
        """Load a versioned configuration without reading referenced secrets.

        Args:
            path: TOML file containing the shared cache configuration.

        Returns:
            A fully defaulted immutable settings value.

        Raises:
            OSError: The configuration file cannot be opened or read.
            TypeError: A configuration scalar or table has the wrong type.
            ValueError: TOML syntax or a configuration value is invalid,
                unknown, unsupported, or unsafe.
        """
        try:
            with Path(path).open('rb') as stream:
                document = tomllib.load(stream)
        except tomllib.TOMLDecodeError:
            raise ValueError('invalid TOML configuration') from None
        return cls._from_mapping(document)

    @classmethod
    def _from_mapping(cls, document: Mapping[str, object]) -> Self:
        """Build settings from a decoded TOML mapping."""
        _reject_unknown_keys(document, _ROOT_KEYS, 'top-level configuration')
        schema_version = _integer(document, 'schema_version', 0)
        cache = _mapping(document.get('cache', {}), 'cache table')
        _reject_unknown_keys(cache, _CACHE_KEYS, 'cache table')

        l1_mapping = _mapping(cache.get('l1', {}), 'cache.l1 table')
        l2_mapping = _mapping(cache.get('l2', {}), 'cache.l2 table')
        limits_mapping = _mapping(cache.get('limits', {}), 'cache.limits table')
        _reject_unknown_keys(l1_mapping, _L1_KEYS, 'cache.l1 table')
        _reject_unknown_keys(l2_mapping, _L2_KEYS, 'cache.l2 table')
        _reject_unknown_keys(limits_mapping, _LIMIT_KEYS, 'cache.limits table')

        l1 = _load_l1(l1_mapping)
        l2 = _load_l2(l2_mapping)
        limits = _load_limits(limits_mapping)
        ttl_rules = _load_ttl_rules(cache.get('ttl_rule', ()))
        defaults = CacheSettings()
        return cls(
            schema_version=schema_version,
            identity_policy=_identity_policy(
                cache.get('identity_policy', defaults.identity_policy)
            ),
            global_safety_max_seconds=_number(
                cache, 'global_safety_max_seconds', defaults.global_safety_max_seconds
            ),
            metrics_digest_secret_env=_optional_string(cache, 'metrics_digest_secret_env'),
            site_namespace=_string(cache, 'site_namespace', defaults.site_namespace),
            enabled_ttl_only_methods=_string_tuple(cache, 'enabled_ttl_only_methods'),
            enable_additional_filter_presets=_boolean(
                cache,
                'enable_additional_filter_presets',
                default=defaults.enable_additional_filter_presets,
            ),
            l1=l1,
            l2=l2,
            limits=limits,
            ttl_rules=ttl_rules,
        )


def _boolean(mapping: Mapping[str, object], key: str, *, default: bool) -> bool:
    """Read one strict TOML boolean."""
    value = mapping.get(key, default)
    return require_bool(value, key)


def _endpoint_identity(host: str, port: int) -> tuple[str, int]:
    """Normalize equivalent IP and DNS endpoint spellings."""
    try:
        normalized = str(ip_address(host))
    except ValueError:
        normalized = host.removesuffix('.').casefold()
    return normalized, port


def _identity_policy(value: object) -> _IdentityPolicy:
    """Read the closed identity-policy vocabulary."""
    if value == 'scripts_only':
        return 'scripts_only'
    if value == 'scripts_and_humans':
        return 'scripts_and_humans'
    raise ValueError('identity policy must be scripts_only or scripts_and_humans')


def _integer(mapping: Mapping[str, object], key: str, default: int) -> int:
    """Read one strict TOML integer."""
    value = mapping.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f'{key} must be an integer')
    return value


def _layer_default_age(settings: CacheSettings, context: TTLContext) -> float:
    """Return the configured default for one active layer and query class."""
    if context.layer == 'hierarchical_l1':
        if not settings.l1.enabled:
            raise ValueError('cannot select an age for disabled L1')
        return settings.l1.hierarchical_max_age_seconds
    if context.layer == 'local_l1':
        if not settings.l1.enabled:
            raise ValueError('cannot select an age for disabled L1')
        return settings.l1.local_only_max_age_seconds
    if not settings.l2.enabled:
        raise ValueError('cannot select an age for disabled L2')
    if context.query_class == 'event_aware':
        return settings.l2.event_aware_max_age_seconds
    if context.query_class == 'ttl_only':
        return settings.l2.ttl_only_max_age_seconds
    if context.query_class == 'schema':
        return settings.l2.schema_max_age_seconds
    raise ValueError('L2 maximum age requires a cacheable query class')


def _mapping(value: object, owner: str) -> Mapping[str, object]:
    """Require a string-keyed decoded TOML table."""
    if not isinstance(value, Mapping):
        raise TypeError(f'{owner} must be a table')
    untyped = cast('Mapping[object, object]', value)
    if not all(isinstance(key, str) for key in untyped):
        raise ValueError(f'{owner} must use string keys')
    return cast('Mapping[str, object]', untyped)


def _number(mapping: Mapping[str, object], key: str, default: float) -> float:
    """Read one finite TOML integer or float."""
    value = mapping.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f'{key} must be a number')
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f'{key} must be finite')
    return result


def _load_l1(mapping: Mapping[str, object]) -> L1Settings:
    """Load the L1 settings table."""
    defaults = L1Settings()
    return L1Settings(
        enabled=_boolean(mapping, 'enabled', default=defaults.enabled),
        hierarchical_max_age_seconds=_number(
            mapping, 'hierarchical_max_age_seconds', defaults.hierarchical_max_age_seconds
        ),
        local_only_max_age_seconds=_number(
            mapping, 'local_only_max_age_seconds', defaults.local_only_max_age_seconds
        ),
        result_budget_bytes=_integer(mapping, 'result_budget_bytes', defaults.result_budget_bytes),
        metadata_budget_bytes=_integer(
            mapping, 'metadata_budget_bytes', defaults.metadata_budget_bytes
        ),
    )


def _optional_number(mapping: Mapping[str, object], key: str) -> float | None:
    """Read an optional finite TOML integer or float."""
    if key not in mapping:
        return None
    return _number(mapping, key, 0)


def _load_limits(mapping: Mapping[str, object]) -> LimitSettings:
    """Load the bounded operational limits table."""
    defaults = LimitSettings()
    return LimitSettings(
        fill_lease_seconds=_number(mapping, 'fill_lease_seconds', defaults.fill_lease_seconds),
        lease_wait_seconds=_number(mapping, 'lease_wait_seconds', defaults.lease_wait_seconds),
        max_dependency_tags=_integer(mapping, 'max_dependency_tags', defaults.max_dependency_tags),
        max_dependency_vector_bytes=_integer(
            mapping, 'max_dependency_vector_bytes', defaults.max_dependency_vector_bytes
        ),
        max_serialized_result_bytes=_integer(
            mapping,
            'max_serialized_result_bytes',
            defaults.max_serialized_result_bytes,
        ),
        scoped_generation_retention_seconds=_number(
            mapping,
            'scoped_generation_retention_seconds',
            defaults.scoped_generation_retention_seconds,
        ),
        connect_timeout_seconds=_number(
            mapping, 'connect_timeout_seconds', defaults.connect_timeout_seconds
        ),
        operation_timeout_seconds=_number(
            mapping, 'operation_timeout_seconds', defaults.operation_timeout_seconds
        ),
        circuit_failure_threshold=_integer(
            mapping, 'circuit_failure_threshold', defaults.circuit_failure_threshold
        ),
        circuit_probe_seconds=_number(
            mapping, 'circuit_probe_seconds', defaults.circuit_probe_seconds
        ),
        incarnation_validation_seconds=_number(
            mapping,
            'incarnation_validation_seconds',
            defaults.incarnation_validation_seconds,
        ),
        signed_url_safety_margin_seconds=_number(
            mapping,
            'signed_url_safety_margin_seconds',
            defaults.signed_url_safety_margin_seconds,
        ),
        transient_image_max_age_seconds=_number(
            mapping,
            'transient_image_max_age_seconds',
            defaults.transient_image_max_age_seconds,
        ),
        clock_error_bound_seconds=_optional_number(mapping, 'clock_error_bound_seconds'),
    )


def _reject_duplicate_rules(rules: tuple[TTLRule, ...]) -> None:
    """Reject duplicate matches rather than choose by declaration order."""
    matches = [rule._match_tuple for rule in rules]  # pyright: ignore[reportPrivateUsage]  # noqa: SLF001
    if len(matches) != len(set(matches)):
        raise ValueError('configuration contains a duplicate TTL rule match')


def _reject_unknown_keys(
    mapping: Mapping[str, object], allowed: frozenset[str], owner: str
) -> None:
    """Reject schema extensions so every process interprets one closed document."""
    if not mapping.keys() <= allowed:
        raise ValueError(f'{owner} contains an unknown key')


def _require_cache_components(l1: object, l2: object, limits: object) -> None:
    """Require immutable validated nested settings values."""
    if not isinstance(l1, L1Settings):
        raise TypeError('L1 settings must be an immutable L1Settings value')
    if not isinstance(l2, L2Settings):
        raise TypeError('L2 settings must be an immutable L2Settings value')
    if not isinstance(limits, LimitSettings):
        raise TypeError('limit settings must be an immutable LimitSettings value')


def _require_nonnegative_int(value: object, owner: str) -> None:
    """Require a nonnegative integer."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f'{owner} must be an integer')
    if value < 0:
        raise ValueError(f'{owner} must be nonnegative')


def _require_nonnegative_number(value: object, owner: str) -> None:
    """Require a finite nonnegative number."""
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise TypeError(f'{owner} must be a finite number')
    if value < 0:
        raise ValueError(f'{owner} must be nonnegative')


def _require_safe_ascii(value: object, owner: str) -> str:
    """Require nonempty printable ASCII without whitespace."""
    text = require_nonempty_string(value, owner)
    if not text.isascii() or any(
        character.isspace() or not character.isprintable() for character in text
    ):
        raise ValueError(f'{owner} must be nonempty printable ASCII without whitespace')
    return text


def _require_environment_name(value: object, owner: str) -> str:
    """Require one nonempty portable ASCII environment-variable name."""
    name = _require_safe_ascii(value, owner)
    if not (name[0].isalpha() or name[0] == '_') or not all(
        character.isalnum() or character == '_' for character in name
    ):
        raise ValueError(f'{owner} must be a portable environment name')
    return name


def _require_key_prefix(value: object) -> str:
    """Require one short literal ASCII Valkey namespace prefix."""
    prefix = _require_safe_ascii(value, 'Valkey key prefix')
    if len(prefix) > _MAX_KEY_PREFIX_LENGTH or any(
        character in '*?[]|\\/' for character in prefix
    ):
        raise ValueError('Valkey key prefix must be short literal ASCII')
    return prefix


def _require_string_tuple(value: object, owner: str) -> tuple[str, ...]:
    """Require an immutable tuple containing only strings."""
    if not isinstance(value, tuple):
        raise TypeError(f'{owner} must be a tuple')
    values = cast('tuple[object, ...]', value)
    if not all(isinstance(item, str) for item in values):
        raise TypeError(f'{owner} must contain only strings')
    return cast('tuple[str, ...]', values)


def _require_ttl_rule_tuple(value: object) -> tuple[TTLRule, ...]:
    """Require an immutable tuple containing only validated TTL rules."""
    if not isinstance(value, tuple):
        raise TypeError('TTL rules must be a tuple')
    values = cast('tuple[object, ...]', value)
    if not all(isinstance(item, TTLRule) for item in values):
        raise TypeError('TTL rules must contain only immutable TTLRule values')
    return cast('tuple[TTLRule, ...]', values)


def _require_valkey_host(value: object) -> str:
    """Require a credential-free literal Valkey host name or address."""
    host = _require_safe_ascii(value, 'Valkey host')
    try:
        ip_address(host)
    except ValueError:
        pass
    else:
        return host
    dns_name = host.removesuffix('.')
    labels = dns_name.split('.')
    if (
        not dns_name
        or len(dns_name) > _MAX_DNS_NAME_LENGTH
        or ':' in host
        or any(
            not label
            or len(label) > _MAX_DNS_LABEL_LENGTH
            or not label[0].isalnum()
            or not label[-1].isalnum()
            or any(not character.isalnum() and character != '-' for character in label)
            for label in labels
        )
        or (len(labels) > 1 and all(label.isdecimal() for label in labels))
    ):
        raise ValueError('Valkey host must be a DNS name or raw IP address')
    return host


def _require_valkey_port(value: object) -> None:
    """Require one TCP port number."""
    require_positive_int(value, 'Valkey port')
    if cast('int', value) > _MAX_TCP_PORT:
        raise ValueError('Valkey port must not exceed 65,535')


def _rule_matches(
    rule: TTLRule,
    *,
    method: str,
    entity_type: str | None,
    query_class: str,
    identity_kind: str,
) -> bool:
    """Return whether every configured rule field matches one lookup context."""
    return (
        (rule.method is None or rule.method == method)
        and (rule.entity_type is None or rule.entity_type == entity_type)
        and (rule.query_class is None or rule.query_class == query_class)
        and (rule.identity_kind is None or rule.identity_kind == identity_kind)
    )


def _rule_precedence(rule: TTLRule) -> int:
    """Return the documented strength of one validated rule shape."""
    return _RULE_PRECEDENCE[rule._match_key]  # pyright: ignore[reportPrivateUsage]  # noqa: SLF001


def _select_ttl_rule(
    rules: Sequence[TTLRule],
    *,
    method: str,
    entity_type: str | None,
    query_class: str,
    identity_kind: str,
) -> TTLRule | None:
    """Select one configured rule using the documented specificity order."""
    matches = [
        rule
        for rule in rules
        if _rule_matches(
            rule,
            method=method,
            entity_type=entity_type,
            query_class=query_class,
            identity_kind=identity_kind,
        )
    ]
    if not matches:
        return None
    return max(matches, key=_rule_precedence)


def _string(mapping: Mapping[str, object], key: str, default: str) -> str:
    """Read one TOML string."""
    value = mapping.get(key, default)
    if not isinstance(value, str):
        raise TypeError(f'{key} must be a string')
    return value


def _optional_string(mapping: Mapping[str, object], key: str) -> str | None:
    """Read an optional TOML string."""
    if key not in mapping:
        return None
    return _string(mapping, key, '')


def _load_ttl_rules(value: object) -> tuple[TTLRule, ...]:
    """Load repeated TTL-rule tables in declaration order."""
    if not isinstance(value, list | tuple):
        raise TypeError('cache.ttl_rule must be an array of tables')
    items = cast('list[object] | tuple[object, ...]', value)
    rules: list[TTLRule] = []
    for item in items:
        mapping = _mapping(item, 'cache.ttl_rule record')
        _reject_unknown_keys(mapping, _TTL_RULE_KEYS, 'cache.ttl_rule record')
        if 'maximum_age_seconds' not in mapping:
            raise ValueError('TTL rule requires maximum_age_seconds')
        rules.append(
            TTLRule(
                _number(mapping, 'maximum_age_seconds', 0),
                method=_optional_string(mapping, 'method'),
                entity_type=_optional_string(mapping, 'entity_type'),
                query_class=_optional_string(mapping, 'query_class'),
                identity_kind=_optional_string(mapping, 'identity_kind'),
            )
        )
    loaded = tuple(rules)
    _reject_duplicate_rules(loaded)
    return loaded


def _string_tuple(mapping: Mapping[str, object], key: str) -> tuple[str, ...]:
    """Read an ordered TOML string array."""
    value = mapping.get(key, ())
    if not isinstance(value, list | tuple):
        raise TypeError(f'{key} must be an array of strings')
    items = cast('list[object] | tuple[object, ...]', value)
    if not all(isinstance(item, str) for item in items):
        raise TypeError(f'{key} must be an array of strings')
    return tuple(cast('list[str] | tuple[str, ...]', items))


def _validate_ttl_only_methods(methods: tuple[str, ...]) -> None:
    """Accept only composite methods whose documented response shape carries no URL."""
    if not set(methods) <= URL_PROVEN_TTL_ONLY_METHODS:
        raise ValueError('configuration contains an unknown TTL-only method')
    if len(methods) != len(set(methods)):
        raise ValueError('TTL-only method names must be unique')


def _valkey_mode(value: object) -> _ValkeyMode:
    """Narrow one validated value to the configured deployment vocabulary."""
    if isinstance(value, str) and value in get_args(_ValkeyMode):
        return cast('_ValkeyMode', value)
    raise ValueError('Valkey mode must be standalone, sentinel, or managed')


def parse_sentinel_address(value: object) -> tuple[str, int]:
    """Parse one credential-free Sentinel `host:port` address.

    Args:
        value: ASCII DNS name or raw IP address followed by a decimal TCP port;
            hosts containing colons require brackets.

    Returns:
        The unbracketed host and integer port, without DNS normalization.

    Raises:
        TypeError: The address is not a string.
        ValueError: The address violates the configured host or port grammar.
    """
    text = _require_safe_ascii(value, 'Valkey sentinel address')
    host: str
    port_text: str
    if text.startswith('['):
        closing = text.find(']')
        if closing < 0 or closing + 1 >= len(text) or text[closing + 1] != ':':
            raise ValueError('Valkey sentinel address must use host:port')
        host = text[1:closing]
        port_text = text[closing + 2 :]
    else:
        host, separator, port_text = text.rpartition(':')
        if not separator or ':' in host:
            raise ValueError('Valkey sentinel address must use host:port')
    _require_valkey_host(host)
    if not port_text.isdecimal():
        raise ValueError('Valkey sentinel address must use host:port')
    port = int(port_text)
    _require_valkey_port(port)
    return host, port


def _load_valkey_role(mapping: Mapping[str, object], role: str) -> ValkeyRoleSettings:
    """Load one complete Valkey role table."""
    defaults = ValkeySettings().result
    mode = _valkey_mode(_string(mapping, 'mode', defaults.mode))
    connection_keys = {'sentinels', 'service_name'} if mode == 'sentinel' else {'host'}
    required = {'password_env'} | connection_keys
    if not required <= mapping.keys():
        raise ValueError(f'Valkey {role} role is missing required connection settings')
    sentinels = _string_tuple(mapping, 'sentinels')
    host = _string(mapping, 'host', parse_sentinel_address(sentinels[0])[0] if sentinels else '')
    return ValkeyRoleSettings(
        mode=mode,
        host=host,
        port=_integer(mapping, 'port', defaults.port),
        database=_integer(mapping, 'database', defaults.database),
        tls=_boolean(mapping, 'tls', default=defaults.tls),
        username=_optional_string(mapping, 'username'),
        password_env=_string(mapping, 'password_env', ''),
        key_prefix=_string(mapping, 'key_prefix', defaults.key_prefix),
        sentinels=sentinels,
        service_name=_optional_string(mapping, 'service_name'),
    )


def _load_valkey(mapping: Mapping[str, object]) -> ValkeySettings:
    """Load the required pair of Valkey role tables."""
    _reject_unknown_keys(mapping, _VALKEY_KEYS, 'cache.l2.valkey table')
    if set(mapping) != set(_VALKEY_KEYS):
        raise ValueError('cache.l2.valkey requires result and metadata role tables')
    result = _mapping(mapping['result'], 'cache.l2.valkey.result table')
    metadata = _mapping(mapping['metadata'], 'cache.l2.valkey.metadata table')
    _reject_unknown_keys(result, _VALKEY_ROLE_KEYS, 'cache.l2.valkey.result table')
    _reject_unknown_keys(metadata, _VALKEY_ROLE_KEYS, 'cache.l2.valkey.metadata table')
    return ValkeySettings(
        result=_load_valkey_role(result, 'result'),
        metadata=_load_valkey_role(metadata, 'metadata'),
    )


def _load_l2(mapping: Mapping[str, object]) -> L2Settings:
    """Load the L2 settings table."""
    defaults = L2Settings()
    enabled = _boolean(mapping, 'enabled', default=defaults.enabled)
    if enabled and 'valkey' not in mapping:
        raise ValueError('enabled L2 requires cache.l2.valkey role tables')
    valkey = (
        ValkeySettings()
        if 'valkey' not in mapping
        else _load_valkey(_mapping(mapping['valkey'], 'cache.l2.valkey table'))
    )
    return L2Settings(
        enabled=enabled,
        backend=_string(mapping, 'backend', defaults.backend),
        valkey=valkey,
        event_aware_max_age_seconds=_number(
            mapping, 'event_aware_max_age_seconds', defaults.event_aware_max_age_seconds
        ),
        ttl_only_max_age_seconds=_number(
            mapping, 'ttl_only_max_age_seconds', defaults.ttl_only_max_age_seconds
        ),
        schema_max_age_seconds=_number(
            mapping, 'schema_max_age_seconds', defaults.schema_max_age_seconds
        ),
        result_budget_bytes=_integer(mapping, 'result_budget_bytes', defaults.result_budget_bytes),
        metadata_budget_bytes=_integer(
            mapping, 'metadata_budget_bytes', defaults.metadata_budget_bytes
        ),
        acknowledge_mutually_trusted_clients=_boolean(
            mapping,
            'acknowledge_mutually_trusted_clients',
            default=defaults.acknowledge_mutually_trusted_clients,
        ),
    )


def select_maximum_age(
    settings: CacheSettings,
    context: TTLContext,
    per_call_override: float | None = None,
) -> float:
    """Select one ordinary age and apply every post-selection safety cap.

    Args:
        settings: Immutable loaded cache settings.
        context: Explicit call, identity, query-class, layer, and cap facts.
        per_call_override: Optional positive age that wins over configured
            rules.

    Returns:
        Zero for an explicit scoped bypass or an exhausted safety cap;
        otherwise, a positive age no larger than any applicable safety cap.

    Raises:
        TypeError: A supplied age is not a finite number.
        ValueError: The layer is disabled, the query class is not cacheable, or
            the per-call override is not positive.
    """
    if context.scoped_bypass:
        return 0.0
    default_age = _layer_default_age(settings, context)
    configured_rule = _select_ttl_rule(
        settings.ttl_rules,
        method=context.method,
        entity_type=context.entity_type,
        query_class=context.query_class,
        identity_kind=context.identity_kind,
    )
    configured_age = (
        default_age if configured_rule is None else configured_rule.maximum_age_seconds
    )
    if per_call_override is not None:
        require_positive_number(per_call_override, 'per-call maximum-age override')
        configured_age = per_call_override

    ages = [configured_age, settings.global_safety_max_seconds]
    if context.response_deadline_seconds is not None:
        ages.append(context.response_deadline_seconds)
    if context.remaining_l2_seconds is not None:
        ages.append(context.remaining_l2_seconds)
    if context.layer == 'hierarchical_l1':
        ages.append(settings.l1.hierarchical_max_age_seconds)
    return min(ages)
