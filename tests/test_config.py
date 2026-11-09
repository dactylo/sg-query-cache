from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import pytest

import sg_query_cache._config as config
from sg_query_cache._codec import MAX_BYTES
from sg_query_cache._config import (
    CacheLayer,
    CacheSettings,
    L1Settings,
    L2Settings,
    LimitSettings,
    TTLContext,
    ValkeyRoleSettings,
    ValkeySettings,
    _TTLRule,
    parse_sentinel_address,
    select_maximum_age,
)
from sg_query_cache._model import DependencyPlan, ExactCall, Identity, IdentityKind


def _context(  # noqa: PLR0913 - keyword builder mirrors every TTLContext field
    *,
    method: str = 'find',
    entity_type: str | None = 'Asset',
    query_class: str = 'event_aware',
    identity_kind: IdentityKind = 'human',
    layer: CacheLayer = 'l2',
    scoped_bypass: bool = False,
    response_deadline_seconds: float | None = None,
    remaining_l2_seconds: float | None = None,
) -> TTLContext:
    return TTLContext(
        method=method,
        entity_type=entity_type,
        query_class=query_class,
        identity_kind=identity_kind,
        layer=layer,
        scoped_bypass=scoped_bypass,
        response_deadline_seconds=response_deadline_seconds,
        remaining_l2_seconds=remaining_l2_seconds,
    )


def _precedence_settings() -> CacheSettings:
    return CacheSettings(
        l2=L2Settings(enabled=True),
        ttl_rules=(
            _TTLRule(700, identity_kind='human'),
            _TTLRule(600, query_class='event_aware'),
            _TTLRule(500, entity_type='Asset'),
            _TTLRule(400, method='find'),
            _TTLRule(300, method='find', entity_type='Asset'),
        ),
    )


def test_cache_settings_define_every_numeric_default() -> None:
    settings = CacheSettings()

    assert settings.schema_version == 1
    assert settings.global_safety_max_seconds == 3600
    assert settings.l1.hierarchical_max_age_seconds == 10
    assert settings.l1.local_only_max_age_seconds == 30
    assert settings.l1.result_budget_bytes == 134_217_728
    assert settings.l1.metadata_budget_bytes == 16_777_216
    assert settings.l2.event_aware_max_age_seconds == 600
    assert settings.l2.ttl_only_max_age_seconds == 60
    assert settings.l2.schema_max_age_seconds == 1800
    assert settings.l2.result_budget_bytes == 8_589_934_592
    assert settings.l2.metadata_budget_bytes == 1_073_741_824
    assert settings.limits.fill_lease_seconds == 30
    assert settings.limits.lease_wait_seconds == 2
    assert settings.limits.max_dependency_tags == 32
    assert settings.limits.max_dependency_vector_bytes == 2048
    assert settings.limits.max_serialized_result_bytes == 1_048_576
    assert settings.limits.scoped_generation_retention_seconds == 7200
    assert settings.limits.connect_timeout_seconds == 0.1
    assert settings.limits.operation_timeout_seconds == 0.25
    assert settings.limits.circuit_failure_threshold == 5
    assert settings.limits.circuit_probe_seconds == 10
    assert settings.limits.incarnation_validation_seconds == 30
    assert settings.limits.signed_url_safety_margin_seconds == 30
    assert settings.limits.transient_image_max_age_seconds == 5
    assert settings.limits.clock_error_bound_seconds is None


def test_configuration_accepts_explicit_policy_options(tmp_path: Path) -> None:
    path = tmp_path / 'cache.toml'
    path.write_text(
        'schema_version = 1\n'
        '[cache]\n'
        "identity_policy = 'scripts_and_humans'\n"
        "enabled_ttl_only_methods = ['work_schedule_read']\n"
        'enable_additional_filter_presets = true\n'
        "metrics_digest_secret_env = 'METRICS_KEY'\n"
        '[[cache.ttl_rule]]\n'
        "method = 'find'\n"
        "entity_type = 'Asset'\n"
        'maximum_age_seconds = 300\n'
        '[cache.l2]\n'
        'enabled = true\n'
        'acknowledge_mutually_trusted_clients = true\n'
        '[cache.l2.valkey.result]\n'
        "host = 'cache-results'\n"
        "password_env = 'RESULT_PASSWORD'\n"
        '[cache.l2.valkey.metadata]\n'
        "host = 'cache-metadata'\n"
        "password_env = 'METADATA_PASSWORD'\n"
        '[cache.limits]\n'
        'clock_error_bound_seconds = 0.5\n',
        encoding='utf-8',
    )

    settings = CacheSettings.from_toml(path)

    assert settings.identity_policy == 'scripts_and_humans'
    assert settings.enabled_ttl_only_methods == ('work_schedule_read',)
    assert settings.enable_additional_filter_presets
    assert settings.metrics_digest_secret_env == 'METRICS' + '_KEY'
    assert settings.ttl_rules == (_TTLRule(300, method='find', entity_type='Asset'),)
    assert settings.limits.clock_error_bound_seconds == 0.5


@pytest.mark.parametrize(
    'entity_match', ['', "entity_type = 'Asset'\n"], ids=['method', 'method-and-entity']
)
def test_configuration_rejects_duplicate_ttl_rule_matches(
    tmp_path: Path, entity_match: str
) -> None:
    path = tmp_path / 'cache.toml'
    path.write_text(
        'schema_version = 1\n'
        '[[cache.ttl_rule]]\n'
        "method = 'find'\n"
        f'{entity_match}'
        'maximum_age_seconds = 30\n'
        '[[cache.ttl_rule]]\n'
        "method = 'find'\n"
        f'{entity_match}'
        'maximum_age_seconds = 60\n',
        encoding='utf-8',
    )

    with pytest.raises(ValueError, match='duplicate TTL rule'):
        CacheSettings.from_toml(path)


def test_configuration_rejects_non_string_method_array_item_with_type_error(
    tmp_path: Path,
) -> None:
    path = tmp_path / 'cache.toml'
    path.write_text(
        'schema_version = 1\n[cache]\nenabled_ttl_only_methods = [1]\n',
        encoding='utf-8',
    )

    with pytest.raises(TypeError, match='array of strings'):
        CacheSettings.from_toml(path)


def test_configured_rule_cannot_lengthen_hierarchical_l1() -> None:
    settings = CacheSettings(
        l2=L2Settings(enabled=True),
        ttl_rules=(_TTLRule(600, method='find'),),
    )

    assert (
        select_maximum_age(
            settings,
            _context(layer='hierarchical_l1', identity_kind='script'),
        )
        == 10
    )
    assert (
        select_maximum_age(
            settings,
            _context(layer='l2', identity_kind='script'),
        )
        == 600
    )


def test_context_derives_explicit_lookup_facts_from_task_interfaces() -> None:
    call = ExactCall(
        'find',
        (
            ('entity_type', 'Asset'),
            ('filters', ()),
        ),
    )
    identity = Identity('human', 'studio-sso', 'human:42', entity_id=42)
    plan = DependencyPlan('event_aware', (), 'supported')

    context = TTLContext.from_call(call, identity, plan, layer='l2')

    assert context.method == 'find'
    assert context.entity_type == 'Asset'
    assert context.query_class == 'event_aware'
    assert context.identity_kind == 'human'
    assert context.layer == 'l2'


def test_dependency_tag_limit_allows_local_33_and_shared_32() -> None:
    local = CacheSettings(limits=LimitSettings(max_dependency_tags=33))
    shared = CacheSettings(
        l1=L1Settings(enabled=False),
        l2=L2Settings(enabled=True),
        limits=LimitSettings(max_dependency_tags=32),
    )

    assert local.limits.max_dependency_tags == 33
    assert shared.limits.max_dependency_tags == 32


def test_disabled_layer_cannot_select_an_age() -> None:
    settings = CacheSettings(l1=L1Settings(enabled=False), l2=L2Settings(enabled=True))

    with pytest.raises(ValueError, match='disabled'):
        select_maximum_age(settings, _context(layer='local_l1'))


def test_enabled_l2_loads_two_standalone_valkey_roles(tmp_path: Path) -> None:
    path = tmp_path / 'cache.toml'
    path.write_text(
        'schema_version = 1\n'
        '[cache.l1]\n'
        'enabled = false\n'
        '[cache.l2]\n'
        'enabled = true\n'
        '[cache.l2.valkey.result]\n'
        "mode = 'standalone'\n"
        "host = 'cache-results'\n"
        'port = 6379\n'
        'database = 0\n'
        'tls = true\n'
        "username = 'sg-query-cache'\n"
        "password_env = 'SG_QUERY_CACHE_RESULT_PASSWORD'\n"
        "key_prefix = 'sgqc'\n"
        '[cache.l2.valkey.metadata]\n'
        "mode = 'standalone'\n"
        "host = 'cache-metadata'\n"
        'port = 6380\n'
        'database = 1\n'
        'tls = false\n'
        "username = 'metadata-user'\n"
        "password_env = 'SG_QUERY_CACHE_METADATA_PASSWORD'\n"
        "key_prefix = 'metadata-prefix'\n",
        encoding='utf-8',
    )

    settings = config.CacheSettings.from_toml(path)

    assert settings.l2.valkey.result == config.ValkeyRoleSettings(
        'cache-results',
        'SG_QUERY_CACHE_RESULT_PASSWORD',
        mode='standalone',
        port=6379,
        database=0,
        tls=True,
        username='sg-query-cache',
        key_prefix='sgqc',
    )
    assert settings.l2.valkey.metadata == config.ValkeyRoleSettings(
        'cache-metadata',
        'SG_QUERY_CACHE_METADATA_PASSWORD',
        mode='standalone',
        port=6380,
        database=1,
        tls=False,
        username='metadata-user',
        key_prefix='metadata-prefix',
    )


def test_enabled_l2_requires_both_valkey_role_tables(tmp_path: Path) -> None:
    path = tmp_path / 'cache.toml'
    path.write_text(
        'schema_version = 1\n[cache.l2]\nenabled = true\n',
        encoding='utf-8',
    )

    with pytest.raises(ValueError, match=r'requires cache\.l2\.valkey'):
        config.CacheSettings.from_toml(path)


def test_equivalent_dns_spellings_are_one_endpoint() -> None:
    with pytest.raises(ValueError, match='separate endpoints'):
        ValkeySettings(
            result=ValkeyRoleSettings('cache.example', 'RESULT_PASSWORD'),
            metadata=ValkeyRoleSettings('CACHE.EXAMPLE.', 'METADATA_PASSWORD'),
        )


def test_identity_policy_must_be_a_string(tmp_path: Path) -> None:
    path = tmp_path / 'cache.toml'
    path.write_text(
        'schema_version = 1\n[cache]\nidentity_policy = 987654321\n',
        encoding='utf-8',
    )

    with pytest.raises(TypeError) as error:
        CacheSettings.from_toml(path)

    assert 'identity_policy' in str(error.value)
    assert '987654321' not in str(error.value)


def test_invalid_valkey_mode_precedes_mode_specific_requirements(tmp_path: Path) -> None:
    path = tmp_path / 'cache.toml'
    path.write_text(
        'schema_version = 1\n'
        '[cache.l2.valkey.result]\n'
        "mode = 'unknown'\n"
        "password_env = 'RESULT_PASSWORD'\n"
        '[cache.l2.valkey.metadata]\n'
        "host = 'cache-metadata'\n"
        "password_env = 'METADATA_PASSWORD'\n",
        encoding='utf-8',
    )

    with pytest.raises(ValueError, match=r'cache\.l2\.valkey\.result\.mode must be'):
        CacheSettings.from_toml(path)


def test_minimal_toml_uses_defaults(tmp_path: Path) -> None:
    path = tmp_path / 'cache.toml'
    path.write_text(
        "schema_version = 1\n[cache]\nsite_namespace = 'production'\n", encoding='utf-8'
    )

    assert CacheSettings.from_toml(path) == CacheSettings()


def test_minimal_valkey_tables_use_dataclass_defaults(tmp_path: Path) -> None:
    path = tmp_path / 'cache.toml'
    path.write_text(
        'schema_version = 1\n'
        '[cache.l2]\n'
        'enabled = true\n'
        '[cache.l2.valkey.result]\n'
        "host = 'cache-results'\n"
        "password_env = 'SG_QUERY_CACHE_RESULT_PASSWORD'\n"
        '[cache.l2.valkey.metadata]\n'
        "host = 'cache-metadata'\n"
        "password_env = 'SG_QUERY_CACHE_METADATA_PASSWORD'\n",
        encoding='utf-8',
    )

    assert CacheSettings.from_toml(path) == CacheSettings(l2=L2Settings(enabled=True))


def test_role_configuration_supports_sentinel_and_managed_but_not_mixed_modes() -> None:
    sentinel = config.ValkeyRoleSettings(
        'unused.example',
        'PASSWORD',
        mode='sentinel',
        sentinels=('one.example:26379', 'two.example:26379', 'three.example:26379'),
        service_name='cache-primary',
    )
    managed = config.ValkeyRoleSettings('primary.example', 'PASSWORD', mode='managed')

    assert sentinel.service_name == 'cache-primary'
    assert managed.host == 'primary.example'
    with pytest.raises(ValueError, match='same deployment mode'):
        config.ValkeySettings(result=sentinel, metadata=managed)


def test_selection_does_not_mutate_loaded_settings() -> None:
    settings = _precedence_settings()
    expected = _precedence_settings()

    select_maximum_age(settings, _context(), per_call_override=200)

    assert settings == expected
    assert settings.ttl_rules == expected.ttl_rules


def test_sentinel_configuration_requires_three_distinct_addresses_and_service() -> None:
    base = config.ValkeyRoleSettings(
        'unused.example',
        'PASSWORD',
        mode='sentinel',
        sentinels=('one.example:26379', 'two.example:26379', 'three.example:26379'),
        service_name='cache-primary',
    )

    with pytest.raises(ValueError, match='three distinct'):
        replace(base, sentinels=('one.example:26379',) * 3)
    with pytest.raises(ValueError, match='three distinct'):
        replace(
            base,
            sentinels=(
                '[::1]:26379',
                '[0:0:0:0:0:0:0:1]:26379',
                '[0000:0000:0000:0000:0000:0000:0000:0001]:26379',
            ),
        )
    with pytest.raises(ValueError, match='service_name'):
        replace(base, service_name=None)


def test_sentinel_role_tables_load_without_direct_hosts(tmp_path: Path) -> None:
    path = tmp_path / 'cache.toml'
    path.write_text(
        """schema_version = 1
[cache.l1]
enabled = false
[cache.l2]
enabled = true
[cache.l2.valkey.result]
mode = "sentinel"
sentinels = ["one:26379", "two:26379", "three:26379"]
service_name = "result-primary"
password_env = "RESULT_PASSWORD"
[cache.l2.valkey.metadata]
mode = "sentinel"
sentinels = ["one:26379", "two:26379", "three:26379"]
service_name = "metadata-primary"
password_env = "METADATA_PASSWORD"
""",
        encoding='utf-8',
    )

    settings = CacheSettings.from_toml(path)

    assert settings.l2.valkey.result.host == 'one'
    assert settings.l2.valkey.metadata.service_name == 'metadata-primary'


def test_serialized_result_limit_cannot_exceed_the_codec_ceiling() -> None:
    assert (
        LimitSettings(max_serialized_result_bytes=MAX_BYTES).max_serialized_result_bytes
        == MAX_BYTES
    )
    with pytest.raises(ValueError, match='codec ceiling'):
        LimitSettings(max_serialized_result_bytes=MAX_BYTES + 1)


def test_shared_human_cache_requires_trust_acknowledgement(tmp_path: Path) -> None:
    path = tmp_path / 'cache.toml'
    path.write_text(
        'schema_version = 1\n'
        '[cache]\n'
        "identity_policy = 'scripts_and_humans'\n"
        '[cache.l2]\n'
        'enabled = true\n'
        '[cache.l2.valkey.result]\n'
        "host = 'cache-results'\n"
        "password_env = 'RESULT_PASSWORD'\n"
        '[cache.l2.valkey.metadata]\n'
        "host = 'cache-metadata'\n"
        "password_env = 'METADATA_PASSWORD'\n",
        encoding='utf-8',
    )

    with pytest.raises(ValueError, match='acknowledge_mutually_trusted_clients'):
        CacheSettings.from_toml(path)


def test_shared_human_caching_requires_mutual_trust_acknowledgement() -> None:
    with pytest.raises(ValueError, match='acknowledge_mutually_trusted_clients'):
        CacheSettings(
            identity_policy='scripts_and_humans',
            l1=L1Settings(enabled=False),
            l2=L2Settings(enabled=True),
        )


@pytest.mark.parametrize(
    ('address', 'expected'),
    [
        ('cache.example:26379', ('cache.example', 26379)),
        ('CACHE.EXAMPLE.:026379', ('CACHE.EXAMPLE.', 26379)),
        ('127.0.0.1:1', ('127.0.0.1', 1)),
        ('[::1]:65535', ('::1', 65535)),
        ('[cache.example]:26379', ('cache.example', 26379)),
    ],
)
def test_shared_sentinel_parser_preserves_accepted_address_grammar(
    address: str,
    expected: tuple[str, int],
) -> None:
    assert parse_sentinel_address(address) == expected


@pytest.mark.parametrize(
    'address',
    [
        '::1:26379',
        'cache:0',
        'cache:65536',
        'cache:+1',
        'cache:1 ',
        'cache:\u0661',
        'user@cache:26379',
        '[::1]26379',
        'cache_name:26379',
    ],
)
def test_shared_sentinel_parser_rejects_invalid_addresses(address: str) -> None:
    with pytest.raises(ValueError, match='Valkey'):
        parse_sentinel_address(address)


@pytest.mark.parametrize(
    ('contents', 'match'),
    [
        ('schema_version = 2\n[cache]\n', 'schema_version'),
        ('schema_version = 1\nunknown = 1\n[cache]\n', 'unknown key'),
        ('schema_version = 1\n[cache]\nunknown = 1\n', 'unknown key'),
        (
            "schema_version = 1\n[cache]\nidentity_policy = 'everyone'\n",
            r'cache\.identity_policy',
        ),
        ("schema_version = 1\n[cache]\nsite_namespace = ''\n", r'cache\.site_namespace'),
        (
            "schema_version = 1\n[cache]\nsite_namespace = 'prod-\N{SNOWMAN}'\n",
            r'cache\.site_namespace',
        ),
        (
            "schema_version = 1\n[cache]\nenabled_ttl_only_methods = ['unknown']\n",
            'TTL-only method',
        ),
        (
            "schema_version = 1\n[cache]\nenabled_ttl_only_methods = ['EventLogEntry']\n",
            'TTL-only method',
        ),
        (
            'schema_version = 1\n[cache.l1]\nenabled = false\n[cache.l2]\nenabled = false\n',
            r'cache\.l1\.enabled or cache\.l2\.enabled',
        ),
        (
            (
                'schema_version = 1\n[cache.limits]\n'
                'clock_error_bound_seconds = 30\n'
                'signed_url_safety_margin_seconds = 30\n'
            ),
            r'cache\.limits\.clock_error_bound_seconds',
        ),
    ],
)
def test_configuration_rejects_unsafe_values(tmp_path: Path, contents: str, match: str) -> None:
    path = tmp_path / 'cache.toml'
    path.write_text(contents, encoding='utf-8')

    with pytest.raises(ValueError, match=match) as captured:
        CacheSettings.from_toml(path)

    assert type(captured.value) is ValueError
    assert all(
        value not in str(captured.value)
        for value in ('everyone', 'prod-\N{SNOWMAN}', 'EventLogEntry')
    )
    if match == 'unknown key':
        assert 'allowed keys:' in str(captured.value)


@pytest.mark.parametrize(
    ('result_host', 'metadata_host'),
    [
        ('::1', '0:0:0:0:0:0:0:1'),
        ('2001:DB8::1', '2001:db8:0:0:0:0:0:1'),
    ],
)
def test_equivalent_ip_spellings_are_one_endpoint(
    result_host: str,
    metadata_host: str,
) -> None:
    with pytest.raises(ValueError, match='separate endpoints'):
        ValkeySettings(
            result=ValkeyRoleSettings(result_host, 'RESULT_PASSWORD'),
            metadata=ValkeyRoleSettings(metadata_host, 'METADATA_PASSWORD'),
        )


def test_shared_valkey_rejects_dependency_tag_limit_above_v1_ceiling() -> None:
    with pytest.raises(
        ValueError,
        match=r'cache\.limits\.max_dependency_tags cannot exceed the v1 ceiling',
    ) as captured:
        CacheSettings(
            l1=L1Settings(enabled=False),
            l2=L2Settings(enabled=True),
            limits=LimitSettings(max_dependency_tags=33),
        )

    assert '33' not in str(captured.value)


@pytest.mark.parametrize(
    'method',
    ['activity_stream_read', 'note_thread_read', 'preferences_read', 'summarize', 'text_search'],
)
def test_unproven_composite_methods_cannot_be_enabled(method: str) -> None:
    with pytest.raises(ValueError, match='unknown TTL-only method'):
        CacheSettings(enabled_ttl_only_methods=(method,))


@pytest.mark.parametrize(
    ('rule', 'match'),
    [
        (lambda: _TTLRule(30), 'match field'),
        (lambda: _TTLRule(0, method='find'), 'positive'),
        (lambda: _TTLRule(3601, method='find'), '3,600'),
        (lambda: _TTLRule(30, method='', entity_type='Asset'), 'nonempty'),
        (
            lambda: _TTLRule(30, method='find', query_class='ttl_only'),
            'match combination',
        ),
    ],
)
def test_ttl_rules_reject_unsafe_or_ambiguous_matches(
    rule: Callable[[], object], match: str
) -> None:
    with pytest.raises((TypeError, ValueError), match=match) as captured:
        rule()

    assert str(captured.value).startswith('cache.ttl_rule')
    assert 'Asset' not in str(captured.value)
    assert 'find' not in str(captured.value)


@pytest.mark.parametrize('query_class', ['event-aware', 'EVENT_AWARE', 'bypass', ''])
def test_ttl_rule_rejects_unknown_query_classes(query_class: str) -> None:
    with pytest.raises(ValueError, match=r'query_class|nonempty'):
        _TTLRule(60, query_class=query_class)


@pytest.mark.parametrize('query_class', ['event_aware', 'ttl_only', 'schema'])
def test_ttl_rule_accepts_the_closed_query_class_vocabulary(query_class: str) -> None:
    assert _TTLRule(60, query_class=query_class).query_class == query_class


@pytest.mark.parametrize(
    'build',
    [
        lambda: CacheSettings(
            enabled_ttl_only_methods=['summarize']  # type: ignore[arg-type]
        ),
        lambda: CacheSettings(
            ttl_rules=[_TTLRule(30, method='find')]  # type: ignore[arg-type]
        ),
    ],
)
def test_settings_values_reject_mutable_nested_containers(build: Callable[[], object]) -> None:
    with pytest.raises(TypeError, match=r'tuple|immutable'):
        build()


@pytest.mark.parametrize(
    'host',
    [
        'localhost',
        'cache-results',
        'cache-results.internal.example',
        'cache-results.internal.example.',
        '127.0.0.1',
        '2001:db8::1',
        '::1',
    ],
)
def test_valkey_roles_accept_dns_names_and_raw_ip_literals(host: str) -> None:
    role = config.ValkeyRoleSettings(host, 'PASSWORD')

    assert role.host == host


def test_valkey_roles_must_use_separate_endpoints() -> None:
    role = config.ValkeyRoleSettings(
        'cache',
        'PASSWORD',
        port=6379,
        database=0,
    )

    with pytest.raises(ValueError, match='separate endpoints'):
        config.ValkeySettings(result=role, metadata=role)


@pytest.mark.parametrize(
    ('build', 'match'),
    [
        (lambda: config.ValkeyRoleSettings('https://cache-results', 'PASSWORD'), 'host'),
        (lambda: config.ValkeyRoleSettings('cache-results:6379', 'PASSWORD'), 'host'),
        (lambda: config.ValkeyRoleSettings('[::1]', 'PASSWORD'), 'host'),
        (lambda: config.ValkeyRoleSettings('cache_results', 'PASSWORD'), 'host'),
        (lambda: config.ValkeyRoleSettings('-cache.example', 'PASSWORD'), 'host'),
        (lambda: config.ValkeyRoleSettings('cache-.example', 'PASSWORD'), 'host'),
        (lambda: config.ValkeyRoleSettings('cache..example', 'PASSWORD'), 'host'),
        (lambda: config.ValkeyRoleSettings('999.999.999.999', 'PASSWORD'), 'host'),
        (lambda: config.ValkeyRoleSettings('?', 'PASSWORD'), 'host'),
        (lambda: config.ValkeyRoleSettings('cache-results', 'PASSWORD', port=0), 'port'),
        (lambda: config.ValkeyRoleSettings('cache-results', 'PASSWORD', port=65_536), 'port'),
        (lambda: config.ValkeyRoleSettings('cache-results', 'PASSWORD', database=-1), 'database'),
        (
            lambda: config.ValkeyRoleSettings(
                'cache-results',
                'PASSWORD',
                mode='sentinel',
            ),
            'mode',
        ),
    ],
)
def test_valkey_roles_reject_invalid_standalone_coordinates(
    build: Callable[[], object], match: str
) -> None:
    with pytest.raises((TypeError, ValueError), match=match):
        build()


@pytest.mark.parametrize(
    ('context', 'per_call_override', 'expected'),
    [
        (_context(scoped_bypass=True), 200, 0),
        (_context(), 200, 200),
        (_context(), 900, 900),
        (_context(), None, 300),
        (_context(entity_type='Shot'), None, 400),
        (_context(method='find_one'), None, 500),
        (_context(method='find_one', entity_type='Shot'), None, 600),
        (
            _context(method='find_one', entity_type='Shot', query_class='ttl_only'),
            None,
            700,
        ),
        (
            _context(
                method='find_one',
                entity_type='Shot',
                query_class='ttl_only',
                identity_kind='script',
            ),
            None,
            60,
        ),
    ],
)
def test_maximum_age_uses_all_eight_precedence_levels_in_order(
    context: TTLContext,
    per_call_override: float | None,
    expected: float,
) -> None:
    assert select_maximum_age(
        _precedence_settings(),
        context,
        per_call_override=per_call_override,
    ) == pytest.approx(expected)


@pytest.mark.parametrize(
    ('context', 'expected'),
    [
        (_context(layer='hierarchical_l1'), 10),
        (_context(layer='local_l1'), 30),
        (_context(layer='l2'), 600),
        (_context(layer='l2', query_class='ttl_only'), 60),
        (_context(layer='l2', query_class='schema'), 1800),
    ],
)
def test_layer_and_query_class_choose_the_documented_default(
    context: TTLContext,
    expected: float,
) -> None:
    settings = CacheSettings(l2=L2Settings(enabled=True))

    assert select_maximum_age(settings, context) == pytest.approx(expected)


@pytest.mark.parametrize(
    ('settings', 'context', 'expected'),
    [
        (
            CacheSettings(
                global_safety_max_seconds=700,
                l2=L2Settings(enabled=True),
                ttl_rules=(_TTLRule(900, identity_kind='human'),),
            ),
            _context(),
            700,
        ),
        (
            CacheSettings(
                l2=L2Settings(enabled=True),
                ttl_rules=(_TTLRule(900, identity_kind='human'),),
            ),
            _context(response_deadline_seconds=250),
            250,
        ),
        (
            CacheSettings(
                l2=L2Settings(enabled=True),
                ttl_rules=(
                    _TTLRule(900, identity_kind='human'),
                    _TTLRule(300, method='find', entity_type='Asset'),
                ),
            ),
            _context(remaining_l2_seconds=120),
            120,
        ),
        (
            CacheSettings(
                l2=L2Settings(enabled=True),
                ttl_rules=(_TTLRule(100, method='find', entity_type='Asset'),),
            ),
            _context(response_deadline_seconds=250, remaining_l2_seconds=200),
            100,
        ),
    ],
)
def test_post_selection_caps_only_shorten_the_winning_age(
    settings: CacheSettings,
    context: TTLContext,
    expected: float,
) -> None:
    assert select_maximum_age(settings, context) == pytest.approx(expected)


@pytest.mark.parametrize('per_call_override', [0, -1, True, float('nan'), float('inf')])
def test_per_call_override_must_be_finite_and_positive(
    per_call_override: float,
) -> None:
    with pytest.raises((TypeError, ValueError), match='per-call'):
        select_maximum_age(
            CacheSettings(),
            _context(layer='local_l1'),
            per_call_override=per_call_override,
        )


@pytest.mark.parametrize(
    ('contents', 'error_type', 'message'),
    [
        ('schema_version = "sensitive-marker"', TypeError, 'schema_version must be an integer'),
        ('schema_version = 917', ValueError, 'schema_version is unsupported'),
        pytest.param(
            '[cache.l1]\nhierarchical_max_age_seconds = ' + '9' * 400,
            OverflowError,
            'cache.l1.hierarchical_max_age_seconds must fit a finite number',
            id='numeric-conversion-overflow',
        ),
        (
            '[cache]\nidentity_policy = "sensitive-marker"',
            ValueError,
            'cache.identity_policy must be scripts_only or scripts_and_humans',
        ),
        (
            '[cache]\nsite_namespace = " sensitive-marker "',
            ValueError,
            'cache.site_namespace must be nonempty ASCII without outer whitespace',
        ),
        (
            '[cache]\nglobal_safety_max_seconds = -917',
            ValueError,
            'cache.global_safety_max_seconds must be positive',
        ),
        (
            '[cache]\nenabled_ttl_only_methods = ["sensitive-marker"]',
            ValueError,
            'cache.enabled_ttl_only_methods contains an unknown TTL-only method',
        ),
        (
            '[cache.l1]\nenabled = "sensitive-marker"',
            TypeError,
            'cache.l1.enabled must be a boolean',
        ),
        (
            '[cache.l1]\nresult_budget_bytes = -917',
            ValueError,
            'cache.l1.result_budget_bytes must be positive',
        ),
        (
            '[cache.l2]\nbackend = "sensitive-marker"',
            ValueError,
            'cache.l2.backend must be valkey',
        ),
        (
            '[cache.l2]\nschema_max_age_seconds = -917',
            ValueError,
            'cache.l2.schema_max_age_seconds must be positive',
        ),
        (
            '[cache.limits]\nmax_dependency_tags = "sensitive-marker"',
            TypeError,
            'cache.limits.max_dependency_tags must be an integer',
        ),
        (
            '[cache.limits]\noperation_timeout_seconds = -917',
            ValueError,
            'cache.limits.operation_timeout_seconds must be positive',
        ),
        (
            '[cache.limits]\nclock_error_bound_seconds = 917',
            ValueError,
            (
                'cache.limits.clock_error_bound_seconds must be smaller than '
                'cache.limits.signed_url_safety_margin_seconds'
            ),
        ),
        (
            '[cache.l1]\nenabled = false',
            ValueError,
            'cache.l1.enabled or cache.l2.enabled must be true',
        ),
        (
            '[[cache.ttl_rule]]\nmethod = "sensitive-marker"',
            ValueError,
            'cache.ttl_rule requires maximum_age_seconds',
        ),
        (
            '[[cache.ttl_rule]]\nmaximum_age_seconds = "sensitive-marker"',
            TypeError,
            'cache.ttl_rule.maximum_age_seconds must be a number',
        ),
        (
            '[[cache.ttl_rule]]\nmaximum_age_seconds = -917\nmethod = "sensitive-marker"',
            ValueError,
            'cache.ttl_rule.maximum_age_seconds must be positive',
        ),
        (
            '[[cache.ttl_rule]]\nmaximum_age_seconds = 1\nquery_class = "sensitive-marker"',
            ValueError,
            'cache.ttl_rule.query_class must be event_aware, ttl_only, or schema',
        ),
        (
            '[[cache.ttl_rule]]\nmaximum_age_seconds = 1\nmethod = ""',
            ValueError,
            'cache.ttl_rule.method must be nonempty',
        ),
    ],
)
def test_settings_errors_name_the_toml_path(
    tmp_path: Path, contents: str, error_type: type[Exception], message: str
) -> None:
    path = tmp_path / 'cache.toml'
    prefix = '' if contents.startswith('schema_version') else 'schema_version = 1\n'
    path.write_text(prefix + contents, encoding='utf-8')

    with pytest.raises(error_type) as captured:
        CacheSettings.from_toml(path)

    assert type(captured.value) is error_type
    assert str(captured.value) == message
    assert 'sensitive-marker' not in str(captured.value)
    assert '917' not in str(captured.value)


@pytest.mark.parametrize(
    ('table', 'allowed'),
    [
        ('', 'cache, schema_version'),
        (
            'cache.l1',
            (
                'enabled, hierarchical_max_age_seconds, '
                'local_only_max_age_seconds, metadata_budget_bytes, result_budget_bytes'
            ),
        ),
        (
            'cache.l2',
            (
                'acknowledge_mutually_trusted_clients, backend, enabled, '
                'event_aware_max_age_seconds, metadata_budget_bytes, '
                'result_budget_bytes, schema_max_age_seconds, ttl_only_max_age_seconds, valkey'
            ),
        ),
        (
            'cache.l2.valkey.result',
            (
                'database, host, key_prefix, mode, password_env, port, '
                'sentinels, service_name, tls, username'
            ),
        ),
    ],
)
def test_unknown_settings_keys_list_the_table_and_allowed_keys(
    tmp_path: Path, table: str, allowed: str
) -> None:
    contents = 'schema_version = 1\n'
    if table:
        contents += f'[{table}]\n'
    contents += 'sensitive-marker = "sensitive-value"\n'
    if table == 'cache.l2.valkey.result':
        contents += '[cache.l2.valkey.metadata]\n'
    path = tmp_path / 'cache.toml'
    path.write_text(contents, encoding='utf-8')

    with pytest.raises(ValueError, match='cache') as captured:
        CacheSettings.from_toml(path)

    owner = f'{table} table' if table else 'root table'
    assert str(captured.value) == f'{owner} contains an unknown key; allowed keys: {allowed}'
    assert 'sensitive' not in str(captured.value)


@pytest.mark.parametrize('role', ['result', 'metadata'])
@pytest.mark.parametrize(
    ('fields', 'error_type', 'message'),
    [
        (
            'host = "sensitive-marker@host"',
            ValueError,
            'host must be a DNS name or raw IP address',
        ),
        ('port = "sensitive-marker"', TypeError, 'port must be an integer'),
        ('port = 65917', ValueError, 'port must not exceed 65,535'),
        ('tls = "sensitive-marker"', TypeError, 'tls must be a boolean'),
        ('mode = "sensitive-marker"', ValueError, 'mode must be standalone, sentinel, or managed'),
        (
            'password_env = "sensitive-marker"',
            ValueError,
            'password_env must be a portable environment name',
        ),
        (
            (
                'mode = "sentinel"\nsentinels = '
                '["sensitive-marker"]\nservice_name = "sensitive-service"'
            ),
            ValueError,
            'sentinels must use host:port',
        ),
        (
            (
                'mode = "sentinel"\nsentinels = '
                '["sensitive-marker:26379"]\nservice_name = "sensitive-service"'
            ),
            ValueError,
            'sentinels requires three distinct sentinel addresses in sentinel mode',
        ),
        pytest.param(
            'mode = "sentinel"\nsentinels = ["sensitive-marker:'
            + '9' * 5000
            + '"]\nservice_name = "sensitive-service"',
            ValueError,
            'sentinels must use host:port',
            id='sentinel-port-conversion-limit',
        ),
        ('mode = "sentinel"', ValueError, 'missing required keys: sentinels, service_name'),
        ('', ValueError, 'missing required keys: host, password_env'),
    ],
)
def test_role_settings_errors_name_the_toml_path(
    tmp_path: Path, role: str, fields: str, error_type: type[Exception], message: str
) -> None:
    contents = 'schema_version = 1\n'
    for name in ('result', 'metadata'):
        contents += f'[cache.l2.valkey.{name}]\n'
        if name == role:
            if fields:
                if not fields.startswith('host ='):
                    contents += f'host = "{name}-host"\n'
                if not fields.startswith('password_env ='):
                    contents += 'password_env = "PASSWORD"\n'
            contents += fields + '\n'
        else:
            contents += f'host = "{name}-host"\npassword_env = "PASSWORD"\n'
    path = tmp_path / 'cache.toml'
    path.write_text(contents, encoding='utf-8')

    with pytest.raises(error_type) as captured:
        CacheSettings.from_toml(path)

    separator = ' ' if message.startswith('missing') else '.'
    assert str(captured.value) == f'cache.l2.valkey.{role}{separator}{message}'
    assert type(captured.value) is error_type
    assert 'sensitive' not in str(captured.value)
    assert '65917' not in str(captured.value)


def test_work_schedule_read_is_the_only_enablable_composite_method() -> None:
    assert CacheSettings(
        enabled_ttl_only_methods=('work_schedule_read',)
    ).enabled_ttl_only_methods == ('work_schedule_read',)
