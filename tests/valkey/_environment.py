from collections.abc import Mapping

import pytest

from sg_query_cache._config import (
    CacheSettings,
    L1Settings,
    L2Settings,
    LimitSettings,
    ValkeyRoleSettings,
    ValkeySettings,
    _ValkeyMode,
    parse_sentinel_address,
)

RETENTION_SECONDS = 0.1


def required(name: str, environ: Mapping[str, str]) -> str:
    value = environ.get(name)
    if not value:
        pytest.fail(f'required disposable-service setting {name} is unset')
    return value


def _boolean(name: str, environ: Mapping[str, str]) -> bool:
    value = required(name, environ).lower()
    if value == 'true':
        return True
    if value == 'false':
        return False
    pytest.fail(f'{name} must be true or false')


def integer(name: str, environ: Mapping[str, str]) -> int:
    try:
        return int(required(name, environ))
    except ValueError:
        pytest.fail(f'{name} must be an integer')


def _role(
    role: str,
    environ: Mapping[str, str],
    *,
    mode: _ValkeyMode = 'standalone',
    key_prefix: str = 'sgqc-integration',
) -> ValkeyRoleSettings:
    prefix = f'SG_QUERY_CACHE_TEST_{role.upper()}'
    username = environ.get(f'{prefix}_USERNAME') or None
    if mode == 'sentinel':
        addresses = tuple(
            item.strip()
            for item in required('SG_QUERY_CACHE_TEST_SENTINEL_ADDRESSES', environ).split(',')
            if item.strip()
        )
        return ValkeyRoleSettings(
            parse_sentinel_address(addresses[0])[0],
            f'{prefix}_PASSWORD',
            mode='sentinel',
            database=integer(f'{prefix}_DATABASE', environ),
            tls=_boolean(f'{prefix}_TLS', environ),
            username=username,
            key_prefix=key_prefix,
            sentinels=addresses,
            service_name=required(f'SG_QUERY_CACHE_TEST_SENTINEL_{role.upper()}_SERVICE', environ),
        )
    return ValkeyRoleSettings(
        required(f'{prefix}_HOST', environ),
        f'{prefix}_PASSWORD',
        mode=mode,
        port=integer(f'{prefix}_PORT', environ),
        database=integer(f'{prefix}_DATABASE', environ),
        tls=_boolean(f'{prefix}_TLS', environ),
        username=username,
        key_prefix=key_prefix,
    )


def settings(
    environ: Mapping[str, str],
    limits: LimitSettings | None = None,
) -> CacheSettings:
    return CacheSettings(
        site_namespace='integration',
        l1=L1Settings(enabled=False),
        l2=L2Settings(
            enabled=True,
            valkey=ValkeySettings(
                result=_role('result', environ),
                metadata=_role('metadata', environ),
            ),
        ),
        limits=limits or LimitSettings(scoped_generation_retention_seconds=RETENTION_SECONDS),
    )


def topology_settings(profile: _ValkeyMode, environ: Mapping[str, str]) -> CacheSettings:
    return CacheSettings(
        site_namespace='failover',
        l1=L1Settings(enabled=False),
        l2=L2Settings(
            enabled=True,
            valkey=ValkeySettings(
                result=_role('result', environ, mode=profile, key_prefix='sgqc-failover-result'),
                metadata=_role(
                    'metadata', environ, mode=profile, key_prefix='sgqc-failover-metadata'
                ),
            ),
        ),
    )
