import pytest

from tests.valkey import _integration_helpers as helpers
from tests.valkey._environment import topology_settings


def test_bracketed_ipv6_sentinel_addresses_reach_discovery_without_brackets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environ = {
        'SG_QUERY_CACHE_TEST_SENTINEL_ADDRESSES': '[::1]:26379,[::2]:26380,[::3]:26381',
        'SG_QUERY_CACHE_TEST_METADATA_DATABASE': '0',
        'SG_QUERY_CACHE_TEST_METADATA_TLS': 'false',
        'SG_QUERY_CACHE_TEST_METADATA_PASSWORD': 'secret',
        'SG_QUERY_CACHE_TEST_SENTINEL_METADATA_SERVICE': 'metadata-primary',
        'SG_QUERY_CACHE_TEST_RESULT_DATABASE': '0',
        'SG_QUERY_CACHE_TEST_RESULT_TLS': 'false',
        'SG_QUERY_CACHE_TEST_SENTINEL_RESULT_SERVICE': 'result-primary',
    }
    settings = topology_settings('sentinel', environ)
    assert settings.l2.valkey.metadata.host == '::1'
    seen: list[tuple[tuple[str, int], ...]] = []

    class _Sentinel:
        def __init__(self, addresses: tuple[tuple[str, int], ...], **kwargs: object) -> None:
            _ = kwargs
            seen.append(addresses)

        def master_for(self, service: str) -> object:
            assert service == 'metadata-primary'
            return object()

    monkeypatch.setenv('SG_QUERY_CACHE_TEST_METADATA_PASSWORD', 'secret')
    monkeypatch.setattr(helpers, 'Sentinel', _Sentinel)
    topology = helpers.Topology('sentinel', settings)
    helpers.primary(topology, 'metadata')
    assert seen == [(('::1', 26379), ('::2', 26380), ('::3', 26381))]
