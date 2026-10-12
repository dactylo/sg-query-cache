import pytest
from valkey.exceptions import ConnectionError as ValkeyConnectionError

from sg_query_cache._backend import BackendProtocolError, BackendUnavailable
from sg_query_cache._valkey_proof import PrimaryProofError, require_primary_reply

_REPLICATION_ID = b'a' * 40
_PRIMARY_REPLY = (
    b'# Replication\r\nrole:master\r\nconnected_slaves:0\r\nmaster_replid:'
    + _REPLICATION_ID
    + b'\r\n'
)


@pytest.mark.parametrize(
    'reply',
    [
        b'master_replid:' + _REPLICATION_ID + b'\r\n',
        b'role:master\r\n',
        b'role:master\r\nrole:master\r\nmaster_replid:' + _REPLICATION_ID + b'\r\n',
        b'role:slave\r\nrole:master\r\nmaster_replid:' + _REPLICATION_ID + b'\r\n',
        b'role:master\r\nmaster_replid:'
        + _REPLICATION_ID
        + b'\r\nmaster_replid:'
        + _REPLICATION_ID
        + b'\r\n',
        b'role:master\r\nmaster_replid:' + _REPLICATION_ID + b'\r\nother:\xff\r\n',
        None,
        'role:master\r\nmaster_replid:' + 'a' * 40,
        ['role:master', 'master_replid:' + 'a' * 40],
        b'role:master\r\nmaster_replid:\r\n',
        b'role:master\r\nmaster_replid:bad\x01id\r\n',
        b'role:master\r\nmaster_replid:bad\x0bid\r\n',
    ],
)
def test_malformed_reply_is_protocol_failure(reply: object) -> None:
    with pytest.raises(PrimaryProofError) as caught:
        require_primary_reply(reply)

    assert isinstance(caught.value.failure, BackendProtocolError)


def test_primary_reply_returns_replication_identity() -> None:
    assert require_primary_reply(_PRIMARY_REPLY) == 'a' * 40


def test_replica_reply_is_unavailable_and_invalidates_connection() -> None:
    with pytest.raises(PrimaryProofError) as caught:
        require_primary_reply(b'role:slave\r\nmaster_replid:' + _REPLICATION_ID + b'\r\n')

    assert isinstance(caught.value, ValkeyConnectionError)
    assert isinstance(caught.value.failure, BackendUnavailable)
    assert 'not writable' in str(caught.value.failure)


def test_reply_fields_do_not_leak_into_failure_messages() -> None:
    reply = _PRIMARY_REPLY + b'other:credential-bearing\r\n'

    assert require_primary_reply(reply) == 'a' * 40
    with pytest.raises(PrimaryProofError) as caught:
        require_primary_reply(reply.replace(b'role:master', b'role:slave'))

    assert 'credential-bearing' not in str(caught.value)
    assert 'credential-bearing' not in repr(caught.value.failure)
