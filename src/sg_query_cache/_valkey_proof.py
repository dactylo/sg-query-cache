"""Prove maintained Valkey connections reach a writable primary."""

from valkey.exceptions import ConnectionError as ValkeyConnectionError

from sg_query_cache._backend import BackendProtocolError, BackendUnavailable
from sg_query_cache._valkey_common import primary_replication_id

_PROOF_FIELDS = frozenset(('master_replid', 'role'))


class PrimaryProofError(ValkeyConnectionError):
    """Carry a classified proof failure through the maintained connect hook.

    Subclassing the maintained reconnect-class error lets existing pool
    invalidation discard the pool. `translate_client_error` raises a fresh copy
    of `failure` at the backend boundary.

    Attributes:
        failure: Backend-neutral classification with a fixed message.
    """

    def __init__(self, failure: BackendProtocolError | BackendUnavailable) -> None:
        """Retain the classified failure and reuse its fixed message."""
        super().__init__(str(failure))
        self.failure = failure


def _replication_fields(reply: object) -> dict[str, str]:
    """Read exactly one of each proof field from one raw `INFO replication` reply."""
    malformed = BackendProtocolError('Valkey replication information is malformed')
    if not isinstance(reply, bytes):
        raise malformed
    try:
        text = reply.decode('ascii')
    except UnicodeDecodeError:
        raise malformed from None
    fields: dict[str, str] = {}
    for raw_line in text.split('\n'):
        line = raw_line.removesuffix('\r')
        name, separator, value = line.partition(':')
        if not separator or name not in _PROOF_FIELDS:
            continue
        if name in fields:
            raise malformed
        fields[name] = value
    if fields.keys() != _PROOF_FIELDS:
        raise malformed
    return fields


def require_primary_reply(reply: object) -> str:
    """Return the replication identity a connect-time reply proves.

    Args:
        reply: Raw bulk reply to `INFO replication` on the new socket.

    Raises:
        PrimaryProofError: The reply is malformed or reports a role other than
            `master`.
    """
    try:
        return primary_replication_id(_replication_fields(reply))
    except (BackendProtocolError, BackendUnavailable) as error:
        raise PrimaryProofError(error) from None
