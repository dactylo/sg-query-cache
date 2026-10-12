"""Define the atomic Lua programs used by the Valkey backend."""

from collections.abc import Callable
from hashlib import sha1
from math import floor
from typing import Never, Protocol, TypeVar

from valkey.exceptions import (
    ConnectionError as ValkeyConnectionError,
)
from valkey.exceptions import (
    DataError,
    NoScriptError,
    OutOfMemoryError,
    ResponseError,
)
from valkey.exceptions import (
    TimeoutError as ValkeyTimeoutError,
)

from sg_query_cache._backend import (
    BackendCapacity,
    BackendProtocolError,
    BackendUnavailable,
    StaleIncarnation,
)
from sg_query_cache._config import VALKEY_MAX_DEPENDENCY_TAGS_V1
from sg_query_cache._model import FillLease
from sg_query_cache._valkey_common import (
    ScriptArgument,
    commit_operands,
    decode_commit_reply,
    decode_digest,
    require_loaded_digest,
)
from sg_query_cache._valkey_proof import PrimaryProofError

_T = TypeVar('_T')

_SCRIPT_ERROR_FUNCTIONS = """
local function is_capacity_error(reply)
    return string.sub(reply.err, 1, 4) == 'OOM '
end

local function is_counter_error(reply)
    return reply.err == 'ERR increment or decrement would overflow'
        or reply.err == 'ERR value is not an integer or out of range'
        or string.sub(reply.err, 1, 10) == 'WRONGTYPE '
end
""".strip()

CHANGE_SCRIPT = (
    _SCRIPT_ERROR_FUNCTIONS
    + '\n\n'
    + """
local control = redis.call('GET', KEYS[1])
if not control or control ~= ARGV[1] then
    return {'stale'}
end
if redis.call('EXISTS', KEYS[2]) ~= 1 then
    return {'missing-counter'}
end

local replaced = 0
local absent = 0
for index = 3, #KEYS do
    local global = ARGV[index]
    local materialized = global == '1' or redis.call('EXISTS', KEYS[index]) == 1
    if materialized then
        local ttl
        if global ~= '1' then
            ttl = redis.call('PTTL', KEYS[index])
            if ttl <= 0 then
                ttl = ARGV[2]
            end
        end
        local allocated = redis.pcall('INCR', KEYS[2])
        if type(allocated) == 'table' and allocated.err then
            if is_capacity_error(allocated) then
                return {'capacity'}
            end
            if is_counter_error(allocated) then
                return {'rotation-required'}
            end
            return allocated
        end
        local token = redis.call('GET', KEYS[2])
        local written
        if global == '1' then
            written = redis.pcall('SET', KEYS[index], token)
        else
            written = redis.pcall('SET', KEYS[index], token, 'PX', ttl)
        end
        if type(written) == 'table' and written.err then
            if is_capacity_error(written) then
                return {'capacity'}
            end
            return written
        end
        replaced = replaced + 1
    else
        absent = absent + 1
    end
end
return {'ok', replaced, absent}
""".strip()
)
CHANGE_SCRIPT_SHA = sha1(CHANGE_SCRIPT.encode(), usedforsecurity=False).hexdigest()

COMMIT_SCRIPT = (
    _SCRIPT_ERROR_FUNCTIONS
    + '\n\n'
    + """
if redis.call('GET', KEYS[1]) ~= ARGV[1] then
    return {'rejected'}
end
local written = redis.pcall('SET', KEYS[2], ARGV[2], 'PX', ARGV[3])
if type(written) == 'table' and written.err then
    if is_capacity_error(written) then
        return {'capacity'}
    end
    return written
end
redis.call('DEL', KEYS[1])
return {'committed'}
""".strip()
)
COMMIT_SCRIPT_SHA = sha1(COMMIT_SCRIPT.encode(), usedforsecurity=False).hexdigest()

CONTROL_CREATE_SCRIPT = (
    _SCRIPT_ERROR_FUNCTIONS
    + '\n\n'
    + """
local current = redis.call('GET', KEYS[1])
if current then
    return {'winner', current}
end
if redis.call('EXISTS', KEYS[2]) ~= 0 then
    return {'retry'}
end
local counter = redis.pcall('SET', KEYS[2], '0', 'NX')
if type(counter) == 'table' and counter.err then
    if is_capacity_error(counter) then
        return {'capacity'}
    end
    return counter
end
if not counter then
    return {'retry'}
end
local control = redis.pcall('SET', KEYS[1], ARGV[1], 'NX')
if type(control) == 'table' and control.err then
    redis.call('DEL', KEYS[2])
    if is_capacity_error(control) then
        return {'capacity'}
    end
    return control
end
if not control then
    redis.call('DEL', KEYS[2])
    current = redis.call('GET', KEYS[1])
    return {'winner', current or ''}
end
return {'created', ARGV[1]}
""".strip()
)
CONTROL_CREATE_SCRIPT_SHA = sha1(CONTROL_CREATE_SCRIPT.encode(), usedforsecurity=False).hexdigest()

CONTROL_ROTATE_SCRIPT = (
    _SCRIPT_ERROR_FUNCTIONS
    + '\n\n'
    + """
local current = redis.call('GET', KEYS[1])
local matches = false
if ARGV[1] == '' then
    matches = not current
else
    matches = current == ARGV[1]
end
if not matches then
    return {'winner', current or ''}
end
if redis.call('EXISTS', KEYS[2]) ~= 0 then
    return {'retry'}
end
local counter = redis.pcall('SET', KEYS[2], '0', 'NX')
if type(counter) == 'table' and counter.err then
    if is_capacity_error(counter) then
        return {'capacity'}
    end
    return counter
end
if not counter then
    return {'retry'}
end
local control = redis.pcall('SET', KEYS[1], ARGV[2])
if type(control) == 'table' and control.err then
    redis.call('DEL', KEYS[2])
    if is_capacity_error(control) then
        return {'capacity'}
    end
    return control
end
return {'rotated', ARGV[2]}
""".strip()
)
CONTROL_ROTATE_SCRIPT_SHA = sha1(CONTROL_ROTATE_SCRIPT.encode(), usedforsecurity=False).hexdigest()

LOOKUP_SCRIPT = """
local value = redis.call('GET', KEYS[1])
if not value then
    return {'miss'}
end
local ttl = redis.call('PTTL', KEYS[1])
if ttl <= 0 then
    return {'miss'}
end
return {'hit', value, ttl}
""".strip()
LOOKUP_SCRIPT_SHA = sha1(LOOKUP_SCRIPT.encode(), usedforsecurity=False).hexdigest()

RELEASE_SCRIPT = """
if redis.call('GET', KEYS[1]) ~= ARGV[1] then
    return {'retained'}
end
redis.call('DEL', KEYS[1])
return {'released'}
""".strip()
RELEASE_SCRIPT_SHA = sha1(RELEASE_SCRIPT.encode(), usedforsecurity=False).hexdigest()

RESERVE_SCRIPT = """
local reserved = redis.call('SET', KEYS[1], ARGV[1], 'NX', 'PX', ARGV[2])
if not reserved then
    return {'busy'}
end
return {'reserved'}
""".strip()
RESERVE_SCRIPT_SHA = sha1(RESERVE_SCRIPT.encode(), usedforsecurity=False).hexdigest()

RESOLVE_SCRIPT = (
    _SCRIPT_ERROR_FUNCTIONS
    + '\n\n'
    + """
local control = redis.call('GET', KEYS[1])
if not control or control ~= ARGV[1] then
    return {'stale'}
end
if redis.call('EXISTS', KEYS[2]) ~= 1 then
    return {'missing-counter'}
end
if #KEYS - 2 > __VALKEY_MAX_DEPENDENCY_TAGS_V1__ then
    return {'too-many-tags'}
end

local tokens = {'ok'}
for index = 3, #KEYS do
    local token = redis.call('GET', KEYS[index])
    if not token then
        local allocated = redis.pcall('INCR', KEYS[2])
        if type(allocated) == 'table' and allocated.err then
            if is_capacity_error(allocated) then
                return {'capacity'}
            end
            if is_counter_error(allocated) then
                return {'rotation-required'}
            end
            return allocated
        end
        token = redis.call('GET', KEYS[2])
        local written = redis.pcall('SET', KEYS[index], token)
        if type(written) == 'table' and written.err then
            if is_capacity_error(written) then
                return {'capacity'}
            end
            return written
        end
    end
    if ARGV[index] == '1' then
        redis.call('PERSIST', KEYS[index])
    else
        local expiry = redis.pcall('PEXPIRE', KEYS[index], ARGV[2])
        if type(expiry) == 'table' and expiry.err then
            if is_capacity_error(expiry) then
                return {'capacity'}
            end
            return expiry
        end
    end
    tokens[#tokens + 1] = token
end
return tokens
""".replace('__VALKEY_MAX_DEPENDENCY_TAGS_V1__', str(VALKEY_MAX_DEPENDENCY_TAGS_V1)).strip()
)
RESOLVE_SCRIPT_SHA = sha1(RESOLVE_SCRIPT.encode(), usedforsecurity=False).hexdigest()


class ScriptClient(Protocol):
    """Execute and load synchronous Valkey scripts."""

    def evalsha(self, sha: str, numkeys: int, *keys_and_args: ScriptArgument) -> object:
        """Execute one cached script by digest."""
        ...

    def script_load(self, script: bytes | str) -> object:
        """Load one script and return its server digest."""
        ...


class ResultClient(Protocol):
    """Run result work on one prepared pin through role settlement."""

    def run_operation(self, operation: Callable[[ScriptClient], _T]) -> _T:
        """Execute result work within one observed physical-connection span."""
        ...


def translate_client_error(error: Exception) -> Never:
    """Raise a redacted backend-neutral failure without chaining the client error.

    Raises:
        BackendCapacity: The server reports its `OOM` error code.
        BackendProtocolError: Proof is malformed or the client reports a
            command data error.
        BackendUnavailable: Proof finds a replica or any other client failure
            occurs.
    """
    if isinstance(error, PrimaryProofError):
        raise type(error.failure)(*error.failure.args) from None
    if isinstance(error, (ValkeyConnectionError, ValkeyTimeoutError)):
        raise BackendUnavailable('Valkey operation is unavailable') from None
    if isinstance(error, DataError):
        raise BackendProtocolError('Valkey rejected a backend command') from None
    if isinstance(error, OutOfMemoryError) or (
        isinstance(error, ResponseError) and str(error).startswith('OOM ')
    ):
        raise BackendCapacity('Valkey capacity rejected the operation') from None
    if isinstance(error, ResponseError):
        raise BackendUnavailable('Valkey operation failed') from None
    raise BackendUnavailable('Valkey operation failed') from None


def execute_commit(  # noqa: PLR0913 - explicit commit deadline inputs
    client: ResultClient,
    *,
    prefix: str,
    lease: FillLease,
    payload: bytes,
    deadline: float,
    clock: Callable[[], float],
) -> bool:
    """Commit on one prepared pin, debiting preparation and one script load.

    Args:
        client: Result role that observes the complete operation span.
        prefix: Configured result-role key prefix.
        lease: Exact fill ownership to compare atomically.
        payload: Original encoded result envelope, retained unchanged.
        deadline: Expiration anchored at backend commit entry in `clock` time.
        clock: Backend monotonic clock used immediately before each attempt.

    Returns:
        Whether the live lease admitted the result before deadline exhaustion.

    Raises:
        BackendCapacity: Valkey rejects a command for capacity.
        BackendProtocolError: A command, loaded digest, or reply is malformed.
        BackendUnavailable: The client cannot complete or settle the operation.
    """

    def operation(pinned: ScriptClient) -> bool:
        def attempt() -> bool:
            ttl = floor((deadline - clock()) * 1000)
            if ttl <= 0:
                return False
            operands = commit_operands(prefix, lease, payload, ttl)
            reply = pinned.evalsha(
                COMMIT_SCRIPT_SHA,
                len(operands.keys),
                *operands.keys,
                *operands.args,
            )
            return decode_commit_reply(reply)

        try:
            return attempt()
        except NoScriptError:
            pass
        loaded = decode_digest(pinned.script_load(COMMIT_SCRIPT))
        require_loaded_digest(loaded, COMMIT_SCRIPT_SHA)
        try:
            return attempt()
        except NoScriptError:
            pass
        raise BackendUnavailable('Valkey script cache remained unavailable') from None

    failure: Exception | None = None
    try:
        return client.run_operation(operation)
    except (BackendCapacity, BackendProtocolError, BackendUnavailable, StaleIncarnation):
        raise
    except Exception as error:  # noqa: BLE001 - closed external-client boundary
        failure = error
    translate_client_error(failure)


def execute_script(
    client: ScriptClient,
    *,
    script: str,
    sha: str,
    keys: tuple[str, ...],
    args: tuple[ScriptArgument, ...],
) -> object:
    """Execute a cached script, loading and retrying once after `NOSCRIPT`.

    Args:
        client: Synchronous maintained-client boundary.
        script: Exact Lua source bound to `sha`.
        sha: Expected lowercase SHA-1 digest.
        keys: Ordered Valkey keys supplied before script arguments.
        args: Ordered non-secret script arguments.

    Returns:
        Raw script response for operation-specific decoding.

    Raises:
        BackendCapacity: Valkey rejects the operation for capacity.
        BackendProtocolError: Script loading or command arguments are
            malformed.
        BackendUnavailable: The client cannot complete the operation.
    """
    failure: Exception | None = None
    try:
        return client.evalsha(sha, len(keys), *keys, *args)
    except NoScriptError:
        pass
    except (BackendCapacity, BackendProtocolError, BackendUnavailable, StaleIncarnation):
        raise
    except Exception as error:  # noqa: BLE001 - closed external-client boundary
        failure = error
    if failure is not None:
        translate_client_error(failure)
    failure = None
    try:
        loaded = decode_digest(client.script_load(script))
        require_loaded_digest(loaded, sha)
        return client.evalsha(sha, len(keys), *keys, *args)
    except NoScriptError:
        raise BackendUnavailable('Valkey script cache remained unavailable') from None
    except (BackendCapacity, BackendProtocolError, BackendUnavailable, StaleIncarnation):
        raise
    except Exception as error:  # noqa: BLE001 - closed external-client boundary
        failure = error
    translate_client_error(failure)
