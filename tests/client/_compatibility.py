import inspect
from collections.abc import Callable

EXPLICIT_METHODS = (
    'activity_stream_read',
    'batch',
    'create',
    'delete',
    'find',
    'find_one',
    'follow',
    'note_thread_read',
    'preferences_read',
    'revive',
    'schema_entity_read',
    'schema_field_create',
    'schema_field_delete',
    'schema_field_read',
    'schema_field_update',
    'schema_read',
    'share_thumbnail',
    'summarize',
    'text_search',
    'unfollow',
    'update',
    'update_project_last_accessed',
    'upload',
    'upload_filmstrip_thumbnail',
    'upload_thumbnail',
    'user_subscriptions_create',
    'work_schedule_read',
    'work_schedule_update',
)


def parameter_contract(
    method: Callable[..., object],
) -> tuple[tuple[str, inspect._ParameterKind, object], ...]:
    return tuple(
        (parameter.name, parameter.kind, parameter.default)
        for parameter in inspect.signature(method).parameters.values()
    )
