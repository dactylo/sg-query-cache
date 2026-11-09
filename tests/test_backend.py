import pytest

from sg_query_cache._backend import BackendUnavailable


def test_backend_failure_class_is_read_only_and_optional() -> None:
    failure = BackendUnavailable('deadline')
    assert failure.failure_class is None
    assert str(failure) == 'deadline'
    with pytest.raises(AttributeError):
        object.__setattr__(failure, 'failure_class', 'timeout')


@pytest.mark.parametrize('args', [(), ('message',), ('message', 42)])
def test_backend_unavailable_keeps_unclassified_exception_arguments(
    args: tuple[object, ...],
) -> None:
    failure = BackendUnavailable(*args)
    assert failure.args == args
    assert str(failure) == str(RuntimeError(*args))
