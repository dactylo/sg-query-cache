from collections import OrderedDict
from collections.abc import Iterator
from typing import TypeVar

from typing_extensions import override

_K = TypeVar('_K')


class CountedQueue(OrderedDict[_K, None]):
    def __init__(self, entries: OrderedDict[_K, None]) -> None:
        super().__init__(entries)
        self._visits = 0

    @override
    def __iter__(self) -> Iterator[_K]:
        for key in super().__iter__():
            self._visits += 1
            yield key
