from traceback import format_exception


def exception_text(error: BaseException) -> str:
    parts = [''.join(format_exception(error))]
    current: BaseException | None = error
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        parts.extend((repr(current), str(current)))
        current = current.__cause__ or current.__context__
    return '\n'.join(parts)
