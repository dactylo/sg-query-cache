from valkey.backoff import NoBackoff

class Retry:
    _retries: int

    def __init__(self, backoff: NoBackoff, retries: int) -> None: ...
