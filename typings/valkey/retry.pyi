from valkey.backoff import NoBackoff

class Retry:
    def __init__(self, backoff: NoBackoff, retries: int) -> None: ...
