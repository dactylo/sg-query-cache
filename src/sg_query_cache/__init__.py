"""Cache exact SG query results under explicit identity and freshness contracts.

Wrap a caller-owned SG client with the synchronous or native asynchronous
adapter using validated settings, an explicit effective identity, and matching
response context. Caching provides bounded staleness; it does not authenticate
the supplied identity or make reads causally consistent with SG writes.

Close each owned cache when its work finishes. The async owning factory closes
its constructed cache; directly injected caches remain caller-owned. Async
operations and resource cleanup must use one event loop. Optional Valkey
backends are imported explicitly from `sg_query_cache.valkey` or
`sg_query_cache.valkey_async`.
"""

from sg_query_cache._async_backend import AsyncCacheBackend
from sg_query_cache._async_cache import AsyncQueryCache
from sg_query_cache._async_client import AsyncCachedShotgun
from sg_query_cache._async_local import AsyncLocalBackend
from sg_query_cache._cache import QueryCache
from sg_query_cache._client import CachedShotgun, ClientContext
from sg_query_cache._config import CacheSettings
from sg_query_cache._local import LocalBackend
from sg_query_cache._metrics import MetricsSink, NullMetrics
from sg_query_cache._model import Identity

__all__ = [
    'AsyncCacheBackend',
    'AsyncCachedShotgun',
    'AsyncLocalBackend',
    'AsyncQueryCache',
    'CacheSettings',
    'CachedShotgun',
    'ClientContext',
    'Identity',
    'LocalBackend',
    'MetricsSink',
    'NullMetrics',
    'QueryCache',
]
