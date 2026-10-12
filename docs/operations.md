# Operations

This guide is for operators who run the shared or hierarchical layout: the Valkey services, the SG
event-daemon plugin that invalidates cached results, monitoring, and recovery. The local layout
needs none of it. Settings are described in [configuration](configuration.md); to try the cache
first, start with the [README quick start](../README.md#quick-start).

The package supplies the cache clients, the invalidation plugin, and the `sg-query-cache` operator
commands. Your deployment supplies and operates Valkey.

| If you want to...                            | Start here                                                        |
| -------------------------------------------- | ----------------------------------------------------------------- |
| Provision the result and metadata services   | [Service layout](#service-layout)                                 |
| Check which Valkey deployments are supported | [Supported deployments](#supported-deployments)                   |
| Restrict network and ACL access              | [Network and access control](#network-and-access-control)         |
| Install the invalidation plugin              | [Event-daemon invalidation](#event-daemon-invalidation)           |
| Recover from a failed invalidation callback  | [Failed invalidation callbacks](#failed-invalidation-callbacks)   |
| Flush cached results                         | [Flush or rotate cached results](#flush-or-rotate-cached-results) |
| Plan metadata capacity                       | [Metadata capacity](#metadata-capacity)                           |
| Collect metrics and set alerts               | [Metrics and alerts](#metrics-and-alerts)                         |
| Bound timeouts and shut down async caches    | [Async deadlines and shutdown](#async-deadlines-and-shutdown)     |
| Look up a metric or label name               | [Metric reference](#metric-reference)                             |

## Service layout

The shared and hierarchical layouts use the shared Valkey layer (L2), which needs two independent,
writable Valkey roles:

| Role     | Contents                                                      | Eviction policy | Durability                                              |
| -------- | ------------------------------------------------------------- | --------------- | ------------------------------------------------------- |
| Result   | Serialized results and fill leases                            | `allkeys-lru`   | Persistence is unnecessary                              |
| Metadata | Incarnation, dependency generations, and replication identity | `noeviction`    | Self-managed deployments use AOF `appendfsync everysec` |

A fill lease is the short reservation one process holds while it fetches a missing result, so other
processes wait for that fetch instead of all reading SG. Dependency generations are the version
numbers that decide whether a stored result is still current; the
[consistency guide](consistency.md#how-the-cache-works) explains them. The incarnation identifies
the current contents of the metadata role. Every key includes it, so changing it invalidates every
result under the key prefix. The cache changes it when the metadata primary's replication identity
changes after a failover, because a new primary may have lost recent generation changes.

Do not place both roles at one production endpoint. Losing result data only causes misses. Evicting
or silently losing metadata can make an old result appear current again, so the metadata role must
never evict keys. Managed-service durability controls differ by provider: choose settings that
preserve the metadata role through the provider's documented primary recovery, and test promotion
in a disposable environment.

When a managed service does not expose `maxmemory`, meet the configured byte budgets through node
sizing and capacity monitoring. The L2 `result_budget_bytes` and `metadata_budget_bytes` settings
are sizing targets; the client does not enforce either total. It does enforce the per-result and
dependency-list limits, and each process enforces its own L1 budgets.

## Supported deployments

Standalone Valkey and Valkey Sentinel are supported. Each role connects to one writable primary, so
ElastiCache Serverless and Valkey Cluster are not supported. You can configure a managed primary
endpoint with `mode = "managed"`, but compatibility with specific providers has not been verified,
including node-based, cluster-mode-disabled Multi-AZ ElastiCache.

A managed deployment needs, for each role:

- a separate writable endpoint, with the eviction policy from the
  [service layout](#service-layout);
- primary-only connections, and reconnection or redirection to the new primary after failover;
- the commands listed under [network and access control](#network-and-access-control), including
  Lua scripting;
- a replication identity that changes when a replica is promoted, so the cache detects failover.

For ElastiCache, use one cluster-mode-disabled Multi-AZ replication group per role. Test both roles
in a disposable environment, and check the provider's current failover-test limits before
scheduling repeated tests.

Measure your SG workload, node sizes, latency, and failover behavior before rollout; local tests do
not establish behavior at farm scale. The [benchmark guide](benchmarks.md) measures duplicate-call
savings and cache costs for a synthetic workload.

## Network and access control

Keep `tls = true` unless an isolated test service deliberately lacks TLS. Supply each role's
password only through its configured `password_env` variable. Grant each role's user access only to
that role's key prefix, and only these commands, including those that the cache's scripts run:

```text
GET SET DEL EXISTS INCR PERSIST PEXPIRE PTTL EVALSHA SCRIPT LOAD INFO replication SELECT
```

The `SELECT` permission is needed when a role uses a nonzero `database`. Do not grant broad
administrative or cross-prefix access. The cache uses `INFO replication` to confirm that each new
connection in managed and Sentinel modes reaches a writable primary, and to detect a change of
primary.

In Sentinel mode, each role uses its configured username, password, and TLS setting for every
Sentinel address as well as its primary. Each Sentinel must accept those credentials and transport
settings and allow `SENTINEL MASTERS` for discovery. A Sentinel that rejects discovery prevents
operator commands from completing and causes cache reads to bypass L2.

Both roles use short connect and operation timeouts, 0.1 and 0.25 seconds by default, so an
unavailable Valkey delays reads only briefly. Five consecutive failures open a role's circuit by
default; while it is open, reads go directly to SG. After 10 seconds, one probe tries again with a
fresh connection pool. Success closes the circuit, and failure reopens it.

Backend keys contain SHA-256 digests rather than raw query text, so ordinary key listings do not
show field values or filter text. Hashing is not encryption: someone who can list keys and guess
likely queries can confirm a guess by comparing digests. Protect Valkey with network controls, TLS,
ACLs, and credentials.

## Event-daemon invalidation

In the shared and hierarchical layouts, the sg-event-daemon plugin included in this package turns
SG events into invalidations, so changes made by other clients appear before results reach their
maximum age.

Install `sg-query-cache[valkey]` in the event daemon's environment, then print the plugin's
installed path:

```sh
sg-query-cache daemon-plugin-path
```

Configure the plugin in its own group with synchronous execution and eager ordering, using the path
that command printed:

```toml
[sites.production.groups.cache-invalidation]
source = "primary"
execution = "sync"
ordering = "eager"

[[sites.production.groups.cache-invalidation.plugins]]
id = "sg-query-cache"
path = "/installed/path/reported/by/daemon-plugin-path.py"
config = { config_path = "/etc/sg-query-cache.toml" }
secrets.SG_QUERY_CACHE_METADATA_PASSWORD = { env = "SG_QUERY_CACHE_METADATA_PASSWORD" }
```

- The plugin's callback is synchronous, so the group uses `execution = "sync"`. Eager ordering
  delivers events with the least delay, which shortens the time a changed result can still be
  served.
- The daemon site name, `production` here, must equal `cache.site_namespace`; setup fails
  otherwise.
- `config_path` points to the shared cache settings file. Use an absolute path; a relative path is
  resolved from the daemon worker's working directory.
- The secret's key must equal the metadata role's `password_env`. The plugin uses only the metadata
  role.

When the plugin starts, it connects to the metadata role and flushes the whole site before it
handles any events. Every daemon or plugin-worker restart therefore invalidates all shared results
for the site and can temporarily increase SG load. A separate group keeps other plugins' failures
and restarts apart from cache invalidation, and lets you retry or skip invalidation on its own.
Monitor restart frequency and budget for the refill.

### Failed invalidation callbacks

A failed callback leaves an invalidation unapplied, so affected results can stay cached until their
maximum age. Choose one recovery path:

- For an ordinary transient failure, use the daemon's `retry` command. The callback stays an
  outstanding obligation until it succeeds:

  ```sh
  sg-event-daemon retry --config "$DAEMON_CONFIG" \
      --site production --group cache-invalidation
  ```

- To run the whole current event again under the current callback registration, use
  `restart-event`. It clears earlier completion marks and does not require a cache flush:

  ```sh
  sg-event-daemon restart-event --config "$DAEMON_CONFIG" \
      --site production --group cache-invalidation --reason "$REASON"
  ```

- To skip the failed callback instead, flush the site first and confirm that the flush was
  acknowledged, then skip:

  ```sh
  sg-query-cache flush --config /etc/sg-query-cache.toml \
      --site production --reason daemon_recovery
  sg-event-daemon skip-callback --config "$DAEMON_CONFIG" \
      --site production --group cache-invalidation --reason "$REASON"
  ```

Never run `skip-callback` before the flush succeeds: the skipped invalidation would otherwise be
lost. A successful skip can retire the event in progress, so do not follow it with `restart-event`
as though the commands were one sequence.

## Flush or rotate cached results

Two commands invalidate shared results. Both need `sg-query-cache[valkey]` in the environment that
runs them, and both read the metadata role's password from its `password_env` variable.

To invalidate every cached result for one SG site, flush the site. This changes the site's
generation; it does not delete result data:

```sh
sg-query-cache flush --config /etc/sg-query-cache.toml \
    --site production --reason incident_recovery
```

`--site` is the site namespace and `--reason` is 1 to 64 ASCII characters. The backend classifies
`cache_site_flush` reasons as `daemon_setup`, `daemon_recovery`, or `operator` for any other
reason. The supplied CLI and daemon plugin use `NullMetrics`, so their flushes do not emit this
metric to a sink.

To invalidate every site under the key prefix at once, rotate the incarnation. Confirm the metadata
role's `key_prefix`:

```sh
sg-query-cache rotate-incarnation --config /etc/sg-query-cache.toml \
    --confirm-prefix sgqc-production
```

Rotation is wider than a site flush. Neither command reclaims metadata memory; see
[metadata capacity](#metadata-capacity).

On success, `flush` prints `flush acknowledged site=` followed by a SHA-256 digest of the site
namespace, and `rotate-incarnation` prints `incarnation rotation acknowledged for all sites`. The
exit status tells you what happened:

| Exit | Meaning                                                                                                                                                      |
| ---- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 0    | The metadata role acknowledged the operation.                                                                                                                |
| 1    | The operation was not acknowledged. Check Valkey and run the command again.                                                                                  |
| 2    | The settings file is invalid, unavailable, or has L2 disabled; the `valkey` extra or the metadata password is missing; or `--confirm-prefix` does not match. |

Apart from usage errors, which print the command's usage, each failure prints one line to standard
error without settings values or secrets: `error: flush operation was not acknowledged` or
`error: incarnation rotation was not acknowledged` for exit 1, and
`error: configuration is invalid or unavailable` or
`error: prefix confirmation does not match configuration` for exit 2.

## Metadata capacity

The metadata role uses `noeviction`, so when it is full, Valkey rejects new metadata writes and the
affected cache operations go directly to SG. Nothing removes generation records early. A record for
a specific entity, entity type, field, or project expires after it goes unused for
`scoped_generation_retention_seconds`, 7,200 seconds by default; site-wide, schema, and permission
records never expire. Space returns as those records expire, or when you add capacity to the
metadata service and raise `metadata_budget_bytes` to match.

A site flush or incarnation rotation changes key names and invalidates old results, but neither
deletes old metadata or reclaims its memory, so neither is a capacity remedy. Before the service
fills, monitor its memory headroom, `cache_metadata_write_rejection`, and backend errors.

## Metrics and alerts

Cache components report counters and observations to a metrics sink: any object with
`increment(name, labels)` and `observe(name, value, labels)` methods. Build the safe metrics
boundary from your sink before passing it to cache components:

```python
metrics = QueryCache.build_metrics(settings, os.environ, sink)
```

Pass the returned `metrics` to `LocalBackend`, `ValkeyBackend.from_settings`, and `QueryCache`. The
boundary accepts only documented metric names and safe labels. With the
[metrics correlation secret](configuration.md#metrics-correlation-secret) configured, metrics
emitted by `QueryCache` during a read include `exact_call_digest`. Backend metrics never carry it,
including those emitted during a read. `AsyncCachedShotgun.open()` builds the boundary itself from
its `metrics=` argument. `NullMetrics` discards all metrics.

Alert on:

- sustained `backend_failure`, and its `outcome` values `open` or `reopened`, which mean a role's
  circuit opened;
- `cache_metadata_write_rejection`, which means the metadata service is full;
- `cache_lease_timeout`, which means a wait budget expired; the request may still use a cached
  result, fetch SG, or publish a parsed memo afterward;
- a rising `cache_bypass` rate;
- unexpected `cache_incarnation` activity. Its `validated` outcome is routine: it recurs for each
  metadata connection in every validation interval and after each reconnect;
- `cache_site_flush` activity from cache components configured with a metrics sink. The supplied
  CLI and daemon plugin discard their metrics.

Judge the benefit by comparing `cache_sg_avoided_calls` and `cache_sg_avoided_seconds` with your SG
request durations; hit rate alone does not describe it.

## Async deadlines and shutdown

`limits.operation_timeout_seconds`, 0.25 seconds by default, applies differently in the two
execution models:

- Synchronous Valkey clients use it as a socket timeout, so a reconnect and a script reload can
  each take another full timeout.
- Async clients use it as one cooperative budget for each backend operation, and for each whole
  fence or release cleanup. Reconnecting, loading scripts, and settling share that budget. Each
  backend close gets its own budget.

Neither is a hard wall-clock limit: cancellation cleanup can finish after the deadline. A custom
async backend must propagate cancellation after finite, nonblocking cleanup.

Async metrics show how operations ended. An SG fetch cancelled before it returns a result emits
`cache_fill` with outcome `cancelled`. Cancellation while a result is being stored keeps the
successful fill metric and any valid stored result. An expired backend deadline emits
`backend_failure` with reason `finalization_timeout`; a fill lease whose release failed expires on
its own.

Closing an async cache is final. Close waits, with no deadline of its own, for every admitted
operation, including SG calls and their cleanup, before it closes each backend. If the caller of
close is cancelled, close keeps waiting and then raises the original cancellation.

Close a cache from outside the chain of operations that are using it. The cache rejects a close
from the task that is running one of its operations, but it checks only that task, not its parents.
A parent operation waiting on a child task, or a `TaskGroup` child, that closes the cache can
deadlock. If your service needs a shutdown time limit, cancel and await its request tasks first,
from outside that chain, and then close the cache.

## Metric reference

Every metric name the cache emits:

```text
backend_failure
backend_latency_seconds
cache_bypass
cache_collapse
cache_commit
cache_dependency_broadening
cache_dependency_vector_bytes
cache_dependency_vector_tags
cache_fill
cache_hit
cache_incarnation
cache_invalidation
cache_lease_rejection
cache_lease_timeout
cache_metadata_recovery_seconds
cache_metadata_write_rejection
cache_miss
cache_query_classification
cache_schema_view
cache_scoped_generation
cache_serialized_bytes
cache_sg_avoided_calls
cache_sg_avoided_seconds
cache_site_flush
cache_ttl_selection
cache_url_age_cap
cache_vector_resolution
cache_wait_seconds
generation_metadata_bytes
```

`generation_metadata_bytes` reports L1 metadata only; use Valkey's own memory telemetry for the
metadata role.

The only label names are `backend_operation`, `classification`, `layer`, `method`, `outcome`,
`reason`, and `tag_kind`. Label values are nonempty ASCII of at most 64 characters. `layer` is
`local_l1`, `hierarchical_l1`, or `l2`. When a metrics correlation secret is configured,
`exact_call_digest` is also allowed; it holds only a keyed lowercase HMAC-SHA-256 digest.
Credentials, fields, filters, payloads, queries, and URLs are never labels.

The cache keeps a parsed copy of the SG schema in each process to analyze queries.
`cache_schema_view` reports how each use of that copy was satisfied:

| `outcome`     | Meaning                                                                     |
| ------------- | --------------------------------------------------------------------------- |
| `reuse`       | The in-process copy was still fresh.                                        |
| `revalidated` | The copy's L1 deadline was renewed against the same shared schema result.   |
| `rebuild`     | A schema result was fetched or read from the cache and parsed successfully. |
| `malformed`   | A schema result failed to parse.                                            |
| `refetch`     | After a parse failure, the schema was fetched directly from SG.             |

### Waiting for another fill

A *request* is one `read` or `read_parsed` call, including calls made by the client adapters. A
read started inside a fetch is a separate request.

Waits have two categories. `local` means waiting for another reader in the same cache instance to
finish the same result or parsed value. `distributed` means waiting because another holder has the
fill lease: L2 in shared and hierarchical layouts, or the local backend's lease in an async
local-only cache.

| Metric                | Emission and value                                                                                                                                                  | `outcome` label                                            |
| --------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------- |
| `cache_collapse`      | One increment per request per category, when its first wait starts.                                                                                                 | `local` or `distributed`.                                  |
| `cache_wait_seconds`  | One observation per request per category that waited, including on error or cancellation. The value is the total seconds across all wait episodes in that category. | How the last wait in that category ended, as listed below. |
| `cache_lease_timeout` | At most one increment per request per category whose wait budget expired.                                                                                           | `local` or `distributed`.                                  |

The duration metric has no category label. A request that waits in both categories emits two
observations, even if their outcomes match. Re-election can produce several wait episodes; collapse
counts and duration observations still occur only once per category.

A local wait starts when a reader joins an existing result or parsed-value flight. It ends when
that flight notifies the reader, the budget expires, or the wait is interrupted. A distributed wait
starts when a denied lease reservation leads to polling with budget remaining. It ends when polling
finds a cached result, acquires a lease, exhausts its budget, fails, or is cancelled. Wait time
excludes a subsequent takeover fill or direct SG fetch, and excludes promotion and parsed-memo work
after the wait.

| Last-wait outcome | Meaning                                                                                                                                                    |
| ----------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `completed`       | A local flight notified the reader, or distributed polling found a cached result. Notification does not guarantee that the reader can use a stored result. |
| `takeover`        | Distributed polling acquired the fill lease. Timing stops before the new holder rechecks or fills the cache.                                               |
| `timeout`         | The wait budget expired.                                                                                                                                   |
| `failure`         | The distributed cache path failed while waiting. The request may then fetch SG directly.                                                                   |
| `cancelled`       | The wait was cancelled or otherwise interrupted before another outcome was recorded.                                                                       |

`limits.lease_wait_seconds` supplies a fresh budget for each synchronous local wait episode and for
a distributed wait. Parsed-memo re-election or a later result flight can start another local
episode, so a request's accumulated time in a category can exceed one budget. Async reads keep one
deadline per read and can report a distributed timeout without any distributed wait. A timeout does
not prove that an SG call occurred or that nothing was stored.

Ordinary synchronous `read` calls in shared-only caches have no in-process result flight: each
thread finding a held lease waits as `distributed`. Synchronous `read_parsed` still has a local
`memo:` flight in every topology, including shared-only. A synchronous local-only result path does
not wait for a lease held elsewhere; it fetches without storing. Hierarchical result paths and
every async cache normally coordinate followers locally, while the elected reader may wait for the
deepest lease.
