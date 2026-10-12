# Configuration

This page helps you choose a cache layout, write the settings file that every cache process loads,
and connect the cache in Python. If you have not run the cache yet, start with the
[README quick start](../README.md#quick-start), which uses the local layout and needs no settings
file.

| If you want to...                               | Start here                                                |
| ----------------------------------------------- | --------------------------------------------------------- |
| Choose local, shared, or hierarchical           | [Choose a layout](#choose-a-layout)                       |
| Copy a starting settings file                   | [Layout examples](#layout-examples)                       |
| Connect to Sentinel or a managed Valkey service | [Valkey connection modes](#valkey-connection-modes)       |
| Build the cache from a settings file in Python  | [Build the cache in Python](#build-the-cache-in-python)   |
| Cache reads for human users                     | [Identities and trust](#identities-and-trust)             |
| Change how long results are kept                | [Maximum age rules](#maximum-age-rules)                   |
| Cache results that contain signed URLs          | [Signed URLs](#signed-urls)                               |
| Look up a setting and its default               | [Settings reference](#settings-reference)                 |
| Correlate metrics for one exact call            | [Metrics correlation secret](#metrics-correlation-secret) |

## Choose a layout

The cache has two layers. The in-process layer (L1) lives inside each process and is bounded by
bytes. The shared Valkey layer (L2) lives in Valkey, where every connected process can reuse its
results. Settings enable each layer in its own table, `[cache.l1]` and `[cache.l2]`.

| Layout       | L1  | L2  | Tradeoff                                                                                                                    |
| ------------ | --- | --- | --------------------------------------------------------------------------------------------------------------------------- |
| Local        | On  | Off | Easiest start. Each process has its own cache and sees changes made elsewhere only when results reach their maximum age.    |
| Shared       | Off | On  | Processes reuse each other's results, and SG events invalidate them. Requires separate result and metadata Valkey services. |
| Hierarchical | On  | On  | Shared results plus a short-lived copy in each process. A process can serve its copy briefly after another process writes.  |

The local layout needs no settings file: `CacheSettings()` selects it with the defaults. For any
layout, `CacheSettings.from_toml(path)` loads one TOML file. Every process that shares L2 must load
the same settings, including the same site namespace and Valkey key prefixes, because those
settings determine cache keys and invalidation.

Loading checks the whole file at startup. Every file must declare the configuration schema version,
`schema_version = 1`. Unknown keys, unsupported values, and unsafe combinations raise `ValueError`,
as does invalid TOML syntax; a value or table of the wrong type raises `TypeError`. The
[consistency guide](consistency.md) explains which reads are cached and how each layout bounds
staleness.

## Layout examples

### Local

Local mode is the default. It has one in-process L1 and no sharing between processes.

```toml
schema_version = 1

[cache]
site_namespace = "production"

[cache.l1]
enabled = true

[cache.l2]
enabled = false
```

### Shared

Shared mode disables L1 and enables L2. It requires two independent Valkey services: one for
results and one for metadata, such as dependency generations. The roles have different eviction and
durability needs; see [service layout](operations.md#service-layout). Passwords are read from the
named environment variables only when the clients are created, so password values never belong in
the file.

```toml
schema_version = 1

[cache]
site_namespace = "production"

[cache.l1]
enabled = false

[cache.l2]
enabled = true

[cache.l2.valkey.result]
mode = "standalone"
host = "cache-results.internal"
port = 6379
database = 0
tls = true
username = "sgqc-results"
password_env = "SG_QUERY_CACHE_RESULT_PASSWORD"
key_prefix = "sgqc-production"

[cache.l2.valkey.metadata]
mode = "standalone"
host = "cache-metadata.internal"
port = 6379
database = 0
tls = true
username = "sgqc-metadata"
password_env = "SG_QUERY_CACHE_METADATA_PASSWORD"
key_prefix = "sgqc-production"
```

### Hierarchical

Hierarchical mode enables both layers. L1 keeps a short-lived copy in each process; L2 remains the
shared authority, and an L1 copy never outlives the L2 result it came from.

```toml
schema_version = 1

[cache]
site_namespace = "production"

[cache.l1]
enabled = true
hierarchical_max_age_seconds = 10

[cache.l2]
enabled = true

[cache.l2.valkey.result]
mode = "managed"
host = "results-primary.internal"
password_env = "SG_QUERY_CACHE_RESULT_PASSWORD"
key_prefix = "sgqc-production"

[cache.l2.valkey.metadata]
mode = "managed"
host = "metadata-primary.internal"
password_env = "SG_QUERY_CACHE_METADATA_PASSWORD"
key_prefix = "sgqc-production"
```

## Valkey connection modes

Both roles must use the same `mode`:

- `standalone` connects directly to `host`.
- `managed` also connects directly to `host`, such as a provider's primary endpoint, and checks
  that the endpoint is a writable primary whenever it connects.
- `sentinel` finds the current primary through Valkey Sentinel. It requires `service_name` and at
  least three distinct `host:port` addresses in `sentinels`, because the client accepts a
  Sentinel's answer only when that Sentinel sees at least two others.

Standalone and managed roles require `host`. The role defaults are `port = 6379`, `database = 0`,
`tls = true`, and `key_prefix = "sgqc"`; `username` is optional. A Sentinel role looks like this:

```toml
[cache.l2.valkey.result]
mode = "sentinel"
sentinels = ["result-s1:26379", "result-s2:26379", "result-s3:26379"]
service_name = "sgqc-results"
password_env = "SG_QUERY_CACHE_RESULT_PASSWORD"

[cache.l2.valkey.metadata]
mode = "sentinel"
sentinels = ["metadata-s1:26379", "metadata-s2:26379", "metadata-s3:26379"]
service_name = "sgqc-metadata"
password_env = "SG_QUERY_CACHE_METADATA_PASSWORD"
```

Loading rejects result and metadata roles that point at the same endpoint: the same host, port, and
database, or for Sentinel, the same service name and database. Keep the roles' credentials, service
names, databases, and memory policies independent; in particular, the metadata role must never
evict keys, while the result role should. Supported providers and ACLs are covered in
[operations](operations.md#supported-deployments).

## Build the cache in Python

The `valkey` extra provides the synchronous `ValkeyBackend` in `sg_query_cache.valkey`. This
snippet builds a synchronous cache for whichever layers the settings file enables, then wraps a
client as in the [README quick start](../README.md#cache-a-synchronous-client):

```python
import os
from time import monotonic

from sg_query_cache import CacheSettings, Identity, LocalBackend, NullMetrics, QueryCache
from sg_query_cache.valkey import ValkeyBackend

settings = CacheSettings.from_toml('/etc/sg-query-cache.toml')
identity = Identity(kind='script', issuer='render-farm', namespace='publish-reader')
metrics = NullMetrics()
l1 = LocalBackend(settings, monotonic, metrics) if settings.l1.enabled else None
l2 = ValkeyBackend.from_settings(settings, os.environ, metrics) if settings.l2.enabled else None
cache = QueryCache(settings, identity, l1=l1, l2=l2, metrics=metrics)
```

See [identities and trust](#identities-and-trust) for choosing `Identity` values. Supply exactly
the backends the settings enable; any other combination raises `ValueError`.
`ValkeyBackend.from_settings` reads the role passwords from the mapping you pass, here the process
environment. `cache.close()` closes L1 and then L2. To collect metrics, replace `NullMetrics()`
with the boundary described in [metrics](operations.md#metrics-and-alerts).

For native async code, `AsyncCachedShotgun.open(raw_sg, settings, identity, context)` builds the
layers the settings enable, reading role passwords from the process environment unless you pass
`environ=`. Pass your sink as `metrics=`, and it builds the metrics boundary for you.

When L2 is enabled, the `ClientContext` site namespace must equal `cache.site_namespace`; the
wrapper raises `ValueError` otherwise. If you later change the raw client's `localized`,
`convert_datetimes_to_utc`, or page size, call the wrapper's `replace_context()` with a matching
context, awaiting it in async code, so results fetched under the old options are not reused. With
L2 enabled, a changed context also bumps the site's shared generation and invalidates cached
results for that site across all connected processes. Passing an equal context leaves the
generation unchanged.

## Identities and trust

An identity describes the effective SG permission scope, not a login credential. Use stable
configuration values for `issuer`, `namespace`, and `epoch`. Do not derive them from a rotating API
key, session token, or process ID: identity values are part of every cache key, so a changing value
would stop processes from sharing results.

```python
from sg_query_cache import Identity

script_identity = Identity(
    kind='script',
    issuer='render-farm',
    namespace='publish-reader',
    entity_id=42,
    epoch='permissions-v2',
)

human_identity = Identity(
    kind='human',
    issuer='desktop-login',
    namespace='artist-session',
    entity_id=123,
)
```

`kind` is `script` or `human`. A script identity may include its `ApiUser` ID as `entity_id`; a
human identity requires its `HumanUser` ID. Change `epoch` when a script's permissions change but
its other identity values stay the same, so results cached under the old permissions are not
reused.

Only script identities are cached by default. Human reads go directly to SG unless you set
`identity_policy = "scripts_and_humans"`. A local cache needs nothing more. A shared or
hierarchical cache also requires an explicit trust acknowledgement:

```toml
[cache]
identity_policy = "scripts_and_humans"

[cache.l2]
enabled = true
acknowledge_mutually_trusted_clients = true
```

The acknowledgement states that every process with access to L2 is trusted to declare the correct
identity and site namespace, because any of them can read and change cached data. It is not an
authentication boundary. If revoking a user's access must take effect immediately, bypass the cache
for that user's reads rather than waiting for an event or the maximum age.

## Maximum age rules

Each layer has a default maximum age:

| Layer and result type         | Setting                                 | Default       |
| ----------------------------- | --------------------------------------- | ------------- |
| L1 in the local layout        | `cache.l1.local_only_max_age_seconds`   | 30 seconds    |
| L1 in the hierarchical layout | `cache.l1.hierarchical_max_age_seconds` | 10 seconds    |
| L2, event-aware results       | `cache.l2.event_aware_max_age_seconds`  | 600 seconds   |
| L2, TTL-only results          | `cache.l2.ttl_only_max_age_seconds`     | 60 seconds    |
| L2, schema results            | `cache.l2.schema_max_age_seconds`       | 1,800 seconds |

Event-aware and TTL-only results are described in the
[consistency guide](consistency.md#how-the-cache-works). To override a default for some reads, add
`[[cache.ttl_rule]]` tables. Each rule matches exactly one of these shapes, listed from strongest
to weakest:

1. `method` plus `entity_type`
2. `method`
3. `entity_type`
4. `query_class`, which is `event_aware`, `ttl_only`, or `schema`
5. `identity_kind`, which is `script` or `human`

The strongest matching rule wins, regardless of the order of the tables. Loading rejects two rules
with the same match and rules that mix shapes. Every rule needs `maximum_age_seconds`.

```toml
[[cache.ttl_rule]]
method = "find_one"
entity_type = "Asset"
maximum_age_seconds = 15

[[cache.ttl_rule]]
query_class = "schema"
maximum_age_seconds = 300

[[cache.ttl_rule]]
identity_kind = "human"
maximum_age_seconds = 5
```

An explicit age from a [policy scope](consistency.md#bypass-and-policy-scopes) replaces the rule or
default, and can lengthen it as well as shorten it. Whatever the source, the selected age is then
capped by:

- `global_safety_max_seconds` (3,600 seconds by default, and never more);
- for hierarchical L1, `hierarchical_max_age_seconds`;
- for a result containing a signed URL, the URL's remaining lifetime, as described in
  [signed URLs](#signed-urls);
- for an L2 result copied into L1, the L2 result's remaining lifetime.

A bypass scope wins before any of this, and a cap that is already used up also prevents caching.

## Signed URLs

SG returns some file and thumbnail links as signed URLs that stop working when they expire. To
avoid serving an expired link, the cache stores a result containing a recognized signed URL only
when `limits.clock_error_bound_seconds` is set. Without it, the read still returns SG's result, but
the cache does not store it.

Set the bound to the largest difference you expect between this host's clock and the clock that
signed the URL. It must be zero or more and smaller than `signed_url_safety_margin_seconds`. The
result's maximum age is then the URL's remaining lifetime minus both the safety margin and the
clock error bound.

Results that contain SG's temporary placeholder thumbnail, shown while SG processes a thumbnail,
are kept for at most `transient_image_max_age_seconds`, 5 seconds by default.

## Settings reference

This section lists every setting in configuration schema version 1 with its default. An enabled L2
also needs both Valkey role tables, which have no defaults for their connection values; see the
[layout examples](#layout-examples). Durations are positive seconds, and byte budgets and counts
are positive integers, unless noted.

### Top level and `[cache]`

| Setting                            | Default          | Meaning                                                                                                                       |
| ---------------------------------- | ---------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| `schema_version`                   | Required         | Configuration schema version. Must be `1`.                                                                                    |
| `site_namespace`                   | `"production"`   | Name of the SG site. Shared processes and the event-daemon site must use the same value. Nonempty ASCII without outer spaces. |
| `identity_policy`                  | `"scripts_only"` | `scripts_only` or `scripts_and_humans`. See [identities and trust](#identities-and-trust).                                    |
| `global_safety_max_seconds`        | `3600`           | Upper limit on every maximum age. At most 3,600.                                                                              |
| `enabled_ttl_only_methods`         | `[]`             | Extra read methods to cache as TTL-only. The only accepted value is `"work_schedule_read"`, whose response carries no URLs.   |
| `enable_additional_filter_presets` | `false`          | When `true`, `find` and `find_one` calls with `additional_filter_presets` are cached as TTL-only instead of read directly.    |
| `metrics_digest_secret_env`        | Unset            | Name of the environment variable holding the [metrics correlation secret](#metrics-correlation-secret).                       |

The other composite read methods, such as `summarize` and `text_search`, are always read directly,
because the cache cannot rule out signed URLs in their responses. `EventLogEntry` reads are always
read directly.

### `[[cache.ttl_rule]]`

Each rule needs `maximum_age_seconds`, a positive age of at most 3,600 seconds, and exactly one of
the match shapes in [maximum age rules](#maximum-age-rules), built from these keys:

| Setting         | Matches                                    |
| --------------- | ------------------------------------------ |
| `method`        | An SG read method name, such as `find_one` |
| `entity_type`   | An SG entity type, such as `Asset`         |
| `query_class`   | `event_aware`, `ttl_only`, or `schema`     |
| `identity_kind` | `script` or `human`                        |

### `[cache.l1]`

| Setting                        | Default               | Meaning                                                                                              |
| ------------------------------ | --------------------- | ---------------------------------------------------------------------------------------------------- |
| `enabled`                      | `true`                | Use the in-process layer. At least one layer must be enabled.                                        |
| `local_only_max_age_seconds`   | `30`                  | Default maximum age when L1 is the only layer.                                                       |
| `hierarchical_max_age_seconds` | `10`                  | Default maximum age, and cap, for L1 copies in the hierarchical layout.                              |
| `result_budget_bytes`          | `134217728` (128 MiB) | Accounted result bytes per local backend. When full, the least recently used results are evicted.    |
| `metadata_budget_bytes`        | `16777216` (16 MiB)   | Accounted bytes for generation records and fill leases. When full, operations that need more bypass. |

L1 budgets count the cache's own accounting of stored bytes, not exact process memory.

### `[cache.l2]`

| Setting                                | Default              | Meaning                                                                                            |
| -------------------------------------- | -------------------- | -------------------------------------------------------------------------------------------------- |
| `enabled`                              | `false`              | Use the shared Valkey layer. Requires both role tables.                                            |
| `backend`                              | `"valkey"`           | The only accepted value.                                                                           |
| `event_aware_max_age_seconds`          | `600`                | Default maximum age for event-aware results.                                                       |
| `ttl_only_max_age_seconds`             | `60`                 | Default maximum age for TTL-only results.                                                          |
| `schema_max_age_seconds`               | `1800`               | Default maximum age for schema results.                                                            |
| `result_budget_bytes`                  | `8589934592` (8 GiB) | Sizing target for the result service. Not enforced by the client.                                  |
| `metadata_budget_bytes`                | `1073741824` (1 GiB) | Sizing target for the metadata service. Not enforced by the client.                                |
| `acknowledge_mutually_trusted_clients` | `false`              | Must be `true` to cache human identities in L2. See [identities and trust](#identities-and-trust). |

The two L2 budgets are targets for sizing and monitoring the Valkey services; see
[metadata capacity](operations.md#metadata-capacity). The client does enforce the per-result and
dependency limits in `[cache.limits]`.

### `[cache.l2.valkey.result]` and `[cache.l2.valkey.metadata]`

| Setting        | Default        | Meaning                                                                                                                                            |
| -------------- | -------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| `mode`         | `"standalone"` | `standalone`, `sentinel`, or `managed`. Both roles must match.                                                                                     |
| `host`         | Required       | Host name or IP address. Required for standalone and managed roles.                                                                                |
| `port`         | `6379`         | TCP port.                                                                                                                                          |
| `database`     | `0`            | Database number, zero or more.                                                                                                                     |
| `tls`          | `true`         | Connect with TLS and verify the host name.                                                                                                         |
| `username`     | Unset          | ACL user name.                                                                                                                                     |
| `password_env` | Required       | Name of the environment variable that holds the password, not the password itself. The variable must be set and nonempty when clients are created. |
| `key_prefix`   | `"sgqc"`       | Prefix for every key, up to 64 ASCII characters without `*?[]\|\/`. All clients sharing data must match.                                           |
| `sentinels`    | Unset          | Sentinel mode only: at least three distinct `host:port` addresses.                                                                                 |
| `service_name` | Unset          | Sentinel mode only: the monitored primary's name.                                                                                                  |

### `[cache.limits]`

| Setting                               | Default           | Meaning                                                                                                                                                               |
| ------------------------------------- | ----------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `fill_lease_seconds`                  | `30`              | How long one reader's reservation to fetch a missing result lasts. If that reader fails, another can fetch once the lease expires.                                    |
| `lease_wait_seconds`                  | `2`               | How long other readers of the same missing result wait for that fetch before reading SG themselves without storing the result.                                        |
| `max_dependency_tags`                 | `32`              | Most dependency tags one result can record. At most 32 when L2 is enabled. Larger lists are broadened, cached as TTL-only, or read directly.                          |
| `max_dependency_vector_bytes`         | `2048`            | Most encoded bytes for one result's dependency list.                                                                                                                  |
| `max_serialized_result_bytes`         | `1048576` (1 MiB) | Largest result the cache stores; larger results are returned but not cached. At most 1,048,576.                                                                       |
| `scoped_generation_retention_seconds` | `7200`            | How long an unused generation record for a specific entity, entity type, field, or project stays in metadata. Site-wide, schema, and permission records never expire. |
| `connect_timeout_seconds`             | `0.1`             | Valkey connection timeout.                                                                                                                                            |
| `operation_timeout_seconds`           | `0.25`            | Valkey operation timeout. Synchronous and async clients apply it differently; see [async deadlines](operations.md#async-deadlines-and-shutdown).                      |
| `circuit_failure_threshold`           | `5`               | Consecutive failures that open a role's circuit, after which reads bypass to SG.                                                                                      |
| `circuit_probe_seconds`               | `10`              | Wait before one probe tries an open circuit again.                                                                                                                    |
| `incarnation_validation_seconds`      | `30`              | Approximate interval, with jitter, at which a client rechecks the metadata incarnation.                                                                               |
| `signed_url_safety_margin_seconds`    | `30`              | Time subtracted from a signed URL's remaining lifetime. See [signed URLs](#signed-urls).                                                                              |
| `transient_image_max_age_seconds`     | `5`               | Maximum age of results containing SG's temporary placeholder thumbnail.                                                                                               |
| `clock_error_bound_seconds`           | Unset             | Largest expected clock error. Required to cache results with signed URLs; zero or more and smaller than the safety margin.                                            |

The incarnation identifies the current contents of the metadata service; see
[service layout](operations.md#service-layout). A local-only configuration may raise
`max_dependency_tags` above 32 when its byte limits allow; with L2 enabled, the cache never sends
more than 32 tags to Valkey.

## Metrics correlation secret

By default, metrics carry no per-call identifier. To correlate the metrics emitted during one exact
read, set `metrics_digest_secret_env` to the name of an environment variable that holds a secret of
at least 32 bytes. Loading the settings records only the variable name; the secret is read when you
build the metrics boundary from the settings, the environment, and your sink:

```python
import os

from sg_query_cache import CacheSettings, MetricsSink, QueryCache


def cache_metrics(settings: CacheSettings, sink: MetricsSink) -> MetricsSink:
    return QueryCache.build_metrics(settings, os.environ, sink)
```

The secret must be valid UTF-8 and at least 32 bytes when encoded. If it is missing, not a string,
or too short, construction fails with one error that does not reveal the value. The returned
boundary adds an `exact_call_digest` label, a keyed HMAC-SHA-256 digest, to `QueryCache` read
metrics. Backend metrics never carry this label, including those emitted during a read. The
boundary keeps neither the environment nor the variable name, and never emits the secret or the raw
query. The secret does not affect cache keys or cached results.

Without `metrics_digest_secret_env`, `QueryCache.build_metrics` still validates and forwards your
sink's metrics, without `exact_call_digest`. Passing `NullMetrics()` as the sink discards all
metrics.
