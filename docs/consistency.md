# Consistency

This page explains how fresh cached results are, what the cache keeps apart, and when to read SG
directly. Read it before you decide which reads to cache or choose a shared layout. To try the
cache first, start with the [README quick start](../README.md#quick-start).

| If you want to...                                 | Start here                                                      |
| ------------------------------------------------- | --------------------------------------------------------------- |
| Know what can be stale, and for how long          | [The guarantee in brief](#the-guarantee-in-brief)               |
| Check which methods are cached                    | [Which reads are cached](#which-reads-are-cached)               |
| Force a direct read or change the age for a block | [Bypass and policy scopes](#bypass-and-policy-scopes)           |
| Understand how writes invalidate results          | [Writes](#writes)                                               |
| Compare the layouts                               | [Layout differences](#layout-differences)                       |
| Handle permission changes and revocation          | [Permissions and authorization](#permissions-and-authorization) |
| Look up a cache concept                           | [How the cache works](#how-the-cache-works)                     |
| Check exact age, fence, and isolation rules       | [Detailed contract](#detailed-contract)                         |

## The guarantee in brief

sg-query-cache is a bounded-staleness cache. It reduces duplicate SG requests; it is not a
transaction log, an authorization service, or a way to make reads causally consistent with writes.

- **Every result has a maximum age.** The cache never serves a result older than its maximum age.
  By default that is 30 seconds in the local layout, 10 seconds for the per-process copy in the
  hierarchical layout, and, in Valkey, 600 seconds for event-invalidated results, 60 seconds for
  [TTL-only results](#how-the-cache-works), and 1,800 seconds for schema. You can change these with
  [maximum age rules](configuration.md#maximum-age-rules).
- **Within that age, a result can be stale.** It can miss any change the cache has not been told
  about. In the local layout, that is any change made outside this process. In the shared layouts,
  it is a change whose SG event the event-daemon plugin has not processed yet.
- **Your own writes invalidate.** A write through the same wrapper invalidates the results it can
  affect. If a layer cannot acknowledge the invalidation, that wrapper bypasses it until recovery;
  see [fence acknowledgement](#fence-acknowledgement). Ordinary writes may leave TTL-only results
  cached, so bypass when a read after a write must see the write.
- **Results are isolated.** Calls that differ in method, arguments, response options, site, or
  identity never share a result.
- **Failures fall through to SG.** If the cache is unavailable or its data is unusable, the read
  goes to SG. A cache failure never turns into a stale answer.
- **Bypass when it matters.** Read SG directly when a result must reflect SG right now: an
  authorization decision, a check right after another process writes, or anything where even
  bounded staleness is unacceptable.

The cache does not promise read-your-writes across different clients, causal consistency, or
exactly-once event delivery.

## Which reads are cached

| Method                                                                                     | Treatment                                                                     |
| ------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------- |
| `find`, `find_one`                                                                         | Event-aware when dependencies are clear; otherwise TTL-only, or read directly |
| `schema_read`, `schema_entity_read`, `schema_field_read`                                   | Cached as schema; invalidated by schema writes and schema events              |
| `work_schedule_read`                                                                       | TTL-only if listed in `enabled_ttl_only_methods`; otherwise read directly     |
| `activity_stream_read`, `note_thread_read`, `preferences_read`, `summarize`, `text_search` | Always read directly                                                          |
| Any read of `EventLogEntry`                                                                | Always read directly                                                          |
| Supported writes, such as `create`, `update`, `delete`, `revive`, `batch`, and uploads     | Sent to SG between fences                                                     |
| Anything else                                                                              | Call it on `raw_client`; those calls are neither cached nor fenced            |

A `find` or `find_one` result is cached as TTL-only when its dependencies are uncertain: an unknown
filter or order field, a relative-date filter, `additional_filter_presets` (only when
`enable_additional_filter_presets` is on), or a dependency list that is still too long after
broadening. Broadening replaces many specific tags with fewer, broader ones, such as one
entity-type tag in place of many entity tags; a broadened result is invalidated more often.

A read goes directly to SG, without reading or storing a cached result, when caching it is unsafe
or unsupported: a human identity under `identity_policy = "scripts_only"`, a malformed query,
arguments the cache cannot normalize, a dependency list that does not fit even as TTL-only, or a
selected field, including `id`, whose schema or type the cache does not know. While the schema read
is failing, that last case covers every `find` and `find_one`. A result that is too large or too
deeply nested to store, such as one over `max_serialized_result_bytes`, is returned to the caller
but not cached.

## Bypass and policy scopes

Use `bypass()` when correctness requires a direct read:

```python
with cached_sg.bypass():
    asset = cached_sg.find_one('Asset', [['id', 'is', 123]], ['code'])
```

Inside the block, reads go to SG and neither read nor store cached results. Scopes apply to the
current thread in synchronous code and to the current task in async code; a task created inside the
scope inherits it.

`cache_policy()` sets the maximum age or the caching mode for a block of reads:

```python
with cached_sg.cache_policy(maximum_age_seconds=5):
    shots = cached_sg.find('Shot', [['sg_status_list', 'is', 'ip']], ['code'])
```

Inside this block, a read accepts only a result at most 5 seconds old, and a result stored from it
uses 5 seconds as its maximum age. When scopes nest, the smallest explicit age applies. An explicit
age replaces the configured default or rule, so it can lengthen a result's lifetime as well as
shorten it; the [safety caps](configuration.md#maximum-age-rules) still apply.

Use `mode` to make a read's caching policy more conservative:

| Mode            | Effect                                                                         |
| --------------- | ------------------------------------------------------------------------------ |
| `'event_aware'` | Keeps event-aware reads event-aware; does not promote TTL-only or direct reads |
| `'ttl_only'`    | Stores otherwise event-aware reads as TTL-only; direct reads still bypass      |
| `'bypass'`      | Reads SG directly, just like `bypass()`                                        |

A scope applies its mode only if it is more conservative than the read's existing policy. Schema
reads ignore every mode except bypass.

## Writes

Supported writes through `CachedShotgun` or `AsyncCachedShotgun` are fenced twice: before the
wrapper sends them to SG, and after they succeed or end with an unknown outcome. A `create`,
`update`, or `batch` whose data sets `image` or `filmstrip_image` also uploads a file, so the
wrapper fences the whole site for it, after any failure as well, because part of the change may
have been applied. In async code, cancelling a write after it was sent counts as an unknown
outcome.

Pass your client's transport exception classes as `indeterminate_write_exceptions` when you create
a wrapper, or to either wrapper's `open()` factory. A write that fails with one of them is treated
as an unknown outcome and fenced afterward; other failures are not.

Fences cover only writes made through the wrapper. Writes made through `raw_client`, another
process, the web interface, or the REST API reach the cache only through SG events in the shared
layouts, and otherwise only through the maximum age.

TTL-only results carry no entity or field tags, so an ordinary write may not invalidate them, even
when the same wrapper reads right after writing. Their maximum age bounds the staleness. Use bypass
when a read after a write must see the write.

## Layout differences

| Layout       | Where results live                   | Changes made outside the wrapper                                                                          |
| ------------ | ------------------------------------ | --------------------------------------------------------------------------------------------------------- |
| Local        | This process's L1                    | No event invalidation; the maximum age bounds staleness                                                   |
| Shared       | Valkey L2                            | Writes through other wrappers fence L2; the event-daemon plugin applies other SG changes                  |
| Hierarchical | Valkey L2 with a short-lived L1 copy | L2 invalidation does not purge another wrapper's L1; that copy can remain for up to 10 seconds by default |

An L1 copy never outlives the L2 result it came from. Shared layouts depend on every client
declaring its identity correctly and on prompt event processing. Synchronous and async wrappers can
share L2 when settings, identity, and response options match; see
[interoperability](#interoperability). Each wrapper's L1 remains subject to its own maximum age.

## Permissions and authorization

Permission and visibility changes are dependencies too, so SG events about them invalidate the
affected results. Until that event is processed or the result reaches its maximum age, the same
identity can still see results it no longer has permission to read. If revoking access must take
effect immediately, bypass the cache for the affected reads, or do not cache human identities. When
you change a script's permissions without changing its identity, set a new `epoch` on its
`Identity` or [flush the site](operations.md#flush-or-rotate-cached-results).

The cache does not authenticate identities. In the shared layouts, every client with access to
Valkey is trusted to declare its identity, site namespace, and key prefixes correctly. See
[identities and trust](configuration.md#identities-and-trust).

## How the cache works

These terms appear throughout the documentation.

**Layers.** The in-process layer (L1) is a cache inside each process, bounded by bytes; when it is
full, it evicts the least recently used results. The shared Valkey layer (L2) holds results that
every connected process can reuse. The local layout uses only L1, the shared layout only L2, and
the hierarchical layout keeps a short-lived L1 copy of L2 results in each process. Configuration
and metrics use the names `l1` and `l2`. "The shared layouts" means the shared and hierarchical
layouts together.

**Exact calls.** The cache key covers the method, its normalized arguments, the response options in
`ClientContext`, the site, and the identity. Only identical calls share a result.

**Identity.** An `Identity` names the SG permission scope whose results may be reused: a script or
a human user, plus stable names you choose. The caller declares it. The cache trusts that
declaration and does not authenticate it.

**Dependency tags and generations.** When the cache stores a result, it records what the result
depends on, such as the site, an entity type, a specific entity, a field, or the schema. Each of
these is a dependency tag. Each tag has a generation, a version number kept in the cache's
metadata. A change increments the generation of every affected tag, and a result recorded under an
older generation is no longer served.

**Event-aware and TTL-only results.** Most `find` and `find_one` results are event-aware: they
carry entity and field tags, so writes and SG events invalidate them precisely. When the cache
cannot determine a read's dependencies exactly, it may cache the result as TTL-only. A TTL-only
result depends only on the site and on permissions, so its maximum age is its main freshness bound.

**Fences.** Before the wrapper sends a write to SG, and again after it completes, it increments the
generations the write can affect. This invalidation is called a fence.

**Events.** In the shared layouts, the SG event-daemon plugin applies generation changes for each
SG event, so changes made by other clients, the web interface, or REST also invalidate results. An
event the plugin cannot map exactly invalidates the whole site.

**Site namespace.** `cache.site_namespace` names one SG site. Every process that shares Valkey, and
the event daemon, must use the same name. Flushes apply to one site namespace.

## Detailed contract

### Maximum age selection

Each read selects a maximum age from its configuration and active policy scopes, then applies the
[safety caps](configuration.md#maximum-age-rules). A hit older than the selected age is discarded
and fetched from SG.

Events usually make a change visible before the maximum age, but they are not a causal cursor:
processing one event does not prove that every earlier or related write has been applied.
Event-aware freshness also assumes every script that writes to SG generates events. A script that
genuinely only reads can disable event generation safely.

### Fence acknowledgement

The wrapper fences L2 before L1. If a layer does not acknowledge a fence, the write is still sent,
and that layer serves no hits to this wrapper until the cache replays the unacknowledged changes
and the layer acknowledges them. The cache replays them only during its next fence, which in
practice means the next write through the same wrapper; a read-only process bypasses that layer
until it writes or restarts.

In async code, a fence still in progress does not by itself make reads bypass. A layer that fails a
fence keeps the failed changes immediately, before the next layer is tried. If cancellation or a
failure stops the fence before a layer is reached, that layer also keeps the changes before the
operation exits. Reads then bypass each layer holding changes until a later fence succeeds there.
Layers whose fences are still pending can serve hits under their existing generations and age
limits.

### Cancellation

Once a read observes cancellation, it does not store its result. If a backend has already stored a
result but its reply is lost or cancelled, that valid result can remain. A fill lease, the short
reservation one reader holds while it fetches a missing result, expires on its own if its release
is uncertain. Cancellation does not roll back stored results.

### Value isolation

Each hit returns a new copy of the stored value, so callers cannot modify each other's results. The
copy preserves values, not object classes. An aware `datetime` from a hit carries a fixed-offset
timezone equal to the original offset, not the client's own `tzinfo` class.

### Interoperability

Synchronous and async wrappers build the same keys, dependency tags, and stored values for equal
inputs, so they can share one L2. This does not make different identities or response options
equivalent.

Clients that call the SG REST API directly are outside the Python wrappers. They neither read
cached results nor fence their writes; provide their own caching or read SG directly.
