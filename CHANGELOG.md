# Changelog

## 0.2.0 (2026-11-08)

### Breaking changes

- Pass every `ClientContext` field by keyword. Positional arguments now raise `TypeError`.
- Pass `key_builder`, `l1`, `l2`, and `metrics` to `QueryCache` and `AsyncQueryCache` by keyword.
  Only `settings` and `identity` remain positional.

### Added

- `CachedShotgun.open()` builds the cache layers your settings enable and closes them on exit, like
  `AsyncCachedShotgun.open()`.
- `cache_policy()` accepts `maximum_age_seconds`, the name the settings file uses.
- `backend_failure` metrics carry a `failure_class` label, such as `authentication`, `tls`, or
  `timeout`.
- Node-based, cluster-mode-disabled Multi-AZ ElastiCache for Valkey is supported with synchronous
  and async clients.

### Changed

- Reads that select a calculated field, such as `Task.workload`, are cached as TTL-only results
  instead of always going to SG.
- `find` and `find_one` calls that filter on many exact IDs no longer stall the async event loop:
  10,000 IDs take about 1 ms in the cache instead of 50 ms.
- Creating a synchronous `ValkeyBackend` or `ValkeyMetadataStore` no longer connects to Valkey, so
  your application starts while Valkey is unreachable and reads fall back to SG.
- `flush` uses `cache.site_namespace` from your settings when you omit `--site`, and refuses a
  `--site` that names a different site.
- Setup errors name their cause, such as the settings field at fault, an unset password variable,
  or the missing `valkey` extra.

### Fixed

- Results that include a query field, such as `open_notes_count`, are cached as TTL-only results,
  so a changed count no longer stays cached for the full event-aware age.
- When an entity is deleted in SG, the event-daemon plugin invalidates only the results that depend
  on it, not every shared result for the site.
- An async cache stops waiting on Valkey when Valkey accepts connections but stops replying,
  instead of waiting the full operation timeout on every read.

## 0.1.0 (2026-10-11)

Initial release.
