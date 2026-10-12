# Benchmarks

The benchmark drivers estimate how much the cache could save for a workload like yours and what it
costs per read. Repeated exact calls, the number of identities, payload size, concurrency, and SG
latency all affect the savings. The drivers report synthetic measurements: they do not certify
throughput for a production site or establish compatibility with a managed Valkey provider. To try
the cache itself, start with the [README quick start](../README.md#quick-start).

The drivers live in the repository's `benchmarks/` directory and are not installed with the
package. Run them from the root of a checkout, with a Python 3.11 or newer environment that has the
project's dependencies; the development environment from `make setup` works, through
`.venv/bin/python`. The shared and hierarchical profiles and the `failure-recovery` scenario also
need the `valkey` extra. In the commands below, `python` is that environment's interpreter.

## Run without external services

The synchronous driver simulates SG with a fixed latency and runs a bounded workload against a
local cache:

```sh
python benchmarks/query_cache.py --profile local --scenario hot-key-burst --json
```

It prints one JSON report. Start with these fields:

- `sg_calls_avoided` and `cache_operations.hits`: how many reads the cache answered.
- `complete_hit_latency_ms`: percentiles for a full cache hit.
- `sg_latency_model`: the simulated SG latency the savings are measured against.
- `correctness_violations`: stale, malformed, or over-age results observed. It should be 0; the
  driver exits 1 otherwise, and 2 for invalid options or setup errors.

`--profile` also accepts `shared` and `hierarchical`, which need Valkey services as described
below. Other scenarios cover identity cardinality, one-hit scans, dependency broadening, event
storms, metadata churn, mixed values, failure recovery, fanout, and schema reads. Some scenarios
support only some profiles, and the driver rejects combinations that need controlled local capacity
or failure behavior. `--operations` accepts up to 10,000 and `--workers` up to 64. Run
`python benchmarks/query_cache.py --help` for the scenario names and options.

Each report also includes a separately measured warmed hit, `normal_hit_backend_phases`, which
counts the backend steps in one normal hit. A normal shared hit resolves metadata and then looks up
the result. This probe does not count toward the scenario's throughput.

The native async driver measures local store maintenance, decoding cost, complete wrapper hits, and
reads around writes:

```sh
python benchmarks/async_query_cache.py --scenario local-maintenance --population 50000 --iterations 10000 --json
python benchmarks/async_query_cache.py --scenario decode --iterations 1000 --json
python benchmarks/async_query_cache.py --scenario wrapper-hit --iterations 1000 --json
python benchmarks/async_query_cache.py --scenario mixed-read-write --iterations 1000 --json
```

Async reports separate operation latency, event-loop scheduling gaps, and accounted local storage.
They use controlled stand-ins for the native async SG client and for fences. The `decode` scenario
includes representative and size-limit payloads, and `mixed-read-write` distinguishes fences that
are still pending from failed fences whose changes are retained. Timings vary with payload shape,
scheduling, and host load. Local maintenance visits at most 64 entries per queue, or 192 for a
fully populated operation; that is a bound on work, not on milliseconds.

## Measure with disposable Valkey services

The `shared` and `hierarchical` profiles need two separate, disposable Valkey services, one for
results and one for metadata. The driver runs them only when `SG_QUERY_CACHE_REQUIRE_INTEGRATION=1`
is set, as your confirmation that the services are disposable. For both `RESULT` and `METADATA`,
set `SG_QUERY_CACHE_TEST_<ROLE>_HOST`, `_PORT`, `_DATABASE`, `_TLS`, and `_PASSWORD`; `_USERNAME`
is optional. `_TLS` accepts `true` or `false` in any letter case. For example:

```sh
export SG_QUERY_CACHE_REQUIRE_INTEGRATION=1
export SG_QUERY_CACHE_TEST_RESULT_HOST='<disposable-result-host>'
export SG_QUERY_CACHE_TEST_RESULT_PORT='6379'
export SG_QUERY_CACHE_TEST_RESULT_DATABASE='0'
export SG_QUERY_CACHE_TEST_RESULT_TLS='true'
export SG_QUERY_CACHE_TEST_RESULT_PASSWORD='<disposable-result-password>'
export SG_QUERY_CACHE_TEST_METADATA_HOST='<disposable-metadata-host>'
export SG_QUERY_CACHE_TEST_METADATA_PORT='6379'
export SG_QUERY_CACHE_TEST_METADATA_DATABASE='0'
export SG_QUERY_CACHE_TEST_METADATA_TLS='true'
export SG_QUERY_CACHE_TEST_METADATA_PASSWORD='<disposable-metadata-password>'
python benchmarks/query_cache.py --profile shared --scenario hot-key-burst --json
```

Never point benchmarks at production services. Each shared or hierarchical run writes under its own
key prefix, built from the scenario name and a fresh random token, so runs do not reuse each
other's data. The driver cannot measure the services' total memory use; use the services' own
telemetry for capacity decisions.

These runs connect in standalone mode only, so they do not establish Sentinel failover or
managed-provider behavior; see [supported deployments](operations.md#supported-deployments).
