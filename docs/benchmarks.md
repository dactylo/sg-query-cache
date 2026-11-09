# Benchmarks

Use the benchmark drivers to estimate cache savings for a workload like yours. Repeated exact
calls, identity count, payload size, concurrency, and SG latency all affect the result. The drivers
use synthetic workloads; compare their assumptions with your application before sizing a
deployment. To try the cache itself, start with the [README quick start](../README.md#quick-start).

The drivers live in `benchmarks/` and are not installed with the package. Run them from the root of
a development checkout with Python 3.11 or newer and the project's dependencies. The environment
created by `make setup` works: use `.venv/bin/python` as `python` in the examples below. Shared and
hierarchical profiles, and the `failure-recovery` scenario, also require the `valkey` extra.

## Run without external services

The synchronous driver simulates a fixed SG latency and runs a bounded workload against a local
cache:

```sh
python benchmarks/query_cache.py --profile local --scenario hot-key-burst --json
```

It prints one compact JSON report. Check `sg_calls_avoided` for savings and
`correctness_violations`, which should be 0. See [Read the report](#read-the-report) for the fields
and exit statuses.

The native async driver covers local maintenance, decoding, complete wrapper hits, and reads around
writes:

```sh
python benchmarks/async_query_cache.py --scenario local-maintenance --population 50000 --iterations 10000
python benchmarks/async_query_cache.py --scenario decode --iterations 1000 --json
python benchmarks/async_query_cache.py --scenario wrapper-hit --iterations 1000 --json
python benchmarks/async_query_cache.py --scenario mixed-read-write --iterations 1000 --json
```

It prints indented JSON by default; `--json` selects compact JSON.

## Read the report

### Synchronous reports

| Field                       | Meaning                                                                               |
| --------------------------- | ------------------------------------------------------------------------------------- |
| `sg_calls_avoided`          | Reads answered without an SG call                                                     |
| `cache_operations.hits`     | Cache hits                                                                            |
| `complete_hit_latency_ms`   | Latency percentiles for a complete cache hit                                          |
| `sg_latency_model`          | Simulated SG latency used to calculate savings                                        |
| `correctness_violations`    | Stale, malformed, or over-age results; should be 0                                    |
| `normal_hit_backend_phases` | Backend steps for a separately measured warmed hit, excluded from scenario throughput |

A normal shared hit resolves metadata and then looks up the result. The driver requires `--json`;
it exits 1 for correctness violations and 2 for invalid options or setup errors.

`--profile` accepts `local`, `shared`, or `hierarchical`. Scenarios cover hot-key bursts, identity
cardinality, one-hit scans, dependency broadening, event storms, metadata churn, mixed values,
failure recovery, fanout, and schema reads. Some require controlled local capacity or failure
behavior, so the driver rejects incompatible profile/scenario combinations. `--operations` accepts
up to 10,000 and `--workers` up to 64. Run `python benchmarks/query_cache.py --help` for the
complete scenario names and options.

### Async reports

Async reports separate operation latency, event-loop scheduling gaps, and accounted local storage.
They use controlled stand-ins for the native async SG client and fences.

| Scenario            | What to compare                                                                 |
| ------------------- | ------------------------------------------------------------------------------- |
| `local-maintenance` | Maintenance latency, scheduling gaps, and accounted storage as population grows |
| `decode`            | Decoding cost for representative and size-limit payloads                        |
| `wrapper-hit`       | Complete wrapper-hit cost                                                       |
| `mixed-read-write`  | Reads with pending fences versus failed fences whose changes are retained       |

Timings depend on payload shape, scheduling, and host load. Local maintenance visits at most 64
entries per queue, or 192 for a fully populated operation. That bounds work, not milliseconds.

## Measure with disposable Valkey services

Use two separate, disposable Valkey services: one for results and one for metadata. Never point
benchmarks at production services. Setting `SG_QUERY_CACHE_REQUIRE_INTEGRATION=1` confirms that the
services are disposable and enables the `shared` and `hierarchical` profiles.

For both roles, set these variables; replace `<ROLE>` with `RESULT` or `METADATA`:

| Variable                              | Value                                 |
| ------------------------------------- | ------------------------------------- |
| `SG_QUERY_CACHE_TEST_<ROLE>_HOST`     | Service host                          |
| `SG_QUERY_CACHE_TEST_<ROLE>_PORT`     | TCP port                              |
| `SG_QUERY_CACHE_TEST_<ROLE>_DATABASE` | Database number                       |
| `SG_QUERY_CACHE_TEST_<ROLE>_TLS`      | `true` or `false`, in any letter case |
| `SG_QUERY_CACHE_TEST_<ROLE>_PASSWORD` | Role password                         |
| `SG_QUERY_CACHE_TEST_<ROLE>_USERNAME` | Optional ACL username                 |

For example:

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

The driver connects in standalone mode. Each run uses a separate key prefix built from the scenario
name and a fresh random token, so runs do not reuse data. The driver does not measure total service
memory; use Valkey telemetry for capacity decisions. See
[supported deployments](operations.md#supported-deployments) for production topology requirements.

## Connection setup measurements

The following TLS connection setup measurements use an ElastiCache `cache.t4g.micro` node:

| Concurrent connections | Timeout setting  | Result                                 |
| ---------------------- | ---------------- | -------------------------------------- |
| 1                      | Long timeouts    | About 15 ms to connect                 |
| 50                     | Long timeouts    | 0.25 to 0.4 seconds total              |
| 50 cache clients       | Default timeouts | One client reported Valkey unavailable |

These figures illustrate connection bursts, not steady-state read latency. Reuse backends and
stagger startup as described in [connection recovery](operations.md#connection-recovery).
