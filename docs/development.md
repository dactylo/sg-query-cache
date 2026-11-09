# Development

Use this guide to set up a development checkout, run checks, and update dependencies or the local
Valkey stubs. The [README](../README.md) covers installing and using the library.

## Set up a development checkout

Install these tools before setup:

| Tool    | Required version or configuration                            |
| ------- | ------------------------------------------------------------ |
| Python  | 3.11 or newer, available as `python` on `PATH`               |
| uv      | 0.13.6; setup and lock targets reject other versions         |
| Taplo   | 0.10.0, for TOML blocks in Markdown                          |
| shfmt   | 3.14.1, for shell blocks in Markdown                         |
| Biome   | 2.6.6, for JSON formatting                                   |
| Node.js | 26.12.0, used by Pyright and installed by the CI quality job |

From the repository root, run:

```sh
make setup
make check
```

`make setup` creates `.venv`, installs the hashed pins in `requirements-ci.txt`, and installs the
project in editable mode. To select an interpreter, use `make setup PYTHON=/path/to/python`; setup
never downloads Python.

Run setup again after changing `pyproject.toml` or a lock file, including when `make format`
rewrites `pyproject.toml`. Other targets compare the environment with those files and stop if it is
missing or outdated. Checks never install or update packages.

## Run checks

Run focused tests while editing; use the broader targets when you need the complete checks:

| Command                                       | Purpose                                                 |
| --------------------------------------------- | ------------------------------------------------------- |
| `make test TEST_FLAGS='tests/test_config.py'` | Run one test file                                       |
| `make test`                                   | Run the suite serially                                  |
| `make quality`                                | Run CI's read-only quality checks                       |
| `make check`                                  | Run quality checks and the test suite                   |
| `make format`                                 | Apply formatters; inspect the changes before committing |
| `make benchmark`                              | Run the local hot-key [benchmark](benchmarks.md)        |

Quality checks cover formatting, lint, dependency locks, Pyright for Linux, macOS, and Windows,
trailing whitespace, and an offline zizmor audit of GitHub Actions workflows. The `slow` marker
covers `tests/test_scale.py`. Live checks need separate configuration; see
[Live Valkey integration tests](#live-valkey-integration-tests).

For the same parallel scheduling as CI:

```sh
make test TEST_FLAGS='-n auto --dist worksteal --durations=15'
```

Set `PYTEST_XDIST_AUTO_NUM_WORKERS` when visible CPU count exceeds a runner's CPU quota.

On Python 3.13 or newer, the development lock includes defsort. Inspect definition ordering without
changing files using the project environment's
`defsort --no-incremental --diff src benchmarks tests`. Python 3.11 and 3.12 environments omit this
tool.

Taplo and shfmt must be available when formatting Markdown. If either is missing, mdformat can warn
that it failed to format a code block and still succeed; check its warnings.

To install the operator command with its `valkey` extra, run `make install`. It uses `uv tool`;
`make reinstall` replaces an existing installation.

## Continuous integration

The [CI workflow](../.github/workflows/ci.yml) runs `quality` first, then `test` jobs in parallel.
`package` waits for those jobs to pass; `hosted-validation` reports the overall outcome, including
failures. Pushes run every job on GitHub-hosted runners, with every test.

The `test` matrix covers Python 3.11 to 3.15 on Linux and Python 3.13 on macOS and Windows. All
test jobs have a one-hour hang guard.

CI downloads checksum-verified Linux builds of Taplo and shfmt. The lock excludes the Python Taplo
package that `mdformat-config` would otherwise install: it lacks some ARM wheels, and CI uses the
native binary.

## Dependency updates

The files serve different purposes:

| File                    | Contents                                                                    |
| ----------------------- | --------------------------------------------------------------------------- |
| `pyproject.toml`        | Published dependency ranges and development requirements                    |
| `requirements-ci.txt`   | Exact pins for development tools, the `valkey` extra, and the build backend |
| `requirements-test.txt` | Smaller set for test jobs, constrained by the same pins                     |

Neither requirements file ships with the package. Use the pinned uv version when updating them:

1. Edit dependencies in `pyproject.toml`.
2. Run `make lock-update`. To refresh allowed versions, use
   `make lock-update LOCK_FLAGS=--upgrade`.
3. Review both requirements-file diffs and run `make setup` to synchronize the environment.
4. Run `make lock-check`, also included in `make check`.

The `valkey` extra requires `>=6.1.1,<6.2`. The async backend in `valkey_async.py` subclasses
Valkey's pool and connection classes and relies on their internals. When extending this range,
compare those interfaces and the [local stubs](../typings/README.md) with the new dependency, and
run the Valkey suites plus live connection, cancellation/release, primary-proof, and failover
checks.

Valkey 6.1.1 calls `asyncio.iscoroutinefunction` during async connection setup. Account for that
dependency when considering Python 3.16 support, where that function is scheduled for removal.

## Live Valkey integration tests

Use disposable services containing no other data. The basic integration tests run `FLUSHDB` on both
configured databases, deleting all contents. Script-error tests also use `CONFIG SET`, `REPLICAOF`,
`ACL SETUSER`, and `ACL DELUSER`; their users need those administrative permissions. Local setup
does not run these checks against production services.

Tests marked `integration`, including `tests/valkey/test_integration.py` and
`tests/valkey/test_async_integration.py`, skip unless `SG_QUERY_CACHE_REQUIRE_INTEGRATION=1` is
set. `make integration` sets it and runs every integration test. Prepare the requirements for the
subset you intend to run:

| Test scope                                                                | Additional setup                                                                                                                                                                               |
| ------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Basic result and metadata tests                                           | `SG_QUERY_CACHE_TEST_RESULT_*` and `SG_QUERY_CACHE_TEST_METADATA_*`, as in [shared benchmarks](benchmarks.md#measure-with-disposable-valkey-services)                                          |
| Full integration target                                                   | `SG_QUERY_CACHE_ALT_API_SOURCE` points to an alternate SG API checkout; `SG_EVENT_DAEMON_SOURCE` points to an event-daemon source root from which `.venv` can import `sg_event_daemon.testing` |
| `tests/valkey/test_failover.py` and `tests/valkey/test_async_failover.py` | Dedicated topology configured through `SG_QUERY_CACHE_TEST_FAILOVER_*` and `SG_QUERY_CACHE_TEST_SENTINEL_*`; tests fail without it                                                             |
| ElastiCache tests                                                         | Also require `SG_QUERY_CACHE_REQUIRE_ELASTICACHE=1`                                                                                                                                            |

Run focused test files when you have only the two disposable roles. Use `make integration` once the
full target's requirements are ready.
