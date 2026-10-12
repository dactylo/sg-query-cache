# Development

## Local environment

Install these tools before you set up a checkout:

- Python 3.11 or newer, as `python` on `PATH`. To use another interpreter, run
  `make setup PYTHON=/path/to/python`; setup never downloads Python.
- uv 0.12.23. Setup and the lock targets refuse any other version, so dependency locks resolve the
  same way locally and in CI.
- Taplo 0.10.0 and shfmt 3.14.1. The Markdown formatter uses them for TOML and shell code blocks in
  the docs.
- Biome 2.5.14, which formats the repository's JSON files.
- Node.js 24.21.0, which Pyright uses for type checks. CI installs this version in its quality job.

Then, from the repository root:

```sh
make setup
make check
```

`make setup` creates `.venv` if it does not exist, installs the exact hashed pins from
`requirements-ci.txt`, and installs the project in editable mode. Run it again after changing
`pyproject.toml` or the lock files, including when `make format` rewrites `pyproject.toml`. The
other targets compare `.venv` with those files and refuse to run when it is missing or outdated, so
a check never installs or updates packages.

`make quality` runs the read-only checks that CI's quality job runs: formatting and lint,
dependency locks, Pyright type checks for Linux, macOS, and Windows, and trailing whitespace.
`make check` adds the test suite, and `make test` runs the suite alone. `make format` applies the
formatters; review its changes before committing. `make install` installs the `sg-query-cache`
command with `uv tool`, and `make reinstall` replaces an existing installation.

`requirements-ci.txt` pins the development tools, the `valkey` extra, and the build backend.
`requirements-test.txt` selects the smaller set that test jobs install, constrained by the same
pins. Neither file ships with the package; published dependency ranges stay in `pyproject.toml`.

On Python 3.13 or newer, the development lock also installs defsort for definition-order checks.
Use the project environment's `defsort --no-incremental --diff src benchmarks tests` to inspect
ordering without changing files. Python 3.11 and 3.12 environments omit this development tool.

CI downloads checksum-verified Linux builds of Taplo and shfmt. The lock leaves out the Python
Taplo package that `mdformat-config` would otherwise pull in, because that package lacks some ARM
wheels and CI uses the native binary instead. When Taplo or shfmt is unavailable, mdformat warns
that it could not format a code block but still succeeds, so read its warnings.

## Continuous integration

The [CI workflow](../.github/workflows/ci.yml) runs `quality` first. The `test` jobs then run in
parallel. `package` starts only after those jobs pass; `hosted-validation` reports the outcome of a
hosted run, including failures.

CI uses GitHub-hosted runners unless a run explicitly requests local runners. In a private
repository, the Actions variables `CI_LINUX_RUNNER`, `CI_MACOS_RUNNER`, and `CI_WINDOWS_RUNNER`
select local runners for runs started from Actions > CI > Run workflow with the hosted option
disabled. If a variable is absent, that operating system uses a hosted runner. Pushes, pull
requests, public repositories, and runs with the hosted option enabled use hosted runners for every
job and run every test.

A staging repository sets the Actions variable `CI_RUN_ON_PUSH` to `false`. Its pushes then appear
as `CI / push skipped` and run no jobs; CI runs only on request. A newer requested run on a branch
cancels an older one. Push runs are never cancelled, because the Release workflow needs the tagged
commit's push run.

| Requested run                                                 | Actions > CI > Run workflow | Title               |
| ------------------------------------------------------------- | --------------------------- | ------------------- |
| Quick: local runners, slow tests omitted on macOS and Windows | hosted off, slow tests off  | `CI / local`        |
| Full: local runners, every test                               | hosted off, slow tests on   | `CI / local (full)` |
| Hosted: GitHub-hosted runners, every test                     | hosted on                   | `CI / hosted`       |

The `test` matrix covers Python 3.11 to 3.15 on Linux and Python 3.13 on macOS and Windows. All
test jobs have a one-hour hang guard, on local and hosted runners, in quick and full runs.

CI runs tests in parallel with pytest-xdist. Set `PYTEST_XDIST_AUTO_NUM_WORKERS` when a runner's
visible CPU count exceeds its CPU quota. For the same scheduling locally, run:

```sh
make test TEST_FLAGS='-n auto --dist worksteal --durations=15'
```

The manual [Release workflow](../.github/workflows/release.yml) publishes the package to PyPI after
checking its version tag and a successful push CI run for the tagged commit.

## Dependency updates

Use the uv version pinned in CI. After changing dependencies in `pyproject.toml`, run
`make lock-update`. To refresh allowed versions, run `make lock-update LOCK_FLAGS=--upgrade`.
Review both requirements file diffs and run `make lock-check`; `make check` includes that check.

The `valkey` extra is capped at the tested minor family (`>=6.1.1,<6.2`) because `valkey_async.py`
subclasses Valkey's asyncio pool and connection classes and relies on their internals. The stubs in
`typings/` follow 6.1.1. Raise the cap only after the Valkey suites and live acceptance checks for
connection, cancellation and release, primary proof, and failover pass on the new family, and the
stubs are compared with it. Revisit the cap when Valkey 6.2 is released as final and before
claiming Python 3.16 support: Valkey 6.1.1 calls `asyncio.iscoroutinefunction`, which Python 3.16
is slated to remove, so every async connect would fail there.

## Project notes

`make test` runs the suite serially. `make benchmark` runs the local hot-key
[benchmark](benchmarks.md) scenario.

The `slow` marker covers `tests/test_scale.py`. Service integration requires
`SG_QUERY_CACHE_REQUIRE_INTEGRATION=1`; ElastiCache failover has a separate explicit gate. Run
focused tests while editing. Live service checks have separate configuration and do not run against
production services as part of local setup.

### Live Valkey integration tests

The tests marked `integration`, including `tests/valkey/test_integration.py` and
`tests/valkey/test_async_integration.py`, exercise real Valkey. They skip unless
`SG_QUERY_CACHE_REQUIRE_INTEGRATION=1` is set; `make integration` sets it and runs every
integration test. The basic tests read the same `SG_QUERY_CACHE_TEST_RESULT_*` and
`SG_QUERY_CACHE_TEST_METADATA_*` variables as the
[shared benchmarks](benchmarks.md#measure-with-disposable-valkey-services). They run `FLUSHDB` on
both configured databases, deleting everything in them, so use disposable services that hold no
other data. The script error tests also run `CONFIG SET`, `REPLICAOF`, `ACL SETUSER`, and
`ACL DELUSER`, so the configured users need those administrative commands. The full integration
target also requires `SG_QUERY_CACHE_ALT_API_SOURCE` to point to an alternate SG API checkout, and
`SG_EVENT_DAEMON_SOURCE` to point to an event-daemon source root. The current `.venv` must be able
to import `sg_event_daemon.testing` from that source root. The failover tests in
`tests/valkey/test_failover.py` also need a dedicated failover topology described by further
`SG_QUERY_CACHE_TEST_FAILOVER_*` and `SG_QUERY_CACHE_TEST_SENTINEL_*` variables, and fail without
them; run focused test files when you have only the two disposable roles. The ElastiCache tests
also require `SG_QUERY_CACHE_REQUIRE_ELASTICACHE=1`.

Offline skips and controlled transport tests do not establish TLS, Sentinel failover, or
managed-provider behavior.
