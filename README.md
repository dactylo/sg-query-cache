# sg-query-cache

sg-query-cache reduces repeated Autodesk Flow Production Tracking API reads from Python workers.
When render jobs, pipeline tools, or services ask the same question many times, the cache answers
the repeats from memory or from a shared Valkey service instead of sending another request to the
site. It wraps your existing synchronous or native asynchronous client and keeps its method
signatures, so your code changes only where it creates the client.

This documentation uses SG for Flow Production Tracking, formerly named Shotgun and ShotGrid.

Caching trades freshness for fewer requests. Every cached answer has a maximum age, 30 seconds by
default for a single process. Writes made through the wrapper invalidate the cached answers they
can affect, and a shared deployment also invalidates answers when SG reports a change. Until then,
a change made elsewhere can be missing from a cached answer. When a read must reflect SG right now,
bypass the cache for that read. The [consistency guide](docs/consistency.md) states the guarantee
in full.

## When to use it

The cache helps when processes repeat exact reads: the same method, filters, fields, and client
options, under the same SG permissions, within seconds or minutes of each other. It does not speed
up a read it has not seen before. It caches `find`, `find_one`, and schema reads, plus
`work_schedule_read` if you enable it; other reads go straight to SG.

Choose a layout:

| Layout          | Use when                                            | What it needs, and how fresh it is                                                                                                                      |
| --------------- | --------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Local (default) | One process repeats reads                           | Nothing beyond this package. Each process keeps its own in-memory cache, and changes made elsewhere appear within 30 seconds by default.                |
| Shared          | Several processes should reuse each other's results | Two Valkey services and the SG event-daemon plugin, which invalidates results as SG reports changes. Most results are kept up to 10 minutes by default. |
| Hierarchical    | Shared results plus a fast per-process copy         | The same as shared. Each process can serve its own copy for up to 10 seconds by default after another process changes the data.                         |

Start with the local layout to measure how many of your reads repeat. A shared layout trusts every
client that can reach its Valkey services: the cache does not check that a caller's declared
identity matches its SG credentials. Share Valkey only among processes you trust to declare their
identities correctly.

## Installation

sg-query-cache requires Python 3.11 or newer and an SG client in the same environment:

- For synchronous code, the official `shotgun_api3` package or `sg-python-api`.
- For native async code, `sg-python-api[async]`.

The official package and sg-python-api both provide the `shotgun_api3` import name, so install only
one of them. The shared and hierarchical layouts also need two Valkey services; see
[service layout](docs/operations.md#service-layout).

From a checkout, install the library into the environment that runs your SG client:

```sh
python -m pip install .
```

For the shared or hierarchical layout, include the `valkey` extra. It adds the Valkey client that
the cache and the `sg-query-cache` operator commands use. The extra accepts valkey-py 6.1.x, the
release family this version is tested with:

```sh
python -m pip install '.[valkey]'
```

## Quick start

### Cache a synchronous client

Set `SHOTGUN_URL`, `SHOTGUN_SCRIPT_NAME`, and `SHOTGUN_API_KEY` for a script account with read
access, then run:

```python
import os
from time import monotonic

from shotgun_api3 import Shotgun
from sg_query_cache import (
    CacheSettings,
    CachedShotgun,
    ClientContext,
    Identity,
    LocalBackend,
    NullMetrics,
    QueryCache,
)

site_url = os.environ['SHOTGUN_URL']
script_name = os.environ['SHOTGUN_SCRIPT_NAME']
raw_sg = Shotgun(site_url, script_name=script_name, api_key=os.environ['SHOTGUN_API_KEY'])

settings = CacheSettings()
identity = Identity(kind='script', issuer='shotgun-script', namespace=script_name)
cache = QueryCache(settings, identity, l1=LocalBackend(settings, monotonic, NullMetrics()))
context = ClientContext(
    site_url=site_url,
    site_namespace=settings.site_namespace,
    localized=raw_sg.config.localized,
    convert_datetimes_to_utc=raw_sg.config.convert_datetimes_to_utc,
    records_per_page=raw_sg.config.records_per_page,
)
sg = CachedShotgun(raw_sg, cache, context)


def timed_read(label: str) -> None:
    started = monotonic()
    assets = sg.find('Asset', [['sg_status_list', 'is', 'ip']], ['code'])
    print(f'{label}: {len(assets)} assets in {monotonic() - started:.3f} s')


try:
    timed_read('first read')
    timed_read('second read')
    with sg.bypass():
        timed_read('bypass')
finally:
    cache.close()
    raw_sg.close()
```

You should see output like this, with your own counts and times:

```text
first read: 42 assets in 0.412 s
second read: 42 assets in 0.001 s
bypass: 42 assets in 0.198 s
```

The first read goes to SG. It also loads the site schema, which the cache uses to work out what
each query depends on. The second, identical read comes from the in-process cache. Inside
`bypass()`, reads always go to SG and leave the cache unchanged. A new process starts with an empty
local cache.

The example sets up four pieces:

- `CacheSettings()` selects the local layout with default limits. Shared layouts load their
  settings from a TOML file; see [configuration](docs/configuration.md).
- `Identity` names the SG permission scope whose results may be reused. Here every process running
  this script shares one identity named after the script; give scripts with different permissions
  different identities. Use stable configuration values, never an API key, session token, or
  process ID, because identity values become part of every cache key.
- `QueryCache` with a `LocalBackend` is the cache itself. The wrapper closes neither the cache nor
  the SG client, so the example closes both.
- `ClientContext` records the raw client's response options, such as page size, so results fetched
  with different options are never mixed. Keep it matched to the raw client's configuration.

### Use a native async client

With `sg-python-api[async]`, `AsyncCachedShotgun.open()` builds the cache from your settings and
closes it on exit. You open and close the SG client yourself:

```python
import asyncio
import os

from shotgun_api3 import AsyncShotgun
from sg_query_cache import AsyncCachedShotgun, CacheSettings, ClientContext, Identity


async def main() -> None:
    site_url = os.environ['SHOTGUN_URL']
    script_name = os.environ['SHOTGUN_SCRIPT_NAME']
    settings = CacheSettings()
    identity = Identity(kind='script', issuer='shotgun-script', namespace=script_name)
    async with AsyncShotgun(
        site_url, script_name=script_name, api_key=os.environ['SHOTGUN_API_KEY']
    ) as raw_sg:
        context = ClientContext(
            site_url=site_url,
            site_namespace=settings.site_namespace,
            localized=raw_sg.config.localized,
            convert_datetimes_to_utc=raw_sg.config.convert_datetimes_to_utc,
            records_per_page=raw_sg.config.records_per_page,
        )
        async with AsyncCachedShotgun.open(raw_sg, settings, identity, context) as sg:
            filters = [['sg_status_list', 'is', 'ip']]
            for label in ('first read', 'second read'):
                assets = await sg.find('Asset', filters, ['code'])
                print(f'{label}: {len(assets)} assets')
            with sg.bypass():
                assets = await sg.find('Asset', filters, ['code'])
                print(f'bypass: {len(assets)} assets')


asyncio.run(main())
```

As in the synchronous example, the second read comes from the cache and the bypassed read goes to
SG. `bypass()` is an ordinary `with` block, even in async code.

Use each async cache on one event loop. `bypass()` and other policy scopes apply to the current
task, and a task created inside a scope inherits it. Before you close caches in a service with
long-running tasks, read [async shutdown](docs/operations.md#async-deadlines-and-shutdown).

## Documentation

| If you want to...                                   | Read                                                          |
| --------------------------------------------------- | ------------------------------------------------------------- |
| Understand how fresh results are and when to bypass | [Consistency](docs/consistency.md)                            |
| Choose a layout and write the settings file         | [Configuration](docs/configuration.md)                        |
| Cache reads for human users                         | [Identities](docs/configuration.md#identities-and-trust)      |
| Change how long results are kept                    | [Maximum age rules](docs/configuration.md#maximum-age-rules)  |
| Run Valkey and the invalidation plugin              | [Operations](docs/operations.md)                              |
| Flush cached results after an incident              | [Recovery](docs/operations.md#flush-or-rotate-cached-results) |
| Collect metrics and set alerts                      | [Metrics](docs/operations.md#metrics-and-alerts)              |
| Measure savings for your workload                   | [Benchmarks](docs/benchmarks.md)                              |
| See changes by version                              | [Changelog](CHANGELOG.md)                                     |

## Development

See the [development guide](docs/development.md) for the required tools, CI, and dependency locks.
With those tools installed, run:

```sh
make setup
make check
```

The checks never contact SG or Valkey; live Valkey integration tests are separate and need
disposable services.

## License

sg-query-cache is available under the [MIT License](LICENSE). It is an independent project and is
not affiliated with or endorsed by Autodesk. Flow Production Tracking is a product of Autodesk,
Inc.
