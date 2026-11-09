# Local Valkey type stubs

These partial stubs provide portable Pyright types for the optional `valkey` dependency. They cover
the interfaces this project uses within its supported `>=6.1.1,<6.2` range; see
[dependency updates](../docs/development.md#dependency-updates). Python imports the installed
Valkey package at runtime. The stubs affect static analysis only.

## What the stubs cover

Valkey supplies `py.typed`, but some interfaces used here have incomplete annotations. The local
declarations include the sync and async connection and pool inheritance used by the backends.

| Interface                 | Coverage                                                                              |
| ------------------------- | ------------------------------------------------------------------------------------- |
| Sync `Valkey` constructor | The project's keyword calls, not all positional or optional parameters                |
| Sync commands             | Members needed for structural checks                                                  |
| Async `Valkey`            | Lifecycle and pool ownership; command calls use the project's typed protocol boundary |

## Update the stubs

Keep declarations limited to imported interfaces and methods used by the project. When upgrading
Valkey or using another part of its API, compare the relevant runtime signatures and inheritance
with these declarations, then run the tracked Pyright check with `make type-check`.
