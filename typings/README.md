# Local Valkey type stubs

These partial stubs cover the Valkey interfaces used by this project. They provide portable Pyright
types for the optional `valkey` dependency. Version 6.1.1 ships `py.typed`, but some interfaces
used here have incomplete annotations. The declarations follow the installed 6.1.1 API, including
the synchronous and asynchronous connection and pool inheritance used by the backends.
The `valkey` extra is capped to the family these stubs follow; see
[dependency updates](../docs/development.md#dependency-updates).

The `Valkey` constructor models the project's keyword calls, not its full positional or optional
parameter list. The retained sync command members support structural checks. The async `Valkey`
stub models lifecycle and pool ownership, while command calls use the project's typed protocol
boundary. Keep the stubs limited to the imported interfaces and exercised methods. When upgrading
Valkey or using another part of its API, compare the relevant runtime signatures and inheritance
with the declarations here, then run the tracked Pyright check. These files are only for static
analysis; Python imports the installed Valkey package at runtime.
