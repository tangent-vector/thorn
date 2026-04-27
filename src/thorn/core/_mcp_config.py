"""Brain+daemon-shared MCP server configuration dataclass.

``MCPServerConfig`` is the bare per-server config record produced by
the unified context-gathering pipeline (see
:mod:`thorn.runtime._context_layers`) and consumed by anything that
needs to talk to an MCP server.  It deliberately lives in its own
module, separate from any MCP wire-protocol client/server code, so
that:

* The runtime context-gathering pipeline can construct instances
  without pulling in the optional ``mcp`` package.
* The toolhost daemon (Phase C.1's eventual MCP client) can import
  the type without going through ``thorn.core._mcp`` and the
  brain-side ``MCPToolSource`` plumbing that lives there today.

The dataclass itself does not depend on the ``mcp`` package; that
dependency only enters the picture when something actually tries to
connect to a server.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class MCPServerConfig:
    """Connection parameters for a single MCP server.

    For stdio transport, set *command* (and optionally *args* / *env*).
    For HTTP transport, set *url*.

    Equality and hashing follow the standard ``@dataclass`` rules and
    are stable across processes for byte-identical fields, which is
    what the brain's per-prompt deduplication and the daemon's
    identity-keyed ``MCPHost`` rely on.
    """

    name: str
    command: str | None = None
    args: list[str] = field(default_factory=list)
    env: dict[str, str] | None = None
    url: str | None = None

    def __post_init__(self) -> None:
        if not self.command and not self.url:
            raise ValueError(
                f"MCPServerConfig {self.name!r}: "
                "must specify either 'command' (stdio) or 'url' (HTTP)"
            )


def mcp_server_config_identity(
    config: MCPServerConfig,
) -> tuple[str, str | None, tuple[str, ...], tuple[tuple[str, str], ...] | None, str | None]:
    """Return a stable, hashable identity tuple for *config*.

    Two configs that produce equal identity tuples are treated as the
    same MCP server by every Thorn component that needs an
    "is this server already loaded?" answer:

    * The brain's per-prompt context-gathering pipeline uses it to
      dedupe configs harvested from overlapping ``mcp.json`` files.
    * The toolhost daemon's ``MCPHost`` uses it as the cache key for
      at-most-one-process-per-unique-config.

    Every field that influences the running server is part of the
    identity, ``env`` included: rotating a credential by editing the
    ``env`` block in ``mcp.json`` produces a new identity, so the
    daemon spins up a fresh process and the stale one drains and
    exits.  ``args`` is normalised to a tuple and ``env`` to a sorted
    tuple of pairs so dict / list iteration order does not perturb
    the result.
    """
    env_key: tuple[tuple[str, str], ...] | None
    env_key = (
        tuple(sorted(config.env.items()))
        if config.env is not None
        else None
    )
    return (
        config.name,
        config.command,
        tuple(config.args),
        env_key,
        config.url,
    )


__all__ = ["MCPServerConfig", "mcp_server_config_identity"]
