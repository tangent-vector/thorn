"""Brain-side per-prompt MCP tool discovery.

Phase C.0 produces a deduplicated list of ``MCPServerConfig`` instances
as part of the per-prompt context bundle (see
:class:`thorn.runtime._prompt_assembly.AssembledPromptContext`).
Phase C.1's brain-side responsibility is to turn that list into
:class:`~thorn.core._loop._WrappedTool` instances ready for the agent
loop's split router, with all MCP execution flowing through the
daemon (no brain-side ``ClientSession`` involvement).

This module owns three discrete jobs:

1. *Inventory*: ask the daemon-backed sandbox executor for each
   server's tool list (``list_mcp_server_tools``).  Failures from any
   single server warn-and-skip so one broken MCP server does not
   block the rest of the prompt.
2. *Name resolution*: enforce the policy "unprefixed name when
   unique; ``<server>__<tool>`` on collision; error on true ambiguity"
   across the union of built-in tool names plus tool names from every
   surviving MCP server.
3. *Wrapping*: build a :class:`_WrappedTool` per MCP tool with
   ``venue=SANDBOX``, the (possibly prefixed) registry name, and the
   ``mcp_server_config`` / ``mcp_tool_name`` metadata that the loop
   forwards to the daemon executor on dispatch.  The wrapper's local
   ``execute`` callable raises a clear error if invoked, because MCP
   tools have no in-process execution path.

The discovery pipeline is intentionally a free function rather than
a class.  It is stateless, has no lifecycle of its own, and is run
once per prompt: instantiating an object would buy nothing.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from thorn.core._mcp_config import MCPServerConfig

if TYPE_CHECKING:
    from thorn.core._loop import _WrappedTool

logger = logging.getLogger(__name__)


async def discover_mcp_tools(
    *,
    sandbox_executor: Any,
    mcp_configs: list[MCPServerConfig],
    builtin_tool_names: set[str],
) -> list["_WrappedTool"]:
    """Resolve *mcp_configs* into ``_WrappedTool`` instances.

    Parameters
    ----------
    sandbox_executor:
        The agent's :class:`~thorn.toolhost.DaemonToolExecutor`.  Must
        expose ``list_mcp_server_tools(MCPServerConfig)``; passed by
        type rather than annotated to avoid a hard import dependency
        from this runtime-level module on the toolhost package.  Any
        executor that does not advertise that surface raises
        ``AttributeError`` here, which we treat as "MCP unsupported"
        and log+skip the entire MCP step.
    mcp_configs:
        The MCP server configs to spin up for this prompt, in the
        order produced by :func:`assemble_prompt_context` (outer-most
        first; identity-deduplicated already).
    builtin_tool_names:
        Names of the agent's already-prepared built-in tools.  MCP
        tool names that collide with a built-in are *always* prefixed
        (built-ins win the unprefixed slot); MCP tool names that
        collide with another MCP tool from a different server are
        prefixed only on the colliding side(s).

    Returns
    -------
    list of :class:`_WrappedTool` (possibly empty).  The caller
    appends these to its already-prepared tool list before handing
    everything to the agent loop.

    Failure handling
    ----------------
    * Per-server inventory failure (e.g.
      :class:`~thorn.toolhost.MCPServerUnavailableError`): logged at
      WARNING, server omitted from this prompt, others proceed.
    * Daemon does not advertise the MCP feature flag at all (one
      ``AttributeError`` from ``list_mcp_server_tools``): logged at
      WARNING, every config skipped.
    * True ambiguity post-prefixing (two surviving servers with the
      same name and tool): logged at ERROR, the colliding tools are
      dropped but the rest of the discovered tools survive.
    """
    if not mcp_configs:
        return []

    list_method = getattr(sandbox_executor, "list_mcp_server_tools", None)
    if list_method is None:
        logger.warning(
            "MCP configs found (%d) but the sandbox executor does not "
            "expose list_mcp_server_tools; skipping MCP tool discovery",
            len(mcp_configs),
        )
        return []

    # Phase 1: inventory.  Each surviving entry is the (config, list of
    # OpenAI-style tool schemas) pair the rest of the function works
    # over.  We also normalise per-server tool name uniqueness so a
    # buggy server that lists the same tool twice does not double-count
    # toward the collision tally below.
    per_server: list[tuple[MCPServerConfig, list[dict[str, Any]]]] = []
    for cfg in mcp_configs:
        try:
            schemas = await list_method(cfg)
        except Exception as exc:
            logger.warning(
                "MCP server %r: inventory failed (%s); skipping",
                cfg.name,
                exc,
            )
            continue
        unique_schemas: list[dict[str, Any]] = []
        seen_local: set[str] = set()
        for schema in schemas:
            name = _schema_name(schema)
            if not name or name in seen_local:
                continue
            seen_local.add(name)
            unique_schemas.append(schema)
        per_server.append((cfg, unique_schemas))

    if not per_server:
        return []

    # Phase 2: collision analysis.  We need to know, for every MCP
    # tool name, how many *distinct* MCP servers expose it -- so a
    # name unique within a single server but shared across two
    # servers still triggers prefixing on both sides.  The built-in
    # name set always forces prefixing for any matching MCP tool.
    mcp_provider_count: dict[str, int] = {}
    for _cfg, schemas in per_server:
        for schema in schemas:
            name = _schema_name(schema)
            if not name:
                continue
            mcp_provider_count[name] = mcp_provider_count.get(name, 0) + 1

    # Phase 3: build _WrappedTool instances under the resolution
    # policy.  ``registered_names`` enforces the post-prefixing
    # uniqueness guarantee; collisions there are the "true ambiguity"
    # case that earns an ERROR log and a skip.
    from thorn.core._executor import ToolVenue
    from thorn.core._loop import _WrappedTool

    wrapped: list[_WrappedTool] = []
    registered_names: set[str] = set()
    for cfg, schemas in per_server:
        for schema in schemas:
            tool_name = _schema_name(schema)
            if not tool_name:
                continue
            collides_with_builtin = tool_name in builtin_tool_names
            collides_with_other_mcp = mcp_provider_count.get(tool_name, 0) > 1
            if collides_with_builtin or collides_with_other_mcp:
                exposed = f"{cfg.name}__{tool_name}"
            else:
                exposed = tool_name

            if exposed in registered_names or exposed in builtin_tool_names:
                logger.error(
                    "MCP tool %r from server %r collides with an existing "
                    "tool name %r even after server-name prefixing; dropping. "
                    "Rename the offending server entry in mcp.json to "
                    "disambiguate.",
                    tool_name, cfg.name, exposed,
                )
                continue
            registered_names.add(exposed)

            wrapped.append(
                _WrappedTool(
                    schema=_relabel_schema(schema, exposed),
                    execute=_make_unsupported_execute(exposed),
                    venue=ToolVenue.SANDBOX,
                    mcp_server_config=cfg,
                    mcp_tool_name=tool_name,
                )
            )

    return wrapped


def _schema_name(schema: dict[str, Any]) -> str:
    """Pull the function name from an OpenAI-style tool schema, or ``""``."""
    if not isinstance(schema, dict):
        return ""
    func = schema.get("function")
    if not isinstance(func, dict):
        return ""
    name = func.get("name")
    return name if isinstance(name, str) else ""


def _relabel_schema(schema: dict[str, Any], name: str) -> dict[str, Any]:
    """Return a shallow copy of *schema* with ``function.name`` set to *name*.

    The original daemon-supplied schema may be reused across prompts
    (the daemon caches its tool lists), so we never mutate it in
    place: that would silently change the cached copy and break
    subsequent prompts that need the unprefixed name.
    """
    func = dict(schema.get("function") or {})
    func["name"] = name
    return {
        "type": schema.get("type", "function"),
        "function": func,
    }


def _make_unsupported_execute(exposed_name: str) -> Any:
    """Build an ``execute`` stub that errors if anything calls it.

    MCP tools route through the daemon executor (``venue=SANDBOX``);
    the wrapped tool's local ``execute`` is only reached if a caller
    bypasses the split router and dispatches in-process, which would
    indicate a misconfiguration of the executor router for the agent.
    Surfacing that as a clear ``RuntimeError`` is preferable to a
    silent in-process attempt that would have no idea how to talk to
    the MCP server.
    """

    async def _execute(**kwargs: Any) -> str:
        raise RuntimeError(
            f"MCP tool {exposed_name!r} can only be executed via the "
            "daemon-backed sandbox executor; the in-process path is "
            "not supported.  Ensure the agent's runtime has a sandbox "
            "executor configured (see thorn.runtime.Runtime).",
        )

    return _execute


__all__ = ["discover_mcp_tools"]
