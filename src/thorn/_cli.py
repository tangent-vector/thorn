"""CLI entry point: ``thorn run``, ``thorn chat``, and ``thorn serve``."""

from __future__ import annotations

import asyncio
import sys
from contextlib import AsyncExitStack
from typing import Any

import click
from rich.console import Console

from thorn._context import (
    ConsoleEventSink,
    ExecutionContext,
    Verbosity,
    set_context,
    reset_context,
)
from thorn._agent import Agent
from thorn._discovery import discover_tools, find_thorn_dirs
from thorn._func import _prepare_tools, prompt
from thorn._loop import run_agent_loop, _WrappedTool
from thorn._provider import load_provider_from_env
from thorn._tools import ALL_BUILTIN_TOOLS
from thorn.errors import SkillError, ThornError

console = Console()


def _resolve_verbosity(verbose: int, quiet: bool) -> Verbosity:
    """Map ``-v``/``-q`` CLI flags to a :class:`Verbosity` level."""
    if quiet:
        return Verbosity.QUIET
    if verbose >= 2:
        return Verbosity.DEBUG
    if verbose == 1:
        return Verbosity.VERBOSE
    return Verbosity.NORMAL


def _build_context(
    verbosity: Verbosity = Verbosity.NORMAL,
    trace_file: Any | None = None,
    workspace: str | None = None,
) -> ExecutionContext:
    """Create an execution context from environment variables.

    When *trace_file* is an open file handle, a :class:`JsonLinesSink`
    is composed alongside the console sink so that a structured JSONL
    trace is written in parallel.

    *workspace* overrides the workspace root.  When ``None``, the
    heuristic in :func:`thorn.infer_workspace_root` is used.
    """
    from pathlib import Path

    from thorn import infer_workspace_root
    from thorn._context import EventSink
    from thorn._file_access import load_global_ignores

    provider = load_provider_from_env()
    console_sink: EventSink = ConsoleEventSink(verbosity=verbosity)

    if trace_file is not None:
        from thorn._trace import CompositeEventSink, JsonLinesSink
        sink: EventSink = CompositeEventSink([
            console_sink, JsonLinesSink(trace_file),
        ])
    else:
        sink = console_sink

    ws_root = Path(workspace).resolve() if workspace else infer_workspace_root()
    global_ignores = load_global_ignores(ws_root)

    return ExecutionContext(
        provider=provider,
        event_sink=sink,
        workspace_root=ws_root,
        global_ignores=global_ignores,
    )


async def _collect_all_tools(
    exit_stack: AsyncExitStack,
    *,
    no_tools: bool,
    no_discover: bool,
    no_mcp: bool,
) -> list[_WrappedTool]:
    """Assemble tools from all sources: builtins, discovery, and MCP servers.

    MCP sessions are registered on *exit_stack* so they stay alive for
    the duration of the caller's async block.
    """
    raw: list[Any] = []

    if not no_tools:
        raw.extend(ALL_BUILTIN_TOOLS)

    thorn_dirs = find_thorn_dirs() if (not no_discover or not no_mcp) else []

    if not no_discover:
        discovered = discover_tools()
        if discovered:
            names = [getattr(fn, "__name__", "?") for fn in discovered]
            console.print(
                f"[dim]discovered tools:[/dim] {', '.join(names)}",
                highlight=False,
            )
        raw.extend(discovered)

    tools = _prepare_tools(raw)

    if not no_mcp:
        try:
            from thorn._mcp import MCPToolSource, load_mcp_configs

            configs = load_mcp_configs(thorn_dirs)
            if configs:
                mcp_source: MCPToolSource = await exit_stack.enter_async_context(
                    MCPToolSource(configs)
                )
                mcp_tools = mcp_source.tools
                if mcp_tools:
                    names = [
                        t.schema.get("function", {}).get("name", "?")
                        for t in mcp_tools
                    ]
                    console.print(
                        f"[dim]MCP tools:[/dim] {', '.join(names)}",
                        highlight=False,
                    )
                tools.extend(mcp_tools)
        except ImportError:
            pass

    return tools


def _collect_concierge_prompts() -> list[str]:
    """Return system prompts from any registered ``Concierge`` Agent subclass.

    Called after tool discovery so that ``.thorn/`` modules have had a
    chance to define and register their ``Concierge`` class.
    """
    concierge_cls = Agent._registry.get("Concierge")
    if concierge_cls is None:
        return []
    return concierge_cls._collect_system_prompts()


# ---------------------------------------------------------------------------
# CLI group
# ---------------------------------------------------------------------------

@click.group()
def main() -> None:
    """Thorn — a lightweight agent harness."""


# ---------------------------------------------------------------------------
# thorn init
# ---------------------------------------------------------------------------

_STARTER_TOOLS = '''\
"""Project-specific tools for thorn.

Functions decorated with @tool or @skill are automatically discovered
by `thorn run`, `thorn chat`, and `thorn serve`.
"""

from thorn import tool

# @tool
# async def build_project() -> str:
#     """Build the project and return a summary of the results."""
#     ...
'''

_STARTER_MCP_JSON = '{\n    "mcpServers": {}\n}\n'


@main.command()
@click.option(
    "--with-mcp",
    is_flag=True,
    default=False,
    help="Also create a stub mcp.json for MCP server configuration.",
)
def init(with_mcp: bool) -> None:
    """Create a .thorn/ directory with starter files in the current directory."""
    from pathlib import Path

    thorn_dir = Path.cwd() / ".thorn"

    if thorn_dir.exists():
        console.print(
            f"[yellow].thorn/ already exists at {thorn_dir}[/yellow]"
        )
        sys.exit(1)

    thorn_dir.mkdir()

    tools_path = thorn_dir / "tools.py"
    tools_path.write_text(_STARTER_TOOLS, encoding="utf-8")
    console.print(f"  Created {tools_path}")

    if with_mcp:
        mcp_path = thorn_dir / "mcp.json"
        mcp_path.write_text(_STARTER_MCP_JSON, encoding="utf-8")
        console.print(f"  Created {mcp_path}")

    console.print(
        "\n[green]Initialized .thorn/ directory.[/green] "
        "Edit tools.py to define your project's tools."
    )


# ---------------------------------------------------------------------------
# thorn run
# ---------------------------------------------------------------------------

@main.command()
@click.argument("prompt_text")
@click.option(
    "--no-tools",
    is_flag=True,
    default=False,
    help="Disable built-in tools (file I/O, etc.).",
)
@click.option(
    "--no-discover",
    is_flag=True,
    default=False,
    help="Skip .thorn/ directory discovery.",
)
@click.option(
    "--no-mcp",
    is_flag=True,
    default=False,
    help="Skip MCP server tool sources.",
)
@click.option("-v", "--verbose", count=True, help="Increase output detail (-v, -vv).")
@click.option("-q", "--quiet", is_flag=True, default=False, help="Suppress all output except the final answer.")
@click.option("--trace", "trace_path", type=click.Path(), default=None, help="Write execution trace to a JSONL file.")
@click.option("--workspace", "workspace_path", type=click.Path(exists=True, file_okay=False), default=None, help="Override workspace root directory.")
def run(prompt_text: str, no_tools: bool, no_discover: bool, no_mcp: bool, verbose: int, quiet: bool, trace_path: str | None, workspace_path: str | None) -> None:
    """Execute a single prompt and print the result."""
    trace_file = open(trace_path, "w", encoding="utf-8") if trace_path else None
    try:
        ctx = _build_context(_resolve_verbosity(verbose, quiet), trace_file=trace_file, workspace=workspace_path)
    except ThornError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        if trace_file:
            trace_file.close()
        sys.exit(1)

    async def _run() -> str:
        token = set_context(ctx)
        try:
            async with AsyncExitStack() as stack:
                tools = await _collect_all_tools(
                    stack,
                    no_tools=no_tools,
                    no_discover=no_discover,
                    no_mcp=no_mcp,
                )
                sys_prompts = [
                    "You are executing a single non-interactive request. "
                    "Complete the task and report results concisely. "
                    "Do not offer follow-up actions or ask questions.",
                ]
                sys_prompts.extend(_collect_concierge_prompts())
                return await run_agent_loop(
                    context=ctx,
                    user_prompt=prompt_text,
                    tools=tools,
                    system_prompts=sys_prompts,
                )
        finally:
            reset_context(token)

    try:
        result = asyncio.run(_run())
    except SkillError as exc:
        console.print(f"\n[red]Agent error:[/red] {exc.detail}")
        sys.exit(1)
    except ThornError as exc:
        console.print(f"\n[red]Error:[/red] {exc}")
        sys.exit(1)
    except KeyboardInterrupt:
        console.print("\n[dim]Interrupted.[/dim]")
        sys.exit(130)
    finally:
        if trace_file:
            trace_file.close()


# ---------------------------------------------------------------------------
# thorn chat
# ---------------------------------------------------------------------------

@main.command()
@click.option(
    "--no-tools",
    is_flag=True,
    default=False,
    help="Disable built-in tools.",
)
@click.option(
    "--no-discover",
    is_flag=True,
    default=False,
    help="Skip .thorn/ directory discovery.",
)
@click.option(
    "--no-mcp",
    is_flag=True,
    default=False,
    help="Skip MCP server tool sources.",
)
@click.option("-v", "--verbose", count=True, help="Increase output detail (-v, -vv).")
@click.option("-q", "--quiet", is_flag=True, default=False, help="Suppress all output except the final answer.")
@click.option("--trace", "trace_path", type=click.Path(), default=None, help="Write execution trace to a JSONL file.")
@click.option("--workspace", "workspace_path", type=click.Path(exists=True, file_okay=False), default=None, help="Override workspace root directory.")
def chat(no_tools: bool, no_discover: bool, no_mcp: bool, verbose: int, quiet: bool, trace_path: str | None, workspace_path: str | None) -> None:
    """Start an interactive chat session."""
    trace_file = open(trace_path, "w", encoding="utf-8") if trace_path else None
    try:
        ctx = _build_context(_resolve_verbosity(verbose, quiet), trace_file=trace_file, workspace=workspace_path)
    except ThornError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        if trace_file:
            trace_file.close()
        sys.exit(1)

    ctx.system_prompts.append(
        "You are in an interactive chat session with a human user. "
        "You may ask clarifying questions and suggest next steps."
    )
    messages: list = []

    console.print("[bold]thorn[/bold] interactive chat  (Ctrl+C to exit)\n")

    async def _chat() -> None:
        from thorn._messages import UserMessage, AssistantMessage, ToolResultMessage
        from thorn._loop import _request_completion, _execute_tool_calls, _RESULT_SENTINEL
        from thorn._provider import TextChunk, ToolCallChunk, FinishChunk

        token = set_context(ctx)
        try:
            async with AsyncExitStack() as stack:
                tools = await _collect_all_tools(
                    stack,
                    no_tools=no_tools,
                    no_discover=no_discover,
                    no_mcp=no_mcp,
                )
                ctx.system_prompts.extend(_collect_concierge_prompts())
                tool_schemas = [t.schema for t in tools]
                tool_dispatch = {
                    t.schema.get("function", {}).get("name", ""): t
                    for t in tools
                }

                while True:
                    try:
                        user_input = console.input("[green]you>[/green] ")
                    except EOFError:
                        break
                    if not user_input.strip():
                        continue

                    messages.append(UserMessage(content=user_input))

                    for _ in range(50):
                        text_parts: list[str] = []
                        tool_call_chunks: list = []
                        finish_reason = "stop"

                        response = ctx.provider.complete(
                            ctx.system_prompts, tool_schemas, messages,
                        )
                        async for chunk in response:
                            await ctx.event_sink.on_response_chunk(chunk, scope=ctx.scope)
                            match chunk:
                                case TextChunk():
                                    text_parts.append(chunk.text)
                                case ToolCallChunk():
                                    tool_call_chunks.append(chunk)
                                case FinishChunk():
                                    finish_reason = chunk.reason

                        text = "".join(text_parts)
                        tool_calls = [tc.to_tool_call() for tc in tool_call_chunks]

                        assistant_msg = AssistantMessage(content=text, tool_calls=tool_calls)
                        messages.append(assistant_msg)

                        if not tool_calls:
                            break

                        result_msgs, _ = await _execute_tool_calls(
                            tool_calls=tool_calls,
                            tool_dispatch=tool_dispatch,
                            context=ctx,
                            result_type=None,
                        )
                        for rm in result_msgs:
                            messages.append(rm)

                    console.print()

        finally:
            reset_context(token)

    try:
        asyncio.run(_chat())
    except KeyboardInterrupt:
        console.print("\n[dim]Goodbye.[/dim]")
    except ThornError as exc:
        console.print(f"\n[red]Error:[/red] {exc}")
        sys.exit(1)
    finally:
        if trace_file:
            trace_file.close()


# ---------------------------------------------------------------------------
# thorn serve
# ---------------------------------------------------------------------------

@main.command()
@click.option(
    "--transport",
    type=click.Choice(["stdio", "streamable-http"]),
    default="stdio",
    help="MCP transport to use.",
)
@click.option("--host", default="0.0.0.0", help="Bind host (HTTP transport).")
@click.option("--port", default=8080, type=int, help="Bind port (HTTP transport).")
@click.option(
    "--no-tools",
    is_flag=True,
    default=False,
    help="Disable built-in tools.",
)
@click.option(
    "--no-discover",
    is_flag=True,
    default=False,
    help="Skip .thorn/ directory discovery.",
)
@click.option("--name", default="thorn", help="Server name reported to MCP clients.")
def serve(
    transport: str,
    host: str,
    port: int,
    no_tools: bool,
    no_discover: bool,
    name: str,
) -> None:
    """Start an MCP server exposing thorn tools and skills."""
    try:
        from thorn._mcp import serve_tools
    except ImportError:
        console.print(
            "[red]Error:[/red] MCP support requires the 'mcp' package. "
            "Install it with:  pip install thorn[mcp]"
        )
        sys.exit(1)

    raw: list[Any] = []
    if not no_tools:
        raw.extend(ALL_BUILTIN_TOOLS)
    if not no_discover:
        discovered = discover_tools()
        if discovered:
            names = [getattr(fn, "__name__", "?") for fn in discovered]
            console.print(
                f"[dim]serving tools:[/dim] {', '.join(names)}",
                highlight=False,
            )
        raw.extend(discovered)

    tools = _prepare_tools(raw)
    if not tools:
        console.print("[yellow]Warning:[/yellow] no tools to serve.")

    try:
        asyncio.run(serve_tools(
            tools,
            name=name,
            transport=transport,
            host=host,
            port=port,
        ))
    except KeyboardInterrupt:
        console.print("\n[dim]Server stopped.[/dim]")
    except ThornError as exc:
        console.print(f"\n[red]Error:[/red] {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
