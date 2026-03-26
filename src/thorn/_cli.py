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
    set_context,
    reset_context,
)
from thorn._discovery import discover_tools, find_thorn_dirs
from thorn._func import _prepare_tools, prompt
from thorn._loop import run_agent_loop, _WrappedTool
from thorn._provider import load_provider_from_env
from thorn._tools import ALL_BUILTIN_TOOLS
from thorn.errors import SkillError, ThornError

console = Console()


def _build_context() -> ExecutionContext:
    """Create an execution context from environment variables."""
    provider = load_provider_from_env()
    sink = ConsoleEventSink()
    return ExecutionContext(provider=provider, event_sink=sink)


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


# ---------------------------------------------------------------------------
# CLI group
# ---------------------------------------------------------------------------

@click.group()
def main() -> None:
    """Thorn — a lightweight agent harness."""


# ---------------------------------------------------------------------------
# thorn run
# ---------------------------------------------------------------------------

@main.command()
@click.argument("prompt_text")
@click.option(
    "--no-tools",
    is_flag=True,
    default=False,
    help="Disable built-in tools (file I/O, shell, etc.).",
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
def run(prompt_text: str, no_tools: bool, no_discover: bool, no_mcp: bool) -> None:
    """Execute a single prompt and print the result."""
    try:
        ctx = _build_context()
    except ThornError as exc:
        console.print(f"[red]Error:[/red] {exc}")
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
                return await run_agent_loop(
                    context=ctx,
                    user_prompt=prompt_text,
                    tools=tools,
                    system_prompts=[
                        "You are executing a single non-interactive request. "
                        "Complete the task and report results concisely. "
                        "Do not offer follow-up actions or ask questions.",
                    ],
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
def chat(no_tools: bool, no_discover: bool, no_mcp: bool) -> None:
    """Start an interactive chat session."""
    try:
        ctx = _build_context()
    except ThornError as exc:
        console.print(f"[red]Error:[/red] {exc}")
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
