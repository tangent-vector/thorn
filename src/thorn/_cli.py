"""CLI entry point: ``thorn run``, ``thorn chat``, and ``thorn serve``."""

from __future__ import annotations

import asyncio
import json
import sys
import time
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any

import click
from rich.console import Console

from thorn.core._context import (
    ConsoleEventSink,
    EventSink,
    ExecutionContext,
    Verbosity,
)
from thorn.core._agent import Agent
from thorn.core._discovery import discover_tools, find_thorn_dirs
from thorn.core._func import _prepare_tools, prompt
from thorn.core._loop import run_agent_loop, _WrappedTool
from thorn.core._provider import load_provider_from_env
from thorn.core._tools import ALL_BUILTIN_TOOLS
from thorn.core.errors import SkillError, ThornError
from thorn.runtime import Runtime

console = Console()


async def _rich_ask_user(question: str) -> str:
    """Prompt the user via the rich console, suitable for interactive CLI use.

    Refuses to block on a non-TTY stdin (e.g. piped input) where
    ``console.input()`` would read EOF or garbage.
    """
    if not sys.stdin.isatty():
        raise RuntimeError(
            "ask_user is not available: stdin is not a TTY. "
            "Cannot prompt for user input in a non-interactive context."
        )
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None, lambda: console.input(f"\n[yellow]? {question}[/yellow]\n> "),
    )


def _resolve_verbosity(verbose: int, quiet: bool) -> Verbosity:
    """Map ``-v``/``-q`` CLI flags to a :class:`Verbosity` level."""
    if quiet:
        return Verbosity.QUIET
    if verbose >= 2:
        return Verbosity.DEBUG
    if verbose == 1:
        return Verbosity.VERBOSE
    return Verbosity.NORMAL


def _build_runtime(
    verbosity: Verbosity = Verbosity.NORMAL,
    trace_file: Any | None = None,
    workspace: str | None = None,
    *,
    interactive: bool = True,
) -> Runtime:
    """Create a ``Runtime`` from environment variables.

    When *trace_file* is an open file handle, a :class:`JsonLinesSink`
    is composed alongside the console sink so that a structured JSONL
    trace is written in parallel.

    *workspace* overrides the workspace root.  When ``None``, the
    heuristic in :func:`thorn.infer_workspace_root` is used.
    """
    from pathlib import Path

    from thorn import infer_workspace_root
    from thorn.core._discovery import load_workspace_instructions
    from thorn.core._file_access import load_global_ignores

    provider = load_provider_from_env()
    console_sink: EventSink = ConsoleEventSink(verbosity=verbosity)

    if trace_file is not None:
        from thorn.core._trace import CompositeEventSink, JsonLinesSink
        sink: EventSink = CompositeEventSink([
            console_sink, JsonLinesSink(trace_file),
        ])
    else:
        sink = console_sink

    ws_root = Path(workspace).resolve() if workspace else infer_workspace_root()

    return Runtime(
        provider=provider,
        event_sink=sink,
        workspace_root=ws_root,
        workspace_instructions=load_workspace_instructions(ws_root),
        global_ignores=load_global_ignores(ws_root),
        ask_user_handler=_rich_ask_user if interactive else None,
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
            from thorn.core._mcp import MCPToolSource, load_mcp_configs

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
    try:
        from dotenv import find_dotenv, load_dotenv
        load_dotenv(find_dotenv(usecwd=True))
    except ImportError:
        pass


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

def _write_result_file(
    path: Path,
    outcome: str,
    duration_s: float,
    usage: Any,
    error: str | None,
    trace_path: str | None,
) -> None:
    """Write the structured JSON result produced by ``thorn run --result-file``."""
    if usage is not None:
        token_usage = {
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens,
        }
    else:
        token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    result = {
        "outcome": outcome,
        "duration_s": round(duration_s, 2),
        "token_usage": token_usage,
        "error": error,
        "trace_file": Path(trace_path).name if trace_path else None,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


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
@click.option("--result-file", "result_file_path", type=click.Path(), default=None, help="Write a JSON result summary (outcome, duration, token usage).")
def run(prompt_text: str, no_tools: bool, no_discover: bool, no_mcp: bool, verbose: int, quiet: bool, trace_path: str | None, workspace_path: str | None, result_file_path: str | None) -> None:
    """Execute a single prompt and print the result."""
    trace_file = open(trace_path, "w", encoding="utf-8") if trace_path else None
    try:
        runtime = _build_runtime(_resolve_verbosity(verbose, quiet), trace_file=trace_file, workspace=workspace_path)
    except ThornError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        if trace_file:
            trace_file.close()
        if result_file_path:
            _write_result_file(
                Path(result_file_path), "agent_error", 0.0, None, str(exc), trace_path,
            )
        sys.exit(1)

    ctx_holder: list[ExecutionContext] = []

    async def _run() -> str:
        async with runtime:
            ctx_holder.append(runtime.context)
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
                    context=runtime.context,
                    user_prompt=prompt_text,
                    tools=tools,
                    system_prompts=sys_prompts,
                )

    outcome = "success"
    error_msg: str | None = None
    exit_code = 0
    t0 = time.monotonic()

    try:
        asyncio.run(_run())
    except SkillError as exc:
        outcome, error_msg, exit_code = "agent_error", exc.detail, 1
        console.print(f"\n[red]Agent error:[/red] {exc.detail}")
    except TimeoutError:
        outcome, error_msg, exit_code = "timeout", "timed out", 1
        console.print("\n[red]Error:[/red] timed out")
    except ThornError as exc:
        outcome, error_msg, exit_code = "agent_error", str(exc), 1
        console.print(f"\n[red]Error:[/red] {exc}")
    except KeyboardInterrupt:
        outcome, error_msg, exit_code = "agent_error", "interrupted", 130
        console.print("\n[dim]Interrupted.[/dim]")
    finally:
        duration_s = time.monotonic() - t0
        if trace_file:
            trace_file.close()
        if result_file_path:
            ctx = ctx_holder[0] if ctx_holder else None
            _write_result_file(
                Path(result_file_path), outcome, duration_s,
                ctx.usage if ctx else None, error_msg, trace_path,
            )

    if exit_code:
        sys.exit(exit_code)


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
        runtime = _build_runtime(_resolve_verbosity(verbose, quiet), trace_file=trace_file, workspace=workspace_path)
    except ThornError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        if trace_file:
            trace_file.close()
        sys.exit(1)

    messages: list = []

    console.print("[bold]thorn[/bold] interactive chat  (Ctrl+C to exit)\n")

    async def _chat() -> None:
        from thorn.core._messages import UserMessage, AssistantMessage, ToolResultMessage
        from thorn.core._loop import _request_completion, _execute_tool_calls, _RESULT_SENTINEL
        from thorn.core._provider import TextChunk, ToolCallChunk, FinishChunk

        async with runtime:
            ctx = runtime.context
            ctx.system_prompts.append(
                "You are in an interactive chat session with a human user. "
                "You may ask clarifying questions and suggest next steps."
            )
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
# thorn serve (group: default=gateway, mcp=MCP server)
# ---------------------------------------------------------------------------

@main.group(invoke_without_command=True)
@click.option("-v", "--verbose", count=True, help="Increase output detail (-v, -vv).")
@click.option("-q", "--quiet", is_flag=True, default=False, help="Suppress all output except errors.")
@click.option("--trace", "trace_path", type=click.Path(), default=None, help="Write execution trace to a JSONL file.")
@click.option("--workspace", "workspace_path", type=click.Path(exists=True, file_okay=False), default=None, help="Override workspace root directory.")
@click.pass_context
def serve(
    ctx: click.Context,
    verbose: int,
    quiet: bool,
    trace_path: str | None,
    workspace_path: str | None,
) -> None:
    """Start the Thorn gateway daemon (or an MCP server via 'thorn serve mcp')."""
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    ctx.obj["quiet"] = quiet
    ctx.obj["trace_path"] = trace_path
    ctx.obj["workspace_path"] = workspace_path

    if ctx.invoked_subcommand is not None:
        return

    _serve_gateway(
        verbose=verbose,
        quiet=quiet,
        trace_path=trace_path,
        workspace_path=workspace_path,
    )


def _serve_gateway(
    *,
    verbose: int,
    quiet: bool,
    trace_path: str | None,
    workspace_path: str | None,
) -> None:
    """Run the gateway daemon (called when ``thorn serve`` has no subcommand).

    Loads service configuration from ``.thorn/gateway.json`` and
    instantiates event sources accordingly.
    """
    import logging
    from rich.logging import RichHandler

    from thorn import infer_workspace_root
    from thorn.gateway import (
        EventSource,
        Gateway,
        instantiate_services,
        load_gateway_config,
    )

    verbosity = _resolve_verbosity(verbose, quiet)

    log_level = logging.DEBUG if verbose >= 2 else (logging.WARNING if quiet else logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True)],
    )

    ws_root = Path(workspace_path).resolve() if workspace_path else infer_workspace_root()
    thorn_dir = ws_root / ".thorn"

    try:
        gateway_config = load_gateway_config(thorn_dir)
    except FileNotFoundError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        sys.exit(1)

    try:
        all_services = instantiate_services(gateway_config)
    except (KeyError, ValueError) as exc:
        console.print(f"[red]Error:[/red] {exc}")
        sys.exit(1)

    sources = [s for s in all_services if isinstance(s, EventSource)]
    if not sources:
        console.print(
            "[yellow]Warning:[/yellow] No event sources configured in "
            f"{thorn_dir / 'gateway.json'}. The gateway will start "
            "but will not receive any events."
        )

    trace_file = open(trace_path, "w", encoding="utf-8") if trace_path else None
    try:
        runtime = _build_runtime(
            verbosity, trace_file=trace_file, workspace=workspace_path,
            interactive=False,
        )
    except ThornError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        if trace_file:
            trace_file.close()
        sys.exit(1)

    for service in all_services:
        runtime.register_service(service)

    gateway = Gateway(runtime=runtime, sources=sources)

    service_names = [s.name for s in gateway_config.services]
    console.print(
        f"[bold]thorn serve[/bold]  services: {', '.join(service_names) or '(none)'}"
    )

    try:
        asyncio.run(gateway.run())
    except KeyboardInterrupt:
        console.print("\n[dim]Gateway stopped.[/dim]")
    except ThornError as exc:
        console.print(f"\n[red]Error:[/red] {exc}")
        sys.exit(1)
    finally:
        if trace_file:
            trace_file.close()


@serve.command("bootstrap")
@click.option("--agent-id", required=True, help="Unique identifier for the coordinator agent.")
@click.option("--project-name", required=True, help="Human-readable project name.")
@click.option("--clone-url", required=True, help="HTTPS clone URL for the project.")
@click.option("--default-branch", default="main", help="Default branch name (default: main).")
@click.option("--project-id", type=int, default=None, help="Numeric GitLab project ID.")
@click.option("--token-env", default="GITLAB_TOKEN", help="Environment variable holding the access token.")
@click.option("--url-env", default="GITLAB_URL", help="Environment variable holding the GitLab instance URL.")
@click.option("--workspace", "workspace_path", type=click.Path(file_okay=False), default=".", help="Runtime root directory (default: current dir).")
@click.pass_context
def serve_bootstrap(
    ctx: click.Context,
    agent_id: str,
    project_name: str,
    clone_url: str,
    default_branch: str,
    project_id: int | None,
    token_env: str,
    url_env: str,
    workspace_path: str,
) -> None:
    """Bootstrap a ProjectCoordinator agent in the runtime directory."""
    from pathlib import Path
    from thorn.gateway._bootstrap import bootstrap_coordinator

    runtime_root = Path(workspace_path).resolve()
    aid = bootstrap_coordinator(
        runtime_root=runtime_root,
        agent_id=agent_id,
        project_name=project_name,
        clone_url=clone_url,
        default_branch=default_branch,
        project_id=project_id,
        access_token_env=token_env,
        gitlab_url_env=url_env,
    )
    console.print(f"[green]Bootstrapped coordinator:[/green] {aid}")
    console.print(f"  Identity: {runtime_root / '.thorn' / 'agents' / f'{aid}.json'}")
    console.print(f"  Gateway config: {runtime_root / '.thorn' / 'gateway.json'}")
    console.print(f"  Workspace: {runtime_root / '.thorn' / 'agents' / str(aid)}")
    console.print(
        "\nEnsure your .env file sets the required environment variables "
        f"(e.g. {url_env}, {token_env}) before running 'thorn serve'."
    )


@serve.command("mcp")
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
def serve_mcp(
    transport: str,
    host: str,
    port: int,
    no_tools: bool,
    no_discover: bool,
    name: str,
) -> None:
    """Start an MCP server exposing thorn tools and skills."""
    try:
        from thorn.core._mcp import serve_tools
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
