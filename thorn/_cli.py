"""CLI entry point: ``thorn run`` and ``thorn chat``."""

from __future__ import annotations

import asyncio
import sys

import click
from rich.console import Console

from thorn._context import (
    ConsoleEventSink,
    ExecutionContext,
    set_context,
    reset_context,
)
from thorn._func import _prepare_tools, prompt
from thorn._loop import run_agent_loop
from thorn._provider import load_provider_from_env
from thorn._tools import ALL_BUILTIN_TOOLS
from thorn.errors import SkillError, ThornError

console = Console()


def _build_context() -> ExecutionContext:
    """Create an execution context from environment variables."""
    provider = load_provider_from_env()
    sink = ConsoleEventSink()
    return ExecutionContext(provider=provider, event_sink=sink)


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
def run(prompt_text: str, no_tools: bool) -> None:
    """Execute a single prompt and print the result."""
    try:
        ctx = _build_context()
    except ThornError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        sys.exit(1)

    tools = _prepare_tools(ALL_BUILTIN_TOOLS) if not no_tools else []

    async def _run() -> str:
        token = set_context(ctx)
        try:
            return await run_agent_loop(
                context=ctx,
                user_prompt=prompt_text,
                tools=tools,
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
def chat(no_tools: bool) -> None:
    """Start an interactive chat session."""
    try:
        ctx = _build_context()
    except ThornError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        sys.exit(1)

    tools = _prepare_tools(ALL_BUILTIN_TOOLS) if not no_tools else []
    messages: list = []

    console.print("[bold]thorn[/bold] interactive chat  (Ctrl+C to exit)\n")

    async def _chat() -> None:
        from thorn._messages import UserMessage, AssistantMessage, ToolResultMessage
        from thorn._loop import _request_completion, _execute_tool_calls, _RESULT_SENTINEL
        from thorn._provider import TextChunk, ToolCallChunk, FinishChunk

        token = set_context(ctx)
        try:
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


if __name__ == "__main__":
    main()
