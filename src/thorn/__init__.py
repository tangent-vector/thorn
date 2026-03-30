"""Thorn — a lightweight agent harness for mixing Python and prompts.

Public API summary::

    from thorn import prompt, skill, run

    # Inline prompt (text mode)
    result = await prompt("summarize this")

    # Inline prompt (structured mode)
    items = await prompt[list[str]]("list all code files")

    # Skill decorator
    @skill
    async def check(name: str) -> bool:
        \"\"\"Is the {name} service running?\"\"\"

    # Entry point for scripts
    async def main():
        ...

    run(main())
"""

from __future__ import annotations

import asyncio
from typing import Any, Callable, Coroutine, TypeVar

from thorn._context import (
    ConsoleEventSink,
    EventSink,
    ExecutionContext,
    NullEventSink,
    Scope,
    Verbosity,
    get_context,
    set_context,
)
from thorn._agent import Agent
from thorn._discovery import discover_tools
from thorn._func import prompt, skill, tool, wrap_function
from thorn._retry import bound_retries
from thorn._loop import _WrappedTool
from thorn._provider import LLMProvider, MockProvider, load_provider_from_env
from thorn import tools
from thorn._tools import (
    ALL_BUILTIN_TOOLS,
    ask_user,
    list_directory,
    read_file,
    run_shell,
    write_file,
)
from thorn.errors import (
    AgentFailureError,
    LoopLimitError,
    ProviderError,
    RateLimitError,
    SkillError,
    ThornError,
)

from thorn._trace import CompositeEventSink, JsonLinesSink

try:
    from thorn._mcp import MCPServerConfig, MCPToolSource, load_mcp_configs, serve_tools
except ImportError:
    pass

T = TypeVar("T")

__all__ = [
    # Core API
    "Agent",
    "prompt",
    "skill",
    "tool",
    "run",
    "wrap_function",
    "discover_tools",
    "bound_retries",
    # Context
    "ExecutionContext",
    "EventSink",
    "NullEventSink",
    "ConsoleEventSink",
    "CompositeEventSink",
    "JsonLinesSink",
    "Verbosity",
    "Scope",
    "get_context",
    "set_context",
    # Providers
    "LLMProvider",
    "MockProvider",
    "load_provider_from_env",
    # Built-in tools (also available as thorn.tools.*)
    "tools",
    "read_file",
    "write_file",
    "list_directory",
    "run_shell",
    "ask_user",
    "ALL_BUILTIN_TOOLS",
    # Errors
    "ThornError",
    "SkillError",
    "ProviderError",
    "RateLimitError",
    "LoopLimitError",
    "AgentFailureError",
    # MCP (available when thorn[mcp] is installed)
    "MCPServerConfig",
    "MCPToolSource",
    "load_mcp_configs",
    "serve_tools",
]


def run(
    coro: Coroutine[Any, Any, T],
    *,
    provider: LLMProvider | None = None,
    event_sink: EventSink | None = None,
    system: str | None = None,
) -> T:
    """Run an async workflow with a thorn execution context.

    This is the standard entry point for scripts.  It sets up the
    ``ExecutionContext`` (provider, event sink) and drives the given
    coroutine to completion.

    If *provider* is ``None``, one is loaded from environment variables
    via ``load_provider_from_env()``.
    """
    if provider is None:
        provider = load_provider_from_env()
    if event_sink is None:
        event_sink = NullEventSink()

    system_prompts: list[str] = []
    if system:
        system_prompts.append(system)

    ctx = ExecutionContext(
        provider=provider,
        event_sink=event_sink,
        system_prompts=system_prompts,
    )

    async def _run_with_context() -> T:
        token = set_context(ctx)
        try:
            return await coro
        finally:
            from thorn._context import reset_context
            reset_context(token)

    return asyncio.run(_run_with_context())
