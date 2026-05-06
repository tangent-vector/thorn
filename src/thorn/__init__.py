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

from thorn.core import (
    AdvisoryNode,
    ConsoleEventSink,
    EventSink,
    ExecutionContext,
    NullEventSink,
    Scope,
    StatusProvider,
    UsageTracker,
    Verbosity,
    get_context,
    scoped_status_provider,
    set_context,
    Agent,
    DirectorySeed,
    FileSeed,
    SearchSeed,
    SeedContent,
    discover_tools,
    ArchiveMarkerNode,
    CollapseState,
    CompactionResult,
    HistoryNode,
    HistoryTree,
    HousekeepingNode,
    ModulePath,
    ValidationRule,
    ValidationStatus,
    ValidationTracker,
    FileAccessLevel,
    FileAccessPolicy,
    FileAccessRule,
    RelativeTo,
    prompt,
    skill,
    tool,
    wrap_function,
    bound_retries,
    _WrappedTool,
    LLMProvider,
    MockProvider,
    UsageChunk,
    load_provider_from_env,
    HousekeepingResult,
    perform_housekeeping,
    select_cut_point,
    JOURNAL_TOOLS,
    append_journal_entry,
    list_journal_dates,
    read_journal,
    read_journal_day,
    read_recent_journal,
    write_journal,
    ALL_BUILTIN_TOOLS,
    FILE_READING,
    FILE_WRITING,
    FileEdit,
    create_file,
    delete_file,
    edit_file,
    find_files,
    list_directory,
    move_file,
    read_file,
    search_files,
    AgentFailureError,
    LoopLimitError,
    ProviderError,
    ProviderUnavailableError,
    RateLimitError,
    SkillError,
    ThornError,
    TransientProviderError,
    CompositeEventSink,
    JsonLinesSink,
    EventBus,
    ScopeFilter,
    Subscription,
    accept_all,
    in_session,
)

from thorn import tools
from thorn.runtime import AgentID, Runtime, SessionKey

from thorn.core import MCPServerConfig

try:
    from thorn.core import serve_tools
except ImportError:
    pass

T = TypeVar("T")


def _effective_context_window(
    provider_context_window: int | None,
    user_context_window: int | None,
) -> int | None:
    """Compute the effective context window budget.

    Uses ``min(provider, user)`` when both are available, or whichever
    one is non-None.  Returns ``None`` (no compaction) when neither is
    set.
    """
    if provider_context_window is not None and user_context_window is not None:
        return min(provider_context_window, user_context_window)
    if provider_context_window is not None:
        return provider_context_window
    return user_context_window


__all__ = [
    # Core API
    "Agent",
    "ModulePath",
    "ValidationRule",
    "ValidationStatus",
    "ValidationTracker",
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
    "EventBus",
    "ScopeFilter",
    "Subscription",
    "accept_all",
    "in_session",
    "Verbosity",
    "Scope",
    "UsageTracker",
    "get_context",
    "set_context",
    # File access control
    "FileAccessLevel",
    "FileAccessRule",
    "FileAccessPolicy",
    "RelativeTo",
    # Providers
    "LLMProvider",
    "MockProvider",
    "UsageChunk",
    "load_provider_from_env",
    # Journal
    "JOURNAL_TOOLS",
    "append_journal_entry",
    "list_journal_dates",
    "read_journal",
    "read_journal_day",
    "read_recent_journal",
    "write_journal",
    # Built-in tools (also available as thorn.tools.*)
    "tools",
    "read_file",
    "edit_file",
    "create_file",
    "delete_file",
    "move_file",
    "FileEdit",
    "list_directory",
    "find_files",
    "search_files",
    "ALL_BUILTIN_TOOLS",
    "FILE_READING",
    "FILE_WRITING",
    # Housekeeping
    "HousekeepingResult",
    "perform_housekeeping",
    "select_cut_point",
    # History / compaction
    "ArchiveMarkerNode",
    "CollapseState",
    "CompactionResult",
    "HistoryNode",
    "HistoryTree",
    "HousekeepingNode",
    # Context injection
    "SeedContent",
    "FileSeed",
    "DirectorySeed",
    "SearchSeed",
    # Runtime
    "AgentID",
    "Runtime",
    "SessionKey",
    # Errors
    "ThornError",
    "SkillError",
    "ProviderError",
    "ProviderUnavailableError",
    "RateLimitError",
    "TransientProviderError",
    "LoopLimitError",
    "AgentFailureError",
    # MCP (``MCPServerConfig`` is always importable; ``serve_tools``
    # requires the ``thorn[mcp]`` optional extra).
    "MCPServerConfig",
    "serve_tools",
]


def run(
    coro: Coroutine[Any, Any, T],
    *,
    provider: LLMProvider | None = None,
    event_sink: EventSink | None = None,
    system: str | None = None,
    workspace: str | None = None,
    context_window: int | None = None,
) -> T:
    """Run an async workflow with a thorn execution context.

    This is the standard entry point for scripts.  It sets up the
    ``ExecutionContext`` (provider, event sink) and drives the given
    coroutine to completion.

    If *provider* is ``None``, one is loaded from environment variables
    via ``load_provider_from_env()``.

    *workspace* sets the workspace root for file-access rules.  When
    ``None``, falls back to the resolved current working directory.
    """
    from pathlib import Path

    from thorn.core._file_access import load_global_ignores
    from thorn.core._context import reset_context

    if provider is None:
        provider = load_provider_from_env()
    if event_sink is None:
        event_sink = NullEventSink()

    ws_root = Path(workspace).resolve() if workspace else Path.cwd().resolve()
    global_ignores = load_global_ignores(ws_root)

    system_prompts: list[str] = []
    if system:
        system_prompts.append(system)

    effective_cw = _effective_context_window(
        provider.context_window, context_window,
    )

    ctx = ExecutionContext(
        provider=provider,
        event_sink=event_sink,
        system_prompts=system_prompts,
        workspace_root=ws_root,
        global_ignores=global_ignores,
        context_window=effective_cw,
        agency_root_directory=ws_root,
    )

    async def _run_with_context() -> T:
        token = set_context(ctx)
        try:
            return await coro
        finally:
            reset_context(token)

    return asyncio.run(_run_with_context())
