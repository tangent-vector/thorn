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
from typing import TYPE_CHECKING, Any, Coroutine, TypeVar

from thorn import tools
from thorn.core import (
    Agent,
    AgentFailureError,
    LoopLimitError,
    ProviderError,
    ProviderUnavailableError,
    RateLimitError,
    SkillError,
    ThornError,
    TransientProviderError,
    prompt,
    skill,
    tool,
)

if TYPE_CHECKING:
    from thorn.core import EventSink, LLMProvider

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
    "Agent",
    "prompt",
    "skill",
    "tool",
    "run",
    "tools",
    "ThornError",
    "SkillError",
    "ProviderError",
    "ProviderUnavailableError",
    "RateLimitError",
    "TransientProviderError",
    "LoopLimitError",
    "AgentFailureError",
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

    from thorn.core import (
        ExecutionContext,
        NullEventSink,
        load_provider_from_env,
        set_context,
    )
    from thorn.core._context import reset_context
    from thorn.core._file_access import load_global_ignores

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
