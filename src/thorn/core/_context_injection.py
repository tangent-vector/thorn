"""Context injection for agent bootstrapping.

Pre-populates an agent's HistoryTree with synthetic tool-call entries
packed into a token budget, eliminating the bootstrap cost of
re-discovering project structure.

Two sources contribute seed items, in priority order:

1. **Recommended context** passed explicitly by the caller (e.g. a parent
   agent delegating work).  Items preserve the caller's ordering.
2. **Role-declared seeds** via ``Agent.context_seed_items()``, sorted by
   declared salience.

Items from both sources are deduplicated and greedily packed into the
injection token budget.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from thorn.core._history import (
    DirectoryListCallNode,
    FileReadCallNode,
    ToolCallNode,
    TurnNode,
)
from thorn.core._messages import (
    ToolCall,
    ToolResultMessage,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Seed content types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SeedContent:
    """Base for hashable context seed content descriptors.

    Frozen so instances can be used as dict keys and deduplicated
    across sources.
    """


@dataclass(frozen=True)
class FileSeed(SeedContent):
    """Seed requesting a file read."""
    path: str


@dataclass(frozen=True)
class DirectorySeed(SeedContent):
    """Seed requesting a directory listing."""
    path: str


@dataclass(frozen=True)
class SearchSeed(SeedContent):
    """Seed requesting a codebase search."""
    query: str


# ---------------------------------------------------------------------------
# Injection budget
# ---------------------------------------------------------------------------

INJECTION_BUDGET_FRACTION: float = 0.2
"""Fraction of the context window to reserve for injected briefing."""


def injection_budget(context_window: int | None) -> int:
    """Compute the token budget for context injection.

    Returns 0 when *context_window* is unknown, disabling injection.
    """
    if context_window is None:
        return 0
    return int(context_window * INJECTION_BUDGET_FRACTION)


# ---------------------------------------------------------------------------
# Briefing assembly
# ---------------------------------------------------------------------------

BRIEFING_ASSISTANT_CONTENT = (
    "Let me look at the workspace for relevant context."
)

LOW_INJECTION_SALIENCE: float = 0.1
"""Intrinsic salience assigned to injected history nodes.

Kept low so that the compaction algorithm aggressively reclaims
injected content once the agent has produced its own history.
"""


async def assemble_briefing(
    items: list[SeedContent],
    token_budget: int,
    workspace: Path | None = None,
) -> TurnNode | None:
    """Build a synthetic assistant turn from an ordered list of seed items.

    Items are processed in the given order (caller controls priority).
    For each item, calls the appropriate tool function to source content
    (respecting the active file-access policy).  Failed tool calls are
    silently dropped.  Stops when the token budget is exhausted.

    Returns a ``TurnNode`` containing all successful ``ToolCallNode``
    instances, marked with low intrinsic salience.  The caller is
    responsible for placing the actual user prompt node before this
    turn in the history.  Returns ``None`` if no items could be sourced.
    """
    if not items or token_budget <= 0:
        return None

    tool_call_nodes: list[ToolCallNode] = []
    tokens_used = 0

    for idx, seed in enumerate(items):
        if tokens_used >= token_budget:
            break

        node = await _source_seed_item(
            seed, call_id=f"seed_{idx}", workspace=workspace,
        )
        if node is None:
            continue

        node_cost = node.expanded_token_cost()
        if tokens_used + node_cost > token_budget:
            break

        tool_call_nodes.append(node)
        tokens_used += node_cost

    if not tool_call_nodes:
        return None

    return TurnNode(
        assistant_content=BRIEFING_ASSISTANT_CONTENT,
        tool_call_nodes=tool_call_nodes,
        intrinsic_salience=LOW_INJECTION_SALIENCE,
    )


async def _source_seed_item(
    seed: SeedContent,
    call_id: str,
    workspace: Path | None,
) -> ToolCallNode | None:
    """Call the appropriate tool function to source a single seed item.

    Returns the ``ToolCallNode`` (using the appropriate subclass for
    ``isinstance``-based identification) or ``None`` if the tool call
    failed.
    """
    from thorn.core._tools import list_directory, read_file, search_files

    try:
        if isinstance(seed, FileSeed):
            args = {"path": seed.path}
            content = await read_file(seed.path)
            return FileReadCallNode(
                ToolCall(
                    call_id=call_id,
                    name="read_file",
                    arguments=json.dumps(args),
                ),
                ToolResultMessage(call_id=call_id, content=content),
                intrinsic_salience=LOW_INJECTION_SALIENCE,
            )

        if isinstance(seed, DirectorySeed):
            args = {"path": seed.path}
            content = await list_directory(seed.path)
            return DirectoryListCallNode(
                ToolCall(
                    call_id=call_id,
                    name="list_directory",
                    arguments=json.dumps(args),
                ),
                ToolResultMessage(call_id=call_id, content=content),
                intrinsic_salience=LOW_INJECTION_SALIENCE,
            )

        if isinstance(seed, SearchSeed):
            search_path = str(workspace) if workspace else "."
            args = {"pattern": seed.query, "path": search_path}
            content = await search_files(seed.query, path=search_path)
            return ToolCallNode(
                ToolCall(
                    call_id=call_id,
                    name="search_files",
                    arguments=json.dumps(args),
                ),
                ToolResultMessage(call_id=call_id, content=content),
                intrinsic_salience=LOW_INJECTION_SALIENCE,
            )

    except Exception:
        logger.debug("seed item %s failed, skipping", seed, exc_info=True)
        return None

    return None
