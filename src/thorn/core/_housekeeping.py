"""Harness-driven history housekeeping.

When deterministic compaction (collapse/expand) cannot bring the context
within budget, the housekeeping flow trims old history after giving the
agent a chance to journal important information.

The flow, run inside ``run_agent_loop`` after every assistant turn:

1. Select a *cut point* — everything before it will be removed.
2. Insert a context-boundary marker at the cut point so the agent can
   see what is about to be lost.
3. Run a restricted sub-loop (journal + read tools only) where the
   agent writes anything important to its journal.
4. Trim: replace removed nodes with an ``ArchiveMarkerNode``, wrap
   the housekeeping interaction in a ``HousekeepingNode``.

The entire mechanism is transparent to callers of ``session.prompt()``
— it is plumbing, not porcelain.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from thorn.core._history import (
    ArchiveMarkerNode,
    HistoryNode,
    HistoryTree,
    HousekeepingNode,
    UserPromptNode,
)
from thorn.core._messages import UserMessage

if TYPE_CHECKING:
    from thorn.core._context import ExecutionContext
    from thorn.core._loop import _WrappedTool

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_HOUSEKEEPING_TOOL_ROUNDS: int = 5
"""Maximum tool-call rounds during the housekeeping sub-loop.

Generous enough for several ``write_journal`` calls plus a final
text acknowledgment.
"""

CONTEXT_BOUNDARY_TEXT: str = (
    "[CONTEXT BOUNDARY: Everything above this point will be removed "
    "from your message history after this turn. If any of that content "
    "is important, write it to your journal now.]"
)

HOUSEKEEPING_PROMPT: str = (
    "Your conversation history has grown large. A context boundary "
    "marker has been placed in the history above. Everything above "
    "that marker WILL be trimmed after this turn.\n\n"
    "Please:\n"
    "1. Review the content above the boundary marker.\n"
    "2. Use write_journal to save any important context that would "
    "otherwise be lost.\n"
    "3. If you have nothing to journal, simply acknowledge and continue."
)

_HOUSEKEEPING_TOOL_ALLOWLIST: frozenset[str] = frozenset({
    "write_journal",
    "read_journal",
    "read_file",
})
"""Tools available during housekeeping.

Journal tools let the agent persist important context before trimming;
``read_file`` allows reviewing journal or workspace files for
cross-referencing.
"""


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class HousekeepingResult:
    """Summary of a single housekeeping operation."""

    nodes_trimmed: int
    journal_date: str


# ---------------------------------------------------------------------------
# Cut-point selection
# ---------------------------------------------------------------------------

def select_cut_point(
    history: HistoryTree,
    protected: set[int],
) -> int | None:
    """Find the last node index (inclusive) that should be trimmed.

    Everything from index 0 through the returned index will be removed
    during the post-housekeeping trim.

    Returns ``None`` when nothing can be trimmed (history is empty or
    all nodes are protected).
    """
    if not history.nodes:
        return None

    if not protected:
        # Nothing is protected — the entire history could be trimmed,
        # but this edge case shouldn't arise in practice (the
        # protected-tail defaults guarantee non-empty protection).
        return len(history.nodes) - 1

    first_protected = min(protected)
    if first_protected == 0:
        return None

    return first_protected - 1


# ---------------------------------------------------------------------------
# Tool filtering
# ---------------------------------------------------------------------------

def filter_housekeeping_tools(
    tools: list[_WrappedTool],
) -> list[_WrappedTool]:
    """Restrict a tool list to the housekeeping allowlist."""
    return [
        t for t in tools
        if _tool_schema_name(t) in _HOUSEKEEPING_TOOL_ALLOWLIST
    ]


def _tool_schema_name(t: Any) -> str:
    return t.schema.get("function", {}).get("name", "")


# ---------------------------------------------------------------------------
# Core housekeeping flow
# ---------------------------------------------------------------------------

async def perform_housekeeping(
    *,
    context: ExecutionContext,
    history: HistoryTree,
    all_tools: list[_WrappedTool],
    system_prompts: list[str] | None,
) -> HousekeepingResult | None:
    """Execute one housekeeping cycle on *history*.

    Called by ``run_agent_loop`` when compaction was insufficient.  The
    history tree is mutated in place: old nodes are replaced by an
    ``ArchiveMarkerNode`` and the housekeeping interaction is wrapped
    in a ``HousekeepingNode``.

    Returns a ``HousekeepingResult`` on success, ``None`` if no trimming
    was possible (all nodes protected).
    """
    from thorn.core._loop import run_agent_loop
    from thorn.core.errors import LoopLimitError, SkillError

    protected = history._protected_indices()
    cut_index = select_cut_point(history, protected)
    if cut_index is None:
        logger.warning(
            "housekeeping: all %d nodes are protected; cannot trim",
            len(history.nodes),
        )
        return None

    nodes_to_trim = cut_index + 1

    # Snapshot the node count *before* we start mutating the tree.
    # After we insert the boundary marker, everything at
    # index >= (pre_hk_count + 1) was added by the sub-loop.
    pre_hk_count = len(history.nodes)

    # -- Insert boundary marker right after the last trimmed node -------
    # The agent will see all content above this marker during the
    # housekeeping turn, giving it the opportunity to journal anything
    # important before the trim.
    boundary_node = UserPromptNode(UserMessage(content=CONTEXT_BOUNDARY_TEXT))
    history.nodes.insert(cut_index + 1, boundary_node)

    # -- Run the restricted sub-loop ------------------------------------
    hk_tools = filter_housekeeping_tools(all_tools)

    try:
        await run_agent_loop(
            context=context,
            user_prompt=HOUSEKEEPING_PROMPT,
            tools=hk_tools,
            system_prompts=system_prompts,
            max_tool_rounds=MAX_HOUSEKEEPING_TOOL_ROUNDS,
            history=history,
            _housekeeping=True,
        )
    except (LoopLimitError, SkillError) as exc:
        logger.info(
            "housekeeping sub-loop ended early (%s); proceeding with trim",
            exc,
        )

    # -- Post-housekeeping surgery on the tree --------------------------
    #
    # Layout after the sub-loop (indices are into history.nodes):
    #
    #   [0 .. cut_index]                    trimmed nodes (original)
    #   [cut_index + 1]                     boundary marker (inserted)
    #   [cut_index + 2 .. pre_hk_count]     protected tail (shifted +1)
    #   [pre_hk_count + 1 ..]               hk prompt + agent turn(s)
    #
    # Target layout:
    #   [ArchiveMarkerNode, <protected tail>, HousekeepingNode]

    now = datetime.now(timezone.utc)
    journal_date = now.strftime("%Y-%m-%d")

    archive_marker = ArchiveMarkerNode(
        archived_at=now,
        summary=f"Archived {nodes_to_trim} history nodes",
        node_count=nodes_to_trim,
        journal_date=journal_date,
    )

    protected_start = cut_index + 2
    # pre_hk_count was measured *before* the boundary insert, so the
    # original last node is now at index pre_hk_count (shifted +1).
    # The protected tail spans [protected_start .. pre_hk_count].
    protected_end = pre_hk_count + 1
    protected_tail: list[HistoryNode] = list(
        history.nodes[protected_start:protected_end]
    )

    hk_additions_start = protected_end
    inner_hk_nodes: list[HistoryNode] = (
        [boundary_node] + list(history.nodes[hk_additions_start:])
    )
    hk_node = HousekeepingNode(inner_nodes=inner_hk_nodes)

    history.nodes = [archive_marker] + protected_tail + [hk_node]

    await context.event_sink.on_status(
        f"housekeeping: trimmed {nodes_to_trim} nodes, "
        f"journal date {journal_date}",
        scope=context.scope,
    )

    return HousekeepingResult(
        nodes_trimmed=nodes_to_trim,
        journal_date=journal_date,
    )
