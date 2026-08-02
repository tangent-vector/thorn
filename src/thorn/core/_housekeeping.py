"""Harness-driven history housekeeping.

Two distinct flavours of housekeeping live here:

1. **Context-compaction housekeeping** (:func:`perform_housekeeping`).
   Runs *inside* ``run_agent_loop`` after a normal assistant turn when
   deterministic compaction (collapse/expand) cannot bring the context
   within budget.  Walks through: select a cut point, insert a
   boundary marker, run a restricted sub-loop where the agent can
   journal anything worth keeping, then trim everything above the
   cut.  Transparent to callers of ``session.prompt()`` -- plumbing,
   not porcelain.

2. **Shutdown housekeeping**
   (:func:`perform_shutdown_housekeeping`).  Runs *outside* the agent
   loop when a session is about to be discarded (today: CLI exit).
   Gives the agent one last turn to write anything important to
   journal or memory before the session goes away.  No cut point, no
   history surgery -- the history is being thrown away anyway.  See
   the aspirational architecture doc's "CLI Sessions" section for
   the motivation: *"When the CLI app exits, the session is always
   prompted to perform housekeeping and migrate all relevant
   information out to the journal or durable memory."*
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
    from thorn.core._session import Session

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


# ---------------------------------------------------------------------------
# Shutdown housekeeping
# ---------------------------------------------------------------------------

SHUTDOWN_HOUSEKEEPING_PROMPT: str = (
    "This session is about to end and will be discarded. Before that "
    "happens, review the conversation above and write down anything "
    "worth remembering in your journal.\n\n"
    "Good candidates for a journal entry include:\n"
    "- Decisions that were made or work that was completed.\n"
    "- Open threads or follow-ups that a future session would want "
    "to pick up on.\n"
    "- Observations about the user's preferences, the project, or "
    "the environment that aren't already captured in your memory.\n\n"
    "Use ``write_journal`` (and ``read_journal`` if you want to "
    "check for duplicates first) to record what matters.  If there "
    "is genuinely nothing notable, a brief acknowledgement that the "
    "session ended uneventfully is fine."
)
"""Prompt shown to the agent at session shutdown.

Deliberately low-pressure: we want the agent to write *something* in
the useful case but not feel compelled to manufacture entries when
the session was trivial.  The exact wording is tuned for the base
``Agent`` role which has ``write_journal`` / ``read_journal`` as
default role tools; specialised agents that override those tools
will need to adapt the prompt (out of scope for Phase 5).
"""


async def perform_shutdown_housekeeping(
    session: "Session",
    *,
    extra_system: str | None = None,
) -> None:
    """Run one final journaling turn before *session* is discarded.

    Intended for CLI exit: ``thorn chat`` invokes this after the REPL
    loop ends so the agent has a chance to migrate anything valuable
    from the session's volatile history into its durable journal.
    The session itself can still be persisted afterwards via
    ``save_session`` -- shutdown housekeeping mutates only the
    in-memory history to record the housekeeping turn itself; it
    doesn't trim or surgically modify the tree like
    :func:`perform_housekeeping` does.

    Why this is a separate function from :func:`perform_housekeeping`
    rather than a flag on it:

    - Context-compaction housekeeping's whole point is the trim; it
      selects a cut point, inserts a boundary marker, and reshapes
      history around it.  Shutdown housekeeping has no trim step --
      the history goes in the bin on CLI exit regardless -- so the
      cut-point and ``HousekeepingNode`` machinery would be dead
      weight here.
    - The two flows also want different prompts (*"context is full"*
      vs. *"session is about to end"*) and different call sites
      (inside ``run_agent_loop`` vs. around a ``session.prompt`` call
      at REPL shutdown).  Sharing the function would force the caller
      to know which half to use.

    Behaviour:

    - Invokes ``session.prompt(SHUTDOWN_HOUSEKEEPING_PROMPT,
      tools=[], system=extra_system)``.  Passing ``tools=[]`` strips
      whatever CLI-default extras the REPL was running with (shell,
      git, file writing) so the agent's tool surface shrinks to the
      base ``Agent`` role tools -- which for the CLI's local agent
      is just ``write_journal`` / ``read_journal`` / inbox tools.
      Restricting further (e.g. to the
      :data:`_HOUSEKEEPING_TOOL_ALLOWLIST` used by compaction) would
      require a parallel code path around ``run_agent_loop``
      bypassing ``session.prompt``'s memory/journal/AGENTS.md
      injection; not worth it for Phase 5.
    - All expected exceptions (``SkillError``, ``ThornError``,
      ``LoopLimitError``) are caught and logged at ``INFO`` level.
      Shutdown housekeeping running into trouble must never block
      CLI exit -- the user has already closed their terminal input
      and is waiting for the process to finish.
    - :class:`asyncio.CancelledError` and other ``BaseException``
      subclasses propagate, so ``Ctrl+C`` during housekeeping still
      tears the session down rather than being silently swallowed.
    - Sessions with an empty history are short-circuited: there is
      nothing for the agent to journal, and asking it to do so would
      only burn a provider round.  This is the ``thorn chat`` EOF-at-
      the-prompt path, which is otherwise common enough to care about.
    """
    from thorn.core.errors import LoopLimitError, SkillError, ThornError

    if not session._history.nodes:
        logger.debug("shutdown housekeeping: empty history, skipping")
        return

    try:
        await session.prompt(
            SHUTDOWN_HOUSEKEEPING_PROMPT,
            tools=[],
            system=extra_system,
        )
    except (SkillError, ThornError, LoopLimitError) as exc:
        logger.info("shutdown housekeeping ended early: %s", exc)
