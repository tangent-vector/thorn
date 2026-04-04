"""Salience-based context injection for sub-agent spawning.

Pre-populates a child agent's HistoryTree with synthetic tool-call entries
ranked by salience and packed into a token budget, eliminating the bootstrap
cost of re-discovering project structure.

Three salience sources contribute scored seed items:

- **Source 1** (weight 1.0): agent-declared structural seeds via
  ``context_seed_items()``.
- **Source 2** (weight 0.5): heuristic path/identifier extraction from the
  task prompt text.  Only active for sub-agents.
- **Source 3** (weight 0.1): typed ``ToolCallNode`` instances found in the
  parent agent's history.  Only active for sub-agents.

Scoring proceeds via per-source normalization, cross-source weighted sum,
threshold cutoff, and greedy budget packing.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from thorn._history import (
    DirectoryListCallNode,
    FileReadCallNode,
    ToolCallNode,
    TurnNode,
)
from thorn._messages import (
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

    Frozen so instances can be used as dict keys and merged across
    salience sources.
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
# Prompt text analysis (Source 2)
# ---------------------------------------------------------------------------

_BACKTICK_RE = re.compile(r"`([^`]+)`")
_QUOTED_RE = re.compile(r'"([^"]+)"')

_FILE_EXT_RE = re.compile(
    r"(?<!\w)"
    r"([\w./\\-]+\.(?:h|cpp|c|py|txt|md|json|yaml|yml|toml|cfg|rs|go|java|ts|js))"
    r"(?!\w)",
)

_SLASH_PATH_RE = re.compile(
    r"(?<!\w)"
    r"([\w][\w./-]*(?:/|\\)[\w./-]*[\w])"
    r"(?!\w)",
)


def _looks_like_path(token: str) -> bool:
    """Heuristic: does *token* look like a filesystem path?"""
    if "/" in token or "\\" in token:
        return True
    extensions = (
        ".h", ".cpp", ".c", ".py", ".txt", ".md", ".json",
        ".yaml", ".yml", ".toml", ".cfg", ".rs", ".go",
        ".java", ".ts", ".js",
    )
    return any(token.endswith(ext) for ext in extensions)


def _classify_token(
    token: str,
    workspace: Path | None,
) -> SeedContent | None:
    """Classify a text token as a FileSeed, DirectorySeed, or SearchSeed.

    Tries resolving against the workspace first so that tokens matching
    real filesystem entries are correctly classified even when they lack
    obvious path syntax (e.g. a directory name without slashes).
    """
    if workspace is not None:
        resolved = workspace / token
        if resolved.is_file():
            return FileSeed(path=str(resolved))
        if resolved.is_dir():
            return DirectorySeed(path=str(resolved))

    if _looks_like_path(token):
        return FileSeed(path=token)

    if len(token) >= 2:
        return SearchSeed(query=token)

    return None


def extract_seeds_from_prompt(
    text: str,
    workspace: Path | None,
) -> dict[SeedContent, float]:
    """Extract path references and identifiers from a task prompt.

    Scans backtick-delimited content, double-quoted strings, and bare
    path-like substrings.  Returns ``FileSeed`` for paths (salience 1.0)
    and ``SearchSeed`` for other identifiers (salience 0.5).
    """
    seeds: dict[SeedContent, float] = {}
    seen: set[str] = set()

    for pattern in (_BACKTICK_RE, _QUOTED_RE, _FILE_EXT_RE, _SLASH_PATH_RE):
        for match in pattern.finditer(text):
            token = match.group(1).strip()
            if not token or token in seen:
                continue
            seen.add(token)
            seed = _classify_token(token, workspace)
            if seed is None:
                continue
            salience = 1.0 if isinstance(seed, (FileSeed, DirectorySeed)) else 0.5
            seeds.setdefault(seed, salience)

    return seeds


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

SCORE_THRESHOLD: float = 0.09
"""Minimum composite score for inclusion.

Set just below the Source 3 weight (0.1) so items sourced *only*
from parent state almost never pass on their own.
"""


def normalize_scores(
    scores: dict[SeedContent, float],
) -> dict[SeedContent, float]:
    """Normalize values to sum to 1.0.

    Returns an empty dict if the input is empty or all-zero.
    """
    total = sum(scores.values())
    if total <= 0:
        return {}
    return {k: v / total for k, v in scores.items()}


def merge_sources(
    sources: list[tuple[dict[SeedContent, float], float]],
) -> dict[SeedContent, float]:
    """Compute per-item weighted sum across sources with threshold cutoff.

    Each entry in *sources* is ``(per_source_scores, source_weight)``.
    Per-source scores are normalized to sum to 1.0, then multiplied by
    the weight and summed across sources.  Items appearing in multiple
    sources naturally get boosted.  Items below ``SCORE_THRESHOLD`` are
    excluded.
    """
    merged: dict[SeedContent, float] = {}

    for raw_scores, weight in sources:
        normalized = normalize_scores(raw_scores)
        for item, score in normalized.items():
            merged[item] = merged.get(item, 0.0) + score * weight

    return {k: v for k, v in merged.items() if v >= SCORE_THRESHOLD}


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
    ranked_items: dict[SeedContent, float],
    token_budget: int,
    workspace: Path | None = None,
) -> TurnNode | None:
    """Build a synthetic assistant turn from scored seed items.

    For each item in descending score order, calls the appropriate tool
    function to source content (respecting the active file-access
    policy).  Failed tool calls are silently dropped.  Stops when the
    token budget is exhausted.

    Returns a ``TurnNode`` containing all successful ``ToolCallNode``
    instances, marked with low intrinsic salience.  The caller is
    responsible for placing the actual user prompt node before this
    turn in the history.  Returns ``None`` if no items could be sourced.
    """
    if not ranked_items or token_budget <= 0:
        return None

    sorted_items = sorted(
        ranked_items.items(), key=lambda kv: kv[1], reverse=True,
    )

    tool_call_nodes: list[ToolCallNode] = []
    tokens_used = 0

    for idx, (seed, _score) in enumerate(sorted_items):
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
    from thorn._tools import list_directory, read_file, search_files

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
