"""Hierarchical agent history with expand/collapse semantics.

Provides a tree-structured representation of conversation history that
supports watermark-based automatic compaction via selective collapse.
Each node in the tree can be expanded (showing full content) or collapsed
(showing a short summary), enabling a greedy algorithm to reduce token
usage when the context window fills up.

See ``docs/ideas/hierarchical-context-management.md`` for the broader
vision this module works toward.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable

from thorn._messages import (
    AssistantMessage,
    Message,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)


# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------

CHARS_PER_TOKEN: int = 4
"""Heuristic ratio for rough token estimation.  Only used for *relative*
comparisons when ranking collapse candidates; absolute threshold checks
use real provider-reported usage."""

LONG_CONTENT_THRESHOLD: int = 2000
"""Character count above which a user or assistant message is considered
"long" and eligible for truncation when collapsed."""

TRUNCATED_PREFIX_CHARS: int = 500
"""Characters to retain when truncating long content."""

SUMMARY_CONTENT_PREFIX_CHARS: int = 200
"""Characters to retain from assistant content when generating a turn summary."""

ABBREVIATED_ARG_VALUE_MAX_LEN: int = 60
"""Maximum character length for individual argument values when
abbreviating tool call arguments for detail-collapsed rendering."""

DEFAULT_INTRINSIC_SALIENCE: float = 1.0
"""Baseline salience for user prompts and assistant turns."""

DEFAULT_TOOL_CALL_SALIENCE: float = 0.8
"""Tool calls are slightly lower salience than the surrounding turn
because their results are often the bulkiest, most compressible content."""

DEFAULT_SALIENCE_DECAY_RATE: float = 0.1
"""Exponential decay rate applied per node-position of age."""

DEFAULT_HIGH_WATERMARK: float = 0.8
"""Fraction of context window that triggers compaction."""

DEFAULT_LOW_WATERMARK: float = 0.6
"""Target fraction of context window after compaction."""


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class CollapseState(Enum):
    """Rendering state of a top-level history node."""
    EXPANDED = "expanded"
    COLLAPSED = "collapsed"


class _CandidateKind(Enum):
    """Type of collapse action in the greedy compaction algorithm."""
    TOOL_CALL = "tool_call"
    TURN = "turn"
    USER_PROMPT = "user_prompt"


# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------

def estimate_tokens(text: str) -> int:
    """Rough character-based token estimate for ranking purposes.

    Monotonic with actual token count, which is all the greedy compaction
    algorithm requires.  Absolute threshold checks use real provider
    usage instead.
    """
    return max(1, len(text) // CHARS_PER_TOKEN)


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

def _truncate_content(
    content: str,
    prefix_chars: int = TRUNCATED_PREFIX_CHARS,
) -> str:
    """Truncate long content to a prefix with an omission indicator."""
    if len(content) <= prefix_chars:
        return content
    remaining = len(content) - prefix_chars
    return content[:prefix_chars] + f"\n[... {remaining} more characters]"


def _brief(text: str, max_len: int) -> str:
    """Shorten *text* to at most *max_len* characters."""
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def _first_arg_repr(args: dict[str, Any]) -> str:
    """Short repr of the first argument value, for default tool summaries."""
    if not args:
        return ""
    val = next(iter(args.values()))
    return repr(_brief(str(val), 40))


def _abbreviate_arguments(
    arguments: str,
    max_value_len: int = ABBREVIATED_ARG_VALUE_MAX_LEN,
) -> str:
    """Abbreviate JSON argument values for detail-collapsed rendering.

    Keeps argument structure (keys and short values intact) but shortens
    long string values and replaces large collections with length
    summaries.  The result is always valid JSON.
    """
    try:
        args = json.loads(arguments) if arguments else {}
    except json.JSONDecodeError:
        return _brief(arguments, max_value_len * 2)

    if not isinstance(args, dict):
        return _brief(arguments, max_value_len * 2)

    abbreviated: dict[str, Any] = {}
    for key, value in args.items():
        if isinstance(value, str) and len(value) > max_value_len:
            abbreviated[key] = _brief(value, max_value_len)
        elif isinstance(value, list):
            serialized = json.dumps(value)
            if len(serialized) > max_value_len:
                abbreviated[key] = f"[{len(value)} items]"
            else:
                abbreviated[key] = value
        elif isinstance(value, dict):
            serialized = json.dumps(value)
            if len(serialized) > max_value_len:
                abbreviated[key] = f"{{{len(value)} keys}}"
            else:
                abbreviated[key] = value
        else:
            abbreviated[key] = value

    return json.dumps(abbreviated)


# ---------------------------------------------------------------------------
# Tool-call summary generation (mechanical, no LLM)
# ---------------------------------------------------------------------------

def _tool_call_summary(tool_call: ToolCall, result: ToolResultMessage) -> str:
    """Generate a deterministic one-line summary for a tool call.

    Per-tool handlers produce readable summaries; unknown tools fall back
    to ``tool_name(first_arg)``.
    """
    name = tool_call.name
    try:
        args = json.loads(tool_call.arguments) if tool_call.arguments else {}
    except json.JSONDecodeError:
        args = {}

    if result.is_error:
        first = _first_arg_repr(args)
        detail = _brief(result.content, 60)
        return f"{name}({first}) -> error: {detail}" if first else f"{name}() -> error: {detail}"

    match name:
        case "read_file":
            path = args.get("path", "?")
            line_count = result.content.count("\n") + 1
            return f"read_file('{path}') -> {line_count} lines"
        case "edit_file":
            path = args.get("path", "?")
            edits = args.get("edits", [])
            count = len(edits) if isinstance(edits, list) else "?"
            return f"edit_file('{path}') -> applied {count} edit(s)"
        case "create_file":
            return f"create_file('{args.get('path', '?')}')"
        case "delete_file":
            return f"delete_file('{args.get('path', '?')}')"
        case "move_file":
            return f"move_file('{args.get('source', '?')}' -> '{args.get('destination', '?')}')"
        case "search_files":
            return f"search_files('{args.get('pattern', '?')}') -> {_brief(result.content, 40)}"
        case "find_files":
            entries = [
                line for line in result.content.strip().split("\n")
                if line and not line.startswith("[")
            ]
            return f"find_files('{args.get('pattern', '?')}') -> {len(entries)} entries"
        case "list_directory":
            return f"list_directory('{args.get('path', '.')}')"
        case "run_shell":
            cmd = _brief(str(args.get("command", "?")), 40)
            return f"run_shell({cmd!r})"
        case "return_result":
            return "return_result(...)"
        case _:
            first = _first_arg_repr(args)
            return f"{name}({first})" if first else f"{name}()"


# ---------------------------------------------------------------------------
# Node types
# ---------------------------------------------------------------------------

class ToolCallNode:
    """A single tool call within a turn.

    Owns the ``ToolCall`` (request) and ``ToolResultMessage`` (response).
    Can be *detail-collapsed*, which replaces the result content with a
    short mechanical summary while keeping the tool call itself visible
    in the rendered history.
    """

    __slots__ = (
        "tool_call",
        "result",
        "detail_collapsed",
        "intrinsic_salience",
        "_summary",
        "_expanded_cost",
        "_collapsed_cost",
        "_abbreviated_args",
    )

    def __init__(
        self,
        tool_call: ToolCall,
        result: ToolResultMessage,
        *,
        intrinsic_salience: float = DEFAULT_TOOL_CALL_SALIENCE,
    ) -> None:
        self.tool_call = tool_call
        self.result = result
        self.detail_collapsed = False
        self.intrinsic_salience = intrinsic_salience
        self._summary: str | None = None
        self._expanded_cost: int | None = None
        self._collapsed_cost: int | None = None
        self._abbreviated_args: str | None = None

    def summary(self) -> str:
        if self._summary is None:
            self._summary = _tool_call_summary(self.tool_call, self.result)
        return self._summary

    def expanded_token_cost(self) -> int:
        if self._expanded_cost is None:
            self._expanded_cost = (
                estimate_tokens(self.tool_call.name)
                + estimate_tokens(self.tool_call.arguments)
                + estimate_tokens(self.result.content)
            )
        return self._expanded_cost

    def abbreviated_arguments(self) -> str:
        """Shortened version of the tool call arguments for collapsed rendering."""
        if self._abbreviated_args is None:
            self._abbreviated_args = _abbreviate_arguments(self.tool_call.arguments)
        return self._abbreviated_args

    def collapsed_token_cost(self) -> int:
        if self._collapsed_cost is None:
            self._collapsed_cost = (
                estimate_tokens(self.tool_call.name)
                + estimate_tokens(self.abbreviated_arguments())
                + estimate_tokens(self.summary())
            )
        return self._collapsed_cost

    def token_cost(self) -> int:
        if self.detail_collapsed:
            return self.collapsed_token_cost()
        return self.expanded_token_cost()

    def savings_if_detail_collapsed(self) -> int:
        return max(0, self.expanded_token_cost() - self.collapsed_token_cost())

    @property
    def is_collapsible(self) -> bool:
        return not self.detail_collapsed and self.savings_if_detail_collapsed() > 0

    def render_tool_call(self) -> ToolCall:
        """The ToolCall for inclusion in the parent AssistantMessage.

        When detail-collapsed, arguments are abbreviated to reduce token
        usage while keeping the tool call structurally valid.
        """
        if self.detail_collapsed:
            return ToolCall(
                call_id=self.tool_call.call_id,
                name=self.tool_call.name,
                arguments=self.abbreviated_arguments(),
            )
        return self.tool_call

    def render_result(self) -> ToolResultMessage:
        """The result message; content is replaced with summary when detail-collapsed."""
        if self.detail_collapsed:
            return ToolResultMessage(
                call_id=self.result.call_id,
                content=self.summary(),
            )
        return self.result


class UserPromptNode:
    """A user prompt in the history.

    Short prompts are leaves with no meaningful collapsed form.  Long
    prompts (above ``LONG_CONTENT_THRESHOLD``) can be collapsed to a
    truncated prefix.
    """

    __slots__ = (
        "message",
        "collapse_state",
        "intrinsic_salience",
        "_expanded_cost",
        "_collapsed_cost",
    )

    def __init__(
        self,
        message: UserMessage,
        *,
        intrinsic_salience: float = DEFAULT_INTRINSIC_SALIENCE,
    ) -> None:
        self.message = message
        self.collapse_state = CollapseState.EXPANDED
        self.intrinsic_salience = intrinsic_salience
        self._expanded_cost: int | None = None
        self._collapsed_cost: int | None = None

    @property
    def is_long(self) -> bool:
        return len(self.message.content) > LONG_CONTENT_THRESHOLD

    def summary(self) -> str:
        return _truncate_content(self.message.content)

    def expanded_token_cost(self) -> int:
        if self._expanded_cost is None:
            self._expanded_cost = estimate_tokens(self.message.content)
        return self._expanded_cost

    def collapsed_token_cost(self) -> int:
        if self._collapsed_cost is None:
            self._collapsed_cost = estimate_tokens(self.summary())
        return self._collapsed_cost

    def token_cost(self) -> int:
        if self.collapse_state == CollapseState.COLLAPSED:
            return self.collapsed_token_cost()
        return self.expanded_token_cost()

    def savings_if_collapsed(self) -> int:
        return max(0, self.expanded_token_cost() - self.collapsed_token_cost())

    @property
    def is_collapsible(self) -> bool:
        return self.collapse_state == CollapseState.EXPANDED and self.is_long

    def render(self) -> list[Message]:
        if self.collapse_state == CollapseState.COLLAPSED:
            return [UserMessage(content=self.summary())]
        return [self.message]


class TurnNode:
    """An assistant turn: the assistant's text plus zero or more tool calls.

    Stores the assistant's text content separately from tool call nodes
    so that individual tool calls can be detail-collapsed independently.
    When the entire turn is collapsed, it renders as a single text-only
    ``AssistantMessage`` with a summary.
    """

    __slots__ = (
        "assistant_content",
        "tool_call_nodes",
        "collapse_state",
        "intrinsic_salience",
        "_collapsed_cost",
        "_summary",
    )

    def __init__(
        self,
        assistant_content: str,
        tool_call_nodes: list[ToolCallNode],
        *,
        intrinsic_salience: float = DEFAULT_INTRINSIC_SALIENCE,
    ) -> None:
        self.assistant_content = assistant_content
        self.tool_call_nodes = tool_call_nodes
        self.collapse_state = CollapseState.EXPANDED
        self.intrinsic_salience = intrinsic_salience
        self._collapsed_cost: int | None = None
        self._summary: str | None = None

    def summary(self) -> str:
        if self._summary is None:
            parts: list[str] = []
            if self.assistant_content:
                parts.append(_truncate_content(self.assistant_content, SUMMARY_CONTENT_PREFIX_CHARS))
            for tcn in self.tool_call_nodes:
                parts.append(tcn.summary())
            self._summary = " | ".join(parts) if parts else "[empty turn]"
        return self._summary

    def token_cost(self) -> int:
        if self.collapse_state == CollapseState.COLLAPSED:
            return self._get_collapsed_cost()
        cost = estimate_tokens(self.assistant_content)
        for tcn in self.tool_call_nodes:
            cost += tcn.token_cost()
        return cost

    def _get_collapsed_cost(self) -> int:
        if self._collapsed_cost is None:
            self._collapsed_cost = estimate_tokens(self.summary())
        return self._collapsed_cost

    def savings_if_collapsed(self) -> int:
        """Token savings from collapsing the entire turn in its current state.

        If individual tool calls have already been detail-collapsed, the
        remaining savings are smaller.
        """
        return max(0, self.token_cost() - self._get_collapsed_cost())

    @property
    def is_collapsible(self) -> bool:
        return self.collapse_state == CollapseState.EXPANDED

    @property
    def has_collapsible_tool_calls(self) -> bool:
        return any(tcn.is_collapsible for tcn in self.tool_call_nodes)

    def render(self) -> list[Message]:
        if self.collapse_state == CollapseState.COLLAPSED:
            return [AssistantMessage(content=self.summary())]

        tool_calls = [tcn.render_tool_call() for tcn in self.tool_call_nodes]
        results = [tcn.render_result() for tcn in self.tool_call_nodes]

        msg = AssistantMessage(
            content=self.assistant_content,
            tool_calls=tool_calls,
        )
        return [msg, *results]


# ---------------------------------------------------------------------------
# Type alias for top-level tree nodes
# ---------------------------------------------------------------------------

HistoryNode = UserPromptNode | TurnNode


# ---------------------------------------------------------------------------
# Compaction result
# ---------------------------------------------------------------------------

@dataclass
class CompactionResult:
    """Summary of a compaction operation, suitable for trace events."""
    nodes_collapsed: int
    tool_calls_detail_collapsed: int
    estimated_savings: int
    tokens_before: int
    tokens_after: int


# ---------------------------------------------------------------------------
# Internal: collapse candidate for the greedy algorithm
# ---------------------------------------------------------------------------

@dataclass
class _CollapseCandidate:
    """One possible collapse action, with its cost/benefit trade-off."""
    effective_salience: float
    estimated_savings: int
    kind: _CandidateKind
    parent_turn: TurnNode | None
    apply: Callable[[], None]

    def __lt__(self, other: _CollapseCandidate) -> bool:
        return self.effective_salience < other.effective_salience


# ---------------------------------------------------------------------------
# History tree
# ---------------------------------------------------------------------------

class HistoryTree:
    """Ordered sequence of history nodes with compaction support.

    The tree is the primary representation of an agent's conversation
    history.  It wraps the flat ``list[Message]`` representation,
    providing structure that the compaction algorithm can exploit.

    Typical lifecycle::

        tree.append_user_prompt("write some code")
        # ... agent loop calls tree.render() for each completion ...
        tree.append_turn(assistant_msg, tool_results)
        # ... check usage, maybe tree.compact(...) ...
    """

    __slots__ = ("nodes",)

    def __init__(self) -> None:
        self.nodes: list[HistoryNode] = []

    def append_user_prompt(
        self,
        content: str,
        *,
        intrinsic_salience: float = DEFAULT_INTRINSIC_SALIENCE,
    ) -> UserPromptNode:
        """Append a user prompt and return the new node."""
        node = UserPromptNode(
            UserMessage(content=content),
            intrinsic_salience=intrinsic_salience,
        )
        self.nodes.append(node)
        return node

    def append_turn(
        self,
        assistant_msg: AssistantMessage,
        tool_results: list[ToolResultMessage],
        *,
        intrinsic_salience: float = DEFAULT_INTRINSIC_SALIENCE,
    ) -> TurnNode:
        """Append an assistant turn (message + tool results) and return the node.

        Tool results are matched to tool calls by ``call_id``.  Tool
        calls with no matching result are silently skipped (this can
        happen if ``raise_error`` aborted before producing a result).
        """
        result_by_id = {r.call_id: r for r in tool_results}
        tool_call_nodes: list[ToolCallNode] = []
        for tc in assistant_msg.tool_calls:
            result = result_by_id.get(tc.call_id)
            if result is not None:
                tool_call_nodes.append(ToolCallNode(tc, result))

        node = TurnNode(
            assistant_content=assistant_msg.content,
            tool_call_nodes=tool_call_nodes,
            intrinsic_salience=intrinsic_salience,
        )
        self.nodes.append(node)
        return node

    def render(self) -> list[Message]:
        """Flatten the tree into a valid ``list[Message]`` for the provider.

        Respects each node's current collapse state: collapsed nodes
        emit summary messages, expanded nodes emit full content.  The
        result always satisfies LLM API sequencing constraints (proper
        role alternation, tool results only after their tool calls).
        """
        result: list[Message] = []
        for node in self.nodes:
            result.extend(node.render())
        return result

    def estimated_tokens(self) -> int:
        """Sum of character-heuristic token estimates across all nodes."""
        return sum(node.token_cost() for node in self.nodes)

    def compact(
        self,
        *,
        context_budget: int,
        low_watermark: float = DEFAULT_LOW_WATERMARK,
        overhead_tokens: int = 0,
        actual_prompt_tokens: int | None = None,
        decay_rate: float = DEFAULT_SALIENCE_DECAY_RATE,
    ) -> CompactionResult:
        """Run the greedy collapse algorithm to reduce token usage.

        Collapses nodes in order of ascending effective salience (oldest
        and least important first) until estimated usage drops below
        the target.  Individual tool call results are detail-collapsed
        before entire turns are collapsed, because tool calls within a
        still-expanded turn are lower-salience candidates than the turn
        itself.

        When *actual_prompt_tokens* is supplied (the real token count
        from the provider), the savings target is derived from the
        actual usage and then scaled into the estimated-token domain.
        This avoids systematic under-compaction when the character-based
        heuristic underestimates true token counts.

        Args:
            context_budget: Effective context window in tokens.
            low_watermark: Target usage fraction after compaction.
            overhead_tokens: Estimated fixed token cost outside the
                history (system prompts, tool schemas).
            actual_prompt_tokens: Real prompt token count from the
                provider, used to calibrate the savings target.
            decay_rate: Exponential decay rate for salience.

        Returns:
            A ``CompactionResult`` with actual (recomputed) savings.
        """
        est_total = self.estimated_tokens() + overhead_tokens
        target = int(context_budget * low_watermark)

        if actual_prompt_tokens is not None:
            actual_savings_needed = actual_prompt_tokens - target
            if actual_savings_needed <= 0:
                return CompactionResult(
                    nodes_collapsed=0,
                    tool_calls_detail_collapsed=0,
                    estimated_savings=0,
                    tokens_before=est_total,
                    tokens_after=est_total,
                )
            # Scale the real-token savings target into the estimated-
            # token domain so the greedy loop can compare apples to
            # apples.  The fraction of the prompt we need to shed is
            # the same regardless of the token-estimation method.
            reduction_frac = actual_savings_needed / actual_prompt_tokens
            savings_needed = max(1, int(est_total * reduction_frac))
        else:
            savings_needed = est_total - target
            if savings_needed <= 0:
                return CompactionResult(
                    nodes_collapsed=0,
                    tool_calls_detail_collapsed=0,
                    estimated_savings=0,
                    tokens_before=est_total,
                    tokens_after=est_total,
                )

        protected = self._protected_indices()
        candidates = self._build_candidates(protected, decay_rate)
        candidates.sort()

        nodes_collapsed = 0
        tcs_collapsed = 0
        est_savings = 0

        for c in candidates:
            if est_savings >= savings_needed:
                break

            if c.parent_turn is not None:
                if c.parent_turn.collapse_state == CollapseState.COLLAPSED:
                    continue

            c.apply()
            est_savings = est_total - (self.estimated_tokens() + overhead_tokens)

            if c.kind == _CandidateKind.TOOL_CALL:
                tcs_collapsed += 1
            else:
                nodes_collapsed += 1

        tokens_after = self.estimated_tokens() + overhead_tokens
        return CompactionResult(
            nodes_collapsed=nodes_collapsed,
            tool_calls_detail_collapsed=tcs_collapsed,
            estimated_savings=est_total - tokens_after,
            tokens_before=est_total,
            tokens_after=tokens_after,
        )

    # -- Internal helpers ---------------------------------------------------

    def _protected_indices(self) -> set[int]:
        """Indices of nodes that should not be collapsed.

        Protects the last two nodes (the most recent turn and its
        predecessor) and the most recent ``UserPromptNode`` (so the
        agent doesn't lose sight of what it was asked to do).
        """
        protected: set[int] = set()
        n = len(self.nodes)

        for i in range(max(0, n - 2), n):
            protected.add(i)

        for i in range(n - 1, -1, -1):
            if isinstance(self.nodes[i], UserPromptNode):
                protected.add(i)
                break

        return protected

    def _build_candidates(
        self,
        protected: set[int],
        decay_rate: float,
    ) -> list[_CollapseCandidate]:
        """Enumerate all possible collapse actions with their trade-offs."""
        candidates: list[_CollapseCandidate] = []
        total = len(self.nodes)

        for i, node in enumerate(self.nodes):
            if i in protected:
                continue

            age = total - 1 - i

            if isinstance(node, TurnNode):
                if node.collapse_state == CollapseState.COLLAPSED:
                    continue

                for tcn in node.tool_call_nodes:
                    if not tcn.is_collapsible:
                        continue
                    tc_eff = tcn.intrinsic_salience * math.exp(-decay_rate * age)
                    savings = tcn.savings_if_detail_collapsed()
                    if savings > 0:
                        def _do(t: ToolCallNode = tcn) -> None:
                            t.detail_collapsed = True
                        candidates.append(_CollapseCandidate(
                            effective_salience=tc_eff,
                            estimated_savings=savings,
                            kind=_CandidateKind.TOOL_CALL,
                            parent_turn=node,
                            apply=_do,
                        ))

                eff = node.intrinsic_salience * math.exp(-decay_rate * age)
                savings = node.savings_if_collapsed()
                if savings > 0:
                    def _do_turn(n: TurnNode = node) -> None:
                        n.collapse_state = CollapseState.COLLAPSED
                    # Bump by a small epsilon so tool-call collapses at the
                    # same age are tried first.
                    candidates.append(_CollapseCandidate(
                        effective_salience=eff + 0.001,
                        estimated_savings=savings,
                        kind=_CandidateKind.TURN,
                        parent_turn=None,
                        apply=_do_turn,
                    ))

            elif isinstance(node, UserPromptNode):
                if not node.is_collapsible:
                    continue
                eff = node.intrinsic_salience * math.exp(-decay_rate * age)
                savings = node.savings_if_collapsed()
                if savings > 0:
                    def _do_user(n: UserPromptNode = node) -> None:
                        n.collapse_state = CollapseState.COLLAPSED
                    candidates.append(_CollapseCandidate(
                        effective_salience=eff,
                        estimated_savings=savings,
                        kind=_CandidateKind.USER_PROMPT,
                        parent_turn=None,
                        apply=_do_user,
                    ))

        return candidates
