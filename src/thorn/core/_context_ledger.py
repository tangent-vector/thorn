"""Request-local provider-history projection and accounting.

The stored :class:`~thorn.core._history.HistoryTree` remains the durable audit
record.  This module derives one immutable provider view for a logical request,
including mandatory stale/redundant observation replacement and a soft history
budget.  Provider retries reuse the resulting messages and ledger.
"""

from __future__ import annotations

import json
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Sequence

from thorn.core._file_context_folding import (
    LiveWorkspaceFileSnapshot,
    SearchObservationReplacementPolicy,
    plan_file_observation_replacements,
    search_observation_call_ids_requiring_raw_evidence,
)
from thorn.core._history import (
    DEFAULT_PROTECTED_TAIL_NODES,
    DEFAULT_PROTECTED_TAIL_TOOL_CALLS,
    DEFAULT_SALIENCE_DECAY_RATE,
    CollapseState,
    FileObservationFoldingMode,
    HistoryRenderPlan,
    HistoryTree,
    ToolCallNode,
    ToolResultReplacement,
    ToolResultReplacementReason,
    TurnNode,
    estimate_tokens,
)
from thorn.core._messages import Message
from thorn.core._prompt_visibility import PromptVisibilitySnapshot


@dataclass(frozen=True, order=True)
class EstimatedTokenCount:
    """A non-negative heuristic token count."""

    value: int

    def __post_init__(self) -> None:
        if self.value < 0:
            raise ValueError("estimated token count must be >= 0")


@dataclass(frozen=True, order=True)
class ContextWindowFraction:
    """A fraction of a model context window in the interval ``(0, 1]``."""

    value: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.value) or not 0.0 < self.value <= 1.0:
            raise ValueError("context window fraction must be in (0, 1]")


class HistoryBudgetSource(StrEnum):
    """Which threshold determined a request history budget."""

    FIXED_DEFAULT = "fixed_default"
    RELATIVE_CONTEXT_WINDOW = "relative_context_window"
    FIXED_MAXIMUM = "fixed_maximum"


@dataclass(frozen=True)
class ResolvedHistoryBudget:
    """Effective soft history and optional hard total-prompt limits."""

    history_tokens: EstimatedTokenCount
    source: HistoryBudgetSource
    hard_prompt_tokens: EstimatedTokenCount | None

    def to_json(self) -> dict[str, Any]:
        return {
            "history_tokens": self.history_tokens.value,
            "source": self.source.value,
            "hard_prompt_tokens": (
                self.hard_prompt_tokens.value
                if self.hard_prompt_tokens is not None else None
            ),
        }


@dataclass(frozen=True)
class ContextBudgetPolicy:
    """Typed controls for proactive provider-history projection."""

    default_history_tokens: EstimatedTokenCount
    maximum_history_tokens: EstimatedTokenCount
    soft_context_window_fraction: ContextWindowFraction
    hard_context_window_fraction: ContextWindowFraction
    search_replacement_policy: SearchObservationReplacementPolicy = (
        SearchObservationReplacementPolicy.EXACT_DUPLICATE_ONLY
    )

    def __post_init__(self) -> None:
        if self.default_history_tokens > self.maximum_history_tokens:
            raise ValueError(
                "default history budget must not exceed maximum history budget",
            )
        if (
            self.soft_context_window_fraction
            > self.hard_context_window_fraction
        ):
            raise ValueError(
                "soft context fraction must not exceed hard context fraction",
            )

    def resolve(
        self,
        *,
        context_window: int | None,
        estimated_overhead_tokens: EstimatedTokenCount,
    ) -> ResolvedHistoryBudget:
        """Resolve a history budget for one current request surface."""
        if context_window is None:
            return ResolvedHistoryBudget(
                history_tokens=self.default_history_tokens,
                source=HistoryBudgetSource.FIXED_DEFAULT,
                hard_prompt_tokens=None,
            )
        if context_window <= 0:
            raise ValueError("context_window must be > 0 when provided")

        relative_total = int(
            context_window * self.soft_context_window_fraction.value,
        )
        relative_history = EstimatedTokenCount(max(
            0,
            relative_total - estimated_overhead_tokens.value,
        ))
        if relative_history > self.maximum_history_tokens:
            history_tokens = self.maximum_history_tokens
            source = HistoryBudgetSource.FIXED_MAXIMUM
        else:
            history_tokens = relative_history
            source = HistoryBudgetSource.RELATIVE_CONTEXT_WINDOW
        return ResolvedHistoryBudget(
            history_tokens=history_tokens,
            source=source,
            hard_prompt_tokens=EstimatedTokenCount(int(
                context_window * self.hard_context_window_fraction.value,
            )),
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "default_history_tokens": self.default_history_tokens.value,
            "maximum_history_tokens": self.maximum_history_tokens.value,
            "soft_context_window_fraction": (
                self.soft_context_window_fraction.value
            ),
            "hard_context_window_fraction": (
                self.hard_context_window_fraction.value
            ),
            "search_replacement_policy": self.search_replacement_policy.value,
        }


DEFAULT_CONTEXT_BUDGET_POLICY = ContextBudgetPolicy(
    default_history_tokens=EstimatedTokenCount(12_000),
    maximum_history_tokens=EstimatedTokenCount(24_000),
    soft_context_window_fraction=ContextWindowFraction(0.60),
    hard_context_window_fraction=ContextWindowFraction(0.80),
)

BOUNDED_HISTORY_V2_CONTEXT_BUDGET_POLICY = ContextBudgetPolicy(
    default_history_tokens=EstimatedTokenCount(12_000),
    maximum_history_tokens=EstimatedTokenCount(24_000),
    soft_context_window_fraction=ContextWindowFraction(0.60),
    hard_context_window_fraction=ContextWindowFraction(0.80),
    search_replacement_policy=(
        SearchObservationReplacementPolicy.COMPLETE_CURRENT_READ
    ),
)


class ProviderHistoryDisposition(StrEnum):
    """One provider-view transformation recorded by the ledger."""

    STALE_FILE_OBSERVATION = "stale_file_observation"
    REDUNDANT_FILE_OBSERVATION = "redundant_file_observation"
    FILE_CONTEXT_FOLDED = "file_context_folded"
    DETAIL_COLLAPSED = "detail_collapsed"
    TURN_COLLAPSED = "turn_collapsed"


@dataclass(frozen=True)
class ProviderHistoryTokenDelta:
    """Marginal provider-history token change from one projection step."""

    before: EstimatedTokenCount
    after: EstimatedTokenCount

    @property
    def estimated_token_savings(self) -> EstimatedTokenCount:
        """Return the non-negative reduction caused by this step."""
        return EstimatedTokenCount(max(0, self.before.value - self.after.value))

    @property
    def estimated_token_growth(self) -> EstimatedTokenCount:
        """Return the non-negative increase caused by this step."""
        return EstimatedTokenCount(max(0, self.after.value - self.before.value))


class ProviderContextLedgerEntry(ABC):
    """Base class for typed provider-history projection entries."""

    @abstractmethod
    def provider_history_token_delta(self) -> ProviderHistoryTokenDelta:
        """Return the marginal provider-history change for this step."""
        ...

    @property
    def estimated_token_savings(self) -> EstimatedTokenCount:
        """Return provider-view token savings attributable to this step."""
        return self.provider_history_token_delta().estimated_token_savings

    @property
    def estimated_token_growth(self) -> EstimatedTokenCount:
        """Return provider-view token growth attributable to this step."""
        return self.provider_history_token_delta().estimated_token_growth

    @abstractmethod
    def to_json(self) -> dict[str, Any]:
        """Return a JSON-safe diagnostic representation."""
        ...


@dataclass(frozen=True)
class ToolCallContextLedgerEntry(ProviderContextLedgerEntry):
    """A provider-view transformation associated with one tool call."""

    call_id: str
    tool_name: str
    disposition: ProviderHistoryDisposition
    token_delta: ProviderHistoryTokenDelta
    superseding_call_id: str | None = None

    def provider_history_token_delta(self) -> ProviderHistoryTokenDelta:
        return self.token_delta

    def to_json(self) -> dict[str, Any]:
        return {
            "entry_kind": "tool_call",
            "call_id": self.call_id,
            "tool_name": self.tool_name,
            "disposition": self.disposition.value,
            "estimated_token_savings": self.estimated_token_savings.value,
            "estimated_token_growth": self.estimated_token_growth.value,
            "estimated_provider_history_tokens_before": (
                self.token_delta.before.value
            ),
            "estimated_provider_history_tokens_after": (
                self.token_delta.after.value
            ),
            "superseding_call_id": self.superseding_call_id,
        }


@dataclass(frozen=True)
class FileContextProjectionLedgerEntry(ProviderContextLedgerEntry):
    """One marginal switch to the treatment file-context projection."""

    call_ids: tuple[str, ...]
    token_delta: ProviderHistoryTokenDelta
    disposition: ProviderHistoryDisposition = (
        ProviderHistoryDisposition.FILE_CONTEXT_FOLDED
    )

    def provider_history_token_delta(self) -> ProviderHistoryTokenDelta:
        return self.token_delta

    def to_json(self) -> dict[str, Any]:
        return {
            "entry_kind": "file_context_projection",
            "call_ids": list(self.call_ids),
            "disposition": self.disposition.value,
            "estimated_token_savings": self.estimated_token_savings.value,
            "estimated_token_growth": self.estimated_token_growth.value,
            "estimated_provider_history_tokens_before": (
                self.token_delta.before.value
            ),
            "estimated_provider_history_tokens_after": (
                self.token_delta.after.value
            ),
        }


@dataclass(frozen=True)
class TurnContextLedgerEntry(ProviderContextLedgerEntry):
    """A request-local whole-turn collapse."""

    history_node_index: int
    disposition: ProviderHistoryDisposition
    token_delta: ProviderHistoryTokenDelta

    def provider_history_token_delta(self) -> ProviderHistoryTokenDelta:
        return self.token_delta

    def to_json(self) -> dict[str, Any]:
        return {
            "entry_kind": "turn",
            "history_node_index": self.history_node_index,
            "disposition": self.disposition.value,
            "estimated_token_savings": self.estimated_token_savings.value,
            "estimated_token_growth": self.estimated_token_growth.value,
            "estimated_provider_history_tokens_before": (
                self.token_delta.before.value
            ),
            "estimated_provider_history_tokens_after": (
                self.token_delta.after.value
            ),
        }


@dataclass(frozen=True)
class ProviderContextLedger:
    """Immutable accounting for one provider-visible history projection.

    The baseline count is the ordinary provider rendering, including legacy
    file folding; it is not an unfolded stored-history size.  The protected
    count uses stored node costs because request-local folding can compose
    several nodes into one view, so it is deliberately labelled as stored.
    """

    policy: ContextBudgetPolicy
    resolved_budget: ResolvedHistoryBudget
    estimated_baseline_provider_history_tokens: EstimatedTokenCount
    estimated_history_tokens_after_required_file_projection: EstimatedTokenCount
    estimated_history_tokens_final: EstimatedTokenCount
    estimated_overhead_tokens: EstimatedTokenCount
    estimated_stored_protected_history_tokens: EstimatedTokenCount
    entries: tuple[ProviderContextLedgerEntry, ...]

    def __post_init__(self) -> None:
        expected_before = self.estimated_baseline_provider_history_tokens
        for entry in self.entries:
            token_delta = entry.provider_history_token_delta()
            if token_delta.before != expected_before:
                raise ValueError(
                    "provider context ledger entries must form a contiguous "
                    "marginal token-accounting chain",
                )
            expected_before = token_delta.after
        if expected_before != self.estimated_history_tokens_final:
            raise ValueError(
                "provider context ledger entries must account for the final "
                "provider-history token count",
            )

    @property
    def is_unavoidably_over_budget(self) -> bool:
        """Whether protected/non-collapsible content exceeds the soft target."""
        return (
            self.estimated_history_tokens_final
            > self.resolved_budget.history_tokens
        )

    @property
    def estimated_total_token_savings(self) -> EstimatedTokenCount:
        """Return gross provider-view reductions across projection steps."""
        return EstimatedTokenCount(sum(
            entry.estimated_token_savings.value for entry in self.entries
        ))

    @property
    def estimated_total_token_growth(self) -> EstimatedTokenCount:
        """Return gross provider-view growth across projection steps."""
        return EstimatedTokenCount(sum(
            entry.estimated_token_growth.value for entry in self.entries
        ))

    def to_json(self) -> dict[str, Any]:
        required_file_projection_tokens = (
            self.estimated_history_tokens_after_required_file_projection.value
        )
        disposition_counts = {
            disposition.value: sum(
                1 for entry in self.entries
                if getattr(entry, "disposition", None) is disposition
            )
            for disposition in ProviderHistoryDisposition
        }
        return {
            "policy": self.policy.to_json(),
            "resolved_budget": self.resolved_budget.to_json(),
            "estimated_baseline_provider_history_tokens": (
                self.estimated_baseline_provider_history_tokens.value
            ),
            "estimated_history_tokens_after_required_file_projection": (
                required_file_projection_tokens
            ),
            "estimated_history_tokens_final": (
                self.estimated_history_tokens_final.value
            ),
            "estimated_total_token_savings": (
                self.estimated_total_token_savings.value
            ),
            "estimated_total_token_growth": (
                self.estimated_total_token_growth.value
            ),
            "estimated_overhead_tokens": self.estimated_overhead_tokens.value,
            "estimated_prompt_tokens_final": (
                self.estimated_history_tokens_final.value
                + self.estimated_overhead_tokens.value
            ),
            "estimated_stored_protected_history_tokens": (
                self.estimated_stored_protected_history_tokens.value
            ),
            "is_unavoidably_over_budget": self.is_unavoidably_over_budget,
            "disposition_counts": disposition_counts,
            "entries": [entry.to_json() for entry in self.entries],
        }


@dataclass(frozen=True)
class ProviderHistoryProjection:
    """Provider messages, visibility facts, and ledger from one snapshot."""

    messages: tuple[Message, ...]
    visibility: PromptVisibilitySnapshot
    ledger: ProviderContextLedger


def estimate_provider_history_tokens(messages: Sequence[Message]) -> int:
    """Estimate serialized provider-history tokens using the loop heuristic."""
    return sum(
        estimate_tokens(json.dumps(_message_metric_payload(message)))
        for message in messages
    )


def project_history_for_provider(
    history: HistoryTree,
    *,
    workspace_root: Path | None,
    context_window: int | None,
    estimated_overhead_tokens: int,
    policy: ContextBudgetPolicy = DEFAULT_CONTEXT_BUDGET_POLICY,
) -> ProviderHistoryProjection:
    """Build a deterministic, non-mutating provider view of *history*."""
    overhead = EstimatedTokenCount(estimated_overhead_tokens)
    resolved_budget = policy.resolve(
        context_window=context_window,
        estimated_overhead_tokens=overhead,
    )
    live_file_snapshot = LiveWorkspaceFileSnapshot()

    protected_search_call_ids = frozenset()
    if (
        policy.search_replacement_policy
        is SearchObservationReplacementPolicy.COMPLETE_CURRENT_READ
    ):
        protected_search_call_ids = (
            search_observation_call_ids_requiring_raw_evidence(history.nodes)
        )
    initial_render_plan = HistoryRenderPlan(
        protected_file_observation_call_ids=protected_search_call_ids,
    )
    baseline_rendered = history.render_with_visibility(
        workspace_root=workspace_root,
        render_plan=initial_render_plan,
        live_file_snapshot=live_file_snapshot,
    )
    baseline_tokens = EstimatedTokenCount(
        estimate_provider_history_tokens(baseline_rendered.messages),
    )

    replacements = plan_file_observation_replacements(
        history.nodes,
        workspace_root=workspace_root,
        live_file_snapshot=live_file_snapshot,
        search_replacement_policy=policy.search_replacement_policy,
    )
    tool_nodes = _tool_nodes_by_call_id(history)
    render_plan = initial_render_plan
    rendered = baseline_rendered
    current_tokens = baseline_tokens.value
    entries: list[ProviderContextLedgerEntry] = []
    for replacement in replacements:
        replacement_plan = _with_tool_result_replacement(
            render_plan,
            replacement,
        )
        replacement_rendered = history.render_with_visibility(
            workspace_root=workspace_root,
            render_plan=replacement_plan,
            live_file_snapshot=live_file_snapshot,
        )
        replacement_tokens = estimate_provider_history_tokens(
            replacement_rendered.messages,
        )
        entries.append(_replacement_ledger_entry(
            tool_nodes,
            replacement,
            before_tokens=current_tokens,
            after_tokens=replacement_tokens,
        ))
        render_plan = replacement_plan
        rendered = replacement_rendered
        current_tokens = replacement_tokens

    treatment_plan = _with_file_observation_folding_mode(
        render_plan,
        FileObservationFoldingMode.FOLD_OLDER_SINGLE_OBSERVATIONS,
    )
    treatment_rendered = history.render_with_visibility(
        workspace_root=workspace_root,
        render_plan=treatment_plan,
        live_file_snapshot=live_file_snapshot,
    )
    treatment_tokens = estimate_provider_history_tokens(
        treatment_rendered.messages,
    )
    affected_folding_call_ids = tuple(sorted(
        rendered.folded_file_tool_call_ids
        | treatment_rendered.folded_file_tool_call_ids
    ))
    if (
        treatment_rendered.messages != rendered.messages
        or treatment_rendered.visibility != rendered.visibility
        or treatment_rendered.folded_file_tool_call_ids
            != rendered.folded_file_tool_call_ids
    ):
        entries.append(FileContextProjectionLedgerEntry(
            call_ids=affected_folding_call_ids,
            token_delta=_provider_history_token_delta(
                current_tokens,
                treatment_tokens,
            ),
        ))
    render_plan = treatment_plan
    rendered = treatment_rendered
    current_tokens = treatment_tokens
    required_file_projection_tokens = EstimatedTokenCount(current_tokens)

    for candidate in _projection_candidates(
        history,
        protected_call_ids=protected_search_call_ids,
    ):
        if current_tokens <= resolved_budget.history_tokens.value:
            break
        if candidate.is_no_op(render_plan):
            continue
        candidate_plan = candidate.apply(render_plan)
        candidate_rendered = history.render_with_visibility(
            workspace_root=workspace_root,
            render_plan=candidate_plan,
            live_file_snapshot=live_file_snapshot,
        )
        candidate_tokens = estimate_provider_history_tokens(
            candidate_rendered.messages,
        )
        if candidate_tokens >= current_tokens:
            continue
        entries.append(candidate.ledger_entry(
            before_tokens=current_tokens,
            after_tokens=candidate_tokens,
        ))
        render_plan = candidate_plan
        rendered = candidate_rendered
        current_tokens = candidate_tokens

    stored_protected_tokens = EstimatedTokenCount(sum(
        history.nodes[node_index].token_cost()
        for node_index in _projection_protected_indices(
            history,
            protected_call_ids=protected_search_call_ids,
        )
    ))
    ledger = ProviderContextLedger(
        policy=policy,
        resolved_budget=resolved_budget,
        estimated_baseline_provider_history_tokens=baseline_tokens,
        estimated_history_tokens_after_required_file_projection=(
            required_file_projection_tokens
        ),
        estimated_history_tokens_final=EstimatedTokenCount(current_tokens),
        estimated_overhead_tokens=overhead,
        estimated_stored_protected_history_tokens=stored_protected_tokens,
        entries=tuple(entries),
    )
    return ProviderHistoryProjection(
        messages=tuple(rendered.messages),
        visibility=rendered.visibility,
        ledger=ledger,
    )


class _ProjectionCandidate(ABC):
    @property
    @abstractmethod
    def sort_key(self) -> tuple[float, int, int]:
        ...

    @abstractmethod
    def apply(self, plan: HistoryRenderPlan) -> HistoryRenderPlan:
        ...

    @abstractmethod
    def is_no_op(self, plan: HistoryRenderPlan) -> bool:
        ...

    @abstractmethod
    def ledger_entry(
        self,
        *,
        before_tokens: int,
        after_tokens: int,
    ) -> ProviderContextLedgerEntry:
        ...


@dataclass(frozen=True)
class _ToolCallDetailCandidate(_ProjectionCandidate):
    node_index: int
    tool_call_index: int
    effective_salience: float
    tool_call_node: ToolCallNode

    @property
    def sort_key(self) -> tuple[float, int, int]:
        return (self.effective_salience, self.node_index, self.tool_call_index)

    def apply(self, plan: HistoryRenderPlan) -> HistoryRenderPlan:
        return plan.with_detail_collapsed(self.tool_call_node.tool_call.call_id)

    def is_no_op(self, plan: HistoryRenderPlan) -> bool:
        call_id = self.tool_call_node.tool_call.call_id
        return (
            self.node_index in plan.collapsed_turn_node_indices
            or call_id in plan.detail_collapsed_call_ids
        )

    def ledger_entry(
        self,
        *,
        before_tokens: int,
        after_tokens: int,
    ) -> ProviderContextLedgerEntry:
        return ToolCallContextLedgerEntry(
            call_id=self.tool_call_node.tool_call.call_id,
            tool_name=self.tool_call_node.tool_call.name,
            disposition=ProviderHistoryDisposition.DETAIL_COLLAPSED,
            token_delta=_provider_history_token_delta(
                before_tokens,
                after_tokens,
            ),
        )


@dataclass(frozen=True)
class _TurnCollapseCandidate(_ProjectionCandidate):
    node_index: int
    effective_salience: float

    @property
    def sort_key(self) -> tuple[float, int, int]:
        return (self.effective_salience + 0.001, self.node_index, -1)

    def apply(self, plan: HistoryRenderPlan) -> HistoryRenderPlan:
        return plan.with_turn_collapsed(self.node_index)

    def is_no_op(self, plan: HistoryRenderPlan) -> bool:
        return self.node_index in plan.collapsed_turn_node_indices

    def ledger_entry(
        self,
        *,
        before_tokens: int,
        after_tokens: int,
    ) -> ProviderContextLedgerEntry:
        return TurnContextLedgerEntry(
            history_node_index=self.node_index,
            disposition=ProviderHistoryDisposition.TURN_COLLAPSED,
            token_delta=_provider_history_token_delta(
                before_tokens,
                after_tokens,
            ),
        )


def _projection_protected_indices(
    history: HistoryTree,
    *,
    protected_call_ids: frozenset[str],
) -> frozenset[int]:
    protected_indices = set(history._protected_indices(
        tail_nodes=DEFAULT_PROTECTED_TAIL_NODES,
        tail_tool_calls=DEFAULT_PROTECTED_TAIL_TOOL_CALLS,
    ))
    protected_indices.update(
        node_index
        for node_index, node in enumerate(history.nodes)
        if (
            isinstance(node, TurnNode)
            and any(
                tool_node.tool_call.call_id in protected_call_ids
                for tool_node in node.tool_call_nodes
            )
        )
    )
    return frozenset(protected_indices)


def _projection_candidates(
    history: HistoryTree,
    *,
    protected_call_ids: frozenset[str] = frozenset(),
) -> tuple[_ProjectionCandidate, ...]:
    protected_indices = _projection_protected_indices(
        history,
        protected_call_ids=protected_call_ids,
    )
    candidates: list[_ProjectionCandidate] = []
    total_nodes = len(history.nodes)
    for node_index, node in enumerate(history.nodes):
        if node_index in protected_indices or not isinstance(node, TurnNode):
            continue
        if node.collapse_state is CollapseState.COLLAPSED:
            continue
        age = total_nodes - 1 - node_index
        for tool_call_index, tool_call_node in enumerate(node.tool_call_nodes):
            if not tool_call_node.is_collapsible:
                continue
            candidates.append(_ToolCallDetailCandidate(
                node_index=node_index,
                tool_call_index=tool_call_index,
                effective_salience=(
                    tool_call_node.intrinsic_salience
                    * math.exp(-DEFAULT_SALIENCE_DECAY_RATE * age)
                ),
                tool_call_node=tool_call_node,
            ))
        if node.savings_if_collapsed() > 0:
            candidates.append(_TurnCollapseCandidate(
                node_index=node_index,
                effective_salience=(
                    node.intrinsic_salience
                    * math.exp(-DEFAULT_SALIENCE_DECAY_RATE * age)
                ),
            ))
    return tuple(sorted(candidates, key=lambda candidate: candidate.sort_key))


def _replacement_ledger_entry(
    tool_nodes: dict[str, ToolCallNode],
    replacement: ToolResultReplacement,
    *,
    before_tokens: int,
    after_tokens: int,
) -> ToolCallContextLedgerEntry:
    tool_call_node = tool_nodes.get(replacement.call_id)
    tool_name = (
        tool_call_node.tool_call.name
        if tool_call_node is not None else "unknown"
    )
    disposition = (
        ProviderHistoryDisposition.STALE_FILE_OBSERVATION
        if replacement.reason is ToolResultReplacementReason.STALE_FILE_OBSERVATION
        else ProviderHistoryDisposition.REDUNDANT_FILE_OBSERVATION
    )
    return ToolCallContextLedgerEntry(
        call_id=replacement.call_id,
        tool_name=tool_name,
        disposition=disposition,
        token_delta=_provider_history_token_delta(
            before_tokens,
            after_tokens,
        ),
        superseding_call_id=replacement.superseding_call_id,
    )


def _with_tool_result_replacement(
    plan: HistoryRenderPlan,
    replacement: ToolResultReplacement,
) -> HistoryRenderPlan:
    return HistoryRenderPlan(
        tool_result_replacements=(
            *plan.tool_result_replacements,
            replacement,
        ),
        detail_collapsed_call_ids=plan.detail_collapsed_call_ids,
        collapsed_turn_node_indices=plan.collapsed_turn_node_indices,
        file_observation_folding_mode=plan.file_observation_folding_mode,
        protected_file_observation_call_ids=(
            plan.protected_file_observation_call_ids
        ),
    )


def _with_file_observation_folding_mode(
    plan: HistoryRenderPlan,
    mode: FileObservationFoldingMode,
) -> HistoryRenderPlan:
    return HistoryRenderPlan(
        tool_result_replacements=plan.tool_result_replacements,
        detail_collapsed_call_ids=plan.detail_collapsed_call_ids,
        collapsed_turn_node_indices=plan.collapsed_turn_node_indices,
        file_observation_folding_mode=mode,
        protected_file_observation_call_ids=(
            plan.protected_file_observation_call_ids
        ),
    )


def _provider_history_token_delta(
    before_tokens: int,
    after_tokens: int,
) -> ProviderHistoryTokenDelta:
    return ProviderHistoryTokenDelta(
        before=EstimatedTokenCount(before_tokens),
        after=EstimatedTokenCount(after_tokens),
    )


def _tool_nodes_by_call_id(history: HistoryTree) -> dict[str, ToolCallNode]:
    return {
        tool_call_node.tool_call.call_id: tool_call_node
        for node in history.nodes
        if isinstance(node, TurnNode)
        for tool_call_node in node.tool_call_nodes
    }


def _message_metric_payload(message: Message) -> dict[str, Any]:
    payload: dict[str, Any] = {"role": message.role}
    content = getattr(message, "content", None)
    if content is not None:
        payload["content"] = content
    tool_calls = getattr(message, "tool_calls", None)
    if tool_calls:
        payload["tool_calls"] = [
            {
                "name": tool_call.name,
                "arguments": tool_call.arguments,
            }
            for tool_call in tool_calls
        ]
    call_id = getattr(message, "call_id", None)
    if call_id is not None:
        payload["call_id"] = call_id
    return payload


__all__ = [
    "BOUNDED_HISTORY_V2_CONTEXT_BUDGET_POLICY",
    "ContextBudgetPolicy",
    "ContextWindowFraction",
    "DEFAULT_CONTEXT_BUDGET_POLICY",
    "EstimatedTokenCount",
    "FileContextProjectionLedgerEntry",
    "HistoryBudgetSource",
    "ProviderContextLedger",
    "ProviderContextLedgerEntry",
    "ProviderHistoryDisposition",
    "ProviderHistoryProjection",
    "ProviderHistoryTokenDelta",
    "ResolvedHistoryBudget",
    "ToolCallContextLedgerEntry",
    "TurnContextLedgerEntry",
    "estimate_provider_history_tokens",
    "project_history_for_provider",
]
