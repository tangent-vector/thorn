"""Named provider-history policies for controlled ``thorn run`` trials.

History policy is an invocation variable rather than agent identity. Direct
CLI runs use request-local projection by default, while the baseline remains
available for controlled comparisons without changing chat or gateway
behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from thorn.core._context_ledger import (
    BOUNDED_HISTORY_V2_CONTEXT_BUDGET_POLICY,
    DEFAULT_CONTEXT_BUDGET_POLICY,
    ContextBudgetPolicy,
)


class RunHistoryPolicy(StrEnum):
    """Provider-history projection policies available to ``thorn run``."""

    BASELINE = "baseline"
    BOUNDED_HISTORY_V1 = "bounded-history-v1"
    BOUNDED_HISTORY_V2 = "bounded-history-v2"


DEFAULT_RUN_HISTORY_POLICY = RunHistoryPolicy.BOUNDED_HISTORY_V2
"""History policy selected when ``--history-policy`` is omitted."""


@dataclass(frozen=True)
class RunHistoryPolicyDefinition:
    """Stable policy identity and its optional request-projection policy."""

    policy: RunHistoryPolicy
    context_budget_policy: ContextBudgetPolicy | None

    def to_metadata(self) -> dict[str, str]:
        """Return the policy identity written to evaluation artifacts."""
        return {"history_policy": self.policy.value}


RUN_HISTORY_POLICY_DEFINITIONS: dict[
    RunHistoryPolicy,
    RunHistoryPolicyDefinition,
] = {
    RunHistoryPolicy.BASELINE: RunHistoryPolicyDefinition(
        policy=RunHistoryPolicy.BASELINE,
        context_budget_policy=None,
    ),
    RunHistoryPolicy.BOUNDED_HISTORY_V1: RunHistoryPolicyDefinition(
        policy=RunHistoryPolicy.BOUNDED_HISTORY_V1,
        context_budget_policy=DEFAULT_CONTEXT_BUDGET_POLICY,
    ),
    RunHistoryPolicy.BOUNDED_HISTORY_V2: RunHistoryPolicyDefinition(
        policy=RunHistoryPolicy.BOUNDED_HISTORY_V2,
        context_budget_policy=BOUNDED_HISTORY_V2_CONTEXT_BUDGET_POLICY,
    ),
}


__all__ = [
    "DEFAULT_RUN_HISTORY_POLICY",
    "RUN_HISTORY_POLICY_DEFINITIONS",
    "RunHistoryPolicy",
    "RunHistoryPolicyDefinition",
]
