"""Named validation-convergence policies for controlled ``thorn run`` trials."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from thorn.core._validation_convergence import ValidationConvergencePolicy


class RunValidationConvergencePolicy(StrEnum):
    """Validation-progress treatments available to ``thorn run``."""

    BASELINE = "baseline"
    ACTION_EPOCH_V1 = "action-epoch-v1"
    WORKSPACE_CONTENT_OBSERVE_V2 = "workspace-content-observe-v2"
    WORKSPACE_CONTENT_V2 = "workspace-content-v2"


DEFAULT_RUN_VALIDATION_CONVERGENCE_POLICY = RunValidationConvergencePolicy.BASELINE
"""Validation policy selected when its ``thorn run`` option is omitted."""


@dataclass(frozen=True)
class RunValidationConvergencePolicyDefinition:
    """Stable run identity and its loop-level convergence policy."""

    policy: RunValidationConvergencePolicy
    validation_convergence_policy: ValidationConvergencePolicy

    def to_metadata(self) -> dict[str, str]:
        """Return the policy identity written to evaluation artifacts."""
        return {"validation_convergence_policy": self.policy.value}


RUN_VALIDATION_CONVERGENCE_POLICY_DEFINITIONS: dict[
    RunValidationConvergencePolicy,
    RunValidationConvergencePolicyDefinition,
] = {
    RunValidationConvergencePolicy.BASELINE: (
        RunValidationConvergencePolicyDefinition(
            policy=RunValidationConvergencePolicy.BASELINE,
            validation_convergence_policy=ValidationConvergencePolicy.BASELINE,
        )
    ),
    RunValidationConvergencePolicy.ACTION_EPOCH_V1: (
        RunValidationConvergencePolicyDefinition(
            policy=RunValidationConvergencePolicy.ACTION_EPOCH_V1,
            validation_convergence_policy=(ValidationConvergencePolicy.ACTION_EPOCH_V1),
        )
    ),
    RunValidationConvergencePolicy.WORKSPACE_CONTENT_OBSERVE_V2: (
        RunValidationConvergencePolicyDefinition(
            policy=(
                RunValidationConvergencePolicy.WORKSPACE_CONTENT_OBSERVE_V2
            ),
            validation_convergence_policy=(
                ValidationConvergencePolicy.WORKSPACE_CONTENT_OBSERVE_V2
            ),
        )
    ),
    RunValidationConvergencePolicy.WORKSPACE_CONTENT_V2: (
        RunValidationConvergencePolicyDefinition(
            policy=RunValidationConvergencePolicy.WORKSPACE_CONTENT_V2,
            validation_convergence_policy=(
                ValidationConvergencePolicy.WORKSPACE_CONTENT_V2
            ),
        )
    ),
}


__all__ = [
    "DEFAULT_RUN_VALIDATION_CONVERGENCE_POLICY",
    "RUN_VALIDATION_CONVERGENCE_POLICY_DEFINITIONS",
    "RunValidationConvergencePolicy",
    "RunValidationConvergencePolicyDefinition",
]
