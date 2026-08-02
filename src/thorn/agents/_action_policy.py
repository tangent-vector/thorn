"""Named coding-action policies for controlled ``thorn run`` trials.

Action policy is an invocation variable rather than agent identity.  Keeping it
separate from :class:`~thorn.agents.LocalCodingAgent` lets evaluations change
how a direct request should progress without silently changing ``thorn chat``
or gateway agents that reuse the same coding role and tool surface.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class RunActionPolicy(StrEnum):
    """Model-facing execution contracts available to ``thorn run``."""

    BASELINE = "baseline"
    BOUNDED_ACTION_V1 = "bounded-action-v1"
    SEMANTIC_WORK_V2 = "semantic-work-v2"


DEFAULT_RUN_ACTION_POLICY = RunActionPolicy.BASELINE
"""Action policy selected when ``--action-policy`` is omitted."""


@dataclass(frozen=True)
class RunActionPolicyDefinition:
    """Stable action-policy identity and its optional system prompt block."""

    policy: RunActionPolicy
    system_prompt: str | None

    def to_metadata(self) -> dict[str, str]:
        """Return the policy identity written to evaluation artifacts."""
        return {"action_policy": self.policy.value}


_BOUNDED_ACTION_V1_SYSTEM_PROMPT = """\
[Direct coding execution contract: bounded-action-v1]
Batch a small set of independent read or search observations that you already
know you need into the same assistant response, normally two to four tool
calls. Keep dependent calls sequential. Do not batch state-changing calls or a
call whose arguments depend on an earlier result.

Treat each observation batch as one inspection step. After it, synthesize the
evidence and either act or identify exactly one missing fact. Do not spend more
than two consecutive provider rounds only broadening inspection. The second
round must target the named missing fact; then make the smallest justified edit
or report a concrete blocker.

Reuse successful tool results and provider-visible context while the underlying
file is unchanged. Do not repeat an unchanged read or search for reassurance.
The edit_file and create_file results already show the resulting content, so do
not reread only to verify those writes.

Progress from localization to the smallest justified edit, then the narrowest
relevant test. Repair based on concrete failures, broaden validation only when
warranted, and finish. Do not continue researching after the requested behavior
is implemented and relevant validation passes.
[/Direct coding execution contract]"""


_SEMANTIC_WORK_V2_SYSTEM_PROMPT = """\
[Direct coding execution contract: semantic-work-v2]
Make each provider round advance a meaningful unit of task work rather than
only one mechanical operation. When a unit needs independent reads, searches,
or validation commands whose inputs are already known, request them together
using the clearest supported tool form. One composite tool call may perform
several semantic operations, while several top-level calls may form one batch.
Never split or add calls to reach a call count.

Keep dependent operations sequential. Do not batch state-changing calls or
combine unrelated work. Do not issue a call whose arguments depend on a result
you have not received.

Treat the results for that unit as one evidence step. After it, synthesize all
of the evidence and either complete the next meaningful unit or identify
exactly one missing fact. Do not spend more than two consecutive provider
rounds only broadening inspection. The second round must target the named
missing fact; then make the smallest justified edit or report a concrete
blocker.

Reuse successful tool results and provider-visible context while the underlying
file is unchanged. Do not repeat an unchanged read or search for reassurance.
The edit_file and create_file results already show the resulting content, so do
not reread only to verify those writes.

Progress from localization to the smallest justified edit, then the narrowest
relevant test. Complete each coherent edit or validation unit before switching
to unrelated work. Repair based on concrete failures, broaden validation only
when warranted, and finish. Do not continue researching after the requested
behavior is implemented and relevant validation passes.
[/Direct coding execution contract]"""


RUN_ACTION_POLICY_DEFINITIONS: dict[
    RunActionPolicy,
    RunActionPolicyDefinition,
] = {
    RunActionPolicy.BASELINE: RunActionPolicyDefinition(
        policy=RunActionPolicy.BASELINE,
        system_prompt=None,
    ),
    RunActionPolicy.BOUNDED_ACTION_V1: RunActionPolicyDefinition(
        policy=RunActionPolicy.BOUNDED_ACTION_V1,
        system_prompt=_BOUNDED_ACTION_V1_SYSTEM_PROMPT,
    ),
    RunActionPolicy.SEMANTIC_WORK_V2: RunActionPolicyDefinition(
        policy=RunActionPolicy.SEMANTIC_WORK_V2,
        system_prompt=_SEMANTIC_WORK_V2_SYSTEM_PROMPT,
    ),
}


__all__ = [
    "DEFAULT_RUN_ACTION_POLICY",
    "RUN_ACTION_POLICY_DEFINITIONS",
    "RunActionPolicy",
    "RunActionPolicyDefinition",
]
