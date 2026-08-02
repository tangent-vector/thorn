"""Structured telemetry summaries for session working-set state."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Mapping

from thorn.core._history import estimate_tokens
from thorn.runtime._todo import TodoStatus
from thorn.runtime._working_set import (
    ActiveContextDetailLevel,
    ActiveContextEvidence,
    RenderedWorkingSet,
    ValidationOutcome,
    WorkingSet,
)


class WorkingSetTelemetryKind(StrEnum):
    """Why a working-set telemetry event was emitted."""

    PROMPT_RENDERED = "prompt_rendered"
    FOCUS_UPDATED = "focus_updated"
    GATE_INTERVENTION = "gate_intervention"


@dataclass(frozen=True)
class WorkingSetTodoTelemetry:
    """TODO counts relevant to the current working set."""

    open_count: int = 0
    resolved_count: int = 0
    total_count: int = 0

    def to_json(self) -> dict[str, int]:
        """Return a JSON-serializable representation."""
        return {
            "open_count": self.open_count,
            "resolved_count": self.resolved_count,
            "total_count": self.total_count,
        }


@dataclass(frozen=True)
class WorkingSetActiveContextTelemetry:
    """Compact active-context metrics for trial comparison."""

    entry_count: int
    stale_entry_count: int
    total_salience: int
    evidence_counts: Mapping[ActiveContextEvidence, int] = field(
        default_factory=dict,
    )

    def to_json(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "entry_count": self.entry_count,
            "stale_entry_count": self.stale_entry_count,
            "total_salience": self.total_salience,
            "evidence_counts": {
                evidence.value: count
                for evidence, count in self.evidence_counts.items()
            },
        }


@dataclass(frozen=True)
class WorkingSetValidationTelemetry:
    """Validation state without embedding raw rationale text."""

    outcome: ValidationOutcome | None = None
    summary_estimated_tokens: int | None = None
    has_command: bool = False
    command_estimated_tokens: int | None = None
    has_no_validation_rationale: bool = False
    no_validation_rationale_estimated_tokens: int | None = None

    def to_json(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "outcome": self.outcome.value if self.outcome is not None else None,
            "summary_estimated_tokens": self.summary_estimated_tokens,
            "has_command": self.has_command,
            "command_estimated_tokens": self.command_estimated_tokens,
            "has_no_validation_rationale": self.has_no_validation_rationale,
            "no_validation_rationale_estimated_tokens": (
                self.no_validation_rationale_estimated_tokens
            ),
        }


@dataclass(frozen=True)
class WorkingSetBlockerTelemetry:
    """Blocked-state shape without raw blocker text."""

    present: bool
    summary_estimated_tokens: int | None = None
    unblock_condition_estimated_tokens: int | None = None

    def to_json(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "present": self.present,
            "summary_estimated_tokens": self.summary_estimated_tokens,
            "unblock_condition_estimated_tokens": (
                self.unblock_condition_estimated_tokens
            ),
        }


@dataclass(frozen=True)
class WorkingSetRenderedTelemetry:
    """Rendered prompt-block size metrics."""

    char_count: int | None = None
    estimated_tokens: int | None = None
    diagnostics: tuple[str, ...] = ()

    def to_json(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "char_count": self.char_count,
            "estimated_tokens": self.estimated_tokens,
            "diagnostics": list(self.diagnostics),
            "diagnostic_count": len(self.diagnostics),
        }


@dataclass(frozen=True)
class WorkingSetGateTelemetry:
    """A rejected transition or heuristic gate."""

    name: str
    reason: str

    def to_json(self) -> dict[str, str]:
        """Return a JSON-serializable representation."""
        return {
            "name": self.name,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class WorkingSetTelemetry:
    """One structured working-set telemetry event."""

    kind: WorkingSetTelemetryKind
    phase: str
    focused_inbox_item_id: str | None
    objective_estimated_tokens: int | None
    last_action_estimated_tokens: int | None
    active_context: WorkingSetActiveContextTelemetry
    todo: WorkingSetTodoTelemetry
    validation: WorkingSetValidationTelemetry
    blocker: WorkingSetBlockerTelemetry
    rendered: WorkingSetRenderedTelemetry
    in_progress_not_focused_count: int = 0
    has_override_rationale: bool = False
    override_rationale_estimated_tokens: int | None = None
    gate: WorkingSetGateTelemetry | None = None

    def to_json(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "kind": self.kind.value,
            "phase": self.phase,
            "focused_inbox_item_id": self.focused_inbox_item_id,
            "objective_estimated_tokens": self.objective_estimated_tokens,
            "last_action_estimated_tokens": self.last_action_estimated_tokens,
            "active_context": self.active_context.to_json(),
            "todo": self.todo.to_json(),
            "validation": self.validation.to_json(),
            "blocker": self.blocker.to_json(),
            "rendered": self.rendered.to_json(),
            "in_progress_not_focused_count": self.in_progress_not_focused_count,
            "has_override_rationale": self.has_override_rationale,
            "override_rationale_estimated_tokens": (
                self.override_rationale_estimated_tokens
            ),
            "gate": self.gate.to_json() if self.gate is not None else None,
        }


def todo_telemetry_from_statuses(
    statuses: tuple[TodoStatus, ...],
) -> WorkingSetTodoTelemetry:
    """Build TODO telemetry counts from durable TODO statuses."""
    resolved_count = sum(1 for status in statuses if status.is_resolved)
    open_count = sum(1 for status in statuses if status is TodoStatus.OPEN)
    return WorkingSetTodoTelemetry(
        open_count=open_count,
        resolved_count=resolved_count,
        total_count=len(statuses),
    )


def build_working_set_telemetry(
    *,
    kind: WorkingSetTelemetryKind,
    working_set: WorkingSet,
    rendered: RenderedWorkingSet | None = None,
    todo: WorkingSetTodoTelemetry | None = None,
    in_progress_not_focused_count: int = 0,
    gate: WorkingSetGateTelemetry | None = None,
) -> WorkingSetTelemetry:
    """Summarize working-set state for traces without raw task prose."""
    return WorkingSetTelemetry(
        kind=kind,
        phase=working_set.phase.value,
        focused_inbox_item_id=(
            str(working_set.focused_inbox_item_id)
            if working_set.focused_inbox_item_id is not None else None
        ),
        objective_estimated_tokens=_optional_estimated_tokens(
            working_set.objective,
        ),
        last_action_estimated_tokens=_optional_estimated_tokens(
            working_set.last_action_summary,
        ),
        active_context=_active_context_telemetry(working_set),
        todo=todo or WorkingSetTodoTelemetry(),
        validation=_validation_telemetry(working_set),
        blocker=_blocker_telemetry(working_set),
        rendered=_rendered_telemetry(rendered),
        in_progress_not_focused_count=in_progress_not_focused_count,
        has_override_rationale=working_set.override_rationale is not None,
        override_rationale_estimated_tokens=_optional_estimated_tokens(
            working_set.override_rationale,
        ),
        gate=gate,
    )


def _active_context_telemetry(
    working_set: WorkingSet,
) -> WorkingSetActiveContextTelemetry:
    evidence_counts: Counter[ActiveContextEvidence] = Counter()
    stale_count = 0
    total_salience = 0
    for entry in working_set.active_context:
        total_salience += entry.salience
        if entry.detail_level is ActiveContextDetailLevel.STALE_SUMMARY:
            stale_count += 1
        evidence_counts.update(entry.evidence)

    return WorkingSetActiveContextTelemetry(
        entry_count=len(working_set.active_context),
        stale_entry_count=stale_count,
        total_salience=total_salience,
        evidence_counts=dict(evidence_counts),
    )


def _validation_telemetry(
    working_set: WorkingSet,
) -> WorkingSetValidationTelemetry:
    if working_set.last_validation is not None:
        return WorkingSetValidationTelemetry(
            outcome=working_set.last_validation.outcome,
            summary_estimated_tokens=estimate_tokens(
                working_set.last_validation.summary,
            ),
            has_command=working_set.last_validation.command is not None,
            command_estimated_tokens=_optional_estimated_tokens(
                working_set.last_validation.command,
            ),
        )

    return WorkingSetValidationTelemetry(
        has_no_validation_rationale=(
            working_set.no_validation_rationale is not None
        ),
        no_validation_rationale_estimated_tokens=_optional_estimated_tokens(
            working_set.no_validation_rationale,
        ),
    )


def _blocker_telemetry(working_set: WorkingSet) -> WorkingSetBlockerTelemetry:
    if working_set.blocker is None:
        return WorkingSetBlockerTelemetry(present=False)
    return WorkingSetBlockerTelemetry(
        present=True,
        summary_estimated_tokens=estimate_tokens(working_set.blocker.summary),
        unblock_condition_estimated_tokens=estimate_tokens(
            working_set.blocker.unblock_condition,
        ),
    )


def _rendered_telemetry(
    rendered: RenderedWorkingSet | None,
) -> WorkingSetRenderedTelemetry:
    if rendered is None:
        return WorkingSetRenderedTelemetry()
    return WorkingSetRenderedTelemetry(
        char_count=len(rendered.text),
        estimated_tokens=estimate_tokens(rendered.text),
        diagnostics=rendered.diagnostics,
    )


def _optional_estimated_tokens(value: str | None) -> int | None:
    return estimate_tokens(value) if value is not None else None


__all__ = [
    "WorkingSetActiveContextTelemetry",
    "WorkingSetBlockerTelemetry",
    "WorkingSetGateTelemetry",
    "WorkingSetRenderedTelemetry",
    "WorkingSetTelemetry",
    "WorkingSetTelemetryKind",
    "WorkingSetTodoTelemetry",
    "WorkingSetValidationTelemetry",
    "build_working_set_telemetry",
    "todo_telemetry_from_statuses",
]
