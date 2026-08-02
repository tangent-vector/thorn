"""Compact session working-set state and rendering."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Mapping, NewType, Sequence

from thorn.runtime._notification import NotificationID
from thorn.runtime._todo import TodoItem, TodoStatus

_RENDERED_TEXT_FIELD_LIMIT = 240
_REPEATED_ACCESS_DIAGNOSTIC_SALIENCE = 5

FileContentHash = NewType("FileContentHash", str)


class HandlingPhase(StrEnum):
    """Agent-facing phase for the current focused work."""

    INTAKE = "intake"
    INSPECT = "inspect"
    ACT = "act"
    VALIDATE = "validate"
    CLOSEOUT = "closeout"
    BLOCKED = "blocked"


class ActiveContextKind(StrEnum):
    """Kind of resource represented in the working-set context."""

    DIRECTORY = "directory"
    FILE = "file"
    URL = "url"
    NOTE = "note"


class ActiveContextDetailLevel(StrEnum):
    """How much detail an active-context entry currently carries."""

    DIRECTORY = "directory"
    FILE = "file"
    SPAN = "span"
    STALE_SUMMARY = "stale_summary"


class ActiveContextEvidence(StrEnum):
    """Tool event kind that made an active-context entry salient."""

    READ = "read"
    SEARCH = "search"
    EDIT = "edit"
    CREATE = "create"
    DELETE = "delete"
    MOVE = "move"


class ValidationOutcome(StrEnum):
    """Outcome of the last recorded validation step."""

    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ActiveContextEntry:
    """One compact active-context resource entry."""

    kind: ActiveContextKind
    label: str
    detail_level: ActiveContextDetailLevel
    summary: str | None = None
    salience: int = 1
    evidence: tuple[ActiveContextEvidence, ...] = ()
    content_hash: FileContentHash | None = None

    def to_data(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        result: dict[str, Any] = {
            "kind": self.kind.value,
            "label": self.label,
            "detail_level": self.detail_level.value,
            "salience": self.salience,
        }
        if self.summary is not None:
            result["summary"] = self.summary
        if self.evidence:
            result["evidence"] = [evidence.value for evidence in self.evidence]
        if self.content_hash is not None:
            result["content_hash"] = str(self.content_hash)
        return result

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "ActiveContextEntry":
        """Reconstruct an entry from serialized data."""
        return cls(
            kind=ActiveContextKind(str(data["kind"])),
            label=_required_text(data["label"], "active context label"),
            detail_level=ActiveContextDetailLevel(str(data["detail_level"])),
            summary=_optional_text(data.get("summary")),
            salience=_positive_int(data.get("salience", 1), "active context salience"),
            evidence=tuple(
                ActiveContextEvidence(str(item))
                for item in data.get("evidence", [])
            ),
            content_hash=_optional_file_content_hash(data.get("content_hash")),
        )


@dataclass(frozen=True)
class LastValidation:
    """Compact summary of the most recent validation attempt."""

    outcome: ValidationOutcome
    summary: str
    command: str | None = None

    def to_data(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        result: dict[str, Any] = {
            "outcome": self.outcome.value,
            "summary": self.summary,
        }
        if self.command is not None:
            result["command"] = self.command
        return result

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "LastValidation":
        """Reconstruct validation state from serialized data."""
        return cls(
            outcome=ValidationOutcome(str(data["outcome"])),
            summary=_required_text(data["summary"], "validation summary"),
            command=_optional_text(data.get("command")),
        )


@dataclass(frozen=True)
class WorkingSetBlocker:
    """Reason the current focused work is blocked."""

    summary: str
    unblock_condition: str

    def to_data(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "summary": self.summary,
            "unblock_condition": self.unblock_condition,
        }

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "WorkingSetBlocker":
        """Reconstruct blocker state from serialized data."""
        return cls(
            summary=_required_text(data["summary"], "blocker summary"),
            unblock_condition=_required_text(
                data["unblock_condition"],
                "blocker unblock condition",
            ),
        )


@dataclass(frozen=True)
class WorkingSet:
    """Durable, compact task state rendered into session prompts."""

    phase: HandlingPhase = HandlingPhase.INTAKE
    focused_inbox_item_id: NotificationID | None = None
    objective: str | None = None
    last_action_summary: str | None = None
    active_context: tuple[ActiveContextEntry, ...] = ()
    last_validation: LastValidation | None = None
    no_validation_rationale: str | None = None
    blocker: WorkingSetBlocker | None = None
    override_rationale: str | None = None

    def to_data(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        result: dict[str, Any] = {"phase": self.phase.value}
        if self.focused_inbox_item_id is not None:
            result["focused_inbox_item_id"] = str(self.focused_inbox_item_id)
        if self.objective is not None:
            result["objective"] = self.objective
        if self.last_action_summary is not None:
            result["last_action_summary"] = self.last_action_summary
        if self.active_context:
            result["active_context"] = [
                entry.to_data() for entry in self.active_context
            ]
        if self.last_validation is not None:
            result["last_validation"] = self.last_validation.to_data()
        if self.no_validation_rationale is not None:
            result["no_validation_rationale"] = self.no_validation_rationale
        if self.blocker is not None:
            result["blocker"] = self.blocker.to_data()
        if self.override_rationale is not None:
            result["override_rationale"] = self.override_rationale
        return result

    @classmethod
    def from_data(cls, data: Mapping[str, Any] | None) -> "WorkingSet":
        """Reconstruct working-set state from serialized data."""
        if data is None:
            return cls()
        active_context = tuple(
            ActiveContextEntry.from_data(item)
            for item in data.get("active_context", [])
        )
        focused_raw = data.get("focused_inbox_item_id")
        return cls(
            phase=HandlingPhase(
                str(data.get("phase", HandlingPhase.INTAKE.value)),
            ),
            focused_inbox_item_id=(
                NotificationID(str(focused_raw))
                if focused_raw is not None else None
            ),
            objective=_optional_text(data.get("objective")),
            last_action_summary=_optional_text(
                data.get("last_action_summary"),
            ),
            active_context=active_context,
            last_validation=(
                LastValidation.from_data(data["last_validation"])
                if data.get("last_validation") is not None else None
            ),
            no_validation_rationale=_optional_text(
                data.get("no_validation_rationale"),
            ),
            blocker=(
                WorkingSetBlocker.from_data(data["blocker"])
                if data.get("blocker") is not None else None
            ),
            override_rationale=_optional_text(data.get("override_rationale")),
        )


@dataclass(frozen=True)
class RenderedWorkingSet:
    """Rendered working-set prompt block plus diagnostics."""

    text: str
    diagnostics: tuple[str, ...] = field(default_factory=tuple)


def render_working_set_block(
    working_set: WorkingSet,
    *,
    open_todos: Sequence[TodoItem] = (),
    in_progress_item_ids_not_focused: Sequence[NotificationID] = (),
    todo_diagnostic: str | None = None,
) -> RenderedWorkingSet:
    """Render the compact working-set prompt suffix."""
    diagnostics = list(_working_set_diagnostics(working_set))
    if todo_diagnostic is not None:
        diagnostics.append(todo_diagnostic)

    lines = ["[Current working set]"]
    lines.append(f"Phase: {working_set.phase.value}")
    if working_set.focused_inbox_item_id is not None:
        lines.append(f"Focused inbox item: {working_set.focused_inbox_item_id}")
    else:
        lines.append("Focused inbox item: none")
    if working_set.objective is not None:
        lines.append(f"Objective: {_compact_line(working_set.objective)}")
    if working_set.last_action_summary is not None:
        lines.append(
            f"Last action: {_compact_line(working_set.last_action_summary)}"
        )

    lines.append("Open TODOs:")
    if open_todos:
        for item in open_todos[:10]:
            lines.append(f"- {item.id}: {_compact_line(item.title)}")
        remaining = len(open_todos) - 10
        if remaining > 0:
            lines.append(f"- ... {remaining} more open TODO(s)")
    else:
        lines.append("- none")

    if in_progress_item_ids_not_focused:
        lines.append("In-progress inbox items outside focus:")
        for item_id in in_progress_item_ids_not_focused[:10]:
            lines.append(f"- {item_id}")
        remaining = len(in_progress_item_ids_not_focused) - 10
        if remaining > 0:
            lines.append(f"- ... {remaining} more in-progress item(s)")

    lines.append("Active context:")
    if working_set.active_context:
        for entry in working_set.active_context[:10]:
            lines.append(f"- {_render_active_context_entry(entry)}")
        remaining = len(working_set.active_context) - 10
        if remaining > 0:
            lines.append(f"- ... {remaining} more context item(s)")
    else:
        lines.append("- none")

    if working_set.last_validation is not None:
        lines.append(
            "Last validation: "
            f"{working_set.last_validation.outcome.value}: "
            f"{_compact_line(working_set.last_validation.summary)}"
        )
        if working_set.last_validation.command is not None:
            lines.append(
                "Validation command: "
                f"{_compact_line(working_set.last_validation.command)}"
            )
    elif working_set.no_validation_rationale is not None:
        lines.append(
            "Validation: not run: "
            f"{_compact_line(working_set.no_validation_rationale)}"
        )
    else:
        lines.append("Validation: not recorded")

    if working_set.blocker is not None:
        lines.append(f"Blocker: {_compact_line(working_set.blocker.summary)}")
        lines.append(
            "Unblocked by: "
            f"{_compact_line(working_set.blocker.unblock_condition)}"
        )
    if working_set.override_rationale is not None:
        lines.append(
            "Override rationale: "
            f"{_compact_line(working_set.override_rationale)}"
        )

    if diagnostics:
        lines.append("Diagnostics:")
        for diagnostic in diagnostics:
            lines.append(f"- {diagnostic}")

    lines.append("[/Current working set]")
    return RenderedWorkingSet(
        text="\n".join(lines),
        diagnostics=tuple(diagnostics),
    )


def _working_set_diagnostics(working_set: WorkingSet) -> tuple[str, ...]:
    diagnostics: list[str] = []
    if (
        working_set.phase is HandlingPhase.INTAKE
        and working_set.focused_inbox_item_id is not None
    ):
        diagnostics.append("phase is intake but a focused inbox item is set")
    if (
        working_set.phase is not HandlingPhase.INTAKE
        and working_set.focused_inbox_item_id is None
    ):
        diagnostics.append(
            "phase is active but no focused inbox item is recorded",
        )
    if working_set.phase is HandlingPhase.BLOCKED and working_set.blocker is None:
        diagnostics.append("phase is blocked but no blocker is recorded")
    if (
        working_set.phase is not HandlingPhase.BLOCKED
        and working_set.blocker is not None
    ):
        diagnostics.append("blocker is recorded while phase is not blocked")
    if (
        working_set.last_validation is not None
        and working_set.no_validation_rationale is not None
    ):
        diagnostics.append(
            "both last validation and no-validation rationale are recorded",
        )
    for entry in working_set.active_context:
        if entry.salience < _REPEATED_ACCESS_DIAGNOSTIC_SALIENCE:
            continue
        evidence = _render_active_context_evidence(entry)
        if evidence is None:
            diagnostics.append(
                "active context repeatedly accessed: "
                f"{entry.label} (salience {entry.salience})",
            )
        else:
            diagnostics.append(
                "active context repeatedly accessed: "
                f"{entry.label} ({evidence}; salience {entry.salience})",
            )
    return tuple(diagnostics)


def _render_active_context_entry(entry: ActiveContextEntry) -> str:
    rendered = (
        f"{entry.kind.value} {entry.label} "
        f"({entry.detail_level.value})"
    )
    if entry.summary is not None:
        rendered += f": {_compact_line(entry.summary)}"
    evidence = _render_active_context_evidence(entry)
    if evidence is not None:
        rendered += f" [{evidence}]"
    return rendered


def _render_active_context_evidence(entry: ActiveContextEntry) -> str | None:
    if not entry.evidence:
        return None

    parts: list[str] = []
    for evidence in ActiveContextEvidence:
        count = entry.evidence.count(evidence)
        if count == 0:
            continue
        if count == 1:
            parts.append(evidence.value)
        else:
            parts.append(f"{evidence.value} x{count}")
    return ", ".join(parts) if parts else None


def _required_text(value: object, field_name: str) -> str:
    cleaned = str(value).strip()
    if not cleaned:
        raise ValueError(f"{field_name} must not be empty")
    return cleaned


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    cleaned = str(value).strip()
    return cleaned or None


def _optional_file_content_hash(value: object) -> FileContentHash | None:
    text = _optional_text(value)
    return FileContentHash(text) if text is not None else None


def _positive_int(value: object, field_name: str) -> int:
    try:
        converted = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be an integer") from exc
    if converted < 1:
        raise ValueError(f"{field_name} must be positive")
    return converted


def _compact_line(value: str) -> str:
    line = " ".join(value.split()).strip()
    if len(line) <= _RENDERED_TEXT_FIELD_LIMIT:
        return line
    return line[: _RENDERED_TEXT_FIELD_LIMIT - 3].rstrip() + "..."


def open_todo_items(items: Sequence[TodoItem]) -> tuple[TodoItem, ...]:
    """Return unresolved TODOs in input order."""
    return tuple(item for item in items if item.status is TodoStatus.OPEN)


__all__ = [
    "ActiveContextDetailLevel",
    "ActiveContextEntry",
    "ActiveContextEvidence",
    "ActiveContextKind",
    "FileContentHash",
    "HandlingPhase",
    "LastValidation",
    "RenderedWorkingSet",
    "ValidationOutcome",
    "WorkingSet",
    "WorkingSetBlocker",
    "open_todo_items",
    "render_working_set_block",
]
