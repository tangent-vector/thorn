"""Active-context salience updates derived from tool events."""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping

from thorn.core._shell_inspection import (
    ShellInspectionCommand,
    ShellInspectionKind,
    parse_shell_inspection_command,
)
from thorn.runtime._working_set import (
    ActiveContextDetailLevel,
    ActiveContextEntry,
    ActiveContextEvidence,
    ActiveContextKind,
    FileContentHash,
    WorkingSet,
)

logger = logging.getLogger(__name__)

MAX_ACTIVE_CONTEXT_ENTRIES = 10
MAX_DETAILED_ACTIVE_CONTEXT_ENTRIES = 5
MAX_EVIDENCE_EVENTS = 12
MAX_SEARCH_HIT_CONTEXT_ENTRIES = 5

_LINE_NUMBER_RE = re.compile(r"^\s*(\d+)\|")


@dataclass(frozen=True)
class ActiveContextToolEvent:
    """One completed tool call that may affect working-set context."""

    tool_name: str
    arguments: Mapping[str, Any]
    result: str
    workspace_root: Path | None = None


@dataclass(frozen=True)
class _ContextObservation:
    kind: ActiveContextKind
    label: str
    detail_level: ActiveContextDetailLevel
    summary: str
    evidence: ActiveContextEvidence
    salience_delta: int
    content_hash: FileContentHash | None = None
    stale_existing_span_summary: str | None = None


def record_tool_active_context(
    session: object | None,
    *,
    tool_name: str,
    arguments: Mapping[str, Any],
    result: str,
    is_error: bool,
    workspace_root: Path | None,
) -> None:
    """Update ``session.working_set`` from a successful tool result.

    Context tracking is advisory prompt state.  It should never turn a
    successful tool call into an agent-visible error, so failures are
    logged and ignored at this boundary.
    """
    if session is None or is_error:
        return

    working_set = getattr(session, "working_set", None)
    if not isinstance(working_set, WorkingSet):
        return

    event = ActiveContextToolEvent(
        tool_name=tool_name,
        arguments=arguments,
        result=result,
        workspace_root=workspace_root,
    )
    try:
        next_working_set = update_active_context_from_tool_event(
            working_set,
            event,
        )
    except Exception:
        logger.debug("failed to update active context", exc_info=True)
        return

    if next_working_set != working_set:
        setattr(session, "working_set", next_working_set)


def update_active_context_from_tool_event(
    working_set: WorkingSet,
    event: ActiveContextToolEvent,
) -> WorkingSet:
    """Return *working_set* with active context updated for *event*."""
    observations = _observations_from_event(event)
    if not observations:
        return working_set

    entries = list(working_set.active_context)
    for observation in observations:
        entries = _apply_observation(entries, observation)

    return replace(
        working_set,
        active_context=tuple(_rank_and_bound_entries(entries)),
    )


def _observations_from_event(
    event: ActiveContextToolEvent,
) -> tuple[_ContextObservation, ...]:
    if event.tool_name == "read_file":
        observation = _read_observation(event)
        return (observation,) if observation is not None else ()
    if event.tool_name == "search_files":
        return _search_observations(event)
    if event.tool_name == "edit_file":
        observation = _edit_observation(event)
        return (observation,) if observation is not None else ()
    if event.tool_name == "create_file":
        observation = _create_observation(event)
        return (observation,) if observation is not None else ()
    if event.tool_name == "delete_file":
        observation = _delete_observation(event)
        return (observation,) if observation is not None else ()
    if event.tool_name == "move_file":
        return _move_observations(event)
    if event.tool_name == "run_shell":
        return _shell_inspection_observations(event)
    return ()


def _read_observation(
    event: ActiveContextToolEvent,
) -> _ContextObservation | None:
    raw_path = _text_argument(event.arguments, "path")
    if raw_path is None:
        return None

    resolved_path = _resolved_event_path(raw_path, event.workspace_root)
    label = _display_path(resolved_path, raw_path, event.workspace_root)
    line_span = _line_span(event.result)
    if line_span is None:
        detail_level = ActiveContextDetailLevel.FILE
        summary = "read file"
    else:
        first_line, last_line = line_span
        detail_level = ActiveContextDetailLevel.SPAN
        summary = _line_span_summary("read", first_line, last_line)

    return _ContextObservation(
        kind=ActiveContextKind.FILE,
        label=label,
        detail_level=detail_level,
        summary=summary,
        evidence=ActiveContextEvidence.READ,
        salience_delta=2,
        content_hash=_file_content_hash(resolved_path),
    )


def _search_observations(
    event: ActiveContextToolEvent,
) -> tuple[_ContextObservation, ...]:
    raw_path = _text_argument(event.arguments, "path") or "."
    pattern = _text_argument(event.arguments, "pattern")
    resolved_path = _resolved_event_path(raw_path, event.workspace_root)
    label = _display_path(resolved_path, raw_path, event.workspace_root)

    observations: list[_ContextObservation] = []
    if resolved_path.is_dir():
        observations.append(
            _ContextObservation(
                kind=ActiveContextKind.DIRECTORY,
                label=label,
                detail_level=ActiveContextDetailLevel.DIRECTORY,
                summary=_search_summary(pattern, event.result),
                evidence=ActiveContextEvidence.SEARCH,
                salience_delta=(
                    1 if event.result.startswith("No matches found") else 2
                ),
            ),
        )
        observations.extend(
            _search_hit_observations(event, pattern),
        )
        return tuple(observations)

    observations.append(
        _ContextObservation(
            kind=ActiveContextKind.FILE,
            label=label,
            detail_level=ActiveContextDetailLevel.FILE,
            summary=_search_summary(pattern, event.result),
            evidence=ActiveContextEvidence.SEARCH,
            salience_delta=1 if event.result.startswith("No matches found") else 2,
            content_hash=_file_content_hash(resolved_path),
        ),
    )
    return tuple(observations)


def _search_hit_observations(
    event: ActiveContextToolEvent,
    pattern: str | None,
) -> tuple[_ContextObservation, ...]:
    if event.result.startswith("No matches found"):
        return ()

    observations: list[_ContextObservation] = []
    for raw_hit_path in _search_hit_paths(event.result):
        resolved_path = _resolved_event_path(raw_hit_path, event.workspace_root)
        observations.append(
            _ContextObservation(
                kind=ActiveContextKind.FILE,
                label=_display_path(
                    resolved_path,
                    raw_hit_path,
                    event.workspace_root,
                ),
                detail_level=ActiveContextDetailLevel.FILE,
                summary=_search_hit_summary(pattern),
                evidence=ActiveContextEvidence.SEARCH,
                salience_delta=2,
                content_hash=_file_content_hash(resolved_path),
            ),
        )
        if len(observations) >= MAX_SEARCH_HIT_CONTEXT_ENTRIES:
            break
    return tuple(observations)


def _search_hit_paths(result: str) -> tuple[str, ...]:
    paths: list[str] = []
    for line in result.splitlines():
        stripped = line.strip()
        if not stripped or stripped == "--":
            continue
        if stripped.startswith("[") or "|" in stripped:
            continue
        if not stripped.endswith(":"):
            continue
        paths.append(stripped[:-1])
    return tuple(paths)


def _search_summary(pattern: str | None, result: str) -> str:
    match_summary = _searched_summary(pattern)
    if result.startswith("No matches found"):
        match_summary += "; no matches"
    return match_summary


def _searched_summary(pattern: str | None) -> str:
    if pattern is None:
        return "searched"
    return f"searched for {pattern!r}"


def _search_hit_summary(pattern: str | None) -> str:
    if pattern is None:
        return "search hit"
    return f"search hit for {pattern!r}"


def _edit_observation(
    event: ActiveContextToolEvent,
) -> _ContextObservation | None:
    raw_path = _text_argument(event.arguments, "path")
    if raw_path is None:
        return None

    resolved_path = _resolved_event_path(raw_path, event.workspace_root)
    label = _display_path(resolved_path, raw_path, event.workspace_root)
    result_summary = _first_result_line(event.result) or "edited file"
    stale_summary = None
    if not event.result.startswith("No edits to apply."):
        stale_summary = "previously inspected before file changed"

    return _ContextObservation(
        kind=ActiveContextKind.FILE,
        label=label,
        detail_level=ActiveContextDetailLevel.FILE,
        summary=result_summary,
        evidence=ActiveContextEvidence.EDIT,
        salience_delta=3,
        content_hash=_file_content_hash(resolved_path),
        stale_existing_span_summary=stale_summary,
    )


def _create_observation(
    event: ActiveContextToolEvent,
) -> _ContextObservation | None:
    raw_path = _text_argument(event.arguments, "path")
    if raw_path is None:
        return None

    resolved_path = _resolved_event_path(raw_path, event.workspace_root)
    label = _display_path(resolved_path, raw_path, event.workspace_root)
    return _ContextObservation(
        kind=ActiveContextKind.FILE,
        label=label,
        detail_level=ActiveContextDetailLevel.FILE,
        summary=_first_result_line(event.result) or "created file",
        evidence=ActiveContextEvidence.CREATE,
        salience_delta=3,
        content_hash=_file_content_hash(resolved_path),
    )


def _delete_observation(
    event: ActiveContextToolEvent,
) -> _ContextObservation | None:
    raw_path = _text_argument(event.arguments, "path")
    if raw_path is None:
        return None

    resolved_path = _resolved_event_path(raw_path, event.workspace_root)
    label = _display_path(resolved_path, raw_path, event.workspace_root)
    return _ContextObservation(
        kind=ActiveContextKind.FILE,
        label=label,
        detail_level=ActiveContextDetailLevel.STALE_SUMMARY,
        summary=_first_result_line(event.result) or "deleted file",
        evidence=ActiveContextEvidence.DELETE,
        salience_delta=3,
        stale_existing_span_summary="file deleted after previous context was read",
    )


def _move_observations(
    event: ActiveContextToolEvent,
) -> tuple[_ContextObservation, ...]:
    raw_source = _text_argument(event.arguments, "source")
    raw_destination = _text_argument(event.arguments, "destination")
    if raw_source is None or raw_destination is None:
        return ()

    source_path = _resolved_event_path(raw_source, event.workspace_root)
    destination_path = _resolved_event_path(raw_destination, event.workspace_root)
    source_label = _display_path(source_path, raw_source, event.workspace_root)
    destination_label = _display_path(
        destination_path,
        raw_destination,
        event.workspace_root,
    )
    source = _ContextObservation(
        kind=ActiveContextKind.FILE,
        label=source_label,
        detail_level=ActiveContextDetailLevel.STALE_SUMMARY,
        summary=f"moved to {destination_label}",
        evidence=ActiveContextEvidence.MOVE,
        salience_delta=2,
        stale_existing_span_summary="file moved after previous context was read",
    )
    destination = _ContextObservation(
        kind=ActiveContextKind.FILE,
        label=destination_label,
        detail_level=ActiveContextDetailLevel.FILE,
        summary=_first_result_line(event.result) or f"moved from {source_label}",
        evidence=ActiveContextEvidence.MOVE,
        salience_delta=3,
        content_hash=_file_content_hash(destination_path),
    )
    return source, destination


def _shell_inspection_observations(
    event: ActiveContextToolEvent,
) -> tuple[_ContextObservation, ...]:
    raw_command = _text_argument(event.arguments, "command")
    if raw_command is None:
        return ()
    inspection = parse_shell_inspection_command(raw_command)
    if inspection is None or inspection.path is None:
        return ()
    if inspection.kind is ShellInspectionKind.READ:
        observation = _shell_read_observation(event, inspection)
        return (observation,) if observation is not None else ()
    if inspection.kind is ShellInspectionKind.SEARCH:
        observation = _shell_search_observation(event, inspection)
        return (observation,) if observation is not None else ()
    return ()


def _shell_read_observation(
    event: ActiveContextToolEvent,
    inspection: ShellInspectionCommand,
) -> _ContextObservation | None:
    if inspection.path is None:
        return None
    resolved_path = _resolved_event_path(inspection.path, event.workspace_root)
    label = _display_path(resolved_path, inspection.path, event.workspace_root)
    if inspection.start_line is not None and inspection.end_line is not None:
        detail_level = ActiveContextDetailLevel.SPAN
        summary = _line_span_summary(
            "shell read",
            inspection.start_line,
            inspection.end_line,
        )
    else:
        detail_level = ActiveContextDetailLevel.FILE
        summary = f"shell read via {inspection.command_name}"
    return _ContextObservation(
        kind=ActiveContextKind.FILE,
        label=label,
        detail_level=detail_level,
        summary=summary,
        evidence=ActiveContextEvidence.READ,
        salience_delta=2,
        content_hash=_file_content_hash(resolved_path),
    )


def _shell_search_observation(
    event: ActiveContextToolEvent,
    inspection: ShellInspectionCommand,
) -> _ContextObservation | None:
    if inspection.path is None:
        return None
    resolved_path = _resolved_event_path(inspection.path, event.workspace_root)
    label = _display_path(resolved_path, inspection.path, event.workspace_root)
    is_directory = resolved_path.is_dir()
    return _ContextObservation(
        kind=(
            ActiveContextKind.DIRECTORY
            if is_directory else ActiveContextKind.FILE
        ),
        label=label,
        detail_level=(
            ActiveContextDetailLevel.DIRECTORY
            if is_directory else ActiveContextDetailLevel.FILE
        ),
        summary=(
            f"shell searched for {inspection.pattern!r} "
            f"via {inspection.command_name}"
        ),
        evidence=ActiveContextEvidence.SEARCH,
        salience_delta=2,
        content_hash=None if is_directory else _file_content_hash(resolved_path),
    )


def _apply_observation(
    entries: list[ActiveContextEntry],
    observation: _ContextObservation,
) -> list[ActiveContextEntry]:
    if observation.stale_existing_span_summary is not None:
        entries = [
            _stale_matching_span(entry, observation)
            for entry in entries
        ]

    updated = ActiveContextEntry(
        kind=observation.kind,
        label=observation.label,
        detail_level=observation.detail_level,
        summary=observation.summary,
        salience=observation.salience_delta,
        evidence=(observation.evidence,),
        content_hash=observation.content_hash,
    )

    result: list[ActiveContextEntry] = []
    inserted = False
    for entry in entries:
        if _same_context_entry(entry, updated):
            if not inserted:
                result.insert(0, _merge_entry(entry, updated))
                inserted = True
            continue
        result.append(entry)

    if not inserted:
        result.insert(0, updated)
    return result


def _stale_matching_span(
    entry: ActiveContextEntry,
    observation: _ContextObservation,
) -> ActiveContextEntry:
    if entry.kind is not ActiveContextKind.FILE:
        return entry
    if entry.label != observation.label:
        return entry
    if entry.detail_level is not ActiveContextDetailLevel.SPAN:
        return entry

    return replace(
        entry,
        detail_level=ActiveContextDetailLevel.STALE_SUMMARY,
        summary=observation.stale_existing_span_summary,
        salience=entry.salience + 1,
        evidence=_append_evidence(entry.evidence, observation.evidence),
        content_hash=None,
    )


def _merge_entry(
    existing: ActiveContextEntry,
    observed: ActiveContextEntry,
) -> ActiveContextEntry:
    return replace(
        existing,
        summary=observed.summary or existing.summary,
        salience=existing.salience + observed.salience,
        evidence=_append_many_evidence(existing.evidence, observed.evidence),
        content_hash=observed.content_hash or existing.content_hash,
    )


def _same_context_entry(
    left: ActiveContextEntry,
    right: ActiveContextEntry,
) -> bool:
    return (
        left.kind is right.kind
        and left.label == right.label
        and left.detail_level is right.detail_level
    )


def _rank_and_bound_entries(
    entries: list[ActiveContextEntry],
) -> tuple[ActiveContextEntry, ...]:
    ranked = sorted(
        enumerate(entries),
        key=lambda item: (-item[1].salience, item[0]),
    )
    selected = [entry for _, entry in ranked[:MAX_ACTIVE_CONTEXT_ENTRIES]]
    return tuple(
        _downgrade_detail_for_budget(entry)
        if index >= MAX_DETAILED_ACTIVE_CONTEXT_ENTRIES else entry
        for index, entry in enumerate(selected)
    )


def _downgrade_detail_for_budget(
    entry: ActiveContextEntry,
) -> ActiveContextEntry:
    if entry.detail_level is not ActiveContextDetailLevel.SPAN:
        return entry

    if entry.summary is None:
        summary = "line detail elided by active-context budget"
    else:
        summary = (
            f"{entry.summary}; line detail elided by active-context budget"
        )
    return replace(
        entry,
        detail_level=ActiveContextDetailLevel.FILE,
        summary=summary,
        content_hash=None,
    )


def _append_evidence(
    evidence: tuple[ActiveContextEvidence, ...],
    item: ActiveContextEvidence,
) -> tuple[ActiveContextEvidence, ...]:
    return (*evidence, item)[-MAX_EVIDENCE_EVENTS:]


def _append_many_evidence(
    evidence: tuple[ActiveContextEvidence, ...],
    items: tuple[ActiveContextEvidence, ...],
) -> tuple[ActiveContextEvidence, ...]:
    return (*evidence, *items)[-MAX_EVIDENCE_EVENTS:]


def _text_argument(
    arguments: Mapping[str, Any],
    name: str,
) -> str | None:
    value = arguments.get(name)
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    return cleaned or None


def _resolved_event_path(
    raw_path: str,
    workspace_root: Path | None,
) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path.resolve(strict=False)
    if workspace_root is not None:
        return (workspace_root / path).resolve(strict=False)
    return path.resolve(strict=False)


def _display_path(
    resolved_path: Path,
    raw_path: str,
    workspace_root: Path | None,
) -> str:
    if workspace_root is not None:
        try:
            relative = resolved_path.relative_to(
                workspace_root.resolve(strict=False),
            )
        except ValueError:
            return resolved_path.as_posix()
        rendered = relative.as_posix()
        return rendered if rendered else "."

    raw = Path(raw_path)
    return raw.as_posix() if not raw.is_absolute() else resolved_path.as_posix()


def _file_content_hash(path: Path) -> FileContentHash | None:
    if not path.is_file():
        return None
    try:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None
    return FileContentHash(digest)


def _line_span(result: str) -> tuple[int, int] | None:
    line_numbers: list[int] = []
    for line in result.splitlines():
        match = _LINE_NUMBER_RE.match(line)
        if match is not None:
            line_numbers.append(int(match.group(1)))
    if not line_numbers:
        return None
    return min(line_numbers), max(line_numbers)


def _line_span_summary(
    verb: str,
    first_line: int,
    last_line: int,
) -> str:
    if first_line == last_line:
        return f"{verb} line {first_line}"
    return f"{verb} lines {first_line}-{last_line}"


def _first_result_line(result: str) -> str | None:
    for line in result.splitlines():
        cleaned = line.strip()
        if cleaned:
            return cleaned
    return None


__all__ = [
    "ActiveContextToolEvent",
    "MAX_ACTIVE_CONTEXT_ENTRIES",
    "MAX_DETAILED_ACTIVE_CONTEXT_ENTRIES",
    "record_tool_active_context",
    "update_active_context_from_tool_event",
]
