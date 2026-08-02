"""Rendered-history folding for file-oriented tool output.

This module keeps raw tool results in ``HistoryTree`` but changes the
provider-facing rendering for repeated file reads/searches.  Older raw
tool outputs become compact provenance summaries, and the agent receives
one source-grounded current view per file, rendered from live file
contents under the active execution context.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum, StrEnum
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, Iterable

from thorn.core._edit_result import FileEditResultProvenance
from thorn.core._file_access import FileAccessLevel
from thorn.core._history import (
    CollapseState,
    FileObservationFoldingMode,
    HistoryNode,
    HistoryRenderPlan,
    ToolCallNode,
    ToolResultReplacement,
    ToolResultReplacementReason,
    TurnNode,
)
from thorn.core._messages import (
    AssistantMessage,
    Message,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)
from thorn.core._prompt_visibility import (
    FileContentKey,
    FileLineContent,
    FileLineRange,
    PromptVisibilitySnapshotBuilder,
    RenderedHistory,
    file_content_key_for_path,
    parse_line_numbered_file_content,
)

if TYPE_CHECKING:
    from thorn.core._outline import OutputSpan

FOLDED_FILE_CONTEXT_MAX_FILES: int = 6
FOLDED_FILE_CONTEXT_LINE_BUDGET: int = 80
FOLDED_FILE_CONTEXT_CHAR_BUDGET: int = 20_000
FOLDED_FILE_CONTEXT_AROUND_SALIENT_LINE: int = 2
FOLDED_FILE_CONTEXT_OUTLINE_LINE_BUDGET: int = 18
FOLDED_FILE_MIN_OBSERVATIONS: int = 2
FOLDED_FILE_SALIENCE_DECAY: float = 0.88


class FileLineMarker(Enum):
    """One-character markers used in folded file-context views."""

    READ = "*"
    SEARCH = "!"
    EDIT = "~"
    EDIT_CONTEXT = "-"


class FileObservationResultState(Enum):
    """Whether historical line content still matches the live file."""

    CURRENT = "current"
    STALE = "stale"
    UNKNOWN = "unknown"


class SearchObservationReplacementPolicy(StrEnum):
    """Proof strength used when replacing successful search observations."""

    EXACT_DUPLICATE_ONLY = "exact-duplicate-only"
    COMPLETE_CURRENT_READ = "complete-current-read"


@dataclass(frozen=True)
class FileRange:
    """A 1-based inclusive line range."""

    start_line: int
    end_line: int

    def clamp(self, total_lines: int) -> FileRange | None:
        if total_lines <= 0:
            return None
        start = max(1, self.start_line)
        end = min(total_lines, self.end_line)
        if start > end:
            return None
        return FileRange(start, end)


@dataclass(frozen=True)
class FileObservation:
    """One file-related tool observation used for folded context."""

    call_id: str
    tool_name: str
    display_path: str
    resolved_path: Path
    sequence_index: int
    read_range: FileRange | None = None
    whole_file_read: bool = False
    search_pattern: str | None = None
    search_lines: tuple[int, ...] = ()
    search_result_complete: bool = False
    edited_lines: tuple[int, ...] = ()
    edit_context_lines: tuple[int, ...] = ()
    returned_lines: tuple[FileLineContent, ...] = ()


@dataclass
class FileContextAccumulator:
    """Accumulated salience and provenance for one live file."""

    display_path: str
    resolved_path: Path
    observations: list[FileObservation] = field(default_factory=list)

    def append(self, observation: FileObservation) -> None:
        self.observations.append(observation)

    @property
    def last_sequence_index(self) -> int:
        return max(obs.sequence_index for obs in self.observations)

    @property
    def read_search_observations(self) -> list[FileObservation]:
        return [
            obs for obs in self.observations
            if obs.tool_name in {"read_file", "search_files"}
        ]

    @property
    def latest_returned_read_range(self) -> FileRange | None:
        """Return the most recent contiguous range returned by ``read_file``."""
        for observation in reversed(self.observations):
            if (
                observation.tool_name == "read_file"
                and observation.read_range is not None
            ):
                return observation.read_range
        return None


@dataclass(frozen=True)
class RenderedFoldedFileContext:
    """Folded current-file context plus visibility facts."""

    text: str
    file_key: FileContentKey
    visible_lines: tuple[FileLineContent, ...]
    collapsed_ranges: tuple[FileLineRange, ...]


_SEARCH_RESULT_HEADER_RE = re.compile(r"^(?P<path>.+):$")
_LINE_NUMBER_RE = re.compile(r"^\s*(?P<line>\d+)\|")
_SANDBOX_WORKSPACE_PREFIX = PurePosixPath("/agent/workspace")


class _LiveFileContent:
    """Result of conservatively loading current file content."""


@dataclass(frozen=True)
class _ReadableLiveFileContent(_LiveFileContent):
    lines: tuple[str, ...]


@dataclass(frozen=True)
class _MissingLiveFileContent(_LiveFileContent):
    pass


@dataclass(frozen=True)
class _UnknownLiveFileContent(_LiveFileContent):
    pass


@dataclass
class LiveWorkspaceFileSnapshot:
    """Lazy, request-scoped snapshot of live workspace file contents.

    A path is read at most once.  Missing, unreadable, and readable results are
    all cached so every projection phase reasons from the same file state.
    """

    _content_by_resolved_path: dict[Path, _LiveFileContent] = field(
        default_factory=dict,
    )

    def content_for(self, resolved_path: Path) -> _LiveFileContent:
        """Return the first live-content result observed for *resolved_path*."""
        if resolved_path not in self._content_by_resolved_path:
            self._content_by_resolved_path[resolved_path] = (
                _load_live_file_content(resolved_path)
            )
        return self._content_by_resolved_path[resolved_path]


class _FileObservationCollectionMode(Enum):
    """Which resolved file paths observation collection accepts."""

    LEGACY_READABLE_FILES = "legacy_readable_files"
    PROJECTED_READABLE_FILES = "projected_readable_files"
    PROJECTED_ALLOWED_PATHS = "projected_allowed_paths"


def plan_file_observation_replacements(
    nodes: list[HistoryNode],
    *,
    workspace_root: Path | None = None,
    live_file_snapshot: LiveWorkspaceFileSnapshot | None = None,
    search_replacement_policy: SearchObservationReplacementPolicy = (
        SearchObservationReplacementPolicy.EXACT_DUPLICATE_ONLY
    ),
) -> tuple[ToolResultReplacement, ...]:
    """Plan conservative request-local removal of stale/redundant results.

    The proof pass is independent of the six-file folded-context selector.
    It never mutates stored history and declines to replace results when live
    state or line-numbered output cannot be established exactly.  When a live
    file snapshot is supplied, proofs share its first-observed file versions
    with the other phases of the same provider projection.
    """
    effective_live_file_snapshot = (
        live_file_snapshot or LiveWorkspaceFileSnapshot()
    )
    accumulators = _collect_file_context(
        nodes,
        workspace_root=workspace_root,
        render_plan=HistoryRenderPlan(),
        collection_mode=_FileObservationCollectionMode.PROJECTED_ALLOWED_PATHS,
        live_file_snapshot=effective_live_file_snapshot,
    )
    replacements_by_call_id: dict[str, ToolResultReplacement] = {}

    for accumulator in accumulators.values():
        read_observations = tuple(
            observation
            for observation in accumulator.read_search_observations
            if observation.tool_name == "read_file"
        )
        if not read_observations:
            continue
        live_content = effective_live_file_snapshot.content_for(
            accumulator.resolved_path,
        )
        current_reads: list[FileObservation] = []
        for observation in read_observations:
            state = _observation_state(live_content, observation.returned_lines)
            if state is FileObservationResultState.STALE:
                replacements_by_call_id[observation.call_id] = (
                    _stale_file_observation_replacement(observation)
                )
                continue
            if (
                state is FileObservationResultState.CURRENT
                and observation.returned_lines
            ):
                current_reads.append(observation)

        for observation_index, observation in enumerate(current_reads):
            returned_lines = frozenset(observation.returned_lines)
            for newer_observation in current_reads[observation_index + 1:]:
                if returned_lines <= frozenset(newer_observation.returned_lines):
                    replacements_by_call_id[observation.call_id] = (
                        _redundant_file_observation_replacement(
                            observation,
                            newer_observation,
                        )
                    )
                    break

    if (
        search_replacement_policy
        is SearchObservationReplacementPolicy.COMPLETE_CURRENT_READ
    ):
        for search_node, newer_read in _searches_subsumed_by_current_read(
            nodes,
            accumulators=accumulators,
            live_file_snapshot=effective_live_file_snapshot,
        ):
            search_call_id = search_node.tool_call.call_id
            if search_call_id in replacements_by_call_id:
                continue
            replacements_by_call_id[search_call_id] = ToolResultReplacement(
                call_id=search_call_id,
                content=(
                    f"{_folded_result_summary(search_node)} "
                    "[redundant file observation omitted; current search "
                    f"evidence is retained by tool call {newer_read.call_id}; "
                    "raw result remains in session history.]"
                ),
                reason=ToolResultReplacementReason.REDUNDANT_FILE_OBSERVATION,
                superseding_call_id=newer_read.call_id,
            )

    for older_node, newer_node in _exact_duplicate_search_pairs(nodes):
        older_call_id = older_node.tool_call.call_id
        if older_call_id in replacements_by_call_id:
            continue
        replacements_by_call_id[older_call_id] = ToolResultReplacement(
            call_id=older_call_id,
            content=(
                f"{_folded_result_summary(older_node)} "
                "[redundant file observation omitted; identical result is "
                f"retained by tool call {newer_node.tool_call.call_id}; raw "
                "result remains in session history.]"
            ),
            reason=ToolResultReplacementReason.REDUNDANT_FILE_OBSERVATION,
            superseding_call_id=newer_node.tool_call.call_id,
        )

    call_order = {
        tool_node.tool_call.call_id: sequence_index
        for sequence_index, tool_node in enumerate(_visible_tool_nodes(nodes))
    }
    return tuple(sorted(
        replacements_by_call_id.values(),
        key=lambda replacement: call_order.get(replacement.call_id, -1),
    ))


def _load_live_file_content(path: Path) -> _LiveFileContent:
    try:
        return _ReadableLiveFileContent(tuple(
            path.read_text(encoding="utf-8").splitlines(),
        ))
    except FileNotFoundError:
        return _MissingLiveFileContent()
    except (UnicodeDecodeError, OSError):
        return _UnknownLiveFileContent()


def _observation_state(
    live_content: _LiveFileContent,
    returned_lines: tuple[FileLineContent, ...],
) -> FileObservationResultState:
    if isinstance(live_content, _MissingLiveFileContent):
        return (
            FileObservationResultState.STALE
            if returned_lines else FileObservationResultState.UNKNOWN
        )
    if isinstance(live_content, _UnknownLiveFileContent):
        return FileObservationResultState.UNKNOWN
    if isinstance(live_content, _ReadableLiveFileContent):
        return _returned_lines_state(list(live_content.lines), returned_lines)
    return FileObservationResultState.UNKNOWN


def _stale_file_observation_replacement(
    observation: FileObservation,
) -> ToolResultReplacement:
    return ToolResultReplacement(
        call_id=observation.call_id,
        content=(
            f"read_file({observation.display_path!r}) "
            "[stale file observation omitted; returned lines no longer "
            "match the live workspace; raw result remains in session history.]"
        ),
        reason=ToolResultReplacementReason.STALE_FILE_OBSERVATION,
    )


def _redundant_file_observation_replacement(
    observation: FileObservation,
    newer_observation: FileObservation,
) -> ToolResultReplacement:
    return ToolResultReplacement(
        call_id=observation.call_id,
        content=(
            f"read_file({observation.display_path!r}) "
            "[redundant file observation omitted; identical current lines "
            f"are retained by tool call {newer_observation.call_id}; raw "
            "result remains in session history.]"
        ),
        reason=ToolResultReplacementReason.REDUNDANT_FILE_OBSERVATION,
        superseding_call_id=newer_observation.call_id,
    )


def _visible_tool_nodes(nodes: list[HistoryNode]) -> Iterable[ToolCallNode]:
    for node in nodes:
        if not isinstance(node, TurnNode):
            continue
        if node.collapse_state is CollapseState.COLLAPSED:
            continue
        for tool_node in node.tool_call_nodes:
            if tool_node.detail_collapsed:
                continue
            yield tool_node


def _exact_duplicate_search_pairs(
    nodes: list[HistoryNode],
) -> Iterable[tuple[ToolCallNode, ToolCallNode]]:
    latest_by_signature: dict[tuple[str, str], ToolCallNode] = {}
    duplicate_pairs: list[tuple[ToolCallNode, ToolCallNode]] = []
    search_nodes = [
        tool_node for tool_node in _visible_tool_nodes(nodes)
        if (
            not tool_node.result.is_error
            and _normalized_tool_name(tool_node.tool_call.name) == "search_files"
        )
    ]
    for tool_node in reversed(search_nodes):
        signature = (
            _canonical_tool_arguments(tool_node.tool_call.arguments),
            tool_node.result.content,
        )
        newer_node = latest_by_signature.get(signature)
        if newer_node is not None:
            duplicate_pairs.append((tool_node, newer_node))
            continue
        latest_by_signature[signature] = tool_node
    duplicate_pairs.reverse()
    return tuple(duplicate_pairs)


def _searches_subsumed_by_current_read(
    nodes: list[HistoryNode],
    *,
    accumulators: dict[Path, FileContextAccumulator],
    live_file_snapshot: LiveWorkspaceFileSnapshot,
) -> tuple[tuple[ToolCallNode, FileObservation], ...]:
    """Return searches whose complete evidence is in one newer current read.

    Requiring one superseding read keeps the proof and its provenance exact.
    A search spanning several files therefore stays visible unless a later
    search has an identical result, which is handled separately.
    """
    search_nodes = {
        tool_node.tool_call.call_id: tool_node
        for tool_node in _visible_tool_nodes(nodes)
        if (
            not tool_node.result.is_error
            and _normalized_tool_name(tool_node.tool_call.name) == "search_files"
        )
    }
    search_observations_by_call_id: dict[str, list[FileObservation]] = (
        defaultdict(list)
    )
    accumulator_by_path = {
        accumulator.resolved_path: accumulator
        for accumulator in accumulators.values()
    }
    for accumulator in accumulators.values():
        for observation in accumulator.read_search_observations:
            if observation.tool_name == "search_files":
                search_observations_by_call_id[observation.call_id].append(
                    observation,
                )

    result: list[tuple[ToolCallNode, FileObservation]] = []
    for call_id, search_observations in search_observations_by_call_id.items():
        search_node = search_nodes.get(call_id)
        if search_node is None or len(search_observations) != 1:
            continue
        search_observation = search_observations[0]
        if (
            not search_observation.returned_lines
            or not search_observation.search_result_complete
        ):
            continue
        accumulator = accumulator_by_path[search_observation.resolved_path]
        for candidate in reversed(accumulator.read_search_observations):
            if (
                candidate.tool_name != "read_file"
                or candidate.sequence_index <= search_observation.sequence_index
                or not candidate.returned_lines
            ):
                continue
            live_content = live_file_snapshot.content_for(
                candidate.resolved_path,
            )
            if (
                _observation_state(live_content, candidate.returned_lines)
                is not FileObservationResultState.CURRENT
            ):
                continue
            if frozenset(search_observation.returned_lines) <= frozenset(
                candidate.returned_lines,
            ):
                result.append((search_node, candidate))
                break
    return tuple(result)


def _canonical_tool_arguments(arguments: str) -> str:
    try:
        parsed = json.loads(arguments or "{}")
    except json.JSONDecodeError:
        return arguments
    return json.dumps(parsed, sort_keys=True, separators=(",", ":"))


def render_history_with_file_context_folding(
    nodes: list[HistoryNode],
    *,
    render_plan: HistoryRenderPlan | None = None,
    live_file_snapshot: LiveWorkspaceFileSnapshot | None = None,
) -> list[Message]:
    """Render *nodes* with repeated file-tool output folded."""
    return render_history_with_file_context_visibility(
        nodes,
        render_plan=render_plan,
        live_file_snapshot=live_file_snapshot,
    ).messages


def render_history_with_file_context_visibility(
    nodes: list[HistoryNode],
    *,
    workspace_root: Path | None = None,
    render_plan: HistoryRenderPlan | None = None,
    live_file_snapshot: LiveWorkspaceFileSnapshot | None = None,
) -> RenderedHistory:
    """Render *nodes* with repeated file-tool output folded.

    If no active execution context can safely produce live file views,
    rendering falls back to the ordinary history rendering.  Omitting
    *live_file_snapshot* creates one fresh lazy snapshot for this render call.
    """
    effective_render_plan = render_plan or HistoryRenderPlan()
    effective_live_file_snapshot = (
        live_file_snapshot or LiveWorkspaceFileSnapshot()
    )
    visibility_builder = PromptVisibilitySnapshotBuilder()
    accumulators = _collect_file_context(
        nodes,
        workspace_root=workspace_root,
        render_plan=effective_render_plan,
        collection_mode=(
            _FileObservationCollectionMode.LEGACY_READABLE_FILES
            if effective_render_plan.file_observation_folding_mode
                is FileObservationFoldingMode.LEGACY_MINIMUM
            else _FileObservationCollectionMode.PROJECTED_READABLE_FILES
        ),
        live_file_snapshot=effective_live_file_snapshot,
    )
    newest_file_tool_call_ids = _newest_file_tool_turn_call_ids(
        nodes,
        render_plan=effective_render_plan,
    )
    candidates = [
        acc for acc in accumulators.values()
        if _should_render_folded_context(
            acc,
            folding_mode=effective_render_plan.file_observation_folding_mode,
            newest_file_tool_call_ids=newest_file_tool_call_ids,
        )
    ]
    if not candidates:
        rendered = _render_nodes(
            nodes,
            folded_call_ids=frozenset(),
            render_plan=effective_render_plan,
            visibility_builder=visibility_builder,
            workspace_root=workspace_root,
        )
        return RenderedHistory(
            messages=rendered,
            visibility=visibility_builder.build(),
            folded_file_tool_call_ids=frozenset(),
        )

    candidates.sort(key=lambda acc: acc.last_sequence_index, reverse=True)
    selected = candidates[:FOLDED_FILE_CONTEXT_MAX_FILES]
    selected_contexts = [
        (acc, rendered) for acc in selected
        if (
            rendered := _render_folded_file_context(
                acc,
                effective_live_file_snapshot,
            )
        )
    ]
    protected_call_ids = (
        newest_file_tool_call_ids
        | effective_render_plan.protected_file_observation_call_ids
    )
    protected_call_ids -= _stale_protected_read_call_ids(
        candidates,
        protected_call_ids=protected_call_ids,
        live_file_snapshot=effective_live_file_snapshot,
    )
    folded_call_ids = _foldable_call_ids(
        candidates=candidates,
        selected=[acc for acc, _rendered in selected_contexts],
        protected_call_ids=protected_call_ids,
    )
    if not folded_call_ids:
        rendered = _render_nodes(
            nodes,
            folded_call_ids=frozenset(),
            render_plan=effective_render_plan,
            visibility_builder=visibility_builder,
            workspace_root=workspace_root,
        )
        return RenderedHistory(
            messages=rendered,
            visibility=visibility_builder.build(),
            folded_file_tool_call_ids=frozenset(),
        )

    context_sections = [
        rendered for acc, rendered in selected_contexts
        if any(
            obs.call_id in folded_call_ids
            for obs in acc.read_search_observations
        )
    ]
    rendered = _render_nodes(
        nodes,
        folded_call_ids=folded_call_ids,
        render_plan=effective_render_plan,
        visibility_builder=visibility_builder,
        workspace_root=workspace_root,
    )
    for section in context_sections:
        visibility_builder.record_visible_file_lines(
            section.file_key,
            section.visible_lines,
        )
        visibility_builder.record_collapsed_file_ranges(
            section.file_key,
            section.collapsed_ranges,
        )
    context_text = _join_folded_file_contexts(
        [section.text for section in context_sections],
    )
    if context_text:
        rendered.append(UserMessage(content=context_text))
    return RenderedHistory(
        messages=rendered,
        visibility=visibility_builder.build(),
        folded_file_tool_call_ids=folded_call_ids,
    )


def _foldable_call_ids(
    *,
    candidates: list[FileContextAccumulator],
    selected: list[FileContextAccumulator],
    protected_call_ids: frozenset[str],
) -> frozenset[str]:
    """Return call IDs whose observed file output is fully represented."""
    selected_paths = {acc.resolved_path for acc in selected}
    paths_by_call_id: dict[str, set[Path]] = defaultdict(set)
    for acc in candidates:
        for observation in acc.read_search_observations:
            paths_by_call_id[observation.call_id].add(acc.resolved_path)
    return frozenset(
        call_id for call_id, paths in paths_by_call_id.items()
        if paths <= selected_paths and call_id not in protected_call_ids
    )


def _newest_file_tool_turn_call_ids(
    nodes: list[HistoryNode],
    *,
    render_plan: HistoryRenderPlan,
) -> frozenset[str]:
    """Keep a new turn's file observations raw until a later file turn.

    Legacy folding historically derives live context even from observations in
    durably compacted turns.  Explicit treatment projections instead honor the
    stored collapse state when choosing their newest visible file turn.
    """
    for node_index in range(len(nodes) - 1, -1, -1):
        node = nodes[node_index]
        if not isinstance(node, TurnNode):
            continue
        if (
            node_index in render_plan.collapsed_turn_node_indices
            or (
                render_plan.file_observation_folding_mode
                is not FileObservationFoldingMode.LEGACY_MINIMUM
                and node.collapse_state is CollapseState.COLLAPSED
            )
        ):
            continue
        call_ids = frozenset(
            tool_node.tool_call.call_id
            for tool_node in node.tool_call_nodes
            if (
                not tool_node.result.is_error
                and render_plan.replacement_for(
                    tool_node.tool_call.call_id,
                ) is None
                and _normalized_tool_name(tool_node.tool_call.name)
                in {"read_file", "search_files"}
            )
        )
        if call_ids:
            return call_ids
    return frozenset()


def _stale_protected_read_call_ids(
    candidates: list[FileContextAccumulator],
    *,
    protected_call_ids: frozenset[str],
    live_file_snapshot: LiveWorkspaceFileSnapshot,
) -> frozenset[str]:
    stale_call_ids: set[str] = set()
    for accumulator in candidates:
        for observation in accumulator.read_search_observations:
            if observation.call_id not in protected_call_ids:
                continue
            if observation.tool_name != "read_file":
                continue
            live_content = live_file_snapshot.content_for(
                accumulator.resolved_path,
            )
            live_lines = (
                list(live_content.lines)
                if isinstance(live_content, _ReadableLiveFileContent)
                else None
            )
            result_state = _returned_lines_state(
                live_lines,
                observation.returned_lines,
            )
            if result_state is FileObservationResultState.STALE:
                stale_call_ids.add(observation.call_id)
    return frozenset(stale_call_ids)


def _render_nodes(
    nodes: list[HistoryNode],
    *,
    folded_call_ids: frozenset[str],
    render_plan: HistoryRenderPlan,
    visibility_builder: PromptVisibilitySnapshotBuilder,
    workspace_root: Path | None,
) -> list[Message]:
    messages: list[Message] = []
    for node_index, node in enumerate(nodes):
        if not isinstance(node, TurnNode):
            messages.extend(node.render())
            continue
        messages.extend(_render_turn_node(
            node,
            node_index,
            folded_call_ids,
            render_plan,
            visibility_builder,
            workspace_root,
        ))
    return messages


def _render_turn_node(
    node: TurnNode,
    node_index: int,
    folded_call_ids: frozenset[str],
    render_plan: HistoryRenderPlan,
    visibility_builder: PromptVisibilitySnapshotBuilder,
    workspace_root: Path | None,
) -> list[Message]:
    if node.collapse_state is CollapseState.COLLAPSED:
        return node.render()
    if node_index in render_plan.collapsed_turn_node_indices:
        return [AssistantMessage(content=node.summary())]

    has_request_local_projection = any(
        (
            tool_node.tool_call.call_id in folded_call_ids
            or tool_node.tool_call.call_id
                in render_plan.detail_collapsed_call_ids
            or render_plan.replacement_for(
                tool_node.tool_call.call_id,
            ) is not None
        )
        for tool_node in node.tool_call_nodes
    )
    if not has_request_local_projection:
        messages = node.render()
        _record_visible_tool_results(
            node.tool_call_nodes,
            visibility_builder,
            messages,
            workspace_root,
        )
        return messages

    tool_calls = [
        _render_tool_call(tool_node, render_plan)
        for tool_node in node.tool_call_nodes
    ]
    results = [
        _render_tool_result(
            tcn,
            folded_call_ids,
            render_plan,
            visibility_builder,
            workspace_root,
        )
        for tcn in node.tool_call_nodes
    ]

    advisory_text = node._render_advisories()
    if advisory_text and results:
        last = results[-1]
        results[-1] = ToolResultMessage(
            call_id=last.call_id,
            content=last.content + "\n\n" + advisory_text,
            is_error=last.is_error,
            external_content_peer_statuses=last.external_content_peer_statuses,
        )

    return [
        AssistantMessage(content=node.assistant_content, tool_calls=tool_calls),
        *results,
    ]


def _render_tool_result(
    node: ToolCallNode,
    folded_call_ids: frozenset[str],
    render_plan: HistoryRenderPlan,
    visibility_builder: PromptVisibilitySnapshotBuilder,
    workspace_root: Path | None,
) -> ToolResultMessage:
    replacement = render_plan.replacement_for(node.tool_call.call_id)
    if replacement is not None:
        return ToolResultMessage(
            call_id=node.result.call_id,
            content=replacement.content,
            is_error=node.result.is_error,
            external_content_peer_statuses=(
                node.result.external_content_peer_statuses
            ),
        )

    if node.tool_call.call_id in folded_call_ids:
        summary = _folded_result_summary(node)
        return ToolResultMessage(
            call_id=node.result.call_id,
            content=(
                f"{summary} [folded into current file context below; "
                "raw result remains in session history.]"
            ),
            is_error=node.result.is_error,
            external_content_peer_statuses=(
                node.result.external_content_peer_statuses
            ),
        )

    result = _render_result_with_plan(node, render_plan)
    if node.tool_call.call_id not in render_plan.detail_collapsed_call_ids:
        _record_visible_tool_result(
            node,
            visibility_builder,
            result,
            workspace_root,
        )
    return result


def _render_tool_call(
    node: ToolCallNode,
    render_plan: HistoryRenderPlan,
) -> ToolCall:
    if node.tool_call.call_id not in render_plan.detail_collapsed_call_ids:
        return node.render_tool_call()
    return ToolCall(
        call_id=node.tool_call.call_id,
        name=node.tool_call.name,
        arguments=node.abbreviated_arguments(),
    )


def _render_result_with_plan(
    node: ToolCallNode,
    render_plan: HistoryRenderPlan,
) -> ToolResultMessage:
    if node.tool_call.call_id not in render_plan.detail_collapsed_call_ids:
        return node.render_result()
    return ToolResultMessage(
        call_id=node.result.call_id,
        content=node.summary(),
        is_error=node.result.is_error,
        external_content_peer_statuses=node.result.external_content_peer_statuses,
    )


def _record_visible_tool_results(
    tool_call_nodes: list[ToolCallNode],
    visibility_builder: PromptVisibilitySnapshotBuilder,
    rendered_messages: list[Message],
    workspace_root: Path | None,
) -> None:
    result_by_call_id = {
        message.call_id: message for message in rendered_messages
        if isinstance(message, ToolResultMessage)
    }
    for tool_call_node in tool_call_nodes:
        result = result_by_call_id.get(tool_call_node.tool_call.call_id)
        if result is not None:
            _record_visible_tool_result(
                tool_call_node,
                visibility_builder,
                result,
                workspace_root,
            )


def _record_visible_tool_result(
    node: ToolCallNode,
    visibility_builder: PromptVisibilitySnapshotBuilder,
    result: ToolResultMessage,
    workspace_root: Path | None,
) -> None:
    if result.is_error:
        return
    if _normalized_tool_name(node.tool_call.name) != "read_file":
        return
    args = _tool_arguments(node)
    raw_path = args.get("path")
    if not isinstance(raw_path, str):
        return
    file_key = _visibility_file_key(raw_path, workspace_root)
    if file_key is None:
        return
    lines = parse_line_numbered_file_content(result.content)
    if not lines:
        return
    visibility_builder.record_visible_file_lines(file_key, lines)


def _folded_result_summary(node: ToolCallNode) -> str:
    if _normalized_tool_name(node.tool_call.name) != "search_files":
        return node.summary()
    args = _tool_arguments(node)
    pattern = args.get("pattern", "?")
    path = args.get("path")
    if path is None:
        return f"search_files({pattern!r})"
    return f"search_files({pattern!r}, path={path!r})"


def _collect_file_context(
    nodes: list[HistoryNode],
    *,
    workspace_root: Path | None,
    render_plan: HistoryRenderPlan,
    collection_mode: _FileObservationCollectionMode,
    live_file_snapshot: LiveWorkspaceFileSnapshot,
) -> dict[Path, FileContextAccumulator]:
    accumulators: dict[Path, FileContextAccumulator] = {}
    sequence_index = 0
    for node_index, node in enumerate(nodes):
        if not isinstance(node, TurnNode):
            sequence_index += 1
            continue
        # Stored compaction did not historically remove observations from the
        # legacy live-context accumulator.  Request-local treatment collection
        # is intentionally stricter, while explicit render-plan decisions are
        # authoritative in every mode.
        if (
            node_index in render_plan.collapsed_turn_node_indices
            or (
                collection_mode
                is not _FileObservationCollectionMode.LEGACY_READABLE_FILES
                and node.collapse_state is CollapseState.COLLAPSED
            )
        ):
            sequence_index += len(node.tool_call_nodes)
            continue
        for tool_node in node.tool_call_nodes:
            call_id = tool_node.tool_call.call_id
            if (
                call_id in render_plan.detail_collapsed_call_ids
                or render_plan.replacement_for(call_id) is not None
                or (
                    collection_mode
                    is not _FileObservationCollectionMode.LEGACY_READABLE_FILES
                    and tool_node.detail_collapsed
                )
            ):
                sequence_index += 1
                continue
            for observation in _observations_for_tool_node(
                tool_node,
                sequence_index,
                workspace_root=workspace_root,
                collection_mode=collection_mode,
            ):
                if (
                    collection_mode
                    is not _FileObservationCollectionMode.PROJECTED_ALLOWED_PATHS
                    and not _is_readable_live_file(
                        observation.resolved_path,
                        live_file_snapshot,
                    )
                ):
                    continue
                acc = accumulators.get(observation.resolved_path)
                if acc is None:
                    acc = FileContextAccumulator(
                        display_path=observation.display_path,
                        resolved_path=observation.resolved_path,
                    )
                    accumulators[observation.resolved_path] = acc
                acc.append(observation)
            sequence_index += 1
    return accumulators


def _is_readable_live_file(
    path: Path,
    live_file_snapshot: LiveWorkspaceFileSnapshot,
) -> bool:
    return isinstance(
        live_file_snapshot.content_for(path),
        _ReadableLiveFileContent,
    )


def _observations_for_tool_node(
    tool_node: ToolCallNode,
    sequence_index: int,
    *,
    workspace_root: Path | None,
    collection_mode: _FileObservationCollectionMode,
) -> Iterable[FileObservation]:
    if tool_node.result.is_error:
        return ()
    name = _normalized_tool_name(tool_node.tool_call.name)
    args = _tool_arguments(tool_node)
    if name == "read_file":
        observation = _read_file_observation(
            tool_node,
            args,
            sequence_index,
            workspace_root=workspace_root,
            collection_mode=collection_mode,
        )
        return (observation,) if observation is not None else ()
    if name == "search_files":
        return tuple(_search_file_observations(
            tool_node,
            args,
            sequence_index,
            workspace_root=workspace_root,
            collection_mode=collection_mode,
        ))
    if name in {"edit_file", "create_file"}:
        observation = _write_file_observation(
            tool_node,
            args,
            sequence_index,
            workspace_root=workspace_root,
            collection_mode=collection_mode,
        )
        return (observation,) if observation is not None else ()
    return ()


def _tool_arguments(tool_node: ToolCallNode) -> dict[str, Any]:
    try:
        parsed = json.loads(tool_node.tool_call.arguments or "{}")
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _normalized_tool_name(name: str) -> str:
    return name.lower().replace("-", "_").replace(" ", "_")


def _read_file_observation(
    tool_node: ToolCallNode,
    args: dict[str, Any],
    sequence_index: int,
    *,
    workspace_root: Path | None,
    collection_mode: _FileObservationCollectionMode,
) -> FileObservation | None:
    raw_path = args.get("path")
    if not isinstance(raw_path, str):
        return None
    resolved = _resolve_allowed_file_path(
        raw_path,
        workspace_root,
        collection_mode=collection_mode,
    )
    if resolved is None:
        return None
    display_path = _display_path(raw_path, resolved)

    returned_lines = parse_line_numbered_file_content(
        tool_node.result.content,
    )
    read_range = _contiguous_returned_range(returned_lines)
    if not returned_lines:
        offset = _positive_int(args.get("offset"), default=1)
        limit = _optional_positive_int(args.get("limit"))
        read_range = (
            FileRange(offset, offset + limit - 1)
            if limit is not None else None
        )
    return FileObservation(
        call_id=tool_node.tool_call.call_id,
        tool_name="read_file",
        display_path=display_path,
        resolved_path=resolved,
        sequence_index=sequence_index,
        read_range=read_range,
        whole_file_read=(
            args.get("limit") is None
            and _positive_int(args.get("offset"), default=1) == 1
        ),
        returned_lines=returned_lines,
    )


def _contiguous_returned_range(
    returned_lines: tuple[FileLineContent, ...],
) -> FileRange | None:
    if not returned_lines:
        return None
    first_line = returned_lines[0].line_number
    expected_line = first_line
    for returned_line in returned_lines:
        if returned_line.line_number != expected_line:
            return None
        expected_line += 1
    return FileRange(first_line, expected_line - 1)


def _returned_lines_state(
    live_lines: list[str] | None,
    returned_lines: tuple[FileLineContent, ...],
) -> FileObservationResultState:
    if live_lines is None or not returned_lines:
        return FileObservationResultState.UNKNOWN
    for returned_line in returned_lines:
        line_index = returned_line.line_number - 1
        if line_index < 0 or line_index >= len(live_lines):
            return FileObservationResultState.STALE
        if live_lines[line_index] != returned_line.content:
            return FileObservationResultState.STALE
    return FileObservationResultState.CURRENT


def _search_file_observations(
    tool_node: ToolCallNode,
    args: dict[str, Any],
    sequence_index: int,
    *,
    workspace_root: Path | None,
    collection_mode: _FileObservationCollectionMode,
) -> Iterable[FileObservation]:
    pattern = args.get("pattern")
    pattern_text = pattern if isinstance(pattern, str) else None
    parsed_hits = _parse_search_result_file_lines(tool_node.result.content)
    search_result_complete = _search_result_is_complete(
        tool_node.result.content,
    )
    observations: list[FileObservation] = []
    for display_path, returned_lines in parsed_hits.items():
        resolved = _resolve_allowed_file_path(
            display_path,
            workspace_root,
            collection_mode=collection_mode,
        )
        if resolved is None:
            return ()
        normalized_display_path = _display_path(display_path, resolved)
        observations.append(
            FileObservation(
                call_id=tool_node.tool_call.call_id,
                tool_name="search_files",
                display_path=normalized_display_path,
                resolved_path=resolved,
                sequence_index=sequence_index,
                search_pattern=pattern_text,
                search_lines=tuple(
                    line.line_number for line in returned_lines
                ),
                search_result_complete=search_result_complete,
                returned_lines=returned_lines,
            ),
        )
    return tuple(observations)


def _write_file_observation(
    tool_node: ToolCallNode,
    args: dict[str, Any],
    sequence_index: int,
    *,
    workspace_root: Path | None,
    collection_mode: _FileObservationCollectionMode,
) -> FileObservation | None:
    raw_path = args.get("path")
    if not isinstance(raw_path, str):
        return None
    resolved = _resolve_allowed_file_path(
        raw_path,
        workspace_root,
        collection_mode=collection_mode,
    )
    if resolved is None:
        return None
    display_path = _display_path(raw_path, resolved)
    normalized_tool_name = _normalized_tool_name(tool_node.tool_call.name)
    if normalized_tool_name == "edit_file":
        provenance = FileEditResultProvenance.from_tool_result(
            tool_node.result.content,
        )
        if provenance is None:
            edited_lines = ()
            edit_context_lines = tuple(sorted(
                _parse_line_numbers(tool_node.result.content),
            ))
        else:
            edited_lines = provenance.changed_line_numbers
            edit_context_lines = provenance.deletion_anchor_line_numbers
    else:
        edited_lines = tuple(sorted(
            _parse_line_numbers(tool_node.result.content),
        ))
        edit_context_lines = ()
    return FileObservation(
        call_id=tool_node.tool_call.call_id,
        tool_name=normalized_tool_name,
        display_path=display_path,
        resolved_path=resolved,
        sequence_index=sequence_index,
        edited_lines=edited_lines,
        edit_context_lines=edit_context_lines,
    )


def _parse_search_result_file_lines(
    content: str,
) -> dict[str, tuple[FileLineContent, ...]]:
    result: dict[str, list[FileLineContent]] = defaultdict(list)
    current_path: str | None = None
    for line in content.splitlines():
        header = _SEARCH_RESULT_HEADER_RE.match(line)
        if header and not _LINE_NUMBER_RE.match(line):
            current_path = header.group("path")
            continue
        match = _LINE_NUMBER_RE.match(line)
        if current_path is not None and match:
            parsed_line = parse_line_numbered_file_content(line)
            if parsed_line:
                result[current_path].append(parsed_line[0])
    return {
        path: tuple(sorted(lines, key=lambda item: item.line_number))
        for path, lines in result.items()
    }


def search_observation_call_ids_requiring_raw_evidence(
    nodes: list[HistoryNode],
) -> frozenset[str]:
    """Return successful searches whose result grammar is incomplete.

    A truncation marker or any unparsed line may carry count or scope evidence
    that a current-file view cannot reconstruct. Versioned projections that
    promise complete search/read subsumption keep those results raw.
    """
    return frozenset(
        tool_node.tool_call.call_id
        for tool_node in _visible_tool_nodes(nodes)
        if (
            not tool_node.result.is_error
            and _normalized_tool_name(tool_node.tool_call.name) == "search_files"
            and not _search_result_is_complete(tool_node.result.content)
        )
    )


def _search_result_is_complete(content: str) -> bool:
    current_path: str | None = None
    current_path_has_line = False
    for line in content.splitlines():
        if not line.strip():
            continue
        header = _SEARCH_RESULT_HEADER_RE.match(line)
        if header and not _LINE_NUMBER_RE.match(line):
            if current_path is not None and not current_path_has_line:
                return False
            current_path = header.group("path")
            current_path_has_line = False
            continue
        if line == "--" and current_path is not None:
            continue
        if current_path is not None and _LINE_NUMBER_RE.match(line):
            if not parse_line_numbered_file_content(line):
                return False
            current_path_has_line = True
            continue
        return False
    return current_path is not None and current_path_has_line


def _parse_line_numbers(content: str) -> set[int]:
    result: set[int] = set()
    for line in content.splitlines():
        match = _LINE_NUMBER_RE.match(line)
        if match:
            result.add(int(match.group("line")))
    return result


def _resolve_allowed_file_path(
    raw_path: str,
    workspace_root: Path | None,
    *,
    collection_mode: _FileObservationCollectionMode,
) -> Path | None:
    try:
        from thorn.core._context import get_context

        ctx = get_context()
        resolved = _resolve_live_file_path(raw_path, ctx)
    except RuntimeError:
        if (
            workspace_root is None
            or collection_mode
                is _FileObservationCollectionMode.LEGACY_READABLE_FILES
        ):
            return None
        file_key = file_content_key_for_path(
            raw_path,
            workspace_root=workspace_root,
        )
        if file_key is None:
            return None
        resolved_workspace_root = workspace_root.resolve()
        if not file_key.resolved_path.is_relative_to(resolved_workspace_root):
            return None
        return file_key.resolved_path
    except OSError:
        return None

    if ctx.file_access_policy is not None and ctx.workspace_root is not None:
        if ctx.file_access_policy.check(resolved) < FileAccessLevel.READ:
            return None
    return resolved


def _resolve_live_file_path(raw_path: str, ctx: Any) -> Path:
    from thorn.core._context import resolve_path

    if ctx.workspace_root is not None:
        sandbox_relative = _sandbox_workspace_relative_path(raw_path)
        if sandbox_relative is not None:
            host_relative = _relative_to_active_workspace(
                sandbox_relative,
                ctx.workspace_root,
            )
            return (ctx.workspace_root / host_relative).resolve()
    return resolve_path(raw_path)


def _relative_to_active_workspace(
    sandbox_relative: Path,
    workspace_root: Path,
) -> Path:
    sandbox_parts = sandbox_relative.parts
    workspace_parts = workspace_root.parts
    max_prefix_len = min(len(sandbox_parts), len(workspace_parts))
    for prefix_len in range(max_prefix_len, 0, -1):
        if sandbox_parts[:prefix_len] == workspace_parts[-prefix_len:]:
            return Path(*sandbox_parts[prefix_len:])
    return sandbox_relative


def _sandbox_workspace_relative_path(raw_path: str) -> Path | None:
    posix_path = PurePosixPath(raw_path)
    try:
        relative = posix_path.relative_to(_SANDBOX_WORKSPACE_PREFIX)
    except ValueError:
        return None
    if any(part == ".." for part in relative.parts):
        return None
    return Path(*relative.parts)


def _display_path(raw_path: str, resolved: Path) -> str:
    try:
        from thorn.core._context import get_context

        ctx = get_context()
    except RuntimeError:
        return raw_path
    if ctx.workspace_root is None:
        return raw_path
    try:
        return resolved.relative_to(ctx.workspace_root).as_posix()
    except ValueError:
        return raw_path


def _visibility_file_key(
    raw_path: str | Path,
    workspace_root: Path | None,
) -> FileContentKey | None:
    if workspace_root is None:
        try:
            from thorn.core._context import get_context

            workspace_root = get_context().workspace_root
        except RuntimeError:
            workspace_root = None
    return file_content_key_for_path(raw_path, workspace_root=workspace_root)


def _should_render_folded_context(
    accumulator: FileContextAccumulator,
    *,
    folding_mode: FileObservationFoldingMode,
    newest_file_tool_call_ids: frozenset[str],
) -> bool:
    observations = accumulator.read_search_observations
    if not observations:
        return False
    if any(obs.tool_name == "search_files" for obs in observations):
        return True
    if len(observations) >= FOLDED_FILE_MIN_OBSERVATIONS:
        return True
    if folding_mode is FileObservationFoldingMode.LEGACY_MINIMUM:
        return False
    return all(
        observation.call_id not in newest_file_tool_call_ids
        for observation in observations
    )


def _join_folded_file_contexts(
    sections: list[str],
) -> str:
    if not sections:
        return ""
    return "\n\n".join([
        "[Folded file context]",
        *sections,
    ])


def _render_folded_file_context(
    accumulator: FileContextAccumulator,
    live_file_snapshot: LiveWorkspaceFileSnapshot,
) -> RenderedFoldedFileContext | None:
    live_content = live_file_snapshot.content_for(accumulator.resolved_path)
    if not isinstance(live_content, _ReadableLiveFileContent):
        return None
    lines = list(live_content.lines)

    if not lines:
        return RenderedFoldedFileContext(
            text=(
                f"[Current file context: {accumulator.display_path}]\n"
                f"Sources: {_format_sources(accumulator.observations)}\n"
                "[empty file]"
            ),
            file_key=FileContentKey(accumulator.resolved_path),
            visible_lines=(),
            collapsed_ranges=(),
        )

    line_scores, line_markers, has_whole_file_read = _score_file_lines(
        accumulator.observations,
        total_lines=len(lines),
    )
    required_edit_line_numbers = tuple(
        line_number
        for line_number, _score in sorted(
            line_scores.items(),
            key=lambda item: (-item[1], item[0]),
        )
        if line_markers.get(line_number) in {
            FileLineMarker.EDIT.value,
            FileLineMarker.EDIT_CONTEXT.value,
        }
    )
    spans = _spans_for_folded_context(
        lines,
        accumulator.display_path,
        line_scores,
        required_edit_line_numbers=required_edit_line_numbers,
        latest_returned_read_range=(
            accumulator.latest_returned_read_range
        ),
        has_whole_file_read=has_whole_file_read,
    )
    marker_text = _marker_legend(line_markers)

    from thorn.core._outline import render_outline

    outline = render_outline(
        lines,
        spans,
        char_budget=FOLDED_FILE_CONTEXT_CHAR_BUDGET,
        line_markers=line_markers if line_markers else None,
        required_line_numbers=required_edit_line_numbers,
    )
    parts = [
        f"[Current file context: {accumulator.display_path}]",
        f"Sources: {_format_sources(accumulator.observations)}",
    ]
    if marker_text:
        parts.append(marker_text)
    parts.append(outline.text)
    visible_lines = tuple(
        FileLineContent(line_number=line_number, content=content)
        for line_number, content in outline.visible_lines
    )
    collapsed_ranges = tuple(
        FileLineRange(start_line=start_line, end_line=end_line)
        for start_line, end_line in outline.collapsed_ranges
    )
    return RenderedFoldedFileContext(
        text="\n".join(parts),
        file_key=FileContentKey(accumulator.resolved_path),
        visible_lines=visible_lines,
        collapsed_ranges=collapsed_ranges,
    )


def _score_file_lines(
    observations: list[FileObservation],
    *,
    total_lines: int,
) -> tuple[dict[int, float], dict[int, str], bool]:
    line_scores: dict[int, float] = {}
    marker_sets: dict[int, set[FileLineMarker]] = defaultdict(set)
    latest_index = max(obs.sequence_index for obs in observations)
    has_whole_file_read = False

    for observation in observations:
        decay = FOLDED_FILE_SALIENCE_DECAY ** (latest_index - observation.sequence_index)
        if observation.whole_file_read:
            has_whole_file_read = True

        if observation.read_range is not None:
            _add_range_salience(
                line_scores,
                marker_sets,
                observation.read_range,
                total_lines=total_lines,
                salience=4.0 * decay,
                marker=FileLineMarker.READ,
            )

        for line in observation.search_lines:
            _add_line_salience(
                line_scores,
                marker_sets,
                line,
                total_lines=total_lines,
                salience=5.0 * decay,
                marker=FileLineMarker.SEARCH,
            )

        for line in observation.edited_lines:
            _add_line_salience(
                line_scores,
                marker_sets,
                line,
                total_lines=total_lines,
                salience=6.0 * decay,
                marker=FileLineMarker.EDIT,
            )

        for line in observation.edit_context_lines:
            _add_line_salience(
                line_scores,
                marker_sets,
                line,
                total_lines=total_lines,
                salience=6.0 * decay,
                marker=FileLineMarker.EDIT_CONTEXT,
            )

    line_markers = {
        line: _dominant_marker(markers).value
        for line, markers in marker_sets.items()
    }
    return line_scores, line_markers, has_whole_file_read


def _add_range_salience(
    line_scores: dict[int, float],
    marker_sets: dict[int, set[FileLineMarker]],
    line_range: FileRange,
    *,
    total_lines: int,
    salience: float,
    marker: FileLineMarker,
) -> None:
    clamped = line_range.clamp(total_lines)
    if clamped is None:
        return
    for line in range(clamped.start_line, clamped.end_line + 1):
        _add_line_salience(
            line_scores,
            marker_sets,
            line,
            total_lines=total_lines,
            salience=salience,
            marker=marker,
        )


def _add_line_salience(
    line_scores: dict[int, float],
    marker_sets: dict[int, set[FileLineMarker]],
    line: int,
    *,
    total_lines: int,
    salience: float,
    marker: FileLineMarker,
) -> None:
    if line < 1 or line > total_lines:
        return
    line_scores[line] = line_scores.get(line, 0.0) + salience
    marker_sets[line].add(marker)


def _dominant_marker(markers: set[FileLineMarker]) -> FileLineMarker:
    for marker in (
        FileLineMarker.EDIT,
        FileLineMarker.EDIT_CONTEXT,
        FileLineMarker.SEARCH,
        FileLineMarker.READ,
    ):
        if marker in markers:
            return marker
    return FileLineMarker.READ


def _spans_for_folded_context(
    lines: list[str],
    display_path: str,
    line_scores: dict[int, float],
    *,
    required_edit_line_numbers: tuple[int, ...],
    latest_returned_read_range: FileRange | None,
    has_whole_file_read: bool,
) -> list["OutputSpan"]:
    from thorn.core._outline import (
        compute_outline_spans,
        spans_for_regions,
    )

    total_lines = len(lines)
    if total_lines <= FOLDED_FILE_CONTEXT_LINE_BUDGET:
        return spans_for_regions(total_lines, [(1, total_lines)], context_lines=0)

    # Reserve space for exact edits and conservative edit anchors before the
    # broad latest-read guarantee. Otherwise a large read can consume the
    # whole budget even though the view claims a later edit as a source.
    edited_line_scores = {
        line_number: line_scores[line_number]
        for line_number in required_edit_line_numbers
    }
    regions = _salient_regions(edited_line_scores, total_lines)
    if latest_returned_read_range is not None:
        latest_range = latest_returned_read_range.clamp(total_lines)
        if latest_range is not None:
            regions.append((latest_range.start_line, latest_range.end_line))
    regions.extend(_salient_regions(
        {
            line: score
            for line, score in line_scores.items()
            if line not in edited_line_scores
        },
        total_lines,
    ))
    if has_whole_file_read:
        regions.extend(
            _visible_regions_from_spans(
                compute_outline_spans(
                    lines,
                    line_budget=FOLDED_FILE_CONTEXT_OUTLINE_LINE_BUDGET,
                    file_path=display_path,
                ),
            ),
        )

    if not regions:
        return compute_outline_spans(
            lines,
            line_budget=FOLDED_FILE_CONTEXT_LINE_BUDGET,
            file_path=display_path,
        )

    regions = _limit_regions_to_line_budget(
        regions,
        line_budget=FOLDED_FILE_CONTEXT_LINE_BUDGET,
    )
    return spans_for_regions(total_lines, regions, context_lines=0)


def _salient_regions(
    line_scores: dict[int, float],
    total_lines: int,
) -> list[tuple[int, int]]:
    regions: list[tuple[int, int]] = []
    for line, _score in sorted(
        line_scores.items(),
        key=lambda item: (-item[1], item[0]),
    ):
        start = max(1, line - FOLDED_FILE_CONTEXT_AROUND_SALIENT_LINE)
        end = min(total_lines, line + FOLDED_FILE_CONTEXT_AROUND_SALIENT_LINE)
        regions.append((start, end))
    return regions


def _visible_regions_from_spans(
    spans: Iterable["OutputSpan"],
) -> list[tuple[int, int]]:
    return [
        (span.start_line, span.end_line)
        for span in spans
        if span.visible
    ]


def _limit_regions_to_line_budget(
    regions: list[tuple[int, int]],
    *,
    line_budget: int,
) -> list[tuple[int, int]]:
    selected_lines: set[int] = set()
    for start, end in regions:
        for line in range(start, end + 1):
            if line in selected_lines:
                continue
            selected_lines.add(line)
            if len(selected_lines) == line_budget:
                return _regions_for_selected_lines(selected_lines)
    return _regions_for_selected_lines(selected_lines)


def _regions_for_selected_lines(
    selected_lines: set[int],
) -> list[tuple[int, int]]:
    return _merge_regions([
        (line, line)
        for line in selected_lines
    ])


def _merge_regions(regions: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not regions:
        return []
    result: list[tuple[int, int]] = []
    for start, end in sorted(regions):
        if not result or start > result[-1][1] + 1:
            result.append((start, end))
            continue
        prev_start, prev_end = result[-1]
        result[-1] = (prev_start, max(prev_end, end))
    return result


def _format_sources(observations: list[FileObservation]) -> str:
    parts: list[str] = []
    for observation in observations[:8]:
        if observation.tool_name == "read_file":
            if observation.read_range is None:
                parts.append(f"read_file {observation.call_id} whole file")
            else:
                rng = observation.read_range
                parts.append(
                    f"read_file {observation.call_id} "
                    f"lines {rng.start_line}-{rng.end_line}"
                )
        elif observation.tool_name == "search_files":
            pattern = observation.search_pattern or "?"
            parts.append(
                f"search_files {observation.call_id} pattern={pattern!r}"
            )
        else:
            parts.append(f"{observation.tool_name} {observation.call_id}")
    remaining = len(observations) - len(parts)
    if remaining > 0:
        parts.append(f"{remaining} more")
    return "; ".join(parts)


def _marker_legend(line_markers: dict[int, str]) -> str:
    used = set(line_markers.values())
    labels: list[str] = []
    if FileLineMarker.SEARCH.value in used:
        labels.append("! search hit")
    if FileLineMarker.READ.value in used:
        labels.append("* requested/read line")
    if FileLineMarker.EDIT.value in used:
        labels.append("~ edited line")
    if FileLineMarker.EDIT_CONTEXT.value in used:
        labels.append("- edit boundary/context line")
    if not labels:
        return ""
    return "Legend: " + ", ".join(labels)


def _positive_int(value: Any, *, default: int) -> int:
    if isinstance(value, bool):
        return default
    if isinstance(value, int) and value > 0:
        return value
    return default


def _optional_positive_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int) and value > 0:
        return value
    return None
