"""Agent loop: streaming, tool dispatch, structured-output extraction.

Two modes of operation:

* **Text mode** (``result_type is str``): the assistant's final text
  response is returned directly.
* **Structured mode** (any other ``result_type``): synthetic
  ``return_result`` and ``raise_error`` tools are injected, and the
  loop extracts and validates the typed result.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from thorn.runtime._working_set_telemetry import WorkingSetTelemetry

from thorn.core._context import ExecutionContext
from thorn.core._context_ledger import (
    ProviderContextLedger,
    project_history_for_provider,
)
from thorn.core._executor import (
    ExecutorRouter,
    ToolInvocation,
    ToolRegistry,
    ToolVenue,
    build_default_router,
    build_registry_from_wrapped_tools,
    build_split_router,
)
from thorn.core._history import (
    DEFAULT_HIGH_WATERMARK,
    AdvisoryNode,
    HistoryTree,
    ToolCallNode,
    estimate_tokens,
)
from thorn.core._messages import (
    AssistantMessage,
    Message,
    ToolCall,
    ToolResultMessage,
)
from thorn.core._prompt_trace import (
    PromptTraceContextSource,
    PromptTraceManifest,
)
from thorn.core._prompt_visibility import (
    FileContentVisibilityComparison,
    parse_line_numbered_file_content,
)
from thorn.core._provider import FinishChunk, TextChunk, ToolCallChunk, UsageChunk
from thorn.core._provider_telemetry import (
    ProviderAttemptNextAction,
    ProviderAttemptOutcome,
    ProviderAttemptTelemetry,
    ProviderContextMetrics,
)
from thorn.core._read_file_history import (
    READ_FILE_REUSE_TELEMETRY_SCHEMA_VERSION,
    ReadFileResultHistory,
    ReadFileReuseKind,
    ReadFileReuseObservation,
)
from thorn.core._retry import RetryPolicy
from thorn.core._schema import (
    RAISE_ERROR_SCHEMA,
    make_return_result_schema,
    validate_result,
)
from thorn.core._search_files_history import SearchFilesDuplicateObservation
from thorn.core._shell_inspection import (
    ShellInspectionCommand,
    parse_shell_inspection_command,
)
from thorn.core._validation_convergence import (
    ValidationActionEpochReason,
    ValidationActionEpochTelemetry,
    ValidationActionTool,
    ValidationConvergenceObservation,
    ValidationConvergencePolicy,
    ValidationConvergenceTelemetry,
    ValidationConvergenceTracker,
    ValidationProgressEffect,
    ValidationTelemetry,
    WorkspaceContentConvergenceObservation,
    WorkspaceContentConvergencePolicy,
    WorkspaceContentConvergenceTelemetry,
    WorkspaceContentConvergenceTracker,
    parse_validation_command,
    validation_outcome_from_tool_result,
)
from thorn.core._workspace_content import (
    WorkspaceContentCollectionFailure,
    WorkspaceContentExcludedPath,
    WorkspaceContentExclusionReason,
    WorkspaceContentSnapshot,
    collect_workspace_content_snapshot,
)
from thorn.core.errors import (
    AgentFailureError,
    LoopLimitError,
    LoopNoProgressError,
    LoopRepetitionError,
    ProviderError,
    ProviderUnavailableError,
    RateLimitError,
    SkillError,
    TransientProviderError,
)

logger = logging.getLogger(__name__)

# The default retry policy for provider-level calls is loaded from
# environment variables at module import.  Tests that need to
# exercise backoff boundaries without waiting for real seconds
# monkeypatch this attribute (or pass ``asyncio.sleep`` replacements
# into the retry loop via the test harness).  Operators override by
# setting ``THORN_PROVIDER_RETRY_*`` before the daemon starts.
_DEFAULT_RETRY_POLICY = RetryPolicy.from_env()

DEFAULT_MAX_TOOL_ROUNDS_WITHOUT_PROGRESS = 16
"""Default inner-loop budget for successful but unproductive tool rounds."""

MAX_TOOL_ROUNDS_WITHOUT_PROGRESS_ENV_VAR = "THORN_MAX_TOOL_ROUNDS_WITHOUT_PROGRESS"
"""Environment override for the session default no-progress guard."""

READ_FILE_PRIOR_CONTEXT_HINT_THRESHOLD = 0.95
"""Minimum exact visible-line fraction before read_file gets a hint."""

# Sentinel returned by _execute_tool_calls when the structured-mode
# ``return_result`` tool is invoked.
_RESULT_SENTINEL = object()


class _AssistantRoundKind(StrEnum):
    TEXT = "text"
    TOOL_CALLS = "tool_calls"


@dataclass(frozen=True)
class _ToolCallFingerprint:
    name: str
    arguments: str


@dataclass(frozen=True)
class _ToolResultFingerprint:
    content: str
    is_error: bool


@dataclass(frozen=True)
class _AssistantRoundFingerprint:
    kind: _AssistantRoundKind
    text: str
    tool_calls: tuple[_ToolCallFingerprint, ...] = ()
    tool_results: tuple[_ToolResultFingerprint, ...] = ()


@dataclass
class _ConsecutiveRepetitionTracker:
    limit: int
    previous: _AssistantRoundFingerprint | None = None
    count: int = 0

    def observe(self, fingerprint: _AssistantRoundFingerprint) -> int:
        if self.previous == fingerprint:
            self.count += 1
            return self.count

        self.previous = fingerprint
        self.count = 1
        return self.count

    def reset(self) -> None:
        self.previous = None
        self.count = 0


@dataclass(frozen=True)
class _MaterialWorkingSetSnapshot:
    phase: str
    focused_inbox_item_id: str | None
    objective: str | None
    last_action_summary: str | None
    validation: tuple[str, str, str | None] | None
    no_validation_rationale: str | None
    blocker: tuple[str, str] | None


@dataclass(frozen=True)
class _TodoLifecycleSnapshot:
    item_statuses: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class _ProgressSnapshot:
    working_set: _MaterialWorkingSetSnapshot | None
    todos: _TodoLifecycleSnapshot | None


@dataclass(frozen=True)
class _ResourceAccessFingerprint:
    tool_name: str
    resource: tuple[tuple[str, str], ...]

    def render(self) -> str:
        rendered_args = ", ".join(
            f"{name}={value}" for name, value in self.resource
        )
        return f"{self.tool_name}({rendered_args})"


@dataclass(frozen=True)
class _ToolRoundProgress:
    made_progress: bool
    reason: str
    tool_pattern: str
    resource_accesses: tuple[_ResourceAccessFingerprint, ...] = ()
    validation_telemetry: tuple[ValidationTelemetry, ...] = ()


@dataclass
class _NoProgressTracker:
    rounds_without_progress: int = 0
    recent_tool_patterns: list[str] = field(default_factory=list)
    last_reason: str = ""
    repeated_resource: _ResourceAccessFingerprint | None = None
    repeated_resource_count: int = 0
    warning_emitted: bool = False

    def observe(self, progress: _ToolRoundProgress) -> None:
        if progress.made_progress:
            self.reset()
            return

        self.rounds_without_progress += 1
        self.last_reason = progress.reason
        self.recent_tool_patterns.append(progress.tool_pattern)
        del self.recent_tool_patterns[:-4]
        self._observe_resource_access(progress.resource_accesses)

    def reset(self) -> None:
        self.rounds_without_progress = 0
        self.recent_tool_patterns.clear()
        self.last_reason = ""
        self.repeated_resource = None
        self.repeated_resource_count = 0
        self.warning_emitted = False

    def _observe_resource_access(
        self,
        resource_accesses: tuple[_ResourceAccessFingerprint, ...],
    ) -> None:
        if len(resource_accesses) != 1:
            self.repeated_resource = None
            self.repeated_resource_count = 0
            return

        resource = resource_accesses[0]
        if self.repeated_resource == resource:
            self.repeated_resource_count += 1
            return

        self.repeated_resource = resource
        self.repeated_resource_count = 1


class _WrappedTool:
    """Lightweight wrapper that pairs a tool schema with an execute callback.

    When *call_node_class* is set, ``HistoryTree.append_turn`` will
    construct that ``ToolCallNode`` subclass instead of the base class,
    enabling ``isinstance``-based identification in downstream code.

    ``venue`` names the :class:`~thorn.core._executor.ToolVenue` the
    tool is meant to run in.  The agent loop uses it only indirectly
    (via :class:`~thorn.core._executor.ToolRegistry` + router); for
    non-registry callers (``wrap_function`` etc.) it still defaults to
    ``IN_PROCESS`` which matches today's behavior.

    ``mcp_server_config`` and ``mcp_tool_name`` are populated only for
    tools sourced from an MCP server.  The brain owns name resolution
    (a tool may be exposed to the model under a server-prefixed name to
    avoid collisions); ``mcp_tool_name`` is the *unprefixed* name the
    server itself knows the tool by, which is what the daemon must
    forward to ``ClientSession.call_tool``.  Both are ``None`` for
    built-in tools and are propagated into
    :class:`~thorn.core._executor.ToolRegistryEntry` and onward into
    :class:`~thorn.core._executor.ToolInvocation` so the daemon
    executor can recognise an MCP-routed call.
    """

    __slots__ = (
        "schema",
        "execute",
        "call_node_class",
        "venue",
        "mcp_server_config",
        "mcp_tool_name",
    )

    def __init__(
        self,
        schema: dict[str, Any],
        execute: Any,  # async callable(**kwargs) -> str
        call_node_class: type[ToolCallNode] | None = None,
        venue: ToolVenue = ToolVenue.IN_PROCESS,
        mcp_server_config: Any = None,  # MCPServerConfig | None; Any avoids cyclic import
        mcp_tool_name: str | None = None,
    ) -> None:
        self.schema = schema
        self.execute = execute
        self.call_node_class = call_node_class
        self.venue = venue
        self.mcp_server_config = mcp_server_config
        self.mcp_tool_name = mcp_tool_name


@dataclass(frozen=True)
class ProviderRequestSnapshot:
    """Complete prompt-side state for one logical provider request.

    A snapshot is materialized once before a provider request begins. Provider
    retries reuse it; after tools execute, the next request receives a newly
    projected snapshot. Keeping prompts, tools, trace provenance, and working-
    set telemetry in one value prevents a request from mixing state observed at
    different points in the session.
    """

    system_prompts: tuple[str, ...]
    tools: tuple[_WrappedTool, ...]
    prompt_trace_manifest: PromptTraceManifest | None = None
    working_set_telemetry: WorkingSetTelemetry | None = None


class ProviderRequestProjector(Protocol):
    """Materialize current prompt-side state for a provider request."""

    async def project(self) -> ProviderRequestSnapshot:
        """Return one internally consistent snapshot of current state."""
        ...


@dataclass(frozen=True)
class ToolRoundTermination:
    """Text-mode result returned by a per-prompt terminal policy."""

    text: str


class ToolRoundTerminalPolicy(Protocol):
    """Decide whether a completed tool round ends this text prompt."""

    def evaluate(
        self,
        *,
        assistant_text: str,
        tool_calls: tuple[ToolCall, ...],
        result_messages: tuple[ToolResultMessage, ...],
        session: Any,
    ) -> ToolRoundTermination | None:
        """Return a terminal result, or ``None`` to continue the loop."""
        ...


async def run_agent_loop(
    *,
    context: ExecutionContext,
    user_prompt: str | None,
    tools: list[_WrappedTool] | None = None,
    system_prompts: list[str] | None = None,
    result_type: type | None = None,
    max_tool_rounds: int = 50,
    max_repeated_rounds: int = 3,
    max_tool_rounds_without_progress: int | None = None,
    max_failures: int = 5,
    history: HistoryTree | None = None,
    session: Any = None,
    prompt_trace_manifest: PromptTraceManifest | None = None,
    provider_request_projector: ProviderRequestProjector | None = None,
    tool_round_terminal_policy: ToolRoundTerminalPolicy | None = None,
    _housekeeping: bool = False,
) -> Any:
    """Drive the request -> tool-call -> response cycle.

    Returns a ``str`` in text mode or a validated value of *result_type*
    in structured mode.

    If *history* is provided and *user_prompt* is not ``None``, the
    prompt is appended to the tree before the loop begins.  When
    *user_prompt* is ``None``, the caller is responsible for having
    already populated the history (e.g. after context injection).
    The tree is mutated in place, so the caller retains the accumulated
    history after the call returns (enabling multi-turn patterns).
    If *history* is ``None`` (the default), a fresh tree is created
    internally.

    When *session* is provided, it is passed to status providers so
    they can tailor advisory content to the active session/agent.

    *provider_request_projector* replaces the static *tools*,
    *system_prompts*, and *prompt_trace_manifest* inputs. It is called once
    before each logical provider request, while retries of that request reuse
    the returned snapshot.

    *tool_round_terminal_policy* may end a text-mode prompt after a tool
    round has executed and been appended to history.  It is scoped to this
    prompt invocation; general agent and gateway loops do not infer
    terminality from tool names.

    ``max_repeated_rounds`` aborts a stuck loop before
    ``max_tool_rounds`` when identical text-only structured responses
    or identical tool-error rounds repeat consecutively.

    The *_housekeeping* flag is set by the housekeeping subsystem to
    prevent recursive housekeeping triggers within a sub-loop.
    External callers should not set this.
    """
    if max_repeated_rounds < 2:
        raise ValueError("max_repeated_rounds must be at least 2")
    if (
        max_tool_rounds_without_progress is not None
        and max_tool_rounds_without_progress < 0
    ):
        raise ValueError("max_tool_rounds_without_progress must be >= 0")
    effective_max_tool_rounds_without_progress = (
        max_tool_rounds_without_progress
        if max_tool_rounds_without_progress is not None
        else _default_max_tool_rounds_without_progress(session)
    )

    if provider_request_projector is not None and (
        tools
        or system_prompts
        or prompt_trace_manifest is not None
    ):
        raise ValueError(
            "provider_request_projector replaces tools, system_prompts, "
            "and prompt_trace_manifest",
        )

    structured = result_type is not None and result_type is not str
    if structured and tool_round_terminal_policy is not None:
        raise ValueError(
            "tool_round_terminal_policy is only valid for text-mode prompts",
        )

    structured_instruction: str | None = None
    if structured:
        structured_instruction = (
            "You MUST call the `return_result` tool to deliver your final "
            "answer.  Do NOT respond with plain text — always use the tool.  "
            "If you cannot fulfil the request, call `raise_error` instead."
        )

    static_request_snapshot = ProviderRequestSnapshot(
        system_prompts=tuple(system_prompts or ()),
        tools=tuple(tools or ()),
        prompt_trace_manifest=prompt_trace_manifest,
    )

    # -- conversation history (tree) ---------------------------------------
    if history is None:
        history = HistoryTree()
    if user_prompt is not None:
        history.append_user_prompt(user_prompt)

    context_window = context.context_window

    consecutive_failures = 0
    repetition_tracker = _ConsecutiveRepetitionTracker(
        limit=max_repeated_rounds,
    )
    no_progress_tracker = _NoProgressTracker()
    validation_convergence_tracker = ValidationConvergenceTracker()
    workspace_content_convergence_tracker = WorkspaceContentConvergenceTracker()

    for round_num in range(max_tool_rounds):
        request_snapshot = (
            await provider_request_projector.project()
            if provider_request_projector is not None
            else static_request_snapshot
        )

        all_tools = list(request_snapshot.tools)
        registry = build_registry_from_wrapped_tools(all_tools)
        if context.sandbox_executor is not None:
            router = build_split_router(all_tools, context.sandbox_executor)
        else:
            router = build_default_router(all_tools)

        all_schemas = registry.schemas()
        all_schemas.append(RAISE_ERROR_SCHEMA)
        if structured:
            all_schemas.append(make_return_result_schema(result_type))

        context_system_prompts = list(context.system_prompts)
        per_call_prompts = list(request_snapshot.system_prompts)
        prompts = [*context_system_prompts, *per_call_prompts]
        if structured_instruction is not None:
            prompts.append(structured_instruction)
        provider_prompt_trace_manifest = _final_prompt_trace_manifest(
            context_system_prompts=context_system_prompts,
            per_call_system_prompts=per_call_prompts,
            per_call_prompt_trace_manifest=(
                request_snapshot.prompt_trace_manifest
            ),
            structured_instruction=structured_instruction,
        )
        overhead_tokens = _estimate_overhead(prompts, all_schemas)

        if request_snapshot.working_set_telemetry is not None:
            await context.event_sink.on_working_set_telemetry(
                request_snapshot.working_set_telemetry,
                scope=context.scope,
            )

        request_id = uuid.uuid4().hex
        if context.context_budget_policy is None:
            rendered_history = history.render_with_visibility(
                workspace_root=context.workspace_root,
            )
            rendered = rendered_history.messages
            rendered_visibility = rendered_history.visibility
            rendered_history_tokens = _estimate_messages(rendered)
            history_ledger = None
        else:
            projected_history = project_history_for_provider(
                history,
                workspace_root=context.workspace_root,
                context_window=context_window,
                estimated_overhead_tokens=overhead_tokens,
                policy=context.context_budget_policy,
            )
            rendered = list(projected_history.messages)
            rendered_visibility = projected_history.visibility
            rendered_history_tokens = (
                projected_history.ledger.estimated_history_tokens_final.value
            )
            history_ledger = projected_history.ledger
        context.prompt_visibility.remember(
            request_id,
            rendered_visibility,
        )
        text, tool_calls, _finish, usage = await _request_completion(
            context=context,
            request_id=request_id,
            system_prompts=prompts,
            tool_schemas=all_schemas,
            messages=rendered,
            history_node_count=len(history.nodes),
            estimated_history_tokens=rendered_history_tokens,
            estimated_overhead_tokens=overhead_tokens,
            history_ledger=history_ledger,
            prompt_trace_manifest=provider_prompt_trace_manifest,
            consecutive_failures=consecutive_failures,
            max_failures=max_failures,
        )
        consecutive_failures = 0

        if usage is not None:
            context.usage.add(
                usage.get("prompt_tokens", 0),
                usage.get("completion_tokens", 0),
                usage.get("total_tokens", 0),
            )

        # -- no tool calls: either done (text mode) or nudge (structured) --
        if not tool_calls:
            history.append_turn(AssistantMessage(content=text), [])
            if not structured:
                return text
            history.append_user_prompt(
                "You must call the `return_result` tool with your "
                "answer, or `raise_error` if you cannot proceed.  "
                "Do not reply with plain text."
            )
            await _raise_if_repeated(
                tracker=repetition_tracker,
                fingerprint=_fingerprint_text_round(text),
                context=context,
            )
            continue

        # -- dispatch tool calls -------------------------------------------
        progress_snapshot_before = _progress_snapshot(context, session)
        (
            result_msgs,
            call_node_classes,
            advisory_nodes,
            captured,
            validation_content_snapshots,
        ) = (
            await _execute_tool_calls(
                tool_calls=tool_calls,
                registry=registry,
                router=router,
                context=context,
                result_type=result_type if structured else None,
                session=session,
                source_render_id=request_id,
                validation_convergence_policy=(
                    context.validation_convergence_policy
                ),
            )
        )
        progress_snapshot_after = _progress_snapshot(context, session)

        turn_node = history.append_turn(
            AssistantMessage(content=text, tool_calls=tool_calls),
            result_msgs,
            advisory_nodes=advisory_nodes or None,
            call_node_classes=call_node_classes or None,
        )

        progress = _classify_tool_round_progress(
            tool_calls=tool_calls,
            result_messages=result_msgs,
            captured=captured,
            before=progress_snapshot_before,
            after=progress_snapshot_after,
            validation_convergence_tracker=validation_convergence_tracker,
            workspace_content_convergence_tracker=(
                workspace_content_convergence_tracker
            ),
            validation_convergence_policy=(
                context.validation_convergence_policy
            ),
            validation_content_snapshots=validation_content_snapshots,
            source_render_id=request_id,
        )
        for telemetry in progress.validation_telemetry:
            if isinstance(telemetry, ValidationActionEpochTelemetry):
                await context.event_sink.on_validation_action_epoch(
                    telemetry,
                    scope=context.scope,
                )
                continue
            if isinstance(
                telemetry,
                (
                    ValidationConvergenceTelemetry,
                    WorkspaceContentConvergenceTelemetry,
                ),
            ):
                await context.event_sink.on_validation_convergence(
                    telemetry,
                    scope=context.scope,
                )
                continue
            raise AssertionError(f"unknown validation telemetry: {telemetry!r}")

        if tool_round_terminal_policy is not None:
            termination = tool_round_terminal_policy.evaluate(
                assistant_text=text,
                tool_calls=tuple(tool_calls),
                result_messages=tuple(result_msgs),
                session=session,
            )
            if termination is not None:
                return termination.text

        no_progress_tracker.observe(progress)
        if not progress.made_progress:
            await _warn_if_no_progress(
                tracker=no_progress_tracker,
                max_tool_rounds_without_progress=(
                    effective_max_tool_rounds_without_progress
                ),
                context=context,
                session=session,
                advisory_nodes=turn_node.advisory_nodes,
            )
            await _raise_if_no_progress(
                tracker=no_progress_tracker,
                max_tool_rounds_without_progress=(
                    effective_max_tool_rounds_without_progress
                ),
                context=context,
                session=session,
            )

        # -- compaction check + housekeeping trigger ------------------------
        if context_window is not None and usage is not None:
            prompt_tokens = usage.get("prompt_tokens", 0)
            hard_prompt_threshold = int(
                context_window
                * _hard_context_window_fraction(context)
            )
            if prompt_tokens > hard_prompt_threshold:
                compact_result = history.compact(
                    context_budget=context_window,
                    overhead_tokens=overhead_tokens,
                    actual_prompt_tokens=prompt_tokens,
                )
                if compact_result.estimated_savings > 0:
                    await context.event_sink.on_status(
                        f"compaction: collapsed {compact_result.nodes_collapsed} nodes, "
                        f"{compact_result.tool_calls_detail_collapsed} tool calls detail-collapsed, "
                        f"~{compact_result.estimated_savings} est. tokens saved "
                        f"({compact_result.tokens_before} -> {compact_result.tokens_after})",
                        scope=context.scope,
                    )

                # Escalate to housekeeping when compaction alone
                # cannot bring the context below the high watermark.
                if (
                    not _housekeeping
                    and compact_result.tokens_after
                        > hard_prompt_threshold
                ):
                    from thorn.core._housekeeping import perform_housekeeping

                    await perform_housekeeping(
                        context=context,
                        history=history,
                        all_tools=all_tools,
                        system_prompts=per_call_prompts,
                    )

        if captured is not _RESULT_SENTINEL:
            return captured

        tool_round_fingerprint = _fingerprint_repeated_tool_error_round(
            text=text,
            tool_calls=tool_calls,
            result_messages=result_msgs,
        )
        if tool_round_fingerprint is None:
            repetition_tracker.reset()
        else:
            await _raise_if_repeated(
                tracker=repetition_tracker,
                fingerprint=tool_round_fingerprint,
                context=context,
            )

    raise LoopLimitError(
        f"agent loop exceeded {max_tool_rounds} rounds", max_tool_rounds,
    )


def _default_max_tool_rounds_without_progress(session: Any) -> int:
    if session is None:
        return 0
    if getattr(session, "key", None) is None:
        return 0
    agent = getattr(session, "agent", None)
    if getattr(agent, "id", None) is None:
        return 0
    configured = os.environ.get(MAX_TOOL_ROUNDS_WITHOUT_PROGRESS_ENV_VAR)
    if configured is not None and configured.strip():
        try:
            value = int(configured)
        except ValueError:
            logger.warning(
                "%s must be an integer; using default %d",
                MAX_TOOL_ROUNDS_WITHOUT_PROGRESS_ENV_VAR,
                DEFAULT_MAX_TOOL_ROUNDS_WITHOUT_PROGRESS,
            )
        else:
            if value >= 0:
                return value
            logger.warning(
                "%s must be >= 0; using default %d",
                MAX_TOOL_ROUNDS_WITHOUT_PROGRESS_ENV_VAR,
                DEFAULT_MAX_TOOL_ROUNDS_WITHOUT_PROGRESS,
            )
    return DEFAULT_MAX_TOOL_ROUNDS_WITHOUT_PROGRESS


_ACTION_PROGRESS_TOOL_NAMES = frozenset({
    "create_file",
    "edit_file",
    "delete_file",
    "move_file",
    "write_journal",
    "forge_create_issue",
    "forge_update_issue",
    "forge_post_comment",
    "forge_create_change_request",
    "forge_mark_notification_done",
    "complete_focused_work",
    "park_focused_work",
    "update_inbox_item",
})

_WORKSPACE_MUTATION_TOOL_NAMES = frozenset({
    ValidationActionTool.CREATE_FILE,
    ValidationActionTool.EDIT_FILE,
    ValidationActionTool.DELETE_FILE,
    ValidationActionTool.MOVE_FILE,
})

_OBSERVATION_TOOL_NAMES = frozenset({
    "find_files",
    "forge_get_change_request",
    "forge_get_project_info",
    "forge_list_change_requests",
    "forge_list_comments",
    "forge_list_issues",
    "forge_read_file",
    "forge_read_issue",
    "list_directory",
    "list_inbox_items",
    "list_peers",
    "list_session_todos",
    "peer_by_account",
    "read_file",
    "read_inbox_item",
    "read_journal",
    "search_files",
})

_RUN_SHELL_PROGRESS_PATTERNS = (
    "pytest",
    "ruff",
    "mypy",
    "pyright",
    "npm test",
    "pnpm test",
    "yarn test",
    "cargo test",
    "go test",
    "git commit",
    "git clone",
    "git fetch",
    "git push",
    "git merge",
    "git pull",
    "git rebase",
    "git cherry-pick",
    "git switch -c",
    "git checkout -b",
    "uv sync",
)


def _progress_snapshot(
    context: ExecutionContext,
    session: Any,
) -> _ProgressSnapshot:
    return _ProgressSnapshot(
        working_set=_material_working_set_snapshot(session),
        todos=_todo_lifecycle_snapshot(context, session),
    )


def _workspace_content_excluded_paths(
    context: ExecutionContext,
    session: Any,
) -> tuple[WorkspaceContentExcludedPath, ...]:
    """Return exact framework-owned paths proven by the active runtime."""
    workspace_root = context.workspace_root
    runtime = context.runtime
    agent = getattr(session, "agent", None)
    agent_id = getattr(agent, "id", None)
    paths = getattr(runtime, "paths", None)
    if workspace_root is None or paths is None or agent_id is None:
        return ()
    try:
        toolhost_log = paths.agent_toolhost_log(agent_id).resolve()
        relative_path = toolhost_log.relative_to(workspace_root.resolve())
    except (AttributeError, OSError, ValueError):
        return ()
    return (
        WorkspaceContentExcludedPath(
            relative_path=PurePosixPath(relative_path.as_posix()),
            reason=WorkspaceContentExclusionReason.FRAMEWORK_TOOLHOST_LOG,
        ),
    )


def _material_working_set_snapshot(
    session: Any,
) -> _MaterialWorkingSetSnapshot | None:
    if session is None:
        return None

    try:
        from thorn.runtime._working_set import WorkingSet
    except ImportError:
        return None

    working_set = getattr(session, "working_set", None)
    if not isinstance(working_set, WorkingSet):
        return None

    validation = None
    if working_set.last_validation is not None:
        validation = (
            working_set.last_validation.outcome.value,
            working_set.last_validation.summary,
            working_set.last_validation.command,
        )
    blocker = None
    if working_set.blocker is not None:
        blocker = (
            working_set.blocker.summary,
            working_set.blocker.unblock_condition,
        )

    return _MaterialWorkingSetSnapshot(
        phase=working_set.phase.value,
        focused_inbox_item_id=(
            str(working_set.focused_inbox_item_id)
            if working_set.focused_inbox_item_id is not None else None
        ),
        objective=working_set.objective,
        last_action_summary=working_set.last_action_summary,
        validation=validation,
        no_validation_rationale=working_set.no_validation_rationale,
        blocker=blocker,
    )


def _todo_lifecycle_snapshot(
    context: ExecutionContext,
    session: Any,
) -> _TodoLifecycleSnapshot | None:
    runtime = context.runtime
    if runtime is None or session is None:
        return None
    session_key = getattr(session, "key", None)
    agent = getattr(session, "agent", None)
    agent_id = getattr(agent, "id", None)
    if session_key is None or agent_id is None:
        return None

    try:
        from thorn.runtime._todo import SessionTodoList

        todo_file = runtime.paths.session_todo_file(agent_id, session_key)
        item_statuses = tuple(
            sorted(
                (str(item.id), item.status.value)
                for item in SessionTodoList(todo_file).list_items()
            )
        )
    except (OSError, ValueError):
        logger.debug("failed to snapshot session TODO lifecycle", exc_info=True)
        return None

    return _TodoLifecycleSnapshot(item_statuses=item_statuses)


def _classify_tool_round_progress(
    *,
    tool_calls: list[ToolCall],
    result_messages: list[ToolResultMessage],
    captured: Any,
    before: _ProgressSnapshot,
    after: _ProgressSnapshot,
    validation_convergence_tracker: ValidationConvergenceTracker,
    workspace_content_convergence_tracker: WorkspaceContentConvergenceTracker,
    validation_convergence_policy: ValidationConvergencePolicy,
    validation_content_snapshots: dict[str, WorkspaceContentSnapshot],
    source_render_id: str,
) -> _ToolRoundProgress:
    """Classify whether this tool round advanced work rather than research."""
    tool_pattern = _tool_round_pattern(tool_calls, result_messages)

    result_by_call_id = {
        result.call_id: result for result in result_messages
    }
    successful_results = {
        result.call_id for result in result_messages if not result.is_error
    }
    successful_tool_calls = [
        tool_call
        for tool_call in tool_calls
        if tool_call.call_id in successful_results
    ]

    validation_observations: list[
        ValidationConvergenceObservation | WorkspaceContentConvergenceObservation
    ] = []
    validation_telemetry: list[ValidationTelemetry] = []
    for tool_call in tool_calls:
        result = result_by_call_id.get(tool_call.call_id)
        if result is None:
            continue
        tool_name = _normalize_tool_name(tool_call.name)
        if (
            not result.is_error
            and tool_name in _WORKSPACE_MUTATION_TOOL_NAMES
        ):
            validation_telemetry.append(
                validation_convergence_tracker.advance_workspace_action_epoch(
                    call_id=tool_call.call_id,
                    render_id=source_render_id,
                    reason=(
                        ValidationActionEpochReason.NATIVE_MATERIAL_MUTATION
                    ),
                    tool_name=ValidationActionTool(tool_name),
                ),
            )
            continue
        if tool_name != ValidationActionTool.RUN_SHELL:
            continue

        validation_command = parse_validation_command(tool_call.arguments)
        if validation_command is not None:
            validation_outcome = validation_outcome_from_tool_result(
                content=result.content,
                tool_reported_error=result.is_error,
            )
            if validation_convergence_policy.uses_workspace_content:
                snapshot = validation_content_snapshots.get(tool_call.call_id)
                if snapshot is None:
                    snapshot = WorkspaceContentSnapshot.unknown(
                        WorkspaceContentCollectionFailure.WORKSPACE_UNAVAILABLE,
                    )
                observation = workspace_content_convergence_tracker.observe(
                    validation_command,
                    validation_outcome,
                    snapshot,
                    call_id=tool_call.call_id,
                    render_id=source_render_id,
                    policy=WorkspaceContentConvergencePolicy.from_validation_policy(
                        validation_convergence_policy,
                    ),
                )
            else:
                observation = validation_convergence_tracker.observe(
                    validation_command,
                    validation_outcome,
                    call_id=tool_call.call_id,
                    render_id=source_render_id,
                    policy=validation_convergence_policy,
                )
            validation_observations.append(observation)
            validation_telemetry.append(observation.telemetry)
            continue

        if not result.is_error and not _tool_call_is_observation(tool_call):
            # An opaque shell call may hide a workspace edit. Advancing here
            # avoids attributing later validation repeats to stale state.
            validation_telemetry.append(
                validation_convergence_tracker.advance_workspace_action_epoch(
                    call_id=tool_call.call_id,
                    render_id=source_render_id,
                    reason=(
                        ValidationActionEpochReason.OPAQUE_SHELL_POSSIBLE_MUTATION
                    ),
                    tool_name=ValidationActionTool.RUN_SHELL,
                ),
            )

    ordered_validation_telemetry = tuple(validation_telemetry)
    if captured is not _RESULT_SENTINEL:
        return _ToolRoundProgress(
            made_progress=True,
            reason="structured result returned",
            tool_pattern=tool_pattern,
            validation_telemetry=ordered_validation_telemetry,
        )

    validation_made_progress = any(
        observation.progress_effect
        is ValidationProgressEffect.COUNTS_AS_PROGRESS
        for observation in validation_observations
    )
    validation_deferred = any(
        observation.progress_effect
        is ValidationProgressEffect.DEFER_TO_CONSERVATIVE_CLASSIFIER
        for observation in validation_observations
    )
    policy_effect_applied = (
        validation_convergence_policy.applies_progress_effect
    )

    working_set_changed = _working_set_materially_changed(before, after)

    if not successful_results:
        return _ToolRoundProgress(
            made_progress=policy_effect_applied and validation_deferred,
            reason=(
                "validation outcome was uncertain"
                if policy_effect_applied and validation_deferred
                else "all tool calls failed"
            ),
            tool_pattern=tool_pattern,
            validation_telemetry=ordered_validation_telemetry,
        )

    if working_set_changed:
        return _ToolRoundProgress(
            made_progress=True,
            reason="working set changed materially",
            tool_pattern=tool_pattern,
            validation_telemetry=ordered_validation_telemetry,
        )

    if _todo_lifecycle_changed(before, after):
        return _ToolRoundProgress(
            made_progress=True,
            reason="TODO lifecycle changed",
            tool_pattern=tool_pattern,
            validation_telemetry=ordered_validation_telemetry,
        )

    for tool_call in successful_tool_calls:
        tool_name = _normalize_tool_name(tool_call.name)
        if tool_name in _ACTION_PROGRESS_TOOL_NAMES:
            return _ToolRoundProgress(
                made_progress=True,
                reason=f"{tool_name} performed a material action",
                tool_pattern=tool_pattern,
                validation_telemetry=ordered_validation_telemetry,
            )
        if tool_name != "run_shell":
            continue
        if (
            policy_effect_applied
            and parse_validation_command(tool_call.arguments) is not None
        ):
            continue
        if _run_shell_command_makes_progress(tool_call.arguments):
            return _ToolRoundProgress(
                made_progress=True,
                reason="shell command performed validation or workspace setup",
                tool_pattern=tool_pattern,
                validation_telemetry=ordered_validation_telemetry,
            )

    if policy_effect_applied and (
        validation_made_progress or validation_deferred
    ):
        return _ToolRoundProgress(
            made_progress=True,
            reason=(
                "validation produced a new convergence observation"
                if validation_made_progress
                else "validation outcome was uncertain"
            ),
            tool_pattern=tool_pattern,
            validation_telemetry=ordered_validation_telemetry,
        )

    resource_accesses = tuple(
        access
        for access in (
            _resource_access_for_tool_call(tool_call)
            for tool_call in successful_tool_calls
        )
        if access is not None
    )
    if successful_tool_calls and all(
        _tool_call_is_observation(tool_call)
        for tool_call in successful_tool_calls
    ):
        return _ToolRoundProgress(
            made_progress=False,
            reason="read/search-only round",
            tool_pattern=tool_pattern,
            resource_accesses=resource_accesses,
            validation_telemetry=ordered_validation_telemetry,
        )

    return _ToolRoundProgress(
        made_progress=False,
        reason="successful tool round only used non-material tools",
        tool_pattern=tool_pattern,
        resource_accesses=resource_accesses,
        validation_telemetry=ordered_validation_telemetry,
    )


def _working_set_materially_changed(
    before: _ProgressSnapshot,
    after: _ProgressSnapshot,
) -> bool:
    if before.working_set is None or after.working_set is None:
        return False
    return before.working_set != after.working_set


def _todo_lifecycle_changed(
    before: _ProgressSnapshot,
    after: _ProgressSnapshot,
) -> bool:
    if before.todos is None or after.todos is None:
        return False
    return before.todos != after.todos


def _tool_round_pattern(
    tool_calls: list[ToolCall],
    result_messages: list[ToolResultMessage],
) -> str:
    result_by_call_id = {
        result.call_id: result for result in result_messages
    }
    parts: list[str] = []
    for tool_call in tool_calls:
        result = result_by_call_id.get(tool_call.call_id)
        if result is None:
            suffix = "missing"
        elif result.is_error:
            suffix = "error"
        else:
            suffix = "ok"
        parts.append(f"{_normalize_tool_name(tool_call.name)}:{suffix}")
    return "+".join(parts) if parts else "none"


def _resource_access_for_tool_call(
    tool_call: ToolCall,
) -> _ResourceAccessFingerprint | None:
    tool_name = _normalize_tool_name(tool_call.name)
    parsed = _parse_tool_arguments(tool_call.arguments)
    if parsed is None:
        return None

    match tool_name:
        case "read_file":
            return _resource_access(
                tool_name,
                parsed,
                ("path", "offset", "limit"),
            )
        case "search_files":
            return _resource_access(
                tool_name,
                parsed,
                (
                    "path",
                    "pattern",
                    "glob",
                    "use_regex",
                    "ignore_case",
                    "context_lines",
                ),
            )
        case "list_directory":
            return _resource_access(
                tool_name,
                parsed,
                ("path", "recursive", "max_depth"),
            )
        case "find_files":
            return _resource_access(
                tool_name,
                parsed,
                ("path", "pattern", "type"),
            )
        case "forge_read_file":
            return _resource_access(
                tool_name,
                parsed,
                ("project", "file_path", "ref"),
            )
        case "read_inbox_item":
            return _resource_access(tool_name, parsed, ("item_id",))
        case "list_session_todos":
            return _resource_access(tool_name, parsed, ("status_filter",))
        case "run_shell":
            inspection = _shell_inspection_for_tool_arguments(parsed)
            if inspection is None:
                return None
            return _shell_inspection_resource_access(inspection)
        case _:
            return None


def _tool_call_is_observation(tool_call: ToolCall) -> bool:
    tool_name = _normalize_tool_name(tool_call.name)
    if tool_name in _OBSERVATION_TOOL_NAMES:
        return True
    if tool_name != "run_shell":
        return False
    parsed = _parse_tool_arguments(tool_call.arguments)
    if parsed is None:
        return False
    return _shell_inspection_for_tool_arguments(parsed) is not None


def _shell_inspection_for_tool_arguments(
    arguments: dict[str, Any],
) -> ShellInspectionCommand | None:
    command = arguments.get("command")
    if not isinstance(command, str):
        return None
    return parse_shell_inspection_command(command)


def _shell_inspection_resource_access(
    inspection: ShellInspectionCommand,
) -> _ResourceAccessFingerprint:
    resource: list[tuple[str, str]] = [
        ("kind", inspection.kind.value),
        ("command", inspection.command_name),
    ]
    if inspection.path is not None:
        resource.append(("path", inspection.path))
    if inspection.pattern is not None:
        resource.append(("pattern", inspection.pattern))
    if inspection.line_range_label is not None:
        resource.append(("range", inspection.line_range_label))
    return _ResourceAccessFingerprint(
        tool_name="run_shell",
        resource=tuple(resource),
    )


def _resource_access(
    tool_name: str,
    arguments: dict[str, Any],
    keys: tuple[str, ...],
) -> _ResourceAccessFingerprint:
    resource = tuple(
        (key, _canonical_resource_value(arguments.get(key)))
        for key in keys
        if arguments.get(key) is not None
    )
    return _ResourceAccessFingerprint(tool_name=tool_name, resource=resource)


def _parse_tool_arguments(arguments: str) -> dict[str, Any] | None:
    if not arguments:
        return {}
    try:
        parsed = json.loads(arguments)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _canonical_resource_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if value is None:
        return "none"
    return str(value)


def _run_shell_command_makes_progress(arguments: str) -> bool:
    if not arguments:
        return False

    try:
        parsed = json.loads(arguments)
    except json.JSONDecodeError:
        return False

    command = parsed.get("command")
    if not isinstance(command, str):
        return False
    if parse_shell_inspection_command(command) is not None:
        return False

    normalized = " ".join(command.lower().split())
    return any(pattern in normalized for pattern in _RUN_SHELL_PROGRESS_PATTERNS)


async def _raise_if_no_progress(
    *,
    tracker: _NoProgressTracker,
    max_tool_rounds_without_progress: int,
    context: ExecutionContext,
    session: Any,
) -> None:
    if max_tool_rounds_without_progress == 0:
        return
    if tracker.rounds_without_progress < max_tool_rounds_without_progress:
        return

    diagnostic = _no_progress_diagnostic(tracker, session)
    await context.event_sink.on_status(
        f"loop progress: {diagnostic}",
        scope=context.scope,
    )
    await _emit_no_progress_telemetry(
        context, session, diagnostic, gate_name="no_progress",
    )
    raise LoopNoProgressError(
        f"agent loop made no meaningful progress: {diagnostic}",
        tracker.rounds_without_progress,
    )


async def _warn_if_no_progress(
    *,
    tracker: _NoProgressTracker,
    max_tool_rounds_without_progress: int,
    context: ExecutionContext,
    session: Any,
    advisory_nodes: list[AdvisoryNode],
) -> None:
    warning_threshold = _no_progress_warning_threshold(
        max_tool_rounds_without_progress,
    )
    if warning_threshold is None:
        return
    if tracker.warning_emitted:
        return
    if tracker.rounds_without_progress < warning_threshold:
        return
    if tracker.rounds_without_progress >= max_tool_rounds_without_progress:
        return

    diagnostic = _no_progress_diagnostic(tracker, session)
    content = (
        "[No-progress warning: "
        f"{diagnostic}. This is advisory only. Use the next tool call to "
        "make concrete progress, validate, close out, or declare a blocker; "
        "only keep inspecting if it answers a specific unresolved question.]"
    )
    tracker.warning_emitted = True
    advisory_nodes.append(AdvisoryNode(source="no_progress", content=content))
    await context.event_sink.on_status(
        f"loop progress warning: {diagnostic}",
        scope=context.scope,
    )
    await context.event_sink.on_advisory(
        "no_progress",
        content,
        scope=context.scope,
    )
    await _emit_no_progress_telemetry(
        context, session, diagnostic, gate_name="no_progress_warning",
    )


def _no_progress_warning_threshold(
    max_tool_rounds_without_progress: int,
) -> int | None:
    if max_tool_rounds_without_progress <= 1:
        return None
    return max(1, (max_tool_rounds_without_progress + 1) // 2)


def _no_progress_diagnostic(
    tracker: _NoProgressTracker,
    session: Any,
) -> str:
    parts = [
        "no meaningful progress after "
        f"{tracker.rounds_without_progress} successful tool rounds",
    ]
    phase, focus = _working_set_phase_and_focus(session)
    if phase is not None:
        parts.append(f"phase={phase}")
    if focus is not None:
        parts.append(f"focus={focus}")
    if tracker.recent_tool_patterns:
        parts.append(
            "recent_tools=" + " -> ".join(
                pattern.replace(":ok", "")
                for pattern in tracker.recent_tool_patterns
            ),
        )
    if (
        tracker.repeated_resource is not None
        and tracker.repeated_resource_count > 1
    ):
        parts.append(
            "repeated_resource="
            f"{tracker.repeated_resource.render()} "
            f"x{tracker.repeated_resource_count}"
        )
    if tracker.last_reason:
        parts.append(f"reason={tracker.last_reason}")
    parts.append(
        "expected_next=act, validate, make a material working-set/TODO "
        "change, close out, or declare blocked"
    )
    return "; ".join(parts)


def _working_set_phase_and_focus(session: Any) -> tuple[str | None, str | None]:
    working_set = getattr(session, "working_set", None)
    phase = getattr(getattr(working_set, "phase", None), "value", None)
    focus = getattr(working_set, "focused_inbox_item_id", None)
    return (
        str(phase) if phase is not None else None,
        str(focus) if focus is not None else None,
    )


async def _emit_no_progress_telemetry(
    context: ExecutionContext,
    session: Any,
    diagnostic: str,
    *,
    gate_name: str,
) -> None:
    working_set = getattr(session, "working_set", None)
    try:
        from thorn.runtime._working_set import WorkingSet
        from thorn.runtime._working_set_telemetry import (
            WorkingSetGateTelemetry,
            WorkingSetTelemetryKind,
            build_working_set_telemetry,
        )
    except ImportError:
        return

    if not isinstance(working_set, WorkingSet):
        return

    await context.event_sink.on_working_set_telemetry(
        build_working_set_telemetry(
            kind=WorkingSetTelemetryKind.GATE_INTERVENTION,
            working_set=working_set,
            gate=WorkingSetGateTelemetry(
                name=gate_name,
                reason=diagnostic,
            ),
        ),
        scope=context.scope,
    )


def _fingerprint_text_round(text: str) -> _AssistantRoundFingerprint:
    return _AssistantRoundFingerprint(
        kind=_AssistantRoundKind.TEXT,
        text=text,
    )


def _fingerprint_repeated_tool_error_round(
    *,
    text: str,
    tool_calls: list[ToolCall],
    result_messages: list[ToolResultMessage],
) -> _AssistantRoundFingerprint | None:
    if any(not result.is_error for result in result_messages):
        return None

    return _AssistantRoundFingerprint(
        kind=_AssistantRoundKind.TOOL_CALLS,
        text=text,
        tool_calls=tuple(
            _ToolCallFingerprint(
                name=tool_call.name,
                arguments=_canonicalize_tool_arguments(tool_call.arguments),
            )
            for tool_call in tool_calls
        ),
        tool_results=tuple(
            _ToolResultFingerprint(
                content=result.content,
                is_error=result.is_error,
            )
            for result in result_messages
        ),
    )


def _canonicalize_tool_arguments(arguments: str) -> str:
    if not arguments:
        return ""

    try:
        parsed = json.loads(arguments)
    except json.JSONDecodeError:
        return arguments

    return json.dumps(parsed, sort_keys=True, separators=(",", ":"))


async def _raise_if_repeated(
    *,
    tracker: _ConsecutiveRepetitionTracker,
    fingerprint: _AssistantRoundFingerprint,
    context: ExecutionContext,
) -> None:
    repetitions = tracker.observe(fingerprint)
    if repetitions < tracker.limit:
        return

    await context.event_sink.on_status(
        f"loop repetition: repeated {fingerprint.kind.value} round "
        f"{repetitions} times",
        scope=context.scope,
    )
    raise LoopRepetitionError(
        f"agent loop repeated the same {fingerprint.kind.value} round "
        f"{repetitions} times",
        repetitions,
    )


# ---------------------------------------------------------------------------
# Overhead estimation
# ---------------------------------------------------------------------------

def _estimate_overhead(
    prompts: list[str],
    schemas: list[dict[str, Any]],
) -> int:
    """Rough token estimate for everything outside the history.

    Covers system prompts and tool schemas.  Used only for the compaction
    target computation; absolute trigger checks use real provider usage.
    """
    total = 0
    for p in prompts:
        total += estimate_tokens(p)
    for schema in schemas:
        total += estimate_tokens(json.dumps(schema))
    return total


# ---------------------------------------------------------------------------
# Completion request (with retry)
# ---------------------------------------------------------------------------


async def _sleep_with_backoff(
    policy: RetryPolicy,
    attempt: int,
    *,
    retry_after: float | None,
    reason: str,
    delay: float | None = None,
) -> None:
    """Sleep for ``policy``'s backoff delay, honouring ``retry_after``."""
    if delay is None:
        delay = policy.backoff_delay(attempt, retry_after=retry_after)
    logger.info(
        "%s, retrying in %.1fs (attempt %d)",
        reason, delay, attempt + 1,
    )
    await asyncio.sleep(delay)


def _final_prompt_trace_manifest(
    *,
    context_system_prompts: list[str],
    per_call_system_prompts: list[str],
    per_call_prompt_trace_manifest: PromptTraceManifest | None,
    structured_instruction: str | None,
) -> PromptTraceManifest:
    sources: list[PromptTraceContextSource] = [
        PromptTraceContextSource.from_text(
            surface="runtime_system_prompt",
            label=f"runtime system prompt {index}",
            text=prompt,
        )
        for index, prompt in enumerate(context_system_prompts)
    ]

    per_call_sources = (
        per_call_prompt_trace_manifest.system_prompt_sources
        if per_call_prompt_trace_manifest is not None
        else []
    )
    working_set_telemetry = (
        per_call_prompt_trace_manifest.working_set_telemetry
        if per_call_prompt_trace_manifest is not None
        else None
    )
    if len(per_call_sources) == len(per_call_system_prompts):
        sources.extend(per_call_sources)
    else:
        sources.extend(
            PromptTraceContextSource.from_text(
                surface="per_call_system_prompt",
                label=f"per-call system prompt {index}",
                text=prompt,
            )
            for index, prompt in enumerate(per_call_system_prompts)
        )

    if structured_instruction is not None:
        sources.append(PromptTraceContextSource.from_text(
            surface="structured_result_instruction",
            label="structured result instruction",
            text=structured_instruction,
        ))

    return PromptTraceManifest(
        system_prompt_sources=sources,
        working_set_telemetry=working_set_telemetry,
    ).with_system_prompt_indices()


async def _record_prompt_trace_if_enabled(
    *,
    context: ExecutionContext,
    request_id: str,
    provider_name: str,
    model_name: str | None,
    system_prompts: list[str],
    tool_schemas: list[dict],
    messages: list[Message],
    metrics: ProviderContextMetrics,
    manifest: PromptTraceManifest | None,
) -> None:
    recorder = context.prompt_trace_recorder
    if recorder is None:
        return
    try:
        provider_payload = context.provider.trace_request_payload(
            system_prompts,
            tool_schemas,
            messages,
        )
        if provider_payload is None:
            provider_payload = _generic_prompt_trace_payload(
                system_prompts=system_prompts,
                tool_schemas=tool_schemas,
                messages=messages,
            )
        artifact = recorder.record(
            request_id=request_id,
            provider_name=provider_name,
            model_name=model_name,
            provider_payload=provider_payload,
            context=metrics,
            manifest=manifest,
        )
        if artifact is not None:
            await context.event_sink.on_prompt_trace(
                artifact,
                scope=context.scope,
            )
    except Exception:
        logger.warning(
            "failed to record prompt trace for request %s",
            request_id,
            exc_info=True,
        )


def _generic_prompt_trace_payload(
    *,
    system_prompts: list[str],
    tool_schemas: list[dict],
    messages: list[Message],
) -> dict[str, Any]:
    return {
        "system_prompts": list(system_prompts),
        "tools": list(tool_schemas),
        "messages": [
            _message_metric_payload(message) for message in messages
        ],
    }


async def _request_completion(
    *,
    context: ExecutionContext,
    request_id: str,
    system_prompts: list[str],
    tool_schemas: list[dict],
    messages: list[Message],
    consecutive_failures: int,
    max_failures: int,
    history_node_count: int | None = None,
    estimated_history_tokens: int | None = None,
    estimated_overhead_tokens: int | None = None,
    history_ledger: ProviderContextLedger | None = None,
    prompt_trace_manifest: PromptTraceManifest | None = None,
    retry_policy: RetryPolicy | None = None,
) -> tuple[str, list[ToolCall], str, dict[str, int] | None]:
    """Return ``(text, tool_calls, finish_reason, usage)``.

    Retry policy:

    - :class:`RateLimitError` (HTTP 429 or equivalent) -- retried
      with full-jitter exponential backoff up to
      ``retry_policy.max_rate_limit_retries`` attempts, honouring
      any ``Retry-After`` hint from the server.  Exhaustion raises
      :class:`ProviderUnavailableError`.
    - :class:`TransientProviderError` (transport-level failures,
      502/503/504) -- same backoff shape, bounded by
      ``retry_policy.max_transient_retries``.  Exhaustion raises
      :class:`ProviderUnavailableError`.
    - Non-transient :class:`ProviderError` -- counted against
      ``max_failures``.  Exhaustion raises
      :class:`AgentFailureError`.  These are almost always
      indicative of a configuration or request-shape problem (4xx
      responses other than 429) rather than a network issue, so
      they stay on the agent-loop's own failure budget.

    The ``consecutive_failures`` counter is seeded from the caller
    so that successive calls to this function (one per tool round)
    can share a budget; the caller resets it to zero on a
    successful return.  Rate-limit and transient-retry budgets are
    per-call and do not accumulate across rounds -- a few network
    blips spaced across many rounds should not lock the session
    out of making progress.
    """
    policy = retry_policy or _DEFAULT_RETRY_POLICY
    rate_limit_retries = 0
    transient_retries = 0
    attempt_number = 0
    provider_name = type(context.provider).__name__
    model_name = getattr(getattr(context.provider, "config", None), "model_name", None)
    base_metrics = _provider_context_metrics(
        context=context,
        system_prompts=system_prompts,
        tool_schemas=tool_schemas,
        messages=messages,
        history_node_count=history_node_count,
        estimated_history_tokens=estimated_history_tokens,
        estimated_overhead_tokens=estimated_overhead_tokens,
        history_ledger=history_ledger,
    )
    await _record_prompt_trace_if_enabled(
        context=context,
        request_id=request_id,
        provider_name=provider_name,
        model_name=model_name,
        system_prompts=system_prompts,
        tool_schemas=tool_schemas,
        messages=messages,
        metrics=base_metrics,
        manifest=prompt_trace_manifest,
    )

    while True:
        attempt_number += 1
        attempt_started = time.monotonic()
        time_to_first_chunk_s: float | None = None
        try:
            text_parts: list[str] = []
            tool_call_chunks: list[ToolCallChunk] = []
            finish_reason = "stop"
            usage: dict[str, int] | None = None

            response = context.provider.complete(
                system_prompts, tool_schemas, messages,
            )
            async for chunk in response:
                if time_to_first_chunk_s is None:
                    time_to_first_chunk_s = time.monotonic() - attempt_started
                await context.event_sink.on_response_chunk(
                    chunk, scope=context.scope,
                )
                match chunk:
                    case TextChunk():
                        text_parts.append(chunk.text)
                    case ToolCallChunk():
                        tool_call_chunks.append(chunk)
                    case UsageChunk():
                        usage = {
                            "prompt_tokens": chunk.prompt_tokens,
                            "completion_tokens": chunk.completion_tokens,
                            "total_tokens": chunk.total_tokens,
                        }
                    case FinishChunk():
                        finish_reason = chunk.reason

            duration_s = time.monotonic() - attempt_started
            await context.event_sink.on_completion_end(
                duration_s=duration_s, usage=usage, scope=context.scope,
            )
            await context.event_sink.on_provider_attempt(
                ProviderAttemptTelemetry(
                    request_id=request_id,
                    attempt_number=attempt_number,
                    provider_name=provider_name,
                    model_name=model_name,
                    outcome=ProviderAttemptOutcome.SUCCESS,
                    next_action=ProviderAttemptNextAction.NONE,
                    duration_s=duration_s,
                    time_to_first_chunk_s=time_to_first_chunk_s,
                    context=base_metrics.with_usage(usage),
                ),
                scope=context.scope,
            )

            text = "".join(text_parts)
            tool_calls = [tc.to_tool_call() for tc in tool_call_chunks]
            return text, tool_calls, finish_reason, usage

        except RateLimitError as exc:
            exhausted = rate_limit_retries >= policy.max_rate_limit_retries
            delay = None if exhausted else policy.backoff_delay(
                rate_limit_retries,
                retry_after=exc.retry_after,
            )
            await _emit_provider_failure_attempt(
                context=context,
                request_id=request_id,
                attempt_number=attempt_number,
                provider_name=provider_name,
                model_name=model_name,
                outcome=ProviderAttemptOutcome.RATE_LIMITED,
                next_action=(
                    ProviderAttemptNextAction.RAISE_PROVIDER_UNAVAILABLE
                    if exhausted else ProviderAttemptNextAction.RETRY
                ),
                metrics=base_metrics,
                started_at=attempt_started,
                time_to_first_chunk_s=time_to_first_chunk_s,
                retry_delay_s=delay,
                exc=exc,
            )
            if exhausted:
                raise ProviderUnavailableError(
                    f"rate-limit retries exhausted after "
                    f"{rate_limit_retries + 1} attempt(s): {exc}",
                    attempts=rate_limit_retries + 1,
                ) from exc
            await _sleep_with_backoff(
                policy,
                rate_limit_retries,
                retry_after=exc.retry_after,
                reason="rate limited",
                delay=delay,
            )
            rate_limit_retries += 1

        except TransientProviderError as exc:
            exhausted = transient_retries >= policy.max_transient_retries
            delay = None if exhausted else policy.backoff_delay(
                transient_retries,
                retry_after=exc.retry_after,
            )
            await _emit_provider_failure_attempt(
                context=context,
                request_id=request_id,
                attempt_number=attempt_number,
                provider_name=provider_name,
                model_name=model_name,
                outcome=ProviderAttemptOutcome.TRANSIENT_ERROR,
                next_action=(
                    ProviderAttemptNextAction.RAISE_PROVIDER_UNAVAILABLE
                    if exhausted else ProviderAttemptNextAction.RETRY
                ),
                metrics=base_metrics,
                started_at=attempt_started,
                time_to_first_chunk_s=time_to_first_chunk_s,
                retry_delay_s=delay,
                exc=exc,
            )
            if exhausted:
                raise ProviderUnavailableError(
                    f"transient provider retries exhausted after "
                    f"{transient_retries + 1} attempt(s): {exc}",
                    attempts=transient_retries + 1,
                ) from exc
            await _sleep_with_backoff(
                policy,
                transient_retries,
                retry_after=exc.retry_after,
                reason="transient provider error",
                delay=delay,
            )
            transient_retries += 1

        except ProviderError as exc:
            # Non-transient provider error: counts against the
            # agent-level failure budget rather than the per-call
            # transient budget, so repeated hard failures across
            # multiple tool rounds eventually break out of the
            # loop with :class:`AgentFailureError` (rather than
            # being masked as "transient" forever).
            consecutive_failures += 1
            exhausted = consecutive_failures >= max_failures
            delay = None if exhausted else policy.backoff_delay(
                consecutive_failures - 1,
                retry_after=None,
            )
            await _emit_provider_failure_attempt(
                context=context,
                request_id=request_id,
                attempt_number=attempt_number,
                provider_name=provider_name,
                model_name=model_name,
                outcome=ProviderAttemptOutcome.PROVIDER_ERROR,
                next_action=(
                    ProviderAttemptNextAction.RAISE_AGENT_FAILURE
                    if exhausted else ProviderAttemptNextAction.RETRY
                ),
                metrics=base_metrics,
                started_at=attempt_started,
                time_to_first_chunk_s=time_to_first_chunk_s,
                retry_delay_s=delay,
                exc=exc,
            )
            if exhausted:
                raise AgentFailureError(
                    f"too many consecutive provider failures "
                    f"({consecutive_failures})",
                    consecutive_failures,
                )
            await _sleep_with_backoff(
                policy,
                consecutive_failures - 1,
                retry_after=None,
                reason="provider error",
                delay=delay,
            )


def _provider_context_metrics(
    *,
    context: ExecutionContext,
    system_prompts: list[str],
    tool_schemas: list[dict],
    messages: list[Message],
    history_node_count: int | None,
    estimated_history_tokens: int | None,
    estimated_overhead_tokens: int | None,
    history_ledger: ProviderContextLedger | None = None,
) -> ProviderContextMetrics:
    overhead_tokens = (
        estimated_overhead_tokens
        if estimated_overhead_tokens is not None
        else _estimate_overhead(system_prompts, tool_schemas)
    )
    history_tokens = (
        estimated_history_tokens
        if estimated_history_tokens is not None
        else _estimate_messages(messages)
    )
    context_window = context.context_window
    return ProviderContextMetrics(
        system_prompt_count=len(system_prompts),
        tool_schema_count=len(tool_schemas),
        message_count=len(messages),
        history_node_count=history_node_count,
        context_window=context_window,
        estimated_prompt_tokens=history_tokens + overhead_tokens,
        estimated_history_tokens=history_tokens,
        estimated_overhead_tokens=overhead_tokens,
        high_watermark_tokens=(
            int(
                context_window
                * _hard_context_window_fraction(context)
            )
            if context_window is not None else None
        ),
        history_ledger=history_ledger,
    )


def _hard_context_window_fraction(context: ExecutionContext) -> float:
    policy = context.context_budget_policy
    if policy is None:
        return DEFAULT_HIGH_WATERMARK
    return policy.hard_context_window_fraction.value


def _estimate_messages(messages: list[Message]) -> int:
    total = 0
    for message in messages:
        total += estimate_tokens(json.dumps(_message_metric_payload(message)))
    return total


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


async def _emit_provider_failure_attempt(
    *,
    context: ExecutionContext,
    request_id: str,
    attempt_number: int,
    provider_name: str,
    model_name: str | None,
    outcome: ProviderAttemptOutcome,
    next_action: ProviderAttemptNextAction,
    metrics: ProviderContextMetrics,
    started_at: float,
    time_to_first_chunk_s: float | None,
    retry_delay_s: float | None,
    exc: ProviderError,
) -> None:
    await context.event_sink.on_provider_attempt(
        ProviderAttemptTelemetry(
            request_id=request_id,
            attempt_number=attempt_number,
            provider_name=provider_name,
            model_name=model_name,
            outcome=outcome,
            next_action=next_action,
            duration_s=time.monotonic() - started_at,
            time_to_first_chunk_s=time_to_first_chunk_s,
            context=metrics,
            retry_delay_s=retry_delay_s,
            retry_after_s=getattr(exc, "retry_after", None),
            failure_kind=getattr(exc, "failure_kind", None),
            status_code=getattr(exc, "status_code", None),
            error_type=type(exc).__name__,
            error_message=str(exc),
        ),
        scope=context.scope,
    )


# ---------------------------------------------------------------------------
# Tool dispatch
# ---------------------------------------------------------------------------

def _normalize_tool_name(name: str) -> str:
    """Forgive minor naming mismatches (hyphens vs underscores, casing)."""
    return name.lower().replace("-", "_").replace(" ", "_")


def _scope_session_key(context: ExecutionContext) -> str | None:
    """Return the nearest ``session_key`` tag from the context scope chain."""
    current = context.scope
    while current is not None:
        value = current.metadata.get("session_key")
        if value:
            return str(value)
        current = current.outer
    return None


def _sandbox_per_call_context(context: ExecutionContext) -> dict[str, Any]:
    """Build daemon-only context metadata for sandbox-venue tool calls."""
    per_call: dict[str, Any] = {}
    session_key = _scope_session_key(context)
    if session_key is not None:
        per_call["session_key"] = session_key

    runtime = context.runtime
    agent = context.agent
    workspace_root = context.workspace_root
    agent_id = getattr(agent, "id", None)
    if runtime is None or agent_id is None or workspace_root is None:
        return per_call

    try:
        agent_workspace = runtime.paths.agent_workspace_mount(agent_id)
    except Exception:
        return per_call

    try:
        rel = Path(workspace_root).resolve().relative_to(
            Path(agent_workspace).resolve(),
        )
    except ValueError:
        return per_call

    if str(rel) != ".":
        per_call["workspace_subdir"] = rel.as_posix()
    return per_call


@dataclass(frozen=True)
class ReadFilePriorContextObservation:
    """Facts from comparing a read result against provider-visible context."""

    call_id: str
    render_id: str
    file_path: str
    requested_offset: int
    requested_limit: int | None
    returned_start_line: int
    returned_end_line: int
    comparison: FileContentVisibilityComparison
    hint_emitted: bool


class ReadFileUnclassifiedReason(StrEnum):
    """Why a successful read could not enter the bounded comparison index."""

    INVALID_PATH_ARGUMENT = "invalid_path_argument"
    TRACKER_DECLINED = "tracker_declined"


def _read_file_prior_context_hint(
    *,
    context: ExecutionContext,
    source_render_id: str | None,
    tool_name: str,
    call_id: str,
    arguments: dict[str, Any],
    result_content: str,
    is_error: bool,
) -> tuple[str | None, ReadFilePriorContextObservation | None]:
    """Return a post-execution hint and telemetry facts for read_file output."""
    if is_error:
        return None, None
    if source_render_id is None:
        return None, None
    if _normalize_tool_name(tool_name) != "read_file":
        return None, None

    raw_path = arguments.get("path")
    if not isinstance(raw_path, str):
        return None, None

    returned_lines = parse_line_numbered_file_content(result_content)
    if not returned_lines:
        return None, None

    comparison = context.prompt_visibility.compare_file_content(
        render_id=source_render_id,
        raw_path=raw_path,
        workspace_root=context.workspace_root,
        lines=returned_lines,
    )
    if comparison is None:
        return None, None

    hint_emitted = _read_file_visibility_is_redundant(comparison)
    observation = ReadFilePriorContextObservation(
        call_id=call_id,
        render_id=source_render_id,
        file_path=raw_path,
        requested_offset=arguments.get("offset", 1),
        requested_limit=arguments.get("limit"),
        returned_start_line=returned_lines[0].line_number,
        returned_end_line=returned_lines[-1].line_number,
        comparison=comparison,
        hint_emitted=hint_emitted,
    )
    if not hint_emitted:
        return None, observation

    return (
        f"[Prior context hint: read_file for {raw_path!r} appears redundant. "
        f"{comparison.exact_visible_line_count}/"
        f"{comparison.compared_line_count} returned lines exactly matched "
        "file content that was already visible in the prompt that led to "
        "that tool call. Use the visible context instead of repeating the "
        "same read unless you are checking for intervening changes.]"
    ), observation


def _read_file_session_reuse_hint(
    *,
    context: ExecutionContext,
    session: Any,
    source_render_id: str | None,
    tool_name: str,
    call_id: str,
    arguments: dict[str, Any],
    result_content: str,
    is_error: bool,
) -> tuple[
    str | None,
    ReadFileReuseObservation | None,
    ReadFileUnclassifiedReason | None,
]:
    """Observe a read in every arm and optionally return the R1 advisory."""
    if is_error:
        return None, None, None
    if _normalize_tool_name(tool_name) != "read_file":
        return None, None, None

    raw_path = arguments.get("path")
    if not isinstance(raw_path, str):
        return None, None, ReadFileUnclassifiedReason.INVALID_PATH_ARGUMENT
    returned_lines = parse_line_numbered_file_content(result_content)

    history = _read_file_history_for_session(context=context, session=session)
    observation = history.observe(
        call_id=call_id,
        render_id=source_render_id,
        raw_path=raw_path,
        workspace_root=context.workspace_root,
        returned_lines=returned_lines,
        policy=context.read_file_observation_policy,
    )
    if observation is None:
        return None, None, ReadFileUnclassifiedReason.TRACKER_DECLINED
    if not observation.hint_recommended:
        return None, observation, None

    advisory_policy = context.read_file_reuse_policy
    if advisory_policy is None:
        return None, observation, None
    if observation.covered_fraction < advisory_policy.minimum_covered_fraction:
        return None, observation, None

    if observation.kind is ReadFileReuseKind.EXACT_REPEAT:
        relationship = (
            f"all {observation.returned_line_count} returned lines exactly "
            f"repeat prior tool call {observation.prior_call_id}"
        )
    elif observation.kind is ReadFileReuseKind.FULL_COVERAGE:
        relationship = (
            f"all {observation.returned_line_count} returned lines were "
            "already obtained across prior reads"
        )
    else:
        relationship = (
            f"{observation.matching_prior_line_count}/"
            f"{observation.returned_line_count} returned lines were already "
            "obtained in prior reads"
        )
    return (
        (
            f"[Prior context hint: read_file for {raw_path!r} largely repeats "
            f"unchanged content: {relationship} in content epoch "
            f"{observation.content_epoch}. Use this result and avoid requesting "
            "the covered range again unless the file changes.]"
        ),
        observation,
        None,
    )


def _read_file_history_for_session(
    *,
    context: ExecutionContext,
    session: Any,
) -> ReadFileResultHistory:
    if session is None:
        return context.read_file_history
    history = getattr(session, "_read_file_history", None)
    if isinstance(history, ReadFileResultHistory):
        return history
    return context.read_file_history


def _invalidate_read_file_history_after_mutation(
    *,
    context: ExecutionContext,
    session: Any,
    tool_name: str,
    arguments: dict[str, Any],
    is_error: bool,
) -> None:
    """Invalidate tracked paths after successful first-party file writes."""
    if is_error:
        return

    normalized_tool_name = _normalize_tool_name(tool_name)
    raw_paths: tuple[str, ...] = ()
    if normalized_tool_name == "edit_file":
        edits = arguments.get("edits")
        if not isinstance(edits, list) or not edits:
            return
        raw_path = arguments.get("path")
        if isinstance(raw_path, str):
            raw_paths = (raw_path,)
    elif normalized_tool_name in {"create_file", "delete_file"}:
        raw_path = arguments.get("path")
        if isinstance(raw_path, str):
            raw_paths = (raw_path,)
    elif normalized_tool_name == "move_file":
        source = arguments.get("source")
        destination = arguments.get("destination")
        raw_paths = tuple(
            path for path in (source, destination)
            if isinstance(path, str)
        )

    if not raw_paths:
        return
    _read_file_history_for_session(
        context=context,
        session=session,
    ).invalidate_paths(raw_paths, workspace_root=context.workspace_root)


def _search_files_prior_context_hint(
    *,
    context: ExecutionContext,
    source_render_id: str | None,
    tool_name: str,
    call_id: str,
    arguments: dict[str, Any],
    result_content: str,
    is_error: bool,
) -> tuple[str | None, SearchFilesDuplicateObservation | None]:
    """Return a post-execution hint for exact duplicate search_files output."""
    if is_error:
        return None, None
    if _normalize_tool_name(tool_name) != "search_files":
        return None, None

    observation = context.search_files_history.observe(
        call_id=call_id,
        render_id=source_render_id,
        arguments=arguments,
        result_content=result_content,
        workspace_root=context.workspace_root,
    )
    if observation is None or not observation.hint_emitted:
        return None, observation

    pattern = arguments.get("pattern", "?")
    path_display = arguments.get("path", ".")
    return (
        "[Prior context hint: search_files appears redundant. "
        f"The search for {pattern!r} in {path_display!r} returned exactly "
        f"the same rendered result as prior tool call {observation.prior_call_id}. "
        "Use the visible context instead of repeating the same search unless "
        "you are checking for intervening changes.]"
    ), observation


def _read_file_visibility_is_redundant(
    comparison: FileContentVisibilityComparison,
) -> bool:
    return (
        comparison.compared_line_count > 0
        and comparison.exact_visible_fraction
        >= READ_FILE_PRIOR_CONTEXT_HINT_THRESHOLD
    )


def _prior_context_hint_status(*, tool_name: str, hint: str) -> str:
    return f"prior-context hint for {tool_name}: {hint}"


def _optional_int_tool_argument(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


async def _emit_read_file_prior_context_telemetry(
    *,
    context: ExecutionContext,
    tool_name: str,
    observation: ReadFilePriorContextObservation,
) -> None:
    comparison = observation.comparison
    await context.event_sink.on_prior_context_hint_telemetry(
        tool_name=tool_name,
        hint_kind="read_file_visible_content_match",
        hint_emitted=observation.hint_emitted,
        details={
            "provider_visible": observation.hint_emitted,
            "call_id": observation.call_id,
            "prior_call_id": None,
            "render_id": observation.render_id,
            "file_path": observation.file_path,
            "requested_offset": observation.requested_offset,
            "requested_limit": observation.requested_limit,
            "returned_start_line": observation.returned_start_line,
            "returned_end_line": observation.returned_end_line,
            "returned_line_count": comparison.compared_line_count,
            "matching_visible_line_count": comparison.exact_visible_line_count,
            "visible_line_mismatch_count": comparison.visible_line_mismatch_count,
            "not_visible_line_count": comparison.not_visible_line_count,
        },
        scope=context.scope,
    )


async def _emit_search_files_prior_context_telemetry(
    *,
    context: ExecutionContext,
    tool_name: str,
    observation: SearchFilesDuplicateObservation,
) -> None:
    await context.event_sink.on_prior_context_hint_telemetry(
        tool_name=tool_name,
        hint_kind="search_files_exact_duplicate_result",
        hint_emitted=observation.hint_emitted,
        details={
            "search_key_hash": observation.call_key.fingerprint,
            "result_hash": observation.result_hash,
            "call_id": observation.call_id,
            "render_id": observation.render_id,
            "prior_call_id": observation.prior_call_id,
            "prior_render_id": observation.prior_render_id,
        },
        scope=context.scope,
    )


async def _emit_read_file_session_reuse_telemetry(
    *,
    context: ExecutionContext,
    tool_name: str,
    raw_path: str,
    requested_offset: int | None,
    requested_limit: int | None,
    observation: ReadFileReuseObservation,
    hint_emitted: bool,
) -> None:
    await context.event_sink.on_prior_context_hint_telemetry(
        tool_name=tool_name,
        hint_kind="read_file_session_content_match",
        hint_emitted=hint_emitted,
        details={
            "telemetry_schema_version": (
                READ_FILE_REUSE_TELEMETRY_SCHEMA_VERSION
            ),
            "classification_status": "classified",
            "call_id": observation.call_id,
            "render_id": observation.render_id,
            "prior_call_id": observation.prior_call_id,
            "prior_render_id": observation.prior_render_id,
            "prior_call_ids": list(observation.prior_call_ids),
            "file_path": raw_path,
            "file_key_hash": hashlib.sha256(
                observation.file_key.resolved_path.as_posix().encode("utf-8"),
            ).hexdigest(),
            "requested_offset": requested_offset,
            "requested_limit": requested_limit,
            "returned_start_line": observation.returned_start_line,
            "returned_end_line": observation.returned_end_line,
            "returned_line_count": observation.returned_line_count,
            "content_epoch": observation.content_epoch,
            "content_epoch_advanced": observation.content_epoch_advanced,
            "content_version_hash": observation.content_version_hash,
            "reuse_kind": observation.kind.value,
            "reuse_threshold_met": observation.hint_recommended,
            "hint_eligible": observation.hint_recommended,
            "hint_emitted": hint_emitted,
            "matching_prior_line_count": (
                observation.matching_prior_line_count
            ),
            "new_line_count": observation.new_line_count,
            "covered_fraction": observation.covered_fraction,
            "hint_threshold": (
                context.read_file_observation_policy.minimum_covered_fraction
            ),
            "overlapping_prior_call_count": (
                observation.overlapping_prior_call_count
            ),
            "exact_repeat_count": observation.exact_repeat_count,
        },
        scope=context.scope,
    )


async def _emit_unclassified_read_file_session_telemetry(
    *,
    context: ExecutionContext,
    tool_name: str,
    call_id: str,
    render_id: str | None,
    raw_path: str | None,
    requested_offset: int | None,
    requested_limit: int | None,
    reason: ReadFileUnclassifiedReason,
) -> None:
    """Record a successful native read that cannot be compared safely."""
    await context.event_sink.on_prior_context_hint_telemetry(
        tool_name=tool_name,
        hint_kind="read_file_session_unclassified",
        hint_emitted=False,
        details={
            "telemetry_schema_version": (
                READ_FILE_REUSE_TELEMETRY_SCHEMA_VERSION
            ),
            "classification_status": "unclassified",
            "unclassified_reason": reason.value,
            "call_id": call_id,
            "render_id": render_id,
            "prior_call_id": None,
            "prior_call_ids": [],
            "file_path": raw_path,
            "file_key_hash": None,
            "requested_offset": requested_offset,
            "requested_limit": requested_limit,
            "returned_start_line": None,
            "returned_end_line": None,
            "returned_line_count": None,
            "content_epoch": None,
            "content_epoch_advanced": None,
            "content_version_hash": None,
            "reuse_kind": "unclassified",
            "reuse_threshold_met": False,
            "hint_eligible": False,
            "hint_emitted": False,
            "matching_prior_line_count": None,
            "new_line_count": None,
            "covered_fraction": None,
            "hint_threshold": (
                context.read_file_observation_policy.minimum_covered_fraction
            ),
            "overlapping_prior_call_count": None,
            "exact_repeat_count": None,
        },
        scope=context.scope,
    )


async def _execute_tool_calls(
    *,
    tool_calls: list[ToolCall],
    registry: ToolRegistry,
    router: ExecutorRouter,
    context: ExecutionContext,
    result_type: type | None,
    session: Any = None,
    source_render_id: str | None = None,
    validation_convergence_policy: ValidationConvergencePolicy = (
        ValidationConvergencePolicy.BASELINE
    ),
) -> tuple[
    list[ToolResultMessage],
    dict[str, type[ToolCallNode]],
    list[AdvisoryNode],
    Any,
    dict[str, WorkspaceContentSnapshot],
]:
    """Execute tools and retain pre-validation content snapshots.

    Dispatch walks through the :class:`ToolRegistry` (for name
    resolution, venue, and history-node class) and the
    :class:`ExecutorRouter` (for the actual invocation).  The two are
    split so that Phase A's sandbox work can swap the executor for the
    ``SANDBOX`` venue without any changes to the loop itself.

    *call_node_classes* maps ``call_id`` to the ``ToolCallNode``
    subclass registered on the resolved tool (if any).  Only entries
    with a non-``None`` class are included.

    *advisory_nodes* contains any advisories produced by registered
    status providers for this tool-call round.

    *captured_value* is ``_RESULT_SENTINEL`` unless a ``return_result``
    call was processed (in structured mode).
    """
    results: list[ToolResultMessage] = []
    call_node_classes: dict[str, type[ToolCallNode]] = {}
    advisory_nodes: list[AdvisoryNode] = []
    captured: Any = _RESULT_SENTINEL
    validation_content_snapshots: dict[str, WorkspaceContentSnapshot] = {}

    for tc in tool_calls:
        # -- parse arguments -----------------------------------------------
        try:
            kwargs = json.loads(tc.arguments) if tc.arguments else {}
        except json.JSONDecodeError as exc:
            results.append(ToolResultMessage(
                call_id=tc.call_id,
                content=f"Invalid JSON arguments: {exc}",
                is_error=True,
            ))
            continue

        # -- structured-mode synthetic tools -------------------------------
        if tc.name == "return_result" and result_type is not None:
            raw_value = kwargs.get("value")
            try:
                captured = validate_result(result_type, raw_value)
            except Exception as exc:
                results.append(ToolResultMessage(
                    call_id=tc.call_id,
                    content=f"Validation error: {exc}. Please try again.",
                    is_error=True,
                ))
                continue
            results.append(ToolResultMessage(
                call_id=tc.call_id, content="ok",
            ))
            continue

        if tc.name == "raise_error":
            msg = kwargs.get("message", "unknown error")
            raise SkillError(msg)

        # -- resolve registry entry ----------------------------------------
        entry = registry.get(tc.name)
        if entry is None:
            normalized = _normalize_tool_name(tc.name)
            for candidate in registry.entries():
                if _normalize_tool_name(candidate.name) == normalized:
                    entry = candidate
                    break

        if entry is None:
            results.append(ToolResultMessage(
                call_id=tc.call_id,
                content=f"Unknown tool: {tc.name!r}",
                is_error=True,
            ))
            await context.event_sink.on_tool_end(
                tc.name, error="unknown tool", scope=context.scope,
            )
            continue

        if entry.call_node_class is not None:
            call_node_classes[tc.call_id] = entry.call_node_class

        # MCP-sourced tools may have been exposed to the model under a
        # collision-disambiguating prefix (``<server>__<tool>``); the
        # daemon's ``ClientSession.call_tool`` needs the unprefixed
        # name the server itself knows.  ``entry.mcp_tool_name`` is
        # always set when ``entry.mcp_server_config`` is, so the
        # daemon receives the right name even on collision-prefixed
        # registry names.
        if entry.mcp_server_config is not None:
            wire_tool_name = entry.mcp_tool_name or entry.name
        else:
            wire_tool_name = entry.name
        invocation = ToolInvocation(
            call_id=tc.call_id,
            tool_name=wire_tool_name,
            arguments=kwargs,
            per_call_context=(
                _sandbox_per_call_context(context)
                if entry.venue is ToolVenue.SANDBOX
                else {}
            ),
            mcp_server_config=entry.mcp_server_config,
        )

        # -- execute via the venue's executor ------------------------------
        await context.event_sink.on_tool_start(
            tc.name, kwargs, scope=context.scope,
        )
        executor = router.for_venue(entry.venue)
        if (
            validation_convergence_policy.uses_workspace_content
            and _normalize_tool_name(entry.name) == "run_shell"
            and parse_validation_command(tc.arguments) is not None
        ):
            workspace_root = context.workspace_root
            if workspace_root is None:
                validation_content_snapshots[tc.call_id] = (
                    WorkspaceContentSnapshot.unknown(
                        WorkspaceContentCollectionFailure.WORKSPACE_UNAVAILABLE,
                    )
                )
            else:
                validation_content_snapshots[tc.call_id] = await asyncio.to_thread(
                    collect_workspace_content_snapshot,
                    workspace_root,
                    excluded_paths=_workspace_content_excluded_paths(
                        context,
                        session,
                    ),
                )
        t0 = time.monotonic()
        try:
            outcome = await executor.invoke(invocation)
        except SkillError:
            raise
        except Exception as exc:
            duration_s = time.monotonic() - t0
            logger.exception("error executing tool %s", tc.name)
            results.append(ToolResultMessage(
                call_id=tc.call_id,
                content=f"Error: {exc}",
                is_error=True,
            ))
            await context.event_sink.on_tool_end(
                tc.name, duration_s=duration_s, error=str(exc),
                scope=context.scope,
            )
            continue

        duration_s = time.monotonic() - t0
        _invalidate_read_file_history_after_mutation(
            context=context,
            session=session,
            tool_name=entry.name,
            arguments=kwargs,
            is_error=outcome.is_error,
        )
        hint, read_observation = _read_file_prior_context_hint(
            context=context,
            source_render_id=source_render_id,
            tool_name=entry.name,
            call_id=tc.call_id,
            arguments=kwargs,
            result_content=outcome.content,
            is_error=outcome.is_error,
        )
        if read_observation is not None:
            await _emit_read_file_prior_context_telemetry(
                context=context,
                tool_name=entry.name,
                observation=read_observation,
            )
        (
            session_read_hint,
            session_read_observation,
            session_read_unclassified_reason,
        ) = (
            _read_file_session_reuse_hint(
                context=context,
                session=session,
                source_render_id=source_render_id,
                tool_name=entry.name,
                call_id=tc.call_id,
                arguments=kwargs,
                result_content=outcome.content,
                is_error=outcome.is_error,
            )
        )
        if session_read_observation is not None:
            raw_path = kwargs.get("path")
            if isinstance(raw_path, str):
                session_hint_emitted = (
                    session_read_hint is not None and hint is None
                )
                await _emit_read_file_session_reuse_telemetry(
                    context=context,
                    tool_name=entry.name,
                    raw_path=raw_path,
                    requested_offset=_optional_int_tool_argument(
                        kwargs.get("offset", 1),
                    ),
                    requested_limit=_optional_int_tool_argument(
                        kwargs.get("limit"),
                    ),
                    observation=session_read_observation,
                    hint_emitted=session_hint_emitted,
                )
        if session_read_unclassified_reason is not None:
            raw_path = kwargs.get("path")
            await _emit_unclassified_read_file_session_telemetry(
                context=context,
                tool_name=entry.name,
                call_id=tc.call_id,
                render_id=source_render_id,
                raw_path=raw_path if isinstance(raw_path, str) else None,
                requested_offset=_optional_int_tool_argument(
                    kwargs.get("offset", 1),
                ),
                requested_limit=_optional_int_tool_argument(
                    kwargs.get("limit"),
                ),
                reason=session_read_unclassified_reason,
            )
        if session_read_hint is not None and hint is None:
            hint = session_read_hint
        search_hint, search_observation = _search_files_prior_context_hint(
            context=context,
            source_render_id=source_render_id,
            tool_name=entry.name,
            call_id=tc.call_id,
            arguments=kwargs,
            result_content=outcome.content,
            is_error=outcome.is_error,
        )
        if search_observation is not None:
            await _emit_search_files_prior_context_telemetry(
                context=context,
                tool_name=entry.name,
                observation=search_observation,
            )
        if search_hint is not None:
            hint = search_hint
        if hint is not None:
            advisory_nodes.append(AdvisoryNode(
                source="prior_context",
                content=hint,
            ))
            await context.event_sink.on_status(
                _prior_context_hint_status(tool_name=entry.name, hint=hint),
                scope=context.scope,
            )
        if entry.mcp_server_config is None:
            from thorn.runtime._active_context import record_tool_active_context

            record_tool_active_context(
                session,
                tool_name=entry.name,
                arguments=kwargs,
                result=outcome.content,
                is_error=outcome.is_error,
                workspace_root=context.workspace_root,
            )
        results.append(ToolResultMessage(
            call_id=tc.call_id,
            content=outcome.content,
            is_error=outcome.is_error,
            external_content_peer_statuses=outcome.external_content_peer_statuses,
        ))
        await context.event_sink.on_tool_end(
            tc.name,
            duration_s=duration_s,
            error=outcome.content if outcome.is_error else None,
            scope=context.scope,
        )

    if results:
        for provider in context.status_providers:
            provider.refresh(session)
            text = provider.render_status(session)
            if text:
                advisory_nodes.append(AdvisoryNode(
                    source=provider.source_label,
                    content=text,
                ))
                await context.event_sink.on_advisory(
                    provider.source_label, text, scope=context.scope,
                )

    return (
        results,
        call_node_classes,
        advisory_nodes,
        captured,
        validation_content_snapshots,
    )
