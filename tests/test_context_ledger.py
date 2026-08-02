"""Black-box tests for request-local provider-history projection."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest

from thorn.core._context import (
    ExecutionContext,
    NullEventSink,
    reset_context,
    set_context,
)
from thorn.core._context_ledger import (
    BOUNDED_HISTORY_V2_CONTEXT_BUDGET_POLICY,
    DEFAULT_CONTEXT_BUDGET_POLICY,
    ContextBudgetPolicy,
    ContextWindowFraction,
    EstimatedTokenCount,
    FileContextProjectionLedgerEntry,
    HistoryBudgetSource,
    ProviderHistoryDisposition,
    ToolCallContextLedgerEntry,
    estimate_provider_history_tokens,
    project_history_for_provider,
)
from thorn.core._executor import ToolVenue
from thorn.core._file_context_folding import (
    plan_file_observation_replacements,
)
from thorn.core._func import wrap_function
from thorn.core._history import CollapseState, HistoryTree, TurnNode, estimate_tokens
from thorn.core._loop import run_agent_loop
from thorn.core._messages import (
    AssistantMessage,
    Message,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)
from thorn.core._prompt_trace import PromptTraceRecorder
from thorn.core._provider import (
    FinishChunk,
    LLMProvider,
    MockProvider,
    TextChunk,
    ToolCallChunk,
    UsageChunk,
)
from thorn.core._provider_telemetry import ProviderAttemptTelemetry
from thorn.core._retry import RetryPolicy
from thorn.core._tools import edit_file
from thorn.core.errors import TransientProviderError
from thorn.runtime import Runtime


def _tool_call(call_id: str, name: str, arguments: dict[str, Any]) -> ToolCall:
    return ToolCall(
        call_id=call_id,
        name=name,
        arguments=json.dumps(arguments),
    )


def _append_tool_turn(
    history: HistoryTree,
    *,
    call_id: str,
    name: str,
    arguments: dict[str, Any],
    content: str,
    is_error: bool = False,
) -> None:
    tool_call = _tool_call(call_id, name, arguments)
    history.append_turn(
        AssistantMessage(content=f"calling {name}", tool_calls=[tool_call]),
        [ToolResultMessage(
            call_id=call_id,
            content=content,
            is_error=is_error,
        )],
    )


def _result_content(messages: tuple[Message, ...], call_id: str) -> str:
    return next(
        message.content for message in messages
        if isinstance(message, ToolResultMessage)
        and message.call_id == call_id
    )


def _dispositions(projection) -> dict[str, ProviderHistoryDisposition]:
    return {
        entry.call_id: entry.disposition
        for entry in projection.ledger.entries
        if hasattr(entry, "call_id")
    }


def _assert_valid_tool_protocol(messages: tuple[Message, ...]) -> None:
    for message_index, message in enumerate(messages):
        if not isinstance(message, AssistantMessage) or not message.tool_calls:
            continue
        expected_call_ids = [call.call_id for call in message.tool_calls]
        actual_call_ids: list[str] = []
        following_index = message_index + 1
        while (
            following_index < len(messages)
            and isinstance(messages[following_index], ToolResultMessage)
        ):
            result = messages[following_index]
            assert isinstance(result, ToolResultMessage)
            actual_call_ids.append(result.call_id)
            following_index += 1
        assert actual_call_ids == expected_call_ids

    announced_call_ids = {
        call.call_id
        for message in messages
        if isinstance(message, AssistantMessage)
        for call in message.tool_calls
    }
    result_call_ids = {
        message.call_id
        for message in messages
        if isinstance(message, ToolResultMessage)
    }
    assert result_call_ids == announced_call_ids


class TestContextBudgetPolicy:
    def test_unknown_window_uses_fixed_default(self) -> None:
        policy = ContextBudgetPolicy(
            default_history_tokens=EstimatedTokenCount(12_000),
            maximum_history_tokens=EstimatedTokenCount(24_000),
            soft_context_window_fraction=ContextWindowFraction(0.60),
            hard_context_window_fraction=ContextWindowFraction(0.80),
        )

        resolved = policy.resolve(
            context_window=None,
            estimated_overhead_tokens=EstimatedTokenCount(5_000),
        )

        assert resolved.history_tokens == EstimatedTokenCount(12_000)
        assert resolved.source is HistoryBudgetSource.FIXED_DEFAULT
        assert resolved.hard_prompt_tokens is None

    def test_known_window_subtracts_overhead_and_has_hard_limit(self) -> None:
        policy = ContextBudgetPolicy(
            default_history_tokens=EstimatedTokenCount(12_000),
            maximum_history_tokens=EstimatedTokenCount(24_000),
            soft_context_window_fraction=ContextWindowFraction(0.60),
            hard_context_window_fraction=ContextWindowFraction(0.80),
        )

        relative = policy.resolve(
            context_window=20_000,
            estimated_overhead_tokens=EstimatedTokenCount(5_000),
        )
        capped = policy.resolve(
            context_window=128_000,
            estimated_overhead_tokens=EstimatedTokenCount(5_000),
        )

        assert relative.history_tokens == EstimatedTokenCount(7_000)
        assert relative.source is HistoryBudgetSource.RELATIVE_CONTEXT_WINDOW
        assert relative.hard_prompt_tokens == EstimatedTokenCount(16_000)
        assert capped.history_tokens == EstimatedTokenCount(24_000)
        assert capped.source is HistoryBudgetSource.FIXED_MAXIMUM

    @pytest.mark.parametrize("value", [-0.1, 0.0, 1.1, float("inf")])
    def test_fraction_rejects_invalid_value(self, value: float) -> None:
        with pytest.raises(ValueError):
            ContextWindowFraction(value)

    def test_runtime_and_child_scope_preserve_selected_policy(
        self,
        tmp_path: Path,
    ) -> None:
        runtime = Runtime(
            provider=MockProvider(),
            workspace_root=tmp_path,
            context_budget_policy=DEFAULT_CONTEXT_BUDGET_POLICY,
        )

        context = runtime.create_context()
        child_context = context.push_scope("child")

        assert context.context_budget_policy is DEFAULT_CONTEXT_BUDGET_POLICY
        assert child_context.context_budget_policy is DEFAULT_CONTEXT_BUDGET_POLICY


class TestRequiredFileObservationReplacement:
    def test_single_stale_read_is_replaced_without_mutating_history(
        self,
        tmp_path: Path,
    ) -> None:
        (tmp_path / "example.py").write_text("new alpha\nnew beta\n")
        history = HistoryTree()
        history.append_user_prompt("inspect example.py")
        _append_tool_turn(
            history,
            call_id="old-read",
            name="read_file",
            arguments={"path": "example.py", "offset": 1, "limit": 2},
            content="1| old alpha\n2| old beta",
        )

        projection = project_history_for_provider(
            history,
            workspace_root=tmp_path,
            context_window=None,
            estimated_overhead_tokens=0,
        )

        assert "stale file observation omitted" in _result_content(
            projection.messages,
            "old-read",
        )
        assert _dispositions(projection)["old-read"] is (
            ProviderHistoryDisposition.STALE_FILE_OBSERVATION
        )
        turn = history.nodes[-1]
        assert isinstance(turn, TurnNode)
        assert turn.tool_call_nodes[0].result.content == (
            "1| old alpha\n2| old beta"
        )
        assert turn.collapse_state is CollapseState.EXPANDED
        _assert_valid_tool_protocol(projection.messages)

    def test_read_of_now_missing_allowed_file_is_stale(
        self,
        tmp_path: Path,
    ) -> None:
        history = HistoryTree()
        history.append_user_prompt("inspect deleted.py")
        _append_tool_turn(
            history,
            call_id="deleted-read",
            name="read_file",
            arguments={"path": "deleted.py"},
            content="1| old content",
        )

        projection = project_history_for_provider(
            history,
            workspace_root=tmp_path,
            context_window=None,
            estimated_overhead_tokens=0,
        )

        assert _dispositions(projection)["deleted-read"] is (
            ProviderHistoryDisposition.STALE_FILE_OBSERVATION
        )

    def test_search_only_proof_does_not_load_search_hit_files(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        search_hit = tmp_path / "example.py"
        search_hit.write_text("target\n", encoding="utf-8")
        history = HistoryTree()
        history.append_user_prompt("find target")
        _append_tool_turn(
            history,
            call_id="search",
            name="search_files",
            arguments={"path": ".", "pattern": "target"},
            content="example.py:\n1| target",
        )
        loaded_paths: list[Path] = []
        original_read_text = Path.read_text

        def record_read_text(path: Path, *args: Any, **kwargs: Any) -> str:
            loaded_paths.append(path)
            return original_read_text(path, *args, **kwargs)

        monkeypatch.setattr(Path, "read_text", record_read_text)

        replacements = plan_file_observation_replacements(
            history.nodes,
            workspace_root=tmp_path,
        )

        assert replacements == ()
        assert loaded_paths == []

    def test_treatment_does_not_reintroduce_stored_compacted_file_observations(
        self,
        tmp_path: Path,
    ) -> None:
        (tmp_path / "example.py").write_text(
            "alpha\nbeta\ngamma\n",
            encoding="utf-8",
        )
        history = HistoryTree()
        history.append_user_prompt("inspect example.py")
        for call_id, line_number, content in (
            ("collapsed-turn", 1, "alpha"),
            ("collapsed-detail", 2, "beta"),
            ("newest-read", 3, "gamma"),
        ):
            _append_tool_turn(
                history,
                call_id=call_id,
                name="read_file",
                arguments={
                    "path": "example.py",
                    "offset": line_number,
                    "limit": 1,
                },
                content=f"{line_number}| {content}",
            )
        collapsed_turn = history.nodes[1]
        assert isinstance(collapsed_turn, TurnNode)
        collapsed_turn.collapse_state = CollapseState.COLLAPSED
        collapsed_detail_turn = history.nodes[2]
        assert isinstance(collapsed_detail_turn, TurnNode)
        collapsed_detail_turn.tool_call_nodes[0].detail_collapsed = True

        projection = project_history_for_provider(
            history,
            workspace_root=tmp_path,
            context_window=None,
            estimated_overhead_tokens=0,
        )

        assert not any(
            isinstance(message, Message)
            and getattr(message, "content", "").startswith(
                "[Folded file context]",
            )
            for message in projection.messages
        )

    def test_fully_subsumed_read_is_replaced_but_partial_overlap_is_not(
        self,
        tmp_path: Path,
    ) -> None:
        (tmp_path / "example.py").write_text("one\ntwo\nthree\nfour\n")
        history = HistoryTree()
        history.append_user_prompt("inspect example.py")
        _append_tool_turn(
            history,
            call_id="subsumed",
            name="read_file",
            arguments={"path": "example.py", "offset": 1, "limit": 2},
            content="1| one\n2| two",
        )
        _append_tool_turn(
            history,
            call_id="superset",
            name="read_file",
            arguments={"path": "example.py", "offset": 1, "limit": 3},
            content="1| one\n2| two\n3| three",
        )
        _append_tool_turn(
            history,
            call_id="partial",
            name="read_file",
            arguments={"path": "example.py", "offset": 3, "limit": 2},
            content="3| three\n4| four",
        )

        projection = project_history_for_provider(
            history,
            workspace_root=tmp_path,
            context_window=None,
            estimated_overhead_tokens=0,
        )
        dispositions = _dispositions(projection)

        assert dispositions["subsumed"] is (
            ProviderHistoryDisposition.REDUNDANT_FILE_OBSERVATION
        )
        assert dispositions.get("partial") is not (
            ProviderHistoryDisposition.REDUNDANT_FILE_OBSERVATION
        )
        assert "redundant file observation omitted" in _result_content(
            projection.messages,
            "subsumed",
        )

    def test_redundancy_is_proved_before_six_file_selection(
        self,
        tmp_path: Path,
    ) -> None:
        history = HistoryTree()
        history.append_user_prompt("inspect the workspace")
        for file_index in range(7):
            path = tmp_path / f"file{file_index}.py"
            path.write_text("one\ntwo\nthree\n")
            first_call_id = f"file{file_index}-first"
            second_call_id = f"file{file_index}-second"
            _append_tool_turn(
                history,
                call_id=first_call_id,
                name="read_file",
                arguments={"path": path.name, "offset": 1, "limit": 2},
                content="1| one\n2| two",
            )
            _append_tool_turn(
                history,
                call_id=second_call_id,
                name="read_file",
                arguments={"path": path.name, "offset": 1, "limit": 3},
                content="1| one\n2| two\n3| three",
            )

        projection = project_history_for_provider(
            history,
            workspace_root=tmp_path,
            context_window=None,
            estimated_overhead_tokens=0,
        )

        assert _dispositions(projection)["file0-first"] is (
            ProviderHistoryDisposition.REDUNDANT_FILE_OBSERVATION
        )
        assert "redundant file observation omitted" in _result_content(
            projection.messages,
            "file0-first",
        )
        _assert_valid_tool_protocol(projection.messages)

    def test_treatment_folds_older_single_read_but_preserves_newest(
        self,
        tmp_path: Path,
    ) -> None:
        (tmp_path / "older.py").write_text("older current\n")
        (tmp_path / "newest.py").write_text("newest current\n")
        history = HistoryTree()
        history.append_user_prompt("inspect both files")
        _append_tool_turn(
            history,
            call_id="older-read",
            name="read_file",
            arguments={"path": "older.py"},
            content="1| older current",
        )
        _append_tool_turn(
            history,
            call_id="newest-read",
            name="read_file",
            arguments={"path": "newest.py"},
            content="1| newest current",
        )

        baseline = history.render_with_visibility(workspace_root=tmp_path)
        treatment = project_history_for_provider(
            history,
            workspace_root=tmp_path,
            context_window=None,
            estimated_overhead_tokens=0,
        )

        assert _result_content(tuple(baseline.messages), "older-read") == (
            "1| older current"
        )
        assert "folded into current file context" in _result_content(
            treatment.messages,
            "older-read",
        )
        assert _result_content(treatment.messages, "newest-read") == (
            "1| newest current"
        )
        fold_entry = next(
            entry
            for entry in treatment.ledger.entries
            if isinstance(entry, FileContextProjectionLedgerEntry)
            and "older-read" in entry.call_ids
        )
        baseline_tokens = estimate_provider_history_tokens(baseline.messages)
        treatment_tokens = estimate_provider_history_tokens(treatment.messages)
        assert fold_entry.token_delta.before.value == baseline_tokens
        assert fold_entry.token_delta.after.value == treatment_tokens
        assert (
            baseline_tokens
            + treatment.ledger.estimated_total_token_growth.value
            - treatment.ledger.estimated_total_token_savings.value
        ) == treatment_tokens
        assert treatment.ledger.to_json()["disposition_counts"][
            "file_context_folded"
        ] == 1
        folded_context = "\n".join(
            getattr(message, "content", "") for message in treatment.messages
        )
        assert "[Current file context: older.py]" in folded_context

    def test_exact_duplicate_search_keeps_newer_result(self, tmp_path: Path) -> None:
        (tmp_path / "example.py").write_text("target\n")
        history = HistoryTree()
        history.append_user_prompt("find target")
        for call_id in ("search-old", "search-new"):
            _append_tool_turn(
                history,
                call_id=call_id,
                name="search_files",
                arguments={"path": ".", "pattern": "target"},
                content="example.py:\n1| target",
            )

        projection = project_history_for_provider(
            history,
            workspace_root=tmp_path,
            context_window=None,
            estimated_overhead_tokens=0,
        )

        assert _dispositions(projection)["search-old"] is (
            ProviderHistoryDisposition.REDUNDANT_FILE_OBSERVATION
        )
        assert "identical result is retained" in _result_content(
            projection.messages,
            "search-old",
        )
        assert "1| target" in _result_content(projection.messages, "search-new")

    def test_current_read_subsumes_older_single_file_search(
        self,
        tmp_path: Path,
    ) -> None:
        (tmp_path / "example.py").write_text(
            "before\ntarget\nafter\n",
            encoding="utf-8",
        )
        history = HistoryTree()
        history.append_user_prompt("find and inspect target")
        _append_tool_turn(
            history,
            call_id="search-old",
            name="search_files",
            arguments={"path": ".", "pattern": "target"},
            content="example.py:\n2| target",
        )
        _append_tool_turn(
            history,
            call_id="read-new",
            name="read_file",
            arguments={"path": "example.py", "offset": 1, "limit": 3},
            content="1| before\n2| target\n3| after",
        )

        projection = project_history_for_provider(
            history,
            workspace_root=tmp_path,
            context_window=None,
            estimated_overhead_tokens=0,
            policy=BOUNDED_HISTORY_V2_CONTEXT_BUDGET_POLICY,
        )

        assert _dispositions(projection)["search-old"] is (
            ProviderHistoryDisposition.REDUNDANT_FILE_OBSERVATION
        )
        assert "search evidence is retained" in _result_content(
            projection.messages,
            "search-old",
        )
        assert "2| target" in _result_content(projection.messages, "read-new")

    def test_v1_does_not_gain_v2_search_subsumption(self, tmp_path: Path) -> None:
        (tmp_path / "example.py").write_text("target\n", encoding="utf-8")
        history = HistoryTree()
        history.append_user_prompt("find and inspect target")
        _append_tool_turn(
            history,
            call_id="search-old",
            name="search_files",
            arguments={"path": ".", "pattern": "target"},
            content="example.py:\n1| target",
        )
        _append_tool_turn(
            history,
            call_id="read-new",
            name="read_file",
            arguments={"path": "example.py"},
            content="1| target",
        )

        projection = project_history_for_provider(
            history,
            workspace_root=tmp_path,
            context_window=None,
            estimated_overhead_tokens=0,
            policy=DEFAULT_CONTEXT_BUDGET_POLICY,
        )

        assert "search-old" not in _dispositions(projection)

    def test_partial_newer_read_does_not_subsume_older_search(
        self,
        tmp_path: Path,
    ) -> None:
        (tmp_path / "example.py").write_text(
            "first target\nsecond target\n",
            encoding="utf-8",
        )
        history = HistoryTree()
        history.append_user_prompt("find both targets")
        _append_tool_turn(
            history,
            call_id="search-old",
            name="search_files",
            arguments={"path": ".", "pattern": "target"},
            content="example.py:\n1| first target\n2| second target",
        )
        _append_tool_turn(
            history,
            call_id="read-new",
            name="read_file",
            arguments={"path": "example.py", "offset": 1, "limit": 1},
            content="1| first target",
        )

        projection = project_history_for_provider(
            history,
            workspace_root=tmp_path,
            context_window=None,
            estimated_overhead_tokens=0,
            policy=BOUNDED_HISTORY_V2_CONTEXT_BUDGET_POLICY,
        )

        assert "search-old" not in _dispositions(projection)
        rendered = "\n".join(
            getattr(message, "content", "") for message in projection.messages
        )
        assert "2!| second target" in rendered

    def test_stale_newer_read_does_not_subsume_older_search(
        self,
        tmp_path: Path,
    ) -> None:
        (tmp_path / "example.py").write_text(
            "current target\n",
            encoding="utf-8",
        )
        history = HistoryTree()
        history.append_user_prompt("find and inspect target")
        _append_tool_turn(
            history,
            call_id="search-old",
            name="search_files",
            arguments={"path": ".", "pattern": "target"},
            content="example.py:\n1| current target",
        )
        _append_tool_turn(
            history,
            call_id="read-stale",
            name="read_file",
            arguments={"path": "example.py"},
            content="1| stale target",
        )

        projection = project_history_for_provider(
            history,
            workspace_root=tmp_path,
            context_window=None,
            estimated_overhead_tokens=0,
            policy=BOUNDED_HISTORY_V2_CONTEXT_BUDGET_POLICY,
        )

        assert _dispositions(projection)["read-stale"] is (
            ProviderHistoryDisposition.STALE_FILE_OBSERVATION
        )
        assert "search-old" not in _dispositions(projection)

    def test_multi_file_search_requires_one_complete_superseding_result(
        self,
        tmp_path: Path,
    ) -> None:
        (tmp_path / "first.py").write_text("target one\n", encoding="utf-8")
        (tmp_path / "second.py").write_text("target two\n", encoding="utf-8")
        history = HistoryTree()
        history.append_user_prompt("find all targets")
        _append_tool_turn(
            history,
            call_id="search-old",
            name="search_files",
            arguments={"path": ".", "pattern": "target"},
            content=(
                "first.py:\n1| target one\n"
                "second.py:\n1| target two"
            ),
        )
        _append_tool_turn(
            history,
            call_id="read-first",
            name="read_file",
            arguments={"path": "first.py"},
            content="1| target one",
        )
        _append_tool_turn(
            history,
            call_id="read-second",
            name="read_file",
            arguments={"path": "second.py"},
            content="1| target two",
        )

        projection = project_history_for_provider(
            history,
            workspace_root=tmp_path,
            context_window=None,
            estimated_overhead_tokens=0,
            policy=BOUNDED_HISTORY_V2_CONTEXT_BUDGET_POLICY,
        )

        assert "search-old" not in _dispositions(projection)

    def test_v2_retains_truncated_search_and_its_count_metadata(
        self,
        tmp_path: Path,
    ) -> None:
        (tmp_path / "example.py").write_text("target\n", encoding="utf-8")
        history = HistoryTree()
        history.append_user_prompt("find every target")
        _append_tool_turn(
            history,
            call_id="search-truncated",
            name="search_files",
            arguments={"path": ".", "pattern": "target"},
            content=(
                "example.py:\n1| target\n\n"
                "[Truncated: showing 1 of 9 total matches across 4 files. "
                "Narrow your pattern.]"
            ),
        )
        _append_tool_turn(
            history,
            call_id="read-new",
            name="read_file",
            arguments={"path": "example.py"},
            content="1| target",
        )
        for index in range(8):
            _append_tool_turn(
                history,
                call_id=f"later-{index}",
                name="list_directory",
                arguments={"path": "."},
                content=f"later observation {index}\n" + "x" * 4_000,
            )
        policy = ContextBudgetPolicy(
            default_history_tokens=EstimatedTokenCount(80),
            maximum_history_tokens=EstimatedTokenCount(80),
            soft_context_window_fraction=ContextWindowFraction(0.60),
            hard_context_window_fraction=ContextWindowFraction(0.80),
            search_replacement_policy=(
                BOUNDED_HISTORY_V2_CONTEXT_BUDGET_POLICY.search_replacement_policy
            ),
        )

        projection = project_history_for_provider(
            history,
            workspace_root=tmp_path,
            context_window=None,
            estimated_overhead_tokens=0,
            policy=policy,
        )

        assert "search-truncated" not in _dispositions(projection)
        retained_search = _result_content(
            projection.messages,
            "search-truncated",
        )
        assert "showing 1 of 9 total matches across 4 files" in retained_search
        assert projection.ledger.is_unavoidably_over_budget

    def test_budget_collapse_preserves_action_and_error_summaries(
        self,
        tmp_path: Path,
    ) -> None:
        history = HistoryTree()
        history.append_user_prompt("make and validate a change")
        _append_tool_turn(
            history,
            call_id="edit",
            name="edit_file",
            arguments={
                "path": "example.py",
                "edits": [{"old_text": "old", "new_text": "new"}],
            },
            content="1| new\n" + "x" * 4_000,
        )
        _append_tool_turn(
            history,
            call_id="failed-validation",
            name="run_shell",
            arguments={"command": "run validator"},
            content="validator failed: environment unavailable\n" + "y" * 4_000,
            is_error=True,
        )
        for index in range(8):
            _append_tool_turn(
                history,
                call_id=f"observation-{index}",
                name="list_directory",
                arguments={"path": "."},
                content=f"observation {index}\n" + "z" * 4_000,
            )
        policy = ContextBudgetPolicy(
            default_history_tokens=EstimatedTokenCount(80),
            maximum_history_tokens=EstimatedTokenCount(80),
            soft_context_window_fraction=ContextWindowFraction(0.60),
            hard_context_window_fraction=ContextWindowFraction(0.80),
        )

        projection = project_history_for_provider(
            history,
            workspace_root=tmp_path,
            context_window=None,
            estimated_overhead_tokens=0,
            policy=policy,
        )
        rendered = "\n".join(
            getattr(message, "content", "") for message in projection.messages
        )

        assert "edit_file('example.py') -> applied 1 edit(s)" in rendered
        assert "run_shell('run validator') -> error:" in rendered
        assert projection.ledger.is_unavoidably_over_budget

    def test_duplicate_large_reads_report_marginal_provider_view_savings(
        self,
        tmp_path: Path,
    ) -> None:
        source_lines = [
            f"line {line_number} " + "x" * 80
            for line_number in range(1, 201)
        ]
        (tmp_path / "large.py").write_text(
            "\n".join(source_lines) + "\n",
            encoding="utf-8",
        )
        rendered_result = "\n".join(
            f"{line_number}| {line}"
            for line_number, line in enumerate(source_lines, start=1)
        )
        history = HistoryTree()
        history.append_user_prompt("inspect large.py twice")
        for call_id in ("large-read-old", "large-read-new"):
            _append_tool_turn(
                history,
                call_id=call_id,
                name="read_file",
                arguments={"path": "large.py", "offset": 1, "limit": 200},
                content=rendered_result,
            )
        policy = ContextBudgetPolicy(
            default_history_tokens=EstimatedTokenCount(100_000),
            maximum_history_tokens=EstimatedTokenCount(100_000),
            soft_context_window_fraction=ContextWindowFraction(0.60),
            hard_context_window_fraction=ContextWindowFraction(0.80),
        )
        baseline = history.render_with_visibility(workspace_root=tmp_path)

        projection = project_history_for_provider(
            history,
            workspace_root=tmp_path,
            context_window=None,
            estimated_overhead_tokens=0,
            policy=policy,
        )

        redundant_entry = next(
            entry for entry in projection.ledger.entries
            if isinstance(entry, ToolCallContextLedgerEntry)
            and entry.disposition
                is ProviderHistoryDisposition.REDUNDANT_FILE_OBSERVATION
        )
        replacement = next(
            replacement
            for replacement in plan_file_observation_replacements(
                history.nodes,
                workspace_root=tmp_path,
            )
            if replacement.call_id == "large-read-old"
        )
        raw_stored_result_savings = max(
            0,
            estimate_tokens(rendered_result)
            - estimate_tokens(replacement.content),
        )
        baseline_tokens = estimate_provider_history_tokens(baseline.messages)
        final_tokens = estimate_provider_history_tokens(projection.messages)

        assert redundant_entry.token_delta.before.value == baseline_tokens
        assert redundant_entry.token_delta.after.value == final_tokens
        assert redundant_entry.estimated_token_savings.value != (
            raw_stored_result_savings
        )
        assert projection.ledger.estimated_total_token_growth.value == 0
        assert (
            baseline_tokens
            + projection.ledger.estimated_total_token_growth.value
            - projection.ledger.estimated_total_token_savings.value
        ) == final_tokens
        ledger_json = projection.ledger.to_json()
        assert ledger_json["disposition_counts"] == {
            "stale_file_observation": 0,
            "redundant_file_observation": 1,
            "file_context_folded": 0,
            "detail_collapsed": 0,
            "turn_collapsed": 0,
        }
        assert ledger_json["estimated_total_token_savings"] == (
            baseline_tokens - final_tokens
        )
        assert ledger_json["estimated_total_token_growth"] == 0

    def test_projection_uses_one_lazy_live_file_snapshot(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        target = tmp_path / "changing.py"
        first_version = "first version\nstable line\n"
        second_version = "second version\nstable line\n"
        target.write_text(first_version, encoding="utf-8")
        history = HistoryTree()
        history.append_user_prompt("inspect changing.py")
        for tool_index in range(8):
            _append_tool_turn(
                history,
                call_id=f"large-observation-{tool_index}",
                name="inspect",
                arguments={"target": tool_index},
                content="x" * 2_000,
            )
        _append_tool_turn(
            history,
            call_id="changing-first-line",
            name="read_file",
            arguments={"path": "changing.py", "offset": 1, "limit": 1},
            content="1| first version",
        )
        _append_tool_turn(
            history,
            call_id="changing-second-line",
            name="read_file",
            arguments={"path": "changing.py", "offset": 2, "limit": 1},
            content="2| stable line",
        )
        target_reads = 0
        original_read_text = Path.read_text

        def read_then_mutate(path: Path, *args: Any, **kwargs: Any) -> str:
            nonlocal target_reads
            content = original_read_text(path, *args, **kwargs)
            if path.resolve() != target.resolve():
                return content
            target_reads += 1
            if target_reads == 1:
                target.write_text(second_version, encoding="utf-8")
            return content

        monkeypatch.setattr(Path, "read_text", read_then_mutate)
        policy = ContextBudgetPolicy(
            default_history_tokens=EstimatedTokenCount(200),
            maximum_history_tokens=EstimatedTokenCount(200),
            soft_context_window_fraction=ContextWindowFraction(0.60),
            hard_context_window_fraction=ContextWindowFraction(0.80),
        )

        first_projection = project_history_for_provider(
            history,
            workspace_root=tmp_path,
            context_window=None,
            estimated_overhead_tokens=0,
            policy=policy,
        )

        assert target_reads == 1
        folded_context = next(
            message.content
            for message in first_projection.messages
            if isinstance(message, Message)
            and getattr(message, "content", "").startswith(
                "[Folded file context]",
            )
        )
        assert "1*| first version" in folded_context
        assert "second version" not in folded_context
        assert not any(
            getattr(entry, "disposition", None)
                is ProviderHistoryDisposition.STALE_FILE_OBSERVATION
            for entry in first_projection.ledger.entries
        )
        assert any(
            getattr(entry, "disposition", None)
                in {
                    ProviderHistoryDisposition.DETAIL_COLLAPSED,
                    ProviderHistoryDisposition.TURN_COLLAPSED,
                }
            for entry in first_projection.ledger.entries
        )

        second_projection = project_history_for_provider(
            history,
            workspace_root=tmp_path,
            context_window=None,
            estimated_overhead_tokens=0,
            policy=policy,
        )

        assert target_reads == 2
        assert _dispositions(second_projection)["changing-first-line"] is (
            ProviderHistoryDisposition.STALE_FILE_OBSERVATION
        )

    def test_direct_renders_take_independent_live_file_snapshots(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        target = tmp_path / "changing.py"
        target.write_text("first version\nstable line\n", encoding="utf-8")
        history = HistoryTree()
        history.append_user_prompt("inspect changing.py")
        _append_tool_turn(
            history,
            call_id="changing-first-line",
            name="read_file",
            arguments={"path": "changing.py", "offset": 1, "limit": 1},
            content="1| first version",
        )
        _append_tool_turn(
            history,
            call_id="changing-second-line",
            name="read_file",
            arguments={"path": "changing.py", "offset": 2, "limit": 1},
            content="2| stable line",
        )
        target_reads = 0
        original_read_text = Path.read_text

        def read_then_mutate(path: Path, *args: Any, **kwargs: Any) -> str:
            nonlocal target_reads
            content = original_read_text(path, *args, **kwargs)
            if path.resolve() != target.resolve():
                return content
            target_reads += 1
            if target_reads == 1:
                target.write_text(
                    "second version\nstable line\n",
                    encoding="utf-8",
                )
            return content

        monkeypatch.setattr(Path, "read_text", read_then_mutate)

        context = ExecutionContext(
            provider=MockProvider(),
            workspace_root=tmp_path,
        )
        context_token = set_context(context)
        try:
            first_render = history.render_with_visibility(
                workspace_root=tmp_path,
            )
            second_render = history.render_with_visibility(
                workspace_root=tmp_path,
            )
        finally:
            reset_context(context_token)

        assert target_reads == 2
        first_content = "\n".join(
            getattr(message, "content", "")
            for message in first_render.messages
        )
        second_content = "\n".join(
            getattr(message, "content", "")
            for message in second_render.messages
        )
        assert "first version" in first_content
        assert "second version" not in first_content
        assert "second version" in second_content

    def test_failed_and_unparsed_reads_are_not_removed(self, tmp_path: Path) -> None:
        (tmp_path / "example.py").write_text("current\n")
        history = HistoryTree()
        history.append_user_prompt("inspect example.py")
        calls = [
            _tool_call("stale", "read_file", {"path": "example.py"}),
            _tool_call("failed", "read_file", {"path": "example.py"}),
            _tool_call("unknown", "read_file", {"path": "example.py"}),
        ]
        history.append_turn(
            AssistantMessage(content="batch inspection", tool_calls=calls),
            [
                ToolResultMessage(call_id="stale", content="1| old"),
                ToolResultMessage(
                    call_id="failed",
                    content="permission denied",
                    is_error=True,
                ),
                ToolResultMessage(call_id="unknown", content="unparsed output"),
            ],
        )

        projection = project_history_for_provider(
            history,
            workspace_root=tmp_path,
            context_window=None,
            estimated_overhead_tokens=0,
        )

        assert "stale file observation omitted" in _result_content(
            projection.messages,
            "stale",
        )
        assert _result_content(projection.messages, "failed") == "permission denied"
        assert _result_content(projection.messages, "unknown") == "unparsed output"
        _assert_valid_tool_protocol(projection.messages)

    def test_path_outside_workspace_is_not_inspected_without_active_context(
        self,
        tmp_path: Path,
    ) -> None:
        workspace_root = tmp_path / "workspace"
        workspace_root.mkdir()
        outside_path = tmp_path / "outside.py"
        outside_path.write_text("current secret\n", encoding="utf-8")
        history = HistoryTree()
        history.append_user_prompt("inspect only the workspace")
        _append_tool_turn(
            history,
            call_id="outside-read",
            name="read_file",
            arguments={"path": str(outside_path)},
            content="1| historical secret",
        )

        projection = project_history_for_provider(
            history,
            workspace_root=workspace_root,
            context_window=None,
            estimated_overhead_tokens=0,
        )

        assert _result_content(projection.messages, "outside-read") == (
            "1| historical secret"
        )
        assert "outside-read" not in _dispositions(projection)


class TestSoftHistoryBudget:
    def test_budget_collapses_provider_view_without_mutating_history(self) -> None:
        history = HistoryTree()
        history.append_user_prompt("perform the task")
        for tool_index in range(14):
            _append_tool_turn(
                history,
                call_id=f"observation-{tool_index}",
                name="inspect",
                arguments={"target": f"item-{tool_index}"},
                content=(f"observation {tool_index}\n" + "x" * 4_000),
            )
        history.append_turn(AssistantMessage(content="considering next step"), [])
        policy = ContextBudgetPolicy(
            default_history_tokens=EstimatedTokenCount(1_500),
            maximum_history_tokens=EstimatedTokenCount(1_500),
            soft_context_window_fraction=ContextWindowFraction(0.60),
            hard_context_window_fraction=ContextWindowFraction(0.80),
        )

        projection = project_history_for_provider(
            history,
            workspace_root=None,
            context_window=None,
            estimated_overhead_tokens=0,
            policy=policy,
        )

        assert (
            projection.ledger.estimated_history_tokens_final
            < projection.ledger.estimated_baseline_provider_history_tokens
        )
        assert any(
            getattr(entry, "disposition", None)
            in {
                ProviderHistoryDisposition.DETAIL_COLLAPSED,
                ProviderHistoryDisposition.TURN_COLLAPSED,
            }
            for entry in projection.ledger.entries
        )
        for node in history.nodes:
            if not isinstance(node, TurnNode):
                continue
            assert node.collapse_state is CollapseState.EXPANDED
            assert all(
                not tool_node.detail_collapsed
                for tool_node in node.tool_call_nodes
            )
        _assert_valid_tool_protocol(projection.messages)


class _RetryCapturingProvider(LLMProvider):
    def __init__(self) -> None:
        self.payloads: list[str] = []
        self.call_count = 0

    async def complete(self, system_prompts, tools, messages):
        self.call_count += 1
        self.payloads.append(json.dumps([
            _message_payload(message) for message in messages
        ], sort_keys=True))
        if self.call_count == 1:
            raise TransientProviderError("retry me")
            yield  # pragma: no cover
        yield TextChunk(text="done")
        yield UsageChunk(
            prompt_tokens=100,
            completion_tokens=1,
            total_tokens=101,
        )
        yield FinishChunk(reason="stop")


class _SuccessfulCapturingProvider(LLMProvider):
    def __init__(self) -> None:
        self.messages: list[tuple[Message, ...]] = []

    async def complete(self, system_prompts, tools, messages):
        self.messages.append(tuple(messages))
        yield TextChunk(text="done")
        yield FinishChunk(reason="stop")


class _ToolThenCaptureProvider(MockProvider):
    def __init__(self, tool_call: ToolCallChunk) -> None:
        super().__init__(canned_responses=[
            [tool_call, FinishChunk(reason="tool_calls")],
            [TextChunk(text="done"), FinishChunk(reason="stop")],
        ])
        self.requests: list[tuple[Message, ...]] = []

    async def complete(self, system_prompts, tools, messages):
        self.requests.append(tuple(messages))
        async for chunk in super().complete(system_prompts, tools, messages):
            yield chunk


class _AttemptRecordingSink(NullEventSink):
    def __init__(self) -> None:
        self.attempts: list[ProviderAttemptTelemetry] = []

    async def on_provider_attempt(self, attempt, *, scope=None) -> None:
        self.attempts.append(attempt)


def _message_payload(message: Message) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "role": message.role,
        "content": getattr(message, "content", None),
    }
    if isinstance(message, AssistantMessage):
        payload["tool_calls"] = [
            {
                "call_id": call.call_id,
                "name": call.name,
                "arguments": call.arguments,
            }
            for call in message.tool_calls
        ]
    if isinstance(message, ToolResultMessage):
        payload["call_id"] = message.call_id
    return payload


class TestLoopProjectionSnapshot:
    @pytest.mark.parametrize(
        "context_budget_policy",
        [
            pytest.param(None, id="legacy-history"),
            pytest.param(
                ContextBudgetPolicy(
                    default_history_tokens=EstimatedTokenCount(300),
                    maximum_history_tokens=EstimatedTokenCount(300),
                    soft_context_window_fraction=ContextWindowFraction(0.60),
                    hard_context_window_fraction=ContextWindowFraction(0.80),
                ),
                id="bounded-history",
            ),
        ],
    )
    async def test_actual_edit_ranges_win_line_and_char_budgets(
        self,
        tmp_path: Path,
        context_budget_policy: ContextBudgetPolicy | None,
    ) -> None:
        source_lines = [f"line {line_number}" for line_number in range(1, 221)]
        source_lines[0] = "early long line " + "x" * 25_000
        (tmp_path / "example.py").write_text(
            "\n".join(source_lines),
            encoding="utf-8",
        )

        history = HistoryTree()
        history.append_user_prompt("make the requested change")
        for observation_index in range(7):
            _append_tool_turn(
                history,
                call_id=f"old-observation-{observation_index}",
                name="inspect",
                arguments={"target": f"old-{observation_index}"},
                content="old diagnostic\n" + "x" * 2_000,
            )
        _append_tool_turn(
            history,
            call_id="broad-read",
            name="read_file",
            arguments={"path": "example.py", "offset": 1, "limit": 150},
            content="\n".join(
                f"{line_number}| {source_lines[line_number - 1]}"
                for line_number in range(1, 151)
            ),
        )
        _append_tool_turn(
            history,
            call_id="search-hit",
            name="search_files",
            arguments={"path": "example.py", "pattern": "line 29"},
            content="example.py:\n29| line 29",
        )

        edit_arguments = {
            "path": "example.py",
            "edits": [
                {
                    "old_string": "line 167\nline 168\nline 169",
                    "new_string": "line 167\nedited line 168\nline 169",
                },
                {
                    "old_string": "line 190",
                    "new_string": "edited line 190",
                },
            ],
        }

        provider = _ToolThenCaptureProvider(ToolCallChunk(
            call_id="recent-edit",
            name="edit_file",
            arguments=json.dumps(edit_arguments),
        ))
        context = ExecutionContext(
            provider=provider,
            workspace_root=tmp_path,
            context_budget_policy=context_budget_policy,
        )
        context_token = set_context(context)
        try:
            await run_agent_loop(
                context=context,
                user_prompt="continue",
                tools=[wrap_function(edit_file, venue=ToolVenue.IN_PROCESS)],
                history=history,
            )
        finally:
            reset_context(context_token)

        folded_context = next(
            message.content
            for message in provider.requests[-1]
            if isinstance(message, UserMessage)
            and "[Current file context: example.py]" in message.content
        )
        for line_number in (168, 190):
            assert (
                f"{line_number}~| edited line {line_number}" in folded_context
            )
        assert "167~| line 167" not in folded_context
        assert "169~| line 169" not in folded_context
        assert source_lines[0] not in folded_context

        rendered_lines = [
            (int(match.group("line")), match.group("marker"))
            for line in folded_context.splitlines()
            if (
                match := re.match(
                    r"^\s*(?P<line>\d+)(?P<marker>[*!~ ])\|",
                    line,
                )
            )
        ]
        justified_lines = {
            line_number
            for line_number, marker in rendered_lines
            if marker != " "
        }
        assert {
            line_number
            for line_number, marker in rendered_lines
            if marker == "~"
        } == {168, 190}
        assert justified_lines
        assert all(
            marker != " "
            or any(
                abs(line_number - justified_line) <= 2
                for justified_line in justified_lines
            )
            for line_number, marker in rendered_lines
        )

    async def test_deletion_anchor_reaches_provider_folded_context(
        self,
        tmp_path: Path,
    ) -> None:
        source_lines = [f"line {line_number}" for line_number in range(1, 221)]
        source_lines[0] = "early long line " + "x" * 25_000
        (tmp_path / "example.py").write_text(
            "\n".join(source_lines),
            encoding="utf-8",
        )
        history = HistoryTree()
        history.append_user_prompt("remove the obsolete line")
        _append_tool_turn(
            history,
            call_id="broad-read",
            name="read_file",
            arguments={"path": "example.py", "offset": 1, "limit": 150},
            content="\n".join(
                f"{line_number}| {source_lines[line_number - 1]}"
                for line_number in range(1, 151)
            ),
        )
        _append_tool_turn(
            history,
            call_id="search-hit",
            name="search_files",
            arguments={"path": "example.py", "pattern": "line 29"},
            content="example.py:\n29| line 29",
        )
        provider = _ToolThenCaptureProvider(ToolCallChunk(
            call_id="recent-delete",
            name="edit_file",
            arguments=json.dumps({
                "path": "example.py",
                "edits": [{
                    "old_string": "line 190\n",
                    "new_string": "",
                }],
            }),
        ))
        context = ExecutionContext(provider=provider, workspace_root=tmp_path)
        context_token = set_context(context)
        try:
            await run_agent_loop(
                context=context,
                user_prompt="continue",
                tools=[wrap_function(edit_file, venue=ToolVenue.IN_PROCESS)],
                history=history,
            )
        finally:
            reset_context(context_token)

        folded_context = next(
            message.content
            for message in provider.requests[-1]
            if isinstance(message, UserMessage)
            and "[Current file context: example.py]" in message.content
        )
        assert "190-| line 191" in folded_context
        assert "190~|" not in folded_context
        assert source_lines[0] not in folded_context

    async def test_legacy_edit_result_keeps_context_conservatively_salient(
        self,
        tmp_path: Path,
    ) -> None:
        source_lines = [f"line {line_number}" for line_number in range(1, 41)]
        source_lines[0] = "early long line " + "x" * 25_000
        source_lines[24] = "legacy changed"
        (tmp_path / "example.py").write_text(
            "\n".join(source_lines),
            encoding="utf-8",
        )
        history = HistoryTree()
        history.append_user_prompt("resume the persisted session")
        _append_tool_turn(
            history,
            call_id="broad-read",
            name="read_file",
            arguments={"path": "example.py", "offset": 1, "limit": 20},
            content="\n".join(
                f"{line_number}| {source_lines[line_number - 1]}"
                for line_number in range(1, 21)
            ),
        )
        _append_tool_turn(
            history,
            call_id="search-hit",
            name="search_files",
            arguments={"path": "example.py", "pattern": "line 10"},
            content="example.py:\n10| line 10",
        )
        _append_tool_turn(
            history,
            call_id="legacy-edit",
            name="edit_file",
            arguments={"path": "example.py", "edits": [{}]},
            content="Applied 1 edit(s) to example.py.\n25| legacy changed",
        )
        provider = _SuccessfulCapturingProvider()
        context = ExecutionContext(provider=provider, workspace_root=tmp_path)
        context_token = set_context(context)
        try:
            await run_agent_loop(
                context=context,
                user_prompt="continue",
                tools=[],
                history=history,
            )
        finally:
            reset_context(context_token)

        folded_context = next(
            message.content
            for message in provider.messages[0]
            if isinstance(message, UserMessage)
            and "[Current file context: example.py]" in message.content
        )
        assert "25-| legacy changed" in folded_context
        assert "25~|" not in folded_context
        assert source_lines[0] not in folded_context

    async def test_none_policy_preserves_legacy_provider_rendering(
        self,
        tmp_path: Path,
    ) -> None:
        (tmp_path / "example.py").write_text("current\n")
        history = HistoryTree()
        history.append_user_prompt("earlier prompt")
        _append_tool_turn(
            history,
            call_id="stale-read",
            name="read_file",
            arguments={"path": "example.py"},
            content="1| old",
        )
        provider = _SuccessfulCapturingProvider()
        sink = _AttemptRecordingSink()
        context = ExecutionContext(
            provider=provider,
            event_sink=sink,
            workspace_root=tmp_path,
            context_budget_policy=None,
        )
        context_token = set_context(context)
        try:
            await run_agent_loop(
                context=context,
                user_prompt="finish",
                tools=[],
                history=history,
            )
        finally:
            reset_context(context_token)

        assert _result_content(provider.messages[0], "stale-read") == "1| old"
        assert sink.attempts[0].context.history_ledger is None
        assert "history_ledger" not in sink.attempts[0].context.to_json()

    async def test_retry_reuses_identical_projected_messages_and_ledger(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import thorn.core._loop as loop_module

        monkeypatch.setattr(
            loop_module,
            "_DEFAULT_RETRY_POLICY",
            RetryPolicy(
                base=0.001,
                cap=0.001,
                max_rate_limit_retries=1,
                max_transient_retries=1,
                retry_after_jitter=0.0,
            ),
        )
        (tmp_path / "example.py").write_text("current\n")
        history = HistoryTree()
        history.append_user_prompt("earlier prompt")
        _append_tool_turn(
            history,
            call_id="stale-read",
            name="read_file",
            arguments={"path": "example.py"},
            content="1| old",
        )
        provider = _RetryCapturingProvider()
        sink = _AttemptRecordingSink()
        context = ExecutionContext(
            provider=provider,
            event_sink=sink,
            workspace_root=tmp_path,
            context_budget_policy=DEFAULT_CONTEXT_BUDGET_POLICY,
        )
        context_token = set_context(context)
        try:
            result = await run_agent_loop(
                context=context,
                user_prompt="finish",
                tools=[],
                history=history,
            )
        finally:
            reset_context(context_token)

        assert result == "done"
        assert provider.payloads[0] == provider.payloads[1]
        assert len(sink.attempts) == 2
        first_context = sink.attempts[0].context.to_json()
        second_context = sink.attempts[1].context.to_json()
        assert sink.attempts[0].request_id == sink.attempts[1].request_id
        assert first_context["history_ledger"] == second_context["history_ledger"]
        assert first_context["history_ledger"]["disposition_counts"][
            "stale_file_observation"
        ] == 1

    async def test_prompt_trace_contains_context_ledger(
        self,
        tmp_path: Path,
    ) -> None:
        (tmp_path / "example.py").write_text("current\n")
        history = HistoryTree()
        history.append_user_prompt("earlier prompt")
        _append_tool_turn(
            history,
            call_id="stale-read",
            name="read_file",
            arguments={"path": "example.py"},
            content="1| old",
        )
        trace_path = tmp_path / "thorn.jsonl"
        context = ExecutionContext(
            provider=MockProvider(canned_responses=[[
                TextChunk(text="done"),
                FinishChunk(reason="stop"),
            ]]),
            workspace_root=tmp_path,
            prompt_trace_recorder=PromptTraceRecorder.for_trace_path(trace_path),
            context_budget_policy=DEFAULT_CONTEXT_BUDGET_POLICY,
        )
        context_token = set_context(context)
        try:
            await run_agent_loop(
                context=context,
                user_prompt="finish",
                tools=[],
                history=history,
            )
        finally:
            reset_context(context_token)

        sidecar_path = next(Path(f"{trace_path}.prompts").glob("*.json"))
        sidecar = json.loads(sidecar_path.read_text())
        ledger = sidecar["context"]["history_ledger"]
        assert ledger["resolved_budget"] == {
            "hard_prompt_tokens": None,
            "history_tokens": 12_000,
            "source": "fixed_default",
        }
        assert ledger["disposition_counts"]["stale_file_observation"] == 1
