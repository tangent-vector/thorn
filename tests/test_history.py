"""Tests for thorn.core._history — hierarchical history and compaction."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from thorn.core._context import ExecutionContext, set_context, reset_context
from thorn.core._history import (
    ABBREVIATED_ARG_VALUE_MAX_LEN,
    CHARS_PER_TOKEN,
    DEFAULT_HIGH_WATERMARK,
    DEFAULT_LOW_WATERMARK,
    DEFAULT_PROTECTED_TAIL_NODES,
    DEFAULT_PROTECTED_TAIL_TOOL_CALLS,
    LONG_CONTENT_THRESHOLD,
    TRUNCATED_PREFIX_CHARS,
    ArchiveMarkerNode,
    CollapseState,
    CompactionResult,
    DirectoryListCallNode,
    FileReadCallNode,
    HistoryNode,
    HistoryTree,
    HousekeepingNode,
    ToolCallNode,
    TurnNode,
    UserPromptNode,
    _abbreviate_arguments,
    _tool_call_summary,
    _truncate_content,
    estimate_tokens,
)
from thorn.core._loop import run_agent_loop
from thorn.core._messages import (
    AssistantMessage,
    Message,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)
from thorn.core._provider import (
    FinishChunk,
    MockProvider,
    TextChunk,
    ToolCallChunk,
    UsageChunk,
)
from thorn.core._func import wrap_function
from thorn.core._executor import ToolVenue


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tc(call_id: str, name: str, args: dict) -> ToolCall:
    return ToolCall(call_id=call_id, name=name, arguments=json.dumps(args))


def _result(call_id: str, content: str, is_error: bool = False) -> ToolResultMessage:
    return ToolResultMessage(call_id=call_id, content=content, is_error=is_error)


def _long_text(n_chars: int = LONG_CONTENT_THRESHOLD + 500) -> str:
    return "x" * n_chars


# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------


class TestEstimateTokens:
    def test_empty_string(self):
        assert estimate_tokens("") == 1

    def test_proportional(self):
        short = estimate_tokens("hello")
        long = estimate_tokens("hello world, this is a longer string")
        assert long > short


# ---------------------------------------------------------------------------
# Content truncation
# ---------------------------------------------------------------------------


class TestTruncateContent:
    def test_short_content_unchanged(self):
        assert _truncate_content("short") == "short"

    def test_long_content_truncated(self):
        long = "a" * 1000
        result = _truncate_content(long, prefix_chars=100)
        assert result.startswith("a" * 100)
        assert "900 more characters" in result


# ---------------------------------------------------------------------------
# Argument abbreviation
# ---------------------------------------------------------------------------


class TestAbbreviateArguments:
    def test_short_args_unchanged(self):
        args = json.dumps({"path": "foo.py", "mode": "r"})
        result = _abbreviate_arguments(args)
        assert json.loads(result) == {"path": "foo.py", "mode": "r"}

    def test_long_string_value_truncated(self):
        args = json.dumps({"path": "main.py", "content": "x" * 200})
        result = _abbreviate_arguments(args)
        parsed = json.loads(result)
        assert parsed["path"] == "main.py"
        assert len(parsed["content"]) < 200
        assert parsed["content"].endswith("...")

    def test_large_list_abbreviated(self):
        args = json.dumps({"edits": [{"old": "a", "new": "b"}] * 20})
        result = _abbreviate_arguments(args)
        parsed = json.loads(result)
        assert "20 items" in parsed["edits"]

    def test_small_list_preserved(self):
        args = json.dumps({"tags": [1, 2]})
        result = _abbreviate_arguments(args)
        parsed = json.loads(result)
        assert parsed["tags"] == [1, 2]

    def test_empty_args(self):
        assert _abbreviate_arguments("") == "{}"
        assert _abbreviate_arguments("{}") == "{}"

    def test_invalid_json_truncated(self):
        result = _abbreviate_arguments("not json at all " * 20)
        assert len(result) <= ABBREVIATED_ARG_VALUE_MAX_LEN * 2

    def test_result_is_valid_json(self):
        args = json.dumps({"a": "x" * 200, "b": list(range(50))})
        result = _abbreviate_arguments(args)
        json.loads(result)  # should not raise


# ---------------------------------------------------------------------------
# Tool call summaries
# ---------------------------------------------------------------------------


class TestToolCallSummary:
    def test_read_file(self):
        tc = _tc("c1", "read_file", {"path": "src/foo.py"})
        result = _result("c1", "line1\nline2\nline3")
        summary = _tool_call_summary(tc, result)
        assert "read_file" in summary
        assert "src/foo.py" in summary
        assert "3 lines" in summary

    def test_edit_file(self):
        tc = _tc("c1", "edit_file", {"path": "main.py", "edits": [{}, {}]})
        result = _result("c1", "Applied 2 edits.")
        summary = _tool_call_summary(tc, result)
        assert "edit_file" in summary
        assert "2 edit(s)" in summary

    def test_unknown_tool(self):
        tc = _tc("c1", "custom_tool", {"input": "some value"})
        result = _result("c1", "done")
        summary = _tool_call_summary(tc, result)
        assert "custom_tool" in summary

    def test_error_result(self):
        tc = _tc("c1", "read_file", {"path": "missing.txt"})
        result = _result("c1", "File not found", is_error=True)
        summary = _tool_call_summary(tc, result)
        assert "error" in summary

    def test_return_result(self):
        tc = _tc("c1", "return_result", {"value": 42})
        result = _result("c1", "ok")
        summary = _tool_call_summary(tc, result)
        assert summary == "return_result(...)"


# ---------------------------------------------------------------------------
# ToolCallNode
# ---------------------------------------------------------------------------


class TestToolCallNode:
    def test_expanded_rendering(self):
        tc = _tc("c1", "read_file", {"path": "x.py"})
        result = _result("c1", "file content here")
        node = ToolCallNode(tc, result)

        assert node.detail_collapsed is False
        assert node.render_tool_call() is tc
        assert node.render_result() is result

    def test_detail_collapsed_rendering(self):
        tc = _tc("c1", "read_file", {"path": "x.py"})
        result = _result("c1", "a" * 1000)
        node = ToolCallNode(tc, result)
        node.detail_collapsed = True

        rendered_result = node.render_result()
        assert rendered_result is not result
        assert "read_file" in rendered_result.content
        assert rendered_result.call_id == "c1"

    def test_detail_collapsed_abbreviates_arguments(self):
        big_args = {"path": "main.py", "content": "x" * 5000}
        tc = _tc("c1", "create_file", big_args)
        result = _result("c1", "ok")
        node = ToolCallNode(tc, result)
        node.detail_collapsed = True

        rendered_tc = node.render_tool_call()
        assert rendered_tc is not tc
        assert rendered_tc.call_id == "c1"
        assert rendered_tc.name == "create_file"
        parsed_args = json.loads(rendered_tc.arguments)
        assert parsed_args["path"] == "main.py"
        assert len(parsed_args["content"]) < 5000

    def test_expanded_does_not_abbreviate_arguments(self):
        big_args = {"path": "main.py", "content": "x" * 5000}
        tc = _tc("c1", "create_file", big_args)
        result = _result("c1", "ok")
        node = ToolCallNode(tc, result)

        rendered_tc = node.render_tool_call()
        assert rendered_tc is tc

    def test_savings_positive_for_large_result(self):
        tc = _tc("c1", "read_file", {"path": "big.py"})
        result = _result("c1", "a" * 5000)
        node = ToolCallNode(tc, result)
        assert node.savings_if_detail_collapsed() > 0
        assert node.is_collapsible

    def test_no_savings_for_tiny_result(self):
        tc = _tc("c1", "read_file", {"path": "x.py"})
        result = _result("c1", "ok")
        node = ToolCallNode(tc, result)
        savings = node.savings_if_detail_collapsed()
        assert savings >= 0

    def test_token_cost_changes_with_collapse(self):
        tc = _tc("c1", "read_file", {"path": "x.py"})
        result = _result("c1", "a" * 5000)
        node = ToolCallNode(tc, result)

        expanded_cost = node.token_cost()
        node.detail_collapsed = True
        collapsed_cost = node.token_cost()
        assert collapsed_cost < expanded_cost


# ---------------------------------------------------------------------------
# ToolCallNode subclasses
# ---------------------------------------------------------------------------


class TestToolCallNodeSubclasses:
    def test_file_read_is_subclass(self):
        assert issubclass(FileReadCallNode, ToolCallNode)

    def test_directory_list_is_subclass(self):
        assert issubclass(DirectoryListCallNode, ToolCallNode)

    def test_file_read_isinstance(self):
        tc = _tc("c1", "read_file", {"path": "x.py"})
        result = _result("c1", "file content")
        node = FileReadCallNode(tc, result)
        assert isinstance(node, ToolCallNode)
        assert isinstance(node, FileReadCallNode)
        assert not isinstance(node, DirectoryListCallNode)

    def test_directory_list_isinstance(self):
        tc = _tc("c1", "list_directory", {"path": "."})
        result = _result("c1", "file1\nfile2")
        node = DirectoryListCallNode(tc, result)
        assert isinstance(node, ToolCallNode)
        assert isinstance(node, DirectoryListCallNode)
        assert not isinstance(node, FileReadCallNode)

    def test_subclass_inherits_full_behavior(self):
        tc = _tc("c1", "read_file", {"path": "big.py"})
        result = _result("c1", "a" * 5000)
        node = FileReadCallNode(tc, result)

        assert node.summary() is not None
        assert node.expanded_token_cost() > 0
        assert node.savings_if_detail_collapsed() > 0
        assert node.is_collapsible

        node.detail_collapsed = True
        rendered = node.render_result()
        assert "read_file" in rendered.content

    def test_base_is_not_subclass_instance(self):
        tc = _tc("c1", "read_file", {"path": "x.py"})
        result = _result("c1", "content")
        node = ToolCallNode(tc, result)
        assert not isinstance(node, FileReadCallNode)
        assert not isinstance(node, DirectoryListCallNode)


# ---------------------------------------------------------------------------
# UserPromptNode
# ---------------------------------------------------------------------------


class TestUserPromptNode:
    def test_short_prompt_not_collapsible(self):
        node = UserPromptNode(UserMessage(content="fix the bug"))
        assert not node.is_long
        assert not node.is_collapsible

    def test_long_prompt_is_collapsible(self):
        node = UserPromptNode(UserMessage(content=_long_text()))
        assert node.is_long
        assert node.is_collapsible

    def test_expanded_render(self):
        msg = UserMessage(content="hello")
        node = UserPromptNode(msg)
        rendered = node.render()
        assert len(rendered) == 1
        assert rendered[0] is msg

    def test_collapsed_render(self):
        long = _long_text()
        node = UserPromptNode(UserMessage(content=long))
        node.collapse_state = CollapseState.COLLAPSED
        rendered = node.render()
        assert len(rendered) == 1
        assert len(rendered[0].content) < len(long)

    def test_token_cost_lower_when_collapsed(self):
        node = UserPromptNode(UserMessage(content=_long_text()))
        expanded_cost = node.token_cost()
        node.collapse_state = CollapseState.COLLAPSED
        collapsed_cost = node.token_cost()
        assert collapsed_cost < expanded_cost


# ---------------------------------------------------------------------------
# TurnNode
# ---------------------------------------------------------------------------


class TestTurnNode:
    def test_text_only_render(self):
        node = TurnNode(assistant_content="hello", tool_call_nodes=[])
        rendered = node.render()
        assert len(rendered) == 1
        assert isinstance(rendered[0], AssistantMessage)
        assert rendered[0].content == "hello"
        assert rendered[0].tool_calls == []

    def test_with_tool_calls_render(self):
        tc = _tc("c1", "read_file", {"path": "x.py"})
        result = _result("c1", "content")
        tcn = ToolCallNode(tc, result)
        node = TurnNode(
            assistant_content="let me read that",
            tool_call_nodes=[tcn],
        )
        rendered = node.render()
        assert len(rendered) == 2
        assert isinstance(rendered[0], AssistantMessage)
        assert len(rendered[0].tool_calls) == 1
        assert isinstance(rendered[1], ToolResultMessage)

    def test_collapsed_render(self):
        tc = _tc("c1", "read_file", {"path": "x.py"})
        result = _result("c1", "content")
        tcn = ToolCallNode(tc, result)
        node = TurnNode(
            assistant_content="checking the file",
            tool_call_nodes=[tcn],
        )
        node.collapse_state = CollapseState.COLLAPSED
        rendered = node.render()
        assert len(rendered) == 1
        assert isinstance(rendered[0], AssistantMessage)
        assert rendered[0].tool_calls == []
        assert "read_file" in rendered[0].content

    def test_detail_collapsed_tool_calls(self):
        """When tool calls are detail-collapsed, the turn still renders
        them but with summary content."""
        tc = _tc("c1", "read_file", {"path": "x.py"})
        result = _result("c1", "a" * 5000)
        tcn = ToolCallNode(tc, result)
        tcn.detail_collapsed = True

        node = TurnNode(assistant_content="reading", tool_call_nodes=[tcn])
        rendered = node.render()
        assert len(rendered) == 2
        assert len(rendered[0].tool_calls) == 1
        assert len(rendered[1].content) < 5000

    def test_savings_from_collapse(self):
        tc = _tc("c1", "read_file", {"path": "x.py"})
        result = _result("c1", "a" * 5000)
        tcn = ToolCallNode(tc, result)
        node = TurnNode(assistant_content="reading", tool_call_nodes=[tcn])
        assert node.savings_if_collapsed() > 0

    def test_summary_includes_tool_info(self):
        tc = _tc("c1", "edit_file", {"path": "main.py", "edits": [{}]})
        result = _result("c1", "Applied 1 edit.")
        tcn = ToolCallNode(tc, result)
        node = TurnNode(assistant_content="editing", tool_call_nodes=[tcn])
        summary = node.summary()
        assert "edit_file" in summary


# ---------------------------------------------------------------------------
# HistoryTree — basic operations
# ---------------------------------------------------------------------------


class TestHistoryTree:
    def test_empty_tree(self):
        tree = HistoryTree()
        assert tree.nodes == []
        assert tree.render() == []
        assert tree.estimated_tokens() == 0

    def test_append_user_prompt(self):
        tree = HistoryTree()
        node = tree.append_user_prompt("hello")
        assert isinstance(node, UserPromptNode)
        assert len(tree.nodes) == 1

    def test_append_turn(self):
        tree = HistoryTree()
        msg = AssistantMessage(content="hi")
        node = tree.append_turn(msg, [])
        assert isinstance(node, TurnNode)
        assert node.assistant_content == "hi"

    def test_append_turn_with_tool_calls(self):
        tree = HistoryTree()
        tc = _tc("c1", "read_file", {"path": "x.py"})
        msg = AssistantMessage(content="reading", tool_calls=[tc])
        result = _result("c1", "file content")
        node = tree.append_turn(msg, [result])
        assert len(node.tool_call_nodes) == 1
        assert node.tool_call_nodes[0].tool_call is tc

    def test_render_produces_valid_sequence(self):
        tree = HistoryTree()
        tree.append_user_prompt("question")
        tc = _tc("c1", "read_file", {"path": "x.py"})
        tree.append_turn(
            AssistantMessage(content="reading", tool_calls=[tc]),
            [_result("c1", "content")],
        )
        tree.append_turn(AssistantMessage(content="done"), [])

        rendered = tree.render()
        assert isinstance(rendered[0], UserMessage)
        assert isinstance(rendered[1], AssistantMessage)
        assert isinstance(rendered[2], ToolResultMessage)
        assert isinstance(rendered[3], AssistantMessage)

    def test_estimated_tokens_grows_with_content(self):
        tree = HistoryTree()
        tree.append_user_prompt("short")
        small = tree.estimated_tokens()

        tree.append_user_prompt("a" * 10000)
        large = tree.estimated_tokens()
        assert large > small


# ---------------------------------------------------------------------------
# HistoryTree — call_node_classes in append_turn
# ---------------------------------------------------------------------------


class TestAppendTurnCallNodeClasses:
    def test_uses_registered_subclass(self):
        tree = HistoryTree()
        tc = _tc("c1", "read_file", {"path": "x.py"})
        msg = AssistantMessage(content="reading", tool_calls=[tc])
        result = _result("c1", "file content")

        node = tree.append_turn(
            msg, [result],
            call_node_classes={"c1": FileReadCallNode},
        )

        assert len(node.tool_call_nodes) == 1
        assert isinstance(node.tool_call_nodes[0], FileReadCallNode)

    def test_falls_back_to_base_without_mapping(self):
        tree = HistoryTree()
        tc = _tc("c1", "read_file", {"path": "x.py"})
        msg = AssistantMessage(content="reading", tool_calls=[tc])
        result = _result("c1", "file content")

        node = tree.append_turn(msg, [result])

        assert len(node.tool_call_nodes) == 1
        assert type(node.tool_call_nodes[0]) is ToolCallNode

    def test_falls_back_for_unregistered_call_id(self):
        tree = HistoryTree()
        tc1 = _tc("c1", "read_file", {"path": "x.py"})
        tc2 = _tc("c2", "custom_tool", {"arg": "val"})
        msg = AssistantMessage(content="working", tool_calls=[tc1, tc2])
        r1 = _result("c1", "file content")
        r2 = _result("c2", "tool result")

        node = tree.append_turn(
            msg, [r1, r2],
            call_node_classes={"c1": FileReadCallNode},
        )

        assert len(node.tool_call_nodes) == 2
        assert isinstance(node.tool_call_nodes[0], FileReadCallNode)
        assert type(node.tool_call_nodes[1]) is ToolCallNode

    def test_multiple_different_subclasses(self):
        tree = HistoryTree()
        tc1 = _tc("c1", "read_file", {"path": "x.py"})
        tc2 = _tc("c2", "list_directory", {"path": "."})
        msg = AssistantMessage(content="exploring", tool_calls=[tc1, tc2])
        r1 = _result("c1", "file content")
        r2 = _result("c2", "dir listing")

        node = tree.append_turn(
            msg, [r1, r2],
            call_node_classes={
                "c1": FileReadCallNode,
                "c2": DirectoryListCallNode,
            },
        )

        assert len(node.tool_call_nodes) == 2
        assert isinstance(node.tool_call_nodes[0], FileReadCallNode)
        assert isinstance(node.tool_call_nodes[1], DirectoryListCallNode)

    def test_isinstance_on_recorded_history(self):
        """Full roundtrip: nodes in the tree are identifiable via isinstance."""
        tree = HistoryTree()
        tree.append_user_prompt("read some files")

        tc = _tc("c1", "read_file", {"path": "main.py"})
        msg = AssistantMessage(content="reading", tool_calls=[tc])
        result = _result("c1", "def main(): pass")
        tree.append_turn(
            msg, [result],
            call_node_classes={"c1": FileReadCallNode},
        )

        turn = tree.nodes[1]
        assert isinstance(turn, TurnNode)
        file_reads = [
            tcn for tcn in turn.tool_call_nodes
            if isinstance(tcn, FileReadCallNode)
        ]
        assert len(file_reads) == 1
        assert file_reads[0].tool_call.name == "read_file"


# ---------------------------------------------------------------------------
# HistoryTree — compaction
# ---------------------------------------------------------------------------


class TestCompaction:
    def test_no_compaction_when_under_budget(self):
        tree = HistoryTree()
        tree.append_user_prompt("hello")
        tree.append_turn(AssistantMessage(content="hi"), [])

        result = tree.compact(context_budget=100000, overhead_tokens=0)
        assert result.nodes_collapsed == 0
        assert result.estimated_savings == 0

    def test_compacts_old_tool_calls_first(self):
        """Detail-collapses tool call results before collapsing turns."""
        tree = HistoryTree()
        tree.append_user_prompt("start")

        for i in range(5):
            tc = _tc(f"c{i}", "read_file", {"path": f"file{i}.py"})
            tree.append_turn(
                AssistantMessage(content=f"reading file {i}", tool_calls=[tc]),
                [_result(f"c{i}", "a" * 2000)],
            )

        tree.append_user_prompt("now what?")
        tree.append_turn(AssistantMessage(content="all done"), [])

        tokens_before = tree.estimated_tokens()
        result = tree.compact(
            context_budget=tokens_before,
            low_watermark=0.5,
            overhead_tokens=0,
            protected_tail_nodes=2,
            protected_tail_tool_calls=0,
        )

        assert result.tool_calls_detail_collapsed > 0 or result.nodes_collapsed > 0
        assert result.estimated_savings > 0
        assert tree.estimated_tokens() < tokens_before

    def test_collapses_oldest_turns_first(self):
        """Oldest nodes (highest age, lowest effective salience) are
        collapsed before newer ones."""
        tree = HistoryTree()
        tree.append_user_prompt("start")

        for i in range(10):
            tree.append_turn(
                AssistantMessage(content=f"turn {i} " + "x" * 500),
                [],
            )

        tree.append_user_prompt("continue")
        tree.append_turn(AssistantMessage(content="latest"), [])

        result = tree.compact(
            context_budget=tree.estimated_tokens(),
            low_watermark=0.3,
            overhead_tokens=0,
            protected_tail_nodes=2,
            protected_tail_tool_calls=0,
        )

        assert result.nodes_collapsed > 0
        # The earliest turns should be collapsed
        assert tree.nodes[1].collapse_state == CollapseState.COLLAPSED

    def test_protects_tail_nodes(self):
        """The tail N nodes should never be collapsed."""
        tree = HistoryTree()
        tree.append_user_prompt("start")
        tree.append_turn(
            AssistantMessage(content="x" * 2000), [],
        )
        tree.append_user_prompt("question")
        tree.append_turn(
            AssistantMessage(content="y" * 2000), [],
        )

        tree.compact(
            context_budget=1,
            low_watermark=0.1,
            overhead_tokens=0,
            protected_tail_nodes=2,
            protected_tail_tool_calls=0,
        )

        # Last two nodes should still be expanded
        assert tree.nodes[-1].collapse_state == CollapseState.EXPANDED
        assert tree.nodes[-2].collapse_state == CollapseState.EXPANDED

    def test_protects_most_recent_user_prompt(self):
        """The most recent UserPromptNode is always protected."""
        tree = HistoryTree()
        tree.append_user_prompt("initial task " + "a" * 3000)

        for i in range(5):
            tree.append_turn(
                AssistantMessage(content=f"turn {i} " + "b" * 500),
                [],
            )

        tree.compact(
            context_budget=1,
            low_watermark=0.1,
            overhead_tokens=0,
            protected_tail_nodes=2,
            protected_tail_tool_calls=0,
        )

        # The initial user prompt is the most recent (and only)
        # UserPromptNode, so it should be protected even though
        # it falls outside the tail window.
        assert tree.nodes[0].collapse_state == CollapseState.EXPANDED

    def test_compaction_result_accuracy(self):
        """CompactionResult.tokens_after matches actual tree state."""
        tree = HistoryTree()
        tree.append_user_prompt("start")
        for i in range(5):
            tc = _tc(f"c{i}", "read_file", {"path": f"f{i}.py"})
            tree.append_turn(
                AssistantMessage(content="r", tool_calls=[tc]),
                [_result(f"c{i}", "a" * 3000)],
            )
        tree.append_user_prompt("next")
        tree.append_turn(AssistantMessage(content="done"), [])

        result = tree.compact(
            context_budget=tree.estimated_tokens(),
            low_watermark=0.3,
            overhead_tokens=0,
            protected_tail_nodes=2,
            protected_tail_tool_calls=0,
        )

        assert result.tokens_after == tree.estimated_tokens()

    def test_idempotent_when_already_compact(self):
        """Running compact twice without new content is a no-op."""
        tree = HistoryTree()
        tree.append_user_prompt("start")
        for i in range(3):
            tc = _tc(f"c{i}", "read_file", {"path": f"f{i}.py"})
            tree.append_turn(
                AssistantMessage(content="r", tool_calls=[tc]),
                [_result(f"c{i}", "a" * 2000)],
            )
        tree.append_user_prompt("next")
        tree.append_turn(AssistantMessage(content="done"), [])

        budget = tree.estimated_tokens()
        tree.compact(
            context_budget=budget, low_watermark=0.3, overhead_tokens=0,
            protected_tail_nodes=2, protected_tail_tool_calls=0,
        )
        tokens_after_first = tree.estimated_tokens()

        result2 = tree.compact(
            context_budget=budget, low_watermark=0.3, overhead_tokens=0,
            protected_tail_nodes=2, protected_tail_tool_calls=0,
        )
        assert tree.estimated_tokens() == tokens_after_first
        assert result2.nodes_collapsed == 0
        assert result2.tool_calls_detail_collapsed == 0

    def test_savings_accurate_when_tool_calls_and_turns_both_collapse(self):
        """When both tool-call detail-collapse and full turn collapse
        happen in the same pass, the reported savings must match the
        actual token reduction (no over-counting)."""
        tree = HistoryTree()
        tree.append_user_prompt("start")

        for i in range(4):
            tc = _tc(f"c{i}", "read_file", {"path": f"f{i}.py"})
            tree.append_turn(
                AssistantMessage(content=f"step {i}", tool_calls=[tc]),
                [_result(f"c{i}", "a" * 3000)],
            )

        tree.append_user_prompt("next")
        tree.append_turn(AssistantMessage(content="done"), [])

        tokens_before = tree.estimated_tokens()
        result = tree.compact(
            context_budget=tokens_before,
            low_watermark=0.2,
            overhead_tokens=0,
            protected_tail_nodes=2,
            protected_tail_tool_calls=0,
        )

        actual_after = tree.estimated_tokens()
        assert result.tokens_after == actual_after
        assert result.estimated_savings == tokens_before - actual_after

    def test_render_after_compaction_is_valid(self):
        """Rendered messages after compaction form a valid sequence."""
        tree = HistoryTree()
        tree.append_user_prompt("start")
        for i in range(5):
            tc = _tc(f"c{i}", "read_file", {"path": f"f{i}.py"})
            tree.append_turn(
                AssistantMessage(content=f"step {i}", tool_calls=[tc]),
                [_result(f"c{i}", "a" * 2000)],
            )
        tree.append_user_prompt("continue")
        tree.append_turn(AssistantMessage(content="ok"), [])

        tree.compact(
            context_budget=tree.estimated_tokens(),
            low_watermark=0.3,
            overhead_tokens=0,
            protected_tail_nodes=2,
            protected_tail_tool_calls=0,
        )

        rendered = tree.render()
        # Verify role alternation is valid
        for i, msg in enumerate(rendered):
            if isinstance(msg, ToolResultMessage):
                assert isinstance(rendered[i - 1], AssistantMessage) or isinstance(
                    rendered[i - 1], ToolResultMessage
                )
            elif isinstance(msg, AssistantMessage) and msg.tool_calls:
                # All tool calls must have results following
                expected_ids = {tc.call_id for tc in msg.tool_calls}
                actual_ids = set()
                for j in range(i + 1, len(rendered)):
                    if isinstance(rendered[j], ToolResultMessage):
                        actual_ids.add(rendered[j].call_id)
                    else:
                        break
                assert expected_ids == actual_ids


# ---------------------------------------------------------------------------
# HistoryTree — protected-tail guardrails
# ---------------------------------------------------------------------------


def _build_tool_heavy_tree(
    num_turns: int = 10,
    tool_calls_per_turn: int = 2,
    result_size: int = 2000,
) -> HistoryTree:
    """Build a tree with a user prompt followed by tool-heavy turns.

    Returns a tree with (1 + num_turns) nodes: one ``UserPromptNode``
    at index 0, then *num_turns* ``TurnNode``s each containing
    *tool_calls_per_turn* tool calls with *result_size*-char results.
    """
    tree = HistoryTree()
    tree.append_user_prompt("initial task")
    for t in range(num_turns):
        tcs = []
        results = []
        for c in range(tool_calls_per_turn):
            cid = f"t{t}_c{c}"
            tcs.append(_tc(cid, "read_file", {"path": f"file_{t}_{c}.py"}))
            results.append(_result(cid, "x" * result_size))
        tree.append_turn(
            AssistantMessage(
                content=f"turn {t}",
                tool_calls=tcs,
            ),
            results,
        )
    return tree


class TestProtectedTailGuardrails:
    """Tests for the protected-tail guardrails in compact()."""

    def test_default_tail_nodes_protected(self):
        """With default N=4, the last 4 nodes must never be collapsed
        even under extreme compaction pressure."""
        tree = HistoryTree()
        tree.append_user_prompt("start")
        for i in range(8):
            tree.append_turn(
                AssistantMessage(content=f"turn {i} " + "y" * 500), [],
            )

        tree.compact(
            context_budget=1, low_watermark=0.1, overhead_tokens=0,
            protected_tail_tool_calls=0,
        )

        n = len(tree.nodes)
        for i in range(n - DEFAULT_PROTECTED_TAIL_NODES, n):
            assert tree.nodes[i].collapse_state == CollapseState.EXPANDED

    def test_nodes_outside_tail_can_be_collapsed(self):
        """Nodes older than the tail window are eligible for collapse."""
        tree = HistoryTree()
        tree.append_user_prompt("start")
        for i in range(8):
            tree.append_turn(
                AssistantMessage(content=f"turn {i} " + "y" * 500), [],
            )

        tree.compact(
            context_budget=1, low_watermark=0.1, overhead_tokens=0,
            protected_tail_tool_calls=0,
        )

        collapsed_count = sum(
            1 for node in tree.nodes
            if isinstance(node, TurnNode)
            and node.collapse_state == CollapseState.COLLAPSED
        )
        assert collapsed_count > 0

    def test_most_recent_user_prompt_protected_outside_tail(self):
        """The most recent UserPromptNode is protected even if it falls
        outside the tail_nodes window."""
        tree = HistoryTree()
        tree.append_user_prompt("important task " + "a" * 3000)
        for i in range(8):
            tree.append_turn(
                AssistantMessage(content=f"turn {i} " + "b" * 500), [],
            )

        tree.compact(
            context_budget=1, low_watermark=0.1, overhead_tokens=0,
            protected_tail_nodes=4, protected_tail_tool_calls=0,
        )

        assert tree.nodes[0].collapse_state == CollapseState.EXPANDED

    def test_tool_call_protection_extends_beyond_tail(self):
        """Turns containing recent tool calls are protected even when
        they fall outside the tail_nodes window."""
        tree = _build_tool_heavy_tree(num_turns=8, tool_calls_per_turn=2)
        # 9 nodes: user(0), turn(1)..turn(8)
        # tail_nodes=2 protects {7,8}
        # tail_tool_calls=6 needs 6 tool calls (2 per turn):
        #   turn 8 → 2 (remaining 4), turn 7 → 2 (remaining 2),
        #   turn 6 → 2 (remaining 0)
        # So turns 6,7,8 are protected by tool calls.

        tree.compact(
            context_budget=1, low_watermark=0.1, overhead_tokens=0,
            protected_tail_nodes=2, protected_tail_tool_calls=6,
        )

        for i in [6, 7, 8]:
            assert tree.nodes[i].collapse_state == CollapseState.EXPANDED, (
                f"node {i} should be protected by tool-call guardrail"
            )

    def test_tool_call_protection_counts_across_turns(self):
        """The M-tool-call budget counts across multiple turns, not
        per-turn.  A turn with many tool calls consumes more budget."""
        tree = HistoryTree()
        tree.append_user_prompt("go")
        # Turn 0: 5 tool calls
        tcs = [_tc(f"t0_c{c}", "read_file", {"path": f"f{c}.py"}) for c in range(5)]
        results = [_result(f"t0_c{c}", "x" * 1000) for c in range(5)]
        tree.append_turn(
            AssistantMessage(content="batch read", tool_calls=tcs),
            results,
        )
        # Turn 1: 1 tool call
        tree.append_turn(
            AssistantMessage(
                content="one more",
                tool_calls=[_tc("t1_c0", "read_file", {"path": "extra.py"})],
            ),
            [_result("t1_c0", "x" * 1000)],
        )
        # Turn 2: text-only (no tool calls)
        tree.append_turn(AssistantMessage(content="done " + "z" * 500), [])

        # tail_tool_calls=4: turn 2 has 0 tc, turn 1 has 1 tc
        # (remaining 3), turn 0 has 5 tc (remaining -2) → turn 0 protected
        tree.compact(
            context_budget=1, low_watermark=0.1, overhead_tokens=0,
            protected_tail_nodes=1, protected_tail_tool_calls=4,
        )

        # Turn 0 (index 1) is protected because it contains some of
        # the 4 most recent tool calls.
        assert tree.nodes[1].collapse_state == CollapseState.EXPANDED

    def test_custom_small_guardrails(self):
        """Passing smaller guardrail values allows more aggressive
        compaction."""
        tree = _build_tool_heavy_tree(num_turns=6, tool_calls_per_turn=1)

        tree.compact(
            context_budget=1, low_watermark=0.1, overhead_tokens=0,
            protected_tail_nodes=1, protected_tail_tool_calls=1,
        )

        collapsed = [
            i for i, node in enumerate(tree.nodes)
            if isinstance(node, TurnNode)
            and node.collapse_state == CollapseState.COLLAPSED
        ]
        assert len(collapsed) >= 3, (
            "with tight guardrails, most old turns should be collapsed"
        )

    def test_custom_large_guardrails(self):
        """Passing larger guardrail values protects more nodes."""
        tree = _build_tool_heavy_tree(num_turns=6, tool_calls_per_turn=1)

        tree.compact(
            context_budget=1, low_watermark=0.1, overhead_tokens=0,
            protected_tail_nodes=10, protected_tail_tool_calls=10,
        )

        collapsed = [
            i for i, node in enumerate(tree.nodes)
            if isinstance(node, TurnNode)
            and node.collapse_state == CollapseState.COLLAPSED
        ]
        assert len(collapsed) == 0

    def test_detail_collapse_skipped_for_protected_tool_calls(self):
        """Tool call detail-collapse is also skipped for turns in the
        protected set (the entire turn is off-limits, including its
        individual tool call nodes)."""
        tree = _build_tool_heavy_tree(num_turns=4, tool_calls_per_turn=2)
        # 5 nodes: user(0), turn(1)..turn(4)
        # tail_nodes=4 + tail_tool_calls=8 → all turns protected

        tree.compact(
            context_budget=1, low_watermark=0.1, overhead_tokens=0,
        )

        for node in tree.nodes:
            if isinstance(node, TurnNode):
                for tcn in node.tool_call_nodes:
                    assert not tcn.detail_collapsed

    def test_zero_guardrails_allow_full_compaction(self):
        """Setting both guardrails to 0 restores the most aggressive
        compaction possible (only the most recent UserPromptNode is
        still protected by rule 2)."""
        tree = HistoryTree()
        tree.append_user_prompt("start")
        for i in range(5):
            tree.append_turn(
                AssistantMessage(content=f"turn {i} " + "y" * 500), [],
            )
        tree.append_user_prompt("latest " + "z" * 3000)

        tree.compact(
            context_budget=1, low_watermark=0.1, overhead_tokens=0,
            protected_tail_nodes=0, protected_tail_tool_calls=0,
        )

        # All turns should be collapsed
        for i, node in enumerate(tree.nodes):
            if isinstance(node, TurnNode):
                assert node.collapse_state == CollapseState.COLLAPSED

        # The most recent UserPromptNode ("latest ...") is still protected
        assert tree.nodes[-1].collapse_state == CollapseState.EXPANDED

    def test_empty_tree_is_fine(self):
        """Guardrail logic handles an empty tree without errors."""
        tree = HistoryTree()
        result = tree.compact(context_budget=1000, overhead_tokens=0)
        assert result.nodes_collapsed == 0

    def test_protected_indices_returns_correct_set(self):
        """Directly verify _protected_indices for a known layout."""
        tree = _build_tool_heavy_tree(num_turns=6, tool_calls_per_turn=2)
        # 7 nodes: user(0), turn(1)..turn(6)
        # 12 tool calls total (2 per turn)
        protected = tree._protected_indices(
            tail_nodes=3,
            tail_tool_calls=5,
        )

        # tail_nodes=3 → {4, 5, 6}
        assert {4, 5, 6}.issubset(protected)

        # Most recent UserPromptNode → {0}
        assert 0 in protected

        # tail_tool_calls=5: walk backward
        #   turn 6 → 2 tc (remaining 3), turn 5 → 2 tc (remaining 1),
        #   turn 4 → 2 tc (remaining -1)
        # → turns 4, 5, 6 protected (already in tail)
        # So turns 1, 2, 3 should NOT be protected
        assert 1 not in protected
        assert 2 not in protected
        assert 3 not in protected


# ---------------------------------------------------------------------------
# Integration: compaction trigger in agent loop
# ---------------------------------------------------------------------------


def _text_response(text: str):
    return [TextChunk(text=text), FinishChunk(reason="stop")]


def _tool_call_response(call_id: str, name: str, arguments: str):
    return [
        ToolCallChunk(call_id=call_id, name=name, arguments=arguments),
        FinishChunk(reason="tool_calls"),
    ]


class TestEffectiveContextWindow:
    def test_both_set_takes_min(self):
        from thorn import _effective_context_window
        assert _effective_context_window(128000, 32000) == 32000
        assert _effective_context_window(32000, 128000) == 32000

    def test_only_provider_set(self):
        from thorn import _effective_context_window
        assert _effective_context_window(128000, None) == 128000

    def test_only_user_set(self):
        from thorn import _effective_context_window
        assert _effective_context_window(None, 32000) == 32000

    def test_neither_set(self):
        from thorn import _effective_context_window
        assert _effective_context_window(None, None) is None


class TestProviderContextWindow:
    def test_mock_provider_default_none(self):
        provider = MockProvider()
        assert provider.context_window is None

    def test_mock_provider_configurable(self):
        provider = MockProvider(context_window=128000)
        assert provider.context_window == 128000


class TestCompactionIntegration:
    async def test_compaction_triggers_on_high_usage(self):
        """When prompt_tokens exceed the high watermark, compaction runs."""
        status_messages: list[str] = []

        from thorn.core._context import EventSink, Scope, NullEventSink

        class CaptureSink(NullEventSink):
            async def on_status(self, message, scope=None):
                status_messages.append(message)

        async def big_tool() -> str:
            """Return lots of content."""
            return "a" * 10000

        tool = wrap_function(big_tool, venue=ToolVenue.SANDBOX)

        # Simulate: first call returns tool call, second returns text.
        # The usage on the tool-call round reports high prompt_tokens.
        provider = MockProvider(canned_responses=[
            [
                ToolCallChunk(call_id="c1", name="big_tool", arguments="{}"),
                UsageChunk(prompt_tokens=9000, completion_tokens=100, total_tokens=9100),
                FinishChunk(reason="tool_calls"),
            ],
            [
                TextChunk(text="done"),
                UsageChunk(prompt_tokens=9500, completion_tokens=50, total_tokens=9550),
                FinishChunk(reason="stop"),
            ],
        ])
        ctx = ExecutionContext(
            provider=provider,
            event_sink=CaptureSink(),
            context_window=10000,
        )

        tree = HistoryTree()
        # Pre-populate with enough old history that some falls outside
        # the default protected-tail window and is eligible for compaction.
        for i in range(10):
            tree.append_user_prompt(f"old prompt {i}")
            tc = _tc(f"old{i}", "read_file", {"path": f"old{i}.py"})
            tree.append_turn(
                AssistantMessage(content=f"old reply {i}", tool_calls=[tc]),
                [_result(f"old{i}", "b" * 3000)],
            )

        result = await run_agent_loop(
            context=ctx,
            user_prompt="do something",
            tools=[tool],
            history=tree,
        )
        assert result == "done"

        compaction_msgs = [m for m in status_messages if "compaction" in m]
        assert len(compaction_msgs) > 0

    async def test_no_compaction_without_context_window(self):
        """When context_window is None, compaction never triggers."""
        status_messages: list[str] = []

        from thorn.core._context import NullEventSink

        class CaptureSink(NullEventSink):
            async def on_status(self, message, scope=None):
                status_messages.append(message)

        provider = MockProvider(canned_responses=[
            [
                TextChunk(text="hi"),
                UsageChunk(prompt_tokens=999999, completion_tokens=100, total_tokens=1000099),
                FinishChunk(reason="stop"),
            ],
        ])
        ctx = ExecutionContext(
            provider=provider,
            event_sink=CaptureSink(),
            context_window=None,
        )

        result = await run_agent_loop(
            context=ctx,
            user_prompt="hello",
            tools=[],
        )
        assert result == "hi"
        assert not any("compaction" in m for m in status_messages)

    async def test_agent_compaction_with_context_window(self):
        """Agent.prompt() triggers compaction when context_window is set."""
        from thorn.core._agent import Agent

        status_messages: list[str] = []

        from thorn.core._context import NullEventSink

        class CaptureSink(NullEventSink):
            async def on_status(self, message, scope=None):
                status_messages.append(message)

        async def big_read() -> str:
            """Return lots."""
            return "z" * 8000

        class BigReader(Agent):
            tools = [wrap_function(big_read, venue=ToolVenue.SANDBOX)]

        responses = []
        for i in range(12):
            responses.append([
                ToolCallChunk(call_id=f"c{i}", name="big_read", arguments="{}"),
                UsageChunk(
                    prompt_tokens=3000 * (i + 1),
                    completion_tokens=100,
                    total_tokens=3000 * (i + 1) + 100,
                ),
                FinishChunk(reason="tool_calls"),
            ])
        responses.append([
            TextChunk(text="all done"),
            UsageChunk(prompt_tokens=5000, completion_tokens=50, total_tokens=5050),
            FinishChunk(reason="stop"),
        ])

        provider = MockProvider(canned_responses=responses)
        ctx = ExecutionContext(
            provider=provider,
            event_sink=CaptureSink(),
            context_window=30000,
        )
        token = set_context(ctx)
        try:
            agent = BigReader()
            result = await agent.prompt("read everything")
            assert result == "all done"

            compaction_msgs = [m for m in status_messages if "compaction" in m]
            assert len(compaction_msgs) > 0
        finally:
            reset_context(token)


# ---------------------------------------------------------------------------
# Integration: call_node_class through agent loop
# ---------------------------------------------------------------------------


class TestCallNodeClassIntegration:
    async def test_tool_with_call_node_class_records_subclass_in_history(self):
        """When a tool has call_node_class set, the agent loop records the
        correct subclass in the history tree."""
        async def my_reader(path: str) -> str:
            """Read a file."""
            return "file contents here"

        my_reader._thorn_call_node_class = FileReadCallNode  # type: ignore[attr-defined]

        wrapped = wrap_function(my_reader, venue=ToolVenue.SANDBOX)
        provider = MockProvider(canned_responses=[
            _tool_call_response("c1", "my_reader", '{"path": "x.py"}'),
            _text_response("done reading"),
        ])
        ctx = ExecutionContext(provider=provider)
        history = HistoryTree()

        result = await run_agent_loop(
            context=ctx,
            user_prompt="read the file",
            tools=[wrapped],
            history=history,
        )
        assert result == "done reading"

        turn_nodes = [n for n in history.nodes if isinstance(n, TurnNode)]
        assert len(turn_nodes) == 2  # tool-call turn + final text turn
        tool_turn = turn_nodes[0]
        assert len(tool_turn.tool_call_nodes) == 1
        assert isinstance(tool_turn.tool_call_nodes[0], FileReadCallNode)

    async def test_tool_without_call_node_class_records_base(self):
        """Tools without call_node_class use the base ToolCallNode."""
        async def plain_tool() -> str:
            """Do something."""
            return "result"

        wrapped = wrap_function(plain_tool, venue=ToolVenue.SANDBOX)
        provider = MockProvider(canned_responses=[
            _tool_call_response("c1", "plain_tool", "{}"),
            _text_response("finished"),
        ])
        ctx = ExecutionContext(provider=provider)
        history = HistoryTree()

        await run_agent_loop(
            context=ctx,
            user_prompt="go",
            tools=[wrapped],
            history=history,
        )

        turn_nodes = [n for n in history.nodes if isinstance(n, TurnNode)]
        tool_turn = turn_nodes[0]
        assert len(tool_turn.tool_call_nodes) == 1
        assert type(tool_turn.tool_call_nodes[0]) is ToolCallNode

    async def test_mixed_tools_record_correct_types(self):
        """Mix of tools with and without call_node_class."""
        async def reader(path: str) -> str:
            """Read."""
            return "content"

        reader._thorn_call_node_class = FileReadCallNode  # type: ignore[attr-defined]

        async def lister(path: str) -> str:
            """List."""
            return "a\nb\nc"

        lister._thorn_call_node_class = DirectoryListCallNode  # type: ignore[attr-defined]

        async def plain() -> str:
            """Plain."""
            return "ok"

        provider = MockProvider(canned_responses=[
            [
                ToolCallChunk(call_id="c1", name="reader", arguments='{"path": "x.py"}'),
                ToolCallChunk(call_id="c2", name="lister", arguments='{"path": "."}'),
                ToolCallChunk(call_id="c3", name="plain", arguments="{}"),
                FinishChunk(reason="tool_calls"),
            ],
            _text_response("all done"),
        ])
        ctx = ExecutionContext(provider=provider)
        history = HistoryTree()

        await run_agent_loop(
            context=ctx,
            user_prompt="explore",
            tools=[wrap_function(reader, venue=ToolVenue.SANDBOX), wrap_function(lister, venue=ToolVenue.SANDBOX), wrap_function(plain, venue=ToolVenue.SANDBOX)],
            history=history,
        )

        turn_nodes = [n for n in history.nodes if isinstance(n, TurnNode)]
        tool_turn = turn_nodes[0]
        assert len(tool_turn.tool_call_nodes) == 3
        assert isinstance(tool_turn.tool_call_nodes[0], FileReadCallNode)
        assert isinstance(tool_turn.tool_call_nodes[1], DirectoryListCallNode)
        assert type(tool_turn.tool_call_nodes[2]) is ToolCallNode


# ---------------------------------------------------------------------------
# HistoryNode base class
# ---------------------------------------------------------------------------


class TestHistoryNodeHierarchy:
    def test_user_prompt_is_history_node(self):
        node = UserPromptNode(UserMessage(content="hi"))
        assert isinstance(node, HistoryNode)

    def test_turn_is_history_node(self):
        node = TurnNode(assistant_content="hi", tool_call_nodes=[])
        assert isinstance(node, HistoryNode)

    def test_archive_marker_is_history_node(self):
        node = ArchiveMarkerNode(
            archived_at=datetime.now(timezone.utc),
            summary="test",
            node_count=5,
            journal_date="2026-04-08",
        )
        assert isinstance(node, HistoryNode)

    def test_housekeeping_is_history_node(self):
        node = HousekeepingNode(inner_nodes=[])
        assert isinstance(node, HistoryNode)

    def test_tool_call_node_is_not_history_node(self):
        """ToolCallNode is an inner node within a TurnNode, not a
        top-level HistoryNode."""
        tc = _tc("c1", "read_file", {"path": "x.py"})
        result = _result("c1", "content")
        node = ToolCallNode(tc, result)
        assert not isinstance(node, HistoryNode)

    def test_base_class_render_raises(self):
        with pytest.raises(NotImplementedError):
            HistoryNode().render()

    def test_base_class_token_cost_raises(self):
        with pytest.raises(NotImplementedError):
            HistoryNode().token_cost()


# ---------------------------------------------------------------------------
# ArchiveMarkerNode
# ---------------------------------------------------------------------------


class TestArchiveMarkerNode:
    def _make_marker(self) -> ArchiveMarkerNode:
        return ArchiveMarkerNode(
            archived_at=datetime(2026, 4, 8, 22, 10, 0, tzinfo=timezone.utc),
            summary="Investigated issue #6 and opened MR",
            node_count=12,
            journal_date="2026-04-08",
        )

    def test_render_returns_single_user_message(self):
        node = self._make_marker()
        rendered = node.render()
        assert len(rendered) == 1
        assert isinstance(rendered[0], UserMessage)

    def test_render_text_contains_node_count(self):
        node = self._make_marker()
        text = node.render()[0].content
        assert "12 turns" in text

    def test_render_text_singular_turn(self):
        node = ArchiveMarkerNode(
            archived_at=datetime(2026, 4, 8, 22, 10, 0, tzinfo=timezone.utc),
            summary="brief",
            node_count=1,
            journal_date="2026-04-08",
        )
        text = node.render()[0].content
        assert "1 turn," in text
        assert "1 turns" not in text

    def test_render_text_contains_journal_date(self):
        node = self._make_marker()
        text = node.render()[0].content
        assert "2026-04-08" in text

    def test_render_text_mentions_read_journal(self):
        node = self._make_marker()
        text = node.render()[0].content
        assert "read_journal" in text

    def test_token_cost_positive(self):
        node = self._make_marker()
        assert node.token_cost() > 0

    def test_token_cost_is_consistent(self):
        node = self._make_marker()
        assert node.token_cost() == node.token_cost()

    def test_fields_accessible(self):
        node = self._make_marker()
        assert node.node_count == 12
        assert node.journal_date == "2026-04-08"
        assert node.summary == "Investigated issue #6 and opened MR"
        assert node.archived_at == datetime(2026, 4, 8, 22, 10, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# HousekeepingNode
# ---------------------------------------------------------------------------


class TestHousekeepingNode:
    def test_render_returns_empty_list(self):
        node = HousekeepingNode(inner_nodes=[])
        assert node.render() == []

    def test_token_cost_is_zero(self):
        inner = [
            UserPromptNode(UserMessage(content="x" * 5000)),
            TurnNode(assistant_content="y" * 5000, tool_call_nodes=[]),
        ]
        node = HousekeepingNode(inner_nodes=inner)
        assert node.token_cost() == 0

    def test_inner_nodes_accessible(self):
        prompt = UserPromptNode(UserMessage(content="housekeeping prompt"))
        turn = TurnNode(assistant_content="journaled everything", tool_call_nodes=[])
        node = HousekeepingNode(inner_nodes=[prompt, turn])
        assert len(node.inner_nodes) == 2
        assert isinstance(node.inner_nodes[0], UserPromptNode)
        assert isinstance(node.inner_nodes[1], TurnNode)

    def test_render_empty_even_with_inner_content(self):
        """Inner nodes exist for debugging but never contribute to render."""
        prompt = UserPromptNode(UserMessage(content="big prompt " + "z" * 3000))
        node = HousekeepingNode(inner_nodes=[prompt])
        assert node.render() == []


# ---------------------------------------------------------------------------
# New node types in HistoryTree
# ---------------------------------------------------------------------------


class TestNewNodeTypesInTree:
    def test_archive_marker_renders_in_tree(self):
        tree = HistoryTree()
        marker = ArchiveMarkerNode(
            archived_at=datetime.now(timezone.utc),
            summary="archived old content",
            node_count=8,
            journal_date="2026-04-07",
        )
        tree.nodes.append(marker)
        tree.append_user_prompt("continue working")
        tree.append_turn(AssistantMessage(content="ok"), [])

        rendered = tree.render()
        assert len(rendered) == 3
        assert isinstance(rendered[0], UserMessage)
        assert "8 turns" in rendered[0].content

    def test_housekeeping_node_invisible_in_tree(self):
        tree = HistoryTree()
        tree.append_user_prompt("start")
        tree.append_turn(AssistantMessage(content="working"), [])

        hk = HousekeepingNode(inner_nodes=[
            UserPromptNode(UserMessage(content="housekeeping prompt")),
            TurnNode(assistant_content="done journaling", tool_call_nodes=[]),
        ])
        tree.nodes.append(hk)
        tree.append_user_prompt("next task")

        rendered = tree.render()
        roles = [m.role for m in rendered]
        assert "user" in roles
        assert "assistant" in roles
        # HousekeepingNode contributes nothing
        assert len(rendered) == 3

    def test_estimated_tokens_excludes_housekeeping(self):
        tree = HistoryTree()
        tree.append_user_prompt("start")
        tokens_before = tree.estimated_tokens()

        hk = HousekeepingNode(inner_nodes=[
            UserPromptNode(UserMessage(content="x" * 10000)),
        ])
        tree.nodes.append(hk)
        assert tree.estimated_tokens() == tokens_before

    def test_estimated_tokens_includes_archive_marker(self):
        tree = HistoryTree()
        tokens_empty = tree.estimated_tokens()

        marker = ArchiveMarkerNode(
            archived_at=datetime.now(timezone.utc),
            summary="test",
            node_count=3,
            journal_date="2026-04-08",
        )
        tree.nodes.append(marker)
        assert tree.estimated_tokens() > tokens_empty

    def test_compaction_skips_archive_marker(self):
        """ArchiveMarkerNode should not be collapsed by compaction."""
        tree = HistoryTree()
        marker = ArchiveMarkerNode(
            archived_at=datetime.now(timezone.utc),
            summary="old content",
            node_count=20,
            journal_date="2026-04-07",
        )
        tree.nodes.append(marker)
        for i in range(6):
            tree.append_turn(
                AssistantMessage(content=f"turn {i} " + "y" * 500), [],
            )

        result = tree.compact(
            context_budget=1, low_watermark=0.1, overhead_tokens=0,
            protected_tail_nodes=2, protected_tail_tool_calls=0,
        )

        # The archive marker is still just the archive marker; its type
        # is unchanged and it doesn't have a collapse_state.
        assert isinstance(tree.nodes[0], ArchiveMarkerNode)
        # Other unprotected turns should have been collapsed.
        assert result.nodes_collapsed > 0

    def test_compaction_skips_housekeeping_node(self):
        """HousekeepingNode should not be collapsed by compaction."""
        tree = HistoryTree()
        tree.append_user_prompt("start")
        hk = HousekeepingNode(inner_nodes=[
            UserPromptNode(UserMessage(content="hk prompt")),
        ])
        tree.nodes.append(hk)
        for i in range(6):
            tree.append_turn(
                AssistantMessage(content=f"turn {i} " + "y" * 500), [],
            )

        result = tree.compact(
            context_budget=1, low_watermark=0.1, overhead_tokens=0,
            protected_tail_nodes=2, protected_tail_tool_calls=0,
        )

        assert isinstance(tree.nodes[1], HousekeepingNode)
        assert result.nodes_collapsed > 0
