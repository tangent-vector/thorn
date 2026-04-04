"""Tests for thorn._history — hierarchical history and compaction."""

from __future__ import annotations

import json

import pytest

from thorn._context import ExecutionContext, set_context, reset_context
from thorn._history import (
    ABBREVIATED_ARG_VALUE_MAX_LEN,
    CHARS_PER_TOKEN,
    DEFAULT_HIGH_WATERMARK,
    DEFAULT_LOW_WATERMARK,
    LONG_CONTENT_THRESHOLD,
    TRUNCATED_PREFIX_CHARS,
    CollapseState,
    CompactionResult,
    DirectoryListCallNode,
    FileReadCallNode,
    HistoryTree,
    ToolCallNode,
    TurnNode,
    UserPromptNode,
    _abbreviate_arguments,
    _tool_call_summary,
    _truncate_content,
    estimate_tokens,
)
from thorn._loop import run_agent_loop
from thorn._messages import (
    AssistantMessage,
    Message,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)
from thorn._provider import (
    FinishChunk,
    MockProvider,
    TextChunk,
    ToolCallChunk,
    UsageChunk,
)
from thorn._func import wrap_function


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
        )

        assert result.nodes_collapsed > 0
        # The earliest turns should be collapsed
        assert tree.nodes[1].collapse_state == CollapseState.COLLAPSED

    def test_protects_last_two_nodes(self):
        """The last two nodes should never be collapsed."""
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
        )

        # The initial user prompt is the most recent (and only)
        # UserPromptNode, so it should be protected.
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
        tree.compact(context_budget=budget, low_watermark=0.3, overhead_tokens=0)
        tokens_after_first = tree.estimated_tokens()

        result2 = tree.compact(
            context_budget=budget, low_watermark=0.3, overhead_tokens=0,
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

        from thorn._context import EventSink, Scope, NullEventSink

        class CaptureSink(NullEventSink):
            async def on_status(self, message, scope=None):
                status_messages.append(message)

        async def big_tool() -> str:
            """Return lots of content."""
            return "a" * 10000

        tool = wrap_function(big_tool)

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
        # Pre-populate with some old history so there's something to compact
        for i in range(5):
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

        from thorn._context import NullEventSink

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
        from thorn._agent import Agent

        status_messages: list[str] = []

        from thorn._context import NullEventSink

        class CaptureSink(NullEventSink):
            async def on_status(self, message, scope=None):
                status_messages.append(message)

        async def big_read() -> str:
            """Return lots."""
            return "z" * 8000

        class BigReader(Agent):
            tools = [big_read]

        responses = []
        for i in range(6):
            responses.append([
                ToolCallChunk(call_id=f"c{i}", name="big_read", arguments="{}"),
                UsageChunk(
                    prompt_tokens=8000 * (i + 1),
                    completion_tokens=100,
                    total_tokens=8000 * (i + 1) + 100,
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

        wrapped = wrap_function(my_reader)
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

        wrapped = wrap_function(plain_tool)
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
            tools=[wrap_function(reader), wrap_function(lister), wrap_function(plain)],
            history=history,
        )

        turn_nodes = [n for n in history.nodes if isinstance(n, TurnNode)]
        tool_turn = turn_nodes[0]
        assert len(tool_turn.tool_call_nodes) == 3
        assert isinstance(tool_turn.tool_call_nodes[0], FileReadCallNode)
        assert isinstance(tool_turn.tool_call_nodes[1], DirectoryListCallNode)
        assert type(tool_turn.tool_call_nodes[2]) is ToolCallNode
