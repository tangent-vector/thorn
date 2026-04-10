"""Tests for thorn.core._housekeeping — harness-driven history housekeeping."""

from __future__ import annotations

import json

from thorn.core._context import ExecutionContext, NullEventSink, Scope
from thorn.core._history import (
    ArchiveMarkerNode,
    HistoryTree,
    HousekeepingNode,
    TurnNode,
    UserPromptNode,
)
from thorn.core._housekeeping import (
    CONTEXT_BOUNDARY_TEXT,
    HOUSEKEEPING_PROMPT,
    _HOUSEKEEPING_TOOL_ALLOWLIST,
    filter_housekeeping_tools,
    perform_housekeeping,
    select_cut_point,
)
from thorn.core._loop import _WrappedTool, run_agent_loop
from thorn.core._messages import (
    AssistantMessage,
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tc(call_id: str, name: str, args: dict) -> ToolCall:
    return ToolCall(call_id=call_id, name=name, arguments=json.dumps(args))


def _result(call_id: str, content: str) -> ToolResultMessage:
    return ToolResultMessage(call_id=call_id, content=content)


def _make_history(
    num_turns: int = 10,
    *,
    content_size: int = 500,
) -> HistoryTree:
    """Create a history with alternating UserPromptNode / TurnNode pairs."""
    tree = HistoryTree()
    for i in range(num_turns):
        tree.append_user_prompt(f"prompt {i}")
        tc = _tc(f"c{i}", "read_file", {"path": f"file{i}.py"})
        tree.append_turn(
            AssistantMessage(
                content=f"response {i} " + "x" * content_size,
                tool_calls=[tc],
            ),
            [_result(f"c{i}", "y" * content_size)],
        )
    return tree


def _make_wrapped_tool(name: str) -> _WrappedTool:
    """Create a minimal _WrappedTool with a given schema name."""

    async def _noop(**kwargs: object) -> str:
        return "ok"

    schema = {
        "type": "function",
        "function": {
            "name": name,
            "parameters": {"type": "object", "properties": {}},
        },
    }
    return _WrappedTool(schema=schema, execute=_noop)


class CaptureSink(NullEventSink):
    """Event sink that records status messages for test assertions."""

    def __init__(self) -> None:
        self.messages: list[str] = []

    async def on_status(self, message: str, scope: Scope | None = None) -> None:
        self.messages.append(message)


# ---------------------------------------------------------------------------
# select_cut_point
# ---------------------------------------------------------------------------


class TestSelectCutPoint:
    def test_basic_cut(self):
        tree = _make_history(5)
        protected = {6, 7, 8, 9}
        assert select_cut_point(tree, protected) == 5

    def test_all_protected(self):
        tree = _make_history(3)
        protected = set(range(6))
        assert select_cut_point(tree, protected) is None

    def test_first_node_protected(self):
        tree = _make_history(3)
        protected = {0, 4, 5}
        assert select_cut_point(tree, protected) is None

    def test_no_nodes_protected(self):
        tree = _make_history(3)
        protected: set[int] = set()
        assert select_cut_point(tree, protected) == 5

    def test_empty_history(self):
        tree = HistoryTree()
        assert select_cut_point(tree, set()) is None

    def test_single_unprotected_node(self):
        tree = HistoryTree()
        tree.append_user_prompt("hello")
        tree.append_turn(AssistantMessage(content="hi"), [])
        protected = {1}
        assert select_cut_point(tree, protected) == 0

    def test_only_tail_protected(self):
        """With 10 nodes and last 4 protected, cut point is 5."""
        tree = _make_history(5)
        protected = {6, 7, 8, 9}
        assert select_cut_point(tree, protected) == 5

    def test_scattered_protected_uses_min(self):
        """Cut point is determined by the lowest protected index."""
        tree = _make_history(5)
        protected = {3, 8, 9}
        assert select_cut_point(tree, protected) == 2


# ---------------------------------------------------------------------------
# filter_housekeeping_tools
# ---------------------------------------------------------------------------


class TestFilterHousekeepingTools:
    def test_keeps_only_allowed_tools(self):
        tools = [
            _make_wrapped_tool("write_journal"),
            _make_wrapped_tool("read_journal"),
            _make_wrapped_tool("read_file"),
            _make_wrapped_tool("edit_file"),
            _make_wrapped_tool("search_files"),
            _make_wrapped_tool("run_shell"),
        ]
        filtered = filter_housekeeping_tools(tools)
        names = {t.schema["function"]["name"] for t in filtered}
        assert names == {"write_journal", "read_journal", "read_file"}

    def test_empty_input(self):
        assert filter_housekeeping_tools([]) == []

    def test_no_matching_tools(self):
        tools = [
            _make_wrapped_tool("edit_file"),
            _make_wrapped_tool("run_shell"),
        ]
        assert filter_housekeeping_tools(tools) == []

    def test_allowlist_contents(self):
        assert "write_journal" in _HOUSEKEEPING_TOOL_ALLOWLIST
        assert "read_journal" in _HOUSEKEEPING_TOOL_ALLOWLIST
        assert "read_file" in _HOUSEKEEPING_TOOL_ALLOWLIST


# ---------------------------------------------------------------------------
# perform_housekeeping (unit-level, using MockProvider)
# ---------------------------------------------------------------------------


class TestPerformHousekeeping:
    async def test_trims_and_restructures_history(self):
        """After housekeeping, the tree has [ArchiveMarker, tail, HousekeepingNode]."""
        tree = _make_history(12)
        original_count = len(tree.nodes)

        sink = CaptureSink()
        provider = MockProvider(
            canned_responses=[
                [
                    TextChunk(text="Acknowledged, nothing critical to journal."),
                    UsageChunk(prompt_tokens=100, completion_tokens=20, total_tokens=120),
                    FinishChunk(reason="stop"),
                ],
            ],
        )
        ctx = ExecutionContext(
            provider=provider,
            event_sink=sink,
            context_window=50000,
        )

        result = await perform_housekeeping(
            context=ctx,
            history=tree,
            all_tools=[
                _make_wrapped_tool("write_journal"),
                _make_wrapped_tool("edit_file"),
            ],
            system_prompts=None,
        )

        assert result is not None
        assert result.nodes_trimmed > 1

        assert isinstance(tree.nodes[0], ArchiveMarkerNode)
        assert isinstance(tree.nodes[-1], HousekeepingNode)

        marker = tree.nodes[0]
        assert marker.node_count == result.nodes_trimmed

        hk_node = tree.nodes[-1]
        assert len(hk_node.inner_nodes) > 0

        for node in tree.nodes[1:-1]:
            assert isinstance(node, (UserPromptNode, TurnNode))

        status_msgs = [m for m in sink.messages if "housekeeping" in m]
        assert len(status_msgs) > 0

        assert len(tree.nodes) < original_count

    async def test_returns_none_when_all_protected(self):
        """When all nodes are protected, housekeeping cannot trim."""
        tree = HistoryTree()
        tree.append_user_prompt("only node")
        tree.append_turn(AssistantMessage(content="reply"), [])

        provider = MockProvider()
        ctx = ExecutionContext(provider=provider, context_window=50000)

        result = await perform_housekeeping(
            context=ctx,
            history=tree,
            all_tools=[],
            system_prompts=None,
        )

        assert result is None
        assert len(tree.nodes) == 2

    async def test_boundary_marker_visible_during_housekeeping(self):
        """The LLM completion receives the boundary marker text."""
        tree = _make_history(6)

        received_messages: list[list] = []

        class SpyProvider(MockProvider):
            async def complete(self, system_prompts, tools, messages):
                received_messages.append(list(messages))
                async for chunk in super().complete(system_prompts, tools, messages):
                    yield chunk

        provider = SpyProvider(
            canned_responses=[
                [
                    TextChunk(text="Acknowledged."),
                    UsageChunk(prompt_tokens=100, completion_tokens=20, total_tokens=120),
                    FinishChunk(reason="stop"),
                ],
            ],
        )
        ctx = ExecutionContext(provider=provider, context_window=50000)

        await perform_housekeeping(
            context=ctx,
            history=tree,
            all_tools=[],
            system_prompts=None,
        )

        assert len(received_messages) > 0
        all_content = " ".join(
            m.content for msg_list in received_messages
            for m in msg_list
            if hasattr(m, "content") and isinstance(m.content, str)
        )
        assert "CONTEXT BOUNDARY" in all_content
        assert HOUSEKEEPING_PROMPT[:40] in all_content

    async def test_agent_journal_call_during_housekeeping(self):
        """The agent can call write_journal during housekeeping."""
        tree = _make_history(6)

        journal_calls: list[dict] = []

        async def write_journal(content: str = "") -> str:
            """Append to journal."""
            journal_calls.append({"content": content})
            return "Journal entry appended."

        wj_tool = wrap_function(write_journal)

        provider = MockProvider(
            canned_responses=[
                # Agent calls write_journal
                [
                    ToolCallChunk(
                        call_id="j1",
                        name="write_journal",
                        arguments=json.dumps({"content": "Important context saved."}),
                    ),
                    UsageChunk(prompt_tokens=100, completion_tokens=30, total_tokens=130),
                    FinishChunk(reason="tool_calls"),
                ],
                # Agent acknowledges
                [
                    TextChunk(text="Done journaling."),
                    UsageChunk(prompt_tokens=100, completion_tokens=20, total_tokens=120),
                    FinishChunk(reason="stop"),
                ],
            ],
        )
        ctx = ExecutionContext(provider=provider, context_window=50000)

        result = await perform_housekeeping(
            context=ctx,
            history=tree,
            all_tools=[wj_tool],
            system_prompts=None,
        )

        assert result is not None
        assert len(journal_calls) == 1
        assert journal_calls[0]["content"] == "Important context saved."

    async def test_loop_limit_error_caught_gracefully(self):
        """Housekeeping completes the trim even if the sub-loop exceeds rounds."""
        tree = _make_history(12)

        # Agent keeps making tool calls, never gives a text response
        provider = MockProvider(
            canned_responses=[
                [
                    ToolCallChunk(
                        call_id=f"j{i}",
                        name="write_journal",
                        arguments="{}",
                    ),
                    UsageChunk(prompt_tokens=100, completion_tokens=20, total_tokens=120),
                    FinishChunk(reason="tool_calls"),
                ]
                for i in range(10)
            ],
        )
        ctx = ExecutionContext(provider=provider, context_window=50000)

        result = await perform_housekeeping(
            context=ctx,
            history=tree,
            all_tools=[_make_wrapped_tool("write_journal")],
            system_prompts=None,
        )

        # Trim should still have happened despite the LoopLimitError
        assert result is not None
        assert result.nodes_trimmed > 0
        assert isinstance(tree.nodes[0], ArchiveMarkerNode)
        assert isinstance(tree.nodes[-1], HousekeepingNode)

    async def test_protected_tail_preserved(self):
        """Nodes in the protected tail survive housekeeping intact."""
        tree = _make_history(12)
        protected = tree._protected_indices()
        protected_contents = {
            i: tree.nodes[i] for i in protected
        }

        provider = MockProvider(
            canned_responses=[
                [
                    TextChunk(text="Acknowledged."),
                    UsageChunk(prompt_tokens=100, completion_tokens=20, total_tokens=120),
                    FinishChunk(reason="stop"),
                ],
            ],
        )
        ctx = ExecutionContext(provider=provider, context_window=50000)

        result = await perform_housekeeping(
            context=ctx,
            history=tree,
            all_tools=[],
            system_prompts=None,
        )

        assert result is not None

        # Protected nodes should be present in the middle of the tree
        # (between ArchiveMarkerNode and HousekeepingNode).
        tail_nodes = tree.nodes[1:-1]
        for original_node in protected_contents.values():
            assert original_node in tail_nodes

    async def test_housekeeping_node_renders_empty(self):
        """The HousekeepingNode created by housekeeping renders as nothing."""
        tree = _make_history(6)

        provider = MockProvider(
            canned_responses=[
                [
                    TextChunk(text="Acknowledged."),
                    UsageChunk(prompt_tokens=100, completion_tokens=20, total_tokens=120),
                    FinishChunk(reason="stop"),
                ],
            ],
        )
        ctx = ExecutionContext(provider=provider, context_window=50000)

        await perform_housekeeping(
            context=ctx,
            history=tree,
            all_tools=[],
            system_prompts=None,
        )

        hk_node = tree.nodes[-1]
        assert isinstance(hk_node, HousekeepingNode)
        assert hk_node.render() == []
        assert hk_node.token_cost() == 0


# ---------------------------------------------------------------------------
# Integration with run_agent_loop
# ---------------------------------------------------------------------------


class TestHousekeepingInLoop:
    async def test_housekeeping_triggered_when_compaction_insufficient(self):
        """When compact() can't reduce below high watermark, housekeeping runs."""
        sink = CaptureSink()

        async def dummy_tool() -> str:
            """A dummy tool."""
            return "ok"

        tool = wrap_function(dummy_tool)

        # Build a history that's already large (to ensure compaction
        # can't bring it under the high watermark).
        tree = _make_history(15, content_size=2000)

        # Provider: first response is a tool call with very high prompt_tokens
        # (above high watermark), second is the housekeeping acknowledgment,
        # third is the final response in the main loop.
        provider = MockProvider(
            canned_responses=[
                # Main loop: agent makes a tool call, high usage triggers compaction
                [
                    ToolCallChunk(call_id="c1", name="dummy_tool", arguments="{}"),
                    UsageChunk(prompt_tokens=9000, completion_tokens=100, total_tokens=9100),
                    FinishChunk(reason="tool_calls"),
                ],
                # Housekeeping sub-loop: agent acknowledges
                [
                    TextChunk(text="Acknowledged housekeeping."),
                    UsageChunk(prompt_tokens=2000, completion_tokens=30, total_tokens=2030),
                    FinishChunk(reason="stop"),
                ],
                # Main loop resumes: agent finishes
                [
                    TextChunk(text="Task complete."),
                    UsageChunk(prompt_tokens=2000, completion_tokens=50, total_tokens=2050),
                    FinishChunk(reason="stop"),
                ],
            ],
            context_window=10000,
        )

        ctx = ExecutionContext(
            provider=provider,
            event_sink=sink,
            context_window=10000,
        )

        result = await run_agent_loop(
            context=ctx,
            user_prompt="do something",
            tools=[tool],
            history=tree,
        )

        assert result == "Task complete."
        assert any("housekeeping" in m for m in sink.messages)
        assert isinstance(tree.nodes[0], ArchiveMarkerNode)

    async def test_no_housekeeping_when_compaction_sufficient(self):
        """When compact() brings tokens below watermark, no housekeeping."""
        sink = CaptureSink()

        async def dummy_tool() -> str:
            """A dummy tool."""
            return "ok"

        tool = wrap_function(dummy_tool)

        # Small history that compaction can handle
        tree = _make_history(3)

        provider = MockProvider(
            canned_responses=[
                [
                    ToolCallChunk(call_id="c1", name="dummy_tool", arguments="{}"),
                    UsageChunk(prompt_tokens=9000, completion_tokens=100, total_tokens=9100),
                    FinishChunk(reason="tool_calls"),
                ],
                [
                    TextChunk(text="Done."),
                    UsageChunk(prompt_tokens=3000, completion_tokens=50, total_tokens=3050),
                    FinishChunk(reason="stop"),
                ],
            ],
            context_window=10000,
        )

        ctx = ExecutionContext(
            provider=provider,
            event_sink=sink,
            context_window=10000,
        )

        result = await run_agent_loop(
            context=ctx,
            user_prompt="do something",
            tools=[tool],
            history=tree,
        )

        assert result == "Done."
        assert not any("housekeeping" in m for m in sink.messages)

    async def test_no_recursive_housekeeping_in_subloop(self):
        """The _housekeeping=True flag prevents nested housekeeping."""
        tree = _make_history(15, content_size=2000)

        provider = MockProvider(
            canned_responses=[
                [
                    TextChunk(text="Done in subloop."),
                    UsageChunk(prompt_tokens=9000, completion_tokens=50, total_tokens=9050),
                    FinishChunk(reason="stop"),
                ],
            ],
            context_window=10000,
        )

        ctx = ExecutionContext(
            provider=provider,
            event_sink=CaptureSink(),
            context_window=10000,
        )

        result = await run_agent_loop(
            context=ctx,
            user_prompt="housekeeping task",
            tools=[],
            history=tree,
            _housekeeping=True,
        )

        assert result == "Done in subloop."
        # No housekeeping should have triggered
        assert not any(
            "housekeeping" in m
            for m in ctx.event_sink.messages  # type: ignore[attr-defined]
        )


# ---------------------------------------------------------------------------
# Post-housekeeping history structure
# ---------------------------------------------------------------------------


class TestPostHousekeepingStructure:
    async def test_archive_marker_has_correct_node_count(self):
        tree = _make_history(12)
        protected = tree._protected_indices()
        expected_trimmed = min(protected)

        provider = MockProvider(
            canned_responses=[
                [
                    TextChunk(text="OK"),
                    UsageChunk(prompt_tokens=100, completion_tokens=20, total_tokens=120),
                    FinishChunk(reason="stop"),
                ],
            ],
        )
        ctx = ExecutionContext(provider=provider, context_window=50000)

        result = await perform_housekeeping(
            context=ctx,
            history=tree,
            all_tools=[],
            system_prompts=None,
        )

        assert result is not None
        assert result.nodes_trimmed == expected_trimmed
        assert tree.nodes[0].node_count == expected_trimmed

    async def test_rendered_history_is_coherent(self):
        """Rendered messages after housekeeping form a valid sequence."""
        tree = _make_history(12)

        provider = MockProvider(
            canned_responses=[
                [
                    TextChunk(text="OK"),
                    UsageChunk(prompt_tokens=100, completion_tokens=20, total_tokens=120),
                    FinishChunk(reason="stop"),
                ],
            ],
        )
        ctx = ExecutionContext(provider=provider, context_window=50000)

        await perform_housekeeping(
            context=ctx,
            history=tree,
            all_tools=[],
            system_prompts=None,
        )

        rendered = tree.render()
        assert len(rendered) > 0

        first = rendered[0]
        assert isinstance(first, UserMessage)
        assert "archived" in first.content.lower() or "journal" in first.content.lower()

    async def test_housekeeping_node_inner_nodes_contain_boundary(self):
        """The HousekeepingNode's inner nodes include the boundary marker."""
        tree = _make_history(6)

        provider = MockProvider(
            canned_responses=[
                [
                    TextChunk(text="Ack"),
                    UsageChunk(prompt_tokens=100, completion_tokens=20, total_tokens=120),
                    FinishChunk(reason="stop"),
                ],
            ],
        )
        ctx = ExecutionContext(provider=provider, context_window=50000)

        await perform_housekeeping(
            context=ctx,
            history=tree,
            all_tools=[],
            system_prompts=None,
        )

        hk_node = tree.nodes[-1]
        assert isinstance(hk_node, HousekeepingNode)

        # First inner node should be the boundary marker
        boundary = hk_node.inner_nodes[0]
        assert isinstance(boundary, UserPromptNode)
        assert "CONTEXT BOUNDARY" in boundary.message.content

        # Should also contain the housekeeping prompt and agent response
        assert len(hk_node.inner_nodes) >= 3

    async def test_second_housekeeping_trims_archive_marker(self):
        """A second housekeeping cycle replaces the old ArchiveMarkerNode."""
        tree = _make_history(12)

        # First housekeeping
        provider1 = MockProvider(
            canned_responses=[
                [
                    TextChunk(text="First HK done."),
                    UsageChunk(prompt_tokens=100, completion_tokens=20, total_tokens=120),
                    FinishChunk(reason="stop"),
                ],
            ],
        )
        ctx1 = ExecutionContext(provider=provider1, context_window=50000)
        result1 = await perform_housekeeping(
            context=ctx1,
            history=tree,
            all_tools=[],
            system_prompts=None,
        )
        assert result1 is not None
        assert isinstance(tree.nodes[0], ArchiveMarkerNode)
        first_marker_count = tree.nodes[0].node_count

        # Add more nodes to simulate continued activity
        for i in range(8):
            tree.append_user_prompt(f"new prompt {i}")
            tree.append_turn(
                AssistantMessage(content=f"new response {i} " + "x" * 500),
                [],
            )

        # Second housekeeping
        provider2 = MockProvider(
            canned_responses=[
                [
                    TextChunk(text="Second HK done."),
                    UsageChunk(prompt_tokens=100, completion_tokens=20, total_tokens=120),
                    FinishChunk(reason="stop"),
                ],
            ],
        )
        ctx2 = ExecutionContext(provider=provider2, context_window=50000)
        result2 = await perform_housekeeping(
            context=ctx2,
            history=tree,
            all_tools=[],
            system_prompts=None,
        )

        assert result2 is not None
        assert isinstance(tree.nodes[0], ArchiveMarkerNode)
        # The old marker was trimmed and replaced by a new one
        assert tree.nodes[0].node_count > 0
