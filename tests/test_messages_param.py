"""Tests for conversation history accumulation.

Verifies that:
- ``run_agent_loop`` accumulates history on a provided ``HistoryTree``.
- ``agent.prompt()`` accumulates history on the agent's internal tree,
  enabling multi-turn patterns without external list management.
"""

from __future__ import annotations

import pytest

from thorn.core._agent import Agent
from thorn.core._context import ExecutionContext, get_context, reset_context, set_context
from thorn.core._func import wrap_function
from thorn.core._history import HistoryTree, TurnNode, UserPromptNode
from thorn.core._loop import run_agent_loop
from thorn.core._messages import AssistantMessage, Message, ToolResultMessage, UserMessage
from thorn.core._provider import FinishChunk, MockProvider, TextChunk, ToolCallChunk


def _text_response(text: str):
    return [TextChunk(text=text), FinishChunk(reason="stop")]


def _tool_call_response(call_id: str, name: str, arguments: str):
    return [
        ToolCallChunk(call_id=call_id, name=name, arguments=arguments),
        FinishChunk(reason="tool_calls"),
    ]


# ---------------------------------------------------------------------------
# run_agent_loop: history parameter
# ---------------------------------------------------------------------------


class TestRunAgentLoopHistory:
    async def test_none_creates_internal_tree(self):
        """When history=None, a fresh tree is created internally."""
        provider = MockProvider(canned_responses=[_text_response("hi")])
        ctx = ExecutionContext(provider=provider)
        result = await run_agent_loop(
            context=ctx, user_prompt="hello", tools=[], history=None,
        )
        assert result == "hi"

    async def test_provided_tree_accumulates_history(self):
        """A provided tree should contain nodes after the call."""
        provider = MockProvider(canned_responses=[_text_response("reply")])
        ctx = ExecutionContext(provider=provider)
        tree = HistoryTree()

        result = await run_agent_loop(
            context=ctx, user_prompt="hello", tools=[], history=tree,
        )
        assert result == "reply"
        assert len(tree.nodes) == 2
        assert isinstance(tree.nodes[0], UserPromptNode)
        assert tree.nodes[0].message.content == "hello"
        assert isinstance(tree.nodes[1], TurnNode)
        assert tree.nodes[1].assistant_content == "reply"

    async def test_multi_turn_accumulation(self):
        """Calling run_agent_loop twice with the same tree accumulates
        both turns."""
        provider = MockProvider(canned_responses=[
            _text_response("first reply"),
            _text_response("second reply"),
        ])
        ctx = ExecutionContext(provider=provider)
        tree = HistoryTree()

        await run_agent_loop(
            context=ctx, user_prompt="turn 1", tools=[], history=tree,
        )
        assert len(tree.nodes) == 2

        await run_agent_loop(
            context=ctx, user_prompt="turn 2", tools=[], history=tree,
        )
        assert len(tree.nodes) == 4
        assert isinstance(tree.nodes[2], UserPromptNode)
        assert tree.nodes[2].message.content == "turn 2"
        assert isinstance(tree.nodes[3], TurnNode)
        assert tree.nodes[3].assistant_content == "second reply"

    async def test_tool_calls_appear_in_history(self):
        """Tool call rounds should be visible in the accumulated history."""
        async def double(x: int) -> int:
            """Double."""
            return x * 2

        tool = wrap_function(double)
        provider = MockProvider(canned_responses=[
            _tool_call_response("c1", "double", '{"x": 5}'),
            _text_response("result is 10"),
        ])
        ctx = ExecutionContext(provider=provider)
        tree = HistoryTree()

        await run_agent_loop(
            context=ctx, user_prompt="double 5", tools=[tool],
            history=tree,
        )

        assert isinstance(tree.nodes[0], UserPromptNode)
        assert isinstance(tree.nodes[1], TurnNode)
        assert len(tree.nodes[1].tool_call_nodes) == 1
        assert tree.nodes[1].tool_call_nodes[0].result.content == "10"
        assert isinstance(tree.nodes[2], TurnNode)
        assert tree.nodes[2].assistant_content == "result is 10"

    async def test_prior_history_is_seen_by_provider(self):
        """The provider should see earlier messages when history is reused."""
        seen_messages: list[list[Message]] = []

        class CapturingProvider(MockProvider):
            async def complete(self, system_prompts, tools, messages):
                seen_messages.append(list(messages))
                async for chunk in super().complete(system_prompts, tools, messages):
                    yield chunk

        provider = CapturingProvider(canned_responses=[
            _text_response("first"),
            _text_response("second"),
        ])
        ctx = ExecutionContext(provider=provider)
        tree = HistoryTree()

        await run_agent_loop(
            context=ctx, user_prompt="msg1", tools=[], history=tree,
        )
        await run_agent_loop(
            context=ctx, user_prompt="msg2", tools=[], history=tree,
        )

        # First call: provider sees [UserMessage("msg1")]
        assert len(seen_messages[0]) == 1
        assert seen_messages[0][0].content == "msg1"

        # Second call: provider sees all prior history plus new prompt
        assert len(seen_messages[1]) == 3
        assert seen_messages[1][0].content == "msg1"
        assert seen_messages[1][2].content == "msg2"


# ---------------------------------------------------------------------------
# agent.prompt: history accumulation
# ---------------------------------------------------------------------------


class TestAgentPromptMessages:
    async def test_text_mode_accumulates_on_agent(self):
        """Consecutive prompt() calls accumulate history on agent._history."""
        provider = MockProvider(canned_responses=[
            _text_response("wrote code"),
            _text_response("fixed errors"),
        ])
        ctx = ExecutionContext(provider=provider)
        token = set_context(ctx)
        try:
            agent = Agent()

            await agent.prompt("write code")
            assert len(agent._history.nodes) == 2

            await agent.prompt("fix build errors")
            assert len(agent._history.nodes) == 4
            assert isinstance(agent._history.nodes[2], UserPromptNode)
            assert agent._history.nodes[2].message.content == "fix build errors"
        finally:
            reset_context(token)

    async def test_structured_mode_accumulates_on_agent(self):
        provider = MockProvider(canned_responses=[
            [
                ToolCallChunk(
                    call_id="c1", name="return_result",
                    arguments='{"value": 42}',
                ),
                FinishChunk(reason="stop"),
            ],
            [
                ToolCallChunk(
                    call_id="c2", name="return_result",
                    arguments='{"value": 99}',
                ),
                FinishChunk(reason="stop"),
            ],
        ])
        ctx = ExecutionContext(provider=provider)
        token = set_context(ctx)
        try:
            agent = Agent()

            r1 = await agent.prompt[int]("count things")
            assert r1 == 42
            first_turn_len = len(agent._history.nodes)
            assert first_turn_len >= 2

            r2 = await agent.prompt[int]("count more")
            assert r2 == 99
            assert len(agent._history.nodes) > first_turn_len
        finally:
            reset_context(token)

    async def test_fresh_agent_starts_with_empty_history(self):
        provider = MockProvider(canned_responses=[_text_response("ok")])
        ctx = ExecutionContext(provider=provider)
        token = set_context(ctx)
        try:
            agent = Agent()
            assert len(agent._history.nodes) == 0
            result = await agent.prompt("hello")
            assert result == "ok"
            assert len(agent._history.nodes) == 2
        finally:
            reset_context(token)

    async def test_separate_agents_have_independent_history(self):
        """Two agent instances don't share history."""
        provider = MockProvider(canned_responses=[
            _text_response("reply A"),
            _text_response("reply B"),
        ])
        ctx = ExecutionContext(provider=provider)
        token = set_context(ctx)
        try:
            agent_a = Agent()
            agent_b = Agent()

            await agent_a.prompt("hello A")
            await agent_b.prompt("hello B")

            assert len(agent_a._history.nodes) == 2
            assert len(agent_b._history.nodes) == 2
            assert agent_a._history.nodes[0].message.content == "hello A"
            assert agent_b._history.nodes[0].message.content == "hello B"
        finally:
            reset_context(token)

    async def test_agent_context_set_correctly(self):
        """get_context().agent should be set correctly during agent.prompt()."""
        captured: list = []

        async def capture() -> str:
            """Capture agent."""
            captured.append(get_context().agent)
            return "ok"

        class MyRole(Agent):
            system_prompts = ["Role for {module}."]
            tools = [capture]

        provider = MockProvider(canned_responses=[
            _tool_call_response("c1", "capture", "{}"),
            _text_response("done"),
        ])
        ctx = ExecutionContext(provider=provider)
        token = set_context(ctx)
        try:
            agent = MyRole(module="parser")
            await agent.prompt("do it")

            assert len(captured) == 1
            assert captured[0] is agent
            assert captured[0].module == "parser"
        finally:
            reset_context(token)

    async def test_provider_sees_prior_history(self):
        """On the second prompt() call, the provider should see messages
        from the first call."""
        seen_messages: list[list[Message]] = []

        class CapturingProvider(MockProvider):
            async def complete(self, system_prompts, tools, messages):
                seen_messages.append(list(messages))
                async for chunk in super().complete(system_prompts, tools, messages):
                    yield chunk

        provider = CapturingProvider(canned_responses=[
            _text_response("first"),
            _text_response("second"),
        ])
        ctx = ExecutionContext(provider=provider)
        token = set_context(ctx)
        try:
            agent = Agent()

            await agent.prompt("msg1")
            await agent.prompt("msg2")

            # First call: provider sees [UserMessage("msg1")]
            assert len(seen_messages[0]) == 1
            assert seen_messages[0][0].content == "msg1"

            # Second call: provider sees all prior history plus new prompt
            assert len(seen_messages[1]) == 3
            assert seen_messages[1][0].content == "msg1"
            assert seen_messages[1][2].content == "msg2"
        finally:
            reset_context(token)
