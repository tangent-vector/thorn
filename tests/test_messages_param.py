"""Tests for the ``messages`` parameter on run_agent_loop and agent.prompt.

Verifies that conversation history accumulates across calls when a mutable
list is passed, and that the default (None) remains backwards-compatible.
"""

from __future__ import annotations

import pytest

from thorn._agent import Agent
from thorn._context import ExecutionContext, get_context, reset_context, set_context
from thorn._func import wrap_function
from thorn._loop import run_agent_loop
from thorn._messages import AssistantMessage, Message, ToolResultMessage, UserMessage
from thorn._provider import FinishChunk, MockProvider, TextChunk, ToolCallChunk


def _text_response(text: str):
    return [TextChunk(text=text), FinishChunk(reason="stop")]


def _tool_call_response(call_id: str, name: str, arguments: str):
    return [
        ToolCallChunk(call_id=call_id, name=name, arguments=arguments),
        FinishChunk(reason="tool_calls"),
    ]


# ---------------------------------------------------------------------------
# run_agent_loop: messages parameter
# ---------------------------------------------------------------------------


class TestRunAgentLoopMessages:
    async def test_none_is_backwards_compatible(self):
        """When messages=None, behaviour is identical to before."""
        provider = MockProvider(canned_responses=[_text_response("hi")])
        ctx = ExecutionContext(provider=provider)
        result = await run_agent_loop(
            context=ctx, user_prompt="hello", tools=[], messages=None,
        )
        assert result == "hi"

    async def test_provided_list_accumulates_history(self):
        """A provided list should contain the user message and assistant
        response after the call."""
        provider = MockProvider(canned_responses=[_text_response("reply")])
        ctx = ExecutionContext(provider=provider)
        history: list[Message] = []

        result = await run_agent_loop(
            context=ctx, user_prompt="hello", tools=[], messages=history,
        )
        assert result == "reply"
        assert len(history) == 2
        assert isinstance(history[0], UserMessage)
        assert history[0].content == "hello"
        assert isinstance(history[1], AssistantMessage)
        assert history[1].content == "reply"

    async def test_multi_turn_accumulation(self):
        """Calling run_agent_loop twice with the same list accumulates
        both turns."""
        provider = MockProvider(canned_responses=[
            _text_response("first reply"),
            _text_response("second reply"),
        ])
        ctx = ExecutionContext(provider=provider)
        history: list[Message] = []

        await run_agent_loop(
            context=ctx, user_prompt="turn 1", tools=[], messages=history,
        )
        assert len(history) == 2

        await run_agent_loop(
            context=ctx, user_prompt="turn 2", tools=[], messages=history,
        )
        assert len(history) == 4
        assert isinstance(history[2], UserMessage)
        assert history[2].content == "turn 2"
        assert isinstance(history[3], AssistantMessage)
        assert history[3].content == "second reply"

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
        history: list[Message] = []

        await run_agent_loop(
            context=ctx, user_prompt="double 5", tools=[tool],
            messages=history,
        )

        assert isinstance(history[0], UserMessage)
        assert isinstance(history[1], AssistantMessage)
        assert len(history[1].tool_calls) == 1
        assert isinstance(history[2], ToolResultMessage)
        assert history[2].content == "10"
        assert isinstance(history[3], AssistantMessage)
        assert history[3].content == "result is 10"

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
        history: list[Message] = []

        await run_agent_loop(
            context=ctx, user_prompt="msg1", tools=[], messages=history,
        )
        await run_agent_loop(
            context=ctx, user_prompt="msg2", tools=[], messages=history,
        )

        # First call: provider sees [UserMessage("msg1")]
        assert len(seen_messages[0]) == 1
        assert seen_messages[0][0].content == "msg1"

        # Second call: provider sees all 4 messages (2 from first turn + user + ...)
        # At the point of the second completion call, history has:
        # [user:msg1, assistant:first, user:msg2]
        assert len(seen_messages[1]) == 3
        assert seen_messages[1][0].content == "msg1"
        assert seen_messages[1][2].content == "msg2"


# ---------------------------------------------------------------------------
# agent.prompt: messages parameter
# ---------------------------------------------------------------------------


class TestAgentPromptMessages:
    async def test_text_mode_accumulates(self):
        provider = MockProvider(canned_responses=[
            _text_response("wrote code"),
            _text_response("fixed errors"),
        ])
        ctx = ExecutionContext(provider=provider)
        token = set_context(ctx)
        try:
            agent = Agent()
            history: list[Message] = []

            await agent.prompt("write code", messages=history)
            assert len(history) == 2

            await agent.prompt("fix build errors", messages=history)
            assert len(history) == 4
            assert isinstance(history[2], UserMessage)
            assert history[2].content == "fix build errors"
        finally:
            reset_context(token)

    async def test_structured_mode_accumulates(self):
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
            history: list[Message] = []

            r1 = await agent.prompt[int]("count things", messages=history)
            assert r1 == 42
            first_turn_len = len(history)
            assert first_turn_len >= 2  # at least user + assistant

            r2 = await agent.prompt[int]("count more", messages=history)
            assert r2 == 99
            assert len(history) > first_turn_len
        finally:
            reset_context(token)

    async def test_none_is_backwards_compatible(self):
        provider = MockProvider(canned_responses=[_text_response("ok")])
        ctx = ExecutionContext(provider=provider)
        token = set_context(ctx)
        try:
            agent = Agent()
            result = await agent.prompt("hello", messages=None)
            assert result == "ok"

            result2 = await agent.prompt("hello again")
            assert isinstance(result2, str)
        finally:
            reset_context(token)

    async def test_agent_context_set_with_messages(self):
        """get_context().agent should still be set correctly when using messages."""
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
            history: list[Message] = []
            await agent.prompt("do it", messages=history)

            assert len(captured) == 1
            assert captured[0] is agent
            assert captured[0].module == "parser"
        finally:
            reset_context(token)
