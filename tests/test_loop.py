"""Tests for thorn._loop — the agent loop in text and structured modes."""

from __future__ import annotations

import pytest

from thorn._context import ExecutionContext
from thorn._func import wrap_function
from thorn._loop import _WrappedTool, _normalize_tool_name, run_agent_loop
from thorn._provider import FinishChunk, MockProvider, TextChunk, ToolCallChunk
from thorn.errors import LoopLimitError, SkillError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _text_response(text: str):
    """A canned response that is just text."""
    return [TextChunk(text=text), FinishChunk(reason="stop")]


def _tool_call_response(call_id: str, name: str, arguments: str):
    """A canned response that is a single tool call."""
    return [ToolCallChunk(call_id=call_id, name=name, arguments=arguments),
            FinishChunk(reason="tool_calls")]


# ---------------------------------------------------------------------------
# Text mode
# ---------------------------------------------------------------------------

class TestTextMode:
    async def test_simple_text_response(self):
        provider = MockProvider(canned_responses=[_text_response("hello")])
        ctx = ExecutionContext(provider=provider)
        result = await run_agent_loop(
            context=ctx, user_prompt="say hello", tools=[],
        )
        assert result == "hello"

    async def test_tool_use_then_text(self):
        """Agent calls a tool, gets the result, then gives a text answer."""
        call_log = []

        async def add(a: int, b: int) -> int:
            """Add two numbers."""
            call_log.append((a, b))
            return a + b

        tool = wrap_function(add)
        provider = MockProvider(canned_responses=[
            _tool_call_response("c1", "add", '{"a": 3, "b": 4}'),
            _text_response("The sum is 7"),
        ])
        ctx = ExecutionContext(provider=provider)
        result = await run_agent_loop(
            context=ctx, user_prompt="add 3+4", tools=[tool],
        )
        assert result == "The sum is 7"
        assert call_log == [(3, 4)]

    async def test_unknown_tool_reports_error_to_agent(self):
        """Unknown tool name → error sent back, agent then responds."""
        provider = MockProvider(canned_responses=[
            _tool_call_response("c1", "nonexistent", "{}"),
            _text_response("sorry, I could not do that"),
        ])
        ctx = ExecutionContext(provider=provider)
        result = await run_agent_loop(
            context=ctx, user_prompt="do it", tools=[],
        )
        assert result == "sorry, I could not do that"

    async def test_raise_error_in_text_mode(self):
        """raise_error is available even in text mode (result_type=str)."""
        provider = MockProvider(canned_responses=[
            _tool_call_response("c1", "raise_error", '{"message": "blocked by dep"}'),
        ])
        ctx = ExecutionContext(provider=provider)
        with pytest.raises(SkillError, match="blocked by dep"):
            await run_agent_loop(
                context=ctx, user_prompt="do it", tools=[],
            )

    async def test_tool_exception_reports_error_to_agent(self):
        """Exception in a tool → error result sent back."""
        async def boom() -> str:
            """Blow up."""
            raise ValueError("kaboom")

        tool = wrap_function(boom)
        provider = MockProvider(canned_responses=[
            _tool_call_response("c1", "boom", "{}"),
            _text_response("that failed"),
        ])
        ctx = ExecutionContext(provider=provider)
        result = await run_agent_loop(
            context=ctx, user_prompt="run boom", tools=[tool],
        )
        assert result == "that failed"


# ---------------------------------------------------------------------------
# Structured mode (return_result / raise_error)
# ---------------------------------------------------------------------------

class TestStructuredMode:
    async def test_return_result_bool(self):
        provider = MockProvider(canned_responses=[
            _tool_call_response("c1", "return_result", '{"value": true}'),
        ])
        ctx = ExecutionContext(provider=provider)
        result = await run_agent_loop(
            context=ctx, user_prompt="is it?", tools=[],
            result_type=bool,
        )
        assert result is True

    async def test_return_result_list(self):
        provider = MockProvider(canned_responses=[
            _tool_call_response("c1", "return_result", '{"value": ["a.py", "b.py"]}'),
        ])
        ctx = ExecutionContext(provider=provider)
        result = await run_agent_loop(
            context=ctx, user_prompt="list files", tools=[],
            result_type=list[str],
        )
        assert result == ["a.py", "b.py"]

    async def test_raise_error_becomes_skill_error(self):
        provider = MockProvider(canned_responses=[
            _tool_call_response("c1", "raise_error", '{"message": "no can do"}'),
        ])
        ctx = ExecutionContext(provider=provider)
        with pytest.raises(SkillError, match="no can do"):
            await run_agent_loop(
                context=ctx, user_prompt="do it", tools=[],
                result_type=bool,
            )

    async def test_text_response_nudged_then_tool(self):
        """In structured mode, a text-only response triggers a nudge
        message, after which the agent uses return_result."""
        provider = MockProvider(canned_responses=[
            _text_response("I think true"),  # wrong — should use tool
            _tool_call_response("c1", "return_result", '{"value": true}'),
        ])
        ctx = ExecutionContext(provider=provider)
        result = await run_agent_loop(
            context=ctx, user_prompt="is it?", tools=[],
            result_type=bool,
        )
        assert result is True

    async def test_validation_error_retried(self):
        """If return_result value doesn't validate, the error is sent back
        and the agent can try again."""
        provider = MockProvider(canned_responses=[
            _tool_call_response("c1", "return_result", '{"value": "not a bool"}'),
            _tool_call_response("c2", "return_result", '{"value": false}'),
        ])
        ctx = ExecutionContext(provider=provider)
        result = await run_agent_loop(
            context=ctx, user_prompt="is it?", tools=[],
            result_type=bool,
        )
        assert result is False

    async def test_tool_use_before_return_result(self):
        """Agent can use regular tools before finally calling return_result."""
        async def count_words(text: str) -> int:
            """Count words."""
            return len(text.split())

        tool = wrap_function(count_words)
        provider = MockProvider(canned_responses=[
            _tool_call_response("c1", "count_words", '{"text": "one two three"}'),
            _tool_call_response("c2", "return_result", '{"value": 3}'),
        ])
        ctx = ExecutionContext(provider=provider)
        result = await run_agent_loop(
            context=ctx, user_prompt="how many?", tools=[tool],
            result_type=int,
        )
        assert result == 3


# ---------------------------------------------------------------------------
# Loop limit
# ---------------------------------------------------------------------------

class TestLoopLimit:
    async def test_loop_limit_exceeded(self):
        """If the agent never finishes, LoopLimitError is raised."""
        provider = MockProvider(canned_responses=[
            _tool_call_response("c1", "return_result", '{"value": "bad"}'),
        ] * 5)
        ctx = ExecutionContext(provider=provider)
        with pytest.raises(LoopLimitError):
            await run_agent_loop(
                context=ctx, user_prompt="x", tools=[],
                result_type=bool, max_tool_rounds=5,
            )


# ---------------------------------------------------------------------------
# Tool name normalization
# ---------------------------------------------------------------------------

class TestToolNameNormalization:
    def test_hyphens_to_underscores(self):
        assert _normalize_tool_name("read-file") == "read_file"

    def test_case_insensitive(self):
        assert _normalize_tool_name("ReadFile") == "readfile"

    async def test_fuzzy_name_match_in_dispatch(self):
        """Agent uses 'read-file' but tool is registered as 'read_file'."""
        async def read_file(path: str) -> str:
            """Read a file."""
            return "contents"

        tool = wrap_function(read_file)
        provider = MockProvider(canned_responses=[
            _tool_call_response("c1", "read-file", '{"path": "x.txt"}'),
            _text_response("got it"),
        ])
        ctx = ExecutionContext(provider=provider)
        result = await run_agent_loop(
            context=ctx, user_prompt="read x.txt", tools=[tool],
        )
        assert result == "got it"
