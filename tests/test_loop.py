"""Tests for thorn.core._loop — the agent loop in text and structured modes."""

from __future__ import annotations

import pytest

from thorn.core._context import (
    ExecutionContext,
    scoped_status_provider,
    set_context,
)
from thorn.core._executor import ToolVenue
from thorn.core._func import wrap_function
from thorn.core._history import AdvisoryNode, CollapseState, HistoryTree, TurnNode
from thorn.core._loop import _normalize_tool_name, run_agent_loop
from thorn.core._provider import (
    FinishChunk,
    LLMProvider,
    MockProvider,
    ResponseChunk,
    TextChunk,
    ToolCallChunk,
    UsageChunk,
)
from thorn.core._retry import RetryPolicy
from thorn.core._validation_tracker import ValidationTracker
from thorn.core.errors import (
    AgentFailureError,
    LoopLimitError,
    LoopRepetitionError,
    ProviderError,
    ProviderUnavailableError,
    RateLimitError,
    SkillError,
    TransientProviderError,
)

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

        tool = wrap_function(add, venue=ToolVenue.SANDBOX)
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

        tool = wrap_function(boom, venue=ToolVenue.SANDBOX)
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

        tool = wrap_function(count_words, venue=ToolVenue.SANDBOX)
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


class TestLoopRepetition:
    async def test_repeated_structured_text_raises_repetition_error(self):
        provider = MockProvider(canned_responses=[
            _text_response("I think true"),
            _text_response("I think true"),
            _text_response("I think true"),
        ])
        ctx = ExecutionContext(provider=provider)

        with pytest.raises(LoopRepetitionError) as exc_info:
            await run_agent_loop(
                context=ctx,
                user_prompt="is it?",
                tools=[],
                result_type=bool,
                max_tool_rounds=10,
            )

        assert exc_info.value.repetitions == 3
        assert exc_info.value.rounds == 3

    async def test_repeated_tool_error_round_raises_repetition_error(self):
        provider = MockProvider(canned_responses=[
            _tool_call_response("c1", "missing_tool", '{"value": 1}'),
            _tool_call_response("c2", "missing_tool", '{"value": 1}'),
            _tool_call_response("c3", "missing_tool", '{"value": 1}'),
        ])
        ctx = ExecutionContext(provider=provider)

        with pytest.raises(LoopRepetitionError) as exc_info:
            await run_agent_loop(
                context=ctx,
                user_prompt="observe repeatedly",
                tools=[],
                max_tool_rounds=10,
            )

        assert exc_info.value.repetitions == 3

    async def test_repeated_successful_tool_round_uses_round_limit(self):
        async def observe(value: int) -> str:
            """Observe a value."""
            return f"value={value}"

        tool = wrap_function(observe, venue=ToolVenue.SANDBOX)
        provider = MockProvider(canned_responses=[
            _tool_call_response("c1", "observe", '{"value": 1}'),
            _tool_call_response("c2", "observe", '{"value": 1}'),
            _tool_call_response("c3", "observe", '{"value": 1}'),
            _text_response("done"),
        ])
        ctx = ExecutionContext(provider=provider)

        result = await run_agent_loop(
            context=ctx,
            user_prompt="observe repeatedly",
            tools=[tool],
            max_tool_rounds=10,
        )

        assert result == "done"

    async def test_tool_argument_json_is_canonicalized_for_repetition(self):
        provider = MockProvider(canned_responses=[
            _tool_call_response("c1", "combine", '{"a": 1, "b": 2}'),
            _tool_call_response("c2", "combine", '{"b":2,"a":1}'),
            _tool_call_response("c3", "combine", '{\n  "a": 1,\n  "b": 2\n}'),
        ])
        ctx = ExecutionContext(provider=provider)

        with pytest.raises(LoopRepetitionError):
            await run_agent_loop(
                context=ctx,
                user_prompt="combine repeatedly",
                tools=[],
                max_tool_rounds=10,
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

        tool = wrap_function(read_file, venue=ToolVenue.SANDBOX)
        provider = MockProvider(canned_responses=[
            _tool_call_response("c1", "read-file", '{"path": "x.txt"}'),
            _text_response("got it"),
        ])
        ctx = ExecutionContext(provider=provider)
        result = await run_agent_loop(
            context=ctx, user_prompt="read x.txt", tools=[tool],
        )
        assert result == "got it"


# ---------------------------------------------------------------------------
# Status provider / advisory node integration
# ---------------------------------------------------------------------------

class TestValidationFooter:
    """Advisory node creation from a ValidationTracker status provider.

    These tests mirror the original footer-injection tests but verify
    that the status text now lives in ``AdvisoryNode`` instances on the
    ``TurnNode`` rather than being baked into tool-result content.
    The rendered output still splices advisory text into the last tool
    result (LLM API constraint), so the rendered assertions are
    compatible with the old behaviour.
    """

    async def test_advisory_node_created_from_tracker(self, tmp_path):
        """When a ValidationTracker is attached as a status provider,
        an AdvisoryNode is created on the TurnNode."""
        (tmp_path / "a.cpp").write_text("code")

        tracker = ValidationTracker(root=tmp_path)
        tracker.add_target("build", ["*.cpp"])

        async def greet(name: str) -> str:
            """Say hello."""
            return f"Hello, {name}!"

        wrapped = wrap_function(greet, venue=ToolVenue.SANDBOX)
        provider = MockProvider(canned_responses=[
            _tool_call_response("c1", "greet", '{"name": "world"}'),
            _text_response("done"),
        ])
        ctx = ExecutionContext(provider=provider, status_providers=[tracker])
        history = HistoryTree()
        result = await run_agent_loop(
            context=ctx, user_prompt="greet", tools=[wrapped],
            history=history,
        )
        assert result == "done"

        turn = history.nodes[1]
        assert isinstance(turn, TurnNode)
        assert len(turn.advisory_nodes) == 1
        assert turn.advisory_nodes[0].source == "validation"
        assert "[build: stale (1 files changed)]" in turn.advisory_nodes[0].content

        rendered = history.render()
        tool_results = [m for m in rendered if hasattr(m, "call_id")]
        assert len(tool_results) == 1
        assert "[build: stale (1 files changed)]" in tool_results[0].content

    async def test_no_advisory_without_providers(self):
        """Without any status providers, no advisory nodes are created."""
        async def greet(name: str) -> str:
            """Say hello."""
            return f"Hello, {name}!"

        wrapped = wrap_function(greet, venue=ToolVenue.SANDBOX)
        provider = MockProvider(canned_responses=[
            _tool_call_response("c1", "greet", '{"name": "world"}'),
            _text_response("done"),
        ])
        ctx = ExecutionContext(provider=provider)
        history = HistoryTree()
        result = await run_agent_loop(
            context=ctx, user_prompt="greet", tools=[wrapped],
            history=history,
        )
        assert result == "done"

        turn = history.nodes[1]
        assert isinstance(turn, TurnNode)
        assert turn.advisory_nodes == []

        rendered = history.render()
        tool_results = [m for m in rendered if hasattr(m, "call_id")]
        assert len(tool_results) == 1
        assert tool_results[0].content == "Hello, world!"

    async def test_advisory_reflects_passing_after_record(self, tmp_path):
        """After recording a passing result, the advisory shows passing."""
        (tmp_path / "a.cpp").write_text("code")

        tracker = ValidationTracker(root=tmp_path)
        tracker.add_target("build", ["*.cpp"])
        tracker.record_result("build", passed=True)

        async def noop() -> str:
            """Do nothing."""
            return "ok"

        wrapped = wrap_function(noop, venue=ToolVenue.SANDBOX)
        provider = MockProvider(canned_responses=[
            _tool_call_response("c1", "noop", "{}"),
            _text_response("done"),
        ])
        ctx = ExecutionContext(provider=provider, status_providers=[tracker])
        history = HistoryTree()
        result = await run_agent_loop(
            context=ctx, user_prompt="go", tools=[wrapped],
            history=history,
        )
        assert result == "done"

        rendered = history.render()
        tool_results = [m for m in rendered if hasattr(m, "call_id")]
        assert len(tool_results) == 1
        assert "[all validations passing]" in tool_results[0].content

    async def test_advisory_only_on_last_result_in_round(self, tmp_path):
        """When the LLM issues multiple tool calls in one round,
        advisory text appears only on the last tool result at render time."""
        (tmp_path / "a.cpp").write_text("code")

        tracker = ValidationTracker(root=tmp_path)
        tracker.add_target("build", ["*.cpp"])

        async def t1() -> str:
            """Tool one."""
            return "result1"

        async def t2() -> str:
            """Tool two."""
            return "result2"

        w1 = wrap_function(t1, venue=ToolVenue.SANDBOX)
        w2 = wrap_function(t2, venue=ToolVenue.SANDBOX)
        provider = MockProvider(canned_responses=[
            [
                ToolCallChunk(call_id="c1", name="t1", arguments="{}"),
                ToolCallChunk(call_id="c2", name="t2", arguments="{}"),
                FinishChunk(reason="tool_calls"),
            ],
            _text_response("done"),
        ])
        ctx = ExecutionContext(provider=provider, status_providers=[tracker])
        history = HistoryTree()
        result = await run_agent_loop(
            context=ctx, user_prompt="go", tools=[w1, w2],
            history=history,
        )
        assert result == "done"

        rendered = history.render()
        tool_results = [m for m in rendered if hasattr(m, "call_id")]
        assert len(tool_results) == 2
        assert "[build:" not in tool_results[0].content
        assert "[build: stale" in tool_results[1].content

    async def test_tracker_propagated_via_push_scope(self, tmp_path):
        """push_scope shares the same status_providers list by reference."""
        tracker = ValidationTracker(root=tmp_path)
        tracker.add_target("build", ["*.cpp"])

        provider = MockProvider()
        ctx = ExecutionContext(provider=provider, status_providers=[tracker])
        child = ctx.push_scope("child-scope")
        assert child.validation_tracker is tracker
        assert child.status_providers is ctx.status_providers


# ---------------------------------------------------------------------------
# System-prompt ordering: context.system_prompts then per-call system_prompts
# ---------------------------------------------------------------------------

class TestSystemPromptOrdering:
    """``run_agent_loop`` concatenates ``context.system_prompts`` first
    (the runtime's universal prompts) and then the per-call
    ``system_prompts`` argument (role-level prompts plus the
    context-gathering pipeline's assembled blocks)."""

    async def test_context_prompts_precede_per_call_prompts(self):
        captured: list[list[str]] = []

        class CapturingProvider(MockProvider):
            async def complete(self, system_prompts, tools, messages):
                captured.append(list(system_prompts))
                async for chunk in super().complete(system_prompts, tools, messages):
                    yield chunk

        provider = CapturingProvider(
            canned_responses=[_text_response("ok")],
        )
        ctx = ExecutionContext(
            provider=provider,
            system_prompts=["universal"],
        )
        await run_agent_loop(
            context=ctx,
            user_prompt="hi",
            tools=[],
            system_prompts=["agent-class", "agent-instance"],
        )
        assert len(captured) == 1
        assert captured[0] == ["universal", "agent-class", "agent-instance"]

    async def test_no_per_call_prompts(self):
        captured: list[list[str]] = []

        class CapturingProvider(MockProvider):
            async def complete(self, system_prompts, tools, messages):
                captured.append(list(system_prompts))
                async for chunk in super().complete(system_prompts, tools, messages):
                    yield chunk

        provider = CapturingProvider(
            canned_responses=[_text_response("ok")],
        )
        ctx = ExecutionContext(
            provider=provider,
            system_prompts=["universal"],
        )
        await run_agent_loop(
            context=ctx,
            user_prompt="hi",
            tools=[],
            system_prompts=[],
        )
        assert len(captured) == 1
        assert captured[0] == ["universal"]


# ---------------------------------------------------------------------------
# AdvisoryNode unit tests
# ---------------------------------------------------------------------------

class TestAdvisoryNode:
    """Direct tests for AdvisoryNode behavior independent of the loop."""

    def test_defaults(self):
        node = AdvisoryNode(source="test", content="hello world")
        assert node.source == "test"
        assert node.content == "hello world"
        assert node.collapse_state == CollapseState.EXPANDED
        assert node.intrinsic_salience == 0.2

    def test_summary(self):
        node = AdvisoryNode(source="inbox", content="3 unread messages")
        assert node.summary() == "[inbox advisory]"

    def test_token_cost_expanded_vs_collapsed(self):
        node = AdvisoryNode(source="validation", content="x" * 400)
        expanded = node.expanded_token_cost()
        collapsed = node.collapsed_token_cost()
        assert expanded > collapsed
        assert node.token_cost() == expanded

        node.collapse_state = CollapseState.COLLAPSED
        assert node.token_cost() == collapsed

    def test_savings_if_collapsed(self):
        node = AdvisoryNode(source="test", content="x" * 400)
        savings = node.savings_if_collapsed()
        assert savings > 0

    def test_is_collapsible(self):
        node = AdvisoryNode(source="test", content="x" * 400)
        assert node.is_collapsible

        node.collapse_state = CollapseState.COLLAPSED
        assert not node.is_collapsible


# ---------------------------------------------------------------------------
# TurnNode advisory integration
# ---------------------------------------------------------------------------

class TestTurnNodeAdvisory:
    """Tests for TurnNode with advisory nodes."""

    def test_render_appends_advisory_to_last_tool_result(self):

        tcn = _make_tool_call_node("c1", "greet", "ok")
        adv = AdvisoryNode(source="validation", content="[build: stale]")
        turn = TurnNode(
            assistant_content="",
            tool_call_nodes=[tcn],
            advisory_nodes=[adv],
        )

        msgs = turn.render()
        tool_results = [m for m in msgs if hasattr(m, "call_id")]
        assert len(tool_results) == 1
        assert tool_results[0].content == "ok\n\n[build: stale]"

    def test_render_no_advisories_leaves_result_clean(self):
        tcn = _make_tool_call_node("c1", "greet", "ok")
        turn = TurnNode(
            assistant_content="",
            tool_call_nodes=[tcn],
        )
        msgs = turn.render()
        tool_results = [m for m in msgs if hasattr(m, "call_id")]
        assert tool_results[0].content == "ok"

    def test_collapsed_advisory_renders_summary(self):
        tcn = _make_tool_call_node("c1", "greet", "ok")
        adv = AdvisoryNode(source="inbox", content="3 messages pending")
        adv.collapse_state = CollapseState.COLLAPSED
        turn = TurnNode(
            assistant_content="",
            tool_call_nodes=[tcn],
            advisory_nodes=[adv],
        )
        msgs = turn.render()
        tool_results = [m for m in msgs if hasattr(m, "call_id")]
        assert "[inbox advisory]" in tool_results[0].content

    def test_summary_includes_advisory_labels(self):
        tcn = _make_tool_call_node("c1", "greet", "ok")
        adv = AdvisoryNode(source="validation", content="stale")
        turn = TurnNode(
            assistant_content="",
            tool_call_nodes=[tcn],
            advisory_nodes=[adv],
        )
        summary = turn.summary()
        assert "advisories: validation" in summary

    def test_token_cost_includes_advisories(self):
        tcn = _make_tool_call_node("c1", "greet", "ok")
        adv = AdvisoryNode(source="test", content="x" * 200)
        turn_without = TurnNode(
            assistant_content="",
            tool_call_nodes=[tcn],
        )
        turn_with = TurnNode(
            assistant_content="",
            tool_call_nodes=[tcn],
            advisory_nodes=[adv],
        )
        assert turn_with.token_cost() > turn_without.token_cost()


# ---------------------------------------------------------------------------
# Compaction with advisories
# ---------------------------------------------------------------------------

class TestAdvisoryCompaction:
    def test_advisory_collapsed_before_turn(self):
        """Advisory nodes with low salience should be collapsed before
        their parent turn node."""
        history = HistoryTree()
        history.append_user_prompt("do something")

        adv = AdvisoryNode(source="test", content="x" * 500)

        turn = TurnNode(
            assistant_content="text",
            tool_call_nodes=[_make_tool_call_node("c1", "t", "x" * 500)],
            advisory_nodes=[adv],
        )
        history.nodes.append(turn)

        assert adv.collapse_state == CollapseState.EXPANDED

        history.compact(
            context_budget=100,
            protected_tail_nodes=0,
            protected_tail_tool_calls=0,
        )
        assert adv.collapse_state == CollapseState.COLLAPSED


# ---------------------------------------------------------------------------
# Serialization round-trip
# ---------------------------------------------------------------------------

class TestAdvisorySerialization:
    def test_round_trip(self):
        from thorn.runtime._serializer import deserialize_history, serialize_history

        history = HistoryTree()
        history.append_user_prompt("test")

        adv = AdvisoryNode(source="validation", content="[build: stale]")
        turn = TurnNode(
            assistant_content="ok",
            tool_call_nodes=[_make_tool_call_node("c1", "t", "result")],
            advisory_nodes=[adv],
        )
        history.nodes.append(turn)

        data = serialize_history(history)
        restored = deserialize_history(data)

        restored_turn = restored.nodes[1]
        assert isinstance(restored_turn, TurnNode)
        assert len(restored_turn.advisory_nodes) == 1
        assert restored_turn.advisory_nodes[0].source == "validation"
        assert restored_turn.advisory_nodes[0].content == "[build: stale]"

    def test_backward_compat_no_advisories_key(self):
        """Old history files without an 'advisories' key deserialize fine."""
        from thorn.runtime._serializer import deserialize_history

        data = [{
            "type": "turn",
            "assistant_content": "hello",
            "tool_calls": [],
        }]
        tree = deserialize_history(data)
        turn = tree.nodes[0]
        assert isinstance(turn, TurnNode)
        assert turn.advisory_nodes == []


# ---------------------------------------------------------------------------
# Multi-provider scenarios
# ---------------------------------------------------------------------------

class TestMultiProvider:
    async def test_multiple_status_providers(self):
        """Multiple status providers each produce their own AdvisoryNode."""
        class FakeProvider:
            def __init__(self, label: str, text: str):
                self._label = label
                self._text = text

            @property
            def source_label(self) -> str:
                return self._label

            def refresh(self, session=None) -> None:
                pass

            def render_status(self, session=None) -> str | None:
                return self._text

        p1 = FakeProvider("alpha", "status alpha")
        p2 = FakeProvider("beta", "status beta")

        async def noop() -> str:
            """Noop."""
            return "ok"

        wrapped = wrap_function(noop, venue=ToolVenue.SANDBOX)
        provider = MockProvider(canned_responses=[
            _tool_call_response("c1", "noop", "{}"),
            _text_response("done"),
        ])
        ctx = ExecutionContext(provider=provider, status_providers=[p1, p2])
        history = HistoryTree()
        await run_agent_loop(
            context=ctx, user_prompt="go", tools=[wrapped],
            history=history,
        )
        turn = history.nodes[1]
        assert isinstance(turn, TurnNode)
        assert len(turn.advisory_nodes) == 2
        assert turn.advisory_nodes[0].source == "alpha"
        assert turn.advisory_nodes[1].source == "beta"

    async def test_provider_returning_none_skipped(self):
        """A provider that returns None from render_status produces
        no AdvisoryNode."""
        class SilentProvider:
            @property
            def source_label(self) -> str:
                return "silent"

            def refresh(self, session=None) -> None:
                pass

            def render_status(self, session=None) -> str | None:
                return None

        async def noop() -> str:
            """Noop."""
            return "ok"

        wrapped = wrap_function(noop, venue=ToolVenue.SANDBOX)
        provider = MockProvider(canned_responses=[
            _tool_call_response("c1", "noop", "{}"),
            _text_response("done"),
        ])
        ctx = ExecutionContext(provider=provider, status_providers=[SilentProvider()])
        history = HistoryTree()
        await run_agent_loop(
            context=ctx, user_prompt="go", tools=[wrapped],
            history=history,
        )
        turn = history.nodes[1]
        assert isinstance(turn, TurnNode)
        assert turn.advisory_nodes == []


# ---------------------------------------------------------------------------
# Backward-compat validation_tracker property
# ---------------------------------------------------------------------------

class TestValidationTrackerCompat:
    def test_property_getter_finds_tracker(self, tmp_path):
        tracker = ValidationTracker(root=tmp_path)
        ctx = ExecutionContext(
            provider=MockProvider(),
            status_providers=[tracker],
        )
        assert ctx.validation_tracker is tracker

    def test_property_getter_returns_none_when_absent(self):
        ctx = ExecutionContext(provider=MockProvider())
        assert ctx.validation_tracker is None

    def test_property_setter_adds_tracker(self, tmp_path):
        tracker = ValidationTracker(root=tmp_path)
        ctx = ExecutionContext(provider=MockProvider())
        ctx.validation_tracker = tracker
        assert tracker in ctx.status_providers
        assert ctx.validation_tracker is tracker

    def test_property_setter_replaces_existing_tracker(self, tmp_path):
        old = ValidationTracker(root=tmp_path)
        new = ValidationTracker(root=tmp_path)
        ctx = ExecutionContext(provider=MockProvider(), status_providers=[old])
        ctx.validation_tracker = new
        assert old not in ctx.status_providers
        assert new in ctx.status_providers

    def test_property_setter_none_removes_tracker(self, tmp_path):
        tracker = ValidationTracker(root=tmp_path)
        ctx = ExecutionContext(provider=MockProvider(), status_providers=[tracker])
        ctx.validation_tracker = None
        assert ctx.validation_tracker is None
        assert tracker not in ctx.status_providers


# ---------------------------------------------------------------------------
# scoped_status_provider
# ---------------------------------------------------------------------------

class TestScopedStatusProvider:
    def test_provider_added_and_removed(self, tmp_path):
        tracker = ValidationTracker(root=tmp_path)
        ctx = ExecutionContext(provider=MockProvider())
        token = set_context(ctx)
        try:
            assert tracker not in ctx.status_providers
            with scoped_status_provider(tracker):
                assert tracker in ctx.status_providers
            assert tracker not in ctx.status_providers
        finally:
            from thorn.core._context import reset_context
            reset_context(token)

    def test_provider_removed_on_exception(self, tmp_path):
        tracker = ValidationTracker(root=tmp_path)
        ctx = ExecutionContext(provider=MockProvider())
        token = set_context(ctx)
        try:
            with pytest.raises(RuntimeError):
                with scoped_status_provider(tracker):
                    assert tracker in ctx.status_providers
                    raise RuntimeError("boom")
            assert tracker not in ctx.status_providers
        finally:
            from thorn.core._context import reset_context
            reset_context(token)


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

def _make_tool_call_node(call_id: str, name: str, result_content: str):
    """Create a simple ToolCallNode for testing."""
    from thorn.core._history import ToolCallNode
    from thorn.core._messages import ToolCall, ToolResultMessage

    tc = ToolCall(call_id=call_id, name=name, arguments="{}")
    result = ToolResultMessage(call_id=call_id, content=result_content)
    return ToolCallNode(tc, result)


# ---------------------------------------------------------------------------
# Provider retry behaviour
# ---------------------------------------------------------------------------

class _FlakyProvider(LLMProvider):
    """Provider that raises a scripted sequence of exceptions, then succeeds.

    Each entry in ``errors`` is raised on the corresponding call to
    :meth:`complete`; once the list is exhausted the provider
    behaves like :class:`MockProvider` and returns
    ``success_chunks`` (a canned sequence of response chunks).
    Used to exercise the retry loop without pulling in real
    ``httpx`` machinery.
    """

    def __init__(
        self,
        errors: list[BaseException],
        *,
        success_chunks: list[ResponseChunk] | None = None,
    ) -> None:
        self._errors = list(errors)
        self._success_chunks = success_chunks or [
            TextChunk(text="ok"),
            UsageChunk(
                prompt_tokens=0, completion_tokens=0, total_tokens=0,
            ),
            FinishChunk(reason="stop"),
        ]
        self.call_count = 0

    async def complete(self, system_prompts, tools, messages):
        self.call_count += 1
        if self._errors:
            exc = self._errors.pop(0)
            # ``AsyncGenerator`` bodies must contain at least one
            # ``yield`` for Python to treat them as generators; an
            # unconditional ``raise`` before the yield still makes
            # this a valid async generator because the ``yield``
            # below is unreachable but statically present.
            raise exc
            yield  # pragma: no cover -- see above
        for chunk in self._success_chunks:
            yield chunk


@pytest.fixture
def fast_retry_policy(monkeypatch):
    """Patch the module-level retry policy to wait essentially no time.

    Tests that exercise retry counting do not care about the real
    sleep durations; a policy with ``base`` and ``cap`` of a few
    milliseconds keeps the suite fast while still going through
    the real backoff code path.
    """
    import thorn.core._loop as loop_mod

    fast = RetryPolicy(
        base=0.001,
        cap=0.005,
        max_rate_limit_retries=3,
        max_transient_retries=3,
        retry_after_jitter=0.0,
    )
    monkeypatch.setattr(loop_mod, "_DEFAULT_RETRY_POLICY", fast)
    return fast


class TestTransientRetry:
    """``TransientProviderError`` should be absorbed by the call-site
    retry loop up to the policy's transient budget, and only surface
    as :class:`ProviderUnavailableError` on exhaustion."""

    async def test_transient_then_success(self, fast_retry_policy):
        provider = _FlakyProvider(
            errors=[
                TransientProviderError("disconnect 1"),
                TransientProviderError("disconnect 2"),
            ],
        )
        ctx = ExecutionContext(provider=provider)
        result = await run_agent_loop(
            context=ctx, user_prompt="hi", tools=[],
        )
        assert result == "ok"
        assert provider.call_count == 3

    async def test_transient_exhausted_raises_unavailable(
        self, fast_retry_policy,
    ):
        # One more error than the transient budget: exhausts the
        # inner retry and surfaces as ``ProviderUnavailableError``,
        # NOT as ``AgentFailureError`` (which is reserved for
        # non-transient failures).
        budget = fast_retry_policy.max_transient_retries
        errors = [
            TransientProviderError(f"blip {i}")
            for i in range(budget + 1)
        ]
        provider = _FlakyProvider(errors=errors)
        ctx = ExecutionContext(provider=provider)
        with pytest.raises(ProviderUnavailableError) as exc_info:
            await run_agent_loop(
                context=ctx, user_prompt="hi", tools=[],
            )
        assert exc_info.value.attempts == budget + 1
        assert provider.call_count == budget + 1


class TestRateLimitRetry:
    async def test_rate_limit_then_success(self, fast_retry_policy):
        provider = _FlakyProvider(
            errors=[RateLimitError("slow down")],
        )
        ctx = ExecutionContext(provider=provider)
        result = await run_agent_loop(
            context=ctx, user_prompt="hi", tools=[],
        )
        assert result == "ok"
        assert provider.call_count == 2

    async def test_rate_limit_exhausted_raises_unavailable(
        self, fast_retry_policy,
    ):
        budget = fast_retry_policy.max_rate_limit_retries
        errors = [
            RateLimitError(f"slow {i}") for i in range(budget + 1)
        ]
        provider = _FlakyProvider(errors=errors)
        ctx = ExecutionContext(provider=provider)
        with pytest.raises(ProviderUnavailableError):
            await run_agent_loop(
                context=ctx, user_prompt="hi", tools=[],
            )

    async def test_retry_after_is_honoured(self, monkeypatch):
        # A long ``Retry-After`` value must flow into the sleep
        # call as a floor.  We swap ``asyncio.sleep`` out for a
        # recorder so the test does not actually wait that long.
        import thorn.core._loop as loop_mod

        fast = RetryPolicy(
            base=0.001, cap=0.005,
            max_rate_limit_retries=3, max_transient_retries=3,
            retry_after_jitter=0.0,
        )
        monkeypatch.setattr(loop_mod, "_DEFAULT_RETRY_POLICY", fast)

        sleeps: list[float] = []

        async def _record_sleep(delay):
            sleeps.append(delay)

        monkeypatch.setattr(loop_mod.asyncio, "sleep", _record_sleep)

        provider = _FlakyProvider(
            errors=[RateLimitError("slow", retry_after=7.5)],
        )
        ctx = ExecutionContext(provider=provider)
        result = await run_agent_loop(
            context=ctx, user_prompt="hi", tools=[],
        )
        assert result == "ok"
        # Exactly one backoff sleep (for the single rate-limit
        # retry) and it must be >= retry_after.
        assert len(sleeps) == 1
        assert sleeps[0] >= 7.5


class TestNonTransientProviderError:
    """Raw :class:`ProviderError` (neither transient nor rate-limit)
    keeps the old behavior: counted against ``max_failures`` and
    raising :class:`AgentFailureError` on exhaustion.  This is the
    path for things like HTTP 400 "bad request" that will not clear
    by waiting."""

    async def test_provider_error_exhausted_raises_agent_failure(
        self, fast_retry_policy,
    ):
        provider = _FlakyProvider(
            errors=[
                ProviderError(f"hard fail {i}") for i in range(6)
            ],
        )
        ctx = ExecutionContext(provider=provider)
        with pytest.raises(AgentFailureError) as exc_info:
            await run_agent_loop(
                context=ctx, user_prompt="hi", tools=[],
                max_failures=5,
            )
        assert exc_info.value.failures == 5

    async def test_provider_error_then_success_resets_budget(
        self, fast_retry_policy,
    ):
        # Four transient hard failures (below the default
        # max_failures=5) then success: the loop should recover
        # and return a result without raising.
        provider = _FlakyProvider(
            errors=[ProviderError(f"fail {i}") for i in range(4)],
        )
        ctx = ExecutionContext(provider=provider)
        result = await run_agent_loop(
            context=ctx, user_prompt="hi", tools=[],
        )
        assert result == "ok"
        assert provider.call_count == 5
