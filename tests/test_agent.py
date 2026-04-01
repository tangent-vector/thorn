"""Tests for thorn._agent — Agent base class, MRO collection, prompt accessor."""

from __future__ import annotations

import pytest

from thorn._agent import Agent, _SafeDict
from thorn._context import (
    ExecutionContext,
    get_context,
    reset_context,
    set_context,
)
from thorn._func import skill
from thorn._provider import FinishChunk, MockProvider, TextChunk, ToolCallChunk


# ---------------------------------------------------------------------------
# _SafeDict
# ---------------------------------------------------------------------------


class TestSafeDict:
    def test_present_key(self):
        d = _SafeDict({"x": 1})
        assert d["x"] == 1

    def test_missing_key_returns_template_placeholder(self):
        d = _SafeDict({})
        assert d["missing"] == "{missing}"


# ---------------------------------------------------------------------------
# MRO-based collection
# ---------------------------------------------------------------------------


class TestMROCollection:
    def test_single_inheritance_prompts(self):
        class Base(Agent):
            system_prompts = ["base prompt"]

        class Child(Base):
            system_prompts = ["child prompt"]

        assert Child._collect_system_prompts() == ["base prompt", "child prompt"]

    def test_single_inheritance_tools(self):
        def tool_a() -> str:
            """A."""

        def tool_b() -> str:
            """B."""

        class Base(Agent):
            tools = [tool_a]

        class Child(Base):
            tools = [tool_b]

        assert Child._collect_tools() == [tool_a, tool_b]

    def test_child_without_own_prompts_inherits_parent(self):
        """A subclass that doesn't declare system_prompts in its own __dict__
        should NOT cause parent prompts to be collected twice."""

        class Base(Agent):
            system_prompts = ["base"]

        class Child(Base):
            pass

        assert Child._collect_system_prompts() == ["base"]

    def test_child_without_own_tools_inherits_parent(self):
        def tool_a() -> str:
            """A."""

        class Base(Agent):
            tools = [tool_a]

        class Child(Base):
            pass

        assert Child._collect_tools() == [tool_a]

    def test_diamond_no_double_counting(self):
        """In diamond inheritance, each class's prompts appear exactly once,
        in MRO order."""

        class Base(Agent):
            system_prompts = ["base"]

        class Left(Base):
            system_prompts = ["left"]

        class Right(Base):
            system_prompts = ["right"]

        class Diamond(Left, Right):
            system_prompts = ["diamond"]

        # reversed MRO: object, Agent, Base, Right, Left, Diamond
        # Agent has [] in __dict__, Base has ["base"], Right has ["right"],
        # Left has ["left"], Diamond has ["diamond"]
        assert Diamond._collect_system_prompts() == [
            "base", "right", "left", "diamond",
        ]

    def test_tool_dedup_by_name(self):
        def shared() -> str:
            """Shared."""

        class Base(Agent):
            tools = [shared]

        class Child(Base):
            tools = [shared]

        collected = Child._collect_tools()
        assert len(collected) == 1
        assert collected[0] is shared

    def test_tool_dedup_preserves_first_occurrence(self):
        """When different functions share a __name__, the first (outermost MRO)
        wins."""

        def helper() -> str:
            """V1."""
            return "v1"

        def helper_v2() -> str:
            """V2."""
            return "v2"

        helper_v2.__name__ = "helper"

        class Base(Agent):
            tools = [helper]

        class Child(Base):
            tools = [helper_v2]

        collected = Child._collect_tools()
        assert len(collected) == 1
        assert collected[0] is helper

    def test_bare_agent_has_empty_collections(self):
        assert Agent._collect_system_prompts() == []
        assert Agent._collect_tools() == []


# ---------------------------------------------------------------------------
# Template rendering
# ---------------------------------------------------------------------------


class TestTemplateRendering:
    def test_simple_substitution(self):
        class MyAgent(Agent):
            system_prompts = ["Working on {module}."]

        agent = MyAgent(module="parser")
        assert agent._render_system_prompts() == ["Working on parser."]

    def test_missing_key_preserved(self):
        class MyAgent(Agent):
            system_prompts = ["Working on {module} in {project}."]

        agent = MyAgent(module="parser")
        assert agent._render_system_prompts() == [
            "Working on parser in {project}.",
        ]

    def test_multiple_attributes(self):
        class MyAgent(Agent):
            system_prompts = ["{role}@{module}"]

        agent = MyAgent(role="architect", module="parser.lexer")
        assert agent._render_system_prompts() == ["architect@parser.lexer"]

    def test_inherited_prompts_rendered(self):
        class Base(Agent):
            system_prompts = ["Base: {module}"]

        class Child(Base):
            system_prompts = ["Child: {module} with {extra}"]

        agent = Child(module="parser", extra="detail")
        rendered = agent._render_system_prompts()
        assert rendered == ["Base: parser", "Child: parser with detail"]

    def test_private_attrs_excluded(self):
        class MyAgent(Agent):
            system_prompts = ["{_secret} {public}"]

        agent = MyAgent(public="yes")
        agent._secret = "hidden"
        rendered = agent._render_system_prompts()
        assert rendered == ["{_secret} yes"]


# ---------------------------------------------------------------------------
# agent.prompt (text mode)
# ---------------------------------------------------------------------------


class TestAgentPromptTextMode:
    async def test_basic_prompt(self, ctx):
        class SimpleAgent(Agent):
            system_prompts = ["You are a helper."]

        agent = SimpleAgent()
        result = await agent.prompt("say hello")
        assert isinstance(result, str)
        assert "say hello" in result

    async def test_prompt_with_agent_tools(self, ctx):
        async def helper(x: int) -> int:
            """Double a number."""
            return x * 2

        class ToolAgent(Agent):
            system_prompts = ["Use tools."]
            tools = [helper]

        agent = ToolAgent()
        result = await agent.prompt("help me")
        assert isinstance(result, str)

    async def test_prompt_with_extra_tools(self, ctx):
        async def extra_tool() -> str:
            """Extra."""
            return "extra"

        class SimpleAgent(Agent):
            pass

        agent = SimpleAgent()
        result = await agent.prompt("do it", tools=[extra_tool])
        assert isinstance(result, str)

    async def test_prompt_with_extra_system(self, ctx):
        class SimpleAgent(Agent):
            system_prompts = ["Base system."]

        agent = SimpleAgent()
        result = await agent.prompt("do it", system="Extra instruction.")
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# agent.prompt[T] (structured mode)
# ---------------------------------------------------------------------------


class TestAgentPromptStructuredMode:
    async def test_bool_result(self):
        provider = MockProvider(canned_responses=[[
            ToolCallChunk(
                call_id="c1", name="return_result",
                arguments='{"value": true}',
            ),
            FinishChunk(reason="stop"),
        ]])
        ctx = ExecutionContext(provider=provider)
        token = set_context(ctx)
        try:
            class BoolAgent(Agent):
                system_prompts = ["Answer with true/false."]

            agent = BoolAgent()
            result = await agent.prompt[bool]("is it running?")
            assert result is True
        finally:
            reset_context(token)

    async def test_list_result(self):
        provider = MockProvider(canned_responses=[[
            ToolCallChunk(
                call_id="c1", name="return_result",
                arguments='{"value": ["a", "b", "c"]}',
            ),
            FinishChunk(reason="stop"),
        ]])
        ctx = ExecutionContext(provider=provider)
        token = set_context(ctx)
        try:
            agent = Agent()
            result = await agent.prompt[list[str]]("list items")
            assert result == ["a", "b", "c"]
        finally:
            reset_context(token)


# ---------------------------------------------------------------------------
# Callable system_prompts
# ---------------------------------------------------------------------------


class TestCallableSystemPrompts:
    def test_callable_entry_invoked(self):
        def dynamic_prompt(agent):
            return f"Dynamic: {agent.module}"

        class MyAgent(Agent):
            system_prompts = ["Static.", dynamic_prompt]

        agent = MyAgent(module="parser")
        rendered = agent._render_system_prompts()
        assert rendered == ["Static.", "Dynamic: parser"]

    def test_callable_returning_none_skipped(self):
        def skip_prompt(agent):
            return None

        class MyAgent(Agent):
            system_prompts = ["Keep this.", skip_prompt, "And this."]

        agent = MyAgent()
        rendered = agent._render_system_prompts()
        assert rendered == ["Keep this.", "And this."]

    def test_callable_returning_empty_string_skipped(self):
        def empty_prompt(agent):
            return ""

        class MyAgent(Agent):
            system_prompts = ["Present.", empty_prompt]

        agent = MyAgent()
        rendered = agent._render_system_prompts()
        assert rendered == ["Present."]

    def test_callable_mixed_with_templates(self):
        def dynamic(agent):
            return f"Role count: {len(agent.roles)}"

        class MyAgent(Agent):
            system_prompts = ["Working on {module}.", dynamic]

        agent = MyAgent(module="calc", roles=["a", "b"])
        rendered = agent._render_system_prompts()
        assert rendered == ["Working on calc.", "Role count: 2"]

    def test_callable_inherited_via_mro(self):
        def base_dynamic(agent):
            return "from-base"

        def child_dynamic(agent):
            return "from-child"

        class Base(Agent):
            system_prompts = [base_dynamic]

        class Child(Base):
            system_prompts = [child_dynamic]

        agent = Child()
        rendered = agent._render_system_prompts()
        assert rendered == ["from-base", "from-child"]


# ---------------------------------------------------------------------------
# context.agent propagation
# ---------------------------------------------------------------------------


class TestContextAgent:
    async def test_agent_set_during_prompt(self):
        """get_context().agent should be the Agent instance during prompt execution."""
        captured_agents: list[Agent | None] = []

        async def capture_agent() -> str:
            """Capture the current agent from context."""
            captured_agents.append(get_context().agent)
            return "captured"

        provider = MockProvider(canned_responses=[
            [
                ToolCallChunk(
                    call_id="c1", name="capture_agent", arguments="{}",
                ),
                FinishChunk(reason="tool_calls"),
            ],
            [
                TextChunk(text="done"),
                FinishChunk(reason="stop"),
            ],
        ])
        ctx = ExecutionContext(provider=provider)
        token = set_context(ctx)
        try:
            class TestRole(Agent):
                system_prompts = ["Working on {module}."]
                tools = [capture_agent]

            agent = TestRole(module="parser")
            await agent.prompt("do it")

            assert len(captured_agents) == 1
            assert captured_agents[0] is agent
            assert captured_agents[0].module == "parser"  # type: ignore[attr-defined]
        finally:
            reset_context(token)

    async def test_agent_none_outside(self, ctx):
        assert get_context().agent is None

    async def test_agent_restored_after_prompt(self):
        """After agent.prompt completes, the outer context's agent should be restored."""
        provider = MockProvider()
        ctx = ExecutionContext(provider=provider)
        token = set_context(ctx)
        try:
            assert get_context().agent is None
            agent = Agent()
            await agent.prompt("hello")
            assert get_context().agent is None
        finally:
            reset_context(token)


# ---------------------------------------------------------------------------
# @skill(role=...)
# ---------------------------------------------------------------------------


class TestSkillWithRole:
    async def test_kwargs_forwarded_to_role(self):
        """All bound arguments from the skill call should be forwarded to
        the role constructor."""
        captured_agents: list[Agent | None] = []

        async def capture_agent() -> str:
            """Capture."""
            captured_agents.append(get_context().agent)
            return "ok"

        class TestRole(Agent):
            system_prompts = ["Working on {module}."]
            tools = [capture_agent]

        provider = MockProvider(canned_responses=[
            [
                ToolCallChunk(
                    call_id="c1", name="capture_agent", arguments="{}",
                ),
                FinishChunk(reason="tool_calls"),
            ],
            [TextChunk(text="done"), FinishChunk(reason="stop")],
        ])
        ctx = ExecutionContext(provider=provider)
        token = set_context(ctx)
        try:
            @skill(role=TestRole)
            async def do_work(module: str) -> str:
                """Work on {module}."""

            await do_work(module="parser.lexer")

            assert len(captured_agents) == 1
            agent = captured_agents[0]
            assert isinstance(agent, TestRole)
            assert agent.module == "parser.lexer"  # type: ignore[attr-defined]
        finally:
            reset_context(token)

    async def test_role_and_skill_tools_merged(self):
        """Both role-level and skill-level tools should be available."""
        calls_seen: list[str] = []

        async def role_tool() -> str:
            """A role tool."""
            calls_seen.append("role_tool")
            return "from role"

        async def skill_tool() -> str:
            """A skill tool."""
            calls_seen.append("skill_tool")
            return "from skill"

        class ToolRole(Agent):
            tools = [role_tool]

        provider = MockProvider(canned_responses=[
            [
                ToolCallChunk(
                    call_id="c1", name="role_tool", arguments="{}",
                ),
                ToolCallChunk(
                    call_id="c2", name="skill_tool", arguments="{}",
                ),
                FinishChunk(reason="tool_calls"),
            ],
            [TextChunk(text="done"), FinishChunk(reason="stop")],
        ])
        ctx = ExecutionContext(provider=provider)
        token = set_context(ctx)
        try:
            @skill(role=ToolRole, tools=[skill_tool])
            async def do_work(module: str) -> str:
                """Work on {module}."""

            result = await do_work(module="x")
            assert "role_tool" in calls_seen
            assert "skill_tool" in calls_seen
            assert result == "done"
        finally:
            reset_context(token)

    async def test_role_and_skill_system_prompts_merged(self):
        """Role system prompts come first, then skill-level system prompt."""
        class TestRole(Agent):
            system_prompts = ["Role prompt for {module}."]

        provider = MockProvider()
        ctx = ExecutionContext(provider=provider)
        token = set_context(ctx)
        try:
            @skill(role=TestRole, system="Skill-level system prompt.")
            async def do_work(module: str) -> str:
                """Work on {module}."""

            result = await do_work(module="parser")
            assert isinstance(result, str)
        finally:
            reset_context(token)

    async def test_skill_without_role_unchanged(self, ctx):
        """@skill without role= should work exactly as before."""

        @skill
        async def greet(name: str) -> str:
            """Say hello to {name}."""

        result = await greet("Alice")
        assert "Say hello to Alice" in result

    async def test_structured_result_with_role(self):
        class TestRole(Agent):
            system_prompts = ["You count things."]

        provider = MockProvider(canned_responses=[[
            ToolCallChunk(
                call_id="c1", name="return_result",
                arguments='{"value": 42}',
            ),
            FinishChunk(reason="stop"),
        ]])
        ctx = ExecutionContext(provider=provider)
        token = set_context(ctx)
        try:
            @skill(role=TestRole)
            async def count_items(module: str) -> int:
                """How many items in {module}?"""

            result = await count_items(module="parser")
            assert result == 42
        finally:
            reset_context(token)
