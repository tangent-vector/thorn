"""Tests for thorn.core._agent — Agent base class, MRO collection, prompt accessor."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from thorn.core._agent import Agent, _SafeDict, _derive_stable_agent_id
from thorn.core._context import (
    ExecutionContext,
    get_context,
    reset_context,
    set_context,
)
from thorn.core._func import skill
from thorn.core._journal import JOURNAL_TOOLS
from thorn.runtime._inbox_tools import INBOX_TOOLS

# Framework-default tools appended to every agent, in the order
# ``Agent._collect_tools`` adds them.
_DEFAULT_TOOLS = list(JOURNAL_TOOLS) + list(INBOX_TOOLS)
from thorn.core._messages import AssistantMessage
from thorn.core._provider import FinishChunk, MockProvider, TextChunk, ToolCallChunk
from thorn.core._session import Session
from thorn.runtime._session import AgentID


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

        assert Child._collect_tools() == [tool_a, tool_b] + _DEFAULT_TOOLS

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

        assert Child._collect_tools() == [tool_a] + _DEFAULT_TOOLS

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
        mro_tools = collected[: -len(_DEFAULT_TOOLS)]
        assert len(mro_tools) == 1
        assert mro_tools[0] is shared

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
        mro_tools = collected[: -len(_DEFAULT_TOOLS)]
        assert len(mro_tools) == 1
        assert mro_tools[0] is helper

    def test_nested_tool_list_flattened(self):
        def tool_a() -> str:
            """A."""

        def tool_b() -> str:
            """B."""

        def tool_c() -> str:
            """C."""

        class Base(Agent):
            tools = [[tool_a, tool_b], tool_c]

        collected = Base._collect_tools()
        assert collected == [tool_a, tool_b, tool_c] + _DEFAULT_TOOLS

    def test_nested_toolset_dedup_across_mro(self):
        """A toolset constant included in both parent and child is
        flattened and deduplicated correctly."""

        def tool_a() -> str:
            """A."""

        def tool_b() -> str:
            """B."""

        def tool_c() -> str:
            """C."""

        TOOLSET = [tool_a, tool_b]

        class Base(Agent):
            tools = [TOOLSET]

        class Child(Base):
            tools = [TOOLSET, tool_c]

        collected = Child._collect_tools()
        assert collected == [tool_a, tool_b, tool_c] + _DEFAULT_TOOLS

    def test_single_inheritance_validation_rules(self):
        class Base(Agent):
            validation_rules = ["build"]

        class Child(Base):
            validation_rules = ["test"]

        assert Child._collect_validation_rules() == ["build", "test"]

    def test_validation_rules_dedup(self):
        class Base(Agent):
            validation_rules = ["build", "test"]

        class Child(Base):
            validation_rules = ["build", "lint"]

        collected = Child._collect_validation_rules()
        assert collected == ["build", "test", "lint"]

    def test_child_without_own_validation_rules_inherits(self):
        class Base(Agent):
            validation_rules = ["build"]

        class Child(Base):
            pass

        assert Child._collect_validation_rules() == ["build"]

    def test_diamond_validation_rules(self):
        class Base(Agent):
            validation_rules = ["build"]

        class Left(Base):
            validation_rules = ["test"]

        class Right(Base):
            validation_rules = ["lint"]

        class Diamond(Left, Right):
            validation_rules = ["format"]

        assert Diamond._collect_validation_rules() == [
            "build", "lint", "test", "format",
        ]

    def test_bare_agent_has_empty_collections(self):
        assert Agent._collect_system_prompts() == []
        assert Agent._collect_tools() == list(_DEFAULT_TOOLS)
        assert Agent._collect_validation_rules() == []


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
        from thorn.core._func import wrap_function

        async def helper(x: int) -> int:
            """Double a number."""
            return x * 2

        class ToolAgent(Agent):
            system_prompts = ["Use tools."]
            tools = [wrap_function(helper)]

        agent = ToolAgent()
        result = await agent.prompt("help me")
        assert isinstance(result, str)

    async def test_prompt_with_extra_tools(self, ctx):
        async def extra_tool() -> str:
            """Extra."""
            return "extra"

        class SimpleAgent(Agent):
            pass

        from thorn.core._func import wrap_function

        agent = SimpleAgent()
        result = await agent.prompt("do it", tools=[wrap_function(extra_tool)])
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
            from thorn.core._func import wrap_function

            class TestRole(Agent):
                system_prompts = ["Working on {module}."]
                tools = [wrap_function(capture_agent)]

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

        from thorn.core._func import wrap_function

        class TestRole(Agent):
            system_prompts = ["Working on {module}."]
            tools = [wrap_function(capture_agent)]

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

        from thorn.core._func import wrap_function

        class ToolRole(Agent):
            tools = [wrap_function(role_tool)]

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
            @skill(role=ToolRole, tools=[wrap_function(skill_tool)])
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


# ---------------------------------------------------------------------------
# AGENTS.md injection into system prompts
# ---------------------------------------------------------------------------


class TestAgentsMdInjection:
    """End-to-end coverage for the AGENTS.md side of the unified
    context-gathering pipeline.

    The unit tests in ``tests/test_context_layers.py`` and
    ``tests/test_prompt_assembly.py`` already exercise per-directory
    AGENTS.md loading and outer-to-inner block ordering against
    in-memory fixtures.  The tests here close the loop: they put real
    files on disk under realistic ``home`` / ``workspace`` directories,
    drive a full ``Session.prompt`` round, and assert that what the
    LLM provider actually receives in ``system_prompts`` matches what
    the pipeline is supposed to produce.

    The headline regression target is the historical "silent-drop"
    bug, where an AGENTS.md placed in the agent workspace would be
    silently overridden (and lost) when the agent home also had one.
    Under the unified pipeline both files must reach the provider, in
    outer-to-inner order: agent_home first, then agent_workspace.
    """

    @pytest.mark.asyncio
    async def test_home_and_workspace_agents_md_both_reach_provider(
        self, tmp_path: Path,
    ):
        home_dir = tmp_path / "home"
        workspace_dir = tmp_path / "ws"
        home_dir.mkdir()
        workspace_dir.mkdir()

        (home_dir / "AGENTS.md").write_text(
            "HOME-POLICY-MARKER: project-wide invariants\n",
            encoding="utf-8",
        )
        (workspace_dir / "AGENTS.md").write_text(
            "WORKSPACE-POLICY-MARKER: per-checkout overrides\n",
            encoding="utf-8",
        )

        provider = MockProvider()
        captured_prompts: list[list[str]] = []
        original_complete = provider.complete

        async def tracking_complete(
            system_prompts: list[str],
            tools: list[dict],
            messages: list[Any],
        ):
            captured_prompts.append(list(system_prompts))
            async for chunk in original_complete(system_prompts, tools, messages):
                yield chunk

        provider.complete = tracking_complete  # type: ignore[assignment]

        context = ExecutionContext(provider=provider, workspace_root=workspace_dir)
        token = set_context(context)
        try:
            agent = Agent(workspace=workspace_dir, home=home_dir)
            session = Session(agent=agent)
            await session.prompt("hello")

            assert len(captured_prompts) == 1
            joined = "\n".join(captured_prompts[0])
            # Both files reach the provider -- this is the silent-drop
            # regression assertion.
            assert "HOME-POLICY-MARKER" in joined
            assert "WORKSPACE-POLICY-MARKER" in joined
            # And the home (outer) block precedes the workspace (inner)
            # block, matching the documented outer-to-inner ordering.
            assert (
                joined.index("HOME-POLICY-MARKER")
                < joined.index("WORKSPACE-POLICY-MARKER")
            )
        finally:
            reset_context(token)

    @pytest.mark.asyncio
    async def test_workspace_agents_md_reaches_provider_when_home_has_none(
        self, tmp_path: Path,
    ):
        # Companion to the silent-drop regression: in the *absence* of a
        # home AGENTS.md, the workspace one must still flow through.
        # Guards against the inverse bug -- workspace content being
        # gated on the presence of home content.
        home_dir = tmp_path / "home"
        workspace_dir = tmp_path / "ws"
        home_dir.mkdir()
        workspace_dir.mkdir()

        (workspace_dir / "AGENTS.md").write_text(
            "WORKSPACE-ONLY-MARKER\n", encoding="utf-8",
        )

        provider = MockProvider()
        captured_prompts: list[list[str]] = []
        original_complete = provider.complete

        async def tracking_complete(
            system_prompts: list[str],
            tools: list[dict],
            messages: list[Any],
        ):
            captured_prompts.append(list(system_prompts))
            async for chunk in original_complete(system_prompts, tools, messages):
                yield chunk

        provider.complete = tracking_complete  # type: ignore[assignment]

        context = ExecutionContext(provider=provider, workspace_root=workspace_dir)
        token = set_context(context)
        try:
            agent = Agent(workspace=workspace_dir, home=home_dir)
            session = Session(agent=agent)
            await session.prompt("hello")

            assert len(captured_prompts) == 1
            joined = "\n".join(captured_prompts[0])
            assert "WORKSPACE-ONLY-MARKER" in joined
        finally:
            reset_context(token)


# ---------------------------------------------------------------------------
# MEMORY.md injection into system prompts
# ---------------------------------------------------------------------------


class TestMemoryMdInjection:
    @pytest.mark.asyncio
    async def test_memory_loaded_from_home_not_workspace(self, tmp_path: Path):
        """MEMORY.md is loaded from agent.home, not agent.workspace.

        When the two directories differ, only the home copy should be
        injected into the system prompt.
        """
        home_dir = tmp_path / "home"
        workspace_dir = tmp_path / "ws"
        home_dir.mkdir()
        workspace_dir.mkdir()

        memory_content = (
            "# Agent Memory\n\n"
            "- Project: example-project\n"
            "- Default branch: main\n"
        )
        (home_dir / "MEMORY.md").write_text(memory_content, encoding="utf-8")
        (workspace_dir / "MEMORY.md").write_text(
            "WRONG -- this should not be loaded", encoding="utf-8",
        )

        provider = MockProvider()
        captured_prompts: list[list[str]] = []
        original_complete = provider.complete

        async def tracking_complete(
            system_prompts: list[str],
            tools: list[dict],
            messages: list[Any],
        ):
            captured_prompts.append(list(system_prompts))
            async for chunk in original_complete(system_prompts, tools, messages):
                yield chunk

        provider.complete = tracking_complete  # type: ignore[assignment]

        context = ExecutionContext(provider=provider, workspace_root=workspace_dir)
        token = set_context(context)
        try:
            agent = Agent(workspace=workspace_dir, home=home_dir)
            session = Session(agent=agent)

            session._history.append_user_prompt("Earlier message")
            session._history.append_turn(
                AssistantMessage(content="Earlier response"), [],
            )

            await session.prompt("New message after resume")

            assert len(captured_prompts) == 1
            joined = "\n".join(captured_prompts[0])
            assert "example-project" in joined
            assert "Default branch: main" in joined
            assert "WRONG" not in joined
        finally:
            reset_context(token)

    @pytest.mark.asyncio
    async def test_memory_included_on_resumed_session(self, tmp_path: Path):
        """On a resumed session (non-empty history), MEMORY.md content
        still appears in the system prompts sent to the provider.

        This guards against accidentally gating MEMORY.md loading on the
        same ``if not session._history.nodes`` check that controls
        one-time context injection.
        """
        memory_content = (
            "# Agent Memory\n\n"
            "- Project: example-project\n"
            "- Default branch: main\n"
        )
        (tmp_path / "MEMORY.md").write_text(memory_content, encoding="utf-8")

        provider = MockProvider()

        captured_prompts: list[list[str]] = []
        original_complete = provider.complete

        async def tracking_complete(
            system_prompts: list[str],
            tools: list[dict],
            messages: list[Any],
        ):
            captured_prompts.append(list(system_prompts))
            async for chunk in original_complete(system_prompts, tools, messages):
                yield chunk

        provider.complete = tracking_complete  # type: ignore[assignment]

        context = ExecutionContext(provider=provider, workspace_root=tmp_path)
        token = set_context(context)
        try:
            # home=tmp_path so MEMORY.md is found there directly
            agent = Agent(workspace=tmp_path, home=tmp_path)
            session = Session(agent=agent)

            session._history.append_user_prompt("Earlier message")
            session._history.append_turn(
                AssistantMessage(content="Earlier response"), [],
            )

            await session.prompt("New message after resume")

            assert len(captured_prompts) == 1
            joined = "\n".join(captured_prompts[0])
            assert "example-project" in joined
            assert "Default branch: main" in joined
        finally:
            reset_context(token)


# ---------------------------------------------------------------------------
# Stable agent ID derivation
# ---------------------------------------------------------------------------


class TestDeriveStableAgentId:
    def test_deterministic(self, tmp_path: Path):
        id1 = _derive_stable_agent_id("MyAgent", tmp_path)
        id2 = _derive_stable_agent_id("MyAgent", tmp_path)
        assert id1 == id2

    def test_different_class_name_different_id(self, tmp_path: Path):
        id1 = _derive_stable_agent_id("AgentA", tmp_path)
        id2 = _derive_stable_agent_id("AgentB", tmp_path)
        assert id1 != id2

    def test_different_workspace_different_id(self, tmp_path: Path):
        dir_a = tmp_path / "a"
        dir_b = tmp_path / "b"
        dir_a.mkdir()
        dir_b.mkdir()
        id1 = _derive_stable_agent_id("MyAgent", dir_a)
        id2 = _derive_stable_agent_id("MyAgent", dir_b)
        assert id1 != id2

    def test_returns_agent_id_type(self, tmp_path: Path):
        result = _derive_stable_agent_id("MyAgent", tmp_path)
        assert isinstance(result, AgentID)

    def test_format_includes_lowercase_class_name(self, tmp_path: Path):
        result = _derive_stable_agent_id("MyAgent", tmp_path)
        assert result.startswith("myagent-")

    def test_filesystem_safe(self, tmp_path: Path):
        result = _derive_stable_agent_id("MyAgent", tmp_path)
        assert "/" not in result
        assert "\\" not in result
        assert " " not in result


# ---------------------------------------------------------------------------
# Agent.home property
# ---------------------------------------------------------------------------


class TestAgentHome:
    def test_explicit_home(self, tmp_path: Path):
        home_dir = tmp_path / "my-home"
        agent = Agent(home=home_dir)
        assert agent.home == home_dir

    def test_home_none_without_context(self):
        agent = Agent()
        assert agent.home is None

    def test_home_lazily_resolved_from_context(self, tmp_path: Path):
        ctx = ExecutionContext(
            provider=MockProvider(),
            workspace_root=tmp_path,
            agency_root_directory=tmp_path,
        )
        token = set_context(ctx)
        try:
            agent = Agent(id=AgentID("test-agent"))
            assert agent.home == (
                tmp_path / ".thorn" / "agents" / "test-agent" / "home"
            )
        finally:
            reset_context(token)

    def test_home_derives_stable_id_when_id_is_none(self, tmp_path: Path):
        ctx = ExecutionContext(
            provider=MockProvider(),
            workspace_root=tmp_path,
            agency_root_directory=tmp_path,
        )
        token = set_context(ctx)
        try:
            agent = Agent()
            home = agent.home
            assert home is not None
            # ``home`` is the mounted ``home/`` subtree under the
            # agent's framework dir.
            assert home.name == "home"
            assert home.parent.parent == tmp_path / ".thorn" / "agents"
            # The agent should now have a derived ID
            assert agent.id is not None
            assert isinstance(agent.id, AgentID)
        finally:
            reset_context(token)

    def test_home_falls_back_to_workspace_root_as_agency_root(self, tmp_path: Path):
        """When agency_root_directory is not set, workspace_root is used."""
        ctx = ExecutionContext(
            provider=MockProvider(),
            workspace_root=tmp_path,
        )
        token = set_context(ctx)
        try:
            agent = Agent(id=AgentID("fallback-agent"))
            assert agent.home == (
                tmp_path / ".thorn" / "agents" / "fallback-agent" / "home"
            )
        finally:
            reset_context(token)

    def test_home_none_when_no_workspace_and_no_id(self, tmp_path: Path):
        """An agent without an id and without workspace context gets no home."""
        ctx = ExecutionContext(provider=MockProvider())
        token = set_context(ctx)
        try:
            agent = Agent()
            assert agent.home is None
        finally:
            reset_context(token)

    def test_home_resolved_once(self, tmp_path: Path):
        """Once resolved, home does not change even if context changes."""
        ctx1 = ExecutionContext(
            provider=MockProvider(),
            workspace_root=tmp_path,
            agency_root_directory=tmp_path,
        )
        token1 = set_context(ctx1)
        try:
            agent = Agent(id=AgentID("pinned"))
            first_home = agent.home
            assert first_home is not None
        finally:
            reset_context(token1)

        other = tmp_path / "other"
        other.mkdir()
        ctx2 = ExecutionContext(
            provider=MockProvider(),
            workspace_root=other,
            agency_root_directory=other,
        )
        token2 = set_context(ctx2)
        try:
            assert agent.home == first_home
        finally:
            reset_context(token2)


# ---------------------------------------------------------------------------
# Agent.lock property
# ---------------------------------------------------------------------------


class TestAgentLock:
    def test_lock_returns_same_instance(self):
        agent = Agent()
        lock1 = agent.lock
        lock2 = agent.lock
        assert lock1 is lock2
        assert isinstance(lock1, asyncio.Lock)

    def test_different_agents_have_independent_locks(self):
        agent_a = Agent()
        agent_b = Agent()
        assert agent_a.lock is not agent_b.lock

    def test_lock_is_none_before_first_access(self):
        agent = Agent()
        assert agent._lock is None
        _ = agent.lock
        assert agent._lock is not None


# ---------------------------------------------------------------------------
# agency_root_directory propagation
# ---------------------------------------------------------------------------


class TestAgencyRootPropagation:
    def test_push_scope_preserves_agency_root(self, tmp_path: Path):
        ctx = ExecutionContext(
            provider=MockProvider(),
            agency_root_directory=tmp_path,
        )
        child = ctx.push_scope("child")
        assert child.agency_root_directory == tmp_path

    def test_agency_root_defaults_to_none(self):
        ctx = ExecutionContext(provider=MockProvider())
        assert ctx.agency_root_directory is None


# ---------------------------------------------------------------------------
# Environment summary in system prompts
# ---------------------------------------------------------------------------


class TestEnvironmentPromptInjection:
    """Verify that _run_session_prompt appends a '## Your environment'
    fragment listing the working directory and home directory."""

    @pytest.mark.asyncio
    async def test_both_workspace_and_home_injected(self, tmp_path: Path):
        ws = tmp_path / "ws"
        home = tmp_path / "home"
        ws.mkdir()
        home.mkdir()

        provider = MockProvider()
        captured_prompts: list[list[str]] = []
        original_complete = provider.complete

        async def tracking_complete(
            system_prompts: list[str],
            tools: list[dict],
            messages: list[Any],
        ):
            captured_prompts.append(list(system_prompts))
            async for chunk in original_complete(system_prompts, tools, messages):
                yield chunk

        provider.complete = tracking_complete  # type: ignore[assignment]

        context = ExecutionContext(provider=provider, workspace_root=ws)
        token = set_context(context)
        try:
            agent = Agent(workspace=ws, home=home)
            session = Session(agent=agent)
            await session.prompt("hello")

            assert len(captured_prompts) == 1
            joined = "\n".join(captured_prompts[0])
            assert "## Your environment" in joined
            assert f"Working directory (`.`): {ws}" in joined
            assert f"Home directory (`~`): {home}" in joined
        finally:
            reset_context(token)

    @pytest.mark.asyncio
    async def test_env_fragment_on_resumed_session(self, tmp_path: Path):
        """The environment fragment is present on every turn, not just the
        first (guards against accidentally gating it on empty history)."""
        ws = tmp_path / "ws"
        home = tmp_path / "home"
        ws.mkdir()
        home.mkdir()

        provider = MockProvider()
        captured_prompts: list[list[str]] = []
        original_complete = provider.complete

        async def tracking_complete(
            system_prompts: list[str],
            tools: list[dict],
            messages: list[Any],
        ):
            captured_prompts.append(list(system_prompts))
            async for chunk in original_complete(system_prompts, tools, messages):
                yield chunk

        provider.complete = tracking_complete  # type: ignore[assignment]

        context = ExecutionContext(provider=provider, workspace_root=ws)
        token = set_context(context)
        try:
            agent = Agent(workspace=ws, home=home)
            session = Session(agent=agent)

            session._history.append_user_prompt("Earlier message")
            session._history.append_turn(
                AssistantMessage(content="Earlier response"), [],
            )

            await session.prompt("New message after resume")

            assert len(captured_prompts) == 1
            joined = "\n".join(captured_prompts[0])
            assert "## Your environment" in joined
            assert f"Working directory (`.`): {ws}" in joined
            assert f"Home directory (`~`): {home}" in joined
        finally:
            reset_context(token)

    @pytest.mark.asyncio
    async def test_no_env_fragment_when_nothing_resolved(self):
        provider = MockProvider()
        captured_prompts: list[list[str]] = []
        original_complete = provider.complete

        async def tracking_complete(
            system_prompts: list[str],
            tools: list[dict],
            messages: list[Any],
        ):
            captured_prompts.append(list(system_prompts))
            async for chunk in original_complete(system_prompts, tools, messages):
                yield chunk

        provider.complete = tracking_complete  # type: ignore[assignment]

        context = ExecutionContext(provider=provider)
        token = set_context(context)
        try:
            agent = Agent()
            session = Session(agent=agent)
            await session.prompt("hello")

            assert len(captured_prompts) == 1
            joined = "\n".join(captured_prompts[0])
            assert "## Your environment" not in joined
        finally:
            reset_context(token)
