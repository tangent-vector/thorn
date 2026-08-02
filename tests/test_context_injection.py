"""Tests for thorn.core._context_injection — context injection."""

from __future__ import annotations

import pytest

from thorn.core._agent import Agent
from thorn.core._context import ExecutionContext, reset_context, set_context
from thorn.core._context_injection import (
    BRIEFING_ASSISTANT_CONTENT,
    INJECTION_BUDGET_FRACTION,
    LOW_INJECTION_SALIENCE,
    DirectorySeed,
    FileSeed,
    SearchSeed,
    SeedContent,
    assemble_briefing,
    injection_budget,
)
from thorn.core._history import (
    DirectoryListCallNode,
    FileReadCallNode,
    TurnNode,
    UserPromptNode,
)
from thorn.core._provider import MockProvider

# ---------------------------------------------------------------------------
# Seed types
# ---------------------------------------------------------------------------


class TestSeedTypes:
    def test_file_seed_hashable(self):
        a = FileSeed(path="src/foo.h")
        b = FileSeed(path="src/foo.h")
        assert a == b
        assert hash(a) == hash(b)

    def test_directory_seed_hashable(self):
        a = DirectorySeed(path="src/foo")
        b = DirectorySeed(path="src/foo")
        assert a == b
        assert hash(a) == hash(b)

    def test_search_seed_hashable(self):
        a = SearchSeed(query="Token")
        b = SearchSeed(query="Token")
        assert a == b
        assert hash(a) == hash(b)

    def test_different_types_not_equal(self):
        assert FileSeed(path="foo") != DirectorySeed(path="foo")
        assert FileSeed(path="foo") != SearchSeed(query="foo")

    def test_usable_as_dict_keys(self):
        d: dict[SeedContent, float] = {
            FileSeed(path="a.h"): 1.0,
            DirectorySeed(path="src"): 0.5,
            SearchSeed(query="Token"): 0.3,
        }
        assert len(d) == 3
        assert d[FileSeed(path="a.h")] == 1.0

    def test_equal_instances_merge_in_dict(self):
        d: dict[SeedContent, float] = {}
        d[FileSeed(path="a.h")] = 1.0
        d[FileSeed(path="a.h")] = 2.0
        assert len(d) == 1
        assert d[FileSeed(path="a.h")] == 2.0

    def test_frozen(self):
        seed = FileSeed(path="foo.h")
        with pytest.raises(AttributeError):
            seed.path = "bar.h"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Injection budget
# ---------------------------------------------------------------------------


class TestInjectionBudget:
    def test_with_context_window(self):
        assert injection_budget(100000) == int(100000 * INJECTION_BUDGET_FRACTION)

    def test_without_context_window(self):
        assert injection_budget(None) == 0


# ---------------------------------------------------------------------------
# Assemble briefing
# ---------------------------------------------------------------------------


class TestAssembleBriefing:
    async def test_returns_turn_node(self, tmp_path):
        test_file = tmp_path / "test.txt"
        test_file.write_text("hello world")

        ctx = ExecutionContext(
            provider=MockProvider(),
            workspace_root=tmp_path,
        )
        token = set_context(ctx)
        try:
            items = [FileSeed(path=str(test_file))]
            turn = await assemble_briefing(items, token_budget=50000)
            assert turn is not None
            assert isinstance(turn, TurnNode)
        finally:
            reset_context(token)

    async def test_assistant_content(self, tmp_path):
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")

        ctx = ExecutionContext(
            provider=MockProvider(),
            workspace_root=tmp_path,
        )
        token = set_context(ctx)
        try:
            items = [FileSeed(path=str(test_file))]
            turn = await assemble_briefing(items, token_budget=50000)
            assert turn is not None
            assert turn.assistant_content == BRIEFING_ASSISTANT_CONTENT
        finally:
            reset_context(token)

    async def test_low_intrinsic_salience(self, tmp_path):
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")

        ctx = ExecutionContext(
            provider=MockProvider(),
            workspace_root=tmp_path,
        )
        token = set_context(ctx)
        try:
            items = [FileSeed(path=str(test_file))]
            turn = await assemble_briefing(items, token_budget=50000)
            assert turn is not None
            assert turn.intrinsic_salience == LOW_INJECTION_SALIENCE
        finally:
            reset_context(token)

    async def test_uses_correct_subclass_for_file_seed(self, tmp_path):
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")

        ctx = ExecutionContext(
            provider=MockProvider(),
            workspace_root=tmp_path,
        )
        token = set_context(ctx)
        try:
            items = [FileSeed(path=str(test_file))]
            turn = await assemble_briefing(items, token_budget=50000)
            assert turn is not None
            assert len(turn.tool_call_nodes) == 1
            assert isinstance(turn.tool_call_nodes[0], FileReadCallNode)
        finally:
            reset_context(token)

    async def test_uses_correct_subclass_for_directory_seed(self, tmp_path):
        (tmp_path / "child.txt").touch()

        ctx = ExecutionContext(
            provider=MockProvider(),
            workspace_root=tmp_path,
        )
        token = set_context(ctx)
        try:
            items = [DirectorySeed(path=str(tmp_path))]
            turn = await assemble_briefing(items, token_budget=50000)
            assert turn is not None
            assert isinstance(turn.tool_call_nodes[0], DirectoryListCallNode)
        finally:
            reset_context(token)

    async def test_preserves_item_order(self, tmp_path):
        file_a = tmp_path / "a.txt"
        file_b = tmp_path / "b.txt"
        file_a.write_text("aaa")
        file_b.write_text("bbb")

        ctx = ExecutionContext(
            provider=MockProvider(),
            workspace_root=tmp_path,
        )
        token = set_context(ctx)
        try:
            items = [
                FileSeed(path=str(file_a)),
                FileSeed(path=str(file_b)),
            ]
            turn = await assemble_briefing(items, token_budget=50000)
            assert turn is not None
            assert len(turn.tool_call_nodes) == 2
            assert "aaa" in turn.tool_call_nodes[0].result.content
            assert "bbb" in turn.tool_call_nodes[1].result.content
        finally:
            reset_context(token)

    async def test_respects_token_budget(self, tmp_path):
        for i in range(10):
            (tmp_path / f"file_{i}.txt").write_text("x" * 200)

        ctx = ExecutionContext(
            provider=MockProvider(),
            workspace_root=tmp_path,
        )
        token = set_context(ctx)
        try:
            items = [
                FileSeed(path=str(tmp_path / f"file_{i}.txt"))
                for i in range(10)
            ]
            turn = await assemble_briefing(items, token_budget=500)
            assert turn is not None
            assert 0 < len(turn.tool_call_nodes) < 10
        finally:
            reset_context(token)

    async def test_drops_failed_tool_calls(self, tmp_path):
        real_file = tmp_path / "real.txt"
        real_file.write_text("content")

        ctx = ExecutionContext(
            provider=MockProvider(),
            workspace_root=tmp_path,
        )
        token = set_context(ctx)
        try:
            items = [
                FileSeed(path=str(tmp_path / "nonexistent.txt")),
                FileSeed(path=str(real_file)),
            ]
            turn = await assemble_briefing(items, token_budget=50000)
            assert turn is not None
            assert len(turn.tool_call_nodes) == 1
            assert "content" in turn.tool_call_nodes[0].result.content
        finally:
            reset_context(token)

    async def test_empty_items(self):
        assert await assemble_briefing([], token_budget=50000) is None

    async def test_zero_budget(self):
        items = [FileSeed(path="whatever")]
        assert await assemble_briefing(items, token_budget=0) is None


# ---------------------------------------------------------------------------
# Agent base class hooks
# ---------------------------------------------------------------------------


class TestAgentHooks:
    def test_context_seed_items_default_empty(self):
        agent = Agent()
        assert agent.context_seed_items() == {}


# ---------------------------------------------------------------------------
# Injection trigger in _run_agent_prompt
# ---------------------------------------------------------------------------


class TestInjectionTrigger:
    async def test_fires_on_empty_history(self, tmp_path):
        """Injection should place the user prompt then the briefing turn."""
        test_file = tmp_path / "seed.txt"
        test_file.write_text("seed content here")

        class SeedAgent(Agent):
            def context_seed_items(self):
                from thorn.core._context_injection import FileSeed
                return {FileSeed(path=str(test_file)): 1.0}

        provider = MockProvider()
        ctx = ExecutionContext(
            provider=provider,
            context_window=100000,
            workspace_root=tmp_path,
        )
        token = set_context(ctx)
        try:
            agent = SeedAgent()
            await agent.prompt("do something")

            assert len(agent._default_session._history.nodes) >= 3
            assert isinstance(agent._default_session._history.nodes[0], UserPromptNode)
            assert agent._default_session._history.nodes[0].message.content == "do something"
            assert isinstance(agent._default_session._history.nodes[1], TurnNode)
            briefing = agent._default_session._history.nodes[1]
            assert len(briefing.tool_call_nodes) >= 1
            assert briefing.tool_call_nodes[0].tool_call.call_id.startswith("seed_")
        finally:
            reset_context(token)

    async def test_does_not_fire_on_subsequent_prompts(self, tmp_path):
        test_file = tmp_path / "seed.txt"
        test_file.write_text("seed content")

        call_count = 0

        class CountingAgent(Agent):
            def context_seed_items(self):
                nonlocal call_count
                call_count += 1
                from thorn.core._context_injection import FileSeed
                return {FileSeed(path=str(test_file)): 1.0}

        provider = MockProvider()
        ctx = ExecutionContext(
            provider=provider,
            context_window=100000,
            workspace_root=tmp_path,
        )
        token = set_context(ctx)
        try:
            agent = CountingAgent()
            await agent.prompt("first prompt")
            first_count = call_count

            await agent.prompt("second prompt")
            assert call_count == first_count
        finally:
            reset_context(token)

    async def test_no_injection_without_context_window(self, tmp_path):
        """When context_window is None, no injection should occur."""
        test_file = tmp_path / "seed.txt"
        test_file.write_text("content")

        class SeedAgent(Agent):
            def context_seed_items(self):
                from thorn.core._context_injection import FileSeed
                return {FileSeed(path=str(test_file)): 1.0}

        provider = MockProvider()
        ctx = ExecutionContext(
            provider=provider,
            context_window=None,
            workspace_root=tmp_path,
        )
        token = set_context(ctx)
        try:
            agent = SeedAgent()
            await agent.prompt("hello")
            injection_turns = [
                n for n in agent._default_session._history.nodes
                if isinstance(n, TurnNode) and len(n.tool_call_nodes) > 0
                and n.tool_call_nodes[0].tool_call.call_id.startswith("seed_")
            ]
            assert len(injection_turns) == 0
        finally:
            reset_context(token)

    async def test_no_injection_without_seed_items(self):
        """Agent with empty context_seed_items and no recommended_context."""
        provider = MockProvider()
        ctx = ExecutionContext(
            provider=provider,
            context_window=100000,
        )
        token = set_context(ctx)
        try:
            agent = Agent()
            await agent.prompt("hello")
            for node in agent._default_session._history.nodes:
                if isinstance(node, TurnNode):
                    for tcn in node.tool_call_nodes:
                        assert not tcn.tool_call.call_id.startswith("seed_")
        finally:
            reset_context(token)


# ---------------------------------------------------------------------------
# Recommended context
# ---------------------------------------------------------------------------


class TestRecommendedContext:
    async def test_recommended_context_injected(self, tmp_path):
        """Recommended context from caller should be injected."""
        rec_file = tmp_path / "recommended.txt"
        rec_file.write_text("recommended content")

        provider = MockProvider()
        ctx = ExecutionContext(
            provider=provider,
            context_window=100000,
            workspace_root=tmp_path,
        )
        token = set_context(ctx)
        try:
            agent = Agent()
            await agent.prompt(
                "do it",
                recommended_context=[FileSeed(path=str(rec_file))],
            )

            briefing_turns = [
                n for n in agent._default_session._history.nodes
                if isinstance(n, TurnNode) and len(n.tool_call_nodes) > 0
                and n.tool_call_nodes[0].tool_call.call_id.startswith("seed_")
            ]
            assert len(briefing_turns) == 1
            assert "recommended content" in briefing_turns[0].tool_call_nodes[0].result.content
        finally:
            reset_context(token)

    async def test_recommended_takes_priority_over_role_seeds(self, tmp_path):
        """Recommended context items appear before role-declared seeds."""
        rec_file = tmp_path / "recommended.txt"
        rec_file.write_text("from caller")
        role_file = tmp_path / "role_seed.txt"
        role_file.write_text("from role")

        class SeedAgent(Agent):
            def context_seed_items(self):
                return {FileSeed(path=str(role_file)): 1.0}

        provider = MockProvider()
        ctx = ExecutionContext(
            provider=provider,
            context_window=100000,
            workspace_root=tmp_path,
        )
        token = set_context(ctx)
        try:
            agent = SeedAgent()
            await agent.prompt(
                "do it",
                recommended_context=[FileSeed(path=str(rec_file))],
            )

            briefing_turns = [
                n for n in agent._default_session._history.nodes
                if isinstance(n, TurnNode) and len(n.tool_call_nodes) > 0
                and n.tool_call_nodes[0].tool_call.call_id.startswith("seed_")
            ]
            assert len(briefing_turns) == 1
            nodes = briefing_turns[0].tool_call_nodes
            assert len(nodes) == 2
            assert "from caller" in nodes[0].result.content
            assert "from role" in nodes[1].result.content
        finally:
            reset_context(token)

    async def test_deduplication_between_recommended_and_role_seeds(self, tmp_path):
        """An item in both recommended and role seeds appears only once."""
        shared_file = tmp_path / "shared.txt"
        shared_file.write_text("shared content")

        class SeedAgent(Agent):
            def context_seed_items(self):
                return {FileSeed(path=str(shared_file)): 1.0}

        provider = MockProvider()
        ctx = ExecutionContext(
            provider=provider,
            context_window=100000,
            workspace_root=tmp_path,
        )
        token = set_context(ctx)
        try:
            agent = SeedAgent()
            await agent.prompt(
                "do it",
                recommended_context=[FileSeed(path=str(shared_file))],
            )

            briefing_turns = [
                n for n in agent._default_session._history.nodes
                if isinstance(n, TurnNode) and len(n.tool_call_nodes) > 0
                and n.tool_call_nodes[0].tool_call.call_id.startswith("seed_")
            ]
            assert len(briefing_turns) == 1
            assert len(briefing_turns[0].tool_call_nodes) == 1
        finally:
            reset_context(token)

    async def test_empty_recommended_falls_back_to_role_seeds(self, tmp_path):
        """When recommended_context is empty, role seeds still work."""
        role_file = tmp_path / "role.txt"
        role_file.write_text("role content")

        class SeedAgent(Agent):
            def context_seed_items(self):
                return {FileSeed(path=str(role_file)): 1.0}

        provider = MockProvider()
        ctx = ExecutionContext(
            provider=provider,
            context_window=100000,
            workspace_root=tmp_path,
        )
        token = set_context(ctx)
        try:
            agent = SeedAgent()
            await agent.prompt("do it", recommended_context=[])

            briefing_turns = [
                n for n in agent._default_session._history.nodes
                if isinstance(n, TurnNode) and len(n.tool_call_nodes) > 0
                and n.tool_call_nodes[0].tool_call.call_id.startswith("seed_")
            ]
            assert len(briefing_turns) == 1
            assert "role content" in briefing_turns[0].tool_call_nodes[0].result.content
        finally:
            reset_context(token)

    async def test_role_seeds_sorted_by_salience(self, tmp_path):
        """Role seeds are sorted by descending salience."""
        low_file = tmp_path / "low.txt"
        low_file.write_text("low priority")
        high_file = tmp_path / "high.txt"
        high_file.write_text("high priority")

        class SeedAgent(Agent):
            def context_seed_items(self):
                return {
                    FileSeed(path=str(low_file)): 0.1,
                    FileSeed(path=str(high_file)): 1.0,
                }

        provider = MockProvider()
        ctx = ExecutionContext(
            provider=provider,
            context_window=100000,
            workspace_root=tmp_path,
        )
        token = set_context(ctx)
        try:
            agent = SeedAgent()
            await agent.prompt("do it")

            briefing_turns = [
                n for n in agent._default_session._history.nodes
                if isinstance(n, TurnNode) and len(n.tool_call_nodes) > 0
                and n.tool_call_nodes[0].tool_call.call_id.startswith("seed_")
            ]
            assert len(briefing_turns) == 1
            nodes = briefing_turns[0].tool_call_nodes
            assert len(nodes) == 2
            assert "high priority" in nodes[0].result.content
            assert "low priority" in nodes[1].result.content
        finally:
            reset_context(token)
