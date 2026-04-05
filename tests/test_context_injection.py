"""Tests for thorn._context_injection — salience-based context injection."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from thorn.core._agent import Agent
from thorn.core._context import ExecutionContext, reset_context, set_context
from thorn.core._context_injection import (
    BRIEFING_ASSISTANT_CONTENT,
    INJECTION_BUDGET_FRACTION,
    LOW_INJECTION_SALIENCE,
    SCORE_THRESHOLD,
    DirectorySeed,
    FileSeed,
    SearchSeed,
    SeedContent,
    assemble_briefing,
    extract_seeds_from_prompt,
    injection_budget,
    merge_sources,
    normalize_scores,
)
from thorn.core._history import (
    CollapseState,
    DirectoryListCallNode,
    FileReadCallNode,
    HistoryTree,
    ToolCallNode,
    TurnNode,
    UserPromptNode,
)
from thorn.core._messages import (
    AssistantMessage,
    ToolCall,
    ToolResultMessage,
)
from thorn.core._provider import MockProvider


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tc(call_id: str, name: str, args: dict) -> ToolCall:
    return ToolCall(call_id=call_id, name=name, arguments=json.dumps(args))


def _result(call_id: str, content: str) -> ToolResultMessage:
    return ToolResultMessage(call_id=call_id, content=content)


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
# Normalize scores
# ---------------------------------------------------------------------------


class TestNormalizeScores:
    def test_sums_to_one(self):
        scores = {FileSeed("a"): 3.0, FileSeed("b"): 1.0}
        normed = normalize_scores(scores)
        assert abs(sum(normed.values()) - 1.0) < 1e-9

    def test_preserves_relative_ordering(self):
        scores = {FileSeed("a"): 3.0, FileSeed("b"): 1.0}
        normed = normalize_scores(scores)
        assert normed[FileSeed("a")] > normed[FileSeed("b")]

    def test_single_item(self):
        scores = {FileSeed("a"): 5.0}
        normed = normalize_scores(scores)
        assert normed[FileSeed("a")] == pytest.approx(1.0)

    def test_empty_input(self):
        assert normalize_scores({}) == {}

    def test_all_zero(self):
        assert normalize_scores({FileSeed("a"): 0.0, FileSeed("b"): 0.0}) == {}


# ---------------------------------------------------------------------------
# Merge sources
# ---------------------------------------------------------------------------


class TestMergeSources:
    def test_single_source(self):
        src = {FileSeed("a"): 1.0}
        merged = merge_sources([(src, 1.0)])
        assert FileSeed("a") in merged
        assert merged[FileSeed("a")] == pytest.approx(1.0)

    def test_multiple_sources_weighted(self):
        s1 = {FileSeed("a"): 1.0}
        s2 = {FileSeed("a"): 1.0, FileSeed("b"): 1.0}
        merged = merge_sources([(s1, 1.0), (s2, 0.5)])
        # a: 1.0*1.0 + 0.5*0.5 = 1.25
        assert merged[FileSeed("a")] == pytest.approx(1.25)
        # b: 0.0*1.0 + 0.5*0.5 = 0.25
        assert merged[FileSeed("b")] == pytest.approx(0.25)

    def test_threshold_excludes_low_scoring(self):
        src = {FileSeed("a"): 0.01, FileSeed("b"): 10.0}
        merged = merge_sources([(src, 0.1)])
        # a: 0.1 * 0.01/10.01 ~= 0.0001 → below threshold
        assert FileSeed("a") not in merged
        assert FileSeed("b") in merged

    def test_items_in_multiple_sources_boosted(self):
        s1 = {FileSeed("shared"): 1.0, FileSeed("only1"): 1.0}
        s2 = {FileSeed("shared"): 1.0}
        merged = merge_sources([(s1, 1.0), (s2, 0.5)])
        assert merged[FileSeed("shared")] > merged[FileSeed("only1")]

    def test_empty_sources(self):
        assert merge_sources([]) == {}

    def test_source_with_empty_scores(self):
        assert merge_sources([({}, 1.0)]) == {}


# ---------------------------------------------------------------------------
# Prompt text analysis
# ---------------------------------------------------------------------------


class TestExtractSeedsFromPrompt:
    def test_backtick_path(self):
        seeds = extract_seeds_from_prompt(
            "Fix the bug in `src/parser/lexer.h`", None,
        )
        assert FileSeed(path="src/parser/lexer.h") in seeds

    def test_quoted_path(self):
        seeds = extract_seeds_from_prompt(
            'Read the file "main.cpp" and fix it.', None,
        )
        assert FileSeed(path="main.cpp") in seeds

    def test_bare_path_with_slash(self):
        seeds = extract_seeds_from_prompt(
            "Check src/utils/helper.h for errors", None,
        )
        assert FileSeed(path="src/utils/helper.h") in seeds

    def test_bare_path_with_extension(self):
        seeds = extract_seeds_from_prompt(
            "Review parser.cpp", None,
        )
        assert FileSeed(path="parser.cpp") in seeds

    def test_backtick_identifier_becomes_search(self):
        seeds = extract_seeds_from_prompt(
            "Fix the `TokenStream` class", None,
        )
        assert SearchSeed(query="TokenStream") in seeds

    def test_no_paths_or_identifiers(self):
        seeds = extract_seeds_from_prompt("Hello world", None)
        assert seeds == {}

    def test_resolves_against_workspace(self, tmp_path):
        (tmp_path / "real_file.h").touch()
        (tmp_path / "real_dir").mkdir()

        seeds = extract_seeds_from_prompt(
            "Read `real_file.h` and list `real_dir`", tmp_path,
        )

        file_seeds = {s for s in seeds if isinstance(s, FileSeed)}
        dir_seeds = {s for s in seeds if isinstance(s, DirectorySeed)}
        assert any(s.path == str(tmp_path / "real_file.h") for s in file_seeds)
        assert any(s.path == str(tmp_path / "real_dir") for s in dir_seeds)

    def test_deduplication(self):
        seeds = extract_seeds_from_prompt(
            "Read `foo.h` and then review `foo.h` again", None,
        )
        file_seeds = [s for s in seeds if isinstance(s, FileSeed) and s.path == "foo.h"]
        assert len(file_seeds) == 1


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
            items = {FileSeed(path=str(test_file)): 1.0}
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
            items = {FileSeed(path=str(test_file)): 1.0}
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
            items = {FileSeed(path=str(test_file)): 1.0}
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
            items = {FileSeed(path=str(test_file)): 1.0}
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
            items = {DirectorySeed(path=str(tmp_path)): 1.0}
            turn = await assemble_briefing(items, token_budget=50000)
            assert turn is not None
            assert isinstance(turn.tool_call_nodes[0], DirectoryListCallNode)
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
            items = {
                FileSeed(path=str(tmp_path / f"file_{i}.txt")): 1.0 - i * 0.05
                for i in range(10)
            }
            # Budget sized to fit a few files but not all ten
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
            items = {
                FileSeed(path=str(tmp_path / "nonexistent.txt")): 1.0,
                FileSeed(path=str(real_file)): 0.9,
            }
            turn = await assemble_briefing(items, token_budget=50000)
            assert turn is not None
            assert len(turn.tool_call_nodes) == 1
            assert "content" in turn.tool_call_nodes[0].result.content
        finally:
            reset_context(token)

    async def test_empty_items(self):
        assert await assemble_briefing({}, token_budget=50000) is None

    async def test_zero_budget(self):
        items = {FileSeed(path="whatever"): 1.0}
        assert await assemble_briefing(items, token_budget=0) is None


# ---------------------------------------------------------------------------
# Agent base class hooks
# ---------------------------------------------------------------------------


class TestAgentParentAndHooks:
    def test_parent_defaults_to_none(self):
        agent = Agent()
        assert agent._parent is None

    def test_context_seed_items_default_empty(self):
        agent = Agent()
        assert agent.context_seed_items() == {}

    def test_extract_salient_default_empty(self):
        agent = Agent()
        tree = HistoryTree()
        assert agent.extract_salient_items_from_history(tree) == {}

    def test_parent_settable(self):
        parent = Agent()
        child = Agent()
        child._parent = parent
        assert child._parent is parent


# ---------------------------------------------------------------------------
# Extract salient items from history
# ---------------------------------------------------------------------------


class TestExtractSalientItemsFromHistory:
    """Test the extraction pattern using a custom Agent subclass
    that mirrors the ModuleDeveloper implementation."""

    def _make_extractor(self):
        """Create an agent subclass with extract logic for testing."""

        class Extractor(Agent):
            def extract_salient_items_from_history(
                self, history: HistoryTree,
            ) -> dict:
                from thorn.core._context_injection import DirectorySeed, FileSeed

                seeds: dict = {}
                turn_nodes = [
                    n for n in history.nodes if isinstance(n, TurnNode)
                ]
                total = len(turn_nodes)
                if total == 0:
                    return seeds
                for i, turn in enumerate(turn_nodes):
                    if turn.collapse_state == CollapseState.COLLAPSED:
                        continue
                    recency = (i + 1) / total
                    for tcn in turn.tool_call_nodes:
                        if tcn.detail_collapsed:
                            continue
                        try:
                            args = json.loads(tcn.tool_call.arguments)
                        except (json.JSONDecodeError, AttributeError):
                            continue
                        if isinstance(tcn, FileReadCallNode):
                            path = args.get("path")
                            if path:
                                key = FileSeed(path=path)
                                seeds[key] = max(
                                    seeds.get(key, 0.0), recency,
                                )
                        elif isinstance(tcn, DirectoryListCallNode):
                            path = args.get("path", ".")
                            key = DirectorySeed(path=path)
                            seeds[key] = max(
                                seeds.get(key, 0.0), recency,
                            )
                return seeds

        return Extractor()

    def test_finds_file_read_nodes(self):
        agent = self._make_extractor()
        tree = HistoryTree()
        tree.append_user_prompt("do something")

        tc = _tc("c1", "read_file", {"path": "src/foo.h"})
        tree.append_turn(
            AssistantMessage(content="reading", tool_calls=[tc]),
            [_result("c1", "file content")],
            call_node_classes={"c1": FileReadCallNode},
        )

        seeds = agent.extract_salient_items_from_history(tree)
        assert FileSeed(path="src/foo.h") in seeds

    def test_finds_directory_list_nodes(self):
        agent = self._make_extractor()
        tree = HistoryTree()
        tree.append_user_prompt("explore")

        tc = _tc("c1", "list_directory", {"path": "src/"})
        tree.append_turn(
            AssistantMessage(content="listing", tool_calls=[tc]),
            [_result("c1", "file1\nfile2")],
            call_node_classes={"c1": DirectoryListCallNode},
        )

        seeds = agent.extract_salient_items_from_history(tree)
        assert DirectorySeed(path="src/") in seeds

    def test_skips_collapsed_turns(self):
        agent = self._make_extractor()
        tree = HistoryTree()
        tree.append_user_prompt("start")

        tc = _tc("c1", "read_file", {"path": "old.h"})
        turn = tree.append_turn(
            AssistantMessage(content="reading", tool_calls=[tc]),
            [_result("c1", "old content")],
            call_node_classes={"c1": FileReadCallNode},
        )
        turn.collapse_state = CollapseState.COLLAPSED

        seeds = agent.extract_salient_items_from_history(tree)
        assert FileSeed(path="old.h") not in seeds

    def test_skips_detail_collapsed_tool_calls(self):
        agent = self._make_extractor()
        tree = HistoryTree()
        tree.append_user_prompt("start")

        tc = _tc("c1", "read_file", {"path": "big.h"})
        turn = tree.append_turn(
            AssistantMessage(content="reading", tool_calls=[tc]),
            [_result("c1", "x" * 5000)],
            call_node_classes={"c1": FileReadCallNode},
        )
        turn.tool_call_nodes[0].detail_collapsed = True

        seeds = agent.extract_salient_items_from_history(tree)
        assert FileSeed(path="big.h") not in seeds

    def test_recency_ranking(self):
        agent = self._make_extractor()
        tree = HistoryTree()
        tree.append_user_prompt("start")

        tc1 = _tc("c1", "read_file", {"path": "old.h"})
        tree.append_turn(
            AssistantMessage(content="r1", tool_calls=[tc1]),
            [_result("c1", "old")],
            call_node_classes={"c1": FileReadCallNode},
        )

        tc2 = _tc("c2", "read_file", {"path": "new.h"})
        tree.append_turn(
            AssistantMessage(content="r2", tool_calls=[tc2]),
            [_result("c2", "new")],
            call_node_classes={"c2": FileReadCallNode},
        )

        seeds = agent.extract_salient_items_from_history(tree)
        assert seeds[FileSeed(path="new.h")] > seeds[FileSeed(path="old.h")]

    def test_ignores_base_tool_call_nodes(self):
        agent = self._make_extractor()
        tree = HistoryTree()
        tree.append_user_prompt("start")

        tc = _tc("c1", "custom_tool", {"data": "value"})
        tree.append_turn(
            AssistantMessage(content="custom", tool_calls=[tc]),
            [_result("c1", "result")],
        )

        seeds = agent.extract_salient_items_from_history(tree)
        assert seeds == {}

    def test_empty_history(self):
        agent = self._make_extractor()
        tree = HistoryTree()
        seeds = agent.extract_salient_items_from_history(tree)
        assert seeds == {}


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

            # History: UserPrompt (actual), TurnNode (briefing), TurnNode (LLM response)
            assert len(agent._history.nodes) >= 3
            assert isinstance(agent._history.nodes[0], UserPromptNode)
            assert agent._history.nodes[0].message.content == "do something"
            assert isinstance(agent._history.nodes[1], TurnNode)
            briefing = agent._history.nodes[1]
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
            # context_seed_items should not have been called again
            assert call_count == first_count
        finally:
            reset_context(token)

    async def test_source2_and_3_only_for_sub_agents(self, tmp_path):
        """Root agents (no _parent) should only use Source 1."""
        test_file = tmp_path / "seed.txt"
        test_file.write_text("content")

        extract_calls: list[HistoryTree] = []

        class TrackingAgent(Agent):
            def context_seed_items(self):
                from thorn.core._context_injection import FileSeed
                return {FileSeed(path=str(test_file)): 1.0}

            def extract_salient_items_from_history(self, history):
                extract_calls.append(history)
                return {}

        provider = MockProvider()
        ctx = ExecutionContext(
            provider=provider,
            context_window=100000,
            workspace_root=tmp_path,
        )
        token = set_context(ctx)
        try:
            # Root agent (no parent)
            agent = TrackingAgent()
            await agent.prompt("do it")
            assert len(extract_calls) == 0

            # Sub-agent (has parent)
            parent = Agent()
            parent._history.append_user_prompt("parent prompt")
            child = TrackingAgent()
            child._parent = parent
            await child.prompt("child task")
            assert len(extract_calls) == 1
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
            # Only user prompt + response, no injection nodes
            injection_turns = [
                n for n in agent._history.nodes
                if isinstance(n, TurnNode) and len(n.tool_call_nodes) > 0
                and n.tool_call_nodes[0].tool_call.call_id.startswith("seed_")
            ]
            assert len(injection_turns) == 0
        finally:
            reset_context(token)

    async def test_no_injection_without_seed_items(self):
        """Agent with empty context_seed_items and no parent should not inject."""
        provider = MockProvider()
        ctx = ExecutionContext(
            provider=provider,
            context_window=100000,
        )
        token = set_context(ctx)
        try:
            agent = Agent()
            await agent.prompt("hello")
            # Should just have user prompt + response
            for node in agent._history.nodes:
                if isinstance(node, TurnNode):
                    for tcn in node.tool_call_nodes:
                        assert not tcn.tool_call.call_id.startswith("seed_")
        finally:
            reset_context(token)
