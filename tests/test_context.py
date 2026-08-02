"""Tests for thorn.core._context — scope chain, ExecutionContext, contextvar management."""

from __future__ import annotations

from pathlib import Path

import pytest

from thorn.core._agent import Agent
from thorn.core._context import (
    ConsoleEventSink,
    ExecutionContext,
    Scope,
    Verbosity,
    get_context,
    reset_context,
    resolve_path,
    set_context,
    shell_env,
)
from thorn.core._provider import MockProvider
from thorn.core._validation_convergence import ValidationConvergencePolicy

# ---------------------------------------------------------------------------
# Scope chain
# ---------------------------------------------------------------------------

class TestScope:
    def test_single_scope_depth(self):
        s = Scope(description="root")
        assert s.depth == 1

    def test_nested_depth(self):
        outer = Scope(description="outer")
        inner = Scope(description="inner", outer=outer)
        assert inner.depth == 2

    def test_chain_ordering_outermost_first(self):
        a = Scope(description="a")
        b = Scope(description="b", outer=a)
        c = Scope(description="c", outer=b)
        chain = c.chain()
        assert [s.description for s in chain] == ["a", "b", "c"]

    def test_single_scope_chain_is_itself(self):
        s = Scope(description="only")
        assert s.chain() == [s]

    def test_metadata_preserved(self):
        s = Scope(description="x", metadata={"key": "val"})
        assert s.metadata == {"key": "val"}


# ---------------------------------------------------------------------------
# ExecutionContext.push_scope
# ---------------------------------------------------------------------------

class TestExecutionContextPushScope:
    def test_push_creates_child_scope(self):
        ctx = ExecutionContext(provider=MockProvider())
        child = ctx.push_scope("level-1")
        assert child.scope is not None
        assert child.scope.description == "level-1"
        assert child.scope.outer is None  # parent had no scope

    def test_push_nests_under_existing_scope(self):
        ctx = ExecutionContext(provider=MockProvider())
        child1 = ctx.push_scope("level-1")
        child2 = child1.push_scope("level-2")
        assert child2.scope.depth == 2
        assert child2.scope.outer.description == "level-1"

    def test_push_does_not_mutate_parent(self):
        ctx = ExecutionContext(provider=MockProvider())
        _ = ctx.push_scope("child")
        assert ctx.scope is None

    def test_push_preserves_system_prompts(self):
        ctx = ExecutionContext(
            provider=MockProvider(),
            system_prompts=["You are helpful."],
        )
        child = ctx.push_scope("x")
        assert child.system_prompts == ["You are helpful."]

    def test_push_shares_provider(self):
        provider = MockProvider()
        ctx = ExecutionContext(provider=provider)
        child = ctx.push_scope("x")
        assert child.provider is provider

    def test_push_metadata_passed_through(self):
        ctx = ExecutionContext(provider=MockProvider())
        child = ctx.push_scope("x", tag="foo")
        assert child.scope.metadata == {"tag": "foo"}

    def test_push_preserves_validation_convergence_policy(self):
        context = ExecutionContext(
            provider=MockProvider(),
            validation_convergence_policy=(
                ValidationConvergencePolicy.ACTION_EPOCH_V1
            ),
        )

        child = context.push_scope("validation")

        assert (
            child.validation_convergence_policy
            is ValidationConvergencePolicy.ACTION_EPOCH_V1
        )

# ---------------------------------------------------------------------------
# ContextVar management
# ---------------------------------------------------------------------------

class TestContextVar:
    def test_get_without_set_raises_runtime_error(self):
        with pytest.raises(RuntimeError, match="No thorn ExecutionContext"):
            get_context()

    def test_set_and_get_round_trip(self):
        ctx = ExecutionContext(provider=MockProvider())
        token = set_context(ctx)
        try:
            assert get_context() is ctx
        finally:
            reset_context(token)

    def test_reset_restores_previous(self):
        ctx_a = ExecutionContext(provider=MockProvider())
        token_a = set_context(ctx_a)
        try:
            ctx_b = ExecutionContext(provider=MockProvider())
            token_b = set_context(ctx_b)
            assert get_context() is ctx_b
            reset_context(token_b)
            assert get_context() is ctx_a
        finally:
            reset_context(token_a)


# ---------------------------------------------------------------------------
# resolve_path
# ---------------------------------------------------------------------------

class TestResolvePath:
    def test_absolute_path_returned_as_is(self, tmp_path: Path):
        """Absolute paths pass through without workspace resolution."""
        absolute = tmp_path / "some" / "file.txt"
        result = resolve_path(str(absolute))
        assert result == absolute.resolve()

    def test_relative_path_resolved_against_workspace(self, tmp_path: Path):
        """Relative paths are joined to the active workspace_root."""
        ctx = ExecutionContext(
            provider=MockProvider(),
            workspace_root=tmp_path,
        )
        token = set_context(ctx)
        try:
            result = resolve_path("subdir/file.txt")
            assert result == (tmp_path / "subdir" / "file.txt").resolve()
        finally:
            reset_context(token)

    def test_dot_resolved_to_workspace_root(self, tmp_path: Path):
        """The special path '.' resolves to the workspace root."""
        ctx = ExecutionContext(
            provider=MockProvider(),
            workspace_root=tmp_path,
        )
        token = set_context(ctx)
        try:
            result = resolve_path(".")
            assert result == tmp_path.resolve()
        finally:
            reset_context(token)

    def test_no_context_falls_back_to_cwd(self):
        """Without an active context, falls back to process CWD."""
        result = resolve_path("relative/path")
        expected = (Path.cwd() / "relative" / "path").resolve()
        assert result == expected

    def test_context_without_workspace_falls_back_to_cwd(self):
        """Context with workspace_root=None falls back to process CWD."""
        ctx = ExecutionContext(
            provider=MockProvider(),
            workspace_root=None,
        )
        token = set_context(ctx)
        try:
            result = resolve_path("relative/path")
            expected = (Path.cwd() / "relative" / "path").resolve()
            assert result == expected
        finally:
            reset_context(token)

    def test_accepts_path_object(self, tmp_path: Path):
        """resolve_path also accepts a Path object, not just str."""
        ctx = ExecutionContext(
            provider=MockProvider(),
            workspace_root=tmp_path,
        )
        token = set_context(ctx)
        try:
            result = resolve_path(Path("foo/bar"))
            assert result == (tmp_path / "foo" / "bar").resolve()
        finally:
            reset_context(token)

    # -- tilde expansion ---------------------------------------------------

    def test_tilde_resolves_to_agent_home(self, tmp_path: Path):
        """Bare ``~`` resolves to agent.home when an agent is active."""
        home = tmp_path / "agent-home"
        home.mkdir()
        agent = Agent(home=home)
        ctx = ExecutionContext(
            provider=MockProvider(),
            workspace_root=tmp_path,
            agent=agent,
        )
        token = set_context(ctx)
        try:
            assert resolve_path("~") == home.resolve()
        finally:
            reset_context(token)

    def test_tilde_slash_resolves_to_agent_home_subpath(self, tmp_path: Path):
        """``~/subdir/file.txt`` resolves under agent.home."""
        home = tmp_path / "agent-home"
        home.mkdir()
        agent = Agent(home=home)
        ctx = ExecutionContext(
            provider=MockProvider(),
            workspace_root=tmp_path,
            agent=agent,
        )
        token = set_context(ctx)
        try:
            result = resolve_path("~/subdir/file.txt")
            assert result == (home / "subdir" / "file.txt").resolve()
        finally:
            reset_context(token)

    def test_tilde_without_agent_falls_back_to_os_home(self, tmp_path: Path):
        """When no agent is active, ~ falls back to OS-level expansion."""
        import os
        ctx = ExecutionContext(
            provider=MockProvider(),
            workspace_root=tmp_path,
        )
        token = set_context(ctx)
        try:
            result = resolve_path("~/somefile")
            expected = Path(os.path.expanduser("~/somefile")).resolve()
            assert result == expected
        finally:
            reset_context(token)

    def test_tilde_without_agent_home_falls_back_to_os(self, tmp_path: Path):
        """Agent with home=None falls back to OS tilde expansion."""
        import os
        agent = Agent()  # no home set, no context to derive from
        agent._home_resolved = True  # prevent lazy resolution
        ctx = ExecutionContext(
            provider=MockProvider(),
            workspace_root=tmp_path,
            agent=agent,
        )
        token = set_context(ctx)
        try:
            result = resolve_path("~/somefile")
            expected = Path(os.path.expanduser("~/somefile")).resolve()
            assert result == expected
        finally:
            reset_context(token)

    def test_tilde_in_middle_of_path_not_expanded(self, tmp_path: Path):
        """A ``~`` that is not at the start should not trigger expansion."""
        home = tmp_path / "agent-home"
        home.mkdir()
        agent = Agent(home=home)
        ctx = ExecutionContext(
            provider=MockProvider(),
            workspace_root=tmp_path,
            agent=agent,
        )
        token = set_context(ctx)
        try:
            result = resolve_path("foo~bar")
            assert result == (tmp_path / "foo~bar").resolve()
        finally:
            reset_context(token)

    def test_tilde_no_context_falls_back_to_os(self):
        """When no context is active at all, ~ still uses OS expansion."""
        import os
        result = resolve_path("~/test")
        expected = Path(os.path.expanduser("~/test")).resolve()
        assert result == expected

    def test_relative_path_not_affected_by_tilde(self, tmp_path: Path):
        """A relative path without ~ is still resolved against workspace."""
        home = tmp_path / "agent-home"
        home.mkdir()
        agent = Agent(home=home)
        ctx = ExecutionContext(
            provider=MockProvider(),
            workspace_root=tmp_path,
            agent=agent,
        )
        token = set_context(ctx)
        try:
            result = resolve_path("subdir/file.txt")
            assert result == (tmp_path / "subdir" / "file.txt").resolve()
        finally:
            reset_context(token)


# ---------------------------------------------------------------------------
# shell_env
# ---------------------------------------------------------------------------


class TestShellEnv:
    def test_returns_none_without_context(self):
        """No context → inherit process environment unchanged."""
        assert shell_env() is None

    def test_returns_none_without_agent(self, tmp_path: Path):
        """Context without an agent → no HOME override."""
        ctx = ExecutionContext(
            provider=MockProvider(),
            workspace_root=tmp_path,
        )
        token = set_context(ctx)
        try:
            assert shell_env() is None
        finally:
            reset_context(token)

    def test_returns_none_when_agent_has_no_home(self, tmp_path: Path):
        """Agent with home=None → no HOME override."""
        agent = Agent()
        agent._home_resolved = True
        ctx = ExecutionContext(
            provider=MockProvider(),
            workspace_root=tmp_path,
            agent=agent,
        )
        token = set_context(ctx)
        try:
            assert shell_env() is None
        finally:
            reset_context(token)

    def test_overrides_home_when_agent_has_home(self, tmp_path: Path):
        """Agent with a home directory → HOME is overridden."""
        import os
        home = tmp_path / "agent-home"
        home.mkdir()
        agent = Agent(home=home)
        ctx = ExecutionContext(
            provider=MockProvider(),
            workspace_root=tmp_path,
            agent=agent,
        )
        token = set_context(ctx)
        try:
            env = shell_env()
            assert env is not None
            assert env["HOME"] == str(home)
            # Other env vars should still be present.
            assert env.get("PATH") == os.environ.get("PATH")
        finally:
            reset_context(token)


# ---------------------------------------------------------------------------
# ConsoleEventSink._scope_label
# ---------------------------------------------------------------------------

class TestConsoleScopeLabel:
    def _make_sink(self) -> ConsoleEventSink:
        return ConsoleEventSink(verbosity=Verbosity.NORMAL)

    def test_agent_only(self):
        agent = Agent(name="reviewer")
        scope = Scope(description="agent:Agent", metadata={"agent": agent})
        label = self._make_sink()._scope_label(scope)
        assert label == "Agent('reviewer')"

    def test_agent_with_session_key(self):
        agent = Agent(name="reviewer")
        scope = Scope(
            description="agent:Agent",
            metadata={"agent": agent, "session_key": "gitlab/123/issue/42"},
        )
        label = self._make_sink()._scope_label(scope)
        assert label == "Agent('reviewer') [gitlab/123/issue/42]"

    def test_no_agent_falls_back_to_description(self):
        scope = Scope(description="skill:summarize")
        label = self._make_sink()._scope_label(scope)
        assert label == "skill:summarize"

    def test_no_agent_with_session_key(self):
        scope = Scope(
            description="skill:summarize",
            metadata={"session_key": "github/456/change-request/7"},
        )
        label = self._make_sink()._scope_label(scope)
        assert label == "skill:summarize [github/456/change-request/7]"

    def test_empty_session_key_omitted(self):
        agent = Agent(name="bot")
        scope = Scope(
            description="agent:Agent",
            metadata={"agent": agent, "session_key": ""},
        )
        label = self._make_sink()._scope_label(scope)
        assert label == "Agent('bot')"

    def test_none_session_key_omitted(self):
        agent = Agent(name="bot")
        scope = Scope(
            description="agent:Agent",
            metadata={"agent": agent, "session_key": None},
        )
        label = self._make_sink()._scope_label(scope)
        assert label == "Agent('bot')"


# ---------------------------------------------------------------------------
# ConsoleEventSink._summarize_tool_args
# ---------------------------------------------------------------------------

class TestSummarizeToolArgs:
    def test_known_single_key_tool(self):
        result = ConsoleEventSink._summarize_tool_args(
            "run_shell", {"command": "git status", "timeout": 120},
        )
        assert result == "git status"

    def test_known_multi_key_tool(self):
        result = ConsoleEventSink._summarize_tool_args(
            "move_file", {"source": "a.py", "destination": "b.py"},
        )
        assert result == "source=a.py destination=b.py"

    def test_search_files_shows_pattern_and_path(self):
        result = ConsoleEventSink._summarize_tool_args(
            "search_files", {"pattern": "TODO", "path": "src/"},
        )
        assert result == "pattern=TODO path=src/"

    def test_unknown_tool_returns_none(self):
        result = ConsoleEventSink._summarize_tool_args(
            "some_unknown_tool", {"question": "hello?"},
        )
        assert result is None

    def test_known_tool_missing_keys_returns_none(self):
        result = ConsoleEventSink._summarize_tool_args(
            "read_file", {},
        )
        assert result is None

    def test_long_value_truncated(self):
        long_cmd = "x" * 200
        result = ConsoleEventSink._summarize_tool_args(
            "run_shell", {"command": long_cmd},
        )
        assert result is not None
        assert len(result) == ConsoleEventSink._TOOL_SUMMARY_MAX_LEN + 1  # +1 for ellipsis
        assert result.endswith("\u2026")

    def test_partial_keys_present(self):
        """When only some summary keys have values, the missing ones are skipped."""
        result = ConsoleEventSink._summarize_tool_args(
            "find_files", {"pattern": "*.py"},
        )
        assert result == "pattern=*.py"
