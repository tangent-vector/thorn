"""Tests for thorn.core._context — scope chain, ExecutionContext, contextvar management."""

from __future__ import annotations

import pytest

from pathlib import Path

from thorn.core._context import (
    ExecutionContext,
    NullEventSink,
    Scope,
    get_context,
    reset_context,
    resolve_path,
    set_context,
)
from thorn.core._provider import MockProvider


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

    def test_push_propagates_workspace_instructions(self):
        ctx = ExecutionContext(
            provider=MockProvider(),
            workspace_instructions="Be concise.",
        )
        child = ctx.push_scope("x")
        assert child.workspace_instructions == "Be concise."

    def test_push_propagates_none_workspace_instructions(self):
        ctx = ExecutionContext(provider=MockProvider())
        child = ctx.push_scope("x")
        assert child.workspace_instructions is None


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
