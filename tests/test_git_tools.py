"""Tests for git tool functions (src/thorn/tools/git.py)."""

from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path

import pytest

from thorn.core._context import ExecutionContext, reset_context, set_context
from thorn.core._provider import MockProvider


def _init_upstream(path: Path) -> str:
    """Create a bare git repo with one commit, return its file:// URL."""
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "--bare", str(path)], check=True, capture_output=True)

    # Create a temporary clone to add a commit, since bare repos have
    # no working tree.
    work = path.parent / "upstream_work"
    subprocess.run(["git", "clone", str(path), str(work)], check=True, capture_output=True)
    (work / "README.md").write_text("hello\n")
    subprocess.run(["git", "add", "README.md"], cwd=str(work), check=True, capture_output=True)
    subprocess.run(
        ["git", "-c", "user.name=test", "-c", "user.email=test@test", "commit", "-m", "init"],
        cwd=str(work),
        check=True,
        capture_output=True,
    )
    subprocess.run(["git", "push"], cwd=str(work), check=True, capture_output=True)

    return f"file://{path}"


@pytest.fixture
def upstream(tmp_path: Path) -> str:
    """A bare upstream repo with one commit; returns the file:// URL."""
    return _init_upstream(tmp_path / "upstream.git")


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    ws = tmp_path / "workspace"
    ws.mkdir()
    return ws


@pytest.fixture
def ctx_with_workspace(workspace: Path):
    """ExecutionContext whose workspace_root points at the test workspace."""
    provider = MockProvider()
    context = ExecutionContext(provider=provider, workspace_root=workspace)
    token = set_context(context)
    yield context
    reset_context(token)


class TestGitClone:
    async def test_normal_clone_produces_working_tree(
        self, upstream: str, workspace: Path, ctx_with_workspace: ExecutionContext,
    ) -> None:
        from thorn.tools.git import git_clone

        result = await git_clone(upstream, "repo")
        clone_dir = workspace / "repo"

        assert clone_dir.is_dir()
        assert (clone_dir / ".git").is_dir(), "normal clone should have a .git directory"
        assert (clone_dir / "README.md").is_file(), "normal clone should have checked-out files"
        assert "Cloned" in result

    async def test_bare_clone_produces_bare_repo(
        self, upstream: str, workspace: Path, ctx_with_workspace: ExecutionContext,
    ) -> None:
        from thorn.tools.git import git_clone

        result = await git_clone(upstream, "repo.git", bare=True)
        clone_dir = workspace / "repo.git"

        assert clone_dir.is_dir()
        assert not (clone_dir / ".git").exists(), "bare clone should not have a .git subdirectory"
        assert (clone_dir / "HEAD").is_file(), "bare clone should have HEAD at the top level"
        assert not (clone_dir / "README.md").exists(), "bare clone should not have checked-out files"
        assert "Cloned" in result

    async def test_fetch_when_directory_exists(
        self, upstream: str, workspace: Path, ctx_with_workspace: ExecutionContext,
    ) -> None:
        from thorn.tools.git import git_clone

        await git_clone(upstream, "repo")
        result = await git_clone(upstream, "repo")

        assert "Fetched" in result

    async def test_bare_fetch_when_directory_exists(
        self, upstream: str, workspace: Path, ctx_with_workspace: ExecutionContext,
    ) -> None:
        from thorn.tools.git import git_clone

        await git_clone(upstream, "repo.git", bare=True)
        result = await git_clone(upstream, "repo.git", bare=True)

        assert "Fetched" in result

    async def test_default_bare_is_false(
        self, upstream: str, workspace: Path, ctx_with_workspace: ExecutionContext,
    ) -> None:
        """Verify that the default clone is non-bare (a working tree)."""
        from thorn.tools.git import git_clone

        await git_clone(upstream, "repo")
        clone_dir = workspace / "repo"

        assert (clone_dir / ".git").is_dir()
        assert (clone_dir / "README.md").is_file()
