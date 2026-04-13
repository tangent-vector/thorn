"""Tests for thorn.tools.git -- Git subprocess tools."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from thorn.tools.git import (
    GIT_TOOLS,
    GitError,
    _inject_url_credentials,
    _run_git,
    git_branch,
    git_clone,
    git_commit,
    git_diff,
    git_fetch,
    git_log,
    git_pull,
    git_push,
    git_status,
    git_worktree_add,
    git_worktree_remove,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def git_repo(tmp_path: Path) -> Path:
    """Create a minimal git repository with one commit."""
    import subprocess

    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@test.com"],
        cwd=repo, check=True, capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=repo, check=True, capture_output=True,
    )
    (repo / "README.md").write_text("# Test\n")
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "initial"],
        cwd=repo, check=True, capture_output=True,
    )
    return repo


@pytest.fixture()
def bare_repo(tmp_path: Path, git_repo: Path) -> Path:
    """Clone the git_repo as a bare repository."""
    import subprocess

    bare = tmp_path / "bare.git"
    subprocess.run(
        ["git", "clone", "--bare", str(git_repo), str(bare)],
        check=True, capture_output=True,
    )
    return bare


# ---------------------------------------------------------------------------
# Credential injection
# ---------------------------------------------------------------------------


class TestInjectUrlCredentials:
    _CTX_PATH = "thorn.core._context.get_context"

    def test_no_context_returns_url_unchanged(self) -> None:
        url = "https://gitlab.example.com/group/project.git"
        with patch(self._CTX_PATH, side_effect=RuntimeError):
            assert _inject_url_credentials(url) == url

    def test_no_agent_returns_url_unchanged(self, tmp_path: Path) -> None:
        from thorn.core._context import ExecutionContext
        from thorn.core._provider import MockProvider

        ctx = ExecutionContext(
            provider=MockProvider(), agent=None, runtime=None,
            workspace_root=tmp_path,
        )
        with patch(self._CTX_PATH, return_value=ctx):
            url = "https://gitlab.example.com/group/project.git"
            assert _inject_url_credentials(url) == url

    def test_no_project_metadata_returns_url_unchanged(self, tmp_path: Path) -> None:
        from thorn.core._agent import Agent
        from thorn.core._context import ExecutionContext
        from thorn.core._provider import MockProvider

        agent = Agent(metadata={})
        ctx = ExecutionContext(
            provider=MockProvider(), agent=agent, runtime=None,
            workspace_root=tmp_path,
        )
        with patch(self._CTX_PATH, return_value=ctx):
            url = "https://gitlab.example.com/group/project.git"
            assert _inject_url_credentials(url) == url

    def test_gitlab_rewrites_https_url_via_forge(
        self, tmp_path: Path,
    ) -> None:
        from thorn.core._agent import Agent
        from thorn.core._context import ExecutionContext
        from thorn.core._provider import MockProvider
        from thorn.runtime import Runtime
        from thorn.tools.forge import GitLabForgeService, GitLabForgeServiceConfig
        from thorn.tools.forge import ProjectService, ProjectServiceConfig

        runtime = Runtime(provider=MockProvider(), workspace_root=tmp_path)
        runtime.register_service(
            GitLabForgeService(
                GitLabForgeServiceConfig(
                    url="https://gitlab.example.com", token="glpat-abc123",
                ),
                service_name="gl-forge",
            ),
        )
        runtime.register_service(
            ProjectService(
                ProjectServiceConfig(forge="gl-forge", native_id="1"),
                service_name="my-proj",
            ),
        )
        agent = Agent(metadata={"project": "my-proj"})
        ctx = ExecutionContext(
            provider=MockProvider(), agent=agent, runtime=runtime,
            workspace_root=tmp_path,
        )
        with patch(self._CTX_PATH, return_value=ctx):
            result = _inject_url_credentials(
                "https://gitlab.example.com/group/project.git",
            )
        assert result == (
            "https://oauth2:glpat-abc123@gitlab.example.com/group/project.git"
        )

    def test_github_uses_x_access_token(
        self, tmp_path: Path,
    ) -> None:
        from thorn.core._agent import Agent
        from thorn.core._context import ExecutionContext
        from thorn.core._provider import MockProvider
        from thorn.runtime import Runtime
        from thorn.tools._github_connection import GitHubConnectionConfig, GitHubPatAuth
        from thorn.tools.forge import GitHubForgeService, ProjectService, ProjectServiceConfig

        runtime = Runtime(provider=MockProvider(), workspace_root=tmp_path)
        runtime.register_service(
            GitHubForgeService(
                GitHubConnectionConfig(auth=GitHubPatAuth(token="ghp_testtok")),
                service_name="gh-forge",
            ),
        )
        runtime.register_service(
            ProjectService(
                ProjectServiceConfig(forge="gh-forge", native_id="o/r"),
                service_name="proj",
            ),
        )
        agent = Agent(metadata={"project": "proj"})
        ctx = ExecutionContext(
            provider=MockProvider(), agent=agent, runtime=runtime,
            workspace_root=tmp_path,
        )
        with patch(self._CTX_PATH, return_value=ctx):
            result = _inject_url_credentials(
                "https://github.com/o/r.git",
            )
        assert result == "https://x-access-token:ghp_testtok@github.com/o/r.git"

    def test_non_https_url_unchanged(self, tmp_path: Path) -> None:
        from thorn.core._agent import Agent
        from thorn.core._context import ExecutionContext
        from thorn.core._provider import MockProvider
        from thorn.runtime import Runtime
        from thorn.tools.forge import GitLabForgeService, GitLabForgeServiceConfig
        from thorn.tools.forge import ProjectService, ProjectServiceConfig

        runtime = Runtime(provider=MockProvider(), workspace_root=tmp_path)
        runtime.register_service(
            GitLabForgeService(
                GitLabForgeServiceConfig(
                    url="https://gitlab.example.com", token="t",
                ),
                service_name="gl-forge",
            ),
        )
        runtime.register_service(
            ProjectService(
                ProjectServiceConfig(forge="gl-forge", native_id="1"),
                service_name="my-proj",
            ),
        )
        agent = Agent(metadata={"project": "my-proj"})
        ctx = ExecutionContext(
            provider=MockProvider(), agent=agent, runtime=runtime,
            workspace_root=tmp_path,
        )
        with patch(self._CTX_PATH, return_value=ctx):
            url = "git@gitlab.example.com:group/project.git"
            assert _inject_url_credentials(url) == url


# ---------------------------------------------------------------------------
# _run_git helper
# ---------------------------------------------------------------------------


class TestRunGit:
    async def test_successful_command(self, git_repo: Path) -> None:
        code, output = await _run_git("status", cwd=str(git_repo))
        assert code == 0

    async def test_failure_raises_git_error(self, tmp_path: Path) -> None:
        with pytest.raises(GitError) as exc_info:
            await _run_git("log", cwd=str(tmp_path))
        assert exc_info.value.returncode != 0

    async def test_check_false_returns_nonzero(self, tmp_path: Path) -> None:
        code, _ = await _run_git("log", cwd=str(tmp_path), check=False)
        assert code != 0


# ---------------------------------------------------------------------------
# git_status
# ---------------------------------------------------------------------------


class TestGitStatus:
    async def test_clean_repo(self, git_repo: Path) -> None:
        result = await git_status(str(git_repo))
        assert "clean" in result.lower()

    async def test_dirty_repo(self, git_repo: Path) -> None:
        (git_repo / "new_file.txt").write_text("hello\n")
        result = await git_status(str(git_repo))
        assert "new_file.txt" in result


# ---------------------------------------------------------------------------
# git_diff
# ---------------------------------------------------------------------------


class TestGitDiff:
    async def test_no_changes(self, git_repo: Path) -> None:
        result = await git_diff(str(git_repo))
        assert "no" in result.lower() and "changes" in result.lower()

    async def test_unstaged_changes(self, git_repo: Path) -> None:
        (git_repo / "README.md").write_text("# Updated\n")
        result = await git_diff(str(git_repo))
        assert "Updated" in result

    async def test_staged_changes(self, git_repo: Path) -> None:
        import subprocess

        (git_repo / "README.md").write_text("# Staged\n")
        subprocess.run(
            ["git", "add", "README.md"], cwd=git_repo,
            check=True, capture_output=True,
        )
        result = await git_diff(str(git_repo), staged=True)
        assert "Staged" in result


# ---------------------------------------------------------------------------
# git_branch
# ---------------------------------------------------------------------------


class TestGitBranch:
    async def test_create_branch(self, git_repo: Path) -> None:
        result = await git_branch(str(git_repo), "feature-x")
        assert "feature-x" in result

    async def test_duplicate_branch_fails(self, git_repo: Path) -> None:
        await git_branch(str(git_repo), "feature-y")
        with pytest.raises(GitError):
            await git_branch(str(git_repo), "feature-y")


# ---------------------------------------------------------------------------
# git_commit
# ---------------------------------------------------------------------------


class TestGitCommit:
    async def test_commit_new_file(self, git_repo: Path) -> None:
        (git_repo / "new.txt").write_text("content\n")
        result = await git_commit(str(git_repo), "add new file")
        assert "new file" in result.lower() or "add new" in result.lower()

    async def test_nothing_to_commit_fails(self, git_repo: Path) -> None:
        with pytest.raises(GitError):
            await git_commit(str(git_repo), "empty")


# ---------------------------------------------------------------------------
# git_log
# ---------------------------------------------------------------------------


class TestGitLog:
    async def test_shows_initial_commit(self, git_repo: Path) -> None:
        result = await git_log(str(git_repo))
        assert "initial" in result

    async def test_max_count(self, git_repo: Path) -> None:
        (git_repo / "a.txt").write_text("a\n")
        await git_commit(str(git_repo), "second commit")
        result = await git_log(str(git_repo), max_count=1)
        assert "second" in result
        assert "initial" not in result


# ---------------------------------------------------------------------------
# git_clone
# ---------------------------------------------------------------------------


class TestGitClone:
    async def test_clone_bare(self, git_repo: Path, tmp_path: Path) -> None:
        dest = str(tmp_path / "clone.git")
        result = await git_clone(str(git_repo), dest)
        assert "Cloned" in result
        assert os.path.isdir(dest)

    async def test_fetch_existing(self, git_repo: Path, tmp_path: Path) -> None:
        dest = str(tmp_path / "clone.git")
        await git_clone(str(git_repo), dest)
        result = await git_clone(str(git_repo), dest)
        assert "Fetched" in result


# ---------------------------------------------------------------------------
# git_fetch
# ---------------------------------------------------------------------------


class TestGitFetch:
    async def test_fetch_updates_bare_repo(
        self, bare_repo: Path, git_repo: Path,
    ) -> None:
        """Fetch into a bare clone picks up new commits from the source."""
        import subprocess

        (git_repo / "after_clone.txt").write_text("new\n")
        subprocess.run(
            ["git", "add", "-A"], cwd=git_repo, check=True, capture_output=True,
        )
        subprocess.run(
            ["git", "commit", "-m", "post-clone commit"],
            cwd=git_repo, check=True, capture_output=True,
        )

        result = await git_fetch(str(bare_repo), remote="origin")
        assert "Fetched" in result

    async def test_fetch_nonexistent_remote_fails(self, git_repo: Path) -> None:
        with pytest.raises(GitError):
            await git_fetch(str(git_repo), remote="nonexistent")


# ---------------------------------------------------------------------------
# git_pull
# ---------------------------------------------------------------------------


class TestGitPull:
    async def test_pull_into_worktree(
        self, bare_repo: Path, git_repo: Path, tmp_path: Path,
    ) -> None:
        """Pull brings remote changes into a worktree's working tree."""
        import subprocess

        wt = tmp_path / "worktree"
        branch_proc = subprocess.run(
            ["git", "branch", "--show-current"],
            cwd=git_repo, check=True, capture_output=True, text=True,
        )
        default_branch = branch_proc.stdout.strip()

        await git_worktree_add(
            str(bare_repo), str(wt), "pull-test", start_point=default_branch,
        )

        (git_repo / "pulled.txt").write_text("from upstream\n")
        subprocess.run(
            ["git", "add", "-A"], cwd=git_repo, check=True, capture_output=True,
        )
        subprocess.run(
            ["git", "commit", "-m", "upstream commit"],
            cwd=git_repo, check=True, capture_output=True,
        )

        await git_fetch(str(bare_repo), remote="origin")
        result = await git_pull(str(wt), remote="origin", branch=default_branch)
        assert "Pulled" in result
        assert (wt / "pulled.txt").exists()

    async def test_pull_no_branch_tracking(self, git_repo: Path) -> None:
        """Pull without explicit branch on a repo with no tracking raises."""
        with pytest.raises(GitError):
            await git_pull(str(git_repo))


# ---------------------------------------------------------------------------
# git_push
# ---------------------------------------------------------------------------


class TestGitPush:
    async def test_push_to_local_remote(
        self, git_repo: Path, tmp_path: Path,
    ) -> None:
        import subprocess

        remote_bare = tmp_path / "remote.git"
        subprocess.run(
            ["git", "init", "--bare", str(remote_bare)],
            check=True, capture_output=True,
        )
        subprocess.run(
            ["git", "remote", "add", "origin", str(remote_bare)],
            cwd=git_repo, check=True, capture_output=True,
        )
        (git_repo / "push_test.txt").write_text("push me\n")
        await git_commit(str(git_repo), "for push test")

        current_branch_proc = subprocess.run(
            ["git", "branch", "--show-current"],
            cwd=git_repo, check=True, capture_output=True, text=True,
        )
        branch = current_branch_proc.stdout.strip()
        result = await git_push(str(git_repo), branch)
        assert "Pushed" in result


# ---------------------------------------------------------------------------
# git_worktree_add / git_worktree_remove
# ---------------------------------------------------------------------------


class TestGitWorktree:
    async def test_add_and_remove(
        self, bare_repo: Path, tmp_path: Path,
    ) -> None:
        wt = str(tmp_path / "worktree")
        result = await git_worktree_add(
            str(bare_repo), wt, "wt-branch",
            start_point="HEAD",
        )
        assert "Created worktree" in result
        assert os.path.isdir(wt)
        assert os.path.isfile(os.path.join(wt, "README.md"))

        result = await git_worktree_remove(str(bare_repo), wt)
        assert "Removed" in result
        assert not os.path.isdir(wt)


# ---------------------------------------------------------------------------
# GIT_TOOLS list
# ---------------------------------------------------------------------------


class TestGitToolsList:
    def test_all_tools_have_thorn_tool_marker(self) -> None:
        for fn in GIT_TOOLS:
            assert getattr(fn, "_thorn_tool", False), (
                f"{fn.__name__} is missing the @tool decorator"  # type: ignore[union-attr]
            )

    def test_expected_count(self) -> None:
        assert len(GIT_TOOLS) == 11


# ---------------------------------------------------------------------------
# Workspace-aware path resolution
# ---------------------------------------------------------------------------


class TestGitWorkspaceResolution:
    """Verify that git tools resolve relative paths against the workspace."""

    @pytest.fixture()
    def workspace_repo(self, tmp_path: Path) -> tuple[Path, Path]:
        """Create a workspace with a git repo inside it.

        Returns (workspace, git_repo).
        """
        import subprocess

        workspace = tmp_path / "agent_workspace"
        workspace.mkdir()
        repo = workspace / "myrepo"
        repo.mkdir()
        subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
        subprocess.run(
            ["git", "config", "user.email", "test@test.com"],
            cwd=repo, check=True, capture_output=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Test"],
            cwd=repo, check=True, capture_output=True,
        )
        (repo / "README.md").write_text("# Workspace test\n")
        subprocess.run(["git", "add", "-A"], cwd=repo, check=True, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "initial"],
            cwd=repo, check=True, capture_output=True,
        )
        return workspace, repo

    @pytest.fixture()
    def ctx_token(self, workspace_repo: tuple[Path, Path]):
        from thorn.core._context import ExecutionContext, set_context, reset_context
        from thorn.core._provider import MockProvider

        workspace, _repo = workspace_repo
        ctx = ExecutionContext(
            provider=MockProvider(),
            workspace_root=workspace,
        )
        token = set_context(ctx)
        yield token
        reset_context(token)

    async def test_git_status_relative_to_workspace(
        self, workspace_repo: tuple[Path, Path], ctx_token,
    ) -> None:
        _workspace, repo = workspace_repo
        result = await git_status("myrepo")
        assert "clean" in result.lower()

    async def test_git_log_relative_to_workspace(
        self, workspace_repo: tuple[Path, Path], ctx_token,
    ) -> None:
        result = await git_log("myrepo")
        assert "initial" in result

    async def test_git_commit_relative_to_workspace(
        self, workspace_repo: tuple[Path, Path], ctx_token,
    ) -> None:
        workspace, repo = workspace_repo
        (repo / "new.txt").write_text("hello\n")
        result = await git_commit("myrepo", "add new file")
        assert "new file" in result.lower() or "add new" in result.lower()

    async def test_git_clone_relative_to_workspace(
        self, workspace_repo: tuple[Path, Path], ctx_token,
    ) -> None:
        workspace, repo = workspace_repo
        result = await git_clone(str(repo), "repos/cloned.git")
        assert "Cloned" in result
        assert (workspace / "repos" / "cloned.git").is_dir()

    async def test_git_worktree_relative_to_workspace(
        self, workspace_repo: tuple[Path, Path], ctx_token,
    ) -> None:
        """Worktree paths resolve against workspace, not the bare repo CWD."""
        import subprocess

        workspace, repo = workspace_repo
        bare = workspace / "repos" / "bare.git"
        subprocess.run(
            ["git", "clone", "--bare", str(repo), str(bare)],
            check=True, capture_output=True,
        )
        result = await git_worktree_add(
            "repos/bare.git",
            "worktrees/my-branch",
            "my-branch",
            start_point="HEAD",
        )
        assert "Created worktree" in result
        wt_path = workspace / "worktrees" / "my-branch"
        assert wt_path.is_dir(), (
            f"Worktree should exist at {wt_path}, "
            f"not inside the bare repo"
        )
        assert (wt_path / "README.md").is_file()
