"""Tests for thorn.tools.git -- Git subprocess tools."""

from __future__ import annotations

import base64
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from thorn.tools.git import (
    GIT_TOOLS,
    GitError,
    _git_auth_env_for_current_agent,
    _git_identity_env,
    _run_git,
    git_add,
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
# Credential injection (per-call env vars, no URL embedding)
# ---------------------------------------------------------------------------


def _expected_basic_header(username: str, token: str) -> str:
    """Build the AUTHORIZATION header value our code is expected to produce."""
    encoded = base64.b64encode(f"{username}:{token}".encode()).decode()
    return f"AUTHORIZATION: basic {encoded}"


class TestGitAuthEnv:
    """Tests for `_git_auth_env_for_current_agent` -- the per-call HTTPS
    credential injection mechanism that replaces URL-embedded tokens."""

    _CTX_PATH = "thorn.core._context.get_context"

    def test_no_context_returns_empty(self) -> None:
        with patch(self._CTX_PATH, side_effect=RuntimeError):
            assert _git_auth_env_for_current_agent() == {}

    def test_no_agent_returns_empty(self, tmp_path: Path) -> None:
        from thorn.core._context import ExecutionContext
        from thorn.core._provider import MockProvider

        ctx = ExecutionContext(
            provider=MockProvider(), agent=None, runtime=None,
            workspace_root=tmp_path,
        )
        with patch(self._CTX_PATH, return_value=ctx):
            assert _git_auth_env_for_current_agent() == {}

    def test_no_project_metadata_returns_empty(self, tmp_path: Path) -> None:
        from thorn.core._agent import Agent
        from thorn.core._context import ExecutionContext
        from thorn.core._provider import MockProvider

        agent = Agent(metadata={})
        ctx = ExecutionContext(
            provider=MockProvider(), agent=agent, runtime=None,
            workspace_root=tmp_path,
        )
        with patch(self._CTX_PATH, return_value=ctx):
            assert _git_auth_env_for_current_agent() == {}

    def test_unknown_project_service_returns_empty(self, tmp_path: Path) -> None:
        from thorn.core._agent import Agent
        from thorn.core._context import ExecutionContext
        from thorn.core._provider import MockProvider
        from thorn.runtime import Runtime

        runtime = Runtime(provider=MockProvider(), workspace_root=tmp_path)
        agent = Agent(metadata={"project": "no-such-proj"})
        ctx = ExecutionContext(
            provider=MockProvider(), agent=agent, runtime=runtime,
            workspace_root=tmp_path,
        )
        with patch(self._CTX_PATH, return_value=ctx):
            assert _git_auth_env_for_current_agent() == {}

    def test_gitlab_legacy_token(self, tmp_path: Path) -> None:
        """No agent account on the forge -> falls back to forge-config token."""
        from thorn.core._agent import Agent
        from thorn.core._context import ExecutionContext
        from thorn.core._provider import MockProvider
        from thorn.runtime import Runtime
        from thorn.tools.forge import (
            GitLabForgeService,
            GitLabForgeServiceConfig,
            ProjectService,
            ProjectServiceConfig,
        )

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
            env = _git_auth_env_for_current_agent()

        assert env["GIT_CONFIG_COUNT"] == "1"
        assert env["GIT_CONFIG_KEY_0"] == (
            "http.https://gitlab.example.com/.extraheader"
        )
        assert env["GIT_CONFIG_VALUE_0"] == _expected_basic_header(
            "oauth2", "glpat-abc123",
        )
        assert "glpat-abc123" not in env["GIT_CONFIG_KEY_0"], (
            "URL prefix should not contain the token"
        )

    def test_github_legacy_token_strips_api_subdomain(
        self, tmp_path: Path,
    ) -> None:
        """github.com's API host is api.github.com but git URLs use github.com."""
        from thorn.core._agent import Agent
        from thorn.core._context import ExecutionContext
        from thorn.core._provider import MockProvider
        from thorn.runtime import Runtime
        from thorn.tools._github_connection import (
            GitHubConnectionConfig,
            GitHubPatAuth,
        )
        from thorn.tools.forge import (
            GitHubForgeService,
            ProjectService,
            ProjectServiceConfig,
        )

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
            env = _git_auth_env_for_current_agent()

        assert env["GIT_CONFIG_KEY_0"] == "http.https://github.com/.extraheader"
        assert env["GIT_CONFIG_VALUE_0"] == _expected_basic_header(
            "x-access-token", "ghp_testtok",
        )

    def test_gitlab_account_token_takes_precedence(self, tmp_path: Path) -> None:
        from thorn.core._account import (
            AgentAccountsConfig,
            ForgeAccountConfig,
            GitLabCredentials,
        )
        from thorn.core._agent import Agent
        from thorn.core._context import ExecutionContext
        from thorn.core._provider import MockProvider
        from thorn.runtime import Runtime
        from thorn.tools.forge import (
            GitLabForgeService,
            GitLabForgeServiceConfig,
            ProjectService,
            ProjectServiceConfig,
        )

        runtime = Runtime(provider=MockProvider(), workspace_root=tmp_path)
        runtime.register_service(
            GitLabForgeService(
                GitLabForgeServiceConfig(
                    url="https://gitlab.example.com", token="forge-config-token",
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
        accounts = AgentAccountsConfig(forge_accounts=[
            ForgeAccountConfig(
                forge="gl-forge",
                credentials=GitLabCredentials(token="account-token"),
                git_user_name="bot",
                git_user_email="bot@thorn",
            ),
        ])
        agent = Agent(metadata={"project": "my-proj"}, accounts=accounts)
        ctx = ExecutionContext(
            provider=MockProvider(), agent=agent, runtime=runtime,
            workspace_root=tmp_path,
        )
        with patch(self._CTX_PATH, return_value=ctx):
            env = _git_auth_env_for_current_agent()

        assert env["GIT_CONFIG_VALUE_0"] == _expected_basic_header(
            "oauth2", "account-token",
        )
        assert "forge-config-token" not in env["GIT_CONFIG_VALUE_0"]


class TestRunGitAuthInjection:
    """Black-box tests that `_run_git(..., auth=True)` actually merges the
    auth env vars into the subprocess environment."""

    _CTX_PATH = "thorn.core._context.get_context"

    async def test_auth_false_does_not_set_git_config_vars(
        self, git_repo: Path, tmp_path: Path,
    ) -> None:
        from thorn.core._agent import Agent
        from thorn.core._context import ExecutionContext
        from thorn.core._provider import MockProvider
        from thorn.runtime import Runtime
        from thorn.tools.forge import (
            GitLabForgeService,
            GitLabForgeServiceConfig,
            ProjectService,
            ProjectServiceConfig,
        )

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

        captured: dict[str, str | None] = {}

        async def fake_subprocess_exec(*args, env=None, **kwargs):
            captured["GIT_CONFIG_COUNT"] = (
                env.get("GIT_CONFIG_COUNT") if env else None
            )

            class _Proc:
                returncode = 0

                async def communicate(self):
                    return (b"", b"")

            return _Proc()

        with patch(self._CTX_PATH, return_value=ctx), patch(
            "asyncio.create_subprocess_exec", side_effect=fake_subprocess_exec,
        ):
            await _run_git("status", cwd=str(git_repo), auth=False)

        assert captured["GIT_CONFIG_COUNT"] is None

    async def test_auth_true_merges_git_config_vars(
        self, git_repo: Path, tmp_path: Path,
    ) -> None:
        from thorn.core._agent import Agent
        from thorn.core._context import ExecutionContext
        from thorn.core._provider import MockProvider
        from thorn.runtime import Runtime
        from thorn.tools.forge import (
            GitLabForgeService,
            GitLabForgeServiceConfig,
            ProjectService,
            ProjectServiceConfig,
        )

        runtime = Runtime(provider=MockProvider(), workspace_root=tmp_path)
        runtime.register_service(
            GitLabForgeService(
                GitLabForgeServiceConfig(
                    url="https://gitlab.example.com", token="abc",
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

        captured: dict[str, str | None] = {}

        async def fake_subprocess_exec(*args, env=None, **kwargs):
            assert env is not None
            captured["GIT_CONFIG_COUNT"] = env.get("GIT_CONFIG_COUNT")
            captured["GIT_CONFIG_KEY_0"] = env.get("GIT_CONFIG_KEY_0")
            captured["GIT_CONFIG_VALUE_0"] = env.get("GIT_CONFIG_VALUE_0")

            class _Proc:
                returncode = 0

                async def communicate(self):
                    return (b"", b"")

            return _Proc()

        with patch(self._CTX_PATH, return_value=ctx), patch(
            "asyncio.create_subprocess_exec", side_effect=fake_subprocess_exec,
        ):
            await _run_git("status", cwd=str(git_repo), auth=True)

        assert captured["GIT_CONFIG_COUNT"] == "1"
        assert captured["GIT_CONFIG_KEY_0"] == (
            "http.https://gitlab.example.com/.extraheader"
        )
        assert captured["GIT_CONFIG_VALUE_0"] == _expected_basic_header(
            "oauth2", "abc",
        )

    async def test_auth_true_appends_to_existing_git_config(
        self, git_repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """If the parent process already exports GIT_CONFIG_*, our entries
        are appended after the existing ones rather than overwriting them."""
        from thorn.core._agent import Agent
        from thorn.core._context import ExecutionContext
        from thorn.core._provider import MockProvider
        from thorn.runtime import Runtime
        from thorn.tools.forge import (
            GitLabForgeService,
            GitLabForgeServiceConfig,
            ProjectService,
            ProjectServiceConfig,
        )

        monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
        monkeypatch.setenv("GIT_CONFIG_KEY_0", "user.email")
        monkeypatch.setenv("GIT_CONFIG_VALUE_0", "site@policy.example")

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
        agent = Agent(metadata={
            "project": "my-proj",
            "git_user_name": "bot",
            "git_user_email": "bot@thorn",
        })
        ctx = ExecutionContext(
            provider=MockProvider(), agent=agent, runtime=runtime,
            workspace_root=tmp_path,
        )

        captured: dict[str, str | None] = {}

        async def fake_subprocess_exec(*args, env=None, **kwargs):
            assert env is not None
            for k in (
                "GIT_CONFIG_COUNT",
                "GIT_CONFIG_KEY_0", "GIT_CONFIG_VALUE_0",
                "GIT_CONFIG_KEY_1", "GIT_CONFIG_VALUE_1",
            ):
                captured[k] = env.get(k)

            class _Proc:
                returncode = 0

                async def communicate(self):
                    return (b"", b"")

            return _Proc()

        with patch(self._CTX_PATH, return_value=ctx), patch(
            "asyncio.create_subprocess_exec", side_effect=fake_subprocess_exec,
        ):
            await _run_git("status", cwd=str(git_repo), auth=True)

        assert captured["GIT_CONFIG_COUNT"] == "2"
        assert captured["GIT_CONFIG_KEY_0"] == "user.email"
        assert captured["GIT_CONFIG_VALUE_0"] == "site@policy.example"
        assert captured["GIT_CONFIG_KEY_1"] == (
            "http.https://gitlab.example.com/.extraheader"
        )


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
# _git_identity_env
# ---------------------------------------------------------------------------


class TestGitIdentityEnv:
    _CTX_PATH = "thorn.core._context.get_context"

    def test_returns_none_without_context(self) -> None:
        with patch(self._CTX_PATH, side_effect=RuntimeError):
            assert _git_identity_env() is None

    def test_returns_none_without_agent(self, tmp_path: Path) -> None:
        from thorn.core._context import ExecutionContext
        from thorn.core._provider import MockProvider

        ctx = ExecutionContext(
            provider=MockProvider(), agent=None, runtime=None,
            workspace_root=tmp_path,
        )
        with patch(self._CTX_PATH, return_value=ctx):
            assert _git_identity_env() is None

    def test_returns_none_when_agent_has_no_git_metadata(
        self, tmp_path: Path,
    ) -> None:
        from thorn.core._agent import Agent
        from thorn.core._context import ExecutionContext
        from thorn.core._provider import MockProvider

        agent = Agent(metadata={"project": "foo"})
        ctx = ExecutionContext(
            provider=MockProvider(), agent=agent, runtime=None,
            workspace_root=tmp_path,
        )
        with patch(self._CTX_PATH, return_value=ctx):
            assert _git_identity_env() is None

    def test_injects_name_and_email(self, tmp_path: Path) -> None:
        from thorn.core._agent import Agent
        from thorn.core._context import ExecutionContext
        from thorn.core._provider import MockProvider

        agent = Agent(metadata={
            "git_user_name": "thorn-bot",
            "git_user_email": "bot@thorn.dev",
        })
        ctx = ExecutionContext(
            provider=MockProvider(), agent=agent, runtime=None,
            workspace_root=tmp_path,
        )
        with patch(self._CTX_PATH, return_value=ctx):
            env = _git_identity_env()

        assert env is not None
        assert env["GIT_AUTHOR_NAME"] == "thorn-bot"
        assert env["GIT_COMMITTER_NAME"] == "thorn-bot"
        assert env["GIT_AUTHOR_EMAIL"] == "bot@thorn.dev"
        assert env["GIT_COMMITTER_EMAIL"] == "bot@thorn.dev"

    def test_injects_name_only(self, tmp_path: Path) -> None:
        from thorn.core._agent import Agent
        from thorn.core._context import ExecutionContext
        from thorn.core._provider import MockProvider

        agent = Agent(metadata={"git_user_name": "thorn-bot"})
        ctx = ExecutionContext(
            provider=MockProvider(), agent=agent, runtime=None,
            workspace_root=tmp_path,
        )
        with patch(self._CTX_PATH, return_value=ctx):
            env = _git_identity_env()

        assert env is not None
        assert env["GIT_AUTHOR_NAME"] == "thorn-bot"
        assert "GIT_AUTHOR_EMAIL" not in env

    def test_inherits_existing_env(self, tmp_path: Path) -> None:
        """The returned env dict should be a copy of os.environ plus the
        git identity vars, not *only* the git identity vars."""
        from thorn.core._agent import Agent
        from thorn.core._context import ExecutionContext
        from thorn.core._provider import MockProvider

        agent = Agent(metadata={
            "git_user_name": "bot",
            "git_user_email": "bot@x",
        })
        ctx = ExecutionContext(
            provider=MockProvider(), agent=agent, runtime=None,
            workspace_root=tmp_path,
        )
        with patch(self._CTX_PATH, return_value=ctx):
            env = _git_identity_env()

        assert env is not None
        assert "PATH" in env


class TestGitIdentityEnvWithAccounts:
    """Tests for _git_identity_env using account-based identity."""

    _CTX_PATH = "thorn.core._context.get_context"

    def test_prefers_account_identity_over_metadata(
        self, tmp_path: Path,
    ) -> None:
        from thorn.core._account import (
            AgentAccountsConfig,
            ForgeAccountConfig,
            GitLabCredentials,
        )
        from thorn.core._agent import Agent
        from thorn.core._context import ExecutionContext
        from thorn.core._provider import MockProvider
        from thorn.runtime import Runtime
        from thorn.tools.forge import (
            GitLabForgeService,
            GitLabForgeServiceConfig,
            ProjectService,
            ProjectServiceConfig,
        )

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
        accounts = AgentAccountsConfig(forge_accounts=[
            ForgeAccountConfig(
                forge="gl-forge",
                credentials=GitLabCredentials(token="t"),
                git_user_name="account-bot",
                git_user_email="account-bot@thorn",
            ),
        ])
        agent = Agent(
            metadata={
                "project": "my-proj",
                "git_user_name": "metadata-bot",
                "git_user_email": "metadata-bot@thorn",
            },
            accounts=accounts,
        )
        ctx = ExecutionContext(
            provider=MockProvider(), agent=agent, runtime=runtime,
            workspace_root=tmp_path,
        )
        with patch(self._CTX_PATH, return_value=ctx):
            env = _git_identity_env()

        assert env is not None
        assert env["GIT_AUTHOR_NAME"] == "account-bot"
        assert env["GIT_AUTHOR_EMAIL"] == "account-bot@thorn"

    def test_falls_back_to_metadata_when_no_account(
        self, tmp_path: Path,
    ) -> None:
        from thorn.core._agent import Agent
        from thorn.core._context import ExecutionContext
        from thorn.core._provider import MockProvider

        agent = Agent(metadata={
            "git_user_name": "metadata-bot",
            "git_user_email": "metadata-bot@thorn",
        })
        ctx = ExecutionContext(
            provider=MockProvider(), agent=agent, runtime=None,
            workspace_root=tmp_path,
        )
        with patch(self._CTX_PATH, return_value=ctx):
            env = _git_identity_env()

        assert env is not None
        assert env["GIT_AUTHOR_NAME"] == "metadata-bot"
        assert env["GIT_AUTHOR_EMAIL"] == "metadata-bot@thorn"

    def test_account_identity_without_runtime_falls_back(
        self, tmp_path: Path,
    ) -> None:
        """Account is set but no runtime -> can't resolve project -> fallback."""
        from thorn.core._account import (
            AgentAccountsConfig,
            ForgeAccountConfig,
            GitLabCredentials,
        )
        from thorn.core._agent import Agent
        from thorn.core._context import ExecutionContext
        from thorn.core._provider import MockProvider

        accounts = AgentAccountsConfig(forge_accounts=[
            ForgeAccountConfig(
                forge="gl-forge",
                credentials=GitLabCredentials(token="t"),
                git_user_name="account-bot",
                git_user_email="account-bot@thorn",
            ),
        ])
        agent = Agent(
            metadata={
                "project": "my-proj",
                "git_user_name": "metadata-bot",
                "git_user_email": "metadata-bot@thorn",
            },
            accounts=accounts,
        )
        ctx = ExecutionContext(
            provider=MockProvider(), agent=agent, runtime=None,
            workspace_root=tmp_path,
        )
        with patch(self._CTX_PATH, return_value=ctx):
            env = _git_identity_env()

        assert env is not None
        assert env["GIT_AUTHOR_NAME"] == "metadata-bot"


class TestGitCommitWithIdentity:
    """Verify that git_commit works when identity comes from agent metadata
    rather than the local git config."""

    async def test_commit_with_agent_identity(self, tmp_path: Path) -> None:
        """A repo with no local user.name/email still commits successfully
        when the agent has git identity metadata."""
        import subprocess
        from thorn.core._agent import Agent
        from thorn.core._context import ExecutionContext, set_context, reset_context
        from thorn.core._provider import MockProvider

        repo = tmp_path / "id-repo"
        repo.mkdir()
        subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)

        agent = Agent(metadata={
            "git_user_name": "test-bot",
            "git_user_email": "test-bot@thorn",
        })
        ctx = ExecutionContext(
            provider=MockProvider(), agent=agent, runtime=None,
            workspace_root=tmp_path,
        )
        token = set_context(ctx)
        try:
            (repo / "file.txt").write_text("hello\n")
            await git_add(str(repo))
            result = await git_commit(str(repo), "identity test commit")
            assert "identity test" in result.lower()

            log_output = subprocess.run(
                ["git", "log", "--format=%an <%ae>", "-1"],
                cwd=repo, check=True, capture_output=True, text=True,
            ).stdout.strip()
            assert "test-bot" in log_output
            assert "test-bot@thorn" in log_output
        finally:
            reset_context(token)


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
# git_add
# ---------------------------------------------------------------------------


class TestGitAdd:
    async def test_add_all_stages_new_file(self, git_repo: Path) -> None:
        (git_repo / "staged.txt").write_text("x\n")
        result = await git_add(str(git_repo))
        assert "staged" in result.lower() or result == "Staged changes."

    async def test_add_explicit_paths(self, git_repo: Path) -> None:
        (git_repo / "keep.txt").write_text("k\n")
        (git_repo / "skip.txt").write_text("s\n")
        await git_add(str(git_repo), paths=["keep.txt"])
        result = await git_status(str(git_repo))
        assert "keep.txt" in result
        assert "skip.txt" in result

    async def test_empty_paths_returns_error_message(self, git_repo: Path) -> None:
        result = await git_add(str(git_repo), paths=[])
        assert "Error" in result

    async def test_path_escapes_repo_rejected(self, git_repo: Path) -> None:
        with pytest.raises(ValueError, match="escapes"):
            await git_add(str(git_repo), paths=["../outside"])

    async def test_absolute_path_rejected(self, git_repo: Path) -> None:
        with pytest.raises(ValueError, match="relative"):
            await git_add(str(git_repo), paths=["/absolute/path"])


# ---------------------------------------------------------------------------
# git_commit
# ---------------------------------------------------------------------------


class TestGitCommit:
    async def test_commit_new_file(self, git_repo: Path) -> None:
        (git_repo / "new.txt").write_text("content\n")
        await git_add(str(git_repo))
        result = await git_commit(str(git_repo), "add new file")
        assert "new file" in result.lower() or "add new" in result.lower()

    async def test_nothing_to_commit_fails(self, git_repo: Path) -> None:
        with pytest.raises(GitError):
            await git_commit(str(git_repo), "empty")

    async def test_commit_appends_remainder_when_untracked_remain(
        self, git_repo: Path,
    ) -> None:
        (git_repo / "in_commit.txt").write_text("in\n")
        (git_repo / "left_out.txt").write_text("out\n")
        await git_add(str(git_repo), paths=["in_commit.txt"])
        result = await git_commit(str(git_repo), "partial")
        assert "Remaining changes after commit" in result
        assert "left_out.txt" in result


# ---------------------------------------------------------------------------
# git_log
# ---------------------------------------------------------------------------


class TestGitLog:
    async def test_shows_initial_commit(self, git_repo: Path) -> None:
        result = await git_log(str(git_repo))
        assert "initial" in result

    async def test_max_count(self, git_repo: Path) -> None:
        (git_repo / "a.txt").write_text("a\n")
        await git_add(str(git_repo))
        await git_commit(str(git_repo), "second commit")
        result = await git_log(str(git_repo), max_count=1)
        assert "second" in result
        assert "initial" not in result


# ---------------------------------------------------------------------------
# git_clone
# ---------------------------------------------------------------------------


class TestGitClone:
    async def test_clone_into_new_directory(
        self, git_repo: Path, tmp_path: Path,
    ) -> None:
        dest = str(tmp_path / "clone.git")
        result = await git_clone(str(git_repo), dest)
        assert "Cloned" in result
        assert os.path.isdir(dest)

    async def test_re_clone_into_existing_directory_fails(
        self, git_repo: Path, tmp_path: Path,
    ) -> None:
        """Re-cloning into the same destination must fail loudly rather than
        silently swap to a `git fetch` (which would leave the working tree
        stale and was the source of confusing 'git fetch --all' errors)."""
        dest = str(tmp_path / "clone.git")
        await git_clone(str(git_repo), dest)
        with pytest.raises(GitError) as exc_info:
            await git_clone(str(git_repo), dest)
        assert "already exists" in exc_info.value.output

    async def test_clone_does_not_embed_token_in_remote_url(
        self, git_repo: Path, tmp_path: Path,
    ) -> None:
        """Regression guard: even when an agent context is active and would
        provide a token, the resulting clone's remote.origin.url must be the
        plain URL we passed -- credentials are injected per-call via env
        vars, not baked into .git/config."""
        from thorn.core._agent import Agent
        from thorn.core._context import ExecutionContext, reset_context, set_context
        from thorn.core._provider import MockProvider
        from thorn.runtime import Runtime
        from thorn.tools.forge import (
            GitLabForgeService,
            GitLabForgeServiceConfig,
            ProjectService,
            ProjectServiceConfig,
        )

        runtime = Runtime(provider=MockProvider(), workspace_root=tmp_path)
        runtime.register_service(
            GitLabForgeService(
                GitLabForgeServiceConfig(
                    url="https://gitlab.example.com",
                    token="should-not-appear-in-config",
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
        token_handle = set_context(ctx)
        try:
            dest = str(tmp_path / "clone.git")
            await git_clone(str(git_repo), dest)
        finally:
            reset_context(token_handle)

        config = (Path(dest) / ".git" / "config").read_text()
        assert "should-not-appear-in-config" not in config, (
            "Token leaked into .git/config -- per-call credential injection "
            "regressed to URL embedding"
        )


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
        await git_add(str(git_repo))
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
        assert len(GIT_TOOLS) == 12


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
        await git_add("myrepo")
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
