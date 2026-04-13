"""Tests for thorn.tools.github -- GitHub API tools.

Uses mock objects instead of a real GitHub instance.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from thorn.tools._github_connection import GitHubConnectionConfig, GitHubPatAuth
from thorn.tools.github import (
    GITHUB_TOOLS,
    GitHubClient,
    github_create_pull_request,
    github_get_pull_request,
    github_get_repo_info,
    github_list_comments,
    github_list_pull_requests,
    github_mark_notification_read,
    github_post_comment,
    github_read_file,
    github_read_issue,
    set_client,
)


# ---------------------------------------------------------------------------
# GitHubConnectionConfig
# ---------------------------------------------------------------------------


class TestGitHubConnectionConfig:
    def test_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GITHUB_TOKEN", "ghp_secret")
        monkeypatch.delenv("GITHUB_URL", raising=False)
        config = GitHubConnectionConfig.from_env()
        assert config.auth.kind == "pat"
        assert config.auth.token == "ghp_secret"
        assert config.base_url == "https://api.github.com"

    def test_from_env_custom_url(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GITHUB_TOKEN", "ghp_secret")
        monkeypatch.setenv("GITHUB_URL", "https://github.example.com/api/v3")
        config = GitHubConnectionConfig.from_env()
        assert config.base_url == "https://github.example.com/api/v3"

    def test_from_env_missing_token(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        monkeypatch.delenv("GITHUB_URL", raising=False)
        with pytest.raises(ValueError, match="GITHUB_TOKEN"):
            GitHubConnectionConfig.from_env()


# ---------------------------------------------------------------------------
# Mock object factories
# ---------------------------------------------------------------------------


def _make_mock_label(name: str) -> MagicMock:
    label = MagicMock()
    label.name = name
    return label


def _make_mock_user(login: str = "alice", user_type: str = "User") -> MagicMock:
    user = MagicMock()
    user.login = login
    user.type = user_type
    return user


def _make_mock_branch_ref(ref: str) -> MagicMock:
    branch = MagicMock()
    branch.ref = ref
    return branch


def _make_mock_issue(**overrides: Any) -> MagicMock:
    issue = MagicMock()
    issue.number = overrides.get("number", 42)
    issue.title = overrides.get("title", "Fix the widget")
    issue.state = overrides.get("state", "open")
    issue.body = overrides.get("body", "It is broken.")
    issue.labels = overrides.get(
        "labels",
        [_make_mock_label("bug"), _make_mock_label("priority:high")],
    )
    issue.assignees = overrides.get(
        "assignees",
        [_make_mock_user("alice")],
    )
    issue.html_url = overrides.get(
        "html_url", "https://github.com/octocat/hello-world/issues/42"
    )
    return issue


def _make_mock_pr(**overrides: Any) -> MagicMock:
    pr = MagicMock()
    pr.number = overrides.get("number", 7)
    pr.title = overrides.get("title", "Fix widget")
    pr.state = overrides.get("state", "open")
    pr.body = overrides.get("body", "Fixes #42")
    pr.html_url = overrides.get(
        "html_url",
        "https://github.com/octocat/hello-world/pull/7",
    )
    pr.head = _make_mock_branch_ref(overrides.get("head", "fix-widget"))
    pr.base = _make_mock_branch_ref(overrides.get("base", "main"))
    pr.mergeable = overrides.get("mergeable", True)
    pr.mergeable_state = overrides.get("mergeable_state", "clean")
    pr.merged = overrides.get("merged", False)
    pr.user = overrides.get("user", _make_mock_user("bob"))
    return pr


def _make_mock_comment(**overrides: Any) -> MagicMock:
    comment = MagicMock()
    comment.id = overrides.get("id", 100)
    comment.user = overrides.get("user", _make_mock_user("reviewer"))
    comment.body = overrides.get("body", "Please fix the naming.")
    comment.created_at = overrides.get(
        "created_at",
        datetime(2026, 4, 8, 10, 0, 0, tzinfo=timezone.utc),
    )
    return comment


def _make_mock_content_file(**overrides: Any) -> MagicMock:
    content = MagicMock()
    content.path = overrides.get("path", "README.md")
    content.decoded_content = overrides.get(
        "decoded_content", b"# Hello World\n",
    )
    return content


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_client() -> GitHubClient:
    """Build a GitHubClient with a mocked PyGithub backend.

    Patches ``_Github``, ``_GHAuth``, and ``_HAS_GITHUB`` so that tests
    work regardless of whether ``PyGithub`` is installed.
    """
    with (
        patch("thorn.tools.github._Github") as mock_gh_cls,
        patch("thorn.tools.github._HAS_GITHUB", True),
    ):
        mock_gh_instance = MagicMock()
        mock_gh_cls.return_value = mock_gh_instance

        config = GitHubConnectionConfig(
            base_url="https://api.github.com",
            auth=GitHubPatAuth(token="test-token"),
        )
        client = GitHubClient(config)
    return client


@pytest.fixture(autouse=True)
def _install_mock_client(mock_client: GitHubClient) -> None:
    """Install the mock client as the module-level client for all tests."""
    set_client(mock_client)
    yield
    set_client(None)


# ---------------------------------------------------------------------------
# GitHubClient methods
# ---------------------------------------------------------------------------


REPO = "octocat/hello-world"


class TestGitHubClientGetIssue:
    def test_returns_dict(self, mock_client: GitHubClient) -> None:
        mock_issue = _make_mock_issue()
        mock_client._gh.get_repo.return_value.get_issue.return_value = mock_issue

        result = mock_client.get_issue(REPO, 42)
        assert result["number"] == 42
        assert result["title"] == "Fix the widget"
        assert result["labels"] == ["bug", "priority:high"]
        assert result["assignees"] == ["alice"]


class TestGitHubClientPostComment:
    def test_posts_comment(self, mock_client: GitHubClient) -> None:
        mock_issue = _make_mock_issue()
        mock_client._gh.get_repo.return_value.get_issue.return_value = mock_issue

        mock_client.post_comment(REPO, 42, "Fixed!")
        mock_issue.create_comment.assert_called_once_with("Fixed!")


class TestGitHubClientCreatePR:
    def test_returns_dict(self, mock_client: GitHubClient) -> None:
        mock_pr = _make_mock_pr()
        mock_repo = mock_client._gh.get_repo.return_value
        mock_repo.create_pull.return_value = mock_pr

        result = mock_client.create_pull_request(REPO, "fix-widget", "Fix widget")
        assert result["number"] == 7
        assert result["head"] == "fix-widget"

    def test_passes_arguments(self, mock_client: GitHubClient) -> None:
        mock_pr = _make_mock_pr()
        mock_repo = mock_client._gh.get_repo.return_value
        mock_repo.create_pull.return_value = mock_pr

        mock_client.create_pull_request(
            REPO, "feat", "Add feature", base="develop", body="Details",
        )
        mock_repo.create_pull.assert_called_once_with(
            title="Add feature", body="Details", head="feat", base="develop",
        )


class TestGitHubClientGetPR:
    def test_returns_dict(self, mock_client: GitHubClient) -> None:
        mock_pr = _make_mock_pr()
        mock_client._gh.get_repo.return_value.get_pull.return_value = mock_pr

        result = mock_client.get_pull_request(REPO, 7)
        assert result["mergeable"] is True
        assert result["mergeable_state"] == "clean"
        assert result["merged"] is False


class TestGitHubClientListPRs:
    def test_returns_list_of_dicts(self, mock_client: GitHubClient) -> None:
        mock_client._gh.get_repo.return_value.get_pulls.return_value = [
            _make_mock_pr(number=1, title="PR one"),
            _make_mock_pr(number=2, title="PR two"),
        ]
        result = mock_client.list_pull_requests(REPO)
        assert len(result) == 2
        assert result[0]["title"] == "PR one"


class TestGitHubClientGetRepoInfo:
    def test_returns_dict(self, mock_client: GitHubClient) -> None:
        mock_repo = mock_client._gh.get_repo.return_value
        mock_repo.full_name = "octocat/hello-world"
        mock_repo.name = "hello-world"
        mock_repo.clone_url = "https://github.com/octocat/hello-world.git"
        mock_repo.ssh_url = "git@github.com:octocat/hello-world.git"
        mock_repo.default_branch = "main"
        mock_repo.html_url = "https://github.com/octocat/hello-world"
        mock_repo.description = "A test repo"

        result = mock_client.get_repo_info(REPO)
        assert result["full_name"] == "octocat/hello-world"
        assert result["default_branch"] == "main"
        assert result["description"] == "A test repo"


class TestGitHubClientReadFile:
    def test_returns_dict(self, mock_client: GitHubClient) -> None:
        mock_content = _make_mock_content_file()
        mock_client._gh.get_repo.return_value.get_contents.return_value = mock_content

        result = mock_client.read_file(REPO, "README.md")
        assert result["file_path"] == "README.md"
        assert result["content"] == "# Hello World\n"

    def test_directory_raises(self, mock_client: GitHubClient) -> None:
        mock_client._gh.get_repo.return_value.get_contents.return_value = [
            _make_mock_content_file(path="dir/a.txt"),
            _make_mock_content_file(path="dir/b.txt"),
        ]
        with pytest.raises(ValueError, match="directory"):
            mock_client.read_file(REPO, "dir")


class TestGitHubClientListComments:
    def test_returns_list_of_dicts(self, mock_client: GitHubClient) -> None:
        comments = [
            _make_mock_comment(id=1, body="First comment"),
            _make_mock_comment(id=2, body="Second comment"),
        ]
        mock_issue = _make_mock_issue()
        mock_issue.get_comments.return_value = comments
        mock_client._gh.get_repo.return_value.get_issue.return_value = mock_issue

        result = mock_client.list_comments(REPO, 42)
        assert len(result) == 2
        assert result[0]["body"] == "First comment"
        assert result[1]["body"] == "Second comment"

    def test_includes_bot_flag(self, mock_client: GitHubClient) -> None:
        bot_user = _make_mock_user("dependabot[bot]", user_type="Bot")
        comments = [_make_mock_comment(user=bot_user, body="Bump version")]
        mock_issue = _make_mock_issue()
        mock_issue.get_comments.return_value = comments
        mock_client._gh.get_repo.return_value.get_issue.return_value = mock_issue

        result = mock_client.list_comments(REPO, 42)
        assert result[0]["is_bot"] is True


class TestGitHubClientMarkNotificationRead:
    def test_calls_api(self, mock_client: GitHubClient) -> None:
        with patch("thorn.tools.github.httpx") as mock_httpx:
            mock_response = MagicMock()
            mock_response.raise_for_status = MagicMock()
            mock_httpx.patch.return_value = mock_response

            mock_client.mark_notification_read("12345")

            mock_httpx.patch.assert_called_once()
            call_args = mock_httpx.patch.call_args
            assert "/notifications/threads/12345" in call_args[0][0]


# ---------------------------------------------------------------------------
# @tool functions
# ---------------------------------------------------------------------------


class TestGitHubReadIssueTool:
    async def test_formats_output(self, mock_client: GitHubClient) -> None:
        mock_issue = _make_mock_issue()
        mock_client._gh.get_repo.return_value.get_issue.return_value = mock_issue

        result = await github_read_issue(REPO, 42)
        assert "Fix the widget" in result
        assert "#42" in result
        assert "bug" in result
        assert "alice" in result


class TestGitHubPostCommentTool:
    async def test_posts_and_confirms(self, mock_client: GitHubClient) -> None:
        mock_issue = _make_mock_issue()
        mock_client._gh.get_repo.return_value.get_issue.return_value = mock_issue

        result = await github_post_comment(REPO, "Issue", 42, "Hello!")
        assert "Issue" in result
        assert "#42" in result


class TestGitHubCreatePullRequestTool:
    async def test_creates_and_formats(self, mock_client: GitHubClient) -> None:
        mock_pr = _make_mock_pr()
        mock_client._gh.get_repo.return_value.create_pull.return_value = mock_pr

        result = await github_create_pull_request(REPO, "fix-widget", "Fix widget")
        assert "#7" in result
        assert "fix-widget" in result


class TestGitHubGetPullRequestTool:
    async def test_formats_output(self, mock_client: GitHubClient) -> None:
        mock_pr = _make_mock_pr()
        mock_client._gh.get_repo.return_value.get_pull.return_value = mock_pr

        result = await github_get_pull_request(REPO, 7)
        assert "#7" in result
        assert "clean" in result

    async def test_shows_merged_indicator(self, mock_client: GitHubClient) -> None:
        mock_pr = _make_mock_pr(state="closed", merged=True)
        mock_client._gh.get_repo.return_value.get_pull.return_value = mock_pr

        result = await github_get_pull_request(REPO, 7)
        assert "(merged)" in result


class TestGitHubListPullRequestsTool:
    async def test_formats_list(self, mock_client: GitHubClient) -> None:
        mock_client._gh.get_repo.return_value.get_pulls.return_value = [
            _make_mock_pr(number=1, title="First PR"),
            _make_mock_pr(number=2, title="Second PR"),
        ]
        result = await github_list_pull_requests(REPO)
        assert "2" in result
        assert "First PR" in result
        assert "Second PR" in result

    async def test_empty_list(self, mock_client: GitHubClient) -> None:
        mock_client._gh.get_repo.return_value.get_pulls.return_value = []
        result = await github_list_pull_requests(REPO)
        assert "No" in result


class TestGitHubListCommentsTool:
    async def test_formats_comments(self, mock_client: GitHubClient) -> None:
        comments = [
            _make_mock_comment(id=1, body="Looks good overall."),
            _make_mock_comment(
                id=2,
                body="Please rename the variable.",
                user=_make_mock_user("alice"),
            ),
        ]
        mock_issue = _make_mock_issue()
        mock_issue.get_comments.return_value = comments
        mock_client._gh.get_repo.return_value.get_issue.return_value = mock_issue

        result = await github_list_comments(REPO, "PullRequest", 7)
        assert "[reviewer]" in result
        assert "Looks good overall." in result
        assert "[alice]" in result
        assert "Please rename the variable." in result

    async def test_filters_bot_comments_by_default(
        self, mock_client: GitHubClient,
    ) -> None:
        bot_user = _make_mock_user("dependabot[bot]", user_type="Bot")
        comments = [
            _make_mock_comment(id=1, body="Real comment"),
            _make_mock_comment(id=2, body="Bump version", user=bot_user),
        ]
        mock_issue = _make_mock_issue()
        mock_issue.get_comments.return_value = comments
        mock_client._gh.get_repo.return_value.get_issue.return_value = mock_issue

        result = await github_list_comments(REPO, "Issue", 42)
        assert "Real comment" in result
        assert "Bump version" not in result

    async def test_includes_bot_comments_when_requested(
        self, mock_client: GitHubClient,
    ) -> None:
        bot_user = _make_mock_user("dependabot[bot]", user_type="Bot")
        comments = [
            _make_mock_comment(id=1, body="Real comment"),
            _make_mock_comment(id=2, body="Bump version", user=bot_user),
        ]
        mock_issue = _make_mock_issue()
        mock_issue.get_comments.return_value = comments
        mock_client._gh.get_repo.return_value.get_issue.return_value = mock_issue

        result = await github_list_comments(
            REPO, "Issue", 42, include_bot_comments=True,
        )
        assert "Real comment" in result
        assert "Bump version" in result

    async def test_empty_comments(self, mock_client: GitHubClient) -> None:
        mock_issue = _make_mock_issue()
        mock_issue.get_comments.return_value = []
        mock_client._gh.get_repo.return_value.get_issue.return_value = mock_issue

        result = await github_list_comments(REPO, "Issue", 42)
        assert "No comments" in result
        assert "#42" in result


class TestGitHubGetRepoInfoTool:
    async def test_formats_output(self, mock_client: GitHubClient) -> None:
        mock_repo = mock_client._gh.get_repo.return_value
        mock_repo.full_name = "octocat/hello-world"
        mock_repo.name = "hello-world"
        mock_repo.clone_url = "https://github.com/octocat/hello-world.git"
        mock_repo.ssh_url = "git@github.com:octocat/hello-world.git"
        mock_repo.default_branch = "main"
        mock_repo.html_url = "https://github.com/octocat/hello-world"
        mock_repo.description = "A test repo"

        result = await github_get_repo_info(REPO)
        assert "octocat/hello-world" in result
        assert "main" in result
        assert "A test repo" in result


class TestGitHubReadFileTool:
    async def test_formats_output(self, mock_client: GitHubClient) -> None:
        mock_content = _make_mock_content_file()
        mock_client._gh.get_repo.return_value.get_contents.return_value = mock_content

        result = await github_read_file(REPO, "README.md")
        assert "README.md" in result
        assert "# Hello World" in result


class TestGitHubMarkNotificationReadTool:
    async def test_confirms_action(self, mock_client: GitHubClient) -> None:
        with patch("thorn.tools.github.httpx") as mock_httpx:
            mock_response = MagicMock()
            mock_response.raise_for_status = MagicMock()
            mock_httpx.patch.return_value = mock_response

            result = await github_mark_notification_read("12345")
            assert "12345" in result


# ---------------------------------------------------------------------------
# GITHUB_TOOLS list
# ---------------------------------------------------------------------------


class TestGitHubToolsList:
    def test_all_tools_have_thorn_tool_marker(self) -> None:
        for fn in GITHUB_TOOLS:
            assert getattr(fn, "_thorn_tool", False), (
                f"{fn.__name__} is missing the @tool decorator"  # type: ignore[union-attr]
            )

    def test_expected_count(self) -> None:
        assert len(GITHUB_TOOLS) == 9

    def test_includes_repo_info(self) -> None:
        names = [getattr(fn, "__name__", "?") for fn in GITHUB_TOOLS]
        assert "github_get_repo_info" in names
        assert "github_read_file" in names
