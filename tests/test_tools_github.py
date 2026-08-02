"""Tests for thorn.tools.github -- the GitHubClient wrapper.

The agent-facing ``@tool`` functions have moved to
:mod:`thorn.tools.forge`; this module now only owns the low-level
``GitHubClient`` plus its connection-config models, so the tests focus
on the client's translation between PyGithub objects and plain dicts.
Mocks stand in for ``PyGithub`` so the tests run without a real GitHub.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from thorn.tools._credential_scopes import (
    BroadCredentialScopeWarning,
    MissingCredentialScopeWarning,
)
from thorn.tools._github_connection import GitHubConnectionConfig, GitHubPatAuth
from thorn.tools.github import GitHubClient, GitHubNotificationThreadID

# ---------------------------------------------------------------------------
# Mock object factories
# ---------------------------------------------------------------------------


def _make_mock_label(name: str) -> MagicMock:
    label = MagicMock()
    label.name = name
    return label


def _make_mock_user(
    login: str = "alice",
    user_type: str = "User",
    *,
    user_id: int = 12345,
    name: str = "Alice",
) -> MagicMock:
    user = MagicMock()
    user.id = user_id
    user.login = login
    user.name = name
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


class _FakeGHAuth:
    """Small stand-in for the PyGithub ``Auth`` module used by tests."""

    class Token:
        def __init__(self, token: str) -> None:
            self.token = str(token)

    class AppInstallationAuth:
        def __init__(self, token: str) -> None:
            self.token = token

    class AppAuth:
        def __init__(self, app_id: int, private_key: str) -> None:
            self.app_id = app_id
            self.private_key = private_key

        def get_installation_auth(
            self, installation_id: int,
        ) -> "_FakeGHAuth.AppInstallationAuth":
            return _FakeGHAuth.AppInstallationAuth(
                f"installation-token-{installation_id}",
            )


@pytest.fixture()
def mock_client() -> GitHubClient:
    """Build a GitHubClient with a mocked PyGithub backend.

    Patches ``_Github``, ``_GHAuth``, and ``_HAS_GITHUB`` so that tests
    work regardless of whether ``PyGithub`` is installed.
    """
    with (
        patch("thorn.tools.github._Github") as mock_gh_cls,
        patch("thorn.tools.github._GHAuth", _FakeGHAuth),
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


# ---------------------------------------------------------------------------
# GitHubClient methods
# ---------------------------------------------------------------------------


class TestGitHubNotificationThreadID:
    def test_accepts_ascii_digits(self) -> None:
        thread_id = GitHubNotificationThreadID.parse("12345")
        assert thread_id.value == "12345"

    def test_rejects_path_like_value(self) -> None:
        with pytest.raises(ValueError, match="GitHub notification thread ID"):
            GitHubNotificationThreadID("../../repos/octocat/hello-world")

    def test_rejects_unicode_decimal_digits(self) -> None:
        with pytest.raises(ValueError, match="GitHub notification thread ID"):
            GitHubNotificationThreadID.parse("\u0661\u0662\u0663")


REPO = "octocat/hello-world"


class TestGitHubClientGetUserByLogin:
    def test_returns_immutable_id_and_display_metadata(
        self, mock_client: GitHubClient,
    ) -> None:
        mock_client._gh.get_user.return_value = _make_mock_user(
            "ada",
            user_id=1001,
            name="Ada Lovelace",
        )

        result = mock_client.get_user_by_login("ada")

        mock_client._gh.get_user.assert_called_once_with("ada")
        assert result == {
            "id": 1001,
            "login": "ada",
            "name": "Ada Lovelace",
        }


class TestGitHubClientInspectCredentialScopes:
    def test_warns_for_broad_classic_repo_and_high_risk_scopes(
        self,
        mock_client: GitHubClient,
    ) -> None:
        response = MagicMock()
        response.headers = {
            "X-OAuth-Scopes": "repo, delete_repo",
        }

        with patch("thorn.tools.github.httpx.get", return_value=response):
            inspection = mock_client.inspect_credential_scopes()

        assert inspection.observed_scopes == ("delete_repo", "repo")
        broad_warnings = [
            warning for warning in inspection.warnings
            if isinstance(warning, BroadCredentialScopeWarning)
        ]
        assert len(broad_warnings) == 2
        assert any("classic 'repo'" in w.summary for w in broad_warnings)
        assert any("'delete_repo'" in w.summary for w in broad_warnings)

    def test_warns_when_classic_scopes_do_not_cover_repo_or_notifications(
        self,
        mock_client: GitHubClient,
    ) -> None:
        response = MagicMock()
        response.headers = {
            "X-OAuth-Scopes": "repo:status",
        }

        with patch("thorn.tools.github.httpx.get", return_value=response):
            inspection = mock_client.inspect_credential_scopes()

        missing_warnings = [
            warning for warning in inspection.warnings
            if isinstance(warning, MissingCredentialScopeWarning)
        ]
        assert len(missing_warnings) == 2
        assert any("repository access" in w.summary for w in missing_warnings)
        assert any("notification access" in w.summary for w in missing_warnings)


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

            thread_id = GitHubNotificationThreadID.parse("12345")
            mock_client.mark_notification_read(thread_id)

            mock_httpx.patch.assert_called_once()
            call_args = mock_httpx.patch.call_args
            assert (
                call_args[0][0]
                == "https://api.github.com/notifications/threads/12345"
            )

    def test_rejects_path_like_id_before_api(
        self,
        mock_client: GitHubClient,
    ) -> None:
        with patch("thorn.tools.github.httpx") as mock_httpx:
            with pytest.raises(ValueError, match="GitHub notification thread ID"):
                mock_client.mark_notification_read("../../repos/octocat/hello-world")

            mock_httpx.patch.assert_not_called()


class TestGitHubClientMarkNotificationDone:
    def test_calls_api(self, mock_client: GitHubClient) -> None:
        with patch("thorn.tools.github.httpx") as mock_httpx:
            mock_response = MagicMock()
            mock_response.raise_for_status = MagicMock()
            mock_httpx.delete.return_value = mock_response

            mock_client.mark_notification_done("12345")

            mock_httpx.delete.assert_called_once()
            call_args = mock_httpx.delete.call_args
            assert (
                call_args[0][0]
                == "https://api.github.com/notifications/threads/12345"
            )

    def test_rejects_path_like_id_before_api(
        self,
        mock_client: GitHubClient,
    ) -> None:
        with patch("thorn.tools.github.httpx") as mock_httpx:
            with pytest.raises(ValueError, match="GitHub notification thread ID"):
                mock_client.mark_notification_done(
                    "../../repos/octocat/hello-world"
                )

            mock_httpx.delete.assert_not_called()
