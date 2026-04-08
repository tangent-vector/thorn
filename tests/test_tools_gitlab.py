"""Tests for thorn.tools.gitlab -- GitLab API tools.

Uses mock objects instead of a real GitLab instance.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from thorn.tools.gitlab import (
    GITLAB_TOOLS,
    GitLabClient,
    GitLabConfig,
    create_merge_request,
    get_client,
    get_merge_request,
    gitlab_mark_todo_done,
    list_merge_requests,
    post_comment,
    read_issue,
    set_client,
)


# ---------------------------------------------------------------------------
# GitLabConfig
# ---------------------------------------------------------------------------


class TestGitLabConfig:
    def test_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GITLAB_URL", "https://gitlab.example.com")
        monkeypatch.setenv("GITLAB_TOKEN", "glpat-secret")
        config = GitLabConfig.from_env()
        assert config.url == "https://gitlab.example.com"
        assert config.token == "glpat-secret"

    def test_from_env_missing_url(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("GITLAB_URL", raising=False)
        monkeypatch.delenv("GITLAB_TOKEN", raising=False)
        with pytest.raises(ValueError, match="GITLAB_URL"):
            GitLabConfig.from_env()

    def test_from_env_missing_token(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GITLAB_URL", "https://gitlab.example.com")
        monkeypatch.delenv("GITLAB_TOKEN", raising=False)
        with pytest.raises(ValueError, match="GITLAB_TOKEN"):
            GitLabConfig.from_env()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_mock_issue(**overrides: Any) -> MagicMock:
    issue = MagicMock()
    issue.iid = overrides.get("iid", 42)
    issue.title = overrides.get("title", "Fix the widget")
    issue.state = overrides.get("state", "opened")
    issue.description = overrides.get("description", "It is broken.")
    issue.labels = overrides.get("labels", ["bug", "priority::high"])
    issue.assignees = overrides.get("assignees", [{"username": "alice"}])
    issue.web_url = overrides.get(
        "web_url", "https://gitlab.example.com/proj/-/issues/42"
    )
    return issue


def _make_mock_mr(**overrides: Any) -> MagicMock:
    mr = MagicMock()
    mr.iid = overrides.get("iid", 7)
    mr.title = overrides.get("title", "Fix widget")
    mr.state = overrides.get("state", "opened")
    mr.description = overrides.get("description", "Fixes #42")
    mr.web_url = overrides.get(
        "web_url", "https://gitlab.example.com/proj/-/merge_requests/7"
    )
    mr.source_branch = overrides.get("source_branch", "fix-widget")
    mr.target_branch = overrides.get("target_branch", "main")
    mr.merge_status = overrides.get("merge_status", "can_be_merged")
    mr.author = overrides.get("author", {"username": "bob"})
    return mr


@pytest.fixture()
def mock_client() -> GitLabClient:
    """Build a GitLabClient with a mocked python-gitlab backend.

    Patches both ``_gitlab_lib`` and ``_HAS_GITLAB`` so that tests
    work regardless of whether ``python-gitlab`` is installed.
    """
    with (
        patch("thorn.tools.gitlab._gitlab_lib") as mock_gl_mod,
        patch("thorn.tools.gitlab._HAS_GITLAB", True),
    ):
        mock_gl_instance = MagicMock()
        mock_gl_mod.Gitlab.return_value = mock_gl_instance

        config = GitLabConfig(url="https://gitlab.example.com", token="test-token")
        client = GitLabClient(config)
    return client


@pytest.fixture(autouse=True)
def _install_mock_client(mock_client: GitLabClient) -> None:
    """Install the mock client as the module-level client for all tests."""
    set_client(mock_client)
    yield
    set_client(None)


# ---------------------------------------------------------------------------
# GitLabClient methods
# ---------------------------------------------------------------------------


class TestGitLabClientGetIssue:
    def test_returns_dict(self, mock_client: GitLabClient) -> None:
        mock_issue = _make_mock_issue()
        mock_client._gl.projects.get.return_value.issues.get.return_value = mock_issue

        result = mock_client.get_issue(1, 42)
        assert result["iid"] == 42
        assert result["title"] == "Fix the widget"
        assert result["labels"] == ["bug", "priority::high"]
        assert result["assignees"] == ["alice"]


class TestGitLabClientPostNote:
    def test_post_issue_note(self, mock_client: GitLabClient) -> None:
        mock_project = MagicMock()
        mock_client._gl.projects.get.return_value = mock_project
        mock_client.post_note(1, "Issue", 42, "Fixed!")
        mock_project.issues.get.assert_called_once_with(42)
        mock_project.issues.get.return_value.notes.create.assert_called_once_with(
            {"body": "Fixed!"}
        )

    def test_post_mr_note(self, mock_client: GitLabClient) -> None:
        mock_project = MagicMock()
        mock_client._gl.projects.get.return_value = mock_project
        mock_client.post_note(1, "MergeRequest", 7, "LGTM")
        mock_project.mergerequests.get.assert_called_once_with(7)

    def test_invalid_type_raises(self, mock_client: GitLabClient) -> None:
        with pytest.raises(ValueError, match="Unsupported noteable_type"):
            mock_client.post_note(1, "Snippet", 1, "oops")


class TestGitLabClientCreateMR:
    def test_returns_dict(self, mock_client: GitLabClient) -> None:
        mock_mr = _make_mock_mr()
        mock_client._gl.projects.get.return_value.mergerequests.create.return_value = (
            mock_mr
        )
        result = mock_client.create_merge_request(1, "fix-widget", "Fix widget")
        assert result["iid"] == 7
        assert result["source_branch"] == "fix-widget"


class TestGitLabClientGetMR:
    def test_returns_dict(self, mock_client: GitLabClient) -> None:
        mock_mr = _make_mock_mr()
        mock_client._gl.projects.get.return_value.mergerequests.get.return_value = (
            mock_mr
        )
        result = mock_client.get_merge_request(1, 7)
        assert result["merge_status"] == "can_be_merged"


class TestGitLabClientListMRs:
    def test_returns_list_of_dicts(self, mock_client: GitLabClient) -> None:
        mock_client._gl.projects.get.return_value.mergerequests.list.return_value = [
            _make_mock_mr(iid=1, title="MR one"),
            _make_mock_mr(iid=2, title="MR two"),
        ]
        result = mock_client.list_merge_requests(1)
        assert len(result) == 2
        assert result[0]["title"] == "MR one"


# ---------------------------------------------------------------------------
# @tool functions
# ---------------------------------------------------------------------------


class TestReadIssueTool:
    async def test_formats_output(self, mock_client: GitLabClient) -> None:
        mock_issue = _make_mock_issue()
        mock_client._gl.projects.get.return_value.issues.get.return_value = mock_issue

        result = await read_issue(1, 42)
        assert "Fix the widget" in result
        assert "#42" in result
        assert "bug" in result
        assert "alice" in result


class TestPostCommentTool:
    async def test_posts_and_confirms(self, mock_client: GitLabClient) -> None:
        mock_project = MagicMock()
        mock_client._gl.projects.get.return_value = mock_project

        result = await post_comment(1, "Issue", 42, "Hello!")
        assert "Issue" in result
        assert "#42" in result


class TestCreateMergeRequestTool:
    async def test_creates_and_formats(self, mock_client: GitLabClient) -> None:
        mock_mr = _make_mock_mr()
        mock_client._gl.projects.get.return_value.mergerequests.create.return_value = (
            mock_mr
        )
        result = await create_merge_request(1, "fix-widget", "Fix widget")
        assert "!7" in result
        assert "fix-widget" in result


class TestGetMergeRequestTool:
    async def test_formats_output(self, mock_client: GitLabClient) -> None:
        mock_mr = _make_mock_mr()
        mock_client._gl.projects.get.return_value.mergerequests.get.return_value = (
            mock_mr
        )
        result = await get_merge_request(1, 7)
        assert "!7" in result
        assert "can_be_merged" in result


class TestListMergeRequestsTool:
    async def test_formats_list(self, mock_client: GitLabClient) -> None:
        mock_client._gl.projects.get.return_value.mergerequests.list.return_value = [
            _make_mock_mr(iid=1, title="First MR"),
            _make_mock_mr(iid=2, title="Second MR"),
        ]
        result = await list_merge_requests(1)
        assert "2" in result
        assert "First MR" in result
        assert "Second MR" in result

    async def test_empty_list(self, mock_client: GitLabClient) -> None:
        mock_client._gl.projects.get.return_value.mergerequests.list.return_value = []
        result = await list_merge_requests(1)
        assert "No" in result


# ---------------------------------------------------------------------------
# GITLAB_TOOLS list
# ---------------------------------------------------------------------------


class TestGitLabToolsList:
    def test_all_tools_have_thorn_tool_marker(self) -> None:
        for fn in GITLAB_TOOLS:
            assert getattr(fn, "_thorn_tool", False), (
                f"{fn.__name__} is missing the @tool decorator"  # type: ignore[union-attr]
            )

    def test_expected_count(self) -> None:
        assert len(GITLAB_TOOLS) == 8

    def test_includes_project_info(self) -> None:
        names = [getattr(fn, "__name__", "?") for fn in GITLAB_TOOLS]
        assert "gitlab_get_project_info" in names
        assert "gitlab_read_file" in names
