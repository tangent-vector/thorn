"""Tests for thorn.tools.gitlab -- the GitLabClient wrapper.

The agent-facing ``@tool`` functions have moved to
:mod:`thorn.tools.forge`; this module now only owns the low-level
``GitLabClient`` plus its ``GitLabConfig``, so the tests focus on the
client's translation between python-gitlab objects and plain dicts.
Mocks stand in for ``python-gitlab`` so the tests run without a real
GitLab.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from thorn.tools.gitlab import GitLabClient, GitLabConfig


# ---------------------------------------------------------------------------
# Mock factories & fixtures
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


class TestGitLabClientListNotes:
    def _make_mock_note(self, **overrides: Any) -> MagicMock:
        note = MagicMock()
        note.id = overrides.get("id", 100)
        note.author = overrides.get("author", {"username": "reviewer"})
        note.body = overrides.get("body", "Please fix the naming.")
        note.created_at = overrides.get("created_at", "2026-04-08T10:00:00Z")
        note.system = overrides.get("system", False)
        return note

    def test_returns_list_of_dicts(self, mock_client: GitLabClient) -> None:
        notes = [
            self._make_mock_note(id=1, body="First comment"),
            self._make_mock_note(id=2, body="Second comment"),
        ]
        mock_project = MagicMock()
        mock_project.mergerequests.get.return_value.notes.list.return_value = notes
        mock_client._gl.projects.get.return_value = mock_project

        result = mock_client.list_notes(1, "MergeRequest", 7)
        assert len(result) == 2
        assert result[0]["body"] == "First comment"
        assert result[1]["body"] == "Second comment"

    def test_issue_notes(self, mock_client: GitLabClient) -> None:
        notes = [self._make_mock_note(id=10, body="Issue comment")]
        mock_project = MagicMock()
        mock_project.issues.get.return_value.notes.list.return_value = notes
        mock_client._gl.projects.get.return_value = mock_project

        result = mock_client.list_notes(1, "Issue", 42)
        assert len(result) == 1
        assert result[0]["author"] == "reviewer"

    def test_includes_system_flag(self, mock_client: GitLabClient) -> None:
        notes = [self._make_mock_note(system=True, body="added label ~bug")]
        mock_project = MagicMock()
        mock_project.issues.get.return_value.notes.list.return_value = notes
        mock_client._gl.projects.get.return_value = mock_project

        result = mock_client.list_notes(1, "Issue", 42)
        assert result[0]["system"] is True

    def test_invalid_type_raises(self, mock_client: GitLabClient) -> None:
        with pytest.raises(ValueError, match="Unsupported noteable_type"):
            mock_client.list_notes(1, "Snippet", 1)
