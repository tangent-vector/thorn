"""Tests for the low-level GitLab API client wrapper."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import pytest

from thorn.tools.gitlab import (
    GitLabClient,
    GitLabConfig,
    GitLabProjectLookupError,
)


class _GitLabNotFound(Exception):
    response_code = 404


def _make_client() -> tuple[GitLabClient, MagicMock]:
    mock_gl_instance = MagicMock()
    with (
        patch("thorn.tools.gitlab._gitlab_lib") as mock_gl_mod,
        patch("thorn.tools.gitlab._HAS_GITLAB", True),
    ):
        mock_gl_mod.Gitlab.return_value = mock_gl_instance
        client = GitLabClient(
            GitLabConfig(url="https://gitlab.example.com", token="test"),
        )
    return client, mock_gl_instance


def _project_with_issue(project_id: int, issue_iid: int) -> SimpleNamespace:
    issue = SimpleNamespace(
        iid=issue_iid,
        title="Fix it",
        state="opened",
        description="",
        labels=[],
        assignees=[],
        web_url=f"https://gitlab.example.com/tfoley/thorn/-/issues/{issue_iid}",
        author=None,
    )
    project = SimpleNamespace(id=project_id)
    project.issues = MagicMock()
    project.issues.get.return_value = issue
    return project


def _project_info(project_id: int) -> SimpleNamespace:
    return SimpleNamespace(
        id=project_id,
        name="thorn",
        name_with_namespace="Theresa Foley / thorn",
        path_with_namespace="tfoley/thorn",
        http_url_to_repo="https://gitlab.example.com/tfoley/thorn.git",
        ssh_url_to_repo="git@gitlab.example.com:tfoley/thorn.git",
        default_branch="main",
        web_url="https://gitlab.example.com/tfoley/thorn",
        description="",
    )


class TestGitLabProjectLookup:
    def test_path_lookup_404_falls_back_to_exact_search_result(self) -> None:
        client, mock_gl = _make_client()
        project = _project_with_issue(project_id=264873, issue_iid=7)
        mock_gl.projects.get.side_effect = [
            _GitLabNotFound("not found"),
            project,
            project,
        ]
        mock_gl.projects.list.return_value = [
            SimpleNamespace(id=264873, path_with_namespace="tfoley/thorn"),
        ]

        result = client.get_issue("tfoley/thorn", 7)

        assert result["iid"] == 7
        assert mock_gl.projects.get.mock_calls[:2] == [
            call("tfoley/thorn"),
            call(264873),
        ]
        mock_gl.projects.list.assert_called_once_with(
            search="thorn",
            simple=True,
            iterator=True,
        )

        client.get_issue("tfoley/thorn", 8)
        assert mock_gl.projects.get.mock_calls[2] == call(264873)
        mock_gl.projects.list.assert_called_once()

    def test_numeric_string_uses_numeric_id_without_search(self) -> None:
        client, mock_gl = _make_client()
        mock_gl.projects.get.return_value = _project_info(264873)

        result = client.get_project_info("264873")

        assert result["id"] == 264873
        mock_gl.projects.get.assert_called_once_with(264873)
        mock_gl.projects.list.assert_not_called()

    def test_unresolved_path_suggests_numeric_native_id(self) -> None:
        client, mock_gl = _make_client()
        mock_gl.projects.get.side_effect = _GitLabNotFound("not found")
        mock_gl.projects.list.return_value = []

        with pytest.raises(GitLabProjectLookupError) as exc_info:
            client.get_project_info("tfoley/missing")

        message = str(exc_info.value)
        assert "native_id" in message
        assert "numeric GitLab project ID" in message
