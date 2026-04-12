"""Tests for the unified forge abstraction layer (thorn.tools.forge).

Covers the ForgeClient adapters, ForgeService/ProjectService service
types, the FORGE_TOOLS toolset, and the runtime integration.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from thorn.core._provider import MockProvider
from thorn.runtime import Runtime
from thorn.tools.forge import (
    FORGE_TOOLS,
    CommentTargetKind,
    ForgeService,
    ForgeServiceConfig,
    GitHubForgeClient,
    GitLabForgeClient,
    ProjectService,
    ProjectServiceConfig,
    get_forge_for_project,
)


# ---------------------------------------------------------------------------
# GitLabForgeClient
# ---------------------------------------------------------------------------


class TestGitLabForgeClient:
    def _make_client(self) -> tuple[GitLabForgeClient, MagicMock]:
        mock_gl = MagicMock()
        return GitLabForgeClient(mock_gl), mock_gl

    def test_get_issue_normalizes_state(self):
        client, mock_gl = self._make_client()
        mock_gl.get_issue.return_value = {
            "iid": 7,
            "title": "Fix bug",
            "state": "opened",
            "web_url": "https://gl.example.com/p/issues/7",
            "description": "Broken",
            "labels": ["bug"],
            "assignees": ["alice"],
        }
        result = client.get_issue("42", 7)
        mock_gl.get_issue.assert_called_once_with(42, 7)
        assert result["id"] == 7
        assert result["state"] == "open"
        assert result["url"] == "https://gl.example.com/p/issues/7"

    def test_post_comment_maps_change_request_target(self):
        client, mock_gl = self._make_client()
        client.post_comment("42", "ChangeRequest", 3, "looks good")
        mock_gl.post_note.assert_called_once_with(42, "MergeRequest", 3, "looks good")

    def test_post_comment_passes_issue_through(self):
        client, mock_gl = self._make_client()
        client.post_comment("42", "Issue", 7, "noted")
        mock_gl.post_note.assert_called_once_with(42, "Issue", 7, "noted")

    def test_create_change_request(self):
        client, mock_gl = self._make_client()
        mock_gl.create_merge_request.return_value = {
            "iid": 1,
            "title": "My MR",
            "state": "opened",
            "web_url": "https://gl.example.com/p/merge_requests/1",
            "source_branch": "feat",
            "target_branch": "main",
        }
        result = client.create_change_request(
            "42", "feat", "My MR", "main", "desc",
        )
        assert result["id"] == 1
        assert result["state"] == "open"
        assert result["source_branch"] == "feat"

    def test_list_change_requests_maps_state(self):
        client, mock_gl = self._make_client()
        mock_gl.list_merge_requests.return_value = [
            {
                "iid": 1,
                "title": "MR 1",
                "state": "opened",
                "web_url": "https://gl.example.com/p/merge_requests/1",
                "source_branch": "feat",
                "author": "alice",
            },
        ]
        result = client.list_change_requests("42", "open")
        mock_gl.list_merge_requests.assert_called_once_with(42, "opened")
        assert len(result) == 1
        assert result[0]["state"] == "open"

    def test_mark_notification_done_converts_to_int(self):
        client, mock_gl = self._make_client()
        client.mark_notification_done("99")
        mock_gl.mark_todo_done.assert_called_once_with(99)

    def test_get_change_request(self):
        client, mock_gl = self._make_client()
        mock_gl.get_merge_request.return_value = {
            "iid": 5,
            "title": "CR 5",
            "state": "merged",
            "web_url": "https://gl.example.com/p/merge_requests/5",
            "description": "stuff",
            "source_branch": "feat",
            "target_branch": "main",
            "merge_status": "can_be_merged",
        }
        result = client.get_change_request("42", 5)
        assert result["id"] == 5
        assert result["state"] == "merged"

    def test_list_comments(self):
        client, mock_gl = self._make_client()
        mock_gl.list_notes.return_value = [
            {
                "id": 1,
                "author": "bob",
                "body": "Nice work",
                "created_at": "2025-01-01T00:00:00Z",
                "system": False,
            },
        ]
        result = client.list_comments("42", "ChangeRequest", 5)
        mock_gl.list_notes.assert_called_once_with(42, "MergeRequest", 5)
        assert len(result) == 1
        assert result[0]["author"] == "bob"
        assert result[0]["is_system"] is False

    def test_get_project_info(self):
        client, mock_gl = self._make_client()
        mock_gl.get_project_info.return_value = {
            "name_with_namespace": "Org / Repo",
            "path_with_namespace": "org/repo",
            "http_url_to_repo": "https://gl.example.com/org/repo.git",
            "ssh_url_to_repo": "git@gl.example.com:org/repo.git",
            "default_branch": "main",
            "web_url": "https://gl.example.com/org/repo",
            "description": "A project",
        }
        result = client.get_project_info("42")
        assert result["name"] == "Org / Repo"
        assert result["clone_url"] == "https://gl.example.com/org/repo.git"

    def test_read_file(self):
        client, mock_gl = self._make_client()
        mock_gl.read_file.return_value = {
            "file_path": "README.md",
            "ref": "main",
            "content": "# Hello",
        }
        result = client.read_file("42", "README.md", "main")
        mock_gl.read_file.assert_called_once_with(42, "README.md", "main")
        assert result["content"] == "# Hello"


# ---------------------------------------------------------------------------
# GitHubForgeClient
# ---------------------------------------------------------------------------


class TestGitHubForgeClient:
    def _make_client(self) -> tuple[GitHubForgeClient, MagicMock]:
        mock_gh = MagicMock()
        return GitHubForgeClient(mock_gh), mock_gh

    def test_get_issue(self):
        client, mock_gh = self._make_client()
        mock_gh.get_issue.return_value = {
            "number": 7,
            "title": "Fix bug",
            "state": "open",
            "html_url": "https://github.com/org/repo/issues/7",
            "body": "Broken",
            "labels": ["bug"],
            "assignees": ["alice"],
        }
        result = client.get_issue("org/repo", 7)
        assert result["id"] == 7
        assert result["state"] == "open"

    def test_get_change_request_merged(self):
        client, mock_gh = self._make_client()
        mock_gh.get_pull_request.return_value = {
            "number": 3,
            "title": "My PR",
            "state": "closed",
            "html_url": "https://github.com/org/repo/pull/3",
            "body": "desc",
            "head": "feat",
            "base": "main",
            "mergeable": None,
            "mergeable_state": "unknown",
            "merged": True,
        }
        result = client.get_change_request("org/repo", 3)
        assert result["state"] == "merged"

    def test_get_change_request_closed_not_merged(self):
        client, mock_gh = self._make_client()
        mock_gh.get_pull_request.return_value = {
            "number": 4,
            "title": "Closed PR",
            "state": "closed",
            "html_url": "https://github.com/org/repo/pull/4",
            "body": "",
            "head": "feat",
            "base": "main",
            "mergeable": None,
            "mergeable_state": "unknown",
            "merged": False,
        }
        result = client.get_change_request("org/repo", 4)
        assert result["state"] == "closed"

    def test_list_change_requests_merged_filters(self):
        client, mock_gh = self._make_client()
        mock_gh.list_pull_requests.return_value = [
            {"number": 1, "title": "Merged", "state": "closed",
             "html_url": "u1", "head": "a", "author": "a", "merged": True},
            {"number": 2, "title": "Closed", "state": "closed",
             "html_url": "u2", "head": "b", "author": "b", "merged": False},
        ]
        result = client.list_change_requests("org/repo", "merged")
        mock_gh.list_pull_requests.assert_called_once_with("org/repo", "closed")
        assert len(result) == 1
        assert result[0]["id"] == 1
        assert result[0]["state"] == "merged"

    def test_list_change_requests_all_normalizes_merged(self):
        client, mock_gh = self._make_client()
        mock_gh.list_pull_requests.return_value = [
            {"number": 1, "title": "Open PR", "state": "open",
             "html_url": "u1", "head": "a", "author": "a", "merged": False},
            {"number": 2, "title": "Merged PR", "state": "closed",
             "html_url": "u2", "head": "b", "author": "b", "merged": True},
            {"number": 3, "title": "Closed PR", "state": "closed",
             "html_url": "u3", "head": "c", "author": "c", "merged": False},
        ]
        result = client.list_change_requests("org/repo", "all")
        mock_gh.list_pull_requests.assert_called_once_with("org/repo", "all")
        assert len(result) == 3
        assert result[0]["state"] == "open"
        assert result[1]["state"] == "merged"
        assert result[2]["state"] == "closed"

    def test_list_change_requests_closed_normalizes_merged(self):
        client, mock_gh = self._make_client()
        mock_gh.list_pull_requests.return_value = [
            {"number": 2, "title": "Merged PR", "state": "closed",
             "html_url": "u2", "head": "b", "author": "b", "merged": True},
            {"number": 3, "title": "Closed PR", "state": "closed",
             "html_url": "u3", "head": "c", "author": "c", "merged": False},
        ]
        result = client.list_change_requests("org/repo", "closed")
        mock_gh.list_pull_requests.assert_called_once_with("org/repo", "closed")
        assert len(result) == 2
        assert result[0]["state"] == "merged"
        assert result[1]["state"] == "closed"

    def test_post_comment(self):
        client, mock_gh = self._make_client()
        client.post_comment("org/repo", "ChangeRequest", 3, "looks good")
        mock_gh.post_comment.assert_called_once_with("org/repo", 3, "looks good")

    def test_mark_notification_done(self):
        client, mock_gh = self._make_client()
        client.mark_notification_done("12345")
        mock_gh.mark_notification_read.assert_called_once_with("12345")


# ---------------------------------------------------------------------------
# ForgeService
# ---------------------------------------------------------------------------


class TestForgeService:
    def test_name_and_type(self):
        config = ForgeServiceConfig(url="https://gl.example.com", token="t")
        svc = ForgeService(
            config, service_name="my-gitlab", forge_type="gitlab",
        )
        assert svc.name == "my-gitlab"
        assert svc.forge_type == "gitlab"

    def test_unknown_forge_type_raises(self):
        config = ForgeServiceConfig(url="https://example.com", token="t")
        svc = ForgeService(
            config, service_name="bad", forge_type="mercurial",
        )
        with pytest.raises(ValueError, match="mercurial"):
            _ = svc.client


# ---------------------------------------------------------------------------
# ProjectService
# ---------------------------------------------------------------------------


class TestProjectService:
    def test_properties(self):
        config = ProjectServiceConfig(
            forge="gl",
            native_id="214768",
            path="lace/lace",
            clone_url="https://gl.example.com/lace/lace.git",
            default_branch="main",
        )
        svc = ProjectService(config, service_name="lace")
        assert svc.name == "lace"
        assert svc.forge_name == "gl"
        assert svc.native_id == "214768"
        assert svc.path == "lace/lace"
        assert svc.clone_url == "https://gl.example.com/lace/lace.git"
        assert svc.default_branch == "main"

    def test_get_forge_client_resolves_via_runtime(self, tmp_path: Path):
        mock_forge_client = MagicMock()
        forge_config = ForgeServiceConfig(
            url="https://gl.example.com", token="t",
        )
        forge_svc = ForgeService(
            forge_config, service_name="gl", forge_type="gitlab",
        )
        forge_svc._client = mock_forge_client

        proj_config = ProjectServiceConfig(
            forge="gl", native_id="42", path="org/repo",
        )
        proj_svc = ProjectService(proj_config, service_name="my-proj")

        runtime = Runtime(
            provider=MockProvider(), workspace_root=tmp_path,
        )
        runtime.register_service(forge_svc)
        runtime.register_service(proj_svc)

        client, native_id = proj_svc.get_forge_client(runtime)
        assert client is mock_forge_client
        assert native_id == "42"


# ---------------------------------------------------------------------------
# get_forge_for_project
# ---------------------------------------------------------------------------


class TestGetForgeForProject:
    def test_resolves_project_and_forge(self, tmp_path: Path):
        mock_forge_client = MagicMock()
        forge_config = ForgeServiceConfig(
            url="https://gl.example.com", token="t",
        )
        forge_svc = ForgeService(
            forge_config, service_name="gl", forge_type="gitlab",
        )
        forge_svc._client = mock_forge_client

        proj_config = ProjectServiceConfig(
            forge="gl", native_id="99",
        )
        proj_svc = ProjectService(proj_config, service_name="test-proj")

        runtime = Runtime(
            provider=MockProvider(), workspace_root=tmp_path,
        )
        runtime.register_service(forge_svc)
        runtime.register_service(proj_svc)

        client, native_id = get_forge_for_project(runtime, "test-proj")
        assert client is mock_forge_client
        assert native_id == "99"

    def test_missing_project_raises_key_error(self, tmp_path: Path):
        runtime = Runtime(
            provider=MockProvider(), workspace_root=tmp_path,
        )
        with pytest.raises(KeyError, match="no-proj"):
            get_forge_for_project(runtime, "no-proj")

    def test_non_project_service_raises_type_error(self, tmp_path: Path):
        from thorn.core._service import Service

        class NotAProject(Service):
            Config = type("C", (), {})  # type: ignore[assignment]

            def __init__(self) -> None:
                pass

            @property
            def name(self) -> str:
                return "not-a-project"

        runtime = Runtime(
            provider=MockProvider(), workspace_root=tmp_path,
        )
        runtime.register_service(NotAProject())
        with pytest.raises(TypeError, match="NotAProject"):
            get_forge_for_project(runtime, "not-a-project")


# ---------------------------------------------------------------------------
# Runtime.get_forge_for_project
# ---------------------------------------------------------------------------


class TestRuntimeGetForgeForProject:
    def test_delegates_to_module_function(self, tmp_path: Path):
        mock_client = MagicMock()
        forge_svc = ForgeService(
            ForgeServiceConfig(url="https://example.com", token="t"),
            service_name="gh",
            forge_type="github",
        )
        forge_svc._client = mock_client

        proj_svc = ProjectService(
            ProjectServiceConfig(forge="gh", native_id="org/repo"),
            service_name="my-proj",
        )

        runtime = Runtime(
            provider=MockProvider(), workspace_root=tmp_path,
        )
        runtime.register_service(forge_svc)
        runtime.register_service(proj_svc)

        client, nid = runtime.get_forge_for_project("my-proj")
        assert client is mock_client
        assert nid == "org/repo"


# ---------------------------------------------------------------------------
# FORGE_TOOLS
# ---------------------------------------------------------------------------


class TestFORGE_TOOLS:
    def test_has_nine_tools(self):
        assert len(FORGE_TOOLS) == 9

    def test_tool_names(self):
        names = {getattr(t, "__name__", str(t)) for t in FORGE_TOOLS}
        expected = {
            "forge_read_issue",
            "forge_post_comment",
            "forge_create_change_request",
            "forge_get_change_request",
            "forge_list_change_requests",
            "forge_list_comments",
            "forge_get_project_info",
            "forge_read_file",
            "forge_mark_notification_done",
        }
        assert names == expected


# ---------------------------------------------------------------------------
# FORGE_TOOLS integration (with mock runtime)
# ---------------------------------------------------------------------------


class TestForgeToolsIntegration:
    """Test that forge tools resolve services from the runtime context."""

    def _setup_runtime(
        self, tmp_path: Path,
    ) -> tuple[Runtime, MagicMock]:
        mock_forge_client = MagicMock()

        forge_svc = ForgeService(
            ForgeServiceConfig(url="https://gl.example.com", token="t"),
            service_name="gl",
            forge_type="gitlab",
        )
        forge_svc._client = mock_forge_client

        proj_svc = ProjectService(
            ProjectServiceConfig(
                forge="gl", native_id="42", path="org/repo",
            ),
            service_name="test-proj",
        )

        runtime = Runtime(
            provider=MockProvider(), workspace_root=tmp_path,
        )
        runtime.register_service(forge_svc)
        runtime.register_service(proj_svc)
        return runtime, mock_forge_client

    @pytest.mark.asyncio
    async def test_forge_read_issue(self, tmp_path: Path):
        from thorn.tools.forge import forge_read_issue

        runtime, mock_client = self._setup_runtime(tmp_path)
        mock_client.get_issue.return_value = {
            "id": 7,
            "title": "Bug",
            "state": "open",
            "url": "https://gl.example.com/issues/7",
            "description": "broken",
            "labels": ["bug"],
            "assignees": ["alice"],
        }

        async with runtime:
            result = await forge_read_issue("test-proj", 7)

        assert "Bug" in result
        assert "open" in result
        mock_client.get_issue.assert_called_once_with("42", 7)

    @pytest.mark.asyncio
    async def test_forge_post_comment(self, tmp_path: Path):
        from thorn.tools.forge import forge_post_comment

        runtime, mock_client = self._setup_runtime(tmp_path)

        async with runtime:
            result = await forge_post_comment(
                "test-proj", "Issue", 7, "looks good",
            )

        assert "Posted comment" in result
        mock_client.post_comment.assert_called_once_with(
            "42", "Issue", 7, "looks good",
        )

    @pytest.mark.asyncio
    async def test_forge_create_change_request(self, tmp_path: Path):
        from thorn.tools.forge import forge_create_change_request

        runtime, mock_client = self._setup_runtime(tmp_path)
        mock_client.create_change_request.return_value = {
            "id": 1,
            "title": "My CR",
            "state": "open",
            "url": "https://gl.example.com/mr/1",
            "source_branch": "feat",
            "target_branch": "main",
        }

        async with runtime:
            result = await forge_create_change_request(
                "test-proj", "feat", "My CR", "desc", "main",
            )

        assert "My CR" in result
        assert "feat" in result

    @pytest.mark.asyncio
    async def test_forge_mark_notification_done(self, tmp_path: Path):
        from thorn.tools.forge import forge_mark_notification_done

        runtime, mock_client = self._setup_runtime(tmp_path)

        async with runtime:
            result = await forge_mark_notification_done("test-proj", "99")

        assert "99" in result
        mock_client.mark_notification_done.assert_called_once_with("99")

    @pytest.mark.asyncio
    async def test_tool_without_runtime_raises(self, tmp_path: Path):
        from thorn.core._context import ExecutionContext, set_context, reset_context

        ctx = ExecutionContext(provider=MockProvider())
        token = set_context(ctx)
        try:
            from thorn.tools.forge import forge_read_issue

            with pytest.raises(RuntimeError, match="No Runtime"):
                await forge_read_issue("test-proj", 7)
        finally:
            reset_context(token)
