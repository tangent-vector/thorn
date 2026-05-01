"""Tests for the unified forge abstraction layer (thorn.tools.forge).

Covers the ForgeClient adapters, forge host services, ProjectService,
the FORGE_TOOLS toolset, and the runtime integration.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from thorn.core._provider import MockProvider
from thorn.runtime import Runtime
from thorn.tools._github_connection import GitHubConnectionConfig, GitHubPatAuth
from thorn.tools.forge import (
    FORGE_TOOLS,
    CommentTargetKind,
    ForkConfig,
    GitHubForgeClient,
    GitHubForgeService,
    GitLabForgeClient,
    GitLabForgeService,
    GitLabForgeServiceConfig,
    ProjectService,
    ProjectServiceConfig,
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

    def test_path_native_id_forwarded_verbatim(self):
        """A path-style native id (``group/project``) is passed through
        unchanged.  ``python-gitlab`` accepts both numeric IDs and
        URL-encoded ``namespace/path`` strings; the new gateway config
        derives the latter from the fork URL so we no longer have to
        force operators to write the numeric form."""
        client, mock_gl = self._make_client()
        mock_gl.get_issue.return_value = {
            "iid": 7,
            "title": "Fix bug",
            "state": "opened",
            "web_url": "https://gl.example.com/group/project/issues/7",
            "description": "",
            "labels": [],
            "assignees": [],
        }
        client.get_issue("group/project", 7)
        mock_gl.get_issue.assert_called_once_with("group/project", 7)

    def test_subgroup_native_id_forwarded_verbatim(self):
        """Subgroup paths (``group/sub/project``) are also passed
        through verbatim; python-gitlab handles URL-encoding."""
        client, mock_gl = self._make_client()
        mock_gl.get_project_info.return_value = {
            "name_with_namespace": "Group / Sub / Project",
            "path_with_namespace": "group/sub/project",
            "http_url_to_repo": "https://gl.example.com/group/sub/project.git",
            "ssh_url_to_repo": "",
            "default_branch": "main",
            "web_url": "",
            "description": "",
        }
        client.get_project_info("group/sub/project")
        mock_gl.get_project_info.assert_called_once_with("group/sub/project")

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

    def test_create_issue(self):
        client, mock_gl = self._make_client()
        mock_gl.create_issue.return_value = {
            "iid": 10,
            "title": "New feature",
            "state": "opened",
            "web_url": "https://gl.example.com/p/issues/10",
            "description": "Please add X",
            "labels": ["enhancement"],
            "assignees": ["alice"],
        }
        result = client.create_issue(
            "42", "New feature", "Please add X", ["enhancement"], ["alice"],
        )
        mock_gl.create_issue.assert_called_once_with(
            42, title="New feature", description="Please add X",
            labels=["enhancement"], assignees=["alice"],
        )
        assert result["id"] == 10
        assert result["state"] == "open"
        assert result["url"] == "https://gl.example.com/p/issues/10"

    def test_list_issues_maps_state(self):
        client, mock_gl = self._make_client()
        mock_gl.list_issues.return_value = [
            {
                "iid": 1, "title": "Bug", "state": "opened",
                "web_url": "https://gl.example.com/p/issues/1",
                "labels": ["bug"], "assignees": ["bob"],
                "author": "alice",
            },
        ]
        result = client.list_issues("42", "open", None)
        mock_gl.list_issues.assert_called_once_with(42, "opened", None)
        assert len(result) == 1
        assert result[0]["state"] == "open"
        assert result[0]["id"] == 1

    def test_list_issues_passes_labels(self):
        client, mock_gl = self._make_client()
        mock_gl.list_issues.return_value = []
        client.list_issues("42", "all", ["bug", "urgent"])
        mock_gl.list_issues.assert_called_once_with(42, "all", ["bug", "urgent"])

    def test_update_issue_maps_state_to_event(self):
        client, mock_gl = self._make_client()
        mock_gl.update_issue.return_value = {
            "iid": 7, "title": "Fixed", "state": "closed",
            "web_url": "https://gl.example.com/p/issues/7",
            "description": "", "labels": [], "assignees": [],
        }
        result = client.update_issue(
            "42", 7, title="Fixed", description=None,
            state="closed", labels=None, assignees=None,
        )
        mock_gl.update_issue.assert_called_once_with(
            42, 7, title="Fixed", description=None,
            state="close", labels=None, assignees=None,
        )
        assert result["id"] == 7
        assert result["title"] == "Fixed"

    def test_update_issue_maps_open_to_reopen(self):
        client, mock_gl = self._make_client()
        mock_gl.update_issue.return_value = {
            "iid": 7, "title": "Bug", "state": "opened",
            "web_url": "https://gl.example.com/p/issues/7",
            "description": "", "labels": [], "assignees": [],
        }
        client.update_issue(
            "42", 7, title=None, description=None,
            state="open", labels=None, assignees=None,
        )
        mock_gl.update_issue.assert_called_once_with(
            42, 7, title=None, description=None,
            state="reopen", labels=None, assignees=None,
        )


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

    def test_mark_notification_done_delegates(self):
        client, mock_gh = self._make_client()
        client.mark_notification_done("12345")
        mock_gh.mark_notification_done.assert_called_once_with("12345")

    def test_create_issue(self):
        client, mock_gh = self._make_client()
        mock_gh.create_issue.return_value = {
            "number": 15,
            "title": "Add tests",
            "state": "open",
            "html_url": "https://github.com/org/repo/issues/15",
            "body": "We need tests",
            "labels": ["testing"],
            "assignees": ["bob"],
        }
        result = client.create_issue(
            "org/repo", "Add tests", "We need tests",
            ["testing"], ["bob"],
        )
        mock_gh.create_issue.assert_called_once_with(
            "org/repo", title="Add tests", body="We need tests",
            labels=["testing"], assignees=["bob"],
        )
        assert result["id"] == 15
        assert result["state"] == "open"
        assert result["url"] == "https://github.com/org/repo/issues/15"

    def test_list_issues(self):
        client, mock_gh = self._make_client()
        mock_gh.list_issues.return_value = [
            {
                "number": 1, "title": "Bug", "state": "open",
                "html_url": "https://github.com/org/repo/issues/1",
                "labels": ["bug"], "assignees": ["alice"],
                "author": "bob",
            },
        ]
        result = client.list_issues("org/repo", "open", None)
        mock_gh.list_issues.assert_called_once_with("org/repo", "open", None)
        assert len(result) == 1
        assert result[0]["id"] == 1

    def test_update_issue(self):
        client, mock_gh = self._make_client()
        mock_gh.update_issue.return_value = {
            "number": 7, "title": "Renamed", "state": "open",
            "html_url": "https://github.com/org/repo/issues/7",
            "body": "updated", "labels": [], "assignees": [],
        }
        result = client.update_issue(
            "org/repo", 7, title="Renamed", description="updated",
            state=None, labels=None, assignees=None,
        )
        mock_gh.update_issue.assert_called_once_with(
            "org/repo", 7, title="Renamed", body="updated",
            state=None, labels=None, assignees=None,
        )
        assert result["id"] == 7
        assert result["title"] == "Renamed"


# ---------------------------------------------------------------------------
# Forge host services
# ---------------------------------------------------------------------------


class TestGitLabForgeService:
    def test_name(self):
        config = GitLabForgeServiceConfig(url="https://gl.example.com", token="t")
        svc = GitLabForgeService(config, service_name="my-gitlab")
        assert svc.name == "my-gitlab"

    def test_clone_url_for_returns_empty(self):
        """GitLab native IDs are numeric, so clone URLs can't be derived."""
        config = GitLabForgeServiceConfig(url="https://gl.example.com", token="t")
        svc = GitLabForgeService(config, service_name="gl")
        assert svc.clone_url_for("214768") == ""


class TestForgeBrokerCredentialPlans:
    """Each forge service emits *two* :class:`BrokerCredentialPlan`
    entries per PAT credential: one for API routing (``Bearer <pat>``
    against the API host) and one for git HTTPS routing
    (``Basic <base64(x:pat)>`` against the web host with
    ``git_extra_header_host`` set).  Pinning the shape here means a
    future tweak to the forge services' routing story has to
    acknowledge the git-HTTPS path, rather than silently dropping
    it."""

    def _gl_account(self) -> Any:
        from thorn.core._credentials import Credential
        from thorn.tools.forge import GitLabAccountConfig

        return GitLabAccountConfig(
            service="gl",
            credentials=[
                Credential(kind="gitlab-pat", env_var_name="GITLAB_TOKEN"),
            ],
        )

    def _gh_account(self) -> Any:
        from thorn.core._credentials import Credential
        from thorn.tools.forge import GitHubAccountConfig

        return GitHubAccountConfig(
            service="gh",
            credentials=[Credential(kind="pat", env_var_name="GITHUB_TOKEN")],
        )

    def test_gitlab_emits_api_and_git_plans(self) -> None:
        svc = GitLabForgeService(
            GitLabForgeServiceConfig(url="https://gitlab.com", token="t"),
            service_name="gl",
        )
        plans = svc.broker_credential_plans(self._gl_account())
        assert len(plans) == 2
        api, git = plans
        assert api.path_pattern == "/api/*"
        assert api.host_pattern == "gitlab.com"
        assert api.value_transform is None
        assert api.git_extra_header_host is None

        assert git.path_pattern == "/*"
        assert git.host_pattern == "gitlab.com"
        assert git.git_extra_header_host == "gitlab.com"
        # The transform must produce ``base64("x:<raw>")`` so that a
        # ``HeaderInjection(value_format="Basic {value}")`` yields
        # the ``Authorization: Basic <...>`` git expects.
        import base64
        assert git.value_transform("hunter2") == (
            base64.b64encode(b"x:hunter2").decode()
        )

    def test_github_splits_api_and_git_hosts(self) -> None:
        svc = GitHubForgeService(
            GitHubConnectionConfig(
                base_url="https://api.github.com",
                auth=GitHubPatAuth(token="t"),
            ),
            service_name="gh",
        )
        plans = svc.broker_credential_plans(self._gh_account())
        assert len(plans) == 2
        api, git = plans
        # GitHub.com splits the two: API on api.github.com, git on github.com.
        assert api.host_pattern == "api.github.com"
        assert git.host_pattern == "github.com"
        assert git.git_extra_header_host == "github.com"
        import base64
        assert git.value_transform("hunter2") == (
            base64.b64encode(b"x:hunter2").decode()
        )

    def test_github_enterprise_collapses_api_and_git_hosts(self) -> None:
        svc = GitHubForgeService(
            GitHubConnectionConfig(
                base_url="https://github.example.com/api/v3",
                auth=GitHubPatAuth(token="t"),
            ),
            service_name="ghe",
        )
        plans = svc.broker_credential_plans(self._gh_account())
        assert len(plans) == 2
        api, git = plans
        assert api.host_pattern == "github.example.com"
        assert git.host_pattern == "github.example.com"
        assert git.git_extra_header_host == "github.example.com"


class TestGitHubForgeServiceCloneUrl:
    def test_derives_clone_url_from_api_url(self):
        cfg = GitHubConnectionConfig(
            base_url="https://api.github.com",
            auth=GitHubPatAuth(token="tok"),
        )
        svc = GitHubForgeService(cfg, service_name="gh")
        assert svc.clone_url_for("owner/repo") == "https://github.com/owner/repo.git"

    def test_github_enterprise_url(self):
        cfg = GitHubConnectionConfig(
            base_url="https://api.github.example.com",
            auth=GitHubPatAuth(token="tok"),
        )
        svc = GitHubForgeService(cfg, service_name="ghe")
        assert svc.clone_url_for("org/proj") == "https://github.example.com/org/proj.git"

    def test_non_api_prefixed_url(self):
        """When base_url doesn't have an 'api.' prefix, use the host as-is."""
        cfg = GitHubConnectionConfig(
            base_url="https://github.example.com/api/v3",
            auth=GitHubPatAuth(token="tok"),
        )
        svc = GitHubForgeService(cfg, service_name="gh")
        assert svc.clone_url_for("o/r") == "https://github.example.com/o/r.git"


# ---------------------------------------------------------------------------
# ProjectService
# ---------------------------------------------------------------------------


class TestProjectService:
    def test_single_fork_properties(self):
        """A single-fork project exposes the primary fork via top-level accessors."""
        config = ProjectServiceConfig(
            forks=[
                ForkConfig(
                    forge="gl",
                    native_id="lace/lace",
                    name="origin",
                    clone_url="https://gl.example.com/lace/lace.git",
                ),
            ],
            default_branch="main",
        )
        svc = ProjectService(config, service_name="lace")
        assert svc.name == "lace"
        assert svc.forge_name == "gl"
        assert svc.native_id == "lace/lace"
        assert svc.clone_url == "https://gl.example.com/lace/lace.git"
        # ``default_branch`` is the *configured* override; resolution
        # of the live value is via ``resolve_default_branch``.
        assert svc.default_branch == "main"

        forks = svc.forks
        assert len(forks) == 1
        assert forks[0].forge == "gl"
        assert forks[0].native_id == "lace/lace"
        assert forks[0].name == "origin"

    def test_fork_based_properties(self):
        """New fork-based config provides correct primary fork accessors."""
        config = ProjectServiceConfig(
            forks=[
                ForkConfig(
                    forge="gh", native_id="owner/upstream",
                    name="upstream", clone_url="https://github.com/owner/upstream.git",
                ),
                ForkConfig(
                    forge="gh", native_id="bot/fork",
                    name="origin", clone_url="https://github.com/bot/fork.git",
                ),
            ],
            default_branch="main",
        )
        svc = ProjectService(config, service_name="my-proj")
        assert svc.forge_name == "gh"
        assert svc.native_id == "owner/upstream"
        assert svc.clone_url == "https://github.com/owner/upstream.git"
        assert len(svc.forks) == 2

    def test_get_fork_by_name(self):
        config = ProjectServiceConfig(
            forks=[
                ForkConfig(forge="gh", native_id="a/repo", name="upstream"),
                ForkConfig(forge="gh", native_id="b/repo", name="origin"),
            ],
        )
        svc = ProjectService(config, service_name="p")
        assert svc.get_fork("origin").native_id == "b/repo"
        assert svc.get_fork("upstream").native_id == "a/repo"
        assert svc.get_fork().native_id == "a/repo"

    def test_get_fork_missing_raises(self):
        config = ProjectServiceConfig(
            forks=[ForkConfig(forge="gh", native_id="a/repo", name="upstream")],
        )
        svc = ProjectService(config, service_name="p")
        with pytest.raises(KeyError, match="no-such-fork"):
            svc.get_fork("no-such-fork")

    def test_get_fork_no_forks_raises(self):
        config = ProjectServiceConfig()
        svc = ProjectService(config, service_name="p")
        with pytest.raises(KeyError, match="no forks"):
            svc.get_fork()

    def test_resolve_default_branch_uses_per_fork_override(self):
        """A per-fork override wins over project- and forge-level values."""
        mock_client = MagicMock()
        proj_svc = ProjectService(
            ProjectServiceConfig(
                forks=[ForkConfig(
                    forge="gl", native_id="org/repo",
                    default_branch="develop",
                )],
                default_branch="main",
            ),
            service_name="proj",
        )
        # Per-fork override returned without ever calling the forge.
        assert proj_svc.resolve_default_branch(mock_client) == "develop"
        mock_client.get_project_info.assert_not_called()

    def test_resolve_default_branch_uses_project_override(self):
        mock_client = MagicMock()
        proj_svc = ProjectService(
            ProjectServiceConfig(
                forks=[ForkConfig(forge="gl", native_id="org/repo")],
                default_branch="trunk",
            ),
            service_name="proj",
        )
        assert proj_svc.resolve_default_branch(mock_client) == "trunk"
        mock_client.get_project_info.assert_not_called()

    def test_resolve_default_branch_falls_back_to_forge_lookup(self):
        """When no override is set, resolve via the forge API.  The
        result should be cached for subsequent calls."""
        mock_client = MagicMock()
        mock_client.get_project_info.return_value = {
            "default_branch": "develop",
        }
        proj_svc = ProjectService(
            ProjectServiceConfig(
                forks=[ForkConfig(forge="gl", native_id="org/repo")],
            ),
            service_name="proj",
        )

        assert proj_svc.resolve_default_branch(mock_client) == "develop"
        mock_client.get_project_info.assert_called_once_with("org/repo")

        # Second call hits the cache, not the forge.
        mock_client.get_project_info.reset_mock()
        assert proj_svc.resolve_default_branch(mock_client) == "develop"
        mock_client.get_project_info.assert_not_called()

    def test_resolve_default_branch_defaults_to_main_when_forge_returns_empty(self):
        """If the forge can't tell us, fall back to ``"main"`` rather
        than returning an empty string that callers would treat as
        "not configured"."""
        mock_client = MagicMock()
        mock_client.get_project_info.return_value = {"default_branch": ""}
        proj_svc = ProjectService(
            ProjectServiceConfig(
                forks=[ForkConfig(forge="gl", native_id="org/repo")],
            ),
            service_name="proj",
        )
        assert proj_svc.resolve_default_branch(mock_client) == "main"


# ---------------------------------------------------------------------------
# FORGE_TOOLS
# ---------------------------------------------------------------------------


class TestFORGE_TOOLS:
    def test_has_twelve_tools(self):
        assert len(FORGE_TOOLS) == 12

    def test_tool_names(self):
        names = {getattr(t, "__name__", str(t)) for t in FORGE_TOOLS}
        expected = {
            "forge_read_issue",
            "forge_create_issue",
            "forge_list_issues",
            "forge_update_issue",
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
    """Test that forge tools resolve services from the runtime context.

    All forge tools authenticate via the agent's account on the
    forge, so each test wires up a runtime *and* an agent with a
    matching :class:`GitLabAccountConfig` on the ``"gl"`` service.
    The forge service's ``authenticated_client`` is patched to
    return a mock client so we can assert on the call shape without
    actually constructing a real GitLab connection.
    """

    def _setup_runtime(
        self, tmp_path: Path,
    ) -> tuple[Runtime, MagicMock, "Agent"]:
        from thorn.core._account import AgentAccountsConfig
        from thorn.core._agent import Agent
        from thorn.core._credentials import Credential
        from thorn.tools.forge import GitLabAccountConfig

        mock_forge_client = MagicMock()

        forge_svc = GitLabForgeService(
            GitLabForgeServiceConfig(url="https://gl.example.com"),
            service_name="gl",
        )
        forge_svc.authenticated_client = MagicMock(  # type: ignore[method-assign]
            return_value=mock_forge_client,
        )

        proj_svc = ProjectService(
            ProjectServiceConfig(forks=[
                ForkConfig(forge="gl", native_id="42"),
            ]),
            service_name="test-proj",
        )

        runtime = Runtime(
            provider=MockProvider(), workspace_root=tmp_path,
        )
        runtime.register_service(forge_svc)
        runtime.register_service(proj_svc)

        agent = Agent(
            metadata={"project": "test-proj"},
            accounts=AgentAccountsConfig(accounts=[
                GitLabAccountConfig(
                    service="gl",
                    credentials=[Credential(
                        kind="gitlab-pat",
                        env_var_name="THORN_TEST_GL_TOKEN",
                    )],
                ),
            ]),
        )
        return runtime, mock_forge_client, agent

    @staticmethod
    def _bind_agent(runtime: Runtime, agent: "Agent") -> None:
        runtime.context.agent = agent

    @pytest.mark.asyncio
    async def test_forge_read_issue(self, tmp_path: Path):
        from thorn.tools.forge import forge_read_issue

        runtime, mock_client, agent = self._setup_runtime(tmp_path)
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
            self._bind_agent(runtime, agent)
            result = await forge_read_issue("test-proj", 7)

        assert "Bug" in result
        assert "open" in result
        mock_client.get_issue.assert_called_once_with("42", 7)

    @pytest.mark.asyncio
    async def test_forge_post_comment(self, tmp_path: Path):
        from thorn.tools.forge import forge_post_comment

        runtime, mock_client, agent = self._setup_runtime(tmp_path)

        async with runtime:
            self._bind_agent(runtime, agent)
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

        runtime, mock_client, agent = self._setup_runtime(tmp_path)
        mock_client.create_change_request.return_value = {
            "id": 1,
            "title": "My CR",
            "state": "open",
            "url": "https://gl.example.com/mr/1",
            "source_branch": "feat",
            "target_branch": "main",
        }

        async with runtime:
            self._bind_agent(runtime, agent)
            result = await forge_create_change_request(
                "test-proj", "feat", "My CR", "desc", "main",
            )

        assert "My CR" in result
        assert "feat" in result

    @pytest.mark.asyncio
    async def test_forge_mark_notification_done(self, tmp_path: Path):
        from thorn.tools.forge import forge_mark_notification_done

        runtime, mock_client, agent = self._setup_runtime(tmp_path)

        async with runtime:
            self._bind_agent(runtime, agent)
            result = await forge_mark_notification_done("test-proj", "99")

        assert "99" in result
        mock_client.mark_notification_done.assert_called_once_with("99")

    @pytest.mark.asyncio
    async def test_forge_create_issue(self, tmp_path: Path):
        from thorn.tools.forge import forge_create_issue

        runtime, mock_client, agent = self._setup_runtime(tmp_path)
        mock_client.create_issue.return_value = {
            "id": 10,
            "title": "New feature",
            "state": "open",
            "url": "https://gl.example.com/issues/10",
            "description": "Please add X",
            "labels": ["enhancement"],
            "assignees": ["alice"],
        }

        async with runtime:
            self._bind_agent(runtime, agent)
            result = await forge_create_issue(
                "test-proj", "New feature", "Please add X",
                ["enhancement"], ["alice"],
            )

        assert "New feature" in result
        assert "#10" in result
        mock_client.create_issue.assert_called_once_with(
            "42", title="New feature", description="Please add X",
            labels=["enhancement"], assignees=["alice"],
        )

    @pytest.mark.asyncio
    async def test_forge_list_issues(self, tmp_path: Path):
        from thorn.tools.forge import forge_list_issues

        runtime, mock_client, agent = self._setup_runtime(tmp_path)
        mock_client.list_issues.return_value = [
            {
                "id": 1, "title": "Bug", "state": "open",
                "url": "https://gl.example.com/issues/1",
                "labels": ["bug"], "assignees": ["bob"],
                "author": "alice",
            },
            {
                "id": 2, "title": "Feature", "state": "open",
                "url": "https://gl.example.com/issues/2",
                "labels": [], "assignees": [],
                "author": "carol",
            },
        ]

        async with runtime:
            self._bind_agent(runtime, agent)
            result = await forge_list_issues("test-proj", "open")

        assert "2 open issue(s)" in result
        assert "Bug" in result
        assert "Feature" in result
        mock_client.list_issues.assert_called_once_with("42", "open", None)

    @pytest.mark.asyncio
    async def test_forge_list_issues_empty(self, tmp_path: Path):
        from thorn.tools.forge import forge_list_issues

        runtime, mock_client, agent = self._setup_runtime(tmp_path)
        mock_client.list_issues.return_value = []

        async with runtime:
            self._bind_agent(runtime, agent)
            result = await forge_list_issues("test-proj", "closed")

        assert "No closed issues" in result

    @pytest.mark.asyncio
    async def test_forge_update_issue_simple(self, tmp_path: Path):
        from thorn.tools.forge import forge_update_issue

        runtime, mock_client, agent = self._setup_runtime(tmp_path)
        mock_client.update_issue.return_value = {
            "id": 7, "title": "Renamed", "state": "open",
            "url": "https://gl.example.com/issues/7",
            "description": "", "labels": [], "assignees": [],
        }

        async with runtime:
            self._bind_agent(runtime, agent)
            result = await forge_update_issue(
                "test-proj", 7, title="Renamed",
            )

        assert "Renamed" in result
        assert "#7" in result
        mock_client.update_issue.assert_called_once_with(
            "42", 7, title="Renamed", description=None,
            state=None, labels=None, assignees=None,
        )

    @pytest.mark.asyncio
    async def test_forge_update_issue_add_remove_labels(self, tmp_path: Path):
        from thorn.tools.forge import forge_update_issue

        runtime, mock_client, agent = self._setup_runtime(tmp_path)
        mock_client.get_issue.return_value = {
            "id": 7, "title": "Bug", "state": "open",
            "url": "https://gl.example.com/issues/7",
            "description": "", "labels": ["bug", "triage"],
            "assignees": ["alice"],
        }
        mock_client.update_issue.return_value = {
            "id": 7, "title": "Bug", "state": "open",
            "url": "https://gl.example.com/issues/7",
            "description": "", "labels": ["bug", "confirmed"],
            "assignees": ["alice"],
        }

        async with runtime:
            self._bind_agent(runtime, agent)
            result = await forge_update_issue(
                "test-proj", 7,
                add_labels=["confirmed"],
                remove_labels=["triage"],
            )

        assert "confirmed" in result
        call_kwargs = mock_client.update_issue.call_args
        labels_arg = call_kwargs[1]["labels"] if call_kwargs[1] else call_kwargs[0][5]
        assert "bug" in labels_arg
        assert "confirmed" in labels_arg
        assert "triage" not in labels_arg

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

    # ------------------------------------------------------------------
    # External-content envelope wrapping (peer-identity-and-trust)
    # ------------------------------------------------------------------
    #
    # The threat model only holds if user-authored text returned from
    # forge tools is labelled with the same envelope shape the gateway
    # formatter uses on incoming notifications.  These tests pin down
    # that contract: every tool that returns user-authored body text
    # wraps it in ``[external-content ...]`` and stamps the right
    # peer status.

    @pytest.mark.asyncio
    async def test_forge_read_issue_wraps_description_with_envelope(
        self, tmp_path: Path,
    ) -> None:
        """The issue description (user-authored) is wrapped; metadata is not."""
        from thorn.gateway._peer import PeerAccount, PeerKind, PeerRegistry, PeerSpec
        from thorn.tools.forge import forge_read_issue

        runtime, mock_client, agent = self._setup_runtime(tmp_path)
        mock_client.get_issue.return_value = {
            "id": 7,
            "title": "Bug",
            "state": "open",
            "url": "https://gl.example.com/issues/7",
            "description": "Bad things happen.",
            "labels": ["bug"],
            "assignees": ["alice"],
            "author": {"id": 12345, "username": "alice-handle", "name": "Alice"},
            "created_at": "2026-04-30T12:34:56Z",
        }
        runtime.peer_registry = PeerRegistry([
            PeerSpec(
                id="alice",
                name="Alice",
                kind=PeerKind.HUMAN,
                accounts=[PeerAccount(service="gl", account_id="12345")],
            ),
        ])

        async with runtime:
            self._bind_agent(runtime, agent)
            result = await forge_read_issue("test-proj", 7)

        # Title/state/labels stay as plain prose.
        assert "Issue #7: Bug" in result
        assert "State: open" in result
        # Body got wrapped with the matching peer status and a
        # blockquote rendition.
        assert "[external-content" in result
        assert "kind=issue_body" in result
        assert "peer=yes" in result
        assert "> Bad things happen." in result

    @pytest.mark.asyncio
    async def test_forge_read_issue_wraps_with_unknown_peer_status_when_no_registry(
        self, tmp_path: Path,
    ) -> None:
        """An empty peer registry yields ``peer=unknown`` rather than ``peer=no``.

        Reporting NON_PEER on an empty registry would be misleading --
        "no peers configured" is not the same as "this user is not a
        peer."
        """
        from thorn.tools.forge import forge_read_issue

        runtime, mock_client, agent = self._setup_runtime(tmp_path)
        mock_client.get_issue.return_value = {
            "id": 1, "title": "T", "state": "open",
            "url": "https://x", "description": "Body",
            "labels": [], "assignees": [],
            "author": {"id": 99, "username": "stranger"},
            "created_at": "2026-04-30T00:00:00Z",
        }

        async with runtime:
            self._bind_agent(runtime, agent)
            result = await forge_read_issue("test-proj", 1)

        assert "peer=unknown" in result

    @pytest.mark.asyncio
    async def test_forge_read_issue_labels_non_peer_when_registry_has_others(
        self, tmp_path: Path,
    ) -> None:
        """A registry with peers, but no match for *this* author, yields ``peer=no``."""
        from thorn.gateway._peer import PeerAccount, PeerRegistry, PeerSpec
        from thorn.tools.forge import forge_read_issue

        runtime, mock_client, agent = self._setup_runtime(tmp_path)
        mock_client.get_issue.return_value = {
            "id": 1, "title": "T", "state": "open",
            "url": "https://x", "description": "Body",
            "labels": [], "assignees": [],
            "author": {"id": 99, "username": "stranger"},
            "created_at": "2026-04-30T00:00:00Z",
        }
        runtime.peer_registry = PeerRegistry([
            PeerSpec(
                id="someone-else",
                accounts=[PeerAccount(service="gl", account_id="11111")],
            ),
        ])

        async with runtime:
            self._bind_agent(runtime, agent)
            result = await forge_read_issue("test-proj", 1)

        assert "peer=no" in result

    @pytest.mark.asyncio
    async def test_forge_get_change_request_wraps_description(
        self, tmp_path: Path,
    ) -> None:
        from thorn.tools.forge import forge_get_change_request

        runtime, mock_client, agent = self._setup_runtime(tmp_path)
        mock_client.get_change_request.return_value = {
            "id": 1, "title": "MR", "state": "open",
            "url": "https://x", "description": "PR body text.",
            "source_branch": "feat", "target_branch": "main",
            "author": {"id": 1, "username": "alice"},
            "created_at": "2026-04-30T00:00:00Z",
        }

        async with runtime:
            self._bind_agent(runtime, agent)
            result = await forge_get_change_request("test-proj", 1)

        assert "[external-content" in result
        assert "kind=pr_body" in result
        assert "> PR body text." in result

    @pytest.mark.asyncio
    async def test_forge_list_comments_wraps_each_comment(
        self, tmp_path: Path,
    ) -> None:
        """Each comment is its own envelope; per-author peer status is independent."""
        from thorn.gateway._peer import PeerAccount, PeerRegistry, PeerSpec
        from thorn.tools.forge import forge_list_comments

        runtime, mock_client, agent = self._setup_runtime(tmp_path)
        mock_client.list_comments.return_value = [
            {
                "author": "alice",
                "author_user": {
                    "id": 12345,
                    "username": "alice",
                    "name": "Alice",
                },
                "created_at": "2026-04-30T00:00:00Z",
                "body": "First comment.",
                "is_system": False,
            },
            {
                "author": "stranger",
                "author_user": {
                    "id": 99999,
                    "username": "stranger",
                    "name": "Stranger",
                },
                "created_at": "2026-04-30T00:01:00Z",
                "body": "Suspicious comment.",
                "is_system": False,
            },
        ]
        runtime.peer_registry = PeerRegistry([
            PeerSpec(
                id="alice",
                name="Alice",
                accounts=[PeerAccount(service="gl", account_id="12345")],
            ),
        ])

        async with runtime:
            self._bind_agent(runtime, agent)
            result = await forge_list_comments(
                "test-proj", "Issue", 1,
            )

        assert result.count("[external-content") == 2
        # Distinct peer statuses for the two authors.
        assert "peer=yes" in result
        assert "peer=no" in result
        assert "> First comment." in result
        assert "> Suspicious comment." in result

    @pytest.mark.asyncio
    async def test_forge_list_comments_no_results_message_unchanged(
        self, tmp_path: Path,
    ) -> None:
        from thorn.tools.forge import forge_list_comments

        runtime, mock_client, agent = self._setup_runtime(tmp_path)
        mock_client.list_comments.return_value = []

        async with runtime:
            self._bind_agent(runtime, agent)
            result = await forge_list_comments(
                "test-proj", "Issue", 42,
            )

        assert "No comments" in result
        assert "[external-content" not in result


class TestResolveWithAccounts:
    """``_resolve`` (the internal helper underneath every forge tool)
    is the single point that ties together project lookup, account
    resolution, and authenticated client construction.  The tests
    here pin down the contract: an agent with a matching account on
    the forge gets an authenticated client; an agent without one
    surfaces a clear error rather than silently falling through.
    """

    @pytest.mark.asyncio
    async def test_resolve_uses_account_credentials(self, tmp_path: Path):
        """An agent with an account on the forge causes ``_resolve``
        to call ``forge_svc.authenticated_client(account)`` and pass
        the resulting client to the tool body."""
        from thorn.core._account import AgentAccountsConfig
        from thorn.core._agent import Agent
        from thorn.core._credentials import Credential
        from thorn.tools.forge import GitLabAccountConfig, forge_read_issue

        mock_account_client = MagicMock()
        mock_account_client.get_issue.return_value = {
            "id": 5,
            "title": "Account Bug",
            "state": "open",
            "url": "https://gl.example.com/issues/5",
            "description": "via account",
            "labels": [],
            "assignees": [],
        }

        forge_svc = GitLabForgeService(
            GitLabForgeServiceConfig(url="https://gl.example.com"),
            service_name="gl",
        )
        forge_svc.authenticated_client = MagicMock(  # type: ignore[method-assign]
            return_value=mock_account_client,
        )

        proj_svc = ProjectService(
            ProjectServiceConfig(forks=[
                ForkConfig(forge="gl", native_id="42"),
            ]),
            service_name="test-proj",
        )

        runtime = Runtime(provider=MockProvider(), workspace_root=tmp_path)
        runtime.register_service(forge_svc)
        runtime.register_service(proj_svc)

        agent = Agent(
            metadata={"project": "test-proj"},
            accounts=AgentAccountsConfig(accounts=[
                GitLabAccountConfig(
                    service="gl",
                    credentials=[Credential(
                        kind="gitlab-pat",
                        env_var_name="THORN_TEST_GL_TOKEN",
                    )],
                ),
            ]),
        )

        async with runtime:
            runtime.context.agent = agent
            result = await forge_read_issue("test-proj", 5)

        assert "Account Bug" in result
        mock_account_client.get_issue.assert_called_once_with("42", 5)
        # The forge service should have been asked for an
        # authenticated client using the agent's account; we don't
        # care about the exact account instance, just that it was
        # called once.
        forge_svc.authenticated_client.assert_called_once()

    @pytest.mark.asyncio
    async def test_resolve_raises_when_agent_has_no_account(
        self, tmp_path: Path,
    ):
        """An agent with no matching account on the forge surfaces a
        clear ``KeyError`` rather than silently falling back to
        unauthenticated access (which is what the pre-account-driven
        code did, and which is exactly what the new design rejects)."""
        from thorn.core._agent import Agent
        from thorn.tools.forge import forge_read_issue

        forge_svc = GitLabForgeService(
            GitLabForgeServiceConfig(url="https://gl.example.com"),
            service_name="gl",
        )
        proj_svc = ProjectService(
            ProjectServiceConfig(forks=[
                ForkConfig(forge="gl", native_id="42"),
            ]),
            service_name="test-proj",
        )

        runtime = Runtime(provider=MockProvider(), workspace_root=tmp_path)
        runtime.register_service(forge_svc)
        runtime.register_service(proj_svc)

        agent = Agent(metadata={"project": "test-proj"})

        async with runtime:
            runtime.context.agent = agent
            with pytest.raises(KeyError, match="no account"):
                await forge_read_issue("test-proj", 5)
