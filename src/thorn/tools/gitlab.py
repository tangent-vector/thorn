"""GitLab API client used by forge services and the TODOs event source.

Provides :class:`GitLabClient`, a thin wrapper around ``python-gitlab``
that exposes purpose-built methods so that downstream code never has to
touch ``python-gitlab`` objects directly.

This module deliberately does **not** define agent-facing ``@tool``
functions of its own.  Agent-facing forge operations live in
:mod:`thorn.tools.forge` as project-name-based tools (``forge_*``) that
resolve credentials from the current agent's
:class:`~thorn.core._account.ForgeAccountConfig` and the forge service
registered in the runtime.  Forge URL and auth therefore come from
``.thorn/gateway.json`` and the agent's identity JSON, never from
process-wide environment variable singletons.

Requires ``python-gitlab`` (install via ``pip install thorn[gitlab]``).
The module gracefully defers the import error until a tool is actually
called, so importing ``thorn.tools.gitlab`` never fails.
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, Field

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional dependency guard
# ---------------------------------------------------------------------------

try:
    import gitlab as _gitlab_lib
    from gitlab.v4.objects import ProjectIssue, ProjectMergeRequest

    _HAS_GITLAB = True
except ImportError:
    _gitlab_lib = None  # type: ignore[assignment]
    _HAS_GITLAB = False


def _require_gitlab() -> None:
    """Raise a clear error if python-gitlab is not installed."""
    if not _HAS_GITLAB:
        raise ImportError(
            "python-gitlab is required for GitLab tools. "
            "Install it with: pip install thorn[gitlab]"
        )


# ---------------------------------------------------------------------------
# Configuration and client
# ---------------------------------------------------------------------------


class GitLabConfig(BaseModel):
    """Connection settings for a single GitLab instance.

    Built up by :class:`~thorn.tools.forge.GitLabForgeService` (which
    pulls ``url`` from the gateway forge entry and ``token`` from an
    agent's credentials) and consumed by :class:`GitLabClient`.
    """

    url: str = Field(description="GitLab instance URL (no trailing slash)")
    token: str = Field(description="Personal Access Token with 'api' scope")


class GitLabClient:
    """High-level GitLab client wrapping ``python-gitlab``.

    Provides purpose-built methods so that tool functions never touch
    ``python-gitlab`` objects directly.
    """

    def __init__(self, config: GitLabConfig) -> None:
        _require_gitlab()
        self._gl = _gitlab_lib.Gitlab(  # type: ignore[union-attr]
            url=config.url,
            private_token=config.token,
        )

    def check_connection(self) -> dict[str, Any]:
        """Authenticate and return info about the current user.

        Raises ``gitlab.exceptions.GitlabAuthenticationError`` on bad tokens.
        """
        self._gl.auth()
        user = self._gl.user
        assert user is not None
        return {
            "id": user.id,
            "username": user.username,
            "name": user.name,
            "web_url": user.web_url,
        }

    def get_issue(self, project_id: int, issue_iid: int) -> dict[str, Any]:
        """Fetch a single issue and return its key fields as a dict."""
        project = self._gl.projects.get(project_id)
        issue: ProjectIssue = project.issues.get(issue_iid)  # type: ignore[assignment]
        return {
            "iid": issue.iid,
            "title": issue.title,
            "state": issue.state,
            "description": issue.description or "",
            "labels": list(issue.labels),
            "assignees": [a["username"] for a in issue.assignees],
            "web_url": issue.web_url,
        }

    def create_issue(
        self,
        project_id: int,
        title: str,
        description: str = "",
        labels: list[str] | None = None,
        assignees: list[str] | None = None,
    ) -> dict[str, Any]:
        """Create an issue and return its key fields as a dict.

        *assignees* should be a list of usernames.  GitLab's create-issue
        endpoint accepts ``assignee_ids``, so we look up each username
        via the project's members (or broader user search) to resolve IDs.
        For simplicity, we pass ``labels`` as a comma-separated string
        (which GitLab's API accepts).
        """
        project = self._gl.projects.get(project_id)
        data: dict[str, Any] = {"title": title, "description": description}
        if labels:
            data["labels"] = ",".join(labels)
        if assignees:
            user_ids: list[int] = []
            for username in assignees:
                users = self._gl.users.list(username=username)
                if users:
                    user_ids.append(users[0].id)
            if user_ids:
                data["assignee_ids"] = user_ids
        issue: ProjectIssue = project.issues.create(data)  # type: ignore[assignment]
        log.info("Created issue #%d in project %d", issue.iid, project_id)
        return {
            "iid": issue.iid,
            "title": issue.title,
            "state": issue.state,
            "description": issue.description or "",
            "labels": list(issue.labels),
            "assignees": [a["username"] for a in issue.assignees],
            "web_url": issue.web_url,
        }

    def list_issues(
        self,
        project_id: int,
        state: str = "opened",
        labels: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """List issues in a project filtered by *state* and optionally *labels*."""
        project = self._gl.projects.get(project_id)
        kwargs: dict[str, Any] = {"state": state, "order_by": "created_at", "sort": "desc"}
        if labels:
            kwargs["labels"] = ",".join(labels)
        issues = project.issues.list(**kwargs, iterator=True)
        return [
            {
                "iid": issue.iid,
                "title": issue.title,
                "state": issue.state,
                "web_url": issue.web_url,
                "labels": list(issue.labels),
                "assignees": [a["username"] for a in issue.assignees],
                "author": issue.author["username"] if issue.author else None,
            }
            for issue in issues
        ]

    def update_issue(
        self,
        project_id: int,
        issue_iid: int,
        title: str | None = None,
        description: str | None = None,
        state: str | None = None,
        labels: list[str] | None = None,
        assignees: list[str] | None = None,
    ) -> dict[str, Any]:
        """Edit an issue's fields and return the updated issue as a dict.

        Only the fields that are not ``None`` are changed.  *labels* and
        *assignees* are **replacement** sets.  *state* should be a
        GitLab-native state event (``"close"`` or ``"reopen"``).
        """
        project = self._gl.projects.get(project_id)
        issue: ProjectIssue = project.issues.get(issue_iid)  # type: ignore[assignment]
        if title is not None:
            issue.title = title
        if description is not None:
            issue.description = description
        if state is not None:
            issue.state_event = state
        if labels is not None:
            issue.labels = labels
        if assignees is not None:
            user_ids: list[int] = []
            for username in assignees:
                users = self._gl.users.list(username=username)
                if users:
                    user_ids.append(users[0].id)
            issue.assignee_ids = user_ids
        issue.save()
        log.info("Updated issue #%d in project %d", issue_iid, project_id)
        return {
            "iid": issue.iid,
            "title": issue.title,
            "state": issue.state,
            "description": issue.description or "",
            "labels": list(issue.labels),
            "assignees": [a["username"] for a in issue.assignees],
            "web_url": issue.web_url,
        }

    def post_note(
        self,
        project_id: int,
        noteable_type: str,
        noteable_iid: int,
        body: str,
    ) -> None:
        """Post a note/comment on an issue or merge request."""
        project = self._gl.projects.get(project_id)
        if noteable_type == "Issue":
            noteable = project.issues.get(noteable_iid)
        elif noteable_type == "MergeRequest":
            noteable = project.mergerequests.get(noteable_iid)
        else:
            raise ValueError(f"Unsupported noteable_type: {noteable_type!r}")
        noteable.notes.create({"body": body})
        log.info(
            "Posted note on %s !%d in project %d",
            noteable_type, noteable_iid, project_id,
        )

    def create_merge_request(
        self,
        project_id: int,
        source_branch: str,
        title: str,
        target_branch: str = "main",
        description: str = "",
    ) -> dict[str, Any]:
        """Open a merge request and return its key fields as a dict."""
        project = self._gl.projects.get(project_id)
        mr: ProjectMergeRequest = project.mergerequests.create(  # type: ignore[assignment]
            {
                "source_branch": source_branch,
                "target_branch": target_branch,
                "title": title,
                "description": description,
            }
        )
        return {
            "iid": mr.iid,
            "title": mr.title,
            "state": mr.state,
            "web_url": mr.web_url,
            "source_branch": mr.source_branch,
            "target_branch": mr.target_branch,
        }

    def get_merge_request(self, project_id: int, mr_iid: int) -> dict[str, Any]:
        """Fetch a single merge request and return its key fields."""
        project = self._gl.projects.get(project_id)
        mr: ProjectMergeRequest = project.mergerequests.get(mr_iid)  # type: ignore[assignment]
        return {
            "iid": mr.iid,
            "title": mr.title,
            "state": mr.state,
            "description": mr.description or "",
            "web_url": mr.web_url,
            "source_branch": mr.source_branch,
            "target_branch": mr.target_branch,
            "merge_status": mr.merge_status,
        }

    def list_merge_requests(
        self,
        project_id: int,
        state: str = "opened",
    ) -> list[dict[str, Any]]:
        """List merge requests in a project filtered by *state*."""
        project = self._gl.projects.get(project_id)
        mrs = project.mergerequests.list(state=state, iterator=True)
        return [
            {
                "iid": mr.iid,
                "title": mr.title,
                "state": mr.state,
                "web_url": mr.web_url,
                "source_branch": mr.source_branch,
                "author": mr.author["username"] if mr.author else None,
            }
            for mr in mrs
        ]

    def get_project_info(self, project_id: int) -> dict[str, Any]:
        """Fetch project metadata: name, clone URL, default branch, etc."""
        project = self._gl.projects.get(project_id)
        return {
            "id": project.id,
            "name": project.name,
            "name_with_namespace": project.name_with_namespace,
            "path_with_namespace": project.path_with_namespace,
            "http_url_to_repo": project.http_url_to_repo,
            "ssh_url_to_repo": project.ssh_url_to_repo,
            "default_branch": project.default_branch,
            "web_url": project.web_url,
            "description": project.description or "",
        }

    def read_file(
        self,
        project_id: int,
        file_path: str,
        ref: str = "HEAD",
    ) -> dict[str, Any]:
        """Read a file from a repository via the GitLab API."""
        project = self._gl.projects.get(project_id)
        f = project.files.get(file_path=file_path, ref=ref)
        return {
            "file_path": f.file_path,
            "ref": ref,
            "content": f.decode().decode("utf-8", errors="replace"),
        }

    def list_notes(
        self,
        project_id: int,
        noteable_type: str,
        noteable_iid: int,
    ) -> list[dict[str, Any]]:
        """List notes (comments) on an issue or merge request.

        Returns all notes in chronological order.  Each dict contains
        ``id``, ``author`` (username string), ``body``, ``created_at``,
        and ``system`` (bool -- ``True`` for auto-generated notes like
        label changes).
        """
        project = self._gl.projects.get(project_id)
        if noteable_type == "Issue":
            noteable = project.issues.get(noteable_iid)
        elif noteable_type == "MergeRequest":
            noteable = project.mergerequests.get(noteable_iid)
        else:
            raise ValueError(f"Unsupported noteable_type: {noteable_type!r}")

        raw_notes = noteable.notes.list(
            sort="asc", order_by="created_at", iterator=True,
        )
        return [
            {
                "id": note.id,
                "author": note.author["username"] if note.author else "unknown",
                "body": note.body,
                "created_at": note.created_at,
                "system": getattr(note, "system", False),
            }
            for note in raw_notes
        ]

    def mark_todo_done(self, todo_id: int) -> None:
        """Mark a GitLab TODO as done by its numeric ID.

        Uses the raw HTTP API because ``python-gitlab``'s
        ``TodoManager`` does not support ``.get()`` by ID.
        """
        self._gl.http_post(f"/todos/{todo_id}/mark_as_done")


__all__ = [
    "GitLabClient",
    "GitLabConfig",
]
