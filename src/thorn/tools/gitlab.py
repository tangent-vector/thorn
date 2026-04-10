"""GitLab API operations as ``@tool``-decorated functions for thorn agents.

Provides tools for interacting with a GitLab instance: reading issues,
posting comments, creating merge requests, and querying MR details.

Requires ``python-gitlab`` (install via ``pip install thorn[gitlab]``).
The module gracefully defers the import error until a tool is actually
called, so importing ``thorn.tools.gitlab`` never fails.

Usage::

    from thorn.tools import gitlab

    agent = Agent(..., tools=[gitlab.GITLAB_TOOLS, ...])

Configuration is loaded from environment variables:

- ``GITLAB_URL``   -- GitLab instance URL (e.g. ``https://gitlab.example.com``)
- ``GITLAB_TOKEN`` -- Personal Access Token with ``api`` scope

Adapted from ``thorn-bot/src/thorn_bot/_gitlab.py``.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Literal

from pydantic import BaseModel, Field

from thorn.core._func import tool

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
    """Configuration for connecting to a GitLab instance.

    Typically loaded from environment variables via
    ``GitLabConfig.from_env()``.
    """

    url: str = Field(description="GitLab instance URL (no trailing slash)")
    token: str = Field(description="Personal Access Token with 'api' scope")

    @classmethod
    def from_env(cls) -> GitLabConfig:
        """Load configuration from ``GITLAB_URL`` and ``GITLAB_TOKEN``
        environment variables.

        Raises ``ValueError`` if either variable is not set.
        """
        url = os.environ.get("GITLAB_URL")
        token = os.environ.get("GITLAB_TOKEN")
        missing = [
            name
            for name, val in [("GITLAB_URL", url), ("GITLAB_TOKEN", token)]
            if not val
        ]
        if missing:
            raise ValueError(
                f"Missing required environment variable(s): {', '.join(missing)}. "
                "Set GITLAB_URL and GITLAB_TOKEN to use GitLab tools."
            )
        return cls(url=url, token=token)  # type: ignore[arg-type]


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


# ---------------------------------------------------------------------------
# Module-level client accessor
# ---------------------------------------------------------------------------

_client: GitLabClient | None = None


def get_client() -> GitLabClient:
    """Return the module-level ``GitLabClient``, creating it lazily.

    Configuration is loaded from environment variables on first access.
    Subsequent calls return the same client instance.
    """
    global _client
    if _client is None:
        config = GitLabConfig.from_env()
        _client = GitLabClient(config)
    return _client


def set_client(client: GitLabClient | None) -> None:
    """Replace the module-level client (useful for testing or custom configs)."""
    global _client
    _client = client


# ---------------------------------------------------------------------------
# @tool functions
# ---------------------------------------------------------------------------

NoteableKind = Literal["Issue", "MergeRequest"]


@tool
async def read_issue(project_id: int, issue_iid: int) -> str:
    """Read a GitLab issue, returning its title, description, labels, and assignees.

    *project_id* is the numeric project ID.  *issue_iid* is the
    issue number within the project (the ``#N`` in the UI).
    """
    client = get_client()
    info = await asyncio.to_thread(client.get_issue, project_id, issue_iid)
    lines = [
        f"Issue #{info['iid']}: {info['title']}",
        f"State: {info['state']}",
        f"Labels: {', '.join(info['labels']) or '(none)'}",
        f"Assignees: {', '.join(info['assignees']) or '(none)'}",
        f"URL: {info['web_url']}",
        "",
        info["description"] or "(no description)",
    ]
    return "\n".join(lines)


@tool
async def post_comment(
    project_id: int,
    noteable_type: NoteableKind,
    noteable_iid: int,
    body: str,
) -> str:
    """Post a comment on a GitLab issue or merge request.

    *noteable_type* must be either ``"Issue"`` or ``"MergeRequest"``.
    *noteable_iid* is the issue/MR number within the project.
    """
    client = get_client()
    await asyncio.to_thread(
        client.post_note, project_id, noteable_type, noteable_iid, body,
    )
    return f"Posted comment on {noteable_type} #{noteable_iid} in project {project_id}."


@tool
async def create_merge_request(
    project_id: int,
    source_branch: str,
    title: str,
    description: str = "",
    target_branch: str = "main",
) -> str:
    """Create a new merge request on GitLab.

    Opens an MR from *source_branch* into *target_branch* (default
    ``main``) in the specified project.
    """
    client = get_client()
    info = await asyncio.to_thread(
        client.create_merge_request,
        project_id=project_id,
        source_branch=source_branch,
        title=title,
        target_branch=target_branch,
        description=description,
    )
    return (
        f"Created MR !{info['iid']}: {info['title']}\n"
        f"  {info['source_branch']} -> {info['target_branch']}\n"
        f"  URL: {info['web_url']}"
    )


@tool
async def get_merge_request(project_id: int, mr_iid: int) -> str:
    """Read details of a GitLab merge request.

    Returns the MR title, state, branches, merge status, and description.
    """
    client = get_client()
    info = await asyncio.to_thread(
        client.get_merge_request, project_id, mr_iid,
    )
    lines = [
        f"MR !{info['iid']}: {info['title']}",
        f"State: {info['state']}",
        f"Branches: {info['source_branch']} -> {info['target_branch']}",
        f"Merge status: {info['merge_status']}",
        f"URL: {info['web_url']}",
        "",
        info["description"] or "(no description)",
    ]
    return "\n".join(lines)


@tool
async def list_merge_requests(
    project_id: int,
    state: Literal["opened", "closed", "merged", "all"] = "opened",
) -> str:
    """List merge requests in a GitLab project.

    Filters by *state* (default ``"opened"``).  Returns a formatted
    list of MR numbers, titles, and authors.
    """
    client = get_client()
    mrs = await asyncio.to_thread(
        client.list_merge_requests, project_id, state,
    )
    if not mrs:
        return f"No {state} merge requests in project {project_id}."
    lines = []
    for mr in mrs:
        author = mr["author"] or "unknown"
        lines.append(f"  !{mr['iid']}: {mr['title']} ({mr['state']}, by {author})")
    header = f"{len(mrs)} {state} merge request(s) in project {project_id}:"
    return "\n".join([header, *lines])


@tool
async def gitlab_get_project_info(project_id: int) -> str:
    """Get information about a GitLab project.

    Returns the project's name, clone URL (HTTPS), default branch,
    namespace path, and web URL.
    """
    client = get_client()
    info = await asyncio.to_thread(client.get_project_info, project_id)
    lines = [
        f"Project: {info['name_with_namespace']}",
        f"Path: {info['path_with_namespace']}",
        f"Clone URL (HTTPS): {info['http_url_to_repo']}",
        f"Clone URL (SSH): {info['ssh_url_to_repo']}",
        f"Default branch: {info['default_branch']}",
        f"Web URL: {info['web_url']}",
    ]
    if info["description"]:
        lines.append(f"Description: {info['description']}")
    return "\n".join(lines)


@tool
async def gitlab_read_file(
    project_id: int,
    file_path: str,
    ref: str = "HEAD",
) -> str:
    """Read a file from a GitLab repository via the API.

    Useful for inspecting files without cloning the entire repository.
    *ref* can be a branch name, tag, or commit SHA.
    """
    client = get_client()
    info = await asyncio.to_thread(client.read_file, project_id, file_path, ref)
    return f"--- {info['file_path']} (ref: {info['ref']}) ---\n{info['content']}"


@tool
async def list_notes(
    project_id: int,
    noteable_type: NoteableKind,
    noteable_iid: int,
    include_system_notes: bool = False,
) -> str:
    """List comments/notes on a GitLab issue or merge request.

    Returns all human-authored notes in chronological order.  Useful
    for reading reviewer feedback, discussion threads, and prior
    comments.

    *noteable_type* must be ``"Issue"`` or ``"MergeRequest"``.
    Set *include_system_notes* to ``True`` to also show auto-generated
    notes (label changes, assignments, etc.).
    """
    client = get_client()
    notes = await asyncio.to_thread(
        client.list_notes, project_id, noteable_type, noteable_iid,
    )
    if not include_system_notes:
        notes = [n for n in notes if not n["system"]]

    if not notes:
        kind = "issue" if noteable_type == "Issue" else "MR"
        return f"No comments on {kind} #{noteable_iid} in project {project_id}."

    lines: list[str] = []
    for note in notes:
        lines.append(f"[{note['author']}] ({note['created_at']}):")
        lines.append(note["body"])
        lines.append("")
    return "\n".join(lines)


@tool
async def gitlab_mark_todo_done(todo_id: int) -> str:
    """Mark a GitLab TODO item as done.

    *todo_id* is the numeric ID of the TODO (not the issue/MR number).
    This is a GitLab-specific operation -- it marks one of your
    GitLab TODO notifications as resolved.
    """
    client = get_client()
    await asyncio.to_thread(client.mark_todo_done, todo_id)
    return f"Marked GitLab TODO {todo_id} as done."


GITLAB_TOOLS: list[object] = [
    read_issue,
    post_comment,
    create_merge_request,
    get_merge_request,
    list_merge_requests,
    list_notes,
    gitlab_get_project_info,
    gitlab_read_file,
    gitlab_mark_todo_done,
]
"""All GitLab tools as a list, suitable for use in ``tools=[GITLAB_TOOLS, ...]``."""

__all__ = [
    "GitLabConfig",
    "GitLabClient",
    "get_client",
    "set_client",
    "NoteableKind",
    "read_issue",
    "post_comment",
    "create_merge_request",
    "get_merge_request",
    "list_merge_requests",
    "list_notes",
    "gitlab_get_project_info",
    "gitlab_read_file",
    "gitlab_mark_todo_done",
    "GITLAB_TOOLS",
]
