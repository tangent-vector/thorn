"""Unified forge abstraction layer.

Provides a forge-neutral interface (``ForgeClient``) with GitLab and
GitHub adapter implementations, plus the service types that represent
forge connections and project definitions in a Thorn agency.

The ``FORGE_TOOLS`` toolset exposes nine ``@tool`` functions that
resolve the backing ``ForgeClient`` through the ambient
``ExecutionContext.runtime`` -- agents never need to know which forge
backend is in use.

Architecture:

    ForgeClient  (Protocol)
        GitLabForgeClient  (wraps thorn.tools.gitlab.GitLabClient)
        GitHubForgeClient  (wraps thorn.tools.github.GitHubClient)

    ForgeService (Service)  -- represents a forge host (URL + creds)
    ProjectService (Service)  -- represents a project on a forge
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, ClassVar, Literal, Protocol

from pydantic import BaseModel, Field

from thorn.core._context import get_context
from thorn.core._func import tool
from thorn.core._service import Service

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ForgeClient protocol
# ---------------------------------------------------------------------------

CommentTargetKind = Literal["Issue", "ChangeRequest"]
"""Forge-neutral target type for comments."""

ChangeRequestState = Literal["open", "closed", "merged", "all"]
"""Normalized state vocabulary for change requests."""


class ForgeClient(Protocol):
    """Unified interface for forge operations.

    Implementations translate between normalized names/states and the
    native forge API.  ``native_project_id`` is the forge-native
    identifier (stringified int for GitLab, ``owner/repo`` for GitHub).
    """

    def get_issue(
        self, native_project_id: str, issue_id: int,
    ) -> dict[str, Any]: ...

    def post_comment(
        self, native_project_id: str, target_type: str,
        target_id: int, body: str,
    ) -> None: ...

    def create_change_request(
        self, native_project_id: str, source_branch: str,
        title: str, target_branch: str, description: str,
    ) -> dict[str, Any]: ...

    def get_change_request(
        self, native_project_id: str, cr_id: int,
    ) -> dict[str, Any]: ...

    def list_change_requests(
        self, native_project_id: str, state: str,
    ) -> list[dict[str, Any]]: ...

    def list_comments(
        self, native_project_id: str, target_type: str, target_id: int,
    ) -> list[dict[str, Any]]: ...

    def get_project_info(
        self, native_project_id: str,
    ) -> dict[str, Any]: ...

    def read_file(
        self, native_project_id: str, file_path: str, ref: str,
    ) -> dict[str, Any]: ...

    def mark_notification_done(
        self, notification_id: str,
    ) -> None: ...


# ---------------------------------------------------------------------------
# GitLab adapter
# ---------------------------------------------------------------------------


class GitLabForgeClient:
    """``ForgeClient`` adapter wrapping ``GitLabClient``.

    Translates terminology (MR -> change request, note -> comment),
    maps ``native_project_id`` from string to int, and normalizes
    state vocabulary (``"opened"`` <-> ``"open"``).
    """

    def __init__(self, gl_client: Any) -> None:
        self._gl = gl_client

    @staticmethod
    def _pid(native_project_id: str) -> int:
        return int(native_project_id)

    @staticmethod
    def _target_type(target_type: str) -> str:
        if target_type == "ChangeRequest":
            return "MergeRequest"
        return target_type

    @staticmethod
    def _to_normalized_state(gl_state: str) -> str:
        if gl_state == "opened":
            return "open"
        return gl_state

    @staticmethod
    def _from_normalized_state(state: str) -> str:
        if state == "open":
            return "opened"
        return state

    def get_issue(
        self, native_project_id: str, issue_id: int,
    ) -> dict[str, Any]:
        raw = self._gl.get_issue(self._pid(native_project_id), issue_id)
        return {
            "id": raw["iid"],
            "title": raw["title"],
            "state": self._to_normalized_state(raw["state"]),
            "url": raw["web_url"],
            "description": raw.get("description", ""),
            "labels": raw.get("labels", []),
            "assignees": raw.get("assignees", []),
        }

    def post_comment(
        self, native_project_id: str, target_type: str,
        target_id: int, body: str,
    ) -> None:
        self._gl.post_note(
            self._pid(native_project_id),
            self._target_type(target_type),
            target_id,
            body,
        )

    def create_change_request(
        self, native_project_id: str, source_branch: str,
        title: str, target_branch: str, description: str,
    ) -> dict[str, Any]:
        raw = self._gl.create_merge_request(
            self._pid(native_project_id),
            source_branch=source_branch,
            title=title,
            target_branch=target_branch,
            description=description,
        )
        return {
            "id": raw["iid"],
            "title": raw["title"],
            "state": self._to_normalized_state(raw["state"]),
            "url": raw["web_url"],
            "source_branch": raw["source_branch"],
            "target_branch": raw["target_branch"],
        }

    def get_change_request(
        self, native_project_id: str, cr_id: int,
    ) -> dict[str, Any]:
        raw = self._gl.get_merge_request(
            self._pid(native_project_id), cr_id,
        )
        return {
            "id": raw["iid"],
            "title": raw["title"],
            "state": self._to_normalized_state(raw["state"]),
            "url": raw["web_url"],
            "description": raw.get("description", ""),
            "source_branch": raw["source_branch"],
            "target_branch": raw["target_branch"],
            "merge_status": raw.get("merge_status", ""),
        }

    def list_change_requests(
        self, native_project_id: str, state: str,
    ) -> list[dict[str, Any]]:
        gl_state = self._from_normalized_state(state)
        raw_list = self._gl.list_merge_requests(
            self._pid(native_project_id), gl_state,
        )
        return [
            {
                "id": mr["iid"],
                "title": mr["title"],
                "state": self._to_normalized_state(mr["state"]),
                "url": mr["web_url"],
                "source_branch": mr["source_branch"],
                "author": mr.get("author"),
            }
            for mr in raw_list
        ]

    def list_comments(
        self, native_project_id: str, target_type: str, target_id: int,
    ) -> list[dict[str, Any]]:
        raw_notes = self._gl.list_notes(
            self._pid(native_project_id),
            self._target_type(target_type),
            target_id,
        )
        return [
            {
                "author": n["author"],
                "created_at": n["created_at"],
                "body": n["body"],
                "is_system": n.get("system", False),
            }
            for n in raw_notes
        ]

    def get_project_info(
        self, native_project_id: str,
    ) -> dict[str, Any]:
        raw = self._gl.get_project_info(self._pid(native_project_id))
        return {
            "name": raw.get("name_with_namespace", raw.get("name", "")),
            "path": raw.get("path_with_namespace", ""),
            "clone_url": raw.get("http_url_to_repo", ""),
            "ssh_url": raw.get("ssh_url_to_repo", ""),
            "default_branch": raw.get("default_branch", ""),
            "url": raw.get("web_url", ""),
            "description": raw.get("description", ""),
        }

    def read_file(
        self, native_project_id: str, file_path: str, ref: str,
    ) -> dict[str, Any]:
        return self._gl.read_file(
            self._pid(native_project_id), file_path, ref,
        )

    def mark_notification_done(self, notification_id: str) -> None:
        self._gl.mark_todo_done(int(notification_id))


# ---------------------------------------------------------------------------
# GitHub adapter
# ---------------------------------------------------------------------------


class GitHubForgeClient:
    """``ForgeClient`` adapter wrapping ``GitHubClient``.

    Translates terminology (PR -> change request) and synthesizes the
    ``"merged"`` state from ``state == "closed" and merged == True``.
    """

    def __init__(self, gh_client: Any) -> None:
        self._gh = gh_client

    @staticmethod
    def _normalize_cr_state(raw: dict[str, Any]) -> str:
        """Derive normalized state from GitHub's state + merged flag."""
        if raw.get("merged"):
            return "merged"
        gh_state = raw.get("state", "open")
        return gh_state

    def get_issue(
        self, native_project_id: str, issue_id: int,
    ) -> dict[str, Any]:
        raw = self._gh.get_issue(native_project_id, issue_id)
        return {
            "id": raw["number"],
            "title": raw["title"],
            "state": raw["state"],
            "url": raw["html_url"],
            "description": raw.get("body", ""),
            "labels": raw.get("labels", []),
            "assignees": raw.get("assignees", []),
        }

    def post_comment(
        self, native_project_id: str, target_type: str,
        target_id: int, body: str,
    ) -> None:
        self._gh.post_comment(native_project_id, target_id, body)

    def create_change_request(
        self, native_project_id: str, source_branch: str,
        title: str, target_branch: str, description: str,
    ) -> dict[str, Any]:
        raw = self._gh.create_pull_request(
            native_project_id,
            head=source_branch,
            title=title,
            base=target_branch,
            body=description,
        )
        return {
            "id": raw["number"],
            "title": raw["title"],
            "state": raw["state"],
            "url": raw["html_url"],
            "source_branch": raw["head"],
            "target_branch": raw["base"],
        }

    def get_change_request(
        self, native_project_id: str, cr_id: int,
    ) -> dict[str, Any]:
        raw = self._gh.get_pull_request(native_project_id, cr_id)
        return {
            "id": raw["number"],
            "title": raw["title"],
            "state": self._normalize_cr_state(raw),
            "url": raw["html_url"],
            "description": raw.get("body", ""),
            "source_branch": raw["head"],
            "target_branch": raw["base"],
            "mergeable": raw.get("mergeable"),
            "mergeable_state": raw.get("mergeable_state", ""),
        }

    def list_change_requests(
        self, native_project_id: str, state: str,
    ) -> list[dict[str, Any]]:
        if state == "merged":
            raw_list = self._gh.list_pull_requests(
                native_project_id, "closed",
            )
            raw_list = [pr for pr in raw_list if pr.get("merged")]
        elif state == "open":
            raw_list = self._gh.list_pull_requests(
                native_project_id, "open",
            )
        else:
            raw_list = self._gh.list_pull_requests(
                native_project_id, state,
            )

        return [
            {
                "id": pr["number"],
                "title": pr["title"],
                "state": state if state == "merged" else pr["state"],
                "url": pr["html_url"],
                "source_branch": pr.get("head", ""),
                "author": pr.get("author"),
            }
            for pr in raw_list
        ]

    def list_comments(
        self, native_project_id: str, target_type: str, target_id: int,
    ) -> list[dict[str, Any]]:
        raw_comments = self._gh.list_comments(
            native_project_id, target_id,
        )
        return [
            {
                "author": c["author"],
                "created_at": c["created_at"],
                "body": c["body"],
                "is_system": c.get("is_bot", False),
            }
            for c in raw_comments
        ]

    def get_project_info(
        self, native_project_id: str,
    ) -> dict[str, Any]:
        raw = self._gh.get_repo_info(native_project_id)
        return {
            "name": raw.get("full_name", raw.get("name", "")),
            "path": raw.get("full_name", ""),
            "clone_url": raw.get("clone_url", ""),
            "ssh_url": raw.get("ssh_url", ""),
            "default_branch": raw.get("default_branch", ""),
            "url": raw.get("html_url", ""),
            "description": raw.get("description", ""),
        }

    def read_file(
        self, native_project_id: str, file_path: str, ref: str,
    ) -> dict[str, Any]:
        return self._gh.read_file(native_project_id, file_path, ref)

    def mark_notification_done(self, notification_id: str) -> None:
        self._gh.mark_notification_read(notification_id)


# ---------------------------------------------------------------------------
# ForgeService
# ---------------------------------------------------------------------------


class ForgeServiceConfig(BaseModel):
    """Configuration for a forge service (GitLab or GitHub host)."""

    url: str = Field(description="API base URL for the forge instance")
    token: str = Field(description="Access token with API scope")


class ForgeService(Service):
    """A named connection to a forge host (GitLab or GitHub).

    Owns a ``ForgeClient`` that provides the unified API for all
    operations against this forge instance.
    """

    Config: ClassVar[type[BaseModel]] = ForgeServiceConfig

    def __init__(
        self,
        config: ForgeServiceConfig,
        *,
        service_name: str,
        forge_type: str,
    ) -> None:
        self._config = config
        self._service_name = service_name
        self._forge_type = forge_type
        self._client: ForgeClient | None = None

    @property
    def name(self) -> str:
        return self._service_name

    @property
    def forge_type(self) -> str:
        return self._forge_type

    @property
    def client(self) -> ForgeClient:
        """Lazily-constructed ``ForgeClient`` for this forge."""
        if self._client is None:
            self._client = self._build_client()
        return self._client

    def _build_client(self) -> ForgeClient:
        if self._forge_type == "gitlab":
            from thorn.tools.gitlab import GitLabClient, GitLabConfig

            gl_config = GitLabConfig(
                url=self._config.url, token=self._config.token,
            )
            return GitLabForgeClient(GitLabClient(gl_config))
        elif self._forge_type == "github":
            from thorn.tools.github import GitHubClient, GitHubConfig

            gh_config = GitHubConfig(
                base_url=self._config.url, token=self._config.token,
            )
            return GitHubForgeClient(GitHubClient(gh_config))
        else:
            raise ValueError(
                f"Unknown forge type: {self._forge_type!r}. "
                f"Expected 'gitlab' or 'github'."
            )


# ---------------------------------------------------------------------------
# ProjectService
# ---------------------------------------------------------------------------


class ProjectServiceConfig(BaseModel):
    """Configuration for a project service."""

    forge: str = Field(
        description="Name of the forge service this project belongs to",
    )
    native_id: str = Field(
        description=(
            "Forge-native project identifier "
            "(numeric string for GitLab, owner/repo for GitHub)"
        ),
    )
    path: str = Field(
        default="",
        description="Human-readable project path (e.g. 'lace/lace')",
    )
    clone_url: str = Field(
        default="",
        description="HTTPS clone URL for the repository",
    )
    default_branch: str = Field(
        default="main",
        description="Default branch name",
    )


class ProjectService(Service):
    """A named project within a forge.

    Carries the project's native identifier, clone URL, and default
    branch.  Resolves its ``ForgeClient`` by looking up its forge
    service on the ``Runtime``.
    """

    Config: ClassVar[type[BaseModel]] = ProjectServiceConfig

    def __init__(
        self,
        config: ProjectServiceConfig,
        *,
        service_name: str,
    ) -> None:
        self._config = config
        self._service_name = service_name

    @property
    def name(self) -> str:
        return self._service_name

    @property
    def forge_name(self) -> str:
        """Name of the forge service this project belongs to."""
        return self._config.forge

    @property
    def native_id(self) -> str:
        return self._config.native_id

    @property
    def path(self) -> str:
        return self._config.path

    @property
    def clone_url(self) -> str:
        return self._config.clone_url

    @property
    def default_branch(self) -> str:
        return self._config.default_branch

    def get_forge_client(self, runtime: Any) -> tuple[ForgeClient, str]:
        """Resolve this project's ``ForgeClient`` via the Runtime.

        Returns ``(client, native_id)`` so callers can make API calls
        using the forge-native project identifier.
        """
        forge_service: ForgeService = runtime.get_service(self.forge_name)
        if not isinstance(forge_service, ForgeService):
            raise TypeError(
                f"Service {self.forge_name!r} is a "
                f"{type(forge_service).__name__}, not a ForgeService"
            )
        return forge_service.client, self.native_id


# ---------------------------------------------------------------------------
# Runtime convenience method (added via monkey-free helper)
# ---------------------------------------------------------------------------


def get_forge_for_project(
    runtime: Any, project_name: str,
) -> tuple[ForgeClient, str]:
    """Look up a project service and return its ``(ForgeClient, native_id)``.

    This is the primary entry point for forge tools: given a project
    name from a tool parameter, obtain the client and native ID needed
    to make API calls.
    """
    project: ProjectService = runtime.get_service(project_name)
    if not isinstance(project, ProjectService):
        raise TypeError(
            f"Service {project_name!r} is a "
            f"{type(project).__name__}, not a ProjectService"
        )
    return project.get_forge_client(runtime)


# ---------------------------------------------------------------------------
# FORGE_TOOLS
# ---------------------------------------------------------------------------


def _resolve(project: str) -> tuple[ForgeClient, str]:
    """Resolve the ForgeClient + native ID from the execution context."""
    ctx = get_context()
    if ctx.runtime is None:
        raise RuntimeError(
            "No Runtime available in the current ExecutionContext. "
            "Forge tools require a Runtime with registered services."
        )
    return get_forge_for_project(ctx.runtime, project)


@tool
async def forge_read_issue(project: str, issue_id: int) -> str:
    """Read an issue from a project.

    *project* is the name of the project service.  *issue_id* is the
    issue number within the project.
    """
    client, native_id = _resolve(project)
    info = await asyncio.to_thread(client.get_issue, native_id, issue_id)
    lines = [
        f"Issue #{info['id']}: {info['title']}",
        f"State: {info['state']}",
        f"Labels: {', '.join(info.get('labels', [])) or '(none)'}",
        f"Assignees: {', '.join(info.get('assignees', [])) or '(none)'}",
        f"URL: {info['url']}",
        "",
        info.get("description") or "(no description)",
    ]
    return "\n".join(lines)


@tool
async def forge_post_comment(
    project: str,
    target_type: CommentTargetKind,
    target_id: int,
    body: str,
) -> str:
    """Post a comment on an issue or change request.

    *target_type* must be ``"Issue"`` or ``"ChangeRequest"``.
    *target_id* is the issue or change-request number.
    """
    client, native_id = _resolve(project)
    await asyncio.to_thread(
        client.post_comment, native_id, target_type, target_id, body,
    )
    return (
        f"Posted comment on {target_type} "
        f"#{target_id} in project {project!r}."
    )


@tool
async def forge_create_change_request(
    project: str,
    source_branch: str,
    title: str,
    description: str = "",
    target_branch: str = "main",
) -> str:
    """Create a new change request (merge request / pull request).

    Opens a change request from *source_branch* into *target_branch*
    (default ``main``) in the specified project.
    """
    client, native_id = _resolve(project)
    info = await asyncio.to_thread(
        client.create_change_request,
        native_id,
        source_branch=source_branch,
        title=title,
        target_branch=target_branch,
        description=description,
    )
    return (
        f"Created change request #{info['id']}: {info['title']}\n"
        f"  {info['source_branch']} -> {info['target_branch']}\n"
        f"  URL: {info['url']}"
    )


@tool
async def forge_get_change_request(project: str, cr_id: int) -> str:
    """Read details of a change request (merge request / pull request).

    Returns the title, state, branches, and description.
    """
    client, native_id = _resolve(project)
    info = await asyncio.to_thread(
        client.get_change_request, native_id, cr_id,
    )
    lines = [
        f"Change request #{info['id']}: {info['title']}",
        f"State: {info['state']}",
        f"Branches: {info['source_branch']} -> {info['target_branch']}",
        f"URL: {info['url']}",
        "",
        info.get("description") or "(no description)",
    ]
    return "\n".join(lines)


@tool
async def forge_list_change_requests(
    project: str,
    state: ChangeRequestState = "open",
) -> str:
    """List change requests (merge requests / pull requests) in a project.

    Filters by *state* (default ``"open"``).  Valid states are
    ``"open"``, ``"closed"``, ``"merged"``, and ``"all"``.
    """
    client, native_id = _resolve(project)
    crs = await asyncio.to_thread(
        client.list_change_requests, native_id, state,
    )
    if not crs:
        return f"No {state} change requests in project {project!r}."
    lines = []
    for cr in crs:
        author = cr.get("author") or "unknown"
        lines.append(
            f"  #{cr['id']}: {cr['title']} ({cr['state']}, by {author})"
        )
    header = f"{len(crs)} {state} change request(s) in project {project!r}:"
    return "\n".join([header, *lines])


@tool
async def forge_list_comments(
    project: str,
    target_type: CommentTargetKind,
    target_id: int,
    include_system: bool = False,
) -> str:
    """List comments on an issue or change request.

    *target_type* must be ``"Issue"`` or ``"ChangeRequest"``.
    Set *include_system* to ``True`` to also show auto-generated
    comments (label changes, bot comments, etc.).
    """
    client, native_id = _resolve(project)
    comments = await asyncio.to_thread(
        client.list_comments, native_id, target_type, target_id,
    )
    if not include_system:
        comments = [c for c in comments if not c.get("is_system", False)]

    if not comments:
        kind = "issue" if target_type == "Issue" else "change request"
        return (
            f"No comments on {kind} #{target_id} in project {project!r}."
        )

    lines: list[str] = []
    for c in comments:
        lines.append(f"[{c['author']}] ({c['created_at']}):")
        lines.append(c["body"])
        lines.append("")
    return "\n".join(lines)


@tool
async def forge_get_project_info(project: str) -> str:
    """Get information about a project.

    Returns the project's name, clone URLs, default branch, and
    description.
    """
    client, native_id = _resolve(project)
    info = await asyncio.to_thread(client.get_project_info, native_id)
    lines = [
        f"Project: {info['name']}",
        f"Path: {info['path']}",
        f"Clone URL (HTTPS): {info['clone_url']}",
        f"Clone URL (SSH): {info['ssh_url']}",
        f"Default branch: {info['default_branch']}",
        f"URL: {info['url']}",
    ]
    if info.get("description"):
        lines.append(f"Description: {info['description']}")
    return "\n".join(lines)


@tool
async def forge_read_file(
    project: str,
    file_path: str,
    ref: str = "HEAD",
) -> str:
    """Read a file from a project's repository via the forge API.

    Useful for inspecting files without cloning the entire repository.
    *ref* can be a branch name, tag, or commit SHA.
    """
    client, native_id = _resolve(project)
    info = await asyncio.to_thread(
        client.read_file, native_id, file_path, ref,
    )
    return f"--- {info['file_path']} (ref: {info['ref']}) ---\n{info['content']}"


@tool
async def forge_mark_notification_done(
    project: str,
    notification_id: str,
) -> str:
    """Mark a notification as done/read.

    *notification_id* is the forge-native notification identifier
    (numeric string for both GitLab TODOs and GitHub notification
    threads).
    """
    client, _native_id = _resolve(project)
    await asyncio.to_thread(
        client.mark_notification_done, notification_id,
    )
    return (
        f"Marked notification {notification_id} as done "
        f"(project {project!r})."
    )


FORGE_TOOLS: list[object] = [
    forge_read_issue,
    forge_post_comment,
    forge_create_change_request,
    forge_get_change_request,
    forge_list_change_requests,
    forge_list_comments,
    forge_get_project_info,
    forge_read_file,
    forge_mark_notification_done,
]
"""All forge-neutral tools as a list, suitable for ``tools=[FORGE_TOOLS, ...]``."""


__all__ = [
    "ChangeRequestState",
    "CommentTargetKind",
    "FORGE_TOOLS",
    "ForgeClient",
    "ForgeService",
    "ForgeServiceConfig",
    "GitHubForgeClient",
    "GitLabForgeClient",
    "ProjectService",
    "ProjectServiceConfig",
    "forge_create_change_request",
    "forge_get_change_request",
    "forge_get_project_info",
    "forge_list_change_requests",
    "forge_list_comments",
    "forge_mark_notification_done",
    "forge_post_comment",
    "forge_read_file",
    "forge_read_issue",
    "get_forge_for_project",
]
