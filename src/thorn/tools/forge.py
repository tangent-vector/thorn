"""Unified forge abstraction layer.

Provides a forge-neutral interface (``ForgeClient``) with GitLab and
GitHub adapter implementations, plus the service types that represent
forge connections and project definitions in a Thorn agency.

The ``FORGE_TOOLS`` toolset exposes twelve ``@tool`` functions that
resolve the backing ``ForgeClient`` through the ambient
``ExecutionContext.runtime`` -- agents never need to know which forge
backend is in use.

Architecture:

    ForgeClient  (Protocol)
        GitLabForgeClient  (wraps thorn.tools.gitlab.GitLabClient)
        GitHubForgeClient  (wraps thorn.tools.github.GitHubClient)

    ForgeHostService (Service)  -- GitLabForgeService or GitHubForgeService
    ProjectService (Service)  -- represents a project on a forge
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Any, ClassVar, Literal, Protocol

from pydantic import BaseModel, Field

from thorn.core._account import ForgeCredentials, GitLabCredentials
from thorn.core._context import get_context
from thorn.core._func import tool
from thorn.core._service import Service
from thorn.tools._github_connection import (
    GitHubAppAuth,
    GitHubConnectionConfig,
    GitHubPatAuth,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ForgeClient protocol
# ---------------------------------------------------------------------------

CommentTargetKind = Literal["Issue", "ChangeRequest"]
"""Forge-neutral target type for comments."""

ChangeRequestState = Literal["open", "closed", "merged", "all"]
"""Normalized state vocabulary for change requests."""

IssueState = Literal["open", "closed", "all"]
"""Normalized state vocabulary for issues."""


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

    def create_issue(
        self, native_project_id: str, title: str, description: str,
        labels: list[str], assignees: list[str],
    ) -> dict[str, Any]: ...

    def list_issues(
        self, native_project_id: str, state: str,
        labels: list[str] | None,
    ) -> list[dict[str, Any]]: ...

    def update_issue(
        self, native_project_id: str, issue_id: int,
        title: str | None, description: str | None,
        state: str | None, labels: list[str] | None,
        assignees: list[str] | None,
    ) -> dict[str, Any]: ...


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
    def _pid(native_project_id: str) -> int | str:
        """Coerce a config-supplied native ID into the form python-gitlab expects.

        ``python-gitlab``'s ``projects.get(...)`` accepts either a
        numeric ID or a URL-encoded ``namespace/project/path`` string.
        Configurations historically required the numeric form; the new
        gateway config shape derives the path-based form from the
        fork's URL, so pass either through unchanged (numeric digits
        get parsed as ``int`` for parity with the legacy behaviour;
        anything else is forwarded verbatim).
        """
        if native_project_id.isdigit():
            return int(native_project_id)
        return native_project_id

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

    def create_issue(
        self, native_project_id: str, title: str, description: str,
        labels: list[str], assignees: list[str],
    ) -> dict[str, Any]:
        raw = self._gl.create_issue(
            self._pid(native_project_id),
            title=title,
            description=description,
            labels=labels or None,
            assignees=assignees or None,
        )
        return {
            "id": raw["iid"],
            "title": raw["title"],
            "state": self._to_normalized_state(raw["state"]),
            "url": raw["web_url"],
            "description": raw.get("description", ""),
            "labels": raw.get("labels", []),
            "assignees": raw.get("assignees", []),
        }

    def list_issues(
        self, native_project_id: str, state: str,
        labels: list[str] | None,
    ) -> list[dict[str, Any]]:
        gl_state = self._from_normalized_state(state)
        raw_list = self._gl.list_issues(
            self._pid(native_project_id), gl_state, labels,
        )
        return [
            {
                "id": issue["iid"],
                "title": issue["title"],
                "state": self._to_normalized_state(issue["state"]),
                "url": issue["web_url"],
                "labels": issue.get("labels", []),
                "assignees": issue.get("assignees", []),
                "author": issue.get("author"),
            }
            for issue in raw_list
        ]

    def update_issue(
        self, native_project_id: str, issue_id: int,
        title: str | None, description: str | None,
        state: str | None, labels: list[str] | None,
        assignees: list[str] | None,
    ) -> dict[str, Any]:
        gl_state = None
        if state is not None:
            gl_state = {"open": "reopen", "closed": "close"}.get(state, state)
        raw = self._gl.update_issue(
            self._pid(native_project_id), issue_id,
            title=title, description=description,
            state=gl_state, labels=labels, assignees=assignees,
        )
        return {
            "id": raw["iid"],
            "title": raw["title"],
            "state": self._to_normalized_state(raw["state"]),
            "url": raw["web_url"],
            "description": raw.get("description", ""),
            "labels": raw.get("labels", []),
            "assignees": raw.get("assignees", []),
        }


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
                "state": self._normalize_cr_state(pr),
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
        self._gh.mark_notification_done(notification_id)

    def create_issue(
        self, native_project_id: str, title: str, description: str,
        labels: list[str], assignees: list[str],
    ) -> dict[str, Any]:
        raw = self._gh.create_issue(
            native_project_id,
            title=title,
            body=description,
            labels=labels or None,
            assignees=assignees or None,
        )
        return {
            "id": raw["number"],
            "title": raw["title"],
            "state": raw["state"],
            "url": raw["html_url"],
            "description": raw.get("body", ""),
            "labels": raw.get("labels", []),
            "assignees": raw.get("assignees", []),
        }

    def list_issues(
        self, native_project_id: str, state: str,
        labels: list[str] | None,
    ) -> list[dict[str, Any]]:
        raw_list = self._gh.list_issues(native_project_id, state, labels)
        return [
            {
                "id": issue["number"],
                "title": issue["title"],
                "state": issue["state"],
                "url": issue["html_url"],
                "labels": issue.get("labels", []),
                "assignees": issue.get("assignees", []),
                "author": issue.get("author"),
            }
            for issue in raw_list
        ]

    def update_issue(
        self, native_project_id: str, issue_id: int,
        title: str | None, description: str | None,
        state: str | None, labels: list[str] | None,
        assignees: list[str] | None,
    ) -> dict[str, Any]:
        raw = self._gh.update_issue(
            native_project_id, issue_id,
            title=title, body=description,
            state=state, labels=labels, assignees=assignees,
        )
        return {
            "id": raw["number"],
            "title": raw["title"],
            "state": raw["state"],
            "url": raw["html_url"],
            "description": raw.get("body", ""),
            "labels": raw.get("labels", []),
            "assignees": raw.get("assignees", []),
        }


# ---------------------------------------------------------------------------
# Forge host services (GitLab vs GitHub)
# ---------------------------------------------------------------------------


class GitLabForgeServiceConfig(BaseModel):
    """Configuration for a ``gitlab`` forge service."""

    url: str = Field(description="GitLab API base URL")
    token: str = Field(description="Personal access token with API scope")


class ForgeHostService(Service, ABC):
    """API client plus credentials for HTTPS git against this forge host.

    Concrete subclasses implement two families of methods:

    **Legacy (credentials baked into the service config):**
    ``client`` property and ``git_https_password()`` — used by
    existing code that reads credentials from the forge service
    config.  These will be removed once all consumers migrate to
    the account-based methods below.

    **Account-based (credentials passed in):**
    ``authenticated_client(credentials)`` and
    ``git_https_password_for(credentials)`` — the target API.
    Consumers resolve the agent's :class:`ForgeAccountConfig` for
    this forge, then pass its credentials here.
    """

    @property
    @abstractmethod
    def client(self) -> ForgeClient:
        """Forge-neutral client for issues, MRs/PRs, etc.

        .. deprecated::
            Use :meth:`authenticated_client` with explicit credentials.
        """

    @abstractmethod
    def git_https_password(self) -> str:
        """Password (or token) for embedding in HTTPS git URLs.

        .. deprecated::
            Use :meth:`git_https_password_for` with explicit credentials.
        """

    @abstractmethod
    def authenticated_client(
        self,
        credentials: ForgeCredentials,
    ) -> ForgeClient:
        """Build a forge client authenticated with the given credentials.

        Unlike the ``client`` property, this does not rely on
        credentials stored in the service config — they come from the
        agent's :class:`ForgeAccountConfig` instead.
        """

    @abstractmethod
    def git_https_password_for(
        self,
        credentials: ForgeCredentials,
    ) -> str:
        """Return the HTTPS password/token for git operations.

        The credential material is extracted from *credentials*
        (typically a PAT token string or an installation access token).
        """

    @abstractmethod
    def clone_url_for(self, native_id: str) -> str:
        """Derive the HTTPS clone URL for a project on this forge.

        Each forge backend knows how to construct a clone URL from
        its base URL and the forge-native project identifier.
        """


class GitLabForgeService(ForgeHostService):
    """Connection to a GitLab instance."""

    Config: ClassVar[type[BaseModel]] = GitLabForgeServiceConfig

    def __init__(self, config: GitLabForgeServiceConfig, *, service_name: str) -> None:
        self._config = config
        self._service_name = service_name
        self._client: ForgeClient | None = None

    @property
    def name(self) -> str:
        return self._service_name

    @property
    def url(self) -> str:
        """GitLab instance API base URL."""
        return self._config.url

    @property
    def client(self) -> ForgeClient:
        if self._client is None:
            from thorn.tools.gitlab import GitLabClient, GitLabConfig

            gl_config = GitLabConfig(
                url=self._config.url, token=self._config.token,
            )
            self._client = GitLabForgeClient(GitLabClient(gl_config))
        return self._client

    def git_https_password(self) -> str:
        return self._config.token

    def _extract_gitlab_token(self, credentials: ForgeCredentials) -> str:
        if not isinstance(credentials, GitLabCredentials):
            raise TypeError(
                f"GitLabForgeService requires GitLabCredentials, "
                f"got {type(credentials).__name__}"
            )
        return credentials.token

    def authenticated_client(
        self,
        credentials: ForgeCredentials,
    ) -> ForgeClient:
        from thorn.tools.gitlab import GitLabClient, GitLabConfig

        token = self._extract_gitlab_token(credentials)
        gl_config = GitLabConfig(url=self._config.url, token=token)
        return GitLabForgeClient(GitLabClient(gl_config))

    def git_https_password_for(
        self,
        credentials: ForgeCredentials,
    ) -> str:
        return self._extract_gitlab_token(credentials)

    def clone_url_for(self, native_id: str) -> str:
        """Derive an HTTPS clone URL for a GitLab project.

        With the path-based ``native_id`` shape used by the new
        gateway config (e.g. ``"group/subgroup/project"``), the clone
        URL is just the GitLab instance URL with that path appended
        and a ``.git`` suffix.

        Returns an empty string when the native_id is purely numeric
        (legacy callers): we don't have enough information to
        reconstruct the path without a forge API call.
        """
        if not native_id or native_id.isdigit():
            return ""
        from urllib.parse import urlparse

        parsed = urlparse(self._config.url)
        host = parsed.hostname or ""
        scheme = parsed.scheme or "https"
        if not host:
            return ""
        return f"{scheme}://{host}/{native_id}.git"


class GitHubForgeService(ForgeHostService):
    """Connection to GitHub or GitHub Enterprise (PAT or GitHub App)."""

    Config: ClassVar[type[BaseModel]] = GitHubConnectionConfig

    def __init__(self, config: GitHubConnectionConfig, *, service_name: str) -> None:
        self._config = config
        self._service_name = service_name
        self._forge_client: ForgeClient | None = None
        self._github_client: Any = None

    @property
    def name(self) -> str:
        return self._service_name

    @property
    def base_url(self) -> str:
        """GitHub REST API base URL."""
        return self._config.base_url

    @property
    def client(self) -> ForgeClient:
        if self._forge_client is None:
            from thorn.tools.github import GitHubClient

            self._github_client = GitHubClient(self._config)
            self._forge_client = GitHubForgeClient(self._github_client)
        return self._forge_client

    def git_https_password(self) -> str:
        _ = self.client
        assert self._github_client is not None
        return self._github_client.bearer_token_for_http()

    def _build_connection_config(
        self,
        credentials: ForgeCredentials,
    ) -> GitHubConnectionConfig:
        if not isinstance(credentials, (GitHubPatAuth, GitHubAppAuth)):
            raise TypeError(
                f"GitHubForgeService requires GitHubPatAuth or "
                f"GitHubAppAuth, got {type(credentials).__name__}"
            )
        return GitHubConnectionConfig(
            base_url=self._config.base_url,
            auth=credentials,
        )

    def authenticated_client(
        self,
        credentials: ForgeCredentials,
    ) -> ForgeClient:
        from thorn.tools.github import GitHubClient

        conn = self._build_connection_config(credentials)
        gh_client = GitHubClient(conn)
        return GitHubForgeClient(gh_client)

    def git_https_password_for(
        self,
        credentials: ForgeCredentials,
    ) -> str:
        from thorn.tools.github import GitHubClient

        conn = self._build_connection_config(credentials)
        gh_client = GitHubClient(conn)
        return gh_client.bearer_token_for_http()

    def clone_url_for(self, native_id: str) -> str:
        """Derive an HTTPS clone URL for a GitHub repository.

        For GitHub, ``native_id`` is ``owner/repo``, and the clone
        URL is ``https://<host>/<owner>/<repo>.git``.  The host is
        derived from the API base URL.
        """
        from urllib.parse import urlparse

        parsed = urlparse(self._config.base_url)
        host = parsed.hostname or "github.com"
        if host.startswith("api."):
            host = host[4:]
        return f"https://{host}/{native_id}.git"


# ---------------------------------------------------------------------------
# ProjectService
# ---------------------------------------------------------------------------


class ForkConfig(BaseModel):
    """A single fork of a project hosted on a forge.

    Built by :func:`thorn.gateway._config.instantiate_services` from a
    :class:`thorn.gateway._config.ForkSpec` -- ``native_id`` and
    ``clone_url`` are derived from the user-facing ``url`` so that
    consumers of ``ForkConfig`` have all the forge-specific identifiers
    they need without re-parsing.

    The ``name`` becomes the git remote name when the project is
    cloned locally (defaults are picked at config-resolution time:
    ``"origin"`` for single-fork projects, the forge name otherwise).
    """

    forge: str = Field(description="Name of the forge service hosting this fork")
    native_id: str = Field(
        description=(
            "Forge-native project identifier "
            "(``namespace/path`` for GitLab, ``owner/repo`` for GitHub)"
        ),
    )
    name: str = Field(
        default="",
        description="Local remote name for this fork (e.g. 'origin')",
    )
    clone_url: str = Field(
        default="",
        description="HTTPS clone URL for this fork",
    )
    default_branch: str = Field(
        default="",
        description=(
            "Per-fork override for the default branch.  When empty, "
            "the project-level default is used; when that is also "
            "empty, the value is fetched from the forge API on first "
            "access (see :meth:`ProjectService.resolve_default_branch`)."
        ),
    )


class ProjectServiceConfig(BaseModel):
    """Configuration for a project service.

    A project has one or more forks.  ``default_branch`` is an
    optional project-level override; when omitted, the value is
    resolved lazily by querying the forge API for the primary fork
    (see :meth:`ProjectService.resolve_default_branch`).
    """

    forks: list[ForkConfig] = Field(
        default_factory=list,
        description=(
            "List of forks of this project.  Defaults to an empty "
            "list so that a ``ProjectServiceConfig`` can be constructed "
            "incrementally; methods that need at least one fork (e.g. "
            ":meth:`ProjectService.get_fork`, "
            ":attr:`ProjectService.forge_name`) raise a clear "
            "``KeyError`` / ``ValueError`` when the list is still empty."
        ),
    )
    default_branch: str = Field(
        default="",
        description=(
            "Project-level default branch override.  When empty, "
            "the value is looked up from the forge for the primary "
            "fork on first access and cached for the process lifetime."
        ),
    )


class ProjectService(Service):
    """A named project within a forge (possibly with multiple forks).

    Carries the project's fork list and convenience accessors for
    the primary fork.  Resolves its ``ForgeClient`` by looking up the
    forge service on the ``Runtime``.

    The default branch for any given fork is resolved by
    :meth:`resolve_default_branch`, which consults a cascade of
    overrides before falling back to a forge API lookup.
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
        # Per-fork default-branch cache (filled by
        # ``resolve_default_branch`` on first lookup).  Keys are fork
        # names.  The cache is process-scoped because the default
        # branch on a forge changes infrequently and a stale value
        # is acceptable until the gateway restarts.
        self._default_branch_cache: dict[str, str] = {}

    @property
    def name(self) -> str:
        return self._service_name

    @property
    def forks(self) -> list[ForkConfig]:
        """The effective fork list for this project."""
        return list(self._config.forks)

    @property
    def forge_name(self) -> str:
        """Name of the forge hosting the primary fork."""
        forks = self.forks
        if not forks:
            raise ValueError(
                f"Project {self.name!r} has no forks configured"
            )
        return forks[0].forge

    @property
    def native_id(self) -> str:
        """Native identifier of the primary fork."""
        forks = self.forks
        if not forks:
            raise ValueError(
                f"Project {self.name!r} has no forks configured"
            )
        return forks[0].native_id

    @property
    def clone_url(self) -> str:
        """Clone URL of the primary fork."""
        forks = self.forks
        if not forks:
            return ""
        return forks[0].clone_url

    @property
    def default_branch(self) -> str:
        """Project-level default branch override (may be empty).

        This returns only the *configured* override; it does **not**
        trigger a forge API lookup.  Consumers that need a definitive
        value should call :meth:`resolve_default_branch`, which
        cascades through overrides and falls back to a live lookup.
        """
        return self._config.default_branch

    def resolve_default_branch(
        self,
        runtime: Any,
        fork_name: str = "",
    ) -> str:
        """Resolve the default branch for *fork_name*, looking it up if needed.

        Resolution cascade:

        1. Per-fork override on :attr:`ForkConfig.default_branch`.
        2. Project-level override on :attr:`default_branch`.
        3. Process-cached previous lookup.
        4. Live ``get_project_info`` call against the fork's forge.

        The result of step 4 is cached per fork for subsequent calls.
        Raises :class:`KeyError` for an unknown fork name.
        """
        fork = self.get_fork(fork_name)

        if fork.default_branch:
            return fork.default_branch
        if self._config.default_branch:
            return self._config.default_branch

        cache_key = fork.name
        cached = self._default_branch_cache.get(cache_key)
        if cached:
            return cached

        forge_svc = runtime.get_service(fork.forge)
        if not isinstance(forge_svc, ForgeHostService):
            raise TypeError(
                f"Service {fork.forge!r} is a "
                f"{type(forge_svc).__name__}, not a ForgeHostService"
            )
        info = forge_svc.client.get_project_info(fork.native_id)
        resolved = info.get("default_branch") or "main"
        self._default_branch_cache[cache_key] = resolved
        return resolved

    def get_fork(self, fork_name: str = "") -> ForkConfig:
        """Look up a fork by name, defaulting to the first fork.

        Raises :class:`KeyError` when the named fork doesn't exist.
        """
        forks = self.forks
        if not forks:
            raise KeyError(
                f"Project {self.name!r} has no forks configured"
            )
        if not fork_name:
            return forks[0]
        for fork in forks:
            if fork.name == fork_name:
                return fork
        raise KeyError(
            f"No fork named {fork_name!r} in project {self.name!r}. "
            f"Available: {[f.name for f in forks]}"
        )

    def get_forge_client(self, runtime: Any) -> tuple[ForgeClient, str]:
        """Resolve this project's ``ForgeClient`` via the Runtime.

        Returns ``(client, native_id)`` for the primary fork so
        callers can make API calls using the forge-native project
        identifier.
        """
        forge_service: ForgeHostService = runtime.get_service(self.forge_name)
        if not isinstance(forge_service, ForgeHostService):
            raise TypeError(
                f"Service {self.forge_name!r} is a "
                f"{type(forge_service).__name__}, not a ForgeHostService"
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
    """Resolve an authenticated ForgeClient + native ID for *project*.

    Uses the current agent's :class:`ForgeAccountConfig` to
    authenticate.  Falls back to the legacy ``ForgeHostService.client``
    property when the agent has no account on the project's forge
    (backward compat with old-style configs where credentials live on
    the forge service itself).
    """
    from thorn.core._account import resolve_forge_account

    ctx = get_context()
    if ctx.runtime is None:
        raise RuntimeError(
            "No Runtime available in the current ExecutionContext. "
            "Forge tools require a Runtime with registered services."
        )

    project_svc: ProjectService = ctx.runtime.get_service(project)
    if not isinstance(project_svc, ProjectService):
        raise TypeError(
            f"Service {project!r} is a "
            f"{type(project_svc).__name__}, not a ProjectService"
        )

    forge_svc: ForgeHostService = ctx.runtime.get_service(
        project_svc.forge_name,
    )
    if not isinstance(forge_svc, ForgeHostService):
        raise TypeError(
            f"Service {project_svc.forge_name!r} is a "
            f"{type(forge_svc).__name__}, not a ForgeHostService"
        )

    agent = ctx.agent
    if agent is not None:
        try:
            account = resolve_forge_account(agent, forge_svc.name)
            return (
                forge_svc.authenticated_client(account.credentials),
                project_svc.native_id,
            )
        except KeyError:
            pass

    return forge_svc.client, project_svc.native_id


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
async def forge_create_issue(
    project: str,
    title: str,
    description: str = "",
    labels: list[str] | None = None,
    assignees: list[str] | None = None,
) -> str:
    """Create a new issue in a project.

    *project* is the name of the project service.  *labels* and
    *assignees* are optional lists of label names and usernames.
    """
    client, native_id = _resolve(project)
    info = await asyncio.to_thread(
        client.create_issue,
        native_id,
        title=title,
        description=description,
        labels=labels or [],
        assignees=assignees or [],
    )
    lines = [
        f"Created issue #{info['id']}: {info['title']}",
        f"State: {info['state']}",
        f"URL: {info['url']}",
    ]
    if info.get("labels"):
        lines.append(f"Labels: {', '.join(info['labels'])}")
    if info.get("assignees"):
        lines.append(f"Assignees: {', '.join(info['assignees'])}")
    return "\n".join(lines)


@tool
async def forge_list_issues(
    project: str,
    state: IssueState = "open",
    labels: list[str] | None = None,
) -> str:
    """List issues in a project.

    Filters by *state* (default ``"open"``).  Optionally filter by
    *labels* (list of label names -- issues must have **all** listed
    labels).
    """
    client, native_id = _resolve(project)
    issues = await asyncio.to_thread(
        client.list_issues, native_id, state, labels,
    )
    if not issues:
        return f"No {state} issues in project {project!r}."
    lines: list[str] = []
    for issue in issues:
        author = issue.get("author") or "unknown"
        lines.append(
            f"  #{issue['id']}: {issue['title']} ({issue['state']}, by {author})"
        )
    header = f"{len(issues)} {state} issue(s) in project {project!r}:"
    return "\n".join([header, *lines])


@tool
async def forge_update_issue(
    project: str,
    issue_id: int,
    title: str | None = None,
    description: str | None = None,
    state: IssueState | None = None,
    add_labels: list[str] | None = None,
    remove_labels: list[str] | None = None,
    add_assignees: list[str] | None = None,
    remove_assignees: list[str] | None = None,
) -> str:
    """Update an existing issue.

    All fields are optional -- only the ones provided are changed.
    Labels and assignees support add/remove semantics: the tool reads
    the current values and merges the requested changes before sending
    the update.
    """
    client, native_id = _resolve(project)

    labels: list[str] | None = None
    assignees: list[str] | None = None
    if add_labels or remove_labels or add_assignees or remove_assignees:
        current = await asyncio.to_thread(
            client.get_issue, native_id, issue_id,
        )
        if add_labels or remove_labels:
            current_labels = set(current.get("labels", []))
            current_labels |= set(add_labels or [])
            current_labels -= set(remove_labels or [])
            labels = sorted(current_labels)
        if add_assignees or remove_assignees:
            current_assignees = set(current.get("assignees", []))
            current_assignees |= set(add_assignees or [])
            current_assignees -= set(remove_assignees or [])
            assignees = sorted(current_assignees)

    info = await asyncio.to_thread(
        client.update_issue,
        native_id, issue_id,
        title=title, description=description,
        state=state, labels=labels, assignees=assignees,
    )
    lines = [
        f"Updated issue #{info['id']}: {info['title']}",
        f"State: {info['state']}",
        f"URL: {info['url']}",
    ]
    if info.get("labels"):
        lines.append(f"Labels: {', '.join(info['labels'])}")
    if info.get("assignees"):
        lines.append(f"Assignees: {', '.join(info['assignees'])}")
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
    forge_create_issue,
    forge_list_issues,
    forge_update_issue,
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
    "ForkConfig",
    "ForgeHostService",
    "GitHubForgeService",
    "GitLabForgeService",
    "GitHubForgeClient",
    "GitLabForgeClient",
    "GitLabForgeServiceConfig",
    "IssueState",
    "ProjectService",
    "ProjectServiceConfig",
    "forge_create_change_request",
    "forge_create_issue",
    "forge_get_change_request",
    "forge_get_project_info",
    "forge_list_change_requests",
    "forge_list_comments",
    "forge_list_issues",
    "forge_mark_notification_done",
    "forge_post_comment",
    "forge_read_file",
    "forge_read_issue",
    "forge_update_issue",
    "get_forge_for_project",
]
