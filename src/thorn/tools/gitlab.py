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

Requires ``python-gitlab`` (install via ``uv pip install 'thorn-agent[gitlab]'``).
The module gracefully defers the import error until a tool is actually
called, so importing ``thorn.tools.gitlab`` never fails.
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, Field

from thorn.tools._credential_scopes import (
    BroadCredentialScopeWarning,
    CredentialScopeInspection,
    CredentialScopeWarning,
    MissingCredentialScopeWarning,
)

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
            "Install it with: uv pip install 'thorn-agent[gitlab]'"
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


GitLabProjectRef = int | str
"""GitLab project reference accepted by Thorn's GitLab client.

Numeric IDs are the most reliable native API key on self-hosted
GitLab.  Human-readable ``path_with_namespace`` strings are still the
configuration-facing shape; the client resolves them to numeric IDs
when an instance rejects path-based project lookups.
"""

_GITLAB_API_ACCESS_SCOPES = frozenset({"api", "read_api"})
_GITLAB_REPOSITORY_WRITE_SCOPES = frozenset({"api", "write_repository"})
_GITLAB_HIGH_RISK_SCOPES = frozenset({
    "admin_mode",
    "create_runner",
    "k8s_proxy",
    "manage_runner",
    "sudo",
    "write_registry",
})


class GitLabProjectLookupError(RuntimeError):
    """Raised when a path-style GitLab project ref cannot be resolved."""


class GitLabProjectResolver:
    """Resolve GitLab project refs against a python-gitlab projects manager."""

    def __init__(self, projects_manager: Any) -> None:
        self._projects_manager = projects_manager
        self._project_path_id_cache: dict[str, int] = {}

    def get_project(self, project_ref: GitLabProjectRef) -> Any:
        """Return a python-gitlab project, resolving path refs if needed."""
        direct_ref = _direct_project_ref(project_ref)
        if isinstance(direct_ref, int):
            return self._projects_manager.get(direct_ref)

        cached_id = self._project_path_id_cache.get(direct_ref)
        if cached_id is not None:
            return self._projects_manager.get(cached_id)

        try:
            project = self._projects_manager.get(direct_ref)
        except Exception as exc:
            if not _is_gitlab_not_found(exc):
                raise
            resolved_id = self.resolve_project_id(direct_ref)
            return self._projects_manager.get(resolved_id)

        project_id = getattr(project, "id", None)
        if isinstance(project_id, int):
            self._project_path_id_cache[direct_ref] = project_id
        return project

    def resolve_project_id(self, project_path: str) -> int:
        """Resolve ``path_with_namespace`` to GitLab's numeric project ID."""
        normalized_path = _normalize_gitlab_project_path(project_path)
        cached_id = self._project_path_id_cache.get(normalized_path)
        if cached_id is not None:
            return cached_id

        search_term = normalized_path.rsplit("/", 1)[-1]
        projects = self._projects_manager.list(
            search=search_term,
            simple=True,
            iterator=True,
        )
        for project in projects:
            candidate_path = str(
                getattr(project, "path_with_namespace", "") or "",
            ).strip("/")
            if candidate_path != normalized_path:
                continue
            project_id = int(getattr(project, "id"))
            self._project_path_id_cache[normalized_path] = project_id
            return project_id

        raise GitLabProjectLookupError(
            f"GitLab project {normalized_path!r} could not be resolved "
            "to a numeric project ID via project search.  If this is "
            "a self-hosted GitLab instance that rejects path-based API "
            "project lookups, set this fork's `native_id` in gateway.json "
            "to the numeric GitLab project ID."
        )


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
        self._project_resolver = GitLabProjectResolver(self._gl.projects)

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

    def inspect_credential_scopes(self) -> CredentialScopeInspection:
        """Inspect observable GitLab token scopes for preflight warnings.

        GitLab's ``personal_access_tokens/self`` endpoint exposes the
        authenticating token's scopes on modern instances.  Callers treat
        failures from this method as advisory, because older or restricted
        instances may not expose the endpoint even when normal project
        API access works.
        """
        raw_token = self._gl.http_get("/personal_access_tokens/self")
        if not isinstance(raw_token, dict):
            return CredentialScopeInspection()
        raw_scopes = raw_token.get("scopes", [])
        if not isinstance(raw_scopes, list):
            return CredentialScopeInspection()
        observed_scopes = tuple(sorted(
            str(scope).strip()
            for scope in raw_scopes
            if str(scope).strip()
        ))
        return CredentialScopeInspection(
            observed_scopes=observed_scopes,
            warnings=_gitlab_scope_warnings(observed_scopes),
        )

    def get_user_by_username(self, username: str) -> dict[str, Any]:
        """Resolve a GitLab username to immutable account identity fields."""
        users = self._gl.users.list(username=username)
        for user in users:
            if str(getattr(user, "username", "")) != username:
                continue
            return {
                "id": getattr(user, "id", None),
                "username": getattr(user, "username", ""),
                "name": getattr(user, "name", "") or "",
                "web_url": getattr(user, "web_url", "") or "",
            }
        raise LookupError(f"GitLab user {username!r} could not be resolved")

    def get_issue(
        self, project_id: GitLabProjectRef, issue_iid: int,
    ) -> dict[str, Any]:
        """Fetch a single issue and return its key fields as a dict."""
        project = self._get_project(project_id)
        issue: ProjectIssue = project.issues.get(issue_iid)  # type: ignore[assignment]
        author = getattr(issue, "author", None) or None
        return {
            "iid": issue.iid,
            "title": issue.title,
            "state": issue.state,
            "description": issue.description or "",
            "labels": list(issue.labels),
            "assignees": [a["username"] for a in issue.assignees],
            "web_url": issue.web_url,
            # Author surfaced as the raw GitLab user dict so the
            # forge tool wrapper can pull both the immutable id
            # (`id`) and textual handle (`username`).
            "author": author,
            "created_at": getattr(issue, "created_at", None),
        }

    def create_issue(
        self,
        project_id: GitLabProjectRef,
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
        project = self._get_project(project_id)
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
        log.info("Created issue #%d in project %s", issue.iid, project_id)
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
        project_id: GitLabProjectRef,
        state: str = "opened",
        labels: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """List issues in a project filtered by *state* and optionally *labels*."""
        project = self._get_project(project_id)
        kwargs: dict[str, Any] = {
            "state": state,
            "order_by": "created_at",
            "sort": "desc",
        }
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
                "author_user": issue.author or None,
            }
            for issue in issues
        ]

    def update_issue(
        self,
        project_id: GitLabProjectRef,
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
        project = self._get_project(project_id)
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
        author = getattr(issue, "author", None) or None
        log.info("Updated issue #%d in project %s", issue_iid, project_id)
        return {
            "iid": issue.iid,
            "title": issue.title,
            "state": issue.state,
            "description": issue.description or "",
            "labels": list(issue.labels),
            "assignees": [a["username"] for a in issue.assignees],
            "web_url": issue.web_url,
            "author": author,
            "created_at": getattr(issue, "created_at", None),
        }

    def post_note(
        self,
        project_id: GitLabProjectRef,
        noteable_type: str,
        noteable_iid: int,
        body: str,
    ) -> None:
        """Post a note/comment on an issue or merge request."""
        project = self._get_project(project_id)
        if noteable_type == "Issue":
            noteable = project.issues.get(noteable_iid)
        elif noteable_type == "MergeRequest":
            noteable = project.mergerequests.get(noteable_iid)
        else:
            raise ValueError(f"Unsupported noteable_type: {noteable_type!r}")
        noteable.notes.create({"body": body})
        log.info(
            "Posted note on %s !%d in project %s",
            noteable_type, noteable_iid, project_id,
        )

    def create_merge_request(
        self,
        project_id: GitLabProjectRef,
        source_branch: str,
        title: str,
        target_branch: str = "main",
        description: str = "",
    ) -> dict[str, Any]:
        """Open a merge request and return its key fields as a dict."""
        project = self._get_project(project_id)
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

    def get_merge_request(
        self, project_id: GitLabProjectRef, mr_iid: int,
    ) -> dict[str, Any]:
        """Fetch a single merge request and return its key fields."""
        project = self._get_project(project_id)
        mr: ProjectMergeRequest = project.mergerequests.get(mr_iid)  # type: ignore[assignment]
        author = getattr(mr, "author", None) or None
        return {
            "iid": mr.iid,
            "title": mr.title,
            "state": mr.state,
            "description": mr.description or "",
            "web_url": mr.web_url,
            "source_branch": mr.source_branch,
            "target_branch": mr.target_branch,
            "merge_status": mr.merge_status,
            "author": author,
            "created_at": getattr(mr, "created_at", None),
        }

    def list_merge_requests(
        self,
        project_id: GitLabProjectRef,
        state: str = "opened",
    ) -> list[dict[str, Any]]:
        """List merge requests in a project filtered by *state*."""
        project = self._get_project(project_id)
        mrs = project.mergerequests.list(state=state, iterator=True)
        return [
            {
                "iid": mr.iid,
                "title": mr.title,
                "state": mr.state,
                "web_url": mr.web_url,
                "source_branch": mr.source_branch,
                "author": mr.author["username"] if mr.author else None,
                "author_user": mr.author or None,
            }
            for mr in mrs
        ]

    def get_project_info(self, project_id: GitLabProjectRef) -> dict[str, Any]:
        """Fetch project metadata: name, clone URL, default branch, etc."""
        project = self._get_project(project_id)
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
        project_id: GitLabProjectRef,
        file_path: str,
        ref: str = "HEAD",
    ) -> dict[str, Any]:
        """Read a file from a repository via the GitLab API."""
        project = self._get_project(project_id)
        f = project.files.get(file_path=file_path, ref=ref)
        return {
            "file_path": f.file_path,
            "ref": ref,
            "content": f.decode().decode("utf-8", errors="replace"),
        }

    def list_notes(
        self,
        project_id: GitLabProjectRef,
        noteable_type: str,
        noteable_iid: int,
    ) -> list[dict[str, Any]]:
        """List notes (comments) on an issue or merge request.

        Returns all notes in chronological order.  Each dict contains
        ``id``, ``author`` (username string), ``body``, ``created_at``,
        and ``system`` (bool -- ``True`` for auto-generated notes like
        label changes).
        """
        project = self._get_project(project_id)
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
                # Keep ``author`` as the username string for backwards
                # compatibility with existing call sites that render
                # it directly; surface the raw author dict separately
                # under ``author_user`` so the forge tool wrapper has
                # access to the immutable id and bot flag.
                "author": note.author["username"] if note.author else "unknown",
                "author_user": note.author or None,
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

    def _get_project(self, project_ref: GitLabProjectRef) -> Any:
        """Return a python-gitlab project, resolving path refs if needed."""
        return self._project_resolver.get_project(project_ref)

    def resolve_project_id(self, project_path: str) -> int:
        """Resolve ``path_with_namespace`` to GitLab's numeric project ID."""
        return self._project_resolver.resolve_project_id(project_path)


def _direct_project_ref(project_ref: GitLabProjectRef) -> GitLabProjectRef:
    if isinstance(project_ref, int):
        return project_ref
    stripped_ref = project_ref.strip()
    if stripped_ref.isdigit():
        return int(stripped_ref)
    return _normalize_gitlab_project_path(stripped_ref)


def _normalize_gitlab_project_path(project_path: str) -> str:
    normalized_path = project_path.strip().strip("/")
    if normalized_path.endswith(".git"):
        normalized_path = normalized_path[: -len(".git")]
    if not normalized_path or "/" not in normalized_path:
        raise GitLabProjectLookupError(
            f"GitLab project reference {project_path!r} is not a numeric "
            "ID and does not look like a path_with_namespace value.  "
            "Use a human project path such as 'group/project' or set "
            "`native_id` to the numeric GitLab project ID."
        )
    return normalized_path


def _is_gitlab_not_found(exc: BaseException) -> bool:
    return str(getattr(exc, "response_code", "")) == "404"


def _gitlab_scope_warnings(
    observed_scopes: tuple[str, ...],
) -> tuple[CredentialScopeWarning, ...]:
    scopes = frozenset(observed_scopes)
    warnings: list[CredentialScopeWarning] = []

    if "api" in scopes:
        warnings.append(BroadCredentialScopeWarning(
            summary="GitLab token advertises the broad 'api' scope.",
            detail=(
                "Use this only when write-side GitLab API operations are "
                "needed, and prefer a dedicated project/group/service token "
                "limited to the repositories this agency manages."
            ),
        ))

    for high_risk_scope in sorted(scopes & _GITLAB_HIGH_RISK_SCOPES):
        warnings.append(BroadCredentialScopeWarning(
            summary=(
                "GitLab token advertises high-risk scope "
                f"{high_risk_scope!r}."
            ),
            detail=(
                "Unattended agents should not be able to use admin mode, "
                "sudo, manage runners, access Kubernetes proxies, or write "
                "unrelated package/registry resources."
            ),
        ))

    if scopes.isdisjoint(_GITLAB_API_ACCESS_SCOPES):
        warnings.append(MissingCredentialScopeWarning(
            summary="GitLab token does not advertise API access.",
            detail=(
                "TODO polling, project reads, comments, issues, and merge "
                "request operations may fail without API-equivalent access."
            ),
        ))

    if scopes.isdisjoint(_GITLAB_REPOSITORY_WRITE_SCOPES):
        warnings.append(MissingCredentialScopeWarning(
            summary="GitLab token does not advertise repository write access.",
            detail=(
                "Branch pushes may fail unless git HTTPS uses another "
                "credential with write_repository-equivalent access. Run "
                "`thorn serve preflight --write-check` to verify push/delete."
            ),
        ))

    return tuple(warnings)


__all__ = [
    "GitLabClient",
    "GitLabConfig",
    "GitLabProjectResolver",
    "GitLabProjectLookupError",
    "GitLabProjectRef",
]
