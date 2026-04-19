"""GitHub API client used by forge services and the notifications source.

Provides :class:`GitHubClient`, a thin wrapper around ``PyGithub`` plus a
few raw REST calls (e.g. notification thread mark-as-read/done) that
PyGithub does not expose.

This module deliberately does **not** define agent-facing ``@tool``
functions of its own.  Agent-facing forge operations live in
:mod:`thorn.tools.forge` as project-name-based tools (``forge_*``) that
resolve credentials from the current agent's
:class:`~thorn.core._account.ForgeAccountConfig` and the forge service
registered in the runtime.  Forge URL and auth therefore come from
``.thorn/gateway.json`` and the agent's identity JSON, never from
process-wide environment variable singletons.

Requires ``PyGithub`` (install via ``pip install thorn[github]``).
The module gracefully defers the import error until a tool is actually
called, so importing ``thorn.tools.github`` never fails.
"""

from __future__ import annotations

import logging
from typing import Any, Literal

import httpx

from thorn.tools._github_connection import (
    GitHubAppAuth,
    GitHubConnectionConfig,
    GitHubPatAuth,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional dependency guard
# ---------------------------------------------------------------------------

try:
    from github import Auth as _GHAuth
    from github import Github as _Github

    _HAS_GITHUB = True
except ImportError:
    _GHAuth = None  # type: ignore[assignment,misc]
    _Github = None  # type: ignore[assignment,misc]
    _HAS_GITHUB = False


def _require_github() -> None:
    """Raise a clear error if PyGithub is not installed."""
    if not _HAS_GITHUB:
        raise ImportError(
            "PyGithub is required for GitHub tools. "
            "Install it with: pip install thorn[github]"
        )


def build_pygithub_auth(auth: GitHubPatAuth | GitHubAppAuth) -> Any:
    """Build a PyGithub ``Auth`` object (PAT or app installation)."""
    _require_github()
    assert _GHAuth is not None
    if auth.kind == "pat":
        return _GHAuth.Token(auth.token)
    app = _GHAuth.AppAuth(auth.app_id, auth.private_key_pem)
    return app.get_installation_auth(auth.installation_id)


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


CommentableKind = Literal["Issue", "PullRequest"]


class GitHubClient:
    """High-level GitHub client wrapping ``PyGithub``.

    Provides purpose-built methods so that tool functions never touch
    ``PyGithub`` objects directly.
    """

    def __init__(self, config: GitHubConnectionConfig) -> None:
        _require_github()
        assert _GHAuth is not None
        assert _Github is not None
        self._auth = build_pygithub_auth(config.auth)
        self._gh = _Github(base_url=config.base_url, auth=self._auth)
        self._base_url = config.base_url
        self._github_auth_module = _GHAuth

    def bearer_token_for_http(self) -> str:
        """Return the current bearer token for REST calls outside PyGithub.

        For installation auth, PyGithub refreshes the installation token
        as needed; this reads the latest value.
        """
        auth = self._auth
        gh_auth = self._github_auth_module
        if isinstance(auth, gh_auth.Token):
            return auth.token
        if isinstance(auth, gh_auth.AppInstallationAuth):
            return auth.token
        raise TypeError(f"Unsupported auth type: {type(auth)!r}")

    def check_connection(self) -> dict[str, Any]:
        """Verify credentials and return a small identity dict for logging.

        Personal access tokens use ``GET /user``. GitHub App *installation*
        tokens cannot call that endpoint (403); we use ``GET /rate_limit``
        instead, which installation tokens may access.
        """
        gh_auth = self._github_auth_module
        auth = self._auth
        if isinstance(auth, gh_auth.Token):
            user = self._gh.get_user()
            return {
                "login": user.login,
                "name": user.name,
                "html_url": user.html_url,
            }
        if isinstance(auth, gh_auth.AppInstallationAuth):
            self._gh.get_rate_limit()
            return {
                "login": "(GitHub App installation)",
                "name": "",
                "html_url": "",
            }
        raise TypeError(f"Unsupported auth type: {type(auth)!r}")

    def get_issue(self, repo: str, issue_number: int) -> dict[str, Any]:
        """Fetch a single issue and return its key fields as a dict."""
        repository = self._gh.get_repo(repo)
        issue = repository.get_issue(number=issue_number)
        return {
            "number": issue.number,
            "title": issue.title,
            "state": issue.state,
            "body": issue.body or "",
            "labels": [label.name for label in issue.labels],
            "assignees": [a.login for a in issue.assignees],
            "html_url": issue.html_url,
        }

    def create_issue(
        self,
        repo: str,
        title: str,
        body: str = "",
        labels: list[str] | None = None,
        assignees: list[str] | None = None,
    ) -> dict[str, Any]:
        """Create an issue and return its key fields as a dict."""
        repository = self._gh.get_repo(repo)
        kwargs: dict[str, Any] = {"title": title, "body": body}
        if labels:
            kwargs["labels"] = labels
        if assignees:
            kwargs["assignees"] = assignees
        issue = repository.create_issue(**kwargs)
        log.info("Created issue #%d in %s", issue.number, repo)
        return {
            "number": issue.number,
            "title": issue.title,
            "state": issue.state,
            "body": issue.body or "",
            "labels": [label.name for label in issue.labels],
            "assignees": [a.login for a in issue.assignees],
            "html_url": issue.html_url,
        }

    def list_issues(
        self,
        repo: str,
        state: str = "open",
        labels: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """List issues in a repository filtered by *state* and optionally *labels*.

        Only returns actual issues (excludes pull requests, which GitHub
        co-mingles in the issues endpoint).
        """
        repository = self._gh.get_repo(repo)
        kwargs: dict[str, Any] = {"state": state, "sort": "created", "direction": "desc"}
        if labels:
            kwargs["labels"] = labels
        issues = repository.get_issues(**kwargs)
        return [
            {
                "number": issue.number,
                "title": issue.title,
                "state": issue.state,
                "html_url": issue.html_url,
                "labels": [label.name for label in issue.labels],
                "assignees": [a.login for a in issue.assignees],
                "author": issue.user.login if issue.user else None,
            }
            for issue in issues
            if issue.pull_request is None
        ]

    def update_issue(
        self,
        repo: str,
        issue_number: int,
        title: str | None = None,
        body: str | None = None,
        state: str | None = None,
        labels: list[str] | None = None,
        assignees: list[str] | None = None,
    ) -> dict[str, Any]:
        """Edit an issue's fields and return the updated issue as a dict.

        Only the fields that are not ``None`` are changed.  *labels* and
        *assignees* are **replacement** sets (not additive); callers that
        want add/remove semantics should merge before calling.
        """
        repository = self._gh.get_repo(repo)
        issue = repository.get_issue(number=issue_number)
        kwargs: dict[str, Any] = {}
        if title is not None:
            kwargs["title"] = title
        if body is not None:
            kwargs["body"] = body
        if state is not None:
            kwargs["state"] = state
        if labels is not None:
            kwargs["labels"] = labels
        if assignees is not None:
            kwargs["assignees"] = assignees
        issue.edit(**kwargs)
        log.info("Updated issue #%d in %s", issue_number, repo)
        return {
            "number": issue.number,
            "title": issue.title,
            "state": issue.state,
            "body": issue.body or "",
            "labels": [label.name for label in issue.labels],
            "assignees": [a.login for a in issue.assignees],
            "html_url": issue.html_url,
        }

    def post_comment(
        self,
        repo: str,
        commentable_number: int,
        body: str,
    ) -> None:
        """Post a comment on an issue or pull request.

        GitHub's API unifies issue and PR comments under the issues
        endpoint, so this works for both.
        """
        repository = self._gh.get_repo(repo)
        issue = repository.get_issue(number=commentable_number)
        issue.create_comment(body)
        log.info(
            "Posted comment on #%d in %s",
            commentable_number, repo,
        )

    def create_pull_request(
        self,
        repo: str,
        head: str,
        title: str,
        base: str = "main",
        body: str = "",
    ) -> dict[str, Any]:
        """Open a pull request and return its key fields as a dict."""
        repository = self._gh.get_repo(repo)
        pr = repository.create_pull(
            title=title,
            body=body,
            head=head,
            base=base,
        )
        return {
            "number": pr.number,
            "title": pr.title,
            "state": pr.state,
            "html_url": pr.html_url,
            "head": pr.head.ref,
            "base": pr.base.ref,
        }

    def get_pull_request(self, repo: str, pr_number: int) -> dict[str, Any]:
        """Fetch a single pull request and return its key fields."""
        repository = self._gh.get_repo(repo)
        pr = repository.get_pull(number=pr_number)
        return {
            "number": pr.number,
            "title": pr.title,
            "state": pr.state,
            "body": pr.body or "",
            "html_url": pr.html_url,
            "head": pr.head.ref,
            "base": pr.base.ref,
            "mergeable": pr.mergeable,
            "mergeable_state": pr.mergeable_state,
            "merged": pr.merged,
        }

    def list_pull_requests(
        self,
        repo: str,
        state: str = "open",
    ) -> list[dict[str, Any]]:
        """List pull requests in a repository filtered by *state*."""
        repository = self._gh.get_repo(repo)
        prs = repository.get_pulls(state=state, sort="created", direction="desc")
        return [
            {
                "number": pr.number,
                "title": pr.title,
                "state": pr.state,
                "html_url": pr.html_url,
                "head": pr.head.ref,
                "author": pr.user.login if pr.user else None,
            }
            for pr in prs
        ]

    def get_repo_info(self, repo: str) -> dict[str, Any]:
        """Fetch repository metadata: name, clone URL, default branch, etc."""
        repository = self._gh.get_repo(repo)
        return {
            "full_name": repository.full_name,
            "name": repository.name,
            "clone_url": repository.clone_url,
            "ssh_url": repository.ssh_url,
            "default_branch": repository.default_branch,
            "html_url": repository.html_url,
            "description": repository.description or "",
        }

    def read_file(
        self,
        repo: str,
        file_path: str,
        ref: str = "HEAD",
    ) -> dict[str, Any]:
        """Read a file from a repository via the GitHub Contents API."""
        repository = self._gh.get_repo(repo)
        contents = repository.get_contents(file_path, ref=ref)
        if isinstance(contents, list):
            raise ValueError(
                f"{file_path!r} is a directory, not a file. "
                "Use list_directory or similar to inspect directory contents."
            )
        return {
            "file_path": contents.path,
            "ref": ref,
            "content": contents.decoded_content.decode("utf-8", errors="replace"),
        }

    def list_comments(
        self,
        repo: str,
        commentable_number: int,
    ) -> list[dict[str, Any]]:
        """List comments on an issue or pull request.

        Returns comments in chronological order.  Each dict contains
        ``id``, ``author`` (login string), ``body``, ``created_at``,
        and ``is_bot`` (bool).
        """
        repository = self._gh.get_repo(repo)
        issue = repository.get_issue(number=commentable_number)
        raw_comments = issue.get_comments()
        return [
            {
                "id": comment.id,
                "author": comment.user.login if comment.user else "unknown",
                "body": comment.body,
                "created_at": comment.created_at.isoformat(),
                "is_bot": (
                    getattr(comment.user, "type", "") == "Bot"
                    if comment.user
                    else False
                ),
            }
            for comment in raw_comments
        ]

    def mark_notification_read(self, thread_id: str) -> None:
        """Mark a GitHub notification thread as read.

        Uses ``httpx`` directly because PyGithub does not expose a
        public method to fetch a single notification by thread ID.
        """
        response = httpx.patch(
            f"{self._base_url}/notifications/threads/{thread_id}",
            headers={
                "Authorization": f"Bearer {self.bearer_token_for_http()}",
                "Accept": "application/vnd.github+json",
            },
        )
        response.raise_for_status()

    def mark_notification_done(self, thread_id: str) -> None:
        """Mark a GitHub notification thread as done.

        "Done" removes the notification from the inbox entirely,
        equivalent to the "Done" button in the GitHub notifications UI.
        """
        response = httpx.delete(
            f"{self._base_url}/notifications/threads/{thread_id}",
            headers={
                "Authorization": f"Bearer {self.bearer_token_for_http()}",
                "Accept": "application/vnd.github+json",
            },
        )
        response.raise_for_status()


__all__ = [
    "CommentableKind",
    "GitHubClient",
    "GitHubConnectionConfig",
    "build_pygithub_auth",
]
