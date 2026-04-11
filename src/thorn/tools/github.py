"""GitHub API operations as ``@tool``-decorated functions for thorn agents.

Provides tools for interacting with a GitHub instance: reading issues,
posting comments, creating pull requests, and querying PR details.

Requires ``PyGithub`` (install via ``pip install thorn[github]``).
The module gracefully defers the import error until a tool is actually
called, so importing ``thorn.tools.github`` never fails.

Usage::

    from thorn.tools import github

    agent = Agent(..., tools=[github.GITHUB_TOOLS, ...])

Configuration is loaded from environment variables:

- ``GITHUB_TOKEN`` -- Personal Access Token or GitHub App token
- ``GITHUB_URL``   -- API base URL (default ``https://api.github.com``;
  override for GitHub Enterprise)
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Literal

import httpx
from pydantic import BaseModel, Field

from thorn.core._func import tool

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


# ---------------------------------------------------------------------------
# Configuration and client
# ---------------------------------------------------------------------------

_DEFAULT_BASE_URL = "https://api.github.com"


class GitHubConfig(BaseModel):
    """Configuration for connecting to a GitHub instance.

    Typically loaded from environment variables via
    ``GitHubConfig.from_env()``.
    """

    base_url: str = Field(
        default=_DEFAULT_BASE_URL,
        description="GitHub API base URL (override for GitHub Enterprise)",
    )
    token: str = Field(description="Personal Access Token or GitHub App token")

    @classmethod
    def from_env(cls) -> GitHubConfig:
        """Load configuration from environment variables.

        - ``GITHUB_TOKEN`` (required)
        - ``GITHUB_URL`` (optional; defaults to ``https://api.github.com``)

        Raises ``ValueError`` if ``GITHUB_TOKEN`` is not set.
        """
        token = os.environ.get("GITHUB_TOKEN")
        if not token:
            raise ValueError(
                "Missing required environment variable: GITHUB_TOKEN. "
                "Set GITHUB_TOKEN to use GitHub tools."
            )
        base_url = os.environ.get("GITHUB_URL", _DEFAULT_BASE_URL)
        return cls(base_url=base_url, token=token)


class GitHubClient:
    """High-level GitHub client wrapping ``PyGithub``.

    Provides purpose-built methods so that tool functions never touch
    ``PyGithub`` objects directly.
    """

    def __init__(self, config: GitHubConfig) -> None:
        _require_github()
        auth = _GHAuth.Token(config.token)
        self._gh = _Github(base_url=config.base_url, auth=auth)
        self._base_url = config.base_url
        self._token = config.token

    def check_connection(self) -> dict[str, Any]:
        """Authenticate and return info about the current user.

        Raises on bad tokens.
        """
        user = self._gh.get_user()
        return {
            "login": user.login,
            "name": user.name,
            "html_url": user.html_url,
        }

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
                "Authorization": f"Bearer {self._token}",
                "Accept": "application/vnd.github+json",
            },
        )
        response.raise_for_status()


# ---------------------------------------------------------------------------
# Module-level client accessor
# ---------------------------------------------------------------------------

_client: GitHubClient | None = None


def get_client() -> GitHubClient:
    """Return the module-level ``GitHubClient``, creating it lazily.

    Configuration is loaded from environment variables on first access.
    Subsequent calls return the same client instance.
    """
    global _client
    if _client is None:
        config = GitHubConfig.from_env()
        _client = GitHubClient(config)
    return _client


def set_client(client: GitHubClient | None) -> None:
    """Replace the module-level client (useful for testing or custom configs)."""
    global _client
    _client = client


# ---------------------------------------------------------------------------
# @tool functions
# ---------------------------------------------------------------------------

CommentableKind = Literal["Issue", "PullRequest"]


@tool
async def github_read_issue(repo: str, issue_number: int) -> str:
    """Read a GitHub issue, returning its title, body, labels, and assignees.

    *repo* is the repository in ``owner/repo`` format (e.g.
    ``"octocat/hello-world"``).  *issue_number* is the issue number
    (the ``#N`` in the UI).
    """
    client = get_client()
    info = await asyncio.to_thread(client.get_issue, repo, issue_number)
    lines = [
        f"Issue #{info['number']}: {info['title']}",
        f"State: {info['state']}",
        f"Labels: {', '.join(info['labels']) or '(none)'}",
        f"Assignees: {', '.join(info['assignees']) or '(none)'}",
        f"URL: {info['html_url']}",
        "",
        info["body"] or "(no description)",
    ]
    return "\n".join(lines)


@tool
async def github_post_comment(
    repo: str,
    commentable_type: CommentableKind,
    commentable_number: int,
    body: str,
) -> str:
    """Post a comment on a GitHub issue or pull request.

    *repo* is the repository in ``owner/repo`` format.
    *commentable_type* must be ``"Issue"`` or ``"PullRequest"``
    (GitHub's API unifies these, but the type clarifies intent).
    *commentable_number* is the issue or PR number.
    """
    client = get_client()
    await asyncio.to_thread(
        client.post_comment, repo, commentable_number, body,
    )
    return (
        f"Posted comment on {commentable_type} "
        f"#{commentable_number} in {repo}."
    )


@tool
async def github_create_pull_request(
    repo: str,
    head: str,
    title: str,
    body: str = "",
    base: str = "main",
) -> str:
    """Create a new pull request on GitHub.

    Opens a PR from *head* branch into *base* branch (default
    ``main``) in the specified repository.  *repo* is in
    ``owner/repo`` format.
    """
    client = get_client()
    info = await asyncio.to_thread(
        client.create_pull_request,
        repo=repo,
        head=head,
        title=title,
        base=base,
        body=body,
    )
    return (
        f"Created PR #{info['number']}: {info['title']}\n"
        f"  {info['head']} -> {info['base']}\n"
        f"  URL: {info['html_url']}"
    )


@tool
async def github_get_pull_request(repo: str, pr_number: int) -> str:
    """Read details of a GitHub pull request.

    Returns the PR title, state, branches, mergeable status, and body.
    *repo* is in ``owner/repo`` format.
    """
    client = get_client()
    info = await asyncio.to_thread(
        client.get_pull_request, repo, pr_number,
    )
    merged_indicator = " (merged)" if info["merged"] else ""
    lines = [
        f"PR #{info['number']}: {info['title']}",
        f"State: {info['state']}{merged_indicator}",
        f"Branches: {info['head']} -> {info['base']}",
        f"Mergeable: {info['mergeable']} ({info['mergeable_state']})",
        f"URL: {info['html_url']}",
        "",
        info["body"] or "(no description)",
    ]
    return "\n".join(lines)


@tool
async def github_list_pull_requests(
    repo: str,
    state: Literal["open", "closed", "all"] = "open",
) -> str:
    """List pull requests in a GitHub repository.

    Filters by *state* (default ``"open"``).  Returns a formatted
    list of PR numbers, titles, and authors.  *repo* is in
    ``owner/repo`` format.
    """
    client = get_client()
    prs = await asyncio.to_thread(
        client.list_pull_requests, repo, state,
    )
    if not prs:
        return f"No {state} pull requests in {repo}."
    lines = []
    for pr in prs:
        author = pr["author"] or "unknown"
        lines.append(
            f"  #{pr['number']}: {pr['title']} ({pr['state']}, by {author})"
        )
    header = f"{len(prs)} {state} pull request(s) in {repo}:"
    return "\n".join([header, *lines])


@tool
async def github_list_comments(
    repo: str,
    commentable_type: CommentableKind,
    commentable_number: int,
    include_bot_comments: bool = False,
) -> str:
    """List comments on a GitHub issue or pull request.

    Returns human-authored comments in chronological order.  Useful
    for reading reviewer feedback, discussion threads, and prior
    comments.

    *repo* is in ``owner/repo`` format.
    *commentable_type* must be ``"Issue"`` or ``"PullRequest"``.
    Set *include_bot_comments* to ``True`` to also show comments
    from bot accounts.
    """
    client = get_client()
    comments = await asyncio.to_thread(
        client.list_comments, repo, commentable_number,
    )
    if not include_bot_comments:
        comments = [c for c in comments if not c["is_bot"]]

    if not comments:
        kind = "issue" if commentable_type == "Issue" else "PR"
        return f"No comments on {kind} #{commentable_number} in {repo}."

    lines: list[str] = []
    for comment in comments:
        lines.append(f"[{comment['author']}] ({comment['created_at']}):")
        lines.append(comment["body"])
        lines.append("")
    return "\n".join(lines)


@tool
async def github_get_repo_info(repo: str) -> str:
    """Get information about a GitHub repository.

    Returns the repository's full name, clone URLs, default branch,
    and description.  *repo* is in ``owner/repo`` format.
    """
    client = get_client()
    info = await asyncio.to_thread(client.get_repo_info, repo)
    lines = [
        f"Repository: {info['full_name']}",
        f"Clone URL (HTTPS): {info['clone_url']}",
        f"Clone URL (SSH): {info['ssh_url']}",
        f"Default branch: {info['default_branch']}",
        f"Web URL: {info['html_url']}",
    ]
    if info["description"]:
        lines.append(f"Description: {info['description']}")
    return "\n".join(lines)


@tool
async def github_read_file(
    repo: str,
    file_path: str,
    ref: str = "HEAD",
) -> str:
    """Read a file from a GitHub repository via the API.

    Useful for inspecting files without cloning the entire repository.
    *repo* is in ``owner/repo`` format.  *ref* can be a branch name,
    tag, or commit SHA.
    """
    client = get_client()
    info = await asyncio.to_thread(client.read_file, repo, file_path, ref)
    return f"--- {info['file_path']} (ref: {info['ref']}) ---\n{info['content']}"


@tool
async def github_mark_notification_read(thread_id: str) -> str:
    """Mark a GitHub notification thread as read.

    *thread_id* is the notification thread ID (a numeric string from
    the GitHub Notifications API).  This marks the notification as
    read so it no longer appears in your unread notifications.
    """
    client = get_client()
    await asyncio.to_thread(client.mark_notification_read, thread_id)
    return f"Marked GitHub notification thread {thread_id} as read."


GITHUB_TOOLS: list[object] = [
    github_read_issue,
    github_post_comment,
    github_create_pull_request,
    github_get_pull_request,
    github_list_pull_requests,
    github_list_comments,
    github_get_repo_info,
    github_read_file,
    github_mark_notification_read,
]
"""All GitHub tools as a list, suitable for use in ``tools=[GITHUB_TOOLS, ...]``."""

__all__ = [
    "GitHubConfig",
    "GitHubClient",
    "get_client",
    "set_client",
    "CommentableKind",
    "github_read_issue",
    "github_post_comment",
    "github_create_pull_request",
    "github_get_pull_request",
    "github_list_pull_requests",
    "github_list_comments",
    "github_get_repo_info",
    "github_read_file",
    "github_mark_notification_read",
    "GITHUB_TOOLS",
]
