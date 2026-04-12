"""GitHub Notifications polling event source.

Polls the GitHub Notifications API for new notifications (e.g.
@-mentions, assignments, review requests) and converts them into
:class:`~thorn.gateway._event.IncomingEvent` objects for the gateway.

Requires ``PyGithub`` (install via ``pip install thorn[github]``).
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Awaitable, Callable
from typing import Any

import httpx
from pydantic import BaseModel, Field

from thorn.gateway._event import EventSource, IncomingEvent
from thorn.runtime._session import SessionKey

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional dependency guard (same pattern as thorn.tools.github)
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
    if not _HAS_GITHUB:
        raise ImportError(
            "PyGithub is required for the GitHub event source. "
            "Install it with: pip install thorn[github]"
        )


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_DEFAULT_BASE_URL = "https://api.github.com"


class GitHubSourceConfig(BaseModel):
    """Configuration for the GitHub Notifications event source.

    Typically loaded from ``gateway.json`` via
    :func:`~thorn.gateway._config.instantiate_sources`, or from
    environment variables via :meth:`from_env`.
    """

    token: str = Field(
        description="Personal Access Token with 'notifications' and 'repo' scopes",
    )
    base_url: str = Field(
        default=_DEFAULT_BASE_URL,
        description="GitHub API base URL (override for GitHub Enterprise)",
    )
    repository: str = Field(
        description="Repository in owner/repo format to filter notifications",
    )
    app_slug: str = Field(
        default="",
        description=(
            "GitHub App slug (e.g. 'my-thorn-app'), communicated to "
            "the agent so it knows its own @-mention identity"
        ),
    )
    poll_interval: int = Field(
        default=30,
        ge=5,
        description="Seconds between polling cycles",
    )

    @classmethod
    def from_env(cls) -> GitHubSourceConfig:
        """Load configuration from environment variables.

        - ``GITHUB_TOKEN`` -- Personal Access Token (required)
        - ``GITHUB_URL`` -- API base URL (default ``https://api.github.com``)
        - ``THORN_GITHUB_REPOSITORY`` -- repository in ``owner/repo``
          format (required)
        - ``THORN_GITHUB_APP_SLUG`` -- GitHub App slug (default empty)
        - ``THORN_POLL_INTERVAL`` -- seconds between polls (default ``30``)
        """
        token = os.environ.get("GITHUB_TOKEN")
        repository = os.environ.get("THORN_GITHUB_REPOSITORY")
        missing = [
            name
            for name, val in [
                ("GITHUB_TOKEN", token),
                ("THORN_GITHUB_REPOSITORY", repository),
            ]
            if not val
        ]
        if missing:
            raise ValueError(
                f"Missing required environment variable(s): "
                f"{', '.join(missing)}. "
                "Set GITHUB_TOKEN and THORN_GITHUB_REPOSITORY "
                "to use the GitHub event source."
            )
        return cls(
            token=token,  # type: ignore[arg-type]
            base_url=os.environ.get("GITHUB_URL", _DEFAULT_BASE_URL),
            repository=repository,  # type: ignore[arg-type]
            app_slug=os.environ.get("THORN_GITHUB_APP_SLUG", ""),
            poll_interval=int(os.environ.get("THORN_POLL_INTERVAL", "30")),
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_subject_number(subject_url: str) -> int | None:
    """Extract the issue/PR number from a notification subject URL.

    URLs look like ``https://api.github.com/repos/owner/repo/issues/42``
    or ``https://api.github.com/repos/owner/repo/pulls/7``.
    Returns ``None`` if the trailing segment is not numeric.
    """
    segment = subject_url.rstrip("/").rsplit("/", 1)[-1]
    if segment.isdigit():
        return int(segment)
    return None


def _fetch_body(url: str, token: str) -> str | None:
    """Fetch the ``body`` field from a GitHub API URL.

    Used to retrieve comment or issue/PR bodies from
    ``latest_comment_url`` or ``subject.url``.  Returns ``None``
    on any failure so callers can fall back gracefully.
    """
    try:
        response = httpx.get(
            url,
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github+json",
            },
            timeout=15.0,
        )
        response.raise_for_status()
        data = response.json()
        return data.get("body") or None
    except Exception:
        log.debug("Failed to fetch body from %s", url, exc_info=True)
        return None


# ---------------------------------------------------------------------------
# Event formatting
# ---------------------------------------------------------------------------


def _make_session_key(notification: Any) -> SessionKey:
    """Derive a filesystem-safe session key from a GitHub notification.

    Format: ``github_{repo_id}_{subject_type}_{number}``

    Falls back to the thread ID when the subject URL does not contain
    a numeric issue/PR number.
    """
    repo_id = notification.repository.id
    subject_type = notification.subject.type
    subject_url = notification.subject.url or ""
    number = _extract_subject_number(subject_url)
    if number is None:
        return SessionKey(
            f"github_{repo_id}_{subject_type}_{notification.id}"
        )
    return SessionKey(f"github_{repo_id}_{subject_type}_{number}")


def _format_event_content(
    notification: Any,
    comment_body: str | None,
) -> str:
    """Build a human-readable prompt from a GitHub notification."""
    repo = notification.repository
    repo_id = repo.id
    full_name = repo.full_name
    clone_url = getattr(repo, "clone_url", "")
    default_branch = getattr(repo, "default_branch", "main")
    html_url = getattr(repo, "html_url", "")
    subject_type = notification.subject.type
    subject_url = notification.subject.url or ""
    number = _extract_subject_number(subject_url)
    reason = notification.reason
    thread_id = str(notification.id)

    target_label = (
        f"{subject_type} #{number}"
        if number is not None
        else f"{subject_type} (thread {thread_id})"
    )

    lines = [
        f"GitHub notification: you were {reason} on "
        f"{target_label} in repository {full_name} "
        f"(repo_id={repo_id}).",
        "",
        f"Thread ID: {thread_id}",
        f"Reason: {reason}",
        f"Target: {target_label}",
    ]
    if clone_url:
        lines.append(f"Clone URL: {clone_url}")
    if default_branch:
        lines.append(f"Default branch: {default_branch}")
    if html_url:
        lines.append(f"Repository URL: {html_url}")
    lines.append("")

    if comment_body:
        lines.append("Comment body:")
        lines.append(comment_body)
        lines.append("")

    lines.append(
        "Respond to the notification as appropriate, then mark the "
        "notification as read using github_mark_notification_read."
    )
    return "\n".join(lines)


def _make_event(
    notification: Any,
    comment_body: str | None,
) -> IncomingEvent:
    """Convert a GitHub notification into an ``IncomingEvent``."""
    repo = notification.repository
    subject_url = notification.subject.url or ""
    number = _extract_subject_number(subject_url)
    return IncomingEvent(
        source="github",
        session_key=_make_session_key(notification),
        content=_format_event_content(notification, comment_body),
        metadata={
            "thread_id": str(notification.id),
            "repo_full_name": repo.full_name,
            "repo_id": repo.id,
            "subject_type": notification.subject.type,
            "subject_number": number,
            "reason": notification.reason,
            "clone_url": getattr(repo, "clone_url", ""),
            "default_branch": getattr(repo, "default_branch", "main"),
            "html_url": getattr(repo, "html_url", ""),
        },
    )


# ---------------------------------------------------------------------------
# EventSource implementation
# ---------------------------------------------------------------------------


class GitHubNotificationsSource(EventSource):
    """Polls the GitHub Notifications API and emits events for new items."""

    Config = GitHubSourceConfig

    def __init__(self, config: GitHubSourceConfig, *, service_name: str = "") -> None:
        _require_github()
        self._config = config
        self._service_name = service_name
        self._gh = _Github(  # type: ignore[misc]
            base_url=config.base_url,
            auth=_GHAuth.Token(config.token),  # type: ignore[union-attr]
        )
        self._seen: set[str] = set()
        self._stop_event: asyncio.Event | None = None

    @property
    def name(self) -> str:
        return self._service_name

    async def start(
        self,
        on_event: Callable[[IncomingEvent], Awaitable[None]],
    ) -> None:
        self._stop_event = asyncio.Event()

        user_info = await asyncio.to_thread(self._check_connection)
        log.info(
            "GitHub source authenticated as %s (%s)",
            user_info["login"], user_info["name"],
        )
        log.info(
            "Polling GitHub notifications every %ds (repo=%s)",
            self._config.poll_interval,
            self._config.repository,
        )

        while not self._stop_event.is_set():
            try:
                await self._poll_once(on_event)
            except Exception:
                log.exception("GitHub poll cycle failed")

            try:
                await asyncio.wait_for(
                    self._stop_event.wait(),
                    timeout=self._config.poll_interval,
                )
                break
            except asyncio.TimeoutError:
                pass

    async def stop(self) -> None:
        if self._stop_event is not None:
            self._stop_event.set()

    def _check_connection(self) -> dict[str, Any]:
        user = self._gh.get_user()
        return {
            "login": user.login,
            "name": user.name,
            "html_url": user.html_url,
        }

    async def _poll_once(
        self,
        on_event: Callable[[IncomingEvent], Awaitable[None]],
    ) -> None:
        notifications = await asyncio.to_thread(self._get_notifications)
        new_notifications = [
            n for n in notifications if str(n.id) not in self._seen
        ]
        if new_notifications:
            log.info(
                "Found %d new notification(s) out of %d unread",
                len(new_notifications), len(notifications),
            )
        for notification in new_notifications:
            self._seen.add(str(notification.id))
            comment_body = await asyncio.to_thread(
                self._fetch_comment_body, notification,
            )
            event = _make_event(notification, comment_body)
            await on_event(event)

    def _get_notifications(self) -> list[Any]:
        """Fetch unread, participating notifications filtered by repository."""
        all_notifications = list(
            self._gh.get_user().get_notifications(participating=True),
        )
        if self._config.repository:
            return [
                n for n in all_notifications
                if n.repository.full_name == self._config.repository
            ]
        return all_notifications

    def _fetch_comment_body(self, notification: Any) -> str | None:
        """Retrieve the body text associated with a notification.

        Tries ``latest_comment_url`` first, then falls back to
        ``subject.url``.
        """
        latest_url = getattr(
            notification.subject, "latest_comment_url", None,
        )
        if latest_url:
            body = _fetch_body(latest_url, self._config.token)
            if body:
                return body

        subject_url = getattr(notification.subject, "url", None)
        if subject_url:
            body = _fetch_body(subject_url, self._config.token)
            if body:
                return body

        return None


__all__ = [
    "GitHubSourceConfig",
    "GitHubNotificationsSource",
]
