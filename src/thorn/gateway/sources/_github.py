"""GitHub notifications event source.

Polls ``GET /notifications`` (the user-scoped Notifications API) to
discover @-mentions, assignments, review requests, and other activity
directed at the authenticated bot user.

The Notifications API requires a personal access token (classic) with
the ``notifications`` scope.  GitHub App installation tokens cannot
access this API.

See `Notifications <https://docs.github.com/en/rest/activity/notifications>`_.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from collections.abc import Awaitable, Callable
from typing import Any

import httpx
from pydantic import BaseModel, Field

from thorn.gateway._event import EventSource, IncomingEvent
from thorn.gateway._routing import Noteable, NoteableKind, route_github_event

log = logging.getLogger(__name__)

_DEFAULT_API_BASE = "https://api.github.com"

# Matches the trailing numeric ID in a GitHub API URL such as
# ``https://api.github.com/repos/owner/repo/issues/42`` or
# ``…/pulls/7``.
_SUBJECT_NUMBER_RE = re.compile(r"/(\d+)$")

_SUBJECT_TYPE_TO_NOTEABLE: dict[str, NoteableKind] = {
    "Issue": NoteableKind.ISSUE,
    "PullRequest": NoteableKind.CHANGE_REQUEST,
}


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class GitHubNotificationsSourceConfig(BaseModel):
    """Settings for the GitHub Notifications API poller.

    Unlike the old repository-events config, this is user-scoped: a
    single poller sees all notifications for the authenticated user,
    across every repository the token can access.
    """

    token: str = Field(
        description="Personal access token (classic) with 'notifications' scope",
    )
    base_url: str = Field(
        default=_DEFAULT_API_BASE,
        description="GitHub REST API base URL",
    )
    poll_interval: int = Field(
        default=30,
        ge=5,
        description="Seconds between polling cycles",
    )
    native_id_to_project_name: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Mapping from GitHub native repository ID (owner/repo) to "
            "logical project name, used for project-name-based session keys."
        ),
    )

    @classmethod
    def from_env(cls) -> GitHubNotificationsSourceConfig:
        """Load from environment variables.

        - ``GITHUB_TOKEN`` -- Personal access token (required)
        - ``GITHUB_API_URL`` -- REST API base URL (optional)
        - ``THORN_POLL_INTERVAL`` -- seconds between polls (default 30)
        """
        token = os.environ.get("GITHUB_TOKEN")
        if not token:
            raise ValueError(
                "GITHUB_TOKEN is required for the GitHub notifications source "
                "(personal access token with 'notifications' scope).",
            )
        base_url = os.environ.get("GITHUB_API_URL", _DEFAULT_API_BASE)
        return cls(
            token=token,
            base_url=base_url.rstrip("/"),
            poll_interval=int(os.environ.get("THORN_POLL_INTERVAL", "30")),
        )


# ---------------------------------------------------------------------------
# Notification formatting
# ---------------------------------------------------------------------------


def _extract_noteable_from_notification(
    subject_type: str,
    subject_url: str,
) -> Noteable | None:
    """Map a notification's ``subject.type`` and ``subject.url`` to a :class:`Noteable`.

    Returns ``None`` for subject types that don't correspond to a
    noteable (e.g. ``Commit``, ``Release``).
    """
    kind = _SUBJECT_TYPE_TO_NOTEABLE.get(subject_type)
    if kind is None:
        return None
    match = _SUBJECT_NUMBER_RE.search(subject_url)
    if match is None:
        return None
    return Noteable(kind, int(match.group(1)))


def _format_notification_content(
    *,
    repo_full_name: str,
    repo_id: int,
    clone_url: str,
    default_branch: str,
    html_url: str,
    subject_type: str,
    subject_title: str,
    reason: str,
    thread_id: str,
    updated_at: str,
    comment_body: str,
) -> str:
    """Build a human-readable prompt from a GitHub notification thread."""
    lines = [
        f"GitHub notification: you were {reason} on "
        f"{subject_type} \"{subject_title}\" in project {repo_full_name} "
        f"(repo_id={repo_id}).",
        "",
        f"Notification ID: {thread_id}",
        f"Reason: {reason}",
        f"Target: {subject_type} — {subject_title}",
        f"Updated: {updated_at}",
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

    lines.extend([
        "Respond to the notification as appropriate, then mark the "
        "notification as done using forge_mark_notification_done.",
    ])
    return "\n".join(lines)


def _make_incoming_event(
    *,
    thread: dict[str, Any],
    comment_body: str,
    native_id_to_project_name: dict[str, str],
) -> IncomingEvent:
    """Convert a GitHub notification thread dict into an :class:`IncomingEvent`."""
    repo = thread["repository"]
    repo_id: int = repo["id"]
    repo_full_name: str = repo["full_name"]
    clone_url: str = repo.get("clone_url") or ""
    default_branch: str = repo.get("default_branch") or "main"
    html_url: str = repo.get("html_url") or ""

    subject = thread["subject"]
    subject_type: str = subject["type"]
    subject_title: str = subject["title"]
    subject_url: str = subject.get("url") or ""
    reason: str = thread.get("reason") or ""
    thread_id: str = thread["id"]
    updated_at: str = thread.get("updated_at") or ""

    project_name = native_id_to_project_name.get(repo_full_name, "")

    noteable = _extract_noteable_from_notification(subject_type, subject_url)

    session_key = route_github_event(
        repo_id=repo_id,
        noteable=noteable,
        event_type=subject_type,
        event_id=thread_id,
        project_name=project_name,
    )

    content = _format_notification_content(
        repo_full_name=repo_full_name,
        repo_id=repo_id,
        clone_url=clone_url,
        default_branch=default_branch,
        html_url=html_url,
        subject_type=subject_type,
        subject_title=subject_title,
        reason=reason,
        thread_id=thread_id,
        updated_at=updated_at,
        comment_body=comment_body,
    )

    return IncomingEvent(
        source="github",
        session_key=session_key,
        content=content,
        metadata={
            "notification_id": thread_id,
            "reason": reason,
            "subject_type": subject_type,
            "subject_title": subject_title,
            "repo_full_name": repo_full_name,
            "repo_id": repo_id,
            "clone_url": clone_url,
            "default_branch": default_branch,
            "html_url": html_url,
            "project_name": project_name,
            "updated_at": updated_at,
        },
    )


# ---------------------------------------------------------------------------
# EventSource implementation
# ---------------------------------------------------------------------------


class GitHubNotificationsSource(EventSource):
    """Polls the GitHub Notifications API and emits events for new threads.

    Uses a personal access token; GitHub App installation tokens are
    not supported (the Notifications API requires user-scoped auth).
    """

    Config = GitHubNotificationsSourceConfig

    def __init__(
        self,
        config: GitHubNotificationsSourceConfig,
        *,
        service_name: str = "",
    ) -> None:
        self._config = config
        self._service_name = service_name
        self._http = httpx.Client(
            base_url=config.base_url.rstrip("/"),
            headers={
                "Authorization": f"Bearer {config.token}",
                "Accept": "application/vnd.github+json",
            },
            timeout=30.0,
        )
        self._seen_thread_ids: set[str] = set()
        self._primed: bool = False
        self._last_modified: str | None = None
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
            "GitHub notifications source authenticated as %s",
            user_info["login"],
        )
        log.info(
            "Polling GitHub notifications every %ds",
            self._config.poll_interval,
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
        """Verify the PAT by calling ``GET /user``."""
        resp = self._http.get("/user")
        resp.raise_for_status()
        data = resp.json()
        return {
            "login": data["login"],
            "name": data.get("name", ""),
            "html_url": data.get("html_url", ""),
        }

    async def _poll_once(
        self,
        on_event: Callable[[IncomingEvent], Awaitable[None]],
    ) -> None:
        new_events = await asyncio.to_thread(self._fetch_new_notifications)
        if new_events:
            log.info("Found %d new notification(s)", len(new_events))
        for ev in new_events:
            await on_event(ev)

    def _fetch_new_notifications(self) -> list[IncomingEvent]:
        """Return new notifications since last poll."""
        headers: dict[str, str] = {}
        if self._last_modified is not None:
            headers["If-Modified-Since"] = self._last_modified

        resp = self._http.get(
            "/notifications",
            params={"participating": "true", "all": "false", "per_page": "50"},
            headers=headers,
        )

        if resp.status_code == 304:
            return []
        resp.raise_for_status()

        if "Last-Modified" in resp.headers:
            self._last_modified = resp.headers["Last-Modified"]

        threads: list[dict[str, Any]] = resp.json()

        if not self._primed:
            for t in threads:
                self._seen_thread_ids.add(t["id"])
            self._primed = True
            log.info(
                "GitHub source: primed with %d existing notification(s); "
                "only newer activity will be delivered",
                len(self._seen_thread_ids),
            )
            return []

        incoming: list[IncomingEvent] = []
        for thread in threads:
            tid = thread["id"]
            if tid in self._seen_thread_ids:
                continue
            self._seen_thread_ids.add(tid)

            comment_body = self._fetch_latest_comment(thread)

            incoming.append(
                _make_incoming_event(
                    thread=thread,
                    comment_body=comment_body,
                    native_id_to_project_name=self._config.native_id_to_project_name,
                ),
            )

        # Process oldest-first so prompts follow chronological order.
        incoming.sort(key=lambda e: e.metadata.get("updated_at", ""))
        return incoming

    def _fetch_latest_comment(self, thread: dict[str, Any]) -> str:
        """Fetch the body of the latest comment on the notification thread.

        Returns an empty string if the URL is missing or the request
        fails (best-effort).
        """
        url = (thread.get("subject") or {}).get("latest_comment_url") or ""
        if not url:
            return ""
        try:
            resp = self._http.get(url)
            resp.raise_for_status()
            return resp.json().get("body") or ""
        except Exception:
            log.debug(
                "Failed to fetch latest comment for thread %s",
                thread.get("id", "?"),
                exc_info=True,
            )
            return ""


__all__ = [
    "GitHubNotificationsSourceConfig",
    "GitHubNotificationsSource",
]
