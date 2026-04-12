"""GitLab TODOs polling event source.

Polls the GitLab Todos API for new notifications (e.g. @-mentions,
assignments, review requests) and converts them into
:class:`~thorn.gateway._event.IncomingEvent` objects for the gateway.

Adapted from ``thorn-bot/src/thorn_bot/_poller.py`` and
``thorn-bot/src/thorn_bot/_daemon.py``.
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Awaitable, Callable
from typing import Any

from pydantic import BaseModel, Field

from thorn.gateway._event import EventSource, IncomingEvent
from thorn.runtime._session import SessionKey

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional dependency guard (same pattern as thorn.tools.gitlab)
# ---------------------------------------------------------------------------

try:
    import gitlab as _gitlab_lib

    _HAS_GITLAB = True
except ImportError:
    _gitlab_lib = None  # type: ignore[assignment]
    _HAS_GITLAB = False


def _require_gitlab() -> None:
    if not _HAS_GITLAB:
        raise ImportError(
            "python-gitlab is required for the GitLab event source. "
            "Install it with: pip install thorn[gitlab]"
        )


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class GitLabSourceConfig(BaseModel):
    """Configuration for the GitLab TODOs event source.

    Typically loaded from environment variables via
    :meth:`from_env`.
    """

    url: str = Field(description="GitLab instance URL (no trailing slash)")
    token: str = Field(description="Personal Access Token with 'api' scope")
    username: str = Field(
        default="thorn-bot",
        description="Bot's GitLab username, for filtering self-mentions",
    )
    poll_interval: int = Field(
        default=30,
        ge=5,
        description="Seconds between polling cycles",
    )

    @classmethod
    def from_env(cls) -> GitLabSourceConfig:
        """Load configuration from environment variables.

        - ``GITLAB_URL`` -- GitLab instance URL
        - ``GITLAB_TOKEN`` -- Personal Access Token
        - ``THORN_GITLAB_USERNAME`` -- bot username (default ``thorn-bot``)
        - ``THORN_POLL_INTERVAL`` -- seconds between polls (default ``30``)
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
                "Set GITLAB_URL and GITLAB_TOKEN to use the GitLab event source."
            )
        return cls(
            url=url,  # type: ignore[arg-type]
            token=token,  # type: ignore[arg-type]
            username=os.environ.get("THORN_GITLAB_USERNAME", "thorn-bot"),
            poll_interval=int(os.environ.get("THORN_POLL_INTERVAL", "30")),
        )


# ---------------------------------------------------------------------------
# Event formatting
# ---------------------------------------------------------------------------


def _format_event_content(todo: Any) -> str:
    """Build a human-readable prompt from a GitLab TODO object."""
    project = todo.project
    project_id = project["id"]
    project_name = project.get("path_with_namespace", str(project_id))
    clone_url = project.get("http_url_to_repo", "")
    default_branch = project.get("default_branch", "main")
    web_url = project.get("web_url", "")
    noteable_type = todo.target_type
    noteable_iid = todo.target["iid"]
    action = todo.action_name
    body = getattr(todo, "body", "")

    lines = [
        f"GitLab notification: you were {action} on "
        f"{noteable_type} #{noteable_iid} in project {project_name} "
        f"(project_id={project_id}).",
        "",
        f"TODO ID: {todo.id}",
        f"Action: {action}",
        f"Target: {noteable_type} #{noteable_iid}",
    ]
    if clone_url:
        lines.append(f"Clone URL: {clone_url}")
    if default_branch:
        lines.append(f"Default branch: {default_branch}")
    if web_url:
        lines.append(f"Project URL: {web_url}")
    lines.append("")

    if body:
        lines.append("Comment body:")
        lines.append(body)
        lines.append("")

    lines.extend([
        "Respond to the notification as appropriate, then mark the "
        "notification as done using forge_mark_notification_done.",
    ])
    return "\n".join(lines)


def _make_session_key(todo: Any) -> SessionKey:
    """Derive a filesystem-safe session key from a TODO.

    Format: ``gitlab_<project_id>_<noteable_type>_<iid>``
    """
    project_id = todo.project["id"]
    noteable_type = todo.target_type
    noteable_iid = todo.target["iid"]
    return SessionKey(f"gitlab_{project_id}_{noteable_type}_{noteable_iid}")


def _make_event(todo: Any) -> IncomingEvent:
    """Convert a GitLab TODO into an ``IncomingEvent``."""
    project = todo.project
    return IncomingEvent(
        source="gitlab",
        session_key=_make_session_key(todo),
        content=_format_event_content(todo),
        metadata={
            "todo_id": todo.id,
            "project_id": project["id"],
            "noteable_type": todo.target_type,
            "noteable_iid": todo.target["iid"],
            "action_name": todo.action_name,
            "clone_url": project.get("http_url_to_repo", ""),
            "default_branch": project.get("default_branch", "main"),
            "web_url": project.get("web_url", ""),
        },
    )


# ---------------------------------------------------------------------------
# EventSource implementation
# ---------------------------------------------------------------------------


class GitLabTODOsSource(EventSource):
    """Polls the GitLab Todos API and emits events for new TODOs."""

    Config = GitLabSourceConfig

    def __init__(self, config: GitLabSourceConfig, *, service_name: str = "") -> None:
        _require_gitlab()
        self._config = config
        self._service_name = service_name
        self._gl = _gitlab_lib.Gitlab(  # type: ignore[union-attr]
            url=config.url,
            private_token=config.token,
        )
        self._seen: set[int] = set()
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
            "GitLab source authenticated as %s (%s)",
            user_info["username"], user_info["name"],
        )
        log.info(
            "Polling GitLab TODOs every %ds", self._config.poll_interval,
        )

        while not self._stop_event.is_set():
            try:
                await self._poll_once(on_event)
            except Exception:
                log.exception("GitLab poll cycle failed")

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
        self._gl.auth()
        user = self._gl.user
        assert user is not None
        return {
            "id": user.id,
            "username": user.username,
            "name": user.name,
            "web_url": user.web_url,
        }

    async def _poll_once(
        self,
        on_event: Callable[[IncomingEvent], Awaitable[None]],
    ) -> None:
        todos = await asyncio.to_thread(self._get_pending_todos)
        new_todos = [t for t in todos if t.id not in self._seen]
        if new_todos:
            log.info(
                "Found %d new TODO(s) out of %d pending",
                len(new_todos), len(todos),
            )
        for todo in new_todos:
            self._seen.add(todo.id)
            event = _make_event(todo)
            await on_event(event)

    def _get_pending_todos(self) -> list[Any]:
        return list(self._gl.todos.list(state="pending", iterator=True))


__all__ = [
    "GitLabSourceConfig",
    "GitLabTODOsSource",
]
