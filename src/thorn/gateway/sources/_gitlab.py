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
from collections.abc import Awaitable, Callable
from typing import Any

from pydantic import BaseModel, Field

from thorn.gateway._event import EventSource, IncomingEvent
from thorn.gateway._routing import Noteable, NoteableKind, route_gitlab_todo
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

    Built from a gateway forge entry plus an agent's GitLab credentials
    by :func:`~thorn.gateway._config.infer_event_sources`.
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
    project_id_to_name: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Mapping from stringified GitLab project ID to logical "
            "project name, used for project-name-based session keys."
        ),
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

    lines.append(
        "Respond to the notification as appropriate.  The GitLab TODO "
        "has already been marked done on your behalf.",
    )
    return "\n".join(lines)


_GITLAB_NOTEABLE_KINDS: dict[str, NoteableKind] = {
    "Issue": NoteableKind.ISSUE,
    "MergeRequest": NoteableKind.CHANGE_REQUEST,
}


def _noteable_from_todo(todo: Any) -> Noteable:
    """Extract a :class:`Noteable` from a GitLab TODO object."""
    kind = _GITLAB_NOTEABLE_KINDS.get(todo.target_type)
    if kind is None:
        raise ValueError(
            f"Unsupported GitLab noteable type: {todo.target_type!r}"
        )
    return Noteable(kind=kind, number=todo.target["iid"])


def _make_session_key(
    todo: Any,
    project_id_to_name: dict[str, str] | None = None,
) -> SessionKey:
    """Derive a session key from a TODO.

    Thin wrapper for backward compatibility with tests that call this
    directly; delegates to :func:`route_gitlab_todo`.
    """
    pid = todo.project["id"]
    proj_name = (project_id_to_name or {}).get(str(pid), "")
    return route_gitlab_todo(
        project_id=pid,
        noteable=_noteable_from_todo(todo),
        project_name=proj_name,
    )


def _make_external_key(gitlab_url: str, todo_id: int) -> str:
    """Build the source-namespaced ``external_key`` for a GitLab TODO.

    The key is globally unique across GitLab instances: a TODO ID is
    only unique within its instance, so we include the URL to avoid
    cross-instance collisions when a single agency polls multiple
    GitLabs.  The URL is included verbatim (no normalization) because
    the config is the source of truth for what "this GitLab" means;
    if the same GitLab is configured with two textually different
    URLs, that is the operator's responsibility to deduplicate.
    """
    return f"gitlab:{gitlab_url}:todo:{todo_id}"


def _make_event(
    todo: Any,
    project_id_to_name: dict[str, str] | None = None,
    *,
    gitlab_url: str = "",
) -> IncomingEvent:
    """Convert a GitLab TODO into an ``IncomingEvent``."""
    project = todo.project
    pid = project["id"]
    proj_name = (project_id_to_name or {}).get(str(pid), "")
    session_key = route_gitlab_todo(
        project_id=pid,
        noteable=_noteable_from_todo(todo),
        project_name=proj_name,
    )
    return IncomingEvent(
        source="gitlab",
        session_key=session_key,
        content=_format_event_content(todo),
        metadata={
            "todo_id": todo.id,
            "project_id": pid,
            "project_name": proj_name,
            "noteable_type": todo.target_type,
            "noteable_iid": todo.target["iid"],
            "action_name": todo.action_name,
            "clone_url": project.get("http_url_to_repo", ""),
            "default_branch": project.get("default_branch", "main"),
            "web_url": project.get("web_url", ""),
        },
        external_key=_make_external_key(gitlab_url, todo.id),
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
        id_to_name = self._config.project_id_to_name
        for todo in new_todos:
            self._seen.add(todo.id)
            event = _make_event(
                todo,
                project_id_to_name=id_to_name,
                gitlab_url=self._config.url,
            )
            try:
                await on_event(event)
            except Exception:
                # Intentionally swallow per-TODO errors so one bad
                # TODO doesn't poison the whole poll cycle.  Skip
                # mark-done: without a confirmed post we want the
                # TODO to reappear on the next poll so we can retry.
                log.exception(
                    "Failed to post event for GitLab TODO %s", todo.id,
                )
                continue

            # Proactively mark the TODO as done on GitLab's side once
            # it has been safely handed off to the gateway.  This
            # happens regardless of whether the gateway deduplicated
            # the post: the point is to keep GitLab from surfacing
            # the same TODO on every poll.  If an earlier copy is
            # already in flight (dedup path) the agent will still
            # handle it; if mark_as_done itself fails we just log and
            # move on -- the `_seen` cache prevents a re-emit within
            # this process, and a restart will retry via the normal
            # pending-TODOs path.
            try:
                await asyncio.to_thread(todo.mark_as_done)
            except Exception:
                log.exception(
                    "Failed to mark GitLab TODO %s as done", todo.id,
                )

    def _get_pending_todos(self) -> list[Any]:
        return list(self._gl.todos.list(state="pending", iterator=True))


__all__ = [
    "GitLabSourceConfig",
    "GitLabTODOsSource",
]
