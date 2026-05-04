"""GitLab polling event source.

Polls the GitLab Todos API for new notifications (e.g. @-mentions,
assignments, review requests) and converts them into
:class:`~thorn.gateway._event.RawIncomingEvent` objects for the
gateway-side notification formatter to wrap and route.

For configured projects, also polls GitLab project events for issue
closures and merge-request merges.  GitLab does not create user TODOs
for every structural transition Thorn cares about, so project-event
polling fills the gap without requiring operators to expose a webhook
endpoint.

The source captures the TODO ``author`` (numeric ``id``, ``username``,
and bot flag) into an :class:`~thorn.gateway._actor.ActorIdentity` so
the gateway's peer registry can look up the actor without
re-fetching the user from GitLab.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field

from thorn.gateway._actor import ActorIdentity
from thorn.gateway._event import (
    ContextItem,
    ContextItemKind,
    EventKind,
    EventSource,
    RawIncomingEvent,
)
from thorn.gateway._routing import Noteable, NoteableKind, route_gitlab_todo
from thorn.runtime._session import SessionKey

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class _ProjectEventRecord:
    """A GitLab project event plus the Thorn project name it belongs to."""

    project_name: str
    event: Any

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
    """Configuration for the GitLab TODOs and project-events source.

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
            "Mapping from a GitLab project's identifier (either its "
            "numeric ID as a string, or its full path-with-namespace "
            "such as 'group/project') to the logical project name "
            "used for session-key routing.  Both key forms are "
            "consulted at lookup time so the producer of this dict "
            "can use whichever it has handy.  A non-empty mapping "
            "also enables supplementary project-event polling for "
            "closed issues and merged merge requests."
        ),
    )
    forge_name: str = Field(
        default="gitlab",
        description=(
            "Service name stamped on every "
            "``ActorIdentity.service`` produced by this source.  "
            "Should match the ``forges[].name`` entry in "
            "``gateway.json`` so peer matching and per-forge "
            "trigger policy line up."
        ),
    )


# ---------------------------------------------------------------------------
# Event classification
# ---------------------------------------------------------------------------


# GitLab TODO ``action_name`` values that represent direct
# message-like activity from a human.  Anything not in this set is
# treated as structural (an assignment changed, CI failed, an
# approval is required, ...).  Same conservative-default rationale
# as the GitHub source: structural-with-banner is the carve-out the
# operator can disable, while a missed conversational classification
# would silently drop legitimate events.
_CONVERSATIONAL_ACTIONS: frozenset[str] = frozenset({
    "mentioned",
    "directly_addressed",
    "review_requested",
})


def _classify_action(action_name: str) -> EventKind:
    """Map a GitLab TODO ``action_name`` to an :class:`EventKind`."""
    if action_name in _CONVERSATIONAL_ACTIONS:
        return EventKind.CONVERSATIONAL
    return EventKind.STRUCTURAL


def _kind_for_target(target_type: str, action_name: str) -> ContextItemKind:
    """Classify a context item by its containing target type."""
    if action_name in _CONVERSATIONAL_ACTIONS:
        return ContextItemKind.COMMENT
    if target_type == "Issue":
        return ContextItemKind.ISSUE_BODY
    if target_type == "MergeRequest":
        return ContextItemKind.PR_BODY
    return ContextItemKind.COMMENT


# ---------------------------------------------------------------------------
# Actor extraction
# ---------------------------------------------------------------------------


def _actor_from_todo_author(
    author: Any,
    *,
    service: str,
) -> ActorIdentity | None:
    """Build an :class:`ActorIdentity` from a GitLab TODO ``author`` field.

    *author* may be a ``dict`` (when the TODO was deserialised from
    JSON) or a ``Mapping``-like object (depending on ``python-gitlab``
    version).  GitLab user objects expose ``id``, ``username``,
    ``name``, and ``bot``; we capture all four where available, with
    the numeric id as the immutable ``account_id`` and the
    ``username`` as a fallback for textual operator config.
    """
    if author is None:
        return None
    if isinstance(author, dict):
        get = author.get
    else:
        # ``python-gitlab`` user objects expose attributes; fall back
        # to ``getattr`` so this works regardless of the source's
        # serialisation style.
        def get(key: str, default: Any = None) -> Any:
            return getattr(author, key, default)

    user_id = get("id")
    username = (get("username") or "").strip()
    name = get("name") or ""
    bot_flag = get("bot")

    if user_id is None and not username:
        return None

    account_id = str(user_id) if user_id is not None else username
    secondary: tuple[str, ...] = (
        (username,) if username and username != account_id else ()
    )

    is_bot: bool | None = None
    if bot_flag is not None:
        is_bot = bool(bot_flag)

    return ActorIdentity(
        service=service,
        account_id=account_id,
        display_name=name or username,
        is_bot=is_bot,
        secondary_account_ids=secondary,
    )


# ---------------------------------------------------------------------------
# Event content
# ---------------------------------------------------------------------------


def _make_summary(todo: Any) -> str:
    """Build the harness-controlled summary line(s) for a GitLab TODO.

    Source-controlled, *not* attacker-controlled: the comment body
    (which *is* attacker-controlled) is surfaced via a separate
    :class:`ContextItem`, never inlined into this string.
    """
    project = todo.project
    project_id = project["id"]
    project_name = project.get("path_with_namespace", str(project_id))
    clone_url = project.get("http_url_to_repo", "")
    default_branch = project.get("default_branch", "main")
    web_url = project.get("web_url", "")
    noteable_type = todo.target_type
    noteable_iid = todo.target["iid"]
    action = todo.action_name

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


def _lookup_project_name(
    todo_project: dict[str, Any],
    project_id_to_name: dict[str, str] | None,
) -> str:
    """Resolve the logical project name for a GitLab TODO's project.

    The mapping may be keyed by either the numeric project ID (as a
    string) or the project's full ``path_with_namespace``; both forms
    are consulted so that path-based ``gateway.json`` configurations
    work without requiring an extra API round-trip to translate
    paths into numeric IDs.
    """
    if not project_id_to_name:
        return ""
    pid = todo_project.get("id")
    path = todo_project.get("path_with_namespace", "")
    if path and path in project_id_to_name:
        return project_id_to_name[path]
    if pid is not None and str(pid) in project_id_to_name:
        return project_id_to_name[str(pid)]
    return ""


def _make_session_key(
    todo: Any,
    project_id_to_name: dict[str, str] | None = None,
) -> SessionKey:
    """Derive a session key from a TODO.

    Thin wrapper for backward compatibility with tests that call this
    directly; delegates to :func:`route_gitlab_todo`.
    """
    pid = todo.project["id"]
    proj_name = _lookup_project_name(todo.project, project_id_to_name)
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


def _make_project_event_external_key(gitlab_url: str, event_id: int) -> str:
    """Build a source-namespaced key for a GitLab project event."""
    return f"gitlab:{gitlab_url}:project-event:{event_id}"


def _make_raw_event(
    todo: Any,
    project_id_to_name: dict[str, str] | None = None,
    *,
    gitlab_url: str = "",
    forge_name: str = "gitlab",
) -> RawIncomingEvent:
    """Convert a GitLab TODO into a :class:`RawIncomingEvent`.

    Captures the TODO's ``author`` as the primary actor and the
    TODO's ``body`` (when present) as a single :class:`ContextItem`.
    Classification of structural vs. conversational follows
    :func:`_classify_action`.
    """
    project = todo.project
    pid = project["id"]
    proj_name = _lookup_project_name(project, project_id_to_name)
    session_key = route_gitlab_todo(
        project_id=pid,
        noteable=_noteable_from_todo(todo),
        project_name=proj_name,
    )

    actor = _actor_from_todo_author(
        getattr(todo, "author", None), service=forge_name,
    )

    body = getattr(todo, "body", "") or ""
    items: tuple[ContextItem, ...] = ()
    if body:
        items = (
            ContextItem(
                body=body,
                kind=_kind_for_target(todo.target_type, todo.action_name),
                actor=actor,
                timestamp=getattr(todo, "created_at", "") or "",
            ),
        )

    return RawIncomingEvent(
        source="gitlab",
        session_key=session_key,
        kind=_classify_action(todo.action_name),
        primary_actor=actor,
        summary=_make_summary(todo),
        items=items,
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


_PROJECT_EVENT_NOTEABLE_KINDS: dict[str, NoteableKind] = {
    "Issue": NoteableKind.ISSUE,
    "MergeRequest": NoteableKind.CHANGE_REQUEST,
}


def _field_value(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _make_project_event_raw_event(
    project_event: Any,
    *,
    project_name: str = "",
    gitlab_url: str = "",
    forge_name: str = "gitlab",
) -> RawIncomingEvent:
    """Convert a GitLab project event into a structural gateway event."""
    target_type = str(_field_value(project_event, "target_type", ""))
    noteable_kind = _PROJECT_EVENT_NOTEABLE_KINDS.get(target_type)
    if noteable_kind is None:
        raise ValueError(
            f"Unsupported GitLab project event target_type: {target_type!r}"
        )

    event_id = int(_field_value(project_event, "id"))
    project_id = int(_field_value(project_event, "project_id"))
    noteable_iid = int(_field_value(project_event, "target_iid"))
    action_name = str(_field_value(project_event, "action_name", ""))
    target_title = str(_field_value(project_event, "target_title", ""))
    created_at = str(_field_value(project_event, "created_at", "") or "")

    actor = _actor_from_todo_author(
        _field_value(project_event, "author"), service=forge_name,
    )
    noteable = Noteable(kind=noteable_kind, number=noteable_iid)
    session_key = route_gitlab_todo(
        project_id=project_id,
        noteable=noteable,
        project_name=project_name,
    )
    display_target = (
        f"merge request !{noteable_iid}"
        if noteable_kind is NoteableKind.CHANGE_REQUEST
        else f"issue #{noteable_iid}"
    )
    summary = (
        f"GitLab project event: {display_target} was {action_name} "
        f"in project {project_name or project_id}."
    )
    if target_title:
        summary = f"{summary}\nTitle: {target_title}"
    return RawIncomingEvent(
        source="gitlab",
        session_key=session_key,
        kind=EventKind.STRUCTURAL,
        primary_actor=actor,
        summary=summary,
        metadata={
            "project_event_id": event_id,
            "project_id": project_id,
            "project_name": project_name,
            "noteable_type": target_type,
            "noteable_iid": noteable_iid,
            "action_name": action_name,
            "target_title": target_title,
            "created_at": created_at,
        },
        external_key=_make_project_event_external_key(gitlab_url, event_id),
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
        self._seen_project_events: set[int] = set()
        self._project_events_baselined = False
        self._project_path_id_cache: dict[str, int] = {}
        self._stop_event: asyncio.Event | None = None

    @property
    def name(self) -> str:
        return self._service_name

    async def start(
        self,
        on_event: Callable[[RawIncomingEvent], Awaitable[None]],
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
        on_event: Callable[[RawIncomingEvent], Awaitable[None]],
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
            event = _make_raw_event(
                todo,
                project_id_to_name=id_to_name,
                gitlab_url=self._config.url,
                forge_name=self._config.forge_name,
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
            # happens regardless of whether the gateway delivered,
            # deduplicated, or *dropped* the event: drops are
            # terminal in the formatter's contract, so we want the
            # platform to stop resurfacing the entity in all three
            # cases.  If an earlier copy is already in flight (dedup
            # path) the agent will still handle it; if mark_as_done
            # itself fails we just log and move on -- the ``_seen``
            # cache prevents a re-emit within this process, and a
            # restart will retry via the normal pending-TODOs path.
            try:
                await asyncio.to_thread(todo.mark_as_done)
            except Exception:
                log.exception(
                    "Failed to mark GitLab TODO %s as done", todo.id,
                )

        await self._poll_project_events_once(on_event)

    async def _poll_project_events_once(
        self,
        on_event: Callable[[RawIncomingEvent], Awaitable[None]],
    ) -> None:
        project_events = await asyncio.to_thread(self._get_project_events)
        if not self._project_events_baselined:
            self._seen_project_events.update(
                _project_event_id(record.event) for record in project_events
            )
            self._project_events_baselined = True
            if project_events:
                log.info(
                    "Baselined %d GitLab project event(s); future "
                    "closed-issue and merged-MR events will wake the gateway.",
                    len(project_events),
                )
            return

        new_events = [
            record
            for record in project_events
            if _project_event_id(record.event) not in self._seen_project_events
        ]
        if new_events:
            log.info("Found %d new GitLab project event(s)", len(new_events))
        for record in new_events:
            project_event = record.event
            event_id = _project_event_id(project_event)
            try:
                raw_event = _make_project_event_raw_event(
                    project_event,
                    project_name=record.project_name,
                    gitlab_url=self._config.url,
                    forge_name=self._config.forge_name,
                )
                await on_event(raw_event)
            except Exception:
                log.exception(
                    "Failed to post GitLab project event %s", event_id,
                )
                continue
            self._seen_project_events.add(event_id)

    def _get_pending_todos(self) -> list[Any]:
        return list(self._gl.todos.list(state="pending", iterator=True))

    def _get_project_events(self) -> list[_ProjectEventRecord]:
        events: list[_ProjectEventRecord] = []
        for project_ref, project_name in self._config.project_id_to_name.items():
            project = self._get_project_for_events(project_ref)
            for event in project.events.list(
                target_type="issue", action="closed", per_page=50,
            ):
                events.append(_ProjectEventRecord(project_name, event))
            for event in project.events.list(
                target_type="merge_request", action="merged", per_page=50,
            ):
                events.append(_ProjectEventRecord(project_name, event))
        return events

    def _get_project_for_events(self, project_ref: str) -> Any:
        project_ref = _coerce_gitlab_project_ref(project_ref)
        if isinstance(project_ref, int):
            return self._gl.projects.get(project_ref)

        cached_id = self._project_path_id_cache.get(project_ref)
        if cached_id is not None:
            return self._gl.projects.get(cached_id)

        try:
            project = self._gl.projects.get(project_ref)
        except Exception as exc:
            if not _is_gitlab_not_found(exc):
                raise
            resolved_id = self._resolve_project_id(project_ref)
            return self._gl.projects.get(resolved_id)

        project_id = _field_value(project, "id")
        if project_id is not None:
            self._project_path_id_cache[project_ref] = int(project_id)
        return project

    def _resolve_project_id(self, project_path: str) -> int:
        normalized_path = _normalize_gitlab_project_path(project_path)
        cached_id = self._project_path_id_cache.get(normalized_path)
        if cached_id is not None:
            return cached_id

        search_term = normalized_path.rsplit("/", 1)[-1]
        projects = self._gl.projects.list(
            search=search_term,
            simple=True,
            iterator=True,
        )
        for project in projects:
            candidate_path = str(
                _field_value(project, "path_with_namespace", "") or "",
            ).strip("/")
            if candidate_path != normalized_path:
                continue
            project_id = int(_field_value(project, "id"))
            self._project_path_id_cache[normalized_path] = project_id
            return project_id

        raise RuntimeError(
            f"GitLab project {normalized_path!r} could not be resolved to "
            "a numeric project ID via project search. Set this fork's "
            "`native_id` in gateway.json to the numeric GitLab project ID."
        )


def _project_event_id(project_event: Any) -> int:
    return int(_field_value(project_event, "id"))


def _coerce_gitlab_project_ref(project_ref: str) -> int | str:
    stripped_ref = project_ref.strip()
    if stripped_ref.isdigit():
        return int(stripped_ref)
    return _normalize_gitlab_project_path(stripped_ref)


def _normalize_gitlab_project_path(project_path: str) -> str:
    normalized_path = project_path.strip().strip("/")
    if normalized_path.endswith(".git"):
        normalized_path = normalized_path[: -len(".git")]
    if not normalized_path or "/" not in normalized_path:
        raise RuntimeError(
            f"GitLab project reference {project_path!r} is not a numeric "
            "ID and does not look like a path_with_namespace value. Set "
            "this fork's `native_id` in gateway.json to the numeric "
            "GitLab project ID."
        )
    return normalized_path


def _is_gitlab_not_found(exc: BaseException) -> bool:
    return str(_field_value(exc, "response_code", "")) == "404"


__all__ = [
    "GitLabSourceConfig",
    "GitLabTODOsSource",
]
