"""GitLab polling event source.

Polls the GitLab Todos API for new notifications (e.g. @-mentions,
assignments, review requests) and converts them into
:class:`~thorn.gateway._event.RawIncomingEvent` objects for the
gateway-side notification formatter to wrap and route.

For configured projects, also polls GitLab project events for issue
closures and merge-request merges.  Those lifecycle events are
record-only by default: the source remembers their IDs so they do not
resurface, but does not hand them to the gateway as prompt work.

The source captures the TODO ``author`` (numeric ``id``, ``username``,
and bot flag) into an :class:`~thorn.gateway._actor.ActorIdentity` so
the gateway's peer registry can look up the actor without
re-fetching the user from GitLab.

GitLab TODO acknowledgement is handoff-based: after the gateway
callback accepts a TODO event, this source marks that TODO done on
GitLab even though the agent may still be handling the local inbox
item.  If gateway handoff raises, the TODO is left pending for a
later poll.  Project-event polling is read-only and does not mutate
upstream project events; record-only acknowledgement is local to
Thorn's seen-event cache.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from thorn.gateway._actor import ActorIdentity
from thorn.gateway._event import (
    ContextItem,
    ContextItemKind,
    EventKind,
    EventSource,
    EventSourceStatusSnapshot,
    EventSourceStatusState,
    RawIncomingEvent,
    event_source_status_timestamp,
)
from thorn.gateway._routing import (
    ForgeServiceName,
    Noteable,
    NoteableKind,
    route_gitlab_todo,
)
from thorn.gateway._workspace_bootstrap import RepositoryCheckoutSpec
from thorn.runtime._session import AgentID, SessionKey
from thorn.tools.gitlab import GitLabProjectResolver

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class _ProjectEventRecord:
    """A GitLab project event plus the Thorn project name it belongs to."""

    project_name: str
    event: Any


class _ProjectEventPollMode(Enum):
    BASELINE = "baseline"
    INCREMENTAL = "incremental"


class _ProjectEventKind(Enum):
    CLOSED_ISSUE = ("issue", "closed")
    MERGED_MERGE_REQUEST = ("merge_request", "merged")

    @property
    def target_type(self) -> str:
        return self.value[0]

    @property
    def action(self) -> str:
        return self.value[1]


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
            "Install it with: uv pip install 'thorn-agent[gitlab]'"
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
    ``username`` as display metadata for envelopes.
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
    owner_agent_id: AgentID | None = None,
) -> RawIncomingEvent:
    """Convert a GitLab TODO into a :class:`RawIncomingEvent`.

    Captures the TODO's ``author`` as the primary actor and the
    TODO's ``body`` (when present) as a single :class:`ContextItem`.
    Classification of structural vs. conversational follows
    :func:`_classify_action`.
    """
    project = todo.project
    pid = project["id"]
    default_branch = project.get("default_branch") or "main"
    proj_name = _lookup_project_name(project, project_id_to_name)
    session_key = route_gitlab_todo(
        project_id=pid,
        noteable=_noteable_from_todo(todo),
        project_name=proj_name,
        forge_name=ForgeServiceName(forge_name),
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
        agent_id=owner_agent_id,
        metadata={
            "todo_id": todo.id,
            "project_id": pid,
            "project_name": proj_name,
            "noteable_type": todo.target_type,
            "noteable_iid": todo.target["iid"],
            "action_name": todo.action_name,
            "clone_url": project.get("http_url_to_repo", ""),
            "default_branch": default_branch,
            "web_url": project.get("web_url", ""),
        },
        workspace_bootstrap=RepositoryCheckoutSpec.from_event_fields(
            clone_url=project.get("http_url_to_repo", ""),
            default_branch=default_branch,
        ),
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
    owner_agent_id: AgentID | None = None,
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
    created_at = str(_field_value(project_event, "created_at", "") or "")

    actor = _actor_from_todo_author(
        _field_value(project_event, "author"), service=forge_name,
    )
    noteable = Noteable(kind=noteable_kind, number=noteable_iid)
    session_key = route_gitlab_todo(
        project_id=project_id,
        noteable=noteable,
        project_name=project_name,
        forge_name=ForgeServiceName(forge_name),
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
    return RawIncomingEvent(
        source="gitlab",
        session_key=session_key,
        kind=EventKind.STRUCTURAL,
        primary_actor=actor,
        summary=summary,
        agent_id=owner_agent_id,
        metadata={
            "project_event_id": event_id,
            "project_id": project_id,
            "project_name": project_name,
            "noteable_type": target_type,
            "noteable_iid": noteable_iid,
            "action_name": action_name,
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

    def __init__(
        self,
        config: GitLabSourceConfig,
        *,
        service_name: str = "",
        owner_agent_id: AgentID | None = None,
    ) -> None:
        _require_gitlab()
        self._config = config
        self._service_name = service_name
        self._owner_agent_id = owner_agent_id
        self._gl = _gitlab_lib.Gitlab(  # type: ignore[union-attr]
            url=config.url,
            private_token=config.token,
        )
        self._seen: set[int] = set()
        self._seen_project_events: set[int] = set()
        self._project_events_baselined = False
        self._project_resolver = GitLabProjectResolver(self._gl.projects)
        self._stop_event: asyncio.Event | None = None
        self._last_poll_started_at: str | None = None
        self._last_poll_finished_at: str | None = None
        self._last_error: str | None = None
        self._last_event_count: int | None = None
        self._poll_count = 0
        self._status_state = EventSourceStatusState.STARTING

    @property
    def name(self) -> str:
        return self._service_name

    def status_snapshot(self) -> EventSourceStatusSnapshot:
        name = self.name or type(self).__name__
        return EventSourceStatusSnapshot(
            name=name,
            source_type=type(self).__name__,
            state=self._status_state,
            last_poll_started_at=self._last_poll_started_at,
            last_poll_finished_at=self._last_poll_finished_at,
            last_error=self._last_error,
            last_event_count=self._last_event_count,
            poll_count=self._poll_count,
        )

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
        self._status_state = EventSourceStatusState.STOPPED

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
        started_at = event_source_status_timestamp()
        self._last_poll_started_at = started_at
        delivered_count = 0
        try:
            todos = await asyncio.to_thread(self._get_pending_todos)
            new_todos = [t for t in todos if t.id not in self._seen]
            if new_todos:
                log.info(
                    "Found %d new TODO(s) out of %d pending",
                    len(new_todos), len(todos),
                )
            id_to_name = self._config.project_id_to_name
            for todo in new_todos:
                event = _make_raw_event(
                    todo,
                    project_id_to_name=id_to_name,
                    gitlab_url=self._config.url,
                    forge_name=self._config.forge_name,
                    owner_agent_id=self._owner_agent_id,
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

                self._seen.add(todo.id)
                delivered_count += 1
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

            delivered_count += await self._poll_project_events_once(on_event)
        except Exception as exc:
            self._record_poll_error(started_at, exc)
            raise
        self._record_poll_success(started_at, delivered_count)

    def _record_poll_success(self, started_at: str, event_count: int) -> None:
        self._poll_count += 1
        self._last_poll_started_at = started_at
        self._last_poll_finished_at = event_source_status_timestamp()
        self._last_event_count = event_count
        self._last_error = None
        self._status_state = EventSourceStatusState.OK

    def _record_poll_error(self, started_at: str, exc: Exception) -> None:
        self._poll_count += 1
        self._last_poll_started_at = started_at
        self._last_poll_finished_at = event_source_status_timestamp()
        self._last_event_count = None
        self._last_error = str(exc) or type(exc).__name__
        self._status_state = EventSourceStatusState.ERROR

    async def _poll_project_events_once(
        self,
        on_event: Callable[[RawIncomingEvent], Awaitable[None]],
    ) -> int:
        poll_mode = (
            _ProjectEventPollMode.INCREMENTAL
            if self._project_events_baselined
            else _ProjectEventPollMode.BASELINE
        )
        project_events = await asyncio.to_thread(
            self._get_project_events,
            poll_mode=poll_mode,
        )
        if poll_mode is _ProjectEventPollMode.BASELINE:
            self._seen_project_events.update(
                _project_event_id(record.event) for record in project_events
            )
            self._project_events_baselined = True
            if project_events:
                log.info(
                    "Baselined %d GitLab project event(s); future "
                    "closed-issue and merged-MR events will be recorded "
                    "without waking the gateway.",
                    len(project_events),
                )
            return 0

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
            self._seen_project_events.add(event_id)
            log.info(
                "Recorded GitLab project lifecycle event %s without "
                "creating prompt work",
                event_id,
            )
        return 0

    def _get_pending_todos(self) -> list[Any]:
        return list(self._gl.todos.list(state="pending", iterator=True))

    def _get_project_events(
        self,
        *,
        poll_mode: _ProjectEventPollMode,
    ) -> list[_ProjectEventRecord]:
        events: list[_ProjectEventRecord] = []
        for project_ref, project_name in self._config.project_id_to_name.items():
            project = self._get_project_for_events(project_ref)
            for event_kind in _ProjectEventKind:
                events.extend(self._get_project_events_of_kind(
                    project,
                    project_name,
                    event_kind=event_kind,
                    poll_mode=poll_mode,
                ))
        return events

    def _get_project_events_of_kind(
        self,
        project: Any,
        project_name: str,
        *,
        event_kind: _ProjectEventKind,
        poll_mode: _ProjectEventPollMode,
    ) -> list[_ProjectEventRecord]:
        query = {
            "target_type": event_kind.target_type,
            "action": event_kind.action,
            "sort": "desc",
            "per_page": 50,
        }
        if poll_mode is _ProjectEventPollMode.INCREMENTAL:
            project_events = project.events.list(**query, iterator=True)
        else:
            # A restart deliberately re-baselines only the newest page.
            # This keeps startup cost bounded without adding a second
            # persistence mechanism for record-only lifecycle events.
            project_events = project.events.list(**query, page=1)

        events: list[_ProjectEventRecord] = []
        for project_event in project_events:
            if (
                poll_mode is _ProjectEventPollMode.INCREMENTAL
                and _project_event_id(project_event) in self._seen_project_events
            ):
                break
            events.append(_ProjectEventRecord(project_name, project_event))
        return events

    def _get_project_for_events(self, project_ref: str) -> Any:
        return self._project_resolver.get_project(project_ref)


def _project_event_id(project_event: Any) -> int:
    return int(_field_value(project_event, "id"))


__all__ = [
    "GitLabSourceConfig",
    "GitLabTODOsSource",
]
