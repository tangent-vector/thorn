"""GitHub notifications event source.

Polls ``GET /notifications`` (the user-scoped Notifications API) to
discover @-mentions, assignments, review requests, and other activity
directed at the authenticated bot user.

Read/unread state lives on GitHub.  After delivering a
:class:`RawIncomingEvent` for a notification thread, the source
``PATCH``-es the thread to ``read`` so the next poll won't see it
again.  At startup the source drains the existing unread set the
same way so the agent isn't flooded with whatever accumulated while
the gateway was down.  No client-side "seen" cache is maintained;
cross-poll deduplication of in-flight events is handled by the
gateway's :class:`~thorn.runtime._in_flight_index.InFlightIndex`.

The source produces *raw* events: it captures the actor (numeric
``user.id`` plus textual ``login``, with the platform's ``type ==
"Bot"`` flag mapped onto :attr:`ActorIdentity.is_bot`) and
classifies the event as structural or conversational based on the
notification's ``reason``.  The gateway-side
:class:`~thorn.gateway._formatter.NotificationFormatter` then
applies peer lookup, the trigger-authorization policy, and content-
envelope wrapping centrally; this source does *not* render prose
beyond a short harness-controlled summary line.

The Notifications API requires a personal access token (classic) with
the ``notifications`` scope.  GitHub App installation tokens cannot
access this API.

See `Notifications <https://docs.github.com/en/rest/activity/notifications>`_.
"""

from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

import httpx
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

# GitHub notification "reason" values that represent direct
# message-like activity from a human or bot, where the natural
# question for the agent is "should I act on what this person
# said?".  Anything not in this set is treated as structural --
# something the agent should know about but should not take
# direction from solely on the basis of.  See:
# https://docs.github.com/en/rest/activity/notifications#about-notification-reasons
_CONVERSATIONAL_REASONS: frozenset[str] = frozenset({
    "mention",
    "comment",
    "review_requested",
    "team_mention",
})


def _classify_reason(reason: str) -> EventKind:
    """Map a GitHub notification ``reason`` to an :class:`EventKind`.

    Defaults to :attr:`EventKind.STRUCTURAL` for unknown reasons --
    the conservative choice, since structural events are still
    delivered (with a non-peer banner) and structural-from-non-peer
    is the carve-out the operator can disable per forge.  A novel
    conversational reason being treated as structural means the
    agent sees one extra "non-peer mentioned you" banner; the
    inverse miscategorisation (treating a novel structural reason
    as conversational) would silently drop legitimate events.
    """
    if reason in _CONVERSATIONAL_REASONS:
        return EventKind.CONVERSATIONAL
    return EventKind.STRUCTURAL


def _kind_for_subject(subject_type: str, reason: str) -> ContextItemKind:
    """Classify a context item by its containing subject type."""
    if reason in _CONVERSATIONAL_REASONS:
        return ContextItemKind.COMMENT
    if subject_type == "Issue":
        return ContextItemKind.ISSUE_BODY
    if subject_type == "PullRequest":
        return ContextItemKind.PR_BODY
    return ContextItemKind.COMMENT


def _actor_from_user(
    user: dict[str, Any] | None,
    *,
    service: str,
) -> ActorIdentity | None:
    """Build an :class:`ActorIdentity` from a GitHub ``user`` object.

    Returns ``None`` when *user* is missing or contains neither an
    immutable id nor a login.  The platform-provided ``type`` field
    (``"User"`` / ``"Bot"``) becomes the ``is_bot`` flag; absence
    of that key leaves ``is_bot`` as ``None`` (i.e. unknown), not
    ``False``, so policy code can distinguish "platform said this
    is a human" from "platform did not say."
    """
    if not user:
        return None
    user_id = user.get("id")
    login = (user.get("login") or "").strip()
    if user_id is None and not login:
        return None

    account_id = str(user_id) if user_id is not None else login
    secondary: tuple[str, ...] = (
        (login,) if login and login != account_id else ()
    )

    is_bot: bool | None = None
    if "type" in user and user.get("type") is not None:
        is_bot = user.get("type") == "Bot"

    return ActorIdentity(
        service=service,
        account_id=account_id,
        display_name=login,
        is_bot=is_bot,
        secondary_account_ids=secondary,
    )


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
    forge_name: str = Field(
        default="github",
        description=(
            "Service name stamped on every "
            "``ActorIdentity.service`` produced by this source.  "
            "Should match the ``forges[].name`` entry in "
            "``gateway.json`` so that peer matching and per-forge "
            "trigger policy line up.  Defaulted to ``\"github\"`` "
            "for back-compat with tests and bare-bones setups."
        ),
    )


# ---------------------------------------------------------------------------
# Latest-comment payload
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _LatestCommentInfo:
    """The fields of a GitHub comment / issue / PR payload we care about.

    GitHub's ``latest_comment_url`` points at one of three kinds of
    resource depending on the thread state: a per-comment endpoint, an
    issue endpoint (when the trigger was the issue body), or a PR
    endpoint.  All three share the relevant fields (``body``,
    ``user``, ``created_at``), so a single shape covers the cases.
    """

    body: str = ""
    user: dict[str, Any] | None = None
    created_at: str = ""


# ---------------------------------------------------------------------------
# Notification -> RawIncomingEvent
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


def _make_summary(
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
) -> str:
    """Build the harness-controlled summary line(s) for a GitHub notification.

    Source-controlled, *not* attacker-controlled: every piece of text
    here is a structural field name or a value pulled from the
    GitHub notification envelope (numeric ids, the platform's reason
    string, etc.).  Body text (which *is* attacker-controlled) is
    surfaced via :class:`ContextItem` items, never inlined here.
    """
    lines = [
        f"GitHub notification: you were {reason} on "
        f"{subject_type} \"{subject_title}\" in project {repo_full_name} "
        f"(repo_id={repo_id}).",
        "",
        f"Notification ID: {thread_id}",
        f"Reason: {reason}",
        f"Target: {subject_type} -- {subject_title}",
        f"Updated: {updated_at}",
    ]
    if clone_url:
        lines.append(f"Clone URL: {clone_url}")
    if default_branch:
        lines.append(f"Default branch: {default_branch}")
    if html_url:
        lines.append(f"Repository URL: {html_url}")
    lines.append("")
    lines.append(
        "Respond to the notification as appropriate.  The GitHub "
        "notification thread has already been marked read on your "
        "behalf.",
    )
    return "\n".join(lines)


def _make_external_key(base_url: str, thread_id: str, updated_at: str) -> str:
    """Build the source-namespaced ``external_key`` for a GitHub thread version.

    GitHub thread IDs are stable per subscription (per issue / PR /
    discussion), not per event.  A thread accumulates new activity
    over its lifetime, and each new event flips the thread back to
    "unread" and bumps its ``updated_at`` timestamp.  Two genuinely
    distinct events on the same thread therefore share a thread ID
    but have different ``updated_at`` values.

    The ``InFlightIndex`` contract (see :mod:`thorn.runtime._in_flight_index`)
    is that two events with the same ``external_key`` represent the
    same logical notification and should be deduplicated.  To match
    that contract, the key encodes a specific *version* of a thread
    (the ``updated_at`` snapshot at the time the source observed it).
    Two distinct versions of the same thread are intentionally treated
    as distinct events; they end up in the same session inbox (the
    session key is per-thread), so the agent sees both items.

    The API base URL is included so that a single agency polling
    multiple GitHub instances (e.g. github.com plus a GitHub
    Enterprise server) does not collide.
    """
    return f"github:{base_url}:thread:{thread_id}:updated:{updated_at}"


def _make_raw_event(
    *,
    thread: dict[str, Any],
    comment_info: _LatestCommentInfo,
    native_id_to_project_name: dict[str, str],
    base_url: str = "",
    forge_name: str = "github",
) -> RawIncomingEvent:
    """Convert a GitHub notification thread + comment payload into a :class:`RawIncomingEvent`.

    The returned event carries:

    - ``primary_actor`` derived from the comment payload's ``user``
      object (or ``None`` if absent), labelled with *forge_name*
      so peer matching uses the right service namespace.
    - ``items`` containing one :class:`ContextItem` for the comment
      body when present, classified as ``COMMENT`` for
      conversational reasons or as ``ISSUE_BODY`` / ``PR_BODY`` for
      structural reasons against an issue/PR subject.
    - ``kind`` derived from the notification's ``reason``.
    - ``summary`` and ``metadata`` carrying the harness-controlled
      structural information the formatter prepends to the rendered
      content.
    """
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

    actor = _actor_from_user(comment_info.user, service=forge_name)

    summary = _make_summary(
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
    )

    items: tuple[ContextItem, ...] = ()
    if comment_info.body:
        items = (
            ContextItem(
                body=comment_info.body,
                kind=_kind_for_subject(subject_type, reason),
                actor=actor,
                timestamp=comment_info.created_at,
            ),
        )

    return RawIncomingEvent(
        source="github",
        session_key=session_key,
        kind=_classify_reason(reason),
        primary_actor=actor,
        summary=summary,
        items=items,
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
        external_key=_make_external_key(base_url, thread_id, updated_at),
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
        # ``_last_modified`` is purely a transport optimisation: it lets
        # the steady-state poll get a 304 from the Notifications API
        # when nothing has changed.  Correctness does *not* depend on
        # it -- if it ever drifted, the worst case is a wasted body
        # parse, since deduplication of in-flight events is handled by
        # the gateway's :class:`~thorn.runtime._in_flight_index.InFlightIndex`,
        # and "this thread has already been processed" is tracked
        # server-side via the Notifications API's read/unread state.
        self._last_modified: str | None = None
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
            "GitHub notifications source authenticated as %s",
            user_info["login"],
        )

        drained = await asyncio.to_thread(self._drain_existing_unread)
        log.info(
            "GitHub source: drained %d pre-existing unread notification(s) "
            "so they are not delivered to the agent",
            drained,
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
        self._status_state = EventSourceStatusState.STOPPED

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
        on_event: Callable[[RawIncomingEvent], Awaitable[None]],
    ) -> None:
        started_at = event_source_status_timestamp()
        self._last_poll_started_at = started_at
        delivered_count = 0
        try:
            new_events = await asyncio.to_thread(self._fetch_new_notifications)
            if new_events:
                log.info("Found %d new notification(s)", len(new_events))
            for ev in new_events:
                try:
                    await on_event(ev)
                except Exception:
                    # One bad event shouldn't poison the whole poll; skip
                    # mark-as-read so the thread resurfaces on the next
                    # poll and we get another shot at posting it.
                    log.exception(
                        "Failed to post event for GitHub thread %s",
                        ev.metadata.get("notification_id", "?"),
                    )
                    continue

                delivered_count += 1
                # Mark the thread read so GitHub doesn't resurface the
                # same notification on every poll.  Runs regardless of
                # whether the gateway delivered, deduplicated, or *dropped*
                # the event -- drops are terminal in the formatter's
                # contract, and we want the *platform* to stop resurfacing
                # the entity in all three cases.
                thread_id = ev.metadata.get("notification_id")
                if thread_id:
                    await asyncio.to_thread(
                        self._mark_thread_read, str(thread_id),
                    )
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

    def _drain_existing_unread(self) -> int:
        """Mark every currently-unread notification as read without emitting events.

        Called once at startup so the agent isn't flooded with whatever
        accumulated while the gateway was down.  We don't track these
        threads client-side: the read/unread bit lives on GitHub.

        Best-effort per thread: a failed PATCH on one thread is logged
        and we keep going.  Threads we fail to mark read here will
        simply resurface on the next steady-state poll and be handled
        normally (delivered + marked read) at that point.

        Returns the number of threads observed (not the number
        successfully PATCHed); the difference, if any, is visible in
        the per-thread error logs.
        """
        threads = self._fetch_unread_thread_list()
        for thread in threads:
            tid = thread.get("id")
            if not tid:
                continue
            self._mark_thread_read(str(tid))
        # The drain just consumed a (possibly cached) Last-Modified
        # snapshot; clear it so the first steady-state poll isn't
        # short-circuited to 304 against pre-drain state.
        self._last_modified = None
        return len(threads)

    def _fetch_unread_thread_list(self) -> list[dict[str, Any]]:
        """GET the current unread notifications, honouring ``If-Modified-Since``.

        Returns ``[]`` on 304.  Updates ``self._last_modified`` from
        the response when present.
        """
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

        return resp.json()

    def _fetch_new_notifications(self) -> list[RawIncomingEvent]:
        """Return one :class:`RawIncomingEvent` per currently-unread thread.

        Has no client-side dedup state: threads we have already
        processed are filtered out by GitHub's own read/unread state
        (we mark each thread read after posting, and the startup drain
        clears the pre-existing unread set).  Cross-poll deduplication
        of *in-flight* notifications is handled by the gateway's
        :class:`~thorn.runtime._in_flight_index.InFlightIndex`.
        """
        threads = self._fetch_unread_thread_list()

        incoming: list[RawIncomingEvent] = []
        for thread in threads:
            comment_info = self._fetch_latest_comment_payload(thread)
            incoming.append(
                _make_raw_event(
                    thread=thread,
                    comment_info=comment_info,
                    native_id_to_project_name=self._config.native_id_to_project_name,
                    base_url=self._config.base_url,
                    forge_name=self._config.forge_name,
                ),
            )

        # Process oldest-first so prompts follow chronological order.
        incoming.sort(key=lambda e: e.metadata.get("updated_at", ""))
        return incoming

    def _mark_thread_read(self, thread_id: str) -> None:
        """Mark a notification thread as read on GitHub.

        Best-effort: a failed PATCH is logged and swallowed.  The
        gateway's :class:`~thorn.runtime._in_flight_index.InFlightIndex`
        deduplicates a re-emit while the original event is still in
        flight, so a transient PATCH failure is harmless: the next
        poll re-fetches the (still-unread) thread, the duplicate
        ``RawIncomingEvent`` is dropped at the gateway, and the source
        retries the PATCH.

        The unhandled edge case is a *persistent* PATCH failure
        combined with the agent finishing the in-flight item before
        the next poll; that would result in a duplicate inbox entry.
        Persistent failures are operator-visible via these log lines.
        """
        try:
            resp = self._http.patch(f"/notifications/threads/{thread_id}")
            resp.raise_for_status()
        except Exception:
            log.exception(
                "Failed to mark GitHub notification thread %s as read",
                thread_id,
            )

    def _fetch_latest_comment_payload(
        self,
        thread: dict[str, Any],
    ) -> _LatestCommentInfo:
        """Fetch body / user / timestamp of the latest comment on the thread.

        Returns an empty :class:`_LatestCommentInfo` if the URL is
        missing or the request fails (best-effort).  The returned
        ``user`` is the raw GitHub user object (``id``, ``login``,
        ``type``, ...) for downstream conversion to an
        :class:`ActorIdentity`.
        """
        url = (thread.get("subject") or {}).get("latest_comment_url") or ""
        if not url:
            return _LatestCommentInfo()
        try:
            resp = self._http.get(url)
            resp.raise_for_status()
            data = resp.json() or {}
            return _LatestCommentInfo(
                body=data.get("body") or "",
                user=data.get("user"),
                created_at=data.get("created_at") or "",
            )
        except Exception:
            log.debug(
                "Failed to fetch latest comment for thread %s",
                thread.get("id", "?"),
                exc_info=True,
            )
            return _LatestCommentInfo()


__all__ = [
    "GitHubNotificationsSourceConfig",
    "GitHubNotificationsSource",
]
