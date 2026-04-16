"""GitHub repository activity event source.

Polls ``GET /repos/{owner}/{repo}/events``, which works with GitHub App
installation access tokens. The Notifications API does not.

See `Repository events <https://docs.github.com/en/rest/activity/events#list-repository-events>`_.

Requires ``PyGithub`` (install via ``pip install thorn[github]``).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from collections.abc import Awaitable, Callable
from typing import Any

from pydantic import Field

from thorn.gateway._event import EventSource, IncomingEvent
from thorn.gateway._routing import Noteable, NoteableKind, route_github_event
from thorn.tools._github_connection import GitHubConnectionConfig
from thorn.tools.github import build_pygithub_auth

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


class GitHubNotificationsSourceConfig(GitHubConnectionConfig):
    """GitHub API auth plus repository events poller settings."""

    repository: str = Field(
        description="Repository in owner/repo format (events are listed for this repo)",
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
    project_name: str = Field(
        default="",
        description=(
            "Logical project name for session-key routing. "
            "When set, session keys use project-name-based format "
            "instead of forge-specific repo IDs."
        ),
    )

    @classmethod
    def from_env(cls) -> GitHubNotificationsSourceConfig:
        """Load from environment (same auth rules as :meth:`GitHubConnectionConfig.from_env`)."""
        conn = GitHubConnectionConfig.from_env()
        repository = os.environ.get("THORN_GITHUB_REPOSITORY")
        if not repository:
            raise ValueError(
                "THORN_GITHUB_REPOSITORY is required for the GitHub event source "
                "(repository in owner/repo format).",
            )
        return cls(
            base_url=conn.base_url,
            auth=conn.auth,
            repository=repository,
            app_slug=os.environ.get("THORN_GITHUB_APP_SLUG", ""),
            poll_interval=int(os.environ.get("THORN_POLL_INTERVAL", "30")),
        )


# ---------------------------------------------------------------------------
# Formatting repository events
# ---------------------------------------------------------------------------



def _payload_summary(event_type: str, payload: dict[str, Any]) -> str:
    """Short human-readable summary of the event payload (best-effort)."""
    if not payload:
        return "(no payload details)"

    try:
        if event_type == "IssuesEvent" and "issue" in payload:
            issue = payload["issue"]
            num = issue.get("number")
            title = issue.get("title", "")
            action = payload.get("action", "")
            return f"Issue #{num} {action}: {title}".strip()
        if event_type == "IssueCommentEvent" and "comment" in payload:
            body = (payload.get("comment") or {}).get("body") or ""
            preview = body.replace("\n", " ")[:200]
            return f"Comment: {preview}" + ("…" if len(body) > 200 else "")
        if event_type == "PullRequestEvent" and "pull_request" in payload:
            pr = payload["pull_request"]
            num = pr.get("number")
            title = pr.get("title", "")
            action = payload.get("action", "")
            return f"Pull request #{num} {action}: {title}".strip()
        if event_type == "PushEvent":
            ref = payload.get("ref", "")
            commits = payload.get("commits") or []
            return f"Push to {ref} ({len(commits)} commit(s))"
        if event_type == "CreateEvent":
            return f"Created {payload.get('ref_type', '')} {payload.get('ref', '')}"
        if event_type == "DeleteEvent":
            return f"Deleted {payload.get('ref_type', '')} {payload.get('ref', '')}"
    except Exception:
        pass

    # Fallback: compact JSON (truncated)
    raw = json.dumps(payload, default=str)[:500]
    return raw + ("…" if len(json.dumps(payload, default=str)) > 500 else "")


def _extract_noteable(
    event_type: str,
    payload: dict[str, Any],
) -> Noteable | None:
    """Identify the issue or pull request a GitHub event pertains to.

    Returns ``None`` for events that are not scoped to a noteable
    (e.g. ``PushEvent``, ``CreateEvent``, ``DeleteEvent``), in which
    case the routing layer falls back to a per-event session key.

    On GitHub, ``IssueCommentEvent`` fires for comments on both issues
    and pull requests.  When ``payload["issue"]`` contains a
    ``pull_request`` key, the comment is on a PR, and we route to the
    change-request session so it shares history with
    ``PullRequestEvent`` / ``PullRequestReviewEvent``.
    """
    try:
        if event_type == "IssuesEvent":
            return Noteable(NoteableKind.ISSUE, payload["issue"]["number"])

        if event_type == "IssueCommentEvent":
            issue = payload["issue"]
            if issue.get("pull_request") is not None:
                return Noteable(NoteableKind.CHANGE_REQUEST, issue["number"])
            return Noteable(NoteableKind.ISSUE, issue["number"])

        if event_type in (
            "PullRequestEvent",
            "PullRequestReviewEvent",
            "PullRequestReviewCommentEvent",
        ):
            return Noteable(
                NoteableKind.CHANGE_REQUEST,
                payload["pull_request"]["number"],
            )
    except (KeyError, TypeError):
        pass

    return None


def _format_repo_event_content(
    *,
    full_name: str,
    repo_id: int,
    clone_url: str,
    default_branch: str,
    html_url: str,
    event_type: str,
    event_id: str,
    actor_login: str,
    created_at: str,
    payload: dict[str, Any],
) -> str:
    summary = _payload_summary(event_type, payload)
    lines = [
        f"GitHub repository activity ({event_type}) in {full_name} (repo_id={repo_id}).",
        "",
        f"Event ID: {event_id}",
        f"Actor: {actor_login}",
        f"Time: {created_at}",
        "",
        "Details:",
        summary,
        "",
    ]
    if clone_url:
        lines.append(f"Clone URL: {clone_url}")
    if default_branch:
        lines.append(f"Default branch: {default_branch}")
    if html_url:
        lines.append(f"Repository URL: {html_url}")
    lines.extend([
        "",
        "Review and respond as appropriate for your project workflow.",
    ])
    return "\n".join(lines)


def _make_incoming_event(
    *,
    repo: Any,
    event_type: str,
    event_id: str,
    actor_login: str,
    created_at: str,
    payload: dict[str, Any],
    project_name: str = "",
) -> IncomingEvent:
    repo_id = repo.id
    full_name = repo.full_name
    clone_url = getattr(repo, "clone_url", "") or ""
    default_branch = getattr(repo, "default_branch", "main") or "main"
    html_url = getattr(repo, "html_url", "") or ""

    content = _format_repo_event_content(
        full_name=full_name,
        repo_id=repo_id,
        clone_url=clone_url,
        default_branch=default_branch,
        html_url=html_url,
        event_type=event_type,
        event_id=event_id,
        actor_login=actor_login,
        created_at=created_at,
        payload=payload,
    )

    session_key = route_github_event(
        repo_id=repo_id,
        noteable=_extract_noteable(event_type, payload),
        event_type=event_type,
        event_id=event_id,
        project_name=project_name,
    )

    return IncomingEvent(
        source="github",
        session_key=session_key,
        content=content,
        metadata={
            "event_id": event_id,
            "event_type": event_type,
            "repo_full_name": full_name,
            "repo_id": repo_id,
            "actor_login": actor_login,
            "created_at": created_at,
            "clone_url": clone_url,
            "default_branch": default_branch,
            "html_url": html_url,
            "project_name": project_name,
        },
    )


# ---------------------------------------------------------------------------
# EventSource implementation
# ---------------------------------------------------------------------------


class GitHubNotificationsSource(EventSource):
    """Polls repository events (compatible with GitHub App installation tokens)."""

    Config = GitHubNotificationsSourceConfig

    def __init__(
        self,
        config: GitHubNotificationsSourceConfig,
        *,
        service_name: str = "",
    ) -> None:
        _require_github()
        self._config = config
        self._service_name = service_name
        self._pygithub_auth = build_pygithub_auth(config.auth)
        self._gh = _Github(  # type: ignore[misc]
            base_url=config.base_url,
            auth=self._pygithub_auth,
        )
        self._seen_event_ids: set[str] = set()
        self._primed: bool = False
        self._stop_event: asyncio.Event | None = None

    @property
    def name(self) -> str:
        return self._service_name

    async def start(
        self,
        on_event: Callable[[IncomingEvent], Awaitable[None]],
    ) -> None:
        self._stop_event = asyncio.Event()

        await asyncio.to_thread(self._check_connection)
        log.info(
            "GitHub source ready (repo=%s); polling repository events every %ds",
            self._config.repository,
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

    def _check_connection(self) -> None:
        """Confirm we can read the configured repository (installation tokens)."""
        assert _GHAuth is not None
        auth = self._pygithub_auth
        if isinstance(auth, _GHAuth.Token):
            # PAT: optional sanity check on the same repo we will poll.
            _ = self._gh.get_repo(self._config.repository)
            log.info("GitHub source: PAT verified; repository %s is readable", self._config.repository)
            return
        if isinstance(auth, _GHAuth.AppInstallationAuth):
            repo = self._gh.get_repo(self._config.repository)
            log.info(
                "GitHub source: app installation verified read access to %s",
                repo.full_name,
            )
            return
        raise TypeError(f"Unsupported PyGithub auth: {type(auth)!r}")

    async def _poll_once(
        self,
        on_event: Callable[[IncomingEvent], Awaitable[None]],
    ) -> None:
        new_events = await asyncio.to_thread(self._fetch_new_events)
        if new_events:
            log.info(
                "Found %d new repository event(s)",
                len(new_events),
            )
        for ev in new_events:
            await on_event(ev)

    def _fetch_new_events(self) -> list[IncomingEvent]:
        """Return new events since last poll (newest-first API order)."""
        repo = self._gh.get_repo(self._config.repository)
        # Up to 100 recent events (paginated list first slice).
        paginated = repo.get_events()
        try:
            batch = list(paginated[:100])
        except Exception:
            batch = []
            for i, ev in enumerate(paginated):
                if i >= 100:
                    break
                batch.append(ev)

        if not self._primed:
            for ev in batch:
                self._seen_event_ids.add(ev.id)
            self._primed = True
            log.info(
                "GitHub source: primed with %d existing event id(s); "
                "only newer activity will be delivered",
                len(self._seen_event_ids),
            )
            return []

        incoming: list[IncomingEvent] = []
        for raw in batch:
            eid = raw.id
            if eid in self._seen_event_ids:
                continue
            self._seen_event_ids.add(eid)
            actor_login = raw.actor.login if raw.actor else "unknown"
            created = raw.created_at.isoformat() if raw.created_at else ""
            payload = raw.payload if isinstance(raw.payload, dict) else {}
            incoming.append(
                _make_incoming_event(
                    repo=repo,
                    event_type=raw.type or "UnknownEvent",
                    event_id=eid,
                    actor_login=actor_login,
                    created_at=created,
                    payload=payload,
                    project_name=self._config.project_name,
                ),
            )

        # Process oldest-first so prompts follow chronological order.
        incoming.sort(key=lambda e: e.metadata.get("created_at", ""))
        return incoming


__all__ = [
    "GitHubNotificationsSourceConfig",
    "GitHubNotificationsSource",
]
