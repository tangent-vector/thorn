"""Centralized session-key derivation for gateway event sources.

Each event source (GitHub, GitLab, etc.) has a ``route_*`` function that
returns a ``SessionKey``.  Workspace paths are **not** computed here —
they are derived mechanically from the session key by the gateway (see
``Gateway._handle_event``).

Session keys use ``/`` separators and lowercase, forge-agnostic
terminology so that they double as a relative directory layout when the
gateway constructs per-session workspace paths.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass

from thorn.runtime._session import SessionKey


# ---------------------------------------------------------------------------
# Forge-agnostic noteable identity
# ---------------------------------------------------------------------------

class NoteableKind(enum.Enum):
    """The kind of forge object a session is scoped to."""

    ISSUE = "issue"
    CHANGE_REQUEST = "change-request"


@dataclass(frozen=True)
class Noteable:
    """Identity of a forge noteable (issue or change request).

    Used by event sources to tell routing which conversation thread an
    event belongs to, so that multiple events about the same noteable
    share a single session.
    """

    kind: NoteableKind
    number: int


# ---------------------------------------------------------------------------
# GitHub
# ---------------------------------------------------------------------------

def route_github_event(
    *,
    repo_id: int,
    noteable: Noteable | None = None,
    event_type: str,
    event_id: str,
) -> SessionKey:
    """Derive a session key for a GitHub repository event.

    When *noteable* is provided (issue or pull request), the key is
    scoped to that noteable so that all events about the same issue or
    PR share one session::

        github/<repo_id>/issue/<number>
        github/<repo_id>/change-request/<number>

    When *noteable* is ``None`` (e.g. ``PushEvent``, ``CreateEvent``),
    the key falls back to a per-event identifier::

        github/<repo_id>/<event_type>/<event_id>
    """
    if noteable is not None:
        return SessionKey(
            f"github/{repo_id}/{noteable.kind.value}/{noteable.number}"
        )

    safe_type = event_type.lower().replace(" ", "_")
    return SessionKey(f"github/{repo_id}/{safe_type}/{event_id}")


# ---------------------------------------------------------------------------
# GitLab
# ---------------------------------------------------------------------------

def route_gitlab_todo(
    *,
    project_id: int,
    noteable: Noteable,
) -> SessionKey:
    """Derive a session key for a GitLab TODO.

    GitLab TODOs are always tied to a noteable (issue or merge
    request), so *noteable* is required::

        gitlab/<project_id>/issue/<iid>
        gitlab/<project_id>/change-request/<iid>
    """
    return SessionKey(
        f"gitlab/{project_id}/{noteable.kind.value}/{noteable.number}"
    )


__all__ = [
    "Noteable",
    "NoteableKind",
    "route_github_event",
    "route_gitlab_todo",
]
