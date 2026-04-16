"""Centralized session-key derivation for gateway event sources.

Each event source (GitHub, GitLab, etc.) has a ``route_*`` function that
returns a ``SessionKey``.  Workspace paths are **not** computed here —
they are derived mechanically from the session key by the gateway (see
``Gateway._handle_event``).

Session keys use ``/`` separators and lowercase, forge-agnostic
terminology so that they double as a relative directory layout when the
gateway constructs per-session workspace paths.

When a ``project_name`` is supplied, the session key uses the project
name as the top-level scope (e.g. ``my-proj/issue/7``), making keys
stable across forge migrations.  When ``project_name`` is omitted, the
legacy forge-specific format is used (e.g. ``github/42/issue/7``) for
backward compatibility with existing sessions.
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
    project_name: str = "",
) -> SessionKey:
    """Derive a session key for a GitHub repository event.

    When *project_name* is given the key uses the project name as the
    top-level scope, dropping the forge-specific prefix::

        <project_name>/issue/<number>
        <project_name>/change-request/<number>
        <project_name>/<event_type>/<event_id>

    When *project_name* is empty (legacy mode), the key preserves the
    original format with ``github/<repo_id>/…`` for backward
    compatibility with existing persisted sessions.
    """
    scope = project_name if project_name else f"github/{repo_id}"

    if noteable is not None:
        return SessionKey(
            f"{scope}/{noteable.kind.value}/{noteable.number}"
        )

    safe_type = event_type.lower().replace(" ", "_")
    return SessionKey(f"{scope}/{safe_type}/{event_id}")


# ---------------------------------------------------------------------------
# GitLab
# ---------------------------------------------------------------------------

def route_gitlab_todo(
    *,
    project_id: int,
    noteable: Noteable,
    project_name: str = "",
) -> SessionKey:
    """Derive a session key for a GitLab TODO.

    When *project_name* is given the key uses the project name as the
    top-level scope::

        <project_name>/issue/<iid>
        <project_name>/change-request/<iid>

    When *project_name* is empty (legacy mode), the key preserves the
    original format with ``gitlab/<project_id>/…``.
    """
    scope = project_name if project_name else f"gitlab/{project_id}"

    return SessionKey(
        f"{scope}/{noteable.kind.value}/{noteable.number}"
    )


__all__ = [
    "Noteable",
    "NoteableKind",
    "route_github_event",
    "route_gitlab_todo",
]
