"""Centralized session-key and workspace derivation for gateway event sources.

Each event source (GitHub, GitLab, etc.) has a ``route_*`` function that
returns **both** a ``SessionKey`` and an optional workspace path in a
single call, keeping the two concerns co-located rather than scattered
across formatting helpers.

When ``workspaces_root`` is ``None``, workspace is always ``None``
(events behave as before the per-session workspace feature).  When
set, workspace paths are deterministic subdirectories under that root
using a documented naming scheme per source.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from thorn.runtime._session import SessionKey


@dataclass(frozen=True)
class SessionRoute:
    """The key + workspace pair determined for an incoming event."""

    session_key: SessionKey
    workspace_root: Path | None


# ---------------------------------------------------------------------------
# GitHub
# ---------------------------------------------------------------------------

def route_github_event(
    *,
    repo_id: int,
    event_type: str,
    event_id: str,
    workspaces_root: Path | None = None,
) -> SessionRoute:
    """Derive session key and workspace for a GitHub repository event.

    Session key: ``github_<repo_id>_<event_type>_<event_id>``
    (spaces in *event_type* are replaced with underscores).

    Workspace: ``<workspaces_root>/github_<repo_id>/`` when
    *workspaces_root* is provided; ``None`` otherwise.  All events for
    the same repository share a workspace so that issue work and PR
    review can operate on the same clone.
    """
    safe_type = event_type.replace(" ", "_")
    key = SessionKey(f"github_{repo_id}_{safe_type}_{event_id}")

    workspace: Path | None = None
    if workspaces_root is not None:
        workspace = workspaces_root / f"github_{repo_id}"

    return SessionRoute(session_key=key, workspace_root=workspace)


# ---------------------------------------------------------------------------
# GitLab
# ---------------------------------------------------------------------------

def route_gitlab_todo(
    *,
    project_id: int,
    noteable_type: str,
    noteable_iid: int,
    workspaces_root: Path | None = None,
) -> SessionRoute:
    """Derive session key and workspace for a GitLab TODO.

    Session key: ``gitlab_<project_id>_<noteable_type>_<iid>``

    Workspace: ``<workspaces_root>/gitlab_<project_id>/`` when
    *workspaces_root* is provided; ``None`` otherwise.  All TODOs for
    the same project share a workspace.
    """
    key = SessionKey(f"gitlab_{project_id}_{noteable_type}_{noteable_iid}")

    workspace: Path | None = None
    if workspaces_root is not None:
        workspace = workspaces_root / f"gitlab_{project_id}"

    return SessionRoute(session_key=key, workspace_root=workspace)


__all__ = [
    "SessionRoute",
    "route_github_event",
    "route_gitlab_todo",
]
