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

    Workspace: ``<workspaces_root>/github_<repo_id>_<event_type>_<event_id>/``
    when *workspaces_root* is provided; ``None`` otherwise.  Each event
    gets its own workspace directory so that concurrent sessions (e.g.
    two different issues, or an issue and a PR review) cannot clobber
    one another's working tree.
    """
    safe_type = event_type.replace(" ", "_")
    key = SessionKey(f"github_{repo_id}_{safe_type}_{event_id}")

    workspace: Path | None = None
    if workspaces_root is not None:
        workspace = workspaces_root / str(key)

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

    Workspace: ``<workspaces_root>/gitlab_<project_id>_<noteable_type>_<iid>/``
    when *workspaces_root* is provided; ``None`` otherwise.  Each
    noteable gets its own workspace directory so that concurrent
    sessions cannot clobber one another's working tree.
    """
    key = SessionKey(f"gitlab_{project_id}_{noteable_type}_{noteable_iid}")

    workspace: Path | None = None
    if workspaces_root is not None:
        workspace = workspaces_root / str(key)

    return SessionRoute(session_key=key, workspace_root=workspace)


__all__ = [
    "SessionRoute",
    "route_github_event",
    "route_gitlab_todo",
]
