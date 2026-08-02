"""Centralized session-key derivation for gateway event sources.

Each event source (GitHub, GitLab, etc.) has a ``route_*`` function that
returns a ``SessionKey``.  Workspace paths are **not** computed here —
they are derived mechanically from the session key by the gateway (see
``Gateway._handle_event``).

Session keys use ``/`` separators and lowercase, forge-agnostic object
terminology so that they double as a relative directory layout when the
gateway constructs per-session workspace paths.

When a ``project_name`` is supplied, the session key includes the
configured forge service as the top-level scope (for example,
``github/my-proj/issue/7``), so same-number noteables from different
forges or service instances do not share context accidentally.  When
``project_name`` is omitted, the legacy forge-specific format is used
(e.g. ``github/42/issue/7``) for backward compatibility with existing
sessions.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass

from thorn.runtime._session import SessionKey

# ---------------------------------------------------------------------------
# Forge-agnostic noteable identity
# ---------------------------------------------------------------------------


class ForgeServiceName(str):
    """Configured forge service name used as a route namespace."""

    def __new__(cls, value: str) -> "ForgeServiceName":
        if not value:
            raise ValueError("Forge service name must be non-empty")
        if "/" in value:
            raise ValueError(
                f"Forge service name must not contain '/': {value!r}"
            )
        return str.__new__(cls, value)


_GITHUB_SERVICE_NAME = ForgeServiceName("github")
_GITLAB_SERVICE_NAME = ForgeServiceName("gitlab")


def _scope_components(
    *,
    project_name: str,
    forge_name: ForgeServiceName,
    legacy_components: tuple[str, str],
) -> tuple[str, ...]:
    if project_name:
        return (str(forge_name), project_name)
    return legacy_components


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
    forge_name: ForgeServiceName = _GITHUB_SERVICE_NAME,
) -> SessionKey:
    """Derive a session key for a GitHub repository event.

    When *project_name* is given the key uses the configured
    *forge_name* followed by the project name as its scope::

        <forge_name>/<project_name>/issue/<number>
        <forge_name>/<project_name>/change-request/<number>
        <forge_name>/<project_name>/<event_type>/<event_id>

    When *project_name* is empty (legacy mode), the key preserves the
    original format with ``github/<repo_id>/…`` for backward
    compatibility with existing persisted sessions.
    """
    scope_components = _scope_components(
        project_name=project_name,
        forge_name=forge_name,
        legacy_components=("github", str(repo_id)),
    )

    if noteable is not None:
        return SessionKey(
            (*scope_components, noteable.kind.value, str(noteable.number))
        )

    safe_type = event_type.lower().replace(" ", "_")
    return SessionKey((*scope_components, safe_type, event_id))


# ---------------------------------------------------------------------------
# GitLab
# ---------------------------------------------------------------------------


def route_gitlab_todo(
    *,
    project_id: int,
    noteable: Noteable,
    project_name: str = "",
    forge_name: ForgeServiceName = _GITLAB_SERVICE_NAME,
) -> SessionKey:
    """Derive a session key for a GitLab TODO.

    When *project_name* is given the key uses the configured
    *forge_name* followed by the project name as its scope::

        <forge_name>/<project_name>/issue/<iid>
        <forge_name>/<project_name>/change-request/<iid>

    When *project_name* is empty (legacy mode), the key preserves the
    original format with ``gitlab/<project_id>/…``.
    """
    scope_components = _scope_components(
        project_name=project_name,
        forge_name=forge_name,
        legacy_components=("gitlab", str(project_id)),
    )

    return SessionKey(
        (*scope_components, noteable.kind.value, str(noteable.number))
    )


__all__ = [
    "ForgeServiceName",
    "Noteable",
    "NoteableKind",
    "route_github_event",
    "route_gitlab_todo",
]
