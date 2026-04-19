"""Agency directory layout: home vs. workspace paths.

``AgencyPaths`` captures the two directory roots that an agency needs:

- **home_root**: where agent state lives (identities, sessions, memory,
  journals, service queues, and ``gateway.json``).  In CLI mode this is
  ``{cwd}/.thorn/`` (auto-nested inside the workspace); in gateway mode
  it is a user-chosen directory taken verbatim — typically
  ``~/.thorn/`` per the local-agency convention from the architecture
  doc — with **no** ``.thorn/`` subdirectory appended.

- **workspace_root**: where agent sessions do their work (clone
  repositories, edit files, run builds).  In CLI mode this is ``{cwd}``
  itself; in gateway mode it is a separate directory whose path is
  recorded in ``gateway.json`` (top-level ``"workspace"`` field) so that
  ``thorn serve`` can locate it from the agency home alone.

The distinction matters because CLI mode deliberately nests the home
directory *under* the workspace (``{cwd}/.thorn/`` inside the project
checkout), while gateway mode keeps the two roots fully independent —
an agency typically lives at ``~/.thorn/`` and serves a workspace
elsewhere on the filesystem.
"""

from __future__ import annotations

import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from thorn.runtime._session import AgentID, SessionKey


# ---------------------------------------------------------------------------
# Filesystem-safe name encoding
# ---------------------------------------------------------------------------

def safe_dirname(name: str) -> str:
    """Encode an identifier into a filesystem-safe directory/file name.

    Uses URL percent-encoding, keeping only alphanumerics and the
    characters ``_``, ``-``, and ``.`` unescaped.  In particular, ``/``,
    ``:``, and whitespace are percent-encoded so that arbitrary agent
    IDs, session keys, and service names map to a single flat
    directory component each.

    The round trip ``unsafe_dirname(safe_dirname(x)) == x`` holds for
    any input string.
    """
    return urllib.parse.quote(str(name), safe="_-.")


def unsafe_dirname(dirname: str) -> str:
    """Recover the original identifier from a :func:`safe_dirname` result."""
    return urllib.parse.unquote(dirname)


# ---------------------------------------------------------------------------
# AgencyPaths
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AgencyPaths:
    """Directory roots for an agency's home (state) and workspace (work).

    Provides helpers for deriving per-agent and per-session paths from
    the two roots.
    """

    home_root: Path
    """Root directory for agent state (identities, sessions, memory)."""

    workspace_root: Path
    """Root directory where agent sessions do their work."""

    @property
    def agents_root(self) -> Path:
        """Root directory for the ``SessionStore`` (agent identities + sessions)."""
        return self.home_root / "agents"

    def agent_home(self, agent_id: AgentID) -> Path:
        """Home directory for a specific agent (memory, journals, etc.).

        Lives under ``agents_root`` so that it is co-located with the
        agent's identity file and session directories.
        """
        return self.agents_root / str(agent_id)

    def session_workspace(
        self,
        agent_id: AgentID,
        session_key: SessionKey,
    ) -> Path:
        """Working directory for a specific agent session."""
        return self.workspace_root / str(agent_id) / str(session_key)

    # ------------------------------------------------------------------
    # Session persistent state (metadata + inbox)
    #
    # These paths mirror ``SessionStore``'s layout under
    # ``home_root/agents/...`` so that a session's durable inbox lives
    # next to its ``session.json`` and ``history.json``.  All
    # user-supplied identifiers are passed through :func:`safe_dirname`
    # to produce a single directory component per level.
    # ------------------------------------------------------------------

    def agent_sessions_dir(self, agent_id: AgentID) -> Path:
        """Root directory for an agent's persisted sessions.

        Returns ``<agents_root>/<agent-id>/sessions``.  Matches the
        layout used by ``SessionStore``.
        """
        return self.agents_root / safe_dirname(agent_id) / "sessions"

    def session_metadata_dir(
        self,
        agent_id: AgentID,
        session_key: SessionKey,
    ) -> Path:
        """Directory holding a session's ``session.json`` and ``history.json``.

        The inbox and any other per-session persistent state live
        under this directory as subdirectories.
        """
        return self.agent_sessions_dir(agent_id) / safe_dirname(session_key)

    def session_inbox_dir(
        self,
        agent_id: AgentID,
        session_key: SessionKey,
    ) -> Path:
        """Directory holding a session's durable inbox.

        Returns ``<session_metadata_dir>/inbox``.  The subdirectory
        ``inbox/errored/`` holds RSVP-less errored items (see
        :meth:`session_inbox_errored_dir`).
        """
        return self.session_metadata_dir(agent_id, session_key) / "inbox"

    def session_inbox_errored_dir(
        self,
        agent_id: AgentID,
        session_key: SessionKey,
    ) -> Path:
        """``errored/`` sibling of a session's inbox, for operator-inspection items."""
        return self.session_inbox_dir(agent_id, session_key) / "errored"

    # ------------------------------------------------------------------
    # Service persistent state
    #
    # Services live side-by-side with agents under the home root:
    # ``home_root/services/<service-name>/``.  Each service's
    # notification queue is a ``queue/`` subdirectory.  Future
    # service-local state (poll cursors, credentials caches, etc.)
    # will live alongside ``queue/`` inside the same service dir.
    # ------------------------------------------------------------------

    @property
    def services_root(self) -> Path:
        """Root directory for all service state (``<home_root>/services``)."""
        return self.home_root / "services"

    def service_dir(self, service_name: str) -> Path:
        """Directory for a specific service's state."""
        return self.services_root / safe_dirname(service_name)

    def service_queue_dir(self, service_name: str) -> Path:
        """Directory holding a service's notification queue.

        Returns ``<service_dir>/queue``.  Fresh arrivals and returning
        RSVPs both land here; RSVP-less errored items go to the
        ``queue/errored/`` subdirectory.
        """
        return self.service_dir(service_name) / "queue"

    def service_queue_errored_dir(self, service_name: str) -> Path:
        """``errored/`` sibling of a service's notification queue."""
        return self.service_queue_dir(service_name) / "errored"

    # ------------------------------------------------------------------
    # Filesystem enumeration
    #
    # These helpers walk the on-disk layout to discover every queue
    # directory that currently exists.  They are used by the startup
    # sweep (for crash recovery) and by the in-flight-index rebuild.
    # They intentionally rely *only* on directory structure, so they
    # do not need ``SessionStore``, an agent registry, or any
    # service-discovery mechanism to work.
    # ------------------------------------------------------------------

    def iter_session_inbox_dirs(self) -> Iterator[Path]:
        """Yield every session inbox directory that currently exists on disk.

        Walks ``<agents_root>/<agent-id>/sessions/<session-key>/inbox``
        for every agent and session directory found.  Yields the main
        inbox directory; the caller is expected to look for the
        ``errored/`` subdirectory separately (via rglob or by probing
        :meth:`session_inbox_errored_dir`) as appropriate.

        Non-existent agents, non-existent ``sessions/`` dirs, and
        missing ``inbox/`` dirs are silently skipped.
        """
        agents_root = self.agents_root
        if not agents_root.exists():
            return
        for agent_dir in agents_root.iterdir():
            if not agent_dir.is_dir():
                continue
            sessions_dir = agent_dir / "sessions"
            if not sessions_dir.is_dir():
                continue
            for session_dir in sessions_dir.iterdir():
                if not session_dir.is_dir():
                    continue
                inbox_dir = session_dir / "inbox"
                if inbox_dir.is_dir():
                    yield inbox_dir

    def iter_service_queue_dirs(self) -> Iterator[Path]:
        """Yield every service notification queue directory on disk.

        Walks ``<services_root>/<service-name>/queue`` for every
        service directory found.  Missing ``services/`` roots and
        missing ``queue/`` dirs are silently skipped.
        """
        services_root = self.services_root
        if not services_root.exists():
            return
        for service_dir in services_root.iterdir():
            if not service_dir.is_dir():
                continue
            queue_dir = service_dir / "queue"
            if queue_dir.is_dir():
                yield queue_dir

    def iter_session_inbox_locations(
        self,
    ) -> Iterator[tuple[AgentID, SessionKey, Path]]:
        """Yield ``(agent_id, session_key, inbox_dir)`` for every inbox on disk.

        Decodes the directory names via :func:`unsafe_dirname` so the
        returned identifiers are the original application-level values
        (with ``/`` and other special characters restored).  Useful for
        sweep and address-aware enumeration paths that need more than
        just the raw directory.
        """
        agents_root = self.agents_root
        if not agents_root.exists():
            return
        for agent_dir in agents_root.iterdir():
            if not agent_dir.is_dir():
                continue
            sessions_dir = agent_dir / "sessions"
            if not sessions_dir.is_dir():
                continue
            agent_id = AgentID(unsafe_dirname(agent_dir.name))
            for session_dir in sessions_dir.iterdir():
                if not session_dir.is_dir():
                    continue
                inbox_dir = session_dir / "inbox"
                if not inbox_dir.is_dir():
                    continue
                session_key = SessionKey(unsafe_dirname(session_dir.name))
                yield (agent_id, session_key, inbox_dir)

    def iter_service_queue_locations(self) -> Iterator[tuple[str, Path]]:
        """Yield ``(service_name, queue_dir)`` for every service queue on disk.

        Mirrors :meth:`iter_session_inbox_locations` for service
        queues.  The service name is decoded from its directory via
        :func:`unsafe_dirname`.
        """
        services_root = self.services_root
        if not services_root.exists():
            return
        for service_dir in services_root.iterdir():
            if not service_dir.is_dir():
                continue
            queue_dir = service_dir / "queue"
            if not queue_dir.is_dir():
                continue
            service_name = unsafe_dirname(service_dir.name)
            yield (service_name, queue_dir)

    @classmethod
    def for_cli(cls, cwd: Path) -> AgencyPaths:
        """Construct paths for CLI mode (``thorn run`` / ``thorn chat``).

        Home is ``{cwd}/.thorn/``, workspace is ``{cwd}`` itself.
        """
        return cls(
            home_root=cwd / ".thorn",
            workspace_root=cwd,
        )

    @classmethod
    def for_gateway(
        cls,
        agency_dir: Path,
        workspace_dir: Path,
    ) -> AgencyPaths:
        """Construct paths for gateway mode (``thorn serve``).

        Home and workspace are fully independent directories.
        *agency_dir* is used as ``home_root`` verbatim — no ``.thorn/``
        subdirectory is auto-nested.  *workspace_dir* is normally
        sourced from the ``"workspace"`` field of ``gateway.json``,
        possibly overridden by the ``--workspace`` CLI option.
        """
        return cls(
            home_root=agency_dir,
            workspace_root=workspace_dir,
        )


__all__ = [
    "AgencyPaths",
    "safe_dirname",
    "unsafe_dirname",
]
