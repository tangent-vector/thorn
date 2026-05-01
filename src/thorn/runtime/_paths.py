"""Agency directory layout: home vs. workspace paths.

``AgencyPaths`` captures the two directory roots that an agency needs:

- **home_root**: where framework state lives (agent identities,
  sessions, service queues, and ``gateway.json``).  In CLI mode this
  is ``{cwd}/.thorn/`` (auto-nested inside the workspace); in gateway
  mode it is a user-chosen directory taken verbatim — typically
  ``~/.thorn/`` per the local-agency convention from the architecture
  doc — with **no** ``.thorn/`` subdirectory appended.

- **workspace_root**: where agent sessions do their work (clone
  repositories, edit files, run builds).  In CLI mode this is ``{cwd}``
  itself; in gateway mode it is a separate directory whose path is
  recorded in ``gateway.json`` (top-level ``"workspace"`` field) so that
  ``thorn serve`` can locate it from the agency home alone.

Per-agent layout (Phase-A sandbox / "agent-touchable vs framework-owned"
split) lives under these two roots as::

    <home_root>/
        agents/
            <safe-agent-id>/
                agent.json              # identity (framework-owned)
                home/                   # agent-touchable, mounted
                    MEMORY.md
                    journal/
                    ...
                sessions/               # framework-owned, never mounted
                    <safe-session-key>/
                        session.json
                        history.json
                        inbox/
        services/
            <safe-service-name>/
                queue/

    <workspace_root>/
        agents/
            <safe-agent-id>/
                workspace/              # agent-touchable, mounted
                    <safe-session-key>/
                        ...             # per-session CWD
                control/                # rendezvous (socket, etc.);
                    toolhost.sock       # never agent-visible
                    toolhost.log

The `home/` and `workspace/` subdirectories of each agent are the only
parts the tool-execution sandbox sees at all.  `agent.json`, `sessions/`,
and `control/` are framework-owned and stay out of the sandbox.  This
distinction is what makes Phase B's "brain on the host, tools in a
container" split safe: compromising a tool can corrupt the mounted
subtrees, but cannot reach framework metadata or Thorn's own
coordination state.
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
# Session-key path encoding
# ---------------------------------------------------------------------------

SESSION_STATE_DIR = "_state"
"""Sentinel subdirectory wrapping a session's framework-owned files.

Per-session metadata (``session.json``, ``history.json``, ``inbox/``,
``errored/``) lives under ``<sessions>/<key-as-path>/_state/`` rather
than directly under ``<sessions>/<key-as-path>/``.  This guarantees
that hierarchical session keys cannot collide on disk with the
framework files of an enclosing session: e.g. session keys ``a/b`` and
``a/b/inbox`` both work because the former's framework files live
under ``a/b/_state/inbox/`` while the latter's live under
``a/b/inbox/_state/``.

Workspace and (future) session-key-home paths use the bare
``<key-as-path>`` with no sentinel, so the agent sees clean
unadorned directory names from inside its sandbox.
"""


def session_key_path(session_key: SessionKey) -> Path:
    """Render a :class:`SessionKey` as a relative on-disk path.

    Each component is passed through :func:`safe_dirname` to keep
    individual segments filesystem-safe, but the slashes *between*
    components are preserved as real directory separators.  So a
    session key with components ``('cli', 'foo', 'abc123')`` becomes
    the path ``cli/foo/abc123`` -- three real directory segments,
    not ``cli%2Ffoo%2Fabc123`` (one URL-quoted segment).

    This is deliberately a module-level function and not a method on
    :class:`AgencyPaths`: the encoding is a property of the session
    key alone, not of any particular agency layout.
    """
    return Path(*(safe_dirname(component) for component in session_key.components))


def session_key_from_path(rel_path: Path) -> SessionKey:
    """Recover a :class:`SessionKey` from the path produced by
    :func:`session_key_path`.

    *rel_path* must be relative; passing an absolute path is an error
    because the session-key-to-path mapping has no concept of an
    anchor.  Each path part is decoded via :func:`unsafe_dirname` and
    fed back into the :class:`SessionKey` constructor.
    """
    if rel_path.is_absolute():
        raise ValueError(
            f"session_key_from_path expects a relative path, got: {rel_path!r}"
        )
    parts = tuple(unsafe_dirname(part) for part in rel_path.parts)
    return SessionKey(parts)


# ---------------------------------------------------------------------------
# Legacy-layout detection
# ---------------------------------------------------------------------------

class LegacyLayoutError(RuntimeError):
    """Raised when an on-disk agency uses the pre-Phase-A layout.

    Phase A does not ship an in-place migrator (see the plan's
    *No automatic migration* section).  Operators are expected to
    either re-create the agency from scratch or hand-migrate.  This
    exception exists so that running against a legacy layout fails
    loudly rather than silently producing half-migrated state.
    """


# ---------------------------------------------------------------------------
# AgencyPaths
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AgencyPaths:
    """Directory roots for an agency's home (state) and workspace (work).

    Provides helpers for deriving per-agent and per-session paths from
    the two roots.  The helpers are the single source of truth for the
    on-disk layout: other subsystems (``SessionStore``, the inbox
    sweep, the tool-host daemon) obtain their paths from an
    ``AgencyPaths`` instance instead of stitching them together from
    raw roots.
    """

    home_root: Path
    """Root directory for framework state (identities, sessions, queues)."""

    workspace_root: Path
    """Root directory where agent sessions do their work."""

    # ------------------------------------------------------------------
    # Per-agent paths (home side)
    # ------------------------------------------------------------------

    @property
    def agents_root(self) -> Path:
        """Root directory under which per-agent state lives.

        Returns ``<home_root>/agents``.  Each agent owns a
        subdirectory named ``safe_dirname(agent_id)``.
        """
        return self.home_root / "agents"

    def agent_framework_dir(self, agent_id: AgentID) -> Path:
        """Per-agent framework dir: ``<agents_root>/<safe-agent-id>/``.

        Holds ``agent.json`` (identity), ``sessions/``, and the
        ``home/`` mount subtree.  Framework-owned; the container
        sandbox only sees ``home/`` within this directory.
        """
        return self.agents_root / safe_dirname(agent_id)

    def agent_identity_file(self, agent_id: AgentID) -> Path:
        """Path to an agent's persisted identity (``agent.json``).

        Lives inside :meth:`agent_framework_dir`, so renaming or
        deleting the framework dir atomically disposes of identity and
        sessions together.
        """
        return self.agent_framework_dir(agent_id) / "agent.json"

    def agent_home_mount(self, agent_id: AgentID) -> Path:
        """The ``home/`` subtree mounted into the agent's sandbox.

        Returns ``<agent_framework_dir>/home``.  Holds ``MEMORY.md``,
        ``journal/``, MCP config, and any other agent-authored state.
        This is the only part of the agent's framework dir that is
        ever exposed to tool-executing processes.
        """
        return self.agent_framework_dir(agent_id) / "home"

    def agent_home(self, agent_id: AgentID) -> Path:
        """Alias of :meth:`agent_home_mount` (historical name).

        Kept so existing call sites that say "the agent's home
        directory" continue to read naturally.  Under the Phase-A
        layout this is no longer the agent's entire framework dir;
        only the mounted ``home/`` subtree.
        """
        return self.agent_home_mount(agent_id)

    # ------------------------------------------------------------------
    # Per-agent paths (workspace side)
    # ------------------------------------------------------------------

    def agent_workspace_dir(self, agent_id: AgentID) -> Path:
        """Per-agent workspace container: ``<workspace_root>/agents/<id>/``.

        Parent of both :meth:`agent_workspace_mount` and
        :meth:`agent_control_dir`.  Rarely referenced directly by
        callers; exists so ``agents/<id>/`` has a single path helper.
        """
        return self.workspace_root / "agents" / safe_dirname(agent_id)

    def agent_workspace_mount(self, agent_id: AgentID) -> Path:
        """The ``workspace/`` subtree mounted into the agent's sandbox.

        Returns ``<agent_workspace_dir>/workspace``.  Session CWDs
        live as subdirectories here (see :meth:`session_workspace`).
        Equivalent to the container-side ``/workspace`` mount once
        Phase B lands.
        """
        return self.agent_workspace_dir(agent_id) / "workspace"

    def agent_control_dir(self, agent_id: AgentID) -> Path:
        """Per-agent brain-to-daemon rendezvous dir.

        Returns ``<agent_workspace_dir>/control``.  Holds the Unix
        socket the tool-host daemon listens on plus any other
        framework-owned per-agent rendezvous files.  Agent-invisible:
        the sandbox never mounts this as a visible path.
        """
        return self.agent_workspace_dir(agent_id) / "control"

    def agent_toolhost_socket(self, agent_id: AgentID) -> Path:
        """Path to an agent's tool-host Unix-domain socket.

        Lives under :meth:`agent_control_dir`.  The same path is used
        by the brain-side ``DaemonToolExecutor`` (to connect) and by
        the in-container daemon (to listen).
        """
        return self.agent_control_dir(agent_id) / "toolhost.sock"

    def agent_toolhost_log(self, agent_id: AgentID) -> Path:
        """Daemon-side log file, co-located with the tool-host socket.

        Separate from any brain-side logging; the brain does not
        consume this file.  It is intended for developer inspection
        when a daemon misbehaves.
        """
        return self.agent_control_dir(agent_id) / "toolhost.log"

    def session_workspace(
        self,
        agent_id: AgentID,
        session_key: SessionKey,
    ) -> Path:
        """Working directory for a specific agent session.

        Returns ``<agent_workspace_mount>/<key-as-path>``, where
        ``<key-as-path>`` is :func:`session_key_path` applied to the
        session key.  ``/`` in the key becomes a real directory
        separator; only individual segments are filesystem-quoted.
        """
        return self.agent_workspace_mount(agent_id) / session_key_path(session_key)

    # ------------------------------------------------------------------
    # Session persistent state (metadata + inbox)
    #
    # These paths live under the agent's *framework* dir, not its
    # *workspace* dir: they are bookkeeping the agent cannot touch.
    # ------------------------------------------------------------------

    def agent_sessions_dir(self, agent_id: AgentID) -> Path:
        """Root directory for an agent's persisted sessions.

        Returns ``<agent_framework_dir>/sessions``.
        """
        return self.agent_framework_dir(agent_id) / "sessions"

    def agent_sandbox_dir(self, agent_id: AgentID) -> Path:
        """Per-agent sandbox-artefact directory.

        Returns ``<agent_framework_dir>/sandbox``.  Holds gateway-
        rendered files the agent's sandbox container needs but that
        should not live under a bind-mounted subtree (so they stay
        invisible to the agent and can be mounted read-only at a
        fixed in-container path without fighting the operator's
        workspace layout).

        Today's sole inhabitant is the per-agent ephemeral
        ``gitconfig`` rendered for broker-routed git HTTPS; the
        gateway writes it during broker registration and bind-mounts
        it at :data:`~thorn.sandbox._container.CONTAINER_GIT_CONFIG_PATH`.
        """
        return self.agent_framework_dir(agent_id) / "sandbox"

    def session_state_root(
        self,
        agent_id: AgentID,
        session_key: SessionKey,
    ) -> Path:
        """The session's bare on-disk root, *without* the ``_state``
        sentinel.

        Returns ``<agent_sessions_dir>/<key-as-path>``.  This is the
        subtree the framework owns for a particular session; framework
        files themselves live one level deeper, under
        :meth:`session_metadata_dir`.

        Useful for callers that need to clean up an entire session's
        on-disk footprint at once (delete this and everything below).
        """
        return self.agent_sessions_dir(agent_id) / session_key_path(session_key)

    def session_metadata_dir(
        self,
        agent_id: AgentID,
        session_key: SessionKey,
    ) -> Path:
        """Directory holding a session's framework-owned files.

        Returns ``<session_state_root>/_state``.  ``session.json``,
        ``history.json``, and ``inbox/`` all live directly inside this
        directory.

        The ``_state`` sentinel exists so that hierarchical session
        keys cannot collide on disk: e.g. session ``a/b`` keeps its
        framework files at ``a/b/_state/`` while session ``a/b/inbox``
        keeps its at ``a/b/inbox/_state/``.  Without the sentinel,
        the second session's root would collide with the first
        session's inbox subdirectory.  See :data:`SESSION_STATE_DIR`
        for the rationale.
        """
        return self.session_state_root(agent_id, session_key) / SESSION_STATE_DIR

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

    def _iter_agent_dirs(self) -> Iterator[Path]:
        """Yield every candidate agent framework directory on disk.

        Returns every immediate subdirectory of
        :attr:`agents_root`.  Stale/spurious entries are harmless: the
        downstream probes (``sessions/``, ``inbox/``) filter out dirs
        that don't carry session state.  The explicit legacy-layout
        diagnostic lives in :meth:`detect_legacy_layout`; this helper
        is pointedly permissive so rebuild sweeps can pick up sessions
        whose ``agent.json`` was never written or was removed.
        """
        agents_root = self.agents_root
        if not agents_root.exists():
            return
        for entry in agents_root.iterdir():
            if not entry.is_dir():
                continue
            yield entry

    def _iter_session_state_dirs(
        self,
    ) -> Iterator[tuple[AgentID, SessionKey, Path]]:
        """Yield every session's ``_state`` directory currently on disk.

        Walks ``<agent_framework_dir>/sessions/`` recursively to find
        any subdirectory whose final component is :data:`SESSION_STATE_DIR`.
        Each such hit identifies one session: the session-key path is
        the rel-path from ``sessions/`` up to (but not including) the
        ``_state`` segment.

        Yields ``(agent_id, session_key, state_dir)`` triples with
        identifiers decoded from their on-disk form.  Sessions whose
        ``_state`` directory is missing or whose path includes the
        sentinel name as a non-terminal segment are silently skipped.
        """
        for agent_dir in self._iter_agent_dirs():
            sessions_dir = agent_dir / "sessions"
            if not sessions_dir.is_dir():
                continue
            agent_id = AgentID(unsafe_dirname(agent_dir.name))
            for state_dir in sessions_dir.rglob(SESSION_STATE_DIR):
                if not state_dir.is_dir():
                    continue
                if state_dir.name != SESSION_STATE_DIR:
                    continue
                rel = state_dir.parent.relative_to(sessions_dir)
                if not rel.parts:
                    # Defensive: a `_state` directory directly under
                    # `sessions/` has no enclosing key.
                    continue
                if SESSION_STATE_DIR in rel.parts:
                    # Defensive: a session-key segment that happens to
                    # be literally `_state` would round-trip
                    # ambiguously; refuse to decode such on-disk shapes
                    # rather than emit a wrong key.
                    continue
                session_key = session_key_from_path(rel)
                yield (agent_id, session_key, state_dir)

    def iter_session_inbox_dirs(self) -> Iterator[Path]:
        """Yield every session inbox directory that currently exists on disk.

        Yields ``<session_metadata_dir>/inbox`` for every session that
        carries an ``inbox/`` subdirectory.  See
        :meth:`iter_session_inbox_locations` for a richer variant that
        also yields the decoded ``(agent_id, session_key)`` pair.

        Non-existent agents, non-existent ``sessions/`` dirs, and
        sessions without an ``inbox/`` subdirectory are silently
        skipped.
        """
        for _agent_id, _session_key, state_dir in self._iter_session_state_dirs():
            inbox_dir = state_dir / "inbox"
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

        Decodes both the agent-id and the session-key from their
        on-disk path representation so that the returned identifiers
        are the original application-level values (with ``/`` between
        session-key components restored).  Useful for sweep and
        address-aware enumeration paths that need more than just the
        raw directory.
        """
        for agent_id, session_key, state_dir in self._iter_session_state_dirs():
            inbox_dir = state_dir / "inbox"
            if inbox_dir.is_dir():
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

    # ------------------------------------------------------------------
    # Legacy-layout detection
    # ------------------------------------------------------------------

    def detect_legacy_layout(self) -> list[Path]:
        """Return a list of paths that match the pre-Phase-A layout.

        The pre-Phase-A layout placed identity files at
        ``<agents_root>/<id>.json`` (sibling to the agent dir) and
        session workspaces at ``<workspace_root>/<id>/<session>/``
        (agent dir directly under ``workspace_root`` with no
        ``agents/`` prefix).  Both are recognisable without any deep
        scan.

        Returns an empty list when the on-disk shape is compatible
        with the current layout (including the "fresh install, no
        agents yet" case).
        """
        offenders: list[Path] = []
        agents_root = self.agents_root
        if agents_root.is_dir():
            # Legacy identity files: ``agents_root/<id>.json``.
            for entry in agents_root.iterdir():
                if entry.is_file() and entry.suffix == ".json":
                    offenders.append(entry)
            # Legacy agent dirs: ``agents_root/<id>/`` without
            # ``agent.json``, but with ``sessions/`` present.
            for entry in agents_root.iterdir():
                if not entry.is_dir():
                    continue
                if (entry / "agent.json").exists():
                    continue
                if (entry / "sessions").is_dir():
                    offenders.append(entry)

        # Legacy workspace layout: ``workspace_root/<id>/<session>/``
        # with no ``agents/`` prefix.  The new layout always puts
        # agent workspace dirs under ``workspace_root/agents/``, so if
        # we see dirs directly under workspace_root whose names look
        # like agent IDs *and* that match a known agent, that's a
        # legacy workspace.  We use a conservative probe: for every
        # agent we know about (via the post-migration layout), if
        # there's also a sibling dir in ``workspace_root`` with the
        # same name, flag the sibling.
        if agents_root.is_dir() and self.workspace_root.exists():
            known_agent_dirs = {
                entry.name
                for entry in agents_root.iterdir()
                if entry.is_dir() and (entry / "agent.json").is_file()
            }
            for entry in self.workspace_root.iterdir():
                if entry.is_dir() and entry.name in known_agent_dirs:
                    offenders.append(entry)

        return offenders

    def raise_if_legacy_layout(self) -> None:
        """Raise :class:`LegacyLayoutError` when legacy layout is detected.

        No-op when the layout is current or when there is no agency
        data on disk yet.
        """
        offenders = self.detect_legacy_layout()
        if not offenders:
            return
        formatted = "\n  ".join(str(p) for p in offenders)
        raise LegacyLayoutError(
            "Pre-Phase-A agency layout detected. Phase A ships no "
            "in-place migrator; re-create the agency from scratch or "
            "hand-migrate to the new layout "
            "(home_root/agents/<id>/{agent.json,home/,sessions/} + "
            "workspace_root/agents/<id>/{workspace/,control/}). "
            "Offending paths:\n  " + formatted
        )

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

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
    "LegacyLayoutError",
    "SESSION_STATE_DIR",
    "safe_dirname",
    "session_key_from_path",
    "session_key_path",
    "unsafe_dirname",
]
