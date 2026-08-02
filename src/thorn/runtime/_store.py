"""Filesystem-backed store for agent identities and sessions.

The ``SessionStore`` manages persisted agent identities and their
sessions on disk.  It is a thin wrapper around
:class:`~thorn.runtime._paths.AgencyPaths`: every path it touches is
derived from that single source of truth rather than reconstructed
locally.

The on-disk shape is::

    <home_root>/
        agents/
            <safe-agent-id>/
                agent.json              # identity
                home/                   # agent-authored state (mounted)
                sessions/
                    <safe-session-key>/
                        session.json
                        history.json
                        inbox/

Identity files live *inside* each agent's framework dir (previous
versions used ``<agents_root>/<id>.json`` sibling files; Phase A moved
the identity file inside the dir so the whole agent lives under one
renameable subtree).
"""

from __future__ import annotations

import shutil
from pathlib import Path

from thorn.core._agent import Agent
from thorn.core._session import Session
from thorn.runtime._paths import (
    SESSION_STATE_DIR,
    AgencyPaths,
    safe_dirname,
    session_key_from_path,
    unsafe_dirname,
)
from thorn.runtime._serializer import (
    _SESSION_FILE,
    JsonSessionSerializer,
    SessionSerializer,
)
from thorn.runtime._session import AgentID, SessionKey

# Backward-compatible aliases for code that imported the previously-private
# helpers from this module.  New callers should import from
# ``thorn.runtime._paths`` directly.
_safe_dirname = safe_dirname
_unsafe_dirname = unsafe_dirname


class SessionStore:
    """Filesystem-backed store for agent identities and sessions.

    The store consumes an :class:`AgencyPaths` instance and uses it as
    the single source of truth for on-disk layout; nothing here
    stitches together paths from raw roots.
    """

    def __init__(
        self,
        paths: AgencyPaths,
        serializer: SessionSerializer | None = None,
    ) -> None:
        self._paths = paths
        self._serializer: SessionSerializer = serializer or JsonSessionSerializer()

    @property
    def paths(self) -> AgencyPaths:
        """The :class:`AgencyPaths` this store is layered on top of."""
        return self._paths

    @property
    def root(self) -> Path:
        """Root directory under which all agents live.

        Kept for backward-compat with older call sites that treated the
        store's ``root`` as ``<home_root>/agents``.  New code should
        reach through :attr:`paths` for the specific path it needs.
        """
        return self._paths.agents_root

    # -- agent identity persistence -----------------------------------------

    def save_agent(self, agent: Agent) -> None:
        """Persist agent identity to disk.

        The agent must have a non-None ``id``.
        """
        if agent.id is None:
            raise ValueError("Cannot save an agent without an id")
        identity_path = self._paths.agent_identity_file(agent.id)
        identity_path.parent.mkdir(parents=True, exist_ok=True)
        self._serializer.save_agent(agent, identity_path)

    def load_agent(self, agent_id: AgentID | str) -> Agent:
        """Load a previously persisted agent identity.

        The agent's ``home`` and ``workspace`` are derived from the
        layout via :class:`AgencyPaths`.  ``home`` points at the
        mounted ``home/`` subtree (``<agent_framework_dir>/home``);
        ``workspace`` points at the mounted ``workspace/`` subtree
        (``<agent_workspace_mount>``).

        Raises ``KeyError`` if no agent with the given ID exists.
        """
        if not isinstance(agent_id, AgentID):
            agent_id = AgentID(agent_id)
        identity_path = self._paths.agent_identity_file(agent_id)
        if not identity_path.exists():
            raise KeyError(f"No agent found for id {agent_id!r}")
        agent = self._serializer.load_agent(identity_path)
        agent._workspace = self._paths.agent_workspace_mount(agent_id)
        agent._workspace_resolved = True
        agent._home = self._paths.agent_home_mount(agent_id)
        agent._home_resolved = True
        return agent

    def agent_exists(self, agent_id: AgentID | str) -> bool:
        """Check whether an agent with the given ID has been persisted."""
        if not isinstance(agent_id, AgentID):
            agent_id = AgentID(agent_id)
        return self._paths.agent_identity_file(agent_id).exists()

    def list_agent_ids(self) -> list[AgentID]:
        """Return all persisted agent IDs, sorted alphabetically.

        Walks ``<agents_root>/`` and returns the IDs of every
        subdirectory that contains an ``agent.json`` file.
        """
        agents_root = self._paths.agents_root
        if not agents_root.exists():
            return []
        return sorted(
            AgentID(unsafe_dirname(entry.name))
            for entry in agents_root.iterdir()
            if entry.is_dir() and (entry / "agent.json").is_file()
        )

    def delete_agent(self, agent_id: AgentID | str) -> None:
        """Remove a persisted agent identity and all its sessions.

        No-op if the agent does not exist.  Both the framework dir
        (identity + sessions) and the workspace dir (session CWDs +
        control dir) are removed.
        """
        if not isinstance(agent_id, AgentID):
            agent_id = AgentID(agent_id)
        framework_dir = self._paths.agent_framework_dir(agent_id)
        if framework_dir.exists():
            shutil.rmtree(framework_dir)
        workspace_dir = self._paths.agent_workspace_dir(agent_id)
        if workspace_dir.exists():
            shutil.rmtree(workspace_dir)

    # -- session persistence ------------------------------------------------

    def save_session(self, session: Session) -> None:
        """Persist a session to disk.

        The session must have a non-None ``key``, and its agent must
        have a non-None ``id``.
        """
        if session.key is None:
            raise ValueError("Cannot save a session without a key")
        if session.agent.id is None:
            raise ValueError("Cannot save a session for an agent without an id")
        directory = self._paths.session_metadata_dir(session.agent.id, session.key)
        directory.mkdir(parents=True, exist_ok=True)
        self._serializer.save_session(session, directory)

    def load_session(self, agent: Agent, key: SessionKey | str) -> Session:
        """Load a previously persisted session for the given agent.

        Raises ``KeyError`` if no session with the given key exists
        under the agent.  Existence here is determined by directory
        presence so that custom serializers with non-default file
        layouts continue to work; callers that want the stronger
        "metadata is present" check should use :meth:`session_exists`.
        """
        if not isinstance(key, SessionKey):
            key = SessionKey(key)
        if agent.id is None:
            raise ValueError("Cannot load a session for an agent without an id")
        directory = self._paths.session_metadata_dir(agent.id, key)
        if not directory.exists():
            raise KeyError(f"No session found for key {key!r} under agent {agent.id!r}")
        return self._serializer.load_session(directory, agent)

    def session_exists(self, agent_id: AgentID | str, key: SessionKey | str) -> bool:
        """Check whether a session has been persisted under the given agent.

        A session "exists" when its ``session.json`` metadata file is
        present.  The bare directory is not enough: callers (notably the
        CLI's lock-then-load flow) routinely mkdir the session directory
        as a side effect of acquiring an advisory lock, and treating
        such a directory as a real session would cause ``load_session``
        to fail attempting to read non-existent metadata.

        Note: this check is specific to the default
        :class:`JsonSessionSerializer` layout.  Custom serializers that
        store their metadata under a different filename will report
        their sessions as nonexistent here -- but ``load_session``
        still succeeds for them based on directory presence alone.
        Removing that asymmetry would require extending the serializer
        protocol; for now it is documented rather than fixed.
        """
        if not isinstance(agent_id, AgentID):
            agent_id = AgentID(agent_id)
        if not isinstance(key, SessionKey):
            key = SessionKey(key)
        metadata_dir = self._paths.session_metadata_dir(agent_id, key)
        return (metadata_dir / _SESSION_FILE).is_file()

    def list_session_keys(self, agent_id: AgentID | str) -> list[SessionKey]:
        """Return all persisted session keys for the given agent, sorted.

        Mirrors :meth:`session_exists`: a session is only counted if
        its ``session.json`` is present in its ``_state/`` directory.
        Lock-only or otherwise empty session directories are skipped.
        See :meth:`session_exists` for the same caveat about custom
        serializer layouts.

        Walks the ``sessions/`` tree recursively to find every
        ``_state/`` marker, since session keys can be hierarchical
        (e.g. ``proj/issues/42``) and produce nested directories.
        """
        if not isinstance(agent_id, AgentID):
            agent_id = AgentID(agent_id)
        sessions_dir = self._paths.agent_sessions_dir(agent_id)
        if not sessions_dir.exists():
            return []
        keys: list[SessionKey] = []
        for state_dir in sessions_dir.rglob(SESSION_STATE_DIR):
            if not state_dir.is_dir() or state_dir.name != SESSION_STATE_DIR:
                continue
            if not (state_dir / _SESSION_FILE).is_file():
                continue
            rel = state_dir.parent.relative_to(sessions_dir)
            if not rel.parts or SESSION_STATE_DIR in rel.parts:
                continue
            keys.append(session_key_from_path(rel))
        return sorted(keys, key=str)

    def delete_session(self, agent_id: AgentID | str, key: SessionKey | str) -> None:
        """Remove a persisted session.

        Removes only the session's ``_state/`` directory, *not* the
        enclosing key-as-path subtree.  Hierarchical session keys
        (e.g. ``proj/issue/42``) share directories with sibling and
        child sessions; tearing down the parent path would obliterate
        unrelated sessions.

        No-op if the session does not exist.
        """
        if not isinstance(agent_id, AgentID):
            agent_id = AgentID(agent_id)
        if not isinstance(key, SessionKey):
            key = SessionKey(key)
        directory = self._paths.session_metadata_dir(agent_id, key)
        if directory.exists():
            shutil.rmtree(directory)


__all__ = [
    "SessionStore",
]
