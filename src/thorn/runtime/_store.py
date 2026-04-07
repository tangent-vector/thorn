"""Filesystem-backed store for agent identities and sessions.

The ``SessionStore`` manages a directory of persisted agent identities
and their sessions, using a hierarchical layout:

    <root>/
        <agent-id>.json                 -- agent identity
        <agent-id>/                     -- agent workspace area
            sessions/
                <session-key>/
                    session.json        -- session timestamps, metadata
                    history.json        -- conversation history
            ...                         -- agent's personal memory, etc.

Directory and file names are URL-encoded via :func:`_safe_dirname` so
that arbitrary identifiers map to valid filesystem paths.
"""

from __future__ import annotations

import shutil
import urllib.parse
from pathlib import Path

from thorn.core._agent import Agent
from thorn.core._session import Session
from thorn.runtime._serializer import JsonSessionSerializer, SessionSerializer
from thorn.runtime._session import AgentID, SessionKey


def _safe_dirname(key: str) -> str:
    """Encode an identifier into a filesystem-safe directory/file name."""
    return urllib.parse.quote(str(key), safe="_-.")


def _unsafe_dirname(dirname: str) -> str:
    """Recover the original identifier from an encoded name."""
    return urllib.parse.unquote(dirname)


class SessionStore:
    """Filesystem-backed store for agent identities and sessions.

    Agent identity is stored as ``<root>/<agent-id>.json``.
    Session data lives under ``<root>/<agent-id>/sessions/<session-key>/``.
    """

    def __init__(
        self,
        root: Path,
        serializer: SessionSerializer | None = None,
    ) -> None:
        self._root = root
        self._serializer: SessionSerializer = serializer or JsonSessionSerializer()

    @property
    def root(self) -> Path:
        return self._root

    # -- path helpers -------------------------------------------------------

    def _agent_identity_path(self, agent_id: AgentID) -> Path:
        return self._root / f"{_safe_dirname(agent_id)}.json"

    def _agent_dir(self, agent_id: AgentID) -> Path:
        return self._root / _safe_dirname(agent_id)

    def _sessions_dir(self, agent_id: AgentID) -> Path:
        return self._agent_dir(agent_id) / "sessions"

    def _session_dir(self, agent_id: AgentID, key: SessionKey) -> Path:
        return self._sessions_dir(agent_id) / _safe_dirname(key)

    # -- agent identity persistence -----------------------------------------

    def save_agent(self, agent: Agent) -> None:
        """Persist agent identity to disk.

        The agent must have a non-None ``id``.
        """
        if agent.id is None:
            raise ValueError("Cannot save an agent without an id")
        self._root.mkdir(parents=True, exist_ok=True)
        self._serializer.save_agent(agent, self._agent_identity_path(agent.id))

    def load_agent(self, agent_id: AgentID | str) -> Agent:
        """Load a previously persisted agent identity.

        Raises ``KeyError`` if no agent with the given ID exists.
        """
        if not isinstance(agent_id, AgentID):
            agent_id = AgentID(agent_id)
        path = self._agent_identity_path(agent_id)
        if not path.exists():
            raise KeyError(f"No agent found for id {agent_id!r}")
        return self._serializer.load_agent(path)

    def agent_exists(self, agent_id: AgentID | str) -> bool:
        """Check whether an agent with the given ID has been persisted."""
        if not isinstance(agent_id, AgentID):
            agent_id = AgentID(agent_id)
        return self._agent_identity_path(agent_id).exists()

    def list_agent_ids(self) -> list[AgentID]:
        """Return all persisted agent IDs, sorted alphabetically."""
        if not self._root.exists():
            return []
        return sorted(
            AgentID(_unsafe_dirname(p.stem))
            for p in self._root.iterdir()
            if p.is_file() and p.suffix == ".json"
        )

    def delete_agent(self, agent_id: AgentID | str) -> None:
        """Remove a persisted agent identity and all its sessions.

        No-op if the agent does not exist.
        """
        if not isinstance(agent_id, AgentID):
            agent_id = AgentID(agent_id)
        identity_path = self._agent_identity_path(agent_id)
        if identity_path.exists():
            identity_path.unlink()
        agent_dir = self._agent_dir(agent_id)
        if agent_dir.exists():
            shutil.rmtree(agent_dir)

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
        directory = self._session_dir(session.agent.id, session.key)
        directory.mkdir(parents=True, exist_ok=True)
        self._serializer.save_session(session, directory)

    def load_session(self, agent: Agent, key: SessionKey | str) -> Session:
        """Load a previously persisted session for the given agent.

        Raises ``KeyError`` if no session with the given key exists
        under the agent.
        """
        if not isinstance(key, SessionKey):
            key = SessionKey(key)
        if agent.id is None:
            raise ValueError("Cannot load a session for an agent without an id")
        directory = self._session_dir(agent.id, key)
        if not directory.exists():
            raise KeyError(f"No session found for key {key!r} under agent {agent.id!r}")
        return self._serializer.load_session(directory, agent)

    def session_exists(self, agent_id: AgentID | str, key: SessionKey | str) -> bool:
        """Check whether a session exists under the given agent."""
        if not isinstance(agent_id, AgentID):
            agent_id = AgentID(agent_id)
        if not isinstance(key, SessionKey):
            key = SessionKey(key)
        return self._session_dir(agent_id, key).is_dir()

    def list_session_keys(self, agent_id: AgentID | str) -> list[SessionKey]:
        """Return all persisted session keys for the given agent, sorted."""
        if not isinstance(agent_id, AgentID):
            agent_id = AgentID(agent_id)
        sessions_dir = self._sessions_dir(agent_id)
        if not sessions_dir.exists():
            return []
        return sorted(
            SessionKey(_unsafe_dirname(d.name))
            for d in sessions_dir.iterdir()
            if d.is_dir()
        )

    def delete_session(self, agent_id: AgentID | str, key: SessionKey | str) -> None:
        """Remove a persisted session.

        No-op if the session does not exist.
        """
        if not isinstance(agent_id, AgentID):
            agent_id = AgentID(agent_id)
        if not isinstance(key, SessionKey):
            key = SessionKey(key)
        directory = self._session_dir(agent_id, key)
        if directory.exists():
            shutil.rmtree(directory)


__all__ = [
    "SessionStore",
]
