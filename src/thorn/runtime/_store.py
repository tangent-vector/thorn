"""Filesystem-backed session store.

The ``SessionStore`` manages a directory of persisted agents, one
subdirectory per session key.  Serialization is delegated to a
pluggable ``SessionSerializer`` (defaulting to ``JsonSessionSerializer``).

Storage convention::

    <root>/
        <session-key-1>/
            session.json
            history.json
        <session-key-2>/
            ...

Directory names are URL-encoded via :func:`_safe_dirname` so that
arbitrary session keys (including ones with ``/``, ``:``, etc.) map
to valid filesystem paths.  Well-behaved keys that stick to
alphanumerics, ``_``, ``-``, and ``.`` pass through unchanged.
"""

from __future__ import annotations

import shutil
import urllib.parse
from pathlib import Path

from thorn.core._agent import Agent
from thorn.runtime._serializer import JsonSessionSerializer, SessionSerializer
from thorn.runtime._session import SessionKey


def _safe_dirname(key: SessionKey) -> str:
    """Encode a session key into a filesystem-safe directory name."""
    return urllib.parse.quote(str(key), safe="_-.")


def _unsafe_dirname(dirname: str) -> SessionKey:
    """Recover the original session key from an encoded directory name."""
    return SessionKey(urllib.parse.unquote(dirname))


class SessionStore:
    """Filesystem-backed store for agent sessions.

    Each agent is persisted in its own subdirectory under *root*,
    named after its ``SessionKey`` (URL-encoded for filesystem safety).
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

    def _session_dir(self, key: SessionKey) -> Path:
        return self._root / _safe_dirname(key)

    def save(self, agent: Agent) -> None:
        """Persist *agent* to disk, creating directories as needed.

        The agent must have a non-None ``key``.
        """
        if agent.key is None:
            raise ValueError("Cannot save an agent without a key")
        directory = self._session_dir(agent.key)
        directory.mkdir(parents=True, exist_ok=True)
        self._serializer.save(agent, directory)

    def load(self, key: SessionKey | str) -> Agent:
        """Load a previously persisted agent.

        Raises ``KeyError`` if no agent with the given key exists.
        """
        if not isinstance(key, SessionKey):
            key = SessionKey(key)
        directory = self._session_dir(key)
        if not directory.exists():
            raise KeyError(f"No session found for key {key!r}")
        return self._serializer.load(directory)

    def exists(self, key: SessionKey | str) -> bool:
        """Check whether an agent with the given key has been persisted."""
        if not isinstance(key, SessionKey):
            key = SessionKey(key)
        return self._session_dir(key).is_dir()

    def list_keys(self) -> list[SessionKey]:
        """Return all persisted session keys, sorted alphabetically."""
        if not self._root.exists():
            return []
        return sorted(
            _unsafe_dirname(d.name)
            for d in self._root.iterdir()
            if d.is_dir()
        )

    def delete(self, key: SessionKey | str) -> None:
        """Remove a persisted agent and its directory.

        No-op if the session does not exist.
        """
        if not isinstance(key, SessionKey):
            key = SessionKey(key)
        directory = self._session_dir(key)
        if directory.exists():
            shutil.rmtree(directory)


__all__ = [
    "SessionStore",
]
