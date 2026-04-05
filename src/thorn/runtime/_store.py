"""Filesystem-backed session store.

The ``SessionStore`` manages a directory of persisted sessions, one
subdirectory per session key.  Serialization is delegated to a
pluggable ``SessionSerializer`` (defaulting to ``JsonSessionSerializer``).

Storage convention::

    <root>/
        <session-key-1>/
            session.json
            history.json
        <session-key-2>/
            ...
"""

from __future__ import annotations

import shutil
from pathlib import Path

from thorn.runtime._serializer import JsonSessionSerializer, SessionSerializer
from thorn.runtime._session import Session, SessionKey


class SessionStore:
    """Filesystem-backed store for agent sessions.

    Each session is persisted in its own subdirectory under *root*,
    named after its ``SessionKey``.
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
        return self._root / str(key)

    def save(self, session: Session) -> None:
        """Persist *session* to disk, creating directories as needed."""
        directory = self._session_dir(session.key)
        directory.mkdir(parents=True, exist_ok=True)
        self._serializer.save(session, directory)

    def load(self, key: SessionKey | str) -> Session:
        """Load a previously persisted session.

        Raises ``KeyError`` if no session with the given key exists.
        """
        if not isinstance(key, SessionKey):
            key = SessionKey(key)
        directory = self._session_dir(key)
        if not directory.exists():
            raise KeyError(f"No session found for key {key!r}")
        return self._serializer.load(directory)

    def exists(self, key: SessionKey | str) -> bool:
        """Check whether a session with the given key has been persisted."""
        if not isinstance(key, SessionKey):
            key = SessionKey(key)
        return self._session_dir(key).is_dir()

    def list_keys(self) -> list[SessionKey]:
        """Return all persisted session keys, sorted alphabetically."""
        if not self._root.exists():
            return []
        return sorted(
            SessionKey(d.name)
            for d in self._root.iterdir()
            if d.is_dir()
        )

    def delete(self, key: SessionKey | str) -> None:
        """Remove a persisted session and its directory.

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
