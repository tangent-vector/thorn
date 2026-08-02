"""File-based session locking for CLI concurrency.

Provides a simple advisory lock on a session directory to prevent
concurrent CLI invocations from corrupting session state.  When the
desired session is already locked, the caller can either fail or
create a fresh session with a distinct key.

Uses ``fcntl.flock`` on POSIX and a lock-file-existence check on
Windows (best-effort, not bulletproof on Windows).
"""

from __future__ import annotations

import logging
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

logger = logging.getLogger(__name__)

_LOCK_FILENAME = ".lock"


class SessionLockError(Exception):
    """Raised when a session is already locked by another process."""


@contextmanager
def session_lock(session_dir: Path) -> Generator[None, None, None]:
    """Acquire an advisory lock on *session_dir* for the lifetime of the block.

    On POSIX, uses ``fcntl.flock(LOCK_EX | LOCK_NB)`` for
    non-blocking exclusive lock.  On Windows, falls back to creating a
    lock file with ``O_CREAT | O_EXCL``.

    Raises :class:`SessionLockError` if the session is already locked.
    """
    session_dir.mkdir(parents=True, exist_ok=True)
    lock_path = session_dir / _LOCK_FILENAME

    if sys.platform == "win32":
        yield from _win32_lock(lock_path)
    else:
        yield from _posix_lock(lock_path)


def _posix_lock(lock_path: Path) -> Generator[None, None, None]:
    import fcntl

    fd = os.open(str(lock_path), os.O_CREAT | os.O_WRONLY)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        os.close(fd)
        raise SessionLockError(
            f"Session is locked by another process: {lock_path.parent}"
        )
    try:
        os.write(fd, str(os.getpid()).encode())
        yield
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        except OSError:
            pass
        os.close(fd)
        try:
            lock_path.unlink(missing_ok=True)
        except OSError:
            pass


def _win32_lock(lock_path: Path) -> Generator[None, None, None]:
    try:
        fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise SessionLockError(
            f"Session is locked by another process: {lock_path.parent}"
        )
    try:
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        yield
    finally:
        try:
            lock_path.unlink(missing_ok=True)
        except OSError:
            pass


__all__ = [
    "SessionLockError",
    "session_lock",
]
