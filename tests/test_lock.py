"""Tests for thorn.runtime._lock session locking."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from thorn.runtime._lock import SessionLockError, session_lock


class TestSessionLock:
    def test_acquires_and_releases(self, tmp_path: Path):
        session_dir = tmp_path / "session"
        with session_lock(session_dir):
            assert session_dir.exists()
            lock_file = session_dir / ".lock"
            assert lock_file.exists()

    def test_creates_directory(self, tmp_path: Path):
        session_dir = tmp_path / "deep" / "nested" / "session"
        assert not session_dir.exists()
        with session_lock(session_dir):
            assert session_dir.exists()

    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX flock semantics")
    def test_concurrent_lock_raises(self, tmp_path: Path):
        session_dir = tmp_path / "session"
        with session_lock(session_dir):
            with pytest.raises(SessionLockError):
                with session_lock(session_dir):
                    pass  # should not get here

    def test_lock_released_after_block(self, tmp_path: Path):
        session_dir = tmp_path / "session"
        with session_lock(session_dir):
            pass
        # Should be able to re-acquire after release
        with session_lock(session_dir):
            pass
