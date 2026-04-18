"""The ``DurableQueue`` primitive: per-item, crash-safe notification storage.

A ``DurableQueue`` stores each notification as a standalone JSON file
under a single root directory.  All mutations are implemented via
atomic :func:`os.replace` or :func:`os.rename`, so a process crash at
any point leaves a consistent filesystem state:

- A partial ``post`` leaves a ``.tmp-<id>.json`` sidecar and no live
  file at ``<id>.json``.  Startup sweeps discard leftover temp files.
- A partial ``update_status`` leaves the previous live file unchanged
  plus a ``.tmp-<id>.json`` sidecar that is similarly swept.
- A partial ``move_to`` leaves the file in either the source queue or
  the target queue -- never both, never neither.
- ``delete`` is a single ``unlink``.

All higher-level handling logic (the two-step handling lifecycle, the
RSVP vs. fresh distinction, retry/backoff, etc.) is built on top of
these operations.  This module owns *only* the filesystem contract.

The queue itself is stateless beyond its root path: every operation
hits disk.  No in-memory cache, no lock.  Concurrency is mediated at
a higher layer (the scheduler guarantees one mutator per session).

File layout under a queue's root directory::

    <root>/
      <ulid>.json        -- live notifications (any transient status)
      .tmp-<ulid>.json   -- write-then-rename sidecars (orphaned if crash)

Sibling directories such as ``errored/`` are *not* part of this
primitive; callers construct a separate ``DurableQueue`` pointing at
``<root>/errored`` and use :meth:`move_to` to park errored items there.

Notes:

- Every operation requires the queue's root to be writable; the
  directory is created lazily on first ``post``.
- ``_rename`` is a single method rather than an inline call so tests
  can patch it to simulate a mid-operation crash.  The plan's Phase 1
  contract requires that these seams exist for crash testing.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable

from thorn.runtime._notification import (
    Notification,
    NotificationSpec,
    NotificationStatus,
)

if TYPE_CHECKING:
    from thorn.runtime._in_flight_index import InFlightIndex


_LIVE_SUFFIX = ".json"
_TEMP_PREFIX = ".tmp-"


class DurableQueue:
    """Per-item, file-backed queue of :class:`Notification` values.

    Construct with a *root_dir* that identifies where the queue's
    files live.  The directory is created on first write; listing an
    uninitialized queue returns an empty list.

    The class is intentionally small: it knows how to post, list, get,
    mutate, move, and delete.  Anything semantically richer (the
    two-step handling dance, the RSVP flow, source-level dedup policy)
    is built on top using these operations.

    An optional :class:`~thorn.runtime.InFlightIndex` can be supplied
    at construction time.  When present, every :meth:`post` adds the
    notification's ``external_key`` to the index, and every
    :meth:`delete` removes it.  Status mutations and moves do not
    touch the index: the item is still "in flight" from a source's
    perspective, just in a different location or state.  Ordering is
    carefully chosen to be crash-safe: posts write the file first
    then update the index, and deletes unlink the file first then
    update the index, so that at no point can the index claim a key
    is in flight when no file exists, or claim a key is absent while
    a file for it still exists and could get duplicated.
    """

    def __init__(
        self,
        root_dir: Path,
        *,
        in_flight_index: "InFlightIndex | None" = None,
    ) -> None:
        self._root = Path(root_dir)
        self._in_flight_index = in_flight_index

    @property
    def root_dir(self) -> Path:
        """The directory that stores this queue's notification files."""
        return self._root

    # ------------------------------------------------------------------
    # Post

    def post(self, spec: NotificationSpec) -> Notification:
        """Promote *spec* to a ``Notification`` and persist it durably.

        Returns the freshly-created :class:`Notification` (with its
        assigned ULID and ``status=pending``).  The returned object is
        already on disk by the time this method returns normally.

        Atomicity: writes the full JSON body to a ``.tmp-<ulid>.json``
        sidecar, then renames it into place.  A crash before the
        rename leaves no live file; the sidecar is orphaned.

        If an :class:`~thorn.runtime.InFlightIndex` was supplied to
        this queue and *spec* carries an ``external_key``, the key is
        added to the index after the file has landed on disk.  A
        crash between rename and index update is harmless: the next
        rebuild repopulates the index from the filesystem.
        """
        notification = Notification.from_spec(spec)
        self._write_atomic(notification)
        if self._in_flight_index is not None and notification.external_key is not None:
            self._in_flight_index.add(notification.external_key)
        return notification

    # ------------------------------------------------------------------
    # Read

    def list(
        self,
        *,
        status: NotificationStatus | Iterable[NotificationStatus] | None = None,
    ) -> list[Notification]:
        """Return every notification currently in the queue.

        Results are sorted by ULID, which sorts them by post time.
        Pass ``status`` to filter: a single :class:`NotificationStatus`
        includes only that state; an iterable includes any matching
        state; ``None`` (default) includes everything.

        Temp-file sidecars and hidden files are skipped.  If the root
        directory does not exist, returns ``[]``.
        """
        if not self._root.exists():
            return []

        allowed: set[NotificationStatus] | None
        if status is None:
            allowed = None
        elif isinstance(status, NotificationStatus):
            allowed = {status}
        else:
            allowed = set(status)

        results: list[Notification] = []
        for path in sorted(self._root.iterdir(), key=lambda p: p.name):
            if not self._is_live_file(path):
                continue
            notification = self._load(path)
            if allowed is None or notification.status in allowed:
                results.append(notification)
        return results

    def get(self, notification_id: str) -> Notification:
        """Return the notification with *notification_id*.

        Raises ``KeyError`` if no such file exists in this queue.
        """
        path = self._path_for(notification_id)
        if not path.exists():
            raise KeyError(notification_id)
        return self._load(path)

    def __contains__(self, notification_id: object) -> bool:
        if not isinstance(notification_id, str):
            return False
        return self._path_for(notification_id).exists()

    # ------------------------------------------------------------------
    # Mutate

    def update_status(
        self,
        notification_id: str,
        status: NotificationStatus,
        **fields: Any,
    ) -> Notification:
        """Atomically change a notification's ``status`` and other fields.

        *fields* may include any field declared on
        :class:`Notification` (e.g. ``notes``, ``error_reason``,
        ``attempt_count``).  Unknown field names raise ``TypeError``
        via :func:`dataclasses.replace`.

        Atomicity: writes a new temp sidecar with the updated
        representation, then renames over the existing live file.  A
        crash before rename leaves the previous live file untouched;
        a crash during rename is not possible on POSIX because
        :func:`os.replace` is atomic.
        """
        current = self.get(notification_id)
        updated = current.with_updates(status=status, **fields)
        self._write_atomic(updated)
        return updated

    def move_to(self, notification_id: str, target: DurableQueue) -> Notification:
        """Atomically move the notification file to *target*'s directory.

        The notification's contents are preserved exactly -- this is a
        pure rename, not a mutation.  Returns the loaded notification
        (for callers who want the value; the file is already in its
        new location by the time this returns).

        Target root is created on demand.  Raises ``KeyError`` if the
        notification does not exist in this queue.  If *target* is the
        same queue as ``self``, this is a no-op that still returns the
        loaded notification.
        """
        src = self._path_for(notification_id)
        if not src.exists():
            raise KeyError(notification_id)
        if target is self or target._root == self._root:
            return self._load(src)
        target._root.mkdir(parents=True, exist_ok=True)
        dst = target._path_for(notification_id)
        self._rename(src, dst)
        return target._load(dst)

    def delete(self, notification_id: str) -> None:
        """Remove the notification's file from the queue.

        Raises ``KeyError`` if no such notification is present.  A
        crash has nothing to corrupt here: either the ``unlink`` ran
        or it didn't.

        If an :class:`~thorn.runtime.InFlightIndex` was supplied to
        this queue, the deleted notification's ``external_key`` (if
        any) is removed from the index *after* the unlink completes.
        The order matters for crash safety: unlinking first means a
        crash between the two steps leaves a stale key in the
        in-memory index, which is harmless -- the next rebuild
        recovers.  The reverse order could let a source re-post while
        the old file still exists, producing a genuine duplicate.
        """
        path = self._path_for(notification_id)
        if not path.exists():
            raise KeyError(notification_id)
        # Load the key BEFORE unlinking, because after the unlink we
        # have no way to recover it.  Tolerate files that can't be
        # parsed -- the unlink still needs to happen.
        external_key: str | None = None
        if self._in_flight_index is not None:
            try:
                external_key = self._load(path).external_key
            except (OSError, ValueError, KeyError):
                external_key = None
        path.unlink()
        if self._in_flight_index is not None and external_key is not None:
            self._in_flight_index.remove(external_key)

    # ------------------------------------------------------------------
    # Maintenance

    def cleanup_temp_files(self) -> int:
        """Remove any orphaned ``.tmp-<ulid>.json`` sidecars.

        Returns the number of files removed.  Intended to be called by
        the startup sweep after a restart; there is no point calling
        it during normal operation, because a running queue has no
        long-lived temp files.
        """
        if not self._root.exists():
            return 0
        removed = 0
        for path in self._root.iterdir():
            if path.is_file() and path.name.startswith(_TEMP_PREFIX):
                path.unlink()
                removed += 1
        return removed

    # ------------------------------------------------------------------
    # Internals

    def _path_for(self, notification_id: str) -> Path:
        return self._root / f"{notification_id}{_LIVE_SUFFIX}"

    def _temp_path_for(self, notification_id: str) -> Path:
        return self._root / f"{_TEMP_PREFIX}{notification_id}{_LIVE_SUFFIX}"

    def _is_live_file(self, path: Path) -> bool:
        if not path.is_file():
            return False
        name = path.name
        if name.startswith("."):
            # Hides temp sidecars (`.tmp-*.json`) and any other dotfiles.
            return False
        return name.endswith(_LIVE_SUFFIX)

    def _write_atomic(self, notification: Notification) -> None:
        self._root.mkdir(parents=True, exist_ok=True)
        temp_path = self._temp_path_for(notification.id)
        final_path = self._path_for(notification.id)
        payload = json.dumps(
            notification.to_json(),
            indent=2,
            ensure_ascii=False,
        ) + "\n"
        temp_path.write_text(payload, encoding="utf-8")
        # os.replace is atomic on POSIX and Windows: it either succeeds
        # and the old live file (if any) is gone, or it fails and both
        # files remain (temp and original). The test-visible seam is
        # _rename, which tests patch to simulate a mid-operation crash.
        self._rename(temp_path, final_path)

    def _rename(self, src: Path, dst: Path) -> None:
        """Atomic rename.  Extracted so tests can patch it to crash."""
        os.replace(src, dst)

    def _load(self, path: Path) -> Notification:
        data = json.loads(path.read_text(encoding="utf-8"))
        return Notification.from_json(data)


__all__ = [
    "DurableQueue",
]
