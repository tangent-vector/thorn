"""In-memory index of in-flight notification keys.

The :class:`InFlightIndex` is a thread-safe set of ``external_key``
values for notifications that currently exist anywhere in the agency
(any session inbox, any service notification queue, including their
``errored/`` subdirectories).  Event sources consult the index to
avoid posting a duplicate notification for the same external entity
(for example, a GitLab TODO whose previous notification has not yet
been fully handled).

Design properties:

- **Derived data.**  The index is not persisted separately; its
  contents are fully reconstructable from the durable queue
  filesystem.  On :class:`~thorn.runtime.Runtime` entry, the startup
  sweep rebuilds the index by scanning every queue directory.  See
  :func:`rebuild_in_flight_index`.

- **In-memory, thread-safe.**  The backing structure is a plain
  :class:`set` guarded by a :class:`threading.Lock`.  Operations are
  O(1) and fast enough to call from any code path (source polling,
  queue mutations, handler runs).

- **Updated only on post/delete.**  A notification's key enters the
  index when it is posted, and leaves when it is deleted.  Status
  mutations and moves between queues do *not* touch the index,
  because the item is still in flight -- just in a different location
  or state.

- **No TTL, no capacity limit.**  The index's size is bounded by the
  number of unhandled/in-flight notifications currently in the
  agency, which the rest of the system keeps small by construction.
  Errored RSVP-less items keep their keys in the index until an
  operator removes them (so that sources do not re-post them).

- **Opt-in from sources.**  Keys are only added for notifications
  whose :class:`~thorn.runtime.NotificationSpec` carries an
  ``external_key``.  Sources that mark their external side-effect as
  done at post time (e.g. the GitLab source that marks the TODO done
  immediately after posting) don't strictly need to supply a key, but
  doing so is cheap insurance.

Concurrency model:

- Source check-then-post is assumed to be *serialized within a single
  source*.  Two concurrent checks for the same key from a single
  source would be a source bug; we do not attempt to prevent that
  race here because fixing it requires source-level synchronization
  anyway.  Cross-source duplicates are not a concern because
  ``external_key`` values are expected to be source-namespaced (for
  example, ``"gitlab:todo:99"``).

- Crash safety: if a post completes but the process dies before the
  in-memory add runs, the next restart rebuilds the index from disk.
  Conversely, a delete that runs the file unlink before updating the
  index (the chosen order) can leak a stale key in memory; this is
  harmless because the key will simply be absent on the next rebuild.
  The *opposite* order (remove from index, then unlink) would risk
  the source posting a duplicate before the file was truly gone, so
  we carefully order file operations before index operations on
  delete.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Iterable, Iterator

from thorn.runtime._notification import Notification
from thorn.runtime._paths import AgencyPaths


class InFlightIndex:
    """Thread-safe set of ``external_key`` values for in-flight notifications.

    The index starts empty.  Typical lifecycle:

    1. Construct at :class:`~thorn.runtime.Runtime` startup.
    2. Populate by calling :func:`rebuild_in_flight_index`.
    3. Attach to each durable queue as it is created (so subsequent
       posts and deletes keep the index up to date).
    4. Consulted by sources before posting, via :meth:`__contains__`.
    """

    def __init__(self) -> None:
        self._keys: set[str] = set()
        self._lock = threading.Lock()

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, str):
            return False
        with self._lock:
            return key in self._keys

    def __len__(self) -> int:
        with self._lock:
            return len(self._keys)

    def contains(self, key: str) -> bool:
        """Return ``True`` if *key* is currently in flight."""
        with self._lock:
            return key in self._keys

    def add(self, key: str) -> None:
        """Mark *key* as in flight.

        Idempotent: adding a key that is already present is a no-op.
        """
        with self._lock:
            self._keys.add(key)

    def remove(self, key: str) -> None:
        """Remove *key* from the index.

        Forgiving: removing a key that is not present is a no-op.
        This matches the expected crash-recovery behavior where the
        filesystem is authoritative and an index lagging the
        filesystem is fine.
        """
        with self._lock:
            self._keys.discard(key)

    def snapshot(self) -> frozenset[str]:
        """Return an immutable snapshot of the current key set."""
        with self._lock:
            return frozenset(self._keys)

    def clear(self) -> None:
        """Drop all keys.  Used during rebuild before repopulating."""
        with self._lock:
            self._keys.clear()

    def bulk_add(self, keys: Iterable[str]) -> None:
        """Add every key in *keys* under a single lock acquisition."""
        with self._lock:
            self._keys.update(keys)


# ---------------------------------------------------------------------------
# Rebuild from filesystem
# ---------------------------------------------------------------------------

def rebuild_in_flight_index(paths: AgencyPaths) -> InFlightIndex:
    """Scan all queue directories under *paths* and return a fresh index.

    Walks:

    - Every session inbox (via :meth:`AgencyPaths.iter_session_inbox_dirs`),
      including each inbox's ``errored/`` subdirectory.
    - Every service notification queue
      (via :meth:`AgencyPaths.iter_service_queue_dirs`), including its
      ``errored/`` subdirectory.

    For each notification file found, the ``external_key`` (if any)
    is added to the index.  Temp-file sidecars and files that fail to
    parse as notifications are silently skipped: a crashed partial
    write should not prevent the rest of the index from rebuilding.
    """
    index = InFlightIndex()
    index.bulk_add(_iter_keys_in_agency(paths))
    return index


def _iter_keys_in_agency(paths: AgencyPaths) -> Iterator[str]:
    """Yield every ``external_key`` present in any agency queue directory."""
    for queue_dir in paths.iter_session_inbox_dirs():
        yield from _iter_keys_in_queue_dir(queue_dir)
    for queue_dir in paths.iter_service_queue_dirs():
        yield from _iter_keys_in_queue_dir(queue_dir)


def _iter_keys_in_queue_dir(root: Path) -> Iterator[str]:
    """Yield every ``external_key`` from notification files under *root*.

    Walks *root* recursively to pick up the main directory and the
    ``errored/`` subdirectory with a single pass.  Temp sidecars
    (files whose name begins with ``.``) and files that fail to parse
    are skipped without raising.
    """
    if not root.exists():
        return
    for path in root.rglob("*.json"):
        if not path.is_file():
            continue
        if path.name.startswith("."):
            # Matches the filter used by DurableQueue._is_live_file:
            # any dotfile (notably .tmp-*.json sidecars) is not a
            # committed notification and must be ignored.
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            notification = Notification.from_json(data)
        except (OSError, ValueError, KeyError):
            # A file that fails to read or parse is either an
            # operator artifact or the victim of a bizarre partial
            # write that escaped our crash-safety guarantees.  Either
            # way, do not block the rebuild over it.
            continue
        if notification.external_key is not None:
            yield notification.external_key


__all__ = [
    "InFlightIndex",
    "rebuild_in_flight_index",
]
