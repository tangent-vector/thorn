"""Unit tests for the in-flight-key index and path enumerators.

Covers:

- :class:`AgencyPaths` additions for session inbox and service queue
  paths, plus their filesystem enumerators.
- :class:`InFlightIndex` as a plain thread-safe set-of-strings:
  contains, add, remove, idempotence, length, snapshot immutability.
- Wiring between the index and :class:`DurableQueue`,
  :class:`SessionInbox`, :class:`NotificationQueue`: post adds,
  delete removes, status mutation does not touch the index, move_to
  does not touch the index.
- :func:`rebuild_in_flight_index` repopulates the index from the
  filesystem across session inboxes, service queues, and their
  ``errored/`` subdirectories, and tolerates unparseable artifacts.
"""

from __future__ import annotations

from pathlib import Path

from thorn.core._agent import Agent
from thorn.runtime._address import ServiceAddress, SessionAddress
from thorn.runtime._in_flight_index import (
    InFlightIndex,
    rebuild_in_flight_index,
)
from thorn.runtime._inbox import SessionInbox
from thorn.runtime._notification import (
    NotificationSpec,
    NotificationStatus,
)
from thorn.runtime._notification_queue import NotificationQueue
from thorn.runtime._paths import (
    AgencyPaths,
    safe_dirname,
    unsafe_dirname,
)
from thorn.runtime._queue import DurableQueue
from thorn.runtime._session import AgentID, SessionKey
from thorn.runtime._store import SessionStore

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _paths(tmp_path: Path) -> AgencyPaths:
    return AgencyPaths.for_cli(tmp_path)


def _spec(
    target,
    *,
    external_key: str | None = None,
    content: str = "hello",
) -> NotificationSpec:
    return NotificationSpec(
        source="test",
        content=content,
        target=target,
        rsvp_to=None,
        external_key=external_key,
    )


# ---------------------------------------------------------------------------
# AgencyPaths: encoding and new paths
# ---------------------------------------------------------------------------

class TestSafeDirname:
    """``safe_dirname`` / ``unsafe_dirname`` round-trip special characters."""

    def test_passthrough_simple(self) -> None:
        assert safe_dirname("bot") == "bot"
        assert safe_dirname("agent-1.v2_final") == "agent-1.v2_final"

    def test_encodes_slash(self) -> None:
        encoded = safe_dirname("proj/mr/42")
        assert "/" not in encoded
        assert unsafe_dirname(encoded) == "proj/mr/42"

    def test_encodes_colon_and_space(self) -> None:
        encoded = safe_dirname("gitlab:demo group:issue 99")
        assert ":" not in encoded
        assert " " not in encoded
        assert unsafe_dirname(encoded) == "gitlab:demo group:issue 99"

    def test_roundtrip_arbitrary(self) -> None:
        original = "strange/name:with spaces&punct?!"
        assert unsafe_dirname(safe_dirname(original)) == original


class TestSessionPaths:
    """New per-session paths under ``AgencyPaths``."""

    def test_session_metadata_dir_matches_store_layout(self, tmp_path: Path) -> None:
        # Matches SessionStore.save_session: framework files live under
        # ``<sessions>/<key-as-path>/_state/`` so the inbox sits
        # alongside session.json/history.json for the same session.
        from thorn.runtime._paths import (
            SESSION_STATE_DIR,
            session_key_path,
        )
        paths = _paths(tmp_path)
        agent = AgentID("coord")
        session = SessionKey("proj/mr/42")
        expected = (
            tmp_path
            / ".thorn"
            / "agents"
            / safe_dirname(agent)
            / "sessions"
            / session_key_path(session)
            / SESSION_STATE_DIR
        )
        assert paths.session_metadata_dir(agent, session) == expected

    def test_session_metadata_dir_agrees_with_session_store(self, tmp_path: Path) -> None:
        # Cross-check against the actual SessionStore path computation
        # so that a future refactor of one and not the other would fail.

        paths = _paths(tmp_path)
        store = SessionStore(paths)
        agent = AgentID("coord")
        session = SessionKey("proj/mr/42")
        # Cross-check: the store's save/load paths land under the same
        # ``session_metadata_dir`` that ``AgencyPaths`` advertises.
        store_agent = Agent(id=agent, name="coord")
        store.save_agent(store_agent)
        from thorn.core._session import Session as _Session
        store.save_session(_Session(agent=store_agent, key=session))
        assert (paths.session_metadata_dir(agent, session) / "session.json").is_file()

    def test_session_inbox_dir_is_inside_metadata(self, tmp_path: Path) -> None:
        paths = _paths(tmp_path)
        agent = AgentID("coord")
        session = SessionKey("proj/mr/42")
        inbox = paths.session_inbox_dir(agent, session)
        assert inbox.name == "inbox"
        assert inbox.parent == paths.session_metadata_dir(agent, session)

    def test_session_inbox_errored_dir(self, tmp_path: Path) -> None:
        paths = _paths(tmp_path)
        agent = AgentID("coord")
        session = SessionKey("proj/mr/42")
        errored = paths.session_inbox_errored_dir(agent, session)
        assert errored.name == "errored"
        assert errored.parent == paths.session_inbox_dir(agent, session)

    def test_agent_sessions_dir(self, tmp_path: Path) -> None:
        paths = _paths(tmp_path)
        agent = AgentID("coord")
        assert paths.agent_sessions_dir(agent) == (
            tmp_path / ".thorn" / "agents" / safe_dirname(agent) / "sessions"
        )


class TestServicePaths:
    """New service paths under ``AgencyPaths``."""

    def test_services_root(self, tmp_path: Path) -> None:
        paths = _paths(tmp_path)
        assert paths.services_root == tmp_path / ".thorn" / "services"

    def test_service_dir_encoded(self, tmp_path: Path) -> None:
        paths = _paths(tmp_path)
        assert paths.service_dir("gitlab-primary") == (
            paths.services_root / "gitlab-primary"
        )
        # Service names with special chars get encoded.
        assert paths.service_dir("x:y/z") == paths.services_root / safe_dirname("x:y/z")

    def test_service_queue_dir(self, tmp_path: Path) -> None:
        paths = _paths(tmp_path)
        q = paths.service_queue_dir("gitlab-primary")
        assert q.name == "queue"
        assert q.parent == paths.service_dir("gitlab-primary")

    def test_service_queue_errored_dir(self, tmp_path: Path) -> None:
        paths = _paths(tmp_path)
        errored = paths.service_queue_errored_dir("gitlab-primary")
        assert errored.name == "errored"
        assert errored.parent == paths.service_queue_dir("gitlab-primary")


class TestPathEnumeration:
    """Filesystem enumerators for session inboxes and service queues."""

    def test_iter_session_inbox_dirs_empty_when_root_absent(self, tmp_path: Path) -> None:
        paths = _paths(tmp_path)
        assert list(paths.iter_session_inbox_dirs()) == []

    def test_iter_session_inbox_dirs_finds_created_inboxes(self, tmp_path: Path) -> None:
        paths = _paths(tmp_path)
        a1, s1 = AgentID("coord"), SessionKey("s1")
        a2, s2a = AgentID("worker"), SessionKey("task/7")
        a2, s2b = AgentID("worker"), SessionKey("task/8")
        inbox_paths = [
            paths.session_inbox_dir(a1, s1),
            paths.session_inbox_dir(a2, s2a),
            paths.session_inbox_dir(a2, s2b),
        ]
        for p in inbox_paths:
            p.mkdir(parents=True)
        found = set(paths.iter_session_inbox_dirs())
        assert found == set(inbox_paths)

    def test_iter_session_inbox_dirs_skips_non_inbox_siblings(self, tmp_path: Path) -> None:
        paths = _paths(tmp_path)
        agent, session = AgentID("coord"), SessionKey("s1")
        session_dir = paths.session_metadata_dir(agent, session)
        session_dir.mkdir(parents=True)
        # No inbox yet; session exists but should not be yielded.
        assert list(paths.iter_session_inbox_dirs()) == []
        # After the inbox shows up, the enumerator sees it.
        paths.session_inbox_dir(agent, session).mkdir()
        assert list(paths.iter_session_inbox_dirs()) == [
            paths.session_inbox_dir(agent, session)
        ]

    def test_iter_service_queue_dirs_empty_when_root_absent(self, tmp_path: Path) -> None:
        paths = _paths(tmp_path)
        assert list(paths.iter_service_queue_dirs()) == []

    def test_iter_service_queue_dirs_finds_created_queues(self, tmp_path: Path) -> None:
        paths = _paths(tmp_path)
        for name in ("gitlab-primary", "github-primary"):
            paths.service_queue_dir(name).mkdir(parents=True)
        found = set(paths.iter_service_queue_dirs())
        assert found == {
            paths.service_queue_dir("gitlab-primary"),
            paths.service_queue_dir("github-primary"),
        }

    def test_iter_service_queue_dirs_skips_service_without_queue(self, tmp_path: Path) -> None:
        paths = _paths(tmp_path)
        # Service dir exists, but no queue subdir.
        paths.service_dir("bare").mkdir(parents=True)
        assert list(paths.iter_service_queue_dirs()) == []


# ---------------------------------------------------------------------------
# InFlightIndex basics
# ---------------------------------------------------------------------------

class TestInFlightIndexBasics:
    def test_starts_empty(self) -> None:
        idx = InFlightIndex()
        assert len(idx) == 0
        assert "anything" not in idx
        assert idx.snapshot() == frozenset()

    def test_add_and_contains(self) -> None:
        idx = InFlightIndex()
        idx.add("gitlab:todo:1")
        assert "gitlab:todo:1" in idx
        assert idx.contains("gitlab:todo:1")
        assert len(idx) == 1

    def test_add_is_idempotent(self) -> None:
        idx = InFlightIndex()
        idx.add("k")
        idx.add("k")
        assert len(idx) == 1

    def test_remove_forgiving_when_absent(self) -> None:
        idx = InFlightIndex()
        # Removing a missing key must not raise; crash-recovery
        # relies on this being safe to call unconditionally.
        idx.remove("missing")
        assert len(idx) == 0

    def test_remove_after_add(self) -> None:
        idx = InFlightIndex()
        idx.add("k")
        idx.remove("k")
        assert "k" not in idx

    def test_contains_type_safety(self) -> None:
        idx = InFlightIndex()
        idx.add("k")
        # Non-string membership test must return False, not raise.
        assert (42 in idx) is False
        assert (None in idx) is False

    def test_snapshot_is_immutable_copy(self) -> None:
        idx = InFlightIndex()
        idx.add("k1")
        snap = idx.snapshot()
        idx.add("k2")
        # The snapshot is a frozenset, independent of subsequent mutation.
        assert snap == frozenset({"k1"})
        assert isinstance(snap, frozenset)

    def test_clear(self) -> None:
        idx = InFlightIndex()
        idx.bulk_add(["a", "b", "c"])
        assert len(idx) == 3
        idx.clear()
        assert len(idx) == 0

    def test_bulk_add(self) -> None:
        idx = InFlightIndex()
        idx.bulk_add(iter(["a", "b", "c", "a"]))
        assert idx.snapshot() == frozenset({"a", "b", "c"})


# ---------------------------------------------------------------------------
# DurableQueue wiring: base primitive honors the index hook
# ---------------------------------------------------------------------------

class TestDurableQueueIndexWiring:
    def test_post_adds_key_when_present(self, tmp_path: Path) -> None:
        idx = InFlightIndex()
        queue = DurableQueue(tmp_path / "q", in_flight_index=idx)
        queue.post(_spec(ServiceAddress("svc"), external_key="k1"))
        assert "k1" in idx

    def test_post_without_key_is_ignored(self, tmp_path: Path) -> None:
        idx = InFlightIndex()
        queue = DurableQueue(tmp_path / "q", in_flight_index=idx)
        queue.post(_spec(ServiceAddress("svc"), external_key=None))
        assert len(idx) == 0

    def test_post_without_index_is_a_noop(self, tmp_path: Path) -> None:
        # Ensure no index keeps the queue working exactly as before.
        queue = DurableQueue(tmp_path / "q")
        n = queue.post(_spec(ServiceAddress("svc"), external_key="k1"))
        assert queue.get(n.id).external_key == "k1"

    def test_delete_removes_key(self, tmp_path: Path) -> None:
        idx = InFlightIndex()
        queue = DurableQueue(tmp_path / "q", in_flight_index=idx)
        n = queue.post(_spec(ServiceAddress("svc"), external_key="k1"))
        queue.delete(n.id)
        assert "k1" not in idx
        assert len(idx) == 0

    def test_delete_of_keyless_item_still_succeeds(self, tmp_path: Path) -> None:
        idx = InFlightIndex()
        queue = DurableQueue(tmp_path / "q", in_flight_index=idx)
        n = queue.post(_spec(ServiceAddress("svc"), external_key=None))
        queue.delete(n.id)
        assert len(idx) == 0

    def test_update_status_does_not_touch_index(self, tmp_path: Path) -> None:
        # Status mutations preserve "in flight"-ness: the key must stay.
        idx = InFlightIndex()
        queue = DurableQueue(tmp_path / "q", in_flight_index=idx)
        n = queue.post(_spec(ServiceAddress("svc"), external_key="k1"))
        queue.update_status(n.id, NotificationStatus.HANDLED)
        assert "k1" in idx

    def test_move_to_does_not_touch_index(self, tmp_path: Path) -> None:
        # A move between queues keeps the item in flight under a
        # single shared index -- key must persist.
        idx = InFlightIndex()
        src = DurableQueue(tmp_path / "src", in_flight_index=idx)
        dst = DurableQueue(tmp_path / "dst", in_flight_index=idx)
        n = src.post(_spec(ServiceAddress("svc"), external_key="k1"))
        src.move_to(n.id, dst)
        assert "k1" in idx
        # Moves do not invoke post or delete on the destination,
        # even though the destination also references the index.
        assert len(idx) == 1


# ---------------------------------------------------------------------------
# SessionInbox / NotificationQueue forward the index parameter
# ---------------------------------------------------------------------------

class TestInboxAndQueueIndexForwarding:
    def test_session_inbox_adds_on_post(self, tmp_path: Path) -> None:
        idx = InFlightIndex()
        addr = SessionAddress(AgentID("coord"), SessionKey("s1"))
        inbox = SessionInbox(tmp_path / "in", addr, in_flight_index=idx)
        inbox.post(_spec(addr, external_key="k1"))
        assert "k1" in idx

    def test_session_inbox_removes_on_delete(self, tmp_path: Path) -> None:
        idx = InFlightIndex()
        addr = SessionAddress(AgentID("coord"), SessionKey("s1"))
        inbox = SessionInbox(tmp_path / "in", addr, in_flight_index=idx)
        n = inbox.post(_spec(addr, external_key="k1"))
        inbox.delete(n.id)
        assert "k1" not in idx

    def test_notification_queue_adds_on_post(self, tmp_path: Path) -> None:
        idx = InFlightIndex()
        addr = ServiceAddress("svc")
        queue = NotificationQueue(tmp_path / "q", addr, in_flight_index=idx)
        queue.post(_spec(addr, external_key="k1"))
        assert "k1" in idx

    def test_notification_queue_removes_on_delete(self, tmp_path: Path) -> None:
        idx = InFlightIndex()
        addr = ServiceAddress("svc")
        queue = NotificationQueue(tmp_path / "q", addr, in_flight_index=idx)
        n = queue.post(_spec(addr, external_key="k1"))
        queue.delete(n.id)
        assert "k1" not in idx


# ---------------------------------------------------------------------------
# Rebuild from filesystem
# ---------------------------------------------------------------------------

class TestRebuildInFlightIndex:
    def test_empty_agency_produces_empty_index(self, tmp_path: Path) -> None:
        paths = _paths(tmp_path)
        idx = rebuild_in_flight_index(paths)
        assert isinstance(idx, InFlightIndex)
        assert len(idx) == 0

    def test_collects_keys_from_session_inboxes(self, tmp_path: Path) -> None:
        paths = _paths(tmp_path)
        agent, session = AgentID("coord"), SessionKey("s1")
        addr = SessionAddress(agent, session)
        inbox = SessionInbox(paths.session_inbox_dir(agent, session), addr)
        inbox.post(_spec(addr, external_key="k1"))
        inbox.post(_spec(addr, external_key="k2"))
        inbox.post(_spec(addr, external_key=None))

        idx = rebuild_in_flight_index(paths)
        assert idx.snapshot() == frozenset({"k1", "k2"})

    def test_collects_keys_from_service_queues(self, tmp_path: Path) -> None:
        paths = _paths(tmp_path)
        addr = ServiceAddress("gitlab-primary")
        queue = NotificationQueue(paths.service_queue_dir("gitlab-primary"), addr)
        queue.post(_spec(addr, external_key="g:todo:1"))
        queue.post(_spec(addr, external_key="g:todo:2"))

        idx = rebuild_in_flight_index(paths)
        assert idx.snapshot() == frozenset({"g:todo:1", "g:todo:2"})

    def test_collects_keys_across_both_trees(self, tmp_path: Path) -> None:
        paths = _paths(tmp_path)
        agent, session = AgentID("coord"), SessionKey("s1")
        sess_addr = SessionAddress(agent, session)
        svc_addr = ServiceAddress("gitlab-primary")

        inbox = SessionInbox(paths.session_inbox_dir(agent, session), sess_addr)
        inbox.post(_spec(sess_addr, external_key="session-key"))

        queue = NotificationQueue(paths.service_queue_dir("gitlab-primary"), svc_addr)
        queue.post(_spec(svc_addr, external_key="service-key"))

        idx = rebuild_in_flight_index(paths)
        assert idx.snapshot() == frozenset({"session-key", "service-key"})

    def test_collects_keys_from_errored_subdirs(self, tmp_path: Path) -> None:
        # Errored items stay "in flight" from a source's perspective
        # until an operator removes them, so their keys must be picked
        # up on rebuild.  We simulate the step-2 move by posting into
        # the errored directory directly (a second DurableQueue).
        paths = _paths(tmp_path)
        agent, session = AgentID("coord"), SessionKey("s1")
        sess_addr = SessionAddress(agent, session)

        main_inbox = SessionInbox(paths.session_inbox_dir(agent, session), sess_addr)
        errored_inbox = DurableQueue(
            paths.session_inbox_errored_dir(agent, session)
        )
        n = main_inbox.post(_spec(sess_addr, external_key="stuck"))
        main_inbox.update_status(n.id, NotificationStatus.ERRORED)
        main_inbox.move_to(n.id, errored_inbox)

        # Sanity: the main inbox is empty, the errored subdir holds it.
        assert main_inbox.list() == []
        assert errored_inbox.list()[0].external_key == "stuck"

        idx = rebuild_in_flight_index(paths)
        assert "stuck" in idx

    def test_ignores_temp_sidecars(self, tmp_path: Path) -> None:
        # A ".tmp-<ulid>.json" sidecar left over from a crashed write
        # must not contribute to the index; otherwise a partial post
        # would look in-flight forever.
        paths = _paths(tmp_path)
        agent, session = AgentID("coord"), SessionKey("s1")
        inbox_dir = paths.session_inbox_dir(agent, session)
        inbox_dir.mkdir(parents=True)
        (inbox_dir / ".tmp-fakeid.json").write_text(
            '{"external_key": "should-not-land"}'
        )

        idx = rebuild_in_flight_index(paths)
        assert "should-not-land" not in idx

    def test_ignores_unparseable_files(self, tmp_path: Path) -> None:
        paths = _paths(tmp_path)
        agent, session = AgentID("coord"), SessionKey("s1")
        inbox_dir = paths.session_inbox_dir(agent, session)
        inbox_dir.mkdir(parents=True)
        (inbox_dir / "not-a-notification.json").write_text("not json at all")

        # Should not raise; the scan must be forgiving.
        idx = rebuild_in_flight_index(paths)
        assert len(idx) == 0

    def test_wiring_matches_rebuild(self, tmp_path: Path) -> None:
        # End-to-end sanity: an index populated online (via wiring)
        # ends up with the same contents as one rebuilt from the
        # filesystem afterward.  Guards against wiring drift.
        paths = _paths(tmp_path)
        online = InFlightIndex()

        agent, session = AgentID("coord"), SessionKey("s1")
        sess_addr = SessionAddress(agent, session)
        inbox = SessionInbox(
            paths.session_inbox_dir(agent, session),
            sess_addr,
            in_flight_index=online,
        )
        inbox.post(_spec(sess_addr, external_key="a"))
        n_b = inbox.post(_spec(sess_addr, external_key="b"))
        inbox.post(_spec(sess_addr, external_key=None))
        inbox.delete(n_b.id)

        rebuilt = rebuild_in_flight_index(paths)
        assert rebuilt.snapshot() == online.snapshot()

    def test_rebuild_does_not_mutate_existing_queues(self, tmp_path: Path) -> None:
        # Rebuild is read-only; the notifications it scans must still
        # be present afterward.
        paths = _paths(tmp_path)
        agent, session = AgentID("coord"), SessionKey("s1")
        sess_addr = SessionAddress(agent, session)
        inbox = SessionInbox(paths.session_inbox_dir(agent, session), sess_addr)
        inbox.post(_spec(sess_addr, external_key="k"))
        listing_before = inbox.list()

        rebuild_in_flight_index(paths)
        assert [n.id for n in inbox.list()] == [n.id for n in listing_before]
