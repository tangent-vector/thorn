"""Unit tests for the DurableQueue primitive + addressing types.

Covers the Phase 1 "durable-queue" contract from the Session Inbox
Abstraction plan:

- ``Address`` parse/format round-trips for session and service kinds.
- ``AddressBook`` register/resolve/unregister semantics.
- ``Notification`` JSON round-trip preserves all fields.
- ``DurableQueue`` post/list/get/update_status/move_to/delete happy
  paths.
- Crash simulation at the rename seam: verify that a failure mid-way
  through post / update / move leaves the filesystem in a consistent,
  recoverable state (no torn writes, no double-presence).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from thorn.runtime._address import (
    Address,
    AddressBook,
    ServiceAddress,
    SessionAddress,
)
from thorn.runtime._notification import (
    InboxCompletionRationale,
    Notification,
    NotificationSpec,
    NotificationStatus,
)
from thorn.runtime._queue import DurableQueue
from thorn.runtime._session import AgentID, SessionKey

# ---------------------------------------------------------------------------
# Address
# ---------------------------------------------------------------------------


class TestSessionAddress:
    def test_canonical_string_form(self) -> None:
        addr = SessionAddress(AgentID("coord"), SessionKey("proj/issues/42"))
        assert str(addr) == "session:coord/proj/issues/42"

    def test_parse_round_trip(self) -> None:
        original = SessionAddress(AgentID("coord"), SessionKey("proj/issues/42"))
        parsed = Address.parse(str(original))
        assert isinstance(parsed, SessionAddress)
        assert parsed == original

    def test_parse_splits_on_first_slash_only(self) -> None:
        parsed = Address.parse("session:coord/a/b/c")
        assert isinstance(parsed, SessionAddress)
        assert parsed.agent_id == AgentID("coord")
        assert parsed.session_key == SessionKey("a/b/c")

    def test_rejects_missing_session_key(self) -> None:
        with pytest.raises(ValueError):
            Address.parse("session:coord")

    def test_rejects_empty_agent_id(self) -> None:
        with pytest.raises(ValueError):
            Address.parse("session:/key")

    def test_rejects_slash_in_agent_id(self) -> None:
        with pytest.raises(ValueError):
            SessionAddress(AgentID("bad/id"), SessionKey("k"))

    def test_rejects_colon_in_agent_id(self) -> None:
        with pytest.raises(ValueError):
            SessionAddress(AgentID("bad:id"), SessionKey("k"))

    def test_rejects_empty_session_key(self) -> None:
        with pytest.raises(ValueError):
            SessionAddress(AgentID("coord"), SessionKey(""))

    def test_equality_and_hash(self) -> None:
        a = SessionAddress(AgentID("coord"), SessionKey("k"))
        b = SessionAddress(AgentID("coord"), SessionKey("k"))
        c = SessionAddress(AgentID("coord"), SessionKey("other"))
        assert a == b
        assert a != c
        assert hash(a) == hash(b)
        assert {a, b, c} == {a, c}


class TestServiceAddress:
    def test_canonical_string_form(self) -> None:
        addr = ServiceAddress("gitlab-primary")
        assert str(addr) == "service:gitlab-primary"

    def test_parse_round_trip(self) -> None:
        original = ServiceAddress("gitlab-primary")
        parsed = Address.parse(str(original))
        assert isinstance(parsed, ServiceAddress)
        assert parsed == original

    def test_rejects_empty_name(self) -> None:
        with pytest.raises(ValueError):
            ServiceAddress("")
        with pytest.raises(ValueError):
            Address.parse("service:")

    def test_rejects_slash_in_name(self) -> None:
        with pytest.raises(ValueError):
            ServiceAddress("bad/name")

    def test_rejects_colon_in_name(self) -> None:
        with pytest.raises(ValueError):
            ServiceAddress("bad:name")


class TestAddressParseFailures:
    def test_rejects_missing_separator(self) -> None:
        with pytest.raises(ValueError):
            Address.parse("coord/key")

    def test_rejects_unknown_kind(self) -> None:
        with pytest.raises(ValueError):
            Address.parse("agent:coord")


# ---------------------------------------------------------------------------
# AddressBook
# ---------------------------------------------------------------------------


class TestAddressBook:
    def _queue(self, tmp_path: Path, name: str) -> DurableQueue:
        return DurableQueue(tmp_path / name)

    def test_register_and_resolve(self, tmp_path: Path) -> None:
        book = AddressBook()
        addr = ServiceAddress("gitlab")
        q = self._queue(tmp_path, "gitlab")
        book.register(addr, q)

        assert book.resolve(addr) is q
        assert addr in book

    def test_register_conflict_raises(self, tmp_path: Path) -> None:
        book = AddressBook()
        addr = ServiceAddress("gitlab")
        book.register(addr, self._queue(tmp_path, "a"))
        with pytest.raises(ValueError):
            book.register(addr, self._queue(tmp_path, "b"))

    def test_resolve_missing_raises_lookup_error(self) -> None:
        book = AddressBook()
        with pytest.raises(LookupError):
            book.resolve(ServiceAddress("missing"))

    def test_get_missing_returns_none(self) -> None:
        book = AddressBook()
        assert book.get(ServiceAddress("missing")) is None

    def test_unregister_removes_binding(self, tmp_path: Path) -> None:
        book = AddressBook()
        addr = ServiceAddress("gitlab")
        book.register(addr, self._queue(tmp_path, "gitlab"))
        book.unregister(addr)
        assert addr not in book
        with pytest.raises(KeyError):
            book.unregister(addr)

    def test_addresses_lists_registered(self, tmp_path: Path) -> None:
        book = AddressBook()
        a1 = ServiceAddress("s1")
        a2 = SessionAddress(AgentID("coord"), SessionKey("k"))
        book.register(a1, self._queue(tmp_path, "s1"))
        book.register(a2, self._queue(tmp_path, "s2"))
        assert set(book.addresses()) == {a1, a2}


# ---------------------------------------------------------------------------
# Notification serialization
# ---------------------------------------------------------------------------


def _sample_spec(
    *,
    rsvp: Address | None = None,
    external_key: str | None = None,
) -> NotificationSpec:
    return NotificationSpec(
        source="gitlab-poller",
        content="Please review !42",
        target=SessionAddress(AgentID("coord"), SessionKey("proj/mr/42")),
        metadata={"todo_id": 99, "url": "https://gitlab.example/todo/99"},
        rsvp_to=rsvp,
        external_key=external_key,
    )


class TestNotification:
    def test_from_spec_assigns_framework_fields(self) -> None:
        spec = _sample_spec(external_key="gitlab:todo:99")
        n = Notification.from_spec(spec)

        assert n.status is NotificationStatus.PENDING
        assert n.attempt_count == 0
        assert n.notes is None
        assert n.completion_rationale is None
        assert n.error_reason is None
        assert n.source == "gitlab-poller"
        assert n.content == spec.content
        assert n.target == spec.target
        assert n.metadata == dict(spec.metadata)
        assert n.external_key == "gitlab:todo:99"
        # ULID is 26 chars of Crockford base32.
        assert len(n.id) == 26
        assert n.posted_at.tzinfo is not None

    def test_json_round_trip_session_target_no_rsvp(self) -> None:
        n = Notification.from_spec(_sample_spec())
        data = n.to_json()
        payload = json.dumps(data)
        restored = Notification.from_json(json.loads(payload))
        assert restored == n

    def test_json_round_trip_service_rsvp(self) -> None:
        n = Notification.from_spec(
            _sample_spec(rsvp=ServiceAddress("gitlab-primary")),
        )
        rationale = InboxCompletionRationale(
            completed_actions=("opened !42",),
            request_coverage=("responded to every requested change",),
            remaining_work=(),
            self_review="checked the MR and source notification",
            external_follow_up=("posted a status comment",),
        )
        n = n.with_updates(
            status=NotificationStatus.HANDLED,
            notes="responded with comment",
            completion_rationale=rationale,
            attempt_count=1,
        )
        restored = Notification.from_json(n.to_json())
        assert restored == n
        assert restored.completion_rationale == rationale

    def test_with_updates_rejects_unknown_field(self) -> None:
        n = Notification.from_spec(_sample_spec())
        with pytest.raises(TypeError):
            n.with_updates(bogus_field="nope")

    def test_ulid_sort_order_is_monotonic(self) -> None:
        ids = [Notification.from_spec(_sample_spec()).id for _ in range(50)]
        assert ids == sorted(ids), "ULIDs should be lexically ordered by creation time"


# ---------------------------------------------------------------------------
# DurableQueue: happy paths
# ---------------------------------------------------------------------------


class TestDurableQueueHappyPath:
    def test_list_empty_when_dir_missing(self, tmp_path: Path) -> None:
        q = DurableQueue(tmp_path / "never-touched")
        assert q.list() == []

    def test_post_writes_file_and_returns_notification(self, tmp_path: Path) -> None:
        q = DurableQueue(tmp_path / "inbox")
        n = q.post(_sample_spec())

        path = q.root_dir / f"{n.id}.json"
        assert path.is_file()
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["status"] == "pending"
        assert data["source"] == "gitlab-poller"

    def test_list_orders_by_id(self, tmp_path: Path) -> None:
        q = DurableQueue(tmp_path / "inbox")
        posted = [q.post(_sample_spec()) for _ in range(5)]
        listed = q.list()
        assert [n.id for n in listed] == [n.id for n in posted]

    def test_list_filter_single_status(self, tmp_path: Path) -> None:
        q = DurableQueue(tmp_path / "inbox")
        n1 = q.post(_sample_spec())
        n2 = q.post(_sample_spec())
        q.update_status(n2.id, NotificationStatus.IN_PROGRESS)

        pending = q.list(status=NotificationStatus.PENDING)
        in_prog = q.list(status=NotificationStatus.IN_PROGRESS)

        assert [n.id for n in pending] == [n1.id]
        assert [n.id for n in in_prog] == [n2.id]

    def test_list_filter_multi_status(self, tmp_path: Path) -> None:
        q = DurableQueue(tmp_path / "inbox")
        n1 = q.post(_sample_spec())
        n2 = q.post(_sample_spec())
        q.update_status(n2.id, NotificationStatus.HANDLED, notes="done")

        in_flight = q.list(
            status=(NotificationStatus.PENDING, NotificationStatus.IN_PROGRESS),
        )
        assert [n.id for n in in_flight] == [n1.id]

    def test_get_missing_raises_key_error(self, tmp_path: Path) -> None:
        q = DurableQueue(tmp_path / "inbox")
        with pytest.raises(KeyError):
            q.get("no-such-id")

    def test_contains(self, tmp_path: Path) -> None:
        q = DurableQueue(tmp_path / "inbox")
        n = q.post(_sample_spec())
        assert n.id in q
        assert "random-id" not in q

    def test_update_status_mutates_in_place(self, tmp_path: Path) -> None:
        q = DurableQueue(tmp_path / "inbox")
        n = q.post(_sample_spec())

        updated = q.update_status(
            n.id,
            NotificationStatus.HANDLED,
            notes="all good",
            attempt_count=1,
        )

        assert updated.status is NotificationStatus.HANDLED
        assert updated.notes == "all good"
        assert updated.attempt_count == 1
        # Still in the same directory, same file name.
        assert (q.root_dir / f"{n.id}.json").is_file()

        reloaded = q.get(n.id)
        assert reloaded == updated

    def test_update_status_rejects_unknown_field(self, tmp_path: Path) -> None:
        q = DurableQueue(tmp_path / "inbox")
        n = q.post(_sample_spec())
        with pytest.raises(TypeError):
            q.update_status(n.id, NotificationStatus.HANDLED, bogus="x")

    def test_move_to_across_queues(self, tmp_path: Path) -> None:
        src = DurableQueue(tmp_path / "inbox")
        dst = DurableQueue(tmp_path / "gitlab-service")

        n = src.post(_sample_spec(rsvp=ServiceAddress("gitlab-primary")))
        moved = src.move_to(n.id, dst)

        assert not (src.root_dir / f"{n.id}.json").exists()
        assert (dst.root_dir / f"{n.id}.json").is_file()
        assert moved == n  # Content unchanged.

    def test_move_to_creates_target_dir(self, tmp_path: Path) -> None:
        src = DurableQueue(tmp_path / "inbox")
        dst = DurableQueue(tmp_path / "brand" / "new" / "dir")

        n = src.post(_sample_spec())
        src.move_to(n.id, dst)
        assert (dst.root_dir / f"{n.id}.json").is_file()

    def test_move_to_same_queue_is_noop(self, tmp_path: Path) -> None:
        q = DurableQueue(tmp_path / "inbox")
        n = q.post(_sample_spec())
        result = q.move_to(n.id, q)
        assert result == n
        assert (q.root_dir / f"{n.id}.json").is_file()

    def test_move_to_missing_raises(self, tmp_path: Path) -> None:
        src = DurableQueue(tmp_path / "a")
        dst = DurableQueue(tmp_path / "b")
        src.root_dir.mkdir()
        with pytest.raises(KeyError):
            src.move_to("no-such-id", dst)

    def test_delete_removes_file(self, tmp_path: Path) -> None:
        q = DurableQueue(tmp_path / "inbox")
        n = q.post(_sample_spec())
        q.delete(n.id)
        assert not (q.root_dir / f"{n.id}.json").exists()
        assert q.list() == []

    def test_delete_missing_raises(self, tmp_path: Path) -> None:
        q = DurableQueue(tmp_path / "inbox")
        q.root_dir.mkdir()
        with pytest.raises(KeyError):
            q.delete("no-such-id")


# ---------------------------------------------------------------------------
# DurableQueue: queue-local ID validation
# ---------------------------------------------------------------------------

class TestDurableQueueNotificationIDValidation:
    def _write_forged_notification_file(
        self,
        path: Path,
        *,
        notification_id: str = "forged",
    ) -> None:
        notification = Notification.from_spec(_sample_spec())
        data = notification.to_json()
        data["id"] = notification_id
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    @pytest.mark.parametrize(
        "operation",
        ["get", "update_status", "move_to", "delete"],
    )
    def test_path_traversal_ids_are_rejected_before_file_io(
        self, tmp_path: Path, operation: str,
    ) -> None:
        q = DurableQueue(tmp_path / "queue")
        target = DurableQueue(tmp_path / "target")
        q.root_dir.mkdir()
        forged_path = tmp_path / "forged.json"
        self._write_forged_notification_file(forged_path)

        with pytest.raises(KeyError):
            if operation == "get":
                q.get("../forged")
            elif operation == "update_status":
                q.update_status("../forged", NotificationStatus.HANDLED)
            elif operation == "move_to":
                q.move_to("../forged", target)
            elif operation == "delete":
                q.delete("../forged")
            else:  # pragma: no cover - parametrization guard
                raise AssertionError(operation)

        assert forged_path.is_file()
        assert not (q.root_dir / "forged.json").exists()
        assert not (target.root_dir / "forged.json").exists()

    @pytest.mark.parametrize(
        ("notification_id", "forged_filename"),
        [
            (".", "..json"),
            ("..", "...json"),
        ],
    )
    def test_dot_only_ids_are_rejected_before_hidden_file_io(
        self, tmp_path: Path, notification_id: str, forged_filename: str,
    ) -> None:
        q = DurableQueue(tmp_path / "queue")
        forged_path = q.root_dir / forged_filename
        self._write_forged_notification_file(
            forged_path,
            notification_id=notification_id,
        )

        with pytest.raises(KeyError):
            q.get(notification_id)

        assert forged_path.is_file()

    def test_absolute_ids_are_rejected_before_file_io(
        self, tmp_path: Path,
    ) -> None:
        q = DurableQueue(tmp_path / "queue")
        forged_stem = tmp_path / "absolute-forged"
        forged_path = forged_stem.with_suffix(".json")
        self._write_forged_notification_file(
            forged_path,
            notification_id=str(forged_stem),
        )

        with pytest.raises(KeyError):
            q.get(str(forged_stem))

        assert forged_path.is_file()

    def test_contains_returns_false_for_invalid_id(self, tmp_path: Path) -> None:
        q = DurableQueue(tmp_path / "queue")
        assert "../forged" not in q


# ---------------------------------------------------------------------------
# DurableQueue: temp-file hygiene
# ---------------------------------------------------------------------------


class TestDurableQueueTempFiles:
    def test_list_ignores_temp_sidecars(self, tmp_path: Path) -> None:
        q = DurableQueue(tmp_path / "inbox")
        q.root_dir.mkdir()
        # A stray temp file (as if from a crashed post).
        (q.root_dir / ".tmp-FAKE01.json").write_text("{}", encoding="utf-8")
        n = q.post(_sample_spec())

        listed = q.list()
        assert [x.id for x in listed] == [n.id]

    def test_cleanup_temp_files_removes_sidecars(self, tmp_path: Path) -> None:
        q = DurableQueue(tmp_path / "inbox")
        q.root_dir.mkdir()
        (q.root_dir / ".tmp-FAKE01.json").write_text("{}", encoding="utf-8")
        (q.root_dir / ".tmp-FAKE02.json").write_text("{}", encoding="utf-8")
        q.post(_sample_spec())

        removed = q.cleanup_temp_files()
        assert removed == 2
        assert not (q.root_dir / ".tmp-FAKE01.json").exists()
        assert not (q.root_dir / ".tmp-FAKE02.json").exists()

    def test_cleanup_noop_on_missing_dir(self, tmp_path: Path) -> None:
        q = DurableQueue(tmp_path / "never-touched")
        assert q.cleanup_temp_files() == 0


# ---------------------------------------------------------------------------
# DurableQueue: crash interleavings via _rename seam
# ---------------------------------------------------------------------------


class _RenameCrash(RuntimeError):
    """Sentinel used to simulate a crash during an atomic rename."""


def _install_rename_crash(queue: DurableQueue) -> None:
    """Patch *queue* so its next ``_rename`` raises ``_RenameCrash``."""

    def boom(_src: Path, _dst: Path) -> None:
        raise _RenameCrash("simulated mid-operation crash")

    queue._rename = boom  # type: ignore[method-assign]


class TestDurableQueueCrashInterleavings:
    """Verify filesystem invariants under simulated mid-operation crashes.

    Each test patches ``_rename`` on the relevant queue so the "commit"
    step fails, and then asserts that:

    - The originally-present live file (if any) is untouched.
    - The temp sidecar may or may not exist, but it is never mistaken
      for a live notification.
    - Recovery via a fresh :class:`DurableQueue` pointed at the same
      directory succeeds.
    """

    def test_post_crash_leaves_no_live_file(self, tmp_path: Path) -> None:
        q = DurableQueue(tmp_path / "inbox")
        _install_rename_crash(q)

        with pytest.raises(_RenameCrash):
            q.post(_sample_spec())

        # No live file was created.
        live = [p for p in q.root_dir.iterdir() if not p.name.startswith(".")]
        assert live == []
        # list() correctly reports empty.
        assert q.list() == []

    def test_update_crash_leaves_previous_content(self, tmp_path: Path) -> None:
        q = DurableQueue(tmp_path / "inbox")
        n = q.post(_sample_spec())
        _install_rename_crash(q)

        with pytest.raises(_RenameCrash):
            q.update_status(n.id, NotificationStatus.HANDLED, notes="won't stick")

        # Live file still has pending status.
        reloaded = q.get(n.id)
        assert reloaded.status is NotificationStatus.PENDING
        assert reloaded.notes is None

    def test_move_crash_leaves_source_intact(self, tmp_path: Path) -> None:
        src = DurableQueue(tmp_path / "inbox")
        dst = DurableQueue(tmp_path / "service-q")
        n = src.post(_sample_spec(rsvp=ServiceAddress("gitlab-primary")))

        _install_rename_crash(src)
        with pytest.raises(_RenameCrash):
            src.move_to(n.id, dst)

        assert (src.root_dir / f"{n.id}.json").is_file()
        assert not (dst.root_dir / f"{n.id}.json").exists()

    def test_recovery_after_post_crash_works(self, tmp_path: Path) -> None:
        q = DurableQueue(tmp_path / "inbox")
        _install_rename_crash(q)
        with pytest.raises(_RenameCrash):
            q.post(_sample_spec())

        # Simulate restart: fresh queue instance, clean up, then operate.
        fresh = DurableQueue(tmp_path / "inbox")
        fresh.cleanup_temp_files()
        n = fresh.post(_sample_spec())
        assert fresh.list() == [n]

    def test_repeat_update_after_crash_still_atomic(self, tmp_path: Path) -> None:
        q = DurableQueue(tmp_path / "inbox")
        n = q.post(_sample_spec())
        _install_rename_crash(q)
        with pytest.raises(_RenameCrash):
            q.update_status(n.id, NotificationStatus.IN_PROGRESS)

        # Unpatch by replacing with a fresh instance -- the real
        # os.replace is back in play.
        fresh = DurableQueue(tmp_path / "inbox")
        fresh.cleanup_temp_files()
        updated = fresh.update_status(n.id, NotificationStatus.IN_PROGRESS)
        assert updated.status is NotificationStatus.IN_PROGRESS
