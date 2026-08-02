"""Notification types: the payloads that flow through durable queues.

A :class:`NotificationSpec` is the immutable, source-produced
description of a notification about to be posted.  Event sources,
services, and agents construct specs and hand them to a
:class:`~thorn.runtime._queue.DurableQueue` via ``post``.

A :class:`Notification` is the persisted form produced by the queue at
post time.  It carries everything from the spec plus framework-owned
fields -- a ULID ``id``, the ``posted_at`` timestamp, the current
``status`` in the two-step handling lifecycle, attempt counts, and
optional ``notes`` / ``error_reason`` annotations that get filled in
as the notification moves through its lifecycle.

The lifecycle is deliberately file-durable: every transition is
serialized as a new JSON representation that is written atomically
over the previous one, so a process crash can leave a notification in
any individual state but never in a partial/torn state.

The status vocabulary (:class:`NotificationStatus`) encodes the full
handling model:

- ``pending``: freshly posted and awaiting attention.
- ``in_progress``: the receiver has explicitly claimed it.
- ``handled``: receiver completed successfully; step 1 of two-step
  dispatch is done, the post-hoc move/delete is still pending.
- ``errored``: receiver gave up; same step-1/step-2 split as handled.
- ``confirmed``: the RSVP handler at the original sender has completed
  its own work; step 1 done, deletion pending.

The ``errored`` status carries a required ``error_reason``; ``handled``
and ``in_progress`` may carry optional ``notes``.
"""

from __future__ import annotations

import secrets
import threading
import time
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Mapping

from thorn.runtime._address import Address


class NotificationStatus(str, Enum):
    """Lifecycle status of a :class:`Notification`.

    Inherits from ``str`` so JSON serialization is trivial and so the
    textual form (``"pending"``, ``"handled"``, etc.) is canonical.
    """

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    HANDLED = "handled"
    ERRORED = "errored"
    CONFIRMED = "confirmed"


@dataclass(frozen=True)
class InboxCompletionRationale:
    """Structured rationale recorded when an inbox item is completed."""

    completed_actions: tuple[str, ...]
    """Concrete actions performed for the notification."""

    request_coverage: tuple[str, ...]
    """Requested elements or acceptance criteria checked before close-out."""

    self_review: str
    """Summary of the final review performed before marking handled."""

    remaining_work: tuple[str, ...] = ()
    """Follow-up work still visible after close-out, if any.

    Entries here should not describe work required for this
    notification to be considered complete.
    """

    external_follow_up: tuple[str, ...] = ()
    """External comments, MRs, reviewer pings, or similar follow-up."""

    def validation_errors(self) -> tuple[str, ...]:
        """Return human-readable issues that make the rationale incomplete."""
        errors: list[str] = []
        if not _has_meaningful_entry(self.completed_actions):
            errors.append("completed_actions must contain at least one item")
        if not _has_meaningful_entry(self.request_coverage):
            errors.append("request_coverage must contain at least one item")
        if not self.self_review.strip():
            errors.append("self_review must not be empty")

        for field_name, entries in (
            ("completed_actions", self.completed_actions),
            ("request_coverage", self.request_coverage),
            ("remaining_work", self.remaining_work),
            ("external_follow_up", self.external_follow_up),
        ):
            if any(not entry.strip() for entry in entries):
                errors.append(f"{field_name} must not contain blank items")
        return tuple(errors)

    def to_json(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "completed_actions": list(self.completed_actions),
            "request_coverage": list(self.request_coverage),
            "self_review": self.self_review,
            "remaining_work": list(self.remaining_work),
            "external_follow_up": list(self.external_follow_up),
        }

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> InboxCompletionRationale:
        """Reconstruct a rationale from :meth:`to_json` output."""
        return cls(
            completed_actions=_string_tuple(data["completed_actions"]),
            request_coverage=_string_tuple(data["request_coverage"]),
            self_review=str(data["self_review"]),
            remaining_work=_string_tuple(data.get("remaining_work") or ()),
            external_follow_up=_string_tuple(
                data.get("external_follow_up") or (),
            ),
        )

    def to_display_text(self) -> str:
        """Render the rationale for operator-facing text surfaces."""
        lines = ["Completion rationale:"]
        _append_items(lines, "Completed actions", self.completed_actions)
        _append_items(lines, "Request coverage", self.request_coverage)
        _append_items(lines, "Remaining work", self.remaining_work)
        lines.append("Self-review:")
        lines.append(f"- {self.self_review}")
        _append_items(lines, "External follow-up", self.external_follow_up)
        return "\n".join(lines)


def _has_meaningful_entry(entries: tuple[str, ...]) -> bool:
    return any(entry.strip() for entry in entries)


def _string_tuple(raw_entries: Any) -> tuple[str, ...]:
    if isinstance(raw_entries, str):
        return (raw_entries,)
    return tuple(str(entry) for entry in raw_entries)


def _append_items(
    lines: list[str],
    heading: str,
    entries: tuple[str, ...],
) -> None:
    lines.append(f"{heading}:")
    if entries:
        lines.extend(f"- {entry}" for entry in entries)
        return
    lines.append("- None")


# ---------------------------------------------------------------------------
# Notification IDs
# ---------------------------------------------------------------------------


class NotificationID(str):
    """Framework-assigned identifier for a persisted notification."""


_ULID_ALPHABET = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
"""Crockford base32 alphabet used for ULID encoding."""

_ULID_RAND_MASK = (1 << 80) - 1

_ulid_state_lock = threading.Lock()
_ulid_last_ms: int = 0
_ulid_last_rand: int = 0


def _generate_ulid() -> NotificationID:
    """Generate a ULID: 26-character Crockford-base32 identifier.

    The ULID spec encodes 48 bits of millisecond-precision Unix
    timestamp followed by 80 bits of randomness.  ULIDs are
    lexicographically sortable by creation time, which makes directory
    listings naturally ordered by post time without any separate
    index.

    Strict in-process monotonicity is guaranteed using the
    `monotonicity extension
    <https://github.com/ulid/spec#monotonicity>`_: when multiple ULIDs
    are generated within the same millisecond (or if the wall clock
    goes backwards), the random component is treated as a counter and
    incremented by one instead of freshly sampled.  This gives a
    strict ordering for rapid-fire posts, not just an at-ms-precision
    one.

    We implement this inline rather than take a new dependency -- the
    encoding is short and its correctness is load-bearing for queue
    ordering.
    """
    global _ulid_last_ms, _ulid_last_rand
    with _ulid_state_lock:
        ms = time.time_ns() // 1_000_000
        if ms <= _ulid_last_ms:
            # Same millisecond, or clock rewound (NTP adjustment etc.).
            # Pin timestamp to last seen value and bump random bits to
            # preserve strict monotonicity.
            ms = _ulid_last_ms
            _ulid_last_rand = (_ulid_last_rand + 1) & _ULID_RAND_MASK
        else:
            _ulid_last_ms = ms
            _ulid_last_rand = int.from_bytes(secrets.token_bytes(10), "big")
        value = (ms << 80) | _ulid_last_rand
    out: list[str] = []
    for _ in range(26):
        out.append(_ULID_ALPHABET[value & 0x1F])
        value >>= 5
    return NotificationID("".join(reversed(out)))


# ---------------------------------------------------------------------------
# NotificationSpec
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class NotificationSpec:
    """Immutable source-produced description of a notification to post.

    A spec is converted to a :class:`Notification` by the queue's
    ``post`` method, which assigns the ``id``/``posted_at``/``status``
    fields.  The fields on this type are those that originate outside
    the framework -- they describe *what* is being delivered and
    *where*, not *where it is in its lifecycle*.
    """

    source: str
    """Stable identifier of the origin (e.g. ``"gitlab-poller"``)."""

    content: str
    """Human-readable payload that will form the bulk of the prompt
    (for session inboxes) or handler input (for notification queues)."""

    target: Address
    """Address of the queue that should receive this notification."""

    metadata: Mapping[str, Any] = field(default_factory=dict)
    """Arbitrary source-specific data (e.g. a forge TODO id, a URL)
    carried through the lifecycle so handlers and RSVP recipients can
    correlate the notification with external state."""

    rsvp_to: Address | None = None
    """Optional address to which the notification is forwarded once the
    receiver marks it ``handled`` or ``errored``.  ``None`` means no
    RSVP is desired; handled items are deleted, errored items are
    parked in the receiver's ``errored/`` directory."""

    external_key: str | None = None
    """Optional stable identifier the source uses for cross-poll
    deduplication via the ``InFlightIndex``.  Two notifications with
    the same ``external_key`` may not coexist in flight."""


# ---------------------------------------------------------------------------
# Notification (persisted form)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Notification:
    """Persisted form of a notification.

    This is the object that gets serialized to and loaded from the
    per-item JSON files owned by a :class:`DurableQueue`.  Frozen so
    that in-place status mutation goes through
    :func:`dataclasses.replace` and is unambiguously a *new* value --
    the queue is responsible for writing the new value atomically to
    disk.
    """

    id: NotificationID
    """ULID assigned at post time; also the filename stem."""

    source: str
    content: str
    target: Address
    metadata: Mapping[str, Any]
    rsvp_to: Address | None
    external_key: str | None

    posted_at: datetime
    """UTC timestamp at which the queue accepted this notification."""

    status: NotificationStatus
    """Current lifecycle position."""

    attempt_count: int
    """Number of delivery attempts made against this notification.

    Incremented by the scheduler / handler on each prompt or drain
    attempt.  Used as input to the progress-guarantee logic but
    otherwise informational."""

    notes: str | None
    """Optional agent/handler annotation carried to RSVP recipients."""

    completion_rationale: InboxCompletionRationale | None
    """Structured completion evidence recorded for ``handled`` inbox items."""

    error_reason: str | None
    """Populated when ``status == errored``; ``None`` otherwise."""

    # ------------------------------------------------------------------
    # Construction from spec

    @classmethod
    def from_spec(cls, spec: NotificationSpec) -> Notification:
        """Promote a :class:`NotificationSpec` to a fresh ``Notification``.

        Assigns a new ULID, captures the current UTC timestamp, and
        sets ``status`` to :attr:`NotificationStatus.PENDING`.  This
        is the only way the framework creates a ``Notification`` from
        scratch -- all later transitions go through
        :func:`dataclasses.replace`.
        """
        return cls(
            id=_generate_ulid(),
            source=spec.source,
            content=spec.content,
            target=spec.target,
            metadata=dict(spec.metadata),
            rsvp_to=spec.rsvp_to,
            external_key=spec.external_key,
            posted_at=datetime.now(timezone.utc),
            status=NotificationStatus.PENDING,
            attempt_count=0,
            notes=None,
            completion_rationale=None,
            error_reason=None,
        )

    # ------------------------------------------------------------------
    # Serialization

    def to_json(self) -> dict[str, Any]:
        """Return a JSON-serializable dict representation.

        Address values are rendered to their canonical string form;
        timestamps are rendered as ISO-8601 strings; the status enum
        is rendered as its string value.
        """
        return {
            "id": self.id,
            "source": self.source,
            "content": self.content,
            "target": str(self.target),
            "metadata": dict(self.metadata),
            "rsvp_to": str(self.rsvp_to) if self.rsvp_to is not None else None,
            "external_key": self.external_key,
            "posted_at": self.posted_at.isoformat(),
            "status": self.status.value,
            "attempt_count": self.attempt_count,
            "notes": self.notes,
            "completion_rationale": (
                self.completion_rationale.to_json()
                if self.completion_rationale is not None else None
            ),
            "error_reason": self.error_reason,
        }

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> Notification:
        """Reconstruct a ``Notification`` from :meth:`to_json` output.

        Raises ``ValueError`` or ``KeyError`` for malformed input.
        """
        rsvp_raw = data.get("rsvp_to")
        completion_rationale_raw = data.get("completion_rationale")
        return cls(
            id=NotificationID(data["id"]),
            source=data["source"],
            content=data["content"],
            target=Address.parse(data["target"]),
            metadata=dict(data.get("metadata") or {}),
            rsvp_to=Address.parse(rsvp_raw) if rsvp_raw else None,
            external_key=data.get("external_key"),
            posted_at=datetime.fromisoformat(data["posted_at"]),
            status=NotificationStatus(data["status"]),
            attempt_count=int(data.get("attempt_count", 0)),
            notes=data.get("notes"),
            completion_rationale=(
                InboxCompletionRationale.from_json(completion_rationale_raw)
                if completion_rationale_raw is not None else None
            ),
            error_reason=data.get("error_reason"),
        )

    # ------------------------------------------------------------------
    # Convenience

    def with_updates(
        self,
        *,
        status: NotificationStatus | None = None,
        **fields: Any,
    ) -> Notification:
        """Return a copy with the supplied fields updated.

        Convenience wrapper around :func:`dataclasses.replace` so that
        ``queue.update_status`` and related callers can express
        transitions naturally.  Unknown field names raise ``TypeError``
        (via :func:`dataclasses.replace`'s validation).
        """
        updates: dict[str, Any] = dict(fields)
        if status is not None:
            updates["status"] = status
        return replace(self, **updates)


__all__ = [
    "InboxCompletionRationale",
    "Notification",
    "NotificationID",
    "NotificationSpec",
    "NotificationStatus",
]
