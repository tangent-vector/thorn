"""Event abstractions for the Thorn gateway.

Two related shapes flow through the gateway's event pipeline:

- :class:`RawIncomingEvent` -- the *public* boundary type that
  :class:`EventSource` implementations produce.  It carries
  structured raw material (actor identity, kind classification,
  context items) without any pre-rendered prose, leaving formatting
  decisions to the gateway's central
  :class:`~thorn.gateway._formatter.NotificationFormatter`.

- :class:`FormattedEvent` -- the *internal* shape produced by the
  formatter on its way to ``Gateway._handle_event``.  Carries a
  fully-rendered ``content`` string ready to post into a session
  inbox.

Splitting the two keeps trigger-authorization policy and content-
envelope rendering centralised: every source emits the same
structured raw shape, and the security-relevant decisions
(drop / deliver / wrap-with-banner) happen in one place rather
than being re-implemented per source.

Three enums and the :class:`ContextItem` value live alongside the
two events:

- :class:`EventKind` -- structural / conversational / system,
  driving the trigger-authorization carve-out for non-peer events.
- :class:`ContextItemKind` -- comment / issue body / PR body /
  review / harness note, used as metadata on the rendered envelope.
- :class:`ContextItem` -- one quoted chunk to surface to the agent.
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from thorn.core._service import Service
from thorn.gateway._actor import ActorIdentity
from thorn.runtime._session import AgentID, SessionKey


class EventKind(StrEnum):
    """High-level classification of an incoming event.

    Used by :mod:`thorn.gateway._trigger_policy` to decide whether
    the structural-event carve-out applies (a ``STRUCTURAL`` event
    from a non-peer is delivered with a banner; a ``CONVERSATIONAL``
    event from a non-peer is dropped).  ``StrEnum`` rather than
    ``Literal[...]`` so ``kind == EventKind.STRUCTURAL``-style
    comparisons read cleanly in policy code.
    """

    STRUCTURAL = "structural"
    """An entity-shape change: an issue or PR was opened, an
    assignment changed, a label moved.  Worth surfacing to the
    agent even from non-peers because the agent should be
    *aware* of activity on its projects, but not under any
    obligation to act."""

    CONVERSATIONAL = "conversational"
    """A message-like event: a comment, reply, review, or
    direct mention.  Default-deny for non-peers because the
    only reason to surface it would be for the agent to act on
    its content, and acting on instructions from non-peers is
    exactly what the policy aims to prevent."""

    SYSTEM = "system"
    """A harness-internal event with no human author (e.g. a
    scheduled wakeup).  Always delivered."""


class ContextItemKind(StrEnum):
    """What kind of external content a :class:`ContextItem` carries.

    Stamped on the envelope's ``kind=`` attribute and used in
    formatter banner text (e.g. "non-peer comment" vs.
    "non-peer issue body").  Members track the natural surface
    types of forge events; new kinds get added as new sources
    grow.
    """

    COMMENT = "comment"
    ISSUE_BODY = "issue_body"
    PR_BODY = "pr_body"
    REVIEW = "review"
    HARNESS_NOTE = "harness_note"


@dataclass(frozen=True)
class ContextItem:
    """One untrusted-content chunk to surface to the agent.

    Sources populate ``items`` on a :class:`RawIncomingEvent` with
    one entry per piece of human-authored text the agent should
    see (e.g. the body of the comment that triggered the event,
    or the description of a freshly-opened issue).  The formatter
    wraps each item in an envelope using
    :func:`~thorn.gateway._envelope.wrap_external`.

    A ``ContextItem.actor`` *may* differ from the
    ``RawIncomingEvent.primary_actor`` -- e.g. a freshly-opened PR
    review by reviewer X may also include the original PR body by
    author Y, which the agent should see for context.  The
    formatter labels each item with its own peer-status
    independently.
    """

    body: str
    """The raw, pre-envelope text of the chunk.  May be empty;
    the envelope still renders so the agent sees that there *was*
    an item."""

    kind: ContextItemKind
    """What kind of content this is (comment, issue body, etc.)."""

    actor: ActorIdentity | None = None
    """The author of this specific chunk, or ``None`` when no
    actor is identifiable (e.g. a harness-injected note)."""

    timestamp: str = ""
    """Optional ISO-8601 timestamp of when the chunk was authored.
    Empty when the source could not capture this."""


@dataclass(frozen=True)
class RawIncomingEvent:
    """Public boundary type produced by :class:`EventSource` implementations.

    Carries structured raw material that the gateway's formatter
    transforms into a :class:`FormattedEvent` for delivery.  Sources
    must not pre-render prose into this type: the entire point of
    the split is to keep peer-lookup, banner prepending, and
    envelope wrapping in one place.

    Routing-relevant fields (``session_key``, ``agent_id``,
    ``external_key``) are forwarded unchanged to the
    ``FormattedEvent`` once the formatter decides to deliver.

    Attributes:
        source: Stable identifier of the originating system
            (e.g. ``"github"``, ``"gitlab"``).  Carried through
            for logging, metrics, and source-keyed external dedup.
        session_key: The session-inbox routing key.
        kind: Structural vs. conversational vs. system; see
            :class:`EventKind`.
        primary_actor: The "who triggered this event" actor.  ``None``
            for system events or for sources that cannot capture an
            actor (e.g. a webhook that does not include sender info).
        summary: A short, harness-controlled description of what
            happened, suitable as the leading line of the rendered
            notification.  Sources construct this; it is *not*
            attacker-controlled (it embeds field names like
            ``Issue #42`` and the platform's notification reason,
            not user-provided body text).
        items: Zero or more :class:`ContextItem` chunks of
            external content the agent should see.  May be empty
            for purely structural events with no body to quote.
        agent_id: Optional override for which agent should handle
            this event.  ``None`` defers to the gateway's default
            routing logic.
        metadata: Source-specific routing/diagnostic data carried
            through to the formatted event's metadata.  Should
            *not* contain attacker-controlled prose; for that, use
            ``items``.
        external_key: Source-namespaced stable id used by the
            in-flight index for cross-poll deduplication.  Same
            role as on :class:`FormattedEvent`.
    """

    source: str
    session_key: SessionKey
    kind: EventKind
    primary_actor: ActorIdentity | None = None
    summary: str = ""
    items: tuple[ContextItem, ...] = ()
    agent_id: AgentID | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    external_key: str | None = None


@dataclass(frozen=True)
class FormattedEvent:
    """Internal post-formatter shape headed for ``Gateway._handle_event``.

    Identical in structure to the pre-refactor ``IncomingEvent``:
    a fully-rendered ``content`` string plus the routing fields.
    Decoupled from :class:`RawIncomingEvent` so that the formatter
    is the only writer of ``content`` -- sources cannot accidentally
    inject pre-rendered prose by setting ``content`` themselves.

    Attributes:
        source: Same as on :class:`RawIncomingEvent`.
        session_key: Same as on :class:`RawIncomingEvent`.
        content: The rendered prompt text -- summary, banner (if
            any), and one envelope per ``ContextItem``.
        agent_id: Same as on :class:`RawIncomingEvent`.
        metadata: Same as on :class:`RawIncomingEvent`.
        external_key: Same as on :class:`RawIncomingEvent`.
    """

    source: str
    session_key: SessionKey
    content: str
    agent_id: AgentID | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    external_key: str | None = None


class EventSource(Service):
    """Abstract base class for pluggable event inputs.

    Implementations call the *on_event* callback supplied to
    :meth:`start` whenever a new event is detected.  The gateway
    handles routing, formatting, and per-event policy decisions;
    sources are responsible only for capturing actor identity,
    classifying the event kind, and populating context items.

    ``EventSource`` is a :class:`Service` subclass.  Subclasses must
    define a ``Config`` class attribute pointing to a
    :class:`pydantic.BaseModel` subclass that describes the source's
    configuration, and implement :attr:`name`.
    """

    @abstractmethod
    async def start(
        self,
        on_event: Callable[[RawIncomingEvent], Awaitable[None]],
    ) -> None:
        """Begin producing events, invoking *on_event* for each one.

        Must not return until :meth:`stop` is called (i.e. this
        coroutine is the long-running event loop).
        """

    @abstractmethod
    async def stop(self) -> None:
        """Signal the source to shut down gracefully.

        After ``stop()`` returns, no further calls to *on_event*
        should be made.
        """


# Back-compat alias.  The pre-refactor type was named ``IncomingEvent``
# and many tests / callers still spell it that way; ``FormattedEvent``
# is the new canonical name (the type produced *by* the formatter on
# its way to ``Gateway._dispatch_formatted``).  Keeping the alias in
# place lets the test suite migrate at its own pace.  New code should
# spell out ``FormattedEvent`` (post-formatter) or ``RawIncomingEvent``
# (pre-formatter) explicitly.
IncomingEvent = FormattedEvent


__all__ = [
    "ContextItem",
    "ContextItemKind",
    "EventKind",
    "EventSource",
    "FormattedEvent",
    "IncomingEvent",
    "RawIncomingEvent",
]
