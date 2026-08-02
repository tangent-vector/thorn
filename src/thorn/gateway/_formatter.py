"""Notification formatter: ``RawIncomingEvent`` -> ``FormattedEvent``.

The formatter is the single bottleneck where actor-aware decisions
and content-envelope rendering happen.  Sources hand it a
:class:`RawIncomingEvent`; it consults the
:class:`~thorn.gateway._trigger_policy.TriggerAuthorizationPolicy`,
applies the decision, and (when delivering) renders the event into
a :class:`FormattedEvent` whose ``content`` string is ready to post
into a session inbox.

Centralising this here -- rather than letting each source render
its own prose -- means new event sources do not have to re-implement
the security policy, and the same envelope discipline applies
uniformly across sources.

The rendered ``content`` is a vertical concatenation of:

1.  The event's ``summary`` line (always present).
2.  An optional non-peer banner when the policy returned
    :class:`~thorn.gateway._trigger_policy.DeliverWithBanner`.
3.  One :func:`~thorn.gateway._envelope.wrap_external` envelope per
    :class:`ContextItem`, separated by blank lines.

The envelopes are labelled with each item's per-item peer status
(an item authored by a peer different from the event's primary
actor still gets correctly labelled).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from thorn.gateway._actor import (
    ActorIdentity,
    describe_actor_for_log,
)
from thorn.gateway._envelope import PeerStatus, wrap_external
from thorn.gateway._event import (
    ContextItem,
    FormattedEvent,
    RawIncomingEvent,
)
from thorn.gateway._peer import PeerRegistry
from thorn.gateway._trigger_policy import (
    Deliver,
    DeliverWithBanner,
    Drop,
    TriggerAuthorizationPolicy,
)

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class FormatterDelivery:
    """Result of a formatter call when the policy decided to deliver.

    Either of the deliver branches produces this; the difference
    between :class:`~thorn.gateway._trigger_policy.Deliver` and
    :class:`~thorn.gateway._trigger_policy.DeliverWithBanner` is
    captured in the rendered ``event.content`` and not separately
    surfaced.
    """

    event: FormattedEvent


@dataclass(frozen=True)
class FormatterDrop:
    """Result of a formatter call when the policy decided to drop.

    The gateway logs *reason* and discards the event without
    touching session state.  The source's mark-read / mark-done
    behaviour fires uniformly regardless (drops are terminal --
    see plan open question #8).
    """

    reason: str


FormatterResult = FormatterDelivery | FormatterDrop


class NotificationFormatter:
    """Apply trigger-authorization policy and render the envelope-wrapped notification.

    Stateless beyond its (peer registry, policy) construction;
    intended to be safe to share across all event sources running
    in a single gateway instance.
    """

    def __init__(
        self,
        *,
        peer_registry: PeerRegistry,
        policy: TriggerAuthorizationPolicy,
    ) -> None:
        self._peers = peer_registry
        self._policy = policy

    # ------------------------------------------------------------------
    # Public API

    def process(self, raw: RawIncomingEvent) -> FormatterResult:
        """Run policy, render content, return delivery or drop."""
        decision = self._policy.decide(raw)

        if isinstance(decision, Drop):
            log.info(
                "Dropping event from %s (kind=%s, actor=%s, reason=%s)",
                raw.source, raw.kind.value,
                describe_actor_for_log(raw.primary_actor),
                decision.reason,
            )
            return FormatterDrop(reason=decision.reason)

        # Deliver / DeliverWithBanner: render content.
        if isinstance(decision, Deliver):
            banner: str | None = None
            primary_peer_status = (
                PeerStatus.PEER if decision.peer is not None
                else PeerStatus.UNKNOWN
                if raw.primary_actor is None
                else PeerStatus.NON_PEER
            )
            log_peer_id = decision.peer.id if decision.peer is not None else None
        else:
            assert isinstance(decision, DeliverWithBanner)
            banner = decision.banner
            # Banner branch only fires when the actor is a non-peer
            # (or actor is None on a structural event that the
            # carve-out lets through).  The primary item's peer
            # status follows that.
            primary_peer_status = (
                PeerStatus.UNKNOWN
                if raw.primary_actor is None
                else PeerStatus.NON_PEER
            )
            log_peer_id = None

        content = self._render(
            raw,
            banner=banner,
            primary_peer_status=primary_peer_status,
        )

        log.info(
            "Delivering event from %s (kind=%s, actor=%s, items=%d, banner=%s)",
            raw.source,
            raw.kind.value,
            describe_actor_for_log(raw.primary_actor, peer_id=log_peer_id),
            len(raw.items),
            "yes" if banner else "no",
        )

        formatted = FormattedEvent(
            source=raw.source,
            session_key=raw.session_key,
            content=content,
            agent_id=raw.agent_id,
            metadata=dict(raw.metadata),
            workspace_bootstrap=raw.workspace_bootstrap,
            external_key=raw.external_key,
        )
        return FormatterDelivery(event=formatted)

    # ------------------------------------------------------------------
    # Rendering

    def _render(
        self,
        raw: RawIncomingEvent,
        *,
        banner: str | None,
        primary_peer_status: PeerStatus,
    ) -> str:
        """Produce the final ``content`` string for a delivered event."""
        sections: list[str] = []

        if raw.summary:
            sections.append(raw.summary)

        if banner:
            sections.append(banner)

        for item in raw.items:
            sections.append(
                self._render_item(
                    item,
                    raw_source=raw.source,
                    primary_actor=raw.primary_actor,
                    primary_peer_status=primary_peer_status,
                ),
            )

        # Trailing harness note for system events that the gateway
        # adds for legibility.  System-event SUMMARY usually carries
        # everything the agent needs; the harness footer (when one
        # exists) is left to the source via ``items``.

        # Sections are joined by blank lines so the agent sees clear
        # visual separation between the summary, the banner, and
        # each envelope.
        return "\n\n".join(sections)

    def _render_item(
        self,
        item: ContextItem,
        *,
        raw_source: str,
        primary_actor: ActorIdentity | None,
        primary_peer_status: PeerStatus,
    ) -> str:
        """Render one :class:`ContextItem` as a single envelope.

        Per-item actor wins when present; otherwise the event's
        primary actor stands in.  Per-item peer status is
        recomputed from the registry (so a thread that contains a
        peer's reply followed by a non-peer's reply gets each
        labelled correctly).
        """
        if item.actor is not None:
            actor = item.actor
            if item.actor == primary_actor:
                # Same actor -> same peer status as the primary.
                peer_status = primary_peer_status
            else:
                matched = self._peers.lookup_actor(item.actor)
                peer_status = (
                    PeerStatus.PEER if matched is not None
                    else PeerStatus.NON_PEER
                )
        else:
            actor = primary_actor
            # Items with no actor inherit the primary's status; this
            # is a useful approximation for tool-result envelopes
            # where the source did not bother to attribute every
            # chunk separately.
            peer_status = primary_peer_status

        return wrap_external(
            body=item.body,
            actor=actor,
            source=raw_source,
            kind=item.kind.value,
            peer_status=peer_status,
            timestamp=item.timestamp,
        )


__all__ = [
    "FormatterDelivery",
    "FormatterDrop",
    "FormatterResult",
    "NotificationFormatter",
]
