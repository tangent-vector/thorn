"""Peers: who the agent considers a trusted speaker.

A *peer* is a human or bot whose messages the gateway is willing to
treat as instructions for an agent.  A peer is named by a stable
``id`` (unique within the gateway), a current display ``name``
(may change freely), and a list of ``accounts`` -- one per service
the peer has an identity on.

Peer ids are deliberately decoupled from ``name``.  The peer's
on-disk notes directory is keyed on ``id`` (e.g.
``~/peers/<peer_id>/``) so a name change on the human side never
invalidates a history of accumulated notes.  An id may be a
memorable handle (``ada-lovelace``) or a programmatic one
(``peer-7f3a``); the framework rule is that ids are write-once.

Trust model: this is the "personal-assistant" boundary, not
adversarial multi-tenant.  Anyone who can write ``gateway.json`` is
trusted at the operator level.  The peer system is a best-effort
guardrail for trigger authorization (drop events from non-peers
before they reach an agent's inbox) and a labelling input for
content envelopes (so the agent's data-vs-instruction rule has
something to anchor on); the actual security boundary is the
container sandbox + broker.
"""

from __future__ import annotations

import logging
import re
from enum import StrEnum

from pydantic import BaseModel, Field, model_validator

from thorn.gateway._actor import ActorIdentity

log = logging.getLogger(__name__)


class PeerKind(StrEnum):
    """Whether a peer is a human or a bot.

    Used by the trigger-authorization policy: an unmatched actor
    flagged ``is_bot=True`` by the platform is dropped by default
    unless an entry in the peer registry has ``kind=BOT`` and
    matches the actor's account.  A ``StrEnum`` (rather than a
    ``Literal[...]``) so policy code reads as
    ``peer.kind == PeerKind.BOT`` rather than string-compare, and so
    the canonical set of kinds has a single home.
    """

    HUMAN = "human"
    BOT = "bot"


class PeerAccount(BaseModel):
    """One ``(service, account_id)`` pair on a peer's account list.

    Operators write these in ``gateway.json``.  The ``account_id``
    field accepts either the platform-immutable id or the textual
    handle (GitHub ``login``, GitLab ``username``); the registry
    matches both forms against an :class:`ActorIdentity`.

    Future ``thorn serve resolve-peers`` (see TODO.md) will rewrite
    textual ``account_id`` values to their immutable form on demand.
    """

    service: str = Field(
        min_length=1,
        description=(
            "Name of the service entry this account lives on, "
            "matching a `forges[].name` (or other service spec name) "
            "in `gateway.json`."
        ),
    )
    account_id: str = Field(
        min_length=1,
        description=(
            "Account identifier on the named service.  Either the "
            "platform-immutable id (preferred) or the textual user "
            "handle (matched against an actor's secondary ids when "
            "the immutable form is not yet known)."
        ),
    )


class PeerSpec(BaseModel):
    """Operator-declared peer entry.

    The shape that lives under the ``peers:`` array in
    ``gateway.json``.  Validated into a :class:`PeerRegistry` at
    gateway startup.  Field names line up with the registry's public
    surface so the agent-facing peer tools can serialise instances of
    this type directly.
    """

    id: str = Field(
        min_length=1,
        description=(
            "Stable, write-once identifier unique within the gateway. "
            "Used as the on-disk peer-notes directory name "
            "(`~/peers/<id>/`) and as the persistent reference for "
            "agent memory.  Either a memorable handle "
            "(`ada-lovelace`) or a programmatic id (`peer-7f3a`); "
            "must be filesystem-safe."
        ),
    )
    name: str = Field(
        default="",
        description=(
            "Current human-readable display name.  May change freely; "
            "the agent reads this for prose rendering.  Empty allowed "
            "(e.g. for bot peers where the service handle is the "
            "natural label)."
        ),
    )
    kind: PeerKind = Field(
        default=PeerKind.HUMAN,
        description=(
            "Whether this peer is a human or a bot.  Determines "
            "whether the bot-default-deny rule in the "
            "trigger-authorization policy applies."
        ),
    )
    accounts: list[PeerAccount] = Field(
        default_factory=list,
        description=(
            "Per-service identities for this peer.  Multiple accounts "
            "on the same service are allowed (e.g. an old and a new "
            "account during a transition); all are matched against "
            "incoming actors."
        ),
    )

    @model_validator(mode="after")
    def _validate_id(self) -> PeerSpec:
        if not _PEER_ID_PATTERN.match(self.id):
            raise ValueError(
                f"Peer id {self.id!r} must start with an alphanumeric "
                "character and contain only letters, digits, '_', "
                "'-', and '.'.  Peer ids are used as filesystem "
                "directory names under the agent's home."
            )
        return self


# Filesystem-safe id check.  Lives at module level rather than as a
# class attribute because pydantic treats leading-underscore class
# attributes on a ``BaseModel`` as private state, which would break
# direct ``.match`` access.  Deliberately strict: peer ids show up
# as directory names under the agent's home, and shell-fragile
# characters (spaces, glob chars, leading dots) cause more operator
# pain than a tightening of the id grammar.
_PEER_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.\-]*$")


class PeerRegistry:
    """Lookup table for peers, built once at gateway startup.

    Two indexing strategies cover the mismatch between what operators
    write in ``gateway.json`` (often a textual handle) and what
    incoming events carry (typically an immutable id, plus the
    handle as a secondary):

    1.  ``(service, account_id)`` -> peer.  Hit when the operator
        wrote the immutable id, or when the platform happens to
        carry the same handle in both fields.
    2.  ``(service, secondary_id)`` -> peer.  Fallback hit when the
        operator wrote a textual handle and the event's
        ``ActorIdentity.secondary_account_ids`` carries it.

    A registry is intended to be constructed once from a validated
    list of :class:`PeerSpec` entries and treated as immutable for
    the gateway's lifetime.  Re-indexing is cheap; the cost is in
    decoupling registry construction from gateway runtime so that
    later we can hot-reload after a config edit without restarting.
    """

    def __init__(self, peers: list[PeerSpec]) -> None:
        self._peers: dict[str, PeerSpec] = {}
        # ``service`` -> ``account_id`` -> peer.  Lookups intentionally
        # index per-service first so that an account_id collision
        # across two unrelated services cannot cross-match.
        self._by_account: dict[str, dict[str, PeerSpec]] = {}
        # Tracks which (service, account_id) pairs have already
        # surfaced a "you wrote the textual handle" hint, so that
        # repeated events from the same actor do not spam the log.
        self._username_match_hinted: set[tuple[str, str]] = set()

        for spec in peers:
            if spec.id in self._peers:
                raise ValueError(
                    f"Duplicate peer id {spec.id!r} in peer registry; "
                    "peer ids must be unique within the gateway."
                )
            self._peers[spec.id] = spec
            for account in spec.accounts:
                bucket = self._by_account.setdefault(account.service, {})
                # First write wins for a given (service, account_id);
                # if two peers claim the same account, the registry
                # construction is the right place to fail loudly.
                existing = bucket.get(account.account_id)
                if existing is not None and existing.id != spec.id:
                    raise ValueError(
                        f"Account {account.service}:{account.account_id} "
                        f"is claimed by both peer {existing.id!r} and "
                        f"peer {spec.id!r}; an account may belong to "
                        "at most one peer."
                    )
                bucket[account.account_id] = spec
            if not spec.accounts:
                log.warning(
                    "Peer %r has no accounts declared; no incoming "
                    "events will ever match this peer.", spec.id,
                )

    # ------------------------------------------------------------------
    # Lookup

    def get(self, peer_id: str) -> PeerSpec | None:
        """Return the peer with the given id, or ``None``."""
        return self._peers.get(peer_id)

    def lookup_actor(self, actor: ActorIdentity) -> PeerSpec | None:
        """Return the peer matching *actor*, or ``None``.

        Tries the immutable id first; on miss, falls back to each of
        the actor's secondary ids in order.  When a match comes back
        on a secondary id we log a one-time-per-actor hint suggesting
        the operator pin the immutable form in ``gateway.json``,
        because subsequent username changes by that peer would
        silently desync the registry.
        """
        bucket = self._by_account.get(actor.service)
        if bucket is None:
            return None
        peer = bucket.get(actor.account_id)
        if peer is not None:
            return peer
        for secondary in actor.secondary_account_ids:
            peer = bucket.get(secondary)
            if peer is None:
                continue
            key = (actor.service, secondary)
            if key not in self._username_match_hinted:
                self._username_match_hinted.add(key)
                log.info(
                    "Peer %r matched on secondary handle %r for "
                    "service %r.  Consider pinning the immutable id "
                    "(%r) in gateway.json so a future username change "
                    "does not break matching.",
                    peer.id, secondary, actor.service, actor.account_id,
                )
            return peer
        return None

    def lookup_account(self, service: str, account_id: str) -> PeerSpec | None:
        """Return the peer claiming ``(service, account_id)``, or ``None``.

        Plain ``(service, account_id)`` lookup with no ``ActorIdentity``
        in hand.  Useful for the agent-facing ``peer_by_account`` tool
        where the agent has only what the platform reports for an
        actor; matches both the immutable-id and textual-handle forms
        because both are stored in the same per-service bucket.
        """
        bucket = self._by_account.get(service)
        if bucket is None:
            return None
        return bucket.get(account_id)

    def find_by_name(self, query: str) -> list[PeerSpec]:
        """Return peers whose ``name`` contains *query* (case-insensitive).

        Substring match against the display name only.  Returns an
        empty list when *query* is empty (a substring match against
        the empty string would otherwise return every peer, which is
        almost never what callers want).
        """
        if not query:
            return []
        needle = query.casefold()
        return [
            peer
            for peer in self._peers.values()
            if needle in peer.name.casefold()
        ]

    def all_peers(self) -> list[PeerSpec]:
        """Return every peer in stable id-sorted order."""
        return sorted(self._peers.values(), key=lambda p: p.id)


__all__ = [
    "PeerAccount",
    "PeerKind",
    "PeerRegistry",
    "PeerSpec",
]
