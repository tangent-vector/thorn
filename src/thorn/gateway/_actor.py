"""Actor identity: who triggered an event.

An :class:`ActorIdentity` names a single account on a single service.
It is the source-side representation of "who did this thing" -- the
counterpart of a :class:`~thorn.gateway._peer.PeerAccount` in the
gateway's peer registry.

Identity matching is performed against ``account_id`` (intended to be
the platform-immutable id, e.g. the numeric id GitHub assigns to a
user) with ``secondary_account_ids`` providing fallback matches
against mutable forms of the same account (e.g. the GitHub
``login``).  The ``is_bot`` flag reflects the platform's own hint at
the time of capture; it is informational only -- the actual "is this
actor a bot peer?" decision is made by the trigger-authorization
policy.

The ``display_name`` is captured for human-readable rendering inside
notification envelopes only.  It is *never* used for matching and
*never* placed in INFO log lines: long-lived log files become a
paper trail of an old name if a peer changes how they want to be
addressed.  See :func:`describe_actor_for_log` for the canonical
log-safe form.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ActorIdentity:
    """The actor (human or bot) responsible for an event.

    Captured by event sources from platform-native event payloads.
    The gateway's peer registry matches actors to known peers using
    ``service`` plus ``account_id``, falling back to
    ``secondary_account_ids`` so that operators who wrote a textual
    handle in ``gateway.json`` still match events that carry only
    the immutable id (and vice versa).

    Frozen so the value can be carried as part of an immutable
    ``RawIncomingEvent`` and used as a dict key in tests.  The
    ``secondary_account_ids`` field is a tuple (rather than a list)
    for the same reason.
    """

    service: str
    """Service name as declared in ``gateway.json``.

    Matches a ``forges[].name`` (or other service-spec ``name``).
    The peer registry uses this as the lookup namespace so the same
    ``account_id`` on two different services does not cross over.
    """

    account_id: str
    """Stable, platform-immutable account identifier.

    GitHub: stringified numeric ``user.id``.  GitLab: stringified
    numeric ``user.id``.  The invariant we rely on is that this id
    never changes for a given account on the platform -- it is what
    makes peer matching robust against mutable user-facing handles.

    Sources that genuinely cannot obtain an immutable id (e.g. an
    older payload that carries only a username) may put the username
    here; matching falls through to ``secondary_account_ids`` so
    this is not a hard error, just a less robust match.
    """

    display_name: str = ""
    """Free-form human-readable name for envelope rendering.

    May change on the platform side; never used for matching, never
    logged at INFO.  May be empty when the platform does not provide
    a separate display name (e.g. a generic bot account).
    """

    is_bot: bool | None = None
    """Best-effort hint from the platform: is this account a bot?

    GitHub: ``user.type == "Bot"``.  GitLab: ``User.bot``.  ``None``
    when the source could not capture this.  The trigger-authorization
    policy treats ``True`` as "block by default unless explicitly
    listed as a bot peer" -- see :mod:`thorn.gateway._trigger_policy`.
    """

    secondary_account_ids: tuple[str, ...] = ()
    """Mutable identifiers (e.g. usernames) for fallback matching.

    Sources should populate this with the platform's user-facing
    handle (GitHub ``login``, GitLab ``username``) so the registry
    can fall back when the operator wrote the textual form in
    ``gateway.json``.  A tuple (not a list) so the dataclass remains
    frozen and hashable.
    """

    def at_login_form(self) -> str:
        """Return ``@login``-style display for envelope/banner text.

        Prefers a secondary id (which is the user-facing handle for
        GitHub/GitLab), falling back to ``account_id``.  Returned
        with a leading ``@`` so the same shape works regardless of
        platform conventions.
        """
        handle = self.secondary_account_ids[0] if self.secondary_account_ids else self.account_id
        return f"@{handle}"


def describe_actor_for_log(
    actor: ActorIdentity | None,
    *,
    peer_id: str | None = None,
) -> str:
    """Format an actor for INFO-level log output.

    Returns ``"<peer:peer-id>"`` when *peer_id* is supplied (the
    actor matched a known peer), otherwise
    ``"<actor:service:account_id>"``.  Display names are
    deliberately excluded so log files do not become a paper trail
    of an old name if a peer renames themselves on the platform.
    """
    if peer_id is not None:
        return f"<peer:{peer_id}>"
    if actor is None:
        return "<actor:unknown>"
    return f"<actor:{actor.service}:{actor.account_id}>"


__all__ = ["ActorIdentity", "describe_actor_for_log"]
