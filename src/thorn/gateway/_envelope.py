"""Envelope rendering for untrusted external content.

The gateway wraps any text whose author is not the agent itself --
comment bodies, issue/PR descriptions, fetched-thread tool output,
etc. -- in a structured envelope that the agent's system prompt
rule identifies as data, never instructions.

The envelope is a hybrid of two signals:

1.  An outer marker pair (``[external-content nonce=N ...]`` and
    ``[/external-content nonce=N]``) carrying structured per-block
    metadata (peer status, actor, source, kind) and a per-block
    nonce.  The nonce is the answer to "what if the body contains
    the closing marker as plain text?" -- a freshly generated
    8-hex-char nonce cannot be guessed by attacker-controlled
    content because the attacker's text was committed before the
    nonce was minted.

2.  A Markdown blockquote prefix ("``> ``") on every line of body
    text.  This piggy-backs on the model's strong training prior
    that blockquoted material is being quoted, not commanded.
    Blockquote prefixing also escapes naturally -- every line gets
    prefixed regardless of contents -- so we do not have to invent
    a separate body-escape scheme.

Envelopes nest cleanly because the blockquote prefix and the
nonce-bearing markers do not collide: a tool result that contains
multiple comments is rendered as a sequence of envelopes, each
with its own nonce.  The same :func:`wrap_external` helper is
used by the notification formatter (on the way into the agent's
inbox) and by forge tools (on the way out via tool results), so
the agent sees a single consistent labelling regardless of how
external content arrives.
"""

from __future__ import annotations

import secrets
from enum import StrEnum

from thorn.gateway._actor import ActorIdentity


class PeerStatus(StrEnum):
    """Whether the actor of a wrapped block is a known peer.

    Rendered as the ``peer=`` attribute on the envelope's opening
    marker.  Three values rather than a bool because there is a
    real difference between "actor exists and is not a peer"
    (``NON_PEER``) and "actor could not be determined at all"
    (``UNKNOWN``); the agent should treat them differently.
    """

    PEER = "yes"
    NON_PEER = "no"
    UNKNOWN = "unknown"


def _generate_nonce() -> str:
    """Return a fresh 8-hex-char nonce for a single envelope.

    8 hex chars (32 bits) is enough to make collision with
    attacker-controlled body text a non-issue: the attacker would
    have to guess a value that does not yet exist when their text
    is captured.
    """
    return secrets.token_hex(4)


def wrap_external(
    *,
    body: str,
    actor: ActorIdentity | None,
    source: str,
    kind: str,
    peer_status: PeerStatus = PeerStatus.UNKNOWN,
    timestamp: str = "",
    nonce: str | None = None,
) -> str:
    """Render *body* as an external-content envelope.

    The output is a multi-line string of the form::

        [external-content nonce=4f8a91 source=github actor=@alice peer=yes kind=comment]
        > @alice (2026-04-30T12:34Z):
        >
        > <body, every line prefixed with `> `>
        [/external-content nonce=4f8a91]

    Args:
        body: The raw text to wrap.  May be empty, in which case the
            envelope still renders (with an empty blockquote body) so
            the agent can see that there *was* an external item with
            no content rather than silently dropping it.
        actor: The author of *body*, or ``None`` when the source could
            not capture an actor.  Renders as the ``actor=`` marker
            attribute and as the leading attribution line of the
            blockquote.
        source: A short tag identifying the originating system
            (``"github"``, ``"gitlab"``, etc.).  Stamped on the
            opening marker so the agent can see where a quoted
            chunk came from.
        kind: A short tag describing what kind of content this is
            (``"comment"``, ``"issue_body"``, etc.).  Same role as
            *source* -- metadata for the agent.
        peer_status: Whether the actor matched a known peer.  Default
            ``UNKNOWN`` is intended for transitional code paths where
            the formatter has not yet consulted the registry; once
            the registry is in place every call should pass an
            explicit value.
        timestamp: Optional ISO-8601 timestamp; rendered into the
            attribution line if present.
        nonce: Override the auto-generated nonce.  Tests pass an
            explicit nonce so output is deterministic; production
            callers should leave this ``None``.

    Returns:
        The rendered envelope, no trailing newline.  Callers wanting
        to concatenate multiple envelopes should add their own
        separator (typically a blank line).
    """
    if nonce is None:
        nonce = _generate_nonce()

    if actor is not None:
        actor_label = actor.at_login_form()
    else:
        actor_label = "(unknown actor)"

    open_attrs = (
        f"nonce={nonce} source={source} actor={actor_label} "
        f"peer={peer_status.value} kind={kind}"
    )
    close_attrs = f"nonce={nonce}"

    # Attribution line lives inside the blockquote because it is
    # itself "data about the quoted content."  Putting it inside
    # the quote (rather than alongside the opening marker) keeps
    # the rule simple: every line beginning with `> ` is data.
    attribution = actor_label
    if timestamp:
        attribution = f"{attribution} ({timestamp})"
    attribution = f"{attribution}:"

    # Markdown blockquote: prefix every line with "> ", and use a
    # bare ">" for blank lines so the blockquote does not break.
    quoted_body_lines: list[str] = [f"> {attribution}", ">"]
    if body:
        for line in body.splitlines():
            if line:
                quoted_body_lines.append(f"> {line}")
            else:
                quoted_body_lines.append(">")
    else:
        quoted_body_lines.append("> (no body)")

    return "\n".join(
        [
            f"[external-content {open_attrs}]",
            *quoted_body_lines,
            f"[/external-content {close_attrs}]",
        ],
    )


__all__ = ["PeerStatus", "wrap_external"]
