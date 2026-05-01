"""Agent-facing peer-lookup tools.

Three ``@tool`` functions exposed by default to gateway agents:

- :func:`peer_by_account` -- "is the user identified by
  ``(service, account_id)`` one of my peers?"
- :func:`find_peers_by_name` -- "I know somebody is called something
  like 'tess'; who are they?"
- :func:`list_peers` -- "show me the whole peer list."

All three return clean public ``Peer`` shapes -- id, name, kind,
and a list of ``(service, account_id)`` accounts.  No credentials
are exposed and no internal registry shape leaks into the tool
surface; the tool's :class:`pydantic.BaseModel` return types are
the contract.

Tools resolve the peer registry from
``get_context().runtime.peer_registry``.  When invoked outside a
gateway-backed runtime (e.g. unit tests using ``agent.prompt``
without a ``Gateway``), the registry is empty and every lookup
returns ``None`` / an empty list -- the correct strict-default
behaviour for "no peers configured."
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field

from thorn.core._context import get_context
from thorn.core._executor import ToolVenue
from thorn.core._func import tool

# Importing ``thorn.gateway._peer`` at module load time creates a
# circular import: ``thorn.gateway/__init__.py`` re-exports
# ``GatewayAgent`` (which imports ``PEER_TOOLS`` from this module),
# so any code path that loads ``thorn.tools.peers`` cold and then
# tries to reach into the gateway package fails halfway through.
# We dodge the cycle by deferring the gateway imports to the function
# bodies that actually need them (the registry-lookup helpers below)
# and by using ``Literal`` rather than ``PeerKind`` for the public
# ``Peer.kind`` field so the pydantic model can be constructed
# without the enum class loaded.
if TYPE_CHECKING:
    from thorn.gateway._peer import PeerRegistry, PeerSpec


PeerKindLiteral = Literal["human", "bot"]


class PeerAccountView(BaseModel):
    """The ``(service, account_id)`` view of one peer account.

    Public read-only shape for agent-facing tools.  Credentials are
    not part of the peer registry (they live on agent accounts, not
    peer accounts) and could not leak through this type even
    accidentally.
    """

    service: str = Field(description="Service name on which this account lives.")
    account_id: str = Field(
        description=(
            "Account id as the operator declared it.  May be the "
            "platform-immutable id or a textual handle; the registry "
            "matches both forms internally."
        ),
    )


class Peer(BaseModel):
    """Public shape returned by peer-lookup tools.

    Mirrors :class:`thorn.gateway.PeerSpec` minus the operator-only
    fields.  Returned from every tool in this module so the agent
    has a single shape to reason about.

    The ``kind`` field is typed as ``Literal["human", "bot"]`` rather
    than the internal ``PeerKind`` enum so that this module can be
    loaded without dragging in ``thorn.gateway`` (see the import
    block above for context on the cycle).
    """

    id: str = Field(description="Stable, write-once peer id.")
    name: str = Field(description="Current human-readable display name.")
    kind: PeerKindLiteral = Field(
        description="Whether this peer is a human or a bot.",
    )
    accounts: list[PeerAccountView] = Field(
        default_factory=list,
        description="Per-service identities for this peer.",
    )


def _peer_view(spec: PeerSpec) -> Peer:
    """Convert an internal ``PeerSpec`` to the public :class:`Peer` shape."""
    return Peer(
        id=spec.id,
        name=spec.name,
        # ``PeerKind`` is a StrEnum; ``.value`` gives the string form
        # ("human" / "bot") that the public ``Literal`` field accepts.
        kind=spec.kind.value,
        accounts=[
            PeerAccountView(service=a.service, account_id=a.account_id)
            for a in spec.accounts
        ],
    )


def _registry() -> PeerRegistry:
    """Locate the active :class:`PeerRegistry`, raising a clear error if absent."""
    ctx = get_context()
    if ctx.runtime is None:
        raise RuntimeError(
            "Peer-lookup tools require a Runtime in the current "
            "ExecutionContext.  When running an agent outside a "
            "Gateway, no peers are configured -- which is fine, but "
            "the Runtime instance still needs to be in scope."
        )
    return ctx.runtime.peer_registry


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@tool(venue=ToolVenue.IN_PROCESS)
async def peer_by_account(service: str, account_id: str) -> Peer | None:
    """Look up a peer by their account on a specific service.

    *service* matches a ``forges[].name`` (or other service spec
    name) in ``gateway.json``.  *account_id* may be the textual
    handle (e.g. a GitHub login) or the platform-immutable id; the
    registry tries both forms.

    Returns the matching :class:`Peer` or ``None`` if no peer in
    the gateway's peer list claims that account.  A return value of
    ``None`` is the canonical "this user is not one of my peers"
    answer -- the harness's own filtering already drops
    conversational events from non-peers, but the agent can use
    this tool to make a defensive check or to give a clear "I do
    not recognise you" response when a non-peer's content is
    surfaced via a forge tool.
    """
    registry = _registry()
    spec = registry.lookup_account(service, account_id)
    if spec is None:
        return None
    return _peer_view(spec)


@tool(venue=ToolVenue.IN_PROCESS)
async def find_peers_by_name(query: str) -> list[Peer]:
    """Search peers by display name (case-insensitive substring match).

    Returns every peer whose ``name`` field contains *query* as a
    case-insensitive substring.  An empty *query* returns an empty
    list (a substring match against the empty string would otherwise
    return every peer, which is almost never what callers want).

    Use this when an agent has a partial name to resolve into a
    peer id (e.g. for cross-referencing against ``~/peers/<id>/``
    notes).
    """
    registry = _registry()
    return [_peer_view(spec) for spec in registry.find_by_name(query)]


@tool(venue=ToolVenue.IN_PROCESS)
async def list_peers() -> list[Peer]:
    """Return every peer the gateway knows about, sorted by id.

    Intended for use cases where the agent wants to enumerate the
    peer list (e.g. to populate a status reply, or to walk
    ``~/peers/`` and reconcile per-peer notes against the current
    registry).  No pagination: peer lists are operator-curated and
    expected to remain small (<100s of entries) for the foreseeable
    future.
    """
    registry = _registry()
    return [_peer_view(spec) for spec in registry.all_peers()]


PEER_TOOLS = [peer_by_account, find_peers_by_name, list_peers]


__all__ = [
    "PEER_TOOLS",
    "Peer",
    "PeerAccountView",
    "find_peers_by_name",
    "list_peers",
    "peer_by_account",
]
