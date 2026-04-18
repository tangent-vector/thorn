"""Event abstractions for the Thorn gateway.

Defines the two core types that every event source and the gateway
itself operate on:

- ``IncomingEvent``: a frozen data object describing something that
  happened (e.g. an @-mention on GitLab).
- ``EventSource``: abstract base class for pluggable event inputs
  (pollers, webhooks, message queues, etc.).
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from thorn.core._service import Service
from thorn.runtime._session import AgentID, SessionKey


@dataclass(frozen=True)
class IncomingEvent:
    """Pure-data description of an external event.

    The gateway routes each event to the appropriate agent based on
    ``agent_id`` (optional) and ``session_key``.  The ``content`` field
    is the formatted prompt that the agent will receive.

    Workspace paths are never carried on the event — the gateway
    derives them mechanically from the agent ID and session key (see
    ``Gateway._handle_event``).

    Attributes:
        source: Identifies the originating system (e.g. ``"gitlab"``).
        session_key: Determines which session handles this event,
            scoped under the resolved agent.
        content: Human-readable prompt describing what happened.
        agent_id: Optional identifier for the agent that should handle
            this event.  When ``None``, the gateway's default routing
            logic applies.
        metadata: Source-specific data (project IDs, TODO IDs, etc.)
            for use by tools or diagnostics.
        external_key: Optional source-namespaced stable identifier for
            the external entity this event represents (e.g.
            ``"gitlab:https://gitlab.example.com:todo:42"``).  When
            set, the gateway uses the runtime's
            :class:`~thorn.runtime.InFlightIndex` to silently drop the
            event if another notification for the same key is still in
            flight somewhere in the agency.  This is the primary
            mechanism by which sources avoid re-posting the same
            external event across polls (including after a restart,
            thanks to the index's filesystem-backed rebuild).
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
    handles routing and agent prompting.

    ``EventSource`` is a :class:`Service` subclass.  Subclasses must
    define a ``Config`` class attribute pointing to a
    :class:`pydantic.BaseModel` subclass that describes the source's
    configuration, and implement :attr:`name`.
    """

    @abstractmethod
    async def start(
        self,
        on_event: Callable[[IncomingEvent], Awaitable[None]],
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


__all__ = [
    "EventSource",
    "IncomingEvent",
]
