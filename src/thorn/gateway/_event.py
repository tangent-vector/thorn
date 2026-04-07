"""Event abstractions for the Thorn gateway.

Defines the two core types that every event source and the gateway
itself operate on:

- ``IncomingEvent``: a frozen data object describing something that
  happened (e.g. an @-mention on GitLab).
- ``EventSource``: abstract base class for pluggable event inputs
  (pollers, webhooks, message queues, etc.).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from thorn.runtime._session import SessionKey


@dataclass(frozen=True)
class IncomingEvent:
    """Pure-data description of an external event.

    The gateway routes each event to the appropriate agent based on
    ``session_key``.  The ``content`` field is the formatted prompt
    that the agent will receive.

    Attributes:
        source: Identifies the originating system (e.g. ``"gitlab"``).
        session_key: Determines which agent session handles this event.
        content: Human-readable prompt describing what happened.
        metadata: Source-specific data (project IDs, TODO IDs, etc.)
            for use by tools or diagnostics.
    """

    source: str
    session_key: SessionKey
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


class EventSource(ABC):
    """Abstract base class for pluggable event inputs.

    Implementations call the *on_event* callback supplied to
    :meth:`start` whenever a new event is detected.  The gateway
    handles routing and agent prompting.
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
