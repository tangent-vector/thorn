"""Session types for persistent agent conversations.

A ``Session`` wraps an ``Agent`` with lifecycle metadata (key, timestamps,
application-specific metadata).  Sessions are the unit of persistence:
the ``SessionStore`` knows how to serialize and restore them.

``SessionKey`` is a custom ``str`` subclass providing type-level
distinction between session identifiers and arbitrary strings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from thorn.core._agent import Agent


class SessionKey(str):
    """Opaque, unique identifier for an agent session.

    Inherits all ``str`` operations but provides type-level distinction
    so that APIs can require a ``SessionKey`` rather than accepting
    arbitrary strings.
    """


@dataclass
class Session:
    """A single agent conversation context with lifecycle metadata.

    Wraps an ``Agent`` (which owns the conversation ``HistoryTree``)
    with session-level bookkeeping: a unique key, timestamps, and an
    application-specific metadata dict.

    The ``metadata`` dict is separate from ``Agent.metadata`` --
    session metadata describes the conversation context (e.g., which
    GitLab issue this session is for), while agent metadata describes
    the agent instance itself (e.g., role configuration).
    """

    key: SessionKey
    agent: Agent
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc),
    )
    last_active: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc),
    )

    def touch(self) -> None:
        """Update ``last_active`` to the current UTC time."""
        self.last_active = datetime.now(timezone.utc)


__all__ = [
    "Session",
    "SessionKey",
]
