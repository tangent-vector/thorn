"""Session key and agent ID types for persistent agent conversations.

``SessionKey`` is a custom ``str`` subclass providing type-level
distinction between session identifiers and arbitrary strings.

``AgentID`` is a custom ``str`` subclass providing type-level
distinction for agent instance identifiers within a runtime.
"""

from __future__ import annotations


class SessionKey(str):
    """Opaque, unique identifier for an agent session.

    Inherits all ``str`` operations but provides type-level distinction
    so that APIs can require a ``SessionKey`` rather than accepting
    arbitrary strings.
    """


class AgentID(str):
    """Unique identifier for an agent instance within a runtime.

    Inherits all ``str`` operations but provides type-level distinction
    so that APIs can require an ``AgentID`` rather than accepting
    arbitrary strings.
    """


__all__ = [
    "AgentID",
    "SessionKey",
]
