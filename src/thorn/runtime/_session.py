"""Session key type for persistent agent conversations.

``SessionKey`` is a custom ``str`` subclass providing type-level
distinction between session identifiers and arbitrary strings.
"""

from __future__ import annotations


class SessionKey(str):
    """Opaque, unique identifier for an agent session.

    Inherits all ``str`` operations but provides type-level distinction
    so that APIs can require a ``SessionKey`` rather than accepting
    arbitrary strings.
    """


__all__ = [
    "SessionKey",
]
