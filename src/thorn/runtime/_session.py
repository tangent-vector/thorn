"""Session key and agent ID types for persistent agent conversations.

``SessionKey`` is a structured value type whose canonical internal
representation is a tuple of string components.  Its string form is the
``/``-join of those components, and the constructor accepts either a
string (parsed by splitting on ``/``) or an iterable of explicit
components.  It is deliberately **not** a ``str`` subclass: that
historical shape encouraged callers to treat session keys as opaque
strings and silently flattened the hierarchical ``/``-separated form
into a single filesystem-safe atom in places that wanted real directory
nesting.

The only invariants enforced at construction time are structural:

- at least one component,
- every component non-empty,
- no ``/`` inside any single component.

Reservation policies on segment names (e.g. ``_``-prefixed names being
framework-reserved) are intentionally *not* enforced here.  Those
belong at the session-key-template layer (see ``coordination.md``)
where the policy can examine the surrounding template and the operator
intent, not the individual ``SessionKey`` value.

``AgentID`` remains a custom ``str`` subclass: agent IDs are flat
single-segment identifiers and the str-subclass pattern fits them
naturally.
"""

from __future__ import annotations

from typing import Iterable, Iterator


class SessionKey:
    """Hierarchical identifier for an agent session.

    Stored as an immutable sequence of string components.  The string
    representation joins them with ``/`` for display, persistence, and
    on-the-wire use; parsing splits them back.  Components may not
    contain ``/`` or be empty, and the key as a whole must have at
    least one component.

    Two construction forms are supported, both via the same
    ``SessionKey(value)`` call:

    - ``SessionKey("a/b/c")`` -- a single ``str`` is split on ``/``,
      so this produces a key with components ``("a", "b", "c")``.
    - ``SessionKey(("a", "b", "c"))`` -- any non-string iterable is
      treated as the literal component sequence.

    The string form is provided for backward compatibility with the
    many existing call sites that were written when ``SessionKey``
    derived from ``str``.  New code that already has the components
    in hand should prefer the iterable form (it is unambiguous and
    skips the parse step).
    """

    __slots__ = ("_components",)

    def __init__(self, value: str | Iterable[str]) -> None:
        if isinstance(value, str):
            if not value:
                raise ValueError("SessionKey string must be non-empty")
            components: tuple[str, ...] = tuple(value.split("/"))
        else:
            components = tuple(value)
        if not components:
            raise ValueError("SessionKey must have at least one component")
        for component in components:
            if not component:
                raise ValueError(
                    f"SessionKey components must be non-empty: {components!r}"
                )
            if "/" in component:
                raise ValueError(
                    f"SessionKey component must not contain '/': {component!r}"
                )
        object.__setattr__(self, "_components", components)

    @property
    def components(self) -> tuple[str, ...]:
        """The session key's components, outermost first."""
        return self._components

    def __str__(self) -> str:
        return "/".join(self._components)

    def __repr__(self) -> str:
        return f"SessionKey({str(self)!r})"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, SessionKey):
            return self._components == other._components
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self._components)

    def __iter__(self) -> Iterator[str]:
        return iter(self._components)

    def __len__(self) -> int:
        return len(self._components)


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
