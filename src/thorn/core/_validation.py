"""ValidationRule -- a named, registered validation check.

Each ``ValidationRule`` auto-registers itself by name in a global
registry.  Rules can be looked up by name via
``ValidationRule.lookup(name)``.

The global registry is a stopgap; scoped registries will replace it
when scope-based composition is implemented.

Example::

    BUILD = ValidationRule("build", check=build_project)
    TEST  = ValidationRule("test",  check=run_tests)

    # Later, during delegation:
    rule = ValidationRule.lookup("build")
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, ClassVar


class ValidationRule:
    """A named validation check with global auto-registration."""

    _registry: ClassVar[dict[str, ValidationRule]] = {}

    def __init__(self, name: str, *, check: Callable[..., Any]) -> None:
        if not name:
            raise ValueError("ValidationRule name must be non-empty")
        self.name = name
        self.check = check
        ValidationRule._registry[name] = self

    @classmethod
    def lookup(cls, name: str) -> ValidationRule | None:
        """Return the rule registered under *name*, or ``None``."""
        return cls._registry.get(name)

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, ValidationRule) and self.name == other.name

    def __repr__(self) -> str:
        return f"ValidationRule({self.name!r})"
