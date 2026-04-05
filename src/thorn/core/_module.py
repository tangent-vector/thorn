"""ModulePath -- a type-safe position in a module hierarchy.

A module path is a sequence of zero or more string segments.  The
zero-segment path represents the root of the hierarchy.  Segments are
separated by dots in string form, and the root is displayed as ``_``.

Examples::

    ModulePath("calc.parser")        # two segments: ("calc", "parser")
    ModulePath("_.calc.parser")      # same -- leading _ is stripped
    ModulePath(("calc", "parser"))   # tuple form
    ModulePath("")                   # root (zero segments)
    ModulePath.root()                # root (classmethod)

    path = ModulePath("calc")
    child = path.child("parser")     # ModulePath("calc.parser")
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterator

_SEGMENT_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")


@dataclass(frozen=True, init=False)
class ModulePath:
    """A position in the module hierarchy, stored as a tuple of segments.

    Construct from a dotted string, a tuple of segments, or another
    ``ModulePath``.  The constructor follows pathlib-style conventions:
    ``ModulePath("calc.parser")`` Just Works.

    Future consideration: an option for ``_`` in segment names to map
    to ``-`` in filenames (e.g. ``my_module`` -> ``my-module/``).
    """

    segments: tuple[str, ...]

    def __init__(
        self, source: str | tuple[str, ...] | ModulePath = ()
    ) -> None:
        if isinstance(source, ModulePath):
            object.__setattr__(self, "segments", source.segments)
            return

        if isinstance(source, str):
            segments = _parse_dotted(source)
        elif isinstance(source, tuple):
            _validate_segments(segments := source)
        else:
            raise TypeError(
                f"ModulePath() requires str, tuple, or ModulePath, "
                f"got {type(source).__name__}"
            )
        object.__setattr__(self, "segments", segments)

    @classmethod
    def root(cls) -> ModulePath:
        """Return the root (zero-segment) module path."""
        return cls(())

    def child(self, name: str) -> ModulePath:
        """Return a new path with *name* appended as the last segment."""
        _validate_segment(name)
        return ModulePath(self.segments + (name,))

    @property
    def parent(self) -> ModulePath | None:
        """The parent path, or ``None`` for the root."""
        if not self.segments:
            return None
        return ModulePath(self.segments[:-1])

    @property
    def name(self) -> str:
        """The last segment, or the empty string for the root."""
        return self.segments[-1] if self.segments else ""

    @property
    def is_root(self) -> bool:
        """True for the zero-segment root path."""
        return not self.segments

    @property
    def depth(self) -> int:
        """The number of segments (0 for root)."""
        return len(self.segments)

    def __iter__(self) -> Iterator[str]:
        return iter(self.segments)

    def __str__(self) -> str:
        return ".".join(self.segments) if self.segments else "_"

    def __format__(self, format_spec: str) -> str:
        return format(str(self), format_spec)

    def __repr__(self) -> str:
        return f"ModulePath({str(self)!r})"


# ---------------------------------------------------------------------------
# Parsing and validation helpers
# ---------------------------------------------------------------------------


def _parse_dotted(s: str) -> tuple[str, ...]:
    """Parse a dotted string into validated segments."""
    if not s:
        return ()

    parts = s.split(".")

    # Leading "_" is the root prefix -- strip it.
    if parts and parts[0] == "_":
        parts = parts[1:]

    # After stripping, nothing left (input was "_" or "_.") → root.
    if not parts or parts == [""]:
        return ()

    for part in parts:
        _validate_segment(part, original=s)

    return tuple(parts)


def _validate_segment(seg: str, *, original: str | None = None) -> None:
    """Raise ``ValueError`` if *seg* is not a valid segment name."""
    ctx = f" in {original!r}" if original else ""
    if not seg:
        raise ValueError(
            f"empty module-path segment{ctx}: "
            f"double dot or trailing dot"
        )
    if seg == "_":
        raise ValueError(
            "'_' is reserved for the root module path "
            "and cannot be used as a segment name"
        )
    if not _SEGMENT_RE.match(seg):
        raise ValueError(
            f"invalid module-path segment {seg!r}: "
            f"must match [a-zA-Z_][a-zA-Z0-9_]*"
        )


def _validate_segments(segs: tuple[str, ...]) -> None:
    """Validate every element of a segments tuple."""
    for seg in segs:
        if not isinstance(seg, str):
            raise TypeError(
                f"ModulePath segments must be strings, "
                f"got {type(seg).__name__}"
            )
        _validate_segment(seg)
