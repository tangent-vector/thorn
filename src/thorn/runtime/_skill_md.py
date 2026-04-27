"""SKILL.md parsing: a small, focused YAML-frontmatter reader.

A ``SKILL.md`` advertises an agent skill via a YAML frontmatter
block followed by a Markdown body that describes how the skill is
meant to be used.  The frontmatter form is::

    ---
    description: One-line summary the agent sees in the prompt.
    ---

    # Remaining markdown body...

Only ``description`` is required at this iteration; additional
keys (``allowed-tools``, prerequisites, model preferences, ...) are
permitted and silently ignored, so format growth doesn't have to
land in lock-step with parser changes.

This module is deliberately independent of
:mod:`thorn.runtime._context_layers` so the parsing concerns
(frontmatter location, YAML decoding, error categorisation) live in
one place and have their own focused unit tests.  The discovery
walk in the layers module just calls :func:`parse_skill_md` and
turns successful results into :class:`SkillEntry` objects.

Error policy
------------
Parsing surfaces three distinct failure shapes via
:exc:`SkillMdError` subclasses, so callers can decide whether each
is loud or silent.  In the context-gathering pipeline every one of
them produces a logged warning and a skipped skill -- a single
malformed SKILL.md should never tank the entire pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class SkillMdError(Exception):
    """Base class for SKILL.md parse failures.

    Carries the *path* of the offending file so callers can include
    it in log messages without re-threading the path through their
    own error handlers.
    """

    def __init__(self, path: Path, message: str) -> None:
        super().__init__(f"{path}: {message}")
        self.path = path
        self.message = message


class MissingFrontmatterError(SkillMdError):
    """Raised when the file does not begin with a ``---`` fence.

    SKILL.md without frontmatter has no machine-readable
    description and is therefore unusable as a skill advertisement.
    """


class MalformedFrontmatterError(SkillMdError):
    """Raised when the frontmatter block does not parse as YAML, or is not a mapping.

    Wraps the underlying :class:`yaml.YAMLError` (when applicable)
    in *__cause__* so the original parse error is still inspectable.
    """


class MissingDescriptionError(SkillMdError):
    """Raised when frontmatter is well-formed YAML but lacks a usable ``description``.

    Either the key is absent, or its value is not a non-empty
    string after whitespace normalisation.  We're strict on this
    one because the description is the single piece of metadata the
    pipeline propagates into the system prompt -- without it the
    skill is invisible to the agent.
    """


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ParsedSkillMd:
    """The successful output of :func:`parse_skill_md`.

    *description* is the one-line summary lifted from frontmatter,
    already stripped of surrounding whitespace.  *body* is the
    remainder of the file after the closing ``---`` fence (with the
    leading newline that follows the fence removed); callers that
    only need the description -- like the current discovery walk
    -- can ignore it, but it's preserved here so future code that
    wants to surface skill content directly doesn't have to re-read
    the file.
    """

    description: str
    body: str


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

# The frontmatter fence; per the spec the very first line of the
# file must be exactly ``---`` and a matching ``---`` line must
# close the block before any body content.
_FENCE: str = "---"


def _split_frontmatter(text: str, *, path: Path) -> tuple[str, str]:
    """Split *text* into ``(yaml_block, body)`` around its frontmatter fence.

    Raises :exc:`MissingFrontmatterError` if the file does not open
    with a ``---`` line, or :exc:`MalformedFrontmatterError` if the
    opening fence is present but the closing fence is missing
    (so the entire file would be eaten as frontmatter).
    """
    lines = text.splitlines(keepends=True)
    if not lines or lines[0].rstrip("\r\n") != _FENCE:
        raise MissingFrontmatterError(
            path, "file does not start with a '---' frontmatter fence",
        )

    for idx in range(1, len(lines)):
        if lines[idx].rstrip("\r\n") == _FENCE:
            yaml_block = "".join(lines[1:idx])
            body = "".join(lines[idx + 1:])
            return yaml_block, body

    raise MalformedFrontmatterError(
        path, "frontmatter is not closed by a matching '---' fence",
    )


def parse_skill_md(path: Path, text: str) -> ParsedSkillMd:
    """Parse *text* (the contents of *path*) into a :class:`ParsedSkillMd`.

    *path* is taken in addition to *text* purely for error
    reporting; the function does no I/O.  Callers are expected to
    have already done the ``read_text`` so the I/O failure mode
    (warn + skip) lives entirely with the discovery walk and not
    with the parser.

    Raises a :exc:`SkillMdError` subclass on every failure mode.
    Callers that need to skip-with-warn rather than propagate
    should catch :exc:`SkillMdError` itself.
    """
    yaml_block, body = _split_frontmatter(text, path=path)

    try:
        loaded = yaml.safe_load(yaml_block)
    except yaml.YAMLError as exc:
        raise MalformedFrontmatterError(
            path, f"YAML parse failed: {exc}",
        ) from exc

    if loaded is None:
        raise MalformedFrontmatterError(
            path, "frontmatter is empty",
        )
    if not isinstance(loaded, dict):
        raise MalformedFrontmatterError(
            path,
            f"frontmatter must be a mapping, got {type(loaded).__name__}",
        )

    raw_description = loaded.get("description")
    if not isinstance(raw_description, str):
        raise MissingDescriptionError(
            path,
            "frontmatter is missing a string-valued 'description' field",
        )
    description = raw_description.strip()
    if not description:
        raise MissingDescriptionError(
            path, "'description' field is empty",
        )

    return ParsedSkillMd(description=description, body=body)


__all__ = [
    "MalformedFrontmatterError",
    "MissingDescriptionError",
    "MissingFrontmatterError",
    "ParsedSkillMd",
    "SkillMdError",
    "parse_skill_md",
]
