"""Machine-readable provenance for built-in ``edit_file`` results."""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass

_PROVENANCE_PREFIX = "[Edit result provenance: changed lines "
_DELETION_ANCHOR_SEPARATOR = "; deletion anchors "
_PROVENANCE_SUFFIX = "]"
_RANGE_TOKEN_RE = re.compile(r"(?P<start>[1-9]\d*)(?:-(?P<end>[1-9]\d*))?")


@dataclass(frozen=True, order=True)
class ChangedLineRange:
    """A 1-based inclusive range whose final-file text changed."""

    start_line: int
    end_line: int

    def __post_init__(self) -> None:
        if self.start_line < 1 or self.end_line < self.start_line:
            raise ValueError("changed line ranges must be non-empty and 1-based")


@dataclass(frozen=True)
class FileEditResultProvenance:
    """Changed final-file ranges recorded in an ``edit_file`` result.

    The rendered line is deliberately separate from the human-readable file
    excerpt.  Consumers can therefore distinguish lines that truly changed
    from unchanged context included around the edit.
    """

    changed_line_ranges: tuple[ChangedLineRange, ...]
    deletion_anchor_line_numbers: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        previous_end_line = 0
        for changed_range in self.changed_line_ranges:
            if changed_range.start_line <= previous_end_line:
                raise ValueError(
                    "changed line ranges must be ordered and non-overlapping",
                )
            previous_end_line = changed_range.end_line
        if any(
            line_number < 1
            for line_number in self.deletion_anchor_line_numbers
        ):
            raise ValueError("deletion anchor line numbers must be 1-based")
        if tuple(sorted(set(self.deletion_anchor_line_numbers))) != (
            self.deletion_anchor_line_numbers
        ):
            raise ValueError(
                "deletion anchor line numbers must be sorted and unique",
            )

    @property
    def changed_line_numbers(self) -> tuple[int, ...]:
        """Return changed final-file line numbers in ascending order."""
        return tuple(
            line_number
            for changed_range in self.changed_line_ranges
            for line_number in range(
                changed_range.start_line,
                changed_range.end_line + 1,
            )
        )

    def render_result_line(self) -> str:
        """Render the stable provenance line included in tool output."""
        rendered_ranges = ", ".join(
            str(changed_range.start_line)
            if changed_range.start_line == changed_range.end_line
            else f"{changed_range.start_line}-{changed_range.end_line}"
            for changed_range in self.changed_line_ranges
        )
        rendered_anchors = ", ".join(
            str(line_number)
            for line_number in self.deletion_anchor_line_numbers
        )
        return (
            f"{_PROVENANCE_PREFIX}{rendered_ranges or 'none'}"
            f"{_DELETION_ANCHOR_SEPARATOR}{rendered_anchors or 'none'}"
            f"{_PROVENANCE_SUFFIX}"
        )

    @property
    def required_line_numbers(self) -> tuple[int, ...]:
        """Return exact changes followed by surviving deletion anchors."""
        return tuple(dict.fromkeys((
            *self.changed_line_numbers,
            *self.deletion_anchor_line_numbers,
        )))

    @classmethod
    def from_tool_result(
        cls,
        content: str,
    ) -> FileEditResultProvenance | None:
        """Parse provenance, returning ``None`` for legacy/invalid results."""
        provenance_lines = [
            line
            for line in content.splitlines()
            if line.startswith(_PROVENANCE_PREFIX)
        ]
        if len(provenance_lines) != 1:
            return None
        provenance_line = provenance_lines[0]
        if not provenance_line.endswith(_PROVENANCE_SUFFIX):
            return None
        rendered_provenance = provenance_line[
            len(_PROVENANCE_PREFIX):-len(_PROVENANCE_SUFFIX)
        ]
        provenance_parts = rendered_provenance.split(
            _DELETION_ANCHOR_SEPARATOR,
        )
        if len(provenance_parts) != 2:
            return None
        rendered_ranges, rendered_anchors = provenance_parts
        if rendered_ranges == "none":
            parsed_ranges: list[ChangedLineRange] = []
        else:
            parsed_ranges = []
            for range_token in rendered_ranges.split(", "):
                match = _RANGE_TOKEN_RE.fullmatch(range_token)
                if match is None:
                    return None
                start_line = int(match.group("start"))
                end_group = match.group("end")
                end_line = int(end_group) if end_group is not None else start_line
                try:
                    parsed_ranges.append(ChangedLineRange(start_line, end_line))
                except ValueError:
                    return None

        if rendered_anchors == "none":
            parsed_anchors: tuple[int, ...] = ()
        else:
            anchor_tokens = rendered_anchors.split(", ")
            if any(not token.isdecimal() for token in anchor_tokens):
                return None
            parsed_anchors = tuple(int(token) for token in anchor_tokens)
        try:
            return cls(tuple(parsed_ranges), parsed_anchors)
        except ValueError:
            return None


def changed_line_ranges_between(
    original_content: str,
    edited_content: str,
) -> tuple[ChangedLineRange, ...]:
    """Return final-file line ranges changed between two text versions."""
    return file_edit_result_provenance_between(
        original_content,
        edited_content,
    ).changed_line_ranges


def file_edit_result_provenance_between(
    original_content: str,
    edited_content: str,
) -> FileEditResultProvenance:
    """Return exact final changes and surviving anchors for deletions."""
    original_lines = original_content.splitlines()
    edited_lines = edited_content.splitlines()
    matcher = difflib.SequenceMatcher(
        a=original_lines,
        b=edited_lines,
        autojunk=False,
    )
    changed_ranges: list[ChangedLineRange] = []
    deletion_anchor_line_numbers: set[int] = set()
    for operation, _original_start, _original_end, edited_start, edited_end in (
        matcher.get_opcodes()
    ):
        if operation == "equal":
            continue
        if edited_start == edited_end:
            if edited_lines:
                deletion_anchor_line_numbers.add(min(
                    edited_start + 1,
                    len(edited_lines),
                ))
            continue
        changed_range = ChangedLineRange(edited_start + 1, edited_end)
        if (
            changed_ranges
            and changed_range.start_line <= changed_ranges[-1].end_line + 1
        ):
            previous_range = changed_ranges[-1]
            changed_ranges[-1] = ChangedLineRange(
                previous_range.start_line,
                changed_range.end_line,
            )
            continue
        changed_ranges.append(changed_range)
    changed_line_numbers = {
        line_number
        for changed_range in changed_ranges
        for line_number in range(
            changed_range.start_line,
            changed_range.end_line + 1,
        )
    }
    deletion_anchor_line_numbers -= changed_line_numbers
    return FileEditResultProvenance(
        tuple(changed_ranges),
        tuple(sorted(deletion_anchor_line_numbers)),
    )
