"""Conservative classification for shell commands that inspect files."""

from __future__ import annotations

import re
import shlex
from dataclasses import dataclass
from enum import StrEnum


class ShellInspectionKind(StrEnum):
    """Broad kind of read-only shell inspection."""

    READ = "read"
    SEARCH = "search"


@dataclass(frozen=True)
class ShellInspectionCommand:
    """A shell command recognized as read-only file inspection."""

    kind: ShellInspectionKind
    command_name: str
    normalized_command: str
    path: str | None = None
    pattern: str | None = None
    start_line: int | None = None
    end_line: int | None = None
    line_range_label: str | None = None


_CONTROL_CHARS = frozenset("|;&<>")
_SED_PRINT_RANGE_RE = re.compile(r"^(?P<start>\d+)(?:,(?P<end>\d+))?p$")
_OPTIONS_WITH_VALUES = frozenset({
    "-A",
    "-B",
    "-C",
    "-e",
    "-f",
    "-g",
    "-m",
    "-t",
    "--after-context",
    "--before-context",
    "--context",
    "--file",
    "--glob",
    "--max-count",
    "--regexp",
    "--type",
})


def parse_shell_inspection_command(
    command: str,
) -> ShellInspectionCommand | None:
    """Return a read/search classification for simple read-only commands.

    This is intentionally not a general shell parser.  It recognizes a
    bounded set of single-command invocations and declines anything with
    shell control syntax, redirection, or multiple target paths.
    """
    tokens = _safe_split(command)
    if not tokens:
        return None

    command_name = tokens[0].lower()
    normalized_command = " ".join(tokens)
    match command_name:
        case "cat":
            return _parse_cat(tokens, normalized_command)
        case "sed":
            return _parse_sed(tokens, normalized_command)
        case "head":
            return _parse_head(tokens, normalized_command)
        case "tail":
            return _parse_tail(tokens, normalized_command)
        case "grep":
            return _parse_search(
                tokens,
                normalized_command,
                command_name="grep",
                default_path=None,
            )
        case "rg":
            return _parse_search(
                tokens,
                normalized_command,
                command_name="rg",
                default_path=".",
            )
        case _:
            return None


def _safe_split(command: str) -> tuple[str, ...]:
    try:
        tokens = tuple(shlex.split(command, posix=True))
    except ValueError:
        return ()
    if any(_has_shell_control(token) for token in tokens):
        return ()
    return tokens


def _has_shell_control(token: str) -> bool:
    return any(char in token for char in _CONTROL_CHARS)


def _parse_cat(
    tokens: tuple[str, ...],
    normalized_command: str,
) -> ShellInspectionCommand | None:
    operands = _operands_after_options(tokens[1:])
    if len(operands) != 1:
        return None
    return ShellInspectionCommand(
        kind=ShellInspectionKind.READ,
        command_name="cat",
        normalized_command=normalized_command,
        path=operands[0],
    )


def _parse_sed(
    tokens: tuple[str, ...],
    normalized_command: str,
) -> ShellInspectionCommand | None:
    quiet = False
    script: str | None = None
    operands: list[str] = []
    index = 1
    while index < len(tokens):
        token = tokens[index]
        if token in {"-n", "--quiet", "--silent"}:
            quiet = True
            index += 1
            continue
        if token == "-e":
            if index + 1 >= len(tokens):
                return None
            script = tokens[index + 1]
            index += 2
            continue
        if token.startswith("-"):
            return None
        if script is None:
            script = token
        else:
            operands.append(token)
        index += 1

    if not quiet or script is None or len(operands) != 1:
        return None

    match = _SED_PRINT_RANGE_RE.fullmatch(script)
    if match is None:
        return None
    start = int(match.group("start"))
    end = int(match.group("end") or start)
    if end < start:
        return None
    return ShellInspectionCommand(
        kind=ShellInspectionKind.READ,
        command_name="sed",
        normalized_command=normalized_command,
        path=operands[0],
        start_line=start,
        end_line=end,
        line_range_label=f"{start}-{end}",
    )


def _parse_head(
    tokens: tuple[str, ...],
    normalized_command: str,
) -> ShellInspectionCommand | None:
    line_count, operands = _line_count_and_operands(tokens[1:], default=10)
    if len(operands) != 1:
        return None
    return ShellInspectionCommand(
        kind=ShellInspectionKind.READ,
        command_name="head",
        normalized_command=normalized_command,
        path=operands[0],
        start_line=1,
        end_line=line_count,
        line_range_label=f"1-{line_count}",
    )


def _parse_tail(
    tokens: tuple[str, ...],
    normalized_command: str,
) -> ShellInspectionCommand | None:
    line_count, operands = _line_count_and_operands(tokens[1:], default=10)
    if len(operands) != 1:
        return None
    return ShellInspectionCommand(
        kind=ShellInspectionKind.READ,
        command_name="tail",
        normalized_command=normalized_command,
        path=operands[0],
        line_range_label=f"last {line_count}",
    )


def _parse_search(
    tokens: tuple[str, ...],
    normalized_command: str,
    *,
    command_name: str,
    default_path: str | None,
) -> ShellInspectionCommand | None:
    if _has_unsupported_pattern_option(tokens[1:]):
        return None
    operands = _operands_after_options(tokens[1:])
    if not operands:
        return None
    pattern = operands[0]
    paths = operands[1:]
    if not paths:
        if default_path is None:
            return None
        paths = [default_path]
    if len(paths) != 1:
        return None
    return ShellInspectionCommand(
        kind=ShellInspectionKind.SEARCH,
        command_name=command_name,
        normalized_command=normalized_command,
        path=paths[0],
        pattern=pattern,
    )


def _has_unsupported_pattern_option(args: tuple[str, ...]) -> bool:
    return any(
        token in {"-e", "-f", "--file", "--regexp"}
        or token.startswith("--file=")
        or token.startswith("--regexp=")
        for token in args
    )


def _line_count_and_operands(
    args: tuple[str, ...],
    *,
    default: int,
) -> tuple[int, list[str]]:
    line_count = default
    operands: list[str] = []
    index = 0
    while index < len(args):
        token = args[index]
        if token == "--":
            operands.extend(args[index + 1:])
            break
        if token == "-n":
            if index + 1 >= len(args):
                return line_count, []
            parsed = _parse_positive_int(args[index + 1])
            if parsed is None:
                return line_count, []
            line_count = parsed
            index += 2
            continue
        if token.startswith("-") and len(token) > 1:
            parsed = _parse_positive_int(token[1:])
            if parsed is None:
                return line_count, []
            line_count = parsed
            index += 1
            continue
        operands.append(token)
        index += 1
    return line_count, operands


def _operands_after_options(args: tuple[str, ...]) -> list[str]:
    operands: list[str] = []
    index = 0
    while index < len(args):
        token = args[index]
        if token == "--":
            operands.extend(args[index + 1:])
            break
        if token.startswith("--") and "=" in token:
            index += 1
            continue
        if token in _OPTIONS_WITH_VALUES:
            index += 2
            continue
        if token.startswith("-"):
            index += 1
            continue
        operands.append(token)
        index += 1
    return operands


def _parse_positive_int(text: str) -> int | None:
    if not text.isdecimal():
        return None
    try:
        value = int(text)
    except ValueError:
        return None
    return value if value > 0 else None
