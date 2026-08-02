"""Tests for conservative shell inspection classification."""

from __future__ import annotations

from thorn.core._shell_inspection import (
    ShellInspectionKind,
    parse_shell_inspection_command,
)


def test_parse_sed_print_range_read() -> None:
    inspection = parse_shell_inspection_command(
        "sed -n '10,14p' src/app.py",
    )

    assert inspection is not None
    assert inspection.kind is ShellInspectionKind.READ
    assert inspection.command_name == "sed"
    assert inspection.path == "src/app.py"
    assert inspection.start_line == 10
    assert inspection.end_line == 14
    assert inspection.line_range_label == "10-14"


def test_parse_rg_search_with_line_number_flag() -> None:
    inspection = parse_shell_inspection_command(
        "rg -n needle src",
    )

    assert inspection is not None
    assert inspection.kind is ShellInspectionKind.SEARCH
    assert inspection.command_name == "rg"
    assert inspection.pattern == "needle"
    assert inspection.path == "src"


def test_parse_rejects_shell_control_syntax() -> None:
    assert parse_shell_inspection_command(
        "cat src/app.py && rm src/app.py",
    ) is None


def test_parse_rejects_pattern_option_search_forms() -> None:
    assert parse_shell_inspection_command("rg -e needle src") is None
