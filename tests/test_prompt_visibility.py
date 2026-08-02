"""Tests for prompt-render file-content visibility facts."""

from __future__ import annotations

from pathlib import Path

from thorn.core._prompt_visibility import (
    FileLineContent,
    PromptVisibilitySnapshotBuilder,
    PromptVisibilityStore,
    file_content_key_for_path,
    parse_line_numbered_file_content,
)


def test_parse_line_numbered_file_content_accepts_marker_column() -> None:
    parsed = parse_line_numbered_file_content(
        "  7*| alpha\n"
        "  8 |     beta\n"
        "[lines 9-12, 4 lines -- use offset/limit to read]\n",
    )

    assert parsed == (
        FileLineContent(line_number=7, content="alpha"),
        FileLineContent(line_number=8, content="    beta"),
    )


def test_compare_file_content_reports_exact_visibility(tmp_path: Path) -> None:
    file_key = file_content_key_for_path("example.py", workspace_root=tmp_path)
    assert file_key is not None
    builder = PromptVisibilitySnapshotBuilder()
    builder.record_visible_file_lines(
        file_key,
        [
            FileLineContent(line_number=1, content="alpha"),
            FileLineContent(line_number=2, content="beta"),
        ],
    )
    snapshot = builder.build()

    comparison = snapshot.compare_file_content(
        file_key,
        [
            FileLineContent(line_number=1, content="alpha"),
            FileLineContent(line_number=2, content="changed"),
            FileLineContent(line_number=3, content="gamma"),
        ],
    )

    assert comparison.compared_line_count == 3
    assert comparison.exact_visible_line_count == 1
    assert comparison.visible_line_mismatch_count == 1
    assert comparison.not_visible_line_count == 1
    assert comparison.exact_visible_fraction == 1 / 3


def test_visibility_store_bounds_retained_snapshots(tmp_path: Path) -> None:
    file_key = file_content_key_for_path("example.py", workspace_root=tmp_path)
    assert file_key is not None
    builder = PromptVisibilitySnapshotBuilder()
    builder.record_visible_file_lines(
        file_key,
        [FileLineContent(line_number=1, content="alpha")],
    )

    store = PromptVisibilityStore(max_snapshots=1)
    store.remember("old", builder.build())
    store.remember("new", builder.build())

    assert store.get("old") is None
    assert store.get("new") is not None
