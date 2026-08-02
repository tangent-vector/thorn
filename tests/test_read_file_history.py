from __future__ import annotations

from pathlib import Path

from thorn.core._prompt_visibility import (
    FileLineContent,
    file_content_key_for_path,
)
from thorn.core._read_file_history import (
    SESSION_LEDGER_V1_READ_FILE_REUSE_POLICY,
    ReadFileResultHistory,
    ReadFileReuseKind,
)


def _lines(*contents: str, start: int = 1) -> tuple[FileLineContent, ...]:
    return tuple(
        FileLineContent(line_number=line_number, content=content)
        for line_number, content in enumerate(contents, start=start)
    )


class TestReadFileResultHistory:
    def test_successful_read_without_numbered_lines_tracks_content_epoch(
        self,
        tmp_path: Path,
    ) -> None:
        target = tmp_path / "empty.txt"
        target.write_text("", encoding="utf-8")
        history = ReadFileResultHistory()

        observation = history.observe(
            call_id="empty-read",
            render_id="render-1",
            raw_path="empty.txt",
            workspace_root=tmp_path,
            returned_lines=(),
            policy=SESSION_LEDGER_V1_READ_FILE_REUSE_POLICY,
        )

        assert observation is not None
        assert observation.kind is ReadFileReuseKind.NO_RETURNED_LINES
        assert observation.content_epoch == 1
        assert observation.returned_start_line is None
        assert observation.returned_end_line is None
        assert observation.returned_line_count == 0
        assert not observation.hint_recommended
        assert history.retained_file_count == 1

    def test_exact_repeat_is_recognized_without_retaining_file_text(
        self,
        tmp_path: Path,
    ) -> None:
        target = tmp_path / "example.py"
        target.write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
        history = ReadFileResultHistory()

        first = history.observe(
            call_id="first",
            render_id="render-1",
            raw_path="example.py",
            workspace_root=tmp_path,
            returned_lines=_lines("alpha", "beta"),
            policy=SESSION_LEDGER_V1_READ_FILE_REUSE_POLICY,
        )
        repeated = history.observe(
            call_id="repeat",
            render_id="render-2",
            raw_path="example.py",
            workspace_root=tmp_path,
            returned_lines=_lines("alpha", "beta"),
            policy=SESSION_LEDGER_V1_READ_FILE_REUSE_POLICY,
        )

        assert first is not None
        assert first.kind is ReadFileReuseKind.FIRST_OBSERVATION
        assert not first.hint_recommended
        assert repeated is not None
        assert repeated.kind is ReadFileReuseKind.EXACT_REPEAT
        assert repeated.hint_recommended
        assert repeated.matching_prior_line_count == 2
        assert repeated.new_line_count == 0
        assert repeated.prior_call_id == "first"
        assert repeated.prior_render_id == "render-1"
        assert repeated.content_epoch == 1
        assert "alpha" not in repr(history)
        assert "beta" not in repr(history)

    def test_union_of_prior_ranges_can_fully_cover_a_later_read(
        self,
        tmp_path: Path,
    ) -> None:
        target = tmp_path / "example.py"
        target.write_text("one\ntwo\nthree\nfour\n", encoding="utf-8")
        history = ReadFileResultHistory()

        history.observe(
            call_id="upper",
            render_id="r1",
            raw_path="example.py",
            workspace_root=tmp_path,
            returned_lines=_lines("one", "two"),
            policy=SESSION_LEDGER_V1_READ_FILE_REUSE_POLICY,
        )
        history.observe(
            call_id="lower",
            render_id="r2",
            raw_path="example.py",
            workspace_root=tmp_path,
            returned_lines=_lines("three", "four", start=3),
            policy=SESSION_LEDGER_V1_READ_FILE_REUSE_POLICY,
        )
        covered = history.observe(
            call_id="combined",
            render_id="r3",
            raw_path="example.py",
            workspace_root=tmp_path,
            returned_lines=_lines("one", "two", "three", "four"),
            policy=SESSION_LEDGER_V1_READ_FILE_REUSE_POLICY,
        )

        assert covered is not None
        assert covered.kind is ReadFileReuseKind.FULL_COVERAGE
        assert covered.hint_recommended
        assert covered.prior_call_ids == ("upper", "lower")
        assert covered.overlapping_prior_call_count == 2
        assert covered.exact_repeat_count == 0

    def test_partial_overlap_is_measured_without_a_hint(
        self,
        tmp_path: Path,
    ) -> None:
        target = tmp_path / "example.py"
        target.write_text("one\ntwo\nthree\nfour\n", encoding="utf-8")
        history = ReadFileResultHistory()
        history.observe(
            call_id="first",
            render_id=None,
            raw_path="example.py",
            workspace_root=tmp_path,
            returned_lines=_lines("one", "two"),
            policy=SESSION_LEDGER_V1_READ_FILE_REUSE_POLICY,
        )

        partial = history.observe(
            call_id="second",
            render_id=None,
            raw_path="example.py",
            workspace_root=tmp_path,
            returned_lines=_lines("two", "three", "four", start=2),
            policy=SESSION_LEDGER_V1_READ_FILE_REUSE_POLICY,
        )

        assert partial is not None
        assert partial.kind is ReadFileReuseKind.PARTIAL_OVERLAP
        assert partial.matching_prior_line_count == 1
        assert partial.new_line_count == 2
        assert partial.covered_fraction == 1 / 3
        assert not partial.hint_recommended

    def test_documented_threshold_allows_near_complete_coverage_hint(
        self,
        tmp_path: Path,
    ) -> None:
        contents = tuple(f"line {index}" for index in range(1, 21))
        target = tmp_path / "example.py"
        target.write_text("\n".join(contents) + "\n", encoding="utf-8")
        history = ReadFileResultHistory()
        history.observe(
            call_id="nineteen-lines",
            render_id=None,
            raw_path="example.py",
            workspace_root=tmp_path,
            returned_lines=_lines(*contents[:19]),
            policy=SESSION_LEDGER_V1_READ_FILE_REUSE_POLICY,
        )

        near_complete = history.observe(
            call_id="twenty-lines",
            render_id=None,
            raw_path="example.py",
            workspace_root=tmp_path,
            returned_lines=_lines(*contents),
            policy=SESSION_LEDGER_V1_READ_FILE_REUSE_POLICY,
        )

        assert near_complete is not None
        assert near_complete.kind is ReadFileReuseKind.SUBSTANTIAL_COVERAGE
        assert near_complete.covered_fraction == 0.95
        assert near_complete.new_line_count == 1
        assert near_complete.hint_recommended

    def test_changed_file_advances_epoch_and_discards_prior_coverage(
        self,
        tmp_path: Path,
    ) -> None:
        target = tmp_path / "example.py"
        target.write_text("same requested line\nold elsewhere\n", encoding="utf-8")
        history = ReadFileResultHistory()
        history.observe(
            call_id="before",
            render_id=None,
            raw_path="example.py",
            workspace_root=tmp_path,
            returned_lines=_lines("same requested line"),
            policy=SESSION_LEDGER_V1_READ_FILE_REUSE_POLICY,
        )

        target.write_text("same requested line\nnew elsewhere\n", encoding="utf-8")
        after = history.observe(
            call_id="after",
            render_id=None,
            raw_path="example.py",
            workspace_root=tmp_path,
            returned_lines=_lines("same requested line"),
            policy=SESSION_LEDGER_V1_READ_FILE_REUSE_POLICY,
        )

        assert after is not None
        assert after.content_epoch == 2
        assert after.content_epoch_advanced
        assert after.kind is ReadFileReuseKind.FIRST_OBSERVATION
        assert after.matching_prior_line_count == 0
        assert not after.hint_recommended

    def test_explicit_invalidation_advances_epoch_before_identical_content(
        self,
        tmp_path: Path,
    ) -> None:
        target = tmp_path / "example.py"
        target.write_text("alpha\n", encoding="utf-8")
        history = ReadFileResultHistory()
        history.observe(
            call_id="before",
            render_id=None,
            raw_path="example.py",
            workspace_root=tmp_path,
            returned_lines=_lines("alpha"),
            policy=SESSION_LEDGER_V1_READ_FILE_REUSE_POLICY,
        )

        history.invalidate_paths(["example.py"], workspace_root=tmp_path)
        after = history.observe(
            call_id="after",
            render_id=None,
            raw_path="example.py",
            workspace_root=tmp_path,
            returned_lines=_lines("alpha"),
            policy=SESSION_LEDGER_V1_READ_FILE_REUSE_POLICY,
        )

        assert after is not None
        assert after.content_epoch == 2
        assert after.content_epoch_advanced
        assert after.kind is ReadFileReuseKind.FIRST_OBSERVATION
        assert not after.hint_recommended

    def test_live_result_mismatch_is_not_retained_or_used_for_a_hint(
        self,
        tmp_path: Path,
    ) -> None:
        target = tmp_path / "example.py"
        target.write_text("live\n", encoding="utf-8")
        history = ReadFileResultHistory()

        mismatch = history.observe(
            call_id="stale-result",
            render_id=None,
            raw_path="example.py",
            workspace_root=tmp_path,
            returned_lines=_lines("stale"),
            policy=SESSION_LEDGER_V1_READ_FILE_REUSE_POLICY,
        )
        live = history.observe(
            call_id="live-result",
            render_id=None,
            raw_path="example.py",
            workspace_root=tmp_path,
            returned_lines=_lines("live"),
            policy=SESSION_LEDGER_V1_READ_FILE_REUSE_POLICY,
        )

        assert mismatch is not None
        assert mismatch.kind is ReadFileReuseKind.LIVE_CONTENT_MISMATCH
        assert not mismatch.hint_recommended
        assert live is not None
        assert live.kind is ReadFileReuseKind.FIRST_OBSERVATION
        assert not live.hint_recommended

    def test_missing_and_unreadable_files_are_not_recorded(
        self,
        tmp_path: Path,
    ) -> None:
        history = ReadFileResultHistory()

        missing = history.observe(
            call_id="missing",
            render_id=None,
            raw_path="missing.py",
            workspace_root=tmp_path,
            returned_lines=_lines("claimed"),
            policy=SESSION_LEDGER_V1_READ_FILE_REUSE_POLICY,
        )
        binary = tmp_path / "binary.dat"
        binary.write_bytes(b"\xff\xfe")
        unreadable = history.observe(
            call_id="binary",
            render_id=None,
            raw_path="binary.dat",
            workspace_root=tmp_path,
            returned_lines=_lines("claimed"),
            policy=SESSION_LEDGER_V1_READ_FILE_REUSE_POLICY,
        )

        assert missing is None
        assert unreadable is None
        assert history.retained_file_count == 0

    def test_file_and_observation_retention_are_bounded(
        self,
        tmp_path: Path,
    ) -> None:
        history = ReadFileResultHistory(
            max_files=2,
            max_observations_per_file=2,
        )
        for file_index in range(3):
            path = tmp_path / f"file-{file_index}.txt"
            path.write_text("value\n", encoding="utf-8")
            for call_index in range(3):
                history.observe(
                    call_id=f"call-{file_index}-{call_index}",
                    render_id=None,
                    raw_path=path.name,
                    workspace_root=tmp_path,
                    returned_lines=_lines("value"),
                    policy=SESSION_LEDGER_V1_READ_FILE_REUSE_POLICY,
                )

        retained_key = file_content_key_for_path(
            "file-2.txt",
            workspace_root=tmp_path,
        )
        evicted_key = file_content_key_for_path(
            "file-0.txt",
            workspace_root=tmp_path,
        )
        assert retained_key is not None
        assert evicted_key is not None
        assert history.retained_file_count == 2
        assert history.retained_observation_count(retained_key) == 2
        assert history.retained_observation_count(evicted_key) == 0
