"""Tests for thorn.core._validation_tracker — scan-based validation tracking."""

from __future__ import annotations

import pytest

from thorn.core._validation_tracker import (
    FileSnapshot,
    ValidationStatus,
    ValidationTracker,
)


# ---------------------------------------------------------------------------
# FileSnapshot
# ---------------------------------------------------------------------------


class TestFileSnapshot:
    def test_scan_finds_matching_files(self, tmp_path):
        (tmp_path / "a.cpp").write_text("int a;")
        (tmp_path / "b.h").write_text("void b();")
        (tmp_path / "readme.md").write_text("hi")

        snap = FileSnapshot.scan(tmp_path, ["*.cpp", "*.h"])
        assert set(snap.checksums.keys()) == {"a.cpp", "b.h"}

    def test_scan_recursive_glob(self, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        (src / "main.cpp").write_text("int main(){}")
        (tmp_path / "root.txt").write_text("nope")

        snap = FileSnapshot.scan(tmp_path, ["src/**/*.cpp"])
        assert "src/main.cpp" in snap.checksums

    def test_scan_ignores_directories(self, tmp_path):
        d = tmp_path / "src"
        d.mkdir()
        snap = FileSnapshot.scan(tmp_path, ["src"])
        assert snap.checksums == {}

    def test_equal_snapshots_have_zero_diff(self, tmp_path):
        (tmp_path / "f.cpp").write_text("x")
        s1 = FileSnapshot.scan(tmp_path, ["*.cpp"])
        s2 = FileSnapshot.scan(tmp_path, ["*.cpp"])
        assert s1.diff_count(s2) == 0

    def test_modified_file_shows_diff(self, tmp_path):
        f = tmp_path / "f.cpp"
        f.write_text("v1")
        s1 = FileSnapshot.scan(tmp_path, ["*.cpp"])
        f.write_text("v2")
        s2 = FileSnapshot.scan(tmp_path, ["*.cpp"])
        assert s1.diff_count(s2) == 1

    def test_added_file_shows_diff(self, tmp_path):
        (tmp_path / "a.cpp").write_text("a")
        s1 = FileSnapshot.scan(tmp_path, ["*.cpp"])
        (tmp_path / "b.cpp").write_text("b")
        s2 = FileSnapshot.scan(tmp_path, ["*.cpp"])
        assert s1.diff_count(s2) == 1

    def test_removed_file_shows_diff(self, tmp_path):
        f = tmp_path / "a.cpp"
        f.write_text("a")
        (tmp_path / "b.cpp").write_text("b")
        s1 = FileSnapshot.scan(tmp_path, ["*.cpp"])
        f.unlink()
        s2 = FileSnapshot.scan(tmp_path, ["*.cpp"])
        assert s1.diff_count(s2) == 1


# ---------------------------------------------------------------------------
# ValidationTracker — staleness detection
# ---------------------------------------------------------------------------


class TestStaleness:
    def test_no_baseline_is_stale(self, tmp_path):
        (tmp_path / "a.cpp").write_text("code")

        tracker = ValidationTracker(root=tmp_path)
        tracker.add_target("build", ["*.cpp"])
        tracker.refresh()

        target = tracker.targets["build"]
        assert target.stale_file_count == 1
        assert tracker.effective_status("build") == ValidationStatus.STALE

    def test_record_result_then_unchanged_is_not_stale(self, tmp_path):
        (tmp_path / "a.cpp").write_text("code")

        tracker = ValidationTracker(root=tmp_path)
        tracker.add_target("build", ["*.cpp"])
        tracker.record_result("build", passed=True)
        tracker.refresh()

        target = tracker.targets["build"]
        assert target.stale_file_count == 0
        assert tracker.effective_status("build") == ValidationStatus.PASSING

    def test_modify_file_makes_stale(self, tmp_path):
        f = tmp_path / "a.cpp"
        f.write_text("v1")

        tracker = ValidationTracker(root=tmp_path)
        tracker.add_target("build", ["*.cpp"])
        tracker.record_result("build", passed=True)

        f.write_text("v2")
        tracker.refresh()

        assert tracker.targets["build"].stale_file_count == 1
        assert tracker.effective_status("build") == ValidationStatus.STALE

    def test_add_matching_file_makes_stale(self, tmp_path):
        (tmp_path / "a.cpp").write_text("a")

        tracker = ValidationTracker(root=tmp_path)
        tracker.add_target("build", ["*.cpp"])
        tracker.record_result("build", passed=True)

        (tmp_path / "b.cpp").write_text("b")
        tracker.refresh()

        assert tracker.targets["build"].stale_file_count == 1
        assert tracker.effective_status("build") == ValidationStatus.STALE

    def test_delete_baseline_file_makes_stale(self, tmp_path):
        f = tmp_path / "a.cpp"
        f.write_text("a")

        tracker = ValidationTracker(root=tmp_path)
        tracker.add_target("build", ["*.cpp"])
        tracker.record_result("build", passed=True)

        f.unlink()
        tracker.refresh()

        assert tracker.targets["build"].stale_file_count == 1
        assert tracker.effective_status("build") == ValidationStatus.STALE

    def test_non_matching_file_does_not_affect_staleness(self, tmp_path):
        (tmp_path / "a.cpp").write_text("code")

        tracker = ValidationTracker(root=tmp_path)
        tracker.add_target("build", ["*.cpp"])
        tracker.record_result("build", passed=True)

        (tmp_path / "readme.md").write_text("docs")
        tracker.refresh()

        assert tracker.targets["build"].stale_file_count == 0
        assert tracker.effective_status("build") == ValidationStatus.PASSING


# ---------------------------------------------------------------------------
# ValidationTracker — depends_on
# ---------------------------------------------------------------------------


class TestDependencies:
    def test_stale_dep_makes_dependent_stale(self, tmp_path):
        (tmp_path / "a.cpp").write_text("code")

        tracker = ValidationTracker(root=tmp_path)
        tracker.add_target("build", ["*.cpp"])
        tracker.add_target("test", ["*.cpp"], depends_on=["build"])
        tracker.record_result("test", passed=True)
        tracker.refresh()

        assert tracker.effective_status("test") == ValidationStatus.STALE

    def test_failing_dep_makes_dependent_stale(self, tmp_path):
        (tmp_path / "a.cpp").write_text("code")

        tracker = ValidationTracker(root=tmp_path)
        tracker.add_target("build", ["*.cpp"])
        tracker.add_target("test", ["*.cpp"], depends_on=["build"])
        tracker.record_result("build", passed=False, summary="3 errors")
        tracker.record_result("test", passed=True)
        tracker.refresh()

        assert tracker.effective_status("test") == ValidationStatus.STALE

    def test_passing_dep_does_not_block(self, tmp_path):
        (tmp_path / "a.cpp").write_text("code")

        tracker = ValidationTracker(root=tmp_path)
        tracker.add_target("build", ["*.cpp"])
        tracker.add_target("test", ["*.cpp"], depends_on=["build"])
        tracker.record_result("build", passed=True)
        tracker.record_result("test", passed=True)
        tracker.refresh()

        assert tracker.effective_status("build") == ValidationStatus.PASSING
        assert tracker.effective_status("test") == ValidationStatus.PASSING

    def test_cycle_does_not_infinite_loop(self, tmp_path):
        tracker = ValidationTracker(root=tmp_path)
        tracker.add_target("a", ["*.cpp"], depends_on=["b"])
        tracker.add_target("b", ["*.cpp"], depends_on=["a"])
        tracker.refresh()

        status = tracker.effective_status("a")
        assert status == ValidationStatus.STALE

    def test_unknown_target_returns_unknown(self, tmp_path):
        tracker = ValidationTracker(root=tmp_path)
        assert tracker.effective_status("nonexistent") == ValidationStatus.UNKNOWN


# ---------------------------------------------------------------------------
# ValidationTracker — record_result
# ---------------------------------------------------------------------------


class TestRecordResult:
    def test_records_passing(self, tmp_path):
        (tmp_path / "a.cpp").write_text("code")

        tracker = ValidationTracker(root=tmp_path)
        tracker.add_target("build", ["*.cpp"])
        tracker.record_result("build", passed=True)

        target = tracker.targets["build"]
        assert target.last_result == ValidationStatus.PASSING
        assert target.last_summary is None
        assert target.baseline is not None

    def test_records_failing_with_summary(self, tmp_path):
        (tmp_path / "a.cpp").write_text("code")

        tracker = ValidationTracker(root=tmp_path)
        tracker.add_target("build", ["*.cpp"])
        tracker.record_result("build", passed=False, summary="3 errors")

        target = tracker.targets["build"]
        assert target.last_result == ValidationStatus.FAILING
        assert target.last_summary == "3 errors"

    def test_record_unknown_target_is_noop(self, tmp_path):
        tracker = ValidationTracker(root=tmp_path)
        tracker.record_result("nonexistent", passed=True)


# ---------------------------------------------------------------------------
# ValidationTracker — render_status
# ---------------------------------------------------------------------------


class TestRenderStatus:
    def test_no_targets_returns_none(self, tmp_path):
        tracker = ValidationTracker(root=tmp_path)
        assert tracker.render_status() is None

    def test_all_unknown_no_stale_returns_none(self, tmp_path):
        tracker = ValidationTracker(root=tmp_path)
        tracker.add_target("build", ["*.nonexistent"])
        tracker.refresh()
        assert tracker.render_status() is None

    def test_all_passing(self, tmp_path):
        (tmp_path / "a.cpp").write_text("code")

        tracker = ValidationTracker(root=tmp_path)
        tracker.add_target("build", ["*.cpp"])
        tracker.add_target("test", ["*.cpp"], depends_on=["build"])
        tracker.record_result("build", passed=True)
        tracker.record_result("test", passed=True)
        tracker.refresh()

        assert tracker.render_status() == "[all validations passing]"

    def test_stale_from_file_changes(self, tmp_path):
        f = tmp_path / "a.cpp"
        f.write_text("v1")

        tracker = ValidationTracker(root=tmp_path)
        tracker.add_target("build", ["*.cpp"])
        tracker.record_result("build", passed=True)
        f.write_text("v2")
        tracker.refresh()

        status = tracker.render_status()
        assert status is not None
        assert "build: stale (1 files changed)" in status

    def test_stale_multiple_files(self, tmp_path):
        f1 = tmp_path / "a.cpp"
        f2 = tmp_path / "b.cpp"
        f1.write_text("v1")
        f2.write_text("v1")

        tracker = ValidationTracker(root=tmp_path)
        tracker.add_target("build", ["*.cpp"])
        tracker.record_result("build", passed=True)
        f1.write_text("v2")
        f2.write_text("v2")
        tracker.refresh()

        status = tracker.render_status()
        assert status is not None
        assert "build: stale (2 files changed)" in status

    def test_failing_shows_summary(self, tmp_path):
        (tmp_path / "a.cpp").write_text("code")

        tracker = ValidationTracker(root=tmp_path)
        tracker.add_target("build", ["*.cpp"])
        tracker.record_result("build", passed=False, summary="3 errors")
        tracker.refresh()

        status = tracker.render_status()
        assert status is not None
        assert "build: 3 errors" in status

    def test_failing_no_summary(self, tmp_path):
        (tmp_path / "a.cpp").write_text("code")

        tracker = ValidationTracker(root=tmp_path)
        tracker.add_target("build", ["*.cpp"])
        tracker.record_result("build", passed=False)
        tracker.refresh()

        status = tracker.render_status()
        assert status is not None
        assert "build: failing" in status

    def test_blocked_by_failing_dep(self, tmp_path):
        (tmp_path / "a.cpp").write_text("code")

        tracker = ValidationTracker(root=tmp_path)
        tracker.add_target("build", ["*.cpp"])
        tracker.add_target("test", ["*.cpp"], depends_on=["build"])
        tracker.record_result("build", passed=False, summary="3 errors")
        tracker.record_result("test", passed=True)
        tracker.refresh()

        status = tracker.render_status()
        assert status is not None
        assert "build: 3 errors" in status
        assert "test: blocked (build failing)" in status

    def test_blocked_by_stale_dep(self, tmp_path):
        f = tmp_path / "a.cpp"
        f.write_text("v1")

        tracker = ValidationTracker(root=tmp_path)
        tracker.add_target("build", ["*.cpp"])
        tracker.add_target("test", ["*.cpp"], depends_on=["build"])
        tracker.record_result("build", passed=True)
        tracker.record_result("test", passed=True)
        f.write_text("v2")
        tracker.refresh()

        status = tracker.render_status()
        assert status is not None
        assert "build: stale" in status
        assert "test: blocked (build stale)" in status

    def test_mixed_passing_and_stale(self, tmp_path):
        f = tmp_path / "a.cpp"
        f.write_text("v1")
        (tmp_path / "t.test").write_text("test")

        tracker = ValidationTracker(root=tmp_path)
        tracker.add_target("build", ["*.cpp"])
        tracker.add_target("lint", ["*.test"])
        tracker.record_result("build", passed=True)
        tracker.record_result("lint", passed=True)

        f.write_text("v2")
        tracker.refresh()

        status = tracker.render_status()
        assert status is not None
        assert "build: stale" in status
        assert "lint: passing" in status
