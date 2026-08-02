"""Black-box tests for bounded task-workspace content identities."""

from __future__ import annotations

import subprocess
from pathlib import Path, PurePosixPath

import pytest

import thorn.core._workspace_content as workspace_content
from thorn.core._workspace_content import (
    WorkspaceContentCollectionFailure,
    WorkspaceContentExcludedPath,
    WorkspaceContentExclusionReason,
    WorkspaceContentSnapshotLimits,
    collect_workspace_content_snapshot,
)


def _initialize_repository(path: Path) -> None:
    subprocess.run(["git", "init", "--quiet"], cwd=path, check=True)
    subprocess.run(
        ["git", "config", "user.email", "workspace-test@example.invalid"],
        cwd=path,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Workspace Test"],
        cwd=path,
        check=True,
    )
    (path / ".gitignore").write_text(".pytest_cache/\n", encoding="utf-8")
    (path / "tracked.py").write_text("value = 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=path, check=True)
    subprocess.run(
        ["git", "commit", "--quiet", "-m", "baseline"],
        cwd=path,
        check=True,
    )


def test_snapshot_tracks_tracked_and_nonignored_untracked_content(
    tmp_path: Path,
) -> None:
    _initialize_repository(tmp_path)
    baseline = collect_workspace_content_snapshot(tmp_path)

    (tmp_path / "tracked.py").write_text("value = 2\n", encoding="utf-8")
    tracked_edit = collect_workspace_content_snapshot(tmp_path)
    (tmp_path / "tracked.py").write_text("value = 1\n", encoding="utf-8")
    (tmp_path / "new.py").write_text("new = True\n", encoding="utf-8")
    untracked_create = collect_workspace_content_snapshot(tmp_path)
    (tmp_path / "renamed.py").write_bytes((tmp_path / "new.py").read_bytes())
    (tmp_path / "new.py").unlink()
    untracked_rename = collect_workspace_content_snapshot(tmp_path)
    (tmp_path / "renamed.py").unlink()
    restored = collect_workspace_content_snapshot(tmp_path)

    assert baseline.is_known
    assert tracked_edit.is_known
    assert untracked_create.is_known
    assert untracked_rename.is_known
    assert restored.is_known
    assert len(
        {
            baseline.identity,
            tracked_edit.identity,
            untracked_create.identity,
            untracked_rename.identity,
        }
    ) == 4
    assert restored.identity == baseline.identity


def test_snapshot_ignores_git_ignored_and_exact_framework_artifacts(
    tmp_path: Path,
) -> None:
    _initialize_repository(tmp_path)
    baseline = collect_workspace_content_snapshot(tmp_path)

    cache = tmp_path / ".pytest_cache/state"
    cache.parent.mkdir()
    cache.write_text("validator output\n", encoding="utf-8")
    framework_log = tmp_path / "agents/local/control/toolhost.log"
    framework_log.parent.mkdir(parents=True)
    framework_log.write_text("framework output\n", encoding="utf-8")

    without_provenance = collect_workspace_content_snapshot(tmp_path)
    after_artifacts = collect_workspace_content_snapshot(
        tmp_path,
        excluded_paths=(
            WorkspaceContentExcludedPath(
                relative_path=PurePosixPath("agents/local/control/toolhost.log"),
                reason=(
                    WorkspaceContentExclusionReason.FRAMEWORK_TOOLHOST_LOG
                ),
            ),
        ),
    )

    assert without_provenance.identity != baseline.identity
    assert after_artifacts.identity == baseline.identity


def test_snapshot_fails_closed_for_non_git_and_resource_limits(
    tmp_path: Path,
) -> None:
    unavailable = collect_workspace_content_snapshot(tmp_path / "missing")
    assert unavailable.failure is (
        WorkspaceContentCollectionFailure.WORKSPACE_UNAVAILABLE
    )

    non_git = collect_workspace_content_snapshot(tmp_path)
    assert non_git.failure is WorkspaceContentCollectionFailure.NOT_GIT_ROOT
    assert not non_git.is_known

    _initialize_repository(tmp_path)
    path_limited = collect_workspace_content_snapshot(
        tmp_path,
        limits=WorkspaceContentSnapshotLimits(maximum_path_count_per_pass=1),
    )
    listing_limited = collect_workspace_content_snapshot(
        tmp_path,
        limits=WorkspaceContentSnapshotLimits(
            maximum_listing_bytes_per_pass=1,
        ),
    )
    byte_limited = collect_workspace_content_snapshot(
        tmp_path,
        limits=WorkspaceContentSnapshotLimits(
            maximum_content_bytes_per_pass=1,
        ),
    )

    assert path_limited.failure is WorkspaceContentCollectionFailure.PATH_LIMIT
    assert listing_limited.failure is (
        WorkspaceContentCollectionFailure.LISTING_BYTE_LIMIT
    )
    assert byte_limited.failure is WorkspaceContentCollectionFailure.BYTE_LIMIT
    assert not path_limited.is_known
    assert not listing_limited.is_known
    assert not byte_limited.is_known


def test_snapshot_hashes_symlink_target_without_reading_outside_workspace(
    tmp_path: Path,
) -> None:
    _initialize_repository(tmp_path)
    external_file = tmp_path.parent / f"{tmp_path.name}-external-content.txt"
    external_file.write_text("first secret\n", encoding="utf-8")
    (tmp_path / "link.py").symlink_to(external_file)
    baseline = collect_workspace_content_snapshot(tmp_path)

    external_file.write_text("different secret\n", encoding="utf-8")
    changed_external_content = collect_workspace_content_snapshot(tmp_path)

    assert baseline.is_known
    assert changed_external_content.identity == baseline.identity


def test_snapshot_reports_change_between_identity_passes_as_unknown(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _initialize_repository(tmp_path)
    original_collect_pass = workspace_content._collect_pass
    pass_count = 0

    def collect_pass(root, limits, excluded_paths, deadline):
        nonlocal pass_count
        pass_count += 1
        result = original_collect_pass(root, limits, excluded_paths, deadline)
        if pass_count == 1:
            (tmp_path / "tracked.py").write_text("value = 2\n", encoding="utf-8")
        return result

    monkeypatch.setattr(workspace_content, "_collect_pass", collect_pass)

    snapshot = collect_workspace_content_snapshot(tmp_path)

    assert snapshot.failure is WorkspaceContentCollectionFailure.CONCURRENT_CHANGE
    assert not snapshot.is_known
