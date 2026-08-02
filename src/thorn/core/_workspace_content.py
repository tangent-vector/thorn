"""Bounded, privacy-safe identities for task-workspace content.

The validation convergence policy needs to know whether opaque shell actions
actually changed task inputs.  It must not infer that from shell syntax.  This
module instead hashes the tracked and non-ignored untracked content of one Git
workspace twice.  A disagreement, unsupported workspace, or exhausted bound is
reported as unknown so convergence remains conservative.
"""

from __future__ import annotations

import contextlib
import hashlib
import os
import selectors
import stat
import subprocess
import time
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import Protocol


class _Digest(Protocol):
    def update(self, content: bytes) -> None: ...


class WorkspaceContentCollectionFailure(StrEnum):
    """Why a workspace identity could not be established safely."""

    WORKSPACE_UNAVAILABLE = "workspace_unavailable"
    NOT_GIT_ROOT = "not_git_root"
    GIT_ERROR = "git_error"
    PATH_LIMIT = "path_limit"
    LISTING_BYTE_LIMIT = "listing_byte_limit"
    BYTE_LIMIT = "byte_limit"
    TIME_LIMIT = "time_limit"
    UNSUPPORTED_ENTRY = "unsupported_entry"
    IO_ERROR = "io_error"
    CONCURRENT_CHANGE = "concurrent_change"


class WorkspaceContentExclusionReason(StrEnum):
    """Trusted owner of one exact path omitted from task content."""

    FRAMEWORK_TOOLHOST_LOG = "framework_toolhost_log"


@dataclass(frozen=True)
class WorkspaceContentExcludedPath:
    """One provenance-established framework path, never a glob."""

    relative_path: PurePosixPath
    reason: WorkspaceContentExclusionReason

    def __post_init__(self) -> None:
        if self.relative_path.is_absolute() or ".." in self.relative_path.parts:
            raise ValueError("workspace content exclusion must stay below its root")


@dataclass(frozen=True, order=True)
class WorkspaceContentIdentity:
    """SHA-256 identity that never exposes workspace paths or file content."""

    value: str

    def __post_init__(self) -> None:
        if len(self.value) != 64 or any(
            character not in "0123456789abcdef" for character in self.value
        ):
            raise ValueError("workspace content identity must be lowercase SHA-256")


@dataclass(frozen=True)
class WorkspaceContentSnapshotLimits:
    """Fixed work limits for one double-sampled workspace identity."""

    maximum_path_count_per_pass: int = 20_000
    maximum_listing_bytes_per_pass: int = 16 * 1024 * 1024
    maximum_content_bytes_per_pass: int = 128 * 1024 * 1024
    maximum_elapsed_seconds: float = 3.0

    def __post_init__(self) -> None:
        if self.maximum_path_count_per_pass <= 0:
            raise ValueError("maximum_path_count_per_pass must be positive")
        if self.maximum_listing_bytes_per_pass <= 0:
            raise ValueError("maximum_listing_bytes_per_pass must be positive")
        if self.maximum_content_bytes_per_pass <= 0:
            raise ValueError("maximum_content_bytes_per_pass must be positive")
        if self.maximum_elapsed_seconds <= 0:
            raise ValueError("maximum_elapsed_seconds must be positive")


DEFAULT_WORKSPACE_CONTENT_SNAPSHOT_LIMITS = WorkspaceContentSnapshotLimits()


@dataclass(frozen=True)
class WorkspaceContentSnapshot:
    """Known content identity or a typed conservative collection failure."""

    identity: WorkspaceContentIdentity | None
    path_count: int | None
    content_bytes: int | None
    failure: WorkspaceContentCollectionFailure | None

    def __post_init__(self) -> None:
        if self.identity is None:
            if (
                self.failure is None
                or self.path_count is not None
                or self.content_bytes is not None
            ):
                raise ValueError("unknown workspace content requires one failure")
            return
        if self.failure is not None:
            raise ValueError("known workspace content cannot have a failure")
        if self.path_count is None or self.path_count < 0:
            raise ValueError("known workspace content requires a path count")
        if self.content_bytes is None or self.content_bytes < 0:
            raise ValueError("known workspace content requires a byte count")

    @classmethod
    def known(
        cls,
        *,
        identity: WorkspaceContentIdentity,
        path_count: int,
        content_bytes: int,
    ) -> WorkspaceContentSnapshot:
        return cls(
            identity=identity,
            path_count=path_count,
            content_bytes=content_bytes,
            failure=None,
        )

    @classmethod
    def unknown(
        cls,
        failure: WorkspaceContentCollectionFailure,
    ) -> WorkspaceContentSnapshot:
        return cls(
            identity=None,
            path_count=None,
            content_bytes=None,
            failure=failure,
        )

    @property
    def is_known(self) -> bool:
        return self.identity is not None


@dataclass(frozen=True)
class _WorkspaceContentPass:
    identity: WorkspaceContentIdentity
    path_count: int
    content_bytes: int


class _CollectionFailure(Exception):
    def __init__(self, reason: WorkspaceContentCollectionFailure) -> None:
        super().__init__(reason.value)
        self.reason = reason


def collect_workspace_content_snapshot(
    workspace_root: Path,
    *,
    limits: WorkspaceContentSnapshotLimits = (
        DEFAULT_WORKSPACE_CONTENT_SNAPSHOT_LIMITS
    ),
    excluded_paths: tuple[WorkspaceContentExcludedPath, ...] = (),
) -> WorkspaceContentSnapshot:
    """Return a stable task-content identity or a conservative unknown state."""

    try:
        root = workspace_root.resolve()
        if not root.is_dir():
            return WorkspaceContentSnapshot.unknown(
                WorkspaceContentCollectionFailure.WORKSPACE_UNAVAILABLE,
            )
    except OSError:
        return WorkspaceContentSnapshot.unknown(
            WorkspaceContentCollectionFailure.WORKSPACE_UNAVAILABLE,
        )
    deadline = time.monotonic() + limits.maximum_elapsed_seconds
    try:
        _require_exact_git_root(root, deadline)
        first = _collect_pass(root, limits, excluded_paths, deadline)
        second = _collect_pass(root, limits, excluded_paths, deadline)
    except _CollectionFailure as failure:
        return WorkspaceContentSnapshot.unknown(failure.reason)

    if first != second:
        return WorkspaceContentSnapshot.unknown(
            WorkspaceContentCollectionFailure.CONCURRENT_CHANGE,
        )
    return WorkspaceContentSnapshot.known(
        identity=first.identity,
        path_count=first.path_count,
        content_bytes=first.content_bytes,
    )


def _require_exact_git_root(root: Path, deadline: float) -> None:
    completed = _run_git(
        root,
        ("rev-parse", "--show-toplevel", "--path-format=absolute"),
        deadline,
    )
    if completed.returncode != 0:
        raise _CollectionFailure(
            WorkspaceContentCollectionFailure.NOT_GIT_ROOT,
        )
    try:
        git_root = Path(os.fsdecode(completed.stdout.rstrip(b"\n"))).resolve()
    except (OSError, ValueError) as error:
        raise _CollectionFailure(
            WorkspaceContentCollectionFailure.NOT_GIT_ROOT,
        ) from error
    if git_root != root:
        raise _CollectionFailure(WorkspaceContentCollectionFailure.NOT_GIT_ROOT)


def _collect_pass(
    root: Path,
    limits: WorkspaceContentSnapshotLimits,
    excluded_paths: tuple[WorkspaceContentExcludedPath, ...],
    deadline: float,
) -> _WorkspaceContentPass:
    raw_paths = _list_git_paths(root, limits, deadline)
    excluded_relative_paths = frozenset(
        exclusion.relative_path for exclusion in excluded_paths
    )
    included_paths = [
        raw_path
        for raw_path in raw_paths
        if PurePosixPath(os.fsdecode(raw_path)) not in excluded_relative_paths
    ]
    if len(included_paths) > limits.maximum_path_count_per_pass:
        raise _CollectionFailure(WorkspaceContentCollectionFailure.PATH_LIMIT)

    digest = hashlib.sha256()
    content_bytes = 0
    for raw_path in included_paths:
        _require_time_remaining(deadline)
        relative_path = PurePosixPath(os.fsdecode(raw_path))
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise _CollectionFailure(
                WorkspaceContentCollectionFailure.UNSUPPORTED_ENTRY,
            )
        encoded_path = os.fsencode(str(relative_path))
        digest.update(len(encoded_path).to_bytes(8, "big"))
        digest.update(encoded_path)
        content_bytes += _hash_path(
            digest,
            root / Path(relative_path),
            maximum_content_bytes=limits.maximum_content_bytes_per_pass,
            prior_content_bytes=content_bytes,
            deadline=deadline,
        )

    return _WorkspaceContentPass(
        identity=WorkspaceContentIdentity(digest.hexdigest()),
        path_count=len(included_paths),
        content_bytes=content_bytes,
    )


def _list_git_paths(
    root: Path,
    limits: WorkspaceContentSnapshotLimits,
    deadline: float,
) -> list[bytes]:
    """List task paths without allowing Git output to grow memory unbounded."""
    try:
        process = subprocess.Popen(
            (
                "git",
                "-C",
                os.fspath(root),
                "ls-files",
                "--cached",
                "--others",
                "--exclude-standard",
                "-z",
                "--",
            ),
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except OSError as error:
        raise _CollectionFailure(WorkspaceContentCollectionFailure.GIT_ERROR) from error

    stdout = process.stdout
    if stdout is None:
        _terminate_process(process)
        raise _CollectionFailure(WorkspaceContentCollectionFailure.GIT_ERROR)

    listing = bytearray()
    path_count = 0
    selector = selectors.DefaultSelector()
    try:
        os.set_blocking(stdout.fileno(), False)
        selector.register(stdout, selectors.EVENT_READ)
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not selector.select(remaining):
                raise _CollectionFailure(
                    WorkspaceContentCollectionFailure.TIME_LIMIT,
                )
            try:
                chunk = os.read(stdout.fileno(), 64 * 1024)
            except BlockingIOError:
                continue
            if not chunk:
                break
            listing.extend(chunk)
            if len(listing) > limits.maximum_listing_bytes_per_pass:
                raise _CollectionFailure(
                    WorkspaceContentCollectionFailure.LISTING_BYTE_LIMIT,
                )
            path_count += chunk.count(b"\0")
            if path_count > limits.maximum_path_count_per_pass:
                raise _CollectionFailure(
                    WorkspaceContentCollectionFailure.PATH_LIMIT,
                )

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise _CollectionFailure(WorkspaceContentCollectionFailure.TIME_LIMIT)
        try:
            return_code = process.wait(timeout=remaining)
        except subprocess.TimeoutExpired as error:
            raise _CollectionFailure(
                WorkspaceContentCollectionFailure.TIME_LIMIT,
            ) from error
        if return_code != 0:
            raise _CollectionFailure(WorkspaceContentCollectionFailure.GIT_ERROR)
    except _CollectionFailure:
        raise
    except OSError as error:
        raise _CollectionFailure(WorkspaceContentCollectionFailure.IO_ERROR) from error
    finally:
        selector.close()
        with contextlib.suppress(OSError):
            stdout.close()
        _terminate_process(process)

    return [path for path in bytes(listing).split(b"\0") if path]


def _terminate_process(process: subprocess.Popen[bytes]) -> None:
    with contextlib.suppress(OSError):
        if process.poll() is not None:
            return
        process.kill()
    with contextlib.suppress(OSError, subprocess.TimeoutExpired):
        process.wait(timeout=1.0)


def _hash_path(
    digest: _Digest,
    path: Path,
    *,
    maximum_content_bytes: int,
    prior_content_bytes: int,
    deadline: float,
) -> int:
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        digest.update(b"missing\0")
        digest.update((0).to_bytes(8, "big"))
        return 0
    except OSError as error:
        raise _CollectionFailure(WorkspaceContentCollectionFailure.IO_ERROR) from error

    executable_bits = stat.S_IMODE(metadata.st_mode) & 0o111
    digest.update(executable_bits.to_bytes(2, "big"))
    if stat.S_ISLNK(metadata.st_mode):
        try:
            content = os.fsencode(os.readlink(path))
        except OSError as error:
            raise _CollectionFailure(
                WorkspaceContentCollectionFailure.IO_ERROR,
            ) from error
        _require_byte_capacity(
            prior_content_bytes,
            len(content),
            maximum_content_bytes,
        )
        digest.update(b"symlink\0")
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
        return len(content)
    if not stat.S_ISREG(metadata.st_mode):
        raise _CollectionFailure(
            WorkspaceContentCollectionFailure.UNSUPPORTED_ENTRY,
        )

    _require_byte_capacity(
        prior_content_bytes,
        metadata.st_size,
        maximum_content_bytes,
    )
    digest.update(b"file\0")
    digest.update(metadata.st_size.to_bytes(8, "big"))
    descriptor_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    descriptor_flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, descriptor_flags)
        with os.fdopen(descriptor, "rb") as stream:
            opened_metadata = os.fstat(stream.fileno())
            if not stat.S_ISREG(opened_metadata.st_mode):
                raise _CollectionFailure(
                    WorkspaceContentCollectionFailure.UNSUPPORTED_ENTRY,
                )
            bytes_read = 0
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                _require_time_remaining(deadline)
                bytes_read += len(chunk)
                _require_byte_capacity(
                    prior_content_bytes,
                    bytes_read,
                    maximum_content_bytes,
                )
                digest.update(chunk)
            final_metadata = os.fstat(stream.fileno())
    except _CollectionFailure:
        raise
    except OSError as error:
        raise _CollectionFailure(WorkspaceContentCollectionFailure.IO_ERROR) from error
    if (
        opened_metadata.st_dev,
        opened_metadata.st_ino,
        opened_metadata.st_size,
        opened_metadata.st_mtime_ns,
    ) != (
        final_metadata.st_dev,
        final_metadata.st_ino,
        final_metadata.st_size,
        final_metadata.st_mtime_ns,
    ):
        raise _CollectionFailure(
            WorkspaceContentCollectionFailure.CONCURRENT_CHANGE,
        )
    return bytes_read


def _run_git(
    root: Path,
    arguments: tuple[str, ...],
    deadline: float,
) -> subprocess.CompletedProcess[bytes]:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise _CollectionFailure(WorkspaceContentCollectionFailure.TIME_LIMIT)
    try:
        completed = subprocess.run(
            ("git", "-C", os.fspath(root), *arguments),
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=remaining,
        )
    except subprocess.TimeoutExpired as error:
        raise _CollectionFailure(
            WorkspaceContentCollectionFailure.TIME_LIMIT,
        ) from error
    except OSError as error:
        raise _CollectionFailure(WorkspaceContentCollectionFailure.GIT_ERROR) from error
    return completed


def _require_byte_capacity(
    prior_content_bytes: int,
    added_content_bytes: int,
    maximum_content_bytes: int,
) -> None:
    if prior_content_bytes + added_content_bytes > maximum_content_bytes:
        raise _CollectionFailure(WorkspaceContentCollectionFailure.BYTE_LIMIT)


def _require_time_remaining(deadline: float) -> None:
    if time.monotonic() >= deadline:
        raise _CollectionFailure(WorkspaceContentCollectionFailure.TIME_LIMIT)


__all__ = [
    "DEFAULT_WORKSPACE_CONTENT_SNAPSHOT_LIMITS",
    "WorkspaceContentCollectionFailure",
    "WorkspaceContentExcludedPath",
    "WorkspaceContentExclusionReason",
    "WorkspaceContentIdentity",
    "WorkspaceContentSnapshot",
    "WorkspaceContentSnapshotLimits",
    "collect_workspace_content_snapshot",
]
