"""Bounded session memory for successful ``read_file`` observations.

The provider-history renderer decides what old observations remain visible to
the model.  This module deliberately has a different lifetime: it remembers
hashed facts for one Thorn session so that dropping an old tool result does not
make an unchanged read look novel again.

Only canonical paths, file-version hashes, line hashes, ranges, and tool-call
identifiers are retained.  File text is read transiently to establish a
content version and to ensure that the returned tool payload still agrees with
the live file, but raw content is never stored in the ledger.
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Iterable, NewType

from thorn.core._prompt_visibility import (
    FileContentKey,
    FileLineContent,
    file_content_key_for_path,
)

DEFAULT_READ_FILE_REUSE_HINT_THRESHOLD = 0.95
"""Covered-line fraction that qualifies as a near-complete repeat."""

READ_FILE_REUSE_TELEMETRY_SCHEMA_VERSION = 2
"""Schema for arm-independent read observation telemetry."""

ReadFileContentEpoch = NewType("ReadFileContentEpoch", int)
"""Monotonic content generation for one canonical file in a session."""

FileContentVersionHash = NewType("FileContentVersionHash", str)
"""SHA-256 identity of a live file's complete byte content."""

FileLineContentHash = NewType("FileLineContentHash", str)
"""SHA-256 identity of one decoded source line."""


@dataclass(frozen=True)
class ReadFileReusePolicy:
    """Policy for classifying reads observed by session-local memory."""

    minimum_covered_fraction: float = DEFAULT_READ_FILE_REUSE_HINT_THRESHOLD

    def __post_init__(self) -> None:
        if not 0.0 < self.minimum_covered_fraction <= 1.0:
            raise ValueError("minimum_covered_fraction must be in (0.0, 1.0]")


SESSION_LEDGER_V1_READ_FILE_REUSE_POLICY = ReadFileReusePolicy()
"""Frozen observation and advisory policy for controlled CLI evaluations."""


class ReadFileReuseKind(StrEnum):
    """Relationship between a successful read and its current-epoch history."""

    NO_RETURNED_LINES = "no_returned_lines"
    FIRST_OBSERVATION = "first_observation"
    NO_OVERLAP = "no_overlap"
    PARTIAL_OVERLAP = "partial_overlap"
    SUBSTANTIAL_COVERAGE = "substantial_coverage"
    FULL_COVERAGE = "full_coverage"
    EXACT_REPEAT = "exact_repeat"
    LIVE_CONTENT_MISMATCH = "live_content_mismatch"


@dataclass(frozen=True)
class ReadFileResultObservation:
    """One bounded, content-free record of a successful file read."""

    call_id: str
    render_id: str | None
    line_hashes: tuple[tuple[int, FileLineContentHash], ...]


@dataclass(frozen=True)
class ReadFileReuseObservation:
    """Structured comparison facts for trajectory telemetry and hint policy."""

    call_id: str
    render_id: str | None
    file_key: FileContentKey
    content_epoch: ReadFileContentEpoch
    content_version_hash: FileContentVersionHash
    returned_start_line: int | None
    returned_end_line: int | None
    returned_line_count: int
    matching_prior_line_count: int
    new_line_count: int
    overlapping_prior_call_count: int
    exact_repeat_count: int
    prior_call_id: str | None
    prior_render_id: str | None
    prior_call_ids: tuple[str, ...]
    kind: ReadFileReuseKind
    hint_recommended: bool
    content_epoch_advanced: bool

    @property
    def covered_fraction(self) -> float:
        """Fraction of returned lines already observed in this content epoch."""
        if self.returned_line_count == 0:
            return 0.0
        return self.matching_prior_line_count / self.returned_line_count


@dataclass
class _FileReadHistory:
    content_epoch: ReadFileContentEpoch = ReadFileContentEpoch(0)
    content_version_hash: FileContentVersionHash | None = None
    epoch_advanced_since_observation: bool = False
    observations: list[ReadFileResultObservation] = field(default_factory=list)


@dataclass
class ReadFileResultHistory:
    """LRU-bounded per-session index of hashed ``read_file`` results."""

    max_files: int = 128
    max_observations_per_file: int = 16
    max_lines_per_observation: int = 500
    _history_by_file: OrderedDict[FileContentKey, _FileReadHistory] = field(
        default_factory=OrderedDict,
    )

    def __post_init__(self) -> None:
        if self.max_files < 1:
            raise ValueError("max_files must be >= 1")
        if self.max_observations_per_file < 1:
            raise ValueError("max_observations_per_file must be >= 1")
        if self.max_lines_per_observation < 1:
            raise ValueError("max_lines_per_observation must be >= 1")

    def observe(
        self,
        *,
        call_id: str,
        render_id: str | None,
        raw_path: str,
        workspace_root: Path | None,
        returned_lines: Iterable[FileLineContent],
        policy: ReadFileReusePolicy,
    ) -> ReadFileReuseObservation | None:
        """Record a successful read and compare it with the current epoch.

        The real tool call has already completed.  Returning ``None`` means
        the result could not be compared conservatively (for example because
        the live file was missing, unreadable, or the result exceeded the
        native read bound).  No read is ever suppressed by this ledger.
        """
        candidate_lines = tuple(returned_lines)
        if len(candidate_lines) > self.max_lines_per_observation:
            return None
        if len({line.line_number for line in candidate_lines}) != len(
            candidate_lines
        ):
            return None

        file_key = file_content_key_for_path(
            raw_path,
            workspace_root=workspace_root,
        )
        if file_key is None:
            return None

        live_snapshot = _read_live_file_snapshot(file_key.resolved_path)
        if live_snapshot is None:
            return None
        content_version_hash, live_line_hashes = live_snapshot

        file_history = self._history_by_file.get(file_key)
        if file_history is None:
            file_history = _FileReadHistory()
            self._history_by_file[file_key] = file_history
        else:
            self._history_by_file.move_to_end(file_key)

        epoch_advanced = self._advance_epoch_for_version(
            file_history,
            content_version_hash,
        )
        if not candidate_lines:
            observation = self._comparison_for_no_returned_lines(
                call_id=call_id,
                render_id=render_id,
                file_key=file_key,
                file_history=file_history,
                content_version_hash=content_version_hash,
                epoch_advanced=epoch_advanced,
            )
            self._evict_files_over_limit()
            return observation

        current_line_hashes = tuple(
            (line.line_number, _line_hash(line.content))
            for line in candidate_lines
        )
        live_mismatch_count = sum(
            live_line_hashes.get(line_number) != line_hash
            for line_number, line_hash in current_line_hashes
        )
        if live_mismatch_count:
            observation = self._comparison_for_live_mismatch(
                call_id=call_id,
                render_id=render_id,
                file_key=file_key,
                file_history=file_history,
                content_version_hash=content_version_hash,
                current_line_hashes=current_line_hashes,
                epoch_advanced=epoch_advanced,
            )
            self._evict_files_over_limit()
            return observation

        comparison = self._compare_with_current_epoch(
            call_id=call_id,
            render_id=render_id,
            file_key=file_key,
            file_history=file_history,
            content_version_hash=content_version_hash,
            current_line_hashes=current_line_hashes,
            policy=policy,
            epoch_advanced=epoch_advanced,
        )
        file_history.observations.append(ReadFileResultObservation(
            call_id=call_id,
            render_id=render_id,
            line_hashes=current_line_hashes,
        ))
        del file_history.observations[:-self.max_observations_per_file]
        self._evict_files_over_limit()
        return comparison

    def invalidate_paths(
        self,
        raw_paths: Iterable[str],
        *,
        workspace_root: Path | None,
    ) -> None:
        """Advance tracked epochs after a successful native file mutation."""
        for raw_path in raw_paths:
            file_key = file_content_key_for_path(
                raw_path,
                workspace_root=workspace_root,
            )
            if file_key is None:
                continue
            file_history = self._history_by_file.get(file_key)
            if file_history is None:
                continue
            file_history.content_epoch = ReadFileContentEpoch(max(
                1,
                int(file_history.content_epoch) + 1,
            ))
            file_history.content_version_hash = None
            file_history.epoch_advanced_since_observation = True
            file_history.observations.clear()
            self._history_by_file.move_to_end(file_key)

    @property
    def retained_file_count(self) -> int:
        """Number of file identities currently retained (for tests/metrics)."""
        return len(self._history_by_file)

    def retained_observation_count(self, file_key: FileContentKey) -> int:
        """Number of call observations retained for one canonical file."""
        file_history = self._history_by_file.get(file_key)
        if file_history is None:
            return 0
        return len(file_history.observations)

    @staticmethod
    def _advance_epoch_for_version(
        file_history: _FileReadHistory,
        content_version_hash: FileContentVersionHash,
    ) -> bool:
        previous_version_hash = file_history.content_version_hash
        if previous_version_hash == content_version_hash:
            epoch_advanced = file_history.epoch_advanced_since_observation
            file_history.epoch_advanced_since_observation = False
            return epoch_advanced

        if previous_version_hash is not None:
            file_history.content_epoch = ReadFileContentEpoch(
                int(file_history.content_epoch) + 1,
            )
            file_history.observations.clear()
            file_history.epoch_advanced_since_observation = True
        elif file_history.content_epoch == 0:
            file_history.content_epoch = ReadFileContentEpoch(1)
        file_history.content_version_hash = content_version_hash
        epoch_advanced = file_history.epoch_advanced_since_observation
        file_history.epoch_advanced_since_observation = False
        return epoch_advanced

    @staticmethod
    def _comparison_for_no_returned_lines(
        *,
        call_id: str,
        render_id: str | None,
        file_key: FileContentKey,
        file_history: _FileReadHistory,
        content_version_hash: FileContentVersionHash,
        epoch_advanced: bool,
    ) -> ReadFileReuseObservation:
        return ReadFileReuseObservation(
            call_id=call_id,
            render_id=render_id,
            file_key=file_key,
            content_epoch=file_history.content_epoch,
            content_version_hash=content_version_hash,
            returned_start_line=None,
            returned_end_line=None,
            returned_line_count=0,
            matching_prior_line_count=0,
            new_line_count=0,
            overlapping_prior_call_count=0,
            exact_repeat_count=0,
            prior_call_id=None,
            prior_render_id=None,
            prior_call_ids=(),
            kind=ReadFileReuseKind.NO_RETURNED_LINES,
            hint_recommended=False,
            content_epoch_advanced=epoch_advanced,
        )

    @staticmethod
    def _comparison_for_live_mismatch(
        *,
        call_id: str,
        render_id: str | None,
        file_key: FileContentKey,
        file_history: _FileReadHistory,
        content_version_hash: FileContentVersionHash,
        current_line_hashes: tuple[tuple[int, FileLineContentHash], ...],
        epoch_advanced: bool,
    ) -> ReadFileReuseObservation:
        return ReadFileReuseObservation(
            call_id=call_id,
            render_id=render_id,
            file_key=file_key,
            content_epoch=file_history.content_epoch,
            content_version_hash=content_version_hash,
            returned_start_line=current_line_hashes[0][0],
            returned_end_line=current_line_hashes[-1][0],
            returned_line_count=len(current_line_hashes),
            matching_prior_line_count=0,
            new_line_count=len(current_line_hashes),
            overlapping_prior_call_count=0,
            exact_repeat_count=0,
            prior_call_id=None,
            prior_render_id=None,
            prior_call_ids=(),
            kind=ReadFileReuseKind.LIVE_CONTENT_MISMATCH,
            hint_recommended=False,
            content_epoch_advanced=epoch_advanced,
        )

    @staticmethod
    def _compare_with_current_epoch(
        *,
        call_id: str,
        render_id: str | None,
        file_key: FileContentKey,
        file_history: _FileReadHistory,
        content_version_hash: FileContentVersionHash,
        current_line_hashes: tuple[tuple[int, FileLineContentHash], ...],
        policy: ReadFileReusePolicy,
        epoch_advanced: bool,
    ) -> ReadFileReuseObservation:
        current_hashes_by_line = dict(current_line_hashes)
        matching_line_numbers: set[int] = set()
        prior_call_ids: list[str] = []
        overlapping_observations: list[
            tuple[int, ReadFileResultObservation]
        ] = []
        exact_repeats: list[ReadFileResultObservation] = []

        for prior in file_history.observations:
            matching_for_call = sum(
                current_hashes_by_line.get(line_number) == line_hash
                for line_number, line_hash in prior.line_hashes
                if line_number in current_hashes_by_line
            )
            if matching_for_call == 0:
                continue
            overlapping_observations.append((matching_for_call, prior))
            prior_call_ids.append(prior.call_id)
            matching_line_numbers.update(
                line_number
                for line_number, line_hash in prior.line_hashes
                if current_hashes_by_line.get(line_number) == line_hash
            )
            if prior.line_hashes == current_line_hashes:
                exact_repeats.append(prior)

        returned_line_count = len(current_line_hashes)
        matching_prior_line_count = len(matching_line_numbers)
        covered_fraction = matching_prior_line_count / returned_line_count
        hint_recommended = (
            matching_prior_line_count > 0
            and covered_fraction >= policy.minimum_covered_fraction
        )

        kind = ReadFileReuseKind.FIRST_OBSERVATION
        if file_history.observations:
            kind = ReadFileReuseKind.NO_OVERLAP
        if matching_prior_line_count:
            kind = ReadFileReuseKind.PARTIAL_OVERLAP
        if hint_recommended:
            kind = ReadFileReuseKind.SUBSTANTIAL_COVERAGE
        if matching_prior_line_count == returned_line_count:
            kind = ReadFileReuseKind.FULL_COVERAGE
        if exact_repeats:
            kind = ReadFileReuseKind.EXACT_REPEAT

        representative: ReadFileResultObservation | None = None
        if exact_repeats:
            representative = exact_repeats[-1]
        elif overlapping_observations:
            representative = max(
                overlapping_observations,
                key=lambda item: (
                    item[0],
                    file_history.observations.index(item[1]),
                ),
            )[1]

        return ReadFileReuseObservation(
            call_id=call_id,
            render_id=render_id,
            file_key=file_key,
            content_epoch=file_history.content_epoch,
            content_version_hash=content_version_hash,
            returned_start_line=current_line_hashes[0][0],
            returned_end_line=current_line_hashes[-1][0],
            returned_line_count=returned_line_count,
            matching_prior_line_count=matching_prior_line_count,
            new_line_count=returned_line_count - matching_prior_line_count,
            overlapping_prior_call_count=len(overlapping_observations),
            exact_repeat_count=len(exact_repeats),
            prior_call_id=(
                representative.call_id if representative is not None else None
            ),
            prior_render_id=(
                representative.render_id if representative is not None else None
            ),
            prior_call_ids=tuple(prior_call_ids),
            kind=kind,
            hint_recommended=hint_recommended,
            content_epoch_advanced=epoch_advanced,
        )

    def _evict_files_over_limit(self) -> None:
        while len(self._history_by_file) > self.max_files:
            self._history_by_file.popitem(last=False)


def _read_live_file_snapshot(
    path: Path,
) -> tuple[
    FileContentVersionHash,
    dict[int, FileLineContentHash],
] | None:
    try:
        raw_content = path.read_bytes()
        text_content = raw_content.decode("utf-8")
    except (OSError, UnicodeError):
        return None

    content_version_hash = FileContentVersionHash(
        hashlib.sha256(raw_content).hexdigest(),
    )
    line_hashes = {
        line_number: _line_hash(line)
        for line_number, line in enumerate(text_content.splitlines(), start=1)
    }
    return content_version_hash, line_hashes


def _line_hash(content: str) -> FileLineContentHash:
    return FileLineContentHash(
        hashlib.sha256(content.encode("utf-8")).hexdigest(),
    )


__all__ = [
    "DEFAULT_READ_FILE_REUSE_HINT_THRESHOLD",
    "FileContentVersionHash",
    "FileLineContentHash",
    "READ_FILE_REUSE_TELEMETRY_SCHEMA_VERSION",
    "SESSION_LEDGER_V1_READ_FILE_REUSE_POLICY",
    "ReadFileContentEpoch",
    "ReadFileResultHistory",
    "ReadFileReuseKind",
    "ReadFileReuseObservation",
    "ReadFileReusePolicy",
]
