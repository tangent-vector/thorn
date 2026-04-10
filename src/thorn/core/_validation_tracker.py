"""Makefile-style validation tracker with scan-based dirty detection.

The ``ValidationTracker`` maintains named validation targets (e.g.
"build", "test"), each associated with file glob patterns.  It detects
staleness by scanning the filesystem and comparing content checksums
against a baseline snapshot taken when the validation last ran.  A
compact one-line status dashboard is surfaced as a footer on tool-call
responses so the agent always knows what's pending.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class ValidationStatus(Enum):
    """Possible states for a validation target."""

    UNKNOWN = "unknown"
    STALE = "stale"
    PASSING = "passing"
    FAILING = "failing"


@dataclass(frozen=True)
class FileSnapshot:
    """Immutable record of file checksums at a point in time.

    Two snapshots are equal when they have the same set of paths with
    the same checksums; any difference (changed content, added file,
    removed file) means the underlying files have changed.
    """

    checksums: dict[str, str]

    @staticmethod
    def scan(root: Path, patterns: list[str]) -> FileSnapshot:
        """Build a snapshot by globbing *root* for each pattern and
        computing the MD5 digest of every matched file's content."""
        checksums: dict[str, str] = {}
        for pattern in patterns:
            for path in sorted(root.glob(pattern)):
                if not path.is_file():
                    continue
                rel = path.relative_to(root).as_posix()
                if rel not in checksums:
                    checksums[rel] = _md5(path)
        return FileSnapshot(checksums=checksums)

    def diff_count(self, other: FileSnapshot) -> int:
        """Number of files that differ between *self* and *other*.

        Counts changed checksums, files present in only one snapshot,
        and files removed from the other.
        """
        all_paths = set(self.checksums) | set(other.checksums)
        return sum(
            1
            for p in all_paths
            if self.checksums.get(p) != other.checksums.get(p)
        )


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class ValidationTarget:
    """A named validation target with file patterns and dependency info."""

    name: str
    file_patterns: list[str]
    depends_on: list[str] = field(default_factory=list)
    last_result: ValidationStatus = ValidationStatus.UNKNOWN
    last_summary: str | None = None
    baseline: FileSnapshot | None = None
    stale_file_count: int = 0


class ValidationTracker:
    """Manages validation targets, detects staleness, and renders
    a compact status dashboard.

    Conforms to the ``StatusProvider`` protocol so it can be
    registered on an ``ExecutionContext`` alongside other providers.
    """

    def __init__(self, root: Path) -> None:
        self._root = root
        self._targets: dict[str, ValidationTarget] = {}

    @property
    def source_label(self) -> str:
        return "validation"

    @property
    def root(self) -> Path:
        return self._root

    @property
    def targets(self) -> dict[str, ValidationTarget]:
        return self._targets

    def add_target(
        self,
        name: str,
        file_patterns: list[str],
        depends_on: list[str] | None = None,
    ) -> None:
        self._targets[name] = ValidationTarget(
            name=name,
            file_patterns=file_patterns,
            depends_on=list(depends_on) if depends_on else [],
        )

    def record_result(
        self,
        name: str,
        *,
        passed: bool,
        summary: str | None = None,
    ) -> None:
        """Record the outcome of a validation run.

        Sets the target's last_result, stores the summary, and takes a
        fresh snapshot of the target's files as the new baseline.
        """
        target = self._targets.get(name)
        if target is None:
            return
        target.last_result = (
            ValidationStatus.PASSING if passed else ValidationStatus.FAILING
        )
        target.last_summary = summary
        target.baseline = FileSnapshot.scan(self._root, target.file_patterns)
        target.stale_file_count = 0

    def refresh(self, session: object = None) -> None:
        """Scan the filesystem and update staleness for every target.

        The *session* parameter satisfies the ``StatusProvider``
        protocol; this tracker's state is workspace-scoped, so the
        session is ignored.
        """
        for target in self._targets.values():
            current = FileSnapshot.scan(self._root, target.file_patterns)
            if target.baseline is None:
                target.stale_file_count = len(current.checksums)
            else:
                target.stale_file_count = current.diff_count(target.baseline)

    def effective_status(
        self, name: str, _visited: set[str] | None = None,
    ) -> ValidationStatus:
        """Return the target's status accounting for transitive deps."""
        if _visited is None:
            _visited = set()

        if name in _visited:
            return ValidationStatus.STALE
        _visited.add(name)

        target = self._targets.get(name)
        if target is None:
            return ValidationStatus.UNKNOWN

        for dep_name in target.depends_on:
            dep_status = self.effective_status(dep_name, _visited)
            if dep_status in (
                ValidationStatus.FAILING,
                ValidationStatus.STALE,
                ValidationStatus.UNKNOWN,
            ):
                return ValidationStatus.STALE

        if target.stale_file_count > 0:
            return ValidationStatus.STALE

        return target.last_result

    def _blocking_dep(
        self, target: ValidationTarget, _visited: set[str] | None = None,
    ) -> tuple[str, ValidationStatus] | None:
        """Find the first dependency whose status blocks this target.

        Returns ``(dep_name, dep_effective_status)`` or ``None``.
        """
        if _visited is None:
            _visited = set()
        _visited.add(target.name)

        for dep_name in target.depends_on:
            if dep_name in _visited:
                continue
            dep_status = self.effective_status(dep_name)
            if dep_status in (
                ValidationStatus.FAILING,
                ValidationStatus.STALE,
                ValidationStatus.UNKNOWN,
            ):
                return dep_name, dep_status
        return None

    def render_status(self, session: object = None) -> str | None:
        """Produce a one-line status dashboard, or ``None`` if nothing
        useful to show.

        The *session* parameter satisfies the ``StatusProvider``
        protocol; this tracker's state is workspace-scoped, so the
        session is ignored.
        """
        if not self._targets:
            return None

        statuses = {
            name: self.effective_status(name) for name in self._targets
        }

        parts: list[str] = []
        for target in self._targets.values():
            eff = statuses[target.name]

            if eff == ValidationStatus.UNKNOWN and target.stale_file_count == 0:
                continue

            if eff == ValidationStatus.STALE:
                blocker = self._blocking_dep(target)
                if blocker is not None:
                    dep_name, dep_status = blocker
                    parts.append(
                        f"{target.name}: blocked ({dep_name} {dep_status.value})"
                    )
                elif target.stale_file_count > 0:
                    parts.append(
                        f"{target.name}: stale "
                        f"({target.stale_file_count} files changed)"
                    )
                else:
                    parts.append(f"{target.name}: stale")
            elif eff == ValidationStatus.PASSING:
                parts.append(f"{target.name}: passing")
            elif eff == ValidationStatus.FAILING:
                if target.last_summary:
                    parts.append(f"{target.name}: {target.last_summary}")
                else:
                    parts.append(f"{target.name}: failing")

        if not parts:
            return None

        if all(s == ValidationStatus.PASSING for s in statuses.values()):
            return "[all validations passing]"

        return "[" + ", ".join(parts) + "]"
