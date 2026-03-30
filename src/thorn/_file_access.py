"""File access control for thorn agents.

Provides gitignore-style pattern matching to control which files agents
can list, read, and write.  Rules are accumulated via the MRO on
``Agent`` subclasses (like tools and system prompts) and enforced by
the built-in file tools at runtime.

This system protects against agents naively overstepping their role,
not against adversarial prompt injection.  True security requires
OS-level sandboxing, which is out of scope.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path, PurePosixPath
from typing import Sequence

import pathspec


class FileAccessLevel(IntEnum):
    """Tiers of file access, ordered from most to least restrictive."""

    HIDDEN = 0
    NONE = 1
    READ = 2
    WRITE = 3


@dataclass(frozen=True)
class FileAccessRule:
    """A single gitignore-style pattern mapped to an access level.

    Patterns follow ``.gitignore`` semantics (via the ``pathspec``
    library):

    - ``*`` matches anything except ``/``
    - ``**`` matches zero or more path components
    - A leading ``/`` anchors to the workspace root
    - A trailing ``/`` matches only directories
    """

    pattern: str
    access: FileAccessLevel


def _normalise_pattern(pattern: str, workspace: Path | None) -> str:
    """Convert an absolute-path pattern to workspace-relative POSIX form.

    If *pattern* is a relative path or a glob expression (containing
    ``*`` or ``?``), it is returned unchanged.  Only literal absolute
    paths are converted, using *workspace* to strip the prefix.
    """
    if workspace is None:
        return pattern
    # Only convert patterns that look like literal absolute paths
    # (no glob wildcards) — e.g. output from module_source_path().
    if "*" in pattern or "?" in pattern:
        return pattern
    p = Path(pattern)
    if not p.is_absolute():
        return pattern
    try:
        rel = p.resolve().relative_to(workspace.resolve())
        return rel.as_posix()
    except ValueError:
        return pattern


class FileAccessPolicy:
    """Ordered rule list evaluated with last-match-wins semantics.

    When no rule matches a path, *default* is returned (typically
    ``NONE``).  A separate *global_ceiling* policy, when present, caps
    the effective level so that global ignore files can only *reduce*
    access.

    If *workspace* is provided, absolute paths in rule patterns are
    normalised to workspace-relative POSIX paths before matching.
    This lets ``_instance_file_access()`` methods pass the output of
    path-computing functions (which may return absolute paths) directly
    as patterns.
    """

    def __init__(
        self,
        rules: Sequence[FileAccessRule],
        *,
        default: FileAccessLevel = FileAccessLevel.NONE,
        global_ceiling: FileAccessPolicy | None = None,
        workspace: Path | None = None,
    ) -> None:
        self._rules = list(rules)
        self._default = default
        self._global_ceiling = global_ceiling
        self._workspace = workspace

        self._specs: list[tuple[pathspec.PathSpec, FileAccessLevel]] = []
        for rule in self._rules:
            pattern = _normalise_pattern(rule.pattern, workspace)
            spec = pathspec.PathSpec.from_lines("gitwildmatch", [pattern])
            self._specs.append((spec, rule.access))

    @property
    def rules(self) -> list[FileAccessRule]:
        return list(self._rules)

    @property
    def default(self) -> FileAccessLevel:
        return self._default

    def check(self, path: str | PurePosixPath) -> FileAccessLevel:
        """Return the effective access level for *path*.

        *path* should be workspace-relative (forward slashes, no
        leading ``./``).  Every rule is tested; the last match wins.
        If a *global_ceiling* is set, the result is capped.
        """
        path_str = str(path)
        result = self._default
        for spec, access in self._specs:
            if spec.match_file(path_str):
                result = access

        if self._global_ceiling is not None:
            ceiling = self._global_ceiling.check(path_str)
            result = min(result, ceiling)

        return result

    def filter_listing(
        self,
        entries: list[str],
        directory: str | PurePosixPath,
    ) -> list[str]:
        """Return *entries* with ``HIDDEN`` items removed.

        Each entry is checked as ``directory/entry``.
        """
        dir_path = PurePosixPath(directory) if not isinstance(directory, PurePosixPath) else directory
        return [
            e for e in entries
            if self.check(dir_path / e) > FileAccessLevel.HIDDEN
        ]

    def with_ceiling(self, ceiling: FileAccessPolicy) -> FileAccessPolicy:
        """Return a new policy capped by *ceiling*."""
        return FileAccessPolicy(
            self._rules,
            default=self._default,
            global_ceiling=ceiling,
            workspace=self._workspace,
        )


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

def resolve_for_check(raw_path: str, workspace: Path) -> PurePosixPath:
    """Resolve *raw_path* to a workspace-relative POSIX path for matching.

    Follows symlinks and canonicalizes case (important on
    case-insensitive filesystems like Windows NTFS) so that agents
    cannot bypass rules by varying capitalisation.

    Paths outside the workspace are returned as absolute POSIX paths,
    which will not match workspace-relative patterns and therefore
    fall through to the policy default (typically ``NONE``).
    """
    p = Path(raw_path)
    if not p.is_absolute():
        p = Path.cwd() / p
    p = p.resolve()
    try:
        rel = p.relative_to(workspace)
    except ValueError:
        return PurePosixPath(p.as_posix())
    return PurePosixPath(rel.as_posix())


def check_access(
    path: str,
    required: FileAccessLevel,
    *,
    policy: FileAccessPolicy,
    workspace: Path,
) -> Path:
    """Validate that *path* meets *required* access under *policy*.

    Returns the resolved ``Path`` on success; raises
    ``PermissionError`` on failure with a message suitable for
    surfacing to an LLM.
    """
    resolved = resolve_for_check(path, workspace)
    level = policy.check(resolved)
    if level < required:
        raise PermissionError(
            f"Access denied: {path} requires {required.name} access, "
            f"but current policy grants {level.name}"
        )
    return Path(path)


# ---------------------------------------------------------------------------
# Global ignore file loading
# ---------------------------------------------------------------------------

def load_ignore_file(path: Path) -> list[FileAccessRule]:
    """Parse a ``.aiignore`` or ``.thornignore`` file into HIDDEN rules.

    Each non-blank, non-comment line becomes a rule that hides the
    matched paths.  This mirrors ``.gitignore`` syntax: lines starting
    with ``#`` are comments.
    """
    if not path.is_file():
        return []
    rules: list[FileAccessRule] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        rules.append(FileAccessRule(stripped, FileAccessLevel.HIDDEN))
    return rules


def load_global_ignores(workspace: Path) -> FileAccessPolicy | None:
    """Load ``.aiignore`` and ``.thornignore`` from *workspace*.

    Returns ``None`` if no ignore files exist.  ``.thornignore``
    rules are appended after ``.aiignore`` (last match wins), so
    Thorn-specific overrides take precedence.
    """
    rules: list[FileAccessRule] = []
    rules.extend(load_ignore_file(workspace / ".aiignore"))
    rules.extend(load_ignore_file(workspace / ".thornignore"))
    if not rules:
        return None
    return FileAccessPolicy(rules, default=FileAccessLevel.WRITE)
