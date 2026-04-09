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

from dataclasses import dataclass
from enum import Enum, IntEnum
from pathlib import Path
from typing import Mapping, Sequence

import pathspec


class FileAccessLevel(IntEnum):
    """Tiers of file access, ordered from most to least restrictive."""

    HIDDEN = 0
    NONE = 1
    READ = 2
    WRITE = 3


class RelativeTo(Enum):
    """Which root directory a :class:`FileAccessRule` pattern is relative to.

    Each variant names a conceptual root.  The concrete path for each
    root is supplied when constructing a :class:`FileAccessPolicy` via
    the *roots* mapping.
    """

    WORKSPACE = "workspace"
    AGENT_HOME = "agent_home"


@dataclass(frozen=True)
class FileAccessRule:
    """A single gitignore-style pattern mapped to an access level.

    Patterns follow ``.gitignore`` semantics (via the ``pathspec``
    library):

    - ``*`` matches anything except ``/``
    - ``**`` matches zero or more path components
    - A leading ``/`` anchors to the root directory
    - A trailing ``/`` matches only directories

    The *relative_to* field determines which root directory the pattern
    is resolved against.  See :class:`RelativeTo`.
    """

    pattern: str
    access: FileAccessLevel
    relative_to: RelativeTo = RelativeTo.WORKSPACE


def _normalise_pattern(pattern: str, root: Path | None) -> str:
    """Convert an absolute-path pattern to root-relative POSIX form.

    If *pattern* is a relative path or a glob expression (containing
    ``*`` or ``?``), it is returned unchanged.  Only literal absolute
    paths are converted, using *root* to strip the prefix.
    """
    if root is None:
        return pattern
    if "*" in pattern or "?" in pattern:
        return pattern
    p = Path(pattern)
    if not p.is_absolute():
        return pattern
    try:
        rel = p.resolve().relative_to(root)
        return rel.as_posix()
    except ValueError:
        return pattern


class FileAccessPolicy:
    """Ordered rule list evaluated with last-match-wins semantics.

    When no rule matches a path, *default* is returned (typically
    ``NONE``).  A separate *global_ceiling* policy, when present, caps
    the effective level so that global ignore files can only *reduce*
    access.

    *roots* maps each :class:`RelativeTo` variant to the concrete
    directory path it represents.  Rules whose ``relative_to`` variant
    has no entry in *roots* (or maps to ``None``) are silently skipped
    during matching -- they cannot apply without a root to resolve
    against.  Absolute-path patterns in rules are normalised against
    the corresponding root at construction time.
    """

    def __init__(
        self,
        rules: Sequence[FileAccessRule],
        *,
        default: FileAccessLevel = FileAccessLevel.NONE,
        global_ceiling: FileAccessPolicy | None = None,
        roots: Mapping[RelativeTo, Path | None] | None = None,
    ) -> None:
        self._rules = list(rules)
        self._default = default
        self._global_ceiling = global_ceiling
        self._roots: dict[RelativeTo, Path] = {
            k: v.resolve() for k, v in (roots or {}).items() if v is not None
        }

        self._compiled: list[
            tuple[pathspec.PathSpec, FileAccessLevel, Path | None]
        ] = []
        for rule in self._rules:
            root = self._roots.get(rule.relative_to)
            pattern = _normalise_pattern(rule.pattern, root)
            spec = pathspec.PathSpec.from_lines("gitwildmatch", [pattern])
            self._compiled.append((spec, rule.access, root))

    @property
    def rules(self) -> list[FileAccessRule]:
        return list(self._rules)

    @property
    def default(self) -> FileAccessLevel:
        return self._default

    def check(self, path: Path) -> FileAccessLevel:
        """Return the effective access level for *path*.

        *path* must be an absolute, resolved filesystem path.  Each
        rule is tested against *path* made relative to the rule's root
        directory; rules whose root is ``None`` or that do not contain
        *path* are skipped.  Last matching rule wins.

        If a *global_ceiling* is set, the result is capped.
        """
        resolved = path.resolve()
        result = self._default
        for spec, access, root in self._compiled:
            if root is None:
                continue
            try:
                rel = resolved.relative_to(root)
            except ValueError:
                continue
            if spec.match_file(rel.as_posix()):
                result = access

        if self._global_ceiling is not None:
            ceiling = self._global_ceiling.check(resolved)
            result = min(result, ceiling)

        return result

    def filter_listing(
        self,
        entries: list[str],
        directory: Path,
    ) -> list[str]:
        """Return *entries* with ``HIDDEN`` items removed.

        *directory* must be an absolute path.  Each entry is checked
        as ``directory / entry``.
        """
        return [
            e for e in entries
            if self.check(directory / e) > FileAccessLevel.HIDDEN
        ]

    def with_ceiling(self, ceiling: FileAccessPolicy) -> FileAccessPolicy:
        """Return a new policy capped by *ceiling*."""
        return FileAccessPolicy(
            self._rules,
            default=self._default,
            global_ceiling=ceiling,
            roots=self._roots,
        )


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

def resolve_for_check(raw_path: str, workspace: Path) -> Path:
    """Resolve *raw_path* to an absolute, canonical filesystem path.

    Relative paths are resolved against *workspace*.  Symlinks are
    followed and case is canonicalized (important on case-insensitive
    filesystems like Windows NTFS) so that agents cannot bypass rules
    by varying capitalisation.
    """
    p = Path(raw_path)
    if not p.is_absolute():
        p = workspace / p
    return p.resolve()


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

    Ignore rules are always workspace-relative (``RelativeTo.WORKSPACE``
    by default), so the workspace is passed as the sole root.
    """
    rules: list[FileAccessRule] = []
    rules.extend(load_ignore_file(workspace / ".aiignore"))
    rules.extend(load_ignore_file(workspace / ".thornignore"))
    if not rules:
        return None
    return FileAccessPolicy(
        rules,
        default=FileAccessLevel.WRITE,
        roots={RelativeTo.WORKSPACE: workspace},
    )
