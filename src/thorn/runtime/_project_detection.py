"""Project-directory detection for CLI session startup.

When a user runs ``thorn chat`` or ``thorn run`` from a directory deep
inside a project tree, the *session workspace* is the current working
directory but the *logical agent workspace* -- the upper bound of the
context-gathering walk that produces the system prompt -- is the
enclosing project root.  Without a project-aware policy the walk
either stops too soon (missing the project's ``AGENTS.md`` and skill
definitions) or runs all the way to the filesystem root (picking up
unrelated parents).

This module provides three small, deliberately-orthogonal pieces:

- :func:`is_logical_project_directory_path` -- the *predicate*: given
  a path, does it look like a project root?  Knows about
  version-control roots, common language manifests, agent-contract
  files, and a Thorn-agency marker.

- :func:`find_outermost_enclosing_logical_project_directory_path` --
  the *walker*: scan ancestors of a path (inclusive) and return the
  outermost one that satisfies the predicate.  Optionally bounded
  above by a caller-supplied path, so the walk does not escape into
  the user's home directory or higher.

- :func:`pick_logical_agent_workspace_path_for_cli_session` -- the
  *policy*: combine the predicate and walker with Thorn's CLI default
  (upper-bounded by the user's home directory; fall back to the
  session workspace itself if no enclosing project root is found).

The split is deliberate.  Refining the marker set ("recognise
``flake.nix``") changes only the predicate.  Switching from outermost
to innermost (or to a closest-match scoring rule) changes only the
walker.  Adjusting the CLI's fallback / upper-bound choices changes
only the policy function.  Each can be reasoned about and tested in
isolation, and none of the orthogonal concerns leak into the others.
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Marker policy
# ---------------------------------------------------------------------------

# Subdirectories whose presence marks the containing directory as a
# project root.  All entries must be subdirectories (not files) for
# the marker to trigger; this keeps an accidental ``.git`` regular
# file (a submodule pointer file) from registering as a repo root.
_PROJECT_DIRECTORY_MARKERS: tuple[str, ...] = (
    ".git",      # git repo root
    ".hg",       # mercurial repo root
    ".svn",      # subversion checkout root
    ".bzr",      # bazaar repo root
    ".jj",       # jujutsu repo root
    ".thorn",    # workspace-rooted Thorn agency home
)

# Files whose presence marks the containing directory as a project
# root.  All entries must be regular files for the marker to
# trigger.  The list is deliberately conservative: it covers the
# canonical "this directory is a software project" manifests for the
# major language ecosystems, plus the agent-contract files Thorn and
# Claude Code both consume.  Generic files like ``Makefile`` are
# *not* included because they appear too widely outside of true
# project roots.
_PROJECT_FILE_MARKERS: tuple[str, ...] = (
    "AGENTS.md",          # Thorn / OpenAI Codex / Cursor agent contract
    "CLAUDE.md",          # Claude Code agent contract (alias for AGENTS.md)
    "pyproject.toml",     # Python (PEP 518)
    "setup.py",           # Python (legacy)
    "setup.cfg",          # Python (legacy declarative)
    "package.json",       # Node.js
    "Cargo.toml",         # Rust
    "go.mod",             # Go module
    "pom.xml",            # Maven (Java)
    "build.gradle",       # Gradle (Groovy DSL)
    "build.gradle.kts",   # Gradle (Kotlin DSL)
    "Gemfile",            # Ruby (Bundler)
    "composer.json",      # PHP (Composer)
    "mix.exs",            # Elixir
)


def is_logical_project_directory_path(path: Path) -> bool:
    """Return ``True`` when *path* looks like a software project root.

    A path qualifies if it is an existing directory and contains at
    least one of:

    - a known version-control root subdirectory (``.git``, ``.hg``,
      ``.svn``, ``.bzr``, ``.jj``);
    - a Thorn workspace-rooted agency-home subdirectory (``.thorn``);
    - a recognised language-ecosystem manifest file (e.g.
      ``pyproject.toml``, ``package.json``, ``Cargo.toml``, ...);
    - an agent-contract file (``AGENTS.md`` or ``CLAUDE.md``).

    The full marker lists live in :data:`_PROJECT_DIRECTORY_MARKERS`
    and :data:`_PROJECT_FILE_MARKERS` at the top of this module.
    Adjusting the policy is a single-place change there.

    Returns ``False`` -- not raises -- when *path* does not exist,
    is not a directory, or contains no markers, so callers can scan
    arbitrary ancestor paths without worrying about file-system
    races.
    """
    if not path.is_dir():
        return False
    for marker in _PROJECT_DIRECTORY_MARKERS:
        if (path / marker).is_dir():
            return True
    for marker in _PROJECT_FILE_MARKERS:
        if (path / marker).is_file():
            return True
    return False


# ---------------------------------------------------------------------------
# Walker
# ---------------------------------------------------------------------------

def find_outermost_enclosing_logical_project_directory_path(
    path: Path,
    *,
    upper_bound: Path | None = None,
) -> Path | None:
    """Walk up from *path* and return the outermost project ancestor.

    The walk is inclusive of *path* itself: if *path* is already a
    logical project directory, it is the innermost candidate and is
    considered alongside its ancestors.  Among all ancestors that
    satisfy :func:`is_logical_project_directory_path`, the one
    closest to the filesystem root is returned -- "outermost" in the
    "biggest containing project" sense, not "deepest by path
    length".  If no ancestor matches, returns ``None``.

    *upper_bound*, when given, restricts the search to candidates
    that are *strictly below* it: a candidate ``c`` is eligible only
    when ``upper_bound`` is a proper ancestor of ``c``.  Pass
    ``Path.home()`` to keep CLI-session detection from adopting the
    user's home directory (or any directory above it) as the logical
    workspace.  *upper_bound* itself is never returned even if it
    satisfies the predicate; nor is anything above it.

    The "outermost" choice is policy: it favours broader context
    (the whole monorepo over an inner subpackage) which is what
    makes ``AGENTS.md`` files at the project root reliably visible
    to the per-prompt context walk.  Innermost would be a one-line
    change here if the policy ever needs revisiting.
    """
    candidates: list[Path] = []
    for candidate in (path, *path.parents):
        if upper_bound is not None:
            if candidate == upper_bound:
                # Reached the upper bound; nothing at or above it
                # is eligible, so the walk is done.
                break
            try:
                if not candidate.is_relative_to(upper_bound):
                    # Walked past the upper bound (it does not
                    # enclose *path*); treat as unbounded from here.
                    upper_bound = None
            except ValueError:
                upper_bound = None
        if is_logical_project_directory_path(candidate):
            candidates.append(candidate)
    if not candidates:
        return None
    return candidates[-1]


# ---------------------------------------------------------------------------
# CLI policy
# ---------------------------------------------------------------------------

def pick_logical_agent_workspace_path_for_cli_session(
    session_workspace_path: Path,
) -> Path:
    """Choose the logical agent-workspace path for a CLI session.

    Returns the outermost enclosing logical project directory of
    *session_workspace_path* -- i.e. the largest project tree that
    contains the session's working directory -- so long as that
    project root sits *strictly below* the user's home directory.
    Falls back to *session_workspace_path* itself if no enclosing
    project root is found, or if the only matches are at or above
    ``Path.home()``.

    The home-directory upper bound is a safety policy: a stray
    ``.git`` in the user's home (a dotfiles repo, a misplaced
    project) must not promote the entire home directory to the
    agent's workspace, which would expose unrelated personal files
    to the per-prompt context walk.  The fallback to
    *session_workspace_path* is the principle-of-least-surprise
    answer when no project root is identifiable: the session's CWD
    is at least demonstrably the place the user intended to work.

    Tests that need to override the home-directory boundary should
    monkey-patch ``Path.home`` (the same hook the existing CLI
    test-isolation fixture uses).
    """
    discovered = find_outermost_enclosing_logical_project_directory_path(
        session_workspace_path,
        upper_bound=Path.home(),
    )
    if discovered is not None:
        return discovered
    return session_workspace_path


__all__ = [
    "find_outermost_enclosing_logical_project_directory_path",
    "is_logical_project_directory_path",
    "pick_logical_agent_workspace_path_for_cli_session",
]
