"""Tests for `thorn.runtime._project_detection`.

Three orthogonal layers are tested separately:

- the predicate (``is_logical_project_directory_path``);
- the walker (``find_outermost_enclosing_logical_project_directory_path``);
- the CLI policy
  (``pick_logical_agent_workspace_path_for_cli_session``).

This mirrors the module's own three-piece structure so that a future
refinement of any single layer (e.g. adding a marker file) can be
covered by changing only the test cases for that layer.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from thorn.runtime._project_detection import (
    find_outermost_enclosing_logical_project_directory_path,
    is_logical_project_directory_path,
    pick_logical_agent_workspace_path_for_cli_session,
)


# ---------------------------------------------------------------------------
# is_logical_project_directory_path
# ---------------------------------------------------------------------------

class TestIsLogicalProjectDirectoryPath:
    def test_empty_directory_is_not_a_project(self, tmp_path: Path) -> None:
        assert is_logical_project_directory_path(tmp_path) is False

    def test_nonexistent_path_is_not_a_project(self, tmp_path: Path) -> None:
        assert is_logical_project_directory_path(tmp_path / "nope") is False

    def test_regular_file_is_not_a_project(self, tmp_path: Path) -> None:
        f = tmp_path / "thing.txt"
        f.write_text("x")
        assert is_logical_project_directory_path(f) is False

    @pytest.mark.parametrize(
        "marker",
        [".git", ".hg", ".svn", ".bzr", ".jj", ".thorn"],
    )
    def test_directory_marker_triggers(
        self, tmp_path: Path, marker: str,
    ) -> None:
        (tmp_path / marker).mkdir()
        assert is_logical_project_directory_path(tmp_path) is True

    def test_directory_marker_must_be_a_directory(
        self, tmp_path: Path,
    ) -> None:
        # A regular file named ``.git`` (e.g. a submodule pointer)
        # is intentionally NOT recognised as a project root, because
        # the containing directory is the *submodule's* working
        # tree, not a repository in its own right.
        (tmp_path / ".git").write_text("gitdir: ../.git/modules/sub\n")
        assert is_logical_project_directory_path(tmp_path) is False

    @pytest.mark.parametrize(
        "marker",
        [
            "AGENTS.md",
            "CLAUDE.md",
            "pyproject.toml",
            "setup.py",
            "setup.cfg",
            "package.json",
            "Cargo.toml",
            "go.mod",
            "pom.xml",
            "build.gradle",
            "build.gradle.kts",
            "Gemfile",
            "composer.json",
            "mix.exs",
        ],
    )
    def test_file_marker_triggers(
        self, tmp_path: Path, marker: str,
    ) -> None:
        (tmp_path / marker).write_text("# placeholder\n")
        assert is_logical_project_directory_path(tmp_path) is True

    def test_file_marker_must_be_a_file(self, tmp_path: Path) -> None:
        # A directory named ``pyproject.toml`` (highly unusual but
        # legal on POSIX) does not satisfy the file-marker check.
        (tmp_path / "pyproject.toml").mkdir()
        assert is_logical_project_directory_path(tmp_path) is False

    def test_unknown_marker_does_not_trigger(self, tmp_path: Path) -> None:
        (tmp_path / "Makefile").write_text("all:\n")
        (tmp_path / "README.md").write_text("# repo\n")
        # ``Makefile`` is intentionally excluded from the marker set
        # (too widely used outside true project roots), so even with
        # a README it should not register.
        assert is_logical_project_directory_path(tmp_path) is False


# ---------------------------------------------------------------------------
# find_outermost_enclosing_logical_project_directory_path
# ---------------------------------------------------------------------------

class TestFindOutermostEnclosingLogicalProjectDirectoryPath:
    def test_returns_none_when_no_marker_anywhere(self, tmp_path: Path) -> None:
        deep = tmp_path / "a" / "b" / "c"
        deep.mkdir(parents=True)
        assert (
            find_outermost_enclosing_logical_project_directory_path(
                deep, upper_bound=tmp_path,
            )
            is None
        )

    def test_returns_path_itself_when_path_is_a_project_root(
        self, tmp_path: Path,
    ) -> None:
        (tmp_path / ".git").mkdir()
        assert (
            find_outermost_enclosing_logical_project_directory_path(tmp_path)
            == tmp_path
        )

    def test_returns_immediate_parent_when_only_parent_matches(
        self, tmp_path: Path,
    ) -> None:
        project = tmp_path / "proj"
        project.mkdir()
        (project / "pyproject.toml").write_text("")
        cwd = project / "src" / "pkg"
        cwd.mkdir(parents=True)
        assert (
            find_outermost_enclosing_logical_project_directory_path(
                cwd, upper_bound=tmp_path,
            )
            == project
        )

    def test_outermost_wins_when_multiple_ancestors_match(
        self, tmp_path: Path,
    ) -> None:
        # Outer monorepo has its own ``.git``; an inner sub-package
        # has its own ``pyproject.toml``.  The outermost match (the
        # monorepo) should be picked, so the agent gets the broader
        # context.
        outer = tmp_path / "monorepo"
        outer.mkdir()
        (outer / ".git").mkdir()
        inner = outer / "packages" / "subpkg"
        inner.mkdir(parents=True)
        (inner / "pyproject.toml").write_text("")
        cwd = inner / "src"
        cwd.mkdir()
        assert (
            find_outermost_enclosing_logical_project_directory_path(
                cwd, upper_bound=tmp_path,
            )
            == outer
        )

    def test_upper_bound_excludes_itself(self, tmp_path: Path) -> None:
        # When the upper bound is itself a project root, it must
        # NOT be returned -- the search is "strictly below" the
        # bound.  Otherwise a stray ``.git`` in the user's home
        # would let home itself become the workspace.
        (tmp_path / ".git").mkdir()
        cwd = tmp_path / "child"
        cwd.mkdir()
        # No marker on ``cwd`` itself; the only matching ancestor is
        # the upper bound -- which is excluded -- so the walker
        # returns None.
        assert (
            find_outermost_enclosing_logical_project_directory_path(
                cwd, upper_bound=tmp_path,
            )
            is None
        )

    def test_upper_bound_excludes_anything_above_it(
        self, tmp_path: Path,
    ) -> None:
        # An ancestor *above* the upper bound is also excluded, even
        # if the bound itself is not a match.
        (tmp_path / ".git").mkdir()
        bound = tmp_path / "scope"
        bound.mkdir()
        cwd = bound / "deep"
        cwd.mkdir()
        assert (
            find_outermost_enclosing_logical_project_directory_path(
                cwd, upper_bound=bound,
            )
            is None
        )

    def test_upper_bound_that_does_not_enclose_path_is_ignored(
        self, tmp_path: Path,
    ) -> None:
        # When *upper_bound* is not an ancestor of *path* at all,
        # the bound contributes nothing -- the walker behaves as if
        # unbounded.  This is the defensive branch in the walker.
        (tmp_path / ".git").mkdir()
        elsewhere = tmp_path.parent / "elsewhere-bound"
        # Sanity: ``elsewhere`` does not enclose ``tmp_path``.
        assert tmp_path != elsewhere
        result = find_outermost_enclosing_logical_project_directory_path(
            tmp_path, upper_bound=elsewhere,
        )
        assert result == tmp_path

    def test_finds_match_at_path_itself_with_upper_bound(
        self, tmp_path: Path,
    ) -> None:
        outer = tmp_path / "outer"
        outer.mkdir()
        proj = outer / "proj"
        proj.mkdir()
        (proj / "AGENTS.md").write_text("")
        # Path itself is a project; upper bound is its parent.
        # Strictly-below check must NOT exclude the path itself.
        assert (
            find_outermost_enclosing_logical_project_directory_path(
                proj, upper_bound=outer,
            )
            == proj
        )


# ---------------------------------------------------------------------------
# pick_logical_agent_workspace_path_for_cli_session
# ---------------------------------------------------------------------------

class TestPickLogicalAgentWorkspacePathForCliSession:
    @pytest.fixture(autouse=True)
    def _isolate_home(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    ) -> Path:
        """Treat *tmp_path* as the user's home directory.

        The CLI policy uses ``Path.home()`` as its upper bound; tests
        need a controllable one so they can place project roots both
        above and below it without touching the developer's real
        home directory.  Mirrors the ``_isolate_cli_agency_home``
        fixture in ``tests/test_cli.py``.
        """
        monkeypatch.setattr(
            Path, "home", classmethod(lambda _cls: tmp_path),
        )
        return tmp_path

    def test_returns_session_workspace_when_no_project_found(
        self, _isolate_home: Path,
    ) -> None:
        cwd = _isolate_home / "scratch"
        cwd.mkdir()
        # No project markers anywhere below home; CWD itself isn't a
        # project either.
        assert pick_logical_agent_workspace_path_for_cli_session(cwd) == cwd

    def test_returns_enclosing_project_root(self, _isolate_home: Path) -> None:
        project = _isolate_home / "projects" / "myrepo"
        project.mkdir(parents=True)
        (project / ".git").mkdir()
        cwd = project / "src" / "pkg"
        cwd.mkdir(parents=True)
        assert (
            pick_logical_agent_workspace_path_for_cli_session(cwd) == project
        )

    def test_picks_outermost_when_multiple_projects_nest(
        self, _isolate_home: Path,
    ) -> None:
        outer = _isolate_home / "monorepo"
        outer.mkdir()
        (outer / "pyproject.toml").write_text("")
        inner = outer / "subpkg"
        inner.mkdir()
        (inner / "Cargo.toml").write_text("")
        assert (
            pick_logical_agent_workspace_path_for_cli_session(inner)
            == outer
        )

    def test_does_not_promote_home_to_workspace(
        self, _isolate_home: Path,
    ) -> None:
        # A stray ``.git`` in the user's home must never escalate
        # the home dir itself to the agent's workspace.
        (_isolate_home / ".git").mkdir()
        downloads = _isolate_home / "Downloads"
        downloads.mkdir()
        assert (
            pick_logical_agent_workspace_path_for_cli_session(downloads)
            == downloads
        )

    def test_does_not_walk_above_home(self, _isolate_home: Path) -> None:
        # A project marker *above* the user's home is invisible to
        # the CLI policy.  We can't realistically write to the
        # filesystem above tmp_path, but we can fake the same thing
        # by setting ``Path.home()`` *below* a tmp project root.
        outer = _isolate_home / "above-home"
        outer.mkdir()
        (outer / ".git").mkdir()
        # Move the synthetic home one level deeper so the marker is
        # technically above it.
        deeper_home = outer / "home"
        deeper_home.mkdir()
        # We're already inside the autouse fixture; rebind home
        # again for the duration of this test only.
        import pytest as _pytest  # local import to use a fresh monkeypatch
        with _pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                Path, "home", classmethod(lambda _cls: deeper_home),
            )
            cwd = deeper_home / "work"
            cwd.mkdir()
            assert (
                pick_logical_agent_workspace_path_for_cli_session(cwd) == cwd
            )
