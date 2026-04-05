"""Tests for thorn._file_access — file access control system."""

from __future__ import annotations

from pathlib import Path, PurePosixPath

import pytest

from thorn.core._agent import Agent, _default_file_access
from thorn.core._context import ExecutionContext, reset_context, set_context
from thorn.core._file_access import (
    FileAccessLevel,
    FileAccessPolicy,
    FileAccessRule,
    check_access,
    load_global_ignores,
    load_ignore_file,
    resolve_for_check,
)
from thorn.core._provider import MockProvider


# ---------------------------------------------------------------------------
# FileAccessLevel ordering
# ---------------------------------------------------------------------------


class TestFileAccessLevel:
    def test_ordering(self):
        assert FileAccessLevel.HIDDEN < FileAccessLevel.NONE
        assert FileAccessLevel.NONE < FileAccessLevel.READ
        assert FileAccessLevel.READ < FileAccessLevel.WRITE

    def test_comparison_with_ints(self):
        assert FileAccessLevel.READ >= FileAccessLevel.READ
        assert FileAccessLevel.WRITE >= FileAccessLevel.READ
        assert not (FileAccessLevel.NONE >= FileAccessLevel.READ)


# ---------------------------------------------------------------------------
# FileAccessPolicy — basic matching
# ---------------------------------------------------------------------------


class TestFileAccessPolicy:
    def test_default_when_no_rules(self):
        policy = FileAccessPolicy([], default=FileAccessLevel.NONE)
        assert policy.check("anything.txt") == FileAccessLevel.NONE

    def test_single_wildcard_rule(self):
        policy = FileAccessPolicy(
            [FileAccessRule("**", FileAccessLevel.WRITE)],
            default=FileAccessLevel.NONE,
        )
        assert policy.check("foo.txt") == FileAccessLevel.WRITE
        assert policy.check("src/bar.cpp") == FileAccessLevel.WRITE

    def test_last_match_wins(self):
        policy = FileAccessPolicy([
            FileAccessRule("**", FileAccessLevel.WRITE),
            FileAccessRule("*.secret", FileAccessLevel.HIDDEN),
        ])
        assert policy.check("readme.md") == FileAccessLevel.WRITE
        assert policy.check("keys.secret") == FileAccessLevel.HIDDEN

    def test_specific_file_overrides_glob(self):
        policy = FileAccessPolicy([
            FileAccessRule("**", FileAccessLevel.READ),
            FileAccessRule("src/parser.cpp", FileAccessLevel.WRITE),
        ])
        assert policy.check("src/parser.cpp") == FileAccessLevel.WRITE
        assert policy.check("src/parser.h") == FileAccessLevel.READ

    def test_directory_pattern(self):
        policy = FileAccessPolicy([
            FileAccessRule("**", FileAccessLevel.READ),
            FileAccessRule(".thorn/**", FileAccessLevel.HIDDEN),
        ])
        assert policy.check("src/main.cpp") == FileAccessLevel.READ
        assert policy.check(".thorn/roles.py") == FileAccessLevel.HIDDEN
        assert policy.check(".thorn/build_tools.py") == FileAccessLevel.HIDDEN

    def test_override_earlier_restriction(self):
        """A later rule can grant higher access than an earlier restriction."""
        policy = FileAccessPolicy([
            FileAccessRule("**", FileAccessLevel.NONE),
            FileAccessRule("src/**", FileAccessLevel.READ),
            FileAccessRule("src/parser.cpp", FileAccessLevel.WRITE),
        ])
        assert policy.check("README.md") == FileAccessLevel.NONE
        assert policy.check("src/parser.h") == FileAccessLevel.READ
        assert policy.check("src/parser.cpp") == FileAccessLevel.WRITE


# ---------------------------------------------------------------------------
# FileAccessPolicy — filter_listing
# ---------------------------------------------------------------------------


class TestFilterListing:
    def test_hidden_entries_removed(self):
        policy = FileAccessPolicy([
            FileAccessRule("**", FileAccessLevel.READ),
            FileAccessRule(".thorn", FileAccessLevel.HIDDEN),
            FileAccessRule(".thorn/**", FileAccessLevel.HIDDEN),
        ])
        entries = [".thorn", "src", "README.md"]
        filtered = policy.filter_listing(entries, PurePosixPath("."))
        assert ".thorn" not in filtered
        assert "src" in filtered
        assert "README.md" in filtered

    def test_none_entries_visible(self):
        policy = FileAccessPolicy([
            FileAccessRule("**", FileAccessLevel.NONE),
        ])
        entries = ["foo.txt", "bar.txt"]
        filtered = policy.filter_listing(entries, PurePosixPath("."))
        assert filtered == ["foo.txt", "bar.txt"]


# ---------------------------------------------------------------------------
# FileAccessPolicy — global ceiling
# ---------------------------------------------------------------------------


class TestGlobalCeiling:
    def test_ceiling_caps_access(self):
        agent_policy = FileAccessPolicy(
            [FileAccessRule("**", FileAccessLevel.WRITE)],
        )
        ceiling = FileAccessPolicy(
            [FileAccessRule("secrets/**", FileAccessLevel.HIDDEN)],
            default=FileAccessLevel.WRITE,
        )
        capped = agent_policy.with_ceiling(ceiling)

        assert capped.check("readme.md") == FileAccessLevel.WRITE
        assert capped.check("secrets/api_key.txt") == FileAccessLevel.HIDDEN

    def test_ceiling_cannot_grant_more(self):
        agent_policy = FileAccessPolicy(
            [FileAccessRule("**", FileAccessLevel.READ)],
        )
        ceiling = FileAccessPolicy(
            [FileAccessRule("**", FileAccessLevel.WRITE)],
            default=FileAccessLevel.WRITE,
        )
        capped = agent_policy.with_ceiling(ceiling)
        assert capped.check("anything.txt") == FileAccessLevel.READ


# ---------------------------------------------------------------------------
# Absolute path normalisation
# ---------------------------------------------------------------------------


class TestPatternNormalisation:
    def test_absolute_path_normalised(self, tmp_path):
        workspace = tmp_path
        abs_path = str(workspace / "src" / "parser.cpp")
        policy = FileAccessPolicy(
            [FileAccessRule(abs_path, FileAccessLevel.WRITE)],
            default=FileAccessLevel.NONE,
            workspace=workspace,
        )
        assert policy.check("src/parser.cpp") == FileAccessLevel.WRITE
        assert policy.check("src/other.cpp") == FileAccessLevel.NONE

    def test_glob_pattern_not_normalised(self, tmp_path):
        workspace = tmp_path
        policy = FileAccessPolicy(
            [FileAccessRule("src/**/*.cpp", FileAccessLevel.WRITE)],
            default=FileAccessLevel.NONE,
            workspace=workspace,
        )
        assert policy.check("src/parser.cpp") == FileAccessLevel.WRITE

    def test_relative_path_unchanged(self, tmp_path):
        workspace = tmp_path
        policy = FileAccessPolicy(
            [FileAccessRule("src/parser.cpp", FileAccessLevel.WRITE)],
            default=FileAccessLevel.NONE,
            workspace=workspace,
        )
        assert policy.check("src/parser.cpp") == FileAccessLevel.WRITE


# ---------------------------------------------------------------------------
# resolve_for_check
# ---------------------------------------------------------------------------


class TestResolveForCheck:
    def test_relative_path_inside_workspace(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "src").mkdir()
        result = resolve_for_check("src/main.cpp", tmp_path)
        assert result == PurePosixPath("src/main.cpp")

    def test_absolute_path_inside_workspace(self, tmp_path):
        abs_path = str(tmp_path / "src" / "main.cpp")
        result = resolve_for_check(abs_path, tmp_path)
        assert result == PurePosixPath("src/main.cpp")

    def test_path_outside_workspace(self, tmp_path):
        outside = str(Path(tmp_path.root) / "outside" / "file.txt")
        result = resolve_for_check(outside, tmp_path)
        # Should be an absolute POSIX path that doesn't match workspace-relative patterns
        assert result.is_absolute() or str(result).startswith("/") or ":" in str(result)


# ---------------------------------------------------------------------------
# check_access
# ---------------------------------------------------------------------------


class TestCheckAccess:
    def test_allowed_access(self, tmp_path):
        policy = FileAccessPolicy(
            [FileAccessRule("**", FileAccessLevel.WRITE)],
        )
        result = check_access(
            str(tmp_path / "file.txt"),
            FileAccessLevel.WRITE,
            policy=policy,
            workspace=tmp_path,
        )
        assert isinstance(result, Path)

    def test_denied_access_raises(self, tmp_path):
        policy = FileAccessPolicy(
            [FileAccessRule("**", FileAccessLevel.READ)],
        )
        with pytest.raises(PermissionError, match="requires WRITE"):
            check_access(
                str(tmp_path / "file.txt"),
                FileAccessLevel.WRITE,
                policy=policy,
                workspace=tmp_path,
            )

    def test_error_message_includes_path(self, tmp_path):
        policy = FileAccessPolicy(
            [FileAccessRule("**", FileAccessLevel.NONE)],
        )
        with pytest.raises(PermissionError, match="file.txt"):
            check_access(
                str(tmp_path / "file.txt"),
                FileAccessLevel.READ,
                policy=policy,
                workspace=tmp_path,
            )


# ---------------------------------------------------------------------------
# MRO-based file_access collection
# ---------------------------------------------------------------------------


class TestMROFileAccess:
    def test_default_agent_rules(self):
        """Base Agent with no file_access in __dict__ should use the default."""
        rules = Agent._collect_file_access()
        assert len(rules) >= 1
        # Default should grant WRITE to **
        assert any(
            r.pattern == "**" and r.access == FileAccessLevel.WRITE
            for r in rules
        )

    def test_subclass_rules_appended(self):
        class Restricted(Agent):
            file_access = [
                FileAccessRule("**", FileAccessLevel.READ),
            ]

        rules = Restricted._collect_file_access()
        assert any(r.access == FileAccessLevel.READ for r in rules)

    def test_mro_ordering(self):
        class Base(Agent):
            file_access = [FileAccessRule("**", FileAccessLevel.READ)]

        class Child(Base):
            file_access = [FileAccessRule("src/**", FileAccessLevel.WRITE)]

        rules = Child._collect_file_access()
        patterns = [r.pattern for r in rules]
        assert patterns.index("**") < patterns.index("src/**")

    def test_instance_file_access(self):
        class CustomAgent(Agent):
            file_access = [FileAccessRule("**", FileAccessLevel.READ)]

            def _instance_file_access(self):
                return [FileAccessRule(f"src/{self.module}.cpp", FileAccessLevel.WRITE)]

        agent = CustomAgent(module="parser")
        instance_rules = agent._instance_file_access()
        assert len(instance_rules) == 1
        assert instance_rules[0].pattern == "src/parser.cpp"
        assert instance_rules[0].access == FileAccessLevel.WRITE


# ---------------------------------------------------------------------------
# Tool enforcement (integration)
# ---------------------------------------------------------------------------


class TestToolEnforcement:
    async def test_read_file_denied(self, tmp_path):
        from thorn.core._tools import read_file

        p = tmp_path / "secret.txt"
        p.write_text("secret data", encoding="utf-8")

        policy = FileAccessPolicy(
            [FileAccessRule("**", FileAccessLevel.NONE)],
        )
        ctx = ExecutionContext(
            provider=MockProvider(),
            workspace_root=tmp_path,
            file_access_policy=policy,
        )
        token = set_context(ctx)
        try:
            with pytest.raises(PermissionError, match="READ"):
                await read_file(str(p))
        finally:
            reset_context(token)

    async def test_write_file_denied(self, tmp_path):
        from thorn.core._tools import write_file

        policy = FileAccessPolicy(
            [FileAccessRule("**", FileAccessLevel.READ)],
        )
        ctx = ExecutionContext(
            provider=MockProvider(),
            workspace_root=tmp_path,
            file_access_policy=policy,
        )
        token = set_context(ctx)
        try:
            with pytest.raises(PermissionError, match="WRITE"):
                await write_file(str(tmp_path / "file.txt"), "data")
        finally:
            reset_context(token)

    async def test_write_file_allowed(self, tmp_path):
        from thorn.core._tools import write_file

        policy = FileAccessPolicy(
            [FileAccessRule("**", FileAccessLevel.WRITE)],
        )
        ctx = ExecutionContext(
            provider=MockProvider(),
            workspace_root=tmp_path,
            file_access_policy=policy,
        )
        token = set_context(ctx)
        try:
            result = await write_file(str(tmp_path / "file.txt"), "data")
            assert "4" in result
            assert (tmp_path / "file.txt").read_text(encoding="utf-8") == "data"
        finally:
            reset_context(token)

    async def test_edit_file_denied(self, tmp_path):
        from thorn.core._tools import FileEdit, edit_file

        p = tmp_path / "file.txt"
        p.write_text("hello", encoding="utf-8")

        policy = FileAccessPolicy(
            [FileAccessRule("**", FileAccessLevel.READ)],
        )
        ctx = ExecutionContext(
            provider=MockProvider(),
            workspace_root=tmp_path,
            file_access_policy=policy,
        )
        token = set_context(ctx)
        try:
            with pytest.raises(PermissionError, match="WRITE"):
                await edit_file(str(p), [
                    FileEdit(old_string="hello", new_string="bye"),
                ])
        finally:
            reset_context(token)

    async def test_edit_file_allowed(self, tmp_path):
        from thorn.core._tools import FileEdit, edit_file

        p = tmp_path / "file.txt"
        p.write_text("hello", encoding="utf-8")

        policy = FileAccessPolicy(
            [FileAccessRule("**", FileAccessLevel.WRITE)],
        )
        ctx = ExecutionContext(
            provider=MockProvider(),
            workspace_root=tmp_path,
            file_access_policy=policy,
        )
        token = set_context(ctx)
        try:
            result = await edit_file(str(p), [
                FileEdit(old_string="hello", new_string="bye"),
            ])
            assert "Applied" in result
            assert p.read_text(encoding="utf-8") == "bye"
        finally:
            reset_context(token)

    async def test_create_file_denied(self, tmp_path):
        from thorn.core._tools import create_file

        policy = FileAccessPolicy(
            [FileAccessRule("**", FileAccessLevel.READ)],
        )
        ctx = ExecutionContext(
            provider=MockProvider(),
            workspace_root=tmp_path,
            file_access_policy=policy,
        )
        token = set_context(ctx)
        try:
            with pytest.raises(PermissionError, match="WRITE"):
                await create_file(str(tmp_path / "new.txt"), "data")
        finally:
            reset_context(token)

    async def test_create_file_allowed(self, tmp_path):
        from thorn.core._tools import create_file

        policy = FileAccessPolicy(
            [FileAccessRule("**", FileAccessLevel.WRITE)],
        )
        ctx = ExecutionContext(
            provider=MockProvider(),
            workspace_root=tmp_path,
            file_access_policy=policy,
        )
        token = set_context(ctx)
        try:
            result = await create_file(str(tmp_path / "new.txt"), "data")
            assert "Created" in result
            assert (tmp_path / "new.txt").read_text(encoding="utf-8") == "data"
        finally:
            reset_context(token)

    async def test_list_directory_filters_hidden(self, tmp_path):
        from thorn.core._tools import list_directory

        (tmp_path / "visible.txt").touch()
        (tmp_path / ".thorn").mkdir()
        (tmp_path / ".thorn" / "tools.py").touch()

        policy = FileAccessPolicy([
            FileAccessRule("**", FileAccessLevel.READ),
            FileAccessRule(".thorn/**", FileAccessLevel.HIDDEN),
            FileAccessRule(".thorn", FileAccessLevel.HIDDEN),
        ])
        ctx = ExecutionContext(
            provider=MockProvider(),
            workspace_root=tmp_path,
            file_access_policy=policy,
        )
        token = set_context(ctx)
        try:
            result = await list_directory(str(tmp_path))
            assert "visible.txt" in result
            assert ".thorn" not in result
        finally:
            reset_context(token)

    async def test_no_policy_allows_all(self, tmp_path):
        """When no policy is set, tools should work without restriction."""
        from thorn.core._tools import read_file

        p = tmp_path / "open.txt"
        p.write_text("open data", encoding="utf-8")

        ctx = ExecutionContext(provider=MockProvider())
        token = set_context(ctx)
        try:
            result = await read_file(str(p))
            assert "open data" in result
        finally:
            reset_context(token)

    async def test_search_files_denied_single_file(self, tmp_path):
        from thorn.core._tools import search_files

        p = tmp_path / "secret.txt"
        p.write_text("needle in secret", encoding="utf-8")

        policy = FileAccessPolicy(
            [FileAccessRule("**", FileAccessLevel.NONE)],
        )
        ctx = ExecutionContext(
            provider=MockProvider(),
            workspace_root=tmp_path,
            file_access_policy=policy,
        )
        token = set_context(ctx)
        try:
            with pytest.raises(PermissionError, match="READ"):
                await search_files("needle", str(p))
        finally:
            reset_context(token)

    async def test_search_files_hidden_excluded(self, tmp_path):
        """HIDDEN files must not appear in directory search results."""
        from thorn.core._tools import search_files

        (tmp_path / "visible.txt").write_text("needle\n", encoding="utf-8")
        (tmp_path / "secret.txt").write_text("needle\n", encoding="utf-8")

        policy = FileAccessPolicy([
            FileAccessRule("**", FileAccessLevel.READ),
            FileAccessRule("secret.txt", FileAccessLevel.HIDDEN),
        ])
        ctx = ExecutionContext(
            provider=MockProvider(),
            workspace_root=tmp_path,
            file_access_policy=policy,
        )
        token = set_context(ctx)
        try:
            result = await search_files("needle", str(tmp_path))
            assert "visible.txt" in result
            assert "secret.txt" not in result
        finally:
            reset_context(token)

    async def test_search_files_none_excluded(self, tmp_path):
        """Files with NONE access must not leak content in search results."""
        from thorn.core._tools import search_files

        (tmp_path / "public.txt").write_text("needle\n", encoding="utf-8")
        (tmp_path / "private.txt").write_text("needle\n", encoding="utf-8")

        policy = FileAccessPolicy([
            FileAccessRule("**", FileAccessLevel.READ),
            FileAccessRule("private.txt", FileAccessLevel.NONE),
        ])
        ctx = ExecutionContext(
            provider=MockProvider(),
            workspace_root=tmp_path,
            file_access_policy=policy,
        )
        token = set_context(ctx)
        try:
            result = await search_files("needle", str(tmp_path))
            assert "public.txt" in result
            assert "private.txt" not in result
        finally:
            reset_context(token)


# ---------------------------------------------------------------------------
# Global ignore file loading
# ---------------------------------------------------------------------------


class TestGlobalIgnores:
    def test_load_ignore_file(self, tmp_path):
        ignore_file = tmp_path / ".aiignore"
        ignore_file.write_text(
            "# comment\n"
            "*.secret\n"
            "\n"
            ".env\n",
            encoding="utf-8",
        )
        rules = load_ignore_file(ignore_file)
        assert len(rules) == 2
        assert rules[0].pattern == "*.secret"
        assert rules[0].access == FileAccessLevel.HIDDEN
        assert rules[1].pattern == ".env"

    def test_load_ignore_file_missing(self, tmp_path):
        rules = load_ignore_file(tmp_path / "nonexistent")
        assert rules == []

    def test_load_global_ignores_both_files(self, tmp_path):
        (tmp_path / ".aiignore").write_text("*.key\n", encoding="utf-8")
        (tmp_path / ".thornignore").write_text("build/\n", encoding="utf-8")
        policy = load_global_ignores(tmp_path)
        assert policy is not None
        assert len(policy.rules) == 2

    def test_load_global_ignores_none_when_no_files(self, tmp_path):
        policy = load_global_ignores(tmp_path)
        assert policy is None

    def test_thornignore_takes_precedence(self, tmp_path):
        (tmp_path / ".aiignore").write_text("*.log\n", encoding="utf-8")
        (tmp_path / ".thornignore").write_text("*.log\n", encoding="utf-8")
        policy = load_global_ignores(tmp_path)
        assert policy is not None
        # Both should create HIDDEN rules; .thornignore rules come last
        assert policy.rules[-1].pattern == "*.log"


# ---------------------------------------------------------------------------
# run_shell not in ALL_BUILTIN_TOOLS
# ---------------------------------------------------------------------------


class TestRunShellRemoval:
    def test_not_in_all_builtin_tools(self):
        from thorn.core._tools import ALL_BUILTIN_TOOLS

        names = [getattr(t, "__name__", str(t)) for t in ALL_BUILTIN_TOOLS]
        assert "run_shell" not in names

    def test_still_importable(self):
        from thorn.core._tools import run_shell

        assert callable(run_shell)
