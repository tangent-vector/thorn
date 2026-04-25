"""Tests for `thorn.runtime._context_layers`.

Each per-category collector is exercised independently against
tmp-dir fixtures so a refinement to one collector's policy
(e.g. adding a new MEMORY alias) need only update that collector's
tests.  The composition (``collect_context_for_directory``) and
the map (``load_context_layers``) get their own coverage at the
end, focused on shape rather than on individual files.

The skill collector intentionally has minimal coverage here: full
SKILL.md discovery / YAML parsing tests land with the
``skill_md_loader`` plan item.  We just verify the kind-filter
behaviour so the existing plumbing won't regress when that work
arrives.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from thorn.runtime._context_layers import (
    CollectedContext,
    SkillEntry,
    TextContribution,
    collect_agents_md_contribution_for_directory,
    collect_context_for_directory,
    collect_mcp_configs_for_directory,
    collect_memory_md_contribution_for_directory,
    collect_skills_for_directory,
    load_context_layers,
)
from thorn.runtime._context_paths import (
    ContextDirectory,
    ContextDirectoryKind,
)


def _ctx(path: Path, kind: ContextDirectoryKind) -> ContextDirectory:
    return ContextDirectory(path=path, kind=kind)


# ---------------------------------------------------------------------------
# AGENTS.md collector
# ---------------------------------------------------------------------------

class TestCollectAgentsMd:
    def test_loads_agents_md_when_present(self, tmp_path: Path) -> None:
        (tmp_path / "AGENTS.md").write_text("policy text\n")
        result = collect_agents_md_contribution_for_directory(
            _ctx(tmp_path, ContextDirectoryKind.AGENT_HOME),
        )
        assert result == TextContribution(
            text="policy text\n", source_path=tmp_path / "AGENTS.md",
        )

    def test_falls_back_to_claude_md(self, tmp_path: Path) -> None:
        (tmp_path / "CLAUDE.md").write_text("claude-flavoured policy\n")
        result = collect_agents_md_contribution_for_directory(
            _ctx(tmp_path, ContextDirectoryKind.AGENT_WORKSPACE),
        )
        assert result is not None
        assert result.text == "claude-flavoured policy\n"
        assert result.source_path == tmp_path / "CLAUDE.md"

    def test_agents_md_wins_over_claude_md(self, tmp_path: Path) -> None:
        (tmp_path / "AGENTS.md").write_text("primary\n")
        (tmp_path / "CLAUDE.md").write_text("fallback\n")
        result = collect_agents_md_contribution_for_directory(
            _ctx(tmp_path, ContextDirectoryKind.AGENT_HOME),
        )
        assert result is not None
        assert result.text == "primary\n"
        assert result.source_path == tmp_path / "AGENTS.md"

    def test_returns_none_when_neither_present(self, tmp_path: Path) -> None:
        result = collect_agents_md_contribution_for_directory(
            _ctx(tmp_path, ContextDirectoryKind.AGENT_WORKSPACE),
        )
        assert result is None

    def test_does_not_borrow_alias_from_outer_directory(
        self, tmp_path: Path,
    ) -> None:
        # Regression guard for the per-directory fallback rule: the
        # CLAUDE.md alias must only fill the AGENTS.md slot for the
        # *same* directory it lives in, never some outer one.
        outer = tmp_path / "outer"
        outer.mkdir()
        inner = outer / "inner"
        inner.mkdir()
        (outer / "CLAUDE.md").write_text("outer claude\n")
        # ``inner`` has neither file; the collector must return None
        # rather than reaching up to the outer directory.
        assert (
            collect_agents_md_contribution_for_directory(
                _ctx(inner, ContextDirectoryKind.AGENT_WORKSPACE),
            )
            is None
        )

    def test_all_kinds_participate(self, tmp_path: Path) -> None:
        # AGENTS.md is the one category that loads from every kind.
        (tmp_path / "AGENTS.md").write_text("everywhere\n")
        for kind in (
            ContextDirectoryKind.OPERATOR,
            ContextDirectoryKind.AGENT_HOME,
            ContextDirectoryKind.AGENT_WORKSPACE,
        ):
            result = collect_agents_md_contribution_for_directory(
                _ctx(tmp_path, kind),
            )
            assert result is not None
            assert result.text == "everywhere\n"


# ---------------------------------------------------------------------------
# MEMORY.md collector
# ---------------------------------------------------------------------------

class TestCollectMemoryMd:
    def test_loads_from_agent_home_kind(self, tmp_path: Path) -> None:
        (tmp_path / "MEMORY.md").write_text("private notes\n")
        result = collect_memory_md_contribution_for_directory(
            _ctx(tmp_path, ContextDirectoryKind.AGENT_HOME),
        )
        assert result == TextContribution(
            text="private notes\n", source_path=tmp_path / "MEMORY.md",
        )

    def test_excluded_from_workspace(self, tmp_path: Path) -> None:
        (tmp_path / "MEMORY.md").write_text("would be loaded\n")
        result = collect_memory_md_contribution_for_directory(
            _ctx(tmp_path, ContextDirectoryKind.AGENT_WORKSPACE),
        )
        assert result is None

    def test_excluded_from_operator(self, tmp_path: Path) -> None:
        (tmp_path / "MEMORY.md").write_text("would be loaded\n")
        result = collect_memory_md_contribution_for_directory(
            _ctx(tmp_path, ContextDirectoryKind.OPERATOR),
        )
        assert result is None

    def test_returns_none_when_file_missing(self, tmp_path: Path) -> None:
        result = collect_memory_md_contribution_for_directory(
            _ctx(tmp_path, ContextDirectoryKind.AGENT_HOME),
        )
        assert result is None


# ---------------------------------------------------------------------------
# MCP collector
# ---------------------------------------------------------------------------

@pytest.fixture
def _agents_dir(tmp_path: Path) -> Path:
    d = tmp_path / ".agents"
    d.mkdir()
    return d


def _write_mcp_json(agents_dir: Path, payload: dict) -> Path:
    p = agents_dir / "mcp.json"
    p.write_text(json.dumps(payload))
    return p


class TestCollectMcpConfigs:
    def test_loads_stdio_server(
        self, tmp_path: Path, _agents_dir: Path,
    ) -> None:
        _write_mcp_json(_agents_dir, {
            "mcpServers": {
                "fs": {
                    "command": "fs-server",
                    "args": ["--root", "/data"],
                },
            },
        })
        result = collect_mcp_configs_for_directory(
            _ctx(tmp_path, ContextDirectoryKind.AGENT_WORKSPACE),
        )
        assert len(result) == 1
        assert result[0].name == "fs"
        assert result[0].command == "fs-server"
        assert result[0].args == ["--root", "/data"]

    def test_loads_http_server(
        self, tmp_path: Path, _agents_dir: Path,
    ) -> None:
        _write_mcp_json(_agents_dir, {
            "mcpServers": {
                "remote": {"url": "https://mcp.example.com"},
            },
        })
        result = collect_mcp_configs_for_directory(
            _ctx(tmp_path, ContextDirectoryKind.AGENT_HOME),
        )
        assert len(result) == 1
        assert result[0].name == "remote"
        assert result[0].url == "https://mcp.example.com"

    def test_excluded_from_operator(
        self, tmp_path: Path, _agents_dir: Path,
    ) -> None:
        _write_mcp_json(_agents_dir, {
            "mcpServers": {"fs": {"command": "fs-server"}},
        })
        result = collect_mcp_configs_for_directory(
            _ctx(tmp_path, ContextDirectoryKind.OPERATOR),
        )
        assert result == []

    def test_no_mcp_json_returns_empty(self, tmp_path: Path) -> None:
        # No .agents/ at all -- harmless empty result.
        result = collect_mcp_configs_for_directory(
            _ctx(tmp_path, ContextDirectoryKind.AGENT_WORKSPACE),
        )
        assert result == []

    def test_env_var_expansion(
        self,
        tmp_path: Path,
        _agents_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("MY_TOKEN", "secret-value")
        _write_mcp_json(_agents_dir, {
            "mcpServers": {
                "github": {
                    "command": "gh-mcp",
                    "env": {"GITHUB_TOKEN": "$MY_TOKEN"},
                },
            },
        })
        result = collect_mcp_configs_for_directory(
            _ctx(tmp_path, ContextDirectoryKind.AGENT_WORKSPACE),
        )
        assert len(result) == 1
        assert result[0].env == {"GITHUB_TOKEN": "secret-value"}

    def test_missing_env_var_skips_one_server(
        self,
        tmp_path: Path,
        _agents_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # ``THIS_VAR_IS_NOT_SET`` is not in the environment; that
        # server should be skipped.  The sibling that resolves
        # cleanly should still be returned.
        monkeypatch.delenv("THIS_VAR_IS_NOT_SET", raising=False)
        _write_mcp_json(_agents_dir, {
            "mcpServers": {
                "needs_env": {
                    "command": "x",
                    "env": {"K": "$THIS_VAR_IS_NOT_SET"},
                },
                "ok": {"command": "y"},
            },
        })
        result = collect_mcp_configs_for_directory(
            _ctx(tmp_path, ContextDirectoryKind.AGENT_WORKSPACE),
        )
        assert [c.name for c in result] == ["ok"]

    def test_invalid_server_config_is_skipped(
        self, tmp_path: Path, _agents_dir: Path,
    ) -> None:
        # Server with neither ``command`` nor ``url`` fails
        # MCPServerConfig validation; collector should skip it,
        # not raise.
        _write_mcp_json(_agents_dir, {
            "mcpServers": {
                "broken": {"args": ["nope"]},
                "ok": {"command": "y"},
            },
        })
        result = collect_mcp_configs_for_directory(
            _ctx(tmp_path, ContextDirectoryKind.AGENT_WORKSPACE),
        )
        assert [c.name for c in result] == ["ok"]

    def test_malformed_json_returns_empty(
        self, tmp_path: Path, _agents_dir: Path,
    ) -> None:
        (_agents_dir / "mcp.json").write_text("{ not valid json")
        result = collect_mcp_configs_for_directory(
            _ctx(tmp_path, ContextDirectoryKind.AGENT_WORKSPACE),
        )
        assert result == []

    def test_missing_mcp_servers_key_returns_empty(
        self, tmp_path: Path, _agents_dir: Path,
    ) -> None:
        _write_mcp_json(_agents_dir, {"unrelated": {}})
        result = collect_mcp_configs_for_directory(
            _ctx(tmp_path, ContextDirectoryKind.AGENT_WORKSPACE),
        )
        assert result == []


# ---------------------------------------------------------------------------
# Skills collector
# ---------------------------------------------------------------------------

class TestCollectSkills:
    def test_returns_empty_for_now(self, tmp_path: Path) -> None:
        # The discovery+parse implementation lands with the
        # ``skill_md_loader`` plan item.  Until then the collector
        # is a deliberate stub; this test is a placeholder that the
        # later work will replace.
        skills = tmp_path / ".agents" / "skills" / "babysit"
        skills.mkdir(parents=True)
        (skills / "SKILL.md").write_text(
            "---\ndescription: keep PRs merge-ready\n---\nbody\n",
        )
        result = collect_skills_for_directory(
            _ctx(tmp_path, ContextDirectoryKind.AGENT_WORKSPACE),
        )
        assert result == []

    def test_excluded_from_operator(self, tmp_path: Path) -> None:
        result = collect_skills_for_directory(
            _ctx(tmp_path, ContextDirectoryKind.OPERATOR),
        )
        assert result == []


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------

class TestCollectContextForDirectory:
    def test_populates_every_field_when_files_present(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        (tmp_path / "AGENTS.md").write_text("policy\n")
        (tmp_path / "MEMORY.md").write_text("memory\n")
        agents_dir = tmp_path / ".agents"
        agents_dir.mkdir()
        _write_mcp_json(agents_dir, {
            "mcpServers": {"fs": {"command": "fs-server"}},
        })
        bundle = collect_context_for_directory(
            _ctx(tmp_path, ContextDirectoryKind.AGENT_HOME),
        )
        assert bundle.directory.path == tmp_path
        assert bundle.agents_md is not None
        assert bundle.agents_md.text == "policy\n"
        assert bundle.memory_md is not None
        assert bundle.memory_md.text == "memory\n"
        assert [c.name for c in bundle.mcp_configs] == ["fs"]
        assert bundle.skills == []

    def test_workspace_kind_excludes_memory(self, tmp_path: Path) -> None:
        (tmp_path / "AGENTS.md").write_text("p\n")
        (tmp_path / "MEMORY.md").write_text("m\n")
        bundle = collect_context_for_directory(
            _ctx(tmp_path, ContextDirectoryKind.AGENT_WORKSPACE),
        )
        assert bundle.agents_md is not None
        assert bundle.memory_md is None

    def test_operator_kind_keeps_only_agents_md(
        self, tmp_path: Path,
    ) -> None:
        (tmp_path / "AGENTS.md").write_text("p\n")
        (tmp_path / "MEMORY.md").write_text("m\n")
        agents_dir = tmp_path / ".agents"
        agents_dir.mkdir()
        _write_mcp_json(agents_dir, {
            "mcpServers": {"fs": {"command": "fs-server"}},
        })
        bundle = collect_context_for_directory(
            _ctx(tmp_path, ContextDirectoryKind.OPERATOR),
        )
        assert bundle.agents_md is not None
        assert bundle.memory_md is None
        assert bundle.mcp_configs == []
        assert bundle.skills == []

    def test_empty_directory_yields_empty_bundle(
        self, tmp_path: Path,
    ) -> None:
        bundle = collect_context_for_directory(
            _ctx(tmp_path, ContextDirectoryKind.AGENT_HOME),
        )
        assert bundle == CollectedContext(
            directory=_ctx(tmp_path, ContextDirectoryKind.AGENT_HOME),
        )


# ---------------------------------------------------------------------------
# load_context_layers (the map)
# ---------------------------------------------------------------------------

class TestLoadContextLayers:
    def test_one_bundle_per_input_in_order(self, tmp_path: Path) -> None:
        a = tmp_path / "a"
        b = tmp_path / "b"
        c = tmp_path / "c"
        for d in (a, b, c):
            d.mkdir()
        (a / "AGENTS.md").write_text("a-policy\n")
        (c / "AGENTS.md").write_text("c-policy\n")
        # ``b`` has nothing; its bundle should be empty but PRESENT
        # in the result, preserving the outer-to-inner order.
        result = load_context_layers([
            _ctx(a, ContextDirectoryKind.OPERATOR),
            _ctx(b, ContextDirectoryKind.AGENT_HOME),
            _ctx(c, ContextDirectoryKind.AGENT_WORKSPACE),
        ])
        assert [bundle.directory.path for bundle in result] == [a, b, c]
        assert [
            bundle.directory.kind for bundle in result
        ] == [
            ContextDirectoryKind.OPERATOR,
            ContextDirectoryKind.AGENT_HOME,
            ContextDirectoryKind.AGENT_WORKSPACE,
        ]
        assert result[0].agents_md is not None
        assert result[0].agents_md.text == "a-policy\n"
        assert result[1].agents_md is None
        assert result[2].agents_md is not None
        assert result[2].agents_md.text == "c-policy\n"

    def test_empty_input_yields_empty_output(self) -> None:
        assert load_context_layers([]) == []


# ---------------------------------------------------------------------------
# I/O resilience
# ---------------------------------------------------------------------------

class TestIoResilience:
    def test_unreadable_agents_md_returns_none_without_raising(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        agents_md = tmp_path / "AGENTS.md"
        agents_md.write_text("text\n")

        def _fail(self: Path, *args: object, **kwargs: object) -> str:
            raise OSError("simulated I/O failure")

        monkeypatch.setattr(Path, "read_text", _fail)
        # Must not raise.
        result = collect_agents_md_contribution_for_directory(
            _ctx(tmp_path, ContextDirectoryKind.AGENT_HOME),
        )
        assert result is None


# ---------------------------------------------------------------------------
# SkillEntry shape (defensive: shape locked in for the next plan item)
# ---------------------------------------------------------------------------

class TestSkillEntryShape:
    def test_constructible_with_required_fields(
        self, tmp_path: Path,
    ) -> None:
        entry = SkillEntry(
            name="babysit",
            description="keep PRs merge-ready",
            skill_md_path=tmp_path / "SKILL.md",
        )
        assert entry.name == "babysit"
        assert entry.description == "keep PRs merge-ready"
        assert entry.skill_md_path == tmp_path / "SKILL.md"
