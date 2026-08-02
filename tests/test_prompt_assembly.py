"""Tests for `thorn.runtime._prompt_assembly`.

These tests are deliberately filesystem-free: ``assemble_prompt_context``
is a pure function over already-loaded
:class:`~thorn.runtime._context_layers.CollectedContext` bundles, so
each case constructs synthetic bundles in-memory and asserts on the
shape of the resulting :class:`AssembledPromptContext`.

The exhaustive on-disk policy (kind filtering, file fallback, etc.)
is covered in ``test_context_layers.py``; here we focus on the
phase-3 contract: block ordering, header formatting, MCP content-hash
dedup, and skill accumulation.
"""

from __future__ import annotations

from pathlib import PurePosixPath

from thorn.core._mcp_config import MCPServerConfig
from thorn.runtime._context_layers import (
    CollectedContext,
    SkillEntry,
    TextContribution,
)
from thorn.runtime._context_paths import (
    ContextDirectory,
    ContextDirectoryKind,
)
from thorn.runtime._prompt_assembly import (
    AssembledPromptContext,
    assemble_prompt_context,
)


def _layer(
    path: str,
    kind: ContextDirectoryKind = ContextDirectoryKind.AGENT_WORKSPACE,
    *,
    agents_md: TextContribution | None = None,
    memory_md: TextContribution | None = None,
    mcp_configs: list[MCPServerConfig] | None = None,
    skills: list[SkillEntry] | None = None,
) -> CollectedContext:
    """Construct a synthetic ``CollectedContext`` for assembly tests."""
    return CollectedContext(
        directory=ContextDirectory(path=PurePosixPath(path), kind=kind),  # type: ignore[arg-type]
        agents_md=agents_md,
        memory_md=memory_md,
        mcp_configs=list(mcp_configs or []),
        skills=list(skills or []),
    )


# ---------------------------------------------------------------------------
# Empty / trivial cases
# ---------------------------------------------------------------------------

class TestEmpty:
    def test_no_layers_no_inputs_returns_empty(self) -> None:
        result = assemble_prompt_context([])
        assert isinstance(result, AssembledPromptContext)
        assert result.system_prompt_blocks == []
        assert result.mcp_configs == []
        assert result.skills == []

    def test_empty_bundles_produce_no_blocks(self) -> None:
        layers = [
            _layer("/a", ContextDirectoryKind.OPERATOR),
            _layer("/b", ContextDirectoryKind.AGENT_HOME),
            _layer("/c", ContextDirectoryKind.AGENT_WORKSPACE),
        ]
        result = assemble_prompt_context(layers)
        assert result.system_prompt_blocks == []
        assert result.mcp_configs == []
        assert result.skills == []


# ---------------------------------------------------------------------------
# Environment block
# ---------------------------------------------------------------------------

class TestEnvironmentBlock:
    def test_workspace_only(self) -> None:
        result = assemble_prompt_context(
            [], workspace_path=PurePosixPath("/work"),  # type: ignore[arg-type]
        )
        assert len(result.system_prompt_blocks) == 1
        block = result.system_prompt_blocks[0]
        assert "## Your environment" in block
        assert "Working directory" in block and "/work" in block
        assert "Home directory" not in block

    def test_home_only(self) -> None:
        result = assemble_prompt_context(
            [], agent_home_path=PurePosixPath("/home/agent"),  # type: ignore[arg-type]
        )
        block = result.system_prompt_blocks[0]
        assert "Home directory" in block and "/home/agent" in block
        assert "Working directory" not in block

    def test_both_paths_combined(self) -> None:
        result = assemble_prompt_context(
            [],
            workspace_path=PurePosixPath("/work"),  # type: ignore[arg-type]
            agent_home_path=PurePosixPath("/home"),  # type: ignore[arg-type]
        )
        block = result.system_prompt_blocks[0]
        assert "/work" in block and "/home" in block

    def test_neither_path_emits_no_block(self) -> None:
        result = assemble_prompt_context([])
        assert result.system_prompt_blocks == []


# ---------------------------------------------------------------------------
# AGENTS.md blocks
# ---------------------------------------------------------------------------

class TestAgentsMdBlocks:
    def test_single_layer(self) -> None:
        layers = [_layer(
            "/proj",
            agents_md=TextContribution(
                text="be polite",
                source_path=PurePosixPath("/proj/AGENTS.md"),  # type: ignore[arg-type]
            ),
        )]
        result = assemble_prompt_context(layers)
        assert len(result.system_prompt_blocks) == 1
        block = result.system_prompt_blocks[0]
        assert "Agent guidance from" in block
        assert "/proj/AGENTS.md" in block
        assert "be polite" in block

    def test_outer_to_inner_order_preserved(self) -> None:
        layers = [
            _layer(
                "/outer",
                agents_md=TextContribution(
                    text="OUTER",
                    source_path=PurePosixPath("/outer/AGENTS.md"),  # type: ignore[arg-type]
                ),
            ),
            _layer(
                "/inner",
                agents_md=TextContribution(
                    text="INNER",
                    source_path=PurePosixPath("/inner/AGENTS.md"),  # type: ignore[arg-type]
                ),
            ),
        ]
        result = assemble_prompt_context(layers)
        assert len(result.system_prompt_blocks) == 2
        assert "OUTER" in result.system_prompt_blocks[0]
        assert "INNER" in result.system_prompt_blocks[1]

    def test_layers_without_agents_md_skipped(self) -> None:
        layers = [
            _layer("/a"),
            _layer(
                "/b",
                agents_md=TextContribution(
                    text="hello",
                    source_path=PurePosixPath("/b/AGENTS.md"),  # type: ignore[arg-type]
                ),
            ),
            _layer("/c"),
        ]
        result = assemble_prompt_context(layers)
        assert len(result.system_prompt_blocks) == 1
        assert "hello" in result.system_prompt_blocks[0]


# ---------------------------------------------------------------------------
# Skill index block
# ---------------------------------------------------------------------------

class TestSkillIndexBlock:
    def test_skills_listed_with_paths(self) -> None:
        layers = [_layer(
            "/proj",
            skills=[
                SkillEntry(
                    name="babysit",
                    description="Keep PRs merge-ready",
                    skill_md_path=PurePosixPath(  # type: ignore[arg-type]
                        "/proj/.agents/skills/babysit/SKILL.md",
                    ),
                ),
            ],
        )]
        result = assemble_prompt_context(layers)
        assert len(result.system_prompt_blocks) == 1
        block = result.system_prompt_blocks[0]
        assert "# Available skills" in block
        assert "babysit" in block
        assert "Keep PRs merge-ready" in block
        assert "/proj/.agents/skills/babysit/SKILL.md" in block

    def test_no_skills_no_block(self) -> None:
        layers = [_layer("/proj", skills=[])]
        result = assemble_prompt_context(layers)
        assert result.system_prompt_blocks == []

    def test_skills_accumulated_outer_to_inner(self) -> None:
        layers = [
            _layer(
                "/outer",
                skills=[SkillEntry(
                    name="alpha",
                    description="A",
                    skill_md_path=PurePosixPath("/outer/.agents/skills/alpha/SKILL.md"),  # type: ignore[arg-type]
                )],
            ),
            _layer(
                "/inner",
                skills=[SkillEntry(
                    name="beta",
                    description="B",
                    skill_md_path=PurePosixPath("/inner/.agents/skills/beta/SKILL.md"),  # type: ignore[arg-type]
                )],
            ),
        ]
        result = assemble_prompt_context(layers)
        assert [s.name for s in result.skills] == ["alpha", "beta"]
        block = result.system_prompt_blocks[0]
        assert block.index("alpha") < block.index("beta")


# ---------------------------------------------------------------------------
# MEMORY.md blocks
# ---------------------------------------------------------------------------

class TestMemoryMdBlocks:
    def test_single_layer(self) -> None:
        layers = [_layer(
            "/home",
            kind=ContextDirectoryKind.AGENT_HOME,
            memory_md=TextContribution(
                text="remember things",
                source_path=PurePosixPath("/home/MEMORY.md"),  # type: ignore[arg-type]
            ),
        )]
        result = assemble_prompt_context(layers)
        assert len(result.system_prompt_blocks) == 1
        block = result.system_prompt_blocks[0]
        assert "Agent memory from" in block
        assert "/home/MEMORY.md" in block
        assert "remember things" in block

    def test_outer_to_inner_order_preserved(self) -> None:
        layers = [
            _layer(
                "/outer-home",
                kind=ContextDirectoryKind.AGENT_HOME,
                memory_md=TextContribution(
                    text="OUTER-MEM",
                    source_path=PurePosixPath("/outer-home/MEMORY.md"),  # type: ignore[arg-type]
                ),
            ),
            _layer(
                "/inner-home",
                kind=ContextDirectoryKind.AGENT_HOME,
                memory_md=TextContribution(
                    text="INNER-MEM",
                    source_path=PurePosixPath("/inner-home/MEMORY.md"),  # type: ignore[arg-type]
                ),
            ),
        ]
        result = assemble_prompt_context(layers)
        # Memory blocks come after the (here absent) skill block; their
        # internal order matches the layer order.
        memory_blocks = [b for b in result.system_prompt_blocks if "memory" in b]
        assert "OUTER-MEM" in memory_blocks[0]
        assert "INNER-MEM" in memory_blocks[1]


# ---------------------------------------------------------------------------
# Journal block
# ---------------------------------------------------------------------------

class TestJournalBlock:
    def test_journal_text_appended_when_provided(self) -> None:
        result = assemble_prompt_context([], journal_text="yesterday: hi")
        assert len(result.system_prompt_blocks) == 1
        block = result.system_prompt_blocks[0]
        assert "Recent activity log" in block
        assert "yesterday: hi" in block

    def test_empty_journal_text_omitted(self) -> None:
        result = assemble_prompt_context([], journal_text="")
        assert result.system_prompt_blocks == []

    def test_none_journal_text_omitted(self) -> None:
        result = assemble_prompt_context([], journal_text=None)
        assert result.system_prompt_blocks == []


# ---------------------------------------------------------------------------
# Block ordering
# ---------------------------------------------------------------------------

class TestBlockOrdering:
    def test_canonical_order_env_agents_skills_memory_journal(self) -> None:
        """Environment, then AGENTS.md, then skills, then MEMORY.md, then journal."""
        layers = [_layer(
            "/d",
            kind=ContextDirectoryKind.AGENT_HOME,
            agents_md=TextContribution(
                text="POLICY",
                source_path=PurePosixPath("/d/AGENTS.md"),  # type: ignore[arg-type]
            ),
            memory_md=TextContribution(
                text="MEMORY",
                source_path=PurePosixPath("/d/MEMORY.md"),  # type: ignore[arg-type]
            ),
            skills=[SkillEntry(
                name="s",
                description="d",
                skill_md_path=PurePosixPath("/d/.agents/skills/s/SKILL.md"),  # type: ignore[arg-type]
            )],
        )]
        result = assemble_prompt_context(
            layers,
            workspace_path=PurePosixPath("/work"),  # type: ignore[arg-type]
            journal_text="JOURNAL",
        )
        assert len(result.system_prompt_blocks) == 5
        env, agents, skills, memory, journal = result.system_prompt_blocks
        assert "Your environment" in env
        assert "POLICY" in agents
        assert "Available skills" in skills
        assert "MEMORY" in memory
        assert "Recent activity log" in journal

    def test_prompt_trace_manifest_tracks_block_provenance(self) -> None:
        layers = [_layer(
            "/d",
            kind=ContextDirectoryKind.AGENT_HOME,
            agents_md=TextContribution(
                text="POLICY",
                source_path=PurePosixPath("/d/AGENTS.md"),  # type: ignore[arg-type]
            ),
            memory_md=TextContribution(
                text="MEMORY",
                source_path=PurePosixPath("/d/MEMORY.md"),  # type: ignore[arg-type]
            ),
            skills=[SkillEntry(
                name="review",
                description="Review code",
                skill_md_path=PurePosixPath("/d/.agents/skills/review/SKILL.md"),  # type: ignore[arg-type]
            )],
        )]

        result = assemble_prompt_context(
            layers,
            workspace_path=PurePosixPath("/work"),  # type: ignore[arg-type]
            journal_text="JOURNAL",
        )

        sources = result.prompt_trace_manifest.system_prompt_sources
        assert [source.surface for source in sources] == [
            "environment",
            "agents_md",
            "skill_index",
            "memory_md",
            "journal",
        ]
        assert sources[1].source_path == "/d/AGENTS.md"
        assert sources[1].directory_kind == "agent_home"
        assert sources[2].metadata["skill_count"] == 1
        assert sources[3].source_path == "/d/MEMORY.md"


# ---------------------------------------------------------------------------
# MCP config dedup
# ---------------------------------------------------------------------------

class TestMcpConfigDedup:
    def test_identical_configs_collapse_to_one(self) -> None:
        cfg = MCPServerConfig(name="github", command="gh-mcp", args=["serve"])
        layers = [
            _layer(
                "/outer",
                kind=ContextDirectoryKind.AGENT_HOME,
                mcp_configs=[cfg],
            ),
            _layer(
                "/inner",
                mcp_configs=[
                    MCPServerConfig(name="github", command="gh-mcp", args=["serve"]),
                ],
            ),
        ]
        result = assemble_prompt_context(layers)
        assert len(result.mcp_configs) == 1
        assert result.mcp_configs[0] is cfg  # outer-most wins

    def test_same_name_different_command_both_kept(self) -> None:
        layers = [
            _layer(
                "/outer",
                kind=ContextDirectoryKind.AGENT_HOME,
                mcp_configs=[MCPServerConfig(name="github", command="orig")],
            ),
            _layer(
                "/inner",
                mcp_configs=[MCPServerConfig(name="github", command="fork")],
            ),
        ]
        result = assemble_prompt_context(layers)
        assert len(result.mcp_configs) == 2
        assert result.mcp_configs[0].command == "orig"
        assert result.mcp_configs[1].command == "fork"

    def test_distinct_servers_all_kept(self) -> None:
        layers = [_layer(
            "/d",
            kind=ContextDirectoryKind.AGENT_HOME,
            mcp_configs=[
                MCPServerConfig(name="a", command="A"),
                MCPServerConfig(name="b", command="B"),
                MCPServerConfig(name="c", url="https://c.example/mcp"),
            ],
        )]
        result = assemble_prompt_context(layers)
        assert [c.name for c in result.mcp_configs] == ["a", "b", "c"]

    def test_env_dict_order_does_not_break_dedup(self) -> None:
        layers = [
            _layer(
                "/outer",
                kind=ContextDirectoryKind.AGENT_HOME,
                mcp_configs=[MCPServerConfig(
                    name="x", command="cmd",
                    env={"A": "1", "B": "2"},
                )],
            ),
            _layer(
                "/inner",
                mcp_configs=[MCPServerConfig(
                    name="x", command="cmd",
                    env={"B": "2", "A": "1"},
                )],
            ),
        ]
        result = assemble_prompt_context(layers)
        assert len(result.mcp_configs) == 1

    def test_brain_dedup_key_matches_daemon_identity(self) -> None:
        """Brain-side dedup and daemon-side ``MCPHost`` cache must agree.

        The contract: any two configs the brain collapses to one
        entry must hash to the same key the daemon uses for its
        per-process MCP cache, otherwise the daemon would spin up
        two processes for what the brain treats as the same server.
        Both sides go through
        :func:`thorn.core._mcp_config.mcp_server_config_identity`,
        so this test is a regression guard against either side
        diverging.
        """
        from thorn.core._mcp_config import mcp_server_config_identity
        from thorn.runtime._prompt_assembly import _mcp_config_dedup_key

        equal_pairs = [
            (
                MCPServerConfig(name="x", command="cmd"),
                MCPServerConfig(name="x", command="cmd"),
            ),
            (
                MCPServerConfig(
                    name="x", command="cmd",
                    args=["a", "b"],
                    env={"A": "1", "B": "2"},
                ),
                MCPServerConfig(
                    name="x", command="cmd",
                    args=["a", "b"],
                    env={"B": "2", "A": "1"},  # different dict order
                ),
            ),
            (
                MCPServerConfig(name="docs", url="https://x/mcp"),
                MCPServerConfig(name="docs", url="https://x/mcp"),
            ),
        ]
        for left, right in equal_pairs:
            assert _mcp_config_dedup_key(left) == _mcp_config_dedup_key(right)
            assert (
                _mcp_config_dedup_key(left)
                == mcp_server_config_identity(right)
            )

        distinct_pairs = [
            # Different env value -> different identity.
            (
                MCPServerConfig(name="x", command="cmd", env={"K": "1"}),
                MCPServerConfig(name="x", command="cmd", env={"K": "2"}),
            ),
            # Different args -> different identity.
            (
                MCPServerConfig(name="x", command="cmd", args=["a"]),
                MCPServerConfig(name="x", command="cmd", args=["a", "b"]),
            ),
            # env=None vs env={} are different identities (the dedup
            # key reflects whether env was specified at all).
            (
                MCPServerConfig(name="x", command="cmd"),
                MCPServerConfig(name="x", command="cmd", env={}),
            ),
        ]
        for left, right in distinct_pairs:
            assert _mcp_config_dedup_key(left) != _mcp_config_dedup_key(right)
            assert (
                _mcp_config_dedup_key(left)
                != mcp_server_config_identity(right)
            )


# ---------------------------------------------------------------------------
# Skill accumulation surface
# ---------------------------------------------------------------------------

class TestSkillAccumulation:
    def test_skills_field_includes_all_from_all_layers(self) -> None:
        layers = [
            _layer(
                "/a",
                kind=ContextDirectoryKind.AGENT_HOME,
                skills=[SkillEntry(
                    name="s1", description="d1",
                    skill_md_path=PurePosixPath("/a/.agents/skills/s1/SKILL.md"),  # type: ignore[arg-type]
                )],
            ),
            _layer(
                "/b",
                skills=[SkillEntry(
                    name="s2", description="d2",
                    skill_md_path=PurePosixPath("/b/.agents/skills/s2/SKILL.md"),  # type: ignore[arg-type]
                )],
            ),
        ]
        result = assemble_prompt_context(layers)
        assert [s.name for s in result.skills] == ["s1", "s2"]

    def test_skills_with_duplicate_names_both_surface(self) -> None:
        """No-dedup policy at this iteration -- duplicates surface, see plan."""
        layers = [
            _layer(
                "/a",
                kind=ContextDirectoryKind.AGENT_HOME,
                skills=[SkillEntry(
                    name="dup", description="from a",
                    skill_md_path=PurePosixPath("/a/.agents/skills/dup/SKILL.md"),  # type: ignore[arg-type]
                )],
            ),
            _layer(
                "/b",
                skills=[SkillEntry(
                    name="dup", description="from b",
                    skill_md_path=PurePosixPath("/b/.agents/skills/dup/SKILL.md"),  # type: ignore[arg-type]
                )],
            ),
        ]
        result = assemble_prompt_context(layers)
        assert len(result.skills) == 2
