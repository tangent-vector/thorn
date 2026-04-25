"""System-prompt and tool assembly: phase 3 of the per-prompt pipeline.

Phase 1 (:mod:`thorn.runtime._context_paths`) produces an outer-to-inner
list of :class:`ContextDirectory`.  Phase 2
(:mod:`thorn.runtime._context_layers`) loads each directory into a
:class:`CollectedContext`.  Phase 3 -- this module -- folds those bundles
into the final shape the agent loop wants:

- A list of system-prompt fragments, in canonical block order, ready to
  hand to ``run_agent_loop(system_prompts=...)``.
- A deduplicated list of MCP server configs to spin up.
- A deduplicated list of agent skills (presented in the prompt and,
  eventually, also turned into tools).

Block ordering
--------------
The aspirational design (``docs/aspirational/context-gathering.md``)
prescribes the following outer-to-inner order for the assembled
prompt.  Block kinds that have no contributions are simply omitted;
every block that is emitted carries its own provenance header so the
agent (and a human reading the prompt) can see where each contribution
came from.

1. Environment block (Thorn-runtime-injected facts about the session's
   filesystem layout: working directory, agent home).  Optional --
   omitted when the caller passes no environment paths.
2. ``AGENTS.md`` contributions, outer-to-inner.  Each contribution is
   one block with a short provenance header above the file's content.
3. Skill index block (one combined block listing every discovered
   skill with its description and the path to its ``SKILL.md``).
4. ``MEMORY.md`` contributions, outer-to-inner.  Same provenance
   treatment as ``AGENTS.md``.
5. Recent journal entries (one block, content already rendered by
   :func:`thorn.core._journal.read_recent_journal`).

The "super-global" Thorn-system prompts (item 1 in the doc) are *not*
emitted by this module: they are agent-provided
(``Agent._render_system_prompts``) and per-call
(``extra_system``), and they sit upstream of the assembled blocks in
the caller's final concatenation.  Persona / ``SOUL.md`` is similarly
not yet implemented; when it lands, it will slot in between the
super-global prompts and the AGENTS.md blocks.

Dedup policy
------------
- **MCP configs**: deduplicated by **content hash** across every
  layer.  Two ``.agents/mcp.json`` files that declare an identical
  server (same name, command, args, env, url) collapse to one entry,
  with the outer-most occurrence kept.  Configs that share a name but
  differ in any other field are *both* kept; resolving that conflict
  is left to the MCP layer (which today just registers both, and the
  consequences are the user's problem).
- **Skills**: no dedup at this iteration.  Skill discovery is still a
  stub (see ``skill_md_loader`` in the unified-context-gathering
  plan); when it lands, the dedup policy will be revisited alongside
  the discovery code so the two designs align.

This module is **pure** in the sense that it does not touch the
filesystem.  All I/O (loading ``AGENTS.md``, reading the journal,
parsing ``mcp.json``) happens in earlier phases or in the caller --
``assemble_prompt_context`` only re-arranges and formats values it is
handed.  That keeps the assembly trivially testable and makes the
boundary with phase 2 explicit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from thorn.runtime._context_layers import (
    CollectedContext,
    SkillEntry,
    TextContribution,
)

if TYPE_CHECKING:
    from thorn.core._mcp import MCPServerConfig


# ---------------------------------------------------------------------------
# Output type
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AssembledPromptContext:
    """The final shape of phase 3: blocks, MCP configs, and skills.

    Consumers (today: ``_run_session_prompt``) splice
    *system_prompt_blocks* into the system-prompt list they hand to
    :func:`thorn.core._loop.run_agent_loop`, register *mcp_configs*
    against the per-prompt :class:`MCPToolSource` (or equivalent),
    and (eventually) emit *skills* as tools.  Today the skill list is
    only used to render the skill-index block in the prompt; the
    "skills as tools" plumbing arrives with the SKILL.md loader.

    Each entry of *system_prompt_blocks* is a single fully-formatted
    string ready to ship -- no further wrapping or header insertion
    is expected of the caller.  The list may be empty if every layer
    was empty.
    """

    system_prompt_blocks: list[str]
    mcp_configs: list[MCPServerConfig]
    skills: list[SkillEntry]


# ---------------------------------------------------------------------------
# Block formatters
# ---------------------------------------------------------------------------

def _format_environment_block(
    *,
    workspace_path: Path | None,
    agent_home_path: Path | None,
) -> str | None:
    """Render the "Your environment" block, or ``None`` if both paths are missing.

    Mirrors the layout the legacy ``_run_session_prompt`` produced
    (``- Working directory (`.`): ...``) so existing prompt-aware
    agents see no surface-level change.  Lifted into its own function
    so a future shape revision -- e.g. adding ``HOME``, an inbox path,
    or a session-key reminder -- is a single-place change.
    """
    lines: list[str] = []
    if workspace_path is not None:
        lines.append(f"- Working directory (`.`): {workspace_path}")
    if agent_home_path is not None:
        lines.append(f"- Home directory (`~`): {agent_home_path}")
    if not lines:
        return None
    return "## Your environment\n\n" + "\n".join(lines)


def _format_agents_md_block(contribution: TextContribution) -> str:
    """Render one ``AGENTS.md`` contribution as a system-prompt block.

    The provenance header is brief so the contribution's own headings
    (which usually start at ``#``) still dominate the block visually,
    but explicit so multiple AGENTS.md files in a layered walk are
    distinguishable when the prompt is dumped for debugging.
    """
    return (
        f"# Agent guidance from `{contribution.source_path}`\n\n"
        f"{contribution.text}"
    )


def _format_memory_md_block(contribution: TextContribution) -> str:
    """Render one ``MEMORY.md`` contribution as a system-prompt block.

    Same provenance treatment as :func:`_format_agents_md_block`; the
    distinct header text helps the agent (and a reader) tell policy
    blocks apart from memory blocks at a glance.
    """
    return (
        f"# Agent memory from `{contribution.source_path}`\n\n"
        f"{contribution.text}"
    )


def _format_skill_index_block(skills: list[SkillEntry]) -> str | None:
    """Render the discovered skills as a single index block, or ``None`` if empty.

    Each skill is presented as a bullet with its name, one-line
    description, and the path to its ``SKILL.md``; the agent is
    expected to read the SKILL.md (with whatever read-file tool is
    available) before invoking the skill.  We deliberately do *not*
    inline the SKILL.md body here -- skills can be large, and the
    point of the index is to advertise their existence cheaply.
    """
    if not skills:
        return None
    lines = [
        "# Available skills",
        "",
        "The following skills are available.  Each has a `SKILL.md`",
        "with full instructions; read it before using the skill.",
        "",
    ]
    for skill in skills:
        lines.append(f"- **{skill.name}**: {skill.description}")
        lines.append(f"  Path: `{skill.skill_md_path}`")
    return "\n".join(lines)


def _format_journal_block(journal_text: str) -> str:
    """Wrap the journal text in its prompt-block header.

    The header text matches what the legacy ``_run_session_prompt``
    produced so behaviour is unchanged at the prompt's surface.
    """
    return (
        "# Recent activity log (shared across all sessions)\n\n"
        f"{journal_text}"
    )


# ---------------------------------------------------------------------------
# Dedup helpers
# ---------------------------------------------------------------------------

def _mcp_config_dedup_key(config: MCPServerConfig) -> tuple[Any, ...]:
    """Return a hashable key that identifies *config* by full content.

    Two configs with the same key are considered semantically
    identical and collapse to one entry in the assembled output.
    Differences in any field -- including ``name`` -- produce
    distinct keys, so two ``github`` servers with different commands
    still both appear in the output.

    The ``env`` dict, if present, is normalised to a sorted tuple of
    ``(key, value)`` pairs so dict iteration order is irrelevant to
    the dedup decision.
    """
    env_key: tuple[tuple[str, str], ...] | None
    env_key = (
        tuple(sorted(config.env.items()))
        if config.env is not None else None
    )
    return (
        config.name,
        config.command,
        tuple(config.args),
        env_key,
        config.url,
    )


def _dedup_mcp_configs_outer_first(
    configs: list[MCPServerConfig],
) -> list[MCPServerConfig]:
    """Drop configs whose content key already appeared in *configs*.

    Iteration is in input order (outer-first per phase 1's contract),
    so the surviving entry for any key is the **outer-most**
    occurrence.  This matches the convention used elsewhere in the
    pipeline ("outermost wins for duplicates").
    """
    seen: set[tuple[Any, ...]] = set()
    kept: list[MCPServerConfig] = []
    for config in configs:
        key = _mcp_config_dedup_key(config)
        if key in seen:
            continue
        seen.add(key)
        kept.append(config)
    return kept


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

def assemble_prompt_context(
    layers: list[CollectedContext],
    *,
    workspace_path: Path | None = None,
    agent_home_path: Path | None = None,
    journal_text: str | None = None,
) -> AssembledPromptContext:
    """Fold per-layer bundles into the final block list, MCP set, and skills.

    *layers* is the output of
    :func:`thorn.runtime._context_layers.load_context_layers`,
    already in outer-to-inner order.  Empty bundles are tolerated and
    silently produce no contributions.

    *workspace_path* and *agent_home_path*, when provided, drive the
    optional environment block at the head of the prompt.  Pass
    ``None`` (default) to suppress the block entirely -- callers that
    want to render their own environment header upstream can do so.

    *journal_text*, when provided, is rendered as the final block.
    The caller is responsible for the actual journal lookup
    (typically :func:`thorn.core._journal.read_recent_journal` against
    the agent's ``home/journal``); this function only owns block
    *placement*, not journal *loading*.  Passing an empty string is
    treated the same as ``None``.

    The returned :class:`AssembledPromptContext` is a value object;
    consumers should not mutate its lists in place.
    """
    blocks: list[str] = []

    env_block = _format_environment_block(
        workspace_path=workspace_path,
        agent_home_path=agent_home_path,
    )
    if env_block is not None:
        blocks.append(env_block)

    for layer in layers:
        if layer.agents_md is not None:
            blocks.append(_format_agents_md_block(layer.agents_md))

    accumulated_skills: list[SkillEntry] = []
    for layer in layers:
        accumulated_skills.extend(layer.skills)

    skills_block = _format_skill_index_block(accumulated_skills)
    if skills_block is not None:
        blocks.append(skills_block)

    for layer in layers:
        if layer.memory_md is not None:
            blocks.append(_format_memory_md_block(layer.memory_md))

    if journal_text:
        blocks.append(_format_journal_block(journal_text))

    accumulated_mcp_configs: list[MCPServerConfig] = []
    for layer in layers:
        accumulated_mcp_configs.extend(layer.mcp_configs)
    deduped_mcp_configs = _dedup_mcp_configs_outer_first(
        accumulated_mcp_configs,
    )

    return AssembledPromptContext(
        system_prompt_blocks=blocks,
        mcp_configs=deduped_mcp_configs,
        skills=accumulated_skills,
    )


__all__ = [
    "AssembledPromptContext",
    "assemble_prompt_context",
]
