"""System-prompt and tool assembly: phase 3 of request projection.

Phase 1 (:mod:`thorn.runtime._context_paths`) produces an outer-to-inner
list of :class:`ContextDirectory`.  Phase 2
(:mod:`thorn.runtime._context_layers`) loads each directory into a
:class:`CollectedContext`.  Phase 3 -- this module -- folds those bundles
into the final shape the agent loop wants:

- A list of system-prompt fragments, in canonical block order, ready for the
  next provider-request snapshot.
- A deduplicated list of MCP server configs to spin up.
- A deduplicated list of agent skills (presented in the prompt and,
  eventually, also turned into tools).

Block ordering
--------------
The shipped pipeline (see ``docs/context-gathering.md``) emits the
following outer-to-inner block order.  Block kinds that have no
contributions are simply omitted; every block that is emitted carries
its own provenance header so the agent (and a human reading the
prompt) can see where each contribution came from.

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
- **Skills**: no dedup at this iteration.  Skill discovery accumulates
  every :class:`SkillEntry` produced by phase 2 in walk order; when
  two layers happen to advertise a same-named skill, both currently
  appear in the index block.  Revisit if/when the index becomes
  large enough that duplicate entries are confusing -- the cheapest
  policy is "outermost wins by ``name``", matching MCP.

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

from thorn.core._prompt_trace import (
    PromptTraceContextSource,
    PromptTraceManifest,
)
from thorn.runtime._context_layers import (
    CollectedContext,
    SkillEntry,
    TextContribution,
)

if TYPE_CHECKING:
    from thorn.core._mcp_config import MCPServerConfig


# ---------------------------------------------------------------------------
# Output type
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AssembledPromptContext:
    """The final shape of phase 3: blocks, MCP configs, and skills.

    Consumers (today: the session provider-request projector) splice
    *system_prompt_blocks* into the next request's system-prompt list, hand
    *mcp_configs* to
    :func:`thorn.runtime._mcp_tools.discover_mcp_tools` (which routes
    every list/call through the per-agent ``thorn-toolhost`` daemon),
    and use *skills* to render the skill-index block.  Skills are
    advertised, not auto-invoked: the agent reads the ``SKILL.md``
    body (via its own file-read tool) when it decides a skill is
    relevant.  No per-skill tool wrapper is generated.

    Each entry of *system_prompt_blocks* is a single fully-formatted
    string ready to ship -- no further wrapping or header insertion
    is expected of the caller.  The list may be empty if every layer
    was empty.
    """

    system_prompt_blocks: list[str]
    mcp_configs: list[MCPServerConfig]
    skills: list[SkillEntry]
    prompt_trace_manifest: PromptTraceManifest = field(
        default_factory=PromptTraceManifest,
    )


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


def _append_prompt_block(
    blocks: list[str],
    sources: list[PromptTraceContextSource],
    *,
    text: str,
    surface: str,
    label: str,
    source_path: Path | None = None,
    directory_kind: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Append a rendered prompt block and its trace provenance together."""
    blocks.append(text)
    sources.append(PromptTraceContextSource.from_text(
        surface=surface,
        label=label,
        text=text,
        source_path=source_path,
        directory_kind=directory_kind,
        metadata=metadata,
    ))


# ---------------------------------------------------------------------------
# Dedup helpers
# ---------------------------------------------------------------------------

def _mcp_config_dedup_key(config: MCPServerConfig) -> tuple[Any, ...]:
    """Return a hashable key that identifies *config* by full content.

    Thin alias around
    :func:`thorn.core._mcp_config.mcp_server_config_identity` so the
    brain (per-request dedup) and the daemon (per-process MCP cache)
    agree byte-for-byte on what counts as "the same" server.  Kept as
    a local name to preserve the existing call sites and to document
    that the dedup decision is intentionally identity-driven (every
    field, ``env`` included, contributes to the key).
    """
    from thorn.core._mcp_config import mcp_server_config_identity

    return mcp_server_config_identity(config)


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
    sources: list[PromptTraceContextSource] = []

    env_block = _format_environment_block(
        workspace_path=workspace_path,
        agent_home_path=agent_home_path,
    )
    if env_block is not None:
        _append_prompt_block(
            blocks,
            sources,
            text=env_block,
            surface="environment",
            label="runtime environment",
            metadata={
                "workspace_path": (
                    str(workspace_path) if workspace_path is not None else None
                ),
                "agent_home_path": (
                    str(agent_home_path) if agent_home_path is not None else None
                ),
            },
        )

    for layer in layers:
        if layer.agents_md is not None:
            text = _format_agents_md_block(layer.agents_md)
            _append_prompt_block(
                blocks,
                sources,
                text=text,
                surface="agents_md",
                label=f"Agent guidance from `{layer.agents_md.source_path}`",
                source_path=layer.agents_md.source_path,
                directory_kind=layer.directory.kind.value,
            )

    accumulated_skills: list[SkillEntry] = []
    for layer in layers:
        accumulated_skills.extend(layer.skills)

    skills_block = _format_skill_index_block(accumulated_skills)
    if skills_block is not None:
        _append_prompt_block(
            blocks,
            sources,
            text=skills_block,
            surface="skill_index",
            label="available skills index",
            metadata={
                "skill_count": len(accumulated_skills),
                "skill_paths": [
                    str(skill.skill_md_path) for skill in accumulated_skills
                ],
            },
        )

    for layer in layers:
        if layer.memory_md is not None:
            text = _format_memory_md_block(layer.memory_md)
            _append_prompt_block(
                blocks,
                sources,
                text=text,
                surface="memory_md",
                label=f"Agent memory from `{layer.memory_md.source_path}`",
                source_path=layer.memory_md.source_path,
                directory_kind=layer.directory.kind.value,
            )

    if journal_text:
        text = _format_journal_block(journal_text)
        _append_prompt_block(
            blocks,
            sources,
            text=text,
            surface="journal",
            label="recent activity log",
        )

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
        prompt_trace_manifest=PromptTraceManifest(
            system_prompt_sources=sources,
        ),
    )


__all__ = [
    "AssembledPromptContext",
    "assemble_prompt_context",
]
