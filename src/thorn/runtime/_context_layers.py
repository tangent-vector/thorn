"""Per-directory context loading: phase 2 of the per-prompt pipeline.

Phase 1 (:mod:`thorn.runtime._context_paths`) produces the ordered
list of :class:`ContextDirectory` entries to consider.  Phase 2 walks
that list and, for each entry, loads the per-category contributions
(agent policy, agent memory, MCP server configs, agent skills) into a
:class:`CollectedContext` bundle.  Phase 3 (:mod:`_prompt_assembly`,
forthcoming) consumes the resulting list of bundles to assemble the
final system prompt and tool list.

Architecture
------------
The module is organised **kind-of-content first**, not
kind-of-directory first.  ``collect_context_for_directory`` is a
flat composition: one call per per-category attribute on
:class:`CollectedContext`.  Each per-category collector
(``collect_agents_md_contribution_for_directory``, etc.) owns its
own kind-filtering and fallback policy and is responsible for
returning ``None``/empty when the directory's kind disqualifies it.

The inversion matters because the policy questions in this codebase
historically split along category lines, not directory lines:

- "Should ``MEMORY.md`` ever come from the workspace?"  -- that is
  a memory-collector question.
- "Should AGENTS.md ever fall back to CLAUDE.md, and only in the
  same directory?" -- that is an agents-md-collector question.
- "Which MCP config sources can safely reference host state?" -- that
  is an MCP-collector question.

Adding a new category, or refining the policy of an existing one,
is therefore a single-place change in the relevant collector.
``collect_context_for_directory`` itself only needs editing when the
*shape* of :class:`CollectedContext` changes.

I/O policy
----------
All collectors are best-effort: missing files return ``None`` /
empty list silently; unreadable files log a warning and return the
same.  The pipeline never raises for filesystem reasons, because
it runs on every prompt and any noisy failure mode would tank
session throughput.

Per-kind file policy (current iteration)
----------------------------------------
========================  =========  ============  ================
Category                  OPERATOR   AGENT_HOME    AGENT_WORKSPACE
========================  =========  ============  ================
``AGENTS.md`` (+CLAUDE)   yes        yes           yes
``MEMORY.md``             no         yes           no
MCP configs               no         yes           yes
Skills                    no         yes           yes
========================  =========  ============  ================

Rationale: the operator dir exists to inject **must-include
policy**, not to ship MCP servers or memory contributions on the
operator's behalf.  ``MEMORY.md`` is agent-private state and does
not belong under workspace trees the operator may share across
agents.  MCP and skills are per-project tooling, expected to live
under whichever directory the project's tools live in.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from thorn.runtime._context_paths import (
    ContextDirectory,
    ContextDirectoryKind,
)
from thorn.runtime._skill_md import (
    SkillMdError,
    parse_skill_md,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TextContribution:
    """A single piece of textual context loaded from disk.

    Carries the loaded text alongside the file it came from so the
    assembler in phase 3 can render a useful provenance header
    (e.g. "from ``<path>``") above each block.  The collector that
    produced this contribution is responsible for the actual
    ``read_text`` call; consumers should treat *text* as already
    sanitised (UTF-8 decoded, trailing whitespace untouched).
    """

    text: str
    source_path: Path


@dataclass(frozen=True)
class SkillEntry:
    """A discovered agent skill from ``<dir>/.agents/skills/<name>/SKILL.md``.

    The *name* is the immediate skill directory's basename (the
    "babysit" in ``.agents/skills/babysit/``).  *description* is
    the one-line summary lifted from the SKILL.md YAML frontmatter
    (see :mod:`thorn.runtime._skill_md` for the parser contract).
    *skill_md_path* is the absolute path the agent uses to actually
    read the skill body via the ``Read`` tool.

    The dataclass is intentionally minimal -- additional metadata
    fields (allowed-tools, prerequisites, model preferences) can be
    added as the SKILL.md format grows, without disturbing the
    pipeline's outer shape.
    """

    name: str
    description: str
    skill_md_path: Path


@dataclass(frozen=True)
class CollectedContext:
    """All context loaded from a single :class:`ContextDirectory`.

    One instance per directory in the phase-1 output, regardless of
    whether anything was actually found.  Empty bundles are kept
    so downstream code can preserve outer-to-inner ordering and
    diagnose "why didn't this show up?" by inspecting the bundle
    list directly.

    Each per-category field is independently populated by its own
    collector function.  The *directory* attribute is carried
    through so phase 3 can use the kind tag for block ordering and
    headers without re-deriving it.
    """

    directory: ContextDirectory
    agents_md: TextContribution | None = None
    memory_md: TextContribution | None = None
    mcp_configs: list[Any] = field(default_factory=list)
    skills: list[SkillEntry] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _read_text_or_warn(path: Path) -> str | None:
    """Return ``path.read_text(...)`` or ``None`` on any I/O failure.

    Logs a warning so unreadable files don't disappear silently, but
    never raises -- the pipeline runs on every prompt and the cost
    of bubbling an OS error here would be a wedged session.
    """
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        logger.warning("failed to read %s", path, exc_info=True)
        return None


def _load_first_existing_text(
    directory: Path,
    candidate_filenames: tuple[str, ...],
) -> TextContribution | None:
    """Return the first readable file from *candidate_filenames* in *directory*.

    Used to implement the per-directory ``AGENTS.md`` -> ``CLAUDE.md``
    fallback (and any future extension to the candidate list).  The
    fallback is *intra-directory only*: if a directory has neither
    file, the next outer directory is consulted independently --
    we do not "borrow" CLAUDE.md from one directory for an
    AGENTS.md slot in another.
    """
    for name in candidate_filenames:
        candidate = directory / name
        if candidate.is_file():
            text = _read_text_or_warn(candidate)
            if text is not None:
                return TextContribution(
                    text=text, source_path=candidate,
                )
    return None


# ---------------------------------------------------------------------------
# AGENTS.md / CLAUDE.md collector
# ---------------------------------------------------------------------------

# Filenames considered for the agent-policy slot, in priority order.
# Per-directory only: a directory must supply *its own* AGENTS.md
# (or fallback alias) to contribute, never borrow from a sibling.
_AGENTS_MD_CANDIDATE_FILENAMES: tuple[str, ...] = (
    "AGENTS.md",
    "CLAUDE.md",
)


def collect_agents_md_contribution_for_directory(
    directory: ContextDirectory,
) -> TextContribution | None:
    """Load the agent-policy contribution from *directory*.

    All three :class:`ContextDirectoryKind` values participate.
    Within the directory, ``AGENTS.md`` wins over ``CLAUDE.md`` if
    both exist (CLAUDE.md is a fallback alias only, per the
    aspirational doc).  No cross-tool aliasing (``.cursor/rules/``
    etc.) at this iteration -- see the follow-ups doc for that.
    """
    return _load_first_existing_text(
        directory.path, _AGENTS_MD_CANDIDATE_FILENAMES,
    )


# ---------------------------------------------------------------------------
# MEMORY.md collector
# ---------------------------------------------------------------------------

# Kinds that MEMORY.md is allowed to come from.  Workspace dirs are
# excluded per the design doc's "only load MEMORY.md files from
# locations under the agent's home directory".  Operator dirs are
# excluded because they exist for must-include *policy*, not for
# the agent's private state.
_MEMORY_MD_ALLOWED_KINDS: frozenset[ContextDirectoryKind] = frozenset({
    ContextDirectoryKind.AGENT_HOME,
})


def collect_memory_md_contribution_for_directory(
    directory: ContextDirectory,
) -> TextContribution | None:
    """Load ``MEMORY.md`` from *directory*, if its kind permits.

    Only ``AGENT_HOME`` directories contribute memory; ``OPERATOR``
    and ``AGENT_WORKSPACE`` directories early-out.  ``MEMORY.md``
    has no aliases at this iteration: a missing file just returns
    ``None``.  Adding a ``CLAUDE_MEMORY.md`` (or similar) alias is
    a single-list extension here.
    """
    if directory.kind not in _MEMORY_MD_ALLOWED_KINDS:
        return None
    return _load_first_existing_text(directory.path, ("MEMORY.md",))


# ---------------------------------------------------------------------------
# MCP config collector
# ---------------------------------------------------------------------------

# Kinds that contribute MCP server configs.  Excludes OPERATOR for
# the same reason as MEMORY.md: the operator dir is for must-include
# policy, not arbitrary tool registration.  AGENT_HOME and
# AGENT_WORKSPACE both contribute -- a project's ``.agents/mcp.json``
# in the workspace is the existing convention; an agent-home
# ``.agents/mcp.json`` is the new path for agent-private MCP tools.
#
# These directories are sandbox-writable.  The host must therefore
# treat their MCP configs as untrusted data and must not resolve
# strings such as ``$GITHUB_TOKEN`` against host process environment.
_MCP_ALLOWED_KINDS: frozenset[ContextDirectoryKind] = frozenset({
    ContextDirectoryKind.AGENT_HOME,
    ContextDirectoryKind.AGENT_WORKSPACE,
})

# Subdirectory under each context directory where MCP configs live.
# Matches Claude Desktop / Cursor convention.
_MCP_CONFIG_REL_PATH: Path = Path(".agents") / "mcp.json"


def collect_mcp_configs_for_directory(
    directory: ContextDirectory,
) -> list[Any]:
    """Load MCP server configs from ``<dir>/.agents/mcp.json``.

    Returns a list of :class:`thorn.core._mcp_config.MCPServerConfig`
    instances (typed loosely as ``list[Any]`` because the dataclass
    lives in a sibling module that this one does not import at
    module scope; see the lazy import in the body).

    Per-server behaviour:

    - Strings such as ``$GITHUB_TOKEN`` are preserved literally.
      MCP config files loaded here come from sandbox-writable
      directories, so resolving them against the host environment
      would cross the credential-isolation boundary.
    - Servers that fail validation (missing both ``command`` and
      ``url``, etc.) are **skipped** (logged at WARNING).

    Excludes ``OPERATOR`` directories.  No-op when ``.agents/mcp.json``
    does not exist or is unreadable.
    """
    if directory.kind not in _MCP_ALLOWED_KINDS:
        return []

    mcp_json = directory.path / _MCP_CONFIG_REL_PATH
    if not mcp_json.is_file():
        return []

    raw_text = _read_text_or_warn(mcp_json)
    if raw_text is None:
        return []

    try:
        raw = json.loads(raw_text)
    except json.JSONDecodeError:
        logger.warning("failed to parse %s as JSON", mcp_json, exc_info=True)
        return []

    servers = raw.get("mcpServers") if isinstance(raw, dict) else None
    if not isinstance(servers, dict):
        logger.warning("'mcpServers' in %s is missing or not a mapping", mcp_json)
        return []

    # Lazy import keeps this module's import graph small for callers
    # that walk directories without ever finding an ``mcp.json`` to
    # parse.  ``_mcp_config`` itself has no MCP-package dependency,
    # so the laziness is a cosmetic preference, not a correctness
    # requirement.
    from thorn.core._mcp_config import MCPServerConfig

    configs: list[MCPServerConfig] = []
    for name, spec in servers.items():
        if not isinstance(spec, dict):
            logger.warning(
                "MCP server %r in %s is not a mapping; skipping",
                name, mcp_json,
            )
            continue

        try:
            configs.append(MCPServerConfig(
                name=name,
                command=spec.get("command"),
                args=spec.get("args", []),
                env=spec.get("env"),
                url=spec.get("url"),
            ))
        except (ValueError, TypeError) as exc:
            logger.warning(
                "invalid MCP server config %r in %s: %s",
                name, mcp_json, exc,
            )

    return configs


# ---------------------------------------------------------------------------
# Skills collector
# ---------------------------------------------------------------------------

# Kinds that contribute skills.  Same rationale as MCP: skills are
# per-project (or per-agent-home) tooling, not operator policy.
_SKILLS_ALLOWED_KINDS: frozenset[ContextDirectoryKind] = frozenset({
    ContextDirectoryKind.AGENT_HOME,
    ContextDirectoryKind.AGENT_WORKSPACE,
})

# Subdirectory under each context directory where skills live.
_SKILLS_REL_PATH: Path = Path(".agents") / "skills"


def collect_skills_for_directory(
    directory: ContextDirectory,
) -> list[SkillEntry]:
    """Discover agent skills under ``<dir>/.agents/skills/``.

    Each immediate subdirectory containing a ``SKILL.md`` file is
    one skill; the entry's *name* is the subdirectory basename, the
    *description* is sourced from the SKILL.md YAML frontmatter
    (see :mod:`thorn.runtime._skill_md` for the parser contract),
    and *skill_md_path* points at the SKILL.md file itself.

    Excludes ``OPERATOR`` directories.

    Per-skill failure modes (no SKILL.md, parse error, missing
    description) all log a warning and skip the offending skill;
    they never tank the rest of the walk.  This matches the broader
    "best-effort context loading" policy of the pipeline -- a typo
    in one SKILL.md should not erase every skill the agent has.

    Returned entries are ordered by skill name (ASCII-sort) so the
    skill-index block in the assembled prompt is stable across runs
    and across filesystems with different ``readdir`` ordering.
    """
    if directory.kind not in _SKILLS_ALLOWED_KINDS:
        return []

    skills_root = directory.path / _SKILLS_REL_PATH
    if not skills_root.is_dir():
        return []

    entries: list[SkillEntry] = []
    for skill_dir in sorted(
        skills_root.iterdir(), key=lambda p: p.name,
    ):
        if not skill_dir.is_dir():
            continue
        skill_md = skill_dir / "SKILL.md"
        if not skill_md.is_file():
            logger.warning(
                "skill directory %s has no SKILL.md; skipping",
                skill_dir,
            )
            continue

        text = _read_text_or_warn(skill_md)
        if text is None:
            continue

        try:
            parsed = parse_skill_md(skill_md, text)
        except SkillMdError as exc:
            logger.warning(
                "failed to parse %s as a SKILL.md: %s",
                skill_md, exc.message,
            )
            continue

        entries.append(SkillEntry(
            name=skill_dir.name,
            description=parsed.description,
            skill_md_path=skill_md,
        ))

    return entries


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------

def collect_context_for_directory(
    directory: ContextDirectory,
) -> CollectedContext:
    """Assemble the per-directory bundle from each per-category collector.

    A flat composition: one call per :class:`CollectedContext`
    field.  Each collector handles its own kind-filtering and
    fallback policy, so this function never branches on
    ``directory.kind``.  Adding a new category is a two-step change:
    add a field on :class:`CollectedContext` and a collector call
    here.
    """
    return CollectedContext(
        directory=directory,
        agents_md=collect_agents_md_contribution_for_directory(directory),
        memory_md=collect_memory_md_contribution_for_directory(directory),
        mcp_configs=collect_mcp_configs_for_directory(directory),
        skills=collect_skills_for_directory(directory),
    )


def load_context_layers(
    directories: list[ContextDirectory],
) -> list[CollectedContext]:
    """Map :func:`collect_context_for_directory` over every directory.

    Returns one :class:`CollectedContext` per input, in the same
    order, including bundles where every category is empty.
    Phase 3 is responsible for filtering out empties at assembly
    time -- keeping them here preserves provenance for diagnostic
    purposes ("why is *this* directory in the walk?").
    """
    return [collect_context_for_directory(d) for d in directories]


__all__ = [
    "CollectedContext",
    "SkillEntry",
    "TextContribution",
    "collect_agents_md_contribution_for_directory",
    "collect_context_for_directory",
    "collect_mcp_configs_for_directory",
    "collect_memory_md_contribution_for_directory",
    "collect_skills_for_directory",
    "load_context_layers",
]
