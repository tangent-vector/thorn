Per-Prompt Context Gathering
============================

Every time a Thorn session is about to be prompted, it runs a single
unified pipeline that decides what context to include in the system
prompt and which tools to make available for the round.  The pipeline
is the one source of truth for that decision regardless of whether the
session was started by `thorn chat`, `thorn run`, or the gateway --
reducing the surface area where CLI and gateway behaviour can drift.

This document describes the shipped shape of that pipeline.  It
supersedes an earlier aspirational design doc that was used during
the refactor's planning phase; whenever this document and any older
docstring or comment disagree, this document wins.


Official prompt-source model
----------------------------

The official model is: Thorn-owned infrastructure guidance comes
first, then call-site steering, then the context discovered by this
pipeline.  The discovered context is the only supported way for
operator-, agent-, and workspace-owned files to contribute recurring
prompt text.

The final `system_prompts` list passed to the provider is assembled in
this order:

1. `ExecutionContext.system_prompts`.  This is a runtime-level slot
   for non-file prompt fragments supplied by the caller.  It is not
   populated from gateway or agent JSON.
2. Static role prompts returned by `agent._render_system_prompts()`.
   The same class-role mechanism is still load-bearing for
   first-party Thorn behavior: `thorn chat` and `thorn run` use
   `LocalCodingAgent` for their tool bundle, while `thorn serve`
   still uses `GatewayAgent` and `ProjectCoordinator` when a
   persisted agent JSON names `"agent_class": "ProjectCoordinator"`.
3. Per-call `extra_system` text.  `thorn chat` uses this for the
   interactive-chat steering fragment; ordinary gateway notification
   handling does not expose this as persisted operator configuration.
4. The context blocks assembled by this pipeline: environment,
   `AGENTS.md` / `CLAUDE.md`, skill index, `MEMORY.md`, and recent
   journal entries, in the phase-3 order documented below.

For gateway and CLI sessions running under a `Runtime`, the first
`AGENTS.md` location in the pipeline is the operator directory:
`<agency-home>/agents/<agent-id>/AGENTS.md`.  That directory is
framework-owned and is outside the agent's sandbox-visible `home/`
mount, so it is the current place for operator policy that the agent
must not be able to edit through its normal tools.  The next
`AGENTS.md` locations are the agent-home chain and then the workspace
chain.  `MEMORY.md`, MCP configs, and skills continue to follow their
per-kind policy in the table below.

The static role-prompt path is intentionally transitional.  The
remaining migration work is tracked in `ROADMAP.md` and in
`docs/plans/gateway-config-followups.md`: split framework-owned
gateway guidance from operator-authored role policy, stop selecting
agent behavior through `agent_class`, and derive gateway tool
availability from runtime/account configuration rather than a Python
class name.


Why a single pipeline
---------------------

Before the unification there were several parallel paths assembling
session prompts: an inline `load_workspace_instructions` walker for
`AGENTS.md`, a `load_agent_memory` walker for `MEMORY.md`, a
hand-rolled environment block injected at one specific call site, an
ad-hoc `WorkspaceConfig` for MCP server discovery, and a separate
copy of all of that in the gateway's session-creation path.  Each
walker had its own rules for which directories it considered, which
of them silently dropped contributions in edge cases, and which the
caller had to remember to invoke at the right time.  The most
visible regression that fell out of that fragmentation was the
"AGENTS.md silent-drop" bug, where a workspace `AGENTS.md` was
overwritten (and lost) by the agent-home one.

The unified pipeline replaces all of those paths.  There is exactly
one place where directories are enumerated, exactly one place where
each category of context is loaded, and exactly one place where the
final blocks are formatted.


The three phases
----------------

The pipeline is organised as three pure-ish phases, each owning a
single concern.  All three live under `src/thorn/runtime/` and are
called from `_run_session_prompt` in `src/thorn/core/_agent.py`.

### Phase 1 -- directory enumeration (`_context_paths.py`)

Inputs: the directory roots that bound the walk (operator dir,
agent home, session-key home, logical agent workspace, session
workspace; any of which may be `None`).
Output: a flat `list[ContextDirectory]`, ordered outer-to-inner,
deduplicated by path.

`ContextDirectoryKind` is a string enum with three values:

- `OPERATOR` -- `<agency-home>/agents/<agent-id>/`.  Outside the
  agent's sandbox; reserved for human-operator policy injection.
- `AGENT_HOME` -- a directory on the chain from the agent's home
  down into the session-key home.  The agent's "skull": private
  state that travels with the agent across workspaces.
- `AGENT_WORKSPACE` -- a directory on the chain from the logical
  agent workspace down into the session workspace.  The agent's
  "desk": where tool-driven work happens.

Within a layer, the walk is *inclusive of both bounds*, so when
`outer == inner` you get a single entry.  Across layers, paths are
deduplicated; the outermost kind wins on the rare collision (a
defensive pass; in practice the layers don't overlap).

This phase performs **no filesystem I/O**.  Whether any given
directory actually exists on disk is phase 2's problem.

### Phase 2 -- per-directory loading (`_context_layers.py`)

Inputs: a `list[ContextDirectory]` from phase 1.
Output: a `list[CollectedContext]` of identical length and order,
each carrying the per-category contributions found in its
directory.

`CollectedContext` is an array-of-structs bundle, not a
structure-of-arrays:

```
@dataclass(frozen=True)
class CollectedContext:
    directory: ContextDirectory
    agents_md: TextContribution | None
    memory_md: TextContribution | None
    mcp_configs: list[MCPServerConfig]
    skills: list[SkillEntry]
```

Adding a new category is a one-place edit on this dataclass plus a
new collector function (e.g. `collect_X_contribution_for_directory`).
Each collector owns its own kind-filtering and fallback policy.

The policy table for the categories that ship today:

| Category                | OPERATOR | AGENT_HOME | AGENT_WORKSPACE |
|-------------------------|----------|------------|-----------------|
| `AGENTS.md` (+CLAUDE.md fallback) | yes | yes | yes |
| `MEMORY.md`             | no       | yes        | no              |
| MCP server configs      | no       | yes        | yes             |
| Skills (`SKILL.md`)     | no       | yes        | yes             |

The `CLAUDE.md` alias is a *same-directory* fallback only -- it
never borrows from an outer directory's `CLAUDE.md` to fill an
inner directory's `AGENTS.md` slot.  That keeps the per-directory
policy genuinely per-directory.

I/O is best-effort throughout: missing files yield `None` / empty
list silently, unreadable files log a warning and yield the same.
The pipeline runs on every prompt, so any noisy failure mode would
tank session throughput.

### Phase 3 -- final assembly (`_prompt_assembly.py`)

Inputs: a `list[CollectedContext]` from phase 2, plus optional
ambient inputs (`workspace_path`, `agent_home_path`, `journal_text`)
that contribute their own blocks.
Output: an `AssembledPromptContext`:

```
@dataclass(frozen=True)
class AssembledPromptContext:
    system_prompt_blocks: list[str]
    mcp_configs: list[MCPServerConfig]
    skills: list[SkillEntry]
```

This phase touches no filesystem.  All formatting, ordering, and
deduplication happens here.

The block order in `system_prompt_blocks` is:

1. Environment block -- working directory, agent-home directory.
   One block, generated from `workspace_path` and `agent_home_path`.
   Omitted when neither is supplied.
2. `AGENTS.md` contributions, outer-to-inner.  Each contribution
   becomes its own block with a provenance header above the file's
   content (so a human reading the prompt can tell which `AGENTS.md`
   it came from).
3. Skill index block -- one combined block listing every discovered
   skill with its description and the path to its `SKILL.md`.
   Skills are *advertised*, not auto-invoked: the agent reads a
   `SKILL.md` body via the ordinary file-read tool when the
   description suggests it is relevant.  No per-skill tool wrapper
   is generated.
4. `MEMORY.md` contributions, outer-to-inner, same provenance
   treatment as `AGENTS.md`.
5. Recent journal entries -- one block, content already rendered
   by `read_recent_journal` in the caller.

Phase 3 owns only the context blocks in the list above.  Runtime
system prompts, static role prompts, and `extra_system` sit upstream
of this module in the caller's final concatenation.  A persona slot
(`SOUL.md` / `PERSONA.md`) from the aspirational design is not yet
emitted; when it lands it will sit between upstream Thorn-owned
prompts and the `AGENTS.md` blocks.

#### Dedup

- **MCP configs** are deduplicated by **content hash** across every
  layer.  Two `.agents/mcp.json` files that declare an identical
  server (same name, command, args, env, url) collapse to one
  entry; the outermost occurrence wins.  Configs that share a name
  but differ in any other field are *both* kept; resolving that
  conflict is left to the MCP layer.
- **Skills** are accumulated outer-to-inner with no deduplication.
  When two layers happen to advertise a same-named skill, both
  appear in the index.  If the index becomes large enough that
  duplicates are confusing, the cheapest fix is "outermost wins
  by name" -- matching MCP.


The directory ontology
----------------------

The pipeline talks about directories using the following terms.
Codebase naming may not always match exactly, but these are the
intended meanings.

- **agency home** -- the top-level directory that contains
  `gateway.json` (which arguably should be named `agency.json`).
  In CLI mode this defaults to `~/.thorn/`.

- **operator directory** --
  `<agency-home>/agents/<agent-id>/`.  *Not* the agent home: this
  directory sits outside what the agent itself can see or modify
  once container isolation is in place (only `home/` gets mounted
  in).  It exists so that the human operator of an agency can
  inject context that *must* be included no matter what the agent
  does.

- **agent home** -- `<agency-home>/agents/<agent-id>/home/`.  The
  agent's persistent private space; contains `MEMORY.md`, the
  `journal/` directory, and any topical sub-trees the operator or
  agent has organised under it.

- **session-key home** -- `<agent-home>/<session-key-as-path>`.
  Used to surface topical memory and skills organised by session
  shape: a session whose key is `projects/foo/issues/123` will
  pick up context from `<agent-home>/projects/`,
  `<agent-home>/projects/foo/`, and so on.

- **logical agent workspace** -- in gateway mode, the per-agent
  workspace mount; in CLI mode, the value computed by
  `pick_logical_agent_workspace_path_for_cli_session` at startup
  (see below).  This is the *upper bound* of the workspace-side
  walk.

- **session workspace** -- the directory where tool work
  actually happens.  In gateway mode this is
  `<agent-workspace>/<session-key-as-path>`; in CLI mode it is
  always the CWD when `thorn chat` / `thorn run` was invoked.


How `SessionKey` maps to disk
-----------------------------

`SessionKey` is no longer a `str` subclass.  Internally it stores
`components: tuple[str, ...]`; the `/` separators in the
human-readable rendering map to actual directory separators on disk
via `session_key_path()` (with per-segment quoting for characters
that aren't filesystem-safe inside a single segment).  Round-trips
through `session_key_from_path()` recover the original
`SessionKey`.

The session's framework-managed metadata (`session.json`,
`history.json`, `inbox/`, `errored/`) lives under a sentinel
subdirectory `_state/`:

    <sessions-root>/<session-key-as-path>/_state/session.json

The sentinel exists to prevent collisions between hierarchical
session keys.  Without it, sessions `a/b` and `a/b/inbox` would
both want to put their `inbox/` directory at the same path; with
it, the former's metadata lives under `a/b/_state/inbox/` while
the latter's lives under `a/b/inbox/_state/`.

Reservation policy on session-key segment names (the "no
underscore-prefixed components" rule, no clashes with `_state` or
similar) is *not* enforced at the `SessionKey` type level.  That
policy belongs in the session-key *templates* (see
`docs/aspirational/coordination.md`), so per-template validation
can decide which segments are user-supplied (and need escaping)
vs. literal.


CLI vs gateway: where the inputs come from
------------------------------------------

The pipeline itself doesn't care whether it's running under
`thorn chat`, `thorn run`, or the gateway -- they all hand it the
same five optional path inputs.  Where those paths come from does
differ:

| Input                          | Gateway                                        | CLI |
|--------------------------------|------------------------------------------------|-----|
| operator dir                   | `<agency-home>/agents/<agent-id>/`             | same shape, defaults to `~/.thorn/agents/local/` |
| agent home                     | `<agency-home>/agents/<agent-id>/home/`        | `~/.thorn/agents/local/home/` |
| session-key home               | `<agent-home>/<session-key-as-path>`           | same |
| logical agent workspace        | per-agent workspace mount from `gateway.json`  | output of `pick_logical_agent_workspace_path_for_cli_session(cwd)` |
| session workspace              | `<agent-workspace>/<session-key-as-path>`      | the CWD at invocation time |

The CLI's logical-agent-workspace policy is split across three
deliberately-orthogonal pieces in `_project_detection.py`:

- `is_logical_project_directory_path(path)` -- the *predicate*:
  does the path look like a project root?  Returns `True` for
  VCS roots (`.git`, `.hg`, ...), Thorn agency markers (`.thorn/`),
  language manifests (`pyproject.toml`, `package.json`, ...), and
  agent-contract files (`AGENTS.md`, `CLAUDE.md`).
- `find_outermost_enclosing_logical_project_directory_path(path,
  upper_bound=...)` -- the *walker*: scan ancestors and return the
  outermost one matching the predicate, optionally bounded above.
- `pick_logical_agent_workspace_path_for_cli_session(cwd)` -- the
  *policy*: outermost project root strictly below `Path.home()`,
  fall back to the session workspace itself if no enclosing
  project root is found.

Each can be tweaked in isolation: refining marker recognition only
touches the predicate, switching outermost↔innermost only touches
the walker, and adjusting the upper-bound or fallback only touches
the policy.

Each `Session` carries its `logical_agent_workspace_path` as a
field, persisted by the JSON serializer alongside `workspace_root`.


Tests
-----

Each phase is covered by black-box tests against documented
behaviour:

- `tests/test_context_paths.py` -- phase 1, including all three
  layers, partial inputs, the unrelated-pair branch, and dedup
  edge cases.
- `tests/test_context_layers.py` -- phase 2, every collector
  independently against tmp-dir fixtures, plus
  `collect_context_for_directory` composition; explicit regression
  guards for the AGENTS.md/CLAUDE.md alias rule and per-kind
  exclusions.
- `tests/test_prompt_assembly.py` -- phase 3, block ordering,
  MCP content-hash dedup edge cases, skill accumulation order.
- `tests/test_skill_md.py` -- the YAML-frontmatter parser in
  isolation.
- `tests/test_project_detection.py` -- the CLI policy split.
- `tests/test_agent.py::TestAgentsMdInjection` and
  `::TestMemoryMdInjection` and `::TestEnvironmentPromptInjection`
  -- end-to-end through `_run_session_prompt`, asserting against
  the `system_prompts` actually handed to the LLM provider.  The
  AGENTS.md case is the explicit silent-drop regression test.
- `tests/test_paths.py::TestSessionKeyPathEncoding` -- the
  session-key-as-path fix, with an inline comment naming the
  historical `safe_dirname(session_key)` bug it guards against.


Future work
-----------

These are extensions of the pipeline's ontology or mechanism, not
loose ends from the refactor. Selected product-level follow-ups live in
`ROADMAP.md`.

### Operator-policy population

The pipeline already considers
`<agency-home>/agents/<agent-id>/AGENTS.md` (the operator
directory) as an `OPERATOR`-kind contribution when a prompt is
running under a `Runtime`.  What's not yet shipped is gateway-side
provisioning that *populates* that file: today the directory exists
for framework state (`agent.json`, `sessions/`, `home/`), but
agencies don't put policy there by default.  Once we have a use case
(e.g. mandatory forge-credential warnings, or an agency-wide
code-of-conduct), the gateway should learn to write a default
`AGENTS.md` into that directory at agent-creation time.

### Cross-tool aliasing for context sources

Today the pipeline recognises `AGENTS.md` (with `CLAUDE.md` as a
same-directory fallback) and `.agents/skills/`, `.agents/mcp.json`.
Other tools in the ecosystem use parallel conventions:
`.cursor/rules/*.mdc`, `.claude/agents/*.md`, `.claude/skills/`,
`.claude/mcp.json`, etc.  A future iteration should fold these in
as additional sources of the same conceptual contributions, using
the same per-category collectors (rather than a parallel set of
walkers).

### Persona slot

The aspirational design carved out a slot for a per-agent
`SOUL.md` / `PERSONA.md` between the super-global prompts and the
`AGENTS.md` blocks.  Not yet implemented; landing it is a phase-3
formatter addition plus a phase-2 collector following the same
shape as `collect_agents_md_contribution_for_directory`, scoped to
`AGENT_HOME` only.

### Topical-memory walks via session-key templates

The current `AGENT_HOME` walk goes from the agent home down to the
session-key home and considers every directory on that chain.
That's already a powerful "topical memory" mechanism for sessions
whose keys naturally form a hierarchy.  When session-key
*templates* (see `docs/aspirational/coordination.md`) ship, the
pipeline should learn to consider additional template-derived
paths under the agent home -- e.g. a session keyed
`projects/foo/issues/123` could pull in
`<agent-home>/projects/foo/templates/issue/` automatically.

### A "logical agent workspace" rung in the runtime ontology

The `logical_agent_workspace_path` smuggled onto `Session` is a
near-term workaround.  The deeper question -- "what is the
ontological rung between Agent and Session that owns one container,
one workspace mount, one set of MCP-server connections, and so
on?" -- is still unanswered.  Today gateway agents have exactly
one such rung (the agent itself) while CLI agents may have many
(one per project the user runs Thorn from).  Container isolation
will eventually force a real type here.  Scoped out of the
context-gathering refactor by design; tracked in `ROADMAP.md` so it
doesn't get lost.

### Skills/tools discovery unification

The pipeline owns `SKILL.md` discovery and `mcp.json` discovery
today.  `@tool` / `@skill` Python decorator discovery (in
`thorn.core._discovery.discover_tools`) still has its own ancestor
walker that runs separately. Tracked in `ROADMAP.md`; the natural
unification is to route `discover_tools` through the same
phase-1/phase-2 walkers, so directory selection and dedup live in
one place.

### Agent-role policy migration

`Agent.system_prompts` (a ClassVar) and the `_render_system_prompts`
template-substitution machinery predate the unified pipeline.
`thorn.gateway._agents.ProjectCoordinator._COORDINATOR_SYSTEM_PROMPT`
is real, in-use content baked into Python; `LocalCodingAgent`,
`GatewayAgent`, and `ProjectCoordinator` are the first-party class
paths that remain load-bearing today.  Migrating that content into
discovered context requires first deciding which parts are
non-overridable Thorn infrastructure and which parts are
operator-authored role policy. Tracked in `ROADMAP.md` and in
`docs/plans/gateway-config-followups.md`.
