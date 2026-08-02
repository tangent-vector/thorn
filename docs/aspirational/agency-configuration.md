Agency Configuration and Operator UX
====================================

This document describes a target design for replacing the current
`thorn serve bootstrap` path with a more useful operator-facing configuration
model.
It is aspirational and does not match the implementation today.

The short version:

- `thorn serve bootstrap` should be treated as design debt.
- The agency configuration file remains the source of truth.
- CLI commands should provide small, safe, typed mutation and validation
  primitives.
- Agent skills should provide the friendly onboarding and configuration
  experience.
- Direct human or agent edits to the agency configuration should remain a
  normal, supported workflow.


Problem frame
-------------

The current bootstrap command was built as a narrow golden path: create a
pre-configured project coordinator, infer a forge from a single project URL,
write `gateway.json`, write an agent identity, and point the operator at a few
environment variables.

That shape no longer matches how Thorn is being dogfooded.
Recent trials have generally hand-authored agency configuration instead,
because the bootstrap path bakes in policy choices and defaults that are too
specific:

- It assumes a project-coordinator-shaped agency.
- It assumes a particular relationship between projects, forges, and agent
  identity.
- It uses `gateway.json` as the one file it knows how to write, even though
  the file is really agency-level configuration.
- It hides too much schema knowledge behind a one-shot command, then leaves
  operators editing the generated output anyway.
- It gives new users an attractive but misleading starting point.

The problem is not that bootstrap needs more flags.
The problem is that bootstrap is trying to be both configuration generator and
operator onboarding experience.
Those should be separate.


Goals and non-goals
-------------------

Goals:

1. Make agency setup incremental.
   Operators and agents should be able to create an agency, add a project,
   add a peer, add an account, validate, and start the daemon as separate
   steps.
2. Keep the configuration file authoritative.
   CLI commands mutate or validate config; they do not become a second hidden
   state store.
3. Make common mutations safe.
   Commands should generate IDs, avoid duplicate entries, preserve unrelated
   fields, keep secrets out of files, and validate after writing.
4. Let agents provide the friendly experience.
   Thorn should ship skills that teach agents how to configure an agency using
   the docs, CLI plumbing, and direct file edits.
5. Preserve direct editing.
   Complex configuration should not require an enormous command surface.
   Humans and agents should be able to edit the config file directly and then
   run validation.
6. Align the CLI with the agency-daemon direction.
   Starting and stopping a local or server agency should feel like operating
   one daemon, not like choosing between unrelated CLI and gateway modes.

Non-goals:

- Building a full interactive wizard into Thorn.
- Making CLI subcommands expressive enough to cover every possible
  configuration edit.
- Solving every daemon-control-plane detail already covered by
  `agency-control-plane.md`.
- Removing JSON/state serializers that are framework-owned rather than
  operator-authored.
- Making this a blocker for all config-schema cleanup work.


Core model
----------

### Config file as source of truth

The agency configuration file is the durable source of truth for services,
projects, peers, agents, default agent selection, sandbox policy, broker
policy, and related operator-managed configuration.

The preferred human-facing name should become `agency.*`, not `gateway.*`,
because the same configuration is used by local and server agency modes.
`gateway.*` should remain accepted as a compatibility name during migration.

Format discovery and YAML support are tracked separately, but this design
assumes the eventual loader can read exactly one of:

```text
agency.yaml
agency.json
gateway.yaml
gateway.json
```

Everything after parsing should still flow through the same Pydantic models.

### CLI as plumbing

The CLI should offer reliable plumbing for operations that benefit from code:

- Create a minimally valid agency layout.
- Generate stable or random identifiers.
- Add or update common config entries.
- Normalize paths and URLs.
- Keep secrets represented by references instead of literal values.
- Detect duplicates and ambiguity.
- Validate the whole agency after mutation.
- Print machine-readable output for agents and scripts.

The CLI should not attempt to encode every advanced policy decision as a flag.
When a change is rare, cross-cutting, or easier to review in context, editing
YAML directly is the better interface.

### Agent skills as porcelain

Thorn should ship strong skills that agents can use to guide users through
agency setup and maintenance.
Those skills are the friendly interface.

An agent using the skills can:

- Ask the user clarifying questions.
- Explain tradeoffs.
- Read and edit the config file.
- Use `thorn agency ...` commands for common safe mutations.
- Run `thorn agency check`.
- Interpret validation errors.
- Produce a small review of what changed.

This gives Thorn an agent-guided onboarding experience without hard-coding a
fragile wizard into the CLI itself.

### Direct edits are first-class

Direct file edits are not a failure mode.
They are the right answer for advanced configuration.

The system should make direct edits safe by providing:

- Clear docs.
- Good examples.
- Strong Pydantic validation.
- `thorn agency check`.
- Diff-friendly YAML.
- Agent skills that understand the schema and conventions.


Proposed command shape
----------------------

The exact command names can change, but the shape should separate
agency-configuration plumbing from daemon lifecycle.

### Agency configuration commands

`thorn agency` is the operator/admin command group for agency configuration:

```text
thorn agency init [path]
thorn agency check [--agency path]
thorn agency show [--agency path] [--json]

thorn agency agent list [--agency path] [--json]
thorn agency agent add <agent-id> [--agency path]
thorn agency agent set-default <agent-id> [--agency path]
thorn agency agent account add <agent-id> <service> [--agency path] ...

thorn agency peer list [--agency path] [--json]
thorn agency peer add <peer-id-or-name> [--agency path]
thorn agency peer account add <peer-id> <service> [--agency path] ...

thorn agency forge list [--agency path] [--json]
thorn agency forge add <name> [--type type] [--url url] [--agency path]

thorn agency project list [--agency path] [--json]
thorn agency project add <name> --url <url> [--agency path]
```

The `agency` word is a little long, but it earns its keep by making the
operator boundary obvious.
These commands mutate agency configuration, not session state or ordinary
agent workspaces.

For agent-driven usage, long command chains are acceptable.
The agent is not inconvenienced by typing.
For humans, the common cases should stay short enough, and direct YAML editing
is always available for complex changes.

### Daemon lifecycle commands

Lifecycle commands should be short because they are frequent:

```text
thorn up [--agency path]
thorn down [--agency path]
thorn restart [--agency path]
thorn status [--agency path] [--json]
thorn logs [--agency path] [--follow]
```

`thorn serve` can remain the foreground/server-mode command for compatibility.
The product direction, though, should be that `thorn up` starts the agency
daemon in the background and `thorn down` stops it.

This aligns with the control-plane direction:

- `thorn chat` and `thorn run` should eventually attach to an agency daemon.
- `thorn serve` is one way to run that daemon in the foreground.
- `thorn up` is the operator-friendly way to ensure a daemon is running.

### Deprecated bootstrap command

`thorn serve bootstrap` should be deprecated rather than expanded.

The replacement flow should be:

```text
thorn agency init .thorn
thorn agency project add slang --url https://example.com/org/slang
thorn agency peer add example-user
thorn agency check
thorn up --agency .thorn
```

During migration, bootstrap can remain as a compatibility command, but its help
text should make it clear that new users should prefer `thorn agency init` and
the agency configuration commands.
Eventually it should be hidden or removed.


Command behavior principles
---------------------------

### Mutations validate after writing

Every mutating `thorn agency ...` command should parse, modify, write, reload,
and validate the agency configuration.

If validation fails after a mutation, the command should not leave the operator
guessing.
Possible strategies:

- Write to a temporary file first, validate, then replace.
- Create a backup before replacing.
- Print the exact path written and the validation error.
- Support `--dry-run` to print the proposed change.

The exact persistence strategy can be decided during implementation, but
silent partial writes should be avoided.

### Preserve human-authored formatting when practical

YAML makes agency configuration more humane, but programmatic mutation can
destroy comments and formatting.

The first implementation can accept some formatting churn for simple commands,
but the long-term goal should be to preserve comments and local formatting
where practical.
If preserving formatting proves too expensive with the chosen parser, the CLI
should be conservative about how much it rewrites and should prefer direct
append/update operations for common lists.

### Secrets are references, not values

Configuration commands should not encourage operators to paste literal tokens
into the agency file.

Commands that add accounts should prefer environment-variable references,
credential broker references, or future credential-store references.
If a command accepts a literal secret as an ergonomic shortcut, it should store
it only in the appropriate credential store, never inline in `agency.yaml`.

### IDs are explicit

Commands can generate IDs, but they should make the result visible.

For example, `thorn agency peer add example-user` may create a stable peer entry
with a generated internal ID if no immutable service account is known yet.
The command should print the peer ID and the path it changed so a human or
agent can refer to it later.

### JSON output is for agents and scripts

Every read-oriented command should support `--json`.

Agent skills should prefer JSON output when they need to inspect current
state, because it avoids parsing human tables.


Agent skills
------------

The repository should include skills that are useful even to agents not
running inside Thorn.

Candidate skills:

- `thorn-agency-configuration`
  - Explains the agency config model.
  - Teaches the `agency.*` / `gateway.*` file discovery rules.
  - Documents common CLI commands.
  - Defines when to use direct YAML edits.
  - Requires `thorn agency check` after changes.
- `thorn-operator-onboarding`
  - Guides a user through creating an initial agency.
  - Helps choose a workspace path, default agent ID, projects, peers, and
    service accounts.
  - Produces a short checklist of remaining credential/environment work.
- `thorn-agency-review`
  - Reviews a proposed agency configuration for obvious mistakes.
  - Checks for inline secrets, duplicate peers/accounts, missing default agent,
    missing workspace, unsupported forge URLs, and unclear authority boundaries.

These skills should treat CLI commands as tools, not as the whole interface.
For common edits, they can call `thorn agency ...`.
For advanced edits, they can modify the config file directly and then run
validation.


Relationship to existing docs
-----------------------------

This document sits between several existing design threads:

- `agency-control-plane.md` describes the daemon/control protocol direction.
  This document borrows that destination but focuses on operator UX and
  configuration.
- `architecture.md` describes agencies, agents, services, peers, and sessions.
  This document assumes those concepts but changes how an operator creates and
  edits their configuration.
- The YAML agency-config issue tracks the narrower file-discovery and parsing
  change: preferring `agency.*` while retaining `gateway.*`.
- `startup_flow.md` describes current `thorn serve` behavior and will need an
  update once `agency.*`, `thorn up`, and agency-daemon lifecycle commands
  exist.


Phasing
-------

A plausible sequence of independently reviewable phases:

- **Phase 1 - Agency config loader cleanup.**
  Support `agency.yaml`, `agency.json`, `gateway.yaml`, and `gateway.json`,
  with ambiguity diagnostics and shared Pydantic validation.

- **Phase 2 - `thorn agency check` and `show`.**
  Add read-only commands that load, validate, summarize, and print JSON for an
  existing agency.
  This gives direct config editing a safety net before mutation commands
  exist.

- **Phase 3 - `thorn agency init`.**
  Create a minimal agency directory with a valid agency config, an agents tree,
  a default agent, and a workspace reference.
  Do not infer a project, forge, or peer unless the user asks.

- **Phase 4 - Common mutation commands.**
  Add project, forge, peer, agent, and account commands for the most common
  operations.
  Each command should validate after mutation and support JSON output where
  useful.

- **Phase 5 - Repository skills.**
  Add the agency-configuration and onboarding skills to the Thorn repo.
  The skills should use the commands where they help and direct YAML edits
  where they are clearer.

- **Phase 6 - Daemon lifecycle commands.**
  Add `thorn up`, `thorn down`, `thorn restart`, and `thorn logs` over the
  agency-daemon/control-plane work.
  Keep `thorn serve` as the foreground compatibility path.

- **Phase 7 - Bootstrap deprecation.**
  Update help text and docs to steer new users away from `thorn serve
  bootstrap`.
  Later, hide or remove the command after the replacement flow is proven.


Open questions
--------------

- Should `thorn agency init` default to `./.thorn`, `~/.thorn`, or require an
  explicit path in non-interactive contexts?
- Should config mutation commands preserve YAML comments, or is normalized
  output acceptable for the first version?
- What is the right account-add UX once credentials move beyond environment
  variable references?
- Should `service` be a user-facing command group, or should the first
  operator UX expose concrete groups like `forge`, `project`, and `peer`
  instead?
- How soon should `agent.yaml` become preferred over `agent.json`, and is
  agent identity operator-authored config or framework-owned state?
- Should `thorn up` be implemented before the full control plane, using the
  current `thorn serve` implementation as the child process, or wait for the
  daemon protocol?


Provisional decisions
---------------------

- Do not expand `thorn serve bootstrap`; replace it.
- Prefer `agency.*` terminology for agency-level configuration.
- Use `thorn agency ...` for operator/admin configuration plumbing.
- Keep direct config editing supported and expected.
- Ship skills as the porcelain layer for friendly onboarding and
  configuration changes.
- Keep lifecycle commands short and top-level: `thorn up`, `thorn down`,
  `thorn restart`, `thorn logs`.
- Use validation as the contract between direct edits, CLI mutations, and
  agent-guided changes.
