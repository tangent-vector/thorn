# Gateway Configuration Cleanup — Follow-ups

## Status and provenance

This document captures the **bigger-picture follow-ups** that were
intentionally deferred from the gateway-config-ergonomics cleanup
PR (see *Cross-references* at the bottom for the originating plan).
That PR took the "small/safe items only" slice — URL/name
inference, fork-name defaults, default-branch lookup, `id`/`name`
deduplication, accounts unification, and a matching bootstrap
rewrite — under a clean-break policy on the on-disk format.  Each follow-up below was scoped out of that PR
because it deserves its own design discussion and would have dwarfed
the cleanup work.

The originating plan listed four follow-ups (labelled A–D) plus a
suggested sequencing.  This document re-presents them in the
recommended order of execution, with enough design detail to pick
any one of them up cold.  Each section ends with explicit open
questions to settle before implementation starts.

The ordering matters: the URL→forge plumbing landed in the cleanup
PR is a prerequisite for B, the unified `accounts` array is a
prerequisite for C, and A and D both want to land after C so that
"this agent's role" is no longer entangled with "this agent has a
single project."

## The four follow-ups, in recommended sequence

| # | Code name              | Removes from config                         | Depends on                |
|---|------------------------|---------------------------------------------|---------------------------|
| 1 | URL→forge credentials  | `metadata.project` (one of two consumers)   | Cleanup PR (URL plumbing) |
| 2 | `git`/`email` services | `git_user_name` / `git_user_email` per-acct | Cleanup PR (`accounts[]`) |
| 3 | Drop `agent_class`     | `agent_class`                               | (independent, but large)  |
| 4 | Multi-project agents   | `metadata.project` (the remaining consumer) | 1 + 3                     |

Items 1 and 4 between them retire `metadata.project` entirely;
item 1 handles the easy half (credential injection), item 4 handles
the harder half (git identity selection across multiple projects),
and item 3 is the prerequisite for item 4 because the routing
rules that replace `metadata.project` only make sense once
`agent_class` is gone and "what an agent does" lives in its
`AGENTS.md` and `accounts` instead.

---

## Follow-up 1 — URL→forge credential lookup

### Goal

Stop using `agent.metadata["project"]` to find which forge a git
operation should authenticate against.  Use the URL of the request
itself (or, failing that, the project that's already attached to
the calling session) to pick the right account on the agent.

### Why now

The Cleanup PR established that every fork has a `url` whose host
identifies its forge, and that `ForgeSpec.url` carries the
human-facing instance URL.  That makes URL-host → forge-service →
account a clean lookup chain that doesn't need any
out-of-band project metadata.

### Current shape (pre-change)

`_git_auth_env_for_current_agent` in
[`src/thorn/tools/git.py`](../../src/thorn/tools/git.py) does
roughly:

```python
project_name = agent.metadata.get("project")
project_svc  = ctx.runtime.get_service(project_name)
forge_svc    = ctx.runtime.get_service(project_svc.forge_name)
account      = resolve_forge_account(agent, forge_svc.name)
token        = forge_svc.git_https_password_for(account.credentials)
```

The same cascade appears in `_resolve_git_identity` for the commit
identity.  Both are gated on the agent declaring exactly one
project in its metadata, which is exactly the constraint we want
to remove.

### Proposed shape (post-change)

Two layered lookups, in priority order:

1. **URL-driven**: when a git command is being issued for a known
   remote URL (clone, push, fetch, pull all know the URL because
   they pass it on the command line), parse the URL host and look
   up the matching `ForgeHostService` directly.  The URL → forge
   match uses the same host-comparison helper that already powers
   forge synthesis in `_resolve_forges_and_projects`.
2. **Session-attached project fallback**: when the URL is not
   available (e.g. `git commit`), use the project service that the
   current session was routed for.  The session key already
   carries the project name (it's the first segment of the keys
   produced by event sources), so this is a one-line extraction
   rather than a metadata lookup.

In both cases the *agent* is selected as before (it's the agent
the session is running on), and `resolve_forge_account` filters
the agent's `accounts[]` by service name to find the credential.

### Open questions

- **Multiple matching accounts on the same forge.**  If an agent
  has two PATs for `github`, which one wins?  Either we forbid
  this entirely (one account per service), or we pick the first
  match deterministically.  Forbidding is cleaner; check whether
  any plausible workflow needs the second account before locking
  it in.
- **Out-of-band fetches.**  `git fetch` against a submodule's URL
  may resolve to a forge the agent has no account on.  The
  current code falls through to `forge_svc.git_https_password()`
  (the forge-wide token).  After this change, that fallback path
  goes away with `metadata.project`; we need to decide whether
  unauthenticated fetches against unknown-forge URLs are
  acceptable (probably yes, they work for public repos).
- **GHE / self-hosted GitLab host matching.**  Hostname comparison
  has to be case-insensitive and ignore the `api.` prefix the
  same way `_host_of` already does.  Worth pulling that into a
  named helper rather than open-coding the comparison at each
  call site.

### Test plan sketch

- New unit tests in `tests/test_tools_git.py`:
  - `test_git_auth_uses_url_to_pick_forge` — agent has accounts
    on two forges, the URL host disambiguates.
  - `test_git_auth_falls_back_to_session_project` — `git commit`
    in a session whose key starts with `proj-a/...` picks
    `proj-a`'s forge.
  - `test_git_auth_unauthenticated_for_unknown_url` — confirms
    we don't 500 when the URL points at a forge the agent has
    no account on.
- Update existing tests that synthesise an agent with a single
  `metadata.project` to use the new URL/session-driven path.

---

## Follow-up 2 — `git` and `email` as first-class services

### Goal

Stop attaching `git_user_name` and `git_user_email` to *forge*
accounts.  Make them properties of dedicated `git` and `email`
account entries in the unified `accounts` array (which already
exists, courtesy of the Cleanup PR).

### Why now

Every dogfood agent has a single git identity that they want to
use regardless of which forge they're committing through.  The
current shape — git identity per forge — actively works against
that, since adding a second forge account means restating the
same name and email.

### Proposed schema

Agent JSON gains two new account `service` values, each with its
own model:

```json
{
  "accounts": [
    { "service": "git",
      "user_name": "Thorn Agent",
      "user_email": "thorn@example.com" },
    { "service": "email",
      "address": "thorn@example.com" },
    { "service": "github",
      "credentials": { "kind": "pat", "token": "$GITHUB_TOKEN" } },
    { "service": "gitlab",
      "credentials": { "kind": "gitlab-pat", "token": "$GITLAB_TOKEN" } }
  ]
}
```

The forge accounts no longer carry `git_user_name`/`git_user_email`
fields at all.  The `git` account is the single source of truth
for the agent's commit identity; the `email` account is reserved
for future email-tool integrations.

### Defaulting cascade

`git.user_email` defaults to `email.address` when the agent has an
`email` account but no explicit git email.  This matches what most
people would actually configure by hand.

A bare agent (no `git` account, no `email` account) gets no
identity injected into git commands; the git CLI falls back to its
own ambient defaults, which is the right behaviour for tests and
local exploration.

### Implementation sketch

- Add `GitIdentityAccountConfig` and `EmailAccountConfig` (or
  similar) in `thorn.core._account` as additional members of the
  `accounts[]` discriminated union.  The discriminator is
  `service` (already in place); each new model declares its own
  `service: Literal["git" | "email"]`.
- Introduce `resolve_git_identity(agent) -> tuple[str, str]`
  alongside `resolve_forge_account`, replacing the per-forge name
  and email lookup in `_resolve_git_identity`.
- Migrate `bootstrap_coordinator` to emit a single `git` account
  (and optionally an `email` account when an email is supplied)
  rather than wedging the identity into the forge account.

### Why this depends on the Cleanup PR

The Cleanup PR's unified `accounts: list[…]` discriminated on
`service` is the foundation for this — without it, adding a `git`
service would have meant inventing the discriminator from scratch.
With it, this is a pure additive change in the `accounts` shape
plus a single helper.

### Open questions

- **Coexistence during migration.**  Do we accept agent JSON that
  still carries `git_user_name`/`git_user_email` on a forge
  account, or do we hard-error and rely on the bootstrap to
  rewrite?  Given the clean-break policy that's already in place
  for the gateway config, hard-error is consistent.
- **Multiple `git` accounts.**  Probably forbid; an agent has *one*
  git identity at a time.  Worth re-checking once we look at the
  multi-project case (follow-up 4) — if a single agent ever
  legitimately wants different identities per project we'll need
  a way to select.
- **`email` account model.**  We don't ship any email tools yet,
  so the model can stay minimal (just `address`).  Future SMTP
  credentials would extend it.

### Test plan sketch

- `tests/test_account.py`: round-trip and discriminator tests for
  the new `git` and `email` account models.
- `tests/test_tools_git.py`: `_resolve_git_identity` returns the
  `git` account's identity in preference to the forge accounts'
  fields (which are no longer present).
- A bootstrap test confirming the new account shape is emitted by
  default.

---

## Follow-up 3 — Drop `agent_class` from agent JSON

### Goal

Stop selecting an agent's behaviour via a class name in the agent
JSON.  The Python class hierarchy (`Agent` and friends) stays;
what goes away is using the JSON's `agent_class: "ProjectCoordinator"`
to swap in a different default tool list and system prompt.

### Why now

This is the single largest piece of "magic" left in agent
configuration.  `agent_class` selects the entirety of
`_COORDINATOR_SYSTEM_PROMPT` (~180 lines of role guidance) and a
specific bundle of tool sets, both baked into
[`src/thorn/gateway/_agents.py`](../../src/thorn/gateway/_agents.py).
That's a lot of behaviour hidden behind a single string.

### Proposed shape

Two pieces split apart:

1. **Role/system-prompt** moves out of Python and into the agent's
   home directory.  An `AGENTS.md` in the agent's home is already
   loaded as a system-prompt fragment; the gateway-specific
   workflow guidance currently in `_COORDINATOR_SYSTEM_PROMPT`
   becomes part of the boilerplate that the bootstrap writes into
   that `AGENTS.md` (or, if we want operators to be able to share
   role definitions across agents, into a referenced file under
   the agency home).
2. **Tool list** is determined by the runtime from the agent's
   declared accounts.  The rule is: an agent that has any forge
   account on a registered forge gets `FORGE_TOOLS`, `GIT_TOOLS`,
   `FILE_READING`, `FILE_WRITING`, and `run_shell` automatically.
   This collapses "is a coordinator" into "has accounts on at
   least one forge", which is what we actually mean.

After both changes, agent JSON looks like:

```json
{
  "name": "thorn",
  "metadata": {},
  "accounts": [ ... ]
}
```

The `Agent` class stays in code; the runtime always instantiates
the base `Agent` and lets memory + accounts drive behaviour.

### Implementation sketch

- Drop the `_resolve_agent_class` indirection in the serializer;
  `load_agent` always returns an `Agent`.
- Move the gateway-relevant portions of `_COORDINATOR_SYSTEM_PROMPT`
  (the workflow guidance, journal/MEMORY conventions) to a
  template file (e.g. `src/thorn/gateway/_coordinator_role.md`).
  The bootstrap copies the relevant template into the agent's
  home `AGENTS.md` at creation time.
- Add a method on `Runtime` (or a free function that takes the
  agent and its registered services) that returns the toolset
  derived from the agent's accounts.  Hook it where the agent is
  registered with a session.

### What needs to be decided up-front

- **Where does the framework-injected scaffolding end and the
  user-authored role description begin?**  The current
  `_COORDINATOR_SYSTEM_PROMPT` mixes both.  Worth doing a careful
  pass to label each section as "framework" or "user", then
  splitting accordingly.  The framework half stays in code (and
  is auto-prepended by the runtime); the user half goes into the
  copied template.
- **Tool default policy.**  Is the rule literally "any forge
  account → coordinator toolset"?  Or do we want richer tagging
  (e.g. "gateway-managed agents get the workflow tools, plain
  agents do not")?  Erring towards the simple rule and revisiting
  if it causes problems.
- **Backwards compatibility.**  Per the Cleanup PR's clean-break
  policy, agent JSONs that still carry `agent_class` would
  hard-error after this change.  The bootstrap is the migration
  path.

### Open questions

- **Custom roles.**  If a user wants a non-coordinator role (e.g.
  a reviewer agent that posts comments but never opens MRs), do
  they get it by writing a different `AGENTS.md` in the agent's
  home, or do we still need a richer role-selection mechanism?
  Leaning towards "just `AGENTS.md`" for v1 and adding more
  structure only if needed.

### Test plan sketch

- New test: bootstrap writes the framework system prompt into the
  agent's home `AGENTS.md`, and a fresh agent loads with the
  expected effective system prompts.
- Update existing `TestProjectCoordinator` tests to assert the
  composed prompt content rather than checking
  `isinstance(agent, ProjectCoordinator)`.

---

## Follow-up 4 — Multi-project agents

### Goal

Let a single agent be responsible for any number of projects.
Routing decides which project a given event belongs to using
real routing rules (per the
[architecture sketch](../aspirational/architecture.md#routing-to-sessions))
rather than `agent.metadata["project"]`.

### Why now

This is the natural endpoint of follow-ups 1 and 3.  Once the
URL-driven credential lookup is in place (1), git identity is
agent-level (2), and roles aren't tied to a single project's
class (3), there's nothing left forcing a 1:1 agent-to-project
relationship except `metadata.project` itself — which exists
solely to disambiguate.

### Implementation sketch

- Routing: the gateway's session-key construction already
  includes the project name as a path segment.  The agent
  receiving the event learns the project the event is for from
  the session key, not from its own metadata.
- Per-session project context: introduce a small helper that
  returns "the project service this session is operating on" by
  consulting the session key.  Replaces every `metadata["project"]`
  read in `thorn.tools.git` and elsewhere.
- Agent-level metadata: `metadata.project` is removed entirely.
  The bootstrap stops emitting it.

### Open questions

- **How does an agent decide which projects to take notifications
  for?**  Two plausible answers:
  - The agent has accounts on the relevant forges; any
    notification on any project hosted on those forges is fair
    game.  Simplest; aligns with how event-source inference
    already works.
  - Explicit project-allow-list or routing rules in agent
    config.  More flexible but reintroduces something
    metadata-shaped.
  Pick the simple answer first; revisit if dogfood says we need
  the more granular control.
- **Memory per project.**  The
  [memory keys](../aspirational/architecture.md#memory-keys)
  story already gestures at per-project memory directories
  (`~/projects/{name}/...`).  This follow-up is the right time
  to make those conventions concrete.

### Test plan sketch

- A single agent with accounts on both `github` and `gitlab`
  receives events for two different projects and produces the
  right session keys / workspaces for each.
- The git identity used for commits is consistent across
  projects (same `git` account; follow-up 2's invariant holds
  when there are multiple projects in play).

---

## What this document deliberately does not cover

- **Routing rule schemas.**  The
  [architecture doc](../aspirational/architecture.md) already
  sketches the routing/memory key shapes; this document defers
  to it rather than restating.
- **Email tooling.**  Follow-up 2 introduces an `email` account
  type as a placeholder, but the actual email tools (SMTP,
  IMAP, etc.) are out of scope here — the account model is
  introduced now so it's available when those tools land.
- **Concurrency / multi-process agencies.**  Orthogonal to the
  config shape, tracked separately in `TODO.md`.

## Cross-references

- The originating plan (canonical source for this work, not edited
  here): `~/.cursor/plans/gateway_config_ergonomics_cleanup_46556edc.plan.md`.
- Architecture sketch for routing rules and memory keys:
  [`docs/aspirational/architecture.md`](../aspirational/architecture.md).
- Implementation anchors in current code:
  - `src/thorn/gateway/_bootstrap.py` — the only first-party
    writer of `gateway.json` and agent JSON; every follow-up
    touches this.
  - `src/thorn/gateway/_agents.py` — `_COORDINATOR_SYSTEM_PROMPT`
    and the `ProjectCoordinator` class (both retired by
    follow-up 3).
  - `src/thorn/tools/git.py` — `_resolve_git_identity` and
    `_git_auth_env_for_current_agent` (rewritten by follow-ups
    1 and 2).
  - `src/thorn/core/_account.py` — `accounts[]` discriminated
    union (extended by follow-up 2).
  - `src/thorn/runtime/_serializer.py` — `_resolve_agent_class`
    (retired by follow-up 3).
