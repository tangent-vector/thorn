Multi-Session Coordination
==========================

This document describes a target design for how a Thorn agent coordinates work that spans multiple sessions, where notifications about world-changing events arrive at one session but the next decision needs to happen elsewhere.
It is a sibling of `architecture.md` and follows the same spirit: a "point on the horizon" that may not match the current implementation but should guide incremental work toward the intended shape.

It does not yet match the implementation.
At the time of writing:

- Session keys use a path-shaped form (`<project>/issue/<n>`, `<project>/change-request/<n>`) but `_routing.py` is hard-coded per forge: there is no template registry that defines what session keys are *allowed* to exist, no machinery for extracting bindings back out of a key, and no notion of one session being an ancestor of another.
- There is no cross-session notification mechanism: events only enter from external `EventSource` instances; there is no internal sender, no timer/heartbeat source.
- Memory is shared agent-wide.
  Session-aligned memory scopes are convention only.
- A single per-noteable session sees both "the notification arrived here" and "decide what to do next", which is the source of two observed failure modes that motivate this design.

Motivating failure modes
------------------------

Two distinct symptoms with a shared root cause have been observed during multi-session dogfooding runs:

- **Silent idle after a downstream change.**
  An agent doing work in a per-noteable session (e.g., for a particular change request) receives a notification that the change was merged.
  Even if the agent has previously planned out a chain of work and recorded that planning to memory, the session that finishes the merge often goes idle rather than initiating the next step, because no part of the system marks that session as responsible for what comes next.

- **Duplicated work.**
  Two distinct sessions independently observe events suggesting the same downstream task should now begin (e.g., one sees an issue close, another sees a change-request merge that closed it).
  Each, in good faith, starts the same work in parallel, with no awareness of the other.

The shared root cause is a routing/ownership mismatch: today's session keys partition by *conversation* (one session per noteable), but the agent's *agendas* are cross-conversation (one agenda per development push, spanning many noteables).
World-changing events land at "where the change happened", not at "where the next decision is made".
There is no session that structurally owns the agenda, and there is no rule that tells per-noteable sessions where to send a "this changed" report so that someone else owns the next decision.

Goals and non-goals
-------------------

Goals:

1. Unblock the agent across multi-step assignments — when work in one session unblocks work in another, the next step actually starts, without user prompting.
2. Reduce duplicated work — when several sessions observe the same underlying change, only one of them dispatches downstream work.
3. Generalize beyond forge-shaped workflows — the same machinery should support DM/chat-driven coordination later, not only forge events.

Non-goals:

- Collapsing the per-noteable session model.
  Per-noteable threading is the right context shape for per-noteable work.
- Making the forge the system of record.
  The agent's memory remains authoritative; forge state is a projection.
- Anything that requires an agency-wide lock or other heavy-weight serialization mechanism.

Core model
----------

### Session keys

Session keys are path-shaped, human-readable strings — the same shape they have today.
Examples:

- `projects/tiny-talk/forks/gitlab-master/issues/42`
- `projects/tiny-talk/forks/gitlab-master/change-requests/47`
- `peers/tess/dms/telegram`

Keys are what surfaces to humans (in logs, in directory paths, in URLs) and to agents (in tool inputs and outputs).
They are the *only* form a session key takes on the wire and on disk.
They map directly onto filesystem paths for workspaces and memory, with no escaping or rewriting.

The framework does *not* require segments to be `<kind>:<value>` pairs.
A session key is just a `/`-separated path whose meaning comes from the **session template** that the key is an instance of, not from the key text itself.
In particular, path components that are not `{}` splices in the template have no semantic significance — they are decoration chosen by the template author for human readability.
In `peers/{peer}/dms/{service}`, the literal `peers` is not linked to the memory-key key `peer`; it is just a readable connective.
A template author who preferred `p/{peer}/d/{service}` would get exactly the same behavior.

Backwards compatibility: existing on-disk sessions already use this shape; nothing about the on-disk layout changes.
What changes is that the framework gains the ability to recognize a key as an *instance of a declared template*, which unlocks ancestor relationships, address validation, and the rest of this design.

### Session templates

A **session template** is the central concept of this design.
A template enumerates a class of allowed sessions: what their keys look like, what incoming events they match, how keys and events map bidirectionally onto each other, and what policy applies to sessions that are instances of it.

A template carries:

- **Name** — a stable identifier (e.g., `project_issue`).
- **Match shape** — the memory-key-shaped pattern this template matches against incoming events.
  A set of required tags, plus a set of required keys where each key's value is either a literal or a `{wildcard}` binding.
  Matches the routing-rule design in `architecture.md`; see *Memory-scope keys* below for the deeper framing.
- **Session key template** — a `/`-separated path string with `{}`-named splices, e.g., `projects/{project}/forks/{fork}/issues/{issue}`.
  Splices must be entire path segments or sub-strings of a segment (`projects/{project}/issues/{fork}-{number}` is valid).
  By convention, splice names match the memory-key key names from the match shape, so `[peer=*, dms, service=*]` pairs with `peers/{peer}/dms/{service}` (not `peers/{p}/dms/{s}`).
  Path text outside splices is decoration; the template author chooses it for readability.
  The bidirectional mapping between memory-scope keys and session keys is entirely determined by this field plus the match shape — no separate "inverse mapping" is needed.
  The framework validates that every wildcard binding in the match shape is spliced into the session key template (so inversion can fully recover those bindings).
- **Parent template** (required for any non-root template) — name of the template whose sessions are this template's parents.
  The framework validates that the parent's match shape is in fact a generalization of this template's match shape (subset of required tags; subset of required keys; wildcards in the parent for any keys this template binds to specific values).
  See *Forward pointers* for the implicit-parent-inference variant we considered and deferred.
- **Heartbeat policy** — none / interval (see *Heartbeats*; deferred to its own push, see *Forward pointers*).
- **Cross-tree-send policy** — `subtree-only` (default) / `siblings-allowed` / `unrestricted`.

The registry of templates is per-agent — see *On-disk layout* below for where the file lives.
The same registry is consulted for both inbound routing and outbound `send_notification` validation.

The set of memory-scope-key key names and tags the framework recognizes is not fixed — it is whatever the agency's templates collectively reference.

### Memory-scope keys

The architecture doc gestures at a concept it calls a *memory key* — a structured value of the form `(tags: set[str], kvs: dict[str, str])`, used to identify the semantic scope of routing decisions and (eventually) memory operations.
This design adopts that concept under the slightly more specific name **memory-scope key**, because in this design it is most useful as the description of a *scope* — a region of conceptual space that some piece of context, policy, or state applies to.

By convention, memory-key keys are always singular nouns — `peer=sean`, `project=slang`, `issue=42` — and wildcard variants reuse those names as the binding identifier: `peer=*` binds to the name `peer`.
Everywhere in this document that a value is "shaped like a memory key" (match shapes, wildcard bindings, typed-shape notation) follows this convention.

The connection to session templates is direct:

- A template's *match shape* **is** a memory-scope-key shape with wildcards.
  An incoming event has an associated memory-scope key (its tags + the kvs the event source attached); inbound routing finds the most specific template whose match shape accepts the event's memory-scope key.
- A template's *session key template* is a path-shaped rendering of that same memory-scope key.
  The splices use the same key names, so a match shape `[peer=*, dms, service=*]` pairs naturally with a session key template like `peers/{peer}/dms/{service}`.

Together the two fields give a clean bidirectional mapping without needing a separately-specified inverse:

- *Memory-scope key → session key*: substitute each wildcard binding from the memory-scope key into the matching splice in the session key template.
- *Session key → memory-scope key*: regex-match the session key against the session key template, pulling the splice values out by name; those become the wildcard-positions of the match shape, and the match shape's own literals and tags fill in the rest.

Non-splice text in the session key template is pure decoration and plays no role in either direction.

#### Typed-shape notation

We do not introduce a `MemoryKey` Python datatype yet; the actual representation stays inside whichever Pydantic models the template registry uses.
What we do adopt is a compact human-readable notation for memory-scope keys — useful in match-shape declarations, in documentation, in logs, and (later) in things like journal-entry tagging or scope-keyed memory operations.
The notation looks like:

- `project:{project}/fork:{fork}/issue:{issue}` — kvs `project={project}, fork={fork}, issue={issue}`, no tags.
- `peer:{peer}/dms/service:{service}` — kvs `peer={peer}, service={service}`, tag `dms`.
- `service:gitlab/dms` — kvs `service=gitlab`, tag `dms`.

Read each segment as either a `kind:value` pair (a kvs entry; `value` may be a `{wildcard}` that binds to the name to its left), or a bare `tag` (an entry in the tags set).

Inside a template definition this notation can be supplied as syntactic sugar for the *match shape* only.
It does *not* derive a session key template; the session key template is always written separately, because its non-splice text is a readability choice that nothing else can make for you.
This keeps the shorthand simple and avoids the pluralization-rule problem altogether.

The same notation will be useful well beyond this design once memory-scope keys are made first-class — see *Forward pointers* for the longer arc.

### The hierarchy of coordination sessions

Three runtime roles for sessions, distinguished by which template they are an instance of:

- **Noteable sessions** — leaves of the tree (`projects/<p>/forks/<f>/issues/<n>` and `projects/<p>/forks/<f>/change-requests/<n>`).
  Their only job is to do their own work and report to their parent.
- **Fork coordinator** — `projects/<p>/forks/<f>`.
  Owns the dependency graph for its fork: which issues are ready, which are blocked, which are in flight, which CRs close which issues.
  Single point of decision for "what should happen next *in this fork*".
- **Project coordinator** — `projects/<p>`.
  Owns the cross-fork agenda: "the umbrella push", sync between forks, project-level priorities.
  Often the human's primary interaction point for setting direction.

Most projects in practice will only have one fork active at a time, so the fork coordinator does the heavy lifting and the project coordinator stays quiet.
Multi-fork projects (e.g., syncing master forks across forges) benefit from having both.

On-disk layout
--------------

This section describes where the new state introduced by this design lives on disk.
It builds on the per-agent directory layout already described in `architecture.md`:

- `agents/<name>/agent.json` — agent's static configuration (framework-managed).
- `agents/<name>/home/` — agent's writable memory.
- `agents/<name>/sessions/` — session state (framework-managed).

The new pieces:

- **`agents/<name>/session-templates.json`** — the agent's session template registry, sitting alongside `agent.json`.
  A single JSON file containing the structured fields of every template the agent recognizes (name, match shape, session key template, parent, policy fields).
  Lives outside the agent's writable home: the framework owns the templates file, the operator/human edits it.
  The agent cannot rewrite its own routing through ordinary memory writes.
  See *Forward pointers* for the question of when (and how safely) we might let agents modify their own templates.

- **Per-template content under the agent's home, at template-scope paths.**
  For each template, the framework derives a *scope path* by replacing every `{...}` splice in the session key template with the sentinel `_`.
  For example, template `projects/{project}/forks/{fork}/issues/{issue}` has scope path `projects/_/forks/_/issues/_/`, and template `projects/{project}/forks/{fork}` has scope path `projects/_/forks/_/`.
  Markdown files at the scope path under `agents/<name>/home/` provide per-template content:
  - `agents/<name>/home/projects/_/forks/_/issues/_/AGENTS.md` — system-prompt guidance for every issue session.
  - `agents/<name>/home/projects/_/forks/_/AGENTS.md` — system-prompt guidance for every fork-coordinator session.
  - `agents/<name>/home/projects/_/AGENTS.md` — system-prompt guidance for every project-coordinator session.

The structured-config-vs-textual-content split keeps the registry file small and JSON-friendly, and means the heavy textual content uses the same Markdown format the agent already reads and writes everywhere else.

The auto-loading convention from `architecture.md` (a session at key `K` includes `MEMORY.md` / `AGENTS.md` files found at prefix paths of `K` under the agent's home) is extended here to walk both axes.
A session at `projects/foo/forks/bar/issues/42` picks up:

- *Instance-axis* files at `projects/foo/forks/bar/issues/42/`, then `projects/foo/forks/bar/`, then `projects/foo/`.
- *Template-scope-axis* files at `projects/_/forks/_/issues/_/`, then `projects/_/forks/_/`, then `projects/_/`.

Layered together these give the agent a clean way to author guidance at the right level of specificity: instance-specific notes for one issue, template-shape guidance for "how I behave on every issue", and ancestor-level guidance for "how I behave when coordinating any fork".

A few small conventions and caveats:

- **Splice collapsing within a segment.**
  When a template has multiple splices in a single segment (e.g., `issues/{fork}-{issue}`), all of them collapse to a single `_` in the scope path (so `issues/_`, not `issues/_-_`).
  Slightly lossy but keeps filesystem paths clean; the small lost expressiveness has not yet mattered for any concrete template we have considered.
- **Sentinel choice.**
  `_` is unambiguously safe on every relevant filesystem and reads naturally.
  The only potential collision is a real entity literally named `_` (e.g., a project named `_`); we accept that as a documented constraint — agencies should not name entities `_`.
- **`.agents/` and friends.**
  Future per-template content (skills directories, tool definitions, etc.) follows the same convention: `agents/<name>/home/<scope-path>/.agents/skills/`, etc.
  This is gestured at as a forward-pointer item; not part of Phase 1.

The notification policy
-----------------------

### Send rule (default: ancestor/descendant only)

An agent may only send a notification to a session that is an ancestor or descendant of the current session.
Ancestry lives at the *template* level (each template declares its parent), and the template tree is projected onto the session-key space using the (match shape, session key template) pair of each template: a session S' is an ancestor of S iff S's template chains up to S''s template, and S''s key is what you get by re-rendering S''s session key template using the splice values extracted from S's key (restricted to the splices S' actually binds).
The cleanest way to think about this is "ancestry of templates first, ancestry of sessions follows mechanically".

So:

- A noteable session may report up to its fork coordinator (and grandparent project coordinator), and may post to existing sibling noteables only if the template explicitly opts into `siblings-allowed`.
- A fork coordinator may dispatch down to any noteable in its fork, and may report up to the project coordinator.
- A project coordinator may dispatch down to any fork coordinator, and to grand-descendant noteables when needed (escape hatch — kept available for cases where the fork coordinator would just be a pass-through).

Disallowing cross-tree hops by default forces work-causing reports through a session that sees the bigger picture, which is precisely what serializes "what's next?" decisions and prevents duplication.
Templates may opt into looser policies when a real workload needs a fast path.

### Address-validation rule (overlaid)

The send rule is layered on top of an address-validation rule:

- Sending to an *existing* session: tool accepts a literal address; checks the runtime address book for binding; checks ancestor/descendant relationship; rejects on either failure.
- Sending to a *new* session: tool requires `template_name` + `bindings`; key is composed from the template; validated against the ancestor/descendant policy *as if the new session existed at that key*.

This combination gives the agent total reach within its subtree (no per-target whitelisting needed) while making cross-tree communication a deliberate template-level decision rather than a per-call one.

Tools
-----

Three new agent-facing tools, modeled after the existing inbox tools:

- `list_sessions(pattern: str = "./**", *, addressable: bool = False)` — lists existing sessions matching a glob-like pattern interpreted relative to the current session.
  `.` is the current session, `..` is its parent, `./**` is the current session and all its descendants, `../*` is the current session and its siblings, and so on.
  Patterns are over clean session-key paths (e.g., `projects/foo/**`, `./forks/*`).
  Results are annotated with whether each session is *addressable* from the current session under the active policy; passing `addressable=True` filters out non-addressable results.
  This tool also surfaces templates whose target key would lie under the pattern (so the agent can see what new sessions could be created by sending to them); template results are similarly annotated/filtered by addressability.
- `send_notification(target_or_template, content, *, bindings=None, metadata=None)` — posts a fresh notification.
  Sender (the current session address) is stamped into the notification automatically; the receiver sees it in the rendered prompt as "from session `<key>`".
- `forward_notification(item_id, target_or_template, *, bindings=None, notes="")` — moves an existing inbox item via the existing handling-transition machinery.
  Original source is preserved; the forwarder is appended to the forward trail (see *Forwarding semantics*, below).
  This is the report-up channel's primary tool: a noteable session marks its merged-MR notification handled-by-forwarding to its fork coordinator.

All three honor the policy.
All three are added to every agent's tool set the same way the existing inbox tools are today.

### Forwarding semantics

A notification carries:

- An immutable `source` — the inbound `EventSource` name (e.g., `"gitlab-todos"`) when it first entered the system, or the originating session address when it was created via `send_notification`.
  This identifies where the underlying world-event came from and is never overwritten by forwards.
- A mutable `forward_trail: list[ForwardEntry]` — empty for un-forwarded items, appended on each forward.
  Each `ForwardEntry` carries `(forwarder_session_address, notes, timestamp)`.

The rendered prompt for a forwarded item shows the original source plus a compact trail, e.g.:

> Originally from `gitlab-todos`.
> Forwarded by `projects/foo/forks/gitlab-master/change-requests/47` ("MR 47 merged; closes #23").
> Forwarded by `projects/foo/forks/gitlab-master` ("rolling up to project for cross-fork review").

For dispatch, the *current target* address is what matters for routing decisions; the trail is informational.
RSVP-style return paths (today's dispatch redirect to original sender) follow the original source, not the trail.

Prompt conventions
------------------

Tool affordances are not enough on their own; the agent needs to know its role.
The system prompt for each session includes:

- Its own session address.
- The name of the template it is an instance of, plus its parent address (if any).
- A role statement keyed to the template.
  Noteable sessions are told "do not initiate work outside your scope; report changes upward".
  Coordinator sessions are told "you are the dispatcher for this scope; serialize incoming reports into outgoing dispatches".

Children of the current session are deliberately *not* enumerated in the system prompt by default.
A coordinator with many children would otherwise pay prompt-bloat cost on every round.
The agent uses `list_sessions("./*")` (or a deeper pattern) when it needs visibility into its descendants.
If experience shows that some level of children-in-prompt visibility is needed for steady-state work, we will add a bounded "up to N most-recently-active children" injection rather than enumerating all of them.

These prompt conventions are not enforcement; they are the convention that makes the design work.
The send-rule policy enforces *reachability*; the prompt convention enforces *responsibility*.

Heartbeats
----------

Optional, low-rate, per-template.

A timer-driven internal `EventSource` produces heartbeats for sessions whose template declares an interval.
By default:

- Coordinator sessions: low-rate heartbeat (e.g., every few hours) — backstop in case a report-up was lost or never sent.
- Noteable sessions: no heartbeat — they only run when there is something to do.

Heartbeat content is a generic "scan your scope for assignments that may have stalled" prompt; the coordinator session is in the right scope to do something useful with it.

Heartbeats are a backstop, not load-bearing.
The report-up channel is the steady-state mechanism.

Forge state as projection
-------------------------

Coordinator sessions are encouraged (via prompt + tool affordances) to mirror their dependency graph onto the relevant umbrella issue: a checklist in the umbrella body, labels on sub-issues (`ready`, `in-progress`, `blocked`), comments on transitions.
This gives human users observability without making the forge the source of truth.
The coordinator's `MEMORY.md` (under `~/projects/<p>/forks/<f>/` per the architecture doc's session-aligned memory scopes) holds the authoritative graph.

Memory layout
-------------

Memory layout mirrors the session-key paths directly:

- `~/projects/<p>/MEMORY.md` for project-level state (the umbrella push, cross-fork concerns).
- `~/projects/<p>/forks/<f>/MEMORY.md` for fork-level state (the dependency graph, in-flight work).
- `~/projects/<p>/forks/<f>/issues/<n>/MEMORY.md` for per-noteable notes.

The architecture-doc convention of automatically loading `MEMORY.md` files from prefix paths gives each session the right context for free.
A noteable session sees its own notes plus its fork's plus its project's.

These per-instance `MEMORY.md` files coexist with the per-template-scope `AGENTS.md` files described in *On-disk layout*: the loader walks both axes, so a session sees instance-axis memory at literal-key paths and template-axis guidance at scope paths (with wildcards as `_`).

How the design resolves the failure modes
-----------------------------------------

- **Problem 1 (silent idle after merge):**
  The change-request session, on receiving the MR-merged notification, marks the item handled by forwarding it to its fork coordinator with a note ("MR 47 merged; closes #23").
  The fork coordinator wakes, looks at its dependency graph, dispatches a `start_work` notification to whichever issue session is now ready (or escalates to the project coordinator if cross-fork).
  The change-request session goes idle by design — it has no further responsibility.
  The wake-up of the next-step session is structural, not reliant on the change-request session "remembering" what to do next.

- **Problem 2 (duplicated work):**
  The fork coordinator is the single point of dispatch within a fork.
  Even if two distinct noteable sessions independently observe events suggesting "Z is unblocked", both can only *report up*.
  The fork coordinator's per-session serialization (one prompt round at a time, enforced by the runtime scheduler) means at most one start-work decision happens per round.
  The send-rule policy makes cross-noteable shortcuts impossible without an explicit template opt-in.

Trade-offs and risks
--------------------

- **Coordinator hot spots.**
  A slow round in a fork coordinator stalls dispatch for that fork.
  This is the cost of resolving problem 2; the per-agent concurrency cap keeps it scoped (other forks and other projects still run).
  Acceptable.
- **Two-hop latency.**
  Report-up through the coordinator adds a hop versus a peer-to-peer fast path.
  The default policy disallows the fast path; templates can opt in if a real workload demands it.
- **Convention dependence.**
  The "do not initiate" / "you are the dispatcher" distinction lives in prompts.
  If the agent ignores it, problem 2 can re-emerge.
  Mitigation: short, repeated, role-keyed prompt content; failure-mode review during dogfooding.
- **Migration.**
  Existing sessions on disk already use a path-shaped key; no on-disk rewrite is needed.
  What does change is the *interpretation* of those keys: a legacy session whose key matches a newly-declared template becomes an instance of that template, and inherits the template's parent, policy, etc. automatically.
  A legacy session whose key matches no template stays unattached to the hierarchy — it can still receive notifications via its existing inbox, but is excluded from ancestor-based send/forward operations until a template covering it is declared.
- **Multi-coordinator-per-agent.**
  A single agent will now host O(projects × forks) coordinator sessions in addition to the noteable ones.
  Today's per-agent scheduler should handle this fine; worth measuring with a multi-project agency to confirm.

Phasing
-------

A proposed sequence of independently reviewable phases.
Each is later expanded into its own plan document.

- **Phase 1 — Session template registry.**
  Replace the hard-coded per-forge routing functions with declared templates, each specified by a match shape and a session key template, plus an explicit parent template (required for any non-root template) and the basic policy fields.
  Templates live in a per-agent `agents/<name>/session-templates.json` (alongside `agent.json`) and are loaded by the framework at startup.
  Extend the auto-loading convention so per-template `AGENTS.md` content lives at template-scope paths under the agent's home (e.g., `agents/<name>/home/projects/_/forks/_/issues/_/AGENTS.md`).
  Existing path-shaped session keys keep their on-disk form; what they gain is template-based interpretation that yields parent/ancestor/descendant relationships.
  No agent-visible behavior change in terms of new tools yet — the visible change is that the system-prompt scaffolding can be authored at template scope.
- **Phase 2 — Cross-session send/forward tools with policy enforcement.**
  Build `list_sessions`, `send_notification`, `forward_notification`.
  Sender stamping.
  Address validation against templates and the address book.
  Tests for ancestor/descendant policy.
- **Phase 3 — Coordinator sessions (fork-level first).**
  Add fork-coordinator templates.
  Update prompt conventions for noteable vs coordinator roles.
  Establish the report-up convention via `forward_notification`.
  End-to-end vertical slice on one project.
- **Phase 4 — Project-coordinator sessions.**
  Same machinery, one level up.
  Forge mirroring of the agenda graph onto umbrella issues.
- **Phase 5 — Timer/heartbeat event source.**
  Optional backstop.
  Per-template interval declaration.

Ancestor-existence invariant
----------------------------

Templates declare a `parent` template name; that declaration defines a tree of templates.
At session-creation time, the runtime enforces the invariant that **if a session exists, then every ancestor session along its template chain also exists**.

Concretely: when an inbound event causes the runtime to materialize a session for template T with bindings B, the runtime first ensures that the session for T's parent template (rendered from the subset of B that the parent template binds) exists, recursing up to a template with no parent.
Each ancestor session is created with an empty inbox; it does not receive a notification of its own.
This means the project coordinator and fork coordinator are always alive by the time a noteable session is created under them, ready to receive forwarded notifications without a special bootstrap step.

This invariant resolves what would otherwise be a separate "how do we bootstrap the project coordinator?" question: there is no separate bootstrap step, because ancestor materialization is part of every session creation.

The invariant assumes a strict tree of templates.
A future cross-cutting coordinator (e.g., a `topics/authentication` session that should be a parent of issue sessions across multiple projects) would require generalizing to a DAG.
That generalization is deferred: every concrete coordination role identified so far (project, fork, noteable, peer-DM) fits a tree, and the simplification is worth keeping until a real workload demands otherwise.

Resolved decisions
------------------

Decisions made during the design discussion that are now part of the spec above:

- **Session keys use the clean human-readable form** (e.g., `projects/foo/forks/gitlab-master/issues/42`).
  Typed `kind:value` notation is descriptive only, used inside template definitions as syntactic sugar for match shapes; it does not appear in keys themselves, on the wire, or on disk.
- **Session-key text outside splices has no semantic significance.**
  A session key is a `/`-separated path; path components that are not `{}` splices in the template are decoration chosen by the author for human readability, with no framework-level meaning.
  This eliminates any need for pluralization rules, derived key templates, or other machinery that tries to guess at session-key text from memory-key shape.
- **Templates are specified by a (match shape, session key template) pair.**
  Both directions of translation between a memory-scope key and a session key are derivable from that pair; no separately-specified inverse mapping or key parser is needed.
  The framework validates that every wildcard binding in the match shape is spliced into the session key template.
- **Memory-key and session-key-splice names match, and are singular.**
  Memory-key keys use singular nouns (`peer`, not `peers` or `p`), and session key templates reuse those same names for their splices (`peers/{peer}/dms/{service}`).
- **Issue and change-request as siblings under the fork.**
  Both `projects/<p>/forks/<f>/issues/<n>` and `projects/<p>/forks/<f>/change-requests/<n>` are children of `projects/<p>/forks/<f>`.
  No PR-tracker level between fork and CR.
- **Per-template cross-tree send policy.**
  Each template declares its own `subtree-only` / `siblings-allowed` / `unrestricted` policy at registration.
  Default is `subtree-only`.
- **Forwarding trail semantics.**
  See *Forwarding semantics* under *Tools*: original `source` is immutable; each forward appends a `ForwardEntry` to a `forward_trail` list; RSVP-style return paths follow the original source.
- **Ancestor sessions are auto-materialized** with the descendant, per the invariant above.
- **Listing sessions is a general-purpose `list_sessions(pattern, addressable=False)` tool** with glob-relative patterns and an addressability annotation/filter, rather than a dedicated `list_addressable_sessions` tool.
- **Children are not enumerated in the system prompt by default.**
  Bounded enumeration may be added later if experience demands it.
- **Memory-scope keys are a first-class concept**, even though no `MemoryKey` Python datatype is introduced in the initial phases.
  A template's match shape is a memory-scope key shape; the typed-shape shorthand (`project:{project}/fork:{fork}/issue:{issue}`) is the human-readable notation for memory-scope keys, used only for match shapes — the session key template is always written separately.
- **Explicit parent declarations only.**
  A template that has any parent must declare that parent by name; the framework validates the declared parent's match shape is in fact a generalization of the child's.
  Implicit inference from match-shape generality is recorded as a forward-pointer item but not built.
- **Per-agent template registry at `agents/<name>/session-templates.json`** (alongside `agent.json`, outside the agent's writable home).
  Single JSON file containing structured fields only; multi-line textual content (`AGENTS.md`, etc.) lives separately in the agent's home at template-scope paths.
- **Template-scope-path convention for per-template content.**
  Each template's splices-as-`_` form names a directory under the agent's home where `AGENTS.md` (and future per-template assets) live; the `MEMORY.md` / `AGENTS.md` auto-loading convention walks both the instance axis and the template-scope axis.

Forward pointers
----------------

These are larger themes that arose during design discussion but are deliberately *not* part of the spec above.
They are recorded here so they are not lost; each is likely to want its own design pass and its own push.

- **Templates as the unit of role specialization (replacing `Agent` sub-classes).**
  A session template is a natural place to attach not just routing/policy info but also the system-prompt content, skills, and tool sets that should apply to sessions of that kind.
  In the limit, this would replace the current `Agent` sub-class mechanism: a single underlying agent "wears many hats" by virtue of which template a given session is an instance of, with neither user nor agent needing to think about class hierarchies.
  This is a much bigger reframing than what this doc currently captures, and we should settle the smaller mechanics first before pulling it in.

- **Templated rendering and richer per-template content.**
  The *On-disk layout* section already places per-template `AGENTS.md` at template-scope paths under the agent's home.
  Two natural extensions remain for later: an `AGENTS.md.jinja` variant that lets a session's actual bindings appear in the rendered prompt (`{{project}}`, `{{fork}}`, etc.), and per-template skills/tools under a sibling `.agents/` directory at the same scope path.
  Both depend on the templates-as-roles reframing above and should be designed jointly with it.

- **Memory-scope keys as a first-class datatype.**
  This design treats memory-scope keys (`(tags, kvs)`) as a concept and adopts the typed-shape notation, but does not yet introduce a `MemoryKey` Python datatype.
  A future push could promote it: a real type with parse/format from typed-shape strings, used to tag journal entries, to scope free-form memory blobs, and to drive a "find me everything I know about scope `peer:{peer}`" semantic search across the agent's memory.
  Scope-keyed memory becomes especially valuable for cross-cutting concerns (a particular human collaborator, a particular service) that don't fit cleanly into the session-tree shape.

- **Agent self-modification of session templates.**
  In the initial phases the template registry (`session-templates.json`) is operator-managed and outside the agent's writable home — the agent can author per-template *content* but cannot reshape the routing.
  A later push could give the agent tools to propose new templates, edit existing ones, or split/merge them, with appropriate guardrails (e.g., a "draft" template registry the agent writes to, with a separate trust gate before draft templates become live).
  This is on a different axis from "letting the agent edit its `AGENTS.md`" because routing changes have system-wide effects: they redirect future events, change ancestry relationships, and change which sessions other sessions are allowed to address.
  Worth doing eventually, but not before we have lived with operator-managed templates long enough to know what kinds of changes the agent actually needs to make.

- **Implicit parent inference from match-shape generality.**
  A template whose match shape is a strict subset of another's (subset of required tags, subset of required keys, with the parent using wildcards or the same literal value for any keys the child binds) is a natural ancestor candidate.
  The framework could infer the parent template automatically rather than requiring an explicit `parent:` declaration.
  Open issues with the implicit approach: tie-breaking when multiple templates are equally-general; edge cases like a template with no required tags or keys claiming parenthood of everything; the risk that a small change to one template's match shape silently re-parents another.
  Provisional preference: keep explicit `parent:` as the authoritative mechanism, but have the framework *validate* that the declared parent's match shape is in fact a generalization of the child's — and consider implicit inference later if explicit declarations turn out to be tedious in practice.

- **Scheduled tasks (heartbeats are one case).**
  The heartbeat machinery sketched in the *Heartbeats* section is really one specific case of a more general "scheduled tasks" feature: per-session and per-template lists of tasks, each carrying either a one-shot UTC timestamp or a `cron`-style recurrence, plus a prompt string that becomes the content of the resulting notification.
  Scheduled tasks should be configurable as ordinary files in the agent-accessible memory hierarchy (e.g., `timers.yaml` under the relevant session-key path or template-shape path) so that the agent itself can add, remove, and edit them via ordinary file-write tools.
  The runtime watches (or periodically scans) those files and fires notifications accordingly.
  This is big enough that it deserves its own push.
  When it lands, the heartbeat mechanism in this doc collapses into "a default scheduled task on coordinator templates".

- **Template inheritance of configuration.**
  If templates form a tree (or a DAG) and carry meaningful policy/configuration, it is natural for a child template to inherit configuration from its ancestors.
  Whether template-inheritance and session-ancestry should align mechanically (a child template's session inherits from its parent template's *corresponding* session) or independently (a child template inherits configuration from its parent template, full stop) is a question worth working through.
  Do not need to answer it for the initial phases.
