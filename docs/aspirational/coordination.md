Multi-Session Coordination
==========================

This document describes a target design for how a Thorn agent coordinates work that spans multiple sessions, where notifications about world-changing events arrive at one session but the next decision needs to happen elsewhere.
It is a sibling of `architecture.md` and follows the same spirit: a "point on the horizon" that may not match the current implementation but should guide incremental work toward the intended shape.

It does not yet match the implementation.
At the time of writing:

- Session keys use untyped, positional segments (`<project>/issue/<n>`, `<project>/change-request/<n>`) and `_routing.py` is hard-coded per forge.
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

### Typed-segment session keys

Today: `tiny-talk/issue/7` (forge-agnostic but positional).

Proposed: `project:tiny-talk/fork:gitlab-master/issue:42`.
Each segment is `<kind>:<id>`.

Properties this gives us:

- **Structural ancestry.**
  The parent of any session is its key with the rightmost segment dropped: `project:tiny-talk/fork:gitlab-master` is the parent of the issue session, and `project:tiny-talk` is the parent of the fork.
  No template lookup needed.
- **Cross-fork visibility.**
  A project session can address two of its fork descendants (e.g., when `gitlab-master` and `github-mirror` need to be kept in sync) without any of them needing knowledge of each other.
- **Free routing for the file system.**
  The runtime already turns a session key into a workspace directory.
  The typed form maps cleanly onto a directory layout — `agents/<id>/sessions/project:foo/fork:gitlab-master/issue:42/` — and sidesteps any ambiguity about what an intermediate path means.
- **Backwards compatibility.**
  Existing on-disk sessions keep their legacy keys; the framework treats untyped legacy keys as opaque (no parent computable) and refuses to fit them into the hierarchy.
  Migration is opt-in per session.

Constraint: segment kinds are drawn from a declared vocabulary (`project`, `fork`, `issue`, `change-request`, `peer`, `dms`, ...).
Unknown kinds are rejected at parse.

### Routing template registry

The hard-coded per-forge routing functions become data-driven.
A template is a small record:

- Name (e.g., `project_issue`).
- A match shape: required tags + key/value patterns on incoming events (matches the design in `architecture.md`).
- A session-key template: `project:{p}/fork:{f}/issue:{n}`.
- An optional `parent` template name, which lets us declare hierarchies explicitly when multiple templates produce the same logical level.
- A heartbeat policy (none / interval) for sessions matching the template.
- A cross-tree-send policy: `subtree-only` (default) / `siblings-allowed` / `unrestricted`.

Templates are declared in `gateway.json` (or a sibling `routing.json`) alongside forges and projects.
The same registry is consulted for both inbound routing and outbound `send_notification` validation.

### The hierarchy of coordination sessions

Three runtime roles for sessions, distinguished by the depth at which the template lands them:

- **Noteable sessions** — leaves of the tree (`project:.../fork:.../issue:N` and `change-request:N`).
  Their only job is to do their own work and report to their parent.
- **Fork coordinator** — `project:.../fork:F`.
  Owns the dependency graph for its fork: which issues are ready, which are blocked, which are in flight, which CRs close which issues.
  Single point of decision for "what should happen next *in this fork*".
- **Project coordinator** — `project:P`.
  Owns the cross-fork agenda: "the umbrella push", sync between forks, project-level priorities.
  Often the human's primary interaction point for setting direction.

Most projects in practice will only have one fork active at a time, so the fork coordinator does the heavy lifting and the project coordinator stays quiet.
Multi-fork projects (e.g., syncing master forks across forges) benefit from having both.

The notification policy
-----------------------

### Send rule (default: ancestor/descendant only)

An agent may only send a notification to a session that is an ancestor or descendant of the current session in the typed-key tree.
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
> Forwarded by `project:foo/fork:gitlab-master/change-request:47` ("MR 47 merged; closes #23").
> Forwarded by `project:foo/fork:gitlab-master` ("rolling up to project for cross-fork review").

For dispatch, the *current target* address is what matters for routing decisions; the trail is informational.
RSVP-style return paths (today's dispatch redirect to original sender) follow the original source, not the trail.

Prompt conventions
------------------

Tool affordances are not enough on their own; the agent needs to know its role.
The system prompt for each session includes:

- Its own session address (typed form).
- Its parent address (if any).
- A role statement keyed to the session's deepest typed segment.
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

Falls out of the typed-key form:

- `~/projects/<p>/MEMORY.md` for project-level state (the umbrella push, cross-fork concerns).
- `~/projects/<p>/forks/<f>/MEMORY.md` for fork-level state (the dependency graph, in-flight work).
- `~/projects/<p>/forks/<f>/issues/<n>/MEMORY.md` for per-noteable notes.

The architecture-doc convention of automatically loading `MEMORY.md` files from prefix paths gives each session the right context for free.
A noteable session sees its own notes plus its fork's plus its project's.

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
  Existing sessions on disk use the old key shape.
  Acceptable to leave them on the legacy regime; new sessions adopt typed segments going forward.
  We add a small adapter so the inbox tools work on both.
- **Multi-coordinator-per-agent.**
  A single agent will now host O(projects × forks) coordinator sessions in addition to the noteable ones.
  Today's per-agent scheduler should handle this fine; worth measuring with a multi-project agency to confirm.

Phasing
-------

A proposed sequence of independently reviewable phases.
Each is later expanded into its own plan document.

- **Phase 1 — Typed-segment keys + template registry.**
  Replace hard-coded routing with declared templates.
  Add a key parser that yields parent/ancestor/descendant relationships.
  Backwards-compat for legacy keys.
  No agent-visible behavior change yet.
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

Concretely: when an inbound event causes the runtime to materialize a session for template T with bindings B, the runtime first ensures that the session for T's parent template (with B restricted to the parent template's parameters) exists, recursing up to a template with no parent.
Each ancestor session is created with an empty inbox; it does not receive a notification of its own.
This means the project coordinator and fork coordinator are always alive by the time a noteable session is created under them, ready to receive forwarded notifications without a special bootstrap step.

This invariant resolves what would otherwise be a separate "how do we bootstrap the project coordinator?" question: there is no separate bootstrap step, because ancestor materialization is part of every session creation.

The invariant assumes a strict tree of templates.
A future cross-cutting coordinator (e.g., a `topic:authentication` session that should be a parent of issue sessions across multiple projects) would require generalizing to a DAG.
That generalization is deferred: every concrete coordination role identified so far (project, fork, noteable, peer-DM) fits a tree, and the simplification is worth keeping until a real workload demands otherwise.

Resolved decisions
------------------

Decisions made during the design discussion that are now part of the spec above:

- **Issue and change-request as siblings under the fork.**
  `issue:N` and `change-request:N` are both children of `project:.../fork:F`.
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

Things still worth pinning down before Phase 1
----------------------------------------------

- Whether the typed-segment vocabulary (`project`, `fork`, `issue`, `change-request`, `peer`, `dms`) is fixed in the framework or extensible by an agency's configuration.
  Most likely answer: small fixed set in the framework, with templates referencing them by name; agency-defined kinds added later if a real need appears.
- The exact on-disk file layout for declared templates and template parents (a separate `routing.json` vs. a section inside `gateway.json`).
  This is a small concern but the answer affects Phase 1's data model and the shape of `_config.py` loading code.
- A migration strategy for the small number of existing on-disk sessions on legacy keys: do we leave them on the legacy regime indefinitely, or write a one-shot `thorn migrate-sessions` command that rewrites them into typed form?
  Provisional take: leave them; new sessions adopt typed segments; the legacy adapter is small.
