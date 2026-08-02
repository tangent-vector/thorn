Scoped Agent Memory
===================

This document describes a target design for Thorn's long-term memory model.
It is a sibling of `architecture.md` and `coordination.md`: a provisional
"point on the horizon" that should guide incremental work, not a description
of the current implementation.

It does not yet match the implementation.
At the time of writing:

- An agent's memory is effectively its writable home directory.
- All sessions of an agent see the same agent-home memory.
- Session-aligned memory paths are convention only.
- Session state, message history, tool trajectory, and journals are persisted
  as session/runtime state rather than modeled as scoped memory with the same
  privacy rules as the memories derived from them.
- There is no runtime-enforced distinction between memory that belongs to one
  user, project, team, organization, matter, or other privacy boundary.
- There is no supported way for a local agency and a gateway agency to share
  the same user-scoped backing memory store.
- The sandbox/container model is still too coarse: a single container per
  agent cannot safely represent sessions that have different memory scopes
  available to them.

The goal of this document is to describe the shape of a future system where
agents still experience memory primarily as a filesystem, but the runtime owns
scope selection, access policy, backing-store credentials, indexing, auditing,
and concurrent write reconciliation.


Goals and non-goals
-------------------

Goals:

1. Preserve the filesystem illusion for agents.
   Agents should read, write, search, and reorganize memory using ordinary file
   operations whenever possible.
2. Make memory scopes explicit.
   A session should see only the memory scopes that its session shape and
   caller authority allow it to see.
3. Keep the scope vocabulary open-ended.
   Thorn should not hard-code a closed hierarchy such as user/project/team/org.
   Agencies should be able to define scopes such as per-attorney,
   per-client, per-attorney-client, and per-attorney-client-matter without
   asking the framework for new built-in categories.
4. Connect memory access to the coordination model.
   The same tags and key/value pairs that route events to session templates
   should also help select prompt context, memory mounts, sandbox shape, and
   write policy.
5. Allow the same logical memory scope to be used from multiple agencies.
   A user should be able to point a local agency and a gateway agency at the
   same backing store, subject to explicit credentials and policy.
6. Support staged, auditable writes.
   The runtime should be able to inspect, grade, reject, merge, or upstream
   memory edits after an agent writes files, without giving the agent direct
   access to backend credentials.
7. Treat session history as memory.
   The messages, tool calls, intermediate reasoning artifacts, journals, and
   trajectories that led to a memory being written can be as sensitive as the
   memory itself.
   They must be scoped and retained under comparable privacy policy.
8. Leave room for semantic search and other indexes while keeping files as the
   source of truth.

Non-goals:

- Designing a general document database API as the primary agent-facing memory
  interface.
- Freezing one canonical internal layout for every memory scope.
  A few conventions are useful, but topic organization inside a scope should
  stay flexible.
- Solving all credential sharing and legal/compliance policy questions in the
  first implementation.
- Making every backend support every feature.
  Local directories, git repositories, object stores, databases, and virtual
  filesystems will have different consistency and audit tradeoffs.


Core distinction: scopes vs. topics
-----------------------------------

The most important distinction is between **memory scopes** and **memory
topics**.

A **memory scope** is an encapsulation and authority boundary.
It answers questions like:

- Who or what owns this memory?
- Which sessions are allowed to read it?
- Which sessions are allowed to propose or commit writes to it?
- Which backend credentials are needed to materialize it?
- Which audit and review policy applies to changes?

A **memory topic** is just a way of organizing information inside a scope.
It answers a different question: what is this file or directory about?

The difference matters because a scope can contain memories about many topics.
For example:

- A user-scoped memory may contain notes about several projects.
- A project-scoped memory may contain notes about several people.
- An attorney-scoped memory may contain general notes about a client, but that
  does not make the notes client-scoped.
- An attorney-client-matter scope may contain notes about a second project or a
  third party, but the privacy boundary remains the matter scope.

Path layout should reinforce this distinction.
The path segment that identifies a mounted scope should not also be treated as
the first topic segment inside that scope.

Bad shape:

```text
~/memory/project/current/
~/memory/project/slang/
```

In that shape, `current` might mean "the project scope selected for this
session", while `slang` might mean "the topic named Slang".
Those are different concepts occupying the same path position.

Better shape:

```text
/memory/scopes/<scope-mount-id>/
```

The root under `/memory/scopes/` names an available memory scope.
Everything below that root is ordinary content within the scope and may be
organized by topic, date, person, project, matter, or any other convention the
agent and operator find useful.


Shape keys, session keys, and memory scopes
-------------------------------------------

`coordination.md` introduces **memory-scope keys** as structured values shaped
like:

```text
(tags: set[str], kvs: dict[str, str])
```

Those keys are used to match incoming events to session templates and to map
between event context and path-shaped session keys.

This document keeps that concept but draws a sharper line:

- A **shape key** is the structured tags-plus-kvs description of an event,
  session, or template match.
- A **session key** is the path-shaped durable identity of a session.
- A **memory scope** is an actual durable store with access policy and a
  backing backend.

The existing name "memory-scope key" is still understandable because shape
keys are often used to select memory scopes.
Long term, "shape key" may be the better generic name: the same structured
value also drives routing, context gathering, sandbox selection, coordination
policy, and prompt scaffolding.

The framework should not have a fixed list of recognized key names.
If an agency defines templates and memory grants using keys like `peer`,
`project`, `matter`, `client`, `attorney`, `tenant`, or `case`, those keys are
part of that agency's vocabulary.
The runtime's job is to parse, match, validate, and render shapes; it should
not decide that one taxonomy is universal.


Session shape as the common language
------------------------------------

The coordination design already points in the right direction:

1. An incoming event arrives with tags and key/value pairs.
2. The session-template registry selects the best matching template.
3. The template renders a path-shaped session key.
4. The session key can be inverted through the template to recover the
   session's structured shape.

Memory should attach to that same pipeline.
Given the selected session shape, the runtime can derive:

- Which prompt context sources are eligible.
- Which memory scopes should be mounted.
- Which scopes are read-only, staged-writable, or direct-writable.
- Which credentials the gateway may use to materialize backing stores.
- Which sandbox/container shape is required.
- Which write-review policy applies after the agent edits files.

This lets memory access and session routing use one coherent language.
A DM session shaped like:

```text
peer:{peer}/dms/service:{service}
```

might mount memory scopes selected by `peer:{peer}`, by `service:{service}`,
or by a more specific pair such as `peer:{peer}/service:{service}`.

A legal-work session shaped like:

```text
attorney:{attorney}/client:{client}/matter:{matter}
```

might mount all of these, each with different modes:

```text
attorney:{attorney}
client:{client}
attorney:{attorney}/client:{client}
attorney:{attorney}/client:{client}/matter:{matter}
```

Those are scopes, not topic paths.
The matter-scoped store may contain files about multiple topics, and the
attorney-scoped store may contain topic folders for several clients, but their
permission boundaries remain distinct.


Session history as scoped memory
--------------------------------

Session state is part of memory for privacy and audit purposes.
The transcript, tool trajectory, retrieved context, intermediate artifacts,
journal entries, and final memories for a session are not merely operational
logs.
They are often the evidence trail for how a memory was created, and they may
contain the same sensitive facts as the memory itself.

Every session should therefore have a de facto **innermost memory scope**.
That scope is created with the session and is at least as narrow as the
session shape itself.
The runtime writes the session's durable history there, including:

- User, peer, service, and system notifications delivered to the session.
- Model-visible prompt context, with provenance.
- Tool-call trajectory and important tool results.
- Session-authored journal entries.
- Staged memory diffs proposed by the session.
- Runtime review and promotion results for those diffs.

This innermost scope does not have to be exposed to the agent as an ordinary
writable directory in the first implementation.
It may begin as framework-owned session state, but it should be modeled as a
memory scope so that access, retention, export, deletion, indexing, and audit
policy can be reasoned about in the same terms as other memory.

Ancestor and sibling sessions should not automatically receive full access to
another session's history just because they share an agent.
Coordination may require summaries, forwarded notifications, or selected
artifacts to move between sessions, but those transfers should be explicit
memory writes or notifications with their own provenance.

This also affects memory promotion.
If a staged write from a narrow matter scope is promoted into a broader scope,
the review system may need to consult the session-history scope that produced
the edit.
That does not imply the broader scope should receive the raw transcript.
The history remains governed by its own scope policy.


Agent-visible filesystem model
------------------------------

The aspirational sandbox view is:

```text
/agent/home/
/agent/workspace/
/memory/scopes/<scope-mount-id>/
/memory/aliases/<alias> -> ../scopes/<scope-mount-id>
```

`/agent/home/` remains the agent's own home: persona, durable agent-local
notes, journals, skills, and other agent-owned state.

`/agent/workspace/` remains the place for current work.

`/memory/scopes/` contains the memory scopes selected for this execution.
Each child is a mounted or materialized scope.
The child name is a **mount ID**, not a topic.
It may be human-readable, such as:

```text
/memory/scopes/peer=alex/
/memory/scopes/project=slang/
/memory/scopes/attorney=lee+client=acme+matter=patent-42/
```

or opaque, such as:

```text
/memory/scopes/scope-01j9t4k3m7q2/
```

depending on the agency's privacy and readability preferences.

`/memory/aliases/` can provide convenient session-relative names such as
`current-peer` or `current-matter`, but aliases should not be the authoritative
identity of a scope.
They are conveniences layered over `/memory/scopes/`.

For execution venues that cannot provide a real top-level `/memory`, an
equivalent `~/memory-scopes/` view is acceptable as a compatibility fallback.
The important rule is conceptual: the visible roots identify memory scopes,
and topic organization begins inside each scope root.

Each scope root should have a small framework-owned metadata surface that is
visible to the agent, for example:

```text
/memory/scopes/<scope-mount-id>/.thorn/scope.json
/memory/scopes/<scope-mount-id>/MEMORY.md
```

`scope.json` describes the mounted scope, access mode, source shape, and
backend class in machine-readable form.
It is framework-owned and should not be writable by the agent.

`MEMORY.md` is agent-authored scope content.
By convention, it describes the scope's purpose and internal organization.
The runtime may include it in the system prompt when that scope is selected,
subject to context budget and policy.


Memory scope definitions and grants
-----------------------------------

An agency configuration should be able to define memory-scope families and the
rules that grant them to sessions.
The exact schema is deferred, but the conceptual fields are:

- **Name**: stable operator-facing name for the rule.
- **Grant shape**: shape-key pattern a session must match before the scope is
  eligible.
- **Scope key template**: shape-key template that names the actual memory
  scope to materialize.
- **Mount ID template**: how to render a path-safe mount ID.
- **Access mode**: none, read-only, staged-write, direct-write, or another
  explicit mode.
- **Backend**: local directory, git repository, object store, database-backed
  virtual filesystem, or another provider.
- **Credential policy**: which authority may release backend credentials for
  this grant.
- **Context policy**: whether `MEMORY.md`, indexes, summaries, or other files
  from the scope may be included in the system prompt.
- **Write-review policy**: what happens to diffs written by a session.

Illustrative sketch:

```yaml
memory_scopes:
  - name: peer-memory
    grant_shape: "peer:{peer}"
    scope_key: "peer:{peer}"
    mount_id: "peer={peer}"
    access: staged-write
    backend:
      kind: git
      remote: "ssh://memory.example/peers/{peer}.git"

  - name: matter-memory
    grant_shape: "attorney:{attorney}/client:{client}/matter:{matter}"
    scope_key: "attorney:{attorney}/client:{client}/matter:{matter}"
    mount_id: "attorney={attorney}+client={client}+matter={matter}"
    access: staged-write
    backend:
      kind: git
      remote: "ssh://memory.example/matters/{attorney}/{client}/{matter}.git"

  - name: organization-reference
    grant_shape: "organization:{organization}"
    scope_key: "organization:{organization}"
    mount_id: "organization={organization}"
    access: read-only
    backend:
      kind: object-store
      bucket: "thorn-memory-{organization}"
```

This example uses familiar categories, but those categories are not built into
Thorn.
They are names chosen by the agency configuration.

The same mechanism should support a local agency and a gateway agency sharing
one user-scoped memory store: both agencies refer to the same logical scope
and backend locator, but each must have its own explicit credential grant.


Access modes
------------

Memory access should not be modeled as a boolean.
Useful modes include at least:

- **Mounted read-only**: the session can inspect the scope but cannot write.
- **Mounted staged-write**: the session writes to an isolated working view.
  The gateway later reviews and promotes the diff.
- **Mounted direct-write**: writes go directly to the backing store.
  This should be rare and reserved for low-risk or ephemeral scopes.
- **Prompt-only**: selected summaries or retrieved snippets can enter the
  prompt, but the filesystem tree is not mounted.
- **Index-only**: the session can search derived indexes but cannot read the
  full underlying files unless a retrieval result is explicitly promoted.
- **No access**: the scope exists but is not available to this session.

The default for durable long-term memory should be staged-write.
It gives agents the ergonomic illusion of normal file writes while preserving
a review point before those writes become authoritative memory.

Broad or sensitive scopes should be especially conservative.
For example, a session with access to a narrow matter scope may be allowed to
write that matter scope, but attempts to write a broader attorney or
organization scope may be forced into proposal-only review.
If a session has access to known-sensitive scopes, policy may also require
extra scrutiny before it writes any broader shared scope, because leakage from
the narrow scope into the broad one is one of the main risks.


Sandbox and execution boundaries
--------------------------------

The "one sandbox container per agent" model is too coarse for scoped memory.
Two sessions of the same agent may need different visible memory scopes, even
when they share the same base tools and model configuration.

The runtime needs an intermediate execution shape between Agent and Session.
That shape likely includes:

- Agent identity.
- Session template and session shape.
- Workspace root.
- Visible memory scope set.
- Mount access modes.
- Tool and MCP-server set.
- Credential-release policy.
- Environment injection.
- Sandbox image and resource limits.

Two sessions may share a warm sandbox only if their execution shapes are
compatible.
At minimum, the visible memory scope set and access modes must match.
If one session can see a sensitive client scope and another cannot, they must
not reuse the same mutable container state.

This connects directly to the "logical agent workspace" gap called out in
`context-gathering.md`.
That missing rung should probably become the owner of one sandbox/container,
one workspace mount, one memory mount set, one MCP/toolhost connection set, and
one credential envelope.


Backends
--------

The agent-facing abstraction should be filesystem-first, but the backing store
should be pluggable.

Likely backend families:

- **Local directory**: simplest single-machine backend.
  Good for local agencies and tests.
- **Git repository**: strong short-term candidate for durable scoped memory.
  Provides history, diffs, merges, review, and a familiar operator workflow.
- **Object store plus manifest/cache**: useful for shared gateway/local memory
  where S3/R2/MinIO-style storage is operationally easier than git hosting.
- **Database-backed virtual filesystem**: useful when the backend needs
  transactions, permissions, or structured indexing.
- **Remote filesystem mount**: NFS, SMB, 9p, virtiofs, or similar, where an
  operator already has that infrastructure.
- **FUSE-backed virtual filesystem**: strongest filesystem illusion when
  practical, because ordinary reads, writes, renames, and shell tools all see
  the same abstraction.

The backend interface should distinguish:

- Materialize a read view.
- Materialize an isolated writable view.
- Summarize current state.
- Produce a diff from a writable view.
- Review or classify a diff.
- Promote a diff.
- Merge concurrent promoted changes.
- Push or sync to an upstream source of truth.
- Update derived indexes.

Not every backend will support all operations directly.
The framework can emulate some operations by copying to a temporary directory,
but emulation should be explicit because it changes performance and
consistency expectations.


Git-backed scopes
-----------------

A git-backed scope is a promising early implementation because it gives Thorn
many useful behaviors without inventing a complete storage system.

The gateway, not the agent, owns the credentialed git remote.
When a session receives staged-write access to a git-backed scope, the gateway
can:

1. Fetch the latest source-of-truth branch.
2. Create an isolated working copy or worktree for the session.
3. Mount or copy only the working tree into the sandbox.
4. Let the agent edit ordinary files.
5. After the turn or session, compute the equivalent of `git add -A` plus a
   diff.
6. Run policy checks over that diff.
7. Commit accepted changes with session/source metadata.
8. Merge accepted changes back into the scope's integration branch.
9. Push upstream when policy allows.

The sandbox does not need remote credentials.
It also does not necessarily need to see `.git/`.
For many memory workflows, the agent should see a normal directory of files
while the gateway owns the repository mechanics outside the sandbox.

This gives a natural answer to concurrent writes.
Each session writes its own isolated view.
The gateway merges accepted diffs.
Conflicts become explicit review items rather than silent last-writer-wins
overwrites.

Useful commit metadata includes:

- Agent ID.
- Session key.
- Session shape.
- Memory scope key.
- Source event or caller identity.
- Timestamp.
- Review policy result.
- Summary of accepted and rejected paths.

Git is not the final answer for every scope.
Large binary assets, high-write-volume stores, and low-latency shared indexes
may want other backends.
But git is a good first backend because users of a coding agent are likely to
understand its review and recovery model.


FUSE strategy and fallbacks
---------------------------

FUSE is attractive because it preserves the filesystem abstraction even when
the backing store is not a normal directory.
If an agent uses `cat`, `rg`, `sed`, a Python script, or a shell pipeline, all
of those operations see the same mounted memory.

The preferred FUSE shape, when available, is host-side:

1. The gateway mounts the virtual filesystem on the host or sandbox manager
   side.
2. The sandbox container receives that mount as a bind mount.
3. Backend credentials stay with the gateway/FUSE process, not inside the
   container.

This avoids giving the container `/dev/fuse` or broad mount capabilities.
Running FUSE inside the sandbox would usually require extra privileges such as
`CAP_SYS_ADMIN`, which is a poor default for an agent container.

Host-side FUSE still has operational caveats:

- The host OS must support FUSE.
- The container runtime must allow the mounted path to be bind-mounted.
- UID/GID mapping and `allow_other` behavior must be correct so the sandbox
  user can read the mount.
- If scopes are mounted after a container starts, mount propagation has to be
  configured deliberately.
- Windows and some managed container platforms may not support the desired
  shape at all.

Fallbacks should be first-class, not afterthoughts:

- Materialized per-session working directories, with gateway-side sync and
  reconciliation.
- Git working trees or exported checkouts mounted as ordinary directories.
- Object-store sync caches mounted as ordinary directories.
- Read-only snapshots plus a separate write-proposal directory.
- Brain-side or toolhost-mediated file operations for backends that cannot be
  safely exposed as a real filesystem.

The main tradeoff is coverage.
FUSE and ordinary mounts preserve the abstraction for every program the agent
runs.
Toolhost-mediated file operations only preserve it for tools that go through
the Thorn protocol.
That makes FUSE or materialized directories preferable for the default memory
surface, even if internal tools exist for management and search.


Filesystem-first, not filesystem-only
-------------------------------------

Thorn should be all-in on a filesystem-first memory model, but that does not
mean every supporting operation has to be a raw file edit.

Primary agent behaviors should work through files:

- Reading memories.
- Writing notes.
- Reorganizing topic directories.
- Grepping and scanning.
- Using ordinary scripts over mounted memory.

Support operations may still be exposed as tools or commands when that is the
clearer interface:

- List mounted scopes and their access modes.
- Explain why a scope is or is not available to the current session.
- Search semantic indexes.
- Show pending memory diffs.
- Propose promotion of staged writes.
- Report merge conflicts.

Those support operations should point back to files and paths rather than
creating a separate memory API that competes with the filesystem.
For example, semantic search can be exposed as a `memory-search` command or a
small tool that returns scope IDs and file paths.
The retrieved facts still live in files; the search index is derived state.


Context gathering and indexes
-----------------------------

Mounting a scope into the sandbox is not the same as injecting all of its
content into the prompt.

Each granted scope should have a context policy.
Possible policies include:

- Include only the scope root `MEMORY.md`.
- Include `MEMORY.md` files along a topic path selected by the session shape.
- Include recent journal entries from that scope.
- Include retrieval results from a text or semantic index.
- Mount the scope but include nothing automatically.

Prompt assembly must preserve provenance.
If content from multiple scopes is included, the prompt should make clear which
scope and path each contribution came from.
This matters for privacy, for write decisions, and for debugging model
behavior.

Indexes should be scope-aware.
A semantic search over available memory should never retrieve from scopes the
session could not otherwise access.
Search results should carry:

- Scope mount ID.
- Scope key or redacted scope label.
- File path.
- Backend revision, commit, or snapshot ID when available.
- Retrieval score and retrieval mode.

Files remain the source of truth.
Indexes are derived artifacts that can be rebuilt, invalidated, or versioned
per backend.


Write review and promotion
--------------------------

The runtime should assume that agents will sometimes write memory to the wrong
place.
The design should make that recoverable.

For staged-write scopes, the write path is:

1. The session receives an isolated writable view of the scope.
2. The agent edits files normally.
3. The runtime computes a diff.
4. Policy checks classify the diff.
5. Accepted changes are promoted to the authoritative scope.
6. Rejected or questionable changes become review artifacts.

Policy checks may include:

- Path-level checks: broad-scope writes, protected files, generated metadata.
- Content checks: secrets, credentials, PII, privileged information, scope
  mismatch.
- Provenance checks: whether the session had access to more sensitive scopes
  whose content might have leaked.
- Quality checks: whether the edit appears useful, coherent, and appropriate
  to the target scope.
- Conflict checks: whether another session changed the same files.

Rejected changes should not disappear.
They should be available as review diffs, and the agent may receive a
notification explaining what was rejected and why, when policy allows.

For high-trust local-only agencies, staged writes may be promoted
automatically after basic checks.
For shared or sensitive agencies, promotion may require human or operator
approval.


Credentials and authority
-------------------------

Agents should not receive raw backend credentials merely because a memory
scope is visible in their sandbox.

The gateway or agency daemon should own credential release.
A session can receive a mounted memory scope only when all of these checks pass:

- The session shape matches a memory grant.
- The caller or source event is authorized for that grant.
- The agent is authorized for that grant.
- The requested access mode is permitted.
- The backend credential can be released to the gateway-side materialization
  process without exposing it to the sandbox.

This is especially important for user-scoped memory shared across local and
gateway agencies.
The fact that a scope's backing store is the same does not mean every agency,
agent, or session may use the same credentials in the same way.

Credential use should be audited with enough detail to answer:

- Which session caused the scope to be materialized?
- Which user, peer, service, or operator authority allowed it?
- Which backend credential was used, in redacted form?
- Which revision or snapshot was mounted?
- Which writes were proposed and promoted?


Open design questions
---------------------

### Naming the shared shape concept

`coordination.md` currently uses "memory-scope key" for tags-plus-kvs values.
This document suggests "shape key" may be a better generic term because the
same object drives routing, session identity, context gathering, memory scope
selection, and sandbox shape.

We should decide whether to rename the concept before it becomes a concrete
Python type and configuration schema.

### Canonical mount path

`/memory/scopes/<scope-mount-id>/` is the preferred sandbox view in this
document because it makes scoped memory visibly distinct from the agent's
private home.

`~/memory-scopes/<scope-mount-id>/` remains a plausible compatibility shape,
especially for non-container execution venues.
The important decision is not the exact prefix; it is that the first
meaningful path segment identifies a scope rather than a topic.

### Agent home relationship

The agent home should remain agent-owned memory, but it should not be the only
memory store.
Open question: should agent home itself become just another mounted memory
scope, perhaps with a reserved scope key like `agent:{agent}`?

That would simplify the model but may blur the useful distinction between
agent-private state and externally-scoped memory.

### How visible should backend identity be?

Human-readable mount IDs help agents choose where to write.
Opaque mount IDs reduce accidental disclosure of sensitive scope labels in
paths and logs.

The likely answer is policy-dependent:

- Local personal agencies can prefer readable paths.
- Shared or sensitive agencies can prefer opaque IDs plus a visible
  `scope.json`/prompt description that reveals only the labels the session is
  authorized to see.

### How direct should semantic memory feel?

Search should probably be available through commands or tools, but retrieved
content should still be represented as scoped file paths.
The unresolved question is how much the agent should be encouraged to write
structured index hints, summaries, or metadata files directly versus letting
background processes infer them from ordinary notes.


Phasing
-------

A plausible sequence of independently reviewable phases:

- **Phase 1 - Terminology and configuration sketch.**
  Settle the names for shape keys, memory scopes, mount IDs, access modes, and
  memory grants.
  Update aspirational docs so `architecture.md`, `coordination.md`, and this
  document do not contradict each other.

- **Phase 2 - Memory grants without new backends.**
  Teach the runtime to derive a session shape from declared templates and use
  it to select scope grants.
  Initially, granted scopes can be ordinary local directories mounted or
  symlinked into the sandbox.

- **Phase 3 - Session-history scopes.**
  Model each session's transcript, trajectory, journal entries, prompt
  context, and review artifacts as an innermost memory scope, even if the
  initial storage remains the existing framework-owned session state.
  Establish access, retention, export, and deletion policy in the same terms
  used for other scopes.

- **Phase 4 - Execution shape and sandbox partitioning.**
  Introduce the missing rung between Agent and Session that owns one sandbox,
  one workspace mount, one memory mount set, and one credential envelope.
  Prevent sessions with incompatible memory grants from sharing mutable
  sandbox state.

- **Phase 5 - Git-backed staged-write scopes.**
  Add a git backend where the gateway materializes per-session working views,
  computes diffs, runs review policy, commits accepted changes, and merges
  them into the authoritative scope.

- **Phase 6 - Context gathering from selected scopes.**
  Extend prompt assembly so selected scopes can contribute root `MEMORY.md`
  files, topic-specific `MEMORY.md` files, journal entries, and provenance
  labels according to scope policy.

- **Phase 7 - Scope-aware search indexes.**
  Add text and/or semantic indexes as derived state.
  Ensure every retrieval query is filtered by the session's available scopes
  and every result carries scope/path provenance.

- **Phase 8 - Shared and virtual backends.**
  Add object-store, database-backed, remote-filesystem, or FUSE-backed scopes
  once the core policy and staging model are stable.


Provisional decisions
---------------------

- Memory scopes are authority boundaries; topics are content organization
  inside a scope.
- The scope vocabulary is agency-defined, not hard-coded by Thorn.
- Session shape should be the common language connecting routing, context
  gathering, memory selection, sandbox shape, and credentials.
- Agents should see scoped memory through a filesystem-first interface.
- `/memory/scopes/<scope-mount-id>/` is the preferred future sandbox namespace,
  with `~/memory-scopes/` as a compatibility fallback.
- Every session should have an innermost memory scope for its transcript,
  trajectory, journal entries, prompt context, and review artifacts.
- Durable scope writes should default to staged-write, not direct-write.
- Git-backed scopes are a strong early backend because they provide diffs,
  history, review, merging, and operator familiarity.
- FUSE is attractive when practical, but materialized directories and
  git-backed working views must be first-class fallbacks.
- Search and semantic retrieval should be scope-aware derived indexes over
  files, not replacements for files as the source of truth.
