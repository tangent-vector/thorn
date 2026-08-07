# Current Thorn Architecture

This document describes the architecture implemented by Thorn today. It is a
present-tense reference, not a target design. Documents under
[`docs/aspirational/`](aspirational/) describe possible future systems and do
not override this one.

## Domain vocabulary

The following distinctions are architectural policy:

- An **agent** is one configured participant with an identity, policy, durable
  state, sessions, and a tool/runtime context.
- An **agency** is a configured collection of agents and their associated
  state. “Agency” refers both to its persisted representation and to the active
  runtime embodiment of that configuration.
- An **agency home** is the on-disk root for configuration and framework-owned
  state. The default local agency home is `~/.thorn`.
- A **session** is a durable conversation and unit of serialized prompt work
  for one agent. Different sessions belonging to an agent may run concurrently.
- A **gateway** is the service-facing operating mode of a running agency. The
  agency runs as a daemon and remote peers interact with it indirectly through
  configured services, currently GitHub or GitLab.

A gateway therefore hosts or exposes an agency. It is not a synonym for the
agency, the agency home, the configuration file, or the runtime shared with CLI
operation.

“Gateway” remains Thorn's current public label for this remote daemon-service
mode, but the name may be revisited. Its possible replacement is a naming
question; it is not a reason to broaden the term until it becomes
interchangeable with “agency.”

Existing identifiers such as `gateway.json`, `GatewayConfig`, `Gateway`, and
the `thorn.gateway` package predate this clarified vocabulary. They identify
real compatibility surfaces and implementation modules, but they should not be
used to infer the domain model. Their responsibilities need deliberate review
before any compatibility-breaking rename.

## One runtime, two interaction modes

Every `thorn run`, `thorn chat`, and `thorn serve` invocation creates a
`Runtime`. The runtime owns the provider connection, agent and session loading,
the session store, event delivery, service registry, and tool-executor
lifecycle.

### Local CLI interaction

`thorn run` and `thorn chat` are direct interactions with a locally running
agency:

1. Thorn selects `~/.thorn` (or `--agency`) as the agency home and the current
   directory (or `--workspace`) as the workspace.
2. It loads or creates the local CLI agent and persists it in the agency home.
3. It creates a fresh `cli/<workspace>/<id>` session unless `--resume` names an
   existing one.
4. The user's turn is posted to the session inbox and dispatched through the
   same per-agent scheduler abstraction used by the gateway.
5. The session invokes the LLM and dispatches each tool by its declared venue:
   in-process tools run in the trusted CLI process, while sandbox tools go
   through the subprocess toolhost.

Local CLI configuration is currently partly implicit: provider settings come
from command environment, and no `agency.yaml` is required for the first run.
The resulting agent and session state are nevertheless an agency, not a
separate persistence model.

The local CLI's default sandbox-venue toolhost is a subprocess running with the
invoking user's OS authority. This provides process separation and the common
tool protocol, but not a security boundary against hostile tool calls.

### Gateway interaction

`thorn serve` runs an agency as a long-lived service:

1. It loads exactly one supported agency configuration file from the agency
   home and loads each persisted agent identity.
2. It creates configured services and infers forge event sources from the
   agents' accounts.
3. It reconstructs in-flight work and repairs recoverable crash-leftover inbox
   state before polling begins.
4. Event sources hand raw GitHub or GitLab events to the gateway orchestration
   layer.
5. The formatter resolves the actor, enforces peer-trigger policy, wraps
   external content, and routes accepted work to an agent/session inbox.
6. A per-agent scheduler allows different sessions to run concurrently while
   serializing work within each session.
7. The agent loop calls the configured LLM and dispatches each tool according
   to its declared venue: sandbox tools go through the agent's toolhost, while
   in-process tools run in the trusted host process.

The default gateway configuration runs the LLM-facing process and in-process
tools on the host, and runs the sandbox toolhost inside a per-agent OCI
container. A bundled credential broker injects service credentials into
matching outbound requests without putting literal forge tokens in
agent-visible state. Operators can explicitly choose a subprocess toolhost or
an externally managed broker, with the resulting operator-dependent boundaries
described in the [threat model](threat-model.md).

## Component flow

```text
 Direct CLI turn                          Remote forge activity
        |                                         |
        |                                  event source polling
        |                                         |
        +-------------------+   +-----------------+
                            v   v
                 +------------------------+
                 | Agency runtime         |
                 |                        |
                 | service registry       |
                 | agents and sessions    |
                 | address book           |
                 | durable inboxes        |
                 | per-agent schedulers   |
                 +-----------+------------+
                             |
                   prompt/context assembly
                             |
                    +--------v--------+       +------------------+
                    | LLM-facing      +------>| LLM provider API |
                    | agent loop      |       +------------------+
                    +--------+--------+
                             |
                    venue-selected dispatch
                 +-----------+-----------+
                 |                       |
          IN_PROCESS tools          SANDBOX tools
                 |                       |
       +---------v----------+   +--------v-------------+
       | trusted host       |   | per-agent toolhost   |
       | forge, peer, inbox |   | subprocess or OCI    |
       | and control tools  |   | container            |
       +---------+----------+   +-----+-----------+-----+
                 |                    |           |
           service APIs          workspace   service APIs
                                                  ^
                                                  |
                                        credential broker
                                        (gateway default)
```

The “brain” comprises the runtime, prompt/context assembly, and LLM-facing
agent loop. It remains outside the tool sandbox. Tools tagged `IN_PROCESS`,
including forge, peer, inbox, and coordination controls, also remain outside
the sandbox because they need trusted runtime state. Tools tagged `SANDBOX`,
including filesystem, shell, journal, and optional MCP tools, cross a typed
protocol into the toolhost.

Container mode gives the toolhost writable mounts for the agent-visible home
and workspace plus a framework-owned control channel. Brokered operation adds
read-only CA and generated Git-configuration mounts; development mode can add a
read-only Thorn source mount. The [threat model](threat-model.md) defines which
of these boundaries are security-relevant.

## Persistent layout

An agency uses two roots:

- The **agency home** stores the agency configuration, agent identities,
  agent-authored home content, sessions, inboxes, queues, and other
  framework-owned state.
- The **workspace root** stores agent workspaces and the control files used to
  communicate with toolhost processes or containers.

A simplified layout is:

```text
<agency-home>/
  agency.yaml                    # preferred config name when a file is used
  agents/<agent-id>/
    agent.json                   # framework-owned identity and account refs
    home/                        # agent-visible durable files
    sessions/<session-key>/      # framework-owned history and inbox state
  services/                      # framework-owned durable service queues

<workspace-root>/
  agents/<agent-id>/
    workspace/<session-key>/     # agent-visible session work
    control/                     # framework/toolhost rendezvous
```

Compatibility configuration names `agency.json`, `gateway.yaml`, and
`gateway.json` are also accepted. An agency home must contain no more than one
of the supported files. CLI-only use may have no agency configuration file at
all because its current configuration is constructed from CLI defaults and the
environment.

Only the agent `home/`, workspace, and toolhost control subtrees are writable
host-directory mounts in the normal container sandbox. Brokered operation can
also mount a CA certificate and generated placeholder-only Git configuration
read-only. Agent identities, session histories, durable inboxes, and service
queues remain framework-owned and unmounted.

## Prompt and context path

Before each LLM request, Thorn assembles framework guidance, role guidance,
operator/project instructions, skills, memory files, journal entries, and
session history through one context pipeline. The pipeline is shared by CLI
and gateway sessions. [`context-gathering.md`](context-gathering.md) documents
its current implementation in detail; its “future work” sections remain
proposals rather than current guarantees.

The agent loop then alternates between provider completions and tool calls until
the request completes or a configured policy stops it. Provider-facing history,
context budgets, read-reuse advice, validation tracking, and tracing are parts
of this loop; several remain experimental controls rather than stable user
contracts.

## Trust and security boundaries

Thorn separates two questions:

1. **Who may trigger work?** Gateway event formatting resolves configured peers
   and applies trigger policy before treating external prose as an instruction.
   This is a trust boundary inside one agency, not tenant isolation.
2. **What can a tool execution venue reach?** The container sandbox limits the
   filesystem, credentials, and network available to `SANDBOX` tools.
   `IN_PROCESS` tools deliberately run in the trusted host process and are
   constrained by their application logic and configured account authority,
   not by the container. Venue assignment is therefore part of the security
   design rather than an implementation detail.

Important current limitations:

- Peers within an agency are assumed to share a trust domain; Thorn does not
  defend mutually hostile tenants from one another.
- The LLM-facing host process is trusted and is outside the tool sandbox.
- Local CLI subprocess tools act with the local user's authority.
- `MEMORY.md` and journals are durable context files, not runtime-enforced
  scoped memory with ownership and promotion controls.
- Forge notification acknowledgement happens after durable handoff, not after
  the agent completes the work. The local inbox becomes the recovery source of
  truth.

See the [threat model](threat-model.md) and
[startup/recovery flow](startup_flow.md) for operational detail.

## Source organization

The main implementation packages currently have these responsibilities:

| Package | Current responsibility |
|---|---|
| `thorn.agents` | First-party local agent roles. |
| `thorn.core` | Agent loop, providers, prompt context, tools contracts, sessions, and low-level execution policy. |
| `thorn.runtime` | Persistent agency paths/state, agent/session lifecycle, inbox routing, scheduling, and recovery. |
| `thorn.gateway` | Agency configuration loading, daemon orchestration, forge event sources, routing, peer policy, preflight, and broker lifecycle. |
| `thorn.toolhost` | Tool protocol and the daemon that executes tools outside the LLM-facing process. |
| `thorn.sandbox` | OCI image/runtime adapters and container-backed toolhost hosting. |
| `thorn.tools` | Built-in filesystem, shell, Git, forge, coordination, and supporting tools. |

This table describes the code as it exists; it is not a claim that every
boundary is already clean. In particular, the clarified agency/gateway model
and several large modules expose decomposition and naming debt that should be
addressed through focused architectural review rather than size-only splits.

The library-style Python API exists to support the CLI and gateway. It is not a
stable external compatibility contract.
