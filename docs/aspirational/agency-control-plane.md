Agency Control Plane
====================

This document describes a target design for unifying the interactive CLI
experience (`thorn chat` and `thorn run`) with the long-running gateway
experience (`thorn serve`).

It is aspirational. It does not match the implementation today, and it should
not block the near-term goal of a first Thorn version worth sharing with
maintainer-led evaluators. The intent is to give future CLI, gateway,
monitoring, and remote-access work a common direction so incremental changes do
not deepen the current split.

Problem frame
-------------

Thorn currently has two user-facing ways to drive an agency:

- **Interactive CLI commands** create a runtime in the foreground process,
  create a local agent/session, wire an in-process scheduler and prompt router,
  and then bridge the result back to the terminal.
- **The gateway** creates a long-running runtime, loads persisted agents and
  services from `gateway.json`, starts event sources, preloads broker/sandbox
  state, writes heartbeat status, and drives sessions from durable inboxes.

Recent CLI/gateway unification work has removed some of the deepest behavioral
divergence: CLI turns now use session inboxes and `AgentScheduler`, just like
the gateway. The remaining split is still important. The CLI owns runtime
lifecycle, scheduler lifecycle, event-bus subscription, toolhost startup, and
the request/reply bridge directly. The gateway owns those same concerns for
service-driven sessions, but behind a different top-level object and a
different entry path.

The result is a recurring design tax:

- Features that should be agency-wide need to answer "does this work in the
  CLI path and in the gateway path?"
- Long-lived state, provider health, sandbox/broker readiness, and monitoring
  are naturally daemon-shaped, but CLI sessions still recreate them per
  process.
- Users reasonably expect `thorn chat` to attach to a persistent agent session,
  including one on a remote machine, but today the CLI is primarily a local
  in-process driver.
- Rich monitoring wants an API, but exposing a web server directly from a
  gateway raises authentication and authorization questions that are not worth
  solving casually.

Terminology
-----------

This document uses **agency daemon** as the generic term for the long-running
process that loads an agency, owns its runtime, schedules sessions, and exposes
a control plane.

`Gateway` remains a useful implementation name for the event-source router, but
the desired product shape is slightly broader:

- `thorn serve` starts an agency daemon with service event sources enabled.
- A future local-agency launcher starts the same kind of agency daemon for a
  user's local `~/.thorn` agency, possibly without forge polling.
- `thorn chat`, `thorn run`, `thorn status`, and future monitor UIs become
  clients of the agency daemon whenever possible.

The important point is not that every daemon must be called "gateway". The
important point is that there should be one agency-runtime owner, and everything
else should talk to it through an explicit control plane.

Goals and non-goals
-------------------

Goals:

1. Make `thorn chat`, `thorn run`, and `thorn serve` layer over the same
   daemon-owned runtime model.
2. Support local interactive sessions without starting a fresh runtime per
   terminal command.
3. Support attaching a CLI to an already-running server agency, including over
   SSH.
4. Provide a control API that can power both terminal commands and a richer
   local monitoring UI.
5. Use operating-system and SSH authentication boundaries before inventing web
   authentication.
6. Keep the design compatible with multiple simultaneous users and multiple
   simultaneous CLI sessions.

Non-goals:

- Do not expose an unauthenticated HTTP API from a gateway.
- Do not design a full browser authentication system as part of this work.
- Do not make this a blocker for the first shareable version of Thorn.
- Do not preserve the current library-style API surface merely to keep external
  embedders working. The CLI and gateway remain the priority surfaces.
- Do not require cross-platform socket support in the first version. Thorn can
  deliberately target Unix-like hosts for this phase.

Current implementation facts
----------------------------

The current implementation already has several pieces this design can build on:

- `thorn run` and `thorn chat` default their agency home to `~/.thorn` and
  create fresh `cli/<workspace>/<uuid>` session keys.
- CLI commands use `SessionInbox`, `AgentScheduler`, and per-session event-bus
  filters, but still create their own `Runtime` in the foreground process.
- `thorn serve` loads `gateway.json`, resolves workspace, registers services,
  validates accounts, infers event sources, constructs `Gateway`, and owns the
  long-running `Runtime`.
- The gateway startup path rebuilds in-flight indexes, runs startup sweep,
  preloads persisted agents, starts broker/sandbox machinery, activates
  sessions with pending work, and writes heartbeat status.
- Operator status is currently a hybrid filesystem inspection path: heartbeat
  JSON, durable inbox directories, broker stack discovery, sandbox container
  listing, and queue summaries.
- The toolhost already uses a framed JSON protocol over Unix-domain sockets for
  brain-to-tool execution. The agency control plane should be separate from the
  toolhost protocol, but the existing protocol is useful evidence that Thorn
  can test and maintain a framed Unix-socket protocol.

Target model
------------

At the center is one long-running agency daemon per loaded agency.

The daemon owns:

- `Runtime`
- loaded agents
- per-agent schedulers
- durable session inboxes
- provider health monitor
- event sources
- sandbox/toolhost pools
- broker bindings
- heartbeat and live status snapshots
- control-plane listener sockets

Clients own only presentation and local interaction:

- `thorn chat` owns terminal input/output, slash-command UI, local cancellation
  gestures, and optional local trace rendering.
- `thorn run` owns the one-shot command-line request/reply shape.
- `thorn status` owns text and JSON rendering.
- `thorn monitor` owns a local web server and browser UI.

The daemon treats CLI input as another event source. A user prompt submitted by
`thorn chat` becomes a durable notification in the target session inbox and is
scheduled by the same machinery that handles forge or future Slack events. The
client receives streamed events and a completion result through the control
protocol, but it is not the owner of the prompt round.

Control endpoints
-----------------

The daemon should expose at least two logical control endpoints:

- **User endpoint**: create and attach to sessions, submit chat turns, receive
  streamed events for sessions the caller is allowed to see, cancel a caller's
  active turn, and inspect caller-relevant status.
- **Operator endpoint**: inspect full daemon status, event-source state,
  provider health, broker/sandbox state, all inboxes, all sessions, and perform
  recovery actions such as requeueing parked items.

These can be separate Unix sockets with different filesystem permissions, or a
single socket with method-level authorization based on peer credentials. The
two-socket shape is simpler to explain and administer:

```text
<runtime-dir>/thorn/<agency-id>/
  user.sock
  operator.sock
```

Local single-user agencies can make both sockets user-private. Shared server
agencies can make the user socket available to a Unix group for users allowed
to talk to agents, and the operator socket available only to an operator group.

The daemon should still perform method-level authorization even when
filesystem permissions are correct. Filesystem permissions answer "may this
process connect?" Method authorization answers "which operation may this
authenticated identity perform?"

Socket location and discovery
-----------------------------

Socket paths are operationally awkward because Unix-domain socket paths are
short. Thorn should avoid putting live sockets under deep agency or workspace
paths.

For local single-user agencies:

- Prefer `$XDG_RUNTIME_DIR/thorn/<agency-id>/`.
- If `$XDG_RUNTIME_DIR` is unavailable, fall back to a user-private directory
  under `/tmp`, with mode `0700`, and warn in diagnostics.
- Store only runtime rendezvous files here: sockets, pid files, short-lived
  lock files, and descriptors.

For server agencies:

- Let the operator configure a short control directory, for example
  `/run/thorn/<agency-name>/` or another host-local path.
- Require explicit ownership and permissions in deployment docs.
- Do not rely on an NFS-mounted home directory for live sockets.

Agency discovery should be explicit before it is clever:

1. `--connect unix:/path/to/user.sock`
2. `--agency /path/to/agency-home`, which reads a daemon descriptor from the
   agency home and verifies that the live daemon serves the same agency.
3. Default local agency lookup for `~/.thorn`.

The descriptor file in the agency home should not be the authority for access.
It is only a locator and sanity-check aid. Connecting to the socket and passing
the daemon handshake is the real check.

The descriptor should include:

- schema version
- agency home path
- workspace path
- daemon pid
- daemon start timestamp
- control socket paths
- daemon instance ID
- Thorn version

On connect, the client should ask the daemon for the same fields and reject a
stale descriptor if the instance ID or agency paths do not match.

Authentication and authorization
--------------------------------

For the first Unix-only phase, authentication should be based on the connecting
process identity:

- On Linux, use Unix peer credentials (`SO_PEERCRED` or equivalent).
- On platforms where peer credentials are unavailable or inconsistent, fall
  back to filesystem permissions only and mark the platform unsupported for
  shared server agencies.
- Record UID, GID, PID, socket path, and resolved Thorn user identity in audit
  events for control-plane calls.

Authorization should be explicit:

- A caller connected to the user endpoint can create or attach to sessions only
  under identities/policies granted to that caller.
- A caller connected to the operator endpoint can inspect and recover all
  agency state if their UID/GID passes operator policy.
- A future peer/user mapping layer should connect OS users, CLI users, and
  service peers, but the first implementation can keep OS-user authorization
  separate from cross-service identity.

Filesystem permissions alone are not enough for future features like "this user
may chat with agent A but not agent B" or "this user can inspect their sessions
but not broker state." The protocol should leave room for those checks even if
v1 has coarse policy.

Control protocol
----------------

The control protocol should be framed, versioned, and transport-neutral. A
Unix socket, SSH stdio proxy, and future local web bridge should all carry the
same logical frames.

The first frame in both directions should be `hello`, carrying:

- protocol major/minor
- Thorn version
- daemon instance ID
- agency home
- workspace root
- supported feature flags
- authenticated OS identity as seen by the daemon
- endpoint kind (`user` or `operator`)

The initial operation set should be deliberately small:

- `get_status`: live equivalent of `thorn status`
- `list_agents`
- `list_sessions`
- `open_session`: create or attach to a session
- `submit_turn`: append a user notification and schedule the session
- `stream_events`: subscribe to event-bus output for a session or operator
  scope
- `cancel_turn`: request cancellation of a running turn initiated by the same
  client, where policy allows
- `close_subscription`
- `requeue_inbox_item`: operator-only equivalent of `thorn inbox requeue`

Each long-running operation should have a request ID. Events should be
correlated to request ID, session address, and daemon instance ID so clients can
recover cleanly when they reconnect.

The protocol should support these event classes from the beginning:

- prompt/session lifecycle
- model response chunks
- tool start/end
- advisory/status messages
- turn completion
- turn failure
- daemon shutdown
- authorization failure

This should not be a Python object remoting protocol. The wire model should be
small, structural, and versioned.

CLI behavior
------------

`thorn chat` should become a client UI.

Default local behavior:

1. Resolve the local agency (`~/.thorn` unless overridden).
2. Find a live daemon descriptor.
3. If no daemon is live, start one in local-agency mode.
4. Connect to the user endpoint.
5. Open a fresh CLI session by default, or attach to a requested session.
6. Stream only events visible to that session into the terminal.

`thorn run` should use the same daemon path when possible:

1. Open a fresh one-shot session.
2. Submit exactly one turn.
3. Stream progress according to verbosity.
4. Return the final answer and exit code.

This leaves room for a compatibility fallback:

- If no daemon can be started and the user explicitly requests in-process mode,
  keep the current foreground runtime path for development and tests.
- The default product path should become daemon-backed once the control plane is
  stable.

Session persistence and resume should be resolved as part of this work. Today,
fresh CLI sessions are a useful default. Once CLI is daemon-backed, users will
also expect:

- `thorn sessions list`
- `thorn chat --resume <session-key>`
- `/resume <session-key>` inside chat
- `/save` or `/pin` for sessions that should outlive normal housekeeping

Remote access over SSH
----------------------

Remote access should be SSH-first.

The most robust first shape is not "open a TCP port" and not "expose a web UI."
It is a stdio proxy:

```console
$ thorn chat --connect ssh://host/path/to/agency
```

implemented roughly as:

```console
$ ssh host thorn control proxy --agency /path/to/agency
```

The remote `thorn control proxy` command connects to the daemon's Unix socket
on the remote host and relays the framed control protocol over stdin/stdout.
The local client then speaks the same protocol it would speak to a local Unix
socket.

This has several advantages:

- SSH handles host authentication, user authentication, encryption, bastions,
  and existing enterprise access policy.
- No gateway TCP listener is required.
- No browser-facing authentication system is required.
- The remote daemon still sees the OS identity of the SSH login user.
- The protocol remains the same after connection setup.

OpenSSH also supports Unix-domain socket forwarding. Thorn may later offer an
`ssh+unix` endpoint that uses socket forwarding directly, but the stdio proxy
is easier to make reliable and easier to explain. It also avoids cleaning up
temporary local forwarded sockets.

Monitoring and web UI
---------------------

A rich web monitor is desirable, but the gateway should not start by serving a
browser UI directly to the network.

The safer first shape is:

```console
$ thorn monitor --agency ~/.thorn
```

or:

```console
$ thorn monitor --connect ssh://host/path/to/agency
```

`thorn monitor` would:

1. Connect to the daemon through the same control protocol as the CLI.
2. Start a local loopback-only HTTP server on the user's machine.
3. Mint a random per-process bearer token.
4. Open a browser to `http://127.0.0.1:<port>/?token=...`.
5. Bridge browser WebSocket/SSE requests to the daemon control protocol.

This keeps the hard authentication problem outside the gateway. The browser
only talks to a same-user local helper. The helper talks to the daemon through
Unix sockets or SSH, reusing those access controls.

The first monitor can be read-only:

- daemon liveness
- agents
- sessions
- active turns
- inbox counts
- pending/in-progress/errored items
- provider health
- source poller status
- broker/sandbox status
- recent event stream

Operator mutations such as requeue, cancel, or force-stop should be added only
after the read-only monitor has proven useful and the authorization model is
settled.

Lifecycle
---------

Local agency daemon lifecycle should be unsurprising:

- Auto-start on first local CLI connection when no live daemon exists.
- Stay alive while clients are connected or work is pending.
- Idle-exit after a short configurable timeout when there are no clients,
  event sources, pending work, or scheduled maintenance tasks.
- Write a descriptor and heartbeat while running.
- Remove live sockets on graceful shutdown.
- Treat stale descriptors as recoverable.

Server agency lifecycle should be operator-managed:

- Usually started by `thorn serve`, systemd, a shell wrapper, or a container
  supervisor.
- No auto-exit by default.
- No implicit daemon start by random users unless explicitly enabled.
- Control directory ownership and group access are part of deployment
  configuration.

`thorn serve` can remain the command that starts the server-mode agency daemon.
The conceptual shift is that event-source polling is one service the daemon can
run, not the defining feature of the daemon itself.

Failure modes
-------------

The control plane should handle these failure modes explicitly:

- **Client disconnect during a turn**: the notification remains in the session
  inbox and the daemon decides whether the turn continues, cancels, or becomes
  detached based on operation policy.
- **Daemon restart**: startup sweep and inbox activation recover pending work;
  clients reconnect using the descriptor and instance ID mismatch tells them
  they are talking to a new daemon instance.
- **Stale socket path**: client reports stale descriptor and suggests cleanup or
  daemon restart.
- **Authorization failure**: daemon returns a structured denial, logs the peer
  identity, and does not leak object existence beyond policy.
- **SSH transport loss**: local client exits or retries; remote daemon state is
  unchanged except for any explicit cancel policy.
- **Monitor helper exit**: browser UI dies, daemon keeps running.
- **Operator socket unavailable**: read-only filesystem fallback can still
  power `thorn status`, but mutation commands require the control API.

Implementation phases
---------------------

### Phase 0 - Keep first-share scope disciplined

Do not block the first shareable version on this design.

Continue using the existing filesystem-backed operator commands and
foreground-CLI path while the evaluator story is being polished. The important
short-term requirement is to avoid adding new features that assume the CLI and
gateway will remain separate forever.

### Phase 1 - Define endpoint and protocol types

Add internal modules for:

- endpoint addresses (`unix:`, `ssh:`, and in-process test transport)
- framed control protocol models
- daemon hello/capabilities
- authenticated peer identity
- authorization result types

No CLI command needs to switch yet. This phase is mostly contract and tests.

### Phase 2 - Add read-only control socket to `thorn serve`

Expose:

- `hello`
- `get_status`
- `list_agents`
- `list_sessions`
- event-stream subscription for operator status

Keep `thorn status` filesystem fallback, but prefer the socket when available.
This gives immediate value while keeping mutation risk low.

### Phase 3 - Add session turn API

Teach the daemon to accept user-originated session turns over the user socket:

- open/create session
- submit user notification
- stream session events
- return turn completion

At this point `thorn chat --connect unix:/...` can exist without changing the
default `thorn chat` behavior.

### Phase 4 - Local agency daemon

Teach default `thorn chat` and `thorn run` to:

- discover `~/.thorn`
- find/start a local agency daemon
- connect over the user socket
- run with the daemon-backed session API

Keep an explicit foreground/in-process escape hatch for tests and debugging.

### Phase 5 - SSH proxy transport

Add:

- `thorn control proxy --agency <path>`
- `ssh://` endpoint parsing
- client-side subprocess transport over SSH stdio

Make `thorn chat --connect ssh://host/path/to/agency` work before building any
web UI.

### Phase 6 - Local monitor bridge

Add `thorn monitor` as a local web bridge over the control protocol.

Start read-only. Add operator mutations later, after the policy model has been
tested from CLI commands.

Deferred questions
------------------

These questions should become tracking issues before implementation:

- What is the stable on-disk format for a daemon descriptor?
- What is the endpoint URL grammar (`unix:`, `ssh://`, `ssh+unix://`, maybe
  later `https://`)?
- How does a local daemon choose its workspace when a single `~/.thorn` agency
  serves CLI sessions launched from many repositories?
- What is the missing runtime "logical agent workspace" rung, and should local
  daemon sessions force that decision?
- What are the exact ownership and group-permission recommendations for shared
  server agencies?
- How do CLI users map to Thorn peers, if at all, in the first version?
- What is the cancellation policy when a user disconnects mid-turn?
- Which operator mutations belong in `thorn monitor`, and which should remain
  terminal-only?
- Should the control protocol reuse any code from the toolhost framed protocol,
  or only share design conventions?
- How much of `thorn status` should continue to work without a live daemon?

Recommendation
--------------

Adopt the agency-daemon/control-plane direction as the long-term architecture,
but keep it out of the first-share critical path.

The practical next step is to implement the smallest read-only control socket
for `thorn serve` and let `thorn status` prefer it. That tests the endpoint,
protocol, authentication, and authorization shape without changing how agents
do work. Once that feels solid, move `thorn chat --connect unix:/...` onto the
same protocol, then make the local agency daemon the default.

The web UI should be layered last, through a local bridge process, not served
directly from the gateway.
