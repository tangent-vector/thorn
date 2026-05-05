# Sandbox Phase B — Wrap the Daemon in a Real Container

## Status and provenance

This document captures the **as-built** shape of Phase B of the
sandbox tool-execution roadmap.  It is the post-implementation
counterpart to `~/.cursor/plans/sandbox_phase_b_plan_9f0dbc4a.plan.md`
(itself extracted from the parent
`sandbox_tool_execution_1c0c334b.plan.md`).  The planning document
laid out the design space, called out the open questions, and
recommended an answer for each; this document records the answers
that were actually settled on, the resulting code shape, and the
items that were intentionally deferred to Phases C–F.

When Phase A landed, the `thorn-toolhost` daemon ran as a
*subprocess* on the host, talking to the gateway over a
Unix-domain socket in `<agent>/control/toolhost.sock`.  The
abstraction was deliberately built so the host-side launch path
could be swapped for a container without touching the gateway, the
brain, or the tool implementations themselves.  Phase B does
exactly that swap.

## Goal & non-goals

**Goal.** Per-agent isolation of tool execution by running
`thorn-toolhost` inside an OCI container.  The brain (gateway,
provider, scheduler) stays on the host.  Communication remains
over the same Unix socket, now bind-mounted into the container at
`/agent/control/toolhost.sock`.  No protocol changes, no
behavioral changes to tools.

**Explicit non-goals (handed to later phases):**

* MCP server lifecycle *inside* the container — Phase C.
* Credential broker (OneCLI integration) and the matching network
  egress policy that bounds the container to the broker — Phase D.
* Capability drops, userns remapping, resource limits — Phase E.
* Production network-restricted defaults.  Phase B uses default
  OCI networking so existing env-injected credentials continue to
  reach upstream APIs through the container.

## Architectural shape

```
host:                                 container (per agent):
  Gateway                                thorn-toolhost
    │                                       │
  Runtime                                bind: /agent/control/toolhost.sock
    │                                       │
  per-agent DaemonToolExecutor              │
    │                                       │
  ContainerDaemonHost  ── adapter ──>  OCI runtime (podman|docker)
                              │
                              └── creates the container above with
                                  bind-mounts:
                                  • <home>     -> /agent/home  (rw)
                                  • <workspace>-> /agent/workspace (rw)
                                  • <control>  -> /agent/control (rw)
                                    (socket is bound *inside* and
                                    appears on host via this mount)
```

The Phase-A `SubprocessDaemonHost` survives unchanged and remains
the default when no `sandbox` block is configured.  Selection
between hosts is per-agent at build time; both can coexist in a
single agency.

## What landed

### Code surface (high-level)

| Component                         | Module                                  | Purpose                                                                   |
|-----------------------------------|-----------------------------------------|---------------------------------------------------------------------------|
| `DaemonHost` protocol             | `thorn/toolhost/_host.py`               | The "how to launch and tear down a daemon" abstraction.                  |
| `SubprocessDaemonHost`            | `thorn/toolhost/_host.py`               | Phase-A behaviour, unchanged in semantics.                               |
| `OCIRuntimeAdapter` protocol      | `thorn/sandbox/_runtime.py`             | The "how to talk to a real OCI CLI" abstraction.                         |
| `PodmanAdapter` / `DockerAdapter` | `thorn/sandbox/_runtime.py`             | CLI shells around `podman` / `docker`; chosen via `select_oci_runtime`.  |
| `FakeOCIRuntimeAdapter`           | `thorn/sandbox/_runtime.py`             | In-memory adapter for tests; the workhorse for fast CI.                  |
| `ContainerDaemonHost`             | `thorn/sandbox/_container.py`           | The container-flavoured `DaemonHost`; image check, run, two-stage probe. |
| `SandboxConfig` (agency)          | `thorn/gateway/_config.py`              | `gateway.json` `sandbox` block: backend, runtime, default image, env.    |
| `AgentSandboxOverride` (agent)    | `thorn/gateway/_config.py`              | `agent.json` `sandbox` block: per-agent overrides.                       |
| `ResolvedSandboxConfig`           | `thorn/sandbox/_resolve.py`             | Result of merging agency defaults with per-agent overrides.              |
| Runtime backend selection        | `thorn/runtime/_runtime.py`             | `_build_sandbox_executor` picks the right host based on resolved config. |
| `Gateway._preload_sandbox_executors` | `thorn/gateway/_gateway.py`         | Eagerly starts every per-agent host on `_startup` (hard-fail on errors). |
| `Dockerfile.sandbox`              | repo root                               | Minimal image; `python -m thorn.toolhost` entrypoint, bundles `git`.     |
| `thorn sandbox build` CLI         | `thorn/_cli.py`                         | Operator-facing image build; tags `thorn-sandbox:<thorn-version>`.       |
| `thorn sandbox status` CLI        | `thorn/_cli.py`                         | Reports image presence and per-agent container state.                    |

### Configuration surface

`gateway.json` gains an optional `sandbox` block:

```json
{
  "sandbox": {
    "backend": "container",
    "oci_runtime": "podman",
    "image": "thorn-sandbox:0.1.0",
    "env_passthrough": ["LANG", "TZ"],
    "dev_mount_runtime": false,
    "container_ready_timeout_s": 30
  }
}
```

* `backend`: `"subprocess"` (Phase A behaviour) or `"container"`
  (Phase B).  When the `sandbox` block is *present*, the default
  is `"container"`; absence of the block defaults to
  `"subprocess"`.
* `oci_runtime`: `"podman"` or `"docker"`; if omitted, prefers
  `podman`, falling back to `docker`.
* `image`: agency-wide default sandbox image.  If omitted, falls
  back to `thorn-sandbox:<thorn-version>`.
* `env_passthrough`: explicit allowlist of host env vars that
  should be forwarded into every container.  Phase B intentionally
  treats env-passthrough as **opt-in**.  Per the design
  discussion: until Phase D removes service credentials from the
  brain, the only safe policy is "pass nothing unless asked."
* `dev_mount_runtime`: when `true`, bind-mount the host's
  `thorn` source tree into the container under
  `/opt/thorn-venv/lib/python3.12/site-packages/thorn`.  Useful
  during local development of Phase B itself; off by default
  because it is a development-only convenience.
* `container_ready_timeout_s`: how long to wait for the OCI
  runtime to report the container as `running` before failing
  startup.

`agent.json` gains an optional matching `sandbox` block.  Every
field is an override of the agency-level value:

```json
{
  "sandbox": {
    "image": "rust-sandbox:1",
    "env_passthrough": ["RUST_LOG"],
    "extra_env": {"AGENT_FLAVOR": "rust"}
  }
}
```

Merge rules (see `thorn.sandbox._resolve.resolve_sandbox_config`):

* `backend`, `image`, `container_ready_timeout_s`: per-agent value
  *overrides* the agency value when set.
* `env_passthrough`: **additive** (agency list ∪ agent list).  This
  is intentional — agency operators are typically maintaining a
  baseline, and per-agent additions should not have to repeat it.
* `extra_env`: per-agent literal env vars (key→value pairs).  No
  agency equivalent; if the agency wants to set literal env vars
  globally, it can either bake them into a custom image or add a
  per-agent block to each agent (intentional friction — literal
  env vars are agency-scope smell).

### Image strategy

Phase B ships **one** opinionated image: the minimal
`thorn-sandbox:<version>` defined by `Dockerfile.sandbox`.  The 90%
case (a single-agent agency) uses it as-is; the rest are expected
to compose:

```dockerfile
FROM thorn-sandbox:0.1.0
RUN apt-get update && apt-get install -y rustc cargo
```

…and override `agent.sandbox.image` in their `agent.json`.

The image is intentionally small.  It bundles only `git`, the
`thorn` package, and its runtime dependencies.  Heavier toolchains
(language runtimes, build tools) are the operator's responsibility
to compose in.

The image is run with `--user <host-uid>:<host-gid>` so bind-mount
writes land with the operator's ownership (no `chown` dance, no
in-image `thorn` user, no root-owned junk in the workspace).

### Image-missing behaviour: hard-fail with remediation

Phase B's stance is hard-fail at gateway startup if the configured
sandbox image is not in the local cache.  No silent auto-build.
The error message names the image and the exact command to run:

```
sandbox image 'thorn-sandbox:0.1.0' is not present in the local
podman cache.  Run `thorn sandbox build --tag thorn-sandbox:0.1.0`
(or omit --tag for the default), or set `sandbox.image` in
gateway.json to an image that has been built/pulled, then restart
the gateway.
```

The decision to favour hard-fail over auto-build was a pragmatic
one: silent auto-build is convenient when it works, but when it
silently runs against the wrong cache or with the wrong context it
produces a subtle, hard-to-diagnose failure mode.  Making the
build a named operator action keeps post-hoc diagnosis simple.

### Eager startup, parallel preload

`Gateway._startup` calls `executor.start()` on every per-agent
sandbox executor *before* it activates any sessions.  This means a
broken sandbox image, a misconfigured runtime, or a permissions
problem on the control directory produces a clear failure on the
gateway operator's console — not on the first inbound webhook
delivery five minutes later.

Preloads run in parallel via `asyncio.gather(return_exceptions=True)`;
all failures are logged, and the first one is re-raised so
`thorn serve` exits non-zero.

### Operator CLI

* `thorn sandbox build [--tag T] [--dockerfile F] [--context D] [--runtime podman|docker]`:
  build the sandbox image.  Defaults to building
  `thorn-sandbox:<thorn-version>` from the repo's
  `Dockerfile.sandbox`.
* `thorn sandbox status [--agency PATH]`: report the agency's
  configured runtime, whether the default image is present, and
  the live state of every container whose name starts with the
  `thorn-agent-` prefix.  Read-only.

### Testing strategy

Two layers, both kept fast on every `pytest` run:

1. **Fast unit tests against `FakeOCIRuntimeAdapter`** — exercise
   `ContainerDaemonHost`'s lifecycle, the resolver's merge logic,
   the runtime's backend selection, the gateway's eager preload,
   and the CLI commands.  No real container ever runs.
2. **Opt-in smoke tests against a real runtime** — gated by
   `pytest.mark.requires_podman` / `requires_docker` and
   *deselected by default* via `addopts` in `pyproject.toml`.  Run
   with `pytest -m requires_podman` (or `-m requires_docker`) when
   you want end-to-end coverage; they pull `busybox`, run a
   container, inspect, list, stop, and remove it.  Round-trip is
   under 25s per runtime.

The smoke tests intentionally do *not* drive
`ContainerDaemonHost` against a real `thorn-sandbox` image — that
would require either a pre-built image or a 3-minute build
on every smoke run.  The deeper integration is covered by the
fake-adapter tests, which exercise the same control-flow.

## Deferred items (handed to Phases C/D/E)

These were either out of scope for Phase B or revealed during
implementation as worth a dedicated follow-up.

### Phase C — MCP server lifecycle inside the container

Phase B leaves the MCP server running on the host (or wherever
Phase A left it).  Phase C will move the lifecycle inside the
container, so an MCP-using tool's network reach is bounded by the
container's network policy rather than the host's.

### Phase D — Credential broker (OneCLI) and egress policy

Phase B punts on credential isolation: env-passthrough is opt-in
and `extra_env` is supported as a literal-value mechanism, but
both are honest about the fact that anything passed in is visible
to the daemon and to all tools running under it. Phase D
integrates [OneCLI](https://github.com/onecli/onecli) as the
credential broker — placeholder strings live in the container's
env, real credentials live in OneCLI, container HTTPS traffic
flows through OneCLI for substitution — and pairs that with a
broker-only egress policy that bounds the container's outbound to
the broker via OCI-network membership (operators create the
broker network with `internal: true`; sandbox containers join only
that network).  Per the roadmap's *Resolved opens*, we
deliberately do not ship a Thorn-internal alternative broker;
OneCLI is the chosen solution.

Phase D also introduces the `ServiceCredential` newtype and the
`assert_no_literal_credentials` audit invariant: post-registration,
no literal-state credential is reachable from agent state. The
audit fails loudly if anything survived, so a refactor that
silently leaks a credential to the container is a noisy test
failure rather than a silent secret-handling regression. See
`docs/plans/sandbox-phase-d.md` for the as-built retro.

Two Phase D items are deliberately deferred to follow-ups:

* **Allow-list enforcement.** `sandbox.planned_egress_allowlist`
  parses as future intent, but the firewall mechanism (per-agent
  network + iptables vs. netns vs. OCI-runtime hooks) is open
  question R3 and pending. The gateway logs a warning at startup
  when the list is non-empty so operators are not surprised by the
  enforcement gap; the active-sounding `sandbox.egress_allowlist`
  key is rejected.
* **Real-runtime broker smoke.** Phase B's
  `requires_podman` / `requires_docker` smoke covers the OCI
  adapter; an end-to-end smoke that brings up the bundled
  compose and asserts both successful credential substitution
  *and* that direct upstream egress is blocked is a follow-up.

The env-passthrough mechanism remains useful for non-credential
values (locale, timezone, dev-mode toggles). The *guidance* has
shifted: with the broker enabled and the sandbox backend in
container mode, credentials are no longer routed through env at
all.

### Phase E — Hardening

Phase B uses default OCI networking and does not drop capabilities,
remap user namespaces, or set memory/CPU limits.  Phase E is the
dedicated hardening phase (capability drops, userns remap, resource
limits, formal threat-model review); the seams are already in
place (`ContainerSpec.extra_run_args`, the per-agent override
block, the resolver) so Phase E mostly adds new fields rather than
restructuring anything.  Egress restriction is *not* a Phase E
deliverable -- it lands with the broker in Phase D, and Phase E
revisits it as part of the threat-model review.

### Smoke-test image build

The opt-in smoke tests cover the OCI adapter against a real
runtime, but do not exercise `ContainerDaemonHost` against a real
`thorn-sandbox` image.  A future "extended smoke" target could
build the image once per CI run and exercise the full
container-host control-flow against it; this was deferred as
unnecessary for Phase B's "is the wiring correct?" question.

### `--entrypoint` JSON-array form on docker

`PodmanAdapter` and `DockerAdapter` share an entrypoint encoder
that emits the JSON-array form (`--entrypoint '["sleep"]'`).
Podman accepts that form; docker does not.  In production we use
the image's baked-in entrypoint and only set `command`, so this
does not bite us, but a follow-up should split the encoder so
docker gets a string-form entrypoint.  The smoke test works around
this by relying on the image's default entrypoint.

## Cross-references

* Originating roadmap:
  `~/.cursor/plans/sandbox_tool_execution_1c0c334b.plan.md`
* Phase B detailed plan (planning artifact):
  `~/.cursor/plans/sandbox_phase_b_plan_9f0dbc4a.plan.md`
* Phase-A daemon: `src/thorn/toolhost/`
* Per-agent path layout: `src/thorn/runtime/_paths.py`
* Sandbox executor pool: `src/thorn/runtime/_runtime.py`
