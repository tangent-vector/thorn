# Sandbox Phase E — Hardening

## Status and provenance

This document captures the **as-built** shape of Phase E of the
sandbox tool-execution roadmap. It is the post-implementation
counterpart to `~/.cursor/plans/sandbox_phase_e_df3e55ca.plan.md`
(itself derived from the parent
`sandbox_tool_execution_1c0c334b.plan.md`).

Phase E is the "lock in privilege drops, resource bounds, and
the identity model; write the threat model down" pass.  It does
not introduce new capabilities for the agent or change the
brain↔daemon split; it tightens the configuration surface and
the run-flag set so a Thorn deployment that follows the
defaults gets the load-bearing security properties (G1
credential isolation, G2 rm-rf containment) plus a meaningful
defense-in-depth posture without operator effort.

**Phase E ships in two pushes.**  This document covers push 1
(the implementation slices); push 2 is the end-to-end
dogfooding pass.  The threat-model document
([sandbox-threat-model.md](sandbox-threat-model.md)) was
written first and is the authoritative source for what the
sandbox actually promises; this retro records the as-built
implementation that delivers those promises.

## Goal & non-goals

**Goal.**  Land the conservative-by-default hardening surface
(capability drops, ``no-new-privileges``, read-only rootfs +
tmpfs scratch, resource limits, identity correctness on
rootless podman) plus the threat-model document, with
operator-friendly per-agent overrides for every Phase-E
hardening field.

**Explicit non-goals (rejected or deferred):**

* **"Remap in-container uid to a host uid distinct from the
  gateway operator."**  Rejected as inconsistent with Phase
  B's already-shipped ``--user $(id -u):$(id -g)`` choice and
  with the project's "have docker, run thorn serve, it works"
  UX target.  See the *Identity model* section of the threat
  model for the full rationale.
* **Allow-list enforcement (R3 from Phase D).**  Out of scope
  for Phase E.  The broker-only baseline (Phase D's
  ``thorn-broker`` ``internal: true`` network) already gives
  the agent unfettered HTTP/HTTPS to the internet via the
  broker; R3 is for the niche case of "operator wants this
  specific agent to bypass the broker for a particular
  protocol/host" and remains a deferred follow-up.
* **Custom seccomp profile.**  Default OCI seccomp is
  sufficient for the threat model; a custom profile is a
  maintenance burden the marginal hardening doesn't justify.
* **Whole-system MAC** (AppArmor / SELinux profiles).  Out
  of scope; rely on the host's distro-default profiles.
* **GitHub App auth via broker**, **admin-event broker
  registration**, **operator runbook docs**.  Phase D
  follow-ups, orthogonal to Phase E.

## Architectural shape

Phase E does not change the architecture; it just populates
``ContainerSpec`` with the new fields and threads them
through the existing pipeline:

```
gateway.json sandbox            agent.json sandbox
        │                              │
        └──────────┬───────────────────┘
                   ▼
          resolve_sandbox_config
                   │
                   ▼
         ResolvedSandboxConfig
                   │
                   ▼
        Runtime._build_sandbox_executor
                   │
                   ▼
         ContainerHostConfig
                   │
                   ▼
        ContainerDaemonHost._build_container_spec
                   │
                   ▼
            ContainerSpec
                   │
                   ▼
   _CLIRuntimeAdapter._build_run_args
                   │
                   ▼
      <runtime> run --cap-drop=ALL --cap-add=NET_RAW
                  --security-opt=no-new-privileges
                  --read-only --tmpfs /tmp:size=1G,mode=1777
                  --tmpfs /var/tmp:size=256M,mode=1777
                  --memory 2G --cpus 2.0 --pids-limit 512
                  --userns=keep-id  (podman only)
                  --network thorn-broker  (Phase D)
                  ...
```

Each box is a layer that already existed; Phase E just adds
new fields to each and a few lines of merge / emission code.

## What landed

### Identity model: ratified, plus a latent rootless-podman fix

[`PodmanAdapter`](../../src/thorn/sandbox/_runtime.py) now
defaults ``--userns=keep-id`` into every container's run args.
This fixes a latent correctness bug: rootless podman without
``keep-id`` does *not* keep operator-uid bind-mount ownership
when ``--user $(host_uid)`` is also passed -- the host uid
maps to a sub-uid inside the container, so files written
through bind mounts land owned by an unprivileged sub-uid the
operator can't read without a chown.  Phase B's
bind-mount-ownership claim quietly assumed ``keep-id`` was set
(or that you ran rootful docker).  Phase E makes the claim
true unconditionally on the podman path.

The defaulting is suppressed when the operator has already
specified ``--userns=...`` in
``ContainerSpec.extra_run_args``, on the "operator knows what
they're doing" principle (rootful podman, custom userns
layouts, intentional non-keep-id mappings all live here).

Module surface: a new ``_runtime_specific_default_run_args``
hook on ``_CLIRuntimeAdapter`` lets a subclass contribute
adapter-specific run flags before the operator's
``extra_run_args``; ``PodmanAdapter`` overrides it,
``DockerAdapter`` inherits the empty default.

### ContainerSpec hardening fields

| Field                      | Type                          | Default | Run flag                              |
|----------------------------|-------------------------------|---------|---------------------------------------|
| `capabilities_drop`        | `tuple[str, ...]`             | `()`    | `--cap-drop=<each>`                   |
| `capabilities_add`         | `tuple[str, ...]`             | `()`    | `--cap-add=<each>`                    |
| `security_opts`            | `tuple[str, ...]`             | `()`    | `--security-opt=<each>`               |
| `read_only_root`           | `bool`                        | `False` | `--read-only`                         |
| `tmpfs_mounts`             | `tuple[Tmpfs, ...]`           | `()`    | `--tmpfs <target>[:<options>]`        |
| `memory_limit`             | `str \| None`                 | `None`  | `--memory <value>`                    |
| `cpu_limit`                | `float \| None`               | `None`  | `--cpus <value>`                      |
| `pid_limit`                | `int \| None`                 | `None`  | `--pids-limit <value>`                |

Every field's default is "do not emit the corresponding flag",
so test fixtures and small smokes can build a `ContainerSpec`
without thinking about hardening; the production gateway
populates them via `ContainerHostConfig`, whose defaults
encode the conservative-by-default policy.

A small `Tmpfs` dataclass mirrors `Mount`: `target` plus
optional comma-separated `options`.  `--tmpfs <target>:<options>`
is the form both runtimes accept; an empty `options` yields
just `--tmpfs <target>` and inherits the runtime's defaults.

### Configuration surface (new fields)

Both `gateway.json sandbox` and `agent.json sandbox` gain the
following fields.  The merge rule for each is documented on
[`resolve_sandbox_config`](../../src/thorn/sandbox/_resolve.py):

| Field                | Agency default          | Override semantics                   |
|----------------------|-------------------------|--------------------------------------|
| `capabilities_drop`  | `["ALL"]`               | additive (merge agency + agent, dedup) |
| `capabilities_add`   | `[]`                    | additive                             |
| `security_opts`      | `["no-new-privileges"]` | additive                             |
| `read_only_root`     | `true`                  | per-agent replace when not `None`    |
| `memory_limit`       | `"2G"`                  | per-agent replace when not `None`    |
| `cpu_limit`          | `2.0`                   | per-agent replace when not `None`    |
| `pid_limit`          | `512`                   | per-agent replace when not `None`    |

Examples that operators are expected to copy into their
`gateway.json` / `agent.json`:

Agency-wide override of the resource ceiling for a
heavyweight workload:

```json
{
  "sandbox": {
    "memory_limit": "16G",
    "cpu_limit": 8.0,
    "pid_limit": 4096
  }
}
```

Per-agent override for a dogfooding agent that needs a
writable rootfs and an extra cap:

```json
{
  "sandbox": {
    "read_only_root": false,
    "capabilities_add": ["NET_RAW"]
  }
}
```

Per the Phase E plan, both files are operator-controlled, so
per-agent overrides are an operator-convenience knob (not a
security boundary).  Agencies that want a uniform agency-wide
policy simply do not set per-agent fields.

### Tmpfs scratch defaults

When `read_only_root` is true (the agency default), the
runtime auto-populates the container's `tmpfs_mounts` with
`DEFAULT_TMPFS_MOUNTS` from
[`_container.py`](../../src/thorn/sandbox/_container.py):

* `/tmp` (size=1 GiB, mode=1777)
* `/var/tmp` (size=256 MiB, mode=1777)

Sizes are roomy-but-bounded: 1 GiB at `/tmp` covers typical
`pip` / `cargo` / `npm` install scratch and per-test
artifacts; 256 MiB at `/var/tmp` covers the rarer "long-lived
temp" use case.  Mode 1777 matches the standard
sticky-world-writable layout `/tmp` is expected to have, so
user-mode software that checks permissions does not get
surprised.

When an agent opts out of `read_only_root`, the runtime drops
the tmpfs mounts too (they exist only to keep canonical
scratch paths writable when the rootfs is locked down -- with
a writable rootfs they are redundant).

### Threat-model document

Land at [docs/plans/sandbox-threat-model.md](sandbox-threat-model.md).
Three things distinguish it from a generic "things we did" list:

1. **Goals foregrounded.**  G1 (credential isolation) and G2
   (rm-rf containment) are stated as the load-bearing claims;
   everything else is explicitly defense-in-depth.  The
   document makes this distinction so a future reviewer
   knows which changes warrant a real threat-model rerun and
   which are routine engineering decisions.
2. **Identity model explicit.**  We run as the operator's uid
   inside and outside the container, by design, and the
   document records *why* (operator workflows depend on it,
   uid separation does not deliver G1/G2, etc.) so the
   roadmap's stale "remap to distinct uid" item cannot be
   silently re-introduced.
3. **"What changes invalidate this analysis"** section.  A
   short list of the kinds of PRs that should trigger a real
   re-review (mounting a path other than the three blessed
   ones, granting a cap, loosening the network policy,
   running as root).  This is the maintainability handle.

### Testing strategy

Phase E lands 33 new tests across three layers:

* **`tests/sandbox/test_runtime.py`** -- adapter-level pinning
  of the run-flag emission (caps, security-opts, tmpfs,
  resource limits, read-only) and of the `--userns=keep-id`
  defaulting on PodmanAdapter (with operator override
  suppression).
* **`tests/sandbox/test_resolve.py`** -- merge-rule pinning
  for every Phase-E field: additive lists, scalar replace,
  default-inheritance, and the "agency can remove caps via
  `null`" path.
* **`tests/sandbox/test_runtime_wiring.py`** -- end-to-end
  wiring from `SandboxConfig` / `AgentSandboxOverride`
  through the resolver into the `ContainerHostConfig`,
  including the tmpfs-defaults logic.
* **`tests/sandbox/test_container.py`** -- pin that
  `_build_container_spec` threads every Phase-E field from
  `ContainerHostConfig` to `ContainerSpec` verbatim.

The existing Phase B `test_smoke_mcp_container.py` was
simplified to demonstrate that operators no longer need to
pass `--userns=keep-id` explicitly; the adapter handles it.

The opt-in real-runtime smokes (`requires_podman`,
`requires_docker`) continue to cover the OCI adapter against
real runtimes.  Phase E does **not** ship a new
real-runtime smoke; that is the upcoming push 2 (dogfooding)
deliverable.

## Deferred items

These were either out of scope for Phase E push 1 or
explicitly deferred per the Phase E plan.

### End-to-end dogfooding (push 2)

The plan's `phase-e-checkpoint` todo splits Phase E into two
pushes: implementation (push 1, this document) and
dogfooding (push 2).  The dogfooding pass:

* Brings up the bundled `deploy/broker.compose.yml`.
* Builds the Phase-E sandbox image.
* Runs a real Thorn gateway + a single agent with a real
  one-prompt task.
* Asserts both that the hardening flags actually take effect
  against a real runtime *and* that an agent can do real
  work under them.
* Catches integration cracks accumulated across Phases A–E
  without major dogfooding in between.

This is intentionally treated as "get it working, fix what
breaks" rather than a tick-box test; surprises are expected
and budgeted for.

### Allow-list enforcement (R3 from Phase D)

Schema is parsed (Phase D) and the gateway warns at startup
when the list is non-empty.  Enforcement is open question
R3 in the Phase D retro and **not** in scope for Phase E
per the *Non-goals* section above.  The broker-only
baseline (Phase D's `internal: true` network) already
delivers "no internet except via broker" enforcement; R3 is
for niche direct-egress exceptions that don't have a known
use case yet.

### Operator runbook documentation

Per the Phase D retro, an operator-facing runbook ("how do I
deploy with the broker", "how do I rotate the admin key",
"how do I debug a substitution failure", "how do I raise
the resource limits") remains a follow-up.  Phase E adds
two more questions to that backlog ("what happens if a
tool tries to install distro packages?", "how do I let an
agent have a writable rootfs?") but does not write the
runbook.

### GitHub App authentication via broker

`register_agent_with_broker` rejects `GitHubAppAuth` with a
clear diagnostic.  Unchanged from Phase D.

### Admin-event broker registration

Agents created via admin events post-startup do not register
with the broker.  Unchanged from Phase D.

## Cross-references

* Originating roadmap: `~/.cursor/plans/sandbox_tool_execution_1c0c334b.plan.md`
* Phase E detailed plan (planning artifact):
  `~/.cursor/plans/sandbox_phase_e_df3e55ca.plan.md`
* Threat model: [sandbox-threat-model.md](sandbox-threat-model.md)
* Phase B retro: [sandbox-phase-b.md](sandbox-phase-b.md)
* Phase D retro: [sandbox-phase-d.md](sandbox-phase-d.md)
* Phase E code surface anchored in:
  * [src/thorn/gateway/_config.py](../../src/thorn/gateway/_config.py)
    (`SandboxConfig` and `AgentSandboxOverride` Phase-E fields)
  * [src/thorn/sandbox/_resolve.py](../../src/thorn/sandbox/_resolve.py)
    (`ResolvedSandboxConfig` and `_additive_str_list` merge helper)
  * [src/thorn/sandbox/_runtime.py](../../src/thorn/sandbox/_runtime.py)
    (`ContainerSpec` Phase-E fields, `Tmpfs`,
    `_runtime_specific_default_run_args` hook,
    `PodmanAdapter` keep-id default)
  * [src/thorn/sandbox/_container.py](../../src/thorn/sandbox/_container.py)
    (`ContainerHostConfig` Phase-E fields,
    `DEFAULT_TMPFS_MOUNTS`, `_build_container_spec`)
  * [src/thorn/runtime/_runtime.py](../../src/thorn/runtime/_runtime.py)
    (`Runtime._build_sandbox_executor` Phase-E wiring)
