# Sandbox Threat Model

## Status and provenance

This document captures the threat model for the per-agent
tool-execution sandbox introduced across Phases A–E of the
`sandbox_tool_execution` roadmap. It is the answer to "what
does the sandbox actually promise, and what would invalidate
those promises?" and is intended to be short enough to read in
one sitting.

The audience is two-fold:

1. **Operators** deploying a Thorn gateway, who need to know
   what the sandbox protects against and what they remain
   responsible for outside it.
2. **Future Thorn contributors**, who need to know which
   invariants are load-bearing for the sandbox's claims and
   which are defense-in-depth, so a refactor doesn't quietly
   weaken something the model relied on.

Cross-references:

* Phase B retro: [sandbox-phase-b.md](sandbox-phase-b.md)
* Phase D retro: [sandbox-phase-d.md](sandbox-phase-d.md)
* Phase E retro: [sandbox-phase-e.md](sandbox-phase-e.md)
* Originating roadmap: `~/.cursor/plans/sandbox_tool_execution_1c0c334b.plan.md`

## What the sandbox promises (load-bearing claims)

The sandbox is designed to deliver two specific guarantees,
in the presence of an adversarial agent:

**G1. Credential isolation.** A tool running inside the sandbox
cannot read or exfiltrate any of the operator's credentials —
SSH keys, cloud tokens, GitHub PATs, GitLab PATs, LLM-provider
keys, or anything else that the operator keeps on the host.
The agent's tools, even if compromised, cannot reach those
credentials' on-disk locations and cannot use the operator's
network identity to call upstream APIs as the operator.

**G2. Containment of dumb mistakes.** A tool running inside the
sandbox cannot delete, modify, or otherwise corrupt any file on
the operator's host outside a bounded, agent-scoped subtree. A
runaway `rm -rf` from a confused agent eats only the agent's
own home and workspace; the operator's actual home directory,
their `/etc`, their other projects, and the framework's own
state are untouched.

Everything else the sandbox does is defense in depth, *not*
load-bearing. We harden the container further (capability
drops, read-only rootfs, resource limits, etc.) to narrow the
blast radius of bugs in the layers that deliver G1 and G2,
but the security claim is grounded in just two boundaries:
filesystem mount selection and network policy.

## What the sandbox does *not* promise

To keep G1 and G2 honest, this list is also short:

* **No protection against kernel exploits or hypervisor
  escapes.** A 0-day in the Linux kernel that lets a non-root
  process gain caps anyway, or a container-escape bug in the
  OCI runtime, defeats every sandbox property. Mitigation is
  "keep your kernel and runtime patched", which is the
  operator's responsibility.
* **No protection against operator self-foot-shooting.** If
  the operator runs an executable the agent wrote (or sourced
  from somewhere unspecified) without thinking about it, the
  sandbox does not protect them. This is the same disposition
  as cloning a random GitHub repo and running its setup
  script: the agent's `/agent/workspace` should be treated
  with the same caution a developer treats unfamiliar code.
* **No protection against the broker being compromised.** The
  credential broker (OneCLI) holds the real credentials; if it
  is compromised, the upstream services those credentials
  unlock are reachable to the attacker. Operators who care
  should treat the broker as a sensitive service (run it in
  its own VM/network, audit access).
* **No protection against gateway compromise.** The gateway
  process holds LLM-provider API keys (and the broker admin
  token). A compromised gateway leaks those. Mitigation is
  "the gateway is a privileged daemon, treat it as such".
* **No promise about LLM provider keys.** The gateway needs
  these to talk to the LLM, and the brain runs in the
  gateway, so they live on the host with the gateway. Per
  the roadmap's *Key decisions*, this is "the operator's
  hosting problem" and outside the sandbox's remit.
* **No promise about the agent's outbound network being
  audit-logged.** The broker logs the substitutions it
  performs; non-substituted HTTPS pass-through traffic is
  not audited at the protocol level. Operators who need
  full network audit can add it at the network layer
  separately.

## How G1 and G2 are delivered

Two boundaries do all the load-bearing work. Everything below
exists to keep these two intact.

### Boundary 1: filesystem mount selection (delivers G1 and G2)

The per-agent sandbox container has access to *exactly* three
host paths via bind mounts, and nothing else:

* `<workspace_root>/agents/<id>/workspace/` → `/agent/workspace`  (rw)
* `<home_root>/agents/<id>/home/` → `/agent/home`  (rw)
* `<workspace_root>/agents/<id>/control/` → `/agent/control`  (rw, never agent-visible)

These three paths are the entire footprint the sandbox can
read or write on the host. In particular:

* The operator's `$HOME`, `/etc`, `/var`, `/root`, and every
  other "real" host directory is **not mounted**. The
  container's view of the host filesystem is the OCI image's
  rootfs (which we control) plus the three bind mounts. There
  is no way for an in-container process to reach anything else.
* The framework's own state — `agent.json`, `sessions/`, the
  agency's `gateway.json`, service queues — lives at
  `<home_root>/agents/<id>/sessions/`, `<home_root>/agents/<id>/agent.json`,
  etc. **None of these paths are mounted.** A compromised tool
  cannot rewrite an agent's identity, replay a session, or
  forge a gateway-config change.
* The agent's framework state is structurally separated from
  its agent-touchable state by the layout helper in
  [src/thorn/runtime/_paths.py](../../src/thorn/runtime/_paths.py).
  This split is what makes the "two parts of the agent dir,
  only one of them mountable" pattern work.

This boundary is **the** thing that delivers G1 and G2:

* G1 holds because the operator's credentials live in
  *unmounted* host paths. They cannot be opened, read, or
  copied by an in-container process. The container's
  environment, separately, contains only placeholder strings
  (Phase D); even those carry no real auth material.
* G2 holds because a runaway `rm -rf /` inside the container
  recurses over the container's rootfs (which is per-launch
  anyway), `/agent/workspace` (the agent's own workspace —
  acceptable), `/agent/home` (the agent's own home —
  acceptable), and `/agent/control` (the rendezvous; agent
  is unaware this directory exists at all). The operator's
  home, the system, and the framework state are not in the
  recursion's path.

### Boundary 2: broker-only network policy (delivers part of G1)

G1 is partly a filesystem property and partly a network
property: even with no on-disk credential access, an
attacker who can reach upstream APIs from the container
might use them under whatever identity their network
position grants. We block this two ways:

* **Network membership.** The container is attached to the
  Docker network `thorn-broker`, which is created with
  `internal: true`. There is no NAT to the host network from
  this network, so the container has no route to the public
  internet. The only IP reachable from inside the container
  is the broker's IP on `thorn-broker`.
* **Credential substitution.** The broker (OneCLI) accepts
  HTTPS via `HTTPS_PROXY`. For hosts that match a registered
  substitution rule, the broker swaps the placeholder string
  the container holds for the real credential, then forwards
  to the real upstream. The container never sees the real
  credential.

The container therefore has full HTTP/HTTPS reach (because
the broker forwards arbitrary HTTPS, not just substituted
hosts) but **does not** have non-HTTP egress (no SSH, no
arbitrary TCP, no DNS to upstreams other than what the
broker resolves on its behalf). The broker is the entire
funnel.

The bundled `deploy/broker.compose.yml` ships exactly this
shape; operators who run the bundled compose get the
broker-only invariant for free. Operators with their own
broker deployment must replicate the network shape
themselves; the threat model assumes they have.

## Defense in depth (not load-bearing)

The following measures narrow blast radius if a bug in
either of the two load-bearing boundaries is found. They
are good practice and expected to be present, but the
sandbox's *promises* (G1, G2) do not depend on them being
correct.

* **Identity (Phase B + Phase E).** The in-container process
  runs as the gateway operator's host uid (no in-container
  root). On rootless podman, `--userns=keep-id` aligns
  secondary in-container uids with operator sub-uids. This
  ensures bind-mount writes land owned by the operator —
  which is necessary for the operator's normal workflows
  (move/copy/git the agency state) and removes the chown
  dance that distinct-uid models force.
* **Capability drops (Phase E).** `--cap-drop=ALL` removes
  every Linux capability from the container's bounding set.
  Combined with the non-root user, this means escalation
  pathways via capability bugs are limited (the attacker
  would need to defeat both the user-id check *and* the cap
  bounding set).
* **No-new-privileges (Phase E).** `--security-opt=no-new-privileges`
  prevents `setuid`/`setgid` binaries from elevating, even if
  one ends up in a derived image.
* **Default OCI seccomp (Phase E).** Both podman and docker
  apply a default seccomp profile that blocks dangerous
  syscalls (`keyctl`, user-namespace creation from unprivileged
  context, etc.). We do not pass `--security-opt=seccomp=unconfined`.
* **Read-only rootfs (Phase E).** `--read-only` plus tmpfs
  at `/tmp` and `/var/tmp` ensures the container's rootfs
  is immutable and per-launch. An attacker who finds a
  write path into the rootfs cannot persist across gateway
  restarts.
* **Resource limits (Phase E).** `--memory`, `--cpus`,
  `--pids-limit` cap the cgroup envelope. A fork bomb or
  memory leak hits the limit and the container OOMs/throttles
  rather than dragging down the host.

## Identity model (why we run as the operator uid)

The in-container process runs as the gateway operator's host
uid (and gid), both inside and outside the container. We
ratify this here because the original Phase E roadmap text
implied a different model ("remap to a host uid distinct
from the gateway's user") that we explicitly do not adopt.

Reasons:

1. **G1 and G2 do not depend on uid separation.** They
   depend on mount selection and network policy. The uid
   choice is about how files written by the in-container
   process are owned on the host; it is not what protects
   the operator's credentials.
2. **The gateway needs full read access to bind-mounted
   state.** The brain process inspects journal files,
   memory documents, and other agent state directly today
   as part of constructing prompts. With distinct-uid
   ownership, every such read becomes either a sudo or a
   separately-managed group/ACL setup; both undermine the
   "have docker, run thorn serve, it works" UX target the
   project commits to.
3. **Operator workflows depend on full read/write access.**
   Operators are expected to copy/move/git the agency
   state (commit memory documents, snapshot a workspace
   for review). With distinct-uid ownership, every such
   action requires manual ownership management.
4. **The marginal hardening from distinct uids is small in
   our threat model.** It buys protection only against
   capability-escalation bugs that *also* defeat
   `cap_drop=ALL` *and* `no-new-privileges` *and* seccomp.
   That stack of "what would need to be true" is not where
   our security investment pays best.

A consequence: the sandbox is not, by itself, suitable for
running fully untrusted code that the operator did not
implicitly approve by running this gateway. The threat
model assumes the operator chose the agency, the agent
configurations, and the tool surface, and is willing to
accept "what this agent does inside its sandbox" as
representative of their own activity for blast-radius
purposes. Multi-tenancy with hostile tenants is out of
scope.

## What changes invalidate this analysis

If a future PR does any of these, this document needs to be
revisited:

* **Mounts a path other than the three above into the
  container.** Adding any mount outside `/agent/{home,workspace,control}`
  potentially breaks G1 or G2.
* **Loosens the network policy** — joins the container to a
  non-internal network, or punches direct egress to upstreams
  outside the broker. Phase D's `egress_allowlist` schema
  is in place but enforcement is deferred; if/when it lands,
  the threat model revision needs to record what the
  allow-list does to the broker-only invariant.
* **Grants any capability** to the container (`--cap-add`).
  Each cap should be justified in the agent or agency
  configuration's documentation.
* **Runs the container as root** (drops the `--user op_uid:op_gid`
  flag). The non-root identity is part of the defense-in-depth
  story; running as root reverts to a much weaker stance.
* **Disables read-only rootfs or seccomp.** Per-agent
  overrides are allowed but should be a deliberate, documented
  choice for a particular agent's workload.
* **Changes the framework-state vs agent-touchable layout.**
  The mount selection in Phase B works because
  `agent.json`, `sessions/`, and `gateway.json` live on
  *unmounted* paths. Any rearrangement of the layout in
  [src/thorn/runtime/_paths.py](../../src/thorn/runtime/_paths.py)
  must preserve that property.

The split between "load-bearing" and "defense in depth" is
not just a writing device: it tells a future reviewer
which changes warrant a real threat-model rerun (the
load-bearing ones, anything in this section) versus which
are routine engineering-quality decisions (the defense-in-depth
ones).
