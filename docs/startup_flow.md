# What `thorn serve` does on startup

This document walks through the bring-up sequence of `thorn serve`
in the default configuration (no `sandbox` block, no `broker` block
in `gateway.json`).  The goal is to give operators a mental model of
what they're seeing in the logs and where to look when something
goes wrong.

The corresponding code lives in
`src/thorn/gateway/_gateway.py::Gateway._startup` and
`src/thorn/gateway/_bundled_broker.py::BundledBrokerSupervisor`.

## TL;DR

1. Load and validate `gateway.json` (auto-fills container sandbox +
   bundled broker defaults).
2. Bring up a per-process OneCLI + Postgres compose stack (`<oci>
   compose up -d --wait`).
3. Discover the bound ports, wait for OneCLI's `/api/health`.
4. Mint (or fetch) the OneCLI admin API key in process memory.
5. Register every loaded agent with the broker; swap each agent's
   literal credentials for broker-issued placeholders.
6. Hand over to the event-source poll loop.

On graceful shutdown:

1. Drain in-flight events; let schedulers finish.
2. Tear down per-agent broker registrations.
3. `<oci> compose down --volumes --remove-orphans` for the broker
   stack -- no broker artefacts survive on disk.

## Step by step

### 1. Load configuration

`thorn serve` reads `gateway.json` from the agency home directory
and validates it against `GatewayConfig`.  The schema applies two
"absent block -> secure default" rules at this point:

* No `sandbox` block -> `SandboxConfig(backend="container")` with
  Phase E hardening (caps drop, read-only rootfs, resource limits).
* No `broker` block, when sandbox resolves to container ->
  `BrokerConfig(mode="bundled")`.

These defaults apply only when `gateway.json` is the thing being
loaded; `thorn run` and `thorn chat` (which never read
`gateway.json`) fall back to subprocess sandboxing as before.

### 2. Bring up the bundled broker stack

When `broker.mode == "bundled"` and the sandbox backend is
`container`, the gateway constructs a `BundledBrokerSupervisor` and
calls `start()` on it.  The supervisor:

* Picks a per-process compose project name
  (`thorn-broker-<short-random>`) so concurrent `thorn serve` runs
  on the same host don't collide.
* Materialises the bundled `broker.compose.yml` (shipped inside the
  installed wheel) to a real on-disk path.
* Detects the OCI runtime (prefers podman, falls back to docker;
  fails loudly when neither is on PATH).
* Runs `<oci> compose -p <project> -f <yaml> up -d --wait` with
  `ONECLI_ADMIN_PORT=0` / `ONECLI_PROXY_PORT=0` so docker picks free
  host ports.  If `broker.bundled_images` is set in `gateway.json`,
  the supervisor passes those OneCLI/Postgres image references into
  compose; otherwise the compose file honors
  `THORN_BUNDLED_BROKER_ONECLI_IMAGE` and
  `THORN_BUNDLED_BROKER_POSTGRES_IMAGE` from the host environment
  before falling back to its built-in defaults.

Cold-start cost on the first run is dominated by image pulls
(typically a couple of minutes); subsequent runs reuse cached
images and complete in ~10 seconds.

Restricted-egress hosts should mirror the default broker images into
a registry the host can reach, then either set the `bundled_images`
block:

```jsonc
{
  "broker": {
    "mode": "bundled",
    "bundled_images": {
      "onecli": "gitlab.example.com:5005/team/mirror/onecli:latest",
      "postgres": "gitlab.example.com:5005/team/mirror/postgres:18-alpine"
    }
  }
}
```

or export the corresponding env vars before `thorn serve`:

```console
$ export THORN_BUNDLED_BROKER_ONECLI_IMAGE=gitlab.example.com:5005/team/mirror/onecli:latest
$ export THORN_BUNDLED_BROKER_POSTGRES_IMAGE=gitlab.example.com:5005/team/mirror/postgres:18-alpine
$ uv run thorn serve
```

Config values win over env vars when both are present.  The compose
project name still uses the `thorn-broker-<short-random>` prefix, so
`thorn broker status` and `thorn broker down` find mirrored-image
stacks the same way they find default-image stacks.

You'll see something like:

```
INFO Bringing up bundled OneCLI broker (this may take ~10s) ...
INFO Bundled broker: bringing up compose project 'thorn-broker-a1b2c3d4'
     (runtime=podman, compose=/tmp/.../broker.compose.yml)
```

### 3. Discover ports + wait for OneCLI

After `compose up` returns, the supervisor reads the actual bound
ports back via `<oci> compose port` and polls
`http://<host>:<admin-port>/api/health` until it returns 200 (with a
default 60s budget).  `compose --wait` already gates on Postgres'
own healthcheck, so the supervisor is only waiting on OneCLI's
Next.js boot here.

### 4. Acquire the admin API key

OneCLI in its single-user "local" mode (`NEXTAUTH_SECRET` unset)
exposes its admin-key endpoints unauthenticated.  The supervisor:

* `GET /api/user/api-key`: if 200, parse `apiKey` out of the body.
* If 404, `POST /api/user/api-key/regenerate` to mint one.

The minted key is held only in process memory and burned on
shutdown -- nothing is written to disk.

The supervisor synthesises a `BrokerConfig` carrying the discovered
URLs / key / `egress_network` and passes it to the rest of the
gateway.  From here on, the broker code paths are identical to the
external-broker case: `BrokerClient` makes admin-API calls against
the supervisor-provided URLs without knowing it's talking to a
managed-by-us stack.

### 5. Register agents with the broker

The existing `_register_broker_bindings` flow (unchanged by the
bundled supervisor) creates an OneCLI agent + secret per loaded
Thorn agent, fetches the substitution-proxy CA cert, and swaps each
agent's literal credentials for broker-issued placeholders.  Per-agent
state lives in OneCLI's Postgres DB only for the lifetime of the
gateway process.

### 6. Sandbox containers

When the gateway later spawns a per-agent sandbox container, it
joins the supervisor's per-project Docker network
(`<project>_thorn-broker`, an `internal: true` bridge with no NAT
to the host network).  Sandboxes therefore reach OneCLI by service
DNS (`onecli:10255`) and have no direct internet egress.  All
upstream HTTP requests flow through the substitution proxy, which
injects the broker-issued credentials and forwards the call.

`sandbox.egress_network` is the only active egress control Thorn
applies today.  `sandbox.planned_egress_allowlist` can record future
direct-egress exceptions, but it is not enforced and the gateway logs
a startup warning when it is non-empty.  The removed
`sandbox.egress_allowlist` key is rejected so operators do not mistake
planned intent for an active allow-list.

### 7. Steady state

After all of the above, the gateway hands over to its event-source
poll loop and per-agent schedulers.  Operator-visible logs from the
bundled broker stop here; everything from here on is the same as
the existing event-driven gateway loop.

## Shutdown

Graceful shutdown (SIGINT / SIGTERM, or `gateway.shutdown()`):

1. Drain in-flight events.
2. Tear down per-agent broker registrations (deletes agents +
   secrets; OneCLI's substitution proxy stops routing for them).
3. Run `<oci> compose -p <project> down --volumes --remove-orphans`
   so the OneCLI + Postgres stack and all anonymous volumes
   evaporate.

Failures during compose-down are logged but do not propagate -- a
hung compose teardown cannot block the gateway from exiting.

## Recovering from a non-graceful shutdown

If `thorn serve` was killed (`kill -9`, OOM kill, host crash) and
left an orphaned compose stack behind:

```console
$ thorn broker status
runtime  project                     status
podman   thorn-broker-a1b2c3d4       running(2)

$ thorn broker down
Tore down 1 stack.
```

`thorn broker status` filters compose projects on the host by the
bundled-broker prefix so it never touches operator-managed stacks.
`thorn broker down` runs `compose down --volumes --remove-orphans`
against each match.
`thorn broker logs --project thorn-broker-a1b2c3d4` prints redacted
OneCLI/Postgres logs for a stack that is still running.

## Operator status and inbox recovery

Use `thorn status --agency ~/.thorn` as the first read-only check
when a gateway looks idle, stuck, or unhealthy.  It summarizes the
agency path, gateway heartbeat, provider health, source poll
snapshots, inbox counts, parked errors, in-flight external keys,
broker stacks, and sandbox containers.  `thorn status --json` prints
the same data without Rich formatting.

When the problem is local queued work, inspect the durable inbox
before mutating it:

```console
$ thorn inbox list --agency ~/.thorn
$ thorn inbox show <item-id> --agency ~/.thorn
```

After fixing the underlying cause, requeue only the local parked item:

```console
$ thorn inbox requeue <item-id> --agency ~/.thorn
```

Requeueing does not touch GitLab/GitHub and does not recreate an
upstream TODO or notification; it moves Thorn's `inbox/errored/`
record back to pending work so the next gateway run can prompt the
session again.

## When things go wrong

Common failure modes and where to look:

* **"No OCI runtime with a 'compose' subcommand found on PATH"**:
  install podman or docker.  If you really do want to run without
  containerised sandboxing (and therefore without the broker), set
  `"sandbox": { "backend": "subprocess" }` in `gateway.json` to opt
  out of both.

* **"OneCLI /api/health did not return 200 within 60s"**: the
  OneCLI image is taking unusually long to boot.  Startup failures
  include a redacted tail of the OneCLI/Postgres compose logs before
  Thorn tears the transient stack down.  For a stack that is still
  running, use `thorn broker logs --project <project>` instead of
  reading raw container logs.  If the image pull itself is slow, pre-pull
  `ghcr.io/onecli/onecli:latest` and `postgres:18-alpine`, or set
  `broker.bundled_images` / the `THORN_BUNDLED_BROKER_*_IMAGE` env
  vars to mirrored images, and retry.

* **Brokered Git reaches OneCLI but fails in TLS/MITM handling**:
  tunnel mode forwards HTTPS bytes without credential substitution;
  MITM mode terminates TLS inside OneCLI so it can inject an upstream
  `Authorization` header for hosts registered by Thorn.  Git HTTPS
  through Thorn's broker should use MITM mode.  TLS or certificate
  failures in the broker log usually mean OneCLI cannot verify the
  upstream host from the VM's egress path; configure
  `ONECLI_HOST_CA_BUNDLE` for the host CA bundle, or set
  `ONECLI_SKIP_VERIFY_HOSTS` only for hosts where bypassing
  verification is acceptable.  `Proxy-Authorization` / HTTP 407
  messages point at missing broker credentials in the sandbox, while
  connection-refused / no-route messages point at host egress or DNS.

* **Bundled broker on a subprocess sandbox**: explicit
  `broker.mode = "bundled"` alongside
  `sandbox.backend = "subprocess"` is rejected at startup with a
  clear error.  Either set the sandbox backend to `container` (or
  omit the block entirely to inherit the secure default), or set
  `broker.enabled = false` to opt out of broker integration.

## Why per-process and fully transient?

Two design constraints drive this:

* **`<agency_home>` should hold only the kind of state you'd
  happily check into git** (gateway config, agent identities,
  journals, sessions).  Binaries, secrets, and compose state
  explicitly do not belong there.
* **`<workspace_root>` is for in-flight work** that ideally
  persists but is recoverable.

Trying to persist the broker across `thorn serve` restarts would
either violate the first constraint (broker DB under
`<agency_home>`) or push state into `<workspace_root>` for
something that doesn't belong there.  Per-process + transient
sidesteps both: there is no broker state to manage, because there
is no broker state that survives the gateway exit.

The cost is a one-time ~10s warm-start (after first-run image
pulls) on every `thorn serve` invocation.  For the single-VM,
hours-to-days-uptime deployment shape this is aimed at, that's
firmly in the noise.
