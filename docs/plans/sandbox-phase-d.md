# Sandbox Phase D — Credential Broker + Egress Policy

## Status and provenance

This document captures the **as-built** shape of Phase D of the
sandbox tool-execution roadmap. It is the post-implementation
counterpart to `~/.cursor/plans/sandbox_phase_d_plan_28d1e156.plan.md`
(itself extracted from the parent
`sandbox_tool_execution_1c0c334b.plan.md`). The planning document
laid out the design space, called out the open questions, and
recommended an answer for each; this document records the answers
that were actually settled on, the resulting code shape, and the
items that were intentionally deferred to Phase E or to follow-up
work.

Phase D moves service credentials out of the brain process and out
of the container's environment, integrating
[OneCLI](https://github.com/onecli/onecli) as the substitution
proxy for outbound HTTPS traffic from the per-agent sandbox
container, and pairs that with an egress policy that bounds the
container's outbound to the broker (plus a schema-only
operator-configurable allow-list).

## Goal & non-goals

**Goal.** When the broker is configured, a sandbox container's
outbound HTTPS traffic flows through OneCLI; the brain reads
operator literal credentials at agent-load, hands them to the
broker, and replaces its in-memory copies with placeholder
`ServiceCredential` instances; the container's tools see
placeholders, the broker sees the real credentials only at request
time, and a per-load delete on shutdown leaves no broker-side
state behind.

**Explicit non-goals (handed to later phases or deliberately
out-of-scope):**

* GitHub App authentication via the broker. The Phase D broker
  client rejects `GitHubAppAuth` credentials with a clear
  diagnostic; App auth keeps the Phase-B env-injection path until
  a follow-up resolves the JWT-minting story (the broker currently
  can substitute a static value but not produce a per-request
  JWT).
* Allow-list **enforcement**. The schema lands in this phase
  (`sandbox.egress_allowlist` parses, the gateway logs a warning
  when non-empty), but the firewall mechanism is open question
  R3 and ships in a follow-up.
* Per-agent admin-event-driven broker registration. The startup
  registration loop covers every agent that the agency has
  persisted; agents that are created post-startup via admin events
  do not currently register with the broker until the gateway is
  restarted. Documented as a Phase D follow-up.
* Capability drops, user-namespace remapping, resource limits.
  These remain Phase E hardening items.
* In-tree credential broker as an alternative. Per the roadmap's
  Resolved opens, we deliberately do not ship one; OneCLI is the
  single chosen solution.

## Architectural shape

```
host:                                    container (per agent):
  Gateway                                  thorn-toolhost
    │                                        │
  Runtime                                  bind: /agent/control/toolhost.sock
    │                                        │
  per-agent DaemonToolExecutor               │
    │                                        │
  ContainerDaemonHost  ── adapter ──>   OCI runtime
                              │
                              └── creates the container above with:
                                  • bind-mounts (Phase B)
                                  • broker CA mounted at
                                    /etc/thorn/onecli-ca.pem (R/O)
                                  • HTTP[S]_PROXY / NO_PROXY env
                                  • SSL_CERT_FILE etc. env
                                  • placeholder credential env
                                    entries (e.g. GITHUB_TOKEN)
                                  • --network thorn-broker
                                    (broker-reachable, internal)

host (gateway side):
  Gateway._register_broker_bindings
    ── BrokerClient.register_agent ───────> OneCLI admin (10254)
    ── BrokerClient.register_secret ──────> OneCLI admin (10254)
    ── ServiceCredential placeholder swap   (in agent.accounts)
    ── BrokerBinding stashed on Gateway

agent runtime:
  in-container tool issues HTTPS request
    ── HTTPS_PROXY routes via OneCLI gateway (10255)
    ── Proxy-Authorization: Basic x:<aoc_token>
    ── OneCLI matches host+path, substitutes real credential,
       forwards to upstream
```

The Phase-A `SubprocessDaemonHost` and Phase-B
`ContainerDaemonHost` survive unchanged in their semantics; broker
integration is **conditional on the container backend**. When the
agency's resolved `sandbox.backend` is `subprocess`,
`Gateway._register_broker_bindings` skips the registration loop
and emits a warning so the operator notices the configuration
mismatch. This keeps single-shot `thorn run` and developer-mode
deployments working unchanged, and avoids the failure mode where a
subprocess daemon would inherit placeholder credentials with no
proxy to substitute against.

## What landed

### Code surface (high-level)

| Component                                | Module                                   | Purpose                                                                       |
|------------------------------------------|------------------------------------------|-------------------------------------------------------------------------------|
| `ServiceCredential` (str subclass)       | `thorn/core/_credentials.py`             | Newtype for tokens with `state ∈ {literal, placeholder}`; redacted `__repr__`. |
| `walk_credentials`                       | `thorn/core/_credentials.py`             | Recursive walker over Pydantic models / dicts / lists yielding credentials.   |
| `assert_no_literal_credentials`          | `thorn/core/_credentials.py`             | Audit invariant: no non-empty literal credential reachable from agent state. |
| `BrokerConfig`                           | `thorn/gateway/_config.py`               | `gateway.json` `broker` block (admin URL, API key, proxy URL, CA path).      |
| `BrokerClient`                           | `thorn/gateway/_broker.py`               | Synchronous httpx client for OneCLI's admin API (R2 endpoints).               |
| `BrokerBinding`                          | `thorn/gateway/_broker.py`               | Per-agent broker state (agent ID, secret IDs, proxy URL, CA path, env).      |
| `register_agent_with_broker`             | `thorn/gateway/_broker.py`               | Maps `ForgeAccountConfig` → secret registrations + agent + token mint.        |
| `Gateway._register_broker_bindings`      | `thorn/gateway/_gateway.py`              | Startup hook: register every loaded agent (gated on container backend).       |
| `Gateway._teardown_broker_bindings`      | `thorn/gateway/_gateway.py`              | Shutdown hook: per-load delete of agents and secrets.                          |
| `Runtime.set_sandbox_broker_binding_lookup` | `thorn/runtime/_runtime.py`           | Gateway installs a callback so the runtime can resolve bindings per agent.    |
| `SandboxBrokerBinding` (Protocol)        | `thorn/runtime/_runtime.py`              | Layering seam: runtime reads three fields, gateway's binding satisfies it.    |
| Container broker fields                  | `thorn/sandbox/_container.py`            | `broker_proxy_url`, `broker_ca_host_path`, `broker_placeholder_env`, `egress_network`. |
| `_broker_env_entries`                    | `thorn/sandbox/_container.py`            | Emits HTTP[S]_PROXY (both cases), NO_PROXY, SSL_CERT_FILE / REQUESTS_CA_BUNDLE / NODE_EXTRA_CA_CERTS / GIT_SSL_CAINFO. |
| `EgressAllowlistEntry`                   | `thorn/gateway/_config.py`               | Typed `(host, port)` entry for `sandbox.egress_allowlist`.                    |
| `Gateway._warn_if_egress_allowlist_unenforced` | `thorn/gateway/_gateway.py`        | Startup warning when allow-list is non-empty (R3 enforcement gap).             |
| Compose bundling                         | `docker-compose.yml`                     | OneCLI + Postgres profile; `thorn-broker` (internal) + `thorn-default`.       |

### Configuration surface

`gateway.json` gains an optional `broker` block:

```json
{
  "broker": {
    "enabled": true,
    "admin_url": "http://onecli:10254",
    "admin_api_key": "$ONECLI_ADMIN_KEY",
    "proxy_url": "http://onecli:10255",
    "ca_certificate_path": "/var/lib/onecli/ca/ca.pem"
  }
}
```

* `enabled`: when `false`, the broker block is treated as absent
  and Phase-B env-injection remains the credential path.
* `admin_url` / `admin_api_key`: where the brain talks to OneCLI's
  admin API (Next.js, port 10254). The API key is `oc_…` form,
  authorised as `Authorization: Bearer …` (R2). The
  `admin_api_key` field passes through `expand_env_vars`, so
  operators can keep the secret out of `gateway.json` on disk.
* `proxy_url`: where the in-container HTTP[S]_PROXY points
  (OneCLI gateway, port 10255).
* `ca_certificate_path`: host-side path to the broker's
  PEM-encoded CA. Bind-mounted R/O at
  `/etc/thorn/onecli-ca.pem` inside every sandbox container.

`gateway.json` also gains two new fields under `sandbox`:

```json
{
  "sandbox": {
    "backend": "container",
    "image": "thorn-sandbox:0.1.0",
    "egress_network": "thorn-broker",
    "egress_allowlist": [
      {"host": "status.internal", "port": 8080}
    ]
  }
}
```

* `egress_network`: name of the OCI network the per-agent sandbox
  joins. Combined with operator-side network setup
  (`internal: true`, broker connected, no other containers), this
  implements the broker-only egress policy without Thorn touching
  iptables. The bundled `docker-compose.yml` ships exactly this
  shape under the name `thorn-broker`.
* `egress_allowlist`: typed `(host, port)` entries the operator
  declares as direct-egress exceptions. **Schema-only** in this
  phase — the gateway logs a warning at startup when the list is
  non-empty so operators are not surprised. Enforcement is
  open question R3 and ships in a follow-up.

`agent.json`'s sandbox override surface deliberately does **not**
include `egress_*` fields: the broker-only invariant is operator
policy and a per-agent escape hatch would defeat it. If a
specialised agent needs different egress, the right answer is a
separate agency, not a per-agent flag.

### Audit invariant: `ServiceCredential` and its lifecycle

The Phase D plan's central invariant is that, post-registration,
**no literal-state `ServiceCredential` is reachable from agent
state**. The implementation:

* `ServiceCredential` is a `str` subclass (so existing call sites
  continue to work) with an explicit `state ∈ {"literal",
  "placeholder"}`. `__repr__` always redacts the value; `str()`
  preserves it for compatibility.
* Pydantic v2 integration uses
  `no_info_plain_validator_function` so re-validating an existing
  `ServiceCredential` preserves its state (without that, Pydantic
  would coerce to plain `str` first and lose the placeholder
  marker on round-trip).
* Empty-string credentials are tolerated as structural shims:
  forge service-level configs use `""` to mean "no service-level
  credential, fill from per-agent account at call time". The
  audit invariant skips empties because they cannot carry auth
  material.
* `walk_credentials` reaches into Pydantic models, dicts, lists,
  and tuples to enumerate every credential reachable from a
  starting object. Used by `assert_no_literal_credentials` and
  exposed for tests.

The brain-side flow (`register_agent_with_broker`):

1. Read each `ForgeAccountConfig`'s literal `ServiceCredential`.
2. Register a corresponding OneCLI secret with the right
   `hostPattern`, `pathPattern`, and `injectionConfig`
   (header for GitLab and GitHub PAT bearers; param-injection
   shapes are also supported for OAuth-style upstreams).
3. Create a OneCLI agent and call `regenerate-token` to get the
   `aoc_…` access token (R2 confirms the create endpoint does
   not return the token).
4. Bind the secrets to the agent.
5. Replace each in-memory literal with a placeholder
   `ServiceCredential` whose value is a non-sensitive
   recognisable string (e.g. `thorn-broker-placeholder-1`).
6. Run `assert_no_literal_credentials(agent)` and fail loudly
   if anything survived.

`BrokerBinding`'s `access_token` is **deliberately** in literal
state: it is a real broker credential, but it is held only by the
gateway in memory and never persisted to disk. The audit
invariant is scoped to credentials reachable from *agent state*,
not from gateway-private bindings.

### Layering: how the binding reaches the sandbox

`thorn.sandbox` cannot import from `thorn.gateway` (the gateway
depends on the runtime, not the other way around). Phase D
introduces a small structural Protocol, `SandboxBrokerBinding`,
exposing only the three fields the runtime needs (`proxy_url`,
`ca_certificate_path`, `placeholder_env`); the gateway's concrete
`BrokerBinding` dataclass satisfies it structurally without an
import.

The runtime exposes `set_sandbox_broker_binding_lookup(callback)`.
The gateway installs `self.broker_binding_for` after
`_register_broker_bindings` populates the bindings dict.
`Runtime._build_daemon_host` consults the lookup only on the
container backend; subprocess construction never touches it.

The gateway's `_ensure_scheduler_for_agent` no longer eagerly
creates the sandbox executor. The previous implementation cached
one as defensive bookkeeping against a "race" between concurrent
prompt rounds, but the dict-based cache runs on a single asyncio
thread without `await` points, so there was no real race to
defend against. Removing the eager call lets `_preload_sandbox_executors`
materialise executors *after* broker bindings are ready, so the
container host config carries the right wiring on first
construction.

### In-container env shape

When the broker fields are populated, the container gets:

| Env var               | Value                                       |
|-----------------------|---------------------------------------------|
| `HTTPS_PROXY`         | `http://x:<aoc_token>@<broker-host>:10255/`|
| `HTTP_PROXY`          | same                                        |
| `https_proxy`         | same (lower-case form, for `git` / `curl`)  |
| `http_proxy`          | same                                        |
| `NO_PROXY`            | `localhost,127.0.0.1,::1,/agent/control`    |
| `no_proxy`            | same                                        |
| `SSL_CERT_FILE`       | `/etc/thorn/onecli-ca.pem`                  |
| `REQUESTS_CA_BUNDLE`  | `/etc/thorn/onecli-ca.pem`                  |
| `NODE_EXTRA_CA_CERTS` | `/etc/thorn/onecli-ca.pem`                  |
| `GIT_SSL_CAINFO`      | `/etc/thorn/onecli-ca.pem`                  |
| `GITHUB_TOKEN`, …     | per-account placeholder strings              |

Both upper- and lower-case forms of `HTTP[S]_PROXY` / `NO_PROXY`
are emitted because client libraries are inconsistent: `curl`
honors lowercase first; Python `requests` and Node honor uppercase;
`git` reads only lowercase. Setting both is harmless and avoids
per-tool surprises.

The four CA-bundle env vars cover OpenSSL-based stacks, Python
`requests`, Node TLS, and Git. Tools using other stacks (Go's
`crypto/tls`, Rust's `rustls` defaults) need image-baked trust
anchors and are addressed when those stacks become a constraint.

### Egress policy (broker-only baseline)

The bundled `docker-compose.yml` defines two networks:

* `thorn-default` (driver: bridge): for the gateway and the
  OneCLI broker to reach the internet (LLM provider, forge APIs).
* `thorn-broker` (driver: bridge, **internal: true**): for the
  per-agent sandbox containers. Because the network is internal,
  it has no NAT to the host, so a container on it cannot reach
  the internet directly — the only reachable destination is the
  broker.

The Thorn gateway sets `extra_run_args = ("--network", "thorn-broker")`
on the per-agent sandbox container, making this the single
network the container joins. There is no iptables work; the
"broker-only" property is a network-membership consequence.

### R1 / R2 spike findings

The `R1` and `R2` research items in the planning document were
resolved by reading OneCLI source ahead of implementation:

* **R1 (proxy auth wire form).** OneCLI's `apps/gateway/src/inject.rs::extract_agent_token`
  reads `Proxy-Authorization: Basic …`, extracting the token from
  the **password** field of `<user>:<token>` (with a legacy
  fallback to the username). This means
  `HTTPS_PROXY=http://x:<aoc_token>@host:port/` Just Works with
  every standard HTTPS-proxy client; we do not need any per-tool
  bearer wiring. The `_compose_proxy_url` helper percent-encodes
  the token and inserts it into the URL's userinfo.
* **R2 (admin API).** The brain authenticates with
  `Authorization: Bearer oc_<key>`. Agent creation
  (`POST /api/agents`) does **not** return the access token; a
  follow-up `POST /api/agents/{id}/regenerate-token` mints it.
  Secrets register with `name`, `type`, `value`, `hostPattern`,
  `pathPattern`, `injectionConfig`. Per-load create-and-delete
  is the chosen lifecycle: each gateway startup creates a fresh
  OneCLI agent and tears it down on shutdown.

`R3` (egress mechanism) remains open at the time of writing;
the broker-only baseline above is in place, and the allow-list
schema parses without enforcement.

### Testing strategy

Two layers, both kept fast on every `pytest` run:

1. **Fast unit / integration tests against `httpx.MockTransport`** —
   pin every OneCLI admin API call's wire shape, exercise
   registration, teardown, secret/agent lifecycles, the audit
   invariant, the runtime's binding lookup, and the container
   spec's broker wiring. The `TestPhaseDAuditFlow` end-to-end
   test in `tests/test_broker.py` drives the entire chain in one
   test (gateway register → placeholder swap → audit → runtime
   lookup → ContainerSpec assembly).
2. **Opt-in real-runtime smoke** — gated by
   `pytest.mark.requires_podman` / `requires_docker`. Phase D
   does not ship a new `requires_*` smoke (the broker would need
   to be running too); a follow-up adds an e2e smoke that brings
   up the bundled compose, runs an HTTPS call from a sandbox
   container, and asserts both successful substitution and that
   a direct upstream call is blocked.

## Deferred items

These were either out of scope for Phase D or revealed during
implementation as worth a dedicated follow-up.

### Allow-list enforcement (R3)

Schema is in place; enforcement is the open question. Three
candidate mechanisms surfaced in the plan:

1. Per-agent OCI network with iptables rules added at container
   creation.
2. Separate network namespace with explicit forwarding rules.
3. OCI-runtime egress hooks (podman / docker custom networks).

The mechanism likely differs between podman and docker. The
contract — broker-only plus `(host, port)` allow-list — is
fixed; the implementation choice is open.

### Admin-event broker registration

Agents created via admin events post-startup do not currently
register with the broker. They run sandboxed but with no
credential injection (a sane fail-safe — no credential leak,
just no authenticated tool calls). A follow-up should hook into
the admin-event scheduler-creation path to register the new
agent on the fly.

### GitHub App authentication

`register_agent_with_broker` rejects `GitHubAppAuth` with a
clear diagnostic. App auth requires JWT minting per request;
OneCLI today substitutes static values. A follow-up either
brings JWT minting into Thorn's broker-side wrapper or
contributes the capability upstream.

### Real-runtime smoke

`tests/sandbox/test_smoke_real_oci.py` exercises Phase B's OCI
adapter against real podman/docker but does not bring up a real
broker. A `requires_podman + requires_compose` smoke would round
out the audit story: bring up the bundled compose, exercise an
HTTPS substitution, confirm direct egress is blocked.

### Documentation: operator runbook

This retro covers the as-built shape. An operator-facing runbook
covering "how do I deploy with the broker", "how do I rotate the
admin key", and "how do I debug a substitution failure" is a
follow-up.

## Cross-references

* Originating roadmap:
  `~/.cursor/plans/sandbox_tool_execution_1c0c334b.plan.md`
* Phase D detailed plan (planning artifact):
  `~/.cursor/plans/sandbox_phase_d_plan_28d1e156.plan.md`
* Phase B retro: `docs/plans/sandbox-phase-b.md`
* OneCLI: <https://github.com/onecli/onecli>
* Phase D code surface anchored in:
  * `src/thorn/core/_credentials.py`
  * `src/thorn/gateway/_broker.py`
  * `src/thorn/gateway/_config.py` (`BrokerConfig`,
    `EgressAllowlistEntry`, `SandboxConfig` egress fields)
  * `src/thorn/gateway/_gateway.py` (registration / teardown
    hooks, lookup install, allow-list warning)
  * `src/thorn/runtime/_runtime.py` (lookup install / consume,
    Protocol)
  * `src/thorn/sandbox/_container.py` (broker fields, env
    helper, egress network plumbing)
