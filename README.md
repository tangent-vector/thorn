Thorn
=====

Thorn is an autonomous coding agent that monitors your GitHub or GitLab
repositories, responds to @-mentions and assignments, and carries out
development tasks: reading issues, cloning repos, making changes,
pushing branches, and opening pull requests.

It runs as a long-lived **gateway** daemon.  The recommended first-run
shape is a host-side `uv run thorn serve` process that manages the
default container sandbox and bundled credential broker for agent
work.

Quick Start
-----------

### Prerequisites

- `uv`
- Python 3.11 or newer
- Docker or Podman for the default sandbox and bundled broker
- An LLM provider exposing an OpenAI-compatible API
- A GitHub or GitLab personal access token (PAT)

### 1. Clone and sync the repository

```console
$ git clone https://github.com/tangent-vector/thorn.git
$ cd thorn
$ uv sync --all-extras
```

### 2. Create a `.env` file

The gateway only reads **secrets** from environment variables; all
other configuration (forge base URL, project metadata, polling cadence,
etc.) lives in `gateway.json` and the per-agent identity JSON files
that `thorn serve bootstrap` writes for you. The CLI loads a `.env`
file from the working tree, so create one at the repository root with
just the secrets:

```dotenv
# LLM provider (any OpenAI-compatible API)
OPENAI_API_URL=https://api.openai.com/v1
OPENAI_API_KEY=sk-...
OPENAI_API_MODEL_NAME=gpt-4o

# GitHub (PAT auth) -- the only required forge secret for GitHub.
GITHUB_TOKEN=ghp_...
```

For GitLab, set `GITLAB_TOKEN` instead:

```dotenv
GITLAB_TOKEN=glpat-...
```

> **Do not commit `.env` to version control.** It contains secrets.

### 3. Bootstrap the agency

Before the gateway can run, it needs an agent identity and service
configuration. The `thorn serve bootstrap` command creates an agency
home, records a separate workspace root for agent sessions, and writes
a project-url-based `gateway.json`:

```console
$ uv run thorn serve bootstrap \
    --agent-id my-coordinator \
    --project-name my-repo \
    --project-url https://github.com/owner/repo \
    --agency-home ~/.thorn \
    --agency-workspace ~/thorn-workspace
```

For a public GitLab project, use its human project URL:

```console
$ uv run thorn serve bootstrap \
    --agent-id my-coordinator \
    --project-name my-repo \
    --project-url https://gitlab.com/group/my-repo \
    --agency-home ~/.thorn \
    --agency-workspace ~/thorn-workspace
```

Pass `--token-env MY_TOKEN_VAR` only when you do not want the default
`GITHUB_TOKEN` or `GITLAB_TOKEN` env var name recorded in the agent
identity.

This writes these files and directories:

| Path | Purpose |
|------|---------|
| `~/.thorn/agents/my-coordinator/agent.json` | Agent identity, including the forge account and a `credentials[*].env_var_name` reference naming the env var that holds the literal token. |
| `~/.thorn/agents/my-coordinator/home/MEMORY.md` | Persistent memory mounted into the agent sandbox. |
| `~/.thorn/gateway.json` | Workspace path and project metadata. No secrets live here. |
| `~/thorn-workspace/my-coordinator/` | Agent workspace prefix where sessions clone repositories and make changes. |

### 4. Run the first-readiness preflight

Before letting a live gateway consume forge notifications, run the
safe Git connectivity preflight from the same agency config:

```console
$ uv run thorn serve --agency ~/.thorn preflight
```

The default preflight starts the configured sandbox and broker path,
then runs `git ls-remote` against each configured project clone URL
from inside the sandbox. It does not start event sources and does not
read, mark done, or otherwise consume GitLab/GitHub notifications.
Add `--write-check` only when you want to verify that the bot
credential can push and delete a temporary branch.

### 5. Start the gateway

```console
$ uv run thorn serve --agency ~/.thorn
```

The gateway will poll the configured repository for events. When it
sees activity (an @-mention on an issue, a new assignment, etc.), it
dispatches the event to the agent, which reads the issue, clones the
repo, and does the requested work.

### 6. Talk to the agent

Open an issue on the configured repository and @-mention the bot
account (the GitHub or GitLab user whose PAT you provided). The agent
will pick up the event on its next poll cycle, read the issue, and
respond.

### Advanced project configuration

GitLab projects should normally be configured with their human project
URL, not the numeric API ID:

```json
{
  "projects": [
    {
      "name": "thorn",
      "url": "https://gitlab.example.com/team/thorn"
    }
  ]
}
```

Thorn derives `team/thorn` from that URL for clone paths and, when a
self-hosted GitLab instance rejects path-based project API lookups,
resolves it to the numeric project ID through GitLab project search.
If that search cannot see the project, add `native_id` as an explicit
fallback while keeping the human URL for clone links:

```json
{
  "projects": [
    {
      "name": "thorn",
      "url": "https://gitlab.example.com/team/thorn",
      "native_id": "264873"
    }
  ]
}
```

For self-hosted GitLab or GitHub Enterprise, declare an explicit forge
entry in `gateway.json` because `thorn serve bootstrap` only infers
the public `github.com` and `gitlab.com` hosts. The agent account's
`service` field must match `forges[].name`.

```json
{
  "workspace": "/home/me/thorn-workspace",
  "forges": [
    {
      "name": "gitlab",
      "type": "gitlab",
      "url": "https://gitlab.example.com"
    }
  ],
  "projects": [
    {
      "name": "thorn",
      "url": "https://gitlab.example.com/team/thorn",
      "native_id": "264873"
    }
  ]
}
```

For GitLab, the inferred source polls two surfaces:

- GitLab TODOs for user-scoped notifications such as mentions, direct
  addresses, review requests, assignments, and other TODO-producing
  actions. These are marked done after Thorn has handed them to the
  gateway. Thorn classifies `mentioned`, `directly_addressed`, and
  `review_requested` TODO actions as conversational; other TODO
  actions are structural. In TODO-only mode (no configured projects),
  merges, auto-closes, and dependency transitions only wake Thorn if
  GitLab also creates a pending TODO for the bot user.
- Project events for configured projects, currently closed issues and
  merged merge requests. The first project-event poll establishes a
  baseline so a fresh gateway startup does not replay historical
  closures; later polls wake the corresponding issue or change-request
  session when a new closure or merge appears.

Configuration
-------------

### Configuration model

Thorn distinguishes **secrets** (held only in environment variables)
from all other configuration (held in JSON on disk).  The on-disk
files live under the agency home you pass with `--agency-home`
(`~/.thorn` in the quick start):

- `gateway.json` -- workspace path, optional forge entries (`name`,
  `type`, `url`, optional `api_url`), and project metadata. Contains
  *no* secrets.
- `agents/<agent-id>/agent.json` -- the agent identity, including an
  `accounts` list whose `credentials[*].env_var_name` field names the
  env var the operator put the literal secret into (e.g.
  `"GITHUB_TOKEN"`). The literal value lives only in the environment;
  the agent state never carries it.

To change a forge URL, edit `gateway.json` (no env-var indirection
needed).  To rotate a secret, change the env var the agent identity
points to and restart the gateway.

### Environment variables

Only secrets are read from the environment:

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_URL` | Yes | Base URL of an OpenAI-compatible LLM API |
| `OPENAI_API_KEY` | Yes | API key for the LLM provider |
| `OPENAI_API_MODEL_NAME` | Yes | Model name (e.g. `gpt-4o`, `claude-sonnet-4-20250514`) |
| `GITHUB_TOKEN` | For GitHub | Personal access token with `repo`, `notifications`, and (if needed) `read:org` scopes.  Default name written by `bootstrap`; any name may be used by editing the agent identity JSON. |
| `GITLAB_TOKEN` | For GitLab | Personal access token with API scope.  Default name written by `bootstrap`; any name may be used by editing the agent identity JSON. |

The polling interval, GitLab username, GitHub repository list, and
similar non-secret values now live in `gateway.json` (or the
inferred-source defaults).  Edit the JSON to change them.

Deployment modes
----------------

Thorn supports two deployment topologies.  The choice mainly comes
down to whether you want the gateway itself to be a container or a
host process.  In both shapes, the OneCLI credential broker
(Phase D) is brought up and torn down automatically by `thorn serve`
-- you do not need to run any compose commands yourself.

### Mode A: just run `thorn serve` (recommended)

The simplest and most portable shape: the gateway runs directly on
the host (a VM, a developer laptop, anywhere `uv run thorn` works).
On startup, the gateway brings up its own dedicated OneCLI + Postgres
stack with anonymous Docker volumes, mints an admin API key in
process memory, joins per-agent sandbox containers to the broker's
network, and -- on graceful shutdown -- runs `compose down --volumes`
so no broker artefacts survive on disk.

The only operator-visible requirement beyond the LLM provider env
vars is having `docker` (or `podman`) installed; `thorn serve`
discovers and uses whichever is present.  See
[`docs/startup_flow.md`](docs/startup_flow.md) for the bring-up
sequence in detail.

```console
$ uv run thorn serve
```

A bare-minimum `gateway.json` (no `sandbox` block, no `broker`
block) gives you secure container sandboxing + bundled broker by
default.  Explicit opt-out is supported:

```jsonc
{
  // Bypass the bundled broker (and the container sandbox);
  // the gateway falls back to subprocess-backed agent execution.
  "sandbox": { "backend": "subprocess" }
}
```

```jsonc
{
  // Keep container sandboxing but do not bring up a broker; agent
  // credentials flow through the legacy env-injection path.
  "broker": { "enabled": false }
}
```

This is the recommended mode for VM deployments (e.g. NVIDIA Brev;
see `deploy/brev/`) and for development.

If a previous `thorn serve` was killed ungracefully (e.g. `kill -9`)
and left an orphaned broker stack behind, `thorn broker status`
lists matching compose projects and `thorn broker down` cleans them
up.

For a single operator view across gateway liveness, source polling,
provider health, broker/sandbox state, in-flight external keys, and
durable inbox counts:

```console
$ uv run thorn status --agency ~/.thorn
```

Add `--json` for scripts and dashboards.  To inspect the local work
queue before requeueing anything:

```console
$ uv run thorn inbox list --agency ~/.thorn
$ uv run thorn inbox show <item-id> --agency ~/.thorn
```

`thorn status` remains useful while the gateway is stopped; live-only
signals such as heartbeat freshness and source polling are reported
as stopped, stale, or unknown from the last heartbeat file.

If the gateway receives a forge notification and then cannot prompt
the coordinator -- for example because the LLM provider key was bad
or the provider was unavailable long enough for the item to be parked
as errored -- fix the underlying operator-side problem and requeue
the durable inbox item:

```console
$ uv run thorn inbox requeue <item-id> --agency ~/.thorn
```

Use `--agent` and `--session` when multiple parked items have the
same ID.  This command only moves Thorn's local `inbox/errored/`
file back to pending work; it does not contact GitLab/GitHub or
re-create upstream TODOs.

Before letting a live gateway consume forge notifications, run a
safe Git connectivity preflight from the same agency config:

```console
$ uv run thorn serve --agency ~/.thorn preflight
```

The default preflight starts the configured sandbox and broker path,
then runs `git ls-remote` against each configured project clone URL
from inside the sandbox. It does not start event sources and does not
read, mark done, or otherwise consume GitLab/GitHub notifications.
Add `--write-check` only when you want to verify that the bot
credential can push and delete a temporary branch.

Sandbox egress is controlled today by `sandbox.egress_network` and
the OCI network topology behind it. The bundled broker uses an
internal network so sandbox containers reach the broker but have no
direct internet egress. `sandbox.planned_egress_allowlist` is only a
record of future direct-egress exceptions; it has no runtime effect
and logs a warning when non-empty. The old active-sounding
`sandbox.egress_allowlist` key is rejected during config loading.

Restricted-egress VMs can mirror the broker images into a reachable
registry and point the bundled stack at those mirrors without local
retagging:

```jsonc
{
  "broker": {
    "mode": "bundled",
    "bundled_images": {
      "onecli": "registry.example.com/mirror/onecli:latest",
      "postgres": "registry.example.com/mirror/postgres:18-alpine"
    }
  }
}
```

For host-wide overrides, set
`THORN_BUNDLED_BROKER_ONECLI_IMAGE` and
`THORN_BUNDLED_BROKER_POSTGRES_IMAGE` in the `thorn serve`
environment.  Config values take precedence over those env vars.
`thorn broker status` and `thorn broker down` still find these
stacks by Thorn's compose project prefix, independent of image names.

### Mode A advanced: external (operator-managed) broker

When you already run an OneCLI broker for other workloads, point
Thorn at it instead of having the gateway manage one of its own:

```jsonc
{
  "broker": {
    "mode": "external",
    "admin_url": "http://my-broker:10254",
    // Names the env var the operator put the literal admin key
    // into; gateway.json never carries the secret value at rest.
    "admin_api_key_env_var": "ONECLI_ADMIN_KEY",
    "proxy_url": "http://my-broker:10255"
  }
}
```

In this mode the gateway never invokes `docker compose`; it just
makes admin-API calls against the URLs you configured.  This is the
right shape when the broker outlives `thorn serve` (e.g. shared
across multiple agencies on a host).

### Mode B: gateway in a container alongside the broker (optional)

If you prefer the gateway itself to be a container, run the wheel
inside any Python image you like and bind-mount `.thorn` and the
host's Docker socket so the gateway can spawn the bundled broker
and sandbox sibling containers.

> **Trade-off.**  Bind-mounting `/var/run/docker.sock` gives the
> gateway container effective root on the host, since anything it
> can ask the Docker daemon to do (launch privileged containers,
> mount arbitrary host paths, etc.) bypasses the container
> boundary.  For single-tenant operator-controlled deployments
> this is fine; do not run this mode where the gateway is exposed
> to untrusted input that could compromise the gateway process.
> Mode A avoids the socket mount entirely and is strictly simpler;
> prefer it unless you have a concrete reason to want the gateway
> in a container.

### Plain Docker (no broker, no sandbox)

If you don't need the credential broker -- e.g. for a quick test
of the gateway against a single forge with env-injected
credentials -- the simplest shape is `docker run` with a `.thorn`
bind mount and a `gateway.json` that explicitly opts out of the
secure defaults (`{"sandbox": {"backend": "subprocess"}}`).
No compose file required.

You can of course write your own minimal compose file too; the
gateway service shape is small:

```yaml
services:
  gateway:
    build: .
    env_file: .env
    environment:
      GIT_AUTHOR_NAME: my-bot
      GIT_COMMITTER_NAME: my-bot
      GIT_AUTHOR_EMAIL: my-bot@users.noreply.github.com
      GIT_COMMITTER_EMAIL: my-bot@users.noreply.github.com
    volumes:
      - ./.thorn:/workspace/.thorn
    restart: unless-stopped
```

Customize `GIT_AUTHOR_NAME` and `GIT_AUTHOR_EMAIL` to match the
identity you want on the agent's commits.

Development Toolchains
----------------------

The Docker image includes the following toolchains so the agent can
build and test projects without needing elevated privileges:

| Language | Tools | Notes |
|----------|-------|-------|
| C / C++ | gcc, g++, make, cmake, pkg-config | Via `build-essential` |
| Rust | rustc, cargo | Via `rustup` (stable channel) |
| Python | python3, pip | From base image (`python:3.12-slim`) |
| JavaScript / TypeScript | node, npm, tsc, ts-node | Node.js 22 LTS via NodeSource |

Architecture
------------

```
┌──────────────────────────────────────────────────┐
│  Docker container                                │
│                                                  │
│  ┌──────────────┐    ┌────────────────────────┐  │
│  │ Event Source  │───>│  ProjectCoordinator    │  │
│  │ (polls forge)│    │  (LLM-powered agent)   │  │
│  └──────────────┘    │                        │  │
│                      │  Tools:                │  │
│                      │  - forge_* (issues,    │  │
│                      │    PRs, comments)      │  │
│                      │  - git_* (clone, push) │  │
│                      │  - file I/O            │  │
│                      │  - run_shell           │  │
│                      └────────────────────────┘  │
│                                                  │
│  ┌──────────────┐    ┌────────────────────────┐  │
│  │ .thorn/      │    │ Agent workspaces       │  │
│  │ gateway.json │    │ (cloned repos, state)  │  │
│  └──────────────┘    └────────────────────────┘  │
└──────────────────────────────────────────────────┘
         │                        │
         ▼                        ▼
   GitHub / GitLab API      LLM Provider API
```

The gateway is forge-agnostic: the agent uses a unified set of
`forge_*` tools that work identically against GitHub and GitLab.

Further Reading
---------------

- [Threat model and trust boundaries](docs/threat-model.md) --
  what the peer registry, content envelopes, and tool-call sandbox
  defend against; the "gossipy co-worker" rule for what to share
  with an agent; operating guidance for public gateways.
- [What `thorn serve` does on startup](docs/startup_flow.md) --
  the bring-up sequence for the bundled credential broker, where
  to look in the logs, and how to recover from a non-graceful
  shutdown.
- [Python library and CLI reference](docs/library.md) -- using Thorn
  as a Python library for building custom agent workflows with
  `@tool`, `@skill`, prompt orchestration, and MCP tool serving.
- `examples/calc/` -- a demonstration project with `.thorn/` tools for
  building, testing, and a multi-agent development workflow.
