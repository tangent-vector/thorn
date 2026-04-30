Thorn
=====

Thorn is an autonomous coding agent that monitors your GitHub or GitLab
repositories, responds to @-mentions and assignments, and carries out
development tasks: reading issues, cloning repos, making changes,
pushing branches, and opening pull requests.

It runs as a long-lived **gateway** daemon -- typically inside a Docker
container -- that polls a forge for activity and dispatches work to an
LLM-powered agent.

Quick Start (Docker)
--------------------

### Prerequisites

- Docker (and optionally Docker Compose)
- An LLM provider exposing an OpenAI-compatible API
- A GitHub personal access token (PAT), or a GitHub App installation

### 1. Clone the repository

```console
$ git clone https://github.com/tangent-vector/thorn.git
$ cd thorn
```

### 2. Create a `.env` file

The gateway only reads **secrets** from environment variables; all
other configuration (forge base URL, project metadata, polling cadence,
etc.) lives in `gateway.json` and the per-agent identity JSON files
that `thorn serve bootstrap` writes for you.  Create a `.env` file at
the repository root with just the secrets:

```dotenv
# LLM provider (any OpenAI-compatible API)
OPENAI_API_URL=https://api.openai.com/v1
OPENAI_API_KEY=sk-...
OPENAI_API_MODEL_NAME=gpt-4o

# GitHub (PAT auth) -- the only required forge secret for GitHub.
GITHUB_TOKEN=ghp_...
```

For GitLab, set `GITLAB_TOKEN` instead (and pass `--forge-type gitlab
--forge-base-url https://your-gitlab.example.com/api/v4` to bootstrap
in step 4).

> **Do not commit `.env` to version control.** It contains secrets.

### 3. Build the Docker image

```console
$ docker build -t thorn-gateway .
```

The image includes development toolchains for C/C++ (gcc, cmake),
Rust (rustup), Python, and Node.js/TypeScript so the agent can build
and test code in these languages.

### 4. Bootstrap the agent

Before the gateway can run, it needs an agent identity and service
configuration. The `thorn serve bootstrap` command creates these inside
a `.thorn/` directory:

```console
$ docker run --rm --env-file .env \
    -v "$(pwd)/.thorn:/workspace/.thorn" \
    thorn-gateway \
    thorn serve bootstrap \
      --agent-id my-coordinator \
      --project-name my-repo \
      --clone-url https://github.com/owner/repo.git \
      --native-project-id owner/repo \
      --forge-type github
```

For a self-hosted GitHub Enterprise instance (or any non-default
host), pass `--forge-base-url https://ghe.example.com/api/v3`.  For
GitLab, `--forge-base-url` is **required** since GitLab has no
canonical default host.

This writes three files under `.thorn/`:

| File | Purpose |
|------|---------|
| `agents/my-coordinator.json` | Agent identity, including the forge account and a `credentials[*].env_var_name` reference (e.g. `"GITHUB_TOKEN"`) naming the env var the operator put the literal token into. |
| `agents/my-coordinator/MEMORY.md` | Persistent memory (project facts, active work) |
| `gateway.json` | Forge entries (with literal `base_url`) and project metadata.  No secrets live here -- the agent identity names env vars from which the gateway reads the literal at use time. |

### 5. Start the gateway

```console
$ docker run --env-file .env \
    -v "$(pwd)/.thorn:/workspace/.thorn" \
    thorn-gateway
```

For sandboxed multi-agent deployments with the credential broker
(Phase D), see [Deployment modes](#deployment-modes) below -- with
the new defaults this is just `thorn serve` against an unpopulated
`broker` / `sandbox` block.

The gateway will poll the configured repository for events. When it
sees activity (an @-mention on an issue, a new assignment, etc.), it
dispatches the event to the agent, which reads the issue, clones the
repo, and does the requested work.

### 6. Talk to the agent

Open an issue on the configured repository and @-mention the bot
account (the GitHub user whose PAT you provided, or the GitHub App).
The agent will pick up the event on its next poll cycle, read the
issue, and respond.

Configuration
-------------

### Configuration model

Thorn distinguishes **secrets** (held only in environment variables)
from all other configuration (held in JSON on disk).  The on-disk
files live under `.thorn/`:

- `gateway.json` -- forge entries (name, type, literal `base_url`) and
  project metadata.  Contains *no* secrets.
- `agents/<agent-id>.json` -- the agent identity, including an
  `accounts` list whose `credentials[*].env_var_name` field names
  the env var the operator put the literal secret into (e.g.
  `"GITHUB_TOKEN"`).  The literal value lives only in the
  environment; the agent state never carries it.

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

- [What `thorn serve` does on startup](docs/startup_flow.md) --
  the bring-up sequence for the bundled credential broker, where
  to look in the logs, and how to recover from a non-graceful
  shutdown.
- [Python library and CLI reference](docs/library.md) -- using Thorn
  as a Python library for building custom agent workflows with
  `@tool`, `@skill`, prompt orchestration, and MCP tool serving.
- `examples/calc/` -- a demonstration project with `.thorn/` tools for
  building, testing, and a multi-agent development workflow.
