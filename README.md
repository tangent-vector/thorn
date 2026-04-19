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
| `agents/my-coordinator.json` | Agent identity, including the forge account and a `$GITHUB_TOKEN` reference for the secret token. |
| `agents/my-coordinator/MEMORY.md` | Persistent memory (project facts, active work) |
| `gateway.json` | Forge entries (with literal `base_url`) and project metadata.  No secrets live here -- the agent identity references them via `$ENV_VAR`. |

### 5. Start the gateway

Using Docker Compose (recommended):

```console
$ docker compose up --build
```

Or directly:

```console
$ docker run --env-file .env \
    -v "$(pwd)/.thorn:/workspace/.thorn" \
    thorn-gateway
```

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
  project metadata.  Contains *no* secrets and *no* `$ENV_VAR`
  references.
- `agents/<agent-id>.json` -- the agent identity, including a
  `forge_accounts` list whose `credentials.token` field is a
  `$ENV_VAR` reference (e.g. `"$GITHUB_TOKEN"`) that the gateway
  resolves at startup.

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

### Docker Compose

The included `docker-compose.yml` is the simplest way to run the
gateway. It reads `.env`, mounts `.thorn/` as a volume (so agent state
persists across restarts), and configures the git identity for commits:

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

- [Python library and CLI reference](docs/library.md) -- using Thorn
  as a Python library for building custom agent workflows with
  `@tool`, `@skill`, prompt orchestration, and MCP tool serving.
- `examples/calc/` -- a demonstration project with `.thorn/` tools for
  building, testing, and a multi-agent development workflow.
