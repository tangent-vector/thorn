# Thorn

[![CI](https://github.com/tangent-vector/thorn/actions/workflows/ci.yml/badge.svg)](https://github.com/tangent-vector/thorn/actions/workflows/ci.yml)

Thorn is an experimental persistent agent harness built on OpenAI-compatible
completion APIs. It runs coding agents in local CLI sessions or as a long-lived
service that responds to GitHub and GitLab activity.

Thorn persists agent identity, sessions, inbox work, and working state across
invocations. Its gateway mode adds forge event sources, concurrent session
scheduling, sandboxed tool execution, and brokered credentials around that same
runtime.

> [!IMPORTANT]
> Thorn is research software under active development. Use dedicated,
> least-privilege forge and provider credentials, review the
> [threat model](docs/threat-model.md), and supervise it until its behavior and
> operating boundaries fit your environment.

## The agency model

An **agency** is a configured collection of agents and their associated state.
The term refers both to the persisted agency on disk and to its active runtime
embodiment. These are two states of the same domain entity.

- An **agent** is one participant in an agency, with its own identity, policy,
  sessions, durable state, and tool context.
- An **agency home** is the on-disk root for agency configuration and
  framework-owned state. It is a storage location, not a separate runtime
  concept.
- `thorn run` and `thorn chat` interact with an agency that is usually persisted
  and run locally under one user's account.
- A **gateway** is the service-facing mode in which a running agency operates as
  a daemon and remote peers interact with it indirectly, currently through
  configured GitHub or GitLab events. A gateway hosts an agency; it is not a
  synonym for the agency or its state.

The current system accepts instructions from multiple configured peers and
projects inside one agency trust domain. It is not an adversarial multi-tenant
service. See the [current architecture](docs/architecture.md) for the complete
execution and trust-boundary model.

## Try the local CLI

You need [uv](https://docs.astral.sh/uv/), Python 3.11 or newer, and an
OpenAI-compatible LLM endpoint. Install the `thorn` executable directly from
GitHub:

```console
$ uv tool install git+https://github.com/tangent-vector/thorn.git
```

From a repository you want Thorn to inspect, provide your endpoint and run one
prompt:

```console
$ OPENAI_API_URL=https://api.example.com/v1 \
    OPENAI_API_KEY=... \
    OPENAI_API_MODEL_NAME=example-model \
    thorn run "Summarize this repository"
```

`thorn run` creates a fresh persistent session. By default, local agency state
lives under `~/.thorn`, while the current directory is the session workspace.
Use `thorn chat` for an interactive session or `thorn sessions list` to find a
session that can be resumed.

The base installation provides the lightweight CLI surface. Gateway operation
uses a source checkout plus the explicit extra for each configured forge.

## Run an agency as a gateway

Gateway operation adds a forge identity, one or more projects and peers, a
long-lived event loop, and a stronger default tool boundary. The recommended
deployment keeps Thorn's LLM-facing process on the host while each agent's
tools run in a separate OCI sandbox.

```console
$ git clone https://github.com/tangent-vector/thorn.git
$ cd thorn
$ uv sync --locked --no-dev --extra github
$ uv run thorn sandbox build
```

Use `--extra gitlab` instead for a GitLab agency. Pass both `--extra github`
and `--extra gitlab` only when one agency connects to both forge types.

After configuring the agency and its credentials, check it before consuming
live forge notifications:

```console
$ uv run thorn agency check --agency ~/.thorn
$ uv run thorn serve --agency ~/.thorn preflight
$ uv run thorn serve --agency ~/.thorn
```

The [agency operations guide](docs/operations.md) covers configuration,
credentials, bootstrap, preflight, deployment choices, status, and recovery.

## Current system shape

```text
 local user                         configured forge peers
     |                                       |
 thorn run / chat                  GitHub / GitLab events
     |                                       |
     +---------------+   +-------------------+
                     v   v
             +------------------+
             |  running agency  |---------> LLM provider API
             |                  |
             | agents           |
             | sessions         |
             | durable inboxes  |
             | schedulers       |
             +--------+---------+
                      |
                tool protocol
                      |
             +--------v---------+
             | per-agent        |
             | toolhost         |
             | subprocess or    |
             | OCI sandbox      |
             +--------+---------+
                      |
               workspace, Git,
              forge APIs (with
             brokered credentials)
```

Both entry paths use the same persistent runtime and session machinery. In the
local CLI today, tools run through a subprocess toolhost with the invoking
user's authority. In the default gateway configuration, tools run inside
per-agent containers and reach credential-bearing services through a broker.
The LLM-facing process and framework-owned state remain outside the tool
sandbox.

## Implemented and exploratory scope

Implemented today:

- persistent CLI and gateway sessions with durable inbox state;
- concurrent per-agent and per-session scheduling;
- GitHub and GitLab event sources and forge tools;
- peer-aware trigger authorization within one agency trust domain;
- subprocess and container-backed tool execution; and
- bundled credential brokering, operator preflight, status, and recovery
  commands.

Still exploratory or incomplete:

- runtime-enforced memory scopes and promotion policy;
- explicit agent-to-agent and cross-session coordination primitives;
- adversarial multi-tenant isolation;
- bounded provider-spend and unattended-work controls; and
- reproducible performance measurements from the existing evaluation adapter.

The [roadmap](ROADMAP.md) distinguishes implemented foundations from these
research directions.

## Documentation

The [documentation index](docs/README.md) separates current reference and
operator material from aspirational designs, research ideas, and historical
implementation plans. Good starting points are:

- [Current architecture](docs/architecture.md)
- [Agency operations](docs/operations.md)
- [Threat model and trust boundaries](docs/threat-model.md)
- [Gateway startup and recovery flow](docs/startup_flow.md)
- [Roadmap](ROADMAP.md)

## Development

Thorn uses `uv` for its development environment and commands:

```console
$ uv sync --all-extras
$ uv run pytest
$ uv run ruff check .
```

Read [CONTRIBUTING.md](CONTRIBUTING.md) before proposing a change. Report
potential vulnerabilities privately as described in [SECURITY.md](SECURITY.md).

The CLI and gateway operating modes are the end-user surfaces. The Python
library API is supporting infrastructure and is not a stable compatibility
contract.

## License

Thorn is available under the [MIT License](LICENSE).
