# Operating a Thorn Agency

This guide covers the current installation, configuration, gateway startup,
status, and recovery paths. For the concepts and component boundaries behind
these procedures, read the [current architecture](architecture.md).

## Two operating modes, one agency model

`thorn run` and `thorn chat` interact directly with a locally running agency.
`thorn serve` keeps an agency running as a daemon so remote peers can interact
with it through configured forge events. Both modes use the same runtime,
persisted agents, sessions, and inbox machinery.

The current CLI path does not require an agency configuration file. It builds
the local agency configuration from command options and provider environment,
stores state under `~/.thorn` by default, and uses the selected working
directory as its workspace.

Gateway operation requires an agency configuration file, persisted agent
identity, forge account, and long-lived workspace.

## Local CLI

### Requirements

- [uv](https://docs.astral.sh/uv/)
- Python 3.11 or newer
- an LLM provider exposing an OpenAI-compatible API

Install the base CLI directly from GitHub:

```console
$ uv tool install git+https://github.com/tangent-vector/thorn.git
```

Run one prompt from the repository that should become the session workspace:

```console
$ OPENAI_API_URL=https://api.example.com/v1 \
    OPENAI_API_KEY=... \
    OPENAI_API_MODEL_NAME=example-model \
    thorn run "Summarize this repository"
```

Use an interactive session with the same environment:

```console
$ thorn chat
```

Each invocation creates a fresh `cli/<workspace-name>/<id>` session unless
`--resume` names an existing session. List persisted local sessions with:

```console
$ thorn sessions list --agency ~/.thorn
```

Resume one with either command:

```console
$ thorn chat --resume cli/my-repo/1a2b3c4d --agency ~/.thorn
$ thorn run --resume cli/my-repo/1a2b3c4d --agency ~/.thorn \
    "Continue from the previous result"
```

A resumed session uses its recorded workspace. Thorn rejects a conflicting
`--workspace` instead of silently moving the conversation to another project.

The local CLI currently uses a subprocess toolhost. Shell-capable tools run
with the invoking user's OS authority, so the subprocess is not a security
boundary for untrusted instructions.

## Gateway requirements

Running an agency as a gateway currently uses a source checkout so the checked-
in sandbox image can be built. You need:

- the local CLI requirements above;
- Docker with Compose, or Podman with a working Compose provider;
- a dedicated GitHub or GitLab identity and least-privilege token; and
- an agency home separate from the workspace where agents perform work.

Clone and install all integration extras:

```console
$ git clone https://github.com/tangent-vector/thorn.git
$ cd thorn
$ uv sync --all-extras --locked
```

Build the versioned sandbox image before starting the agency:

```console
$ uv run thorn sandbox build
```

Thorn chooses Podman when available and otherwise Docker. If both are installed,
select one explicitly and use the same value in agency configuration:

```console
$ uv run thorn sandbox build --runtime docker
```

```yaml
sandbox:
  backend: container
  oci_runtime: docker
```

The bundled broker follows the configured OCI runtime so its network and the
agent sandboxes are not split between Docker and Podman.

The default per-agent sandbox image is deliberately small: it contains Thorn,
Python, Git, uv, and the MCP client, but not general C++, Rust, Node.js, or
project-specific SDKs. Derive an image from the generated
`thorn-sandbox:<version>` tag and select it with `sandbox.image` when an agency
needs additional toolchains. The repository-root `Dockerfile` builds an
optional image for the gateway process itself; it is not the default agent tool
sandbox.

## Configuration files and agency creation

A service-running agency home contains exactly one supported agency
configuration filename:

1. `agency.yaml` (preferred)
2. `agency.json`
3. `gateway.yaml` (compatibility name)
4. `gateway.json` (compatibility name)

The file describes the agency-wide workspace, LLM defaults, services, projects,
peers, sandbox, and broker. Agent identities and their account references live
separately under `agents/<agent-id>/agent.json`.

The code currently has two creation paths:

- `thorn agency init` creates a minimal edit-first `agency.yaml` and workspace.
  It does not yet create or configure an agent.
- `thorn serve bootstrap` is an opinionated convenience path that creates one
  coordinator agent plus a project-oriented `gateway.json`. The compatibility
  filename and `ProjectCoordinator` role are current implementation details,
  not a distinct kind of agency.

Create an empty edit-first agency with:

```console
$ uv run thorn agency init ~/.thorn --workspace ~/thorn-workspace
$ uv run thorn agency check --agency ~/.thorn
$ uv run thorn agency show --agency ~/.thorn --json
```

For the shortest current project-and-agent bootstrap, start from an empty
agency home and use a public GitHub or GitLab project URL:

```console
$ uv run thorn serve bootstrap \
    --agent-id my-coordinator \
    --project-name my-repo \
    --project-url https://github.com/owner/repo \
    --llm-api-url https://api.example.com/v1 \
    --llm-model example-model \
    --llm-api-key-env OPENAI_API_KEY \
    --agency-home ~/.thorn \
    --agency-workspace ~/thorn-workspace
```

Use `https://gitlab.com/group/project` for GitLab. Bootstrap records
`GITHUB_TOKEN` or `GITLAB_TOKEN` as the default token environment-variable name;
pass `--token-env` to choose a different name.

Bootstrap does not decide who should be trusted to instruct the agency. Before
live use, add at least one peer with the forge's immutable numeric account ID to
the generated agency configuration:

```json
{
  "peers": [
    {
      "id": "owner",
      "name": "Repository Owner",
      "kind": "human",
      "accounts": [
        { "service": "github", "account_id": "123456" }
      ]
    }
  ]
}
```

Merge this field into the generated `gateway.json`; do not replace the
workspace, LLM, or project fields already there. Use service `gitlab` for the
public GitLab bootstrap. An empty peer list deliberately refuses to treat any
outside user's conversational event as an instruction.

Complete configuration skeletons live under
[`examples/gateway/`](../examples/gateway/). They cover public GitHub and
GitLab PATs, bundled-broker mirrors, subprocess opt-out, self-hosted GitLab, and
peer account IDs.

### Agency layout

The bootstrap path creates:

| Path | Purpose |
|---|---|
| `<agency-home>/gateway.json` | Compatibility-named agency configuration; contains no literal secrets. |
| `<agency-home>/agents/<agent-id>/agent.json` | Framework-owned agent identity and credential references. |
| `<agency-home>/agents/<agent-id>/home/` | Agent-visible durable files such as `MEMORY.md` and journals. |
| `<agency-home>/agents/<agent-id>/sessions/` | Framework-owned session history and inbox state. |
| `<workspace>/agents/<agent-id>/workspace/` | Agent-visible session workspaces. |
| `<workspace>/agents/<agent-id>/control/` | Framework/toolhost control files. |

## Credentials and configuration

Do not put literal provider or forge credentials in agency configuration,
agent state, or a repository checkout. Configuration records the names of
environment variables that hold secrets.

For example:

```dotenv
OPENAI_API_KEY=...
GITHUB_TOKEN=...
```

For GitLab, use `GITLAB_TOKEN`. Export variables in the service environment or
point Thorn at one explicit dotenv file:

```console
$ uv run thorn --env-file ~/.config/thorn/secrets.env \
    serve --agency ~/.thorn preflight
```

`THORN_ENV_FILE` can name the same file. Thorn does not search for `.env` in the
current working tree.

Use a dedicated bot credential restricted to the intended repositories and
forks. It should be able to read relevant work, create agent-owned branches,
open change requests, comment, and receive or acknowledge notifications. Avoid
credentials that can administer or delete repositories, change organization
settings, merge protected branches, or act across unrelated projects.

Peers are keyed by each forge's immutable numeric user ID. If an older config
uses handles, resolve them before starting live polling:

```console
$ uv run thorn serve --agency ~/.thorn resolve-peers
```

## Preflight and startup

Validate the agency and inspect its resolved shape:

```console
$ uv run thorn agency check --agency ~/.thorn
$ uv run thorn agency show --agency ~/.thorn
```

Before allowing the gateway to consume notifications, run:

```console
$ uv run thorn serve --agency ~/.thorn preflight
```

Preflight validates provider environment, agency and agent account
configuration, event-source inference, forge API access, sandbox and broker
startup, and `git ls-remote` from inside the sandbox. It does not start event
sources or consume forge notifications. Add `--write-check` only against a
purpose-created repository when the bot may create and delete a temporary
branch.

Start the agency in gateway mode:

```console
$ uv run thorn serve --agency ~/.thorn
```

The inferred event sources poll the configured forge identity. Direct mentions,
assignments, review requests, and supported lifecycle events are routed through
peer policy into durable session inboxes.

## Status and recovery

Inspect agency health even while the gateway is stopped:

```console
$ uv run thorn status --agency ~/.thorn
$ uv run thorn inbox list --agency ~/.thorn
$ uv run thorn inbox show <item-id> --agency ~/.thorn
```

Forge notification acknowledgement is based on successful handoff to Thorn,
not completion of the resulting agent work. After handoff, the durable local
inbox is the recovery source of truth. If an item is parked as errored, correct
the operator-side problem and requeue it:

```console
$ uv run thorn inbox requeue <item-id> --agency ~/.thorn
```

Requeueing does not recreate or modify the upstream GitHub notification or
GitLab TODO. See [gateway startup and recovery](startup_flow.md) for the exact
acknowledgement and startup-sweep behavior.

After an ungraceful exit, inspect and remove orphaned bundled-broker stacks:

```console
$ uv run thorn broker status
$ uv run thorn broker down
```

## Deployment and isolation choices

The recommended topology runs `thorn serve` directly on a dedicated host or VM.
Thorn starts the bundled OneCLI/Postgres broker stack, joins per-agent sandbox
containers to its proxy network, and tears the stack down on graceful shutdown.
See [gateway startup and recovery](startup_flow.md) for the lifecycle.

The secure defaults apply when an agency configuration omits both `sandbox` and
`broker`: tool execution uses per-agent containers and credentials use the
bundled broker.

Explicit subprocess opt-out is available for local development:

```yaml
sandbox:
  backend: subprocess
```

This disables the tool sandbox and bundled broker. Shell tools then execute with
the gateway host account's authority. Do not use this topology for a gateway
that processes untrusted external events or shares a host with users who do not
belong to the agency trust domain.

Container execution without a broker is also possible:

```yaml
broker:
  enabled: false
```

Credentials then use the legacy environment-injection path and are visible to
the agent container. The bundled broker is the stronger default.

An externally managed broker can be declared when its lifecycle should outlive
one Thorn process:

```yaml
broker:
  mode: external
  admin_url: https://broker.example.com
  admin_api_key_env_var: ONECLI_ADMIN_KEY
  proxy_url: http://broker.example.com:10255
```

Running the gateway process itself inside a container is possible, but allowing
it to create sibling sandboxes normally requires mounting the host OCI socket.
That gives the gateway container daemon-level authority over the host and is
not the recommended default.

The optional [Brev recipe](../deploy/brev/README.md) documents one VM-oriented
deployment. A provider-neutral CPU-hosted recipe remains roadmap work.

## Additional reference

- [Current architecture](architecture.md)
- [Gateway startup and recovery](startup_flow.md)
- [Threat model](threat-model.md)
- [Configuration examples](../examples/gateway/)
- [Policy templates](policy-templates/README.md)
