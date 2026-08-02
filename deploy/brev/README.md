# Deploying Thorn on NVIDIA Brev

This directory contains an experimental recipe for running `thorn serve` on a
public [NVIDIA Brev](https://brev.nvidia.com/) instance. Thorn uses hosted LLM
APIs and does not itself require a GPU, so Brev is an optional deployment
target rather than the default hosting recommendation.

> [!CAUTION]
> This recipe has been checked against the public Brev CLI surface but has not
> yet completed Thorn's outside-NVIDIA deployment rehearsal. Review the
> commands, instance price, sandbox runtime availability, and credential scope
> before using `--start`.

## Prerequisites

- Linux, macOS, or WSL.
- A normal public Brev account and an authenticated `brev` CLI.
- A dedicated LLM provider key.
- A dedicated, least-privilege GitHub or GitLab bot credential.
- A Thorn agency configured for the repository the bot may access.

Install and authenticate the CLI using Brev's
[current instructions](https://docs.nvidia.com/brev/latest/cli/getting-started):

```console
$ bash -c "$(curl -fsSL https://raw.githubusercontent.com/brevdev/brev-cli/main/bin/install-latest.sh)"
$ brev login
$ brev list
```

## What the helper does

From a Thorn source checkout:

```console
$ ./deploy/brev/deploy.sh \
    --repo https://github.com/tangent-vector/thorn.git
```

By default, the helper:

1. creates the Brev instance from the public repository;
2. runs `deploy/brev/setup.sh` on the instance;
3. installs `uv` when needed and runs `uv sync --all-extras --locked`; and
4. stops before configuring or starting the gateway.

The safe default leaves account creation, credentials, agency review, and
preflight under operator control. Use `--gpu` only when you intentionally want
a particular Brev instance type; Thorn itself does not need GPU compute.

## Configure an agency

Bootstrap on the instance:

```console
$ brev shell thorn-gateway
$ cd /home/ubuntu/workspace/thorn
$ uv run thorn serve bootstrap \
    --agent-id my-coordinator \
    --project-name my-project \
    --project-url https://github.com/owner/repository \
    --llm-api-url https://api.example.com/v1 \
    --llm-model example-model \
    --llm-api-key-env OPENAI_API_KEY \
    --agency-home /home/ubuntu/workspace/thorn/.thorn \
    --agency-workspace /home/ubuntu/workspace/thorn-workspace
```

Alternatively, copy an agency you have already reviewed:

```console
$ ./deploy/brev/deploy.sh \
    --repo https://github.com/tangent-vector/thorn.git \
    --agency-dir /path/to/reviewed-agency
```

Do not point the agent at the Thorn repository itself for a first rehearsal.
Use a purpose-created private scratch repository and a bot identity that cannot
administer repositories, change settings, merge protected branches, or access
unrelated projects.

## Supply secrets

Thorn does not discover `.env` from the repository working tree. Create a
dedicated local file from `deploy/brev/env.template`, keep it outside every Git
checkout, and restrict its permissions:

```console
$ install -m 600 deploy/brev/env.template ~/.config/thorn/brev-secrets.env
$ $EDITOR ~/.config/thorn/brev-secrets.env
```

When explicitly requested, the deploy helper copies that file to
`/home/ubuntu/workspace/.config/thorn/secrets.env` and sets mode `0600`:

```console
$ ./deploy/brev/deploy.sh \
    --repo https://github.com/tangent-vector/thorn.git \
    --env-file ~/.config/thorn/brev-secrets.env
```

This is a pragmatic rehearsal path, not a claim that a plaintext file is an
ideal production secret store. If your Brev account exposes a managed secret
injection mechanism, prefer it and launch Thorn with those variables already
present in the service environment.

## Validate and start

Before starting, confirm that Docker or Podman is available in the Brev
environment. Thorn's default gateway requires an OCI runtime for its sandbox
and bundled credential broker.

Run the steps manually first:

```console
$ brev shell thorn-gateway
$ cd /home/ubuntu/workspace/thorn
$ uv run thorn sandbox build
$ uv run thorn --env-file /home/ubuntu/workspace/.config/thorn/secrets.env \
    serve --agency /home/ubuntu/workspace/thorn/.thorn preflight
$ uv run thorn --env-file /home/ubuntu/workspace/.config/thorn/secrets.env \
    serve --agency /home/ubuntu/workspace/thorn/.thorn
```

After that path is understood, `--start` performs the sandbox build and
non-destructive preflight before launching the gateway in a detached tmux
session:

```console
$ ./deploy/brev/deploy.sh \
    --repo https://github.com/tangent-vector/thorn.git \
    --agency-dir /path/to/reviewed-agency \
    --env-file ~/.config/thorn/brev-secrets.env \
    --start
```

The write-capability extension to preflight is deliberately not automated.
Run `preflight --write-check` yourself only against the scratch repository.

## Operations

| Task | Command |
|---|---|
| Connect | `brev shell thorn-gateway` |
| Attach to gateway | `brev shell thorn-gateway`, then `tmux attach -t thorn` |
| View logs | `brev exec thorn-gateway "tail -f /tmp/thorn-serve.log"` |
| Stop gateway | `brev exec thorn-gateway "tmux kill-session -t thorn"` |
| Stop instance | `brev stop thorn-gateway` |
| Restart instance | `brev start thorn-gateway` |
| Delete instance | `brev delete thorn-gateway` |

Files under `/home/ubuntu/workspace` persist across stops but are deleted with
the instance. Preserve agency state and agent work deliberately, scan retained
logs for secrets, and record the observed provider and instance cost.

If an instance IP changes, run `brev refresh`. If setup is incomplete, connect
and rerun `deploy/brev/setup.sh`; it is intended to be idempotent.
