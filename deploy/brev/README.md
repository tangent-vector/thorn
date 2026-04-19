# Deploying Thorn on NVIDIA Brev

Minimal scripts for running the Thorn gateway on an [NVIDIA Brev](https://brev.nvidia.com/) GPU instance.

## Prerequisites

| Requirement | Notes |
|-------------|-------|
| Linux or WSL | All commands and scripts here assume a Unix environment. On Windows, clone and work from inside WSL -- do **not** run these scripts from a Windows checkout (CRLF line endings will break them). |
| `brev` CLI | Install: `sudo bash -c "$(curl -fsSL https://raw.githubusercontent.com/brevdev/brev-cli/main/bin/install-latest.sh)"` |
| Brev account | Sign up at <https://brev.nvidia.com/> |
| Authenticated CLI | Run `brev login` once after installing. |

## Secrets (recommended)

Thorn only reads **secrets** from environment variables.  Forge base
URLs, repository identifiers, polling intervals, and similar non-secret
configuration are written into `.thorn/gateway.json` and the per-agent
identity JSON by `thorn serve bootstrap`; you do not need to (and
should not) set them as Brev secrets.

Configure credentials with [Brev secrets](https://docs.nvidia.com/brev/latest/cli/advanced-commands#secrets-management) so nothing sensitive is copied to the instance as a `.env` file. Each name becomes an environment variable on the instance. Create one secret per variable you need:

**LLM (required for `thorn run` / `thorn serve`):**

```bash
brev secret create OPENAI_API_URL
brev secret create OPENAI_API_KEY
brev secret create OPENAI_API_MODEL_NAME
# Optional:
# brev secret create OPENAI_MAX_TOKENS
```

**Forge access tokens (set whichever forge(s) you use):**

```bash
# GitHub bot user PAT (default name; configurable via --access-token-env)
brev secret create GITHUB_TOKEN

# GitLab PAT (default name; configurable via --access-token-env)
# brev secret create GITLAB_TOKEN
```

Each `brev secret create` prompts for the value. Secrets are encrypted at rest and **injected when the instance starts** ([Brev docs](https://docs.nvidia.com/brev/latest/cli/advanced-commands#secrets-management)). If you add or change secrets while the instance is already running, **stop and start** the instance (or restart it) so the new values appear in the environment before you rely on them for `thorn serve`.

`deploy/brev/env.template` lists the same names with short comments.

## Quick start (one command)

From the thorn repo root:

```bash
./deploy/brev/deploy.sh --repo https://github.com/your-org/thorn.git
```

This will:

1. Create a Brev instance named `thorn-gateway`
2. Clone the Thorn repo and run `pip install -e ".[github,gitlab]"` via the setup script
3. Start `thorn serve` inside a tmux session on the instance (unless you pass `--no-start`)

**Configure secrets** using the commands above, either before you run `deploy.sh` or afterward. Because injection happens at instance start, if you create or update secrets after the instance already exists, restart the instance, then start the gateway again (or re-run `deploy.sh` without `--no-start` if you use it to launch tmux).

The deploy script **does not** copy a local `.env` file unless you pass `--env-file` explicitly (see below).

### Common options

```bash
# Custom instance name and GPU type
./deploy/brev/deploy.sh --name my-thorn --gpu "nebius.l40sx1.pcie"

# Copy a pre-made agency directory instead of bootstrapping
./deploy/brev/deploy.sh --agency-dir /path/to/.thorn

# Legacy: copy a .env file from your machine to the instance (not recommended for secrets)
./deploy/brev/deploy.sh --env-file ~/secrets/thorn.env

# Don't auto-start the gateway
./deploy/brev/deploy.sh --no-start
```

Run `./deploy/brev/deploy.sh --help` for all options.

## Manual step-by-step

If you prefer to do things by hand, or the automated script doesn't suit your
situation:

### 1. Create the instance

```bash
brev start https://github.com/your-org/thorn.git \
    --name thorn-gateway \
    --setup-repo . \
    --setup-path deploy/brev/setup.sh
```

Or create an empty instance and clone manually:

```bash
brev start --name thorn-gateway
brev shell thorn-gateway
# on the instance:
cd /home/ubuntu/workspace
git clone https://github.com/your-org/thorn.git
cd thorn
pip install -e ".[github,gitlab]"
```

### 2. Configure environment variables

**Option A -- Brev secrets (recommended):**

Use `brev secret create` as described in [Secrets (recommended)](#secrets-recommended). Restart the instance after adding or changing secrets so they are injected.

**Option B -- `.env` file on the instance (quick tests only):**

Avoid storing real secrets in a file on the VM when you can use Brev secrets instead.

```bash
# On your dev machine
cp deploy/brev/env.template .env
# Edit .env with your values, then:
brev copy .env thorn-gateway:/home/ubuntu/workspace/thorn/.env
```

### 3. Set up the agency

The gateway needs a `.thorn/` directory with `gateway.json` and at least one
agent identity.  Three ways to provide this:

**Copy from your dev machine:**

```bash
brev copy -r /path/to/.thorn thorn-gateway:/home/ubuntu/workspace/thorn/.thorn
```

**Bootstrap on the instance:**

```bash
brev shell thorn-gateway
cd /home/ubuntu/workspace/thorn
thorn serve bootstrap \
    --agent-id my-coordinator \
    --project-name my-project \
    --clone-url https://github.com/org/repo.git \
    --forge-type github \
    --native-project-id org/repo
```

For a self-hosted GitHub Enterprise instance pass
`--forge-base-url https://ghe.example.com/api/v3`.  For GitLab,
`--forge-base-url` is **required** (e.g.
`--forge-base-url https://gitlab.example.com/api/v4`).

**Clone from a state repo** (future; not yet implemented):

```bash
cd /home/ubuntu/workspace/thorn
git clone https://github.com/org/thorn-agency.git .thorn
```

### 4. Configure git identity

Thorn's structured git tools inject identity via environment variables,
but agents may also run git via `run_shell`, which inherits the OS-level
git config.  Set these for the user that runs the gateway:

```bash
git config --global user.name "thorn-bot"
git config --global user.email "thorn-bot@users.noreply.github.com"
git config --global credential.helper store
```

If using `bootstrap-instance.sh`, this step is done automatically.

### 5. Start the gateway

```bash
brev shell thorn-gateway
cd /home/ubuntu/workspace/thorn
thorn serve
```

Or run it in a detached tmux session so it survives SSH disconnects:

```bash
brev shell thorn-gateway -- bash -c "
    cd /home/ubuntu/workspace/thorn
    tmux new-session -d -s thorn 'thorn serve 2>&1 | tee /tmp/thorn-serve.log'
"
```

## Managing the instance

| Task | Command |
|------|---------|
| Connect | `brev shell thorn-gateway` |
| Attach to gateway | `brev shell thorn-gateway -- tmux attach -t thorn` |
| View logs | `brev shell thorn-gateway -- tail -f /tmp/thorn-serve.log` |
| Stop gateway | `brev shell thorn-gateway -- tmux kill-session -t thorn` |
| Stop instance | `brev stop thorn-gateway` |
| Restart instance | `brev start thorn-gateway` |
| Delete instance | `brev delete thorn-gateway` |

## Architecture notes

- **No inbound ports required.** The `thorn serve` gateway is a polling async
  loop that makes outbound HTTP requests to the event source API and the LLM.
  It does not listen on any port.
- **`thorn serve mcp`** (a separate command) *does* bind to port 8080 by
  default.  If you need it, forward with
  `brev port-forward thorn-gateway --port 8080:8080`.
- **Persistent storage.** Files under `/home/ubuntu/workspace` survive instance
  stops but are deleted when the instance is deleted.  Push important state to
  Git before deleting.

## Troubleshooting

**`brev start` fails with capacity error:**
Try a different GPU type (`--gpu`), or omit it to use the default.

**`thorn` command not found after setup:**
The setup script installs into the system Python.  Run
`pip install -e ".[github,gitlab]"` manually from `/home/ubuntu/workspace/thorn`.

**Environment variable errors on `thorn serve`:**
Confirm required **secrets** are set in the **same** environment where `thorn serve` runs (interactive shell or tmux). With Brev secrets, restart the instance after changing secrets. Check with `env | grep OPENAI`, `env | grep GITHUB_TOKEN`, or `env | grep GITLAB_TOKEN` as appropriate. If you use a `.env` file instead, it must live in the thorn repo directory (where you run `thorn serve`). Non-secret configuration (forge base URL, project ID, etc.) lives in `.thorn/gateway.json` and `.thorn/agents/<id>.json`; if those are wrong, edit the files rather than the environment.

**Instance IP changed after restart:**
Run `brev refresh` on your dev machine to update SSH config.
