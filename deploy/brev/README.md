# Deploying Thorn on NVIDIA Brev

Minimal scripts for running the Thorn gateway on an [NVIDIA Brev](https://brev.nvidia.com/) GPU instance.

## Prerequisites

| Requirement | Notes |
|-------------|-------|
| Linux or WSL | All commands and scripts here assume a Unix environment. On Windows, clone and work from inside WSL -- do **not** run these scripts from a Windows checkout (CRLF line endings will break them). |
| `brev` CLI | Install: `sudo bash -c "$(curl -fsSL https://raw.githubusercontent.com/brevdev/brev-cli/main/bin/install-latest.sh)"` |
| Brev account | Sign up at <https://brev.nvidia.com/> |
| Authenticated CLI | Run `brev login` once after installing. |

## Quick start (one command)

From the thorn repo root:

```bash
./deploy/brev/deploy.sh --repo https://github.com/your-org/thorn.git
```

This will:
1. Create a Brev instance named `thorn-gateway`
2. Clone the Thorn repo and run `pip install -e ".[github,gitlab]"` via the setup script
3. Copy `.env` from the repo root (if present) to the instance
4. Start `thorn serve` inside a tmux session on the instance

### Common options

```bash
# Custom instance name and GPU type
./deploy/brev/deploy.sh --name my-thorn --gpu "nebius.l40sx1.pcie"

# Copy a pre-made agency directory instead of bootstrapping
./deploy/brev/deploy.sh --agency-dir /path/to/.thorn

# Explicit .env file
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

**Option A -- `.env` file** (simplest for testing):

```bash
# Copy from your dev machine
cp deploy/brev/env.template .env
# Edit .env with your values, then:
brev copy .env thorn-gateway:/home/ubuntu/workspace/thorn/.env
```

**Option B -- Brev secrets** (persists across instance restarts):

```bash
brev secret create OPENAI_API_URL
brev secret create OPENAI_API_KEY
brev secret create OPENAI_API_MODEL_NAME
brev secret create GITHUB_TOKEN
```

Each command prompts for the value interactively.  The secrets are injected as
environment variables every time the instance starts.

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
    --token-env GITHUB_TOKEN \
    --url-env GITHUB_URL
```

**Clone from a state repo** (future; not yet implemented):

```bash
cd /home/ubuntu/workspace/thorn
git clone https://github.com/org/thorn-agency.git .thorn
```

### 4. Start the gateway

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
Make sure your `.env` is in the thorn repo directory (where you run `thorn serve`),
or that Brev secrets are configured.  Run `env | grep OPENAI` to verify.

**Instance IP changed after restart:**
Run `brev refresh` on your dev machine to update SSH config.
