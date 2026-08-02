# Agency Configuration Examples for Gateway Operation

These directories are complete agency-home skeletons for common
`thorn serve` configurations. The files are plain JSON so they can be
validated by tests and used as starting points without stripping comments.

None of the examples contain literal secrets. Provider, forge, and broker
credentials are represented only by environment-variable names; the running
agency reads the literal values from the gateway process environment.

## Public GitHub PAT

`github-pat/` shows the smallest public GitHub shape:

- The compatibility-named `gateway.json` declares a workspace, an
  project URL, and a peer.
- No `forges` block is needed because `github.com` is inferred.
- No `sandbox` or `broker` block is needed because `thorn serve` defaults to
  the container sandbox plus bundled broker.
- `agents/thorn-agent/agent.json` declares a GitHub account whose PAT is read
  from `GITHUB_TOKEN`.

## Public GitLab PAT

`gitlab-pat/` is the matching public GitLab shape:

- `gitlab.com` is inferred from the project URL.
- The agent account uses a `gitlab-pat` credential read from `GITLAB_TOKEN`.
- The project path in the URL is used as the GitLab native project path.

## Lean Coordinator Calibration

For prompt/tool overhead experiments, bootstrap a coordinator with the reduced
agent-computer-interface surface:

```console
$ thorn serve bootstrap \
    --agent-id thorn-agent \
    --project-name tasknote \
    --project-url https://gitlab.com/group/tasknote \
    --agent-class LeanProjectCoordinator \
    --agency-home ./agency \
    --agency-workspace ./workspace
```

For an existing agency, edit `agents/<agent-id>/agent.json` and set
`"agent_class": "LeanProjectCoordinator"`. This is an opt-in calibration role,
not the default production coordinator. It keeps local code tools, shell,
focused-work/TODO/journal tools, and a small Forge subset for issue and merge
request handoff while omitting the broad Forge and peer lookup surfaces.

## Subprocess Opt-Out

`subprocess-opt-out/` shows the explicit escape hatch for running without the
container sandbox:

```json
{
  "sandbox": { "backend": "subprocess" }
}
```

With this backend, Thorn does not auto-start the bundled broker because there
is no sandbox container to receive broker proxy settings.

## Bundled Broker Mirrors

`bundled-broker-mirrors/` keeps the recommended container sandbox plus bundled
broker path, but pins the OneCLI and Postgres images to operator-reachable
mirrors. This is the shape to use on restricted-egress hosts where pulling the
default public images is not allowed.

## Self-Hosted GitLab With Native ID

`self-hosted-gitlab-native-id/` shows the manual shape required for a
self-hosted GitLab instance:

- `forges[]` declares the forge type because non-public hosts are not inferred.
- Agent and peer account `service` values match the explicit forge name.
- The project keeps the human URL for clone links and sets `native_id` to the
  numeric GitLab project ID for API lookup fallback.

## Peer Account IDs

Peer `id` values are Thorn-stable identifiers used for local notes and memory.
Peer `accounts[].account_id` values should use the platform-immutable account
ID when you know it, such as the numeric GitHub or GitLab user ID.

If a peer entry still uses a textual handle (`login` on GitHub, `username` on
GitLab), run `thorn serve --agency <agency-home> resolve-peers` before starting
the gateway. The command resolves those handles through the configured forge,
stores the numeric platform ID in `account_id`, and preserves the old handle as
`display_handle` metadata. Gateway startup rejects handle-only GitHub/GitLab
peer accounts because handles are mutable and are not safe authorization keys.
