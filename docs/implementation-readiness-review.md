# Thorn Implementation Readiness Review

Date: 2026-04-30

Update: the body below preserves the original review findings from this
checkout. Follow-up work on branch `golden-path-smoke-test` landed the first P0
tool-routing fix, implemented the second P0 smoke test, and added an initial
Ruff configuration for branch-scoped linting.

This review is based on a source-level deep dive through the CLI, runtime,
gateway, sandbox/toolhost, forge integrations, tests, and project docs in this
checkout. It is aimed at answering one practical question: what needs to be
true before Thorn is a good thing to show other people for feedback, and later
to try with initial users.

## Executive Summary

Thorn has moved well past a sketch. The implementation contains a real runtime
substrate, persistent agents and sessions, an inbox-backed gateway, event-source
pollers for GitHub/GitLab, a peer-aware trigger policy, contextual prompt
assembly, a daemon toolhost, and an OCI container sandbox with bundled broker
support. The test suite is also substantial: pytest collected 2,399 items in
this checkout, with 2,378 selected under the default marker configuration. The
suite has focused coverage for the runtime, queues, gateway routing, forge
clients, git tools, MCP/toolhost protocols, sandbox wiring, and provider retry
behavior.

The project is still not ready to share with people who expect a smooth
feedback experience. The core issue is integration readiness, not the absence of
individual pieces. Several key paths are either under-validated end to end or
contradict each other across docs, config, and execution venues. The most
important example is the gateway coordinator's tool execution path. A targeted
diagnostic confirmed that `ProjectCoordinator._collect_tools()` currently fails
`_prepare_tools()` because `PEER_TOOLS` are not in the built-in allowlist. Even
after bypassing that allowlist for inspection, the coordinator has 27
sandbox-venue tools missing from the toolhost daemon registry, including all
peer, forge, and git tools.

My recommendation is to treat the current state as strong internal dogfood
software, not yet a public-alpha candidate. It is suitable for maintainers who
can read logs, patch config, and debug toolhost/gateway behavior. Before sharing
with outside reviewers, the P0 list below should be closed or explicitly scoped
out with a known-good "golden path".

## What Looks Solid

- The runtime/inbox/scheduler substrate is thoughtfully decomposed. Durable
  queues, `SessionInbox`, in-flight external-key deduplication, startup sweep,
  provider health gating, and per-agent/per-session concurrency controls all
  have direct test coverage.
- The gateway's trust boundary is coherent. Raw forge content is converted into
  structured `RawIncomingEvent` values, peer filtering is centralized in
  `TriggerAuthorizationPolicy`, and external content is envelope-rendered before
  it reaches the agent.
- The context-gathering pipeline is a real architecture, not just prompt string
  concatenation. `docs/context-gathering.md` accurately describes layered
  discovery of AGENTS/MEMORY/MCP/skills/journal context, and the source reflects
  that shape.
- Forge and git work are no longer one-off GitLab-only code. The unified
  `ForgeHostService` / `ProjectService` abstraction, GitHub/GitLab adapters,
  and forge-neutral toolset are a meaningful base for additional sources.
- The sandbox design has a credible security posture on paper and in code:
  container backend, dropped capabilities, `no-new-privileges`, read-only root,
  resource limits, agent home/workspace/control mounts, and broker integration
  hooks.
- The tests are broad and intent-revealing. They lock down many contracts that
  are easy to regress: queue status transitions, history compaction, provider
  retry, context assembly, peer decisions, forge routing, sandbox hardening
  flags, and toolhost protocol behavior.

## Overall Quality Assessment

The codebase is in a healthy but transitional state. It shows careful
engineering in isolated subsystems, but the product-level path is carrying
multiple generations of design at once:

- Python library API vs. gateway-first product direction.
- Static `Agent.system_prompts` vs. the newer context-gathering pipeline.
- Python callable tool discovery vs. Markdown skills and sandboxed scripts.
- Historical subprocess tool execution vs. current container/broker defaults.
- Single coordinator vertical slice vs. the aspirational multi-agent agency
  architecture.

That transition is manageable, but it needs deliberate consolidation before the
project is shown broadly. Most of the risk is in mismatched assumptions between
subsystems, not in local code quality. Large files such as
`src/thorn/gateway/_config.py`, `src/thorn/tools/forge.py`,
`src/thorn/gateway/_gateway.py`, `src/thorn/core/_history.py`, and
`src/thorn/sandbox/_runtime.py` also make it harder to see the actual contracts
between pieces.

There is no configured lint/type-check command in `pyproject.toml` beyond pytest
settings. For a project with this many typed data models and security-sensitive
execution paths, adding an explicit `ruff` and static typing posture would be a
high-leverage cleanup before wider collaboration.

## P0: Blockers Before Outside Feedback

### 1. Fix the gateway coordinator tool preparation and execution path

This is the top risk I found, and the validation pass confirmed it
mechanically.

`GatewayAgent` contributes `PEER_TOOLS`, and `ProjectCoordinator` adds
`FORGE_TOOLS`, `GIT_TOOLS`, file tools, and `run_shell`
(`src/thorn/gateway/_agents.py`). `Agent._run_session_prompt()` calls
`_prepare_tools()` before entering the agent loop. In this checkout,
`_prepare_tools(GatewayAgent._collect_tools())` and
`_prepare_tools(ProjectCoordinator._collect_tools())` both raise:

```text
TypeError: 'peer_by_account' is not a registered Thorn tool.
```

The immediate cause is that `_known_builtin_tools()` includes core, journal,
inbox, git, and forge tools, but not `thorn.tools.peers.PEER_TOOLS`.

There is a second issue after that allowlist failure. `wrap_function()` defaults
unannotated tools to `ToolVenue.SANDBOX`, and tests explicitly assert that both
git and forge tools default to `SANDBOX` (`tests/test_tool_venues.py`). When a
sandbox executor is present, `run_agent_loop()` uses `build_split_router()` and
sends every non-`IN_PROCESS` tool to the sandbox executor. However,
`thorn.toolhost._server.build_default_registry()` registers only
`ALL_BUILTIN_TOOLS`, `run_shell`, and `JOURNAL_TOOLS`. It does not register
`GIT_TOOLS`, `FORGE_TOOLS`, or `PEER_TOOLS`.

The diagnostic comparison found 38 sandbox-venue tools on `ProjectCoordinator`,
with these 27 absent from the daemon registry:

```text
find_peers_by_name
forge_create_change_request
forge_create_issue
forge_get_change_request
forge_get_project_info
forge_list_change_requests
forge_list_comments
forge_list_issues
forge_mark_notification_done
forge_post_comment
forge_read_file
forge_read_issue
forge_update_issue
git_add
git_branch
git_clone
git_commit
git_diff
git_fetch
git_log
git_pull
git_push
git_status
git_worktree_add
git_worktree_remove
list_peers
peer_by_account
```

Forge and peer tools also need `ExecutionContext.runtime`; the toolhost context
does not set `runtime`. Git auth is degraded without runtime-backed project and
account lookup. This needs a clear design decision:

- Keep runtime/forge/peer tools brain-side by marking them `IN_PROCESS`, while
  leaving file/shell/git subprocess operations sandboxed where appropriate.
- Or extend the daemon protocol/context so the sandbox can safely invoke those
  tools with the needed service graph and credentials.
- Or split forge API calls, peer lookup, and git filesystem operations into
  separate explicit tool venues.

Do not share the gateway path for feedback until a real coordinator prompt can
prepare its tools and execute representative peer, forge, git, file, and shell
calls through the actual router used by `thorn serve`.

### 2. Create a real golden-path end-to-end smoke test

The repository has a section named "End-to-end wiring verification" in
`tests/test_gateway.py`, but those tests mainly verify that the coordinator is
resolved, has expected tools, has memory, and renders prompts. The gateway inbox
tests exercise scheduler/queue behavior, often with a patched dispatcher. The
forge and git tools are tested in isolation.

Before outside feedback, add at least one smoke path that drives the real
vertical slice through the current production-ish wiring:

1. Bootstrap an agency with a fake or local forge service.
2. Start a runtime/gateway with sandbox executor enabled.
3. Deliver a synthetic GitHub/GitLab event through the formatter and inbox.
4. Let a mock provider call representative tools (`forge_read_issue`,
   `git_clone`, `edit_file`, `git_commit`, `forge_create_change_request`,
   `forge_post_comment`).
5. Assert the forge-side effects, git-side effects, session state, and outgoing
   notification behavior.

The current tests make the pieces look individually credible. They do not yet
prove the path a new evaluator will try.

### 3. Bring README and CLI bootstrap back into alignment

The user-facing README quick start is materially stale. It documents bootstrap
flags such as `--clone-url`, `--native-project-id`, `--forge-type`, and
container-mounted `.thorn/` paths. The actual CLI now requires `--project-url`,
`--agency-home`, and `--agency-workspace`, and the bootstrap source writes a
newer project-url-based `gateway.json` with inferred forges.

This is the sort of mismatch that makes a first evaluator fail before reaching
the interesting parts of the system. Update README, `docs/startup_flow.md` if
needed, and any examples to match current commands exactly.

### 4. Make first-run preflight explicit

The gateway config schema now fills absent sandbox config with a container
backend and bundled broker default for `thorn serve`. The runtime fallback still
uses subprocess when no gateway config is involved. The code does have
`thorn sandbox build` and `thorn sandbox status`, and missing sandbox images
produce helpful errors, but the first-run path is still demanding:

- an LLM provider,
- Docker or Podman,
- the default sandbox image built or pulled,
- bundled broker image availability,
- forge tokens,
- a valid agency home/workspace split,
- and correct gateway/agent account config.

Before feedback, provide a `thorn doctor` or equivalent preflight that checks
these items and gives a single actionable report. At minimum, the README should
put `thorn sandbox build`/`thorn sandbox status` in the golden path before
`thorn serve`.

### 5. Fix peer validation with inferred forges

`GatewayConfig._validate_peers()` checks `peers[].accounts[].service` against
`self.forges`. Bootstrap intentionally writes no `forges` block for well-known
GitHub/GitLab project URLs, relying on later forge inference. That means a user
who starts from bootstrap output and then adds a peer account on `"github"` or
`"gitlab"` can hit validation failure unless they also add an explicit forge
entry.

Either validate peers against resolved/inferred forges, or make the bootstrap
write explicit forge entries when peer config is present. This is central to the
trust model, so it needs to be smooth.

### 6. Decide the source acknowledgement semantics

Both GitHub and GitLab sources intentionally mark/dismiss external notifications
after handing them to the gateway, regardless of whether the formatter delivered,
deduplicated, or dropped them. GitHub additionally drains all existing unread
notifications on startup so they are never delivered.

That may be the right production default, but it is risky for early users:
events can disappear from the external platform before the agent has actually
done useful work, and GitHub downtime/backlog behavior is especially surprising.

For initial users, consider a conservative mode:

- no startup drain by default, or a prompted/explicit drain flag;
- mark external items read/done only after session handling reaches a terminal
  state, or record an internal durable audit of every dismissed event;
- surface dropped/deduped events in a status command.

### 7. Close or clearly label the egress allowlist gap

`SandboxConfig.egress_allowlist` is parsed, but enforcement is explicitly not
wired. The gateway logs a warning when it is non-empty. That is acceptable for
internal dogfooding, but not for users who may read "allowlist" as a security
control.

Before sharing, either implement enforcement or rename/document it as planned
configuration that currently has no effect.

## P1: Important Before Initial Users

### Multi-project and multi-coordinator routing

Gateway routing still chooses an explicit `agent_id`, else the first persisted
agent, else a default agent. The code comments correctly call out future
project-based routing. Since bootstrap already supports multiple project
entries, this mismatch should be resolved before anyone tries to use one gateway
for multiple repos.

### Observability and operator control

The project needs a small operator surface for "what is happening right now?":

- active agents and sessions,
- pending/in-progress/errored inbox items,
- source poller health,
- sandbox executor status,
- broker status and active bindings,
- provider health breaker state,
- last handled external keys,
- and a way to inspect/clear stuck work safely.

The TODO already calls out observability; for early users this is not optional.
Without it, every failure becomes filesystem spelunking.

### Public API and documentation consolidation

`docs/library.md` still presents arbitrary `@tool` Python functions as normal
agent tools, and CLI docs mention `.agents/thorn/*.py` discovery. Current
`_prepare_tools()` rejects bare callables unless they are known Thorn built-ins
or explicitly pre-wrapped. That may be the right sandbox-era design, but the
docs need to say so.

Also narrow `thorn.__init__` exports before inviting external code. It currently
re-exports many internals and transitional APIs, which creates accidental public
surface area.

### Context and prompt-source cleanup

The code still carries static `Agent.system_prompts` and class-based agent roles,
while the newer context pipeline discovers AGENTS/MEMORY/MCP/skills. The TODO
already has a good migration plan. This does not have to be finished before
feedback, but the "current official way" should be clear to contributors and
users.

### CLI session persistence and resume

`thorn run` and `thorn chat` mint fresh `cli/<workspace>/<uuid8>` session keys.
That is fine for one-shot local use, but feedback users will expect to resume
or inspect prior work. Add `--resume`, `thorn sessions list`, or a gateway-backed
chat path before making persistent-agent behavior a selling point.

### Provider lifecycle and loop controls

Provider retry and max-token support are in better shape than the TODO implies:
`OpenAIProviderConfig.max_tokens` exists and tests cover `OPENAI_MAX_TOKENS`.
The loop still relies mainly on a fixed `max_tool_rounds` limit and does not
detect repetitive tool-call/text patterns. Add repetition detection and ensure
the provider client's lifecycle is cleanly closed during long-lived gateway use.

### Configuration model UX

The schema is powerful but large. Early users need examples more than fields.
Provide minimal, GitHub PAT, GitLab PAT, subprocess opt-out, bundled broker, and
self-hosted forge examples. Include peer examples using immutable IDs and explain
when `thorn serve resolve-peers` is not yet available.

## P2: Cleanup and Maintainability

- Split the largest modules along stable seams. Good candidates are gateway
  config loading vs. service instantiation vs. source inference; forge service
  abstractions vs. tools; gateway startup vs. routing; and history data
  structures vs. compaction.
- Add lint and format commands to `pyproject.toml`, then enforce them locally
  and in CI. The project currently has pytest config but no visible CI config or
  lint/type config.
- Introduce explicit domain types where the code still leans on strings:
  `ForgeName`, `ProjectName`, `ToolName`, `ExternalKey`, `NotificationID`,
  `PeerID`, `ServiceName`, and similar. This aligns with the repository's own
  engineering instructions and would reduce config/routing mistakes.
- Revisit boolean config fields that may grow policy nuance, especially around
  peer delivery, egress, source acknowledgement, and sandbox/broker modes.
- Make trace output include structured advisory events. `JsonLinesSink` inherits
  the base `on_advisory()` behavior, which records advisories as status text
  instead of a first-class event with `source` and `content`.
- Make file-writing tools more auditable for user-facing operation. `edit_file`
  is intentionally exact-match based, which is good, but broader transaction or
  patch-summary behavior would improve operator confidence.
- Consider making source event strings service-instance-aware instead of
  forge-type-aware. Current policy maps are keyed by event `source` strings such
  as `"github"` and `"gitlab"`, while peer actors use forge service names. The
  code comments already acknowledge this as best-effort for multiple instances.

## Suggested Sharing Plan

### Stage 0: Maintainer Dogfood

Use the current code only with maintainers who can debug it. Prefer one repo,
one coordinator, explicit config, and either a known-good subprocess mode or a
known-good sandbox image. Keep forge permissions narrow and use throwaway test
repositories.

Exit criteria:

- P0 item 1 has a fix or an explicit known-good workaround.
- README commands are correct.
- A smoke test proves the golden path.
- `thorn sandbox status` and broker status are clean on the target machine.

### Stage 1: Friendly Internal Feedback

Invite technically sympathetic reviewers to try a documented, constrained path:
one forge, one project, one coordinator, a short list of peer accounts, and
non-critical tasks.

Exit criteria:

- Preflight/doctor exists or equivalent scripted checks are documented.
- Source acknowledgement behavior is conservative or clearly explained.
- Operators can inspect pending/in-progress/errored work without reading raw
  JSON by hand.

### Stage 2: Initial Users

Only after the gateway has end-to-end tests, stable docs, basic observability,
and clear safety constraints should it be put in front of users who are not
expected to patch Thorn itself.

## Validation Performed

Commands attempted from this checkout:

- Installed missing local tooling with permission: `python3.12-venv` and
  `python3-pip` via apt, then `uv 0.11.8` in a local bootstrap virtualenv
  symlinked at `~/.local/bin/uv`.
- `uv sync --all-extras`: passed, creating `.venv`.
- `uv run pytest`: passed, `2378 passed, 21 deselected, 764 warnings in 76.87s`.
- `uv build`: passed, producing both sdist and wheel. The generated `dist/`
  artifacts were removed after the check.
- `uv run python -m compileall -q src tests`: passed.
- Targeted coordinator tool-preparation diagnostic: failed as described in P0
  item 1; this was an intentional probe outside the pytest suite.
- No project lint/type-check command is configured in `pyproject.toml`; no
  project linter was run. Adding one is a P2 recommendation above.

## Bottom Line

The implementation has a strong foundation and a passing default test suite. The
next step is not to add more surface area; it is to prove and polish the one
path a new person will actually try. Fix the coordinator tool-preparation and
sandbox/tool venue mismatch, add a real gateway golden-path smoke test, update
the README, and add preflight plus basic observability. After that, Thorn should
be in a much better position for targeted feedback.
