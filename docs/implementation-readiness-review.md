# Thorn Implementation Readiness Review

Original review date: 2026-04-30
Reassessed: 2026-05-05

Update: this document has been revised after an extended closed-loop gateway
trial and the follow-up work that landed on `main`. The original review's
framing still holds -- Thorn's main risk is integration readiness, not missing
isolated subsystems -- but several original P0 blockers are now closed or
materially reduced.

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
support. The test suite is also substantial and now includes direct regression
coverage for the original coordinator tool-venue bug plus a golden-path gateway
smoke test.

The closed-loop trial changed the readiness picture in an important way:
Thorn has now completed real issue-to-MR-to-review loops against a test GitLab
project in both subprocess mode and the default container sandbox plus
bundled OneCLI broker mode. That means the core gateway path is no longer just
plausible from unit tests; it has worked under live forge conditions, including
review feedback, MR creation, push/merge, cross-session memory, and brokered
Git in a restricted-egress environment.

The remaining blocker is now productization of the first evaluator experience.
The project is close to being reasonable for maintainer-led friendly feedback,
but not yet ready for self-service users. The biggest risks are stale or
contradictory quick-start docs, incomplete operator visibility/recovery
surfaces, and event-source acknowledgement semantics that can still make work
appear to vanish unless the operator understands the inbox model.

My recommendation is to share aggressively only through a constrained,
maintainer-led path: one project, one coordinator, known-good provider/forge
tokens, `thorn serve preflight` passing, and a human operator watching logs.
Before broader self-service feedback, close the current P0 list below.

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

Ruff is now configured in `pyproject.toml`, and `uv run ruff --version` works in
this checkout. CI still runs only `uv run pytest`, and there is not yet a static
typing posture. For a project with this many typed data models and
security-sensitive execution paths, enforcing Ruff in CI and deciding whether
to add a type checker remain high-leverage cleanup.

## P0: Current Blockers Before Self-Service Feedback

### 1. Bring README and CLI bootstrap back into alignment

The user-facing README quick start is materially stale. It documents bootstrap
flags such as `--clone-url`, `--native-project-id`, `--forge-type`, and
container-mounted `.thorn/` paths. The actual CLI now requires `--project-url`,
`--agency-home`, and `--agency-workspace`, and the bootstrap source writes a
newer project-url-based `gateway.json` with inferred forges.

Later sections of the README do document newer realities such as Mode A
host-side `thorn serve`, `thorn serve preflight`, requeue, mirrored broker
images, and self-hosted GitLab `native_id`. The problem is that the first
quick-start path still sends evaluators down the old route before they reach
those details.

This is the highest embarrassment risk: a first evaluator can fail before
reaching the interesting parts of the system. Update the README and examples so
the supported path is explicit:

1. `uv sync --all-extras`
2. create `.env` with provider and forge token env vars
3. `uv run thorn serve bootstrap --project-url ... --agency-home ... --agency-workspace ...`
4. `uv run thorn serve --agency ... preflight`
5. `uv run thorn serve --agency ...`

Containerized gateway instructions should be secondary and clearly labelled as
optional, with the Docker-socket tradeoff called out.

### 2. Turn preflight into a first-run readiness gate

The gateway config schema now fills absent sandbox config with a container
backend and bundled broker default for `thorn serve`. The runtime fallback still
uses subprocess when no gateway config is involved. The code does have
`thorn serve preflight`, `thorn sandbox build`, `thorn sandbox status`, and
broker status/log cleanup commands, but the first-run path is still demanding:

- an LLM provider,
- Docker or Podman,
- the default sandbox image built or pulled,
- bundled broker image availability,
- forge tokens,
- a valid agency home/workspace split,
- and correct gateway/agent account config.

`thorn serve preflight` now proves sandboxed Git connectivity through the
configured sandbox/broker path without consuming live notifications. That is
the right core check, but the evaluator path still needs either a `thorn doctor`
wrapper or a documented runbook that also verifies provider config, image
availability, forge API access, source inference, and the selected agent's
account validation before live polling starts.

### 3. Add enough observability that operators are not spelunking JSON

The trial's recovery moments were manageable because a maintainer was watching
logs and knew where to look. Friendly feedback users need a small operator
surface for "what is happening right now?":

- active agents and sessions,
- pending/in-progress/errored inbox items,
- source poller health,
- sandbox executor status,
- broker status and active bindings,
- provider health breaker state,
- recent source events and external keys,
- and safe ways to requeue or clear stuck work.

Pieces exist: `thorn inbox requeue`, `thorn broker status`, `thorn broker logs`,
`thorn broker down`, provider health internals, and durable inbox files. They
are not yet one coherent status surface. Before self-service feedback, add a
minimal `thorn status` / `thorn inbox list` / runbook combination that lets an
operator distinguish provider failure, broker failure, source silence,
dedup/drop, and agent-in-progress states quickly.

### 4. Decide and document source acknowledgement and recovery semantics

The trial confirmed the original concern: a provider authentication failure
could leave the GitLab TODO marked done while Thorn parked the local inbox item
as errored, requiring an operator nudge. Follow-up work added
`thorn inbox requeue`, and GitLab/GitHub sources now skip mark-done/read if
posting to the gateway raises. That is a meaningful improvement.

The remaining semantic issue is after successful handoff: sources still mark
external notifications done/read before the agent has actually completed the
work, and GitHub still drains all existing unread notifications at startup. That
may be a defensible daemon default, but early users need either:

- a conservative mode that delays external acknowledgement until local handling
  reaches a terminal state;
- or a clearly documented contract: after handoff, recovery is through Thorn's
  durable inbox, not the forge notification list;
- plus operator commands to inspect and requeue those local items.

This should be settled before telling users they can rely on Thorn for anything
more than throwaway feedback tasks.

### 5. Keep the closed-loop path proven on current `main`

The original tool-routing blocker is closed. Built-in tools now flow through a
single catalog: runtime-dependent forge, peer, and inbox tools execute
`IN_PROCESS`, sandboxed file/shell/journal tools execute in the toolhost, and
dedicated `git_*` tools are gone from the coordinator path in favor of
`run_shell`. `tests/test_tool_venues.py` locks down the catalog/registry
contract and `tests/test_gateway_golden_path.py` drives a bootstrapped
coordinator through issue -> edit -> commit -> push -> change request -> comment
-> inbox update using the composed gateway path.

The live trial also proved default container sandbox plus bundled broker mode
against GitLab after fixes. What is still missing is a cheap release-candidate
rehearsal that starts from a fresh agency on current `main`, runs the documented
quick start, passes `thorn serve preflight`, completes one small issue/MR in
container/broker mode, exercises `thorn inbox requeue` once, and exits with
`thorn broker status` clean. That trial should happen after the README/runbook
cleanup, not before.

### 6. Close or clearly label the egress allowlist gap

`SandboxConfig.egress_allowlist` is parsed, but enforcement is explicitly not
wired. The gateway logs a warning when it is non-empty. That is acceptable for
internal dogfooding, but not for users who may read "allowlist" as a security
control.

Before sharing, either implement enforcement or rename/document it as planned
configuration that currently has no effect.

### Closed Original P0 Items

- **Coordinator tool preparation and venue routing:** fixed by the tool catalog,
  explicit tool venues, in-process runtime tools, and regression tests.
- **Golden-path end-to-end smoke test:** fixed for the subprocess toolhost path
  by `tests/test_gateway_golden_path.py`; live closed-loop evidence covers the
  container/broker path.
- **Peer validation with inferred forges:** fixed by validating peers against
  the resolved service-name set after forge synthesis.
- **Trial follow-ups 1-5:** first-class bundled broker image references,
  robust bundled broker shutdown, broker diagnostics, sandbox/broker Git
  preflight, and inbox requeue have all landed on `main`.
- **Trial follow-ups 6-10:** GitLab project-event polling, coordinator closing
  reference guidance, no-SSH sandbox guidance, self-hosted GitLab native-ID
  resolution, and broader secret redaction have also landed or materially
  improved.

## P1: Important Before Initial Users

### Multi-project and multi-coordinator routing

Gateway routing still chooses an explicit `agent_id`, else the first persisted
agent, else a default agent. The code comments correctly call out future
project-based routing. Since bootstrap already supports multiple project
entries, this mismatch should be resolved before anyone tries to use one gateway
for multiple repos.

### Observability and operator control

Promoted to P0 for self-service feedback. It can be scoped down for a
maintainer-led friendly trial if the runbook says exactly which commands and
files to inspect, but it should not remain a vague future TODO.

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
  and in CI. Ruff is configured, but CI currently runs only pytest and there is
  no type-check command.
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

This stage has succeeded for the GitLab gateway path. Keep dogfooding on one
repo, one coordinator, narrow forge permissions, and a human operator watching
logs while the remaining P0 productization work lands.

Exit criteria:

- Current README quick start matches the CLI and the recommended host-side
  gateway topology.
- `thorn serve preflight` is documented as mandatory before live polling.
- A short operator runbook covers provider failure, broker failure, source
  silence, and inbox requeue.
- `thorn broker status` is clean after shutdown on the target machine.

### Stage 1: Friendly Internal Feedback

Invite technically sympathetic reviewers to try a documented, constrained path:
one forge, one project, one coordinator, a short list of peer accounts, and
non-critical tasks.

Exit criteria:

- A fresh-agency rehearsal on current `main` follows the public quick start,
  passes preflight, completes one issue/MR loop in container/broker mode, tests
  `thorn inbox requeue`, and exits with no orphaned broker stack.
- Source acknowledgement behavior is conservative or clearly documented.
- Operators can inspect pending/in-progress/errored work through CLI/status
  commands or an explicit runbook.

### Stage 2: Initial Users

Only after the gateway has end-to-end tests, stable docs, basic observability,
and clear safety constraints should it be put in front of users who are not
expected to patch Thorn itself.

## Validation Performed

Commands attempted from this checkout:

- Original 2026-04-30 validation: `uv sync --all-extras`, `uv run pytest`,
  `uv build`, and `uv run python -m compileall -q src tests` passed in that
  checkout; the original coordinator tool-preparation diagnostic failed and
  motivated the now-fixed tool catalog work.
- Reassessment inspection confirmed current CLI help for
  `uv run thorn serve bootstrap --help`, `uv run thorn serve preflight --help`,
  and `uv run thorn inbox requeue --help`.
- Reassessment inspection confirmed `uv run ruff --version` works with the
  configured dev dependency.
- Reassessment source/test inspection found current regression coverage in
  `tests/test_tool_venues.py`, `tests/test_gateway_golden_path.py`,
  `tests/test_gateway_preflight.py`, `tests/test_gateway_bundled_broker_shutdown.py`,
  `tests/test_gitlab.py`, `tests/test_bundled_broker.py`, and sandbox runtime
  redaction tests.
- `uv run pytest tests/test_tool_venues.py tests/test_gateway_golden_path.py tests/test_gateway_preflight.py tests/test_gateway_bundled_broker_shutdown.py tests/test_gitlab.py tests/test_bundled_broker.py tests/sandbox/test_runtime.py`:
  passed, `123 passed, 8 warnings in 2.85s`.

The reassessment did not run the full pytest suite or a live gateway trial.
Those should be run before calling a release-candidate rehearsal complete.

## Bottom Line

The implementation has a strong foundation, a real live-gateway success story,
and regression coverage for the original integration blockers. The next step is
not to add more surface area; it is to make the one path a new person will try
boringly correct. Update the quick start, make preflight and recovery part of
the official runbook, add minimal status/observability, settle acknowledgement
semantics, and then run one fresh closed-loop rehearsal on current `main`.
After that, Thorn should be ready for targeted friendly feedback without
looking more fragile than the implementation actually is.
