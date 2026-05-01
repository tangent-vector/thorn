# Closed-Loop Gateway Trial Plan

## Status and provenance

Status: draft plan.

This document sketches a live dogfood trial where a Codex-side
operator acts as a realistic human user of a Thorn gateway.  The
operator creates GitLab issues, reviews change requests, posts
feedback, approves and merges work, watches the gateway logs, and
records cases where Thorn behaves differently from the desired
autonomous-assistant workflow.

The trial is intentionally closed loop:

- Thorn's gateway runs against a real GitLab test project.
- Thorn's project coordinator agent performs normal project work via
  issue and merge-request notifications.
- The Codex-side operator does not commit test-app code directly.
- The Codex-side operator may patch Thorn itself only when an
  experiment-blocking Thorn bug prevents the trial from continuing.

The plan is grounded in the current implementation:

- `thorn serve` loads `gateway.json`, infers event sources from agent
  forge accounts, and runs the gateway daemon.
- GitLab events enter through polling the GitLab TODO API, not
  webhooks.
- Agent-facing forge tools can read/update issues, comments, and
  change requests, but they do not currently approve or merge merge
  requests.  Approval and merge actions stay on the simulated-user
  side of the fence.
- Git operations are performed by the coordinator with `run_shell`
  invoking `git` directly.  The account-level `git_user_name` and
  `git_user_email` fields are persisted metadata today; they do not
  by themselves configure `git commit` in the shell environment.
- Self-hosted GitLab instances need an explicit `forges` entry in
  `gateway.json`; URL-only bootstrap inference currently covers only
  `github.com` and `gitlab.com`.

## Goals

1. Prove that a Thorn gateway can run for an extended live trial
   against a private GitLab test project.
2. Exercise the realistic loop of issue creation, implementation,
   review feedback, revision, approval, merge, and follow-on work.
3. Detect cases where the gateway, event source, scheduler, prompt
   behavior, or agent memory fail to produce the desired workflow.
4. Capture a research journal with timestamps, expected behavior,
   observed behavior, log evidence, and follow-up recommendations.
5. Make small, well-documented Thorn fixes only when required to keep
   the experiment moving.

## Non-goals

- Do not build a synthetic deterministic test in this trial.  The
  existing golden-path smoke test covers that style of validation.
- Do not use this as a reason to broaden Thorn's public library API.
  The CLI and gateway remain the priority surfaces.
- Do not let the simulated user directly implement the test
  application's issues.  The point is to exercise Thorn as the
  implementer.
- Do not over-specify the test app to the Thorn agent.  Issues should
  read like normal human-authored tickets, with acceptance criteria
  and examples but not a prewritten implementation plan.

## Run layout

Use a dedicated trial area outside the active Thorn checkout's normal
state.  If another Thorn agent is active in this repo, use a separate
git worktree for Thorn fixes.

Suggested layout:

```text
thorn-loop-run/
  agency-home/
    gateway.json
    agents/
      thorn-loop-coord/
        agent.json
        home/
          MEMORY.md
          journal/
  agency-workspace/
    agents/
      thorn-loop-coord/
        workspace/
        control/
  logs/
    gateway.log
    gateway.trace.jsonl
  research-journal.md
```

Suggested long-running command:

```console
$ uv sync --all-extras
$ uv run thorn serve \
    --agency /abs/path/thorn-loop-run/agency-home \
    -vv \
    --trace /abs/path/thorn-loop-run/logs/gateway.trace.jsonl \
    2>&1 | tee /abs/path/thorn-loop-run/logs/gateway.log
```

For unattended runs, wrap that command in a small script that:

- exports the required LLM and GitLab token environment variables,
- exports `GIT_AUTHOR_NAME`, `GIT_AUTHOR_EMAIL`,
  `GIT_COMMITTER_NAME`, and `GIT_COMMITTER_EMAIL` to stable bot
  values when running in subprocess mode,
- records the Thorn git commit SHA,
- starts `thorn serve`,
- restarts it only when the operator intentionally asks for a restart,
- appends restart reason and timestamp to the research journal.

## GitLab project setup

Create a private test repository on `gitlab-master.nvidia.com`.

Recommended accounts:

- **Thorn bot account**: Developer access to the project, with a PAT
  exposed to the gateway as `THORN_LOOP_GITLAB_TOKEN`.
- **Simulated user account**: Maintainer access, with authority to
  create issues, review, approve, and merge.

Recommended project settings:

- Protect `main`.
- Require merge requests for changes.
- Allow the Thorn bot to push branches.
- Start with simple merge requirements.  One required approval from a
  non-author is useful if GitLab configuration makes that easy.
- Avoid complex CODEOWNERS, mandatory pipelines, or squash-only merge
  rules for the first pass unless those are deliberate test variables.

Preflight checks before starting the gateway:

1. Thorn bot token can authenticate to GitLab.
2. Thorn bot can clone the repository over HTTPS from the same
   execution mode chosen for the first trial.
3. Thorn bot can make a trivial commit with the intended bot author
   identity.
4. Thorn bot can push a branch.
5. Thorn bot can open an MR.
6. Simulated user can comment, approve, and merge that MR.
7. Mentioning or assigning the Thorn bot creates a pending GitLab TODO
   visible to the bot token.

For subprocess mode, these checks must use host-side git credentials.
The Thorn credential broker is not active in subprocess mode, so the
agent's `run_shell("git clone ...")` path uses whatever the host git
configuration and exported environment provide.  Do not put PATs in
`gateway.json`, `MEMORY.md`, issue text, branch names, or clone URLs
that the agent can later echo into logs or commits.

## CLI for simulated user actions

Preferred tool: `glab`.

Current local observation: `glab` was not on `PATH` when this plan was
drafted.  Treat it as an environment dependency, not an assumption.

Useful setup shape:

```console
$ glab auth login --hostname gitlab-master.nvidia.com
$ glab auth status --hostname gitlab-master.nvidia.com
```

Expected simulated-user operations:

```console
$ glab issue create ...
$ glab issue note <iid> --message "..."
$ glab mr list ...
$ glab mr view <iid> ...
$ glab mr note <iid> --message "..."
$ glab mr approve <iid>
$ glab mr merge <iid>
```

If `glab` is unavailable or unreliable against
`gitlab-master.nvidia.com`, use a small `python-gitlab` wrapper script
for the simulated-user side.  Keep that wrapper outside Thorn's agent
tool surface; it is the human/operator harness, not a Thorn feature.

The fallback wrapper should support only the experiment operations:

- create issue,
- post issue note,
- list open MRs,
- read MR details and notes,
- post MR note,
- approve MR,
- merge MR,
- list bot TODOs for preflight and diagnosis.

## Gateway configuration sketch

For self-hosted GitLab, write explicit config.  The GitLab forge URL
should be the human-facing instance URL; `python-gitlab` appends
`/api/v4` internally.

Example `gateway.json`:

```jsonc
{
  "workspace": "/abs/path/thorn-loop-run/agency-workspace",
  "forges": [
    {
      "name": "gitlab-master",
      "type": "gitlab",
      "url": "https://gitlab-master.nvidia.com",
      "poll_interval": 10
    }
  ],
  "projects": [
    {
      "name": "thorn-loop-app",
      "url": "https://gitlab-master.nvidia.com/<group>/thorn-loop-app"
    }
  ],
  "peers": [
    {
      "id": "sim-user",
      "name": "Simulated User",
      "accounts": [
        {
          "service": "gitlab-master",
          "account_id": "<simulated-user-gitlab-id-or-username>"
        }
      ]
    }
  ],
  "sandbox": {
    "backend": "subprocess"
  }
}
```

Start with the subprocess sandbox while validating GitLab behavior.
In this mode, configure host git credentials for the Thorn bot and
export the git author/committer environment variables in the wrapper
that starts `thorn serve`.  After the basic loop is stable, repeat the
trial with the default container sandbox plus bundled broker, because
credential and network behavior are materially different in that mode.
For container mode, gateway-side forge API calls and the GitLab TODO
source still read the PAT from the gateway process environment, while
the broker registers that same PAT for container-side HTTPS traffic
such as `git clone` and `git push`.  Non-secret commit-author variables
must still be passed into the container through `sandbox.env_passthrough`
or the agent's `sandbox.extra_env`.

Example `agent.json`:

```jsonc
{
  "name": "thorn-loop-coord",
  "agent_class": "ProjectCoordinator",
  "metadata": {
    "project": "thorn-loop-app"
  },
  "accounts": [
    {
      "service": "gitlab-master",
      "credentials": [
        {
          "kind": "gitlab-pat",
          "env_var_name": "THORN_LOOP_GITLAB_TOKEN"
        }
      ],
      "git_user_name": "Thorn Loop Bot",
      "git_user_email": "thorn-loop-bot@example.invalid"
    }
  ]
}
```

Example `MEMORY.md` seed:

```markdown
# thorn-loop-app Coordinator Memory

- Project service: `thorn-loop-app`
- Project URL: https://gitlab-master.nvidia.com/<group>/thorn-loop-app
- GitLab forge service: `gitlab-master`
- Primary branch: `main`

## Active work

No active issues or change requests yet.
```

## Test application sketch

Use a small Python CLI application with real tests and enough
incremental surface area to produce dependencies between issues.

Working name: `tasknote`.

The app is a local JSON-backed task/note tracker.  It should be simple
enough that failures are attributable to Thorn workflow behavior, not
domain complexity.

Seed the repository with only:

- `README.md`,
- `pyproject.toml`,
- a minimal package directory,
- a minimal `tests/` directory,
- `AGENTS.md` instructing contributors to use `uv run pytest`.

Do not prebuild the requested features.  Let Thorn implement them from
issues.

Suggested issue sequence:

1. **Basic CLI and storage**
   Add commands to create and list tasks.  Store data in a JSON file.
   Include tests for command behavior and persistence.
2. **Complete and reopen**
   Add commands to complete and reopen tasks.  Add status filtering to
   the list command.  Depends on issue 1.
3. **Tags**
   Add tags to tasks and tag-based filtering.  Depends on issue 1.
4. **Import and export**
   Add JSON import/export commands that preserve task status and tags.
   Depends on issues 2 and 3.
5. **Error handling and help text**
   Improve errors for missing files, invalid IDs, malformed JSON, and
   common CLI misuse.  Depends on the earlier user-facing commands.

Each issue should include examples and acceptance criteria but avoid
dictating module structure.

## Operating loop

For each issue:

1. Simulated user posts the issue and either assigns the Thorn bot or
   mentions it in the issue body/comment.
2. Wait at least two GitLab poll intervals.
3. Check gateway log and trace for TODO polling, event formatting,
   session dispatch, tool calls, and prompt completion.
4. Check GitLab for bot response, branch push, and MR creation.
5. Review the MR like a human maintainer:
   - inspect the diff,
   - run the tests locally when practical,
   - post targeted review feedback if needed,
   - mention the bot in feedback that requires action,
   - approve and merge only once acceptable.
6. After merge, observe whether Thorn reacts to unblocked work.
7. If Thorn does not react, check whether GitLab produced a TODO for
   the bot.  If no TODO exists, record that as an event-source or
   platform-coverage limitation rather than immediately treating it as
   an agent failure.

Do not hurry the trial by manually posting the next issue until the
research question for the current transition is answered.  The point
is to learn whether Thorn wakes, routes, and remembers correctly.

## Research journal format

Append one entry per meaningful event or decision:

```markdown
## 2026-05-01T14:30:00-07:00 - Issue #2 after !3 merge

Expected:
- After !3 merged, Thorn should notice that issue #2 is unblocked or
  otherwise continue the planned dependency chain without a manual
  nudge.

Observed:
- !3 merged at ...
- Gateway log lines:
  - ...
- GitLab TODO state for bot:
  - ...
- Thorn session/inbox state:
  - ...

Action:
- Waited two poll intervals.
- Did not post a manual nudge.

Classification:
- Thorn bug / GitLab TODO limitation / prompt behavior / harness issue.

Follow-up:
- ...
```

Track at least:

- time from user issue/comment to gateway observation,
- time from observation to first Thorn response,
- time from issue to MR,
- number of review cycles,
- number of manual nudges,
- gateway restarts,
- Thorn fixes made during the trial,
- duplicate work attempts,
- idle-after-merge incidents.

## Failure modes and mitigations

### `glab` is unavailable

Mitigation: install/configure `glab` before the trial, or use a small
`python-gitlab` wrapper for simulated-user actions.  Keep this wrapper
outside the Thorn agent's available tools.

### Self-hosted GitLab bootstrap gap

Current bootstrap cannot infer self-hosted GitLab from a project URL.
Mitigation: hand-author `gateway.json` and `agent.json` for the first
trial.  If repeatability becomes painful, add a small bootstrap
enhancement in Thorn with tests.

### GitLab TODO coverage is incomplete

GitLab may not create TODOs for every event the experiment cares about,
especially merges or indirect dependency changes.

Mitigation:

- Preflight which actions create TODOs for the bot.
- Record no-TODO cases precisely.
- Consider a later Thorn event-source enhancement only after proving
  the platform signal is absent.

### Peer matching is wrong

If the simulated user is not configured as a peer, conversational
comments may be dropped.  Structural events may still arrive with a
non-peer banner, causing the agent to read but not act.

Mitigation:

- Use the simulated user's immutable GitLab ID when possible.
- Username matching can work, but pinning the immutable ID avoids
  future username changes and reduces ambiguity.
- Confirm in logs whether events are delivered, delivered with a
  banner, or dropped.

### Bot self-notifications or bot identity mismatch

If GitLab creates TODOs from the bot's own activity, the gateway may
route unnecessary events.  If GitLab marks the bot actor differently
than expected, the trigger policy may drop bot-authored events.

Mitigation:

- Include TODO inspection in preflight.
- Record the actor fields seen in logs.
- Avoid relying on bot-authored comments as instructions.

### Sandbox or broker blocks git/network work

Subprocess mode and container-plus-broker mode have different failure
surfaces.

Mitigation:

- First validate the GitLab workflow in subprocess mode.
- In subprocess mode, verify host git credential helper behavior and
  git author/committer environment before starting the gateway.
- Then rerun with default container sandboxing and bundled broker.
- For container mode, prebuild/pull required images and confirm the
  bot can clone/push through the broker, and that non-secret git
  author/committer variables reach the container.

### Agent forgets cross-session context

Issue sessions and change-request sessions have separate workspaces
and histories.  Today the coordinator relies on `MEMORY.md` and
journal entries for cross-session continuity.

Mitigation:

- Seed `MEMORY.md` with project basics only.
- Let the trial reveal whether the agent keeps active-work state well.
- Record every case where the agent loses branch, MR, issue, or
  dependency context.

### Silent idle after merge

This is a known motivating failure mode in the coordination design
notes.  A merge event may land in a change-request session, while the
next decision logically belongs to an issue or project-level agenda.

Mitigation:

- Treat this as a primary research target.
- Avoid manual nudges until logs and TODO state have been captured.
- If no platform event exists, note the missing signal separately
  from the agent's response to signals it did receive.

### Duplicate downstream work

Two sessions may independently infer the same issue is unblocked and
start similar work.

Mitigation:

- Create dependencies that can expose this behavior.
- Watch branches, MRs, and session keys for duplicate attempts.
- Record whether duplication came from platform events, memory gaps,
  or missing cross-session coordination.

### LLM/provider instability

Provider outages or long latency can make gateway behavior look like
workflow failure.

Mitigation:

- Record provider errors from gateway logs.
- Use the gateway health monitor logs as evidence.
- Do not classify timeout-induced behavior as coordination failure
  without log support.

## Thorn bugfix policy during the trial

When a Thorn bug blocks progress:

1. Create or switch to a dedicated Thorn fix branch or worktree.
2. Record the failing reproduction in the research journal.
3. Make the smallest fix needed to continue.
4. Add focused tests when feasible.
5. Run `uv run pytest` or a narrower `uv run pytest <test-file>` when
   the full suite is impractical.
6. Commit the fix with a message that includes:
   - observed failure,
   - reproduction context,
   - reason for the fix,
   - tests run.
7. Restart the gateway and record the before/after result.

Do not mix test-app GitLab activity and Thorn fix commits in the same
git history.  The trial's application work happens through GitLab and
the Thorn agent; Thorn framework fixes happen in the Thorn checkout.

## Suggested phases

### Phase 0: Tooling and access

- Install/configure `glab` or prepare the fallback `python-gitlab`
  wrapper.
- Verify both GitLab identities and permissions.
- Verify the bot PAT can see pending TODOs.
- Verify the bot can clone, commit, push, and open an MR from the
  chosen execution mode.
- Verify `uv sync --all-extras` and the chosen `thorn serve` command.

Exit criteria: simulated user can create, review, approve, and merge a
throwaway MR; bot can receive at least one TODO; bot git authentication
and commit identity are known-good for the first trial mode.

### Phase 1: Gateway bring-up

- Create agency home/workspace.
- Write `gateway.json`, `agent.json`, and `MEMORY.md`.
- Start gateway in subprocess sandbox mode.
- Trigger a simple issue mention.
- Confirm logs show GitLab source authentication and polling.

Exit criteria: gateway receives and dispatches one issue event.

### Phase 2: First implementation loop

- Post issue 1.
- Let Thorn implement, push, and open an MR.
- Review and request one small change.
- Confirm Thorn handles feedback on the MR and pushes to the same
  branch.
- Approve and merge.

Exit criteria: one issue completes through MR review and merge with no
manual code edits by the simulated user.

### Phase 3: Dependency chain

- Post issues 2 through 5 with explicit dependencies.
- Merge each MR as it becomes acceptable.
- Observe whether Thorn notices unblocked work without extra nudges.

Exit criteria: at least one dependency transition is observed and
classified with log evidence.

### Phase 4: Container and broker repeat

- Remove the subprocess sandbox override.
- Start the gateway with default container sandbox plus bundled broker.
- Repeat one issue and one review-feedback cycle.

Exit criteria: the same GitLab loop works in the deployment mode we
expect to recommend.

### Phase 5: Synthesis

- Summarize findings into categories:
  - event-source coverage,
  - trigger/peer policy,
  - agent prompt behavior,
  - cross-session memory,
  - sandbox/broker operational issues,
  - GitLab permission and UX issues.
- Convert concrete Thorn issues into follow-up tickets or plan docs.

## Open questions

- Which exact GitLab group/project namespace should host the test repo?
- Which account should be the Thorn bot on `gitlab-master.nvidia.com`?
- Can the simulated user token approve MRs authored by the bot under
  project approval rules?
- Does GitLab create TODOs for the bot on MR merge, issue close, and
  dependency-related comments, or only on direct mention/assignment?
- Should the first trial intentionally use subprocess sandboxing, or
  should it start directly with default container/broker mode despite
  the larger setup surface?
- Do we want the harness to poll GitLab and gateway logs passively, or
  should it also enforce timeouts and write structured JSON events for
  later analysis?
