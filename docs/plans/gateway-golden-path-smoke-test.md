# Gateway Golden-Path End-to-End Smoke Test

## Status and Provenance

Status: implemented.

This plan describes a fast, deterministic smoke test proving that the gateway
can process a realistic issue notification through a bootstrapped coordinator,
the session inbox, the real prompt loop, representative tools, and close-out
state.

The intended first implementation target is:

- Test file: `tests/test_gateway_golden_path.py`
- Test name: `test_bootstrapped_gateway_handles_issue_to_change_request_smoke`
- Default-suite command: `uv run pytest tests/test_gateway_golden_path.py -q`

## Goal

Prove this production-shaped path works in one test:

1. Bootstrap a project coordinator into an agency home/workspace.
2. Load `gateway.json`, instantiate project/forge services, and start `Gateway.run()` with sandbox execution enabled.
3. Deliver a synthetic GitHub issue event as a `RawIncomingEvent`.
4. Let the real gateway formatter, inbox posting, scheduler, default inbox dispatcher, `Session.prompt()`, and agent loop run.
5. Drive the coordinator through representative tool calls using a deterministic provider:
   - `update_focus` to claim the inbox item for inspection
   - `forge_read_issue`
   - `run_shell` for `git clone`
   - `run_shell` for `git checkout -b`
   - `edit_file`
   - `run_shell` to validate the edit
   - `run_shell` for `git add`
   - `run_shell` for `git commit`
   - `run_shell` for `git push`
   - `forge_create_change_request`
   - `forge_post_comment`
   - `update_focus` to record closeout validation evidence
   - `complete_focused_work` to complete the inbox item
6. Assert forge effects, git effects, session/inbox state, and prompt/tool routing invariants.

This is not a unit test for any one component. Its value is that it fails when the components are plausible in isolation but broken when composed.

## Non-Goals

- No real LLM call. The provider is scripted and deterministic.
- No real GitHub or GitLab traffic. Forge calls hit a fake client or fake local service.
- No real container runtime in the default suite. Use the subprocess sandbox backend so the test runs quickly without Docker/Podman.
- No CLI `thorn serve` process. The test uses the same gateway/runtime classes in-process to keep failure reporting direct.
- No broad behavior matrix. This is one issue-to-change-request path, not a complete gateway integration suite.

## Readiness Gates After P0 #1

Before implementing the smoke test, confirm these contracts against the landed P0 #1 work.

1. `ProjectCoordinator` tools prepare successfully:

   ```python
   from thorn.core._func import _prepare_tools
   from thorn.gateway._agents import ProjectCoordinator

   _prepare_tools(ProjectCoordinator._collect_tools())
   ```

2. Tool venue decisions are explicit. The recommended v1 contract is:

   - `IN_PROCESS`: inbox tools, peer tools, forge API tools.
   - `SANDBOX`: file tools, shell tools, journal tools, and git subprocess work via `run_shell`.

   Rationale: forge and peer tools need brain-side `ExecutionContext.runtime`, service registration, peer registry, and typed agent accounts. If P0 #1 instead makes forge/peer tools sandbox-routed, the smoke setup must use process-visible fakes and the daemon must receive enough runtime/service context to execute those tools. Do not hide that with monkeypatching in the brain process, because a real subprocess daemon will not see those patches.

3. Every `SANDBOX` tool exposed to `ProjectCoordinator` is callable through the daemon registry. The smoke will naturally catch missing daemon entries by receiving a tool-result error for `run_shell`, `edit_file`, etc.

4. Sandbox tool calls run relative to the active session workspace, not the broader per-agent workspace. The protocol already has `ToolCallRequest.per_call_context["workspace_subdir"]`; verify the brain side populates it, or fix that before expecting this smoke to pass. The test should call `run_shell(command="git clone <bare_repo_path> repo")` and then assert the clone lives under:

   ```text
   <agency_workspace>/<agent_id>/<session_key>/repo
   ```

5. The subprocess sandbox backend can start during gateway startup with no broker:

   ```python
   config.sandbox = SandboxConfig(backend="subprocess")
   config.broker = None
   runtime = Runtime(..., sandbox_executor_enabled=True, sandbox_config=config.sandbox)
   ```

6. Subprocess toolhost socket paths must stay within the Unix-domain
   socket path limit. The implementation should either keep the
   control-dir socket path short or fall back to a short per-user temp
   path when the agency workspace lives under a deep directory such as
   pytest's default temp tree. This is part of the smoke's value: a
   daemon that crashes before binding its socket fails before the event
   is delivered.

## Test Architecture

### Main Actors

`OneShotRawEventSource`

- Implements `EventSource`.
- Calls the gateway's `on_event` callback once with a synthetic `RawIncomingEvent`.
- Waits for a test-owned `asyncio.Event` set by the scripted provider after the final close-out step.
- Stores any timeout/failure on the source so the test can assert it after `Gateway.run()` returns. Source task exceptions are gathered by the gateway, so do not rely on an exception raised inside `start()` to fail the test directly.

`ScriptedGoldenPathProvider`

- Implements `LLMProvider` directly rather than using `MockProvider`.
- Records every completion request's tool schemas and messages.
- Parses the inbox item id from the first user prompt:

  ```text
  Incoming notification (id: <item_id>, source: github, status: pending):
  ```

- Asserts the first user prompt contains the formatter output, including an `[external-content` envelope.
- Emits one tool call per round, then a final text response.
- Checks the previous `ToolResultMessage` before each next step and records a failure if any tool result has `is_error=True`.
- Sets the done event after the final response or after an internal assertion failure.

`FakeForgeClient`

- In-memory implementation of the `ForgeClient` methods used by this test.
- Required methods:
  - `get_issue(native_project_id, issue_id)`
  - `create_change_request(native_project_id, source_branch, title, target_branch, description)`
  - `post_comment(native_project_id, target_type, target_id, body)`
- Other protocol methods can raise `NotImplementedError`; the test should not call them.
- Records calls in typed lists or small dataclasses so assertions do not scrape strings.

`seed_bare_git_repo`

- Creates a local bare repository and pushes a `main` branch containing a `README.md` with a unique string to replace.
- Uses explicit `git -c user.name=... -c user.email=...` for the seed commit.
- The provider uses the bare repo path as the `git_clone.remote_url`, keeping the smoke offline.

### Bootstrapped Agency

Use real bootstrap output so the test exercises the same persisted coordinator shape an evaluator would create:

```python
bootstrap_coordinator(
    agency_home=agency_home,
    agency_workspace=agency_workspace,
    agent_id="smoke-coord",
    project_name="thorn-smoke",
    project_url="https://github.com/example/thorn-smoke",
    git_user_name="thorn-smoke",
    git_user_email="thorn-smoke@example.invalid",
)
```

Then load and adjust the config for a fast default-suite smoke:

```python
config = load_gateway_config(agency_home)
config.sandbox = SandboxConfig(backend="subprocess")
config.broker = None
```

Set environment variables:

- `GITHUB_TOKEN=fake-token-for-smoke`, so the bootstrapped account has a readable credential if a code path asks for it.
- `GIT_AUTHOR_NAME`, `GIT_AUTHOR_EMAIL`, `GIT_COMMITTER_NAME`, and `GIT_COMMITTER_EMAIL` to stable smoke-test values. This keeps daemon-executed `git commit` independent of host git config while preserving the separate account-identity test coverage elsewhere.

Instantiate services from the config and inject the fake forge client into the GitHub forge service. This injection is valid only if forge tools are `IN_PROCESS` after P0 #1:

```python
services = instantiate_services(config)
for service in services:
    if isinstance(service, GitHubForgeService):
        monkeypatch.setattr(
            service,
            "authenticated_client",
            lambda _account: fake_forge_client,
        )
```

Register all services on the runtime before starting the gateway, so startup account validation can type the coordinator account.

Construct the runtime and gateway with the real startup path:

```python
paths = AgencyPaths.for_gateway(agency_home, agency_workspace)
runtime = Runtime(
    provider=scripted_provider,
    workspace_root=agency_workspace,
    paths=paths,
    sandbox_executor_enabled=True,
    sandbox_config=config.sandbox,
)
for service in services:
    runtime.register_service(service)

source = OneShotRawEventSource(event, done_event=scripted_provider.done)
gateway = Gateway(
    runtime=runtime,
    sources=[source],
    gateway_config=config,
    shutdown_timeout=10.0,
)

await asyncio.wait_for(gateway.run(), timeout=30.0)
```

### Synthetic Event

Use `RawIncomingEvent`, not the legacy `IncomingEvent` alias, so the formatter and trigger policy run:

```python
actor = ActorIdentity(
    service="github",
    account_id="1001",
    secondary_account_ids=("alice",),
    display_name="Alice",
)

event = RawIncomingEvent(
    source="github",
    session_key=SessionKey("github/123/issue/7"),
    kind=EventKind.STRUCTURAL,
    primary_actor=actor,
    summary="Issue #7 opened in example/thorn-smoke: fix the smoke marker",
    items=(
        ContextItem(
            body=(
                "The README still says SMOKE_FAIL. Please change it to "
                "SMOKE_PASS and open a change request."
            ),
            kind=ContextItemKind.ISSUE_BODY,
            actor=actor,
            timestamp="2026-05-01T12:00:00Z",
        ),
    ),
    metadata={
        "notification_id": "thread-7",
        "repo_full_name": "example/thorn-smoke",
        "repo_id": 123,
        "project_name": "thorn-smoke",
        "issue_id": 7,
    },
    external_key="github:https://github.com:thread:thread-7:updated:2026-05-01T12:00:00Z",
)
```

An empty peer registry is acceptable here because this is a structural event. The formatter should deliver it with the non-peer structural treatment, and the content body should be wrapped in an external-content envelope.

## Provider Script

The provider's state machine should be linear and strict. A suggested sequence:

| Step | Tool | Arguments |
| --- | --- | --- |
| 1 | `update_focus` | `{"phase": "inspect", "item_id": "<parsed_item_id>", "objective": "Replace SMOKE_FAIL with SMOKE_PASS and open a change request for issue #7.", "notes": "Starting issue #7 smoke marker fix."}` |
| 2 | `forge_read_issue` | `{"project": "thorn-smoke", "issue_id": 7}` |
| 3 | `run_shell` | `{"command": "git clone <bare_repo_path> repo"}` |
| 4 | `run_shell` | `{"command": "git checkout -b thorn-smoke-fix", "working_directory": "repo"}` |
| 5 | `edit_file` | `{"path": "repo/README.md", "edits": [{"old_string": "SMOKE_FAIL", "new_string": "SMOKE_PASS"}]}` |
| 6 | `run_shell` | `{"command": "grep -q SMOKE_PASS README.md && ! grep -q SMOKE_FAIL README.md", "working_directory": "repo"}` |
| 7 | `run_shell` | `{"command": "git add README.md", "working_directory": "repo"}` |
| 8 | `run_shell` | `{"command": "git commit -m 'Fix smoke marker'", "working_directory": "repo"}` |
| 9 | `run_shell` | `{"command": "git push origin thorn-smoke-fix", "working_directory": "repo"}` |
| 10 | `forge_create_change_request` | `{"project": "thorn-smoke", "source_branch": "thorn-smoke-fix", "title": "Fix smoke marker", "description": "Updates README smoke marker.", "target_branch": "main"}` |
| 11 | `forge_post_comment` | `{"project": "thorn-smoke", "target_type": "Issue", "target_id": 7, "body": "Opened change request #1 for this issue."}` |
| 12 | `update_focus` | `{"phase": "closeout", "validation_outcome": "passed", "validation_summary": "Verified README.md contains SMOKE_PASS and no longer contains SMOKE_FAIL.", "validation_command": "grep -q SMOKE_PASS README.md && ! grep -q SMOKE_FAIL README.md", "notes": "Opened change request #1."}` |
| 13 | `complete_focused_work` | `{"notes": "Opened change request #1.", "completion": {"completed_actions": ["Read issue #7.", "Updated README.md so the smoke marker passes.", "Committed and pushed thorn-smoke-fix.", "Opened change request #1.", "Posted a source issue comment."], "request_coverage": ["The issue requested replacing SMOKE_FAIL with SMOKE_PASS.", "The change request contains that update."], "remaining_work": [], "self_review": "Checked the source issue, the file edit, and the change request before closing the inbox item.", "external_follow_up": ["Opened change request #1.", "Commented on issue #7."]}}` |
| 14 | final text | `"Opened change request #1 and closed the notification."` |

The first completion request should also assert that the exposed schema names include at least:

```python
{
    "forge_read_issue",
    "run_shell",
    "edit_file",
    "forge_create_change_request",
    "forge_post_comment",
    "update_focus",
    "complete_focused_work",
}
```

That schema assertion catches preparation/exposure regressions before the test gets to tool execution. The tool-result checks catch router/daemon registry/runtime regressions.

## Assertions

### Forge Effects

Assert the fake client saw:

- `get_issue("example/thorn-smoke", 7)`
- `create_change_request("example/thorn-smoke", source_branch="thorn-smoke-fix", target_branch="main", ...)`
- `post_comment("example/thorn-smoke", "Issue", 7, body containing "change request #1")`

The fake `create_change_request` should return a URL like `https://github.example.local/example/thorn-smoke/pull/1`, and the comment assertion should confirm the provider used the returned id or URL in its outbound message if the scripted flow is extended to inspect prior tool results.

### Git Effects

Assert the pushed branch exists in the bare repo:

```bash
git --git-dir <bare_repo> show-ref --verify refs/heads/thorn-smoke-fix
```

Assert the branch content changed:

```bash
git --git-dir <bare_repo> show thorn-smoke-fix:README.md
```

Expected content contains `SMOKE_PASS` and not `SMOKE_FAIL`.

Assert the working clone is under the session workspace:

```python
session_workspace = runtime.paths.session_workspace(
    AgentID("smoke-coord"),
    SessionKey("github/123/issue/7"),
)
assert (session_workspace / "repo" / ".git").exists()
assert not (runtime.paths.agent_workspace_mount(AgentID("smoke-coord")) / "repo").exists()
```

This is the deliberate guard for per-call sandbox workspace propagation.

### Inbox and Session State

Assert:

- `runtime.sessions.agent_exists(AgentID("smoke-coord"))`
- `runtime.sessions.session_exists(AgentID("smoke-coord"), SessionKey("github/123/issue/7"))`
- the loaded session has non-empty history with tool calls recorded.
- the session inbox has no `prompt_pending()` items after `complete_focused_work(...)`.
- the external key is no longer present in `runtime.in_flight_index`.
- the provider consumed every scripted step and recorded no failure.
- the one-shot source did not time out waiting for completion.

### Gateway/Formatter Evidence

Assert through the provider's recorded first prompt:

- It starts with the inbox prompt header containing the generated item id.
- It includes the synthetic summary.
- It includes an `[external-content` envelope around the issue body.
- It includes the focused-work instruction to claim the item with `update_focus(phase="inspect", item_id="<item_id>", ...)`, record validation evidence with `update_focus(phase="closeout", ...)`, and finish with `complete_focused_work(completion=...)`.

This ensures the event did not bypass formatter/inbox/default dispatcher plumbing.

## Implementation Sequence

1. Re-check the readiness gates after the P0 #1 branch lands.
2. Add `tests/test_gateway_golden_path.py` with helper classes local to that file.
3. Implement the local bare-repo fixture/helper.
4. Implement the fake forge client and GitHub service injection, or adapt to the landed P0 #1 forge venue decision.
5. Implement `ScriptedGoldenPathProvider`.
6. Implement `OneShotRawEventSource`.
7. Implement the main smoke test using `Gateway.run()`, not private `_handle_event()` calls.
8. Run the targeted smoke:

   ```bash
   uv run pytest tests/test_gateway_golden_path.py -q
   ```

9. Run the related slices:

   ```bash
   uv run pytest tests/test_tool_venues.py tests/test_toolhost_server.py tests/test_toolhost_executor.py tests/test_gateway.py -q
   ```

10. Run the full suite:

    ```bash
    uv run pytest
    ```

## Expected Failure Modes This Test Should Catch

- `ProjectCoordinator` cannot prepare its tools.
- Tool schemas omit a representative project, forge, shell, file, or inbox tool.
- A sandbox-routed tool is absent from the daemon registry.
- A brain-owned tool was incorrectly routed to the daemon and lost access to runtime/service state.
- The daemon executes relative paths in the agent workspace instead of the session workspace.
- `git commit` fails because the test accidentally depends on host git config.
- The gateway bypasses the formatter or inbox dispatcher.
- The agent loop completes without closing the inbox item.
- The in-flight external key remains after a handled no-RSVP notification.

## Deferred Follow-Ups

- Add an opt-in OCI variant under `tests/sandbox/` marked `requires_podman` and/or `requires_docker` after the default subprocess smoke is stable.
- Add a real source adapter smoke that uses `_make_raw_event()` from the GitHub or GitLab source module once fake API payload setup is worth the extra detail.
- Add RSVP/outgoing-queue coverage if `RawIncomingEvent` grows an explicit `rsvp_to` field. For the current gateway event shape, this smoke should treat the forge comment as the outbound user-visible notification and rely on existing RSVP unit/integration tests for queue return-path behavior.
