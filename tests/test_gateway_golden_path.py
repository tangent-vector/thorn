"""Golden-path gateway smoke test.

This test intentionally exercises the composed gateway path rather than
one narrow unit: bootstrapped coordinator, raw event formatting, session
inbox, scheduler, real prompt loop, in-process forge/inbox tools, and
sandboxed file/shell tools through the subprocess toolhost.
"""

from __future__ import annotations

import asyncio
import json
import re
import shlex
import shutil
import subprocess
import sys
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from thorn.core._messages import Message, ToolResultMessage, UserMessage
from thorn.core._provider import (
    FinishChunk,
    LLMProvider,
    ResponseChunk,
    TextChunk,
    ToolCallChunk,
)
from thorn.gateway import (
    ActorIdentity,
    ContextItem,
    ContextItemKind,
    EventKind,
    EventSource,
    Gateway,
    RawIncomingEvent,
    SandboxConfig,
    instantiate_services,
    load_gateway_config,
)
from thorn.gateway._bootstrap import bootstrap_coordinator
from thorn.runtime import (
    AgencyPaths,
    AgentID,
    Runtime,
    SessionAddress,
    SessionInbox,
    SessionKey,
)
from thorn.tools.forge import GitHubForgeService

pytestmark = [
    pytest.mark.skipif(
        sys.platform == "win32",
        reason="thorn-toolhost subprocess transport uses Unix sockets",
    ),
    pytest.mark.skipif(
        shutil.which("git") is None,
        reason="golden-path smoke requires the git CLI",
    ),
]


@dataclass(frozen=True)
class ReadIssueCall:
    native_project_id: str
    issue_id: int


@dataclass(frozen=True)
class CreateChangeRequestCall:
    native_project_id: str
    source_branch: str
    title: str
    target_branch: str
    description: str


@dataclass(frozen=True)
class PostCommentCall:
    native_project_id: str
    target_type: str
    target_id: int
    body: str


class FakeForgeClient:
    """In-memory forge client for the methods the smoke test uses."""

    def __init__(self) -> None:
        self.read_issue_calls: list[ReadIssueCall] = []
        self.create_change_request_calls: list[CreateChangeRequestCall] = []
        self.post_comment_calls: list[PostCommentCall] = []

    def get_issue(
        self,
        native_project_id: str,
        issue_id: int,
    ) -> dict[str, Any]:
        self.read_issue_calls.append(ReadIssueCall(native_project_id, issue_id))
        return {
            "id": issue_id,
            "title": "Fix the smoke marker",
            "state": "open",
            "labels": ["smoke"],
            "assignees": [],
            "url": f"https://github.example.local/{native_project_id}/issues/{issue_id}",
            "description": (
                "The README still says SMOKE_FAIL. Please change it to "
                "SMOKE_PASS and open a change request."
            ),
            "author": {
                "id": 1001,
                "login": "alice",
                "name": "Alice",
                "type": "User",
            },
            "created_at": "2026-05-01T12:00:00Z",
        }

    def create_change_request(
        self,
        native_project_id: str,
        source_branch: str,
        title: str,
        target_branch: str,
        description: str,
    ) -> dict[str, Any]:
        self.create_change_request_calls.append(
            CreateChangeRequestCall(
                native_project_id=native_project_id,
                source_branch=source_branch,
                title=title,
                target_branch=target_branch,
                description=description,
            )
        )
        return {
            "id": 1,
            "title": title,
            "source_branch": source_branch,
            "target_branch": target_branch,
            "url": (
                "https://github.example.local/"
                f"{native_project_id}/pull/1"
            ),
        }

    def post_comment(
        self,
        native_project_id: str,
        target_type: str,
        target_id: int,
        body: str,
    ) -> None:
        self.post_comment_calls.append(
            PostCommentCall(
                native_project_id=native_project_id,
                target_type=target_type,
                target_id=target_id,
                body=body,
            )
        )


class ScriptedGoldenPathProvider(LLMProvider):
    """Strict provider script for issue-to-change-request smoke coverage."""

    _REQUIRED_SCHEMA_NAMES = {
        "forge_read_issue",
        "run_shell",
        "edit_file",
        "forge_create_change_request",
        "forge_post_comment",
        "update_inbox_item",
    }

    def __init__(self, bare_repo: Path, done_event: asyncio.Event) -> None:
        self._bare_repo = bare_repo
        self._done_event = done_event
        self._emitted_tool_steps = 0
        self.item_id: str | None = None
        self.first_prompt: str | None = None
        self.schema_names_seen: set[str] = set()
        self.failure: AssertionError | None = None
        self.completed = False

    async def complete(
        self,
        system_prompts: list[str],
        tools: list[dict[str, Any]],
        messages: list[Message],
    ) -> AsyncIterator[ResponseChunk]:
        try:
            del system_prompts
            self._record_schema_names(tools)
            self._capture_and_check_first_prompt(messages)
            self._check_latest_tool_result(messages)

            if self._emitted_tool_steps < 10:
                self._emitted_tool_steps += 1
                name, arguments = self._tool_step(self._emitted_tool_steps)
                yield ToolCallChunk(
                    call_id=f"smoke-{self._emitted_tool_steps}",
                    name=name,
                    arguments=json.dumps(arguments),
                )
                yield FinishChunk(reason="tool_calls")
                return

            yield TextChunk(
                text="Opened change request #1 and closed the notification."
            )
            yield FinishChunk(reason="stop")
            self.completed = True
            self._done_event.set()
        except AssertionError as exc:
            self.failure = exc
            yield TextChunk(text=f"Scripted smoke provider failed: {exc}")
            yield FinishChunk(reason="stop")
            self._done_event.set()

    def _record_schema_names(self, tools: list[dict[str, Any]]) -> None:
        for schema in tools:
            function = schema.get("function", {})
            name = function.get("name")
            if isinstance(name, str):
                self.schema_names_seen.add(name)

    def _capture_and_check_first_prompt(self, messages: list[Message]) -> None:
        if self.first_prompt is not None:
            return

        user_prompt = _last_user_prompt(messages)
        self.first_prompt = user_prompt
        match = re.search(
            r"Incoming notification \(id: ([^,]+), source: github, "
            r"status: pending\):",
            user_prompt,
        )
        assert match is not None, user_prompt
        self.item_id = match.group(1)

        missing = self._REQUIRED_SCHEMA_NAMES - self.schema_names_seen
        assert not missing, f"missing tool schemas: {sorted(missing)}"
        assert "Issue #7 opened in example/thorn-smoke" in user_prompt
        assert "[external-content" in user_prompt
        assert "SMOKE_FAIL" in user_prompt
        assert "SMOKE_PASS" in user_prompt
        assert (
            f'update_inbox_item("{self.item_id}", "handled")'
            in user_prompt
        )

    def _check_latest_tool_result(self, messages: list[Message]) -> None:
        if self._emitted_tool_steps == 0:
            return

        expected_call_id = f"smoke-{self._emitted_tool_steps}"
        latest_result = next(
            (
                message for message in reversed(messages)
                if isinstance(message, ToolResultMessage)
            ),
            None,
        )
        assert latest_result is not None
        assert latest_result.call_id == expected_call_id
        assert not latest_result.is_error, latest_result.content
        assert not latest_result.content.startswith("[exit code"), (
            latest_result.content
        )
        assert not latest_result.content.startswith("[timed out"), (
            latest_result.content
        )

    def _tool_step(self, step_number: int) -> tuple[str, dict[str, Any]]:
        assert self.item_id is not None

        if step_number == 1:
            return "forge_read_issue", {
                "project": "thorn-smoke",
                "issue_id": 7,
            }
        if step_number == 2:
            return "run_shell", {
                "command": (
                    f"git clone {shlex.quote(str(self._bare_repo))} repo"
                ),
            }
        if step_number == 3:
            return "run_shell", {
                "command": "git checkout -b thorn-smoke-fix",
                "working_directory": "repo",
            }
        if step_number == 4:
            return "edit_file", {
                "path": "repo/README.md",
                "edits": [
                    {
                        "old_string": "SMOKE_FAIL",
                        "new_string": "SMOKE_PASS",
                    },
                ],
            }
        if step_number == 5:
            return "run_shell", {
                "command": "git add README.md",
                "working_directory": "repo",
            }
        if step_number == 6:
            return "run_shell", {
                "command": "git commit -m 'Fix smoke marker'",
                "working_directory": "repo",
            }
        if step_number == 7:
            return "run_shell", {
                "command": "git push origin thorn-smoke-fix",
                "working_directory": "repo",
            }
        if step_number == 8:
            return "forge_create_change_request", {
                "project": "thorn-smoke",
                "source_branch": "thorn-smoke-fix",
                "title": "Fix smoke marker",
                "description": "Updates README smoke marker.",
                "target_branch": "main",
            }
        if step_number == 9:
            return "forge_post_comment", {
                "project": "thorn-smoke",
                "target_type": "Issue",
                "target_id": 7,
                "body": "Opened change request #1 for this issue.",
            }
        if step_number == 10:
            return "update_inbox_item", {
                "item_id": self.item_id,
                "status": "handled",
                "notes": "Opened change request #1.",
            }
        raise AssertionError(f"unexpected smoke step {step_number}")


class OneShotRawEventSource(EventSource):
    """Event source that emits one raw event and waits for test completion."""

    Config = type("Config", (), {})  # type: ignore[assignment]

    def __init__(
        self,
        event: RawIncomingEvent,
        done_event: asyncio.Event,
        *,
        timeout_s: float = 30.0,
    ) -> None:
        self._event = event
        self._done_event = done_event
        self._timeout_s = timeout_s
        self._stop = asyncio.Event()
        self.error: BaseException | None = None

    @property
    def name(self) -> str:
        return "golden-path-smoke"

    async def start(
        self,
        on_event: Callable[[RawIncomingEvent], Awaitable[None]],
    ) -> None:
        try:
            await on_event(self._event)
            await asyncio.wait_for(
                self._wait_until_done_or_stopped(),
                timeout=self._timeout_s,
            )
        except BaseException as exc:
            self.error = exc
        finally:
            self._stop.set()

    async def stop(self) -> None:
        self._stop.set()

    async def _wait_until_done_or_stopped(self) -> None:
        done_task = asyncio.create_task(self._done_event.wait())
        stop_task = asyncio.create_task(self._stop.wait())
        pending: set[asyncio.Task[bool]] = set()
        try:
            _, pending = await asyncio.wait(
                {done_task, stop_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
        finally:
            for task in pending:
                task.cancel()


def run_git(
    *args: str,
    cwd: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )


def seed_bare_git_repo(tmp_path: Path) -> Path:
    bare_repo = tmp_path / "remote.git"
    seed_repo = tmp_path / "seed"

    run_git("init", "--bare", str(bare_repo))
    run_git("init", str(seed_repo))
    run_git("checkout", "-B", "main", cwd=seed_repo)
    (seed_repo / "README.md").write_text("SMOKE_FAIL\n", encoding="utf-8")
    run_git("add", "README.md", cwd=seed_repo)
    run_git(
        "-c",
        "user.name=Seed User",
        "-c",
        "user.email=seed@example.invalid",
        "commit",
        "-m",
        "Seed smoke repo",
        cwd=seed_repo,
    )
    run_git("remote", "add", "origin", str(bare_repo), cwd=seed_repo)
    run_git("push", "origin", "main", cwd=seed_repo)
    run_git(
        "--git-dir",
        str(bare_repo),
        "symbolic-ref",
        "HEAD",
        "refs/heads/main",
    )
    return bare_repo


def _last_user_prompt(messages: list[Message]) -> str:
    for message in reversed(messages):
        if isinstance(message, UserMessage):
            return message.content
    raise AssertionError("provider request did not contain a user prompt")


def _make_raw_github_issue_event() -> RawIncomingEvent:
    actor = ActorIdentity(
        service="github",
        account_id="1001",
        secondary_account_ids=("alice",),
        display_name="Alice",
    )
    return RawIncomingEvent(
        source="github",
        session_key=SessionKey("github/123/issue/7"),
        kind=EventKind.STRUCTURAL,
        primary_actor=actor,
        summary=(
            "Issue #7 opened in example/thorn-smoke: "
            "fix the smoke marker"
        ),
        items=(
            ContextItem(
                body=(
                    "The README still says SMOKE_FAIL. Please change it "
                    "to SMOKE_PASS and open a change request."
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
        external_key=(
            "github:https://github.com:thread:thread-7:"
            "updated:2026-05-01T12:00:00Z"
        ),
    )


@pytest.mark.asyncio
async def test_bootstrapped_gateway_handles_issue_to_change_request_smoke(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GITHUB_TOKEN", "fake-token-for-smoke")
    monkeypatch.setenv("GIT_AUTHOR_NAME", "Thorn Smoke")
    monkeypatch.setenv("GIT_AUTHOR_EMAIL", "thorn-smoke@example.invalid")
    monkeypatch.setenv("GIT_COMMITTER_NAME", "Thorn Smoke")
    monkeypatch.setenv("GIT_COMMITTER_EMAIL", "thorn-smoke@example.invalid")

    bare_repo = seed_bare_git_repo(tmp_path)
    agency_home = tmp_path / "agency-home"
    agency_workspace = tmp_path / "agency-workspace"
    agent_id = AgentID("smoke-coord")
    session_key = SessionKey("github/123/issue/7")

    bootstrap_coordinator(
        agency_home=agency_home,
        agency_workspace=agency_workspace,
        agent_id=str(agent_id),
        project_name="thorn-smoke",
        project_url="https://github.com/example/thorn-smoke",
        git_user_name="thorn-smoke",
        git_user_email="thorn-smoke@example.invalid",
    )

    config = load_gateway_config(agency_home)
    config.sandbox = SandboxConfig(backend="subprocess")
    config.broker = None

    fake_forge_client = FakeForgeClient()
    services = instantiate_services(config)
    patched_github_forge = False
    for service in services:
        if isinstance(service, GitHubForgeService):
            monkeypatch.setattr(
                service,
                "authenticated_client",
                lambda _account, client=fake_forge_client: client,
            )
            patched_github_forge = True
    assert patched_github_forge

    done_event = asyncio.Event()
    provider = ScriptedGoldenPathProvider(bare_repo, done_event)
    paths = AgencyPaths.for_gateway(agency_home, agency_workspace)
    runtime = Runtime(
        provider=provider,
        workspace_root=agency_workspace,
        paths=paths,
        sandbox_executor_enabled=True,
        sandbox_config=config.sandbox,
    )
    for service in services:
        runtime.register_service(service)

    event = _make_raw_github_issue_event()
    source = OneShotRawEventSource(event, done_event=done_event)
    gateway = Gateway(
        runtime=runtime,
        sources=[source],
        gateway_config=config,
        shutdown_timeout=10.0,
    )

    await asyncio.wait_for(gateway.run(), timeout=45.0)

    assert source.error is None
    assert provider.failure is None
    assert provider.completed
    assert provider._emitted_tool_steps == 10

    assert fake_forge_client.read_issue_calls == [
        ReadIssueCall("example/thorn-smoke", 7),
    ]
    assert fake_forge_client.create_change_request_calls == [
        CreateChangeRequestCall(
            native_project_id="example/thorn-smoke",
            source_branch="thorn-smoke-fix",
            title="Fix smoke marker",
            target_branch="main",
            description="Updates README smoke marker.",
        ),
    ]
    assert fake_forge_client.post_comment_calls == [
        PostCommentCall(
            native_project_id="example/thorn-smoke",
            target_type="Issue",
            target_id=7,
            body="Opened change request #1 for this issue.",
        ),
    ]

    run_git(
        "--git-dir",
        str(bare_repo),
        "show-ref",
        "--verify",
        "refs/heads/thorn-smoke-fix",
    )
    readme = run_git(
        "--git-dir",
        str(bare_repo),
        "show",
        "thorn-smoke-fix:README.md",
    ).stdout
    assert "SMOKE_PASS" in readme
    assert "SMOKE_FAIL" not in readme

    session_workspace = runtime.paths.session_workspace(agent_id, session_key)
    assert (session_workspace / "repo" / ".git").exists()
    assert not (runtime.paths.agent_workspace_mount(agent_id) / "repo").exists()

    assert runtime.sessions.agent_exists(agent_id)
    assert runtime.sessions.session_exists(agent_id, session_key)
    agent = runtime.sessions.load_agent(agent_id)
    session = runtime.sessions.load_session(agent, session_key)
    assert session.workspace_root == session_workspace
    assert session._history.nodes

    inbox = SessionInbox(
        runtime.paths.session_inbox_dir(agent_id, session_key),
        SessionAddress(agent_id, session_key),
        in_flight_index=runtime.in_flight_index,
    )
    assert inbox.prompt_pending() == []
    assert event.external_key not in runtime.in_flight_index

    assert provider.first_prompt is not None
    assert provider.item_id is not None
    assert provider.first_prompt.startswith(
        f"Incoming notification (id: {provider.item_id}, "
        "source: github, status: pending):"
    )
    assert "[external-content" in provider.first_prompt
    assert (
        f'update_inbox_item("{provider.item_id}", "handled")'
        in provider.first_prompt
    )
