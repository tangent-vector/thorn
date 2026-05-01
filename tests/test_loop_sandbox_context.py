"""Focused tests for sandbox executor context propagation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from thorn.core._executor import ToolInvocation, ToolInvocationResult, ToolVenue
from thorn.core._loop import _WrappedTool, run_agent_loop
from thorn.core._provider import FinishChunk, MockProvider, ToolCallChunk
from thorn.runtime import AgencyPaths, AgentID, Runtime, SessionKey


class RecordingSandboxExecutor:
    """Tool executor that records sandbox invocations for assertions."""

    def __init__(self) -> None:
        self.invocations: list[ToolInvocation] = []

    async def invoke(
        self,
        invocation: ToolInvocation,
        *,
        on_chunk: Any = None,
    ) -> ToolInvocationResult:
        del on_chunk
        self.invocations.append(invocation)
        return ToolInvocationResult(content="ok")

    async def cancel(self, call_id: str) -> None:
        del call_id

    async def aclose(self) -> None:
        return None


def _tool_call_response(
    call_id: str,
    name: str,
    arguments: str,
) -> list[ToolCallChunk | FinishChunk]:
    return [
        ToolCallChunk(call_id=call_id, name=name, arguments=arguments),
        FinishChunk(reason="tool_calls"),
    ]


def _text_response(text: str):
    from thorn.core._provider import TextChunk

    return [TextChunk(text=text), FinishChunk(reason="stop")]


async def test_sandbox_invocation_carries_session_context(
    tmp_path: Path,
) -> None:
    provider = MockProvider(canned_responses=[
        _tool_call_response("c1", "sandbox_echo", "{}"),
        _text_response("done"),
    ])
    paths = AgencyPaths.for_gateway(
        tmp_path / "agency-home",
        tmp_path / "agency-workspace",
    )
    runtime = Runtime(
        provider=provider,
        workspace_root=paths.workspace_root,
        paths=paths,
    )
    agent = runtime.create_agent(id=AgentID("agent-a"))
    session_key = SessionKey("github/123/issue/7")
    session_workspace = paths.session_workspace(agent.id, session_key)
    session_workspace.mkdir(parents=True)

    sandbox_executor = RecordingSandboxExecutor()
    ctx = runtime.create_context().push_scope(
        "session",
        agent=agent,
        session_key=str(session_key),
    )
    ctx.workspace_root = session_workspace
    ctx.sandbox_executor = sandbox_executor

    async def sandbox_echo() -> str:
        return "unused"

    tool = _WrappedTool(
        schema={
            "type": "function",
            "function": {
                "name": "sandbox_echo",
                "description": "Echo from the sandbox executor.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
            },
        },
        execute=sandbox_echo,
        venue=ToolVenue.SANDBOX,
    )

    result = await run_agent_loop(
        context=ctx,
        user_prompt="call sandbox_echo",
        tools=[tool],
    )

    assert result == "done"
    assert len(sandbox_executor.invocations) == 1
    assert sandbox_executor.invocations[0].per_call_context == {
        "session_key": "github/123/issue/7",
        "workspace_subdir": "github/123/issue/7",
    }
