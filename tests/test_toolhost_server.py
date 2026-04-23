"""Tests for :class:`thorn.toolhost._server.ToolhostServer`.

These exercise the daemon's request loop using in-memory stream pairs
so we get fast, deterministic coverage of the handshake, dispatch,
cancellation, and error paths.  The full end-to-end "real subprocess
across a real Unix socket" flow is exercised separately under the
end-to-end-tests todo.
"""

from __future__ import annotations

import asyncio
import os
import struct
from pathlib import Path

import pytest

from thorn.core._executor import (
    InProcessToolExecutor,
    ToolRegistry,
    ToolRegistryEntry,
    ToolVenue,
)
from thorn.core._loop import _WrappedTool
from thorn.toolhost._protocol import (
    LENGTH_PREFIX_FORMAT,
    PROTOCOL_MAJOR,
    PROTOCOL_MINOR,
    Heartbeat,
    Hello,
    ToolCallCancel,
    ToolCallRequest,
    ToolCallResponse,
    encode_frame,
    read_frame,
    write_frame,
)
from thorn.toolhost._server import (
    ToolhostConfig,
    ToolhostServer,
    build_default_registry,
)


def _schema(name: str) -> dict:
    return {
        "type": "function",
        "function": {"name": name, "parameters": {"type": "object"}},
    }


async def _stream_pair() -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
    rfd, wfd = os.pipe()
    loop = asyncio.get_event_loop()
    reader = asyncio.StreamReader()
    read_proto = asyncio.StreamReaderProtocol(reader)
    await loop.connect_read_pipe(lambda: read_proto, os.fdopen(rfd, "rb", 0))
    write_transport, write_proto = await loop.connect_write_pipe(
        asyncio.streams.FlowControlMixin, os.fdopen(wfd, "wb", 0)
    )
    writer = asyncio.StreamWriter(write_transport, write_proto, None, loop)
    return reader, writer


class _BrainSide:
    """Helpers for driving the daemon from the test side.

    Wraps the brain's reader/writer pair and provides convenience
    methods for the handshake and request/response cycle.
    """

    def __init__(
        self,
        brain_reader: asyncio.StreamReader,
        brain_writer: asyncio.StreamWriter,
    ) -> None:
        self.reader = brain_reader
        self.writer = brain_writer

    async def send_hello(
        self,
        *,
        major: int = PROTOCOL_MAJOR,
        minor: int = PROTOCOL_MINOR,
    ) -> None:
        await write_frame(
            self.writer,
            Hello(
                protocol_major=major,
                protocol_minor=minor,
                thorn_version="brain-test",
            ),
        )

    async def expect_hello(self) -> Hello:
        frame = await read_frame(self.reader)
        assert isinstance(frame, Hello), frame
        return frame

    async def send_request(
        self,
        call_id: str,
        tool_name: str,
        arguments: dict | None = None,
    ) -> None:
        await write_frame(
            self.writer,
            ToolCallRequest(
                call_id=call_id,
                tool_name=tool_name,
                arguments=arguments or {},
            ),
        )

    async def expect_response(self, *, timeout: float = 1.0) -> ToolCallResponse:
        frame = await asyncio.wait_for(read_frame(self.reader), timeout=timeout)
        assert isinstance(frame, ToolCallResponse), frame
        return frame


@pytest.fixture
async def brain_and_server(tmp_path: Path):
    """Spin up a server connected to an in-memory brain side."""
    daemon_reader, brain_writer = await _stream_pair()
    brain_reader, daemon_writer = await _stream_pair()

    started = asyncio.Event()
    completed_inputs: list[str] = []

    async def slow_tool(token: str) -> str:
        completed_inputs.append(token)
        started.set()
        await asyncio.sleep(10)
        return "should-not-arrive"

    async def fast_tool(value: str) -> str:
        return f"echo:{value}"

    async def boom_tool() -> str:
        raise RuntimeError("kaboom")

    registry = ToolRegistry(
        [
            ToolRegistryEntry(
                name="slow",
                schema=_schema("slow"),
                venue=ToolVenue.SANDBOX,
            ),
            ToolRegistryEntry(
                name="echo",
                schema=_schema("echo"),
                venue=ToolVenue.SANDBOX,
            ),
            ToolRegistryEntry(
                name="boom",
                schema=_schema("boom"),
                venue=ToolVenue.SANDBOX,
            ),
        ]
    )
    executor = InProcessToolExecutor(
        {
            "slow": _WrappedTool(schema=_schema("slow"), execute=slow_tool),
            "echo": _WrappedTool(schema=_schema("echo"), execute=fast_tool),
            "boom": _WrappedTool(schema=_schema("boom"), execute=boom_tool),
        }
    )
    config = ToolhostConfig(
        socket_path=tmp_path / "toolhost.sock",
        agent_id="agent-x",
        workspace_root=tmp_path / "workspace",
    )
    server = ToolhostServer(config, registry=registry, executor=executor)
    server_task = asyncio.create_task(server.handle_streams(daemon_reader, daemon_writer))

    brain = _BrainSide(brain_reader, brain_writer)
    try:
        yield brain, server, server_task, started
    finally:
        brain.writer.close()
        try:
            await asyncio.wait_for(server_task, timeout=2.0)
        except (asyncio.TimeoutError, Exception):
            server_task.cancel()
            try:
                await server_task
            except (asyncio.CancelledError, Exception):
                pass


class TestHandshake:
    @pytest.mark.asyncio
    async def test_happy_path_handshake(self, brain_and_server):
        brain, server, _task, _started = brain_and_server
        await brain.send_hello()
        hello = await brain.expect_hello()
        assert hello.protocol_major == PROTOCOL_MAJOR
        assert hello.per_agent_state["agent_id"] == "agent-x"

    @pytest.mark.asyncio
    async def test_major_mismatch_disconnects(self, tmp_path):
        daemon_reader, brain_writer = await _stream_pair()
        brain_reader, daemon_writer = await _stream_pair()
        config = ToolhostConfig(
            socket_path=tmp_path / "s.sock",
            agent_id="x",
        )
        registry, _table = build_default_registry()
        server = ToolhostServer(
            config,
            registry=registry,
            executor=InProcessToolExecutor({}),
        )
        server_task = asyncio.create_task(
            server.handle_streams(daemon_reader, daemon_writer),
        )
        brain = _BrainSide(brain_reader, brain_writer)
        await brain.send_hello(major=PROTOCOL_MAJOR + 99)

        with pytest.raises(asyncio.IncompleteReadError):
            await asyncio.wait_for(read_frame(brain.reader), timeout=1.0)
        await asyncio.wait_for(server_task, timeout=1.0)
        brain.writer.close()

    @pytest.mark.asyncio
    async def test_minor_mismatch_is_tolerated(self, brain_and_server):
        brain, _server, _task, _started = brain_and_server
        await brain.send_hello(minor=PROTOCOL_MINOR + 1)
        await brain.expect_hello()
        await brain.send_request("c1", "echo", {"value": "ok"})
        response = await brain.expect_response()
        assert response.result == "echo:ok"

    @pytest.mark.asyncio
    async def test_first_frame_must_be_hello(self, tmp_path):
        daemon_reader, brain_writer = await _stream_pair()
        brain_reader, daemon_writer = await _stream_pair()
        config = ToolhostConfig(
            socket_path=tmp_path / "s.sock",
            agent_id="x",
        )
        registry, _table = build_default_registry()
        server = ToolhostServer(
            config,
            registry=registry,
            executor=InProcessToolExecutor({}),
        )
        server_task = asyncio.create_task(
            server.handle_streams(daemon_reader, daemon_writer),
        )
        await write_frame(brain_writer, Heartbeat())
        with pytest.raises(asyncio.IncompleteReadError):
            await asyncio.wait_for(read_frame(brain_reader), timeout=1.0)
        await asyncio.wait_for(server_task, timeout=1.0)
        brain_writer.close()


class TestDispatch:
    @pytest.mark.asyncio
    async def test_happy_path_request_response(self, brain_and_server):
        brain, _server, _task, _started = brain_and_server
        await brain.send_hello()
        await brain.expect_hello()

        await brain.send_request("c1", "echo", {"value": "hi"})
        response = await brain.expect_response()
        assert response.call_id == "c1"
        assert response.result == "echo:hi"
        assert response.error is None

    @pytest.mark.asyncio
    async def test_unknown_tool_returns_error(self, brain_and_server):
        brain, _server, _task, _started = brain_and_server
        await brain.send_hello()
        await brain.expect_hello()

        await brain.send_request("c1", "no_such_tool")
        response = await brain.expect_response()
        assert response.error is not None
        assert response.error.kind == "unknown_tool"

    @pytest.mark.asyncio
    async def test_tool_exception_returns_structured_error(self, brain_and_server):
        brain, _server, _task, _started = brain_and_server
        await brain.send_hello()
        await brain.expect_hello()

        await brain.send_request("c1", "boom")
        response = await brain.expect_response()
        assert response.error is not None
        assert response.error.kind == "execution_error"
        assert "kaboom" in response.error.message

    @pytest.mark.asyncio
    async def test_concurrent_requests_resolve_independently(self, brain_and_server):
        brain, _server, _task, _started = brain_and_server
        await brain.send_hello()
        await brain.expect_hello()

        await brain.send_request("c1", "echo", {"value": "a"})
        await brain.send_request("c2", "echo", {"value": "b"})

        responses = {
            (await brain.expect_response()).call_id: None,
            (await brain.expect_response()).call_id: None,
        }
        assert set(responses.keys()) == {"c1", "c2"}

    @pytest.mark.asyncio
    async def test_heartbeat_is_echoed(self, brain_and_server):
        brain, _server, _task, _started = brain_and_server
        await brain.send_hello()
        await brain.expect_hello()

        await write_frame(brain.writer, Heartbeat())
        response = await asyncio.wait_for(read_frame(brain.reader), timeout=1.0)
        assert isinstance(response, Heartbeat)


class TestCancellation:
    @pytest.mark.asyncio
    async def test_cancel_aborts_in_flight_call(self, brain_and_server):
        brain, _server, _task, started = brain_and_server
        await brain.send_hello()
        await brain.expect_hello()

        await brain.send_request("slow1", "slow", {"token": "go"})
        await asyncio.wait_for(started.wait(), timeout=1.0)

        await write_frame(brain.writer, ToolCallCancel(call_id="slow1"))
        response = await brain.expect_response(timeout=2.0)
        assert response.error is not None
        assert response.error.kind == "cancelled"

    @pytest.mark.asyncio
    async def test_cancel_for_unknown_call_is_silent(self, brain_and_server):
        brain, _server, _task, _started = brain_and_server
        await brain.send_hello()
        await brain.expect_hello()

        await write_frame(brain.writer, ToolCallCancel(call_id="never-existed"))
        await brain.send_request("c1", "echo", {"value": "x"})
        response = await brain.expect_response()
        assert response.result == "echo:x"


class TestProtocolViolations:
    @pytest.mark.asyncio
    async def test_malformed_frame_after_handshake_closes_connection(
        self, brain_and_server,
    ):
        brain, _server, server_task, _started = brain_and_server
        await brain.send_hello()
        await brain.expect_hello()

        body = b"{not json"
        brain.writer.write(struct.pack(LENGTH_PREFIX_FORMAT, len(body)) + body)
        await brain.writer.drain()

        with pytest.raises(asyncio.IncompleteReadError):
            await asyncio.wait_for(read_frame(brain.reader), timeout=1.0)
        await asyncio.wait_for(server_task, timeout=1.0)


class TestDefaultRegistry:
    def test_registry_excludes_in_process_allowlist(self):
        registry, table = build_default_registry()
        assert "ask_user" not in registry
        assert "ask_user" not in table

    def test_registry_includes_core_file_tools(self):
        registry, _table = build_default_registry()
        for name in ["read_file", "edit_file", "list_directory", "run_shell"]:
            assert name in registry, name

    def test_every_entry_is_sandbox_venue(self):
        registry, _table = build_default_registry()
        for entry in registry.entries():
            assert entry.venue is ToolVenue.SANDBOX, entry
