"""Tests for :class:`thorn.toolhost.DaemonToolExecutor`.

These exercises use an in-memory stream pair connected to a real
:class:`ToolhostServer` running in the same event loop.  This gives us
fast coverage of multiplexing, cancellation, chunk forwarding, and
crash detection without the cost (or flakiness) of forking a real
subprocess.  The full subprocess flow is exercised under the
``end-to-end-tests`` todo.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest

from thorn.core._executor import (
    InProcessToolExecutor,
    ToolInvocation,
    ToolOutputChunk,
    ToolRegistry,
    ToolRegistryEntry,
    ToolVenue,
)
from thorn.core._loop import _WrappedTool
from thorn.toolhost._executor import (
    DaemonCrashedError,
    DaemonExecutorConfig,
    DaemonToolExecutor,
    DaemonUnavailableError,
)
from thorn.toolhost._protocol import (
    PROTOCOL_MAJOR,
    PROTOCOL_MINOR,
    Hello,
    ToolCallChunk,
    ToolCallResponse,
    write_frame,
)
from thorn.toolhost._server import ToolhostConfig, ToolhostServer


def _schema(name: str) -> dict:
    return {
        "type": "function",
        "function": {"name": name, "parameters": {"type": "object"}},
    }


async def _stream_pair() -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
    rfd, wfd = os.pipe()
    loop = asyncio.get_event_loop()
    reader = asyncio.StreamReader()
    proto = asyncio.StreamReaderProtocol(reader)
    await loop.connect_read_pipe(lambda: proto, os.fdopen(rfd, "rb", 0))
    write_transport, write_proto = await loop.connect_write_pipe(
        asyncio.streams.FlowControlMixin, os.fdopen(wfd, "wb", 0)
    )
    writer = asyncio.StreamWriter(write_transport, write_proto, None, loop)
    return reader, writer


class _ConnectedPair:
    """Convenience wiring: a daemon and its brain talking via in-memory pipes."""

    def __init__(
        self,
        executor: DaemonToolExecutor,
        server: ToolhostServer,
        server_task: asyncio.Task,
    ) -> None:
        self.executor = executor
        self.server = server
        self.server_task = server_task

    async def aclose(self) -> None:
        await self.executor.aclose()
        try:
            await asyncio.wait_for(self.server_task, timeout=2.0)
        except (asyncio.TimeoutError, Exception):
            self.server_task.cancel()
            try:
                await self.server_task
            except (asyncio.CancelledError, Exception):
                pass


async def _connected_pair(
    tmp_path: Path,
    *,
    tools: dict[str, _WrappedTool] | None = None,
    extra_entries: list[ToolRegistryEntry] | None = None,
    config_overrides: dict | None = None,
) -> _ConnectedPair:
    """Wire a :class:`DaemonToolExecutor` to a same-loop :class:`ToolhostServer`."""
    daemon_reader, brain_writer = await _stream_pair()
    brain_reader, daemon_writer = await _stream_pair()

    tools = tools or {}
    entries = list(extra_entries or [])
    if not entries:
        entries = [
            ToolRegistryEntry(
                name=name,
                schema=tool.schema,
                venue=ToolVenue.SANDBOX,
            )
            for name, tool in tools.items()
        ]
    registry = ToolRegistry(entries)
    server = ToolhostServer(
        ToolhostConfig(
            socket_path=tmp_path / "ignored.sock",
            agent_id="agent-x",
        ),
        registry=registry,
        executor=InProcessToolExecutor(tools),
    )
    server_task = asyncio.create_task(
        server.handle_streams(daemon_reader, daemon_writer),
    )

    base_overrides = {
        "heartbeat_interval_s": 0.05,
        "heartbeat_dead_s": 1.0,
        "handshake_timeout_s": 1.0,
    }
    base_overrides.update(config_overrides or {})
    config = DaemonExecutorConfig(
        socket_path=tmp_path / "ignored.sock",
        agent_id="agent-x",
        **base_overrides,
    )
    executor = DaemonToolExecutor(config)
    await executor.adopt_streams(brain_reader, brain_writer)
    return _ConnectedPair(executor, server, server_task)


@pytest.fixture
async def echo_pair(tmp_path: Path):
    async def echo(value: str) -> str:
        return f"echo:{value}"

    pair = await _connected_pair(
        tmp_path,
        tools={"echo": _WrappedTool(schema=_schema("echo"), execute=echo)},
    )
    try:
        yield pair
    finally:
        await pair.aclose()


class TestHandshake:
    @pytest.mark.asyncio
    async def test_records_daemon_hello(self, echo_pair):
        hello = echo_pair.executor.daemon_hello
        assert hello is not None
        assert hello.protocol_major == PROTOCOL_MAJOR
        assert hello.per_agent_state["agent_id"] == "agent-x"

    @pytest.mark.asyncio
    async def test_major_mismatch_raises(self, tmp_path: Path):
        brain_reader, daemon_writer = await _stream_pair()
        daemon_reader, brain_writer = await _stream_pair()

        async def fake_daemon():
            try:
                from thorn.toolhost._protocol import read_frame
                await read_frame(daemon_reader)
            except Exception:
                return
            await write_frame(
                daemon_writer,
                Hello(
                    protocol_major=PROTOCOL_MAJOR + 99,
                    protocol_minor=PROTOCOL_MINOR,
                    thorn_version="x",
                ),
            )

        fake_task = asyncio.create_task(fake_daemon())
        executor = DaemonToolExecutor(
            DaemonExecutorConfig(
                socket_path=tmp_path / "ignored.sock",
                agent_id="x",
                handshake_timeout_s=1.0,
            )
        )
        with pytest.raises(DaemonUnavailableError, match="protocol major"):
            await executor.adopt_streams(brain_reader, brain_writer)
        await fake_task
        await executor.aclose()


class TestRoundTrip:
    @pytest.mark.asyncio
    async def test_invoke_returns_result(self, echo_pair):
        result = await echo_pair.executor.invoke(
            ToolInvocation(call_id="c1", tool_name="echo", arguments={"value": "hi"}),
        )
        assert result.is_error is False
        assert result.content == "echo:hi"

    @pytest.mark.asyncio
    async def test_concurrent_calls_resolve_independently(self, tmp_path: Path):
        gate = asyncio.Event()
        proceed = asyncio.Event()

        async def synchronized(value: str) -> str:
            gate.set()
            await proceed.wait()
            return f"done:{value}"

        pair = await _connected_pair(
            tmp_path,
            tools={
                "sync": _WrappedTool(schema=_schema("sync"), execute=synchronized),
            },
        )
        try:
            t1 = asyncio.create_task(
                pair.executor.invoke(
                    ToolInvocation(call_id="a", tool_name="sync", arguments={"value": "1"}),
                )
            )
            t2 = asyncio.create_task(
                pair.executor.invoke(
                    ToolInvocation(call_id="b", tool_name="sync", arguments={"value": "2"}),
                )
            )
            await gate.wait()
            proceed.set()
            r1, r2 = await asyncio.gather(t1, t2)
            assert {r1.content, r2.content} == {"done:1", "done:2"}
        finally:
            await pair.aclose()

    @pytest.mark.asyncio
    async def test_unknown_tool_yields_error_result(self, echo_pair):
        result = await echo_pair.executor.invoke(
            ToolInvocation(call_id="c1", tool_name="missing", arguments={}),
        )
        assert result.is_error is True
        assert result.error_kind == "unknown_tool"

    @pytest.mark.asyncio
    async def test_tool_exception_yields_error_result(self, tmp_path: Path):
        async def boom() -> str:
            raise RuntimeError("oops")

        pair = await _connected_pair(
            tmp_path,
            tools={"boom": _WrappedTool(schema=_schema("boom"), execute=boom)},
        )
        try:
            result = await pair.executor.invoke(
                ToolInvocation(call_id="c1", tool_name="boom", arguments={}),
            )
            assert result.is_error is True
            assert result.error_kind == "execution_error"
            assert "oops" in result.content
        finally:
            await pair.aclose()


class TestCancellation:
    @pytest.mark.asyncio
    async def test_caller_cancellation_sends_cancel_frame(self, tmp_path: Path):
        async def slow() -> str:
            await asyncio.sleep(10)
            return "never"

        pair = await _connected_pair(
            tmp_path,
            tools={"slow": _WrappedTool(schema=_schema("slow"), execute=slow)},
        )
        try:
            task = asyncio.create_task(
                pair.executor.invoke(
                    ToolInvocation(call_id="c1", tool_name="slow", arguments={}),
                )
            )
            await asyncio.sleep(0.05)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        finally:
            await pair.aclose()

    @pytest.mark.asyncio
    async def test_explicit_cancel_does_not_raise(self, echo_pair):
        await echo_pair.executor.cancel("never-existed")


class TestChunkForwarding:
    @pytest.mark.asyncio
    async def test_chunks_forwarded_to_callback(self, tmp_path: Path):
        pair = await _connected_pair(tmp_path)
        try:
            received: list[ToolOutputChunk] = []

            async def on_chunk(chunk: ToolOutputChunk) -> None:
                received.append(chunk)

            inflight = asyncio.create_task(
                pair.executor.invoke(
                    ToolInvocation(call_id="c1", tool_name="missing", arguments={}),
                    on_chunk=on_chunk,
                )
            )
            await asyncio.sleep(0)

            await pair.executor._handle_frame(
                ToolCallChunk(call_id="c1", stream="stdout", data="hello\n")
            )
            await pair.executor._handle_frame(
                ToolCallChunk(call_id="c1", stream="stdout", data="world\n")
            )

            await inflight
            assert [c.data for c in received] == ["hello\n", "world\n"]
        finally:
            await pair.aclose()


class TestCrashRecovery:
    @pytest.mark.asyncio
    async def test_heartbeat_timeout_fails_in_flight_call(self, tmp_path: Path):
        async def slow() -> str:
            await asyncio.sleep(10)
            return "never"

        # Override the daemon-side server so it stops echoing heartbeats
        # after the handshake.  The executor's heartbeat watchdog should
        # detect the silence and fail the in-flight call.
        daemon_reader, brain_writer = await _stream_pair()
        brain_reader, daemon_writer = await _stream_pair()

        registry = ToolRegistry(
            [ToolRegistryEntry(name="slow", schema=_schema("slow"), venue=ToolVenue.SANDBOX)],
        )
        server = ToolhostServer(
            ToolhostConfig(
                socket_path=tmp_path / "ignored.sock",
                agent_id="agent-x",
            ),
            registry=registry,
            executor=InProcessToolExecutor(
                {"slow": _WrappedTool(schema=_schema("slow"), execute=slow)},
            ),
        )

        from thorn.toolhost._protocol import read_frame

        async def silent_daemon():
            await read_frame(daemon_reader)
            await write_frame(
                daemon_writer,
                Hello(
                    protocol_major=PROTOCOL_MAJOR,
                    protocol_minor=PROTOCOL_MINOR,
                    thorn_version="silent",
                ),
            )
            try:
                await read_frame(daemon_reader)
            except Exception:
                return

        server_task = asyncio.create_task(silent_daemon())

        executor = DaemonToolExecutor(
            DaemonExecutorConfig(
                socket_path=tmp_path / "ignored.sock",
                agent_id="agent-x",
                handshake_timeout_s=1.0,
                heartbeat_interval_s=0.05,
                heartbeat_dead_s=0.3,
            )
        )
        await executor.adopt_streams(brain_reader, brain_writer)

        try:
            task = asyncio.create_task(
                executor.invoke(
                    ToolInvocation(call_id="c1", tool_name="slow", arguments={}),
                )
            )
            with pytest.raises(DaemonCrashedError):
                await asyncio.wait_for(task, timeout=2.0)
        finally:
            await executor.aclose()
            try:
                await asyncio.wait_for(server_task, timeout=1.0)
            except (asyncio.TimeoutError, Exception):
                server_task.cancel()
                try:
                    await server_task
                except (asyncio.CancelledError, Exception):
                    pass

    @pytest.mark.asyncio
    async def test_invoke_after_close_raises(self, echo_pair):
        await echo_pair.executor.aclose()
        with pytest.raises(RuntimeError, match="closed"):
            await echo_pair.executor.invoke(
                ToolInvocation(call_id="c1", tool_name="echo", arguments={"value": "x"})
            )

    @pytest.mark.asyncio
    async def test_invoke_after_close_raises(self, echo_pair):
        await echo_pair.executor.aclose()
        with pytest.raises(RuntimeError, match="closed"):
            await echo_pair.executor.invoke(
                ToolInvocation(call_id="c1", tool_name="echo", arguments={"value": "x"})
            )


class TestSendOrdering:
    @pytest.mark.asyncio
    async def test_response_for_unknown_call_is_dropped(self, echo_pair):
        await echo_pair.executor._handle_frame(
            ToolCallResponse(call_id="never-issued", result="ignored"),
        )
