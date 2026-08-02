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
from thorn.core._mcp_config import MCPServerConfig, mcp_server_config_identity
from thorn.toolhost._executor import (
    DaemonCrashedError,
    DaemonExecutorConfig,
    DaemonToolExecutor,
    DaemonUnavailableError,
    MCPServerUnavailableError,
)
from thorn.toolhost._mcp_host import MCPHost
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
    mcp_host: MCPHost | None = None,
) -> _ConnectedPair:
    """Wire a :class:`DaemonToolExecutor` to a same-loop :class:`ToolhostServer`.

    If *mcp_host* is provided, the daemon side uses it as the MCP host
    -- callers typically pass an instance whose ``_mcp_available`` is
    forced and whose ``_ensure_connected`` is patched to install fake
    sessions, so MCP tests can exercise the real wire path without
    importing the optional ``mcp`` package.
    """
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
        mcp_host=mcp_host,
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
        _server = ToolhostServer(
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


class TestSendOrdering:
    @pytest.mark.asyncio
    async def test_response_for_unknown_call_is_dropped(self, echo_pair):
        await echo_pair.executor._handle_frame(
            ToolCallResponse(call_id="never-issued", result="ignored"),
        )


# ---------------------------------------------------------------------------
# MCP-flavored requests (Phase C.1)
# ---------------------------------------------------------------------------
#
# These exercises route ``ListMCPServerToolsRequest`` and MCP-flavored
# ``ToolCallRequest`` frames through the same in-memory pair fixture used
# above.  The daemon side gets a custom ``MCPHost`` whose
# ``_ensure_connected`` is patched to install a fake session, so we
# cover the real protocol + dispatch path without importing the
# optional ``mcp`` package.

class _FakeMCPTool:
    def __init__(
        self,
        name: str,
        description: str = "",
        input_schema: dict | None = None,
    ) -> None:
        self.name = name
        self.description = description
        self.inputSchema = input_schema or {"type": "object", "properties": {}}


class _FakeListResult:
    def __init__(self, tools: list[_FakeMCPTool]) -> None:
        self.tools = tools


class _FakeBlock:
    def __init__(self, text: str) -> None:
        self.text = text
        self.type = "text"


class _FakeCallResult:
    def __init__(self, text: str) -> None:
        self.content = [_FakeBlock(text)]


class _FakeMCPSession:
    """Minimal stand-in for ``mcp.ClientSession``."""

    def __init__(
        self,
        tools: list[_FakeMCPTool] | None = None,
        call_text: str = "ok",
        call_delay_s: float = 0.0,
    ) -> None:
        self.tools = tools or []
        self.call_text = call_text
        self.call_delay_s = call_delay_s
        self.list_calls = 0
        self.call_calls: list[tuple[str, dict]] = []

    async def list_tools(self) -> _FakeListResult:
        self.list_calls += 1
        return _FakeListResult(self.tools)

    async def call_tool(self, name: str, arguments: dict) -> _FakeCallResult:
        self.call_calls.append((name, dict(arguments)))
        if self.call_delay_s:
            await asyncio.sleep(self.call_delay_s)
        return _FakeCallResult(self.call_text)


def _make_fake_mcp_host(
    sessions: dict[tuple, _FakeMCPSession],
) -> MCPHost:
    host = MCPHost()
    host._mcp_available = True

    async def _fake_ensure_connected(entry):
        if entry.session is not None:
            return
        key = mcp_server_config_identity(entry.config)
        try:
            entry.session = sessions[key]
        except KeyError as exc:
            raise AssertionError(
                f"test did not provide a fake session for config {entry.config!r}",
            ) from exc

    host._ensure_connected = _fake_ensure_connected  # type: ignore[method-assign]
    return host


class TestMCPListAndInvoke:
    @pytest.mark.asyncio
    async def test_list_mcp_server_tools_round_trip(self, tmp_path: Path):
        cfg = MCPServerConfig(name="srv", command="srv-mcp")
        fake = _FakeMCPSession(
            tools=[_FakeMCPTool("alpha"), _FakeMCPTool("beta")],
        )
        host = _make_fake_mcp_host({mcp_server_config_identity(cfg): fake})

        pair = await _connected_pair(tmp_path, mcp_host=host)
        try:
            tools = await pair.executor.list_mcp_server_tools(cfg)
            names = [t["function"]["name"] for t in tools]
            assert names == ["alpha", "beta"]
            # Subsequent call hits the daemon's per-server cache --
            # the underlying fake session is only listed once.
            tools2 = await pair.executor.list_mcp_server_tools(cfg)
            assert tools2 == tools
            assert fake.list_calls == 1
        finally:
            await pair.aclose()

    @pytest.mark.asyncio
    async def test_invoke_routes_through_mcp_when_config_set(
        self, tmp_path: Path,
    ):
        cfg = MCPServerConfig(name="srv", command="srv-mcp")
        fake = _FakeMCPSession(
            tools=[_FakeMCPTool("hello")],
            call_text="hi from fake",
        )
        host = _make_fake_mcp_host({mcp_server_config_identity(cfg): fake})

        pair = await _connected_pair(tmp_path, mcp_host=host)
        try:
            result = await pair.executor.invoke(
                ToolInvocation(
                    call_id="c1",
                    tool_name="hello",
                    arguments={"who": "world"},
                    mcp_server_config=cfg,
                ),
            )
            assert result.is_error is False
            assert result.content == "hi from fake"
            assert fake.call_calls == [("hello", {"who": "world"})]
        finally:
            await pair.aclose()

    @pytest.mark.asyncio
    async def test_invoke_mcp_call_is_cancellable(self, tmp_path: Path):
        cfg = MCPServerConfig(name="srv", command="srv-mcp")
        # call_delay_s gives us time to cancel before completion.
        fake = _FakeMCPSession(call_delay_s=10.0)
        host = _make_fake_mcp_host({mcp_server_config_identity(cfg): fake})

        pair = await _connected_pair(tmp_path, mcp_host=host)
        try:
            task = asyncio.create_task(
                pair.executor.invoke(
                    ToolInvocation(
                        call_id="c1",
                        tool_name="t",
                        arguments={},
                        mcp_server_config=cfg,
                    ),
                ),
            )
            await asyncio.sleep(0.05)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        finally:
            await pair.aclose()

    @pytest.mark.asyncio
    async def test_list_mcp_unavailable_when_daemon_lacks_feature(
        self, tmp_path: Path,
    ):
        # An ``MCPHost`` with mcp_available=False makes the daemon
        # withhold the "mcp" feature flag.  The brain must reject the
        # list request before any frame goes out.  This is also the
        # "new-brain talking to old-daemon" half of the backward-compat
        # contract: a daemon that predates the ``mcp`` feature flag
        # likewise omits the flag and triggers the same error path.
        host = MCPHost()
        host._mcp_available = False

        pair = await _connected_pair(tmp_path, mcp_host=host)
        try:
            with pytest.raises(MCPServerUnavailableError) as exc_info:
                await pair.executor.list_mcp_server_tools(
                    MCPServerConfig(name="srv", command="srv-mcp"),
                )
            assert "mcp" in str(exc_info.value).lower()
        finally:
            await pair.aclose()


class TestBackwardCompat:
    """Both directions of the new-brain/new-daemon contract.

    The executor tests above already cover the full Phase-C.1 path
    (new brain + new daemon) and the mismatched-feature error
    direction (new brain + daemon without ``mcp`` feature).  This
    class explicitly exercises the *other* direction --
    "old-brain talking to a new daemon" -- by hand-rolling a
    minimal brain that does not advertise ``mcp`` and never sends
    MCP frames.  A new daemon must still serve built-in tool calls
    for that brain unchanged.
    """

    @pytest.mark.asyncio
    async def test_old_brain_can_still_call_builtins_on_new_daemon(
        self, tmp_path: Path,
    ):
        from thorn.toolhost._protocol import (
            ToolCallRequest as RawToolCallRequest,
        )
        from thorn.toolhost._protocol import (
            read_frame,
        )

        async def echo(value: str) -> str:
            return f"echo:{value}"

        daemon_reader, brain_writer = await _stream_pair()
        brain_reader, daemon_writer = await _stream_pair()

        registry = ToolRegistry(
            [
                ToolRegistryEntry(
                    name="echo",
                    schema=_schema("echo"),
                    venue=ToolVenue.SANDBOX,
                ),
            ],
        )
        server = ToolhostServer(
            ToolhostConfig(
                socket_path=tmp_path / "ignored.sock",
                agent_id="agent-x",
            ),
            registry=registry,
            executor=InProcessToolExecutor(
                {"echo": _WrappedTool(schema=_schema("echo"), execute=echo)},
            ),
        )
        server_task = asyncio.create_task(
            server.handle_streams(daemon_reader, daemon_writer),
        )
        try:
            # 1. Hand-roll an "old brain" handshake: features=[]
            #    (no ``mcp``), thorn_version="0.1.0".  This is a
            #    pre-Phase-C.1 brain.
            await write_frame(
                brain_writer,
                Hello(
                    protocol_major=PROTOCOL_MAJOR,
                    protocol_minor=0,
                    thorn_version="0.1.0",
                    features=[],
                ),
            )
            daemon_hello = await read_frame(brain_reader)
            assert isinstance(daemon_hello, Hello)
            assert daemon_hello.protocol_major == PROTOCOL_MAJOR

            # 2. Old brain dispatches a built-in tool call with no
            #    ``mcp_server_config`` field.  The daemon must run
            #    the call exactly as it did in Phase A.
            await write_frame(
                brain_writer,
                RawToolCallRequest(
                    call_id="c1",
                    tool_name="echo",
                    arguments={"value": "hi"},
                ),
            )
            response = await read_frame(brain_reader)
            assert isinstance(response, ToolCallResponse)
            assert response.error is None
            assert response.result == "echo:hi"
        finally:
            brain_writer.close()
            try:
                await asyncio.wait_for(server_task, timeout=2.0)
            except (asyncio.TimeoutError, Exception):
                server_task.cancel()
                try:
                    await server_task
                except (asyncio.CancelledError, Exception):
                    pass

    @pytest.mark.asyncio
    async def test_minor_mismatch_is_non_fatal(self, tmp_path: Path):
        """A brain at minor=N and a daemon at minor=N+1 must still talk.

        The handshake code logs but proceeds on minor mismatch; this
        is the contract Phase C.1 relied on when bumping
        ``PROTOCOL_MINOR`` from 0 to 1 (so old brains running against
        a newer daemon still work).
        """
        from thorn.toolhost._protocol import (
            ToolCallRequest as RawToolCallRequest,
        )
        from thorn.toolhost._protocol import (
            read_frame,
        )

        async def echo(value: str) -> str:
            return f"echo:{value}"

        daemon_reader, brain_writer = await _stream_pair()
        brain_reader, daemon_writer = await _stream_pair()

        server = ToolhostServer(
            ToolhostConfig(
                socket_path=tmp_path / "ignored.sock",
                agent_id="agent-x",
            ),
            registry=ToolRegistry(
                [
                    ToolRegistryEntry(
                        name="echo",
                        schema=_schema("echo"),
                        venue=ToolVenue.SANDBOX,
                    ),
                ],
            ),
            executor=InProcessToolExecutor(
                {"echo": _WrappedTool(schema=_schema("echo"), execute=echo)},
            ),
        )
        server_task = asyncio.create_task(
            server.handle_streams(daemon_reader, daemon_writer),
        )
        try:
            await write_frame(
                brain_writer,
                Hello(
                    protocol_major=PROTOCOL_MAJOR,
                    protocol_minor=PROTOCOL_MINOR - 1,
                    thorn_version="older",
                    features=[],
                ),
            )
            daemon_hello = await read_frame(brain_reader)
            assert isinstance(daemon_hello, Hello)
            assert daemon_hello.protocol_minor == PROTOCOL_MINOR

            await write_frame(
                brain_writer,
                RawToolCallRequest(
                    call_id="c1", tool_name="echo", arguments={"value": "hi"},
                ),
            )
            response = await read_frame(brain_reader)
            assert isinstance(response, ToolCallResponse)
            assert response.error is None
            assert response.result == "echo:hi"
        finally:
            brain_writer.close()
            try:
                await asyncio.wait_for(server_task, timeout=2.0)
            except (asyncio.TimeoutError, Exception):
                server_task.cancel()
                try:
                    await server_task
                except (asyncio.CancelledError, Exception):
                    pass
