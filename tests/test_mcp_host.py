"""Tests for ``thorn.toolhost._mcp_host``.

Two layers of coverage:

1. *Conversion helpers* (``_mcp_tool_to_openai_schema``,
   ``_mcp_result_to_string``) -- moved here in Phase C.1 from
   ``tests/test_mcp.py`` when the brain-side ``MCPToolSource`` was
   retired.  The daemon's ``MCPHost`` is now the canonical caller;
   the helpers themselves are private to the module but are
   stable enough to test directly.
2. *MCPHost lifecycle* with a fake ``ClientSession`` -- exercises
   the identity-keyed cache (one process per unique config), the
   per-server call lock (concurrent calls serialise), and clean
   teardown via ``aclose``.  We avoid importing ``mcp`` by patching
   ``MCPHost._ensure_connected`` to install a fake session
   directly; that keeps the test free of the optional ``mcp``
   extra and pure-Python fast.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock

import pytest

from thorn.core._mcp_config import MCPServerConfig
from thorn.toolhost._mcp_host import (
    MCPHost,
    MCPUnavailableError,
    _mcp_result_to_string,
    _mcp_tool_to_openai_schema,
)

# ---------------------------------------------------------------------------
# Schema conversion: MCP ``Tool`` -> OpenAI-style schema
# ---------------------------------------------------------------------------

class TestMcpToolToOpenAISchema:
    def test_translates_full_tool(self):
        mcp_tool = MagicMock()
        mcp_tool.name = "search"
        mcp_tool.description = "Search for things."
        mcp_tool.inputSchema = {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        }

        schema = _mcp_tool_to_openai_schema(mcp_tool)

        assert schema["type"] == "function"
        assert schema["function"]["name"] == "search"
        assert schema["function"]["description"] == "Search for things."
        assert schema["function"]["parameters"] == mcp_tool.inputSchema

    def test_falsy_description_becomes_empty_string(self):
        # Some MCP servers omit the description; the OpenAI side
        # requires a string, never ``None``.
        mcp_tool = MagicMock()
        mcp_tool.name = "noop"
        mcp_tool.description = None
        mcp_tool.inputSchema = {"type": "object", "properties": {}}

        schema = _mcp_tool_to_openai_schema(mcp_tool)
        assert schema["function"]["description"] == ""


# ---------------------------------------------------------------------------
# Result extraction
# ---------------------------------------------------------------------------

class TestMcpResultToString:
    def test_single_text_block(self):
        result = MagicMock()
        block = MagicMock()
        block.text = "hello world"
        result.content = [block]

        assert _mcp_result_to_string(result) == "hello world"

    def test_multiple_text_blocks_join_with_newlines(self):
        result = MagicMock()
        b1, b2 = MagicMock(), MagicMock()
        b1.text = "line 1"
        b2.text = "line 2"
        result.content = [b1, b2]

        assert _mcp_result_to_string(result) == "line 1\nline 2"

    def test_non_text_blocks_skipped(self):
        result = MagicMock()
        text_block = MagicMock()
        text_block.text = "good"
        image_block = MagicMock(spec=[])  # no ``.text`` attribute
        result.content = [image_block, text_block]

        assert _mcp_result_to_string(result) == "good"

    def test_empty_content(self):
        result = MagicMock()
        result.content = []

        assert _mcp_result_to_string(result) == ""


# ---------------------------------------------------------------------------
# MCPHost lifecycle
# ---------------------------------------------------------------------------

class _FakeTool:
    """Stand-in for ``mcp.types.Tool`` with the three attrs we touch."""

    def __init__(self, name: str, description: str | None = "") -> None:
        self.name = name
        self.description = description
        self.inputSchema = {"type": "object", "properties": {}}


class _FakeListResult:
    def __init__(self, tools: list[_FakeTool]) -> None:
        self.tools = tools


class _FakeBlock:
    def __init__(self, text: str) -> None:
        self.text = text


class _FakeCallResult:
    def __init__(self, text: str) -> None:
        self.content = [_FakeBlock(text)]


class _FakeSession:
    """Minimal stand-in for ``mcp.ClientSession``.

    Tracks call counts so tests can assert "the host did not open a
    second session for the same identity".  Each call is an
    ``asyncio.Lock``-aware coroutine so we can inspect serialisation
    behaviour in :class:`TestMCPHostConcurrency`.
    """

    def __init__(
        self,
        tools: list[_FakeTool] | None = None,
        call_text: str = "ok",
    ) -> None:
        self.tools = tools or []
        self.call_text = call_text
        self.list_calls = 0
        self.call_calls: list[tuple[str, dict[str, Any]]] = []

    async def list_tools(self) -> _FakeListResult:
        self.list_calls += 1
        return _FakeListResult(self.tools)

    async def call_tool(
        self, name: str, arguments: dict[str, Any],
    ) -> _FakeCallResult:
        self.call_calls.append((name, dict(arguments)))
        return _FakeCallResult(self.call_text)


def _make_host_with_fake_sessions(
    sessions: dict[tuple, _FakeSession],
) -> MCPHost:
    """Build an ``MCPHost`` whose ``_ensure_connected`` installs a fake.

    *sessions* is keyed by the same identity tuple the host uses
    internally (see :func:`mcp_server_config_identity`).  This lets a
    single test inject distinct fake sessions for distinct configs
    while still exercising the production identity-cache logic.
    """
    from thorn.core._mcp_config import mcp_server_config_identity

    host = MCPHost()
    # Force-enable the MCP path even when the optional package is not
    # installed in the test environment; we never reach the real
    # ``mcp`` import because ``_ensure_connected`` is patched below.
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


@pytest.mark.asyncio
async def test_list_tools_caches_per_identity():
    """A second list call for the same config reuses the cached result."""
    cfg = MCPServerConfig(name="srv", command="srv-mcp")
    fake = _FakeSession(tools=[_FakeTool("a"), _FakeTool("b")])

    from thorn.core._mcp_config import mcp_server_config_identity
    host = _make_host_with_fake_sessions({mcp_server_config_identity(cfg): fake})

    first = await host.list_tools(cfg)
    second = await host.list_tools(cfg)

    assert [s["function"]["name"] for s in first] == ["a", "b"]
    assert second == first
    # Cache hit on the second call: the underlying server was only
    # asked once.
    assert fake.list_calls == 1

    await host.aclose()


@pytest.mark.asyncio
async def test_distinct_configs_get_distinct_sessions():
    """Two configs that hash differently must not share a session."""
    a = MCPServerConfig(name="srv", command="cmd-a")
    b = MCPServerConfig(name="srv", command="cmd-b")
    fake_a = _FakeSession(tools=[_FakeTool("a_tool")], call_text="from-a")
    fake_b = _FakeSession(tools=[_FakeTool("b_tool")], call_text="from-b")

    from thorn.core._mcp_config import mcp_server_config_identity
    host = _make_host_with_fake_sessions({
        mcp_server_config_identity(a): fake_a,
        mcp_server_config_identity(b): fake_b,
    })

    text_a = await host.call_tool(a, "a_tool", {"x": 1})
    text_b = await host.call_tool(b, "b_tool", {"x": 2})

    assert text_a == "from-a"
    assert text_b == "from-b"
    assert fake_a.call_calls == [("a_tool", {"x": 1})]
    assert fake_b.call_calls == [("b_tool", {"x": 2})]

    await host.aclose()


@pytest.mark.asyncio
async def test_concurrent_calls_to_same_server_serialise():
    """Per-entry ``call_lock`` must serialise concurrent calls.

    Many real MCP servers are single-threaded subprocesses; even when
    the SDK is reentrant, parallel ``call_tool`` would race the
    server's own state.
    """
    cfg = MCPServerConfig(name="srv", command="srv-mcp")

    in_flight = 0
    max_in_flight = 0
    barrier = asyncio.Event()

    class _Tracking(_FakeSession):
        async def call_tool(self, name, arguments):  # type: ignore[override]
            nonlocal in_flight, max_in_flight
            in_flight += 1
            max_in_flight = max(max_in_flight, in_flight)
            try:
                # Yield control so any racing caller has a chance to
                # bump in_flight before we decrement; the lock should
                # prevent that.
                await asyncio.sleep(0)
                await barrier.wait()
            finally:
                in_flight -= 1
            return await super().call_tool(name, arguments)

    fake = _Tracking(call_text="done")

    from thorn.core._mcp_config import mcp_server_config_identity
    host = _make_host_with_fake_sessions({mcp_server_config_identity(cfg): fake})

    async def _release_after_first_blocks() -> None:
        # Wait until both caller tasks are queued (they hit the lock),
        # then unblock everything.
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        barrier.set()

    results = await asyncio.gather(
        host.call_tool(cfg, "t", {"i": 0}),
        host.call_tool(cfg, "t", {"i": 1}),
        _release_after_first_blocks(),
    )

    # Both real call results returned ``"done"``; the helper task
    # returned ``None``.
    assert results[0] == "done"
    assert results[1] == "done"
    assert max_in_flight == 1, (
        "two calls were inside call_tool simultaneously; per-server "
        "lock did not serialise them"
    )

    await host.aclose()


@pytest.mark.asyncio
async def test_unavailable_when_mcp_not_installed():
    """Fresh hosts default ``mcp_available`` from the import probe.

    When forced off, every MCP operation raises a clear error.
    """
    host = MCPHost()
    host._mcp_available = False

    cfg = MCPServerConfig(name="srv", command="srv")
    with pytest.raises(MCPUnavailableError):
        await host.list_tools(cfg)
    with pytest.raises(MCPUnavailableError):
        await host.call_tool(cfg, "x", {})

    await host.aclose()


@pytest.mark.asyncio
async def test_aclose_is_idempotent():
    """Re-closing a host must not raise."""
    host = MCPHost()
    await host.aclose()
    await host.aclose()


@pytest.mark.asyncio
async def test_use_after_close_raises():
    """``aclose`` is terminal -- subsequent calls reject loudly."""
    host = MCPHost()
    host._mcp_available = True
    await host.aclose()

    cfg = MCPServerConfig(name="srv", command="srv")
    with pytest.raises(RuntimeError, match="MCPHost is closed"):
        await host.list_tools(cfg)


# ---------------------------------------------------------------------------
# snapshot()
# ---------------------------------------------------------------------------

class TestMCPHostSnapshot:
    @pytest.mark.asyncio
    async def test_empty_host_snapshots_empty(self):
        host = MCPHost()
        try:
            assert host.snapshot() == []
        finally:
            await host.aclose()

    @pytest.mark.asyncio
    async def test_snapshot_after_list_reports_alive_and_tool_count(self):
        cfg = MCPServerConfig(name="srv", command="srv-mcp")
        fake = _FakeSession(tools=[_FakeTool("a"), _FakeTool("b")])

        from thorn.core._mcp_config import mcp_server_config_identity
        host = _make_host_with_fake_sessions(
            {mcp_server_config_identity(cfg): fake},
        )
        try:
            await host.list_tools(cfg)
            snap = host.snapshot()
            assert len(snap) == 1
            entry = snap[0]
            assert entry.name == "srv"
            assert entry.kind == "stdio"
            assert entry.identifier == "srv-mcp"
            assert entry.alive is True
            assert entry.tool_count == 2
            assert entry.last_used_at is not None
            # ISO-8601 with offset; cheap structural check.
            assert "T" in entry.last_used_at
            # Identity hash is short hex (12 chars in production).
            assert len(entry.config_identity) == 12
            int(entry.config_identity, 16)  # parses as hex
        finally:
            await host.aclose()

    @pytest.mark.asyncio
    async def test_snapshot_call_tool_advances_last_used(self):
        cfg = MCPServerConfig(name="srv", command="srv-mcp")
        fake = _FakeSession(tools=[_FakeTool("t")])

        from thorn.core._mcp_config import mcp_server_config_identity
        host = _make_host_with_fake_sessions(
            {mcp_server_config_identity(cfg): fake},
        )
        try:
            await host.list_tools(cfg)
            first = host.snapshot()[0].last_used_at
            assert first is not None

            # Force monotonic clock advancement (real wall-clock can
            # tick at the same microsecond on fast machines).  We
            # patch the module-level _utcnow seam with a fixed later
            # timestamp.
            from datetime import datetime, timezone

            import thorn.toolhost._mcp_host as mcp_host_mod

            later = datetime(2099, 1, 1, tzinfo=timezone.utc)
            original = mcp_host_mod._utcnow
            mcp_host_mod._utcnow = lambda: later  # type: ignore[assignment]
            try:
                await host.call_tool(cfg, "t", {})
            finally:
                mcp_host_mod._utcnow = original  # type: ignore[assignment]

            second = host.snapshot()[0].last_used_at
            assert second == later.isoformat()
            assert second != first
        finally:
            await host.aclose()

    @pytest.mark.asyncio
    async def test_snapshot_distinguishes_kind_for_url_config(self):
        cfg = MCPServerConfig(name="docs", url="https://example.com/mcp")
        host = MCPHost()
        host._mcp_available = True
        try:
            entry = await host._get_or_create_entry(cfg)
            assert entry is not None
            snap = host.snapshot()
            assert len(snap) == 1
            assert snap[0].kind == "http"
            assert snap[0].identifier == "https://example.com/mcp"
            assert snap[0].alive is False
            assert snap[0].tool_count is None
            assert snap[0].last_used_at is None
        finally:
            await host.aclose()
