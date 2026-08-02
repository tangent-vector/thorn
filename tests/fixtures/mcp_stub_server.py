"""Minimal stdio-transport MCP server used by Phase-C.1 e2e tests.

Run with::

    python tests/fixtures/mcp_stub_server.py

Exposes three tools used by ``tests/test_toolhost_e2e.py``:

* ``echo(message)`` -- returns ``"echoed: <message>"`` so a successful
  call's payload is unambiguous in test assertions.
* ``slow_sleep(seconds)`` -- blocking sleep used to give the brain
  time to cancel the in-flight call.  Implemented with the synchronous
  :func:`time.sleep` so the cancellation propagates only to the brain
  / daemon layer (the MCP server keeps running for the rest of its
  sleep, which is fine: the daemon's teardown reaps it on test exit).
* ``add(a, b)`` -- a second always-fast tool so the e2e test can
  verify multi-tool listing without depending on call ordering.

The stub deliberately requires only the ``mcp`` package; no thorn
imports are reachable from here.  That keeps the daemon's spawning
path entirely SDK-driven and matches what a third-party MCP server
would look like to ``thorn-toolhost``.
"""

from __future__ import annotations

import time

from mcp.server.fastmcp import FastMCP

_server = FastMCP("thorn-stub")


@_server.tool()
def echo(message: str) -> str:
    """Echo *message* back with a stable prefix for test assertions."""
    return f"echoed: {message}"


@_server.tool()
def slow_sleep(seconds: float) -> str:
    """Block for *seconds* and then return.

    The blocking sleep here is deliberate: the e2e test cancels the
    brain-side task while this call is in flight.  Using
    :func:`time.sleep` instead of :func:`asyncio.sleep` ensures the
    cancellation has to traverse the brain -> daemon -> MCPHost
    layers, which is the path the test is asserting.
    """
    time.sleep(seconds)
    return f"slept {seconds}s"


@_server.tool()
def add(a: int, b: int) -> int:
    """Return ``a + b`` (used to assert multi-tool discovery)."""
    return a + b


if __name__ == "__main__":
    _server.run()
