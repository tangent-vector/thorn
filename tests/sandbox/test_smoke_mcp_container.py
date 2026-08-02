"""Opt-in smoke test for the full MCP-in-container path (Phase C.1).

Skipped unless all of:

* ``podman`` or ``docker`` is on ``PATH``,
* the default ``thorn-sandbox:<version>`` image is present locally
  (this test deliberately does **not** auto-build it -- per the
  Phase B contract, image build is an explicit operator action via
  ``thorn sandbox build``), and
* the optional ``mcp`` Python package is importable on the host
  (only used here for skip detection; the in-container daemon
  installs ``mcp`` from the image, not from the host).

Verifies the full Phase-C.1 chain over a real OCI runtime:

    DaemonToolExecutor (host) -> Unix socket
        -> thorn-toolhost (in container)
        -> MCPHost (in container)
        -> python -c "<inline FastMCP server>" (subprocess in container)

This complements the in-process unit tests
(``tests/test_mcp_host.py``, ``tests/test_toolhost_executor.py``) and
the host-subprocess e2e tests (``tests/test_toolhost_e2e.py``,
``TestSubprocessMCP``) by adding the only path that exercises a real
OCI container *and* MCP execution at the same time.

Run with::

    pytest -m requires_podman tests/sandbox/test_smoke_mcp_container.py
or::

    pytest -m requires_docker tests/sandbox/test_smoke_mcp_container.py
"""

from __future__ import annotations

import asyncio
import shutil
import sys
import uuid
from pathlib import Path

import pytest

from thorn.core._executor import ToolInvocation
from thorn.core._mcp_config import MCPServerConfig
from thorn.sandbox import (
    ContainerDaemonHost,
    ContainerHostConfig,
    DockerAdapter,
    PodmanAdapter,
    default_sandbox_image_tag,
)
from thorn.toolhost._executor import DaemonExecutorConfig, DaemonToolExecutor

_PODMAN_AVAILABLE = shutil.which("podman") is not None
_DOCKER_AVAILABLE = shutil.which("docker") is not None

_mcp_pkg_available = True
try:  # pragma: no cover - import probe
    import mcp  # noqa: F401
except Exception:  # pragma: no cover
    _mcp_pkg_available = False


# A self-contained FastMCP server expressed as a string the in-container
# daemon can execute via ``python -c``.  Avoids any bind-mount
# gymnastics: the container already has ``python`` and the ``mcp``
# package thanks to ``Dockerfile.sandbox``'s ``[mcp]`` extra, so a
# couple of inline imports is the simplest reproducible payload.
_INLINE_MCP_STUB = (
    "from mcp.server.fastmcp import FastMCP\n"
    "m = FastMCP('inline-stub')\n"
    "@m.tool()\n"
    "def echo(message: str) -> str:\n"
    "    return f'echoed-in-container: {message}'\n"
    "m.run()\n"
)


def _dev_mount_src() -> Path:
    """Return the ``src/`` directory that should be bind-mounted as the
    framework code inside the container.

    Without this, a stale ``thorn-sandbox`` image (built before Phase
    C.1) would run the older daemon and fail the test for the wrong
    reason.  Mirrors :meth:`Runtime._dev_mount_runtime_path`.
    """
    import thorn

    return Path(thorn.__file__).resolve().parent.parent


def _container_name() -> str:
    return f"thorn-mcp-smoke-{uuid.uuid4().hex[:12]}"


async def _smoke(adapter) -> None:
    image = default_sandbox_image_tag()
    if not await adapter.image_exists(image):
        pytest.skip(
            f"sandbox image {image!r} not present; run 'thorn sandbox build' "
            "to enable this smoke test",
        )

    name = _container_name()
    host_dir_root = Path("/tmp") / f"thorn-mcp-smoke-{uuid.uuid4().hex[:12]}"
    home = host_dir_root / "home"
    workspace = host_dir_root / "workspace"
    control = host_dir_root / "control"
    home.mkdir(parents=True)
    workspace.mkdir(parents=True)
    control.mkdir(parents=True)

    # Target uid/gid for the in-container daemon's final identity.
    # The entrypoint trampoline reads this from the
    # ``THORN_SANDBOX_UID`` / ``THORN_SANDBOX_GID`` env vars
    # (injected by :meth:`ContainerDaemonHost._build_container_spec`)
    # and ``setpriv``-drops to it after the one-shot broker CA
    # install -- so the daemon writes the control dir's socket /
    # log / mcp_state.json with the test's ownership and cleanup
    # succeeds.  On rootless podman, the container still needs to
    # boot as the user-namespace root; :class:`PodmanAdapter`'s
    # default ``--userns=keep-id`` suffices for that mapping.
    import os
    user = f"{os.getuid()}:{os.getgid()}"

    cfg = ContainerHostConfig(
        agent_id="smoke-agent",
        container_name=name,
        image=image,
        adapter=adapter,
        host_home_dir=home,
        host_workspace_dir=workspace,
        host_control_dir=control,
        dev_mount_runtime=_dev_mount_src(),
        user=user,
        container_ready_timeout_s=30.0,
    )
    host = ContainerDaemonHost(cfg)

    # Hand the host to the executor so a single ``aclose`` tears
    # everything down -- mirrors how Runtime composes the two in
    # production.
    executor = DaemonToolExecutor(
        DaemonExecutorConfig(
            socket_path=host.socket_path,
            agent_id="smoke-agent",
            connect_timeout_s=20.0,
            handshake_timeout_s=20.0,
            heartbeat_interval_s=0.5,
            heartbeat_dead_s=10.0,
        ),
        host=host,
    )
    try:
        mcp_cfg = MCPServerConfig(
            name="inline-stub",
            command="python",
            args=["-c", _INLINE_MCP_STUB],
        )
        tools = await asyncio.wait_for(
            executor.list_mcp_server_tools(mcp_cfg),
            timeout=60.0,
        )
        names = {t["function"]["name"] for t in tools}
        assert "echo" in names, names

        result = await asyncio.wait_for(
            executor.invoke(
                ToolInvocation(
                    call_id="c1",
                    tool_name="echo",
                    arguments={"message": "smoke"},
                    mcp_server_config=mcp_cfg,
                ),
            ),
            timeout=30.0,
        )
        assert result.is_error is False, result.content
        assert "echoed-in-container: smoke" in result.content
    finally:
        try:
            await asyncio.wait_for(executor.aclose(), timeout=15.0)
        except (asyncio.TimeoutError, Exception):
            pass
        # Best-effort tmpdir cleanup; the container has already
        # released its bind-mounts by the time aclose() returns.
        import os
        if not os.environ.get("THORN_SMOKE_KEEP_DIR"):
            try:
                shutil.rmtree(host_dir_root, ignore_errors=True)
            except Exception:
                pass


@pytest.mark.requires_podman
@pytest.mark.skipif(
    sys.platform == "win32", reason="thorn-toolhost is Unix-only",
)
@pytest.mark.skipif(not _PODMAN_AVAILABLE, reason="podman not on PATH")
@pytest.mark.skipif(
    not _mcp_pkg_available, reason="mcp package not installed",
)
def test_mcp_container_smoke_podman() -> None:
    asyncio.run(_smoke(PodmanAdapter()))


@pytest.mark.requires_docker
@pytest.mark.skipif(
    sys.platform == "win32", reason="thorn-toolhost is Unix-only",
)
@pytest.mark.skipif(not _DOCKER_AVAILABLE, reason="docker not on PATH")
@pytest.mark.skipif(
    not _mcp_pkg_available, reason="mcp package not installed",
)
def test_mcp_container_smoke_docker() -> None:
    asyncio.run(_smoke(DockerAdapter()))
