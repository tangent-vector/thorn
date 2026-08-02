"""``DaemonHost``: how the brain brings up a ``thorn-toolhost`` instance.

The brain-side :class:`~thorn.toolhost._executor.DaemonToolExecutor`
talks to a daemon over a Unix-domain socket, but it does *not* care
how that daemon is brought into existence.  Phase A spawns the daemon
as a subprocess; Phase B wraps the same daemon in an OCI container.
The :class:`DaemonHost` protocol is the seam where those two paths
diverge: the executor sees only ``socket_path`` plus ``start`` /
``stop`` lifecycle hooks.

Each host implementation owns:

* the lifecycle of whatever process supervises the daemon (Python
  subprocess, container, future broker-side process, ...),
* the lifecycle of the socket file (it must exist after ``start``
  completes its readiness contract; it must be cleaned up by
  ``stop``).

The executor owns:

* polling for the socket to actually be connectable,
* the protocol handshake,
* heartbeat liveness detection,
* propagating crashes back as :class:`DaemonCrashedError`.

Phase F-flavored additions (resource limits, capability drops, etc.)
land per-host without touching the executor.

This module is deliberately narrow: a protocol plus the
:class:`SubprocessDaemonHost` implementation that preserves the
exact Phase-A subprocess behavior.  Container hosts live under
:mod:`thorn.sandbox` so the toolhost package stays free of
container-runtime concerns.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@runtime_checkable
class DaemonHost(Protocol):
    """How to bring up (and tear down) a ``thorn-toolhost`` instance.

    A host owns one daemon at a time.  ``start`` brings the daemon
    into existence and waits until the host considers it "running"
    (whatever that means for the implementation -- a launched
    subprocess, a container in the ``running`` state, etc.).  After
    ``start`` returns successfully, the brain-side executor begins
    polling :attr:`socket_path` for connectability.

    ``stop`` is idempotent: calling it on a host that was never
    started, or that already stopped, must be safe and silent.

    Implementations are expected to be single-instance (one daemon
    per host).  Re-starting after a stop is allowed and is how
    automatic crash recovery works on the executor side.
    """

    @property
    def socket_path(self) -> Path:
        """Path to the Unix-domain socket the daemon will bind.

        Stable for the lifetime of the host instance; the executor
        polls this path during connect.  For container hosts, this
        is the host-side path of the bind-mounted socket.
        """
        ...

    async def start(self) -> None:
        """Bring the daemon into existence and wait for host-level readiness.

        Implementations may do significant work here (image checks,
        container start, subprocess spawn).  The contract is:

        * On normal return, the daemon process exists and the host
          considers it "running" by whatever criterion makes sense
          for the implementation.
        * The socket file may or may not be reachable yet; the
          executor's connect-poll handles that wait.
        * Failure raises a clear, host-specific exception that
          identifies which stage failed.

        Calling ``start`` on an already-running host is a programming
        error; the executor calls ``stop`` first when restarting.
        """
        ...

    async def stop(self) -> None:
        """Tear the daemon down and clean up host-level resources.

        Implementations must remove the socket file as part of
        teardown so a subsequent start sees a clean rendezvous
        directory.  Must be idempotent.
        """
        ...


@dataclass
class SubprocessDaemonHostConfig:
    """Configuration for :class:`SubprocessDaemonHost`.

    Carries everything needed to spawn a ``python -m thorn.toolhost``
    subprocess on the host: the rendezvous socket path, the agent
    identity it serves, optional bind-style mounts (passed as CLI
    flags so the daemon resolves them itself), and a few knobs
    matching the daemon's argparse surface.
    """

    socket_path: Path
    agent_id: str
    home_path: Path | None = None
    workspace_root: Path | None = None
    log_path: Path | None = None
    python_executable: str = sys.executable
    max_concurrency: int = 8
    extra_args: tuple[str, ...] = field(default_factory=tuple)


class SubprocessDaemonHost:
    """Spawns ``thorn-toolhost`` as a Python subprocess on the host.

    This is the Phase-A shape: the brain ``fork``s a Python interpreter
    that runs ``thorn-toolhost`` directly on the host, sharing the
    host's filesystem and uid.  No container, no isolation; just a
    cleaner separation of tool execution from the brain's own event
    loop.  Phase B introduces :class:`thorn.sandbox.ContainerDaemonHost`
    as an alternative implementation.

    The ``start`` contract is "subprocess has been launched"; we do
    not wait for the daemon to bind its socket here.  That lets the
    executor's socket-reachable poll be the single source of "is the
    daemon actually serving?" truth, common across host kinds.
    """

    def __init__(self, config: SubprocessDaemonHostConfig) -> None:
        self._config = config
        self._process: asyncio.subprocess.Process | None = None

    @property
    def socket_path(self) -> Path:
        return self._config.socket_path

    @property
    def process(self) -> asyncio.subprocess.Process | None:
        """The supervised subprocess, or ``None`` if not running.

        Exposed for diagnostics (tests inspect ``returncode``); not
        intended for callers to manipulate directly.
        """
        return self._process

    async def start(self) -> None:
        if self._process is not None and self._process.returncode is None:
            raise RuntimeError(
                "SubprocessDaemonHost.start called while a daemon is "
                "already running; call stop first",
            )

        cmd = self._build_command()
        self._process = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )

    async def stop(self) -> None:
        proc = self._process
        if proc is not None and proc.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                proc.terminate()
            try:
                await asyncio.wait_for(proc.wait(), timeout=2.0)
            except asyncio.TimeoutError:
                with contextlib.suppress(ProcessLookupError):
                    proc.kill()
                with contextlib.suppress(Exception):
                    await proc.wait()
        self._process = None
        # Remove the rendezvous socket so a subsequent start sees a
        # clean directory; the daemon also unlinks on shutdown but
        # crashed daemons can leave a stale entry behind.
        with contextlib.suppress(FileNotFoundError):
            os.unlink(self._config.socket_path)

    def _build_command(self) -> list[str]:
        cmd: list[str] = [
            self._config.python_executable,
            "-m",
            "thorn.toolhost",
            "--socket",
            str(self._config.socket_path),
            "--agent-id",
            self._config.agent_id,
            "--max-concurrency",
            str(self._config.max_concurrency),
        ]
        if self._config.home_path is not None:
            cmd.extend(["--home", str(self._config.home_path)])
        if self._config.workspace_root is not None:
            cmd.extend(["--workspace-root", str(self._config.workspace_root)])
        if self._config.log_path is not None:
            cmd.extend(["--log-file", str(self._config.log_path)])
        cmd.extend(self._config.extra_args)
        return cmd


__all__ = [
    "DaemonHost",
    "SubprocessDaemonHost",
    "SubprocessDaemonHostConfig",
]
