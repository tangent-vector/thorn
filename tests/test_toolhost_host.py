"""Tests for the :class:`DaemonHost` abstraction.

Phase B introduces a small protocol that decouples
:class:`DaemonToolExecutor` from the specific way a ``thorn-toolhost``
daemon is brought into existence.  The Phase-A path
(:class:`SubprocessDaemonHost`) preserves the original behavior; this
module covers:

* the construction-time validation that a host's socket path matches
  the executor's connection-level config (catching wiring mistakes
  loudly instead of silently connecting to the wrong path),
* that the executor invokes ``host.start`` on the lazy first-use
  path and ``host.stop`` on teardown,
* that ``DaemonExecutorConfig.to_subprocess_host_config`` round-trips
  the relevant fields verbatim,
* that the in-package :class:`SubprocessDaemonHost` can boot a real
  ``python -m thorn.toolhost`` and shut it down cleanly.

The fake host used here is the same shape Phase B's container host
will adopt: an awaitable ``start``, an awaitable ``stop``, and a
``socket_path`` property.  Keeping the tests against the protocol
(not the concrete subprocess implementation) means they stay valid
when the container host lands.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

from thorn.core._executor import ToolInvocation
from thorn.toolhost._executor import (
    DaemonExecutorConfig,
    DaemonToolExecutor,
    DaemonUnavailableError,
)
from thorn.toolhost._host import (
    DaemonHost,
    SubprocessDaemonHost,
    SubprocessDaemonHostConfig,
)

pytestmark = pytest.mark.skipif(
    sys.platform == "win32", reason="thorn-toolhost is Unix-only",
)


class _RecordingHost:
    """Minimal :class:`DaemonHost` that records start/stop calls.

    Does not actually run a daemon -- the tests that use it pair the
    executor with a same-loop server via ``adopt_streams`` to drive the
    handshake/IPC, then exercise the ``host.start`` / ``host.stop`` path
    in isolation.
    """

    def __init__(self, socket_path: Path) -> None:
        self._socket_path = socket_path
        self.start_calls = 0
        self.stop_calls = 0
        self.start_should_raise: Exception | None = None

    @property
    def socket_path(self) -> Path:
        return self._socket_path

    async def start(self) -> None:
        self.start_calls += 1
        if self.start_should_raise is not None:
            raise self.start_should_raise

    async def stop(self) -> None:
        self.stop_calls += 1


class TestDaemonHostProtocol:
    def test_recording_host_satisfies_protocol(self, tmp_path: Path) -> None:
        host = _RecordingHost(tmp_path / "x.sock")
        # ``DaemonHost`` is ``runtime_checkable`` so this is a real
        # structural check, not just documentation.
        assert isinstance(host, DaemonHost)

    def test_subprocess_host_satisfies_protocol(self, tmp_path: Path) -> None:
        host = SubprocessDaemonHost(
            SubprocessDaemonHostConfig(
                socket_path=tmp_path / "x.sock",
                agent_id="agent-x",
            )
        )
        assert isinstance(host, DaemonHost)


class TestExecutorHostWiring:
    def test_socket_path_mismatch_raises(self, tmp_path: Path) -> None:
        host = _RecordingHost(tmp_path / "host.sock")
        config = DaemonExecutorConfig(
            socket_path=tmp_path / "config.sock",
            agent_id="agent-x",
        )
        with pytest.raises(ValueError, match="socket_path"):
            DaemonToolExecutor(config, host=host)

    def test_default_host_is_subprocess(self, tmp_path: Path) -> None:
        config = DaemonExecutorConfig(
            socket_path=tmp_path / "x.sock",
            agent_id="agent-x",
        )
        executor = DaemonToolExecutor(config)
        assert isinstance(executor.host, SubprocessDaemonHost)
        assert executor.host.socket_path == config.socket_path

    def test_explicit_host_is_used(self, tmp_path: Path) -> None:
        host = _RecordingHost(tmp_path / "x.sock")
        config = DaemonExecutorConfig(
            socket_path=tmp_path / "x.sock",
            agent_id="agent-x",
        )
        executor = DaemonToolExecutor(config, host=host)
        assert executor.host is host


class TestExecutorConfigToHostConfig:
    def test_round_trips_relevant_fields(self, tmp_path: Path) -> None:
        config = DaemonExecutorConfig(
            socket_path=tmp_path / "x.sock",
            agent_id="agent-x",
            home_path=tmp_path / "home",
            workspace_root=tmp_path / "ws",
            log_path=tmp_path / "log",
            python_executable="/opt/python/bin/python3",
            max_concurrency=12,
            extra_args=("--debug",),
        )
        host_cfg = config.to_subprocess_host_config()
        assert host_cfg.socket_path == config.socket_path
        assert host_cfg.agent_id == config.agent_id
        assert host_cfg.home_path == config.home_path
        assert host_cfg.workspace_root == config.workspace_root
        assert host_cfg.log_path == config.log_path
        assert host_cfg.python_executable == config.python_executable
        assert host_cfg.max_concurrency == config.max_concurrency
        assert host_cfg.extra_args == config.extra_args

    def test_omitted_fields_are_none(self, tmp_path: Path) -> None:
        config = DaemonExecutorConfig(
            socket_path=tmp_path / "x.sock",
            agent_id="agent-x",
        )
        host_cfg = config.to_subprocess_host_config()
        assert host_cfg.home_path is None
        assert host_cfg.workspace_root is None
        assert host_cfg.log_path is None
        assert host_cfg.extra_args == ()


class TestExecutorInvokesHostLifecycle:
    @pytest.mark.asyncio
    async def test_invoke_failure_propagates_host_start_error(
        self, tmp_path: Path,
    ) -> None:
        host = _RecordingHost(tmp_path / "x.sock")
        host.start_should_raise = RuntimeError("boom")
        config = DaemonExecutorConfig(
            socket_path=tmp_path / "x.sock",
            agent_id="agent-x",
            connect_timeout_s=0.5,
        )
        executor = DaemonToolExecutor(config, host=host)
        with pytest.raises(DaemonUnavailableError, match="boom"):
            await executor.invoke(
                ToolInvocation(call_id="c1", tool_name="x", arguments={}),
            )
        assert host.start_calls == 1
        # Failed start is followed by a stop so the host can clean up
        # any partial state (a pre-readiness container, a doomed
        # subprocess, ...) before the executor surfaces the error.
        assert host.stop_calls == 1
        await executor.aclose()

    @pytest.mark.asyncio
    async def test_aclose_without_start_does_not_call_host_stop(
        self, tmp_path: Path,
    ) -> None:
        # If the executor never reached the host (e.g. closed before
        # any invoke), it should not call host.stop -- the host was
        # never asked to start, so there's nothing to tear down.
        host = _RecordingHost(tmp_path / "x.sock")
        config = DaemonExecutorConfig(
            socket_path=tmp_path / "x.sock",
            agent_id="agent-x",
        )
        executor = DaemonToolExecutor(config, host=host)
        await executor.aclose()
        assert host.start_calls == 0
        assert host.stop_calls == 0


class TestSubprocessDaemonHostLifecycle:
    @pytest.mark.asyncio
    async def test_start_then_stop_cleans_socket(self, tmp_path: Path) -> None:
        socket_path = tmp_path / "control" / "toolhost.sock"
        host = SubprocessDaemonHost(
            SubprocessDaemonHostConfig(
                socket_path=socket_path,
                agent_id="agent-x",
            )
        )
        await host.start()
        try:
            assert host.process is not None
            assert host.process.returncode is None

            # The daemon binds the socket asynchronously; give it up
            # to a couple of seconds before declaring the start broken.
            for _ in range(200):
                if socket_path.exists():
                    break
                await asyncio.sleep(0.01)
            assert socket_path.exists(), (
                f"daemon never bound {socket_path}",
            )
        finally:
            await host.stop()

        assert not socket_path.exists()
        assert host.process is None

    @pytest.mark.asyncio
    async def test_double_start_raises(self, tmp_path: Path) -> None:
        host = SubprocessDaemonHost(
            SubprocessDaemonHostConfig(
                socket_path=tmp_path / "x.sock",
                agent_id="agent-x",
            )
        )
        await host.start()
        try:
            with pytest.raises(RuntimeError, match="already running"):
                await host.start()
        finally:
            await host.stop()

    @pytest.mark.asyncio
    async def test_stop_without_start_is_noop(self, tmp_path: Path) -> None:
        host = SubprocessDaemonHost(
            SubprocessDaemonHostConfig(
                socket_path=tmp_path / "x.sock",
                agent_id="agent-x",
            )
        )
        await host.stop()
        assert host.process is None
