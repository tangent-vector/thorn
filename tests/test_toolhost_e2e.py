"""End-to-end tests through a real ``thorn-toolhost`` subprocess.

These tests spawn ``python -m thorn.toolhost`` as a child process, let
:class:`DaemonToolExecutor` open the Unix-domain socket, and then drive
the full IPC path: handshake, dispatch, cancellation, crash recovery,
and malformed-frame handling.

They are slower and more failure-modes-rich than the in-memory pair
tests in ``test_toolhost_executor.py``; the latter cover the executor's
multiplexing / cancellation logic in isolation.  This module exists to
catch the things only a real subprocess can fail at: the binary path
through ``argparse``, socket binding, the ``python -m thorn.toolhost``
import surface, signal handling, and so on.
"""

from __future__ import annotations

import asyncio
import struct
import sys
from pathlib import Path

import pytest

from thorn.core._executor import ToolInvocation
from thorn.toolhost._executor import (
    DaemonCrashedError,
    DaemonExecutorConfig,
    DaemonToolExecutor,
    DaemonUnavailableError,
)


pytestmark = [
    pytest.mark.skipif(
        sys.platform == "win32",
        reason="thorn-toolhost is Unix-only",
    ),
]


def _make_config(
    tmp_path: Path,
    *,
    name: str = "agent-x",
    home: Path | None = None,
    workspace: Path | None = None,
    heartbeat_interval_s: float = 0.2,
    heartbeat_dead_s: float = 5.0,
) -> DaemonExecutorConfig:
    """Build a :class:`DaemonExecutorConfig` for a one-off subprocess.

    Each test gets its own socket path under *tmp_path* so the daemons
    do not collide.  The default heartbeat parameters are fast enough
    that a stuck daemon surfaces inside the test's timeout budget but
    loose enough that ordinary scheduling latency does not falsely
    fire the watchdog.
    """
    socket_path = tmp_path / f"{name}.sock"
    log_path = tmp_path / f"{name}.log"
    return DaemonExecutorConfig(
        socket_path=socket_path,
        agent_id=name,
        home_path=home,
        workspace_root=workspace,
        log_path=log_path,
        heartbeat_interval_s=heartbeat_interval_s,
        heartbeat_dead_s=heartbeat_dead_s,
        connect_timeout_s=10.0,
        handshake_timeout_s=10.0,
    )


async def _terminate(executor: DaemonToolExecutor) -> None:
    """Best-effort teardown helper used in test ``finally`` clauses."""
    try:
        await asyncio.wait_for(executor.aclose(), timeout=5.0)
    except (asyncio.TimeoutError, Exception):
        pass


class TestSubprocessHandshake:
    @pytest.mark.asyncio
    async def test_handshake_succeeds_against_real_subprocess(
        self, tmp_path: Path,
    ):
        executor = DaemonToolExecutor(_make_config(tmp_path))
        try:
            await executor._ensure_running()
            hello = executor.daemon_hello
            assert hello is not None
            assert hello.per_agent_state["agent_id"] == "agent-x"
        finally:
            await _terminate(executor)

    @pytest.mark.asyncio
    async def test_socket_file_is_cleaned_up_on_close(
        self, tmp_path: Path,
    ):
        executor = DaemonToolExecutor(_make_config(tmp_path))
        try:
            await executor._ensure_running()
            assert executor.config.socket_path.exists()
        finally:
            await _terminate(executor)
        # Allow the OS a moment to flush; the daemon unlinks on exit
        # and the executor unlinks when it kills the subprocess, so
        # by the time aclose() returns the file should be gone.
        for _ in range(20):
            if not executor.config.socket_path.exists():
                break
            await asyncio.sleep(0.05)
        assert not executor.config.socket_path.exists()


class TestSubprocessRoundTrip:
    @pytest.mark.asyncio
    async def test_read_file_round_trips_through_daemon(
        self, tmp_path: Path,
    ):
        target = tmp_path / "hello.txt"
        target.write_text("hello world\n", encoding="utf-8")

        executor = DaemonToolExecutor(
            _make_config(tmp_path, workspace=tmp_path),
        )
        try:
            result = await asyncio.wait_for(
                executor.invoke(
                    ToolInvocation(
                        call_id="read-1",
                        tool_name="read_file",
                        arguments={"path": str(target)},
                    ),
                ),
                timeout=15.0,
            )
            assert result.is_error is False
            assert "hello world" in result.content
        finally:
            await _terminate(executor)

    @pytest.mark.asyncio
    async def test_create_file_round_trips_through_daemon(
        self, tmp_path: Path,
    ):
        target = tmp_path / "out.txt"
        executor = DaemonToolExecutor(
            _make_config(tmp_path, workspace=tmp_path),
        )
        try:
            result = await asyncio.wait_for(
                executor.invoke(
                    ToolInvocation(
                        call_id="write-1",
                        tool_name="create_file",
                        arguments={
                            "path": str(target),
                            "content": "phase-a\n",
                        },
                    ),
                ),
                timeout=15.0,
            )
            assert result.is_error is False
            assert target.read_text(encoding="utf-8") == "phase-a\n"
        finally:
            await _terminate(executor)

    @pytest.mark.asyncio
    async def test_run_shell_round_trips_through_daemon(
        self, tmp_path: Path,
    ):
        executor = DaemonToolExecutor(
            _make_config(tmp_path, workspace=tmp_path),
        )
        try:
            result = await asyncio.wait_for(
                executor.invoke(
                    ToolInvocation(
                        call_id="sh-1",
                        tool_name="run_shell",
                        arguments={"command": "echo daemon-says-hi"},
                    ),
                ),
                timeout=15.0,
            )
            assert result.is_error is False
            assert "daemon-says-hi" in result.content
        finally:
            await _terminate(executor)

    @pytest.mark.asyncio
    async def test_unknown_tool_returns_error_result(
        self, tmp_path: Path,
    ):
        executor = DaemonToolExecutor(_make_config(tmp_path))
        try:
            result = await asyncio.wait_for(
                executor.invoke(
                    ToolInvocation(
                        call_id="bad-1",
                        tool_name="no_such_tool",
                        arguments={},
                    ),
                ),
                timeout=15.0,
            )
            assert result.is_error is True
            assert result.error_kind == "unknown_tool"
        finally:
            await _terminate(executor)


class TestSubprocessConcurrency:
    @pytest.mark.asyncio
    async def test_many_concurrent_calls_each_get_their_own_response(
        self, tmp_path: Path,
    ):
        executor = DaemonToolExecutor(
            _make_config(tmp_path, workspace=tmp_path),
        )
        try:
            tasks = [
                asyncio.create_task(
                    executor.invoke(
                        ToolInvocation(
                            call_id=f"c-{i}",
                            tool_name="run_shell",
                            arguments={"command": f"echo result-{i}"},
                        ),
                    )
                )
                for i in range(8)
            ]
            results = await asyncio.wait_for(
                asyncio.gather(*tasks), timeout=20.0,
            )
            assert all(not r.is_error for r in results)
            outputs = sorted(r.content.strip() for r in results)
            for i in range(8):
                assert any(f"result-{i}" in r.content for r in results)
            assert len(outputs) == 8
        finally:
            await _terminate(executor)


class TestSubprocessCancellation:
    @pytest.mark.asyncio
    async def test_caller_cancellation_kills_long_running_shell(
        self, tmp_path: Path,
    ):
        executor = DaemonToolExecutor(
            _make_config(tmp_path, workspace=tmp_path),
        )
        try:
            # ``run_shell`` should propagate cancel down to the spawned
            # subprocess; if it doesn't, this test hangs until the
            # outer wait_for fires.
            invoke_task = asyncio.create_task(
                executor.invoke(
                    ToolInvocation(
                        call_id="slow-1",
                        tool_name="run_shell",
                        arguments={"command": "sleep 30"},
                    ),
                )
            )
            # Give the daemon time to actually spawn the sleep.
            await asyncio.sleep(0.5)
            invoke_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(invoke_task, timeout=10.0)
        finally:
            await _terminate(executor)


class TestSubprocessCrashRecovery:
    @pytest.mark.asyncio
    async def test_killing_daemon_fails_in_flight_call(
        self, tmp_path: Path,
    ):
        executor = DaemonToolExecutor(
            _make_config(tmp_path, workspace=tmp_path),
        )
        try:
            await executor._ensure_running()
            host_proc = executor.host.process
            assert host_proc is not None

            slow_task = asyncio.create_task(
                executor.invoke(
                    ToolInvocation(
                        call_id="slow-2",
                        tool_name="run_shell",
                        arguments={"command": "sleep 30"},
                    ),
                )
            )
            await asyncio.sleep(0.3)

            # SIGKILL the daemon out from under the in-flight call.
            host_proc.kill()
            await host_proc.wait()

            with pytest.raises((DaemonCrashedError, asyncio.CancelledError)):
                await asyncio.wait_for(slow_task, timeout=10.0)
        finally:
            await _terminate(executor)

    @pytest.mark.asyncio
    async def test_invoke_after_crash_restarts_daemon(
        self, tmp_path: Path,
    ):
        executor = DaemonToolExecutor(
            _make_config(tmp_path, workspace=tmp_path),
        )
        try:
            await executor._ensure_running()
            first_proc = executor.host.process
            first_pid = first_proc.pid if first_proc is not None else None
            assert first_pid is not None and first_proc is not None

            first_proc.kill()
            await first_proc.wait()

            # Drain the reader-loop's "connection closed" handling
            # so the next invoke spawns a fresh subprocess from a
            # clean state instead of racing the crash detector.
            for _ in range(40):
                if executor._writer is None:
                    break
                await asyncio.sleep(0.05)
            assert executor._writer is None

            result = await asyncio.wait_for(
                executor.invoke(
                    ToolInvocation(
                        call_id="post-crash-1",
                        tool_name="run_shell",
                        arguments={"command": "echo restarted"},
                    ),
                ),
                timeout=15.0,
            )
            assert result.is_error is False
            assert "restarted" in result.content
            second_proc = executor.host.process
            assert second_proc is not None
            assert second_proc.pid != first_pid
        finally:
            await _terminate(executor)


class TestSubprocessMalformedFrames:
    @pytest.mark.asyncio
    async def test_garbage_after_handshake_drops_connection(
        self, tmp_path: Path,
    ):
        """If the brain sends nonsense after the handshake, the daemon
        should drop the connection (per the protocol spec) without
        crashing the process.  We open a raw socket, do the Hello
        handshake by hand, then write a length prefix that promises a
        payload we never deliver -- the daemon should hit
        :class:`asyncio.IncompleteReadError`, close *this* connection,
        and remain available for new ones.  A follow-up connection
        through :class:`DaemonToolExecutor` confirms the daemon is
        still healthy.
        """
        from thorn.toolhost._protocol import (
            PROTOCOL_MAJOR,
            PROTOCOL_MINOR,
            Hello,
            encode_frame,
            read_frame,
        )

        socket_path = tmp_path / "raw.sock"
        log_path = tmp_path / "raw.log"
        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "thorn.toolhost",
            "--socket",
            str(socket_path),
            "--agent-id",
            "raw",
            "--log-file",
            str(log_path),
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )

        try:
            # Wait for the daemon to bind the socket.
            for _ in range(200):
                if socket_path.exists():
                    break
                await asyncio.sleep(0.05)
            assert socket_path.exists(), "daemon did not bind socket in time"

            reader, writer = await asyncio.open_unix_connection(
                path=str(socket_path),
            )

            payload = encode_frame(
                Hello(
                    protocol_major=PROTOCOL_MAJOR,
                    protocol_minor=PROTOCOL_MINOR,
                    thorn_version="raw-test",
                ),
            )
            writer.write(payload)
            await writer.drain()

            response = await asyncio.wait_for(read_frame(reader), timeout=5.0)
            assert isinstance(response, Hello)

            # Promise a 32-byte payload, then close the writer half.
            writer.write(struct.pack(">I", 32))
            await writer.drain()
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass

            # The daemon should still be alive (i.e. the malformed
            # frame closed the connection but did not crash the
            # process).  Give it a beat to actually reach the
            # ``IncompleteReadError`` path, then poll the subprocess
            # status -- ``returncode is None`` means it is still
            # running and serving on the socket.  We do not open a
            # second connection because the daemon's per-process
            # ``_connection_count`` is monotonic (it accepts exactly
            # one connection over its lifetime by design).
            await asyncio.sleep(0.3)
            assert proc.returncode is None, (
                "daemon process exited after malformed frame"
            )
        finally:
            if proc.returncode is None:
                proc.terminate()
                try:
                    await asyncio.wait_for(proc.wait(), timeout=5.0)
                except asyncio.TimeoutError:
                    proc.kill()
                    try:
                        await proc.wait()
                    except Exception:
                        pass


class TestMultiAgentDaemons:
    @pytest.mark.asyncio
    async def test_two_agents_get_independent_daemons(
        self, tmp_path: Path,
    ):
        a = DaemonToolExecutor(_make_config(tmp_path, name="agent-a"))
        b = DaemonToolExecutor(_make_config(tmp_path, name="agent-b"))
        try:
            await a._ensure_running()
            await b._ensure_running()
            a_proc = a.host.process
            b_proc = b.host.process
            assert a_proc is not None and b_proc is not None
            assert a_proc.pid != b_proc.pid
            assert a.config.socket_path != b.config.socket_path
        finally:
            await _terminate(a)
            await _terminate(b)


class TestUnreachableDaemon:
    @pytest.mark.asyncio
    async def test_handshake_failure_surfaces_as_daemon_unavailable(
        self, tmp_path: Path,
    ):
        """If we point the executor at a non-existent ``python``, the
        subprocess fails immediately and the executor should raise
        :class:`DaemonUnavailableError` rather than hang on the socket.
        """
        config = DaemonExecutorConfig(
            socket_path=tmp_path / "never.sock",
            agent_id="absent",
            python_executable="/nonexistent/python",
            connect_timeout_s=2.0,
            handshake_timeout_s=2.0,
        )
        executor = DaemonToolExecutor(config)
        try:
            with pytest.raises(DaemonUnavailableError):
                await executor.invoke(
                    ToolInvocation(
                        call_id="x",
                        tool_name="read_file",
                        arguments={"path": "/etc/hostname"},
                    ),
                )
        finally:
            await _terminate(executor)
