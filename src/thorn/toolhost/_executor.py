"""Brain-side :class:`ToolExecutor` that talks to a ``thorn-toolhost`` daemon.

The agent loop never sees the daemon directly; it only sees a
:class:`ToolExecutor`.  :class:`DaemonToolExecutor` adapts the protocol
defined in :mod:`thorn.toolhost._protocol` to that interface:

* :meth:`invoke` ships a :class:`ToolCallRequest` over the connection
  and ``await``s the matching :class:`ToolCallResponse` keyed by
  ``call_id``.
* :meth:`cancel` ships a :class:`ToolCallCancel` and lets the daemon
  decide what cancellation actually means for the in-flight call.
* :meth:`aclose` shuts the connection (and the subprocess, if we
  started one) cleanly.

The implementation deliberately splits subprocess management from
stream multiplexing so unit tests can drive the executor over an
in-memory stream pair without paying for a real ``fork``.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from thorn.core._executor import (
    OnChunkCallback,
    ToolExecutor,
    ToolInvocation,
    ToolInvocationResult,
)
from thorn.toolhost._protocol import (
    PROTOCOL_MAJOR,
    PROTOCOL_MINOR,
    Frame,
    Heartbeat,
    Hello,
    ProtocolError,
    ToolCallCancel,
    ToolCallChunk,
    ToolCallRequest,
    ToolCallResponse,
    read_frame,
    write_frame,
)

logger = logging.getLogger(__name__)


class DaemonUnavailableError(RuntimeError):
    """Raised when the brain cannot reach a working daemon.

    Either the subprocess never started, the socket never bound, the
    handshake failed, or the daemon crashed before serving a frame.
    Wraps the underlying error so callers do not have to introspect
    ``__cause__`` to render a useful message.
    """


class DaemonCrashedError(RuntimeError):
    """Raised inside :meth:`DaemonToolExecutor.invoke` when the daemon dies.

    Distinguishes a daemon-side crash (transient, retryable: the next
    invocation will spin up a fresh subprocess) from a tool-level
    failure (returned in the response payload).
    """


@dataclass
class DaemonExecutorConfig:
    """Configuration for spawning and supervising the daemon.

    The defaults are tuned for human interactive latency: a five-second
    connect budget covers cold-cache subprocess startup, and the
    twelve-second heartbeat dead window is tight enough that a wedged
    daemon surfaces while the user is still likely watching.
    """

    socket_path: Path
    agent_id: str
    home_path: Path | None = None
    workspace_root: Path | None = None
    log_path: Path | None = None
    python_executable: str = sys.executable
    max_concurrency: int = 8
    connect_timeout_s: float = 5.0
    handshake_timeout_s: float = 5.0
    heartbeat_interval_s: float = 4.0
    heartbeat_dead_s: float = 12.0
    auto_restart: bool = True
    extra_args: tuple[str, ...] = ()


class DaemonToolExecutor(ToolExecutor):
    """``ToolExecutor`` backed by a ``thorn-toolhost`` subprocess.

    Lifecycle::

        executor = DaemonToolExecutor(config)
        result = await executor.invoke(invocation)   # lazily starts on first call
        await executor.aclose()                      # shuts everything down

    The executor is safe to share across coroutines.  Send frames are
    serialized by an ``asyncio.Lock``; per-call replies are matched to
    their callers via futures keyed by ``call_id``.  The daemon's death
    is detected either by the reader task encountering an
    :class:`asyncio.IncompleteReadError` or by the heartbeat watchdog
    failing to receive an echo within ``heartbeat_dead_s`` seconds.
    """

    def __init__(self, config: DaemonExecutorConfig) -> None:
        self._config = config
        self._lock = asyncio.Lock()
        self._writer: asyncio.StreamWriter | None = None
        self._reader: asyncio.StreamReader | None = None
        self._reader_task: asyncio.Task | None = None
        self._heartbeat_task: asyncio.Task | None = None
        self._process: asyncio.subprocess.Process | None = None
        self._owns_process: bool = False
        self._in_flight: dict[str, asyncio.Future[ToolCallResponse]] = {}
        self._chunk_callbacks: dict[str, OnChunkCallback] = {}
        self._daemon_hello: Hello | None = None
        self._closed: bool = False
        self._last_frame_received_at: float = 0.0
        self._send_lock: asyncio.Lock = asyncio.Lock()

    @property
    def config(self) -> DaemonExecutorConfig:
        return self._config

    @property
    def daemon_hello(self) -> Hello | None:
        """The :class:`Hello` we received from the daemon, or ``None``.

        Useful for callers that want to surface the daemon's reported
        version / feature flags.  Reset to ``None`` when the daemon
        connection is reset.
        """
        return self._daemon_hello

    # ------------------------------------------------------------------
    # ToolExecutor interface
    # ------------------------------------------------------------------

    async def invoke(
        self,
        invocation: ToolInvocation,
        *,
        on_chunk: OnChunkCallback | None = None,
    ) -> ToolInvocationResult:
        if self._closed:
            raise RuntimeError("DaemonToolExecutor is closed")

        await self._ensure_running()

        future: asyncio.Future[ToolCallResponse] = asyncio.get_running_loop().create_future()
        self._in_flight[invocation.call_id] = future
        if on_chunk is not None:
            self._chunk_callbacks[invocation.call_id] = on_chunk

        try:
            await self._send(
                ToolCallRequest(
                    call_id=invocation.call_id,
                    tool_name=invocation.tool_name,
                    arguments=dict(invocation.arguments),
                )
            )
        except Exception:
            self._in_flight.pop(invocation.call_id, None)
            self._chunk_callbacks.pop(invocation.call_id, None)
            raise

        try:
            response = await future
        except asyncio.CancelledError:
            with contextlib.suppress(Exception):
                await self._send(ToolCallCancel(call_id=invocation.call_id))
            raise
        finally:
            self._in_flight.pop(invocation.call_id, None)
            self._chunk_callbacks.pop(invocation.call_id, None)

        if response.error is not None:
            return ToolInvocationResult(
                content=response.error.message,
                is_error=True,
                error_kind=response.error.kind,
            )
        return ToolInvocationResult(content=response.result or "")

    async def cancel(self, call_id: str) -> None:
        if self._writer is None:
            return
        with contextlib.suppress(Exception):
            await self._send(ToolCallCancel(call_id=call_id))

    async def aclose(self) -> None:
        self._closed = True
        await self._teardown(reason="aclose")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def _ensure_running(self) -> None:
        if self._writer is not None:
            return
        async with self._lock:
            if self._writer is not None:
                return
            await self._start_subprocess_and_connect()

    async def _start_subprocess_and_connect(self) -> None:
        """Spawn the daemon and complete the handshake.

        Raises :class:`DaemonUnavailableError` on any startup failure,
        wrapping the underlying cause so callers get a clear message
        without having to introspect chained exceptions.
        """
        try:
            self._process = await self._spawn_subprocess()
            self._owns_process = True
            reader, writer = await self._connect_socket()
        except Exception as exc:
            await self._kill_process()
            raise DaemonUnavailableError(
                f"failed to start toolhost daemon: {exc}"
            ) from exc

        try:
            await self._adopt_streams(reader, writer)
        except Exception:
            await self._teardown(reason="handshake_failed")
            raise

    async def _spawn_subprocess(self) -> asyncio.subprocess.Process:
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

        return await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )

    async def _connect_socket(
        self,
    ) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
        """Wait for the daemon's socket to appear, then connect.

        We poll the filesystem because :func:`asyncio.open_unix_connection`
        does not block on a not-yet-bound path and would just raise
        ``FileNotFoundError`` on the first try.  The poll interval is
        coarser than necessary (10ms) to keep CPU use trivial; a real
        cold start tops out at a few hundred milliseconds.
        """
        deadline = time.monotonic() + self._config.connect_timeout_s
        last_exc: Exception | None = None
        while time.monotonic() < deadline:
            if not self._config.socket_path.exists():
                await asyncio.sleep(0.01)
                continue
            try:
                return await asyncio.open_unix_connection(
                    path=str(self._config.socket_path),
                )
            except (FileNotFoundError, ConnectionRefusedError) as exc:
                last_exc = exc
                await asyncio.sleep(0.01)
        raise DaemonUnavailableError(
            f"daemon socket {self._config.socket_path} did not become "
            f"reachable within {self._config.connect_timeout_s}s"
        ) from last_exc

    async def adopt_streams(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        """Adopt caller-provided streams instead of spawning a subprocess.

        Tests use this entry point to drive the executor over an
        in-memory pipe pair connected to a daemon running in the same
        event loop.  Production code should use :meth:`invoke`, which
        spawns the subprocess implicitly.
        """
        if self._writer is not None:
            raise RuntimeError("DaemonToolExecutor is already connected")
        async with self._lock:
            if self._writer is not None:
                raise RuntimeError("DaemonToolExecutor is already connected")
            await self._adopt_streams(reader, writer)

    async def _adopt_streams(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        self._reader = reader
        self._writer = writer
        self._last_frame_received_at = time.monotonic()

        await write_frame(
            writer,
            Hello(
                protocol_major=PROTOCOL_MAJOR,
                protocol_minor=PROTOCOL_MINOR,
                thorn_version="phase-a-brain",
                features=[],
                per_agent_state={
                    "agent_id": self._config.agent_id,
                    "home": (
                        str(self._config.home_path)
                        if self._config.home_path
                        else None
                    ),
                    "workspace_root": (
                        str(self._config.workspace_root)
                        if self._config.workspace_root
                        else None
                    ),
                },
            ),
        )

        try:
            response = await asyncio.wait_for(
                read_frame(reader),
                timeout=self._config.handshake_timeout_s,
            )
        except (asyncio.TimeoutError, asyncio.IncompleteReadError) as exc:
            raise DaemonUnavailableError(
                f"toolhost handshake timed out: {exc}"
            ) from exc
        except ProtocolError as exc:
            raise DaemonUnavailableError(
                f"toolhost sent invalid handshake: {exc}"
            ) from exc

        if not isinstance(response, Hello):
            raise DaemonUnavailableError(
                f"expected Hello from daemon, got {type(response).__name__}",
            )
        if response.protocol_major != PROTOCOL_MAJOR:
            raise DaemonUnavailableError(
                f"daemon protocol major {response.protocol_major} "
                f"incompatible with brain {PROTOCOL_MAJOR}",
            )

        self._daemon_hello = response
        self._reader_task = asyncio.create_task(
            self._reader_loop(),
            name=f"daemon-reader:{self._config.agent_id}",
        )
        self._heartbeat_task = asyncio.create_task(
            self._heartbeat_loop(),
            name=f"daemon-heartbeat:{self._config.agent_id}",
        )

    # ------------------------------------------------------------------
    # I/O loops
    # ------------------------------------------------------------------

    async def _reader_loop(self) -> None:
        assert self._reader is not None
        try:
            while True:
                frame = await read_frame(self._reader)
                self._last_frame_received_at = time.monotonic()
                await self._handle_frame(frame)
        except asyncio.IncompleteReadError:
            await self._fail_all(
                DaemonCrashedError("daemon connection closed unexpectedly"),
                reason="connection_closed",
            )
        except ProtocolError as exc:
            await self._fail_all(
                DaemonCrashedError(f"daemon sent malformed frame: {exc}"),
                reason="protocol_error",
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("daemon reader loop crashed")
            await self._fail_all(
                DaemonCrashedError(f"reader loop error: {exc}"),
                reason="reader_error",
            )

    async def _heartbeat_loop(self) -> None:
        try:
            while True:
                await asyncio.sleep(self._config.heartbeat_interval_s)
                if self._writer is None:
                    return
                if (
                    time.monotonic() - self._last_frame_received_at
                    > self._config.heartbeat_dead_s
                ):
                    await self._fail_all(
                        DaemonCrashedError(
                            "daemon heartbeat timeout: "
                            f"no frame received in "
                            f"{self._config.heartbeat_dead_s}s"
                        ),
                        reason="heartbeat_timeout",
                    )
                    return
                with contextlib.suppress(Exception):
                    await self._send(Heartbeat())
        except asyncio.CancelledError:
            raise

    async def _handle_frame(self, frame: Frame) -> None:
        if isinstance(frame, ToolCallResponse):
            future = self._in_flight.get(frame.call_id)
            if future is None or future.done():
                logger.debug(
                    "daemon: ignoring response for unknown call_id %r",
                    frame.call_id,
                )
                return
            future.set_result(frame)
        elif isinstance(frame, ToolCallChunk):
            callback = self._chunk_callbacks.get(frame.call_id)
            if callback is not None:
                from thorn.core._executor import ToolOutputChunk

                with contextlib.suppress(Exception):
                    await callback(
                        ToolOutputChunk(
                            call_id=frame.call_id,
                            stream=frame.stream,
                            data=frame.data,
                        )
                    )
        elif isinstance(frame, Heartbeat):
            return
        elif isinstance(frame, Hello):
            logger.warning("daemon: unexpected late Hello; ignoring")
        else:
            logger.warning(
                "daemon: ignoring unexpected frame %s",
                type(frame).__name__,
            )

    async def _send(self, frame: Frame) -> None:
        writer = self._writer
        if writer is None:
            raise DaemonCrashedError("daemon connection is not open")
        async with self._send_lock:
            await write_frame(writer, frame)

    # ------------------------------------------------------------------
    # Failure handling
    # ------------------------------------------------------------------

    async def _fail_all(self, exc: Exception, *, reason: str) -> None:
        """Fail every in-flight call with *exc* and reset the connection.

        Called when the reader loop notices the daemon died or sent
        garbage.  The futures get the exception so awaiting callers
        unblock immediately; the next :meth:`invoke` triggers a fresh
        subprocess start when ``auto_restart`` is enabled.
        """
        if self._closed:
            return
        for call_id, future in list(self._in_flight.items()):
            if not future.done():
                future.set_exception(exc)
            self._in_flight.pop(call_id, None)
            self._chunk_callbacks.pop(call_id, None)
        await self._teardown(reason=reason)

    async def _teardown(self, *, reason: str) -> None:
        if self._heartbeat_task is not None:
            self._heartbeat_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._heartbeat_task
            self._heartbeat_task = None

        if self._reader_task is not None and asyncio.current_task() is not self._reader_task:
            self._reader_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._reader_task
        self._reader_task = None

        if self._writer is not None:
            with contextlib.suppress(Exception):
                self._writer.close()
            with contextlib.suppress(Exception):
                await self._writer.wait_closed()
        self._writer = None
        self._reader = None
        self._daemon_hello = None
        self._last_frame_received_at = 0.0

        if self._owns_process:
            await self._kill_process()

        if not self._config.auto_restart:
            self._closed = True

    async def _kill_process(self) -> None:
        proc = self._process
        if proc is None:
            return
        if proc.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                proc.terminate()
            try:
                await asyncio.wait_for(proc.wait(), timeout=2.0)
            except asyncio.TimeoutError:
                with contextlib.suppress(ProcessLookupError):
                    proc.kill()
                with contextlib.suppress(Exception):
                    await proc.wait()
        with contextlib.suppress(FileNotFoundError):
            os.unlink(self._config.socket_path)
        self._process = None
        self._owns_process = False


# Re-exported for convenience.  Aliased so the symbol can be imported
# from the top-level :mod:`thorn.toolhost` package without forcing
# callers to reach into ``_executor``.
__all__ = [
    "DaemonExecutorConfig",
    "DaemonToolExecutor",
    "DaemonUnavailableError",
    "DaemonCrashedError",
]
