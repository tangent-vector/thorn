"""Brain-side :class:`ToolExecutor` that talks to a ``thorn-toolhost`` daemon.

The agent loop never sees the daemon directly; it only sees a
:class:`ToolExecutor`.  :class:`DaemonToolExecutor` adapts the protocol
defined in :mod:`thorn.toolhost._protocol` to that interface:

* :meth:`invoke` ships a :class:`ToolCallRequest` over the connection
  and ``await``s the matching :class:`ToolCallResponse` keyed by
  ``call_id``.
* :meth:`cancel` ships a :class:`ToolCallCancel` and lets the daemon
  decide what cancellation actually means for the in-flight call.
* :meth:`aclose` shuts the connection (and the daemon, if we started
  one) cleanly.

The implementation deliberately splits *daemon hosting* from *stream
multiplexing* so unit tests can drive the executor over an in-memory
stream pair without paying for a real ``fork``, and so that Phase B
can swap a container-host implementation in without touching the
protocol layer.

Hosting concerns -- "what brings the daemon process into existence?
where does the socket live? when is the host considered ready?" --
live behind the :class:`~thorn.toolhost._host.DaemonHost` protocol.
The Phase-A path uses the in-package
:class:`~thorn.toolhost._host.SubprocessDaemonHost`; Phase B's
container path lives at :class:`thorn.sandbox.ContainerDaemonHost`.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
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
from thorn.toolhost._host import (
    DaemonHost,
    SubprocessDaemonHost,
    SubprocessDaemonHostConfig,
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

    Either the host failed to start, the socket never bound, the
    handshake failed, or the daemon crashed before serving a frame.
    Wraps the underlying error so callers do not have to introspect
    ``__cause__`` to render a useful message.
    """


class DaemonCrashedError(RuntimeError):
    """Raised inside :meth:`DaemonToolExecutor.invoke` when the daemon dies.

    Distinguishes a daemon-side crash (transient, retryable: the next
    invocation will spin up a fresh host) from a tool-level failure
    (returned in the response payload).
    """


@dataclass
class DaemonExecutorConfig:
    """Configuration for the brain-side daemon executor.

    Carries two distinct concerns:

    * *Connection-level* knobs the executor uses directly: socket
      path, agent identity, timeouts, heartbeat policy, auto-restart
      flag.
    * *Subprocess-host fallback* knobs (``home_path``,
      ``workspace_root``, ``log_path``, ``python_executable``,
      ``max_concurrency``, ``extra_args``) used when no explicit
      :class:`~thorn.toolhost._host.DaemonHost` is supplied to
      :class:`DaemonToolExecutor`.  In that case the executor builds
      a :class:`~thorn.toolhost._host.SubprocessDaemonHost` from these
      fields, preserving the Phase-A behavior verbatim.

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

    def to_subprocess_host_config(self) -> SubprocessDaemonHostConfig:
        """Return a :class:`SubprocessDaemonHostConfig` mirroring this config.

        Used when no explicit ``host=`` is passed to the executor; lets
        the Phase-A subprocess shape stay a one-line construction
        instead of leaking the host's keyword arguments back into
        the executor's call sites.
        """
        return SubprocessDaemonHostConfig(
            socket_path=self.socket_path,
            agent_id=self.agent_id,
            home_path=self.home_path,
            workspace_root=self.workspace_root,
            log_path=self.log_path,
            python_executable=self.python_executable,
            max_concurrency=self.max_concurrency,
            extra_args=self.extra_args,
        )


class DaemonToolExecutor(ToolExecutor):
    """``ToolExecutor`` backed by a ``thorn-toolhost`` daemon.

    Lifecycle::

        executor = DaemonToolExecutor(config)        # subprocess host (Phase A default)
        # or
        executor = DaemonToolExecutor(config, host=ContainerDaemonHost(...))

        result = await executor.invoke(invocation)   # lazily starts the host on first call
        await executor.aclose()                      # shuts everything down

    The executor is safe to share across coroutines.  Send frames are
    serialized by an ``asyncio.Lock``; per-call replies are matched to
    their callers via futures keyed by ``call_id``.  The daemon's death
    is detected either by the reader task encountering an
    :class:`asyncio.IncompleteReadError` or by the heartbeat watchdog
    failing to receive an echo within ``heartbeat_dead_s`` seconds.
    """

    def __init__(
        self,
        config: DaemonExecutorConfig,
        *,
        host: DaemonHost | None = None,
    ) -> None:
        self._config = config
        if host is None:
            host = SubprocessDaemonHost(config.to_subprocess_host_config())
        self._host = host
        # The host owns the socket path of record.  Validate that the
        # executor's connection-level config agrees, so a mismatched
        # injection (e.g. tests handing in a host whose socket doesn't
        # match the config) is a loud programming error rather than a
        # silent connect-to-the-wrong-place bug.
        if Path(host.socket_path) != Path(config.socket_path):
            raise ValueError(
                f"DaemonHost.socket_path ({host.socket_path}) does not "
                f"match config.socket_path ({config.socket_path})",
            )

        self._lock = asyncio.Lock()
        self._writer: asyncio.StreamWriter | None = None
        self._reader: asyncio.StreamReader | None = None
        self._reader_task: asyncio.Task | None = None
        self._heartbeat_task: asyncio.Task | None = None
        self._host_started: bool = False
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
    def host(self) -> DaemonHost:
        """The :class:`DaemonHost` driving this executor's daemon."""
        return self._host

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

    async def start(self) -> None:
        """Eagerly start the host and complete the handshake.

        Equivalent to issuing a no-op :meth:`invoke` to force the lazy
        path, but without sending a tool-call frame.  Used by the
        gateway during agent pre-load (Phase B) so containers come up
        before the first incoming event hits the agent.

        Idempotent: calling ``start`` on an already-running executor
        is a no-op.  Raises :class:`DaemonUnavailableError` on any
        startup failure.
        """
        if self._closed:
            raise RuntimeError("DaemonToolExecutor is closed")
        await self._ensure_running()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def _ensure_running(self) -> None:
        if self._writer is not None:
            return
        async with self._lock:
            if self._writer is not None:
                return
            await self._start_host_and_connect()

    async def _start_host_and_connect(self) -> None:
        """Start the host and complete the handshake.

        Raises :class:`DaemonUnavailableError` on any startup failure,
        wrapping the underlying cause so callers get a clear message
        without having to introspect chained exceptions.
        """
        try:
            await self._host.start()
            self._host_started = True
            reader, writer = await self._connect_socket()
        except DaemonUnavailableError:
            # Already a host-aware diagnostic; tear the host down and
            # let the caller see the original error verbatim.
            await self._stop_host_quietly()
            raise
        except Exception as exc:
            await self._stop_host_quietly()
            raise DaemonUnavailableError(
                f"failed to start toolhost daemon: {exc}"
            ) from exc

        try:
            await self._adopt_streams(reader, writer)
        except Exception:
            await self._teardown(reason="handshake_failed")
            raise

    async def _connect_socket(
        self,
    ) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
        """Wait for the daemon's socket to appear, then connect.

        We poll the filesystem because :func:`asyncio.open_unix_connection`
        does not block on a not-yet-bound path and would just raise
        ``FileNotFoundError`` on the first try.  The poll interval is
        coarser than necessary (10ms) to keep CPU use trivial; a real
        cold start tops out at a few hundred milliseconds.

        This is the *protocol-level* readiness probe, common across
        host kinds.  Container hosts run their own *host-level*
        readiness probe inside ``host.start()`` (e.g. waiting for
        ``inspect`` to report ``running``); that means cold container
        start latency is paid in ``host.start()``, not here, and the
        executor's connect timeout can stay tight even for containers.
        """
        socket_path = self._host.socket_path
        deadline = time.monotonic() + self._config.connect_timeout_s
        last_exc: Exception | None = None
        while time.monotonic() < deadline:
            if not socket_path.exists():
                await asyncio.sleep(0.01)
                continue
            try:
                return await asyncio.open_unix_connection(path=str(socket_path))
            except (FileNotFoundError, ConnectionRefusedError) as exc:
                last_exc = exc
                await asyncio.sleep(0.01)
        raise DaemonUnavailableError(
            f"daemon socket {socket_path} did not become "
            f"reachable within {self._config.connect_timeout_s}s"
        ) from last_exc

    async def adopt_streams(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        """Adopt caller-provided streams instead of starting the host.

        Tests use this entry point to drive the executor over an
        in-memory pipe pair connected to a daemon running in the same
        event loop.  Production code should use :meth:`invoke`, which
        starts the host implicitly.
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
        host start when ``auto_restart`` is enabled.
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

        if self._host_started:
            await self._stop_host_quietly()

        if not self._config.auto_restart:
            self._closed = True

    async def _stop_host_quietly(self) -> None:
        """Tear the host down, swallowing exceptions for graceful close."""
        try:
            await self._host.stop()
        except Exception:
            logger.exception(
                "DaemonHost.stop raised during teardown; ignoring",
            )
        self._host_started = False


__all__ = [
    "DaemonExecutorConfig",
    "DaemonToolExecutor",
    "DaemonUnavailableError",
    "DaemonCrashedError",
]
