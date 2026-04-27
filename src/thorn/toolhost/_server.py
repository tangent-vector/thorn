"""``thorn-toolhost`` daemon: serve sandbox-venue tool calls over a socket.

The daemon is a tiny Python process that:

1. Binds the per-agent Unix-domain socket given on argv.
2. Accepts a single brain connection, performs the
   :class:`~thorn.toolhost._protocol.Hello` handshake, and refuses on
   protocol-major mismatch.
3. Reads :class:`ToolCallRequest` frames from the socket, dispatches
   each to a bounded set of ``asyncio`` tasks, and replies with a
   matching :class:`ToolCallResponse`.
4. Honors :class:`ToolCallCancel` (cancels the in-flight task; the
   tool's own subprocess plumbing is responsible for propagation).
5. Echoes :class:`Heartbeat` frames so the brain has a positive
   liveness signal.

Phase A keeps everything modest: the daemon ships in the brain's venv,
its tool registry is a static list of known built-ins (everything
except the in-process allow-list), and there is no MCP / discovery
layer.  Later phases swap the executor target for an OCI container and
add MCP server lifecycles inside the daemon itself.
"""

from __future__ import annotations

import asyncio
import logging
import os
import platform
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from thorn.core._context import ExecutionContext, NullEventSink, set_context, reset_context
from thorn.core._executor import (
    InProcessToolExecutor,
    ToolInvocation,
    ToolRegistry,
    ToolRegistryEntry,
    ToolVenue,
)
from thorn.core._loop import _WrappedTool
from thorn.toolhost._protocol import (
    PROTOCOL_MAJOR,
    PROTOCOL_MINOR,
    Frame,
    Heartbeat,
    Hello,
    ProtocolError,
    ToolCallCancel,
    ToolCallError,
    ToolCallRequest,
    ToolCallResponse,
    read_frame,
    write_frame,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ToolhostConfig:
    """Configuration for a running ``thorn-toolhost`` instance.

    The daemon is per-agent: ``socket_path`` and ``agent_id`` together
    identify the rendezvous, and the optional ``home_path`` /
    ``workspace_root`` set the agent's filesystem view (``~`` resolves
    to ``home_path``; relative tool paths resolve under
    ``workspace_root``).  Both can be left ``None`` in tests; tools
    that depend on them will return their usual "no agent home / no
    workspace" errors instead of crashing.
    """

    socket_path: Path
    agent_id: str
    home_path: Path | None = None
    workspace_root: Path | None = None
    log_path: Path | None = None
    max_concurrency: int = 8
    debug: bool = False
    thorn_version: str = "phase-a"
    features: tuple[str, ...] = ()


class _StubAgent:
    """Minimal stand-in for the brain-side ``Agent`` object.

    The daemon never owns a real :class:`~thorn.core._agent.Agent`; the
    only attributes any tool actually reaches for via ``ctx.agent`` are
    ``id``, ``home``, and ``workspace``.  Keeping the surface tiny
    means the daemon does not have to instantiate (or even import) the
    full agent machinery.
    """

    __slots__ = ("id", "home", "workspace")

    def __init__(
        self,
        agent_id: str,
        home: Path | None,
        workspace: Path | None,
    ) -> None:
        self.id = agent_id
        self.home = home
        self.workspace = workspace


class _NullProvider:
    """LLM provider stub.

    Tools never trigger completions (the brain owns prompting), but
    :class:`ExecutionContext` requires a non-``None`` ``provider``
    field.  Calling ``complete`` is a programming error in the daemon
    and is treated as such with an explicit ``RuntimeError`` rather
    than a silent ``None``.
    """

    @property
    def context_window(self) -> int | None:
        return None

    def complete(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover
        raise RuntimeError("toolhost daemon must not invoke an LLM provider")


def build_default_registry() -> tuple[ToolRegistry, dict[str, _WrappedTool]]:
    """Build the static daemon tool registry plus its execute table.

    Loads every built-in tool that is *not* on the in-process
    allow-list (currently empty -- the inbox tools live elsewhere).
    Returns the registry used to schema-validate incoming requests
    and the parallel dict of executable callables consumed by
    :class:`InProcessToolExecutor`.
    """
    from thorn.core._func import wrap_function
    from thorn.core._journal import JOURNAL_TOOLS
    from thorn.core._tools import ALL_BUILTIN_TOOLS, run_shell

    in_process_allowlist: set[str] = set()

    callables: list[Callable[..., Any]] = []
    for fn in ALL_BUILTIN_TOOLS:
        if getattr(fn, "__name__", "") in in_process_allowlist:
            continue
        callables.append(fn)
    callables.append(run_shell)
    callables.extend(JOURNAL_TOOLS)

    entries: list[ToolRegistryEntry] = []
    table: dict[str, _WrappedTool] = {}
    for fn in callables:
        wrapped = wrap_function(fn)
        wrapped.venue = ToolVenue.SANDBOX
        name = wrapped.schema.get("function", {}).get("name", "")
        if not name:
            continue
        if name in table:
            logger.warning("toolhost: ignoring duplicate tool %r", name)
            continue
        table[name] = wrapped
        entries.append(
            ToolRegistryEntry(
                name=name,
                schema=wrapped.schema,
                venue=ToolVenue.SANDBOX,
                call_node_class=wrapped.call_node_class,
            )
        )
    return ToolRegistry(entries), table


@dataclass
class _Connection:
    """Per-connection mutable state held by :class:`ToolhostServer`.

    The write lock serializes writes from multiple concurrent task
    coroutines into the single underlying stream.  ``in_flight`` lets
    :class:`ToolCallCancel` find and cancel a running task, and is
    cleaned up by the task itself in its ``finally``.
    """

    reader: asyncio.StreamReader
    writer: asyncio.StreamWriter
    write_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    in_flight: dict[str, asyncio.Task] = field(default_factory=dict)


class ToolhostServer:
    """Daemon-side tool host.

    A single instance owns the listener, the static tool registry, and
    the per-call dispatch state.  ``serve_forever`` binds the socket
    and accepts exactly one connection (extras are immediately closed
    with a log line); when that connection drops, ``serve_forever``
    returns and the process can exit.
    """

    def __init__(
        self,
        config: ToolhostConfig,
        registry: ToolRegistry | None = None,
        executor: InProcessToolExecutor | None = None,
    ) -> None:
        if registry is None or executor is None:
            built_registry, table = build_default_registry()
            self._registry = registry or built_registry
            self._executor = executor or InProcessToolExecutor(table)
        else:
            self._registry = registry
            self._executor = executor

        self._config = config
        self._semaphore = asyncio.Semaphore(config.max_concurrency)
        self._base_agent = _StubAgent(
            agent_id=config.agent_id,
            home=config.home_path,
            workspace=config.workspace_root,
        )
        self._provider = _NullProvider()
        self._connection_count = 0

    @property
    def config(self) -> ToolhostConfig:
        return self._config

    @property
    def registry(self) -> ToolRegistry:
        return self._registry

    async def serve_forever(self) -> None:
        """Bind the socket, handle one connection, then return."""
        if platform.system() == "Windows":
            raise RuntimeError(
                "thorn-toolhost requires a Unix-like host (got Windows); "
                "use WSL or a Linux container instead.",
            )

        socket_path = self._config.socket_path
        socket_path.parent.mkdir(parents=True, exist_ok=True)
        if socket_path.exists():
            try:
                socket_path.unlink()
            except OSError as exc:
                raise RuntimeError(
                    f"failed to clear stale socket {socket_path}: {exc}",
                ) from exc

        server = await asyncio.start_unix_server(
            self._on_connect,
            path=str(socket_path),
        )

        async with server:
            logger.info("toolhost listening on %s", socket_path)
            try:
                await server.serve_forever()
            except asyncio.CancelledError:
                pass
            finally:
                with suppress(FileNotFoundError):
                    socket_path.unlink()

    async def handle_streams(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        """Run the connection loop on caller-provided streams.

        Exposed so unit tests can drive the daemon over an in-memory
        stream pair without binding a real socket.  Returns when the
        peer closes the connection or when an unrecoverable protocol
        error fires.
        """
        await self._on_connect(reader, writer)

    async def _on_connect(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        self._connection_count += 1
        if self._connection_count > 1:
            logger.warning(
                "toolhost: rejecting concurrent connection #%d",
                self._connection_count,
            )
            writer.close()
            with suppress(Exception):
                await writer.wait_closed()
            return

        connection = _Connection(reader=reader, writer=writer)
        try:
            handshake_ok = await self._perform_handshake(connection)
            if not handshake_ok:
                return
            await self._serve_connection(connection)
        finally:
            for task in list(connection.in_flight.values()):
                task.cancel()
            for task in list(connection.in_flight.values()):
                with suppress(asyncio.CancelledError, Exception):
                    await task
            connection.in_flight.clear()
            with suppress(Exception):
                writer.close()
            with suppress(Exception):
                await writer.wait_closed()

    async def _perform_handshake(self, connection: _Connection) -> bool:
        """Read the brain's :class:`Hello`, send our own, gate on major."""
        try:
            frame = await read_frame(connection.reader)
        except (asyncio.IncompleteReadError, ProtocolError) as exc:
            logger.warning("toolhost: handshake failed: %s", exc)
            return False

        if not isinstance(frame, Hello):
            logger.warning(
                "toolhost: expected Hello as first frame, got %s",
                type(frame).__name__,
            )
            return False

        if frame.protocol_major != PROTOCOL_MAJOR:
            logger.warning(
                "toolhost: protocol major mismatch (brain=%d, daemon=%d); "
                "disconnecting",
                frame.protocol_major,
                PROTOCOL_MAJOR,
            )
            return False

        if frame.protocol_minor != PROTOCOL_MINOR:
            logger.info(
                "toolhost: protocol minor mismatch (brain=%d, daemon=%d); "
                "proceeding",
                frame.protocol_minor,
                PROTOCOL_MINOR,
            )

        await self._send(
            connection,
            Hello(
                protocol_major=PROTOCOL_MAJOR,
                protocol_minor=PROTOCOL_MINOR,
                thorn_version=self._config.thorn_version,
                features=list(self._config.features),
                per_agent_state={
                    "agent_id": self._config.agent_id,
                    "home": str(self._config.home_path) if self._config.home_path else None,
                    "workspace_root": (
                        str(self._config.workspace_root)
                        if self._config.workspace_root
                        else None
                    ),
                },
            ),
        )
        return True

    async def _serve_connection(self, connection: _Connection) -> None:
        """Main per-connection request loop.

        Reads frames sequentially.  ``ToolCallRequest`` frames spawn a
        bounded task; everything else is handled inline so the loop
        stays simple and protocol violations terminate the connection
        promptly.
        """
        while True:
            try:
                frame = await read_frame(connection.reader)
            except asyncio.IncompleteReadError:
                logger.info("toolhost: peer closed connection")
                return
            except ProtocolError as exc:
                logger.warning("toolhost: protocol error: %s; closing", exc)
                return

            if isinstance(frame, ToolCallRequest):
                self._spawn_request(connection, frame)
            elif isinstance(frame, ToolCallCancel):
                self._handle_cancel(connection, frame)
            elif isinstance(frame, Heartbeat):
                await self._send(connection, Heartbeat())
            elif isinstance(frame, Hello):
                logger.warning("toolhost: ignoring unexpected late Hello")
            else:
                logger.warning(
                    "toolhost: unexpected frame %s; closing connection",
                    type(frame).__name__,
                )
                return

    def _spawn_request(
        self,
        connection: _Connection,
        request: ToolCallRequest,
    ) -> None:
        """Launch the per-request task and record it for cancel routing."""
        if request.call_id in connection.in_flight:
            logger.warning(
                "toolhost: duplicate call_id %r; dropping new request",
                request.call_id,
            )
            asyncio.create_task(
                self._send(
                    connection,
                    ToolCallResponse(
                        call_id=request.call_id,
                        error=ToolCallError(
                            kind="duplicate_call_id",
                            message=f"call_id {request.call_id!r} already in flight",
                        ),
                    ),
                )
            )
            return

        task = asyncio.create_task(
            self._run_request(connection, request),
            name=f"toolhost:{request.call_id}",
        )
        connection.in_flight[request.call_id] = task
        task.add_done_callback(
            lambda _t, cid=request.call_id: connection.in_flight.pop(cid, None)
        )

    def _handle_cancel(
        self,
        connection: _Connection,
        cancel: ToolCallCancel,
    ) -> None:
        task = connection.in_flight.get(cancel.call_id)
        if task is None:
            logger.debug(
                "toolhost: cancel for unknown call_id %r; dropping",
                cancel.call_id,
            )
            return
        task.cancel()

    async def _run_request(
        self,
        connection: _Connection,
        request: ToolCallRequest,
    ) -> None:
        """Execute one tool call under the per-call ExecutionContext."""
        async with self._semaphore:
            response = await self._dispatch(request)
            await self._send(connection, response)

    async def _dispatch(self, request: ToolCallRequest) -> ToolCallResponse:
        """Look up *request* in the registry and run it via the executor."""
        entry = self._registry.get(request.tool_name)
        if entry is None:
            return ToolCallResponse(
                call_id=request.call_id,
                error=ToolCallError(
                    kind="unknown_tool",
                    message=f"Unknown tool: {request.tool_name!r}",
                ),
            )

        context = self._build_context(request)
        token = set_context(context)
        try:
            invocation = ToolInvocation(
                call_id=request.call_id,
                tool_name=entry.name,
                arguments=dict(request.arguments),
            )
            try:
                result = await self._executor.invoke(invocation)
            except asyncio.CancelledError:
                logger.info(
                    "toolhost: call %r cancelled mid-flight",
                    request.call_id,
                )
                return ToolCallResponse(
                    call_id=request.call_id,
                    error=ToolCallError(
                        kind="cancelled",
                        message="tool call was cancelled",
                    ),
                )
            except Exception as exc:
                logger.exception(
                    "toolhost: tool %r raised", request.tool_name,
                )
                message = (
                    f"Error: {exc!r}" if self._config.debug else f"Error: {exc}"
                )
                return ToolCallResponse(
                    call_id=request.call_id,
                    error=ToolCallError(
                        kind="execution_error",
                        message=message,
                    ),
                )
        finally:
            reset_context(token)

        if result.is_error:
            return ToolCallResponse(
                call_id=request.call_id,
                error=ToolCallError(
                    kind=result.error_kind or "execution_error",
                    message=result.content,
                ),
            )
        return ToolCallResponse(
            call_id=request.call_id,
            result=result.content,
        )

    def _build_context(self, request: ToolCallRequest) -> ExecutionContext:
        """Construct the per-call :class:`ExecutionContext`.

        ``per_call_context`` may carry a ``workspace_subdir`` (relative
        to the agent's workspace_root) that becomes the effective
        ``ctx.workspace_root`` for this call; ``mkdir -p`` is applied
        on first reference so tools can chdir / write into a never-
        before-seen session directory.
        """
        per_call = request.per_call_context
        workspace = self._config.workspace_root
        subdir = per_call.get("workspace_subdir") if per_call else None
        if workspace is not None and subdir:
            workspace = workspace / subdir
            workspace.mkdir(parents=True, exist_ok=True)

        scope_metadata = (
            dict(per_call.get("scope_metadata") or {}) if per_call else {}
        )
        session_key = per_call.get("session_key") if per_call else None
        if session_key is not None:
            scope_metadata.setdefault("session_key", session_key)

        ctx = ExecutionContext(
            provider=self._provider,
            event_sink=NullEventSink(),
            workspace_root=workspace,
            agent=self._base_agent,
        )
        if scope_metadata or session_key is not None:
            ctx = ctx.push_scope(
                f"toolcall:{request.call_id}",
                **scope_metadata,
            )
        return ctx

    async def _send(
        self,
        connection: _Connection,
        frame: Frame,
    ) -> None:
        async with connection.write_lock:
            try:
                await write_frame(connection.writer, frame)
            except (BrokenPipeError, ConnectionResetError) as exc:
                logger.info("toolhost: write failed (peer gone): %s", exc)


def configure_logging(log_path: Path | None, *, debug: bool) -> None:
    """Set up the daemon's standalone logger.

    Logs go to ``log_path`` (typically ``control/toolhost.log``) plus
    stderr.  The root logger is reconfigured here because the daemon
    runs as its own subprocess; the brain's logging setup does not
    reach into it.
    """
    handlers: list[logging.Handler] = []
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path, encoding="utf-8"))
    handlers.append(logging.StreamHandler())

    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        handlers=handlers,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        force=True,
    )


def run(config: ToolhostConfig) -> None:
    """Synchronous entry point used by ``__main__``.

    Wraps :meth:`ToolhostServer.serve_forever` in :func:`asyncio.run`
    and ensures the socket file is removed on exit so a stale socket
    never blocks a future restart.
    """
    configure_logging(config.log_path, debug=config.debug)

    async def _main() -> None:
        server = ToolhostServer(config)
        await server.serve_forever()

    try:
        asyncio.run(_main())
    finally:
        with suppress(FileNotFoundError):
            os.unlink(config.socket_path)
