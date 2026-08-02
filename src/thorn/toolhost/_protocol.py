"""Framed JSON wire protocol between the brain and ``thorn-toolhost``.

The transport is a single duplex connection (a Unix-domain socket in
production, an ``asyncio`` stream pair in tests).  Each frame is a
4-byte big-endian unsigned length prefix followed by exactly that many
bytes of UTF-8 JSON.  Every JSON object carries a ``"kind"`` field
identifying its frame type; the remaining keys are the type's payload.

Versioning rules:

* ``PROTOCOL_MAJOR`` mismatches between sides are unrecoverable and the
  receiver should disconnect with a diagnostic.
* ``PROTOCOL_MINOR`` differences are logged but tolerated; older minor
  versions ignore unknown frame fields rather than raising.
* New frame kinds are only ever added, never removed; receivers raise
  :class:`ProtocolError` when given an unknown kind so that protocol
  drift is loud rather than silent.

This module deliberately avoids any I/O concept beyond the abstract
``StreamReader`` / ``StreamWriter`` pair from :mod:`asyncio.streams` so
that the wire format can be exercised under fully in-memory tests.
"""

from __future__ import annotations

import asyncio
import base64
import json
import struct
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Union

from thorn.core._mcp_config import MCPServerConfig

PROTOCOL_MAJOR: int = 1
PROTOCOL_MINOR: int = 1

# Feature flag advertised in :class:`Hello.features` by both sides when
# they understand the MCP frames (``ListMCPServerToolsRequest`` /
# ``ListMCPServerToolsResponse``) and the optional ``mcp_server_config``
# field on :class:`ToolCallRequest`.  The brain advertises it
# unconditionally; the daemon advertises it only when the ``mcp``
# Python package is importable.  Brains that wish to issue MCP traffic
# must verify the daemon advertises it before using those frames.
MCP_FEATURE: str = "mcp"

LENGTH_PREFIX_FORMAT: str = "!I"
LENGTH_PREFIX_SIZE: int = struct.calcsize(LENGTH_PREFIX_FORMAT)
MAX_FRAME_SIZE: int = 64 * 1024 * 1024


class ProtocolError(Exception):
    """Raised for any framing or schema violation on the wire.

    Callers are expected to treat this as fatal for the connection (the
    far side either crashed mid-frame, sent garbage, or is speaking an
    incompatible major version).
    """


class FrameKind(str, Enum):
    """Kind tag for every frame on the wire."""

    HELLO = "hello"
    TOOL_CALL_REQUEST = "tool_call_request"
    TOOL_CALL_RESPONSE = "tool_call_response"
    TOOL_CALL_CHUNK = "tool_call_chunk"
    TOOL_CALL_CANCEL = "tool_call_cancel"
    HEARTBEAT = "heartbeat"
    LIST_MCP_SERVER_TOOLS_REQUEST = "list_mcp_server_tools_request"
    LIST_MCP_SERVER_TOOLS_RESPONSE = "list_mcp_server_tools_response"


@dataclass(frozen=True)
class Hello:
    """First frame in each direction after a connection is established.

    ``per_agent_state`` carries any stable, per-agent configuration the
    daemon needs for every call (home path, workspace root, optional
    serialized ``FileAccessPolicy`` data, etc.).  ``features`` is a
    forward-compatibility surface for opt-in extensions
    (``"streaming"``, ``"cancel"``, ``"binary_chunks"``, ...).
    """

    protocol_major: int
    protocol_minor: int
    thorn_version: str
    features: list[str] = field(default_factory=list)
    per_agent_state: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ToolCallError:
    """Structured error payload carried inside a :class:`ToolCallResponse`.

    ``kind`` is a stable classifier used by the brain to decide whether
    a failure is retryable; ``message`` is the human-readable text
    surfaced to the model.
    """

    kind: str
    message: str


@dataclass(frozen=True)
class ToolCallRequest:
    """Brain -> daemon: please execute a tool call.

    ``per_call_context`` carries only what genuinely varies per call
    (session key, workspace subdirectory, scope metadata) -- the stable
    per-agent state lives in the connection-level :class:`Hello`.

    ``mcp_server_config`` routes the call through the daemon's
    :class:`~thorn.toolhost._mcp_host.MCPHost` instead of the static
    built-in registry.  The full server config is sent on every MCP
    call so the daemon does not have to track brain-side server names
    (the brain owns name resolution); the daemon caches connections
    keyed by config identity and serializes calls per-server.  Leave
    ``None`` for built-in tools.
    """

    call_id: str
    tool_name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    per_call_context: dict[str, Any] = field(default_factory=dict)
    mcp_server_config: MCPServerConfig | None = None


@dataclass(frozen=True)
class ToolCallResponse:
    """Daemon -> brain: terminal result for a ``call_id``.

    Exactly one of ``result`` or ``error`` is populated; this is
    enforced at construction time so the brain side does not have to
    defend against malformed combinations.
    """

    call_id: str
    result: str | None = None
    error: ToolCallError | None = None

    def __post_init__(self) -> None:
        if (self.result is None) == (self.error is None):
            raise ProtocolError(
                "ToolCallResponse must carry exactly one of result/error",
            )


@dataclass(frozen=True)
class ToolCallChunk:
    """Daemon -> brain: streaming output chunk for an in-flight call.

    ``stream`` is one of ``"stdout"``, ``"stderr"``, or ``"structured"``.
    Binary data is base64-encoded by callers; the protocol layer treats
    ``data`` as an opaque string.  Phase A reserves the frame but no
    built-in tool emits it yet.
    """

    call_id: str
    stream: str
    data: str

    @classmethod
    def from_bytes(cls, *, call_id: str, stream: str, data: bytes) -> ToolCallChunk:
        """Convenience constructor that base64-encodes raw bytes."""
        return cls(
            call_id=call_id,
            stream=stream,
            data=base64.b64encode(data).decode("ascii"),
        )


@dataclass(frozen=True)
class ToolCallCancel:
    """Brain -> daemon: cancel the in-flight call ``call_id``.

    Cancellation is idempotent at the daemon level -- a cancel for a
    completed or unknown ``call_id`` is silently dropped.  Brain-side
    bookkeeping is responsible for not double-cancelling.
    """

    call_id: str


@dataclass(frozen=True)
class Heartbeat:
    """Bidirectional liveness ping with no payload.

    The brain sends one periodically and treats a missing reply as
    evidence of a wedged daemon.  The daemon may also send heartbeats
    back to give the brain a positive signal that no frame has been
    lost on the wire.
    """


@dataclass(frozen=True)
class ListMCPServerToolsRequest:
    """Brain -> daemon: enumerate the tools served by *server_config*.

    The daemon spawns / connects to the matching MCP server on first
    reference and returns its tool list as OpenAI-style schema dicts
    (``{"type": "function", "function": {...}}``).  The full
    :class:`~thorn.core._mcp_config.MCPServerConfig` is shipped on every
    request so the daemon does not need any prior knowledge of brain-
    side server names; identity-keyed caching inside the daemon
    deduplicates concurrent requests for the same config.

    ``request_id`` is the brain-generated correlation key for the
    matching :class:`ListMCPServerToolsResponse`.
    """

    request_id: str
    server_config: MCPServerConfig


@dataclass(frozen=True)
class ListMCPServerToolsResponse:
    """Daemon -> brain: tool list (or error) for a prior list request.

    Exactly one of ``tools`` or ``error`` is populated; this is
    enforced at construction time.  On error the brain logs and omits
    the offending server's tools from this prompt; the next prompt
    re-attempts naturally because the per-prompt context-gathering
    pipeline does not cache across prompts.
    """

    request_id: str
    tools: list[dict[str, Any]] | None = None
    error: ToolCallError | None = None

    def __post_init__(self) -> None:
        if (self.tools is None) == (self.error is None):
            raise ProtocolError(
                "ListMCPServerToolsResponse must carry exactly one of "
                "tools/error",
            )


Frame = Union[
    Hello,
    ToolCallRequest,
    ToolCallResponse,
    ToolCallChunk,
    ToolCallCancel,
    Heartbeat,
    ListMCPServerToolsRequest,
    ListMCPServerToolsResponse,
]


_KIND_TO_CLASS: dict[FrameKind, type] = {
    FrameKind.HELLO: Hello,
    FrameKind.TOOL_CALL_REQUEST: ToolCallRequest,
    FrameKind.TOOL_CALL_RESPONSE: ToolCallResponse,
    FrameKind.TOOL_CALL_CHUNK: ToolCallChunk,
    FrameKind.TOOL_CALL_CANCEL: ToolCallCancel,
    FrameKind.HEARTBEAT: Heartbeat,
    FrameKind.LIST_MCP_SERVER_TOOLS_REQUEST: ListMCPServerToolsRequest,
    FrameKind.LIST_MCP_SERVER_TOOLS_RESPONSE: ListMCPServerToolsResponse,
}

_CLASS_TO_KIND: dict[type, FrameKind] = {cls: kind for kind, cls in _KIND_TO_CLASS.items()}


def _mcp_server_config_to_payload(config: MCPServerConfig) -> dict[str, Any]:
    """Encode *config* into a JSON-friendly object for the wire.

    Mirrors the dataclass fields one-to-one; ``env`` is preserved as a
    plain dict because callers (e.g. the daemon's identity hash and
    the brain's dedup key) sort it themselves where order matters.
    """
    return {
        "name": config.name,
        "command": config.command,
        "args": list(config.args),
        "env": dict(config.env) if config.env is not None else None,
        "url": config.url,
    }


def _payload_to_mcp_server_config(payload: Any) -> MCPServerConfig:
    """Decode an MCP server config payload into an :class:`MCPServerConfig`.

    Raises :class:`ProtocolError` for any structural problem (missing
    ``name``, non-object ``env``, etc.).  Field-level invariants
    enforced by ``MCPServerConfig.__post_init__`` (e.g. "must specify
    command or url") propagate as :class:`ProtocolError` too so the
    caller does not have to distinguish wire bugs from semantic ones.
    """
    if not isinstance(payload, dict):
        raise ProtocolError(
            f"mcp_server_config must be an object, got "
            f"{type(payload).__name__}",
        )
    try:
        name = str(payload["name"])
    except KeyError as exc:
        raise ProtocolError("mcp_server_config missing 'name'") from exc
    args_payload = payload.get("args") or []
    if not isinstance(args_payload, list):
        raise ProtocolError("mcp_server_config 'args' must be a list")
    env_payload = payload.get("env")
    if env_payload is not None and not isinstance(env_payload, dict):
        raise ProtocolError("mcp_server_config 'env' must be an object or null")
    try:
        return MCPServerConfig(
            name=name,
            command=payload.get("command"),
            args=[str(arg) for arg in args_payload],
            env={str(k): str(v) for k, v in env_payload.items()}
            if env_payload is not None
            else None,
            url=payload.get("url"),
        )
    except ValueError as exc:
        raise ProtocolError(f"invalid mcp_server_config: {exc}") from exc


def _frame_to_payload(frame: Frame) -> dict[str, Any]:
    """Convert *frame* to the JSON-encodable mapping placed on the wire."""
    kind = _CLASS_TO_KIND.get(type(frame))
    if kind is None:
        raise ProtocolError(f"object {frame!r} is not a known protocol frame")

    body: dict[str, Any] = {"kind": kind.value}

    if isinstance(frame, Heartbeat):
        return body

    if isinstance(frame, ToolCallResponse):
        body["call_id"] = frame.call_id
        if frame.error is not None:
            body["error"] = asdict(frame.error)
        else:
            body["result"] = frame.result
        return body

    if isinstance(frame, ToolCallRequest):
        body["call_id"] = frame.call_id
        body["tool_name"] = frame.tool_name
        body["arguments"] = dict(frame.arguments)
        body["per_call_context"] = dict(frame.per_call_context)
        if frame.mcp_server_config is not None:
            body["mcp_server_config"] = _mcp_server_config_to_payload(
                frame.mcp_server_config,
            )
        return body

    if isinstance(frame, ListMCPServerToolsRequest):
        body["request_id"] = frame.request_id
        body["server_config"] = _mcp_server_config_to_payload(frame.server_config)
        return body

    if isinstance(frame, ListMCPServerToolsResponse):
        body["request_id"] = frame.request_id
        if frame.error is not None:
            body["error"] = asdict(frame.error)
        else:
            body["tools"] = list(frame.tools or [])
        return body

    payload = asdict(frame)
    body.update(payload)
    return body


def _payload_to_frame(payload: dict[str, Any]) -> Frame:
    """Convert a decoded JSON object to the matching frame dataclass."""
    if not isinstance(payload, dict):
        raise ProtocolError(
            f"frame payload must be a JSON object, got {type(payload).__name__}",
        )

    kind_value = payload.get("kind")
    if not isinstance(kind_value, str):
        raise ProtocolError(f"frame is missing a string 'kind' field: {payload!r}")

    try:
        kind = FrameKind(kind_value)
    except ValueError as exc:
        raise ProtocolError(f"unknown frame kind: {kind_value!r}") from exc

    if kind is FrameKind.HEARTBEAT:
        return Heartbeat()

    if kind is FrameKind.HELLO:
        try:
            return Hello(
                protocol_major=int(payload["protocol_major"]),
                protocol_minor=int(payload["protocol_minor"]),
                thorn_version=str(payload["thorn_version"]),
                features=list(payload.get("features") or []),
                per_agent_state=dict(payload.get("per_agent_state") or {}),
            )
        except KeyError as exc:
            raise ProtocolError(f"hello frame missing field {exc.args[0]!r}") from exc

    if kind is FrameKind.TOOL_CALL_REQUEST:
        mcp_config_payload = payload.get("mcp_server_config")
        mcp_config = (
            _payload_to_mcp_server_config(mcp_config_payload)
            if mcp_config_payload is not None
            else None
        )
        try:
            return ToolCallRequest(
                call_id=str(payload["call_id"]),
                tool_name=str(payload["tool_name"]),
                arguments=dict(payload.get("arguments") or {}),
                per_call_context=dict(payload.get("per_call_context") or {}),
                mcp_server_config=mcp_config,
            )
        except KeyError as exc:
            raise ProtocolError(
                f"tool_call_request frame missing field {exc.args[0]!r}",
            ) from exc

    if kind is FrameKind.TOOL_CALL_RESPONSE:
        try:
            call_id = str(payload["call_id"])
        except KeyError as exc:
            raise ProtocolError(
                "tool_call_response frame missing 'call_id'",
            ) from exc

        error_payload = payload.get("error")
        if error_payload is not None:
            if not isinstance(error_payload, dict):
                raise ProtocolError("tool_call_response 'error' must be an object")
            try:
                error = ToolCallError(
                    kind=str(error_payload["kind"]),
                    message=str(error_payload["message"]),
                )
            except KeyError as exc:
                raise ProtocolError(
                    f"tool_call_response error missing field {exc.args[0]!r}",
                ) from exc
            return ToolCallResponse(call_id=call_id, error=error)

        if "result" not in payload:
            raise ProtocolError(
                "tool_call_response must carry either 'result' or 'error'",
            )
        return ToolCallResponse(call_id=call_id, result=str(payload["result"]))

    if kind is FrameKind.TOOL_CALL_CHUNK:
        try:
            return ToolCallChunk(
                call_id=str(payload["call_id"]),
                stream=str(payload["stream"]),
                data=str(payload["data"]),
            )
        except KeyError as exc:
            raise ProtocolError(
                f"tool_call_chunk frame missing field {exc.args[0]!r}",
            ) from exc

    if kind is FrameKind.TOOL_CALL_CANCEL:
        try:
            return ToolCallCancel(call_id=str(payload["call_id"]))
        except KeyError as exc:
            raise ProtocolError(
                f"tool_call_cancel frame missing field {exc.args[0]!r}",
            ) from exc

    if kind is FrameKind.LIST_MCP_SERVER_TOOLS_REQUEST:
        try:
            request_id = str(payload["request_id"])
        except KeyError as exc:
            raise ProtocolError(
                "list_mcp_server_tools_request frame missing 'request_id'",
            ) from exc
        try:
            server_payload = payload["server_config"]
        except KeyError as exc:
            raise ProtocolError(
                "list_mcp_server_tools_request frame missing 'server_config'",
            ) from exc
        return ListMCPServerToolsRequest(
            request_id=request_id,
            server_config=_payload_to_mcp_server_config(server_payload),
        )

    if kind is FrameKind.LIST_MCP_SERVER_TOOLS_RESPONSE:
        try:
            request_id = str(payload["request_id"])
        except KeyError as exc:
            raise ProtocolError(
                "list_mcp_server_tools_response frame missing 'request_id'",
            ) from exc

        error_payload = payload.get("error")
        if error_payload is not None:
            if not isinstance(error_payload, dict):
                raise ProtocolError(
                    "list_mcp_server_tools_response 'error' must be an object",
                )
            try:
                error = ToolCallError(
                    kind=str(error_payload["kind"]),
                    message=str(error_payload["message"]),
                )
            except KeyError as exc:
                raise ProtocolError(
                    "list_mcp_server_tools_response error missing field "
                    f"{exc.args[0]!r}",
                ) from exc
            return ListMCPServerToolsResponse(request_id=request_id, error=error)

        if "tools" not in payload:
            raise ProtocolError(
                "list_mcp_server_tools_response must carry either 'tools' "
                "or 'error'",
            )
        tools_payload = payload["tools"]
        if not isinstance(tools_payload, list):
            raise ProtocolError(
                "list_mcp_server_tools_response 'tools' must be a list",
            )
        tools: list[dict[str, Any]] = []
        for tool in tools_payload:
            if not isinstance(tool, dict):
                raise ProtocolError(
                    "list_mcp_server_tools_response 'tools' entries must be "
                    "objects",
                )
            tools.append(dict(tool))
        return ListMCPServerToolsResponse(request_id=request_id, tools=tools)

    raise ProtocolError(f"no decoder for known frame kind {kind!r}")  # pragma: no cover


def encode_frame(frame: Frame) -> bytes:
    """Serialize *frame* to its on-wire bytes (length prefix + JSON).

    The returned ``bytes`` is exactly what should be written to a stream
    in one ``write()`` call.  Raises :class:`ProtocolError` if *frame*
    is unknown or the encoded payload exceeds :data:`MAX_FRAME_SIZE`.
    """
    payload = _frame_to_payload(frame)
    body = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    if len(body) > MAX_FRAME_SIZE:
        raise ProtocolError(
            f"frame payload exceeds {MAX_FRAME_SIZE} bytes (got {len(body)})",
        )
    return struct.pack(LENGTH_PREFIX_FORMAT, len(body)) + body


def decode_frame(body: bytes) -> Frame:
    """Decode a single frame body (without the length prefix) into a dataclass."""
    try:
        payload = json.loads(body.decode("utf-8"))
    except UnicodeDecodeError as exc:
        raise ProtocolError(f"frame body is not valid UTF-8: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ProtocolError(f"frame body is not valid JSON: {exc}") from exc
    return _payload_to_frame(payload)


async def read_frame(reader: asyncio.StreamReader) -> Frame:
    """Read exactly one frame from *reader*.

    Raises :class:`asyncio.IncompleteReadError` (propagated) when the
    far side closed the connection cleanly between frames; raises
    :class:`ProtocolError` for any framing or schema violation.
    """
    prefix = await reader.readexactly(LENGTH_PREFIX_SIZE)
    (length,) = struct.unpack(LENGTH_PREFIX_FORMAT, prefix)
    if length == 0:
        raise ProtocolError("zero-length frame is not permitted")
    if length > MAX_FRAME_SIZE:
        raise ProtocolError(
            f"frame length {length} exceeds maximum {MAX_FRAME_SIZE}",
        )
    body = await reader.readexactly(length)
    return decode_frame(body)


async def write_frame(writer: asyncio.StreamWriter, frame: Frame) -> None:
    """Encode *frame* and write it to *writer*, then drain.

    Centralizing the drain here means callers do not have to remember
    to flush after each frame and cannot accidentally interleave
    writes from different tasks (within a single task the awaits are
    cooperative).
    """
    writer.write(encode_frame(frame))
    await writer.drain()
