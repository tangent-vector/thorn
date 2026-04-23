"""``thorn.toolhost`` -- the per-agent tool-execution daemon and its IPC.

Phase A of the sandbox roadmap introduces a separate ``thorn-toolhost``
process that the brain (the agent loop) talks to over a Unix-domain
socket.  This package holds:

* :mod:`thorn.toolhost._protocol` -- the framed JSON wire format used
  by both ends of the socket.  Independent of any I/O so it can be
  unit-tested in isolation and reused later by both the daemon (this
  package) and the brain-side ``DaemonToolExecutor``.

The daemon entrypoint and the brain-side executor land in subsequent
phase-A todos; this initial commit only ships the protocol primitives.
"""

from __future__ import annotations

from thorn.toolhost._executor import (
    DaemonCrashedError,
    DaemonExecutorConfig,
    DaemonToolExecutor,
    DaemonUnavailableError,
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
    FrameKind,
    Heartbeat,
    Hello,
    ProtocolError,
    ToolCallCancel,
    ToolCallChunk,
    ToolCallError,
    ToolCallRequest,
    ToolCallResponse,
    decode_frame,
    encode_frame,
    read_frame,
    write_frame,
)

__all__ = [
    "PROTOCOL_MAJOR",
    "PROTOCOL_MINOR",
    "DaemonCrashedError",
    "DaemonExecutorConfig",
    "DaemonHost",
    "DaemonToolExecutor",
    "DaemonUnavailableError",
    "Frame",
    "SubprocessDaemonHost",
    "SubprocessDaemonHostConfig",
    "FrameKind",
    "Heartbeat",
    "Hello",
    "ProtocolError",
    "ToolCallCancel",
    "ToolCallChunk",
    "ToolCallError",
    "ToolCallRequest",
    "ToolCallResponse",
    "decode_frame",
    "encode_frame",
    "read_frame",
    "write_frame",
]
