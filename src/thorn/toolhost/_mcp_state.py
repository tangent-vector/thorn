"""Daemon -> CLI MCP state snapshot file.

The toolhost daemon writes a JSON snapshot of its
:class:`~thorn.toolhost._mcp_host.MCPHost` state into the per-agent
control directory (alongside ``toolhost.sock`` / ``toolhost.log``) on
every MCP operation.  The ``thorn sandbox status`` CLI reads that file
on the host side -- it is the only currently-supported brain-out-of-band
read of live MCP state, intentionally so: there is no second protocol
connection (the daemon serves one client) and no ``oci exec`` peeking.

File layout in the control dir::

    <agent_control_dir>/
        toolhost.sock          # Phase A
        toolhost.log           # Phase A
        mcp_state.json         # this module

The schema is versioned (``schema_version: 1``) so future shape changes
can be detected and skipped by older readers without crashing.

Atomicity
---------
Writes go through ``write_atomic_snapshot``, which writes to a sibling
``mcp_state.json.tmp`` and ``os.replace``s into place.  ``os.replace``
is atomic on POSIX, so a CLI reading the file mid-write either sees
the previous snapshot or the new one, never a half-written one.

Both encode and decode are tolerant of a missing field by treating it
as ``None`` / a sensible default; that way an older daemon writing a
slimmer file is still readable by a newer CLI, and vice versa.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


MCP_STATE_FILE_NAME: str = "mcp_state.json"
"""Conventional file name for the snapshot inside the agent's control dir."""

SCHEMA_VERSION: int = 1
"""Bumped whenever the on-disk shape changes incompatibly."""


@dataclass(frozen=True)
class MCPServerState:
    """Live state of a single MCP server hosted by the daemon.

    All fields except ``name`` may be ``None`` to represent "unknown
    yet" (e.g. a server registered but not yet connected has no
    ``tool_count``).  ``identifier`` is human-readable and is intended
    only for diagnostics; it is not a stable identity (use
    ``config_identity`` for that).
    """

    name: str
    """The MCP server's logical name (the ``name`` field from
    ``mcp.json``)."""

    kind: str
    """``"stdio"`` for command-style configs, ``"http"`` for URL-based."""

    identifier: str
    """Human-readable transport hint: the command (with first arg) for
    stdio servers, the URL for HTTP servers."""

    config_identity: str
    """Short hex digest of the full
    :func:`~thorn.core._mcp_config.mcp_server_config_identity` tuple.
    Two snapshots agree on this value iff their underlying configs are
    byte-identical (including ``env``).  Operators use this to
    cross-reference the snapshot with daemon log lines."""

    alive: bool
    """``True`` while the daemon currently holds an open
    ``ClientSession`` for this config."""

    tool_count: int | None
    """Number of tools in the cache for this server, or ``None`` if the
    daemon has not yet listed them."""

    last_used_at: str | None
    """ISO-8601 timestamp of the last ``list_tools`` or ``call_tool``
    against this server, or ``None`` if the entry has only been
    registered (never used)."""


@dataclass
class MCPStateSnapshot:
    """The daemon's view of its MCP fleet at a single point in time.

    Snapshots are produced fresh on every persisted state update; the
    daemon does not retain history.  Readers should treat the file as
    monotonic only in the sense that ``updated_at`` increases.
    """

    updated_at: str
    """ISO-8601 timestamp at which the snapshot was produced."""

    servers: list[MCPServerState] = field(default_factory=list)
    """One entry per server the daemon currently has registered;
    empty when no MCP traffic has happened in this daemon lifetime
    yet."""

    schema_version: int = SCHEMA_VERSION
    """Mirrors :data:`SCHEMA_VERSION` for forward-compat detection."""


# ---------------------------------------------------------------------------
# JSON encode / decode
# ---------------------------------------------------------------------------


def _server_to_payload(state: MCPServerState) -> dict[str, Any]:
    return {
        "name": state.name,
        "kind": state.kind,
        "identifier": state.identifier,
        "config_identity": state.config_identity,
        "alive": state.alive,
        "tool_count": state.tool_count,
        "last_used_at": state.last_used_at,
    }


def _server_from_payload(payload: dict[str, Any]) -> MCPServerState | None:
    """Decode one server entry; returns ``None`` if it lacks a name.

    Tolerant to missing fields so an older daemon's slimmer file still
    decodes; missing fields surface as ``None`` / sensible defaults
    rather than raising.
    """
    if not isinstance(payload, dict):
        return None
    name = payload.get("name")
    if not isinstance(name, str) or not name:
        return None
    return MCPServerState(
        name=name,
        kind=str(payload.get("kind", "")),
        identifier=str(payload.get("identifier", "")),
        config_identity=str(payload.get("config_identity", "")),
        alive=bool(payload.get("alive", False)),
        tool_count=(
            int(payload["tool_count"])
            if payload.get("tool_count") is not None
            else None
        ),
        last_used_at=(
            str(payload["last_used_at"])
            if payload.get("last_used_at") is not None
            else None
        ),
    )


def snapshot_to_payload(snapshot: MCPStateSnapshot) -> dict[str, Any]:
    """Convert *snapshot* to the JSON-encodable mapping written to disk."""
    return {
        "schema_version": snapshot.schema_version,
        "updated_at": snapshot.updated_at,
        "servers": [_server_to_payload(s) for s in snapshot.servers],
    }


def snapshot_from_payload(payload: dict[str, Any]) -> MCPStateSnapshot | None:
    """Decode a JSON payload into a snapshot; ``None`` on any structural problem.

    Returns ``None`` (rather than raising) so the CLI can render
    ``mcp_state: <unreadable>`` without aborting the entire status
    command on a single garbled file.
    """
    if not isinstance(payload, dict):
        return None
    updated_at = payload.get("updated_at")
    if not isinstance(updated_at, str):
        return None

    servers_payload = payload.get("servers", [])
    if not isinstance(servers_payload, list):
        return None
    servers: list[MCPServerState] = []
    for entry in servers_payload:
        decoded = _server_from_payload(entry)
        if decoded is not None:
            servers.append(decoded)

    raw_version = payload.get("schema_version", SCHEMA_VERSION)
    try:
        version = int(raw_version)
    except (TypeError, ValueError):
        version = SCHEMA_VERSION

    return MCPStateSnapshot(
        updated_at=updated_at,
        servers=servers,
        schema_version=version,
    )


# ---------------------------------------------------------------------------
# Atomic write / safe read
# ---------------------------------------------------------------------------


def write_atomic_snapshot(path: Path, snapshot: MCPStateSnapshot) -> None:
    """Write *snapshot* to *path* atomically.

    Strategy: serialize to a sibling temp file in the same directory
    (so the rename is on the same filesystem), ``fsync`` it, then
    ``os.replace`` over the destination.  ``os.replace`` is atomic on
    POSIX, which is the only platform the daemon runs on.

    Failures are propagated to the caller; the daemon's
    ``_persist_mcp_state`` wraps the call in a best-effort try/except
    so a transient I/O hiccup does not break the user's MCP call.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = snapshot_to_payload(snapshot)
    body = json.dumps(payload, indent=2, sort_keys=False)

    tmp_fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as fh:
            fh.write(body)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, path)
    except Exception:
        # Make sure we don't leave the temp file behind on failure.
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def read_snapshot(path: Path) -> MCPStateSnapshot | None:
    """Read *path* and decode; ``None`` if missing, unreadable, or malformed.

    The CLI calls this once per container during ``thorn sandbox
    status``; treating every failure as "no snapshot available"
    preserves the principle that diagnostic commands never crash on
    bad daemon state.
    """
    if not path.exists():
        return None
    try:
        body = path.read_text(encoding="utf-8")
    except OSError as exc:
        logger.warning("read_snapshot: failed to read %s: %s", path, exc)
        return None
    try:
        payload = json.loads(body)
    except json.JSONDecodeError as exc:
        logger.warning("read_snapshot: failed to parse %s: %s", path, exc)
        return None
    return snapshot_from_payload(payload)


__all__ = [
    "MCP_STATE_FILE_NAME",
    "MCPServerState",
    "MCPStateSnapshot",
    "SCHEMA_VERSION",
    "read_snapshot",
    "snapshot_from_payload",
    "snapshot_to_payload",
    "write_atomic_snapshot",
]
