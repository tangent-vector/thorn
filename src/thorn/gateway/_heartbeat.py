"""Gateway heartbeat file used by operator status commands."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

GATEWAY_HEARTBEAT_FILENAME = "gateway-status.json"
GATEWAY_HEARTBEAT_SCHEMA_VERSION = 1


def gateway_heartbeat_path(agency_home: Path) -> Path:
    """Return the heartbeat path for an agency home."""
    return Path(agency_home) / GATEWAY_HEARTBEAT_FILENAME


def gateway_heartbeat_timestamp() -> str:
    """Return the canonical UTC timestamp for heartbeat records."""
    return datetime.now(timezone.utc).isoformat()


def write_gateway_heartbeat(path: Path, payload: dict[str, Any]) -> None:
    """Atomically write *payload* as the current gateway heartbeat."""
    path.parent.mkdir(parents=True, exist_ok=True)
    enriched = {
        "schema_version": GATEWAY_HEARTBEAT_SCHEMA_VERSION,
        **payload,
    }
    temp_path = path.with_name(f".tmp-{path.name}-{os.getpid()}")
    temp_path.write_text(
        json.dumps(enriched, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temp_path, path)


def read_gateway_heartbeat(path: Path) -> dict[str, Any] | None:
    """Read a heartbeat JSON object, returning ``None`` if absent or invalid."""
    if not path.is_file():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(raw, dict):
        return None
    return raw


__all__ = [
    "GATEWAY_HEARTBEAT_FILENAME",
    "GATEWAY_HEARTBEAT_SCHEMA_VERSION",
    "gateway_heartbeat_path",
    "gateway_heartbeat_timestamp",
    "read_gateway_heartbeat",
    "write_gateway_heartbeat",
]
