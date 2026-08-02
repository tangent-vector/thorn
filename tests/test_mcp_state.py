"""Tests for ``thorn.toolhost._mcp_state``.

Two layers of coverage:

1. *Dataclass round-trip* -- a snapshot encoded with
   :func:`snapshot_to_payload` and decoded with
   :func:`snapshot_from_payload` reproduces the original fields, and
   the JSON form is forward-compatible with missing/older fields.
2. *Atomic file I/O* -- :func:`write_atomic_snapshot` materializes
   the file contents at the destination path, and
   :func:`read_snapshot` returns ``None`` (rather than raising) for
   missing or malformed files.  This is the contract the CLI
   ``thorn sandbox status`` relies on for crash-free diagnostics.
"""

from __future__ import annotations

import json
from pathlib import Path

from thorn.toolhost._mcp_state import (
    MCP_STATE_FILE_NAME,
    SCHEMA_VERSION,
    MCPServerState,
    MCPStateSnapshot,
    read_snapshot,
    snapshot_from_payload,
    snapshot_to_payload,
    write_atomic_snapshot,
)


def _sample_snapshot() -> MCPStateSnapshot:
    return MCPStateSnapshot(
        updated_at="2026-04-27T22:39:55+00:00",
        servers=[
            MCPServerState(
                name="github",
                kind="stdio",
                identifier="uvx mcp-server-github",
                config_identity="ab12cd34ef01",
                alive=True,
                tool_count=17,
                last_used_at="2026-04-27T22:39:50+00:00",
            ),
            MCPServerState(
                name="docs",
                kind="http",
                identifier="https://example.com/mcp",
                config_identity="9988aabbccdd",
                alive=False,
                tool_count=None,
                last_used_at=None,
            ),
        ],
    )


class TestPayloadRoundTrip:
    def test_to_from_payload_preserves_fields(self):
        original = _sample_snapshot()
        decoded = snapshot_from_payload(snapshot_to_payload(original))
        assert decoded is not None
        assert decoded.schema_version == original.schema_version
        assert decoded.updated_at == original.updated_at
        assert decoded.servers == original.servers

    def test_default_schema_version_is_one(self):
        snap = MCPStateSnapshot(updated_at="t", servers=[])
        assert snap.schema_version == SCHEMA_VERSION == 1

    def test_from_payload_tolerates_missing_fields(self):
        payload = {
            "updated_at": "2026-04-27T00:00:00Z",
            "servers": [
                {"name": "minimal"},
            ],
        }
        decoded = snapshot_from_payload(payload)
        assert decoded is not None
        assert len(decoded.servers) == 1
        s = decoded.servers[0]
        assert s.name == "minimal"
        assert s.kind == ""
        assert s.identifier == ""
        assert s.config_identity == ""
        assert s.alive is False
        assert s.tool_count is None
        assert s.last_used_at is None

    def test_from_payload_skips_servers_without_name(self):
        payload = {
            "updated_at": "t",
            "servers": [{"kind": "stdio"}, {"name": "ok"}],
        }
        decoded = snapshot_from_payload(payload)
        assert decoded is not None
        assert [s.name for s in decoded.servers] == ["ok"]

    def test_from_payload_returns_none_on_garbage(self):
        assert snapshot_from_payload("not a dict") is None  # type: ignore[arg-type]
        assert snapshot_from_payload({}) is None  # missing updated_at
        assert snapshot_from_payload({"updated_at": 5}) is None  # wrong type
        assert (
            snapshot_from_payload({"updated_at": "t", "servers": "no"}) is None
        )

    def test_from_payload_falls_back_on_bad_schema_version(self):
        payload = {
            "updated_at": "t",
            "servers": [],
            "schema_version": "not-an-int",
        }
        decoded = snapshot_from_payload(payload)
        assert decoded is not None
        assert decoded.schema_version == SCHEMA_VERSION


class TestWriteAtomic:
    def test_writes_file_with_expected_contents(self, tmp_path: Path):
        path = tmp_path / "control" / MCP_STATE_FILE_NAME
        snap = _sample_snapshot()
        write_atomic_snapshot(path, snap)
        assert path.exists()
        on_disk = json.loads(path.read_text(encoding="utf-8"))
        decoded = snapshot_from_payload(on_disk)
        assert decoded == snap

    def test_creates_parent_directory(self, tmp_path: Path):
        # Two levels of missing parent so we know mkdir(parents=True)
        # is doing the work; a single level would also pass on a
        # vanilla mkdir() call.
        path = tmp_path / "a" / "b" / MCP_STATE_FILE_NAME
        snap = MCPStateSnapshot(updated_at="t", servers=[])
        write_atomic_snapshot(path, snap)
        assert path.exists()

    def test_overwrites_existing_file(self, tmp_path: Path):
        path = tmp_path / MCP_STATE_FILE_NAME
        write_atomic_snapshot(
            path,
            MCPStateSnapshot(updated_at="first", servers=[]),
        )
        write_atomic_snapshot(
            path,
            MCPStateSnapshot(updated_at="second", servers=[]),
        )
        assert read_snapshot(path).updated_at == "second"  # type: ignore[union-attr]

    def test_no_temp_file_left_behind_on_success(self, tmp_path: Path):
        path = tmp_path / MCP_STATE_FILE_NAME
        write_atomic_snapshot(
            path, MCPStateSnapshot(updated_at="t", servers=[]),
        )
        leftovers = [
            p for p in tmp_path.iterdir() if p.name.startswith(".")
        ]
        assert leftovers == []


class TestReadSnapshot:
    def test_missing_file_returns_none(self, tmp_path: Path):
        assert read_snapshot(tmp_path / "nope.json") is None

    def test_malformed_json_returns_none(self, tmp_path: Path):
        path = tmp_path / MCP_STATE_FILE_NAME
        path.write_text("{not json", encoding="utf-8")
        assert read_snapshot(path) is None

    def test_structurally_invalid_returns_none(self, tmp_path: Path):
        path = tmp_path / MCP_STATE_FILE_NAME
        path.write_text(json.dumps({"not": "valid"}), encoding="utf-8")
        assert read_snapshot(path) is None

    def test_round_trips_via_disk(self, tmp_path: Path):
        path = tmp_path / MCP_STATE_FILE_NAME
        snap = _sample_snapshot()
        write_atomic_snapshot(path, snap)
        decoded = read_snapshot(path)
        assert decoded == snap
