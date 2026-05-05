"""Tests for read-only operator status collection."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from thorn.gateway._heartbeat import write_gateway_heartbeat
from thorn.gateway._operator_status import (
    GatewayLiveness,
    InboxItemLocation,
    NotificationStatusCounts,
    collect_heartbeat_summary,
    collect_inbox_items,
    collect_operator_status,
    collect_sandbox_status,
)
from thorn.runtime import (
    AddressBook,
    AgencyPaths,
    AgentID,
    NotificationSpec,
    NotificationStatus,
    SessionAddress,
    SessionInbox,
    SessionKey,
    apply_handling_transition,
)


def _paths(tmp_path: Path) -> AgencyPaths:
    return AgencyPaths.for_gateway(
        agency_dir=tmp_path / ".thorn",
        workspace_dir=tmp_path / "workspace",
    )


def _session_address() -> SessionAddress:
    return SessionAddress(AgentID("coordinator"), SessionKey("project/issue/7"))


def _spec(target: SessionAddress, content: str) -> NotificationSpec:
    return NotificationSpec(
        source="test",
        content=content,
        target=target,
        metadata={"kind": content},
        external_key=f"external:{content}",
    )


def test_collect_inbox_items_covers_live_and_parked_statuses(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    address = _session_address()
    inbox = SessionInbox(
        paths.session_inbox_dir(address.agent_id, address.session_key),
        address,
    )

    pending = inbox.post(_spec(address, "pending work"))
    in_progress = inbox.post(_spec(address, "active work"))
    handled = inbox.post(_spec(address, "handled work"))
    errored = inbox.post(_spec(address, "errored awaiting dispatch"))
    confirmed = inbox.post(_spec(address, "confirmed cleanup"))
    parked = inbox.post(_spec(address, "parked failure"))

    inbox.update_status(in_progress.id, NotificationStatus.IN_PROGRESS)
    inbox.update_status(handled.id, NotificationStatus.HANDLED, notes="done")
    inbox.update_status(
        errored.id,
        NotificationStatus.ERRORED,
        error_reason="transient failure",
    )
    inbox.update_status(confirmed.id, NotificationStatus.CONFIRMED)
    apply_handling_transition(
        inbox,
        parked.id,
        NotificationStatus.ERRORED,
        address_book=AddressBook(),
        error_reason="provider unavailable",
    )

    records = collect_inbox_items(paths)

    assert {record.item_id for record in records} == {
        pending.id,
        in_progress.id,
        handled.id,
        errored.id,
        confirmed.id,
        parked.id,
    }
    parked_records = [
        record
        for record in records
        if record.location is InboxItemLocation.PARKED_ERRORED
    ]
    assert [record.item_id for record in parked_records] == [parked.id]
    assert parked_records[0].notification.error_reason == "provider unavailable"

    errored_records = collect_inbox_items(paths, status_filter="errored")
    assert {record.item_id for record in errored_records} == {
        errored.id,
        parked.id,
    }
    only_parked = collect_inbox_items(paths, status_filter="parked_errored")
    assert [record.item_id for record in only_parked] == [parked.id]


def test_heartbeat_summary_classifies_running_and_stopped(
    tmp_path: Path,
) -> None:
    agency_home = tmp_path / ".thorn"
    heartbeat_path = agency_home / "gateway-status.json"

    write_gateway_heartbeat(
        heartbeat_path,
        {
            "status": "running",
            "updated_at": "2999-01-01T00:00:00+00:00",
            "heartbeat_interval_s": 5,
        },
    )
    assert collect_heartbeat_summary(agency_home).liveness is GatewayLiveness.RUNNING

    write_gateway_heartbeat(
        heartbeat_path,
        {
            "status": "stopped",
            "updated_at": "2026-05-05T00:00:00+00:00",
            "heartbeat_interval_s": 5,
        },
    )
    assert collect_heartbeat_summary(agency_home).liveness is GatewayLiveness.STOPPED


@pytest.mark.asyncio
async def test_collect_sandbox_status_reports_subprocess_without_oci_probe() -> None:
    summary = await collect_sandbox_status(
        SimpleNamespace(sandbox=SimpleNamespace(backend="subprocess"))
    )

    assert summary.backend == "subprocess"
    assert summary.image_present is None
    assert summary.containers == ()
    assert summary.error is None


@pytest.mark.asyncio
async def test_collect_operator_status_summarizes_counts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _paths(tmp_path)
    address = _session_address()
    inbox = SessionInbox(
        paths.session_inbox_dir(address.agent_id, address.session_key),
        address,
    )
    inbox.post(_spec(address, "pending"))
    parked = inbox.post(_spec(address, "parked"))
    apply_handling_transition(
        inbox,
        parked.id,
        NotificationStatus.ERRORED,
        address_book=AddressBook(),
        error_reason="bad key",
    )

    async def _broker_status():
        from thorn.gateway._operator_status import BrokerStatusSummary

        return BrokerStatusSummary(stacks=())

    async def _sandbox_status(_gateway_config):
        from thorn.gateway._operator_status import SandboxStatusSummary

        return SandboxStatusSummary(
            runtime_name=None,
            image=None,
            image_present=None,
            containers=(),
            error="not checked",
        )

    monkeypatch.setattr(
        "thorn.gateway._operator_status.collect_broker_status",
        _broker_status,
    )
    monkeypatch.setattr(
        "thorn.gateway._operator_status.collect_sandbox_status",
        _sandbox_status,
    )

    summary = await collect_operator_status(
        agency_home=paths.home_root,
        workspace_root=paths.workspace_root,
        gateway_config=None,
        config_error="no config",
    )

    assert summary.inbox_counts == NotificationStatusCounts(
        pending=1,
        parked_errored=1,
    )
    assert summary.config_error == "no config"
    assert summary.in_flight_external_keys == (
        "external:parked",
        "external:pending",
    )
