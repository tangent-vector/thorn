"""Tests for read-only operator status collection."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

from thorn._cli import main
from thorn.gateway._heartbeat import write_gateway_heartbeat
from thorn.gateway._operator_status import (
    GatewayLiveness,
    InboxItemLocation,
    NotificationStatusCounts,
    OperatorStatusAlertCode,
    OperatorStatusAlertSeverity,
    collect_heartbeat_summary,
    collect_inbox_items,
    collect_operator_status,
    collect_provider_unavailable_sessions,
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
from thorn.runtime._provider_state import PROVIDER_UNAVAILABLE_METADATA_KEY
from thorn.runtime._todo import SessionTodoList


def _paths(tmp_path: Path) -> AgencyPaths:
    return AgencyPaths.for_gateway(
        agency_dir=tmp_path / ".thorn",
        workspace_dir=tmp_path / "workspace",
    )


def _session_address() -> SessionAddress:
    return SessionAddress(AgentID("coordinator"), SessionKey("project/issue/7"))


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


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


def test_collect_inbox_items_includes_linked_todo_summary(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    address = _session_address()
    inbox = SessionInbox(
        paths.session_inbox_dir(address.agent_id, address.session_key),
        address,
    )
    notification = inbox.post(_spec(address, "pending work"))
    todos = SessionTodoList(
        paths.session_todo_file(address.agent_id, address.session_key),
    )
    todos.create(
        title="finish linked work",
        linked_inbox_item_id=notification.id,
    )

    records = collect_inbox_items(paths)

    assert len(records) == 1
    payload = records[0].to_json()
    assert payload["linked_todos"]["open_count"] == 1
    assert payload["linked_todos"]["open_titles"] == ["finish linked work"]


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


def test_collect_provider_unavailable_sessions_reads_session_metadata(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    address = _session_address()
    metadata_dir = paths.session_metadata_dir(
        address.agent_id,
        address.session_key,
    )
    metadata_dir.mkdir(parents=True)
    (metadata_dir / "session.json").write_text(
        (
            "{\n"
            f'  "key": "{address.session_key}",\n'
            '  "created_at": null,\n'
            '  "last_active": null,\n'
            '  "metadata": {\n'
            f'    "{PROVIDER_UNAVAILABLE_METADATA_KEY}": {{\n'
            '      "state": "waiting_on_provider",\n'
            '      "attempts": 9,\n'
            '      "reason": "read timeout"\n'
            "    }\n"
            "  }\n"
            "}\n"
        ),
        encoding="utf-8",
    )

    sessions = collect_provider_unavailable_sessions(paths)

    assert len(sessions) == 1
    assert sessions[0].agent_id == address.agent_id
    assert sessions[0].session_key == address.session_key
    assert sessions[0].metadata["attempts"] == 9


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
    assert summary.provider_unavailable_sessions == ()
    assert summary.config_error == "no config"
    assert summary.in_flight_external_keys == (
        "external:parked",
        "external:pending",
    )
    assert [(alert.code, alert.severity) for alert in summary.alerts] == [
        (
            OperatorStatusAlertCode.GATEWAY_HEARTBEAT_MISSING,
            OperatorStatusAlertSeverity.WARNING,
        ),
        (
            OperatorStatusAlertCode.INBOX_PARKED_ERRORED,
            OperatorStatusAlertSeverity.ERROR,
        ),
        (
            OperatorStatusAlertCode.SANDBOX_STATUS_ERROR,
            OperatorStatusAlertSeverity.WARNING,
        ),
    ]


@pytest.mark.asyncio
async def test_collect_operator_status_alerts_cover_provider_and_source_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _paths(tmp_path)
    address = _session_address()
    inbox = SessionInbox(
        paths.session_inbox_dir(address.agent_id, address.session_key),
        address,
    )
    errored = inbox.post(_spec(address, "errored"))
    inbox.update_status(
        errored.id,
        NotificationStatus.ERRORED,
        error_reason="provider timeout",
    )
    write_gateway_heartbeat(
        paths.home_root / "gateway-status.json",
        {
            "status": "running",
            "updated_at": "2999-01-01T00:00:00+00:00",
            "heartbeat_interval_s": 5,
            "provider_health": {
                "state": "degraded",
                "recent_failure_count": 4,
                "seconds_until_probe": 12.5,
                "probe_in_flight": False,
                "consecutive_probe_failures": 1,
            },
            "sources": [
                {
                    "name": "gitlab",
                    "source_type": "gitlab",
                    "state": "error",
                    "last_error": "401 unauthorized",
                    "last_poll_started_at": None,
                    "last_poll_finished_at": None,
                    "last_event_count": None,
                    "poll_count": 3,
                }
            ],
        },
    )

    async def _broker_status():
        from thorn.gateway._operator_status import BrokerStatusSummary

        return BrokerStatusSummary(stacks=(), error="compose unavailable")

    async def _sandbox_status(_gateway_config):
        from thorn.gateway._operator_status import SandboxStatusSummary

        return SandboxStatusSummary(
            runtime_name=None,
            image=None,
            image_present=None,
            containers=(),
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
    )

    assert [alert.code for alert in summary.alerts] == [
        OperatorStatusAlertCode.PROVIDER_DEGRADED,
        OperatorStatusAlertCode.EVENT_SOURCE_ERROR,
        OperatorStatusAlertCode.INBOX_ERRORED,
        OperatorStatusAlertCode.BROKER_STATUS_ERROR,
    ]
    assert summary.to_json()["alerts"][0] == {
        "code": "provider_degraded",
        "severity": "error",
        "summary": "Provider health is degraded; recent_failures=4 seconds_until_probe=12.5.",
    }


def test_status_cli_renders_and_serializes_alerts(
    runner: CliRunner,
    tmp_path: Path,
) -> None:
    agency = tmp_path / "agency"
    agency.mkdir()
    write_gateway_heartbeat(
        agency / "gateway-status.json",
        {
            "status": "running",
            "updated_at": "2999-01-01T00:00:00+00:00",
            "heartbeat_interval_s": 5,
            "provider_health": {
                "state": "degraded",
                "recent_failure_count": 3,
                "seconds_until_probe": 9.0,
                "probe_in_flight": False,
                "consecutive_probe_failures": 0,
            },
            "sources": [],
        },
    )

    text_result = runner.invoke(main, ["status", "--agency", str(agency)])
    assert text_result.exit_code == 0, text_result.output
    assert "Alerts: 2" in text_result.output
    assert "WARNING" in text_result.output
    assert "sandbox_status_error" in text_result.output
    assert "ERROR" in text_result.output
    assert "provider_degraded" in text_result.output

    json_result = runner.invoke(main, ["status", "--agency", str(agency), "--json"])
    assert json_result.exit_code == 0, json_result.output
    payload = json.loads(json_result.output)
    assert [alert["code"] for alert in payload["alerts"]] == [
        "provider_degraded",
        "sandbox_status_error",
    ]
