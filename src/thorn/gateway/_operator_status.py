"""Read-only operator status collection for the Thorn gateway CLI."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from typing import Any

from thorn.gateway._heartbeat import (
    gateway_heartbeat_path,
    read_gateway_heartbeat,
)
from thorn.runtime._address import SessionAddress
from thorn.runtime._in_flight_index import rebuild_in_flight_index
from thorn.runtime._inbox import SessionInbox
from thorn.runtime._notification import Notification, NotificationStatus
from thorn.runtime._paths import AgencyPaths
from thorn.runtime._queue import DurableQueue
from thorn.runtime._session import AgentID, SessionKey
from thorn.runtime._store import SessionStore

_SUMMARY_CHARS = 96


class InboxItemLocation(StrEnum):
    """Where an operator-visible notification currently lives."""

    LIVE = "live"
    PARKED_ERRORED = "parked_errored"


class GatewayLiveness(StrEnum):
    """Computed liveness of the last gateway heartbeat."""

    UNKNOWN = "unknown"
    RUNNING = "running"
    STALE = "stale"
    STOPPED = "stopped"


@dataclass(frozen=True)
class InboxItemRecord:
    """One notification found in a session inbox tree."""

    agent_id: AgentID
    session_key: SessionKey
    location: InboxItemLocation
    notification: Notification

    @property
    def item_id(self) -> str:
        return self.notification.id

    @property
    def status(self) -> NotificationStatus:
        return self.notification.status

    @property
    def summary(self) -> str:
        return summarize_notification_content(self.notification.content)

    def to_json(self, *, include_content: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "id": self.item_id,
            "agent_id": str(self.agent_id),
            "session_key": str(self.session_key),
            "location": self.location.value,
            "status": self.status.value,
            "source": self.notification.source,
            "posted_at": self.notification.posted_at.isoformat(),
            "attempt_count": self.notification.attempt_count,
            "summary": self.summary,
            "metadata": dict(self.notification.metadata),
            "external_key": self.notification.external_key,
            "notes": self.notification.notes,
            "error_reason": self.notification.error_reason,
        }
        if include_content:
            payload["content"] = self.notification.content
        return payload


@dataclass(frozen=True)
class NotificationStatusCounts:
    """Counts by notification lifecycle state."""

    pending: int = 0
    in_progress: int = 0
    handled: int = 0
    errored: int = 0
    confirmed: int = 0
    parked_errored: int = 0

    @property
    def total(self) -> int:
        return (
            self.pending
            + self.in_progress
            + self.handled
            + self.errored
            + self.confirmed
            + self.parked_errored
        )

    def to_json(self) -> dict[str, int]:
        return {
            "pending": self.pending,
            "in_progress": self.in_progress,
            "handled": self.handled,
            "errored": self.errored,
            "confirmed": self.confirmed,
            "parked_errored": self.parked_errored,
            "total": self.total,
        }


@dataclass(frozen=True)
class ServiceQueueSummary:
    """Counts for one service notification queue."""

    service_name: str
    counts: NotificationStatusCounts

    def to_json(self) -> dict[str, Any]:
        return {
            "service_name": self.service_name,
            "counts": self.counts.to_json(),
        }


@dataclass(frozen=True)
class GatewayHeartbeatSummary:
    """Parsed view of the gateway heartbeat file."""

    path: Path
    liveness: GatewayLiveness
    payload: dict[str, Any] | None
    stale_after_seconds: float | None

    def to_json(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "liveness": self.liveness.value,
            "stale_after_seconds": self.stale_after_seconds,
            "payload": self.payload,
        }


@dataclass(frozen=True)
class BrokerStackSummary:
    """One bundled broker compose stack visible on the host."""

    project_name: str
    runtime_name: str
    status: str

    def to_json(self) -> dict[str, str]:
        return {
            "project_name": self.project_name,
            "runtime_name": self.runtime_name,
            "status": self.status,
        }


@dataclass(frozen=True)
class BrokerStatusSummary:
    """Operator summary for bundled broker stacks."""

    stacks: tuple[BrokerStackSummary, ...]
    error: str | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "stacks": [stack.to_json() for stack in self.stacks],
            "error": self.error,
        }


@dataclass(frozen=True)
class SandboxContainerSummary:
    """One per-agent sandbox container visible on the host."""

    name: str
    status: str
    running: bool
    exit_code: int | None

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "running": self.running,
            "exit_code": self.exit_code,
        }


@dataclass(frozen=True)
class SandboxStatusSummary:
    """Operator summary for sandbox image and containers."""

    runtime_name: str | None
    image: str | None
    image_present: bool | None
    containers: tuple[SandboxContainerSummary, ...]
    backend: str | None = None
    error: str | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "runtime_name": self.runtime_name,
            "image": self.image,
            "image_present": self.image_present,
            "containers": [
                container.to_json() for container in self.containers
            ],
            "error": self.error,
        }


@dataclass(frozen=True)
class OperatorStatusSummary:
    """Top-level status summary rendered by ``thorn status``."""

    agency_home: Path
    workspace_root: Path
    agent_ids: tuple[AgentID, ...]
    session_count: int
    inbox_counts: NotificationStatusCounts
    service_queues: tuple[ServiceQueueSummary, ...]
    in_flight_external_keys: tuple[str, ...]
    heartbeat: GatewayHeartbeatSummary
    broker: BrokerStatusSummary
    sandbox: SandboxStatusSummary
    config_error: str | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "agency_home": str(self.agency_home),
            "workspace_root": str(self.workspace_root),
            "agent_ids": [str(agent_id) for agent_id in self.agent_ids],
            "session_count": self.session_count,
            "inbox_counts": self.inbox_counts.to_json(),
            "service_queues": [
                service_queue.to_json()
                for service_queue in self.service_queues
            ],
            "in_flight_external_keys": list(self.in_flight_external_keys),
            "heartbeat": self.heartbeat.to_json(),
            "broker": self.broker.to_json(),
            "sandbox": self.sandbox.to_json(),
            "config_error": self.config_error,
        }


def summarize_notification_content(content: str) -> str:
    """Return a single-line summary suitable for operator lists."""
    first_line = content.split("\n", 1)[0].strip()
    if not first_line:
        return "(empty content)"
    if len(first_line) <= _SUMMARY_CHARS:
        return first_line
    return first_line[: _SUMMARY_CHARS - 1].rstrip() + "..."


def collect_inbox_items(
    paths: AgencyPaths,
    *,
    agent_id_filter: AgentID | None = None,
    session_key_filter: SessionKey | None = None,
    status_filter: str | None = None,
    item_id_filter: str | None = None,
) -> list[InboxItemRecord]:
    """Collect session inbox items visible to operators."""
    records: list[InboxItemRecord] = []
    for agent_id, session_key, inbox_dir in paths.iter_session_inbox_locations():
        if agent_id_filter is not None and agent_id != agent_id_filter:
            continue
        if session_key_filter is not None and session_key != session_key_filter:
            continue
        address = SessionAddress(agent_id, session_key)
        inbox = SessionInbox(inbox_dir, address)
        records.extend(
            _record_for_notification(
                agent_id=agent_id,
                session_key=session_key,
                location=InboxItemLocation.LIVE,
                notification=notification,
            )
            for notification in inbox.list()
        )
        records.extend(
            _record_for_notification(
                agent_id=agent_id,
                session_key=session_key,
                location=InboxItemLocation.PARKED_ERRORED,
                notification=notification,
            )
            for notification in inbox.errored_items()
        )

    filtered = [
        record for record in records
        if _matches_item_filters(
            record,
            status_filter=status_filter,
            item_id_filter=item_id_filter,
        )
    ]
    return sorted(
        filtered,
        key=lambda record: (
            str(record.agent_id),
            str(record.session_key),
            record.notification.posted_at,
            record.item_id,
            record.location.value,
        ),
    )


def collect_service_queue_summaries(paths: AgencyPaths) -> list[ServiceQueueSummary]:
    """Collect counts for every service notification queue."""
    summaries: list[ServiceQueueSummary] = []
    for service_name, queue_dir in paths.iter_service_queue_locations():
        live_items = DurableQueue(queue_dir).list()
        parked_items = DurableQueue(queue_dir / "errored").list()
        summaries.append(ServiceQueueSummary(
            service_name=service_name,
            counts=_counts_for_notifications(live_items, parked_items),
        ))
    return sorted(summaries, key=lambda summary: summary.service_name)


def collect_heartbeat_summary(agency_home: Path) -> GatewayHeartbeatSummary:
    """Read and classify the gateway heartbeat for *agency_home*."""
    path = gateway_heartbeat_path(agency_home)
    payload = read_gateway_heartbeat(path)
    if payload is None:
        return GatewayHeartbeatSummary(
            path=path,
            liveness=GatewayLiveness.UNKNOWN,
            payload=None,
            stale_after_seconds=None,
        )

    status = str(payload.get("status") or "")
    stale_after = _heartbeat_stale_after_seconds(payload)
    liveness = GatewayLiveness.UNKNOWN
    if status == GatewayLiveness.STOPPED.value:
        liveness = GatewayLiveness.STOPPED
    elif status == GatewayLiveness.RUNNING.value:
        liveness = (
            GatewayLiveness.STALE
            if _heartbeat_is_stale(payload, stale_after)
            else GatewayLiveness.RUNNING
        )
    return GatewayHeartbeatSummary(
        path=path,
        liveness=liveness,
        payload=payload,
        stale_after_seconds=stale_after,
    )


async def collect_broker_status() -> BrokerStatusSummary:
    """Collect bundled broker stack status, never raising to callers."""
    try:
        from thorn.gateway._bundled_broker import list_bundled_broker_stacks

        stacks = await list_bundled_broker_stacks()
    except Exception as exc:
        return BrokerStatusSummary(stacks=(), error=str(exc))
    return BrokerStatusSummary(
        stacks=tuple(
            BrokerStackSummary(
                project_name=stack.project_name,
                runtime_name=stack.runtime_name,
                status=stack.status,
            )
            for stack in stacks
        )
    )


async def collect_sandbox_status(gateway_config: Any | None) -> SandboxStatusSummary:
    """Collect sandbox image/container status, never raising to callers."""
    if gateway_config is None:
        return SandboxStatusSummary(
            runtime_name=None,
            image=None,
            image_present=None,
            containers=(),
            error="gateway configuration unavailable",
        )

    sandbox_config = getattr(gateway_config, "sandbox", None)
    backend = getattr(sandbox_config, "backend", None)
    if backend == "subprocess":
        return SandboxStatusSummary(
            runtime_name=None,
            image=None,
            image_present=None,
            containers=(),
            backend=backend,
        )

    runtime_choice = getattr(sandbox_config, "oci_runtime", None)
    try:
        from thorn.sandbox import (
            default_sandbox_image_tag,
            select_oci_runtime,
        )

        image = (
            getattr(sandbox_config, "image", None)
            or default_sandbox_image_tag()
        )
        adapter = select_oci_runtime(runtime_choice)
        image_present = await adapter.image_exists(image)
        containers = await adapter.list_containers(name_prefix="thorn-agent-")
    except Exception as exc:
        return SandboxStatusSummary(
            runtime_name=runtime_choice,
            image=None,
            image_present=None,
            containers=(),
            backend=backend,
            error=str(exc),
        )

    return SandboxStatusSummary(
        runtime_name=adapter.name,
        image=image,
        image_present=image_present,
        containers=tuple(
            SandboxContainerSummary(
                name=container.name,
                status=container.status,
                running=container.running,
                exit_code=container.exit_code,
            )
            for container in containers
        ),
        backend=backend,
    )


async def collect_operator_status(
    *,
    agency_home: Path,
    workspace_root: Path,
    gateway_config: Any | None,
    config_error: str | None = None,
) -> OperatorStatusSummary:
    """Collect the status rendered by ``thorn status``."""
    paths = AgencyPaths.for_gateway(
        agency_dir=agency_home,
        workspace_dir=workspace_root,
    )
    store = SessionStore(paths)
    agent_ids = tuple(store.list_agent_ids())
    session_count = sum(
        len(store.list_session_keys(agent_id)) for agent_id in agent_ids
    )
    inbox_records = collect_inbox_items(paths)
    in_flight_keys = tuple(sorted(rebuild_in_flight_index(paths).snapshot()))

    return OperatorStatusSummary(
        agency_home=agency_home,
        workspace_root=workspace_root,
        agent_ids=agent_ids,
        session_count=session_count,
        inbox_counts=_counts_for_records(inbox_records),
        service_queues=tuple(collect_service_queue_summaries(paths)),
        in_flight_external_keys=in_flight_keys,
        heartbeat=collect_heartbeat_summary(agency_home),
        broker=await collect_broker_status(),
        sandbox=await collect_sandbox_status(gateway_config),
        config_error=config_error,
    )


def _record_for_notification(
    *,
    agent_id: AgentID,
    session_key: SessionKey,
    location: InboxItemLocation,
    notification: Notification,
) -> InboxItemRecord:
    return InboxItemRecord(
        agent_id=agent_id,
        session_key=session_key,
        location=location,
        notification=notification,
    )


def _matches_item_filters(
    record: InboxItemRecord,
    *,
    status_filter: str | None,
    item_id_filter: str | None,
) -> bool:
    if item_id_filter is not None and record.item_id != item_id_filter:
        return False
    if status_filter is None:
        return True
    if status_filter == InboxItemLocation.PARKED_ERRORED.value:
        return record.location is InboxItemLocation.PARKED_ERRORED
    if status_filter == NotificationStatus.ERRORED.value:
        return record.status is NotificationStatus.ERRORED
    return record.status.value == status_filter


def _counts_for_records(records: list[InboxItemRecord]) -> NotificationStatusCounts:
    live_items = [
        record.notification
        for record in records
        if record.location is InboxItemLocation.LIVE
    ]
    parked_items = [
        record.notification
        for record in records
        if record.location is InboxItemLocation.PARKED_ERRORED
    ]
    return _counts_for_notifications(live_items, parked_items)


def _counts_for_notifications(
    live_items: list[Notification],
    parked_items: list[Notification],
) -> NotificationStatusCounts:
    counts = {status: 0 for status in NotificationStatus}
    for notification in live_items:
        counts[notification.status] += 1
    return NotificationStatusCounts(
        pending=counts[NotificationStatus.PENDING],
        in_progress=counts[NotificationStatus.IN_PROGRESS],
        handled=counts[NotificationStatus.HANDLED],
        errored=counts[NotificationStatus.ERRORED],
        confirmed=counts[NotificationStatus.CONFIRMED],
        parked_errored=len(parked_items),
    )


def _heartbeat_stale_after_seconds(payload: dict[str, Any]) -> float:
    raw = payload.get("heartbeat_interval_s")
    try:
        interval = float(raw)
    except (TypeError, ValueError):
        interval = 5.0
    return max(15.0, interval * 3)


def _heartbeat_is_stale(
    payload: dict[str, Any],
    stale_after_seconds: float,
) -> bool:
    updated_at = payload.get("updated_at")
    if not isinstance(updated_at, str):
        return True
    parsed = _parse_datetime(updated_at)
    if parsed is None:
        return True
    return (
        datetime.now(timezone.utc) - parsed
    ).total_seconds() > stale_after_seconds


def _parse_datetime(value: str) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


__all__ = [
    "BrokerStatusSummary",
    "GatewayHeartbeatSummary",
    "GatewayLiveness",
    "InboxItemLocation",
    "InboxItemRecord",
    "NotificationStatusCounts",
    "OperatorStatusSummary",
    "SandboxStatusSummary",
    "ServiceQueueSummary",
    "collect_broker_status",
    "collect_heartbeat_summary",
    "collect_inbox_items",
    "collect_operator_status",
    "collect_sandbox_status",
    "collect_service_queue_summaries",
    "summarize_notification_content",
]
