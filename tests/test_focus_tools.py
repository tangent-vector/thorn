"""Tests for the unified inbox/working-set focus tool."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

import pytest

from thorn.core._agent import Agent
from thorn.core._context import (
    ExecutionContext,
    NullEventSink,
    reset_context,
    set_context,
)
from thorn.core._provider import MockProvider
from thorn.core._session import Session
from thorn.runtime import (
    HandlingPhase,
    Runtime,
    SessionAddress,
    SessionInbox,
    ValidationOutcome,
    WorkingSet,
)
from thorn.runtime._address import Address
from thorn.runtime._inbox_tools import complete_focused_work, update_focus
from thorn.runtime._notification import (
    InboxCompletionRationale,
    NotificationSpec,
    NotificationStatus,
)
from thorn.runtime._session import AgentID, SessionKey
from thorn.runtime._working_set_telemetry import (
    WorkingSetTelemetry,
    WorkingSetTelemetryKind,
)

AGENT_ID = AgentID("focus-test-agent")
SESSION_KEY = SessionKey("demo/session")


class RecordingSink(NullEventSink):
    def __init__(self) -> None:
        self.statuses: list[str] = []
        self.working_set_events: list[WorkingSetTelemetry] = []

    async def on_status(self, message: str, scope=None) -> None:
        self.statuses.append(message)

    async def on_working_set_telemetry(
        self,
        telemetry: WorkingSetTelemetry,
        *,
        scope=None,
    ) -> None:
        self.working_set_events.append(telemetry)


@pytest.fixture
def sink() -> RecordingSink:
    return RecordingSink()


@pytest.fixture
def runtime(tmp_path: Path, sink: RecordingSink) -> Runtime:
    return Runtime(
        provider=MockProvider(),
        event_sink=sink,
        workspace_root=tmp_path / "ws",
    )


@pytest.fixture
def agent() -> Agent:
    return Agent(id=AGENT_ID, name="focus tester")


@pytest.fixture
def session(runtime: Runtime, agent: Agent) -> Session:
    session = runtime.get_or_create_session(agent, SESSION_KEY)
    runtime.save_session(session)
    return session


@pytest.fixture
def inbox(runtime: Runtime) -> SessionInbox:
    address = SessionAddress(AGENT_ID, SESSION_KEY)
    inbox = SessionInbox(
        runtime.paths.session_inbox_dir(AGENT_ID, SESSION_KEY),
        address,
        in_flight_index=runtime.in_flight_index,
    )
    runtime.address_book.register(address, inbox)
    return inbox


@pytest.fixture
def scoped_ctx(
    runtime: Runtime,
    agent: Agent,
    session: Session,
) -> Iterator[ExecutionContext]:
    base = runtime.create_context()
    scoped = base.push_scope(
        "focus-test-scope",
        agent=agent,
        session=session,
        session_key=str(SESSION_KEY),
    )
    token = set_context(scoped)
    try:
        yield scoped
    finally:
        reset_context(token)


def _spec(content: str = "work item") -> NotificationSpec:
    target: Address = SessionAddress(AGENT_ID, SESSION_KEY)
    return NotificationSpec(
        source="test",
        content=content,
        target=target,
    )


def _completion() -> InboxCompletionRationale:
    return InboxCompletionRationale(
        completed_actions=("finished the requested change",),
        request_coverage=("checked the inbox request",),
        self_review="reviewed the final state",
    )


def _reload_working_set(runtime: Runtime, agent: Agent) -> WorkingSet:
    return runtime.get_or_create_session(agent, SESSION_KEY).working_set


class TestUpdateFocus:
    async def test_start_pending_item_sets_inbox_and_working_set(
        self,
        scoped_ctx: ExecutionContext,
        runtime: Runtime,
        agent: Agent,
        inbox: SessionInbox,
    ) -> None:
        item = inbox.post(_spec("investigate this"))

        result = await update_focus(
            phase="inspect",
            item_id=item.id,
            objective="Understand the report before editing.",
            notes="starting investigation",
        )

        assert "Focus updated" in result
        updated = inbox.get(item.id)
        assert updated.status is NotificationStatus.IN_PROGRESS
        assert updated.notes == "starting investigation"
        working_set = _reload_working_set(runtime, agent)
        assert working_set.phase is HandlingPhase.INSPECT
        assert working_set.focused_inbox_item_id == item.id
        assert working_set.objective == "Understand the report before editing."

    async def test_same_focus_phase_change_allows_omitting_item_id(
        self,
        scoped_ctx: ExecutionContext,
        runtime: Runtime,
        agent: Agent,
        session: Session,
        inbox: SessionInbox,
    ) -> None:
        item = inbox.post(_spec("act next"))
        inbox.update_status(item.id, NotificationStatus.IN_PROGRESS)
        session.working_set = WorkingSet(
            phase=HandlingPhase.INSPECT,
            focused_inbox_item_id=item.id,
            objective="Fix the reported problem.",
        )
        runtime.save_session(session)

        result = await update_focus(phase="act")

        assert "phase=act" in result
        working_set = _reload_working_set(runtime, agent)
        assert working_set.phase is HandlingPhase.ACT
        assert working_set.focused_inbox_item_id == item.id
        assert working_set.objective == "Fix the reported problem."

    async def test_resume_already_in_progress_item_repairs_missing_focus(
        self,
        scoped_ctx: ExecutionContext,
        runtime: Runtime,
        agent: Agent,
        inbox: SessionInbox,
    ) -> None:
        item = inbox.post(_spec("resume"))
        inbox.update_status(item.id, NotificationStatus.IN_PROGRESS)

        result = await update_focus(
            phase="inspect",
            item_id=item.id,
            objective="Resume the existing item.",
        )

        assert "Focus updated" in result
        assert inbox.get(item.id).status is NotificationStatus.IN_PROGRESS
        working_set = _reload_working_set(runtime, agent)
        assert working_set.focused_inbox_item_id == item.id

    async def test_invalid_inbox_id_is_rejected_without_focus_change(
        self,
        scoped_ctx: ExecutionContext,
        runtime: Runtime,
        agent: Agent,
        inbox: SessionInbox,
    ) -> None:
        result = await update_focus(
            phase="inspect",
            item_id="NO-SUCH-ID",
            objective="Investigate something.",
        )

        assert "Error" in result
        assert _reload_working_set(runtime, agent) == WorkingSet()

    async def test_objective_is_required_before_claiming_active_work(
        self,
        scoped_ctx: ExecutionContext,
        runtime: Runtime,
        agent: Agent,
        inbox: SessionInbox,
    ) -> None:
        item = inbox.post(_spec("needs an objective"))

        result = await update_focus(phase="inspect", item_id=item.id)

        assert "Error" in result
        assert "objective is required" in result
        assert inbox.get(item.id).status is NotificationStatus.PENDING
        assert _reload_working_set(runtime, agent) == WorkingSet()

    async def test_objective_gate_can_be_overridden(
        self,
        scoped_ctx: ExecutionContext,
        runtime: Runtime,
        agent: Agent,
        inbox: SessionInbox,
    ) -> None:
        item = inbox.post(_spec("objective later"))

        result = await update_focus(
            phase="inspect",
            item_id=item.id,
            override_rationale="operator asked for triage before objective",
        )

        assert "Focus updated" in result
        working_set = _reload_working_set(runtime, agent)
        assert working_set.phase is HandlingPhase.INSPECT
        assert working_set.override_rationale == (
            "operator asked for triage before objective"
        )

    async def test_item_id_is_required_when_leaving_intake(
        self,
        scoped_ctx: ExecutionContext,
        runtime: Runtime,
        agent: Agent,
        inbox: SessionInbox,
    ) -> None:
        result = await update_focus(
            phase="inspect",
            objective="Need an item before work can start.",
        )

        assert "Error" in result
        assert "item_id is required" in result
        assert _reload_working_set(runtime, agent) == WorkingSet()

    async def test_partial_blocker_fields_are_rejected_before_claiming(
        self,
        scoped_ctx: ExecutionContext,
        runtime: Runtime,
        agent: Agent,
        inbox: SessionInbox,
    ) -> None:
        item = inbox.post(_spec("blocked soon"))

        result = await update_focus(
            phase="blocked",
            item_id=item.id,
            objective="Investigate why this is blocked.",
            blocker_summary="missing credentials",
        )

        assert "Error" in result
        assert "blocker_summary and unblock_condition" in result
        assert inbox.get(item.id).status is NotificationStatus.PENDING
        assert _reload_working_set(runtime, agent) == WorkingSet()

    async def test_blocked_requires_blocker_fields_before_claiming(
        self,
        scoped_ctx: ExecutionContext,
        runtime: Runtime,
        agent: Agent,
        inbox: SessionInbox,
    ) -> None:
        item = inbox.post(_spec("blocked soon"))

        result = await update_focus(
            phase="blocked",
            item_id=item.id,
            objective="Wait for credentials.",
        )

        assert "Error" in result
        assert "blocker_summary and unblock_condition" in result
        assert inbox.get(item.id).status is NotificationStatus.PENDING
        assert _reload_working_set(runtime, agent) == WorkingSet()

    async def test_blocked_records_blocker_fields(
        self,
        scoped_ctx: ExecutionContext,
        runtime: Runtime,
        agent: Agent,
        inbox: SessionInbox,
    ) -> None:
        item = inbox.post(_spec("blocked"))

        result = await update_focus(
            phase="blocked",
            item_id=item.id,
            objective="Wait for credentials.",
            blocker_summary="missing API token",
            unblock_condition="operator configures the token",
        )

        assert "Focus updated" in result
        working_set = _reload_working_set(runtime, agent)
        assert working_set.phase is HandlingPhase.BLOCKED
        assert working_set.blocker is not None
        assert working_set.blocker.summary == "missing API token"

    async def test_switching_focus_requires_override(
        self,
        scoped_ctx: ExecutionContext,
        runtime: Runtime,
        agent: Agent,
        session: Session,
        inbox: SessionInbox,
    ) -> None:
        first = inbox.post(_spec("first"))
        second = inbox.post(_spec("second"))
        inbox.update_status(first.id, NotificationStatus.IN_PROGRESS)
        session.working_set = WorkingSet(
            phase=HandlingPhase.ACT,
            focused_inbox_item_id=first.id,
            objective="Finish the first item.",
        )
        runtime.save_session(session)

        result = await update_focus(
            phase="inspect",
            item_id=second.id,
            objective="Switch to second item.",
        )

        assert "Error" in result
        assert "override_rationale" in result
        assert inbox.get(second.id).status is NotificationStatus.PENDING
        assert _reload_working_set(runtime, agent).focused_inbox_item_id == first.id

    async def test_switching_focus_with_override_claims_new_item(
        self,
        scoped_ctx: ExecutionContext,
        runtime: Runtime,
        agent: Agent,
        session: Session,
        inbox: SessionInbox,
        sink: RecordingSink,
    ) -> None:
        first = inbox.post(_spec("first"))
        second = inbox.post(_spec("second"))
        inbox.update_status(first.id, NotificationStatus.IN_PROGRESS)
        session.working_set = WorkingSet(
            phase=HandlingPhase.ACT,
            focused_inbox_item_id=first.id,
            objective="Finish the first item.",
        )
        runtime.save_session(session)

        result = await update_focus(
            phase="inspect",
            item_id=second.id,
            objective="Switch to second item.",
            override_rationale="operator asked for urgent new work",
        )

        assert "Focus updated" in result
        assert inbox.get(second.id).status is NotificationStatus.IN_PROGRESS
        working_set = _reload_working_set(runtime, agent)
        assert working_set.focused_inbox_item_id == second.id
        assert working_set.override_rationale == (
            "operator asked for urgent new work"
        )
        assert any("outside working-set focus" in msg for msg in sink.statuses)

    async def test_validate_requires_action_summary_without_override(
        self,
        scoped_ctx: ExecutionContext,
        runtime: Runtime,
        agent: Agent,
        session: Session,
        inbox: SessionInbox,
        sink: RecordingSink,
    ) -> None:
        item = inbox.post(_spec("validate"))
        inbox.update_status(item.id, NotificationStatus.IN_PROGRESS)
        session.working_set = WorkingSet(
            phase=HandlingPhase.ACT,
            focused_inbox_item_id=item.id,
            objective="Fix the bug.",
        )
        runtime.save_session(session)

        result = await update_focus(phase="validate")

        assert "Error" in result
        assert "action_summary is required" in result
        assert _reload_working_set(runtime, agent).phase is HandlingPhase.ACT
        telemetry = sink.working_set_events[-1]
        assert telemetry.kind is WorkingSetTelemetryKind.GATE_INTERVENTION
        assert telemetry.gate is not None
        assert telemetry.gate.name == "action_summary_required_for_validate"
        assert telemetry.phase == "act"

    async def test_validate_records_action_summary(
        self,
        scoped_ctx: ExecutionContext,
        runtime: Runtime,
        agent: Agent,
        session: Session,
        inbox: SessionInbox,
        sink: RecordingSink,
    ) -> None:
        item = inbox.post(_spec("validate"))
        inbox.update_status(item.id, NotificationStatus.IN_PROGRESS)
        session.working_set = WorkingSet(
            phase=HandlingPhase.ACT,
            focused_inbox_item_id=item.id,
            objective="Fix the bug.",
        )
        runtime.save_session(session)

        result = await update_focus(
            phase="validate",
            action_summary="patched the parser",
        )

        assert "phase=validate" in result
        working_set = _reload_working_set(runtime, agent)
        assert working_set.phase is HandlingPhase.VALIDATE
        assert working_set.last_action_summary == "patched the parser"
        telemetry = sink.working_set_events[-1]
        assert telemetry.kind is WorkingSetTelemetryKind.FOCUS_UPDATED
        assert telemetry.phase == "validate"
        assert telemetry.last_action_estimated_tokens is not None

    async def test_validate_allows_validation_only_override(
        self,
        scoped_ctx: ExecutionContext,
        runtime: Runtime,
        agent: Agent,
        session: Session,
        inbox: SessionInbox,
    ) -> None:
        item = inbox.post(_spec("validate only"))
        inbox.update_status(item.id, NotificationStatus.IN_PROGRESS)
        session.working_set = WorkingSet(
            phase=HandlingPhase.INSPECT,
            focused_inbox_item_id=item.id,
            objective="Confirm the existing fix.",
        )
        runtime.save_session(session)

        result = await update_focus(
            phase="validate",
            override_rationale="operator asked for validation only",
        )

        assert "phase=validate" in result
        working_set = _reload_working_set(runtime, agent)
        assert working_set.phase is HandlingPhase.VALIDATE
        assert working_set.override_rationale == "operator asked for validation only"

    async def test_closeout_phase_does_not_mark_item_handled(
        self,
        scoped_ctx: ExecutionContext,
        runtime: Runtime,
        agent: Agent,
        session: Session,
        inbox: SessionInbox,
    ) -> None:
        item = inbox.post(_spec("close out soon"))
        inbox.update_status(item.id, NotificationStatus.IN_PROGRESS)
        session.working_set = WorkingSet(
            phase=HandlingPhase.VALIDATE,
            focused_inbox_item_id=item.id,
            objective="Finish and close out.",
        )
        runtime.save_session(session)

        result = await update_focus(
            phase="closeout",
            no_validation_rationale="manual review only",
        )

        assert "phase=closeout" in result
        assert inbox.get(item.id).status is NotificationStatus.IN_PROGRESS
        assert _reload_working_set(runtime, agent).phase is HandlingPhase.CLOSEOUT

    async def test_closeout_requires_validation_or_no_validation(
        self,
        scoped_ctx: ExecutionContext,
        runtime: Runtime,
        agent: Agent,
        session: Session,
        inbox: SessionInbox,
    ) -> None:
        item = inbox.post(_spec("closeout blocked"))
        inbox.update_status(item.id, NotificationStatus.IN_PROGRESS)
        session.working_set = WorkingSet(
            phase=HandlingPhase.VALIDATE,
            focused_inbox_item_id=item.id,
            objective="Finish and close out.",
            last_action_summary="patched the code",
        )
        runtime.save_session(session)

        result = await update_focus(phase="closeout")

        assert "Error" in result
        assert "closeout requires validation evidence" in result
        assert _reload_working_set(runtime, agent).phase is HandlingPhase.VALIDATE

    async def test_closeout_records_validation_evidence(
        self,
        scoped_ctx: ExecutionContext,
        runtime: Runtime,
        agent: Agent,
        session: Session,
        inbox: SessionInbox,
        sink: RecordingSink,
    ) -> None:
        item = inbox.post(_spec("closeout validated"))
        inbox.update_status(item.id, NotificationStatus.IN_PROGRESS)
        session.working_set = WorkingSet(
            phase=HandlingPhase.VALIDATE,
            focused_inbox_item_id=item.id,
            objective="Finish and close out.",
            last_action_summary="patched the code",
        )
        runtime.save_session(session)

        result = await update_focus(
            phase="closeout",
            validation_outcome="passed",
            validation_summary="targeted tests passed",
            validation_command="uv run pytest tests/test_focus_tools.py -q",
        )

        assert "phase=closeout" in result
        working_set = _reload_working_set(runtime, agent)
        assert working_set.phase is HandlingPhase.CLOSEOUT
        assert working_set.last_validation is not None
        assert working_set.last_validation.outcome is ValidationOutcome.PASSED
        assert working_set.last_validation.summary == "targeted tests passed"
        telemetry = sink.working_set_events[-1]
        assert telemetry.kind is WorkingSetTelemetryKind.FOCUS_UPDATED
        assert telemetry.phase == "closeout"
        assert telemetry.validation.outcome is ValidationOutcome.PASSED

    async def test_handled_focused_item_resets_working_set_to_intake(
        self,
        scoped_ctx: ExecutionContext,
        runtime: Runtime,
        agent: Agent,
        session: Session,
        inbox: SessionInbox,
    ) -> None:
        item = inbox.post(_spec("done"))
        inbox.update_status(item.id, NotificationStatus.IN_PROGRESS)
        session.working_set = WorkingSet(
            phase=HandlingPhase.CLOSEOUT,
            focused_inbox_item_id=item.id,
            objective="Close this item.",
            no_validation_rationale="operator requested closeout without tests",
        )
        runtime.save_session(session)

        result = await complete_focused_work(completion=_completion())

        assert "handled" in result
        assert _reload_working_set(runtime, agent) == WorkingSet()

    async def test_handled_focused_item_requires_closeout_evidence(
        self,
        scoped_ctx: ExecutionContext,
        runtime: Runtime,
        agent: Agent,
        session: Session,
        inbox: SessionInbox,
    ) -> None:
        item = inbox.post(_spec("not ready"))
        inbox.update_status(item.id, NotificationStatus.IN_PROGRESS)
        session.working_set = WorkingSet(
            phase=HandlingPhase.CLOSEOUT,
            focused_inbox_item_id=item.id,
            objective="Close this item.",
        )
        runtime.save_session(session)

        result = await complete_focused_work(completion=_completion())

        assert "Error" in result
        assert "validation evidence" in result
        assert inbox.get(item.id).status is NotificationStatus.IN_PROGRESS
        assert _reload_working_set(runtime, agent).phase is HandlingPhase.CLOSEOUT


class TestExceptionalRendering:
    async def test_prompt_renders_in_progress_items_outside_focus(
        self,
        runtime: Runtime,
        agent: Agent,
        session: Session,
        inbox: SessionInbox,
    ) -> None:
        focused = inbox.post(_spec("focused"))
        extra = inbox.post(_spec("not focused"))
        inbox.update_status(focused.id, NotificationStatus.IN_PROGRESS)
        inbox.update_status(extra.id, NotificationStatus.IN_PROGRESS)
        session.working_set = WorkingSet(
            phase=HandlingPhase.INSPECT,
            focused_inbox_item_id=focused.id,
            objective="Work on the focused item.",
        )
        runtime.save_session(session)

        provider = runtime.provider
        captured_prompts: list[list[str]] = []
        original_complete = provider.complete

        async def tracking_complete(
            system_prompts: list[str],
            tools: list[dict],
            messages: list[Any],
        ):
            captured_prompts.append(list(system_prompts))
            async for chunk in original_complete(system_prompts, tools, messages):
                yield chunk

        provider.complete = tracking_complete  # type: ignore[assignment]

        context = runtime.create_context()
        token = set_context(context)
        try:
            await session.prompt("continue")
        finally:
            reset_context(token)

        assert len(captured_prompts) == 1
        joined = "\n".join(captured_prompts[0])
        assert "In-progress inbox items outside focus:" in joined
        assert str(extra.id) in joined
