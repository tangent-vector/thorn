"""Tests for structured event trace sinks."""

from __future__ import annotations

import io
import json
from typing import Any

import pytest

from thorn.core._context import NullEventSink, Scope
from thorn.core._prompt_trace import (
    PromptTraceArtifact,
    PromptTraceCapture,
    PromptTraceContextSource,
    PromptTraceManifest,
    PromptTraceRecorder,
)
from thorn.core._provider_telemetry import (
    ProviderAttemptNextAction,
    ProviderAttemptOutcome,
    ProviderAttemptTelemetry,
    ProviderContextMetrics,
)
from thorn.core._trace import CompositeEventSink, JsonLinesSink
from thorn.core._validation_convergence import (
    ValidationActionEpochReason,
    ValidationActionTool,
    ValidationConvergencePolicy,
    ValidationConvergenceTracker,
    ValidationOutcome,
    WorkspaceContentConvergencePolicy,
    WorkspaceContentConvergenceTracker,
    parse_validation_command,
)
from thorn.core._workspace_content import (
    WorkspaceContentIdentity,
    WorkspaceContentSnapshot,
)
from thorn.core.errors import ProviderFailureKind
from thorn.runtime._working_set import HandlingPhase, WorkingSet
from thorn.runtime._working_set_telemetry import (
    WorkingSetGateTelemetry,
    WorkingSetTelemetryKind,
    WorkingSetTodoTelemetry,
    build_working_set_telemetry,
)


def _jsonl_records(buffer: io.StringIO) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in buffer.getvalue().splitlines()
        if line.strip()
    ]


def _context_metrics() -> ProviderContextMetrics:
    return ProviderContextMetrics(
        system_prompt_count=1,
        tool_schema_count=0,
        message_count=1,
        history_node_count=1,
        context_window=None,
        estimated_prompt_tokens=10,
        estimated_history_tokens=5,
        estimated_overhead_tokens=5,
        high_watermark_tokens=None,
    )


def test_prompt_trace_recorder_redacts_sidecar_payload_by_default(tmp_path):
    recorder = PromptTraceRecorder(tmp_path)

    artifact = recorder.record(
        request_id="req-redacted",
        provider_name="MockProvider",
        model_name=None,
        provider_payload={
            "messages": [
                {"role": "user", "content": "token=glpat-secretvalue"},
            ],
        },
        context=_context_metrics(),
        manifest=None,
    )

    assert artifact is not None
    sidecar = json.loads(artifact.artifact_path.read_text(encoding="utf-8"))
    encoded = json.dumps(sidecar)
    assert "glpat-secretvalue" not in encoded
    assert "<redacted>" in encoded
    assert sidecar["capture_mode"] == "redacted"
    assert sidecar["redaction"]["applied"] is True


def test_prompt_trace_recorder_raw_mode_preserves_sidecar_payload(tmp_path):
    recorder = PromptTraceRecorder(
        tmp_path,
        capture_mode=PromptTraceCapture.RAW,
    )

    artifact = recorder.record(
        request_id="req-raw",
        provider_name="MockProvider",
        model_name=None,
        provider_payload={
            "messages": [
                {"role": "user", "content": "token=glpat-secretvalue"},
            ],
        },
        context=_context_metrics(),
        manifest=None,
    )

    assert artifact is not None
    sidecar = json.loads(artifact.artifact_path.read_text(encoding="utf-8"))
    encoded = json.dumps(sidecar)
    assert "glpat-secretvalue" in encoded
    assert sidecar["capture_mode"] == "raw"
    assert sidecar["redaction"]["applied"] is False


@pytest.mark.asyncio
async def test_json_lines_sink_records_advisory_as_structured_event():
    buffer = io.StringIO()
    sink = JsonLinesSink(buffer)
    outer_scope = Scope(description="agent")
    scope = Scope(description="tool", outer=outer_scope)

    await sink.on_advisory(
        "validation",
        "[build: stale]",
        scope=scope,
    )

    records = _jsonl_records(buffer)
    assert len(records) == 1
    record = records[0]
    assert isinstance(record.pop("timestamp"), str)
    assert record == {
        "event": "advisory",
        "scope": ["agent", "tool"],
        "source": "validation",
        "content": "[build: stale]",
    }


@pytest.mark.asyncio
async def test_json_lines_sink_records_prior_context_hint_telemetry():
    buffer = io.StringIO()
    sink = JsonLinesSink(buffer)
    scope = Scope(description="agent")

    await sink.on_prior_context_hint_telemetry(
        tool_name="search_files",
        hint_kind="search_files_exact_duplicate_result",
        hint_emitted=True,
        details={
            "search_key_hash": "key",
            "result_hash": "result",
            "prior_call_id": "c1",
        },
        scope=scope,
    )

    records = _jsonl_records(buffer)
    assert len(records) == 1
    record = records[0]
    assert isinstance(record.pop("timestamp"), str)
    assert record == {
        "event": "prior_context_hint",
        "scope": ["agent"],
        "tool_name": "search_files",
        "hint_kind": "search_files_exact_duplicate_result",
        "hint_emitted": True,
        "details": {
            "search_key_hash": "key",
            "result_hash": "result",
            "prior_call_id": "c1",
        },
    }


@pytest.mark.asyncio
async def test_composite_event_sink_preserves_advisory_event_shape():
    buffer = io.StringIO()
    recording_sink = _RecordingAdvisorySink()
    sink = CompositeEventSink([
        recording_sink,
        JsonLinesSink(buffer),
    ])

    await sink.on_advisory("inbox", "[pending work]")

    assert recording_sink.advisories == [("inbox", "[pending work]", None)]
    records = _jsonl_records(buffer)
    assert len(records) == 1
    record = records[0]
    assert record["event"] == "advisory"
    assert record["source"] == "inbox"
    assert record["content"] == "[pending work]"


@pytest.mark.asyncio
async def test_json_lines_sink_records_provider_attempt():
    buffer = io.StringIO()
    sink = JsonLinesSink(buffer)
    scope = Scope(description="agent", metadata={"session_key": "s1"})

    await sink.on_provider_attempt(
        ProviderAttemptTelemetry(
            request_id="req-1",
            attempt_number=2,
            provider_name="OpenAIProvider",
            model_name="gpt-test",
            outcome=ProviderAttemptOutcome.TRANSIENT_ERROR,
            next_action=ProviderAttemptNextAction.RETRY,
            duration_s=45.0,
            time_to_first_chunk_s=None,
            context=ProviderContextMetrics(
                system_prompt_count=3,
                tool_schema_count=8,
                message_count=5,
                history_node_count=4,
                context_window=128000,
                estimated_prompt_tokens=25000,
                estimated_history_tokens=12000,
                estimated_overhead_tokens=13000,
                high_watermark_tokens=102400,
            ),
            retry_delay_s=4.5,
            retry_after_s=None,
            failure_kind=ProviderFailureKind.READ_TIMEOUT,
            status_code=None,
            error_type="TransientProviderError",
            error_message="transport error talking to provider: timeout",
        ),
        scope=scope,
    )

    records = _jsonl_records(buffer)
    assert len(records) == 1
    record = records[0]
    assert record["event"] == "provider_attempt"
    assert record["scope"] == ["agent"]
    assert record["outcome"] == "transient_error"
    assert record["next_action"] == "retry"
    assert record["failure_kind"] == "read_timeout"
    assert record["context"]["estimated_prompt_tokens"] == 25000
    assert record["context"]["context_window"] == 128000


@pytest.mark.asyncio
async def test_json_lines_sink_records_prompt_trace_artifact():
    buffer = io.StringIO()
    sink = JsonLinesSink(buffer)
    scope = Scope(description="agent", metadata={"session_key": "s1"})

    await sink.on_prompt_trace(
        PromptTraceArtifact(
            request_id="req-1",
            provider_name="OpenAIProvider",
            model_name="model-a",
            capture_mode=PromptTraceCapture.REDACTED,
            artifact_path="/tmp/trace.jsonl.prompts/req-1.json",
            context=ProviderContextMetrics(
                system_prompt_count=1,
                tool_schema_count=2,
                message_count=3,
                history_node_count=4,
                context_window=100000,
                estimated_prompt_tokens=500,
                estimated_history_tokens=300,
                estimated_overhead_tokens=200,
                high_watermark_tokens=80000,
            ),
            manifest=PromptTraceManifest(system_prompt_sources=[
                PromptTraceContextSource.from_text(
                    surface="agents_md",
                    label="Agent guidance from `/repo/AGENTS.md`",
                    text="policy",
                    source_path="/repo/AGENTS.md",
                    directory_kind="agent_workspace",
                    system_prompt_index=0,
                ),
            ]),
        ),
        scope=scope,
    )

    records = _jsonl_records(buffer)
    assert len(records) == 1
    record = records[0]
    assert record["event"] == "prompt_trace"
    assert record["request_id"] == "req-1"
    assert record["capture_mode"] == "redacted"
    assert record["artifact_path"].endswith("req-1.json")
    source = record["manifest"]["system_prompt_sources"][0]
    assert source["surface"] == "agents_md"
    assert source["source_path"] == "/repo/AGENTS.md"


@pytest.mark.asyncio
async def test_json_lines_sink_records_working_set_telemetry():
    buffer = io.StringIO()
    sink = JsonLinesSink(buffer)
    scope = Scope(description="agent", metadata={"session_key": "s1"})

    await sink.on_working_set_telemetry(
        build_working_set_telemetry(
            kind=WorkingSetTelemetryKind.GATE_INTERVENTION,
            working_set=WorkingSet(phase=HandlingPhase.INSPECT),
            todo=WorkingSetTodoTelemetry(open_count=2, total_count=3),
            gate=WorkingSetGateTelemetry(
                name="action_summary_required_for_validate",
                reason="action_summary is required before moving to validate.",
            ),
        ),
        scope=scope,
    )

    records = _jsonl_records(buffer)
    assert len(records) == 1
    record = records[0]
    assert record["event"] == "working_set_telemetry"
    assert record["kind"] == "gate_intervention"
    assert record["phase"] == "inspect"
    assert record["todo"]["open_count"] == 2
    assert record["gate"]["name"] == "action_summary_required_for_validate"


@pytest.mark.asyncio
async def test_json_lines_sink_records_privacy_safe_validation_convergence():
    buffer = io.StringIO()
    sink = JsonLinesSink(buffer)
    command = parse_validation_command(
        json.dumps({"command": "pytest secret/customer_case.py"}),
    )
    assert command is not None
    telemetry = ValidationConvergenceTracker().observe(
        command,
        ValidationOutcome.SUCCEEDED,
        call_id="validation-call",
        render_id="provider-render",
        policy=ValidationConvergencePolicy.BASELINE,
    ).telemetry

    await sink.on_validation_convergence(telemetry)

    records = _jsonl_records(buffer)
    assert len(records) == 1
    record = records[0]
    assert record["event"] == "validation_convergence"
    assert record["telemetry_schema_version"] == 2
    assert record["call_id"] == "validation-call"
    assert record["render_id"] == "provider-render"
    assert record["workspace_action_epoch"] == 0
    assert record["runner"] == "pytest"
    assert record["validation_scope"] == "targeted"
    assert record["outcome"] == "succeeded"
    assert record["prior_outcome"] is None
    assert record["decision"] == "first_in_action_epoch"
    assert record["progress_effect"] == "counts_as_progress"
    assert record["policy_effect_applied"] is False
    encoded = json.dumps(record)
    assert "secret" not in encoded
    assert "customer_case" not in encoded
    assert "command" not in record
    assert "output" not in record


@pytest.mark.asyncio
async def test_json_lines_sink_records_privacy_safe_workspace_content_convergence():
    buffer = io.StringIO()
    sink = JsonLinesSink(buffer)
    command = parse_validation_command(
        json.dumps({"command": "pytest secret/customer_case.py"}),
    )
    assert command is not None
    telemetry = WorkspaceContentConvergenceTracker().observe(
        command,
        ValidationOutcome.SUCCEEDED,
        WorkspaceContentSnapshot.known(
            identity=WorkspaceContentIdentity("a" * 64),
            path_count=3,
            content_bytes=120,
        ),
        call_id="validation-call",
        render_id="provider-render",
        policy=WorkspaceContentConvergencePolicy.OBSERVE_V2,
    ).telemetry

    await sink.on_validation_convergence(telemetry)

    records = _jsonl_records(buffer)
    assert len(records) == 1
    record = records[0]
    assert record["event"] == "validation_convergence"
    assert record["telemetry_schema_version"] == 3
    assert record["workspace_content_epoch"] == 0
    assert record["workspace_content_identity"] == "a" * 64
    assert record["prior_workspace_content_identity"] is None
    assert record["workspace_content_transition"] == "known_changed"
    assert record["workspace_content_collection_failure"] is None
    assert record["workspace_content_path_count"] == 3
    assert record["workspace_content_bytes"] == 120
    assert record["runner"] == "pytest"
    assert record["policy_effect_applied"] is False
    encoded = json.dumps(record)
    assert "secret" not in encoded
    assert "customer_case" not in encoded
    assert "command" not in record
    assert "output" not in record


@pytest.mark.asyncio
async def test_json_lines_sink_records_privacy_safe_action_epoch_advance():
    buffer = io.StringIO()
    sink = JsonLinesSink(buffer)
    telemetry = ValidationConvergenceTracker().advance_workspace_action_epoch(
        call_id="opaque-call",
        render_id="provider-render",
        reason=ValidationActionEpochReason.OPAQUE_SHELL_POSSIBLE_MUTATION,
        tool_name=ValidationActionTool.RUN_SHELL,
    )

    await sink.on_validation_action_epoch(telemetry)

    records = _jsonl_records(buffer)
    assert records == [
        {
            "timestamp": records[0]["timestamp"],
            "event": "validation_action_epoch",
            "scope": [],
            "telemetry_schema_version": 2,
            "call_id": "opaque-call",
            "render_id": "provider-render",
            "prior_workspace_action_epoch": 0,
            "workspace_action_epoch": 1,
            "reason": "opaque_shell_possible_mutation",
            "tool_name": "run_shell",
        },
    ]


class _RecordingAdvisorySink(NullEventSink):
    def __init__(self) -> None:
        self.advisories: list[tuple[str, str, Scope | None]] = []

    async def on_advisory(
        self,
        source: str,
        content: str,
        *,
        scope: Scope | None = None,
    ) -> None:
        self.advisories.append((source, content, scope))
