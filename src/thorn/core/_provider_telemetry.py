"""Structured telemetry emitted around LLM provider calls."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from thorn.core.errors import ProviderFailureKind

if TYPE_CHECKING:
    from thorn.core._context_ledger import ProviderContextLedger


class ProviderAttemptOutcome(StrEnum):
    """Final outcome of one provider-call attempt."""

    SUCCESS = "success"
    RATE_LIMITED = "rate_limited"
    TRANSIENT_ERROR = "transient_error"
    PROVIDER_ERROR = "provider_error"


class ProviderAttemptNextAction(StrEnum):
    """What the retry loop will do after recording an attempt."""

    NONE = "none"
    RETRY = "retry"
    RAISE_PROVIDER_UNAVAILABLE = "raise_provider_unavailable"
    RAISE_AGENT_FAILURE = "raise_agent_failure"


@dataclass(frozen=True)
class ProviderContextMetrics:
    """Context-size metrics associated with a provider request."""

    system_prompt_count: int
    tool_schema_count: int
    message_count: int
    history_node_count: int | None
    context_window: int | None
    estimated_prompt_tokens: int
    estimated_history_tokens: int
    estimated_overhead_tokens: int
    high_watermark_tokens: int | None
    history_ledger: ProviderContextLedger | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None

    def with_usage(self, usage: dict[str, int] | None) -> "ProviderContextMetrics":
        """Return a copy enriched with provider-reported token usage."""
        if usage is None:
            return self
        return ProviderContextMetrics(
            system_prompt_count=self.system_prompt_count,
            tool_schema_count=self.tool_schema_count,
            message_count=self.message_count,
            history_node_count=self.history_node_count,
            context_window=self.context_window,
            estimated_prompt_tokens=self.estimated_prompt_tokens,
            estimated_history_tokens=self.estimated_history_tokens,
            estimated_overhead_tokens=self.estimated_overhead_tokens,
            high_watermark_tokens=self.high_watermark_tokens,
            history_ledger=self.history_ledger,
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            total_tokens=usage.get("total_tokens"),
        )

    def to_json(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        result = {
            "system_prompt_count": self.system_prompt_count,
            "tool_schema_count": self.tool_schema_count,
            "message_count": self.message_count,
            "history_node_count": self.history_node_count,
            "context_window": self.context_window,
            "estimated_prompt_tokens": self.estimated_prompt_tokens,
            "estimated_history_tokens": self.estimated_history_tokens,
            "estimated_overhead_tokens": self.estimated_overhead_tokens,
            "high_watermark_tokens": self.high_watermark_tokens,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }
        if self.history_ledger is not None:
            result["history_ledger"] = self.history_ledger.to_json()
        return result


@dataclass(frozen=True)
class ProviderAttemptTelemetry:
    """Structured event for one provider request attempt."""

    request_id: str
    attempt_number: int
    provider_name: str
    model_name: str | None
    outcome: ProviderAttemptOutcome
    next_action: ProviderAttemptNextAction
    duration_s: float
    time_to_first_chunk_s: float | None
    context: ProviderContextMetrics
    retry_delay_s: float | None = None
    retry_after_s: float | None = None
    failure_kind: ProviderFailureKind | None = None
    status_code: int | None = None
    error_type: str | None = None
    error_message: str | None = None

    def to_json(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "request_id": self.request_id,
            "attempt_number": self.attempt_number,
            "provider_name": self.provider_name,
            "model_name": self.model_name,
            "outcome": self.outcome.value,
            "next_action": self.next_action.value,
            "duration_s": self.duration_s,
            "time_to_first_chunk_s": self.time_to_first_chunk_s,
            "context": self.context.to_json(),
            "retry_delay_s": self.retry_delay_s,
            "retry_after_s": self.retry_after_s,
            "failure_kind": (
                self.failure_kind.value if self.failure_kind is not None else None
            ),
            "status_code": self.status_code,
            "error_type": self.error_type,
            "error_message": self.error_message,
        }


__all__ = [
    "ProviderAttemptNextAction",
    "ProviderAttemptOutcome",
    "ProviderAttemptTelemetry",
    "ProviderContextMetrics",
]
