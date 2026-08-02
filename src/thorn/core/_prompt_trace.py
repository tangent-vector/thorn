"""Prompt trace sidecar artifacts for provider-bound requests."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from enum import StrEnum
from hashlib import sha256
from pathlib import Path
from typing import Any

from thorn._redaction import REDACTED_SECRET, redact_secrets
from thorn.core._history import estimate_tokens
from thorn.core._provider_telemetry import ProviderContextMetrics

logger = logging.getLogger(__name__)


class PromptTraceCapture(StrEnum):
    """How much prompt payload content a trace sidecar should preserve."""

    REDACTED = "redacted"
    RAW = "raw"


@dataclass(frozen=True)
class PromptTraceContextSource:
    """One provider-bound system-prompt fragment's provenance summary."""

    surface: str
    label: str
    char_count: int
    estimated_tokens: int
    sha256: str
    system_prompt_index: int | None = None
    source_path: str | None = None
    directory_kind: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_text(
        cls,
        *,
        surface: str,
        label: str,
        text: str,
        source_path: Path | str | None = None,
        directory_kind: str | None = None,
        system_prompt_index: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "PromptTraceContextSource":
        """Build a provenance summary without embedding the source text."""
        return cls(
            surface=surface,
            label=label,
            char_count=len(text),
            estimated_tokens=estimate_tokens(text),
            sha256=sha256(text.encode("utf-8")).hexdigest(),
            system_prompt_index=system_prompt_index,
            source_path=str(source_path) if source_path is not None else None,
            directory_kind=directory_kind,
            metadata=dict(metadata or {}),
        )

    def with_system_prompt_index(self, index: int) -> "PromptTraceContextSource":
        """Return this source tagged with its final provider prompt index."""
        return PromptTraceContextSource(
            surface=self.surface,
            label=self.label,
            char_count=self.char_count,
            estimated_tokens=self.estimated_tokens,
            sha256=self.sha256,
            system_prompt_index=index,
            source_path=self.source_path,
            directory_kind=self.directory_kind,
            metadata=dict(self.metadata),
        )

    def to_json(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "surface": self.surface,
            "label": self.label,
            "system_prompt_index": self.system_prompt_index,
            "source_path": self.source_path,
            "directory_kind": self.directory_kind,
            "char_count": self.char_count,
            "estimated_tokens": self.estimated_tokens,
            "sha256": self.sha256,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class PromptTraceManifest:
    """Machine-readable summary of the context surfaces in a prompt."""

    system_prompt_sources: list[PromptTraceContextSource] = field(
        default_factory=list,
    )
    working_set_telemetry: dict[str, Any] | None = None

    def with_system_prompt_indices(
        self,
        *,
        start_index: int = 0,
    ) -> "PromptTraceManifest":
        """Return a copy whose sources have contiguous final prompt indices."""
        return PromptTraceManifest(
            system_prompt_sources=[
                source.with_system_prompt_index(start_index + offset)
                for offset, source in enumerate(self.system_prompt_sources)
            ],
            working_set_telemetry=self.working_set_telemetry,
        )

    def to_json(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "system_prompt_sources": [
                source.to_json() for source in self.system_prompt_sources
            ],
            "working_set_telemetry": self.working_set_telemetry,
        }


@dataclass(frozen=True)
class PromptTraceArtifact:
    """Small event payload pointing at a prompt trace sidecar file."""

    request_id: str
    provider_name: str
    model_name: str | None
    capture_mode: PromptTraceCapture
    artifact_path: Path
    context: ProviderContextMetrics
    manifest: PromptTraceManifest | None = None

    def to_json(self) -> dict[str, Any]:
        """Return a JSON-serializable event payload."""
        return {
            "request_id": self.request_id,
            "provider_name": self.provider_name,
            "model_name": self.model_name,
            "capture_mode": self.capture_mode.value,
            "artifact_path": str(self.artifact_path),
            "context": self.context.to_json(),
            "manifest": (
                self.manifest.to_json() if self.manifest is not None else None
            ),
        }


class PromptTraceRecorder:
    """Writes large prompt payloads as sidecar files for a JSONL trace."""

    def __init__(
        self,
        artifact_dir: Path,
        *,
        capture_mode: PromptTraceCapture = PromptTraceCapture.REDACTED,
    ) -> None:
        self.artifact_dir = artifact_dir
        self.capture_mode = capture_mode

    @classmethod
    def for_trace_path(
        cls,
        trace_path: Path | str,
        *,
        capture_mode: PromptTraceCapture = PromptTraceCapture.REDACTED,
    ) -> "PromptTraceRecorder":
        """Create a recorder whose sidecars live next to *trace_path*."""
        return cls(
            Path(f"{trace_path}.prompts"),
            capture_mode=capture_mode,
        )

    def record(
        self,
        *,
        request_id: str,
        provider_name: str,
        model_name: str | None,
        provider_payload: Any,
        context: ProviderContextMetrics,
        manifest: PromptTraceManifest | None,
    ) -> PromptTraceArtifact | None:
        """Write one prompt trace artifact.

        Trace capture must not break a live gateway.  I/O failures are
        logged and reported by returning ``None``.
        """
        try:
            self.artifact_dir.mkdir(parents=True, exist_ok=True)
            artifact_path = self.artifact_dir / f"{request_id}.json"
            sidecar = {
                "schema_version": 1,
                "request_id": request_id,
                "provider_name": provider_name,
                "model_name": model_name,
                "capture_mode": self.capture_mode.value,
                "redaction": _redaction_metadata(self.capture_mode),
                "context": context.to_json(),
                "manifest": (
                    manifest.to_json() if manifest is not None else None
                ),
                "provider_payload": _payload_for_capture_mode(
                    provider_payload,
                    self.capture_mode,
                ),
            }
            artifact_path.write_text(
                json.dumps(sidecar, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            return PromptTraceArtifact(
                request_id=request_id,
                provider_name=provider_name,
                model_name=model_name,
                capture_mode=self.capture_mode,
                artifact_path=artifact_path,
                context=context,
                manifest=manifest,
            )
        except OSError:
            logger.warning(
                "failed to write prompt trace artifact for request %s",
                request_id,
                exc_info=True,
            )
            return None


def _redaction_metadata(capture_mode: PromptTraceCapture) -> dict[str, Any]:
    if capture_mode is PromptTraceCapture.RAW:
        return {
            "applied": False,
            "warning": (
                "raw prompt capture may contain secrets, private code, "
                "or sensitive conversation content"
            ),
        }
    return {
        "applied": True,
        "method": "recursive JSON-value redaction with thorn._redaction",
    }


def _payload_for_capture_mode(
    provider_payload: Any,
    capture_mode: PromptTraceCapture,
) -> Any:
    if capture_mode is PromptTraceCapture.RAW:
        return provider_payload
    return _redact_jsonish(provider_payload)


def _redact_jsonish(value: Any) -> Any:
    """Redact credential-shaped content while preserving JSON shape."""
    if isinstance(value, str):
        return redact_secrets(value)
    if isinstance(value, list):
        return [_redact_jsonish(item) for item in value]
    if isinstance(value, tuple):
        return [_redact_jsonish(item) for item in value]
    if isinstance(value, dict):
        redacted: dict[Any, Any] = {}
        for key, item in value.items():
            if isinstance(key, str) and _is_secret_field_name(key):
                redacted[key] = REDACTED_SECRET
            else:
                redacted[key] = _redact_jsonish(item)
        return redacted
    try:
        json.dumps(value)
        return value
    except TypeError:
        return redact_secrets(str(value))


_SECRET_FIELD_NAMES = frozenset({
    "api_key",
    "apikey",
    "access_token",
    "accesstoken",
    "token",
    "secret",
    "password",
    "authorization",
})


def _is_secret_field_name(name: str) -> bool:
    normalized = name.lower().replace("-", "_").replace(".", "_")
    if normalized in _SECRET_FIELD_NAMES:
        return True
    return normalized.endswith((
        "_api_key",
        "_apikey",
        "_access_token",
        "_accesstoken",
        "_token",
        "_secret",
        "_password",
        "_authorization",
    ))


__all__ = [
    "PromptTraceArtifact",
    "PromptTraceCapture",
    "PromptTraceContextSource",
    "PromptTraceManifest",
    "PromptTraceRecorder",
]
