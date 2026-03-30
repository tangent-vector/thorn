"""Composite and trace-oriented event sinks.

``CompositeEventSink`` fans events out to multiple child sinks.
``JsonLinesSink`` writes structured JSONL traces for post-hoc analysis.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import IO, Any

from thorn._context import EventSink, Scope
from thorn._provider import ResponseChunk


class CompositeEventSink(EventSink):
    """Broadcasts every event to a list of child sinks."""

    def __init__(self, sinks: list[EventSink]) -> None:
        self._sinks = list(sinks)

    async def on_response_chunk(
        self, chunk: ResponseChunk, scope: Scope | None = None,
    ) -> None:
        for sink in self._sinks:
            await sink.on_response_chunk(chunk, scope=scope)

    async def on_status(
        self, message: str, scope: Scope | None = None,
    ) -> None:
        for sink in self._sinks:
            await sink.on_status(message, scope=scope)

    async def on_scope_enter(self, scope: Scope) -> None:
        for sink in self._sinks:
            await sink.on_scope_enter(scope)

    async def on_scope_exit(
        self, scope: Scope, *, duration_s: float | None = None,
    ) -> None:
        for sink in self._sinks:
            await sink.on_scope_exit(scope, duration_s=duration_s)

    async def on_tool_start(
        self, name: str, arguments: dict[str, Any],
        *, scope: Scope | None = None,
    ) -> None:
        for sink in self._sinks:
            await sink.on_tool_start(name, arguments, scope=scope)

    async def on_tool_end(
        self, name: str, *, duration_s: float | None = None,
        error: str | None = None, scope: Scope | None = None,
    ) -> None:
        for sink in self._sinks:
            await sink.on_tool_end(
                name, duration_s=duration_s, error=error, scope=scope,
            )

    async def on_completion_end(
        self, *, duration_s: float | None = None,
        usage: dict[str, int] | None = None,
        scope: Scope | None = None,
    ) -> None:
        for sink in self._sinks:
            await sink.on_completion_end(
                duration_s=duration_s, usage=usage, scope=scope,
            )


class JsonLinesSink(EventSink):
    """Writes one JSON object per event to a file handle.

    Each record includes an ISO-8601 timestamp, the event type, the
    scope chain (as a list of description strings), and event-specific
    fields.  The file is flushed after every write so partial traces
    are always readable.
    """

    def __init__(self, file: IO[str]) -> None:
        self._file = file

    def _write(self, event: str, scope: Scope | None, **fields: Any) -> None:
        record: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": event,
            "scope": [s.description for s in scope.chain()] if scope else [],
        }
        record.update(fields)
        self._file.write(json.dumps(record, default=str) + "\n")
        self._file.flush()

    # -- abstract required -------------------------------------------------

    async def on_response_chunk(
        self, chunk: ResponseChunk, scope: Scope | None = None,
    ) -> None:
        from thorn._provider import (
            FinishChunk, TextChunk, ToolCallChunk, UsageChunk,
        )

        match chunk:
            case TextChunk():
                self._write("text_chunk", scope, text=chunk.text)
            case ToolCallChunk():
                self._write(
                    "tool_call_chunk", scope,
                    name=chunk.name, call_id=chunk.call_id,
                    arguments=chunk.arguments,
                )
            case UsageChunk():
                self._write(
                    "usage_chunk", scope,
                    prompt_tokens=chunk.prompt_tokens,
                    completion_tokens=chunk.completion_tokens,
                    total_tokens=chunk.total_tokens,
                )
            case FinishChunk():
                self._write("finish_chunk", scope, reason=chunk.reason)

    async def on_status(
        self, message: str, scope: Scope | None = None,
    ) -> None:
        self._write("status", scope, message=message)

    # -- typed events ------------------------------------------------------

    async def on_scope_enter(self, scope: Scope) -> None:
        self._write("scope_enter", scope, description=scope.description)

    async def on_scope_exit(
        self, scope: Scope, *, duration_s: float | None = None,
    ) -> None:
        self._write(
            "scope_exit", scope,
            description=scope.description, duration_s=duration_s,
        )

    async def on_tool_start(
        self, name: str, arguments: dict[str, Any],
        *, scope: Scope | None = None,
    ) -> None:
        self._write("tool_start", scope, name=name, arguments=arguments)

    async def on_tool_end(
        self, name: str, *, duration_s: float | None = None,
        error: str | None = None, scope: Scope | None = None,
    ) -> None:
        self._write(
            "tool_end", scope,
            name=name, duration_s=duration_s, error=error,
        )

    async def on_completion_end(
        self, *, duration_s: float | None = None,
        usage: dict[str, int] | None = None,
        scope: Scope | None = None,
    ) -> None:
        self._write(
            "completion_end", scope,
            duration_s=duration_s, **(usage or {}),
        )
