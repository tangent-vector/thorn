"""LLM provider abstraction.

The provider is a thin transport layer.  It does not manage conversation
state, handle tool calls, or implement retry logic; those concerns belong
to the agent loop in ``_loop.py``.
"""

from __future__ import annotations

import email.utils
import json
import logging
import os
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import httpx

from thorn._redaction import redact_secret_snippet
from thorn.core._messages import (
    AssistantMessage,
    Message,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)
from thorn.core.errors import (
    ProviderError,
    RateLimitError,
    TransientProviderError,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Response chunks
# ---------------------------------------------------------------------------

@dataclass
class ResponseChunk:
    """Base class for incremental pieces of an LLM response."""


@dataclass
class TextChunk(ResponseChunk):
    """A fragment of natural-language text from the assistant."""

    text: str


@dataclass
class ToolCallChunk(ResponseChunk):
    """Notification that the assistant is requesting a tool call."""

    call_id: str
    name: str
    arguments: str

    def to_tool_call(self) -> ToolCall:
        return ToolCall(
            call_id=self.call_id,
            name=self.name,
            arguments=self.arguments,
        )


@dataclass
class FinishChunk(ResponseChunk):
    """Signals that the provider has finished producing the response."""

    reason: str


@dataclass
class UsageChunk(ResponseChunk):
    """Token usage statistics returned by the provider."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


# ---------------------------------------------------------------------------
# Provider interface
# ---------------------------------------------------------------------------

class LLMProvider(ABC):
    """Abstract interface to an LLM chat-completion service.

    The provider does *not* own conversation state.  The caller supplies
    the full message history on every call.
    """

    @property
    def context_window(self) -> int | None:
        """The model's maximum context window in tokens, if known.

        Subclasses should override to report the model's actual limit.
        Returns ``None`` when the limit is unknown.
        """
        return None

    @abstractmethod
    async def complete(
        self,
        system_prompts: list[str],
        tools: list[dict],
        messages: list[Message],
    ) -> AsyncIterator[ResponseChunk]:
        """Submit a chat-completion request and stream the response."""
        ...  # pragma: no cover


# ---------------------------------------------------------------------------
# Mock provider (testing)
# ---------------------------------------------------------------------------

class MockProvider(LLMProvider):
    """Deterministic provider for testing.

    Pre-load with canned responses that are yielded in FIFO order.
    When none remain, echoes the last user message.
    """

    def __init__(
        self,
        canned_responses: list[list[ResponseChunk]] | None = None,
        *,
        context_window: int | None = None,
    ) -> None:
        self.canned_responses: list[list[ResponseChunk]] = (
            list(canned_responses) if canned_responses else []
        )
        self._context_window = context_window

    @property
    def context_window(self) -> int | None:
        return self._context_window

    async def complete(
        self,
        system_prompts: list[str],
        tools: list[dict],
        messages: list[Message],
    ) -> AsyncIterator[ResponseChunk]:
        if self.canned_responses:
            chunks = self.canned_responses.pop(0)
        else:
            last_user_text = ""
            for msg in reversed(messages):
                if isinstance(msg, UserMessage):
                    last_user_text = msg.content
                    break
            chunks = [
                TextChunk(text=f"[mock] {last_user_text}"),
                UsageChunk(prompt_tokens=0, completion_tokens=0, total_tokens=0),
                FinishChunk(reason="stop"),
            ]
        for chunk in chunks:
            yield chunk


# ---------------------------------------------------------------------------
# OpenAI-compatible provider
# ---------------------------------------------------------------------------

@dataclass
class OpenAIProviderConfig:
    """Configuration for connecting to an OpenAI-compatible API."""

    api_url: str
    api_key: str
    model_name: str
    max_tokens: int | None = None


def _message_to_openai(msg: Message) -> dict[str, Any]:
    """Convert a thorn ``Message`` to an OpenAI API message dict."""
    match msg:
        case UserMessage():
            return {"role": "user", "content": msg.content}
        case AssistantMessage():
            result: dict[str, Any] = {"role": "assistant"}
            if msg.content:
                result["content"] = msg.content
            if msg.tool_calls:
                result["tool_calls"] = [
                    {
                        "id": tc.call_id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": tc.arguments,
                        },
                    }
                    for tc in msg.tool_calls
                ]
            elif not msg.content:
                result["content"] = ""
            return result
        case ToolResultMessage():
            return {
                "role": "tool",
                "tool_call_id": msg.call_id,
                "content": msg.content,
            }
        case _:
            return {"role": msg.role, "content": ""}


class OpenAIProvider(LLMProvider):
    """Provider that delegates to an OpenAI-compatible chat-completion API."""

    def __init__(self, config: OpenAIProviderConfig) -> None:
        self.config = config
        self._client = httpx.AsyncClient(
            base_url=config.api_url.rstrip("/"),
            headers={
                "Authorization": f"Bearer {config.api_key}",
                "Content-Type": "application/json",
            },
            timeout=httpx.Timeout(120.0, connect=30.0),
        )

    async def complete(
        self,
        system_prompts: list[str],
        tools: list[dict],
        messages: list[Message],
    ) -> AsyncIterator[ResponseChunk]:
        api_messages: list[dict[str, Any]] = []
        combined_system = "\n\n".join(system_prompts)
        if combined_system:
            api_messages.append({"role": "system", "content": combined_system})

        for msg in messages:
            api_messages.append(_message_to_openai(msg))

        body: dict[str, Any] = {
            "model": self.config.model_name,
            "messages": api_messages,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if self.config.max_tokens is not None:
            body["max_tokens"] = self.config.max_tokens
        if tools:
            body["tools"] = tools

        try:
            async with self._client.stream(
                "POST",
                "/chat/completions",
                json=body,
            ) as response:
                if response.status_code == 429:
                    await response.aread()
                    raise RateLimitError(
                        f"rate limited (HTTP 429): "
                        f"{_provider_error_snippet(response.text)}",
                        retry_after=_parse_retry_after(
                            response.headers.get("retry-after"),
                        ),
                    )
                if response.status_code in _TRANSIENT_HTTP_STATUSES:
                    await response.aread()
                    raise TransientProviderError(
                        f"provider returned transient HTTP "
                        f"{response.status_code}: "
                        f"{_provider_error_snippet(response.text)}",
                        retry_after=_parse_retry_after(
                            response.headers.get("retry-after"),
                        ),
                    )
                if response.status_code != 200:
                    await response.aread()
                    raise ProviderError(
                        f"provider returned HTTP {response.status_code}: "
                        f"{_provider_error_snippet(response.text)}"
                    )

                # Stream parsing lives inside the ``try`` so that
                # transport-level failures that surface mid-stream
                # (notably ``httpx.RemoteProtocolError`` when the
                # server closes the SSE connection early) are
                # converted to a typed transient error the agent
                # loop can retry, rather than propagating as a raw
                # httpx exception that nothing above knows how to
                # categorize.
                async for chunk in _iter_sse_chunks(response):
                    yield chunk

        except httpx.TransportError as exc:
            # ``TransportError`` is the ``httpx`` superclass for
            # connection errors, timeouts, network errors, and
            # protocol errors (including ``RemoteProtocolError``).
            # Wrapping the whole family as transient matches
            # industry practice for REST clients -- the caller
            # will decide whether to retry based on its policy.
            raise TransientProviderError(
                f"transport error talking to provider: "
                f"{_provider_error_snippet(str(exc))}",
            ) from exc


# HTTP status codes that almost always represent a transient
# server-side condition worth retrying at the call site.  500 is
# deliberately excluded: a generic "internal server error" is
# often a server-side bug that will not clear on its own, and we
# would rather surface it quickly through the non-transient
# ``ProviderError`` path than spend a big retry budget on it.  The
# four listed here are the ones spec'd (502/504) or widely
# implemented (503) as transient gateway/overload conditions.
_TRANSIENT_HTTP_STATUSES = frozenset({502, 503, 504})


def _provider_error_snippet(response_text: str) -> str:
    """Bound provider error bodies without leaking credential material."""
    return redact_secret_snippet(response_text, max_chars=500)


def _parse_retry_after(header: str | None) -> float | None:
    """Parse an HTTP ``Retry-After`` header value into seconds.

    The RFC allows two formats:

    - A non-negative integer expressing a delay in seconds, e.g.
      ``Retry-After: 30``.
    - An HTTP-date, e.g. ``Retry-After: Wed, 21 Oct 2015 07:28:00 GMT``,
      interpreted as the absolute time at which the client may
      retry.  We convert the latter to a delay by subtracting the
      current time; negative results clamp to 0.

    Returns ``None`` when the header is absent, empty, or
    unparseable -- callers treat that as "no explicit guidance"
    and fall back to their usual backoff.
    """
    if header is None:
        return None
    raw = header.strip()
    if not raw:
        return None
    try:
        seconds = float(raw)
    except ValueError:
        pass
    else:
        return max(0.0, seconds)
    # ``parsedate_to_datetime`` raises ``ValueError`` (and, on some
    # Python versions, ``TypeError``) for unparseable inputs rather
    # than returning ``None``.  Swallow those to fall through to
    # the "no explicit guidance" path.
    try:
        parsed = email.utils.parsedate_to_datetime(raw)
    except (ValueError, TypeError):
        return None
    if parsed is None:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    now = datetime.now(tz=timezone.utc)
    delta = (parsed - now).total_seconds()
    return max(0.0, delta)


async def _iter_sse_chunks(
    response: httpx.Response,
) -> AsyncIterator[ResponseChunk]:
    """Parse an SSE stream from an OpenAI-compatible API into chunks."""
    pending_tool_calls: dict[int, dict[str, str]] = {}

    async for line in response.aiter_lines():
        if not line.startswith("data: "):
            continue
        data = line[6:]
        if data.strip() == "[DONE]":
            for idx in sorted(pending_tool_calls):
                tc = pending_tool_calls[idx]
                yield ToolCallChunk(
                    call_id=tc.get("id", ""),
                    name=tc.get("name", ""),
                    arguments=tc.get("arguments", ""),
                )
            yield FinishChunk(reason="stop")
            return

        try:
            payload = json.loads(data)
        except json.JSONDecodeError:
            continue

        usage = payload.get("usage")
        if usage is not None:
            yield UsageChunk(
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
                total_tokens=usage.get("total_tokens", 0),
            )

        choices = payload.get("choices", [])
        if not choices:
            continue
        delta = choices[0].get("delta", {})
        finish_reason = choices[0].get("finish_reason")

        if "content" in delta and delta["content"]:
            yield TextChunk(text=delta["content"])

        if "tool_calls" in delta:
            for tc_delta in delta["tool_calls"]:
                idx = tc_delta.get("index", 0)
                if idx not in pending_tool_calls:
                    pending_tool_calls[idx] = {
                        "id": "",
                        "name": "",
                        "arguments": "",
                    }
                entry = pending_tool_calls[idx]
                if "id" in tc_delta:
                    entry["id"] = tc_delta["id"]
                func = tc_delta.get("function", {})
                if "name" in func:
                    entry["name"] = func["name"]
                if "arguments" in func:
                    entry["arguments"] += func["arguments"]

        if finish_reason is not None:
            for idx in sorted(pending_tool_calls):
                tc = pending_tool_calls[idx]
                yield ToolCallChunk(
                    call_id=tc.get("id", ""),
                    name=tc.get("name", ""),
                    arguments=tc.get("arguments", ""),
                )
            pending_tool_calls.clear()
            yield FinishChunk(reason=finish_reason)


def load_provider_from_env() -> LLMProvider:
    """Create an ``OpenAIProvider`` from environment variables.

    Reads ``OPENAI_API_URL``, ``OPENAI_API_KEY``, and
    ``OPENAI_API_MODEL_NAME``.  The caller is responsible for
    loading ``.env`` files beforehand (the CLI does this in its
    ``main()`` group callback).
    """
    api_url = os.environ.get("OPENAI_API_URL")
    api_key = os.environ.get("OPENAI_API_KEY")
    model_name = os.environ.get("OPENAI_API_MODEL_NAME")

    missing = []
    if not api_url:
        missing.append("OPENAI_API_URL")
    if not api_key:
        missing.append("OPENAI_API_KEY")
    if not model_name:
        missing.append("OPENAI_API_MODEL_NAME")

    if missing:
        raise ProviderError(
            f"missing required environment variables: {', '.join(missing)}"
        )

    max_tokens_raw = os.environ.get("OPENAI_MAX_TOKENS")
    max_tokens: int | None = None
    if max_tokens_raw is not None:
        try:
            max_tokens = int(max_tokens_raw)
        except ValueError:
            raise ProviderError(
                f"OPENAI_MAX_TOKENS must be an integer, got {max_tokens_raw!r}"
            )

    config = OpenAIProviderConfig(
        api_url=api_url,  # type: ignore[arg-type]
        api_key=api_key,  # type: ignore[arg-type]
        model_name=model_name,  # type: ignore[arg-type]
        max_tokens=max_tokens,
    )
    return OpenAIProvider(config)
