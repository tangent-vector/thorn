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
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field, JsonValue

from thorn._redaction import redact_secret_snippet
from thorn.core._credentials import ServiceCredential
from thorn.core._messages import (
    AssistantMessage,
    Message,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)
from thorn.core.errors import (
    ProviderError,
    ProviderFailureKind,
    RateLimitError,
    TransientProviderError,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Provider configuration
# ---------------------------------------------------------------------------

class LLMProviderType(StrEnum):
    """Provider backends supported by Thorn configuration."""

    OPENAI = "openai"


class OpenAIReasoningEffort(StrEnum):
    """Reasoning-effort values accepted by supported OpenAI-compatible models."""

    NONE = "none"
    MINIMAL = "minimal"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    XHIGH = "xhigh"

    @classmethod
    def parse_env(cls, raw_value: str) -> "OpenAIReasoningEffort":
        """Parse the legacy CLI environment setting with a useful error."""
        try:
            return cls(raw_value.strip().lower())
        except ValueError as exc:
            supported = ", ".join(value.value for value in cls)
            raise ProviderError(
                "OPENAI_REASONING_EFFORT must be one of: " + supported,
            ) from exc


class OpenAIProviderSettings(BaseModel):
    """Operator-controlled connection settings for an OpenAI-compatible provider.

    The API key itself is never stored in JSON.  ``api_key_env_var`` names
    the environment variable the gateway or CLI process should read when it
    creates the provider.
    """

    model_config = ConfigDict(frozen=True)

    type: Literal[LLMProviderType.OPENAI] = Field(
        description="LLM provider backend.  Only 'openai' is supported today.",
    )
    api_url: str = Field(
        min_length=1,
        description="Base URL for the provider's API.",
    )
    api_key_env_var: str = Field(
        min_length=1,
        description=(
            "Environment variable holding the provider API key.  The "
            "literal key is read from the process environment at startup."
        ),
    )
    request_timeout_s: float = Field(
        default=120.0,
        gt=0,
        description=(
            "HTTP request/read/write timeout in seconds for provider calls."
        ),
    )
    connect_timeout_s: float = Field(
        default=30.0,
        gt=0,
        description="HTTP connection timeout in seconds for provider calls.",
    )


class LLMModelConfig(BaseModel):
    """Model name and provider-interpreted request options."""

    model_config = ConfigDict(frozen=True)

    name: str | None = Field(
        default=None,
        min_length=1,
        description="Provider model name.",
    )
    options: dict[str, JsonValue] = Field(
        default_factory=dict,
        description=(
            "Provider/model-specific request options. Thorn preserves "
            "these as JSON and lets the selected provider interpret them."
        ),
    )

    def merged_with(
        self,
        override: "LLMModelConfig",
    ) -> "LLMModelConfig":
        """Return ``override`` layered on top of this model config."""
        return LLMModelConfig(
            name=override.name if override.name is not None else self.name,
            options={**self.options, **override.options},
        )


class LLMConfig(BaseModel):
    """LLM provider/model configuration from ``gateway.json`` or ``agent.json``.

    A gateway-level config establishes agency defaults.  An agent-level
    config may specify only the fields that differ from the gateway default;
    ``None`` means "inherit" throughout the nested model.
    """

    model_config = ConfigDict(frozen=True)

    provider: OpenAIProviderSettings | None = None
    model: LLMModelConfig | None = None

    def has_operator_config(self) -> bool:
        """Return whether this object carries any JSON-authored settings."""
        return self.provider is not None or self.model is not None

    def merged_with(self, override: "LLMConfig | None") -> "LLMConfig":
        """Return ``override`` layered on top of this config."""
        if override is None or not override.has_operator_config():
            return self

        provider = (
            override.provider
            if override.provider is not None
            else self.provider
        )

        if self.model is None:
            model = override.model
        elif override.model is None:
            model = self.model
        else:
            model = self.model.merged_with(override.model)

        return LLMConfig(provider=provider, model=model)

    def cache_key(self) -> str:
        """Return a stable non-secret key for provider cache lookups."""
        return json.dumps(
            self.model_dump(mode="json", exclude_none=True),
            sort_keys=True,
            separators=(",", ":"),
        )


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

    def trace_request_payload(
        self,
        system_prompts: list[str],
        tools: list[dict],
        messages: list[Message],
    ) -> Any:
        """Return a diagnostic representation of the next provider request."""
        return None

    async def aclose(self) -> None:
        """Release provider resources held across completion calls."""


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

    def trace_request_payload(
        self,
        system_prompts: list[str],
        tools: list[dict],
        messages: list[Message],
    ) -> Any:
        """Return the mock request shape for prompt-trace tests."""
        return {
            "provider": type(self).__name__,
            "system_prompts": list(system_prompts),
            "tools": list(tools),
            "messages": [_message_to_openai(message) for message in messages],
        }


# ---------------------------------------------------------------------------
# OpenAI-compatible provider
# ---------------------------------------------------------------------------

@dataclass
class OpenAIProviderConfig:
    """Configuration for connecting to an OpenAI-compatible API."""

    api_url: str
    api_key: ServiceCredential
    model_name: str
    model_options: dict[str, JsonValue] | None = None
    request_timeout_s: float = 120.0
    connect_timeout_s: float = 30.0

    def __post_init__(self) -> None:
        if not isinstance(self.api_key, ServiceCredential):
            self.api_key = ServiceCredential(self.api_key)
        self.model_options = dict(self.model_options or {})
        if self.request_timeout_s <= 0:
            raise ProviderError(
                "OpenAI request_timeout_s must be > 0, got "
                f"{self.request_timeout_s!r}"
            )
        if self.connect_timeout_s <= 0:
            raise ProviderError(
                "OpenAI connect_timeout_s must be > 0, got "
                f"{self.connect_timeout_s!r}"
            )
        reserved_keys = sorted(
            set(self.model_options) & _OPENAI_PROVIDER_RESERVED_OPTION_KEYS
        )
        if reserved_keys:
            raise ProviderError(
                "OpenAI model options cannot override Thorn-managed request "
                f"fields: {', '.join(reserved_keys)}"
            )


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


def _build_openai_chat_completion_body(
    config: OpenAIProviderConfig,
    system_prompts: list[str],
    tools: list[dict],
    messages: list[Message],
) -> dict[str, Any]:
    """Build the exact JSON body used for OpenAI-compatible requests."""
    api_messages: list[dict[str, Any]] = []
    combined_system = "\n\n".join(system_prompts)
    if combined_system:
        api_messages.append({"role": "system", "content": combined_system})

    for msg in messages:
        api_messages.append(_message_to_openai(msg))

    body: dict[str, Any] = {
        "model": config.model_name,
        "messages": api_messages,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    body.update(config.model_options or {})
    if tools:
        body["tools"] = tools
    return body


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
            timeout=httpx.Timeout(
                config.request_timeout_s,
                connect=config.connect_timeout_s,
            ),
        )

    async def aclose(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.aclose()

    async def complete(
        self,
        system_prompts: list[str],
        tools: list[dict],
        messages: list[Message],
    ) -> AsyncIterator[ResponseChunk]:
        body = _build_openai_chat_completion_body(
            self.config,
            system_prompts,
            tools,
            messages,
        )

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
                        status_code=response.status_code,
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
                        failure_kind=ProviderFailureKind.TRANSIENT_HTTP,
                        status_code=response.status_code,
                    )
                if response.status_code != 200:
                    await response.aread()
                    raise ProviderError(
                        f"provider returned HTTP {response.status_code}: "
                        f"{_provider_error_snippet(response.text)}",
                        status_code=response.status_code,
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
                failure_kind=_transport_failure_kind(exc),
            ) from exc

    def trace_request_payload(
        self,
        system_prompts: list[str],
        tools: list[dict],
        messages: list[Message],
    ) -> Any:
        """Return the exact JSON body submitted to the chat-completions API."""
        return _build_openai_chat_completion_body(
            self.config,
            system_prompts,
            tools,
            messages,
        )


# HTTP status codes that almost always represent a transient
# server-side condition worth retrying at the call site.  500 is
# deliberately excluded: a generic "internal server error" is
# often a server-side bug that will not clear on its own, and we
# would rather surface it quickly through the non-transient
# ``ProviderError`` path than spend a big retry budget on it.  The
# four listed here are the ones spec'd (502/504) or widely
# implemented (503) as transient gateway/overload conditions.
_TRANSIENT_HTTP_STATUSES = frozenset({502, 503, 504})


_OPENAI_PROVIDER_RESERVED_OPTION_KEYS = frozenset({
    "messages",
    "model",
    "stream",
    "stream_options",
    "tools",
})


def _transport_failure_kind(exc: httpx.TransportError) -> ProviderFailureKind:
    """Classify an ``httpx`` transport exception for telemetry."""
    if isinstance(exc, httpx.ConnectTimeout):
        return ProviderFailureKind.CONNECT_TIMEOUT
    if isinstance(exc, httpx.ReadTimeout):
        return ProviderFailureKind.READ_TIMEOUT
    if isinstance(exc, httpx.WriteTimeout):
        return ProviderFailureKind.WRITE_TIMEOUT
    if isinstance(exc, httpx.PoolTimeout):
        return ProviderFailureKind.POOL_TIMEOUT
    if isinstance(exc, httpx.TimeoutException):
        return ProviderFailureKind.TRANSPORT_TIMEOUT
    if isinstance(exc, httpx.RemoteProtocolError):
        return ProviderFailureKind.PROTOCOL_ERROR
    if isinstance(exc, httpx.NetworkError):
        return ProviderFailureKind.NETWORK_ERROR
    return ProviderFailureKind.TRANSPORT_ERROR


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


def load_provider_from_config(
    config: LLMConfig,
    *,
    environ: Mapping[str, str] | None = None,
) -> LLMProvider:
    """Create a provider from structured config plus environment secrets.

    ``config`` supplies non-secret provider/model settings from JSON.
    The only environment value read here is the secret named by
    ``config.provider.api_key_env_var``.  The caller is responsible for
    loading ``.env`` files beforehand (the CLI does this in its
    ``main()`` group callback).
    """
    env = environ if environ is not None else os.environ
    provider_settings = config.provider
    model_settings = config.model

    missing_config = []
    if provider_settings is None:
        missing_config.append("llm.provider")
    if model_settings is None:
        missing_config.append("llm.model")
    elif model_settings.name is None:
        missing_config.append("llm.model.name")
    if missing_config:
        raise ProviderError(
            f"missing required LLM configuration: {', '.join(missing_config)}"
        )

    if provider_settings.type != LLMProviderType.OPENAI:
        raise ProviderError(
            f"unsupported LLM provider type: {provider_settings.type.value}"
        )

    api_url = provider_settings.api_url
    api_key_env_var = provider_settings.api_key_env_var
    api_key = env.get(api_key_env_var)

    missing = []
    if not api_key:
        missing.append(api_key_env_var)

    if missing:
        raise ProviderError(
            f"missing required environment variables: {', '.join(missing)}"
        )

    provider_config = OpenAIProviderConfig(
        api_url=api_url,
        api_key=ServiceCredential(api_key),
        model_name=model_settings.name,
        model_options=model_settings.options,
        request_timeout_s=provider_settings.request_timeout_s,
        connect_timeout_s=provider_settings.connect_timeout_s,
    )
    return OpenAIProvider(provider_config)


def load_provider_from_env() -> LLMProvider:
    """Create an ``OpenAIProvider`` from legacy environment variables."""
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

    max_tokens = None
    max_tokens_raw = os.environ.get("OPENAI_MAX_TOKENS")
    if max_tokens_raw is not None:
        try:
            max_tokens = int(max_tokens_raw)
        except ValueError:
            raise ProviderError(
                f"OPENAI_MAX_TOKENS must be an integer, got {max_tokens_raw!r}"
            ) from None

    model_options: dict[str, JsonValue] = {}
    if max_tokens is not None:
        model_options["max_tokens"] = max_tokens

    reasoning_effort_raw = os.environ.get("OPENAI_REASONING_EFFORT")
    if reasoning_effort_raw is not None:
        reasoning_effort = OpenAIReasoningEffort.parse_env(
            reasoning_effort_raw,
        )
        model_options["reasoning_effort"] = reasoning_effort.value

    provider_config = OpenAIProviderConfig(
        api_url=api_url,
        api_key=ServiceCredential(api_key),
        model_name=model_name,
        model_options=model_options,
    )
    return OpenAIProvider(provider_config)
