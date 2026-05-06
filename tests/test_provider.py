"""Tests for thorn.core._provider — MockProvider, message conversion, SSE parsing."""

from __future__ import annotations

import json

import pytest

from thorn.core._messages import (
    AssistantMessage,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)
from thorn.core._provider import (
    FinishChunk,
    LLMConfig,
    LLMModelConfig,
    LLMProviderType,
    MockProvider,
    OpenAIProvider,
    OpenAIProviderConfig,
    OpenAIProviderSettings,
    TextChunk,
    ToolCallChunk,
    _iter_sse_chunks,
    _message_to_openai,
    _parse_retry_after,
    load_provider_from_config,
    load_provider_from_env,
)
from thorn.core.errors import (
    ProviderError,
    RateLimitError,
    TransientProviderError,
)

# ---------------------------------------------------------------------------
# MockProvider
# ---------------------------------------------------------------------------

class TestMockProvider:
    async def test_canned_response_fifo(self):
        canned = [
            [TextChunk(text="first"), FinishChunk(reason="stop")],
            [TextChunk(text="second"), FinishChunk(reason="stop")],
        ]
        provider = MockProvider(canned_responses=canned)

        chunks1 = [c async for c in provider.complete([], [], [UserMessage(content="a")])]
        chunks2 = [c async for c in provider.complete([], [], [UserMessage(content="b")])]

        assert any(isinstance(c, TextChunk) and c.text == "first" for c in chunks1)
        assert any(isinstance(c, TextChunk) and c.text == "second" for c in chunks2)

    async def test_echo_fallback(self):
        provider = MockProvider()
        chunks = [c async for c in provider.complete([], [], [UserMessage(content="hello")])]
        text_chunks = [c for c in chunks if isinstance(c, TextChunk)]
        assert len(text_chunks) == 1
        assert text_chunks[0].text == "[mock] hello"

    async def test_echo_finds_last_user_message(self):
        provider = MockProvider()
        messages = [
            UserMessage(content="first"),
            AssistantMessage(content="ok"),
            UserMessage(content="second"),
        ]
        chunks = [c async for c in provider.complete([], [], messages)]
        text_chunks = [c for c in chunks if isinstance(c, TextChunk)]
        assert text_chunks[0].text == "[mock] second"

    async def test_canned_tool_call(self):
        canned = [[
            ToolCallChunk(call_id="c1", name="read_file", arguments='{"path":"a.txt"}'),
            FinishChunk(reason="stop"),
        ]]
        provider = MockProvider(canned_responses=canned)
        chunks = [c async for c in provider.complete([], [], [UserMessage(content="x")])]
        tc_chunks = [c for c in chunks if isinstance(c, ToolCallChunk)]
        assert len(tc_chunks) == 1
        assert tc_chunks[0].name == "read_file"


# ---------------------------------------------------------------------------
# _message_to_openai
# ---------------------------------------------------------------------------

class TestMessageToOpenai:
    def test_user_message(self):
        result = _message_to_openai(UserMessage(content="hi"))
        assert result == {"role": "user", "content": "hi"}

    def test_assistant_text_only(self):
        result = _message_to_openai(AssistantMessage(content="reply"))
        assert result == {"role": "assistant", "content": "reply"}

    def test_assistant_with_tool_calls(self):
        msg = AssistantMessage(
            content="",
            tool_calls=[ToolCall(call_id="c1", name="read", arguments="{}")],
        )
        result = _message_to_openai(msg)
        assert result["role"] == "assistant"
        assert len(result["tool_calls"]) == 1
        tc = result["tool_calls"][0]
        assert tc["id"] == "c1"
        assert tc["function"]["name"] == "read"

    def test_assistant_empty(self):
        result = _message_to_openai(AssistantMessage())
        assert result == {"role": "assistant", "content": ""}

    def test_tool_result(self):
        result = _message_to_openai(
            ToolResultMessage(call_id="c1", content="file data"),
        )
        assert result == {"role": "tool", "tool_call_id": "c1", "content": "file data"}


# ---------------------------------------------------------------------------
# _iter_sse_chunks (SSE parsing)
# ---------------------------------------------------------------------------

class _FakeResponse:
    """Minimal stand-in for httpx.Response that yields lines."""

    def __init__(self, lines: list[str]) -> None:
        self._lines = lines

    async def aiter_lines(self):
        for line in self._lines:
            yield line


class TestIterSseChunks:
    async def test_text_only_stream(self):
        lines = [
            'data: {"choices":[{"delta":{"content":"Hello"},"finish_reason":null}]}',
            'data: {"choices":[{"delta":{"content":" world"},"finish_reason":null}]}',
            'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}',
        ]
        chunks = [c async for c in _iter_sse_chunks(_FakeResponse(lines))]
        texts = [c.text for c in chunks if isinstance(c, TextChunk)]
        assert texts == ["Hello", " world"]
        assert any(isinstance(c, FinishChunk) and c.reason == "stop" for c in chunks)

    async def test_tool_call_stream(self):
        lines = [
            'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c1","function":{"name":"my_tool","arguments":""}}]},"finish_reason":null}]}',
            'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\\"x\\":"}}]},"finish_reason":null}]}',
            'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"1}"}}]},"finish_reason":"tool_calls"}]}',
        ]
        chunks = [c async for c in _iter_sse_chunks(_FakeResponse(lines))]
        tc_chunks = [c for c in chunks if isinstance(c, ToolCallChunk)]
        assert len(tc_chunks) == 1
        assert tc_chunks[0].name == "my_tool"
        assert tc_chunks[0].call_id == "c1"
        assert json.loads(tc_chunks[0].arguments) == {"x": 1}

    async def test_done_marker(self):
        lines = [
            'data: {"choices":[{"delta":{"content":"hi"},"finish_reason":null}]}',
            "data: [DONE]",
        ]
        chunks = [c async for c in _iter_sse_chunks(_FakeResponse(lines))]
        assert any(isinstance(c, FinishChunk) for c in chunks)

    async def test_ignores_non_data_lines(self):
        lines = [
            ": keep-alive",
            "",
            'data: {"choices":[{"delta":{"content":"ok"},"finish_reason":"stop"}]}',
        ]
        chunks = [c async for c in _iter_sse_chunks(_FakeResponse(lines))]
        texts = [c.text for c in chunks if isinstance(c, TextChunk)]
        assert texts == ["ok"]

    async def test_ignores_malformed_json(self):
        lines = [
            "data: {not valid json",
            'data: {"choices":[{"delta":{"content":"ok"},"finish_reason":"stop"}]}',
        ]
        chunks = [c async for c in _iter_sse_chunks(_FakeResponse(lines))]
        texts = [c.text for c in chunks if isinstance(c, TextChunk)]
        assert texts == ["ok"]

    async def test_multiple_tool_calls_in_parallel(self):
        lines = [
            'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c1","function":{"name":"foo","arguments":"{}"}},{"index":1,"id":"c2","function":{"name":"bar","arguments":"{}"}}]},"finish_reason":"tool_calls"}]}',
        ]
        chunks = [c async for c in _iter_sse_chunks(_FakeResponse(lines))]
        tc_chunks = [c for c in chunks if isinstance(c, ToolCallChunk)]
        assert len(tc_chunks) == 2
        names = {tc.name for tc in tc_chunks}
        assert names == {"foo", "bar"}


# ---------------------------------------------------------------------------
# load_provider_from_env
# ---------------------------------------------------------------------------

class TestLoadProviderFromEnv:
    def test_missing_vars_raises(self, monkeypatch):
        # Prevent .env file from re-injecting the variables.
        monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **kw: None)
        monkeypatch.delenv("OPENAI_API_URL", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_MODEL_NAME", raising=False)
        with pytest.raises(ProviderError, match="missing required"):
            load_provider_from_env()

    def test_partial_missing_lists_all(self, monkeypatch):
        monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **kw: None)
        monkeypatch.setenv("OPENAI_API_URL", "http://x")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_MODEL_NAME", raising=False)
        with pytest.raises(ProviderError, match="OPENAI_API_KEY") as exc_info:
            load_provider_from_env()
        assert "OPENAI_API_MODEL_NAME" in str(exc_info.value)

    def test_max_tokens_from_env(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_URL", "http://localhost")
        monkeypatch.setenv("OPENAI_API_KEY", "key")
        monkeypatch.setenv("OPENAI_API_MODEL_NAME", "gpt-4")
        monkeypatch.setenv("OPENAI_MAX_TOKENS", "8192")
        provider = load_provider_from_env()
        assert isinstance(provider, OpenAIProvider)
        assert provider.config.model_options == {"max_tokens": 8192}

    def test_max_tokens_absent_from_env(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_URL", "http://localhost")
        monkeypatch.setenv("OPENAI_API_KEY", "key")
        monkeypatch.setenv("OPENAI_API_MODEL_NAME", "gpt-4")
        monkeypatch.delenv("OPENAI_MAX_TOKENS", raising=False)
        provider = load_provider_from_env()
        assert isinstance(provider, OpenAIProvider)
        assert provider.config.model_options == {}

    def test_max_tokens_non_integer_raises(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_URL", "http://localhost")
        monkeypatch.setenv("OPENAI_API_KEY", "key")
        monkeypatch.setenv("OPENAI_API_MODEL_NAME", "gpt-4")
        monkeypatch.setenv("OPENAI_MAX_TOKENS", "not-a-number")
        with pytest.raises(ProviderError, match="OPENAI_MAX_TOKENS"):
            load_provider_from_env()


class TestLoadProviderFromConfig:
    def test_config_supplies_non_secret_provider_and_model_settings(
        self, monkeypatch,
    ):
        monkeypatch.delenv("OPENAI_API_URL", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_MODEL_NAME", raising=False)
        monkeypatch.setenv("THORN_LLM_KEY", "secret-value")

        provider = load_provider_from_config(LLMConfig(
            provider=OpenAIProviderSettings(
                type=LLMProviderType.OPENAI,
                api_url="https://llm.example/v1",
                api_key_env_var="THORN_LLM_KEY",
            ),
            model=LLMModelConfig(
                name="agent-model",
                options={"max_tokens": 2048, "reasoning_effort": "medium"},
            ),
        ))

        assert isinstance(provider, OpenAIProvider)
        assert provider.config.api_url == "https://llm.example/v1"
        assert provider.config.api_key == "secret-value"
        assert provider.config.model_name == "agent-model"
        assert provider.config.model_options == {
            "max_tokens": 2048,
            "reasoning_effort": "medium",
        }

    def test_config_path_ignores_legacy_non_secret_env(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_URL", "https://llm.example/v1")
        monkeypatch.setenv("OPENAI_API_KEY", "key")
        monkeypatch.setenv("OPENAI_API_MODEL_NAME", "env-model")
        monkeypatch.setenv("OPENAI_MAX_TOKENS", "8192")
        monkeypatch.setenv("THORN_LLM_KEY", "configured-key")

        provider = load_provider_from_config(LLMConfig(
            provider=OpenAIProviderSettings(
                type=LLMProviderType.OPENAI,
                api_url="https://configured.example/v1",
                api_key_env_var="THORN_LLM_KEY",
            ),
            model=LLMModelConfig(
                name="configured-model",
            ),
        ))

        assert isinstance(provider, OpenAIProvider)
        assert provider.config.api_url == "https://configured.example/v1"
        assert provider.config.api_key == "configured-key"
        assert provider.config.model_name == "configured-model"
        assert provider.config.model_options == {}

    def test_missing_configured_api_key_env_var_is_reported(self, monkeypatch):
        monkeypatch.delenv("MISSING_LLM_KEY", raising=False)

        with pytest.raises(ProviderError, match="MISSING_LLM_KEY"):
            load_provider_from_config(LLMConfig(
                provider=OpenAIProviderSettings(
                    type=LLMProviderType.OPENAI,
                    api_url="https://llm.example/v1",
                    api_key_env_var="MISSING_LLM_KEY",
                ),
                model=LLMModelConfig(name="model"),
            ))

    def test_missing_configured_provider_and_model_are_reported(self):
        with pytest.raises(ProviderError, match="llm.provider") as exc_info:
            load_provider_from_config(LLMConfig(), environ={})
        assert "llm.model" in str(exc_info.value)


# ---------------------------------------------------------------------------
# OpenAIProvider model options in request body
# ---------------------------------------------------------------------------

class TestOpenAIProviderModelOptions:
    """Verify provider/model-specific options are added to the HTTP body."""

    async def test_max_tokens_included_when_set(self):
        captured_bodies: list[dict] = []

        config = OpenAIProviderConfig(
            api_url="http://localhost:1234",
            api_key="test",
            model_name="test-model",
            model_options={"max_tokens": 4096},
        )
        provider = OpenAIProvider(config)

        from contextlib import asynccontextmanager

        @asynccontextmanager
        async def capturing_stream(method, url, *, json=None, **kwargs):
            captured_bodies.append(json)
            # Yield a fake response that produces a valid SSE stream
            fake = _FakeHttpResponse(200, [
                'data: {"choices":[{"delta":{"content":"ok"},"finish_reason":"stop"}]}',
                "data: [DONE]",
            ])
            yield fake

        provider._client.stream = capturing_stream  # type: ignore[assignment]

        chunks = []
        async for c in provider.complete(["sys"], [], [UserMessage(content="hi")]):
            chunks.append(c)

        assert len(captured_bodies) == 1
        assert captured_bodies[0]["max_tokens"] == 4096

    async def test_max_tokens_omitted_when_none(self):
        captured_bodies: list[dict] = []

        config = OpenAIProviderConfig(
            api_url="http://localhost:1234",
            api_key="test",
            model_name="test-model",
        )
        provider = OpenAIProvider(config)

        from contextlib import asynccontextmanager

        @asynccontextmanager
        async def capturing_stream(method, url, *, json=None, **kwargs):
            captured_bodies.append(json)
            fake = _FakeHttpResponse(200, [
                'data: {"choices":[{"delta":{"content":"ok"},"finish_reason":"stop"}]}',
                "data: [DONE]",
            ])
            yield fake

        provider._client.stream = capturing_stream  # type: ignore[assignment]

        chunks = []
        async for c in provider.complete(["sys"], [], [UserMessage(content="hi")]):
            chunks.append(c)

        assert len(captured_bodies) == 1
        assert "max_tokens" not in captured_bodies[0]

    async def test_reasoning_effort_included_when_set(self):
        captured_bodies: list[dict] = []

        config = OpenAIProviderConfig(
            api_url="http://localhost:1234",
            api_key="test",
            model_name="test-model",
            model_options={"reasoning_effort": "high", "temperature": 0.2},
        )
        provider = OpenAIProvider(config)

        from contextlib import asynccontextmanager

        @asynccontextmanager
        async def capturing_stream(method, url, *, json=None, **kwargs):
            captured_bodies.append(json)
            fake = _FakeHttpResponse(200, [
                'data: {"choices":[{"delta":{"content":"ok"},"finish_reason":"stop"}]}',
                "data: [DONE]",
            ])
            yield fake

        provider._client.stream = capturing_stream  # type: ignore[assignment]

        async for _ in provider.complete(["sys"], [], [UserMessage(content="hi")]):
            pass

        assert len(captured_bodies) == 1
        assert captured_bodies[0]["reasoning_effort"] == "high"
        assert captured_bodies[0]["temperature"] == 0.2

    def test_model_options_cannot_replace_thorn_managed_fields(self):
        with pytest.raises(ProviderError, match="messages"):
            OpenAIProviderConfig(
                api_url="http://localhost:1234",
                api_key="test",
                model_name="test-model",
                model_options={"messages": []},
            )


class _FakeHttpResponse:
    """Minimal stand-in for httpx.Response with status_code and aiter_lines.

    Supports two failure modes for exercising the provider's error
    handling: a non-200 ``status_code`` (with an optional response
    ``text`` and ``headers`` mapping for ``Retry-After`` tests),
    and a mid-stream ``raise_in_stream`` exception that is raised
    after yielding any preamble lines.
    """

    def __init__(
        self,
        status_code: int,
        lines: list[str],
        *,
        text: str = "",
        headers: dict[str, str] | None = None,
        raise_in_stream: BaseException | None = None,
    ) -> None:
        self.status_code = status_code
        self._lines = lines
        self.text = text
        self.headers = headers or {}
        self._raise_in_stream = raise_in_stream

    async def aiter_lines(self):
        for line in self._lines:
            yield line
        if self._raise_in_stream is not None:
            raise self._raise_in_stream

    async def aread(self) -> None:
        pass


# ---------------------------------------------------------------------------
# _parse_retry_after
# ---------------------------------------------------------------------------

class TestParseRetryAfter:
    """``Retry-After`` header parsing accepts seconds and HTTP-dates."""

    def test_none_and_empty_return_none(self):
        assert _parse_retry_after(None) is None
        assert _parse_retry_after("") is None
        assert _parse_retry_after("   ") is None

    def test_integer_seconds(self):
        assert _parse_retry_after("30") == 30.0

    def test_float_seconds(self):
        # RFC technically requires integer seconds, but real-world
        # servers occasionally return fractional values.  Accept
        # them rather than misclassify as "unparseable" and fall
        # back to no hint.
        assert _parse_retry_after("2.5") == 2.5

    def test_negative_seconds_clamp_to_zero(self):
        # A negative "seconds" value is not spec-legal, but a bug
        # in the server should not translate to a negative sleep
        # that fires immediately with complaint; clamp to 0.
        assert _parse_retry_after("-5") == 0.0

    def test_http_date_future(self):
        from datetime import datetime, timedelta, timezone
        from email.utils import format_datetime

        future = datetime.now(tz=timezone.utc) + timedelta(seconds=45)
        header = format_datetime(future, usegmt=True)
        parsed = _parse_retry_after(header)
        # Allow slack for the clock read between format and parse.
        assert parsed is not None
        assert 40.0 <= parsed <= 50.0

    def test_http_date_in_past_clamps_to_zero(self):
        # Past HTTP-date: behavior of a server that happened to
        # pick a date whose value has already elapsed.  Zero means
        # "retry now"; we do not want to return a negative sleep.
        from datetime import datetime, timedelta, timezone
        from email.utils import format_datetime

        past = datetime.now(tz=timezone.utc) - timedelta(seconds=30)
        header = format_datetime(past, usegmt=True)
        assert _parse_retry_after(header) == 0.0

    def test_garbage_returns_none(self):
        # Unrecognised value: the caller treats None as "no hint"
        # and falls back to its own backoff, which is safer than
        # crashing the session.
        assert _parse_retry_after("not a date") is None


# ---------------------------------------------------------------------------
# OpenAIProvider error mapping
# ---------------------------------------------------------------------------

def _mock_streaming_provider(
    response: "_FakeHttpResponse | None" = None,
    *,
    raise_on_stream: BaseException | None = None,
) -> OpenAIProvider:
    """Build an ``OpenAIProvider`` whose ``_client.stream`` is stubbed.

    Either yields a fixed ``_FakeHttpResponse`` or raises the given
    exception from the ``stream()`` context manager itself (used
    for pre-stream transport errors like ``ConnectError``).
    """
    from contextlib import asynccontextmanager

    config = OpenAIProviderConfig(
        api_url="http://localhost:1234",
        api_key="test",
        model_name="test-model",
    )
    provider = OpenAIProvider(config)

    @asynccontextmanager
    async def stub_stream(method, url, *, json=None, **kwargs):
        if raise_on_stream is not None:
            raise raise_on_stream
        yield response

    provider._client.stream = stub_stream  # type: ignore[assignment]
    return provider


class TestOpenAIProviderErrorMapping:
    async def test_remote_protocol_error_mid_stream_becomes_transient(self):
        """A ``RemoteProtocolError`` raised while reading the SSE stream
        must surface as :class:`TransientProviderError` so the agent-loop
        retry logic can absorb it instead of letting a raw ``httpx``
        exception escape into the scheduler.
        """
        import httpx

        response = _FakeHttpResponse(
            200,
            [
                'data: {"choices":[{"delta":{"content":"partial"},'
                '"finish_reason":null}]}',
            ],
            raise_in_stream=httpx.RemoteProtocolError("server disconnected"),
        )
        provider = _mock_streaming_provider(response)
        with pytest.raises(TransientProviderError, match="transport error"):
            async for _ in provider.complete(
                ["sys"], [], [UserMessage(content="hi")],
            ):
                pass

    async def test_connect_error_becomes_transient(self):
        import httpx

        provider = _mock_streaming_provider(
            raise_on_stream=httpx.ConnectError("refused"),
        )
        with pytest.raises(TransientProviderError, match="transport error"):
            async for _ in provider.complete(
                ["sys"], [], [UserMessage(content="hi")],
            ):
                pass

    async def test_timeout_becomes_transient(self):
        import httpx

        provider = _mock_streaming_provider(
            raise_on_stream=httpx.ReadTimeout("timeout"),
        )
        with pytest.raises(TransientProviderError, match="transport error"):
            async for _ in provider.complete(
                ["sys"], [], [UserMessage(content="hi")],
            ):
                pass

    async def test_transport_error_redacts_credentials(self) -> None:
        import httpx

        provider_key = "sk-transport-provider-key-123456"
        provider = _mock_streaming_provider(
            raise_on_stream=httpx.ConnectError(
                f"Authorization: Bearer {provider_key}",
            ),
        )
        with pytest.raises(TransientProviderError) as exc_info:
            async for _ in provider.complete(
                ["sys"], [], [UserMessage(content="hi")],
            ):
                pass

        message = str(exc_info.value)
        assert provider_key not in message
        assert "Authorization: <redacted>" in message

    async def test_429_becomes_rate_limit_with_retry_after(self):
        response = _FakeHttpResponse(
            429, [], text="slow down", headers={"retry-after": "12"},
        )
        provider = _mock_streaming_provider(response)
        with pytest.raises(RateLimitError) as exc_info:
            async for _ in provider.complete(
                ["sys"], [], [UserMessage(content="hi")],
            ):
                pass
        assert exc_info.value.retry_after == 12.0

    async def test_429_without_header_has_no_retry_after(self):
        response = _FakeHttpResponse(429, [], text="no hint", headers={})
        provider = _mock_streaming_provider(response)
        with pytest.raises(RateLimitError) as exc_info:
            async for _ in provider.complete(
                ["sys"], [], [UserMessage(content="hi")],
            ):
                pass
        assert exc_info.value.retry_after is None

    @pytest.mark.parametrize("status", [502, 503, 504])
    async def test_5xx_becomes_transient(self, status: int):
        """502/503/504 are modeled as transport-ish transient errors
        and routed to the retry budget, not the agent-failure budget.
        """
        response = _FakeHttpResponse(
            status, [], text="overloaded",
            headers={"retry-after": "5"} if status == 503 else {},
        )
        provider = _mock_streaming_provider(response)
        with pytest.raises(TransientProviderError) as exc_info:
            async for _ in provider.complete(
                ["sys"], [], [UserMessage(content="hi")],
            ):
                pass
        if status == 503:
            assert exc_info.value.retry_after == 5.0
        else:
            assert exc_info.value.retry_after is None

    async def test_500_is_not_transient(self):
        # 500 is *deliberately* excluded from the transient set:
        # a generic internal error is usually a server bug rather
        # than an overload condition, so we fail faster via the
        # regular :class:`ProviderError` path.
        response = _FakeHttpResponse(500, [], text="kaboom")
        provider = _mock_streaming_provider(response)
        with pytest.raises(ProviderError) as exc_info:
            async for _ in provider.complete(
                ["sys"], [], [UserMessage(content="hi")],
            ):
                pass
        assert not isinstance(exc_info.value, TransientProviderError)

    async def test_400_is_not_transient(self):
        response = _FakeHttpResponse(400, [], text="bad request")
        provider = _mock_streaming_provider(response)
        with pytest.raises(ProviderError) as exc_info:
            async for _ in provider.complete(
                ["sys"], [], [UserMessage(content="hi")],
            ):
                pass
        assert not isinstance(exc_info.value, TransientProviderError)
        assert not isinstance(exc_info.value, RateLimitError)

    @pytest.mark.parametrize(
        ("status", "error_type"),
        [
            (429, RateLimitError),
            (503, TransientProviderError),
            (401, ProviderError),
        ],
    )
    async def test_http_error_bodies_redact_credentials(
        self,
        status: int,
        error_type: type[ProviderError],
    ) -> None:
        provider_api_key = "sk-live-provider-key-123456"
        bearer_token = "nvapi-provider-bearer-token-123456"
        proxy_token = "aoc_proxy_token_for_provider_123456"
        gitlab_pat = "glpat-provider-pat-123456"
        response = _FakeHttpResponse(
            status,
            [],
            text=(
                f"Authorization: Bearer {bearer_token}\n"
                f"Proxy-Authorization: Basic {proxy_token}\n"
                f'{{"api_key": "{provider_api_key}", '
                f'"accessToken": "{proxy_token}", '
                f'"token": "{gitlab_pat}", '
                f'"url": "https://x:{gitlab_pat}@gitlab.example.com/repo"}}'
            ),
        )
        provider = _mock_streaming_provider(response)

        with pytest.raises(error_type) as exc_info:
            async for _ in provider.complete(
                ["sys"], [], [UserMessage(content="hi")],
            ):
                pass

        message = str(exc_info.value)
        for secret in (
            provider_api_key,
            bearer_token,
            proxy_token,
            gitlab_pat,
        ):
            assert secret not in message
        assert "Authorization: <redacted>" in message
        assert "Proxy-Authorization: <redacted>" in message
        assert "<redacted>" in message
