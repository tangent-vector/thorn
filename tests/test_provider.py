"""Tests for thorn.core._provider — MockProvider, message conversion, SSE parsing."""

from __future__ import annotations

import json
import os
from unittest.mock import AsyncMock

import pytest

from thorn.core._messages import (
    AssistantMessage,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)
from thorn.core._provider import (
    FinishChunk,
    MockProvider,
    OpenAIProvider,
    OpenAIProviderConfig,
    TextChunk,
    ToolCallChunk,
    _iter_sse_chunks,
    _message_to_openai,
    load_provider_from_env,
)
from thorn.core.errors import ProviderError


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
        assert provider.config.max_tokens == 8192

    def test_max_tokens_absent_from_env(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_URL", "http://localhost")
        monkeypatch.setenv("OPENAI_API_KEY", "key")
        monkeypatch.setenv("OPENAI_API_MODEL_NAME", "gpt-4")
        monkeypatch.delenv("OPENAI_MAX_TOKENS", raising=False)
        provider = load_provider_from_env()
        assert isinstance(provider, OpenAIProvider)
        assert provider.config.max_tokens is None

    def test_max_tokens_non_integer_raises(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_URL", "http://localhost")
        monkeypatch.setenv("OPENAI_API_KEY", "key")
        monkeypatch.setenv("OPENAI_API_MODEL_NAME", "gpt-4")
        monkeypatch.setenv("OPENAI_MAX_TOKENS", "not-a-number")
        with pytest.raises(ProviderError, match="OPENAI_MAX_TOKENS"):
            load_provider_from_env()


# ---------------------------------------------------------------------------
# OpenAIProvider max_tokens in request body
# ---------------------------------------------------------------------------

class TestOpenAIProviderMaxTokens:
    """Verify that max_tokens is conditionally included in the HTTP body."""

    async def test_max_tokens_included_when_set(self):
        """When max_tokens is configured, the request body includes it."""
        captured_bodies: list[dict] = []

        config = OpenAIProviderConfig(
            api_url="http://localhost:1234",
            api_key="test",
            model_name="test-model",
            max_tokens=4096,
        )
        provider = OpenAIProvider(config)

        original_stream = provider._client.stream

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
        """When max_tokens is None, the request body does not contain the key."""
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


class _FakeHttpResponse:
    """Minimal stand-in for httpx.Response with status_code and aiter_lines."""

    def __init__(self, status_code: int, lines: list[str]) -> None:
        self.status_code = status_code
        self._lines = lines

    async def aiter_lines(self):
        for line in self._lines:
            yield line

    async def aread(self) -> None:
        pass
