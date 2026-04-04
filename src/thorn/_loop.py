"""Agent loop: streaming, tool dispatch, structured-output extraction.

Two modes of operation:

* **Text mode** (``result_type is str``): the assistant's final text
  response is returned directly.
* **Structured mode** (any other ``result_type``): synthetic
  ``return_result`` and ``raise_error`` tools are injected, and the
  loop extracts and validates the typed result.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import time
from typing import Any

from thorn._context import ExecutionContext, Scope
from thorn._history import (
    DEFAULT_HIGH_WATERMARK,
    DEFAULT_LOW_WATERMARK,
    HistoryTree,
    ToolCallNode,
    estimate_tokens,
)
from thorn._messages import (
    AssistantMessage,
    Message,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)
from thorn._provider import FinishChunk, TextChunk, ToolCallChunk, UsageChunk
from thorn._schema import (
    RAISE_ERROR_SCHEMA,
    make_return_result_schema,
    serialize_for_tool_result,
    validate_result,
)
from thorn.errors import (
    AgentFailureError,
    LoopLimitError,
    ProviderError,
    RateLimitError,
    SkillError,
)

logger = logging.getLogger(__name__)

_BACKOFF_BASE = 1.0
_BACKOFF_CAP = 60.0
_BACKOFF_MAX_RETRIES = 5

# Sentinel returned by _execute_tool_calls when the structured-mode
# ``return_result`` tool is invoked.
_RESULT_SENTINEL = object()


class _WrappedTool:
    """Lightweight wrapper that pairs a tool schema with an execute callback.

    When *call_node_class* is set, ``HistoryTree.append_turn`` will
    construct that ``ToolCallNode`` subclass instead of the base class,
    enabling ``isinstance``-based identification in downstream code.
    """

    __slots__ = ("schema", "execute", "call_node_class")

    def __init__(
        self,
        schema: dict[str, Any],
        execute: Any,  # async callable(**kwargs) -> str
        call_node_class: type[ToolCallNode] | None = None,
    ) -> None:
        self.schema = schema
        self.execute = execute
        self.call_node_class = call_node_class


async def run_agent_loop(
    *,
    context: ExecutionContext,
    user_prompt: str | None,
    tools: list[_WrappedTool],
    system_prompts: list[str] | None = None,
    result_type: type | None = None,
    max_tool_rounds: int = 50,
    max_failures: int = 5,
    history: HistoryTree | None = None,
) -> Any:
    """Drive the request -> tool-call -> response cycle.

    Returns a ``str`` in text mode or a validated value of *result_type*
    in structured mode.

    If *history* is provided and *user_prompt* is not ``None``, the
    prompt is appended to the tree before the loop begins.  When
    *user_prompt* is ``None``, the caller is responsible for having
    already populated the history (e.g. after context injection).
    The tree is mutated in place, so the caller retains the accumulated
    history after the call returns (enabling multi-turn patterns).
    If *history* is ``None`` (the default), a fresh tree is created
    internally.
    """
    structured = result_type is not None and result_type is not str

    # -- build schemas and dispatch table ----------------------------------
    all_tools = list(tools)
    tool_dispatch: dict[str, _WrappedTool] = {
        _tool_name(t.schema): t for t in all_tools
    }

    all_schemas = [t.schema for t in all_tools]

    # raise_error is always available so any agent can signal failure.
    all_schemas.append(RAISE_ERROR_SCHEMA)

    # In structured mode, also inject the return_result tool.
    structured_result: list[Any] = []  # mutable box for the captured value
    if structured:
        rr_schema = make_return_result_schema(result_type)
        all_schemas.append(rr_schema)

    # -- assemble system prompts -------------------------------------------
    prompts = list(context.system_prompts)
    if context.workspace_instructions:
        prompts.append(context.workspace_instructions)
    if system_prompts:
        prompts.extend(system_prompts)

    if structured:
        prompts.append(
            "You MUST call the `return_result` tool to deliver your final "
            "answer.  Do NOT respond with plain text — always use the tool.  "
            "If you cannot fulfil the request, call `raise_error` instead."
        )

    # -- conversation history (tree) ---------------------------------------
    if history is None:
        history = HistoryTree()
    if user_prompt is not None:
        history.append_user_prompt(user_prompt)

    # -- compaction configuration ------------------------------------------
    context_window = context.context_window
    overhead_tokens = _estimate_overhead(prompts, all_schemas) if context_window else 0

    consecutive_failures = 0

    for round_num in range(max_tool_rounds):
        rendered = history.render()
        text, tool_calls, _finish, usage = await _request_completion(
            context=context,
            system_prompts=prompts,
            tool_schemas=all_schemas,
            messages=rendered,
            consecutive_failures=consecutive_failures,
            max_failures=max_failures,
        )
        consecutive_failures = 0

        if usage is not None:
            context.usage.add(
                usage.get("prompt_tokens", 0),
                usage.get("completion_tokens", 0),
                usage.get("total_tokens", 0),
            )

        # -- no tool calls: either done (text mode) or nudge (structured) --
        if not tool_calls:
            history.append_turn(AssistantMessage(content=text), [])
            if not structured:
                return text
            history.append_user_prompt(
                "You must call the `return_result` tool with your "
                "answer, or `raise_error` if you cannot proceed.  "
                "Do not reply with plain text."
            )
            continue

        # -- dispatch tool calls -------------------------------------------
        result_msgs, call_node_classes, captured = await _execute_tool_calls(
            tool_calls=tool_calls,
            tool_dispatch=tool_dispatch,
            context=context,
            result_type=result_type if structured else None,
        )

        history.append_turn(
            AssistantMessage(content=text, tool_calls=tool_calls),
            result_msgs,
            call_node_classes=call_node_classes or None,
        )

        # -- compaction check ----------------------------------------------
        if context_window is not None and usage is not None:
            prompt_tokens = usage.get("prompt_tokens", 0)
            if prompt_tokens > context_window * DEFAULT_HIGH_WATERMARK:
                compact_result = history.compact(
                    context_budget=context_window,
                    overhead_tokens=overhead_tokens,
                    actual_prompt_tokens=prompt_tokens,
                )
                if compact_result.estimated_savings > 0:
                    await context.event_sink.on_status(
                        f"compaction: collapsed {compact_result.nodes_collapsed} nodes, "
                        f"{compact_result.tool_calls_detail_collapsed} tool calls detail-collapsed, "
                        f"~{compact_result.estimated_savings} est. tokens saved "
                        f"({compact_result.tokens_before} -> {compact_result.tokens_after})",
                        scope=context.scope,
                    )

        if captured is not _RESULT_SENTINEL:
            return captured

    raise LoopLimitError(
        f"agent loop exceeded {max_tool_rounds} rounds", max_tool_rounds,
    )


# ---------------------------------------------------------------------------
# Overhead estimation
# ---------------------------------------------------------------------------

def _estimate_overhead(
    prompts: list[str],
    schemas: list[dict[str, Any]],
) -> int:
    """Rough token estimate for everything outside the history.

    Covers system prompts and tool schemas.  Used only for the compaction
    target computation; absolute trigger checks use real provider usage.
    """
    total = 0
    for p in prompts:
        total += estimate_tokens(p)
    for schema in schemas:
        total += estimate_tokens(json.dumps(schema))
    return total


# ---------------------------------------------------------------------------
# Completion request (with retry)
# ---------------------------------------------------------------------------

async def _backoff_retry(attempt: int, reason: str = "rate limited") -> None:
    delay = min(_BACKOFF_BASE * (2 ** attempt) + random.uniform(0, 1), _BACKOFF_CAP)
    logger.info("%s, retrying in %.1fs (attempt %d)", reason, delay, attempt + 1)
    await asyncio.sleep(delay)


async def _request_completion(
    *,
    context: ExecutionContext,
    system_prompts: list[str],
    tool_schemas: list[dict],
    messages: list[Message],
    consecutive_failures: int,
    max_failures: int,
) -> tuple[str, list[ToolCall], str, dict[str, int] | None]:
    """Return ``(text, tool_calls, finish_reason, usage)``."""
    rate_limit_retries = 0

    while True:
        try:
            text_parts: list[str] = []
            tool_call_chunks: list[ToolCallChunk] = []
            finish_reason = "stop"
            usage: dict[str, int] | None = None

            t0 = time.monotonic()
            response = context.provider.complete(
                system_prompts, tool_schemas, messages,
            )
            async for chunk in response:
                await context.event_sink.on_response_chunk(
                    chunk, scope=context.scope,
                )
                match chunk:
                    case TextChunk():
                        text_parts.append(chunk.text)
                    case ToolCallChunk():
                        tool_call_chunks.append(chunk)
                    case UsageChunk():
                        usage = {
                            "prompt_tokens": chunk.prompt_tokens,
                            "completion_tokens": chunk.completion_tokens,
                            "total_tokens": chunk.total_tokens,
                        }
                    case FinishChunk():
                        finish_reason = chunk.reason

            duration_s = time.monotonic() - t0
            await context.event_sink.on_completion_end(
                duration_s=duration_s, usage=usage, scope=context.scope,
            )

            text = "".join(text_parts)
            tool_calls = [tc.to_tool_call() for tc in tool_call_chunks]
            return text, tool_calls, finish_reason, usage

        except RateLimitError:
            rate_limit_retries += 1
            if rate_limit_retries > _BACKOFF_MAX_RETRIES:
                raise
            await _backoff_retry(rate_limit_retries - 1)

        except ProviderError:
            consecutive_failures += 1
            if consecutive_failures >= max_failures:
                raise AgentFailureError(
                    f"too many consecutive provider failures ({consecutive_failures})",
                    consecutive_failures,
                )
            await _backoff_retry(
                consecutive_failures - 1, reason="provider error"
            )


# ---------------------------------------------------------------------------
# Tool dispatch
# ---------------------------------------------------------------------------

def _tool_name(schema: dict[str, Any]) -> str:
    return schema.get("function", {}).get("name", "")


def _normalize_tool_name(name: str) -> str:
    """Forgive minor naming mismatches (hyphens vs underscores, casing)."""
    return name.lower().replace("-", "_").replace(" ", "_")


async def _execute_tool_calls(
    *,
    tool_calls: list[ToolCall],
    tool_dispatch: dict[str, _WrappedTool],
    context: ExecutionContext,
    result_type: type | None,
) -> tuple[list[ToolResultMessage], dict[str, type[ToolCallNode]], Any]:
    """Execute tool calls and return (result_messages, call_node_classes, captured_value).

    *call_node_classes* maps ``call_id`` to the ``ToolCallNode``
    subclass registered on the resolved tool (if any).  Only entries
    with a non-``None`` class are included.

    *captured_value* is ``_RESULT_SENTINEL`` unless a ``return_result``
    call was processed (in structured mode).
    """
    results: list[ToolResultMessage] = []
    call_node_classes: dict[str, type[ToolCallNode]] = {}
    captured: Any = _RESULT_SENTINEL

    for tc in tool_calls:
        # -- parse arguments -----------------------------------------------
        try:
            kwargs = json.loads(tc.arguments) if tc.arguments else {}
        except json.JSONDecodeError as exc:
            results.append(ToolResultMessage(
                call_id=tc.call_id,
                content=f"Invalid JSON arguments: {exc}",
                is_error=True,
            ))
            continue

        # -- structured-mode synthetic tools -------------------------------
        if tc.name == "return_result" and result_type is not None:
            raw_value = kwargs.get("value")
            try:
                captured = validate_result(result_type, raw_value)
            except Exception as exc:
                results.append(ToolResultMessage(
                    call_id=tc.call_id,
                    content=f"Validation error: {exc}. Please try again.",
                    is_error=True,
                ))
                continue
            results.append(ToolResultMessage(
                call_id=tc.call_id, content="ok",
            ))
            continue

        if tc.name == "raise_error":
            msg = kwargs.get("message", "unknown error")
            raise SkillError(msg)

        # -- resolve tool --------------------------------------------------
        tool = tool_dispatch.get(tc.name)
        if tool is None:
            normalized = _normalize_tool_name(tc.name)
            for registered_name, t in tool_dispatch.items():
                if _normalize_tool_name(registered_name) == normalized:
                    tool = t
                    break

        if tool is None:
            results.append(ToolResultMessage(
                call_id=tc.call_id,
                content=f"Unknown tool: {tc.name!r}",
                is_error=True,
            ))
            await context.event_sink.on_tool_end(
                tc.name, error="unknown tool", scope=context.scope,
            )
            continue

        if tool.call_node_class is not None:
            call_node_classes[tc.call_id] = tool.call_node_class

        # -- execute -------------------------------------------------------
        await context.event_sink.on_tool_start(
            tc.name, kwargs, scope=context.scope,
        )
        t0 = time.monotonic()
        try:
            result_str = await tool.execute(**kwargs)
        except SkillError:
            raise
        except Exception as exc:
            duration_s = time.monotonic() - t0
            logger.exception("error executing tool %s", tc.name)
            result_str = None
            results.append(ToolResultMessage(
                call_id=tc.call_id,
                content=f"Error: {exc}",
                is_error=True,
            ))
            await context.event_sink.on_tool_end(
                tc.name, duration_s=duration_s, error=str(exc),
                scope=context.scope,
            )
            continue

        duration_s = time.monotonic() - t0
        results.append(ToolResultMessage(
            call_id=tc.call_id,
            content=result_str if isinstance(result_str, str) else serialize_for_tool_result(result_str),
        ))
        await context.event_sink.on_tool_end(
            tc.name, duration_s=duration_s, scope=context.scope,
        )

    if results:
        tracker = context.validation_tracker
        if tracker is not None:
            tracker.refresh()
            status_line = tracker.render_status()
            if status_line:
                last = results[-1]
                results[-1] = ToolResultMessage(
                    call_id=last.call_id,
                    content=last.content + "\n\n" + status_line,
                    is_error=last.is_error,
                )

    return results, call_node_classes, captured
