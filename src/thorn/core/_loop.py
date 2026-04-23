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
import time
from typing import Any

from thorn.core._context import ExecutionContext, Scope
from thorn.core._executor import (
    ExecutorRouter,
    ToolInvocation,
    ToolRegistry,
    ToolVenue,
    build_default_router,
    build_registry_from_wrapped_tools,
    build_split_router,
)
from thorn.core._history import (
    DEFAULT_HIGH_WATERMARK,
    DEFAULT_LOW_WATERMARK,
    AdvisoryNode,
    HistoryTree,
    ToolCallNode,
    estimate_tokens,
)
from thorn.core._messages import (
    AssistantMessage,
    Message,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)
from thorn.core._provider import FinishChunk, TextChunk, ToolCallChunk, UsageChunk
from thorn.core._retry import RetryPolicy
from thorn.core._schema import (
    RAISE_ERROR_SCHEMA,
    make_return_result_schema,
    serialize_for_tool_result,
    validate_result,
)
from thorn.core.errors import (
    AgentFailureError,
    LoopLimitError,
    ProviderError,
    ProviderUnavailableError,
    RateLimitError,
    SkillError,
    TransientProviderError,
)

logger = logging.getLogger(__name__)

# The default retry policy for provider-level calls is loaded from
# environment variables at module import.  Tests that need to
# exercise backoff boundaries without waiting for real seconds
# monkeypatch this attribute (or pass ``asyncio.sleep`` replacements
# into the retry loop via the test harness).  Operators override by
# setting ``THORN_PROVIDER_RETRY_*`` before the daemon starts.
_DEFAULT_RETRY_POLICY = RetryPolicy.from_env()

# Sentinel returned by _execute_tool_calls when the structured-mode
# ``return_result`` tool is invoked.
_RESULT_SENTINEL = object()


class _WrappedTool:
    """Lightweight wrapper that pairs a tool schema with an execute callback.

    When *call_node_class* is set, ``HistoryTree.append_turn`` will
    construct that ``ToolCallNode`` subclass instead of the base class,
    enabling ``isinstance``-based identification in downstream code.

    ``venue`` names the :class:`~thorn.core._executor.ToolVenue` the
    tool is meant to run in.  The agent loop uses it only indirectly
    (via :class:`~thorn.core._executor.ToolRegistry` + router); for
    non-registry callers (``wrap_function`` etc.) it still defaults to
    ``IN_PROCESS`` which matches today's behavior.
    """

    __slots__ = ("schema", "execute", "call_node_class", "venue")

    def __init__(
        self,
        schema: dict[str, Any],
        execute: Any,  # async callable(**kwargs) -> str
        call_node_class: type[ToolCallNode] | None = None,
        venue: ToolVenue = ToolVenue.IN_PROCESS,
    ) -> None:
        self.schema = schema
        self.execute = execute
        self.call_node_class = call_node_class
        self.venue = venue


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
    session: Any = None,
    _housekeeping: bool = False,
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

    When *session* is provided, it is passed to status providers so
    they can tailor advisory content to the active session/agent.

    The *_housekeeping* flag is set by the housekeeping subsystem to
    prevent recursive housekeeping triggers within a sub-loop.
    External callers should not set this.
    """
    structured = result_type is not None and result_type is not str

    # -- filter out ask_user when no handler is available ------------------
    if context.ask_user_handler is None:
        tools = [
            t for t in tools
            if _tool_name(t.schema) != "ask_user"
        ]

    # -- build schemas and registry ----------------------------------------
    all_tools = list(tools)
    registry = build_registry_from_wrapped_tools(all_tools)
    if context.sandbox_executor is not None:
        router = build_split_router(all_tools, context.sandbox_executor)
    else:
        router = build_default_router(all_tools)

    all_schemas = registry.schemas()

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
        result_msgs, call_node_classes, advisory_nodes, captured = (
            await _execute_tool_calls(
                tool_calls=tool_calls,
                registry=registry,
                router=router,
                context=context,
                result_type=result_type if structured else None,
                session=session,
            )
        )

        history.append_turn(
            AssistantMessage(content=text, tool_calls=tool_calls),
            result_msgs,
            advisory_nodes=advisory_nodes or None,
            call_node_classes=call_node_classes or None,
        )

        # -- compaction check + housekeeping trigger ------------------------
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

                # Escalate to housekeeping when compaction alone
                # cannot bring the context below the high watermark.
                if (
                    not _housekeeping
                    and compact_result.tokens_after
                        > int(context_window * DEFAULT_HIGH_WATERMARK)
                ):
                    from thorn.core._housekeeping import perform_housekeeping

                    await perform_housekeeping(
                        context=context,
                        history=history,
                        all_tools=all_tools,
                        system_prompts=system_prompts,
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


async def _sleep_with_backoff(
    policy: RetryPolicy,
    attempt: int,
    *,
    retry_after: float | None,
    reason: str,
) -> None:
    """Sleep for ``policy``'s backoff delay, honouring ``retry_after``."""
    delay = policy.backoff_delay(attempt, retry_after=retry_after)
    logger.info(
        "%s, retrying in %.1fs (attempt %d)",
        reason, delay, attempt + 1,
    )
    await asyncio.sleep(delay)


async def _request_completion(
    *,
    context: ExecutionContext,
    system_prompts: list[str],
    tool_schemas: list[dict],
    messages: list[Message],
    consecutive_failures: int,
    max_failures: int,
    retry_policy: RetryPolicy | None = None,
) -> tuple[str, list[ToolCall], str, dict[str, int] | None]:
    """Return ``(text, tool_calls, finish_reason, usage)``.

    Retry policy:

    - :class:`RateLimitError` (HTTP 429 or equivalent) -- retried
      with full-jitter exponential backoff up to
      ``retry_policy.max_rate_limit_retries`` attempts, honouring
      any ``Retry-After`` hint from the server.  Exhaustion raises
      :class:`ProviderUnavailableError`.
    - :class:`TransientProviderError` (transport-level failures,
      502/503/504) -- same backoff shape, bounded by
      ``retry_policy.max_transient_retries``.  Exhaustion raises
      :class:`ProviderUnavailableError`.
    - Non-transient :class:`ProviderError` -- counted against
      ``max_failures``.  Exhaustion raises
      :class:`AgentFailureError`.  These are almost always
      indicative of a configuration or request-shape problem (4xx
      responses other than 429) rather than a network issue, so
      they stay on the agent-loop's own failure budget.

    The ``consecutive_failures`` counter is seeded from the caller
    so that successive calls to this function (one per tool round)
    can share a budget; the caller resets it to zero on a
    successful return.  Rate-limit and transient-retry budgets are
    per-call and do not accumulate across rounds -- a few network
    blips spaced across many rounds should not lock the session
    out of making progress.
    """
    policy = retry_policy or _DEFAULT_RETRY_POLICY
    rate_limit_retries = 0
    transient_retries = 0

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

        except RateLimitError as exc:
            if rate_limit_retries >= policy.max_rate_limit_retries:
                raise ProviderUnavailableError(
                    f"rate-limit retries exhausted after "
                    f"{rate_limit_retries + 1} attempt(s): {exc}",
                    attempts=rate_limit_retries + 1,
                ) from exc
            await _sleep_with_backoff(
                policy,
                rate_limit_retries,
                retry_after=exc.retry_after,
                reason="rate limited",
            )
            rate_limit_retries += 1

        except TransientProviderError as exc:
            if transient_retries >= policy.max_transient_retries:
                raise ProviderUnavailableError(
                    f"transient provider retries exhausted after "
                    f"{transient_retries + 1} attempt(s): {exc}",
                    attempts=transient_retries + 1,
                ) from exc
            await _sleep_with_backoff(
                policy,
                transient_retries,
                retry_after=exc.retry_after,
                reason="transient provider error",
            )
            transient_retries += 1

        except ProviderError:
            # Non-transient provider error: counts against the
            # agent-level failure budget rather than the per-call
            # transient budget, so repeated hard failures across
            # multiple tool rounds eventually break out of the
            # loop with :class:`AgentFailureError` (rather than
            # being masked as "transient" forever).
            consecutive_failures += 1
            if consecutive_failures >= max_failures:
                raise AgentFailureError(
                    f"too many consecutive provider failures "
                    f"({consecutive_failures})",
                    consecutive_failures,
                )
            await _sleep_with_backoff(
                policy,
                consecutive_failures - 1,
                retry_after=None,
                reason="provider error",
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
    registry: ToolRegistry,
    router: ExecutorRouter,
    context: ExecutionContext,
    result_type: type | None,
    session: Any = None,
) -> tuple[list[ToolResultMessage], dict[str, type[ToolCallNode]], list[AdvisoryNode], Any]:
    """Execute tool calls and return (result_messages, call_node_classes, advisory_nodes, captured_value).

    Dispatch walks through the :class:`ToolRegistry` (for name
    resolution, venue, and history-node class) and the
    :class:`ExecutorRouter` (for the actual invocation).  The two are
    split so that Phase A's sandbox work can swap the executor for the
    ``SANDBOX`` venue without any changes to the loop itself.

    *call_node_classes* maps ``call_id`` to the ``ToolCallNode``
    subclass registered on the resolved tool (if any).  Only entries
    with a non-``None`` class are included.

    *advisory_nodes* contains any advisories produced by registered
    status providers for this tool-call round.

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

        # -- resolve registry entry ----------------------------------------
        entry = registry.get(tc.name)
        if entry is None:
            normalized = _normalize_tool_name(tc.name)
            for candidate in registry.entries():
                if _normalize_tool_name(candidate.name) == normalized:
                    entry = candidate
                    break

        if entry is None:
            results.append(ToolResultMessage(
                call_id=tc.call_id,
                content=f"Unknown tool: {tc.name!r}",
                is_error=True,
            ))
            await context.event_sink.on_tool_end(
                tc.name, error="unknown tool", scope=context.scope,
            )
            continue

        if entry.call_node_class is not None:
            call_node_classes[tc.call_id] = entry.call_node_class

        invocation = ToolInvocation(
            call_id=tc.call_id,
            tool_name=entry.name,
            arguments=kwargs,
        )

        # -- execute via the venue's executor ------------------------------
        await context.event_sink.on_tool_start(
            tc.name, kwargs, scope=context.scope,
        )
        executor = router.for_venue(entry.venue)
        t0 = time.monotonic()
        try:
            outcome = await executor.invoke(invocation)
        except SkillError:
            raise
        except Exception as exc:
            duration_s = time.monotonic() - t0
            logger.exception("error executing tool %s", tc.name)
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
            content=outcome.content,
            is_error=outcome.is_error,
        ))
        await context.event_sink.on_tool_end(
            tc.name,
            duration_s=duration_s,
            error=outcome.content if outcome.is_error else None,
            scope=context.scope,
        )

    advisory_nodes: list[AdvisoryNode] = []
    if results:
        for provider in context.status_providers:
            provider.refresh(session)
            text = provider.render_status(session)
            if text:
                advisory_nodes.append(AdvisoryNode(
                    source=provider.source_label,
                    content=text,
                ))
                await context.event_sink.on_advisory(
                    provider.source_label, text, scope=context.scope,
                )

    return results, call_node_classes, advisory_nodes, captured
