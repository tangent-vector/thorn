"""Execution context, scope chain, and event sinks.

The ``ExecutionContext`` is the ambient state carried via a
``contextvars.ContextVar`` so that user code can call ``prompt()`` and
``@skill`` functions without manually threading provider/sink references.
"""

from __future__ import annotations

import contextvars
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from thorn._provider import LLMProvider, ResponseChunk

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# Scope chain (call-graph observability)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Scope:
    """One level in an immutable linked list of nested execution scopes.

    Each entry describes a logical operation boundary (an agent loop
    invocation, a tool call, a skill execution, ...).  The ``outer``
    pointer links to the enclosing scope, forming a chain from innermost
    to outermost.
    """

    description: str
    outer: Scope | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def chain(self) -> list[Scope]:
        """Return the full chain from outermost to innermost."""
        result: list[Scope] = []
        current: Scope | None = self
        while current is not None:
            result.append(current)
            current = current.outer
        result.reverse()
        return result

    @property
    def depth(self) -> int:
        d = 0
        current: Scope | None = self
        while current is not None:
            d += 1
            current = current.outer
        return d


# ---------------------------------------------------------------------------
# Event sink
# ---------------------------------------------------------------------------

class EventSink(ABC):
    """Receives events produced during agent execution."""

    @abstractmethod
    async def on_response_chunk(
        self,
        chunk: ResponseChunk,
        scope: Scope | None = None,
    ) -> None:
        """Called for each streaming chunk from the LLM provider."""
        ...

    @abstractmethod
    async def on_status(
        self,
        message: str,
        scope: Scope | None = None,
    ) -> None:
        """Called for human-readable status messages."""
        ...


class NullEventSink(EventSink):
    """Silently discards all events."""

    async def on_response_chunk(
        self,
        chunk: ResponseChunk,
        scope: Scope | None = None,
    ) -> None:
        pass

    async def on_status(
        self,
        message: str,
        scope: Scope | None = None,
    ) -> None:
        pass


class ConsoleEventSink(EventSink):
    """Prints streaming text and status updates to the console via *rich*.

    Used by the CLI for interactive feedback.
    """

    def __init__(self) -> None:
        from rich.console import Console
        self._console = Console()
        self._in_text = False

    async def on_response_chunk(
        self,
        chunk: ResponseChunk,
        scope: Scope | None = None,
    ) -> None:
        from thorn._provider import TextChunk, ToolCallChunk, FinishChunk

        match chunk:
            case TextChunk():
                self._console.print(chunk.text, end="", highlight=False)
                self._in_text = True
            case ToolCallChunk():
                if self._in_text:
                    self._console.print()
                    self._in_text = False
                indent = "  " * (scope.depth if scope else 1)
                self._console.print(
                    f"{indent}[dim]tool:[/dim] {chunk.name}", highlight=False,
                )
            case FinishChunk():
                if self._in_text:
                    self._console.print()
                    self._in_text = False

    async def on_status(
        self,
        message: str,
        scope: Scope | None = None,
    ) -> None:
        indent = "  " * (scope.depth if scope else 1)
        self._console.print(f"{indent}[dim]{message}[/dim]", highlight=False)


# ---------------------------------------------------------------------------
# Execution context
# ---------------------------------------------------------------------------

@dataclass
class ExecutionContext:
    """The ambient bundle of services for agent execution.

    Attributes:
        provider:   The LLM provider for submitting completion requests.
        event_sink: Sink that receives streaming execution events.
        scope:      Current position in the nested scope chain.
        system_prompts: Extra system-prompt strings that apply to every
                        agent created under this context.
    """

    provider: LLMProvider
    event_sink: EventSink = field(default_factory=NullEventSink)
    scope: Scope | None = None
    system_prompts: list[str] = field(default_factory=list)

    def push_scope(self, description: str, **metadata: Any) -> ExecutionContext:
        """Return a *new* context with one more scope level pushed."""
        new_scope = Scope(
            description=description,
            outer=self.scope,
            metadata=metadata,
        )
        return ExecutionContext(
            provider=self.provider,
            event_sink=self.event_sink,
            scope=new_scope,
            system_prompts=list(self.system_prompts),
        )


# ---------------------------------------------------------------------------
# ContextVar management
# ---------------------------------------------------------------------------

_current_context: contextvars.ContextVar[ExecutionContext] = contextvars.ContextVar(
    "thorn_context",
)


def get_context() -> ExecutionContext:
    """Return the current ``ExecutionContext``.

    Raises ``RuntimeError`` if no context has been set (i.e. ``prompt()``
    was called outside of a ``thorn.run()`` block).
    """
    try:
        return _current_context.get()
    except LookupError:
        raise RuntimeError(
            "No thorn ExecutionContext is active.  "
            "Make sure you are inside a thorn.run() block or have "
            "called set_context() explicitly."
        ) from None


def set_context(ctx: ExecutionContext) -> contextvars.Token[ExecutionContext]:
    """Set the current ``ExecutionContext`` and return a reset token."""
    return _current_context.set(ctx)


def reset_context(token: contextvars.Token[ExecutionContext]) -> None:
    """Restore the ``ExecutionContext`` to its previous value."""
    _current_context.reset(token)
