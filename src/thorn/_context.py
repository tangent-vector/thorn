"""Execution context, scope chain, and event sinks.

The ``ExecutionContext`` is the ambient state carried via a
``contextvars.ContextVar`` so that user code can call ``prompt()`` and
``@skill`` functions without manually threading provider/sink references.
"""

from __future__ import annotations

import contextvars
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import IntEnum
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
# Verbosity
# ---------------------------------------------------------------------------

class Verbosity(IntEnum):
    """Controls how much detail the console event sink shows."""
    QUIET = 0
    NORMAL = 1
    VERBOSE = 2
    DEBUG = 3


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

    async def on_scope_enter(self, scope: Scope) -> None:
        """Called when an agent/prompt/skill scope is entered."""
        await self.on_status(scope.description, scope=scope)

    async def on_scope_exit(
        self, scope: Scope, *, duration_s: float | None = None,
    ) -> None:
        """Called when an agent/prompt/skill scope is exited."""
        label = scope.description
        if duration_s is not None:
            label += f" ({duration_s:.1f}s)"
        await self.on_status(label, scope=scope)

    async def on_tool_start(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        scope: Scope | None = None,
    ) -> None:
        """Called just before a tool begins executing."""
        await self.on_status(f"tool: {name}", scope=scope)

    async def on_tool_end(
        self,
        name: str,
        *,
        duration_s: float | None = None,
        error: str | None = None,
        scope: Scope | None = None,
    ) -> None:
        """Called after a tool finishes executing."""
        if error:
            await self.on_status(f"tool {name}: error \u2014 {error}", scope=scope)
        else:
            label = f"tool {name}: ok"
            if duration_s is not None:
                label += f" ({duration_s:.1f}s)"
            await self.on_status(label, scope=scope)


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

    async def on_scope_enter(self, scope: Scope) -> None:
        pass

    async def on_scope_exit(
        self, scope: Scope, *, duration_s: float | None = None,
    ) -> None:
        pass

    async def on_tool_start(
        self, name: str, arguments: dict[str, Any], *, scope: Scope | None = None,
    ) -> None:
        pass

    async def on_tool_end(
        self, name: str, *, duration_s: float | None = None,
        error: str | None = None, scope: Scope | None = None,
    ) -> None:
        pass


class ConsoleEventSink(EventSink):
    """Prints streaming text and status updates to the console via *rich*.

    Used by the CLI for interactive feedback.  The *verbosity* parameter
    controls how much detail is shown (see :class:`Verbosity`).
    """

    _UNICODE_SYMBOLS = ("\u25b6", "\u2713", "\u2717", "\u2500")  # ▶ ✓ ✗ ─
    _ASCII_SYMBOLS = (">", "+", "x", "-")

    def __init__(self, verbosity: Verbosity = Verbosity.NORMAL) -> None:
        from rich.console import Console

        self._console = Console()
        self._verbosity = verbosity
        self._in_text = False

        use_unicode = self._console.encoding.lower().startswith("utf")
        syms = self._UNICODE_SYMBOLS if use_unicode else self._ASCII_SYMBOLS
        self._sym_start, self._sym_ok, self._sym_fail, self._sym_rule = syms

    def _safe_print(self, *args: Any, **kwargs: Any) -> None:
        """Print with fallback for terminals that cannot encode all characters."""
        try:
            self._console.print(*args, **kwargs)
        except UnicodeEncodeError:
            sanitized = tuple(
                s.encode("ascii", errors="replace").decode("ascii")
                if isinstance(s, str) else s
                for s in args
            )
            try:
                self._console.print(*sanitized, **kwargs)
            except UnicodeEncodeError:
                pass

    def _indent(self, scope: Scope | None) -> str:
        return "  " * (scope.depth if scope else 1)

    def _end_text(self) -> None:
        if self._in_text:
            self._safe_print()
            self._in_text = False

    @staticmethod
    def _abbreviate_args(arguments: dict[str, Any], max_len: int = 60) -> str:
        """Format tool arguments as key=value pairs, truncating long values."""
        parts: list[str] = []
        for key, value in arguments.items():
            s = json.dumps(value) if not isinstance(value, str) else value
            if len(s) > max_len:
                s = f"<{len(s)} chars>"
            else:
                s = repr(s) if isinstance(value, str) else s
            parts.append(f"{key}={s}")
        return " ".join(parts)

    def _scope_label(self, scope: Scope) -> str:
        """Derive a human-readable label for a scope header.

        If the scope carries an agent, use ``str(agent)``; otherwise
        fall back to the raw scope description.
        """
        agent = scope.metadata.get("agent")
        if agent is not None:
            return str(agent)
        return scope.description

    def _print_rule(self, label: str, scope: Scope, *, suffix: str = "") -> None:
        """Print a horizontal-rule header/footer line for a scope."""
        indent = self._indent(scope.outer)
        rule_char = self._sym_rule
        text = f"{label}{suffix}"
        padding = max(0, 40 - len(indent) - len(text) - 4)
        line = f"{indent}{rule_char}{rule_char} {text} {rule_char * padding}"
        self._safe_print(f"[dim]{line}[/dim]", highlight=False)

    # -- response chunks ---------------------------------------------------

    async def on_response_chunk(
        self,
        chunk: ResponseChunk,
        scope: Scope | None = None,
    ) -> None:
        from thorn._provider import TextChunk, ToolCallChunk, FinishChunk

        match chunk:
            case TextChunk():
                if self._verbosity >= Verbosity.NORMAL:
                    self._safe_print(chunk.text, end="", highlight=False)
                    self._in_text = True
            case ToolCallChunk():
                self._end_text()
                if self._verbosity >= Verbosity.DEBUG:
                    indent = self._indent(scope)
                    self._safe_print(
                        f"{indent}[dim]llm requested:[/dim] {chunk.name}",
                        highlight=False,
                    )
            case FinishChunk():
                self._end_text()

    # -- status (fallback) -------------------------------------------------

    async def on_status(
        self,
        message: str,
        scope: Scope | None = None,
    ) -> None:
        if self._verbosity < Verbosity.NORMAL:
            return
        indent = self._indent(scope)
        self._safe_print(f"{indent}[dim]{message}[/dim]", highlight=False)

    # -- scope lifecycle ---------------------------------------------------

    async def on_scope_enter(self, scope: Scope) -> None:
        if self._verbosity < Verbosity.NORMAL:
            return
        self._end_text()
        self._print_rule(self._scope_label(scope), scope)

    async def on_scope_exit(
        self, scope: Scope, *, duration_s: float | None = None,
    ) -> None:
        if self._verbosity < Verbosity.NORMAL:
            return
        self._end_text()
        suffix = f" ({duration_s:.1f}s)" if duration_s is not None else ""
        self._print_rule(self._scope_label(scope), scope, suffix=suffix)

    # -- tool lifecycle ----------------------------------------------------

    async def on_tool_start(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        scope: Scope | None = None,
    ) -> None:
        if self._verbosity < Verbosity.NORMAL:
            return
        self._end_text()
        indent = self._indent(scope)
        line = f"{indent}{self._sym_start} {name}"
        if self._verbosity >= Verbosity.VERBOSE and arguments:
            line += f" {self._abbreviate_args(arguments)}"
        self._safe_print(line, highlight=False)

    async def on_tool_end(
        self,
        name: str,
        *,
        duration_s: float | None = None,
        error: str | None = None,
        scope: Scope | None = None,
    ) -> None:
        if self._verbosity < Verbosity.NORMAL:
            return
        indent = self._indent(scope)
        if error:
            self._safe_print(
                f"{indent}[red]{self._sym_fail}[/red] {name}: {error}",
                highlight=False,
            )
        else:
            timing = f" ({duration_s:.1f}s)" if duration_s is not None else ""
            self._safe_print(
                f"{indent}[green]{self._sym_ok}[/green] {name}{timing}",
                highlight=False,
            )


# ---------------------------------------------------------------------------
# Execution context
# ---------------------------------------------------------------------------

_UNSET = object()


@dataclass
class ExecutionContext:
    """The ambient bundle of services for agent execution.

    Attributes:
        provider:   The LLM provider for submitting completion requests.
        event_sink: Sink that receives streaming execution events.
        scope:      Current position in the nested scope chain.
        system_prompts: Extra system-prompt strings that apply to every
                        agent created under this context.
        agent:      The current ``Agent`` instance, if running inside one.
    """

    provider: LLMProvider
    event_sink: EventSink = field(default_factory=NullEventSink)
    scope: Scope | None = None
    system_prompts: list[str] = field(default_factory=list)
    agent: Any = None

    def push_scope(
        self, description: str, *, agent: Any = _UNSET, **metadata: Any,
    ) -> ExecutionContext:
        """Return a *new* context with one more scope level pushed.

        The *agent* field is propagated from the parent unless explicitly
        overridden (pass ``agent=<instance>`` or ``agent=None``).
        The resolved agent is also stored in ``scope.metadata["agent"]``
        so that event sinks can access it without the full context.
        """
        resolved_agent = self.agent if agent is _UNSET else agent
        if resolved_agent is not None:
            metadata.setdefault("agent", resolved_agent)
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
            agent=resolved_agent,
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
