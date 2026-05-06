"""Execution context, scope chain, and event sinks.

The ``ExecutionContext`` is the ambient state carried via a
``contextvars.ContextVar`` so that user code can call ``prompt()`` and
``@skill`` functions without manually threading provider/sink references.
"""

from __future__ import annotations

import contextvars
import json
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import IntEnum
from typing import TYPE_CHECKING, Any, Generator, Protocol

from thorn.core._provider import LLMProvider, ResponseChunk

if TYPE_CHECKING:
    from pathlib import Path

    from thorn.core._file_access import FileAccessPolicy
    from thorn.core._session import Session
    from thorn.core._validation_tracker import ValidationTracker


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
# Status provider protocol
# ---------------------------------------------------------------------------

class StatusProvider(Protocol):
    """Cross-cutting system that can inject advisory status text.

    Implementations provide a ``source_label`` for identification and
    a ``refresh`` / ``render_status`` pair that the agent loop calls
    once per tool-call round.

    Both ``refresh`` and ``render_status`` receive the ``Session``
    being annotated so that a provider can inspect the agent's role,
    workspace, or session metadata when deciding what (if anything) to
    report.  Providers that don't care about session identity simply
    ignore the parameter.
    """

    @property
    def source_label(self) -> str:
        """Short identifier, e.g. ``'validation'``, ``'inbox'``."""
        ...

    def refresh(self, session: Session | None) -> None:
        """Re-evaluate current state (called once per tool round).

        *session* is the active session, or ``None`` when running
        outside of a session (e.g. bare ``thorn.run()``).  Providers
        that only apply to certain agents/sessions can check
        ``session.agent`` and short-circuit.
        """
        ...

    def render_status(self, session: Session | None) -> str | None:
        """Status text for the agent, or ``None`` if nothing to report.

        Called after ``refresh``.  A provider that is irrelevant to the
        given session should return ``None``.
        """
        ...


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

    async def on_completion_end(
        self,
        *,
        duration_s: float | None = None,
        usage: dict[str, int] | None = None,
        scope: Scope | None = None,
    ) -> None:
        """Called after each LLM completion round finishes."""
        label = "completion"
        if usage:
            total = usage.get("total_tokens")
            if total is not None:
                label += f": {total} tokens"
        if duration_s is not None:
            label += f" ({duration_s:.1f}s)"
        await self.on_status(label, scope=scope)

    async def on_advisory(
        self,
        source: str,
        content: str,
        *,
        scope: Scope | None = None,
    ) -> None:
        """Called when a status provider emits an advisory for the agent."""
        await self.on_status(f"[{source}] {content}", scope=scope)


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

    async def on_completion_end(
        self, *, duration_s: float | None = None,
        usage: dict[str, int] | None = None,
        scope: Scope | None = None,
    ) -> None:
        pass

    async def on_advisory(
        self,
        source: str,
        content: str,
        *,
        scope: Scope | None = None,
    ) -> None:
        pass


class ConsoleEventSink(EventSink):
    """Prints streaming text and status updates to the console via *rich*.

    Used by the CLI for interactive feedback.  The *verbosity* parameter
    controls how much detail is shown (see :class:`Verbosity`).
    """

    _UNICODE_SYMBOLS = ("\u25b6", "\u2713", "\u2717", "\u2500")  # ▶ ✓ ✗ ─
    _ASCII_SYMBOLS = (">", "+", "x", "-")

    # Tool names mapped to the argument key(s) worth showing at NORMAL
    # verbosity.  Tools not listed here show only their name at NORMAL
    # (and the full argument dump at VERBOSE, as before).
    _TOOL_SUMMARY_KEYS: dict[str, list[str]] = {
        "run_shell": ["command"],
        "read_file": ["path"],
        "edit_file": ["path"],
        "create_file": ["path"],
        "delete_file": ["path"],
        "move_file": ["source", "destination"],
        "find_files": ["pattern", "path"],
        "search_files": ["pattern", "path"],
        "list_directory": ["path"],
    }

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

    _TOOL_SUMMARY_MAX_LEN: int = 80

    @classmethod
    def _summarize_tool_args(
        cls,
        name: str,
        arguments: dict[str, Any],
    ) -> str | None:
        """Return a short summary string for well-known tools, or ``None``.

        Only extracts the most meaningful argument(s) listed in
        ``_TOOL_SUMMARY_KEYS``.  Values are shown bare (no key= prefix)
        when there is a single summary key, or as ``key=value`` pairs
        when there are multiple.
        """
        keys = cls._TOOL_SUMMARY_KEYS.get(name)
        if not keys:
            return None
        values = [arguments.get(k) for k in keys]
        if not any(v is not None for v in values):
            return None
        parts: list[str] = []
        for key, value in zip(keys, values):
            if value is None:
                continue
            s = value if isinstance(value, str) else json.dumps(value)
            if len(s) > cls._TOOL_SUMMARY_MAX_LEN:
                s = s[: cls._TOOL_SUMMARY_MAX_LEN] + "\u2026"
            if len(keys) > 1:
                parts.append(f"{key}={s}")
            else:
                parts.append(s)
        return " ".join(parts)

    def _scope_label(self, scope: Scope) -> str:
        """Derive a human-readable label for a scope header.

        If the scope carries an agent, use ``str(agent)``; otherwise
        fall back to the raw scope description.  When a ``session_key``
        is present in the scope metadata, it is appended in brackets.
        """
        agent = scope.metadata.get("agent")
        base = str(agent) if agent is not None else scope.description
        session_key = scope.metadata.get("session_key")
        if session_key:
            return f"{base} [{session_key}]"
        return base

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
        from thorn.core._provider import TextChunk, ToolCallChunk, FinishChunk

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
        elif arguments:
            summary = self._summarize_tool_args(name, arguments)
            if summary:
                line += f": {summary}"
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

    # -- completion lifecycle -----------------------------------------------

    async def on_completion_end(
        self,
        *,
        duration_s: float | None = None,
        usage: dict[str, int] | None = None,
        scope: Scope | None = None,
    ) -> None:
        if self._verbosity < Verbosity.VERBOSE:
            return
        indent = self._indent(scope)
        label = "completion"
        if usage:
            total = usage.get("total_tokens")
            if total is not None:
                label += f": {total} tokens"
        if duration_s is not None:
            label += f" ({duration_s:.1f}s)"
        self._safe_print(f"{indent}[dim]{label}[/dim]", highlight=False)

    # -- advisory events ----------------------------------------------------

    async def on_advisory(
        self,
        source: str,
        content: str,
        *,
        scope: Scope | None = None,
    ) -> None:
        if self._verbosity < Verbosity.NORMAL:
            return
        self._end_text()
        indent = self._indent(scope)
        self._safe_print(
            f"{indent}[dim][{source}] {content}[/dim]", highlight=False,
        )


# ---------------------------------------------------------------------------
# Usage tracking
# ---------------------------------------------------------------------------

@dataclass
class UsageTracker:
    """Accumulates LLM token usage across completions.

    A single instance is shared across nested scopes so that the root
    context holds the grand totals after a run completes.
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def add(self, prompt: int, completion: int, total: int) -> None:
        self.prompt_tokens += prompt
        self.completion_tokens += completion
        self.total_tokens += total


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
        workspace_root: Resolved absolute path to the workspace directory.
                        File access patterns are matched relative to this.
        file_access_policy: Active file-access policy for the current
                            agent scope.  ``None`` only at the root
                            context before any agent scope is pushed.
        global_ignores: Ceiling policy loaded from ``.aiignore`` /
                        ``.thornignore`` at startup.  Applied as an upper
                        bound on every per-agent policy.
        usage:      Shared accumulator for LLM token counts.
        status_providers: List of cross-cutting ``StatusProvider``
                          instances that inject advisory text into
                          the agent loop after tool-call rounds.
                          Propagated by reference across nested
                          scopes (shared list, not copied).
        agency_root_directory: The top-level directory that owns the
                               ``.thorn/`` directory.  Used to derive
                               agent home paths
                               (``<agency_root>/.thorn/agents/<id>/``).
                               Set once by the ``Runtime`` or
                               ``thorn.run()`` and inherited unchanged
                               by all child contexts.
        runtime: Reference to the :class:`~thorn.runtime.Runtime` that
                 owns this context.  ``None`` for bare contexts created
                 without a runtime (e.g. unit tests).  Propagated
                 unchanged through :meth:`push_scope`.
    """

    provider: LLMProvider
    event_sink: EventSink = field(default_factory=NullEventSink)
    scope: Scope | None = None
    system_prompts: list[str] = field(default_factory=list)
    agent: Any = None
    workspace_root: Path | None = None
    file_access_policy: FileAccessPolicy | None = None
    global_ignores: FileAccessPolicy | None = None
    usage: UsageTracker = field(default_factory=UsageTracker)
    context_window: int | None = None
    status_providers: list[StatusProvider] = field(default_factory=list)
    agency_root_directory: Path | None = None
    runtime: Any = None
    sandbox_executor: Any = None
    """Optional :class:`~thorn.core._executor.ToolExecutor` for ``SANDBOX``.

    When ``None`` (the default), :func:`run_agent_loop` runs every
    venue in-process -- the historical behavior, retained for tests
    and for callers that have not opted into the toolhost daemon.
    When set, the loop builds a split router whose ``SANDBOX`` venue
    dispatches through this executor (typically a per-agent
    :class:`~thorn.toolhost.DaemonToolExecutor`) while ``IN_PROCESS``
    tools (e.g. the inbox tools) still execute inline.
    """

    def add_status_provider(self, provider: StatusProvider) -> None:
        """Register a ``StatusProvider`` for advisory injection."""
        self.status_providers.append(provider)

    @property
    def validation_tracker(self) -> ValidationTracker | None:
        """Backward-compatible accessor for a ``ValidationTracker`` provider."""
        from thorn.core._validation_tracker import ValidationTracker as _VT
        for p in self.status_providers:
            if isinstance(p, _VT):
                return p
        return None

    @validation_tracker.setter
    def validation_tracker(self, tracker: ValidationTracker | None) -> None:
        from thorn.core._validation_tracker import ValidationTracker as _VT
        self.status_providers = [
            p for p in self.status_providers if not isinstance(p, _VT)
        ]
        if tracker is not None:
            self.status_providers.append(tracker)

    def push_scope(
        self,
        description: str,
        *,
        agent: Any = _UNSET,
        file_access_policy: FileAccessPolicy | None = _UNSET,  # type: ignore[assignment]
        **metadata: Any,
    ) -> ExecutionContext:
        """Return a *new* context with one more scope level pushed.

        The *agent* field is propagated from the parent unless explicitly
        overridden (pass ``agent=<instance>`` or ``agent=None``).
        The resolved agent is also stored in ``scope.metadata["agent"]``
        so that event sinks can access it without the full context.

        *file_access_policy* follows the same propagation rule: it
        inherits from the parent unless explicitly supplied.
        """
        resolved_agent = self.agent if agent is _UNSET else agent
        resolved_policy = (
            self.file_access_policy
            if file_access_policy is _UNSET
            else file_access_policy
        )
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
            workspace_root=self.workspace_root,
            file_access_policy=resolved_policy,
            global_ignores=self.global_ignores,
            usage=self.usage,
            context_window=self.context_window,
            status_providers=self.status_providers,
            agency_root_directory=self.agency_root_directory,
            runtime=self.runtime,
            sandbox_executor=self.sandbox_executor,
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


@contextmanager
def scoped_status_provider(provider: StatusProvider):
    """Register *provider* on the current context for the duration of the block.

    Removes the provider from ``status_providers`` on exit, even if an
    exception is raised.
    """
    ctx = get_context()
    ctx.status_providers.append(provider)
    try:
        yield provider
    finally:
        try:
            ctx.status_providers.remove(provider)
        except ValueError:
            pass


def _agent_home() -> Path | None:
    """Return the current agent's home directory, or ``None``.

    Used by :func:`resolve_path` and :func:`shell_env` to map ``~`` to
    the agent's personal state directory rather than the OS-level
    ``$HOME``.
    """
    try:
        ctx = get_context()
        agent = ctx.agent
        if agent is not None:
            home = getattr(agent, "home", None)
            if home is not None:
                return home
    except RuntimeError:
        pass
    return None


def resolve_path(raw: str | Path) -> Path:
    """Resolve *raw* against the active workspace, returning an absolute path.

    Tool implementations should call this on every user-supplied path
    argument so that relative paths are interpreted relative to the
    agent's workspace rather than the process CWD.

    Resolution rules:

    1. **Tilde paths** (``~`` or ``~/…``) are expanded against the
       current agent's :attr:`~Agent.home` directory when one is
       available.  This makes ``~/MEMORY.md`` refer to the agent's
       personal state regardless of the workspace.  When no agent home
       is available, falls back to :func:`os.path.expanduser`.
    2. **Absolute paths** are returned as-is (canonicalized via
       :meth:`~pathlib.Path.resolve`).
    3. **Relative paths** are joined to the current
       ``ExecutionContext.workspace_root`` when a context is active and
       a workspace is set.
    4. If no context or workspace is available, the path is resolved
       against the process CWD (matching default ``pathlib`` behavior).
    """
    from pathlib import Path as _Path

    s = str(raw)

    if s == "~" or s.startswith("~/"):
        home = _agent_home()
        if home is not None:
            if s == "~":
                return home.resolve()
            return (home / s[2:]).resolve()
        # No agent home — fall back to OS-level tilde expansion so
        # that ~/… is still meaningful rather than treated as a
        # literal relative path component named "~".
        import os
        s = os.path.expanduser(s)

    p = _Path(s)
    if p.is_absolute():
        return p.resolve()

    try:
        ctx = get_context()
        if ctx.workspace_root is not None:
            return (ctx.workspace_root / p).resolve()
    except RuntimeError:
        pass

    return (_Path.cwd() / p).resolve()


def shell_env() -> dict[str, str] | None:
    """Build an environment dict for agent shell subprocesses.

    When an agent with a home directory is active, overrides ``$HOME``
    so that shell tilde expansion (``~``) agrees with the built-in file
    tools' :func:`resolve_path`.  Returns ``None`` (inherit the process
    environment unchanged) when no agent home is available.
    """
    home = _agent_home()
    if home is None:
        return None
    import os
    env = os.environ.copy()
    env["HOME"] = str(home)
    return env
