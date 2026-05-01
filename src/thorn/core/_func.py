"""Unified function abstraction: ``prompt``, ``@skill``, ``@tool``, and ``wrap_function``.

This module provides the ways user code interacts with the agent system:

* ``prompt("...")`` / ``prompt[T]("...")`` — inline ad-hoc prompts
* ``@skill`` — decorator that turns a function stub + docstring into a
  prompt-based callable
* ``@tool`` — marker decorator that tags a Python function for
  auto-discovery (body is unchanged)
* ``wrap_function(fn)`` — wraps any Python function so it can be handed
  to an agent as a tool
"""

from __future__ import annotations

import asyncio
import functools
import inspect
import json
import time
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, Callable, Generic, TypeVar, get_type_hints, overload

if TYPE_CHECKING:
    from thorn.core._history import ToolCallNode

from thorn.core._context import ExecutionContext, get_context, reset_context, set_context
from thorn.core._executor import ToolVenue
from thorn.core._loop import _WrappedTool, run_agent_loop
from thorn.core._schema import func_to_tool_schema, serialize_for_tool_result

T = TypeVar("T")


# ---------------------------------------------------------------------------
# wrap_function — turn any Python function into a tool for an agent
# ---------------------------------------------------------------------------

def _build_param_coercers(
    fn: Callable[..., Any],
) -> dict[str, TypeAdapter[Any]]:
    """Build a ``TypeAdapter`` for each typed parameter of *fn*.

    Used at call time to coerce raw JSON-parsed dicts/lists into the
    annotated Python types (dataclasses, Pydantic models, etc.).
    """
    from pydantic import TypeAdapter as _TA

    hints = get_type_hints(fn)
    sig = inspect.signature(fn)
    coercers: dict[str, TypeAdapter[Any]] = {}
    for name, param in sig.parameters.items():
        annotation = hints.get(name, inspect.Parameter.empty)
        if annotation is inspect.Parameter.empty or annotation is Any:
            continue
        coercers[name] = _TA(annotation)
    return coercers


def wrap_function(
    fn: Callable[..., Any],
    *,
    venue: ToolVenue | None = None,
) -> _WrappedTool:
    """Wrap a Python function so it can be used as a tool by an agent.

    The function's name, docstring, and typed parameters are
    introspected to produce an OpenAI-style tool schema.  At
    invocation time the JSON arguments from the model are
    deserialized and passed to the function, and the return value
    is serialized back to a string.

    *venue* tells the executor router whether to dispatch this tool
    in-process (``ToolVenue.IN_PROCESS``) or via the toolhost daemon
    (``ToolVenue.SANDBOX``).  If *fn* was decorated with ``@tool``
    its venue is already stamped onto the function and *venue* may
    be omitted; otherwise *venue* is **required** and there is no
    default.  See :func:`tool` for the rationale: a silent default
    is exactly the failure mode that allowed the daemon registry to
    silently drift away from the brain's tool list.

    If *fn* has a ``_thorn_call_node_class`` attribute (set by
    ``@tool(call_node_class=...)``), it is forwarded to the wrapped
    tool so that history recording uses the correct ``ToolCallNode``
    subclass.
    """
    schema = func_to_tool_schema(fn)
    is_async = asyncio.iscoroutinefunction(fn)
    coercers = _build_param_coercers(fn)

    async def execute(**kwargs: Any) -> str:
        for name, adapter in coercers.items():
            if name in kwargs:
                kwargs[name] = adapter.validate_python(kwargs[name])
        if is_async:
            result = await fn(**kwargs)
        else:
            result = fn(**kwargs)
        return serialize_for_tool_result(result)

    call_node_class = getattr(fn, "_thorn_call_node_class", None)
    # Venue is mandatory: there is no default.  An explicit *venue*
    # argument (used by direct ``wrap_function`` callers) takes
    # precedence over a ``_thorn_venue`` attribute already stamped on
    # the function (the common path for ``@tool``-decorated builtins).
    # A silent default would hide the same kind of layering bug that
    # let ``FORGE_TOOLS`` and ``GIT_TOOLS`` get treated as sandbox-bound
    # but never registered with the toolhost daemon, so calls into them
    # returned ``unknown_tool`` only at first invocation in production.
    # Forcing the choice to be explicit at the tool's definition site
    # is the smallest fence that keeps that drift from recurring.
    resolved_venue = venue if venue is not None else getattr(fn, "_thorn_venue", None)
    if resolved_venue is None:
        name = getattr(fn, "__name__", repr(fn))
        raise TypeError(
            f"{name!r} has no venue.  Decorate it with "
            "@tool(venue=ToolVenue.SANDBOX) for tools that take "
            "untrusted arguments and execute via the toolhost daemon, "
            "or @tool(venue=ToolVenue.IN_PROCESS) for tools that need "
            "brain-side runtime state (agency / credential broker / "
            "peer registry / inbox).  Direct callers of wrap_function "
            "may pass venue= explicitly instead."
        )
    return _WrappedTool(
        schema=schema,
        execute=execute,
        call_node_class=call_node_class,
        venue=resolved_venue,
    )


def _flatten_tools(items: Iterable[Any]) -> Iterable[Any]:
    """Recursively flatten nested iterables of tools into individual items.

    Leaf items (callables, ``_WrappedTool`` instances, and non-iterable
    objects) are yielded as-is.  Lists, tuples, and other non-string
    iterables are recursed into.  This allows toolset constants like
    ``FILE_READING = [read_file, list_directory, search_files]`` to be
    nested inside another tool list::

        tools = [FILE_READING, write_file]
        # flattens to [read_file, list_directory, search_files, write_file]
    """
    for item in items:
        if isinstance(item, _WrappedTool) or callable(item):
            yield item
        elif isinstance(item, Iterable) and not isinstance(item, (str, bytes)):
            yield from _flatten_tools(item)
        else:
            yield item


_KNOWN_BUILTIN_TOOLS: set[Any] | None = None


def _known_builtin_tools() -> set[Any]:
    """Return the set of callable references that are valid Thorn built-ins.

    Used by :func:`_prepare_tools` to enforce the Phase A rule: agents
    may pass references to known built-ins (and pre-packaged toolset
    constants like ``FILE_READING``), but arbitrary user-supplied
    callables are rejected.  The substitute for user-authored behavior
    is skills + scripts invoked via ``run_shell`` (see
    ``docs/plans/sandbox_tool_execution_*.plan.md``).

    The set is built lazily because importing the catalog module at
    decorator-evaluation time would create circular imports against
    this module.  Once built, the catalog is the single source of
    truth: anything not in :data:`thorn.tools._catalog.ALL_BUILTIN_TOOL_FUNCTIONS`
    is rejected as "not a registered Thorn tool", which keeps the
    brain's allowlist and the toolhost daemon's registry from
    drifting apart.
    """
    global _KNOWN_BUILTIN_TOOLS
    if _KNOWN_BUILTIN_TOOLS is not None:
        return _KNOWN_BUILTIN_TOOLS

    from thorn.tools._catalog import ALL_BUILTIN_TOOL_FUNCTIONS

    _KNOWN_BUILTIN_TOOLS = set(ALL_BUILTIN_TOOL_FUNCTIONS)
    return _KNOWN_BUILTIN_TOOLS


def _prepare_tools(raw_tools: list[Any] | None) -> list[_WrappedTool]:
    """Normalise a user-supplied tool list into ``_WrappedTool`` instances.

    Accepts a mix of:
    - ``_WrappedTool`` instances (passed through)
    - references to **known built-in** Thorn tools (auto-wrapped via
      ``wrap_function``)
    - nested iterables of the above (recursively flattened)

    Arbitrary Python callables are *not* accepted: in Phase A of the
    sandbox roadmap the registry is limited to tools the brain and the
    tool-host daemon both know about statically, and there is no
    transitional shim for user-supplied callables.  The forward-looking
    substitute is skills bundled with shell scripts that the agent
    invokes via ``run_shell``.
    """
    if not raw_tools:
        return []
    result: list[_WrappedTool] = []
    builtins = None
    for item in _flatten_tools(raw_tools):
        if isinstance(item, _WrappedTool):
            result.append(item)
            continue
        if not callable(item):
            raise TypeError(
                f"Expected a callable or WrappedTool, got {type(item)!r}"
            )
        if builtins is None:
            builtins = _known_builtin_tools()
        if item not in builtins:
            name = getattr(item, "__name__", repr(item))
            raise TypeError(
                f"{name!r} is not a registered Thorn tool. "
                "Phase A retired user-supplied @tool callables; "
                "wrap the desired behavior as a script and invoke it "
                "via run_shell instead."
            )
        result.append(wrap_function(item))
    return result


# ---------------------------------------------------------------------------
# prompt — the core inline-prompt primitive
# ---------------------------------------------------------------------------

def _push_bare_prompt_scope(
    ctx: ExecutionContext,
    label: str,
    extra_file_access: list[Any] | None,
) -> ExecutionContext:
    """Build a file-access policy for a bare ``prompt()`` call and push a scope.

    Uses the base ``Agent`` default rules (write within workspace,
    ``.thorn/`` read-only) so that bare ``prompt()`` calls don't get
    more permissions than an explicit ``Agent()``.
    """
    from thorn.core._agent import Agent
    from thorn.core._file_access import FileAccessLevel, FileAccessPolicy, RelativeTo

    rules = list(Agent._collect_file_access())
    if extra_file_access:
        rules.extend(extra_file_access)
    policy = FileAccessPolicy(
        rules,
        default=FileAccessLevel.NONE,
        roots={RelativeTo.WORKSPACE: ctx.workspace_root},
    )

    if ctx.global_ignores is not None:
        policy = policy.with_ceiling(ctx.global_ignores)

    return ctx.push_scope(label, file_access_policy=policy)


class _TypedPrompt:
    """Callable returned by ``prompt[T]`` that executes a prompt expecting
    a result of type *T*.
    """

    __slots__ = ("_result_type",)

    def __init__(self, result_type: type) -> None:
        self._result_type = result_type

    async def __call__(
        self,
        text: str,
        *,
        tools: list[Any] | None = None,
        system: str | None = None,
        role: Any | None = None,
        file_access: list[Any] | None = None,
    ) -> Any:
        if role is not None:
            from thorn.core._agent import _run_session_prompt
            agent = role() if isinstance(role, type) else role
            return await _run_session_prompt(
                session=agent._default_session,
                text=text,
                result_type=self._result_type,
                extra_tools=tools,
                extra_system=system,
                extra_file_access=file_access,
            )

        ctx = get_context()
        child = _push_bare_prompt_scope(
            ctx,
            f"prompt[{_type_label(self._result_type)}]",
            file_access,
        )

        await child.event_sink.on_scope_enter(child.scope)
        t0 = time.monotonic()

        sys_prompts: list[str] | None = None
        if system:
            sys_prompts = [system]

        try:
            return await run_agent_loop(
                context=child,
                user_prompt=text,
                tools=_prepare_tools(tools),
                system_prompts=sys_prompts,
                result_type=self._result_type,
            )
        finally:
            await child.event_sink.on_scope_exit(
                child.scope, duration_s=time.monotonic() - t0,
            )


class _PromptAccessor:
    """The ``prompt`` object exposed in the public API.

    Supports three call patterns::

        await prompt("...")               # -> str  (text mode)
        await prompt[list[str]]("...")    # -> list[str]  (structured mode)
        await prompt("...", role=MyRole)  # -> str  (with agent role)
    """

    def __getitem__(self, result_type: type) -> _TypedPrompt:
        return _TypedPrompt(result_type)

    async def __call__(
        self,
        text: str,
        *,
        tools: list[Any] | None = None,
        system: str | None = None,
        role: Any | None = None,
        file_access: list[Any] | None = None,
    ) -> str:
        """Execute a prompt and return the assistant's text response."""
        if role is not None:
            from thorn.core._agent import _run_session_prompt
            agent = role() if isinstance(role, type) else role
            return await _run_session_prompt(
                session=agent._default_session,
                text=text,
                result_type=str,
                extra_tools=tools,
                extra_system=system,
                extra_file_access=file_access,
            )

        ctx = get_context()
        child = _push_bare_prompt_scope(ctx, "prompt", file_access)

        await child.event_sink.on_scope_enter(child.scope)
        t0 = time.monotonic()

        sys_prompts: list[str] | None = None
        if system:
            sys_prompts = [system]

        try:
            return await run_agent_loop(
                context=child,
                user_prompt=text,
                tools=_prepare_tools(tools),
                system_prompts=sys_prompts,
                result_type=str,
            )
        finally:
            await child.event_sink.on_scope_exit(
                child.scope, duration_s=time.monotonic() - t0,
            )


prompt = _PromptAccessor()


# ---------------------------------------------------------------------------
# @skill — decorator that turns a docstring into a prompt-based function
# ---------------------------------------------------------------------------

def _type_label(t: type) -> str:
    """Short human-readable label for a type (for scope descriptions)."""
    if hasattr(t, "__name__"):
        return t.__name__
    return str(t)


def skill(
    fn: Callable[..., Any] | None = None,
    *,
    tools: list[Any] | None = None,
    system: str | None = None,
    role: type | None = None,
) -> Any:
    """Decorator that turns a function stub into a prompt-based skill.

    The function's **docstring** becomes the prompt template (with
    ``{param_name}`` placeholders filled from the call arguments), and
    the **return annotation** determines the expected result type.

    When *role* is an ``Agent`` subclass, the skill automatically
    instantiates it (forwarding the call's bound arguments as kwargs),
    collects its system prompts and tools via MRO, and sets
    ``context.agent`` on the child execution context.

    Can be used bare or with configuration::

        @skill
        async def check(name: str) -> bool:
            \"\"\"Is the {name} service running?\"\"\"

        @skill(tools=[read_file])
        async def lint(path: str) -> list[str]:
            \"\"\"Lint the file at {path}.\"\"\"

        @skill(role=Architect)
        async def architect_module(module: str) -> None:
            \"\"\"Define architecture for module `{module}`.\"\"\"
    """

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        hints = get_type_hints(fn)
        return_type = hints.get("return", str)
        template = inspect.getdoc(fn) or ""

        @functools.wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            sig = inspect.signature(fn)
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            all_kwargs = dict(bound.arguments)

            prompt_text = template.format(**all_kwargs)

            ctx = get_context()

            if role is not None:
                role_instance = role(**all_kwargs)
                sys_prompts = role_instance._render_system_prompts()
                if system:
                    sys_prompts.append(system)

                role_tools = type(role_instance)._collect_tools()
                combined = role_tools + (tools or [])
                prepared = _prepare_tools(combined)

                child = ctx.push_scope(f"skill:{fn.__name__}", agent=role_instance)
                await child.event_sink.on_scope_enter(child.scope)
                t0 = time.monotonic()
                token = set_context(child)
                try:
                    return await run_agent_loop(
                        context=child,
                        user_prompt=prompt_text,
                        tools=prepared,
                        system_prompts=sys_prompts,
                        result_type=return_type,
                    )
                finally:
                    await child.event_sink.on_scope_exit(
                        child.scope, duration_s=time.monotonic() - t0,
                    )
                    reset_context(token)
            else:
                child = ctx.push_scope(f"skill:{fn.__name__}")
                await child.event_sink.on_scope_enter(child.scope)
                t0 = time.monotonic()
                sys_prompts = [system] if system else None
                try:
                    return await run_agent_loop(
                        context=child,
                        user_prompt=prompt_text,
                        tools=_prepare_tools(tools),
                        system_prompts=sys_prompts,
                        result_type=return_type,
                    )
                finally:
                    await child.event_sink.on_scope_exit(
                        child.scope, duration_s=time.monotonic() - t0,
                    )

        wrapper._thorn_skill = True  # type: ignore[attr-defined]
        wrapper._thorn_return_type = return_type  # type: ignore[attr-defined]
        return wrapper

    if fn is not None:
        return decorator(fn)
    return decorator


# ---------------------------------------------------------------------------
# @tool — marker decorator for auto-discovery
# ---------------------------------------------------------------------------

def tool(
    *,
    venue: ToolVenue,
    call_node_class: type[ToolCallNode] | None = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Mark a Python function as a discoverable thorn tool.

    Unlike ``@skill``, this does **not** replace the function body.
    The function keeps its original implementation and is simply
    tagged so that ``.thorn/`` directory discovery can find it.

    *venue* is **required** and has no default.  ``ToolVenue.SANDBOX``
    means the tool is dispatched by the toolhost daemon inside the
    agent's sandbox -- safe to give an agent's untrusted arguments
    because the daemon process is the trust boundary.
    ``ToolVenue.IN_PROCESS`` means the tool runs in the brain process
    and may reach into runtime state (the agency, the credential
    broker, the peer registry, the inbox); the author has thought
    about how the tool treats agent-supplied arguments and is taking
    responsibility for that hygiene.  See ``thorn.tools._catalog``
    for the canonical lists each venue is registered against.

    The bare ``@tool`` form (no parens) is no longer accepted: every
    tool author must pick a venue, and a silent default (whichever
    one we picked) would inevitably be wrong for some tool.  Mistakes
    made early in the lifecycle of a project are cheap; mistakes
    made silently after a tool ships get expensive.

    Usage::

        @tool(venue=ToolVenue.SANDBOX)
        async def grep_codebase(pattern: str, path: str = ".") -> str:
            \"\"\"Search for *pattern* in files under *path*.\"\"\"
            ...

        @tool(venue=ToolVenue.IN_PROCESS, call_node_class=FileReadCallNode)
        async def list_inbox_items() -> list[InboxItem]:
            \"\"\"Return the agent's pending inbox items.\"\"\"
            ...
    """

    def decorator(f: Callable[..., Any]) -> Callable[..., Any]:
        f._thorn_tool = True  # type: ignore[attr-defined]
        f._thorn_venue = venue  # type: ignore[attr-defined]
        if call_node_class is not None:
            f._thorn_call_node_class = call_node_class  # type: ignore[attr-defined]
        return f

    return decorator
