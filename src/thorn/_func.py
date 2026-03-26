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
from typing import Any, Callable, Generic, TypeVar, get_type_hints, overload

from thorn._context import ExecutionContext, get_context, reset_context, set_context
from thorn._loop import _WrappedTool, run_agent_loop
from thorn._schema import func_to_tool_schema, serialize_for_tool_result

T = TypeVar("T")


# ---------------------------------------------------------------------------
# wrap_function — turn any Python function into a tool for an agent
# ---------------------------------------------------------------------------

def wrap_function(fn: Callable[..., Any]) -> _WrappedTool:
    """Wrap a Python function so it can be used as a tool by an agent.

    The function's name, docstring, and typed parameters are introspected
    to produce an OpenAI-style tool schema.  At invocation time the JSON
    arguments from the model are deserialized and passed to the function,
    and the return value is serialized back to a string.
    """
    schema = func_to_tool_schema(fn)
    is_async = asyncio.iscoroutinefunction(fn)

    async def execute(**kwargs: Any) -> str:
        if is_async:
            result = await fn(**kwargs)
        else:
            result = fn(**kwargs)
        return serialize_for_tool_result(result)

    return _WrappedTool(schema=schema, execute=execute)


def _prepare_tools(raw_tools: list[Any] | None) -> list[_WrappedTool]:
    """Normalise a user-supplied tool list into ``_WrappedTool`` instances.

    Accepts a mix of:
    - ``_WrappedTool`` instances (passed through)
    - plain callables (auto-wrapped via ``wrap_function``)
    """
    if not raw_tools:
        return []
    result: list[_WrappedTool] = []
    for item in raw_tools:
        if isinstance(item, _WrappedTool):
            result.append(item)
        elif callable(item):
            result.append(wrap_function(item))
        else:
            raise TypeError(
                f"Expected a callable or WrappedTool, got {type(item)!r}"
            )
    return result


# ---------------------------------------------------------------------------
# prompt — the core inline-prompt primitive
# ---------------------------------------------------------------------------

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
    ) -> Any:
        ctx = get_context()
        child = ctx.push_scope(f"prompt[{_type_label(self._result_type)}]")

        sys_prompts: list[str] | None = None
        if system:
            sys_prompts = [system]

        return await run_agent_loop(
            context=child,
            user_prompt=text,
            tools=_prepare_tools(tools),
            system_prompts=sys_prompts,
            result_type=self._result_type,
        )


class _PromptAccessor:
    """The ``prompt`` object exposed in the public API.

    Supports two call patterns::

        await prompt("...")               # -> str  (text mode)
        await prompt[list[str]]("...")    # -> list[str]  (structured mode)
    """

    def __getitem__(self, result_type: type) -> _TypedPrompt:
        return _TypedPrompt(result_type)

    async def __call__(
        self,
        text: str,
        *,
        tools: list[Any] | None = None,
        system: str | None = None,
    ) -> str:
        """Execute a prompt and return the assistant's text response."""
        ctx = get_context()
        child = ctx.push_scope("prompt")

        sys_prompts: list[str] | None = None
        if system:
            sys_prompts = [system]

        return await run_agent_loop(
            context=child,
            user_prompt=text,
            tools=_prepare_tools(tools),
            system_prompts=sys_prompts,
            result_type=str,
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
                    reset_context(token)
            else:
                child = ctx.push_scope(f"skill:{fn.__name__}")
                sys_prompts = [system] if system else None
                return await run_agent_loop(
                    context=child,
                    user_prompt=prompt_text,
                    tools=_prepare_tools(tools),
                    system_prompts=sys_prompts,
                    result_type=return_type,
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

def tool(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Mark a Python function as a discoverable thorn tool.

    Unlike ``@skill``, this does **not** replace the function body.
    The function keeps its original implementation and is simply tagged
    so that ``.thorn/`` directory discovery can find it::

        @tool
        async def grep_codebase(pattern: str, path: str = ".") -> str:
            \"\"\"Search for *pattern* in files under *path*.\"\"\"
            ...
    """
    fn._thorn_tool = True  # type: ignore[attr-defined]
    return fn
