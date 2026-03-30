"""Agent base class for role-based agent execution.

Subclasses declare ``system_prompts`` and ``tools`` as class variables.
The framework walks the MRO (outermost-first) to collect them, renders
prompt templates against instance attributes, and provides a ``prompt``
accessor for running LLM calls with the role's context.
"""

from __future__ import annotations

from typing import Any, ClassVar


class _SafeDict(dict):
    """Dict subclass that leaves ``{key}`` intact for missing keys."""

    def __missing__(self, key: str) -> str:
        return f"{{{key}}}"


class Agent:
    """Base class for agent roles.

    Subclasses declare ``system_prompts`` and ``tools`` as class variables.
    The framework walks the MRO (outermost-first) to collect them.
    System prompts are string templates rendered against instance attributes.

    Subclasses are automatically registered and can be queried via
    :meth:`get_subclasses`.  Mark intermediate base classes with
    ``abstract=True`` to exclude them from the registry::

        class Developer(Agent, abstract=True):
            ...

        class Architect(Developer):  # registered
            ...
    """

    _registry: ClassVar[dict[str, type[Agent]]] = {}
    _abstract: ClassVar[bool] = True

    system_prompts: ClassVar[list[str]] = []
    tools: ClassVar[list[Any]] = []

    def __init_subclass__(cls, *, abstract: bool = False, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        cls._abstract = abstract
        if not abstract:
            Agent._registry[cls.__name__] = cls

    def __init__(self, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __str__(self) -> str:
        return type(self).__name__

    @classmethod
    def get_subclasses(cls, base: type[Agent] | None = None) -> dict[str, type[Agent]]:
        """Return registered (non-abstract) subclasses, optionally filtered.

        When *base* is provided, only subclasses of *base* are returned.
        When *base* is ``None``, subclasses of the calling class are
        returned -- so ``Agent.get_subclasses()`` returns everything,
        while ``MyBase.get_subclasses()`` returns only its descendants.
        """
        filter_base = base if base is not None else cls
        return {
            name: klass
            for name, klass in Agent._registry.items()
            if issubclass(klass, filter_base)
        }

    @classmethod
    def _collect_system_prompts(cls) -> list[str]:
        """Walk MRO outermost-first, collecting prompts from each class's own ``__dict__``."""
        collected: list[str] = []
        for klass in reversed(cls.__mro__):
            if "system_prompts" in klass.__dict__:
                collected.extend(klass.__dict__["system_prompts"])
        return collected

    @classmethod
    def _collect_tools(cls) -> list[Any]:
        """Walk MRO outermost-first, collecting tools and deduplicating by name."""
        collected: list[Any] = []
        seen_names: set[str] = set()
        for klass in reversed(cls.__mro__):
            if "tools" in klass.__dict__:
                for tool_item in klass.__dict__["tools"]:
                    tool_name = getattr(tool_item, "__name__", str(tool_item))
                    if tool_name not in seen_names:
                        collected.append(tool_item)
                        seen_names.add(tool_name)
        return collected

    def _render_system_prompts(self) -> list[str]:
        """Render collected prompts as templates against instance attributes."""
        attrs = _SafeDict(
            {k: v for k, v in vars(self).items() if not k.startswith("_")}
        )
        return [p.format_map(attrs) for p in type(self)._collect_system_prompts()]

    @property
    def prompt(self) -> _AgentPromptAccessor:
        """Accessor for ``agent.prompt("...")`` and ``agent.prompt[T]("...")``."""
        return _AgentPromptAccessor(self)


class _AgentPromptAccessor:
    """Provides ``agent.prompt("...")`` (text) and ``agent.prompt[T]("...")`` (structured)."""

    __slots__ = ("_agent",)

    def __init__(self, agent: Agent) -> None:
        self._agent = agent

    def __getitem__(self, result_type: type) -> _TypedAgentPrompt:
        return _TypedAgentPrompt(self._agent, result_type)

    async def __call__(
        self,
        text: str,
        *,
        tools: list[Any] | None = None,
        system: str | None = None,
        messages: list | None = None,
    ) -> str:
        return await _run_agent_prompt(
            agent=self._agent,
            text=text,
            result_type=str,
            extra_tools=tools,
            extra_system=system,
            messages=messages,
        )


class _TypedAgentPrompt:
    """Callable returned by ``agent.prompt[T]`` for structured results."""

    __slots__ = ("_agent", "_result_type")

    def __init__(self, agent: Agent, result_type: type) -> None:
        self._agent = agent
        self._result_type = result_type

    async def __call__(
        self,
        text: str,
        *,
        tools: list[Any] | None = None,
        system: str | None = None,
        messages: list | None = None,
    ) -> Any:
        return await _run_agent_prompt(
            agent=self._agent,
            text=text,
            result_type=self._result_type,
            extra_tools=tools,
            extra_system=system,
            messages=messages,
        )


async def _run_agent_prompt(
    *,
    agent: Agent,
    text: str,
    result_type: type,
    extra_tools: list[Any] | None = None,
    extra_system: str | None = None,
    messages: list | None = None,
) -> Any:
    """Shared implementation for agent prompt calls.

    If *messages* is provided, the conversation history accumulates
    across calls — enabling multi-turn patterns where the same agent
    retains context (e.g. write code, then fix build errors).
    """
    import time

    from thorn._context import get_context, reset_context, set_context
    from thorn._func import _prepare_tools, _type_label
    from thorn._loop import run_agent_loop

    sys_prompts = agent._render_system_prompts()
    if extra_system:
        sys_prompts.append(extra_system)

    role_tools = type(agent)._collect_tools()
    combined = role_tools + (extra_tools or [])
    prepared = _prepare_tools(combined)

    ctx = get_context()
    scope_label = f"agent:{type(agent).__name__}"
    if result_type is not str:
        scope_label += f"[{_type_label(result_type)}]"
    child = ctx.push_scope(scope_label, agent=agent)

    await child.event_sink.on_scope_enter(child.scope)
    t0 = time.monotonic()
    token = set_context(child)
    try:
        return await run_agent_loop(
            context=child,
            user_prompt=text,
            tools=prepared,
            system_prompts=sys_prompts,
            result_type=result_type,
            messages=messages,
        )
    finally:
        duration_s = time.monotonic() - t0
        await child.event_sink.on_scope_exit(
            child.scope, duration_s=duration_s,
        )
        reset_context(token)
