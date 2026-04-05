"""Agent base class for role-based agent execution.

Subclasses declare ``system_prompts``, ``tools``, ``file_access``, and
``validation_rules`` as class variables.  The framework walks the MRO
(outermost-first) to collect them, renders prompt templates against
instance attributes, and provides a ``prompt`` accessor for running LLM
calls with the role's context.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    from thorn.core._context_injection import SeedContent
    from thorn.core._file_access import FileAccessRule
    from thorn.core._history import HistoryTree


class _SafeDict(dict):
    """Dict subclass that leaves ``{key}`` intact for missing keys."""

    def __missing__(self, key: str) -> str:
        return f"{{{key}}}"


class Agent:
    """Base class for agent roles.

    Subclasses declare ``system_prompts``, ``tools``, and
    ``validation_rules`` as class variables.  The framework walks the
    MRO (outermost-first) to collect them.  System prompts are string
    templates rendered against instance attributes.

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

    system_prompts: ClassVar[list[Any]] = []
    tools: ClassVar[list[Any]] = []
    file_access: ClassVar[list[FileAccessRule]]
    validation_rules: ClassVar[list[str]] = []

    def __init_subclass__(cls, *, abstract: bool = False, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        cls._abstract = abstract
        if not abstract:
            Agent._registry[cls.__name__] = cls

    def __init__(self, **kwargs: Any) -> None:
        from thorn.core._history import HistoryTree
        self._history: HistoryTree = HistoryTree()
        self._parent: Agent | None = None
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
    def _collect_system_prompts(cls) -> list[Any]:
        """Walk MRO outermost-first, collecting prompts from each class's own ``__dict__``.

        Entries may be plain strings (rendered as templates) or callables
        ``(Agent) -> str | None`` (invoked at render time; falsy results
        are skipped).
        """
        collected: list[Any] = []
        for klass in reversed(cls.__mro__):
            if "system_prompts" in klass.__dict__:
                collected.extend(klass.__dict__["system_prompts"])
        return collected

    @classmethod
    def _collect_tools(cls) -> list[Any]:
        """Walk MRO outermost-first, collecting tools and deduplicating by name.

        Nested iterables (e.g. toolset constants like ``FILE_READING``)
        are flattened before deduplication.
        """
        from thorn.core._func import _flatten_tools

        collected: list[Any] = []
        seen_names: set[str] = set()
        for klass in reversed(cls.__mro__):
            if "tools" in klass.__dict__:
                for tool_item in _flatten_tools(klass.__dict__["tools"]):
                    tool_name = getattr(tool_item, "__name__", str(tool_item))
                    if tool_name not in seen_names:
                        collected.append(tool_item)
                        seen_names.add(tool_name)
        return collected

    @classmethod
    def _collect_validation_rules(cls) -> list[str]:
        """Walk MRO outermost-first, collecting validation rule names and deduplicating."""
        collected: list[str] = []
        seen: set[str] = set()
        for klass in reversed(cls.__mro__):
            if "validation_rules" in klass.__dict__:
                for name in klass.__dict__["validation_rules"]:
                    if name not in seen:
                        collected.append(name)
                        seen.add(name)
        return collected

    @classmethod
    def _collect_file_access(cls) -> list[FileAccessRule]:
        """Walk MRO outermost-first, collecting file-access rules from each class's own ``__dict__``.

        If no class in the MRO defines ``file_access``, the default
        policy (write within workspace, ``.thorn/`` read-only) is
        returned.
        """
        from thorn.core._file_access import FileAccessLevel, FileAccessRule as _FAR  # noqa: F811

        collected: list[_FAR] = []
        found_any = False
        for klass in reversed(cls.__mro__):
            if "file_access" in klass.__dict__:
                collected.extend(klass.__dict__["file_access"])
                found_any = True
        if not found_any:
            collected = _default_file_access()
        return collected

    def _instance_file_access(self) -> list[FileAccessRule]:
        """Return per-instance file-access rules (e.g. based on a module name).

        Subclasses override this to grant dynamic access that depends
        on instance attributes.  The returned rules are appended after
        the MRO-collected class rules, so they have highest priority.
        """
        return []

    def _render_system_prompts(self) -> list[str]:
        """Render collected prompts as templates against instance attributes.

        String entries are rendered via ``str.format_map`` with instance
        attributes (missing keys are preserved as ``{key}``).  Callable
        entries receive the agent instance and should return a string;
        a falsy return value causes the entry to be skipped.
        """
        attrs = _SafeDict(
            {k: v for k, v in vars(self).items() if not k.startswith("_")}
        )
        rendered: list[str] = []
        for p in type(self)._collect_system_prompts():
            if callable(p):
                result = p(self)
                if result:
                    rendered.append(result)
            else:
                rendered.append(p.format_map(attrs))
        return rendered

    def context_seed_items(self) -> dict[SeedContent, float]:
        """Declare structurally relevant content for context injection.

        Subclasses override this to return seed items (e.g. module files)
        that should be pre-loaded when the agent starts a fresh session.
        The default implementation returns nothing.
        """
        return {}

    def extract_salient_items_from_history(
        self,
        history: HistoryTree,
    ) -> dict[SeedContent, float]:
        """Extract salient items from a parent agent's history.

        Subclasses override this to inspect the parent's ``HistoryTree``
        for relevant tool calls and produce ``SeedContent`` items ranked
        by salience.  The default implementation returns nothing.
        """
        return {}

    @property
    def prompt(self) -> _AgentPromptAccessor:
        """Accessor for ``agent.prompt("...")`` and ``agent.prompt[T]("...")``."""
        return _AgentPromptAccessor(self)


def _default_file_access() -> list[FileAccessRule]:
    """The fallback file-access rules when no class in the MRO defines any."""
    from thorn.core._file_access import FileAccessLevel, FileAccessRule
    return [
        FileAccessRule("**", FileAccessLevel.WRITE),
        FileAccessRule(".thorn", FileAccessLevel.READ),
        FileAccessRule(".thorn/**", FileAccessLevel.READ),
    ]


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
        file_access: list[FileAccessRule] | None = None,
    ) -> str:
        return await _run_agent_prompt(
            agent=self._agent,
            text=text,
            result_type=str,
            extra_tools=tools,
            extra_system=system,
            extra_file_access=file_access,
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
        file_access: list[FileAccessRule] | None = None,
    ) -> Any:
        return await _run_agent_prompt(
            agent=self._agent,
            text=text,
            result_type=self._result_type,
            extra_tools=tools,
            extra_system=system,
            extra_file_access=file_access,
        )


async def _run_agent_prompt(
    *,
    agent: Agent,
    text: str,
    result_type: type,
    extra_tools: list[Any] | None = None,
    extra_system: str | None = None,
    extra_file_access: list[FileAccessRule] | None = None,
) -> Any:
    """Shared implementation for agent prompt calls.

    Conversation history accumulates on ``agent._history`` across
    calls, enabling multi-turn patterns where the same agent retains
    context (e.g. write code, then fix build errors).
    """
    import time

    from thorn.core._context import get_context, reset_context, set_context
    from thorn.core._file_access import FileAccessLevel, FileAccessPolicy
    from thorn.core._func import _prepare_tools, _type_label
    from thorn.core._loop import run_agent_loop

    sys_prompts = agent._render_system_prompts()
    if extra_system:
        sys_prompts.append(extra_system)

    role_tools = type(agent)._collect_tools()
    combined = role_tools + (extra_tools or [])
    prepared = _prepare_tools(combined)

    ctx = get_context()

    rules = type(agent)._collect_file_access()
    rules = list(rules)
    rules.extend(agent._instance_file_access())
    if extra_file_access:
        rules.extend(extra_file_access)
    policy = FileAccessPolicy(
        rules, default=FileAccessLevel.NONE, workspace=ctx.workspace_root,
    )

    if ctx.global_ignores is not None:
        policy = policy.with_ceiling(ctx.global_ignores)

    scope_label = f"agent:{type(agent).__name__}"
    if result_type is not str:
        scope_label += f"[{_type_label(result_type)}]"
    child = ctx.push_scope(scope_label, agent=agent, file_access_policy=policy)

    await child.event_sink.on_scope_enter(child.scope)
    t0 = time.monotonic()
    token = set_context(child)
    try:
        # Context injection: pre-populate the child's history with
        # salient content so it doesn't waste turns re-discovering
        # project structure.  This MUST run after set_context(child)
        # because the tool functions we call (read_file, etc.) use
        # get_context() internally to enforce the child's file-access
        # policy.  The user prompt is appended first so the injected
        # assistant turn follows it naturally; run_agent_loop is then
        # called with user_prompt=None to avoid a duplicate.
        user_prompt_for_loop: str | None = text
        if not agent._history.nodes:
            injected = await _inject_context(agent, text, child)
            if injected:
                user_prompt_for_loop = None

        return await run_agent_loop(
            context=child,
            user_prompt=user_prompt_for_loop,
            tools=prepared,
            system_prompts=sys_prompts,
            result_type=result_type,
            history=agent._history,
        )
    finally:
        duration_s = time.monotonic() - t0
        await child.event_sink.on_scope_exit(
            child.scope, duration_s=duration_s,
        )
        reset_context(token)


async def _inject_context(
    agent: Agent,
    text: str,
    ctx: ExecutionContext,
) -> bool:
    """Pre-populate *agent*'s history with salient context.

    Appends the user prompt followed by a synthetic assistant turn
    containing tool calls for salient workspace content.  Returns
    ``True`` if injection occurred (meaning the user prompt is already
    in the history and the caller should not append it again).

    Collects seed items from up to three sources (agent seeds, prompt
    text analysis, parent history), merges and scores them, then
    assembles a synthetic briefing within the injection token budget.

    Source 2 (prompt analysis) and Source 3 (parent history) are only
    active when the agent has a parent, i.e. was spawned via delegation
    rather than being a root-level agent.
    """
    import logging

    from thorn.core._context_injection import (
        assemble_briefing,
        extract_seeds_from_prompt,
        injection_budget,
        merge_sources,
    )

    sources: list[tuple[dict[SeedContent, float], float]] = []

    seed_items = agent.context_seed_items()
    if seed_items:
        sources.append((seed_items, 1.0))

    if agent._parent is not None:
        prompt_items = extract_seeds_from_prompt(text, ctx.workspace_root)
        if prompt_items:
            sources.append((prompt_items, 0.5))

        parent_items = agent.extract_salient_items_from_history(
            agent._parent._history,
        )
        if parent_items:
            sources.append((parent_items, 0.1))

    if not sources:
        return False

    merged = merge_sources(sources)
    budget = injection_budget(ctx.context_window)
    if budget <= 0 or not merged:
        return False

    try:
        briefing = await assemble_briefing(merged, budget, ctx.workspace_root)
        if briefing is None:
            return False
        agent._history.append_user_prompt(text)
        agent._history.nodes.append(briefing)
        return True
    except Exception:
        logging.getLogger(__name__).debug(
            "context injection failed for %s, continuing without",
            agent,
            exc_info=True,
        )
        return False
