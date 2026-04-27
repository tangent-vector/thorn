"""Agent base class for role-based agent execution.

Subclasses declare ``system_prompts``, ``tools``, ``file_access``, and
``validation_rules`` as class variables.  The framework walks the MRO
(outermost-first) to collect them, renders prompt templates against
instance attributes, and provides a ``prompt`` accessor for running LLM
calls with the role's context.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    from pathlib import Path

    from thorn.core._context import ExecutionContext
    from thorn.core._context_injection import SeedContent
    from thorn.core._file_access import FileAccessRule
    from thorn.core._session import Session, _SessionPromptAccessor
    from thorn.runtime._session import AgentID


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
    """Tool functions discovered via :meth:`_collect_tools` MRO walk.

    The base ``Agent`` class provides journal tools
    (``write_journal``, ``read_journal``) by default even when
    ``tools`` is empty; see :meth:`_collect_tools` for details.
    """
    file_access: ClassVar[list[FileAccessRule]]
    validation_rules: ClassVar[list[str]] = []

    def __init__(
        self,
        *,
        name: str | None = None,
        metadata: dict[str, Any] | None = None,
        id: AgentID | None = None,
        workspace: Path | None = None,
        home: Path | None = None,
        **kwargs: Any,
    ) -> None:
        # ``id`` is the persistent identifier (used as the path stem
        # for the agent's identity file and per-agent
        # subdirectories) and ``name`` is the human-facing label.
        # The two are distinct in code so that an in-memory agent
        # constructed without an ``id`` cannot be accidentally
        # persisted; callers that want a one-off agent for tests
        # can therefore pass ``name="x"`` without obtaining a
        # spurious ``AgentID``.  In persisted JSON only ``name`` is
        # written -- the ``id`` is the file's path stem -- but at
        # construction time the two parameters remain independent.
        self.name: str | None = name
        self.metadata: dict[str, Any] = metadata if metadata is not None else {}
        self.id: AgentID | None = id
        self._workspace: Path | None = workspace
        self._workspace_resolved: bool = workspace is not None
        self._home: Path | None = home
        self._home_resolved: bool = home is not None
        self._lock: asyncio.Lock | None = None
        for k, v in kwargs.items():
            setattr(self, k, v)

    @property
    def lock(self) -> asyncio.Lock:
        """Per-agent concurrency lock, created lazily on first access.

        Serializes event handling (and eventually delegate_task) for
        this agent instance, protecting its workspace and session state
        from concurrent mutation.
        """
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    @property
    def workspace(self) -> Path | None:
        """The agent's workspace directory (analogous to ``.``).

        Where the agent performs file I/O.  ``AGENTS.md`` is loaded
        from here.  Resolved lazily: if not explicitly provided at
        construction, inherited from the ambient
        ``ExecutionContext.workspace_root`` on first access.  Once
        resolved, the value is fixed for the lifetime of the instance.
        """
        if not self._workspace_resolved:
            self._workspace_resolved = True
            try:
                from thorn.core._context import get_context
                self._workspace = get_context().workspace_root
            except RuntimeError:
                pass
        return self._workspace

    @property
    def home(self) -> Path | None:
        """The agent's personal state directory — its ``~``.

        Where the agent stores ``MEMORY.md``, ``journal/``, and any
        other instance-specific scratch state.  Both built-in file
        tools and shell subprocesses map ``~`` to this directory, so
        the agent can use ``~/MEMORY.md`` regardless of the current
        workspace.

        By default, derived as
        ``<agency_root>/.thorn/agents/<agent-id>/``, but callers may
        supply an explicit *home* at construction to place it anywhere
        on the filesystem.

        Resolved lazily from the ambient ``ExecutionContext`` on first
        access.  If the agent has no ``id``, a stable one is derived
        from the class name and workspace path so that personal state
        persists across runs.  Returns ``None`` when insufficient
        context is available (no execution context, no workspace).
        """
        if not self._home_resolved:
            self._home_resolved = True
            try:
                from thorn.core._context import get_context
                ctx = get_context()
                agency_root = ctx.agency_root_directory or ctx.workspace_root
                if agency_root is not None:
                    agent_id = self.id
                    if agent_id is None:
                        ws = self.workspace
                        if ws is not None:
                            agent_id = _derive_stable_agent_id(
                                type(self).__name__, ws,
                            )
                            self.id = agent_id
                    if agent_id is not None:
                        from thorn.runtime._paths import safe_dirname
                        self._home = (
                            agency_root / ".thorn" / "agents"
                            / safe_dirname(str(agent_id)) / "home"
                        )
            except RuntimeError:
                pass
        return self._home

    @property
    def _default_session(self) -> Session:
        """Lazily-created default session for the simple API.

        Enables ``agent.prompt("...")`` without explicitly constructing
        a ``Session``.  Multi-turn conversations reuse the same session.
        """
        if self.__dict__.get("_default_session_obj") is None:
            from thorn.core._session import Session
            self._default_session_obj = Session(agent=self)
        return self._default_session_obj

    def __str__(self) -> str:
        cls_name = type(self).__name__
        if self.name:
            return f"{cls_name}({self.name!r})"
        return cls_name

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

    def __init_subclass__(cls, *, abstract: bool = False, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        cls._abstract = abstract
        if not abstract:
            Agent._registry[cls.__name__] = cls

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

        Journal tools (``write_journal``, ``read_journal``) and inbox
        tools (``list_inbox_items``, ``read_inbox_item``,
        ``update_inbox_item``) are always appended unless a tool with
        the same name was already declared in the MRO, so every agent
        gets journaling and inbox capability by default.  A subclass
        that needs to shadow one of these can declare its own
        function with the same ``__name__``.
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

        from thorn.core._journal import JOURNAL_TOOLS
        for tool_item in _flatten_tools(JOURNAL_TOOLS):
            tool_name = getattr(tool_item, "__name__", str(tool_item))
            if tool_name not in seen_names:
                collected.append(tool_item)
                seen_names.add(tool_name)

        from thorn.runtime._inbox_tools import INBOX_TOOLS
        for tool_item in _flatten_tools(INBOX_TOOLS):
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

    @property
    def prompt(self) -> _SessionPromptAccessor:
        """Accessor for ``agent.prompt("...")`` and ``agent.prompt[T]("...")``.

        Delegates to a lazily-created default ``Session`` so that the
        simple ``agent.prompt("...")`` API continues to work without
        requiring an explicit ``Session`` or ``Runtime``.
        """
        from thorn.core._session import _SessionPromptAccessor
        return _SessionPromptAccessor(self._default_session)


def _derive_stable_agent_id(class_name: str, workspace: Path) -> AgentID:
    """Derive a deterministic agent ID from a class name and workspace path.

    Produces a filesystem-safe identifier that is stable across process
    restarts, so that an agent's ``home`` directory (and therefore its
    ``MEMORY.md`` and ``journal/``) persists across ``thorn run``
    invocations in the same project with the same agent class.
    """
    import hashlib

    from thorn.runtime._session import AgentID

    key = f"{class_name}:{workspace.resolve()}"
    digest = hashlib.sha256(key.encode()).hexdigest()[:12]
    return AgentID(f"{class_name.lower()}-{digest}")


def _default_file_access() -> list[FileAccessRule]:
    """The fallback file-access rules when no class in the MRO defines any.

    Rule ordering matters (last match wins):

    1. Grant write access everywhere in the workspace.
    2. Restrict ``.thorn/`` to read-only in the workspace (framework
       internals should not be casually mutated by agents).
    3. Grant write access everywhere in the agent's home directory.
       This overrides the ``.thorn/`` restriction for paths that fall
       under the agent's home, which lives *inside* ``.thorn/``.

    Do not reorder these rules without considering the interactions.
    """
    from thorn.core._file_access import FileAccessLevel, FileAccessRule, RelativeTo
    return [
        FileAccessRule("**", FileAccessLevel.WRITE),
        FileAccessRule(".thorn", FileAccessLevel.READ),
        FileAccessRule(".thorn/**", FileAccessLevel.READ),
        FileAccessRule("**", FileAccessLevel.WRITE, relative_to=RelativeTo.AGENT_HOME),
    ]


async def _run_session_prompt(
    *,
    session: Session,
    text: str,
    result_type: type,
    extra_tools: list[Any] | None = None,
    extra_system: str | None = None,
    extra_file_access: list[FileAccessRule] | None = None,
    recommended_context: list[SeedContent] | None = None,
) -> Any:
    """Shared implementation for session prompt calls.

    Conversation history accumulates on ``session._history`` across
    calls, enabling multi-turn patterns where the same session retains
    context (e.g. write code, then fix build errors).
    """
    import time

    from thorn.core._context import get_context, reset_context, set_context
    from thorn.core._file_access import FileAccessLevel, FileAccessPolicy, RelativeTo
    from thorn.core._func import _prepare_tools, _type_label
    from thorn.core._loop import run_agent_loop

    agent = session.agent

    sys_prompts = agent._render_system_prompts()
    if extra_system:
        sys_prompts.append(extra_system)

    role_tools = type(agent)._collect_tools()
    combined = role_tools + (extra_tools or [])
    prepared = _prepare_tools(combined)

    ctx = get_context()

    workspace = (
        session.workspace_root
        or agent.workspace
        or ctx.workspace_root
    )

    rules = type(agent)._collect_file_access()
    rules = list(rules)
    rules.extend(agent._instance_file_access())
    if extra_file_access:
        rules.extend(extra_file_access)
    policy = FileAccessPolicy(
        rules,
        default=FileAccessLevel.NONE,
        roots={
            RelativeTo.WORKSPACE: workspace,
            RelativeTo.AGENT_HOME: agent.home,
        },
    )

    if ctx.global_ignores is not None:
        policy = policy.with_ceiling(ctx.global_ignores)

    scope_label = f"agent:{type(agent).__name__}"
    if result_type is not str:
        scope_label += f"[{_type_label(result_type)}]"
    child = ctx.push_scope(
        scope_label,
        agent=agent,
        file_access_policy=policy,
        session_key=str(session.key) if session.key is not None else None,
    )

    if ctx.runtime is not None and child.sandbox_executor is None:
        runtime_executor = ctx.runtime.get_or_create_sandbox_executor(agent)
        if runtime_executor is not None:
            child.sandbox_executor = runtime_executor

    # Override workspace_root when the agent has its own workspace,
    # so file tools and policies operate within the agent's scope
    # rather than the parent context's.  The system-prompt blocks
    # (AGENTS.md, MEMORY.md, journal, environment) are no longer
    # injected here -- they are produced by the unified
    # context-gathering pipeline a few lines below.
    if workspace is not None:
        child.workspace_root = workspace

    agent_home = agent.home

    # ----------------------------------------------------------------
    # Unified context-gathering pipeline (per-prompt)
    #
    # Phase 1: identify the outer-to-inner directory chain.
    # Phase 2: load each directory's per-category contributions.
    # Phase 3: assemble the system-prompt blocks, MCP configs, and
    #          skill list ready for the agent loop.
    #
    # This replaces the legacy ``load_workspace_instructions`` /
    # ``load_agent_memory`` / hand-rolled environment block paths
    # that used to live inline above.  Journal loading still happens
    # here (caller-side) but block placement is owned by phase 3.
    # ----------------------------------------------------------------
    from thorn.runtime._context_layers import load_context_layers
    from thorn.runtime._context_paths import gather_context_directories
    from thorn.runtime._paths import session_key_path
    from thorn.runtime._prompt_assembly import assemble_prompt_context

    session_key_home_path = (
        agent_home / session_key_path(session.key)
        if (agent_home is not None and session.key is not None)
        else None
    )
    context_directories = gather_context_directories(
        agent_home_path=agent_home,
        session_key_home_path=session_key_home_path,
        logical_agent_workspace_path=session.logical_agent_workspace_path,
        session_workspace_path=workspace,
    )
    collected_layers = load_context_layers(context_directories)

    journal_text: str | None = None
    if agent_home is not None:
        journal_dir = agent_home / "journal"
        if journal_dir.is_dir():
            from thorn.core._journal import read_recent_journal
            session_key_str = (
                str(session.key) if session.key is not None else None
            )
            journal_text = read_recent_journal(
                journal_dir,
                exclude_session_key=session_key_str,
            )

    assembled = assemble_prompt_context(
        collected_layers,
        workspace_path=workspace,
        agent_home_path=agent_home,
        journal_text=journal_text,
    )
    sys_prompts.extend(assembled.system_prompt_blocks)
    # ``assembled.skills`` is *not* exposed as tools by design: skills
    # are a prompt-side concept, advertised via the skill-index block
    # already included in ``assembled.system_prompt_blocks`` above.
    # The agent reads a SKILL.md body via the ordinary file-read tool
    # if and when the indexed description suggests it is relevant.

    # ----------------------------------------------------------------
    # MCP servers for this prompt round
    #
    # Wrap the agent loop in an AsyncExitStack so any MCP sessions we
    # open are torn down even if the loop raises.  When there are no
    # MCP configs (the common case for tests / library use), the
    # stack is a no-op.  The ``mcp`` package is an optional extra; if
    # it is missing we log a warning and proceed without MCP tools
    # rather than crashing the prompt -- the same behaviour the
    # legacy ``_collect_all_tools`` had.
    # ----------------------------------------------------------------
    from contextlib import AsyncExitStack
    mcp_stack = AsyncExitStack()
    async with mcp_stack:
        if assembled.mcp_configs:
            try:
                from thorn.core._mcp import MCPToolSource
            except ImportError:
                import logging as _logging
                _logging.getLogger(__name__).warning(
                    "MCP configs found but the 'mcp' package is not "
                    "installed; skipping MCP tool registration",
                )
            else:
                mcp_source = await mcp_stack.enter_async_context(
                    MCPToolSource(assembled.mcp_configs),
                )
                prepared.extend(mcp_source.tools)

        await child.event_sink.on_scope_enter(child.scope)
        t0 = time.monotonic()
        token = set_context(child)
        try:
            # Context injection: pre-populate the session's history with
            # relevant content so it doesn't waste turns re-discovering
            # project structure.  This MUST run after set_context(child)
            # because the tool functions we call (read_file, etc.) use
            # get_context() internally to enforce the agent's file-access
            # policy.  The user prompt is appended first so the injected
            # assistant turn follows it naturally; run_agent_loop is then
            # called with user_prompt=None to avoid a duplicate.
            user_prompt_for_loop: str | None = text
            if not session._history.nodes:
                injected = await _inject_context(
                    session, text, child, recommended_context,
                )
                if injected:
                    user_prompt_for_loop = None

            return await run_agent_loop(
                context=child,
                user_prompt=user_prompt_for_loop,
                tools=prepared,
                system_prompts=sys_prompts,
                result_type=result_type,
                history=session._history,
                session=session,
            )
        finally:
            duration_s = time.monotonic() - t0
            await child.event_sink.on_scope_exit(
                child.scope, duration_s=duration_s,
            )
            reset_context(token)


async def _inject_context(
    session: Session,
    text: str,
    ctx: ExecutionContext,
    recommended_context: list[SeedContent] | None = None,
) -> bool:
    """Pre-populate *session*'s history with relevant context.

    Appends the user prompt followed by a synthetic assistant turn
    containing tool calls for workspace content.  Returns ``True`` if
    injection occurred (meaning the user prompt is already in the
    history and the caller should not append it again).

    Builds a priority-ordered list of seed items from two sources:

    1. *recommended_context* from the caller (preserves given order).
    2. Role-declared seeds from ``agent.context_seed_items()`` (sorted
       by descending salience, deduplicated against #1).

    Items are greedily packed into the injection token budget.
    """
    import logging

    from thorn.core._context_injection import (
        assemble_briefing,
        injection_budget,
    )

    budget = injection_budget(ctx.context_window)
    if budget <= 0:
        return False

    agent = session.agent

    seen: set[SeedContent] = set()
    ordered: list[SeedContent] = []

    for item in (recommended_context or []):
        if item not in seen:
            ordered.append(item)
            seen.add(item)

    role_seeds = agent.context_seed_items()
    for item, _salience in sorted(
        role_seeds.items(), key=lambda kv: kv[1], reverse=True,
    ):
        if item not in seen:
            ordered.append(item)
            seen.add(item)

    if not ordered:
        return False

    try:
        briefing = await assemble_briefing(ordered, budget, ctx.workspace_root)
        if briefing is None:
            return False
        session._history.append_user_prompt(text)
        session._history.nodes.append(briefing)
        return True
    except Exception:
        logging.getLogger(__name__).debug(
            "context injection failed for %s, continuing without",
            agent,
            exc_info=True,
        )
        return False
