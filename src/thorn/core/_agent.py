"""Agent base class for role-based agent execution.

Subclasses declare ``system_prompts``, ``tools``, ``file_access``, and
``validation_rules`` as class variables.  The framework walks the MRO
(outermost-first) to collect them, renders prompt templates against
instance attributes, and provides a ``prompt`` accessor for running LLM
calls with the role's context.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    from pathlib import Path

    from thorn.core._context import ExecutionContext
    from thorn.core._context_injection import SeedContent
    from thorn.core._file_access import FileAccessRule
    from thorn.core._loop import (
        ProviderRequestSnapshot,
        ToolRoundTerminalPolicy,
        _WrappedTool,
    )
    from thorn.core._session import Session, _SessionPromptAccessor
    from thorn.runtime._session import AgentID
    from thorn.runtime._working_set import RenderedWorkingSet
    from thorn.runtime._working_set_telemetry import (
        WorkingSetTelemetry,
        WorkingSetTodoTelemetry,
    )


class _SafeDict(dict):
    """Dict subclass that leaves ``{key}`` intact for missing keys."""

    def __missing__(self, key: str) -> str:
        return f"{{{key}}}"


@dataclass(frozen=True)
class _SessionWorkingSetPromptBlock:
    rendered: RenderedWorkingSet
    todo: WorkingSetTodoTelemetry
    in_progress_not_focused_count: int


@dataclass(frozen=True)
class _SessionProviderRequestProjector:
    """Project live session state immediately before a provider request."""

    agent: Agent
    session: Session
    context: ExecutionContext
    workspace_path: Path | None
    agent_home_path: Path | None
    base_tools: tuple[_WrappedTool, ...]
    extra_system: str | None

    async def project(self) -> ProviderRequestSnapshot:
        """Return a consistent prompt/tool snapshot of current disk state."""
        return await _project_session_provider_request(self)


def _agent_uses_container_paths(agent: Agent, ctx: ExecutionContext) -> bool:
    """Return whether model-facing paths should use container mount points."""
    runtime = ctx.runtime
    if runtime is None or not runtime.sandbox_executor_enabled:
        return False

    from thorn.sandbox._resolve import resolve_sandbox_config

    resolved = resolve_sandbox_config(
        runtime.sandbox_config,
        getattr(agent, "sandbox_override", None),
    )
    return resolved.backend == "container"


def _agent_environment_display_paths(
    agent: Agent,
    session: Session,
    ctx: ExecutionContext,
    *,
    workspace_path: Path | None,
    agent_home_path: Path | None,
) -> tuple[Path | PurePosixPath | None, Path | PurePosixPath | None]:
    """Translate host paths into the namespace used by sandbox tools."""
    if not _agent_uses_container_paths(agent, ctx):
        return workspace_path, agent_home_path

    from thorn.sandbox._container import CONTAINER_HOME_DIR, CONTAINER_WORKSPACE_DIR

    display_workspace: PurePosixPath | None = None
    if workspace_path is not None:
        display_workspace = PurePosixPath(CONTAINER_WORKSPACE_DIR)
        agent_workspace_root = session.logical_agent_workspace_path
        runtime = ctx.runtime
        if runtime is not None and agent.id is not None:
            agent_workspace_root = runtime.paths.agent_workspace_mount(agent.id)

        if agent_workspace_root is not None:
            try:
                rel = workspace_path.resolve().relative_to(
                    agent_workspace_root.resolve(),
                )
            except ValueError:
                rel = None
            if rel is not None and str(rel) != ".":
                display_workspace = display_workspace.joinpath(*rel.parts)

    display_home = (
        PurePosixPath(CONTAINER_HOME_DIR)
        if agent_home_path is not None else None
    )
    return display_workspace, display_home


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

        Journal tools (``write_journal``, ``read_journal``), focused
        inbox workflow tools (``list_inbox_items``, ``read_inbox_item``,
        ``update_focus``, ``complete_focused_work``,
        ``park_focused_work``), and session TODO tools are always
        appended unless a tool with the same name was already declared
        in the MRO, so every agent gets journaling, inbox, and TODO
        capability by default.  A subclass that needs to shadow one of
        these can declare its own function with the same ``__name__``.
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

        from thorn.runtime._todo_tools import TODO_TOOLS
        for tool_item in _flatten_tools(TODO_TOOLS):
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
        from thorn.core._file_access import FileAccessRule as _FAR

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
    tool_round_terminal_policy: ToolRoundTerminalPolicy | None = None,
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
        session=session,
        session_key=str(session.key) if session.key is not None else None,
    )

    if ctx.runtime is not None:
        child.provider = ctx.runtime.provider_for_session(session)

    if ctx.runtime is not None and child.sandbox_executor is None:
        runtime_executor = ctx.runtime.get_or_create_sandbox_executor(agent)
        if runtime_executor is not None:
            child.sandbox_executor = runtime_executor

    # Override workspace_root when the agent has its own workspace,
    # so file tools and policies operate within the agent's scope
    # rather than the parent context's. The per-request projector reloads
    # prompt-side context from this scope before every provider request.
    if workspace is not None:
        child.workspace_root = workspace

    provider_request_projector = _SessionProviderRequestProjector(
        agent=agent,
        session=session,
        context=child,
        workspace_path=workspace,
        agent_home_path=agent.home,
        base_tools=tuple(prepared),
        extra_system=extra_system,
    )

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
            result_type=result_type,
            history=session._history,
            session=session,
            provider_request_projector=provider_request_projector,
            tool_round_terminal_policy=tool_round_terminal_policy,
        )
    finally:
        duration_s = time.monotonic() - t0
        await child.event_sink.on_scope_exit(
            child.scope, duration_s=duration_s,
        )
        reset_context(token)


async def _project_session_provider_request(
    projector: _SessionProviderRequestProjector,
) -> ProviderRequestSnapshot:
    """Read live prompt/tool state once for the next provider request."""
    from thorn.core._loop import ProviderRequestSnapshot
    from thorn.core._prompt_trace import (
        PromptTraceContextSource,
        PromptTraceManifest,
    )
    from thorn.runtime._context_layers import load_context_layers
    from thorn.runtime._context_paths import gather_context_directories
    from thorn.runtime._paths import session_key_path
    from thorn.runtime._prompt_assembly import assemble_prompt_context

    agent = projector.agent
    session = projector.session
    context = projector.context

    role_system_prompts = agent._render_system_prompts()
    system_prompts = list(role_system_prompts)
    prompt_trace_sources = [
        PromptTraceContextSource.from_text(
            surface="agent_role_prompt",
            label=f"{type(agent).__name__} role prompt {index}",
            text=prompt,
        )
        for index, prompt in enumerate(role_system_prompts)
    ]
    if projector.extra_system:
        system_prompts.append(projector.extra_system)
        prompt_trace_sources.append(PromptTraceContextSource.from_text(
            surface="extra_system_prompt",
            label="extra system prompt",
            text=projector.extra_system,
        ))

    session_key_home_path = (
        projector.agent_home_path / session_key_path(session.key)
        if projector.agent_home_path is not None and session.key is not None
        else None
    )
    operator_dir = None
    if context.runtime is not None and agent.id is not None:
        operator_dir = context.runtime.paths.agent_framework_dir(agent.id)

    context_directories = gather_context_directories(
        operator_dir=operator_dir,
        agent_home_path=projector.agent_home_path,
        session_key_home_path=session_key_home_path,
        logical_agent_workspace_path=session.logical_agent_workspace_path,
        session_workspace_path=projector.workspace_path,
    )
    collected_layers = load_context_layers(context_directories)

    journal_text: str | None = None
    if projector.agent_home_path is not None:
        journal_dir = projector.agent_home_path / "journal"
        if journal_dir.is_dir():
            from thorn.core._journal import read_recent_journal

            session_key_text = (
                str(session.key) if session.key is not None else None
            )
            journal_text = read_recent_journal(
                journal_dir,
                exclude_session_key=session_key_text,
            )

    display_workspace, display_agent_home = _agent_environment_display_paths(
        agent,
        session,
        context,
        workspace_path=projector.workspace_path,
        agent_home_path=projector.agent_home_path,
    )
    assembled = assemble_prompt_context(
        collected_layers,
        workspace_path=display_workspace,
        agent_home_path=display_agent_home,
        journal_text=journal_text,
    )
    system_prompts.extend(assembled.system_prompt_blocks)
    prompt_trace_sources.extend(
        assembled.prompt_trace_manifest.system_prompt_sources,
    )

    working_set_telemetry: WorkingSetTelemetry | None = None
    working_set_block = _render_session_working_set_block(session, context)
    if working_set_block is not None:
        working_set_render = working_set_block.rendered
        system_prompts.append(working_set_render.text)
        from thorn.runtime._working_set_telemetry import (
            WorkingSetTelemetryKind,
            build_working_set_telemetry,
        )

        working_set_telemetry = build_working_set_telemetry(
            kind=WorkingSetTelemetryKind.PROMPT_RENDERED,
            working_set=session.working_set,
            rendered=working_set_render,
            todo=working_set_block.todo,
            in_progress_not_focused_count=(
                working_set_block.in_progress_not_focused_count
            ),
        )
        prompt_trace_sources.append(PromptTraceContextSource.from_text(
            surface="working_set",
            label="current session working set",
            text=working_set_render.text,
            metadata={
                "phase": session.working_set.phase.value,
                "focused_inbox_item_id": (
                    str(session.working_set.focused_inbox_item_id)
                    if session.working_set.focused_inbox_item_id is not None
                    else None
                ),
                "diagnostic_count": len(working_set_render.diagnostics),
                "active_context_entry_count": len(
                    session.working_set.active_context,
                ),
                "open_todo_count": working_set_block.todo.open_count,
            },
        ))

    tools = list(projector.base_tools)
    if assembled.mcp_configs:
        if context.sandbox_executor is None:
            import logging

            logging.getLogger(__name__).warning(
                "MCP configs found (%d) but the agent has no sandbox "
                "executor; skipping MCP tool registration. Configure a "
                "Runtime with a sandbox executor to enable MCP.",
                len(assembled.mcp_configs),
            )
        else:
            from thorn.runtime._mcp_tools import discover_mcp_tools

            builtin_tool_names = {
                tool.schema.get("function", {}).get("name", "")
                for tool in tools
            }
            builtin_tool_names.discard("")
            tools.extend(await discover_mcp_tools(
                sandbox_executor=context.sandbox_executor,
                mcp_configs=assembled.mcp_configs,
                builtin_tool_names=builtin_tool_names,
            ))

    return ProviderRequestSnapshot(
        system_prompts=tuple(system_prompts),
        tools=tuple(tools),
        prompt_trace_manifest=PromptTraceManifest(
            system_prompt_sources=prompt_trace_sources,
            working_set_telemetry=(
                working_set_telemetry.to_json()
                if working_set_telemetry is not None else None
            ),
        ),
        working_set_telemetry=working_set_telemetry,
    )


def _render_session_working_set_block(
    session: Session,
    ctx: ExecutionContext,
) -> _SessionWorkingSetPromptBlock | None:
    """Render the current working set for sessions with durable identity."""
    if session.key is None and ctx.runtime is None:
        return None

    from thorn.runtime._todo import SessionTodoList
    from thorn.runtime._working_set import (
        open_todo_items,
        render_working_set_block,
    )
    from thorn.runtime._working_set_telemetry import (
        todo_telemetry_from_statuses,
    )

    todo_diagnostic: str | None = None
    todos = ()
    if (
        ctx.runtime is not None
        and session.agent.id is not None
        and session.key is not None
    ):
        todo_file = ctx.runtime.paths.session_todo_file(
            session.agent.id,
            session.key,
        )
        try:
            todos = tuple(SessionTodoList(todo_file).list_items())
        except ValueError as exc:
            todo_diagnostic = f"session TODO list is unreadable: {exc}"

    in_progress_item_ids_not_focused = ()
    if (
        ctx.runtime is not None
        and session.agent.id is not None
        and session.key is not None
    ):
        from thorn.runtime._address import SessionAddress
        from thorn.runtime._inbox import SessionInbox
        from thorn.runtime._notification import NotificationStatus

        address = SessionAddress(session.agent.id, session.key)
        inbox = ctx.runtime.address_book.get(address)
        if isinstance(inbox, SessionInbox):
            focused_id = session.working_set.focused_inbox_item_id
            in_progress_item_ids_not_focused = tuple(
                item.id for item in inbox.prompt_pending()
                if (
                    item.status is NotificationStatus.IN_PROGRESS
                    and item.id != focused_id
                )
            )

    rendered = render_working_set_block(
        session.working_set,
        open_todos=open_todo_items(todos),
        in_progress_item_ids_not_focused=in_progress_item_ids_not_focused,
        todo_diagnostic=todo_diagnostic,
    )
    return _SessionWorkingSetPromptBlock(
        rendered=rendered,
        todo=todo_telemetry_from_statuses(
            tuple(item.status for item in todos),
        ),
        in_progress_not_focused_count=len(in_progress_item_ids_not_focused),
    )


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
