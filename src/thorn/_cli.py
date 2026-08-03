"""CLI entry point: ``thorn run``, ``thorn chat``, and ``thorn serve``."""

from __future__ import annotations

import asyncio
import json
import sys
import time
import uuid
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any

import click
from rich.console import Console

from thorn.agents import LocalCodingAgent
from thorn.agents._action_policy import (
    DEFAULT_RUN_ACTION_POLICY,
    RUN_ACTION_POLICY_DEFINITIONS,
    RunActionPolicy,
    RunActionPolicyDefinition,
)
from thorn.agents._history_policy import (
    DEFAULT_RUN_HISTORY_POLICY,
    RUN_HISTORY_POLICY_DEFINITIONS,
    RunHistoryPolicy,
    RunHistoryPolicyDefinition,
)
from thorn.agents._read_reuse_policy import (
    DEFAULT_RUN_READ_REUSE_POLICY,
    RUN_READ_REUSE_POLICY_DEFINITIONS,
    RunReadReusePolicy,
    RunReadReusePolicyDefinition,
)
from thorn.agents._validation_convergence_policy import (
    DEFAULT_RUN_VALIDATION_CONVERGENCE_POLICY,
    RUN_VALIDATION_CONVERGENCE_POLICY_DEFINITIONS,
    RunValidationConvergencePolicy,
    RunValidationConvergencePolicyDefinition,
)
from thorn.core._agent import Agent
from thorn.core._context import (
    ConsoleEventSink,
    EventSink,
    ExecutionContext,
    Verbosity,
)
from thorn.core._context_ledger import ContextBudgetPolicy
from thorn.core._event_bus import EventBus, in_session
from thorn.core._func import _prepare_tools
from thorn.core._prompt_trace import PromptTraceCapture, PromptTraceRecorder
from thorn.core._provider import (
    LLMConfig,
    LLMModelConfig,
    LLMProviderType,
    OpenAIProviderSettings,
    load_provider_from_config,
    load_provider_from_env,
)
from thorn.core._read_file_history import (
    SESSION_LEDGER_V1_READ_FILE_REUSE_POLICY,
    ReadFileReusePolicy,
)
from thorn.core._session import Session
from thorn.core._tools import ALL_BUILTIN_TOOLS
from thorn.core._validation_convergence import ValidationConvergencePolicy
from thorn.core.errors import SkillError, ThornError
from thorn.gateway._agents import LeanProjectCoordinator, ProjectCoordinator
from thorn.runtime import (
    AgentID,
    AgentScheduler,
    ChatPromptRouter,
    NotificationSpec,
    Runtime,
    SessionAddress,
    SessionInbox,
    SessionKey,
    establish_fresh_cli_direct_focus,
    make_cli_prompt_dispatcher,
)
from thorn.runtime._lock import SessionLockError, session_lock
from thorn.runtime._project_detection import (
    pick_logical_agent_workspace_path_for_cli_session,
)
from thorn.sandbox._warnings import (
    SubprocessSandboxWarning,
    SubprocessSandboxWarningSurface,
    uses_subprocess_sandbox_backend,
)

console = Console()
_LINKED_TODO_SUMMARY_CHARS = 120

THORN_ENV_FILE_ENV_VAR = "THORN_ENV_FILE"


def _emit_json(payload: Any) -> None:
    """Write JSON without Rich wrapping or markup interpretation."""
    click.echo(json.dumps(payload, indent=2, sort_keys=True))


if TYPE_CHECKING:
    from thorn.gateway._agency_admin import AgencyConfigSummary
    from thorn.gateway._config import (
        AgencyConfigFile,
        GatewayConfig,
        SandboxConfig,
    )
    from thorn.gateway._preflight import ForgeAPIPreflightTarget, GitPreflightTarget
    from thorn.runtime import AgencyPaths
    from thorn.tools._credential_scopes import CredentialScopeWarning

CLI_AGENT_ID = AgentID("local")
"""Well-known agent ID for the CLI local coding agent."""


class RunAgentProfile(StrEnum):
    """Built-in agent configurations available to ``thorn run``."""

    LOCAL = "local"
    LEAN_COORDINATOR = "lean-coordinator"
    PROJECT_COORDINATOR = "project-coordinator"


DEFAULT_RUN_AGENT_PROFILE = RunAgentProfile.LOCAL
"""Profile selected by ``thorn run`` when ``--agent-profile`` is omitted."""


class RunPromptDelivery(StrEnum):
    """How a ``thorn run`` prompt enters the selected agent session."""

    DIRECT = "direct"


RUN_PROMPT_DELIVERY = RunPromptDelivery.DIRECT
"""Prompt-delivery mode implemented by the current ``thorn run`` path."""


@dataclass(frozen=True)
class RunAgentProfileDefinition:
    """Stable identity and agent class selected by a run profile."""

    profile: RunAgentProfile
    agent_id: AgentID
    agent_class: type[Agent]

    @property
    def tool_inventory(self) -> tuple[str, ...]:
        """Return the exact ordered tool names exposed to the model."""
        prepared_tools = _prepare_tools(self.agent_class._collect_tools())
        return tuple(
            prepared_tool.schema["function"]["name"]
            for prepared_tool in prepared_tools
        )

    def to_metadata(self) -> dict[str, str | list[str]]:
        """Return the stable profile metadata written to run artifacts."""
        return {
            "agent_profile": self.profile.value,
            "agent_id": str(self.agent_id),
            "agent_class": self.agent_class.__name__,
            "tool_inventory": list(self.tool_inventory),
        }


RUN_AGENT_PROFILE_DEFINITIONS: dict[
    RunAgentProfile,
    RunAgentProfileDefinition,
] = {
    RunAgentProfile.LOCAL: RunAgentProfileDefinition(
        profile=RunAgentProfile.LOCAL,
        agent_id=CLI_AGENT_ID,
        agent_class=LocalCodingAgent,
    ),
    RunAgentProfile.LEAN_COORDINATOR: RunAgentProfileDefinition(
        profile=RunAgentProfile.LEAN_COORDINATOR,
        agent_id=AgentID("cli-lean-coordinator"),
        agent_class=LeanProjectCoordinator,
    ),
    RunAgentProfile.PROJECT_COORDINATOR: RunAgentProfileDefinition(
        profile=RunAgentProfile.PROJECT_COORDINATOR,
        agent_id=AgentID("cli-project-coordinator"),
        agent_class=ProjectCoordinator,
    ),
}


class _WorkspaceState(StrEnum):
    PRESENT = "present"
    MISSING = "missing"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class _CLISessionRecord:
    agent_id: AgentID
    session_key: SessionKey
    created_at: datetime | None
    last_active: datetime | None
    workspace_root: Path | None
    workspace_state: _WorkspaceState

    def to_json(self) -> dict[str, str | None]:
        return {
            "agent_id": str(self.agent_id),
            "key": str(self.session_key),
            "created_at": (
                self.created_at.isoformat()
                if self.created_at is not None
                else None
            ),
            "last_active": (
                self.last_active.isoformat()
                if self.last_active is not None
                else None
            ),
            "workspace_root": (
                str(self.workspace_root)
                if self.workspace_root is not None
                else None
            ),
            "workspace_state": self.workspace_state.value,
        }


@dataclass(frozen=True)
class _CLICommandStartup:
    agency_home: Path
    workspace_root: Path
    resume_session_key: SessionKey | None


@dataclass(frozen=True)
class _PeerResolutionClientError:
    service_name: str
    reason: str


def _workspace_state_for(workspace_root: Path | None) -> _WorkspaceState:
    if workspace_root is None:
        return _WorkspaceState.UNKNOWN
    if workspace_root.is_dir():
        return _WorkspaceState.PRESENT
    return _WorkspaceState.MISSING


def _session_sort_key(record: _CLISessionRecord) -> datetime:
    return (
        record.last_active
        or record.created_at
        or datetime.min.replace(tzinfo=timezone.utc)
    )


def _parse_resume_session_key(raw: str | None) -> SessionKey | None:
    if raw is None:
        return None
    try:
        return SessionKey(raw)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc


def _resolve_cli_sessions_agency_home(agency_path: str | None) -> Path:
    if agency_path is not None:
        agency_home = Path(agency_path).expanduser().resolve()
        if not agency_home.is_dir():
            raise click.ClickException(
                f"Agency home directory does not exist: {agency_home}"
            )
        return agency_home
    return (Path.home() / ".thorn").resolve()


def _load_cli_session_records(
    *,
    agency_home: Path,
    workspace_root: Path,
) -> list[_CLISessionRecord]:
    from thorn.runtime._paths import AgencyPaths
    from thorn.runtime._store import SessionStore

    if not agency_home.is_dir():
        return []

    paths = AgencyPaths(home_root=agency_home, workspace_root=workspace_root)
    store = SessionStore(paths)
    if not store.agent_exists(CLI_AGENT_ID):
        return []

    agent = store.load_agent(CLI_AGENT_ID)
    records: list[_CLISessionRecord] = []
    for session_key in store.list_session_keys(CLI_AGENT_ID):
        if session_key.components[0] != "cli":
            continue
        session = store.load_session(agent, session_key)
        records.append(
            _CLISessionRecord(
                agent_id=CLI_AGENT_ID,
                session_key=session_key,
                created_at=session.created_at,
                last_active=session.last_active,
                workspace_root=session.workspace_root,
                workspace_state=_workspace_state_for(session.workspace_root),
            )
        )
    records.sort(key=_session_sort_key, reverse=True)
    return records


def _load_cli_session_for_resume(
    *,
    agency_home: Path,
    workspace_root: Path,
    agent_id: AgentID,
    session_key: SessionKey,
) -> Session:
    from thorn.runtime._paths import AgencyPaths
    from thorn.runtime._store import SessionStore

    paths = AgencyPaths(home_root=agency_home, workspace_root=workspace_root)
    store = SessionStore(paths)
    if not store.agent_exists(agent_id):
        raise click.ClickException(
            f"No CLI agent {agent_id!s} is persisted under {agency_home}."
        )
    if not store.session_exists(agent_id, session_key):
        raise click.ClickException(
            f"No persisted CLI session {session_key!s} found under "
            f"agent {agent_id!s} in {agency_home}."
        )
    agent = store.load_agent(agent_id)
    return store.load_session(agent, session_key)


def _resolve_cli_command_startup(
    *,
    agency_path: str | None,
    workspace_path: str | None,
    resume_session_key_raw: str | None,
    agent_id: AgentID,
) -> _CLICommandStartup:
    resume_session_key = _parse_resume_session_key(resume_session_key_raw)
    agency_home = (
        _resolve_cli_agency_home(agency_path)
        if resume_session_key is None
        else _resolve_cli_sessions_agency_home(agency_path)
    )
    requested_workspace = (
        Path(workspace_path).resolve()
        if workspace_path is not None
        else None
    )
    if resume_session_key is None:
        return _CLICommandStartup(
            agency_home=agency_home,
            workspace_root=requested_workspace or Path.cwd().resolve(),
            resume_session_key=None,
        )

    provisional_workspace = requested_workspace or Path.cwd().resolve()
    session = _load_cli_session_for_resume(
        agency_home=agency_home,
        workspace_root=provisional_workspace,
        agent_id=agent_id,
        session_key=resume_session_key,
    )
    stored_workspace = (
        session.workspace_root.resolve()
        if session.workspace_root is not None
        else None
    )
    if stored_workspace is not None and requested_workspace is not None:
        if stored_workspace != requested_workspace:
            raise click.ClickException(
                f"CLI session {resume_session_key!s} was created for "
                f"workspace {stored_workspace}, but --workspace resolved "
                f"to {requested_workspace}."
            )

    workspace_root = stored_workspace or requested_workspace or Path.cwd().resolve()
    if not workspace_root.is_dir():
        raise click.ClickException(
            f"Workspace for CLI session {resume_session_key!s} does not "
            f"exist: {workspace_root}"
        )
    return _CLICommandStartup(
        agency_home=agency_home,
        workspace_root=workspace_root,
        resume_session_key=resume_session_key,
    )


def _raise_locked_session(
    *,
    session_key: SessionKey,
    error: SessionLockError,
) -> None:
    raise click.ClickException(
        f"CLI session {session_key!s} is already active in another "
        f"Thorn process: {error}"
    ) from error


def _generate_cli_session_key(workspace_root: Path) -> SessionKey:
    """Generate a fresh session key for a single CLI invocation.

    Every ``thorn run`` and ``thorn chat`` invocation gets a brand-new
    session by default rather than resuming a stable per-workspace one.
    The aspirational architecture doc says:

        Each invocation of ``thorn chat`` or ``thorn run`` creates a
        fresh session by default, with a key that depends on the
        current working directory when the command was run, along
        with a unique ID.

    We encode that as ``cli/<workspace-basename>/<uuid8>`` so that:

    - the top-level ``cli/`` prefix distinguishes ephemeral CLI
      sessions from the named sessions the gateway uses (peers/...,
      projects/..., etc.);
    - the workspace basename makes the on-disk layout scannable
      ("this dir holds chat sessions started from the thorn project"),
      without trying to encode the full absolute path -- basenames
      across different parent dirs can collide, but the uuid tail
      keeps each individual session unique so collisions are
      harmless;
    - the 8-hex-char uuid suffix guarantees uniqueness across
      invocations and is short enough to copy into ``thorn run
      --resume`` or ``thorn chat --resume``.

    Characters in the basename that aren't safe as a single directory
    component are percent-encoded via :func:`safe_dirname`; the rest
    of the key (``cli/`` and the hex tail) is ASCII-safe so the
    encoded form round-trips via :func:`unsafe_dirname`.
    """
    from thorn.runtime._paths import safe_dirname

    basename = workspace_root.name or "root"
    safe_basename = safe_dirname(basename)
    return SessionKey(f"cli/{safe_basename}/{uuid.uuid4().hex[:8]}")

CHAT_SYSTEM_PROMPT = (
    "You are in an interactive chat session with a human user. "
    "You may ask clarifying questions and suggest next steps."
)
"""System prompt fragment appended to the agent's role prompts during
``thorn chat`` turns.  Communicates that the human is on the other side
of the conversation and that asking back is welcome."""


def _ensure_cli_agent(runtime: Runtime) -> LocalCodingAgent:
    """Get or create the CLI local agent, persisting identity to disk.

    The agent is constructed as a :class:`LocalCodingAgent` so that
    its standard tool kit (file I/O, shell, git) flows through the
    ordinary :meth:`Agent._collect_tools` MRO walk -- the dispatcher
    no longer needs to smuggle tools in via ``extra_tools``.
    """
    agent = runtime.get_or_create_agent(
        CLI_AGENT_ID, LocalCodingAgent, name="local",
    )
    runtime.save_agent(agent)
    return agent


def _ensure_run_agent(
    runtime: Runtime,
    profile_definition: RunAgentProfileDefinition,
) -> Agent:
    """Get or create the stable agent selected for a ``thorn run`` profile."""
    agent = runtime.get_or_create_agent(
        profile_definition.agent_id,
        profile_definition.agent_class,
        name=profile_definition.profile.value,
        metadata={"run_agent_profile": profile_definition.profile.value},
    )
    if type(agent) is not profile_definition.agent_class:
        raise ThornError(
            f"Run profile {profile_definition.profile.value!r} expected "
            f"agent class {profile_definition.agent_class.__name__}, but "
            f"persisted agent {profile_definition.agent_id!s} uses "
            f"{type(agent).__name__}."
        )
    agent.metadata["run_agent_profile"] = profile_definition.profile.value
    runtime.save_agent(agent)
    return agent


def _resolve_verbosity(verbose: int, quiet: bool) -> Verbosity:
    """Map ``-v``/``-q`` CLI flags to a :class:`Verbosity` level."""
    if quiet:
        return Verbosity.QUIET
    if verbose >= 2:
        return Verbosity.DEBUG
    if verbose == 1:
        return Verbosity.VERBOSE
    return Verbosity.NORMAL


def _resolve_cli_agency_home(agency_path: str | None) -> Path:
    """Resolve the agency home directory for ``thorn run`` / ``thorn chat``.

    Phase 5 of the CLI/gateway unification switches the CLI default
    agency home from ``{cwd}/.thorn/`` (the old ``for_cli`` nested
    layout) to ``~/.thorn/`` (the local-agency convention from the
    architecture doc).  Home is where agent identity, memory, and
    persisted sessions live; workspace is where the agent does its
    work.  Decoupling the two lets a single local agency serve chats
    launched from different project directories.

    When *agency_path* is supplied (``--agency <dir>``), use it
    verbatim after :func:`~pathlib.Path.expanduser` and
    :func:`~pathlib.Path.resolve`.  Otherwise fall back to
    ``~/.thorn``.  The directory is created if missing -- a fresh
    ``~/.thorn`` is the expected first-run experience.
    """
    if agency_path is not None:
        agency_home = Path(agency_path).expanduser().resolve()
    else:
        agency_home = (Path.home() / ".thorn").resolve()
    agency_home.mkdir(parents=True, exist_ok=True)
    return agency_home


def _build_runtime(
    trace_file: Any | None = None,
    trace_path: str | None = None,
    workspace: str | None = None,
    *,
    paths: "AgencyPaths | None" = None,
    sandbox_executor_enabled: bool = False,
    sandbox_config: "SandboxConfig | None" = None,
    subprocess_tool_workspace_root: Path | None = None,
    llm_config: LLMConfig | None = None,
    prompt_trace_capture: PromptTraceCapture = PromptTraceCapture.REDACTED,
    context_budget_policy: ContextBudgetPolicy | None = None,
    read_file_observation_policy: ReadFileReusePolicy = (
        SESSION_LEDGER_V1_READ_FILE_REUSE_POLICY
    ),
    read_file_reuse_policy: ReadFileReusePolicy | None = None,
    validation_convergence_policy: ValidationConvergencePolicy = (
        ValidationConvergencePolicy.BASELINE
    ),
) -> Runtime:
    """Create a ``Runtime`` whose event sink is an :class:`EventBus`.

    The runtime always carries an :class:`EventBus`, never a bare
    console sink.  Per-command code (``_run``, ``_chat``,
    ``_serve_gateway``) subscribes its own console listener with the
    appropriate scope filter and verbosity; that lets concurrent
    sessions on the same runtime route their output independently
    without each event sink seeing every other session's events.

    When *trace_file* is an open file handle, a :class:`JsonLinesSink`
    is subscribed *here* (without any scope filter) so the trace
    captures everything regardless of which session emitted it.  Trace
    is an operator-audit channel; per-session filtering would defeat
    its purpose.

    When *trace_path* is also supplied, prompt payloads are captured as
    sidecar files under ``<trace_path>.prompts`` and the JSONL trace
    receives small pointer events.

    *workspace* overrides the workspace root.  When ``None``, falls
    back to the resolved current working directory.

    *subprocess_tool_workspace_root* overrides the subprocess toolhost's
    filesystem root.  CLI sessions use their selected workspace directly
    because they do not create gateway-style per-session workspaces.

    *paths* sets the agency directory layout explicitly.  When
    ``None``, falls back to the legacy CLI-nested layout
    (``{ws_root}/.thorn/``) for backwards compatibility with callers
    that haven't yet adopted the Phase-5 ``~/.thorn`` default; all
    production CLI entry points pass *paths* explicitly and do not
    exercise that fallback.
    """
    from pathlib import Path

    from thorn.core._file_access import load_global_ignores
    from thorn.runtime._paths import AgencyPaths

    provider = (
        load_provider_from_config(llm_config)
        if llm_config is not None
        else load_provider_from_env()
    )

    bus = EventBus()
    if trace_file is not None:
        from thorn.core._trace import JsonLinesSink
        bus.subscribe(JsonLinesSink(trace_file))
    prompt_trace_recorder = (
        PromptTraceRecorder.for_trace_path(
            trace_path,
            capture_mode=prompt_trace_capture,
        )
        if trace_path is not None
        else None
    )

    ws_root = Path(workspace).resolve() if workspace else Path.cwd().resolve()

    if paths is None:
        paths = AgencyPaths.for_cli(ws_root)

    return Runtime(
        provider=provider,
        event_sink=bus,
        workspace_root=ws_root,
        global_ignores=load_global_ignores(ws_root),
        paths=paths,
        sandbox_executor_enabled=sandbox_executor_enabled,
        sandbox_config=sandbox_config,
        subprocess_tool_workspace_root=subprocess_tool_workspace_root,
        provider_config=llm_config,
        prompt_trace_recorder=prompt_trace_recorder,
        context_budget_policy=context_budget_policy,
        read_file_observation_policy=read_file_observation_policy,
        read_file_reuse_policy=read_file_reuse_policy,
        validation_convergence_policy=validation_convergence_policy,
    )


def _resolve_prompt_trace_capture(
    *,
    trace_path: str | None,
    trace_raw_prompts: bool,
) -> PromptTraceCapture:
    if trace_raw_prompts and trace_path is None:
        raise click.ClickException(
            "--trace-raw-prompts requires --trace <path>."
        )
    if trace_raw_prompts:
        console.print(
            "[yellow]Warning:[/yellow] raw prompt trace capture is enabled. "
            "Prompt sidecars may contain secrets, private code, and sensitive "
            "conversation content.",
        )
        return PromptTraceCapture.RAW
    return PromptTraceCapture.REDACTED


def _runtime_event_bus(runtime: Runtime) -> EventBus:
    """Return *runtime*'s event sink, asserting it is an :class:`EventBus`.

    Every CLI-built runtime (created via :func:`_build_runtime`) carries
    a bus, so this is essentially a typed accessor.  Lifted out as a
    helper so the assertion is centralised and tests that build a
    runtime by hand get a helpful error message rather than an attribute
    error if they forget the bus.
    """
    sink = runtime.event_sink
    if not isinstance(sink, EventBus):
        raise TypeError(
            f"CLI commands require runtime.event_sink to be an EventBus, "
            f"got {type(sink).__name__}"
        )
    return sink


def _warn_if_cli_uses_subprocess_sandbox(runtime: Runtime) -> None:
    if not runtime.sandbox_executor_enabled:
        return
    if not uses_subprocess_sandbox_backend(runtime.sandbox_config):
        return
    warning = SubprocessSandboxWarning(
        SubprocessSandboxWarningSurface.CLI_COMMAND,
    )
    console.print(f"[bold yellow]{warning.rich_text()}[/bold yellow]")


# ---------------------------------------------------------------------------
# CLI group
# ---------------------------------------------------------------------------

def _load_explicit_env_file(env_file_path: Path | None) -> None:
    if env_file_path is None:
        return

    resolved_path = env_file_path.expanduser().resolve()
    if not resolved_path.is_file():
        raise click.ClickException(
            f"Environment file does not exist: {resolved_path}"
        )

    try:
        from dotenv import load_dotenv
    except ImportError as exc:
        raise click.ClickException(
            "python-dotenv is required to load --env-file."
        ) from exc
    load_dotenv(resolved_path)


@click.group()
@click.version_option(package_name="thorn-agent", prog_name="thorn")
@click.option(
    "--env-file",
    "env_file_path",
    type=click.Path(file_okay=True, dir_okay=False, path_type=Path),
    envvar=THORN_ENV_FILE_ENV_VAR,
    default=None,
    help=(
        "Load environment variables from an explicit dotenv file. "
        f"Can also be set with {THORN_ENV_FILE_ENV_VAR}."
    ),
)
def main(env_file_path: Path | None) -> None:
    """Operate persistent Thorn agencies locally or as gateway services."""
    _load_explicit_env_file(env_file_path)


# ---------------------------------------------------------------------------
# thorn sessions
# ---------------------------------------------------------------------------

@main.group()
def sessions() -> None:
    """Inspect persisted local CLI sessions."""


@sessions.command("list")
@click.option(
    "--agency",
    "agency_path",
    type=click.Path(file_okay=False),
    default=None,
    help=(
        "Agency home directory containing the local CLI agent. "
        "Defaults to ~/.thorn."
    ),
)
@click.option(
    "--json",
    "json_output",
    is_flag=True,
    default=False,
    help="Print machine-readable JSON.",
)
def sessions_list(agency_path: str | None, json_output: bool) -> None:
    """List persisted ``thorn run`` and ``thorn chat`` sessions."""
    agency_home = _resolve_cli_sessions_agency_home(agency_path)
    records = _load_cli_session_records(
        agency_home=agency_home,
        workspace_root=Path.cwd().resolve(),
    )
    if json_output:
        _emit_json([record.to_json() for record in records])
        return
    if not records:
        console.print("[dim]No CLI sessions found.[/dim]")
        return

    console.print(f"[bold]CLI sessions ({len(records)}):[/bold]")
    for record in records:
        last_active = (
            record.last_active.isoformat()
            if record.last_active is not None
            else "unknown"
        )
        workspace = (
            str(record.workspace_root)
            if record.workspace_root is not None
            else "unknown"
        )
        console.print(
            f"  {record.session_key}  last_active={last_active}  "
            f"workspace={workspace} ({record.workspace_state.value})"
        )


# ---------------------------------------------------------------------------
# thorn agency
# ---------------------------------------------------------------------------

def _default_agency_home() -> Path:
    return (Path.home() / ".thorn").resolve()


def _resolve_agency_command_home(agency_path: str | None) -> Path:
    if agency_path is not None:
        return Path(agency_path).expanduser().resolve()
    return _default_agency_home()


def _render_agency_summary(summary: "AgencyConfigSummary") -> None:
    workspace = (
        str(summary.workspace_root)
        if summary.workspace_root is not None
        else "(not configured)"
    )
    console.print(f"  Config: {summary.config_file.path}")
    console.print(f"  Workspace: {workspace}")
    console.print(
        f"  Services: {len(summary.services)}  "
        f"Projects: {len(summary.projects)}  "
        f"Agents: {len(summary.agent_ids)}  "
        f"Peers: {len(summary.peer_ids)}"
    )
    console.print(
        f"  Sandbox: {summary.sandbox_backend or '(none)'}  "
        f"Broker: {summary.broker_mode or '(none)'}"
    )


@main.group()
def agency() -> None:
    """Inspect and initialize Thorn agency configuration."""


@agency.command("init")
@click.argument(
    "agency_path",
    required=False,
    type=click.Path(file_okay=False),
)
@click.option(
    "--workspace",
    "workspace_path",
    type=click.Path(file_okay=False),
    required=True,
    help=(
        "Agency workspace directory.  Recorded in agency.yaml."
    ),
)
@click.option(
    "--json",
    "json_output",
    is_flag=True,
    default=False,
    help="Print machine-readable JSON.",
)
def agency_init(
    agency_path: str | None,
    workspace_path: str,
    json_output: bool,
) -> None:
    """Create a minimal agency home with an agency.yaml config."""
    from thorn.gateway._agency_admin import initialize_agency_home

    agency_home = _resolve_agency_command_home(agency_path)
    workspace_root = Path(workspace_path).expanduser().resolve()

    try:
        result = initialize_agency_home(
            agency_home=agency_home,
            workspace_root=workspace_root,
        )
    except (OSError, ValueError) as exc:
        console.print(f"[red]Error:[/red] {exc}")
        sys.exit(1)

    if json_output:
        _emit_json(result.to_json())
        return

    console.print(f"[green]Created agency:[/green] {result.agency_home}")
    console.print(f"  Config: {result.config_file}")
    console.print(f"  Workspace: {result.workspace_root}")


@agency.command("check")
@click.option(
    "--agency",
    "agency_path",
    type=click.Path(file_okay=False),
    default=None,
    help="Agency home directory.  Defaults to ~/.thorn.",
)
@click.option(
    "--json",
    "json_output",
    is_flag=True,
    default=False,
    help="Print machine-readable JSON.",
)
def agency_check(agency_path: str | None, json_output: bool) -> None:
    """Validate an agency configuration file and persisted agent accounts."""
    from thorn.gateway._agency_admin import summarize_agency_config

    agency_home = _resolve_agency_command_home(agency_path)
    try:
        summary = summarize_agency_config(agency_home)
    except (OSError, ValueError) as exc:
        if json_output:
            _emit_json({
                "status": "error",
                "agency_home": str(agency_home),
                "error": str(exc),
            })
        else:
            console.print(f"[red]Error:[/red] {exc}")
        sys.exit(1)

    if json_output:
        payload = {"status": "ok", **summary.to_json()}
        _emit_json(payload)
        return

    console.print("[green]OK[/green] Agency configuration is valid.")
    _render_agency_summary(summary)


@agency.command("show")
@click.option(
    "--agency",
    "agency_path",
    type=click.Path(file_okay=False),
    default=None,
    help="Agency home directory.  Defaults to ~/.thorn.",
)
@click.option(
    "--json",
    "json_output",
    is_flag=True,
    default=False,
    help="Print machine-readable JSON.",
)
def agency_show(agency_path: str | None, json_output: bool) -> None:
    """Show a resolved summary of the agency configuration."""
    from thorn.gateway._agency_admin import summarize_agency_config

    agency_home = _resolve_agency_command_home(agency_path)
    try:
        summary = summarize_agency_config(agency_home)
    except (OSError, ValueError) as exc:
        if json_output:
            _emit_json({
                "status": "error",
                "agency_home": str(agency_home),
                "error": str(exc),
            })
        else:
            console.print(f"[red]Error:[/red] {exc}")
        sys.exit(1)

    if json_output:
        _emit_json(summary.to_json())
        return

    console.print(f"[bold]Agency:[/bold] {summary.agency_home}")
    _render_agency_summary(summary)


# ---------------------------------------------------------------------------
# thorn inbox
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _ErroredInboxMatch:
    agent_id: AgentID
    session_key: SessionKey
    inbox: SessionInbox


@dataclass(frozen=True)
class _LiveInboxMatch:
    agent_id: AgentID
    session_key: SessionKey
    inbox: SessionInbox


def _resolve_inbox_agency_home(agency_path: str | None) -> Path:
    if agency_path is not None:
        agency_home = Path(agency_path).expanduser().resolve()
    else:
        agency_home = (Path.home() / ".thorn").resolve()
    if not agency_home.is_dir():
        raise click.ClickException(
            f"Agency home directory does not exist: {agency_home}"
        )
    return agency_home


def _inbox_agency_paths(
    agency_home: Path,
    *,
    workspace_root: Path | None = None,
) -> "AgencyPaths":
    from thorn.runtime._paths import AgencyPaths

    return AgencyPaths.for_gateway(
        agency_dir=agency_home,
        workspace_dir=workspace_root or Path.cwd().resolve(),
    )


def _parse_agent_id_filter(agent_id_raw: str | None) -> AgentID | None:
    return AgentID(agent_id_raw) if agent_id_raw else None


def _parse_session_key_filter(session_key_raw: str | None) -> SessionKey | None:
    if session_key_raw is None:
        return None
    try:
        return SessionKey(session_key_raw)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc


def _find_errored_inbox_matches(
    *,
    agency_home: Path,
    item_id: str,
    agent_id_filter: AgentID | None,
    session_key_filter: SessionKey | None,
) -> list[_ErroredInboxMatch]:
    from thorn.runtime._paths import AgencyPaths

    paths = AgencyPaths(
        home_root=agency_home,
        workspace_root=Path.cwd().resolve(),
    )
    matches: list[_ErroredInboxMatch] = []
    for agent_id, session_key, inbox_dir in paths.iter_session_inbox_locations():
        if agent_id_filter is not None and agent_id != agent_id_filter:
            continue
        if session_key_filter is not None and session_key != session_key_filter:
            continue
        inbox = SessionInbox(
            inbox_dir,
            SessionAddress(agent_id, session_key),
        )
        if any(item.id == item_id for item in inbox.errored_items()):
            matches.append(_ErroredInboxMatch(agent_id, session_key, inbox))
    return matches


def _find_live_inbox_matches(
    *,
    agency_home: Path,
    item_id: str,
    agent_id_filter: AgentID | None,
    session_key_filter: SessionKey | None,
) -> list[_LiveInboxMatch]:
    from thorn.runtime._paths import AgencyPaths

    paths = AgencyPaths(
        home_root=agency_home,
        workspace_root=Path.cwd().resolve(),
    )
    matches: list[_LiveInboxMatch] = []
    for agent_id, session_key, inbox_dir in paths.iter_session_inbox_locations():
        if agent_id_filter is not None and agent_id != agent_id_filter:
            continue
        if session_key_filter is not None and session_key != session_key_filter:
            continue
        inbox = SessionInbox(
            inbox_dir,
            SessionAddress(agent_id, session_key),
        )
        if any(item.id == item_id for item in inbox.prompt_pending()):
            matches.append(_LiveInboxMatch(agent_id, session_key, inbox))
    return matches


@main.group()
def inbox() -> None:
    """Inspect and recover session inbox items."""


@inbox.command("list")
@click.option(
    "--agency",
    "agency_path",
    type=click.Path(file_okay=False),
    default=None,
    help=(
        "Agency home directory containing the agents/ tree. "
        "Defaults to ~/.thorn."
    ),
)
@click.option(
    "--agent",
    "agent_id_raw",
    default=None,
    help="Restrict lookup to one agent ID.",
)
@click.option(
    "--session",
    "session_key_raw",
    default=None,
    help="Restrict lookup to one session key.",
)
@click.option(
    "--status",
    "status_filter",
    type=click.Choice([
        "pending",
        "in_progress",
        "handled",
        "errored",
        "confirmed",
        "parked_errored",
    ]),
    default=None,
    help="Restrict output to one inbox status.",
)
@click.option(
    "--json",
    "json_output",
    is_flag=True,
    default=False,
    help="Print machine-readable JSON.",
)
def inbox_list(
    agency_path: str | None,
    agent_id_raw: str | None,
    session_key_raw: str | None,
    status_filter: str | None,
    json_output: bool,
) -> None:
    """List session inbox items without reading raw JSON files."""
    from thorn.gateway._operator_status import collect_inbox_items

    agency_home = _resolve_inbox_agency_home(agency_path)
    paths = _inbox_agency_paths(agency_home)
    records = collect_inbox_items(
        paths,
        agent_id_filter=_parse_agent_id_filter(agent_id_raw),
        session_key_filter=_parse_session_key_filter(session_key_raw),
        status_filter=status_filter,
    )
    if json_output:
        _emit_json([record.to_json() for record in records])
        return
    if not records:
        console.print("[dim]No inbox items found.[/dim]")
        return

    console.print(f"[bold]Inbox items ({len(records)}):[/bold]")
    for record in records:
        linked_todos = _format_linked_todos_for_cli(record.linked_todos)
        linked_todo_suffix = f"  {linked_todos}" if linked_todos else ""
        console.print(
            f"  {record.item_id}  {record.status.value}  "
            f"{record.location.value}  agent={record.agent_id}  "
            f"session={record.session_key}  source={record.notification.source}  "
            f"{record.summary}"
            f"{linked_todo_suffix}"
        )


@inbox.command("show")
@click.argument("item_id")
@click.option(
    "--agency",
    "agency_path",
    type=click.Path(file_okay=False),
    default=None,
    help=(
        "Agency home directory containing the agents/ tree. "
        "Defaults to ~/.thorn."
    ),
)
@click.option(
    "--agent",
    "agent_id_raw",
    default=None,
    help="Restrict lookup to one agent ID.",
)
@click.option(
    "--session",
    "session_key_raw",
    default=None,
    help="Restrict lookup to one session key.",
)
@click.option(
    "--json",
    "json_output",
    is_flag=True,
    default=False,
    help="Print machine-readable JSON.",
)
def inbox_show(
    item_id: str,
    agency_path: str | None,
    agent_id_raw: str | None,
    session_key_raw: str | None,
    json_output: bool,
) -> None:
    """Show one session inbox item and its operator recovery context."""
    from thorn.gateway._operator_status import (
        InboxItemLocation,
        collect_inbox_items,
    )

    agency_home = _resolve_inbox_agency_home(agency_path)
    paths = _inbox_agency_paths(agency_home)
    records = collect_inbox_items(
        paths,
        agent_id_filter=_parse_agent_id_filter(agent_id_raw),
        session_key_filter=_parse_session_key_filter(session_key_raw),
        item_id_filter=item_id,
    )
    if not records:
        raise click.ClickException(
            f"No inbox item {item_id!r} found under {agency_home}."
        )
    if len(records) > 1:
        choices = [
            f"{record.agent_id}:{record.session_key}:{record.location.value}"
            for record in records
        ]
        raise click.ClickException(
            "Multiple inbox items matched. Re-run with --agent and "
            "--session to choose one:\n  "
            + "\n  ".join(choices)
        )

    record = records[0]
    if json_output:
        _emit_json(record.to_json(include_content=True))
        return

    notification = record.notification
    console.print(f"[bold]Inbox item:[/bold] {record.item_id}")
    console.print(f"Agent: {record.agent_id}")
    console.print(f"Session: {record.session_key}")
    console.print(f"Location: {record.location.value}")
    console.print(f"Status: {record.status.value}")
    console.print(f"Source: {notification.source}")
    console.print(f"Posted: {notification.posted_at.isoformat()}")
    if notification.external_key:
        console.print(f"External key: {notification.external_key}")
    if notification.attempt_count:
        console.print(f"Attempts: {notification.attempt_count}")
    if notification.notes:
        console.print(f"Notes: {notification.notes}")
    linked_todos = _format_linked_todos_for_cli(record.linked_todos)
    if linked_todos:
        console.print(f"Linked TODOs: {linked_todos}")
    if notification.completion_rationale:
        console.print(notification.completion_rationale.to_display_text())
    if notification.error_reason:
        console.print(f"Error reason: {notification.error_reason}")
    if notification.metadata:
        console.print(f"Metadata: {dict(notification.metadata)!r}")
    if record.location is InboxItemLocation.PARKED_ERRORED:
        console.print(
            "Requeue: "
            f"uv run thorn inbox requeue {record.item_id} "
            f"--agency {agency_home} "
            f"--agent {record.agent_id} --session {record.session_key}"
        )
    console.print("\n[bold]Content:[/bold]")
    console.print(notification.content)


def _format_linked_todos_for_cli(linked_todos: Any) -> str:
    if linked_todos.total_count == 0:
        return ""
    text = (
        f"TODOs: {linked_todos.open_count} open, "
        f"{linked_todos.resolved_count} resolved"
    )
    if linked_todos.open_titles:
        text += ": " + _truncate_cli_summary("; ".join(linked_todos.open_titles))
    return text


def _truncate_cli_summary(text: str) -> str:
    if len(text) <= _LINKED_TODO_SUMMARY_CHARS:
        return text
    return text[: _LINKED_TODO_SUMMARY_CHARS - 3].rstrip() + "..."


@inbox.command("requeue")
@click.argument("item_id")
@click.option(
    "--agency",
    "agency_path",
    type=click.Path(file_okay=False),
    default=None,
    help=(
        "Agency home directory containing the agents/ tree. "
        "Defaults to ~/.thorn."
    ),
)
@click.option(
    "--agent",
    "agent_id_raw",
    default=None,
    help="Restrict lookup to one agent ID.",
)
@click.option(
    "--session",
    "session_key_raw",
    default=None,
    help="Restrict lookup to one session key.",
)
def inbox_requeue(
    item_id: str,
    agency_path: str | None,
    agent_id_raw: str | None,
    session_key_raw: str | None,
) -> None:
    """Move a parked errored inbox item back to pending work.

    Use this after fixing an operator-side problem such as a bad
    provider key.  The command does not contact upstream event sources
    or re-create GitLab TODOs; it only moves Thorn's durable inbox
    item from ``inbox/errored/`` back into the live session inbox so
    the next gateway run can prompt the coordinator again.
    """
    agent_id_filter = AgentID(agent_id_raw) if agent_id_raw else None
    try:
        session_key_filter = (
            SessionKey(session_key_raw) if session_key_raw else None
        )
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc

    agency_home = _resolve_inbox_agency_home(agency_path)
    matches = _find_errored_inbox_matches(
        agency_home=agency_home,
        item_id=item_id,
        agent_id_filter=agent_id_filter,
        session_key_filter=session_key_filter,
    )
    if not matches:
        raise click.ClickException(
            f"No parked errored inbox item {item_id!r} found under {agency_home}."
        )
    if len(matches) > 1:
        lines = [
            f"{match.agent_id}:{match.session_key}"
            for match in matches
        ]
        raise click.ClickException(
            "Multiple parked errored inbox items matched. Re-run with "
            "--agent and --session to choose one:\n  "
            + "\n  ".join(lines)
        )

    match = matches[0]
    requeued = match.inbox.requeue_errored(item_id)
    console.print(
        "Requeued inbox item "
        f"[bold]{requeued.id}[/bold] for agent "
        f"[bold]{match.agent_id}[/bold], session "
        f"[bold]{match.session_key}[/bold]."
    )


@inbox.command("park")
@click.argument("item_id")
@click.option(
    "--reason",
    required=True,
    help="Operator-visible reason recorded on the parked item.",
)
@click.option(
    "--agency",
    "agency_path",
    type=click.Path(file_okay=False),
    default=None,
    help=(
        "Agency home directory containing the agents/ tree. "
        "Defaults to ~/.thorn."
    ),
)
@click.option(
    "--agent",
    "agent_id_raw",
    default=None,
    help="Restrict lookup to one agent ID.",
)
@click.option(
    "--session",
    "session_key_raw",
    default=None,
    help="Restrict lookup to one session key.",
)
def inbox_park(
    item_id: str,
    reason: str,
    agency_path: str | None,
    agent_id_raw: str | None,
    session_key_raw: str | None,
) -> None:
    """Move a live pending/in-progress inbox item to parked errored."""
    agent_id_filter = AgentID(agent_id_raw) if agent_id_raw else None
    try:
        session_key_filter = (
            SessionKey(session_key_raw) if session_key_raw else None
        )
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc

    agency_home = _resolve_inbox_agency_home(agency_path)
    matches = _find_live_inbox_matches(
        agency_home=agency_home,
        item_id=item_id,
        agent_id_filter=agent_id_filter,
        session_key_filter=session_key_filter,
    )
    if not matches:
        raise click.ClickException(
            f"No live pending/in_progress inbox item {item_id!r} "
            f"found under {agency_home}."
        )
    if len(matches) > 1:
        lines = [
            f"{match.agent_id}:{match.session_key}"
            for match in matches
        ]
        raise click.ClickException(
            "Multiple live inbox items matched. Re-run with --agent "
            "and --session to choose one:\n  "
            + "\n  ".join(lines)
        )

    match = matches[0]
    parked = match.inbox.park_live(item_id, error_reason=reason)
    console.print(
        "Parked inbox item "
        f"[bold]{parked.id}[/bold] for agent "
        f"[bold]{match.agent_id}[/bold], session "
        f"[bold]{match.session_key}[/bold]."
    )


# ---------------------------------------------------------------------------
# thorn status
# ---------------------------------------------------------------------------

@main.command("status")
@click.option(
    "--agency",
    "agency_path",
    type=click.Path(file_okay=False),
    default=None,
    help=(
        "Agency home directory containing gateway.json and agents/. "
        "Defaults to ~/.thorn."
    ),
)
@click.option(
    "--json",
    "json_output",
    is_flag=True,
    default=False,
    help="Print machine-readable JSON.",
)
def operator_status(agency_path: str | None, json_output: bool) -> None:
    """Summarize gateway, inbox, broker, sandbox, and source health."""
    from thorn.gateway import load_gateway_config
    from thorn.gateway._operator_status import collect_operator_status

    agency_home = _resolve_inbox_agency_home(agency_path)
    gateway_config = None
    config_error: str | None = None
    try:
        gateway_config = load_gateway_config(agency_home)
    except Exception as exc:
        config_error = str(exc)

    workspace_root = Path.cwd().resolve()
    if gateway_config is not None:
        resolved_workspace = gateway_config.resolve_workspace(agency_home)
        if resolved_workspace is not None:
            workspace_root = resolved_workspace

    summary = asyncio.run(collect_operator_status(
        agency_home=agency_home,
        workspace_root=workspace_root,
        gateway_config=gateway_config,
        config_error=config_error,
    ))
    if json_output:
        _emit_json(summary.to_json())
        return

    _print_operator_status(summary)


def _print_operator_status(summary: Any) -> None:
    console.print(f"[bold]Agency:[/bold] {summary.agency_home}")
    console.print(f"[bold]Workspace:[/bold] {summary.workspace_root}")
    if summary.config_error:
        console.print(f"[yellow]Config:[/yellow] {summary.config_error}")

    heartbeat = summary.heartbeat
    console.print(
        f"[bold]Gateway:[/bold] {heartbeat.liveness.value} "
        f"[dim]({heartbeat.path})[/dim]"
    )
    payload = heartbeat.payload or {}
    updated_at = payload.get("updated_at")
    if updated_at:
        console.print(f"  updated_at={updated_at}")
    provider = payload.get("provider_health")
    if isinstance(provider, dict):
        console.print(
            "  provider="
            f"{provider.get('state', 'unknown')} "
            f"failures={provider.get('recent_failure_count', '?')} "
            f"probe_in_flight={provider.get('probe_in_flight', '?')}"
        )

    sources = payload.get("sources")
    if isinstance(sources, list):
        console.print(f"[bold]Sources:[/bold] {len(sources)}")
        for source in sources:
            if not isinstance(source, dict):
                continue
            console.print(
                f"  {source.get('name') or source.get('source_type')}  "
                f"{source.get('state', 'unknown')}  "
                f"last_poll={source.get('last_poll_finished_at') or 'never'}  "
                f"events={source.get('last_event_count')}"
            )
            if source.get("last_error"):
                console.print(f"    error={source['last_error']}")
    else:
        console.print("[bold]Sources:[/bold] unknown")

    console.print(
        f"[bold]Agents:[/bold] {len(summary.agent_ids)}  "
        f"[bold]Sessions:[/bold] {summary.session_count}"
    )
    counts = summary.inbox_counts
    console.print(
        "[bold]Inbox:[/bold] "
        f"pending={counts.pending} "
        f"in_progress={counts.in_progress} "
        f"handled={counts.handled} "
        f"errored={counts.errored} "
        f"confirmed={counts.confirmed} "
        f"parked_errored={counts.parked_errored}"
    )
    if counts.parked_errored:
        console.print(
            "  [yellow]Use 'thorn inbox list --status parked_errored' "
            "and 'thorn inbox requeue <item-id>' after fixing the cause.[/yellow]"
        )
    console.print(
        f"[bold]In-flight external keys:[/bold] "
        f"{len(summary.in_flight_external_keys)}"
    )
    provider_waits = summary.provider_unavailable_sessions
    if provider_waits:
        console.print(f"[bold]Provider waits:[/bold] {len(provider_waits)}")
        for session in provider_waits[:5]:
            metadata = session.metadata
            console.print(
                f"  {session.agent_id}/{session.session_key}  "
                f"attempts={metadata.get('attempts', '?')}  "
                f"reason={metadata.get('reason', 'unknown')}"
            )

    if summary.alerts:
        console.print(f"[bold]Alerts:[/bold] {len(summary.alerts)}")
        for alert in summary.alerts:
            style = "red" if alert.severity == "error" else "yellow"
            console.print(
                f"  [{style}]{alert.severity.upper()}[/{style}] "
                f"{alert.code}: {alert.summary}"
            )
    else:
        console.print("[bold]Alerts:[/bold] none")

    broker = summary.broker
    if broker.error:
        console.print(f"[bold]Broker:[/bold] unknown ({broker.error})")
    elif broker.stacks:
        console.print(f"[bold]Broker stacks:[/bold] {len(broker.stacks)}")
        for stack in broker.stacks:
            console.print(
                f"  {stack.project_name}  ({stack.runtime_name})  "
                f"{stack.status}"
            )
    else:
        console.print("[bold]Broker stacks:[/bold] none")

    sandbox = summary.sandbox
    if sandbox.error:
        console.print(f"[bold]Sandbox:[/bold] unknown ({sandbox.error})")
    elif sandbox.backend == "subprocess":
        console.print("[bold]Sandbox:[/bold] backend=subprocess")
    else:
        image_state = "present" if sandbox.image_present else "missing"
        console.print(
            f"[bold]Sandbox:[/bold] backend={sandbox.backend} "
            f"runtime={sandbox.runtime_name} "
            f"image={sandbox.image} ({image_state}) "
            f"containers={len(sandbox.containers)}"
        )


# ---------------------------------------------------------------------------
# thorn run
# ---------------------------------------------------------------------------

def _write_result_file(
    path: Path,
    outcome: str,
    duration_s: float,
    usage: Any,
    error: str | None,
    trace_path: str | None,
    profile_definition: RunAgentProfileDefinition,
    action_policy_definition: RunActionPolicyDefinition,
    history_policy_definition: RunHistoryPolicyDefinition,
    read_reuse_policy_definition: RunReadReusePolicyDefinition,
    validation_convergence_policy_definition: (
        RunValidationConvergencePolicyDefinition
    ),
) -> None:
    """Write the structured JSON result produced by ``thorn run --result-file``."""
    if usage is not None:
        token_usage = {
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens,
        }
    else:
        token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    result = {
        "outcome": outcome,
        "duration_s": round(duration_s, 2),
        "token_usage": token_usage,
        "error": error,
        "trace_file": Path(trace_path).name if trace_path else None,
    }
    result.update(profile_definition.to_metadata())
    result.update(action_policy_definition.to_metadata())
    result.update(history_policy_definition.to_metadata())
    result.update(read_reuse_policy_definition.to_metadata())
    result.update(validation_convergence_policy_definition.to_metadata())
    result["prompt_delivery"] = RUN_PROMPT_DELIVERY.value
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


def _write_run_trace_metadata(
    trace_file: Any,
    profile_definition: RunAgentProfileDefinition,
    action_policy_definition: RunActionPolicyDefinition,
    history_policy_definition: RunHistoryPolicyDefinition,
    read_reuse_policy_definition: RunReadReusePolicyDefinition,
    validation_convergence_policy_definition: (
        RunValidationConvergencePolicyDefinition
    ),
) -> None:
    """Write the selected run configuration before execution events."""
    record: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": "run_metadata",
        "scope": [],
    }
    record.update(profile_definition.to_metadata())
    record.update(action_policy_definition.to_metadata())
    record.update(history_policy_definition.to_metadata())
    record.update(read_reuse_policy_definition.to_metadata())
    record.update(validation_convergence_policy_definition.to_metadata())
    record["prompt_delivery"] = RUN_PROMPT_DELIVERY.value
    trace_file.write(json.dumps(record) + "\n")
    trace_file.flush()


def _parse_run_agent_profile(
    _context: click.Context,
    _parameter: click.Parameter,
    value: str,
) -> RunAgentProfile:
    """Convert Click's validated profile choice into the domain type."""
    return RunAgentProfile(value)


def _parse_run_action_policy(
    _context: click.Context,
    _parameter: click.Parameter,
    value: str,
) -> RunActionPolicy:
    """Convert Click's validated action-policy choice into the domain type."""
    return RunActionPolicy(value)


def _parse_run_history_policy(
    _context: click.Context,
    _parameter: click.Parameter,
    value: str,
) -> RunHistoryPolicy:
    """Convert Click's validated history-policy choice into the domain type."""
    return RunHistoryPolicy(value)


def _parse_run_read_reuse_policy(
    _context: click.Context,
    _parameter: click.Parameter,
    value: str,
) -> RunReadReusePolicy:
    """Convert Click's validated read-reuse choice into the domain type."""
    return RunReadReusePolicy(value)


def _parse_run_validation_convergence_policy(
    _context: click.Context,
    _parameter: click.Parameter,
    value: str,
) -> RunValidationConvergencePolicy:
    """Convert Click's validated convergence policy into the domain type."""
    return RunValidationConvergencePolicy(value)


@main.command()
@click.argument("prompt_text")
@click.option("-v", "--verbose", count=True, help="Increase output detail (-v, -vv).")
@click.option("-q", "--quiet", is_flag=True, default=False, help="Suppress all output except the final answer.")
@click.option("--trace", "trace_path", type=click.Path(), default=None, help="Write execution trace to a JSONL file.")
@click.option(
    "--trace-raw-prompts",
    is_flag=True,
    default=False,
    help=(
        "Write raw prompt payload sidecars next to --trace. "
        "May expose secrets and sensitive content."
    ),
)
@click.option("--workspace", "workspace_path", type=click.Path(exists=True, file_okay=False), default=None, help="Override workspace root directory.")
@click.option("--resume", "resume_session_key_raw", default=None, help="Resume an existing CLI session key.")
@click.option(
    "--agency",
    "agency_path",
    type=click.Path(file_okay=False),
    default=None,
    help=(
        "Agency home directory (holds agent identities, sessions, "
        "and journals).  Defaults to ~/.thorn/; created if missing."
    ),
)
@click.option(
    "--agent-profile",
    type=click.Choice(
        [profile.value for profile in RunAgentProfile],
        case_sensitive=False,
    ),
    default=DEFAULT_RUN_AGENT_PROFILE.value,
    show_default=True,
    callback=_parse_run_agent_profile,
    help="Select the agent role and tool surface for this run.",
)
@click.option(
    "--action-policy",
    type=click.Choice(
        [policy.value for policy in RunActionPolicy],
        case_sensitive=False,
    ),
    default=DEFAULT_RUN_ACTION_POLICY.value,
    show_default=True,
    callback=_parse_run_action_policy,
    help="Select the model-facing execution contract for this run.",
)
@click.option(
    "--history-policy",
    type=click.Choice(
        [policy.value for policy in RunHistoryPolicy],
        case_sensitive=False,
    ),
    default=DEFAULT_RUN_HISTORY_POLICY.value,
    show_default=True,
    callback=_parse_run_history_policy,
    help="Select the provider-visible history projection policy for this run.",
)
@click.option(
    "--read-reuse-policy",
    type=click.Choice(
        [policy.value for policy in RunReadReusePolicy],
        case_sensitive=False,
    ),
    default=DEFAULT_RUN_READ_REUSE_POLICY.value,
    show_default=True,
    callback=_parse_run_read_reuse_policy,
    help="Select the session-local read-reuse advisory policy for this run.",
)
@click.option(
    "--validation-convergence-policy",
    type=click.Choice(
        [policy.value for policy in RunValidationConvergencePolicy],
        case_sensitive=False,
    ),
    default=DEFAULT_RUN_VALIDATION_CONVERGENCE_POLICY.value,
    show_default=True,
    callback=_parse_run_validation_convergence_policy,
    help="Select validation progress convergence tracking for this run.",
)
@click.option("--result-file", "result_file_path", type=click.Path(), default=None, help="Write a JSON result summary (outcome, duration, token usage).")
def run(
    prompt_text: str,
    verbose: int,
    quiet: bool,
    trace_path: str | None,
    trace_raw_prompts: bool,
    workspace_path: str | None,
    resume_session_key_raw: str | None,
    agency_path: str | None,
    agent_profile: RunAgentProfile,
    action_policy: RunActionPolicy,
    history_policy: RunHistoryPolicy,
    read_reuse_policy: RunReadReusePolicy,
    validation_convergence_policy: RunValidationConvergencePolicy,
    result_file_path: str | None,
) -> None:
    """Execute a single prompt and print the result."""
    from thorn.runtime._paths import AgencyPaths

    verbosity = _resolve_verbosity(verbose, quiet)
    prompt_trace_capture = _resolve_prompt_trace_capture(
        trace_path=trace_path,
        trace_raw_prompts=trace_raw_prompts,
    )
    profile_definition = RUN_AGENT_PROFILE_DEFINITIONS[agent_profile]
    action_policy_definition = RUN_ACTION_POLICY_DEFINITIONS[action_policy]
    history_policy_definition = RUN_HISTORY_POLICY_DEFINITIONS[history_policy]
    read_reuse_policy_definition = RUN_READ_REUSE_POLICY_DEFINITIONS[
        read_reuse_policy
    ]
    validation_convergence_policy_definition = (
        RUN_VALIDATION_CONVERGENCE_POLICY_DEFINITIONS[
            validation_convergence_policy
        ]
    )
    trace_file = open(trace_path, "w", encoding="utf-8") if trace_path else None
    if trace_file is not None:
        _write_run_trace_metadata(
            trace_file,
            profile_definition,
            action_policy_definition,
            history_policy_definition,
            read_reuse_policy_definition,
            validation_convergence_policy_definition,
        )
    try:
        startup = _resolve_cli_command_startup(
            agency_path=agency_path,
            workspace_path=workspace_path,
            resume_session_key_raw=resume_session_key_raw,
            agent_id=profile_definition.agent_id,
        )
        paths = AgencyPaths(
            home_root=startup.agency_home,
            workspace_root=startup.workspace_root,
        )
        runtime = _build_runtime(
            trace_file=trace_file,
            trace_path=trace_path,
            workspace=str(startup.workspace_root),
            paths=paths,
            sandbox_executor_enabled=True,
            subprocess_tool_workspace_root=startup.workspace_root,
            prompt_trace_capture=prompt_trace_capture,
            context_budget_policy=history_policy_definition.context_budget_policy,
            read_file_observation_policy=(
                read_reuse_policy_definition.read_file_observation_policy
            ),
            read_file_reuse_policy=(
                read_reuse_policy_definition.read_file_advisory_policy
            ),
            validation_convergence_policy=(
                validation_convergence_policy_definition.validation_convergence_policy
            ),
        )
        _warn_if_cli_uses_subprocess_sandbox(runtime)
    except ThornError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        if trace_file:
            trace_file.close()
        if result_file_path:
            _write_result_file(
                Path(result_file_path), "agent_error", 0.0, None, str(exc), trace_path,
                profile_definition,
                action_policy_definition,
                history_policy_definition,
                read_reuse_policy_definition,
                validation_convergence_policy_definition,
            )
        sys.exit(1)
    except click.ClickException as exc:
        if trace_file:
            trace_file.close()
        if result_file_path:
            _write_result_file(
                Path(result_file_path),
                "agent_error",
                0.0,
                None,
                exc.message,
                trace_path,
                profile_definition,
                action_policy_definition,
                history_policy_definition,
                read_reuse_policy_definition,
                validation_convergence_policy_definition,
            )
        raise

    ctx_holder: list[ExecutionContext] = []

    async def _run() -> str:
        # Phase 2 of the CLI/gateway unification: route ``thorn run``
        # through the in-process ``AgentScheduler`` rather than
        # invoking ``run_agent_loop`` directly.  The notification
        # round-trip is overkill for a one-shot request on its own,
        # but it lets ``thorn run``, ``thorn chat`` (post-Phase 4),
        # and the gateway daemon all drive sessions through the same
        # scheduler+dispatcher pipeline -- which is the whole point
        # of the unification.  A future-based dispatcher
        # (``make_cli_prompt_dispatcher``) bridges the scheduler's
        # fire-and-forget shape back to the synchronous "give me the
        # answer" shape ``thorn run`` needs.
        async with runtime:
            ctx_holder.append(runtime.context)
            agent = _ensure_run_agent(runtime, profile_definition)

            # By default every CLI invocation gets a fresh session
            # under a unique key (``cli/<workspace-basename>/<uuid>``);
            # ``--resume`` opts into loading one of those persisted
            # sessions and appending another one-shot prompt.
            if agent.id is None:
                raise RuntimeError(
                    "CLI agent has no id; cannot build a session inbox"
                )
            session_key = (
                startup.resume_session_key
                or _generate_cli_session_key(runtime.workspace_root)
            )
            # Pick the logical agent-workspace upper bound for this
            # CLI session by scanning ancestors of the session
            # workspace for a project-root marker.  Persisted on
            # the Session so the per-prompt context-gathering walk
            # has a single, well-defined upper bound regardless of
            # whether the session is later resumed from a different
            # CWD.  The session's CWD itself is the workspace's
            # *lower* bound (the inner end of the walk).
            logical_agent_workspace_path = (
                pick_logical_agent_workspace_path_for_cli_session(
                    runtime.workspace_root,
                )
            )

            # Phase 3: the runtime carries an ``EventBus`` rather
            # than a single sink; subscribe a console listener
            # that only fires for *this* session's events so
            # background activity in other sessions (none today
            # under ``thorn run``, but plausible once the local
            # agency lands in Phase 5) won't bleed into our
            # console output.  Subscribing here -- *before* the
            # session is submitted -- ensures we don't miss the
            # initial agent scope-enter event.
            bus = _runtime_event_bus(runtime)
            console_listener: EventSink = ConsoleEventSink(
                verbosity=verbosity,
            )
            with bus.subscribe(
                console_listener,
                scope_filter=in_session(session_key),
            ):
                session = runtime.get_or_create_session(
                    agent,
                    session_key,
                    workspace_root=runtime.workspace_root,
                    logical_agent_workspace_path=(
                        logical_agent_workspace_path
                    ),
                )

                session_address = SessionAddress(agent.id, session_key)
                inbox = SessionInbox(
                    runtime.paths.session_inbox_dir(
                        agent.id, session_key,
                    ),
                    session_address,
                    in_flight_index=runtime.in_flight_index,
                )
                runtime.address_book.register(session_address, inbox)

                loop = asyncio.get_running_loop()
                result_future: asyncio.Future[str] = loop.create_future()

                async def _save_session_async(sess: Session) -> None:
                    runtime.save_session(sess)

                scheduler: AgentScheduler | None = None
                framework_focused_item_id = None
                try:
                    lock_context = (
                        session_lock(
                            runtime.paths.session_metadata_dir(
                                agent.id, session_key,
                            )
                        )
                        if startup.resume_session_key is not None
                        else nullcontext()
                    )
                    try:
                        with lock_context:
                            posted = inbox.post(NotificationSpec(
                                source="user",
                                content=prompt_text,
                                target=session_address,
                            ))
                            if startup.resume_session_key is None:
                                framework_focused_item_id = (
                                    establish_fresh_cli_direct_focus(
                                        session=session,
                                        inbox=inbox,
                                        notification=posted,
                                        address_book=runtime.address_book,
                                    )
                                )
                                if framework_focused_item_id is not None:
                                    runtime.save_session(session)

                            direct_system_prompt = (
                                "You are executing a single "
                                "non-interactive request. Complete the "
                                "task and report results concisely. Do not "
                                "offer follow-up actions or ask questions."
                            )
                            if framework_focused_item_id is not None:
                                direct_system_prompt += (
                                    " When you call complete_focused_work, "
                                    "include the final user-facing report in "
                                    "the accompanying assistant text; a "
                                    "successful completion ends this request."
                                )
                            if action_policy_definition.system_prompt is not None:
                                direct_system_prompt += (
                                    "\n\n" + action_policy_definition.system_prompt
                                )
                            dispatcher = make_cli_prompt_dispatcher(
                                result_future=result_future,
                                extra_system=direct_system_prompt,
                                framework_focused_item_id=(
                                    framework_focused_item_id
                                ),
                                terminal_completion_item_id=(
                                    framework_focused_item_id
                                ),
                            )
                            scheduler = AgentScheduler(
                                agent=agent,
                                prompt_dispatcher=dispatcher,
                                save_session=_save_session_async,
                            )
                            await scheduler.submit(session, inbox)
                            return await result_future
                    except SessionLockError as exc:
                        _raise_locked_session(
                            session_key=session_key,
                            error=exc,
                        )
                finally:
                    # Bounded grace period so a misbehaving
                    # dispatcher cannot wedge process exit
                    # indefinitely.  By the time we get here the
                    # future has resolved, the dispatcher has
                    # returned, and the driver has parked on its
                    # idle wait, so shutdown is essentially
                    # instantaneous in the success case.
                    if scheduler is not None:
                        await scheduler.shutdown(timeout=5.0)
                    if framework_focused_item_id is not None:
                        runtime.save_session(session)

    outcome = "success"
    error_msg: str | None = None
    exit_code = 0
    t0 = time.monotonic()

    try:
        final_answer = asyncio.run(_run())
        if quiet:
            click.echo(final_answer)
    except SkillError as exc:
        outcome, error_msg, exit_code = "agent_error", exc.detail, 1
        console.print(f"\n[red]Agent error:[/red] {exc.detail}")
    except TimeoutError:
        outcome, error_msg, exit_code = "timeout", "timed out", 1
        console.print("\n[red]Error:[/red] timed out")
    except ThornError as exc:
        outcome, error_msg, exit_code = "agent_error", str(exc), 1
        console.print(f"\n[red]Error:[/red] {exc}")
    except click.ClickException as exc:
        outcome, error_msg, exit_code = "agent_error", exc.message, 1
        raise
    except KeyboardInterrupt:
        outcome, error_msg, exit_code = "agent_error", "interrupted", 130
        console.print("\n[dim]Interrupted.[/dim]")
    finally:
        duration_s = time.monotonic() - t0
        if trace_file:
            trace_file.close()
        if result_file_path:
            ctx = ctx_holder[0] if ctx_holder else None
            _write_result_file(
                Path(result_file_path), outcome, duration_s,
                ctx.usage if ctx else None, error_msg, trace_path,
                profile_definition,
                action_policy_definition,
                history_policy_definition,
                read_reuse_policy_definition,
                validation_convergence_policy_definition,
            )

    if exit_code:
        sys.exit(exit_code)


# ---------------------------------------------------------------------------
# thorn chat
# ---------------------------------------------------------------------------

async def _chat_loop(
    *,
    router: ChatPromptRouter,
    scheduler: AgentScheduler,
    session: Session,
    inbox: SessionInbox,
) -> None:
    """Read user input forever and route each turn through the scheduler.

    Returns when the user signals end-of-input (``EOFError``) or when
    a non-Thorn exception escapes ``router.turn`` and reaches the
    caller's ``finally`` block.  ``SkillError`` and ``ThornError`` are
    caught and printed so the REPL stays responsive.

    ``console.input`` is a blocking call; we run it on the default
    executor so the scheduler's drain task and any background event
    listeners get loop time while the user is typing.  Without this,
    the scheduler could not even dispatch the round we just submitted
    until the user hit return for the *next* turn, and the chat REPL
    would no longer be a useful test bed for the scheduler-driven
    pipeline Phase 4 is putting in place.
    """
    loop = asyncio.get_running_loop()
    while True:
        try:
            user_input = await loop.run_in_executor(
                None, lambda: console.input("[green]you>[/green] "),
            )
        except EOFError:
            return
        if not user_input.strip():
            continue

        try:
            await router.turn(
                scheduler=scheduler,
                session=session,
                inbox=inbox,
                prompt_text=user_input,
            )
        except SkillError as exc:
            console.print(f"\n[red]Agent error:[/red] {exc.detail}")
        except ThornError as exc:
            console.print(f"\n[red]Error:[/red] {exc}")
        console.print()


@main.command()
@click.option("-v", "--verbose", count=True, help="Increase output detail (-v, -vv).")
@click.option("-q", "--quiet", is_flag=True, default=False, help="Suppress all output except the final answer.")
@click.option("--trace", "trace_path", type=click.Path(), default=None, help="Write execution trace to a JSONL file.")
@click.option(
    "--trace-raw-prompts",
    is_flag=True,
    default=False,
    help=(
        "Write raw prompt payload sidecars next to --trace. "
        "May expose secrets and sensitive content."
    ),
)
@click.option("--workspace", "workspace_path", type=click.Path(exists=True, file_okay=False), default=None, help="Override workspace root directory.")
@click.option("--resume", "resume_session_key_raw", default=None, help="Resume an existing CLI session key.")
@click.option(
    "--agency",
    "agency_path",
    type=click.Path(file_okay=False),
    default=None,
    help=(
        "Agency home directory (holds agent identities, sessions, "
        "and journals).  Defaults to ~/.thorn/; created if missing."
    ),
)
@click.option(
    "--no-housekeeping",
    is_flag=True,
    default=False,
    help=(
        "Skip the shutdown housekeeping turn that normally runs on "
        "chat exit (the agent's chance to journal anything worth "
        "remembering from this session)."
    ),
)
def chat(
    verbose: int,
    quiet: bool,
    trace_path: str | None,
    trace_raw_prompts: bool,
    workspace_path: str | None,
    resume_session_key_raw: str | None,
    agency_path: str | None,
    no_housekeeping: bool,
) -> None:
    """Start an interactive chat session."""
    from thorn.runtime._paths import AgencyPaths

    verbosity = _resolve_verbosity(verbose, quiet)
    prompt_trace_capture = _resolve_prompt_trace_capture(
        trace_path=trace_path,
        trace_raw_prompts=trace_raw_prompts,
    )
    trace_file = open(trace_path, "w", encoding="utf-8") if trace_path else None
    try:
        startup = _resolve_cli_command_startup(
            agency_path=agency_path,
            workspace_path=workspace_path,
            resume_session_key_raw=resume_session_key_raw,
            agent_id=CLI_AGENT_ID,
        )
        paths = AgencyPaths(
            home_root=startup.agency_home,
            workspace_root=startup.workspace_root,
        )
        runtime = _build_runtime(
            trace_file=trace_file,
            trace_path=trace_path,
            workspace=str(startup.workspace_root),
            paths=paths,
            sandbox_executor_enabled=True,
            subprocess_tool_workspace_root=startup.workspace_root,
            prompt_trace_capture=prompt_trace_capture,
        )
        _warn_if_cli_uses_subprocess_sandbox(runtime)
    except ThornError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        if trace_file:
            trace_file.close()
        sys.exit(1)
    except click.ClickException:
        if trace_file:
            trace_file.close()
        raise

    console.print("[bold]thorn[/bold] interactive chat  (Ctrl+C to exit)\n")

    async def _chat() -> None:
        # Phase 4 of the CLI/gateway unification: route the chat REPL
        # through the in-process ``AgentScheduler`` rather than calling
        # ``session.prompt`` directly.  Each user input is posted as a
        # notification on the session's inbox; the scheduler invokes
        # ``ChatPromptRouter.dispatcher`` which calls ``session.prompt``
        # and resolves a per-turn future the REPL awaits.
        #
        # Each invocation gets a fresh session key unless the caller
        # explicitly asks to resume a persisted local CLI session.
        async with runtime:
            agent = _ensure_cli_agent(runtime)
            if agent.id is None:
                raise RuntimeError(
                    "CLI agent has no id; cannot build a session inbox"
                )

            session_key = (
                startup.resume_session_key
                or _generate_cli_session_key(runtime.workspace_root)
            )
            # Pick the logical agent-workspace upper bound for this
            # CLI session.  See the matching comment on the ``thorn
            # run`` path for the rationale.
            logical_agent_workspace_path = (
                pick_logical_agent_workspace_path_for_cli_session(
                    runtime.workspace_root,
                )
            )
            session = runtime.get_or_create_session(
                agent, session_key,
                workspace_root=runtime.workspace_root,
                logical_agent_workspace_path=(
                    logical_agent_workspace_path
                ),
            )

            # Subscribe a per-session console listener (Phase 3
            # bus pattern); the filter scopes output to events
            # tagged with this session key, so any background
            # session that lands later under a shared local-agency
            # runtime won't bleed into the REPL output.
            bus = _runtime_event_bus(runtime)
            console_listener: EventSink = ConsoleEventSink(
                verbosity=verbosity,
            )
            with bus.subscribe(
                console_listener,
                scope_filter=in_session(session_key),
            ):
                # Per-session inbox + scheduler wiring.  The
                # inbox lives under the agent's data root so a
                # crash leaves the in-flight notification on
                # disk for the startup sweep to reconcile -- a
                # property that becomes load-bearing once Phase
                # 6 introduces the local-agency daemon.
                session_address = SessionAddress(
                    agent.id, session_key,
                )
                inbox = SessionInbox(
                    runtime.paths.session_inbox_dir(
                        agent.id, session_key,
                    ),
                    session_address,
                    in_flight_index=runtime.in_flight_index,
                )
                runtime.address_book.register(
                    session_address, inbox,
                )

                router = ChatPromptRouter(
                    target=session_address,
                    extra_system=CHAT_SYSTEM_PROMPT,
                )

                async def _save_session_async(
                    sess: Session,
                ) -> None:
                    """Async adapter around the sync save.

                    Mirrors the gateway's pattern (named so
                    scheduler exception logs point at a real
                    qualified name).  The save is small;
                    running it on the loop thread is fine
                    because the scheduler already serialises
                    per-session, so there is no contention to
                    offload.
                    """
                    runtime.save_session(sess)

                scheduler = AgentScheduler(
                    agent=agent,
                    prompt_dispatcher=router.dispatcher,
                    save_session=_save_session_async,
                )

                try:
                    lock_context = (
                        session_lock(
                            runtime.paths.session_metadata_dir(
                                agent.id, session_key,
                            )
                        )
                        if startup.resume_session_key is not None
                        else nullcontext()
                    )
                    try:
                        with lock_context:
                            await _chat_loop(
                                router=router,
                                scheduler=scheduler,
                                session=session,
                                inbox=inbox,
                            )
                            # Shutdown housekeeping (Phase 5): gives the
                            # agent one last turn to journal anything
                            # worth remembering before the session is
                            # discarded.  Bypasses the scheduler/router
                            # -- this is a framework-driven action, not a
                            # user turn, and the scheduler's drain task
                            # is about to be torn down.  ``--no-
                            # housekeeping`` skips the turn entirely for
                            # scripted / test use.  We run housekeeping
                            # *only after a clean REPL exit*; if
                            # ``_chat_loop`` raised (``KeyboardInterrupt``,
                            # unexpected error) we skip, because the
                            # agent is unlikely to produce a useful final
                            # turn mid-panic and the user wants their
                            # shell prompt back.
                            if not no_housekeeping:
                                from thorn.core._housekeeping import (
                                    perform_shutdown_housekeeping,
                                )
                                try:
                                    await perform_shutdown_housekeeping(session)
                                    runtime.save_session(session)
                                except ThornError as exc:
                                    # Shutdown housekeeping swallows
                                    # Thorn-level errors internally, but
                                    # save_session could still raise
                                    # (disk full, etc.); log and move on
                                    # rather than block exit.
                                    console.print(
                                        f"[yellow]Shutdown housekeeping "
                                        f"failed to save:[/yellow] {exc}"
                                    )
                    except SessionLockError as exc:
                        _raise_locked_session(
                            session_key=session_key,
                            error=exc,
                        )
                finally:
                    # Bounded grace period mirrors ``thorn
                    # run``: in the success case the driver is
                    # parked on its idle wait, so shutdown is
                    # essentially instantaneous.
                    await scheduler.shutdown(timeout=5.0)

    try:
        asyncio.run(_chat())
    except KeyboardInterrupt:
        console.print("\n[dim]Goodbye.[/dim]")
    except ThornError as exc:
        console.print(f"\n[red]Error:[/red] {exc}")
        sys.exit(1)
    finally:
        if trace_file:
            trace_file.close()


# ---------------------------------------------------------------------------
# thorn serve (group: default=gateway, mcp=MCP server)
# ---------------------------------------------------------------------------

@main.group(invoke_without_command=True)
@click.option("-v", "--verbose", count=True, help="Increase output detail (-v, -vv).")
@click.option("-q", "--quiet", is_flag=True, default=False, help="Suppress all output except errors.")
@click.option("--trace", "trace_path", type=click.Path(), default=None, help="Write execution trace to a JSONL file.")
@click.option(
    "--trace-raw-prompts",
    is_flag=True,
    default=False,
    help=(
        "Write raw prompt payload sidecars next to --trace. "
        "May expose secrets and sensitive content."
    ),
)
@click.option(
    "--agency",
    "agency_path",
    type=click.Path(file_okay=False),
    default=None,
    help=(
        "Agency home directory (contains agency config and the agents/ tree). "
        "Defaults to ~/.thorn."
    ),
)
@click.option(
    "--workspace",
    "workspace_path",
    type=click.Path(file_okay=False),
    default=None,
    help=(
        "Override the agency workspace directory.  Takes precedence over "
        "the 'workspace' field in the agency config."
    ),
)
@click.pass_context
def serve(
    ctx: click.Context,
    verbose: int,
    quiet: bool,
    trace_path: str | None,
    trace_raw_prompts: bool,
    agency_path: str | None,
    workspace_path: str | None,
) -> None:
    """Run an agency in gateway mode (or expose Thorn tools over MCP)."""
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    ctx.obj["quiet"] = quiet
    ctx.obj["trace_path"] = trace_path
    ctx.obj["trace_raw_prompts"] = trace_raw_prompts
    ctx.obj["agency_path"] = agency_path
    ctx.obj["workspace_path"] = workspace_path

    if ctx.invoked_subcommand is not None:
        return

    _serve_gateway(
        verbose=verbose,
        quiet=quiet,
        trace_path=trace_path,
        trace_raw_prompts=trace_raw_prompts,
        agency_path=agency_path,
        workspace_path=workspace_path,
    )


def _resolve_agency_home(agency_path: str | None) -> Path:
    """Resolve the agency home directory for ``thorn serve``.

    Uses *agency_path* when provided; otherwise falls back to
    ``~/.thorn`` (the local-agency convention from the architecture
    doc).  Errors out with a clear message if the directory does not
    exist.
    """
    if agency_path is not None:
        agency_home = Path(agency_path).expanduser().resolve()
    else:
        agency_home = (Path.home() / ".thorn").resolve()

    if not agency_home.is_dir():
        console.print(
            f"[red]Error:[/red] Agency home directory does not exist: {agency_home}\n"
            "Pass --agency <dir> to point at an existing agency home, or run "
            "'thorn serve bootstrap --agency-home <dir> --agency-workspace <dir> ...' "
            "to create a new one."
        )
        sys.exit(1)

    return agency_home


def _resolve_serve_workspace(
    *,
    cli_workspace: str | None,
    config_workspace: Path | None,
    agency_home: Path,
) -> Path:
    """Pick the workspace root for ``thorn serve``.

    CLI ``--workspace`` wins over the value from ``gateway.json``;
    if neither is set, this prints an error and exits.
    """
    if cli_workspace is not None:
        return Path(cli_workspace).expanduser().resolve()

    if config_workspace is not None:
        return config_workspace

    console.print(
        "[red]Error:[/red] No workspace directory configured.\n"
        f"Either pass --workspace <dir> to 'thorn serve', or set the "
        f"'workspace' field in the agency config under {agency_home} (re-run "
        "'thorn serve bootstrap' with --agency-workspace to do that)."
    )
    sys.exit(1)


def _serve_gateway(
    *,
    verbose: int,
    quiet: bool,
    trace_path: str | None,
    trace_raw_prompts: bool,
    agency_path: str | None,
    workspace_path: str | None,
) -> None:
    """Run the gateway daemon (called when ``thorn serve`` has no subcommand).

    Loads ``gateway.json`` from the agency home directory (defaulting
    to ``~/.thorn``) and resolves the workspace root from the CLI
    override or the ``workspace`` field in the config.
    """
    import logging

    from rich.logging import RichHandler

    from thorn.gateway import (
        Gateway,
        infer_event_sources,
        instantiate_services,
        load_gateway_config,
    )

    verbosity = _resolve_verbosity(verbose, quiet)
    prompt_trace_capture = _resolve_prompt_trace_capture(
        trace_path=trace_path,
        trace_raw_prompts=trace_raw_prompts,
    )

    log_level = logging.DEBUG if verbose >= 2 else (logging.WARNING if quiet else logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True)],
    )

    from thorn.runtime._paths import AgencyPaths

    agency_home = _resolve_agency_home(agency_path)

    try:
        gateway_config = load_gateway_config(agency_home)
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"[red]Error:[/red] {exc}")
        sys.exit(1)

    ws_root = _resolve_serve_workspace(
        cli_workspace=workspace_path,
        config_workspace=gateway_config.resolve_workspace(agency_home),
        agency_home=agency_home,
    )

    try:
        all_services = instantiate_services(gateway_config)
    except (KeyError, ValueError) as exc:
        console.print(f"[red]Error:[/red] {exc}")
        sys.exit(1)

    paths = AgencyPaths.for_gateway(
        agency_dir=agency_home,
        workspace_dir=ws_root,
    )

    trace_file = open(trace_path, "w", encoding="utf-8") if trace_path else None
    try:
        runtime = _build_runtime(
            trace_file=trace_file, trace_path=trace_path, workspace=str(ws_root),
            paths=paths,
            sandbox_executor_enabled=True,
            sandbox_config=gateway_config.sandbox,
            llm_config=gateway_config.llm,
            prompt_trace_capture=prompt_trace_capture,
        )
    except ThornError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        if trace_file:
            trace_file.close()
        sys.exit(1)

    # Daemon-mode console listener: no scope filter, since the gateway
    # operator wants to see every session's events (this is the
    # operator log).  Per-session filtering would obscure exactly the
    # cross-cutting view ``-vvv`` is asked for.
    _runtime_event_bus(runtime).subscribe(
        ConsoleEventSink(verbosity=verbosity),
    )

    for service in all_services:
        runtime.register_service(service)

    agents = [
        runtime.sessions.load_agent(aid)
        for aid in runtime.sessions.list_agent_ids()
    ]

    # Service-driven account validation: replace each loaded
    # agent's parse-time UntypedAccountConfig entries with the typed
    # AccountConfig subclass declared by the corresponding service.
    # Doing this before `infer_event_sources` (and broker
    # registration, which happens later inside `gateway.run`) means
    # both code paths see typed accounts and per-service fields
    # like git_user_email survive validation.  Misconfigured
    # accounts (referencing an unknown service) surface here with
    # a clear error.
    from thorn.core._account import validate_agent_accounts

    try:
        for agent in agents:
            validate_agent_accounts(agent, runtime.get_service)
    except ValueError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        sys.exit(1)

    sources = infer_event_sources(gateway_config, agents)

    if not sources:
        console.print(
            "[yellow]Warning:[/yellow] No event sources could be inferred "
            f"from {agency_home / 'gateway.json'} and agent accounts. "
            "The gateway will start but will not receive any events."
        )

    gateway = Gateway(
        runtime=runtime,
        sources=sources,
        gateway_config=gateway_config,
    )

    service_names = [s.name for s in all_services]
    console.print(
        f"[bold]thorn serve[/bold]  services: {', '.join(service_names) or '(none)'}"
    )

    try:
        asyncio.run(gateway.run())
    except KeyboardInterrupt:
        console.print("\n[dim]Gateway stopped.[/dim]")
    except ThornError as exc:
        console.print(f"\n[red]Error:[/red] {exc}")
        sys.exit(1)
    finally:
        if trace_file:
            trace_file.close()


def _load_gateway_config_with_raw_agency_config(
    agency_home: Path,
) -> tuple["GatewayConfig", dict[str, Any], "AgencyConfigFile"]:
    from thorn.gateway._config import (
        GatewayConfig,
        _discover_agency_config_file,
        _parse_agency_config_file,
    )

    config_file = _discover_agency_config_file(agency_home)
    raw = _parse_agency_config_file(config_file)
    return GatewayConfig.model_validate(raw), raw, config_file


def _write_raw_agency_config(
    config_file: "AgencyConfigFile",
    raw_config: dict[str, Any],
) -> None:
    from thorn.gateway._config import AgencyConfigFileFormat

    if config_file.configuration_format is AgencyConfigFileFormat.JSON:
        serialized = json.dumps(raw_config, indent=2, ensure_ascii=False) + "\n"
    elif config_file.configuration_format is AgencyConfigFileFormat.YAML:
        import yaml

        serialized = yaml.safe_dump(
            raw_config,
            sort_keys=False,
            allow_unicode=True,
        )
    else:
        raise ValueError(
            "Unsupported agency config format: "
            f"{config_file.configuration_format.value}"
        )
    config_file.path.write_text(serialized, encoding="utf-8")


def _load_agents_for_peer_resolution(
    *,
    agency_home: Path,
    workspace_root: Path,
    selected_agent_id: AgentID | None,
) -> list["Agent"]:
    from thorn.runtime._paths import AgencyPaths
    from thorn.runtime._store import SessionStore

    paths = AgencyPaths.for_gateway(
        agency_dir=agency_home,
        workspace_dir=workspace_root,
    )
    store = SessionStore(paths)
    agents: list["Agent"] = []
    for agent_id in store.list_agent_ids():
        if selected_agent_id is not None and agent_id != selected_agent_id:
            continue
        agents.append(store.load_agent(agent_id))
    return agents


def _build_peer_resolution_clients(
    *,
    services_by_name: dict[str, Any],
    agents: list["Agent"],
) -> tuple[dict[str, Any], list[_PeerResolutionClientError]]:
    from thorn.core._account import resolve_account, validate_agent_accounts
    from thorn.tools.forge import ForgeHostService

    errors: list[_PeerResolutionClientError] = []
    clients_by_service: dict[str, Any] = {}

    for agent in agents:
        try:
            validate_agent_accounts(agent, services_by_name.__getitem__)
        except ValueError as exc:
            errors.append(_PeerResolutionClientError(
                service_name="(agent accounts)",
                reason=str(exc),
            ))
    if errors:
        return {}, errors

    for service_name, service in services_by_name.items():
        if not isinstance(service, ForgeHostService):
            continue
        service_errors: list[_PeerResolutionClientError] = []
        for agent in agents:
            try:
                account = resolve_account(agent, service_name)
            except KeyError:
                continue
            except TypeError as exc:
                service_errors.append(_PeerResolutionClientError(
                    service_name=service_name,
                    reason=str(exc),
                ))
                continue
            try:
                clients_by_service[service_name] = (
                    service.authenticated_client(account)
                )
                break
            except Exception as exc:
                service_errors.append(_PeerResolutionClientError(
                    service_name=service_name,
                    reason=str(exc),
                ))
        if service_name not in clients_by_service:
            errors.extend(service_errors)

    return clients_by_service, errors


def _serve_resolve_peers(
    *,
    agency_path: str | None,
    agent_id_raw: str | None,
    dry_run: bool,
) -> int:
    from thorn.gateway._config import (
        PeerAccountIDPolicy,
        _resolve_forges_and_projects,
        collect_handle_only_peer_account_problems,
        instantiate_services,
    )
    from thorn.gateway._peer_resolution import (
        apply_peer_account_resolutions,
        resolve_peer_account_handles,
    )

    agency_home = _resolve_agency_home(agency_path)
    try:
        gateway_config, raw_config, config_file = (
            _load_gateway_config_with_raw_agency_config(agency_home)
        )
    except (OSError, ValueError) as exc:
        console.print(f"[red]Error:[/red] {exc}")
        return 1

    try:
        services = instantiate_services(
            gateway_config,
            peer_account_id_policy=PeerAccountIDPolicy.ALLOW_HANDLE_ONLY,
        )
        forge_specs, _resolved_projects = _resolve_forges_and_projects(
            gateway_config,
            peer_account_id_policy=PeerAccountIDPolicy.ALLOW_HANDLE_ONLY,
        )
    except (KeyError, ValueError) as exc:
        console.print(f"[red]Error:[/red] {exc}")
        return 1

    forge_types_by_service = {forge.name: forge.type for forge in forge_specs}
    handle_only_accounts = collect_handle_only_peer_account_problems(
        list(gateway_config.peers),
        forge_types_by_service,
    )
    if not handle_only_accounts:
        console.print("[green]OK[/green] No handle-only peer accounts found.")
        return 0

    selected_agent_id = AgentID(agent_id_raw) if agent_id_raw else None
    workspace_root = gateway_config.resolve_workspace(agency_home) or agency_home
    agents = _load_agents_for_peer_resolution(
        agency_home=agency_home,
        workspace_root=workspace_root,
        selected_agent_id=selected_agent_id,
    )
    if selected_agent_id is not None and not agents:
        console.print(
            f"[red]Error:[/red] No persisted agent {selected_agent_id!s} "
            f"found under {agency_home}."
        )
        return 1

    services_by_name = {service.name: service for service in services}
    clients_by_service, client_errors = _build_peer_resolution_clients(
        services_by_name=services_by_name,
        agents=agents,
    )
    if client_errors:
        console.print("[red]Error:[/red] Could not prepare forge clients:")
        for error in client_errors:
            console.print(f"  {error.service_name}: {error.reason}")
        return 1

    result = resolve_peer_account_handles(
        gateway_config,
        forge_clients_by_service=clients_by_service,
        forge_types_by_service=forge_types_by_service,
    )
    if result.errors:
        console.print("[red]Error:[/red] Some peer handles could not be resolved:")
        for error in result.errors:
            loc = error.location
            console.print(
                f"  peer={loc.peer_id} service={loc.service_name} "
                f"handle={loc.original_account_id}: {error.reason}"
            )
        return 1

    updated = apply_peer_account_resolutions(
        raw_config,
        result.resolutions,
    )
    for resolution in result.resolutions:
        loc = resolution.location
        console.print(
            f"Resolved peer={loc.peer_id} service={loc.service_name} "
            f"{loc.original_account_id} -> {resolution.immutable_account_id}"
        )

    if dry_run:
        console.print(
            f"[yellow]Dry run:[/yellow] {config_file.path} was not modified."
        )
        return 0

    _write_raw_agency_config(config_file, updated)
    console.print(f"[green]Updated:[/green] {config_file.path}")
    return 0


@serve.command("resolve-peers")
@click.option(
    "--agent",
    "agent_id_raw",
    default=None,
    help="Use credentials from one persisted agent ID.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Resolve and report changes without writing the agency config.",
)
@click.pass_context
def serve_resolve_peers(
    ctx: click.Context,
    agent_id_raw: str | None,
    dry_run: bool,
) -> None:
    """Rewrite peer account handles in the agency config to immutable IDs."""
    agency_path = ctx.obj.get("agency_path")
    exit_code = _serve_resolve_peers(
        agency_path=agency_path,
        agent_id_raw=agent_id_raw,
        dry_run=dry_run,
    )
    sys.exit(exit_code)


@serve.command("preflight")
@click.option(
    "--agent",
    "agent_id_raw",
    default=None,
    help="Restrict the probe to one persisted agent ID.",
)
@click.option(
    "--project",
    "project_name",
    default=None,
    help="Restrict the probe to one gateway.json project name.",
)
@click.option(
    "--fork",
    "fork_name",
    default=None,
    help="Restrict the probe to one fork/remote name within the project.",
)
@click.option(
    "--write-check",
    is_flag=True,
    default=False,
    help=(
        "Also push and delete a temporary branch through the broker. "
        "The default is read-only git ls-remote."
    ),
)
@click.option(
    "--timeout",
    "timeout_s",
    type=click.IntRange(min=1),
    default=60,
    show_default=True,
    help=(
        "Per-readiness-operation timeout in seconds. Git probes use this "
        "inside the sandbox; LLM probes use it as a wall-clock timeout."
    ),
)
@click.pass_context
def serve_preflight(
    ctx: click.Context,
    agent_id_raw: str | None,
    project_name: str | None,
    fork_name: str | None,
    write_check: bool,
    timeout_s: int,
) -> None:
    """Run first-run gateway readiness checks before live polling.

    This validates event-source inference, forge API access, and the
    effective LLM model configuration, then starts the configured sandbox
    and broker path and invokes git from inside the sandbox. It does not
    start event sources or touch forge TODO/notification state.
    """
    verbose = ctx.obj.get("verbose", 0)
    quiet = ctx.obj.get("quiet", False)
    trace_path = ctx.obj.get("trace_path")
    trace_raw_prompts = ctx.obj.get("trace_raw_prompts", False)
    agency_path = ctx.obj.get("agency_path")
    workspace_path = ctx.obj.get("workspace_path")

    exit_code = _serve_preflight(
        verbose=verbose,
        quiet=quiet,
        trace_path=trace_path,
        trace_raw_prompts=trace_raw_prompts,
        agency_path=agency_path,
        workspace_path=workspace_path,
        agent_id_raw=agent_id_raw,
        project_name=project_name,
        fork_name=fork_name,
        write_check=write_check,
        timeout_s=timeout_s,
    )
    sys.exit(exit_code)


def _serve_preflight(
    *,
    verbose: int,
    quiet: bool,
    trace_path: str | None,
    trace_raw_prompts: bool,
    agency_path: str | None,
    workspace_path: str | None,
    agent_id_raw: str | None,
    project_name: str | None,
    fork_name: str | None,
    write_check: bool,
    timeout_s: int,
) -> int:
    from thorn.core._account import validate_agent_accounts
    from thorn.gateway import (
        Gateway,
        infer_event_sources,
        instantiate_services,
        load_gateway_config,
    )
    from thorn.gateway._preflight import (
        collect_event_source_preflight_problems,
        collect_forge_api_preflight_targets,
        collect_git_preflight_targets,
    )
    from thorn.runtime._paths import AgencyPaths

    verbosity = _resolve_verbosity(verbose, quiet)
    prompt_trace_capture = _resolve_prompt_trace_capture(
        trace_path=trace_path,
        trace_raw_prompts=trace_raw_prompts,
    )
    agency_home = _resolve_agency_home(agency_path)
    try:
        gateway_config = load_gateway_config(agency_home)
        all_services = instantiate_services(gateway_config)
    except (FileNotFoundError, KeyError, ValueError) as exc:
        console.print(f"[red]Error:[/red] {exc}")
        return 1

    ws_root = _resolve_serve_workspace(
        cli_workspace=workspace_path,
        config_workspace=gateway_config.resolve_workspace(agency_home),
        agency_home=agency_home,
    )
    targets = collect_git_preflight_targets(
        gateway_config,
        project_filter=project_name,
        fork_filter=fork_name,
    )
    if not targets:
        console.print(
            "[red]Error:[/red] No configured project clone URLs matched "
            "the requested preflight filters."
        )
        return 1

    paths = AgencyPaths.for_gateway(
        agency_dir=agency_home,
        workspace_dir=ws_root,
    )
    trace_file = open(trace_path, "w", encoding="utf-8") if trace_path else None
    try:
        runtime = _build_runtime(
            trace_file=trace_file,
            trace_path=trace_path,
            workspace=str(ws_root),
            paths=paths,
            sandbox_executor_enabled=True,
            sandbox_config=gateway_config.sandbox,
            llm_config=gateway_config.llm,
            prompt_trace_capture=prompt_trace_capture,
        )
    except ThornError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        if trace_file:
            trace_file.close()
        return 1

    _runtime_event_bus(runtime).subscribe(
        ConsoleEventSink(verbosity=verbosity),
    )
    for service in all_services:
        runtime.register_service(service)

    selected_agent_id = AgentID(agent_id_raw) if agent_id_raw else None
    agents = []
    for agent_id in runtime.sessions.list_agent_ids():
        if selected_agent_id is not None and agent_id != selected_agent_id:
            continue
        agent = runtime.get_or_create_agent(agent_id)
        try:
            validate_agent_accounts(agent, runtime.get_service)
        except ValueError as exc:
            console.print(f"[red]Error:[/red] {exc}")
            if trace_file:
                trace_file.close()
            return 1
        agents.append(agent)
    if not agents:
        console.print("[red]Error:[/red] No persisted agents matched preflight.")
        if trace_file:
            trace_file.close()
        return 1

    source_problems = collect_event_source_preflight_problems(
        gateway_config,
        agents,
        project_filter=project_name,
        fork_filter=fork_name,
    )
    if source_problems:
        console.print("[red]FAILED[/red] event-source readiness")
        for problem in source_problems:
            console.print(
                f"  agent={problem.agent_id} forge={problem.forge_name}: "
                f"{problem.reason}",
            )
        if trace_file:
            trace_file.close()
        return 1

    try:
        inferred_sources = infer_event_sources(gateway_config, agents)
    except Exception as exc:
        console.print(
            "[red]FAILED[/red] event-source inference: "
            f"{_preflight_redacted_error(exc)}",
        )
        if trace_file:
            trace_file.close()
        return 1

    if not inferred_sources:
        console.print(
            "[red]FAILED[/red] No event sources could be inferred from "
            f"{agency_home / 'gateway.json'} and the selected agent accounts.",
        )
        if trace_file:
            trace_file.close()
        return 1

    console.print("[bold]preflight[/bold] event sources")
    for source in inferred_sources:
        source_name = source.name or type(source).__name__
        console.print(f"  [green]OK[/green] {source_name}")

    api_targets = collect_forge_api_preflight_targets(
        gateway_config,
        agents,
        project_filter=project_name,
        fork_filter=fork_name,
    )
    if _preflight_forge_api_targets(runtime=runtime, targets=api_targets):
        if trace_file:
            trace_file.close()
        return 1

    async def _run() -> int:
        async with runtime:
            failures = await _preflight_agent_llm_targets(
                agents=agents,
                runtime=runtime,
                timeout_s=timeout_s,
            )
            if failures:
                return 1

            gateway = Gateway(
                runtime=runtime,
                sources=[],
                gateway_config=gateway_config,
            )
            try:
                for agent in agents:
                    gateway._ensure_scheduler_for_agent(agent)
                gateway._warn_if_planned_egress_allowlist_configured()
                await gateway._maybe_start_bundled_broker()
                await gateway._register_broker_bindings()
                runtime.set_sandbox_broker_binding_lookup(
                    gateway.broker_binding_for,
                )
                for agent in agents:
                    failures += await _preflight_agent_git_targets(
                        agent=agent,
                        runtime=runtime,
                        targets=targets,
                        write_check=write_check,
                        timeout_s=timeout_s,
                    )
            finally:
                await gateway.shutdown()
        return 1 if failures else 0

    try:
        return asyncio.run(_run())
    except ThornError as exc:
        console.print(f"\n[red]Error:[/red] {exc}")
        return 1
    except Exception as exc:
        console.print(f"\n[red]Error:[/red] {exc}")
        return 1
    finally:
        if trace_file:
            trace_file.close()


def _preflight_forge_api_targets(
    *,
    runtime: Runtime,
    targets: list["ForgeAPIPreflightTarget"],
) -> int:
    from thorn.tools.forge import ForgeHostService

    failures = 0
    for target in targets:
        console.print(
            f"[bold]preflight[/bold] agent={target.agent_id} "
            f"project={target.project_name} fork={target.fork_name} "
            f"mode=forge-api",
        )
        try:
            service = runtime.get_service(target.forge_name)
            if not isinstance(service, ForgeHostService):
                raise TypeError(
                    f"service {target.forge_name!r} is not a forge host service"
            )
            client = service.authenticated_client(target.account)
            client.get_project_info(target.native_project_id)
            credential_scope_warnings = _preflight_credential_scope_warnings(
                client,
            )
        except Exception as exc:
            failures += 1
            console.print(f"  [red]FAILED[/red] {_preflight_redacted_error(exc)}")
            hint = _forge_api_preflight_hint(exc)
            if hint is not None:
                console.print(f"  [yellow]Hint:[/yellow] {hint}")
            continue
        console.print("  [green]OK[/green]")
        for warning in credential_scope_warnings:
            console.print(f"  [yellow]Warning:[/yellow] {warning.summary}")
            if warning.detail:
                console.print(f"  {warning.detail}")
    return failures


def _preflight_credential_scope_warnings(
    client: object,
) -> tuple["CredentialScopeWarning", ...]:
    from thorn.tools._credential_scopes import CredentialScopeWarning

    inspect_scopes = getattr(client, "inspect_credential_scopes", None)
    if inspect_scopes is None:
        return ()
    try:
        inspection = inspect_scopes()
    except Exception:
        return ()
    warnings = getattr(inspection, "warnings", ())
    return tuple(
        warning for warning in warnings
        if isinstance(warning, CredentialScopeWarning)
    )


def _preflight_redacted_error(exc: Exception) -> str:
    from thorn.gateway._preflight import redact_git_preflight_output

    return redact_git_preflight_output(str(exc))


def _forge_api_preflight_hint(exc: Exception) -> str | None:
    from thorn.core._credentials import CredentialMissingError

    if isinstance(exc, CredentialMissingError):
        return (
            "Export the referenced credential env var before starting "
            "`thorn serve`."
        )
    if isinstance(exc, ImportError):
        return (
            "Run `uv sync --all-extras` so the GitHub/GitLab API "
            "dependencies are installed."
        )
    return None


async def _preflight_agent_llm_targets(
    *,
    agents: list["Agent"],
    runtime: Runtime,
    timeout_s: int,
) -> int:
    from thorn.gateway._preflight import (
        LLMPreflightTarget,
        probe_llm_preflight_target,
    )

    failures = 0
    provider_results: dict[int, str | None] = {}
    for agent in agents:
        if agent.id is None:
            console.print("[red]FAILED[/red] agent without an ID cannot preflight")
            failures += 1
            continue

        console.print(
            f"[bold]preflight[/bold] agent={agent.id} mode=llm",
        )
        try:
            provider = runtime.provider_for_agent(agent)
        except Exception as exc:
            failures += 1
            console.print(f"  [red]FAILED[/red] {_preflight_redacted_error(exc)}")
            continue

        provider_identity = id(provider)
        if provider_identity in provider_results:
            previous_error = provider_results[provider_identity]
            if previous_error is None:
                console.print(
                    "  [green]OK[/green] shared provider already checked"
                )
                continue

            failures += 1
            console.print(f"  [red]FAILED[/red] {previous_error}")
            console.print(
                "  [yellow]Note:[/yellow] shared provider already checked"
            )
            continue

        try:
            await probe_llm_preflight_target(
                LLMPreflightTarget(
                    agent_id=agent.id,
                    provider=provider,
                ),
                timeout_s=timeout_s,
            )
        except Exception as exc:
            failures += 1
            redacted_error = _preflight_redacted_error(exc)
            provider_results[provider_identity] = redacted_error
            console.print(f"  [red]FAILED[/red] {redacted_error}")
            continue

        provider_results[provider_identity] = None
        console.print("  [green]OK[/green]")

    return failures


async def _preflight_agent_git_targets(
    *,
    agent: "Agent",
    runtime: Runtime,
    targets: list["GitPreflightTarget"],
    write_check: bool,
    timeout_s: int,
) -> int:
    from thorn.core import ToolInvocation
    from thorn.gateway._preflight import (
        build_git_preflight_command,
        git_preflight_failure_hint,
        redact_git_preflight_output,
    )

    if agent.id is None:
        console.print("[red]FAILED[/red] agent without an ID cannot preflight")
        return 1
    executor = runtime.get_or_create_sandbox_executor(agent)
    if executor is None:
        console.print(
            f"[red]FAILED[/red] agent {agent.id}: sandbox executor disabled"
        )
        return 1
    await executor.start()
    failures = 0
    for target in targets:
        branch_name = None
        if write_check:
            branch_name = f"thorn-preflight/{uuid.uuid4().hex[:12]}"
        console.print(
            f"[bold]preflight[/bold] agent={agent.id} "
            f"project={target.project_name} fork={target.fork_name} "
            f"mode={'push/delete' if write_check else 'ls-remote'}"
        )
        command = build_git_preflight_command(
            clone_url=target.clone_url,
            timeout_s=timeout_s,
            write_check_branch=branch_name,
        )
        result = await executor.invoke(
            ToolInvocation(
                call_id=f"git-preflight-{uuid.uuid4().hex}",
                tool_name="run_shell",
                arguments={
                    "command": command,
                    "timeout": float(timeout_s + 30),
                },
            )
        )
        output = redact_git_preflight_output(result.content)
        failed = (
            result.is_error
            or output.startswith("[exit code")
            or output.startswith("[timed out")
        )
        if not failed:
            console.print("  [green]OK[/green]")
            continue
        failures += 1
        console.print("  [red]FAILED[/red]")
        if output:
            console.print(output)
        hint = git_preflight_failure_hint(output)
        if hint is not None:
            console.print(f"  [yellow]Hint:[/yellow] {hint}")
    return failures


@serve.command("bootstrap")
@click.option("--agent-id", required=True, help="Unique identifier for the coordinator agent.")
@click.option("--project-name", required=True, help="Human-readable project name.")
@click.option(
    "--project-url",
    required=True,
    help=(
        "Human-facing project URL on its forge "
        "(e.g. https://github.com/owner/repo or "
        "https://gitlab.com/group/project).  The forge type, name, "
        "API URL, and the per-fork native ID and clone URL are all "
        "derived from this URL."
    ),
)
@click.option(
    "--token-env",
    default=None,
    help=(
        "Env var holding the access token "
        "(default: GITHUB_TOKEN for github.com, GITLAB_TOKEN otherwise)."
    ),
)
@click.option("--git-user-name", default=None, help="Git author/committer name for this agent (default: agent-id).")
@click.option("--git-user-email", default=None, help="Git author/committer email for this agent (default: <agent-id>@thorn).")
@click.option(
    "--agent-class",
    type=click.Choice(["ProjectCoordinator", "LeanProjectCoordinator"]),
    default="ProjectCoordinator",
    show_default=True,
    help=(
        "Agent implementation to persist in agent.json. "
        "LeanProjectCoordinator exposes fewer tools and a smaller "
        "policy surface."
    ),
)
@click.option(
    "--llm-api-url",
    default=None,
    help="Base URL for the OpenAI-compatible LLM provider API.",
)
@click.option(
    "--llm-model",
    default=None,
    help="Default LLM model name for the agency.",
)
@click.option(
    "--llm-api-key-env",
    default=None,
    help="Env var holding the LLM provider API key.",
)
@click.option(
    "--agency-home",
    "agency_home_path",
    type=click.Path(file_okay=False),
    required=True,
    help=(
        "Agency home directory: holds the agency configuration and "
        "agents/ tree. Created if missing. No .thorn/ subdirectory is "
        "appended."
    ),
)
@click.option(
    "--agency-workspace",
    "agency_workspace_path",
    type=click.Path(file_okay=False),
    required=True,
    help=(
        "Agency workspace directory: where agent sessions do their work.  "
        "Recorded in the agency configuration so 'thorn serve' can locate it.  "
        "Created if missing."
    ),
)
@click.pass_context
def serve_bootstrap(
    ctx: click.Context,
    agent_id: str,
    project_name: str,
    project_url: str,
    token_env: str | None,
    git_user_name: str | None,
    git_user_email: str | None,
    agent_class: str,
    llm_api_url: str | None,
    llm_model: str | None,
    llm_api_key_env: str | None,
    agency_home_path: str,
    agency_workspace_path: str,
) -> None:
    """Create a project-oriented agent in an agency home directory."""
    from pathlib import Path

    from thorn.gateway._bootstrap import bootstrap_coordinator
    from thorn.gateway._config import (
        GATEWAY_CONFIG_FILENAME,
        load_gateway_config,
    )
    from thorn.runtime import AgencyPaths

    agency_home = Path(agency_home_path).expanduser().resolve()
    agency_workspace = Path(agency_workspace_path).expanduser().resolve()
    llm_args = [llm_api_url, llm_model, llm_api_key_env]
    if any(value is not None for value in llm_args) and not all(llm_args):
        console.print(
            "[red]Error:[/red] --llm-api-url, --llm-model, and "
            "--llm-api-key-env must be provided together."
        )
        sys.exit(1)
    llm_config = None
    if all(llm_args):
        assert llm_api_url is not None
        assert llm_model is not None
        assert llm_api_key_env is not None
        llm_config = LLMConfig(
            provider=OpenAIProviderSettings(
                type=LLMProviderType.OPENAI,
                api_url=llm_api_url,
                api_key_env_var=llm_api_key_env,
            ),
            model=LLMModelConfig(name=llm_model),
        )
    try:
        persisted_agent_id = bootstrap_coordinator(
            agency_home=agency_home,
            agency_workspace=agency_workspace,
            agent_id=agent_id,
            project_name=project_name,
            project_url=project_url,
            access_token_env=token_env,
            git_user_name=git_user_name or "",
            git_user_email=git_user_email or "",
            llm_config=llm_config,
            agent_class=agent_class,
        )
        agency_config = load_gateway_config(agency_home)
    except ValueError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        sys.exit(1)

    paths = AgencyPaths.for_gateway(agency_home, agency_workspace)
    agency_config_path = agency_home / GATEWAY_CONFIG_FILENAME
    console.print(f"[green]Bootstrapped agent:[/green] {persisted_agent_id}")
    console.print(f"  Identity: {paths.agent_identity_file(persisted_agent_id)}")
    console.print(f"  Agent home: {paths.agent_home_mount(persisted_agent_id)}")
    console.print(f"  Agency config: {agency_config_path}")
    console.print(
        f"  Agent workspace: {paths.agent_workspace_mount(persisted_agent_id)}"
    )
    if not agency_config.peers:
        console.print(
            "\n[yellow]No trusted peers configured.[/yellow] Gateway mode will "
            "deny conversational forge instructions until you add a peer to "
            f"{agency_config_path}."
        )
    if token_env is None:
        token_env = (
            "GITHUB_TOKEN" if "github.com" in project_url else "GITLAB_TOKEN"
        )
    console.print(
        f"\nSet ${token_env} before running 'thorn serve'."
    )
    if llm_api_key_env is not None:
        console.print(f"Set ${llm_api_key_env} before running 'thorn serve'.")


@serve.command("mcp")
@click.option(
    "--transport",
    type=click.Choice(["stdio", "streamable-http"]),
    default="stdio",
    help="MCP transport to use.",
)
@click.option("--host", default="0.0.0.0", help="Bind host (HTTP transport).")
@click.option("--port", default=8080, type=int, help="Bind port (HTTP transport).")
@click.option("--name", default="thorn", help="Server name reported to MCP clients.")
def serve_mcp(
    transport: str,
    host: str,
    port: int,
    name: str,
) -> None:
    """Start an MCP server exposing built-in Thorn tools."""
    try:
        from thorn.core._mcp import serve_tools
    except ImportError:
        console.print(
            "[red]Error:[/red] MCP support requires the 'mcp' package. "
            "Install it with: uv pip install 'thorn-agent[mcp]'"
        )
        sys.exit(1)

    tools = _prepare_tools(list(ALL_BUILTIN_TOOLS))
    if not tools:
        console.print("[yellow]Warning:[/yellow] no tools to serve.")

    try:
        asyncio.run(serve_tools(
            tools,
            name=name,
            transport=transport,
            host=host,
            port=port,
        ))
    except KeyboardInterrupt:
        console.print("\n[dim]Server stopped.[/dim]")
    except ThornError as exc:
        console.print(f"\n[red]Error:[/red] {exc}")
        sys.exit(1)


# ---------------------------------------------------------------------------
# thorn sandbox (group: build, status)
# ---------------------------------------------------------------------------

@main.group()
def sandbox() -> None:
    """Build and inspect agent tool sandboxes."""


@sandbox.command("build")
@click.option(
    "--tag",
    "tag",
    default=None,
    help=(
        "Image tag to build.  Defaults to thorn-sandbox:<thorn-version>; "
        "match this in gateway.json's sandbox.image to use it."
    ),
)
@click.option(
    "--dockerfile",
    "dockerfile_path",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help=(
        "Path to the Dockerfile.sandbox to build.  Defaults to the "
        "Dockerfile.sandbox shipped with the thorn source tree."
    ),
)
@click.option(
    "--context",
    "context_path",
    type=click.Path(exists=True, file_okay=False),
    default=None,
    help=(
        "Build context directory.  Defaults to the directory containing "
        "the resolved Dockerfile.sandbox."
    ),
)
@click.option(
    "--runtime",
    "runtime_choice",
    type=click.Choice(["podman", "docker"]),
    default=None,
    help=(
        "OCI runtime to use for the build.  Defaults to podman when "
        "available, then docker.  No gateway.json read here -- the build "
        "command is independent of any specific agency configuration."
    ),
)
def sandbox_build(
    tag: str | None,
    dockerfile_path: str | None,
    context_path: str | None,
    runtime_choice: str | None,
) -> None:
    """Build the sandbox image from Dockerfile.sandbox.

    Phase B's image policy is hard-fail on missing image; this is the
    command operators run when the gateway tells them the image is
    missing.  No auto-build happens elsewhere -- making the build an
    explicit, named operator action keeps post-hoc diagnosis simple.
    """
    from thorn.sandbox import (
        OCIRuntimeNotFound,
        build_default_sandbox_image,
        select_oci_runtime,
    )

    try:
        adapter = select_oci_runtime(runtime_choice)  # type: ignore[arg-type]
    except OCIRuntimeNotFound as exc:
        console.print(f"[red]Error:[/red] {exc}")
        sys.exit(1)

    dockerfile = Path(dockerfile_path) if dockerfile_path else None
    context = Path(context_path) if context_path else None

    try:
        resolved_tag = asyncio.run(
            build_default_sandbox_image(
                adapter,
                tag=tag,
                dockerfile=dockerfile,
                context=context,
            ),
        )
    except FileNotFoundError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        sys.exit(1)
    except Exception as exc:
        console.print(f"[red]Error:[/red] sandbox image build failed: {exc}")
        sys.exit(1)

    console.print(
        f"[green]Built sandbox image:[/green] {resolved_tag} "
        f"(runtime: {adapter.name})",
    )


@sandbox.command("status")
@click.option(
    "--agency",
    "agency_path",
    type=click.Path(file_okay=False),
    default=None,
    help=(
        "Agency home directory.  Defaults to ~/.thorn.  The configured "
        "sandbox runtime and per-agent containers are read from this "
        "agency's gateway.json + agents/ tree."
    ),
)
def sandbox_status(agency_path: str | None) -> None:
    """Report sandbox image presence and per-agent container state.

    Reads ``gateway.json`` for the configured runtime + default image,
    then probes the local OCI cache for image presence and queries
    ``ps -a`` (filtered by the ``thorn-agent-`` prefix) for the live
    state of every per-agent container.  Read-only; no containers are
    started, stopped, or removed.

    Live MCP-server state is read from each agent's
    ``<workspace_root>/agents/<id>/control/mcp_state.json`` (written
    by the daemon via :class:`thorn.toolhost.MCPStateSnapshot`) and
    rendered alongside each container.  Missing or unreadable state
    files are silently skipped -- diagnostic commands never crash on
    half-written files.
    """
    from thorn.gateway import load_gateway_config
    from thorn.runtime._paths import AgencyPaths
    from thorn.runtime._store import SessionStore
    from thorn.sandbox import (
        OCIRuntimeNotFound,
        default_sandbox_image_tag,
        select_oci_runtime,
    )
    from thorn.sandbox._container import derive_container_name
    from thorn.toolhost import (
        MCP_STATE_FILE_NAME,
        MCPStateSnapshot,
        read_snapshot,
    )

    agency_home = _resolve_agency_home(agency_path)
    try:
        gateway_config = load_gateway_config(agency_home)
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"[red]Error:[/red] {exc}")
        sys.exit(1)

    sandbox_cfg = getattr(gateway_config, "sandbox", None)
    runtime_choice = (
        sandbox_cfg.oci_runtime if sandbox_cfg is not None else None
    )
    default_image = (
        (sandbox_cfg.image if sandbox_cfg is not None else None)
        or default_sandbox_image_tag()
    )

    try:
        adapter = select_oci_runtime(runtime_choice)
    except OCIRuntimeNotFound as exc:
        console.print(f"[red]Error:[/red] {exc}")
        sys.exit(1)

    async def _gather() -> dict[str, Any]:
        present = await adapter.image_exists(default_image)
        containers = await adapter.list_containers(name_prefix="thorn-agent-")
        return {"present": present, "containers": containers}

    info = asyncio.run(_gather())

    # Build a {container_name -> MCPStateSnapshot} map by enumerating
    # the agents this agency knows about (their canonical IDs live in
    # <home_root>/agents/<safe-id>/agent.json).  We deliberately do
    # *not* try to invert the container name back to an agent ID --
    # derive_container_name's ``[^a-zA-Z0-9_.-] -> _`` substitution is
    # not generally invertible -- so the forward map keeps the lookup
    # honest.  Containers with no matching agent.json get rendered
    # without MCP state, which is the right behavior for orphaned
    # containers and old test debris.
    snapshots_by_container: dict[str, MCPStateSnapshot] = {}
    workspace_root = gateway_config.resolve_workspace(agency_home)
    if workspace_root is not None:
        paths = AgencyPaths(home_root=agency_home, workspace_root=workspace_root)
        try:
            agent_ids = SessionStore(paths).list_agent_ids()
        except Exception as exc:
            console.print(
                f"[yellow]Warning:[/yellow] could not enumerate agents: {exc}"
            )
            agent_ids = []
        for agent_id in agent_ids:
            container_name = derive_container_name(str(agent_id))
            snapshot_path = paths.agent_control_dir(agent_id) / MCP_STATE_FILE_NAME
            snapshot = read_snapshot(snapshot_path)
            if snapshot is not None:
                snapshots_by_container[container_name] = snapshot

    console.print(f"[bold]Agency:[/bold] {agency_home}")
    console.print(f"[bold]Runtime:[/bold] {adapter.name}")
    console.print(
        f"[bold]Default image:[/bold] {default_image} "
        f"({'present' if info['present'] else '[red]missing[/red]'})"
    )
    if not info["present"]:
        console.print(
            "  Run [yellow]thorn sandbox build[/yellow] to build the default image."
        )

    containers = info["containers"]
    if not containers:
        console.print("[bold]Containers:[/bold] (none)")
        return
    console.print(f"[bold]Containers:[/bold] {len(containers)}")
    for state in containers:
        marker = "[green]running[/green]" if state.running else f"[dim]{state.status}[/dim]"
        line = f"  {state.name}  {marker}"
        if state.exit_code is not None and not state.running:
            line += f"  exit={state.exit_code}"
        console.print(line)
        snapshot = snapshots_by_container.get(state.name)
        if snapshot is None or not snapshot.servers:
            continue
        console.print(
            f"    [bold]MCP servers ({len(snapshot.servers)}):[/bold] "
            f"[dim](as of {snapshot.updated_at})[/dim]"
        )
        for server in snapshot.servers:
            alive = (
                "[green]alive[/green]" if server.alive else "[dim]idle[/dim]"
            )
            tool_count = (
                f"tools={server.tool_count}"
                if server.tool_count is not None
                else "tools=?"
            )
            last_used = (
                f"last_used={server.last_used_at}"
                if server.last_used_at
                else "last_used=never"
            )
            console.print(
                f"      {server.name}  ({server.kind})  {alive}  "
                f"{tool_count}  identity={server.config_identity}  "
                f"{last_used}  [dim]{server.identifier}[/dim]"
            )


# ---------------------------------------------------------------------------
# thorn broker (group: status, down)
# ---------------------------------------------------------------------------

@main.group()
def broker() -> None:
    """Inspect or clean up bundled OneCLI broker stacks.

    The bundled broker is normally entirely transparent: ``thorn
    serve`` brings it up on startup and tears it down on shutdown.
    These subcommands exist for the unhappy path -- a previous
    ``thorn serve`` was killed without graceful shutdown (kill -9,
    OOM, host crash) and left compose stacks behind that the next
    startup needs to know about.
    """


@broker.command("status")
def broker_status() -> None:
    """List bundled OneCLI broker compose stacks visible on this host.

    Filters ``docker compose ls`` / ``podman compose ls`` output to
    the ``thorn-broker-*`` project prefix used by
    :class:`BundledBrokerSupervisor`.  Emits one line per stack with
    its compose project name, runtime, and status.

    No exit-code semantics beyond "found something / found nothing"
    -- ``thorn broker status`` exits 0 either way; orphans are an
    expected (if mildly annoying) state of the world, not an error.
    """
    from thorn.gateway._bundled_broker import list_bundled_broker_stacks

    stacks = asyncio.run(list_bundled_broker_stacks())
    if not stacks:
        console.print("[dim]No bundled-broker compose stacks found.[/dim]")
        return
    console.print(
        f"[bold]Bundled-broker stacks ({len(stacks)}):[/bold]"
    )
    for stack in stacks:
        console.print(
            f"  {stack.project_name}  ({stack.runtime_name})  "
            f"[dim]{stack.status}[/dim]"
        )
    console.print(
        "\nRun [yellow]thorn broker down[/yellow] to tear them all down."
    )


@broker.command("down")
@click.option(
    "--project",
    "project_name",
    default=None,
    help=(
        "Tear down only the named compose project (must begin with "
        "'thorn-broker-').  When omitted, every matching stack on the "
        "host is torn down."
    ),
)
def broker_down(project_name: str | None) -> None:
    """Best-effort ``compose down --volumes --remove-orphans`` on stale stacks.

    Used after a non-graceful ``thorn serve`` exit to clean up
    compose stacks that the supervisor did not get a chance to tear
    down itself.  Always safe to run -- it only touches projects
    matching the bundled-broker prefix, and only ever issues
    ``compose down``, never ``compose up``.

    Exits non-zero only when one or more individual ``compose down``
    invocations failed; the CLI emits a per-stack outcome line in all
    cases so the operator can see which (if any) need manual cleanup.
    """
    from thorn.gateway._bundled_broker import (
        BundledBrokerError,
        list_bundled_broker_stacks,
        shutdown_bundled_broker_stack,
    )

    async def _run() -> int:
        stacks = await list_bundled_broker_stacks()
        if project_name is not None:
            stacks = [s for s in stacks if s.project_name == project_name]
            if not stacks:
                console.print(
                    f"[yellow]No matching stack {project_name!r} "
                    "(did you mean a different project?)[/yellow]"
                )
                return 1
        if not stacks:
            console.print(
                "[dim]No bundled-broker compose stacks to tear down.[/dim]"
            )
            return 0
        failures = 0
        for stack in stacks:
            try:
                await shutdown_bundled_broker_stack(stack)
            except BundledBrokerError as exc:
                console.print(
                    f"  [red]FAILED[/red]  {stack.project_name}  ({exc})"
                )
                failures += 1
            else:
                console.print(
                    f"  [green]down[/green]    {stack.project_name}"
                )
        if failures:
            console.print(
                f"\n[red]{failures} stack(s) failed to tear down.[/red]  "
                "You may need to clean them up manually with "
                "`<runtime> compose -p <project> down --volumes`."
            )
            return 1
        return 0

    sys.exit(asyncio.run(_run()))


@broker.command("logs")
@click.option(
    "--project",
    "project_name",
    default=None,
    help=(
        "Collect logs only for the named compose project (must begin with "
        "'thorn-broker-').  When omitted, exactly one matching stack must "
        "be present."
    ),
)
@click.option(
    "--tail",
    type=click.IntRange(min=1),
    default=100,
    show_default=True,
    help="Number of log lines to collect from each broker service.",
)
def broker_logs(project_name: str | None, tail: int) -> None:
    """Print redacted OneCLI/Postgres logs for a bundled broker stack."""
    from thorn.gateway._bundled_broker import (
        BundledBrokerError,
        collect_bundled_broker_stack_logs,
        list_bundled_broker_stacks,
    )

    async def _run() -> int:
        stacks = await list_bundled_broker_stacks()
        if project_name is not None:
            stacks = [s for s in stacks if s.project_name == project_name]
            if not stacks:
                console.print(
                    f"[yellow]No matching stack {project_name!r} "
                    "(did you mean a different project?)[/yellow]"
                )
                return 1
        if not stacks:
            console.print(
                "[dim]No bundled-broker compose stacks found.[/dim]"
            )
            return 1
        if len(stacks) > 1:
            console.print(
                "[yellow]Multiple bundled-broker stacks found; re-run "
                "with --project:[/yellow]"
            )
            for stack in stacks:
                console.print(
                    f"  {stack.project_name}  ({stack.runtime_name})  "
                    f"[dim]{stack.status}[/dim]"
                )
            return 1
        stack = stacks[0]
        try:
            diagnostics = await collect_bundled_broker_stack_logs(
                stack,
                tail=tail,
            )
        except BundledBrokerError as exc:
            console.print(f"[red]FAILED[/red] {exc}")
            return 1
        console.print(
            f"[bold]Bundled-broker logs for {stack.project_name}[/bold] "
            f"({stack.runtime_name})"
        )
        console.print(diagnostics)
        return 0

    sys.exit(asyncio.run(_run()))


if __name__ == "__main__":
    main()
