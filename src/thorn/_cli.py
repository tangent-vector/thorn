"""CLI entry point: ``thorn run``, ``thorn chat``, and ``thorn serve``."""

from __future__ import annotations

import asyncio
import json
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import click
from rich.console import Console

from thorn.agents import LocalCodingAgent
from thorn.core._context import (
    ConsoleEventSink,
    EventSink,
    ExecutionContext,
    Verbosity,
)
from thorn.core._event_bus import EventBus, in_session
from thorn.core._func import _prepare_tools
from thorn.core._provider import load_provider_from_env
from thorn.core._session import Session
from thorn.core._tools import ALL_BUILTIN_TOOLS
from thorn.core.errors import SkillError, ThornError
from thorn.runtime import (
    AgentID,
    AgentScheduler,
    ChatPromptRouter,
    NotificationSpec,
    Runtime,
    SessionAddress,
    SessionInbox,
    SessionKey,
    make_cli_prompt_dispatcher,
)
from thorn.runtime._project_detection import (
    pick_logical_agent_workspace_path_for_cli_session,
)

console = Console()


def _emit_json(payload: Any) -> None:
    """Write JSON without Rich wrapping or markup interpretation."""
    click.echo(json.dumps(payload, indent=2, sort_keys=True))


if TYPE_CHECKING:
    from thorn.core import Agent
    from thorn.gateway._config import SandboxConfig
    from thorn.gateway._preflight import GitPreflightTarget
    from thorn.runtime import AgencyPaths

CLI_AGENT_ID = AgentID("local")
"""Well-known agent ID for the CLI local coding agent."""


def _generate_cli_session_key(workspace_root: Path) -> SessionKey:
    """Generate a fresh session key for a single CLI invocation.

    Phase 5 of the CLI/gateway unification: every ``thorn run`` and
    every ``thorn chat`` invocation gets a brand-new ephemeral session
    rather than resuming a stable per-workspace one.  The aspirational
    architecture doc says:

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
      invocations and is short enough to type or paste in a future
      ``/resume <key>`` slash command.

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
    workspace: str | None = None,
    *,
    paths: "AgencyPaths | None" = None,
    sandbox_executor_enabled: bool = False,
    sandbox_config: "SandboxConfig | None" = None,
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

    *workspace* overrides the workspace root.  When ``None``, falls
    back to the resolved current working directory.

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

    provider = load_provider_from_env()

    bus = EventBus()
    if trace_file is not None:
        from thorn.core._trace import JsonLinesSink
        bus.subscribe(JsonLinesSink(trace_file))

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
    )


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


# ---------------------------------------------------------------------------
# CLI group
# ---------------------------------------------------------------------------

@click.group()
def main() -> None:
    """Thorn — a lightweight agent harness."""
    try:
        from dotenv import find_dotenv, load_dotenv
        load_dotenv(find_dotenv(usecwd=True))
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# thorn inbox
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _ErroredInboxMatch:
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
        console.print(
            f"  {record.item_id}  {record.status.value}  "
            f"{record.location.value}  agent={record.agent_id}  "
            f"session={record.session_key}  source={record.notification.source}  "
            f"{record.summary}"
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
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


@main.command()
@click.argument("prompt_text")
@click.option("-v", "--verbose", count=True, help="Increase output detail (-v, -vv).")
@click.option("-q", "--quiet", is_flag=True, default=False, help="Suppress all output except the final answer.")
@click.option("--trace", "trace_path", type=click.Path(), default=None, help="Write execution trace to a JSONL file.")
@click.option("--workspace", "workspace_path", type=click.Path(exists=True, file_okay=False), default=None, help="Override workspace root directory.")
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
@click.option("--result-file", "result_file_path", type=click.Path(), default=None, help="Write a JSON result summary (outcome, duration, token usage).")
def run(prompt_text: str, verbose: int, quiet: bool, trace_path: str | None, workspace_path: str | None, agency_path: str | None, result_file_path: str | None) -> None:
    """Execute a single prompt and print the result."""
    from thorn.runtime._paths import AgencyPaths

    verbosity = _resolve_verbosity(verbose, quiet)
    trace_file = open(trace_path, "w", encoding="utf-8") if trace_path else None
    try:
        agency_home = _resolve_cli_agency_home(agency_path)
        ws_root = (
            Path(workspace_path).resolve() if workspace_path
            else Path.cwd().resolve()
        )
        paths = AgencyPaths(home_root=agency_home, workspace_root=ws_root)
        runtime = _build_runtime(
            trace_file=trace_file,
            workspace=str(ws_root),
            paths=paths,
            sandbox_executor_enabled=True,
        )
    except ThornError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        if trace_file:
            trace_file.close()
        if result_file_path:
            _write_result_file(
                Path(result_file_path), "agent_error", 0.0, None, str(exc), trace_path,
            )
        sys.exit(1)

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
            agent = _ensure_cli_agent(runtime)

            # Phase 5: every CLI invocation gets a fresh session
            # under a unique key (``cli/<workspace-basename>/<uuid>``).
            # The in-memory session is never persisted
            # (``save_session=None`` below) so the only on-disk
            # artefacts are the session's inbox directory (empty
            # after the dispatcher deletes the one notification it
            # processes) and, in future, shutdown-housekeeping
            # journal entries that the agent may choose to write.
            if agent.id is None:
                raise RuntimeError(
                    "CLI agent has no id; cannot build a session inbox"
                )
            session_key = _generate_cli_session_key(runtime.workspace_root)
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
                session = Session(
                    agent=agent,
                    key=session_key,
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
                dispatcher = make_cli_prompt_dispatcher(
                    result_future=result_future,
                    extra_system=(
                        "You are executing a single non-interactive "
                        "request. Complete the task and report results "
                        "concisely. Do not offer follow-up actions or "
                        "ask questions."
                    ),
                )
                # ``save_session=None``: ``thorn run`` is ephemeral.
                # Phase 5 will revisit when the local-agency defaults
                # land and persistence semantics get redesigned.
                scheduler = AgentScheduler(
                    agent=agent,
                    prompt_dispatcher=dispatcher,
                    save_session=None,
                )
                try:
                    inbox.post(NotificationSpec(
                        source="user",
                        content=prompt_text,
                        target=session_address,
                    ))
                    await scheduler.submit(session, inbox)
                    return await result_future
                finally:
                    # Bounded grace period so a misbehaving
                    # dispatcher cannot wedge process exit
                    # indefinitely.  By the time we get here the
                    # future has resolved, the dispatcher has
                    # returned, and the driver has parked on its
                    # idle wait, so shutdown is essentially
                    # instantaneous in the success case.
                    await scheduler.shutdown(timeout=5.0)

    outcome = "success"
    error_msg: str | None = None
    exit_code = 0
    t0 = time.monotonic()

    try:
        asyncio.run(_run())
    except SkillError as exc:
        outcome, error_msg, exit_code = "agent_error", exc.detail, 1
        console.print(f"\n[red]Agent error:[/red] {exc.detail}")
    except TimeoutError:
        outcome, error_msg, exit_code = "timeout", "timed out", 1
        console.print("\n[red]Error:[/red] timed out")
    except ThornError as exc:
        outcome, error_msg, exit_code = "agent_error", str(exc), 1
        console.print(f"\n[red]Error:[/red] {exc}")
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
@click.option("--workspace", "workspace_path", type=click.Path(exists=True, file_okay=False), default=None, help="Override workspace root directory.")
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
def chat(verbose: int, quiet: bool, trace_path: str | None, workspace_path: str | None, agency_path: str | None, no_housekeeping: bool) -> None:
    """Start an interactive chat session."""
    from thorn.runtime._paths import AgencyPaths

    verbosity = _resolve_verbosity(verbose, quiet)
    trace_file = open(trace_path, "w", encoding="utf-8") if trace_path else None
    try:
        agency_home = _resolve_cli_agency_home(agency_path)
        ws_root = (
            Path(workspace_path).resolve() if workspace_path
            else Path.cwd().resolve()
        )
        paths = AgencyPaths(home_root=agency_home, workspace_root=ws_root)
        runtime = _build_runtime(
            trace_file=trace_file,
            workspace=str(ws_root),
            paths=paths,
            sandbox_executor_enabled=True,
        )
    except ThornError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        if trace_file:
            trace_file.close()
        sys.exit(1)

    console.print("[bold]thorn[/bold] interactive chat  (Ctrl+C to exit)\n")

    async def _chat() -> None:
        # Phase 4 of the CLI/gateway unification: route the chat REPL
        # through the in-process ``AgentScheduler`` rather than calling
        # ``session.prompt`` directly.  Each user input is posted as a
        # notification on the session's inbox; the scheduler invokes
        # ``ChatPromptRouter.dispatcher`` which calls ``session.prompt``
        # and resolves a per-turn future the REPL awaits.
        #
        # Phase 5: each invocation gets a fresh session key, so the
        # implicit per-workspace resume (and the PID-suffixed
        # collision-fallback hack) are gone.  The session lock is no
        # longer useful either -- the uuid-suffixed key cannot collide
        # across processes -- so it's dropped.  An explicit
        # ``--resume <key>`` flow is a Phase 5 follow-up.
        async with runtime:
            agent = _ensure_cli_agent(runtime)
            if agent.id is None:
                raise RuntimeError(
                    "CLI agent has no id; cannot build a session inbox"
                )

            session_key = _generate_cli_session_key(runtime.workspace_root)
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
    "--agency",
    "agency_path",
    type=click.Path(file_okay=False),
    default=None,
    help=(
        "Agency home directory (contains gateway.json and the agents/ tree). "
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
        "the 'workspace' field in gateway.json."
    ),
)
@click.pass_context
def serve(
    ctx: click.Context,
    verbose: int,
    quiet: bool,
    trace_path: str | None,
    agency_path: str | None,
    workspace_path: str | None,
) -> None:
    """Start the Thorn gateway daemon (or an MCP server via 'thorn serve mcp')."""
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    ctx.obj["quiet"] = quiet
    ctx.obj["trace_path"] = trace_path
    ctx.obj["agency_path"] = agency_path
    ctx.obj["workspace_path"] = workspace_path

    if ctx.invoked_subcommand is not None:
        return

    _serve_gateway(
        verbose=verbose,
        quiet=quiet,
        trace_path=trace_path,
        agency_path=agency_path,
        workspace_path=workspace_path,
    )


def _resolve_agency_home(agency_path: str | None) -> Path:
    """Resolve the agency home directory for ``thorn serve``.

    Uses *agency_path* when provided; otherwise falls back to
    ``~/.thorn`` (the local-agency convention from the architecture
    doc).  Errors out with a clear message if the directory does not
    exist or does not contain a ``gateway.json``.
    """
    from thorn.gateway._config import GATEWAY_CONFIG_FILENAME

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

    if not (agency_home / GATEWAY_CONFIG_FILENAME).is_file():
        console.print(
            f"[red]Error:[/red] No {GATEWAY_CONFIG_FILENAME} in agency home: {agency_home}\n"
            "Run 'thorn serve bootstrap --agency-home <dir> --agency-workspace <dir> ...' "
            "to create one."
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
        f"'workspace' field in {agency_home / 'gateway.json'} (re-run "
        "'thorn serve bootstrap' with --agency-workspace to do that)."
    )
    sys.exit(1)


def _serve_gateway(
    *,
    verbose: int,
    quiet: bool,
    trace_path: str | None,
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
    except FileNotFoundError as exc:
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
            trace_file=trace_file, workspace=str(ws_root),
            paths=paths,
            sandbox_executor_enabled=True,
            sandbox_config=gateway_config.sandbox,
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
    help="Per-git-operation timeout in seconds inside the sandbox.",
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
    """Preflight sandboxed git connectivity through the broker.

    This starts the configured sandbox and broker path, invokes git
    from inside the sandbox, and does not start event sources or touch
    forge TODO/notification state.
    """
    verbose = ctx.obj.get("verbose", 0)
    quiet = ctx.obj.get("quiet", False)
    trace_path = ctx.obj.get("trace_path")
    agency_path = ctx.obj.get("agency_path")
    workspace_path = ctx.obj.get("workspace_path")

    exit_code = _serve_preflight(
        verbose=verbose,
        quiet=quiet,
        trace_path=trace_path,
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
        instantiate_services,
        load_gateway_config,
    )
    from thorn.gateway._preflight import collect_git_preflight_targets
    from thorn.runtime._paths import AgencyPaths

    verbosity = _resolve_verbosity(verbose, quiet)
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
            workspace=str(ws_root),
            paths=paths,
            sandbox_executor_enabled=True,
            sandbox_config=gateway_config.sandbox,
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

    async def _run() -> int:
        gateway = Gateway(
            runtime=runtime,
            sources=[],
            gateway_config=gateway_config,
        )
        failures = 0
        async with runtime:
            try:
                for agent in agents:
                    gateway._ensure_scheduler_for_agent(agent)
                gateway._warn_if_egress_allowlist_unenforced()
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
    "--agency-home",
    "agency_home_path",
    type=click.Path(file_okay=False),
    required=True,
    help=(
        "Agency home directory: holds gateway.json and the agents/ tree. "
        "Created if missing.  No .thorn/ subdirectory is appended."
    ),
)
@click.option(
    "--agency-workspace",
    "agency_workspace_path",
    type=click.Path(file_okay=False),
    required=True,
    help=(
        "Agency workspace directory: where agent sessions do their work.  "
        "Recorded as the 'workspace' field of gateway.json so 'thorn serve' "
        "can locate it.  Created if missing."
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
    agency_home_path: str,
    agency_workspace_path: str,
) -> None:
    """Bootstrap a ProjectCoordinator agent in an agency home directory."""
    from pathlib import Path

    from thorn.gateway._bootstrap import bootstrap_coordinator

    agency_home = Path(agency_home_path).expanduser().resolve()
    agency_workspace = Path(agency_workspace_path).expanduser().resolve()
    try:
        aid = bootstrap_coordinator(
            agency_home=agency_home,
            agency_workspace=agency_workspace,
            agent_id=agent_id,
            project_name=project_name,
            project_url=project_url,
            access_token_env=token_env,
            git_user_name=git_user_name or "",
            git_user_email=git_user_email or "",
        )
    except ValueError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        sys.exit(1)

    gateway_config_path = agency_home / "gateway.json"
    console.print(f"[green]Bootstrapped coordinator:[/green] {aid}")
    console.print(f"  Identity: {agency_home / 'agents' / f'{aid}.json'}")
    console.print(f"  Agent home: {agency_home / 'agents' / str(aid)}")
    console.print(f"  Gateway config: {gateway_config_path}")
    console.print(f"  Agent workspace: {agency_workspace / str(aid)}")
    if token_env is None:
        token_env = (
            "GITHUB_TOKEN" if "github.com" in project_url else "GITLAB_TOKEN"
        )
    console.print(
        f"\nSet ${token_env} before running 'thorn serve'."
    )


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
    """Start an MCP server exposing thorn tools and skills."""
    try:
        from thorn.core._mcp import serve_tools
    except ImportError:
        console.print(
            "[red]Error:[/red] MCP support requires the 'mcp' package. "
            "Install it with:  pip install thorn[mcp]"
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
    """Manage Phase-B sandbox container images and runtime status."""


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
    except FileNotFoundError as exc:
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
