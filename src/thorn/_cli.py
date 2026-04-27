"""CLI entry point: ``thorn run``, ``thorn chat``, and ``thorn serve``."""

from __future__ import annotations

import asyncio
import json
import sys
import time
import uuid
from pathlib import Path
from typing import Any

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

    *workspace* overrides the workspace root.  When ``None``, the
    heuristic in :func:`thorn.infer_workspace_root` is used.

    *paths* sets the agency directory layout explicitly.  When
    ``None``, falls back to the legacy CLI-nested layout
    (``{ws_root}/.thorn/``) for backwards compatibility with callers
    that haven't yet adopted the Phase-5 ``~/.thorn`` default; all
    production CLI entry points pass *paths* explicitly and do not
    exercise that fallback.
    """
    from pathlib import Path

    from thorn import infer_workspace_root
    from thorn.core._file_access import load_global_ignores
    from thorn.runtime._paths import AgencyPaths

    provider = load_provider_from_env()

    bus = EventBus()
    if trace_file is not None:
        from thorn.core._trace import JsonLinesSink
        bus.subscribe(JsonLinesSink(trace_file))

    ws_root = Path(workspace).resolve() if workspace else infer_workspace_root()

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
    from thorn import infer_workspace_root
    from thorn.runtime._paths import AgencyPaths

    verbosity = _resolve_verbosity(verbose, quiet)
    trace_file = open(trace_path, "w", encoding="utf-8") if trace_path else None
    try:
        agency_home = _resolve_cli_agency_home(agency_path)
        ws_root = (
            Path(workspace_path).resolve() if workspace_path
            else infer_workspace_root()
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
    from thorn import infer_workspace_root
    from thorn.runtime._paths import AgencyPaths

    verbosity = _resolve_verbosity(verbose, quiet)
    trace_file = open(trace_path, "w", encoding="utf-8") if trace_path else None
    try:
        agency_home = _resolve_cli_agency_home(agency_path)
        ws_root = (
            Path(workspace_path).resolve() if workspace_path
            else infer_workspace_root()
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
    sources = infer_event_sources(gateway_config, agents)

    if not sources:
        console.print(
            "[yellow]Warning:[/yellow] No event sources could be inferred "
            f"from {agency_home / 'gateway.json'} and agent accounts. "
            "The gateway will start but will not receive any events."
        )

    gateway = Gateway(runtime=runtime, sources=sources)

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
    """
    from thorn.gateway import load_gateway_config
    from thorn.sandbox import (
        OCIRuntimeNotFound,
        default_sandbox_image_tag,
        select_oci_runtime,
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


if __name__ == "__main__":
    main()
