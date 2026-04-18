"""Gateway daemon orchestrator.

The ``Gateway`` owns a :class:`~thorn.runtime.Runtime`, a list of
:class:`EventSource` instances, and a tool list.  It is the top-level
object that ``thorn serve`` creates to run the agent daemon.

High-level flow
---------------

The gateway is the *router* of the actor model: it translates incoming
events into durable notifications posted to a session's inbox, then
submits that session to a per-agent scheduler.  The scheduler (one
per agent) owns the concurrency discipline -- one prompt in flight
per session, up to *N* in flight per agent.  Prompt construction,
agent-level tool use, and inbox item close-out all live below the
gateway in :mod:`thorn.runtime`.

Startup:

1. Enter the runtime context.
2. Rebuild the :class:`~thorn.runtime.InFlightIndex` from the on-disk
   state of every queue, so source-level dedup is correct before any
   source starts polling.
3. Run :func:`~thorn.runtime.run_startup_sweep` to heal any
   crash-leftover state (``in_progress`` reverts, stuck ``handled``/
   ``errored`` items re-dispatched, temp files cleaned up).
4. Eagerly load every persisted agent and build an
   :class:`~thorn.runtime.AgentScheduler` for each.  Agents live for
   the lifetime of the process, so paying their load cost up front
   is cheaper than doing it on the first-event hot path and makes
   the steady-state predictable.
5. Start event sources and wait for shutdown.

Per-event handling is :meth:`_handle_event`: resolve the agent, ensure
its scheduler exists, lazily create the session and its inbox, post a
:class:`~thorn.runtime.NotificationSpec` to the inbox, and submit the
session to the scheduler.  No ``agent.lock``: concurrency within an
agent is the scheduler's job now.

Shutdown:

1. Stop sources (no new events).
2. Gracefully shut down every scheduler with a bounded grace period.
   Schedulers cancel in-flight dispatchers on timeout; the startup
   sweep on the next process entry reconciles any resulting
   ``in_progress`` items.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import signal
import sys
from typing import Any

from thorn.core._agent import Agent
from thorn.core._session import Session
from thorn.gateway._event import EventSource, IncomingEvent
from thorn.runtime import (
    DEFAULT_AGENT_CONCURRENCY,
    AgentID,
    AgentScheduler,
    NotificationSpec,
    PromptDispatcher,
    Runtime,
    SessionAddress,
    SessionInbox,
    SessionKey,
    default_progress_evictor,
    inbox_prompt_dispatcher,
    rebuild_in_flight_index,
    run_startup_sweep,
)

log = logging.getLogger(__name__)

_DEFAULT_AGENT_ID = AgentID("default")

DEFAULT_SHUTDOWN_TIMEOUT_SECONDS = 30.0
"""Default grace period handed to :meth:`AgentScheduler.shutdown`.

Long enough to let an in-flight LLM call finish cleanly in most
real-world cases; short enough that a stuck scheduler is not a
deployment incident.  Tests override this with a small value.
"""


class Gateway:
    """Daemon that routes external events to Thorn agents.

    Parameters:
        runtime: The persistent execution environment.  The gateway
            registers session inboxes on its :class:`AddressBook` and
            shares its :class:`InFlightIndex` with every queue it
            creates.
        sources: Event sources to poll / listen on.
        tools: Tools passed to ``session.prompt(..., tools=...)``
            for every event.  Retained for backward compatibility with
            callers that wired tools through the gateway; the scheduler
            now delegates prompt construction to the injected
            :data:`~thorn.runtime.PromptDispatcher`.
        agent_concurrency: Per-agent concurrency cap for the
            :class:`~thorn.runtime.AgentScheduler` of each agent.
            Defaults to :data:`~thorn.runtime.DEFAULT_AGENT_CONCURRENCY`.
        prompt_dispatcher: Override the scheduler's
            :data:`~thorn.runtime.PromptDispatcher`.  Tests use this to
            substitute a dispatcher that closes out inbox items
            automatically; production callers should leave the default
            :func:`~thorn.runtime.inbox_prompt_dispatcher`.
        shutdown_timeout: Grace period (seconds) given to each
            scheduler on gateway shutdown.  ``None`` waits
            indefinitely; ``0`` cancels immediately.  Defaults to
            :data:`DEFAULT_SHUTDOWN_TIMEOUT_SECONDS`.
    """

    def __init__(
        self,
        *,
        runtime: Runtime,
        sources: list[EventSource],
        tools: list[Any] | None = None,
        agent_concurrency: int = DEFAULT_AGENT_CONCURRENCY,
        prompt_dispatcher: PromptDispatcher | None = None,
        shutdown_timeout: float | None = DEFAULT_SHUTDOWN_TIMEOUT_SECONDS,
    ) -> None:
        self._runtime = runtime
        self._sources = sources
        self._tools = list(tools or [])
        self._agent_concurrency = agent_concurrency
        self._prompt_dispatcher: PromptDispatcher = (
            prompt_dispatcher or inbox_prompt_dispatcher
        )
        self._shutdown_timeout = shutdown_timeout

        self._stop_event: asyncio.Event | None = None
        self._source_tasks: list[asyncio.Task[None]] = []
        self._schedulers: dict[AgentID, AgentScheduler] = {}
        self._inboxes: dict[SessionAddress, SessionInbox] = {}
        self._started = False

    async def run(self) -> None:
        """Enter the runtime context and run all sources until shutdown.

        Installs signal handlers on POSIX; falls back to
        ``KeyboardInterrupt`` on Windows.
        """
        self._stop_event = asyncio.Event()

        async with self._runtime:
            self._install_signal_handlers()
            self._startup()

            for source in self._sources:
                task = asyncio.create_task(
                    source.start(self._handle_event),
                )
                self._source_tasks.append(task)

            if self._source_tasks:
                asyncio.create_task(
                    self._stop_when_sources_done(),
                )

            log.info(
                "Gateway started with %d source(s)", len(self._sources),
            )

            try:
                await self._stop_event.wait()
            except (KeyboardInterrupt, asyncio.CancelledError):
                pass
            finally:
                await self.shutdown()

    # ------------------------------------------------------------------
    # Startup
    # ------------------------------------------------------------------

    def _startup(self) -> None:
        """One-time startup: sweep, in-flight index rebuild, agent load.

        Order matters:

        1. Rebuild the in-flight index *first* so the sweep's
           ``delete`` calls update the index we hand out to queues
           later.
        2. Run the sweep, which mutates the filesystem (and the
           rebuilt index) to restore invariants.
        3. Pre-load every persisted agent and stand up its scheduler.
           Sessions themselves stay on disk until their first event
           arrives; we create the :class:`SessionInbox` (and its
           address-book entry) lazily in :meth:`_handle_event`.
        """
        if self._started:
            return

        rebuilt = rebuild_in_flight_index(self._runtime.paths)
        # Merge the rebuilt contents into the runtime's existing
        # index object so any previously-wired consumers keep their
        # reference stable.  The runtime seeds ``in_flight_index``
        # with an empty instance; the merge is effectively a
        # replacement in practice.
        self._runtime.in_flight_index.clear()
        self._runtime.in_flight_index.bulk_add(rebuilt.snapshot())

        report = run_startup_sweep(
            self._runtime.paths,
            self._runtime.address_book,
            in_flight_index=self._runtime.in_flight_index,
        )
        if (
            report.session_handled_dispatched
            or report.session_errored_dispatched
            or report.session_in_progress_reverted
            or report.session_confirmed_cleaned
            or report.service_in_progress_reverted
            or report.service_confirmed_cleaned
            or report.service_misplaced
            or report.temp_files_removed
            or report.dispatch_skipped_unresolved
        ):
            log.info("Startup sweep completed: %s", report)

        for agent_id in self._runtime.sessions.list_agent_ids():
            agent = self._runtime.get_or_create_agent(agent_id)
            self._ensure_scheduler_for_agent(agent)

        self._started = True

    # ------------------------------------------------------------------
    # Agent / scheduler / inbox lifecycle
    # ------------------------------------------------------------------

    def _ensure_scheduler_for_agent(self, agent: Agent) -> AgentScheduler:
        """Return the scheduler for *agent*, creating one on first sight.

        The scheduler is wired with:

        - the injected :data:`PromptDispatcher` (normally the
          inbox-aware default);
        - an async ``save_session`` adapter around
          :meth:`Runtime.save_session`;
        - the canonical :func:`default_progress_evictor` so stalled
          sessions get their oldest item evicted at the N-strikes
          threshold.

        Scheduler instances are kept in the gateway's
        ``_schedulers`` map keyed by :class:`AgentID`.  On shutdown
        the gateway shuts every scheduler down with a bounded grace
        period.
        """
        if agent.id is None:
            raise ValueError("Cannot build a scheduler for an agent without an id")
        existing = self._schedulers.get(agent.id)
        if existing is not None:
            return existing
        scheduler = AgentScheduler(
            agent=agent,
            prompt_dispatcher=self._prompt_dispatcher,
            concurrency=self._agent_concurrency,
            save_session=self._save_session_async,
            progress_evictor=default_progress_evictor(
                self._runtime.address_book,
            ),
        )
        self._schedulers[agent.id] = scheduler
        return scheduler

    async def _save_session_async(self, session: Session) -> None:
        """Async adapter around :meth:`Runtime.save_session`.

        Kept as a method (not a lambda) so the scheduler's own
        exception logging points at a real qualified name.  The
        underlying save is synchronous and small; we do not move it
        to a worker thread because the scheduler already serializes
        per-session, so back-to-back writes don't contend.
        """
        self._runtime.save_session(session)

    def _ensure_inbox(
        self,
        agent: Agent,
        session_key: SessionKey,
    ) -> tuple[SessionAddress, SessionInbox]:
        """Return the :class:`SessionInbox` for ``(agent, session_key)``.

        On first call for a given ``(agent, session_key)`` pair, we:

        - Construct the inbox pointing at
          :meth:`AgencyPaths.session_inbox_dir`, wiring in the
          runtime-wide :class:`InFlightIndex`.
        - Register it in the :class:`AddressBook` under the
          corresponding :class:`SessionAddress` so that step-2
          dispatch (from handling transitions, progress eviction,
          and the startup sweep) can resolve the target.

        Subsequent calls return the same inbox.  A concurrent
        startup-sweep-created inbox is silently replaced so that the
        long-lived gateway instance becomes the canonical registrant.
        """
        if agent.id is None:
            raise ValueError("Cannot build an inbox for an agent without an id")
        address = SessionAddress(agent.id, session_key)
        existing = self._inboxes.get(address)
        if existing is not None:
            return address, existing

        inbox_dir = self._runtime.paths.session_inbox_dir(
            agent.id, session_key,
        )
        inbox = SessionInbox(
            inbox_dir, address,
            in_flight_index=self._runtime.in_flight_index,
        )
        self._inboxes[address] = inbox

        book = self._runtime.address_book
        # The sweep may have registered an ephemeral inbox at this
        # address already (when recovering stuck handled/errored
        # items).  Replace it with our canonical, gateway-owned
        # instance so the rest of the runtime holds a single source
        # of truth.  The two instances point at the same directory,
        # so there is no concurrency hazard from the swap.
        previous = book.get(address)
        if previous is inbox:
            return address, inbox
        if previous is not None:
            book.unregister(address)
        book.register(address, inbox)
        return address, inbox

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    def _resolve_agent(self, event: IncomingEvent) -> Agent:
        """Map an event to the agent instance that should handle it.

        Routing logic (in priority order):

        1. If the event carries an explicit ``agent_id``, use that.
        2. Look for a pre-configured coordinator agent in the runtime
           store.  For the single-coordinator vertical slice, the
           first (and only) persisted agent is used.
        3. Fall back to a bare ``Agent`` with the default ID.

        Future multi-project support would match event metadata (e.g.
        ``project_id``) to the appropriate project-scoped
        coordinator.
        """
        if event.agent_id is not None:
            return self._runtime.get_or_create_agent(event.agent_id)

        persisted_ids = self._runtime.sessions.list_agent_ids()
        if persisted_ids:
            return self._runtime.get_or_create_agent(persisted_ids[0])

        return self._runtime.get_or_create_agent(_DEFAULT_AGENT_ID)

    async def _handle_event(self, event: IncomingEvent) -> None:
        """Post an event to the right inbox and wake its scheduler.

        Does *not* execute the prompt directly.  The session driver
        inside the scheduler picks up the posted notification and
        runs a prompt round at its own pace, under the per-agent
        concurrency cap and per-session serialization guarantee.

        The session workspace is derived from ``AgencyPaths``::

            paths.session_workspace(agent_id, session_key)

        which expands to ``<workspace_root>/<agent_id>/<session_key>/``.
        The directory is pre-created here so the agent always starts
        with a valid working directory.

        Agent identity and session metadata are persisted *before*
        the notification is posted, so ``session_exists`` reports
        the session as soon as an event has arrived for it -- even if
        the first prompt has not yet run (or has crashed).  This is
        a semantic shift from the pre-inbox gateway, where
        persistence was gated on a successful prompt.
        """
        log.info(
            "Handling event from %s (session=%s)",
            event.source, event.session_key,
        )
        agent = self._resolve_agent(event)

        # Persist identity on first sight so the resolved agent is
        # discoverable by a subsequent run even before any prompt has
        # completed.  Idempotent for already-persisted agents.
        if not self._runtime.sessions.agent_exists(agent.id):
            self._runtime.save_agent(agent)

        scheduler = self._ensure_scheduler_for_agent(agent)

        ws = self._runtime.paths.session_workspace(
            agent.id, event.session_key,
        )
        ws.mkdir(parents=True, exist_ok=True)

        session = self._runtime.get_or_create_session(
            agent, event.session_key, workspace_root=ws,
        )
        # Persist session metadata up front so the session directory
        # exists before any prompt round runs.  The scheduler's saver
        # will overwrite ``session.json`` after each successful round
        # with an updated ``last_active`` timestamp.
        self._runtime.save_session(session)

        address, inbox = self._ensure_inbox(agent, event.session_key)

        spec = NotificationSpec(
            source=event.source,
            content=event.content,
            target=address,
            metadata=dict(event.metadata),
        )
        notification = inbox.post(spec)
        log.info(
            "Posted %s to %s (source=%s, session=%s)",
            notification.id, address, event.source, event.session_key,
        )
        await scheduler.submit(session, inbox)

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    async def _stop_when_sources_done(self) -> None:
        """Set the stop event when all source tasks have completed.

        In production, polling sources loop forever and never return
        from ``start()``, so this only fires on signal-driven
        shutdown (where tasks are cancelled) or when a finite source
        is used (e.g. in tests).  This ensures ``run()`` terminates
        cleanly on all platforms without relying on platform-specific
        signal behavior.
        """
        await asyncio.gather(*self._source_tasks, return_exceptions=True)
        if self._stop_event is not None:
            self._stop_event.set()

    async def shutdown(self) -> None:
        """Stop all sources, then drain / cancel every scheduler."""
        log.info("Gateway shutting down ...")
        for source in self._sources:
            try:
                await source.stop()
            except Exception:
                log.exception(
                    "Error stopping source %s", type(source).__name__,
                )

        for task in self._source_tasks:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        self._source_tasks.clear()

        # Schedulers: graceful shutdown with a bounded grace period.
        # ``asyncio.gather`` lets the grace period run in parallel
        # across every agent, so the total shutdown time is bounded
        # by ``shutdown_timeout`` rather than growing linearly with
        # agent count.  Each scheduler internally cancels its
        # drivers on timeout.
        schedulers = list(self._schedulers.values())
        self._schedulers.clear()
        if schedulers:
            await asyncio.gather(
                *(
                    s.shutdown(timeout=self._shutdown_timeout)
                    for s in schedulers
                ),
                return_exceptions=True,
            )
        self._inboxes.clear()
        log.info("Gateway stopped.")

    def _install_signal_handlers(self) -> None:
        """Register SIGINT/SIGTERM to trigger a clean shutdown on POSIX."""
        if sys.platform == "win32":
            return
        loop = asyncio.get_running_loop()
        assert self._stop_event is not None
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, self._stop_event.set)


__all__ = [
    "DEFAULT_SHUTDOWN_TIMEOUT_SECONDS",
    "Gateway",
]
