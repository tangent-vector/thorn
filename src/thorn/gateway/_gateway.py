"""Gateway daemon orchestrator.

The ``Gateway`` owns a :class:`~thorn.runtime.Runtime`, a list of
:class:`EventSource` instances, and a tool list.  It is the top-level
object that ``thorn serve`` creates to run the agent daemon.

The gateway's job is to resolve the right agent for each incoming event,
find or create a session under that agent, prompt it with a description
of what happened, and save the session afterward.  The *agent* decides
what actions to take using its tools.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import signal
import sys
from typing import Any

from thorn.core._agent import Agent
from thorn.gateway._event import EventSource, IncomingEvent
from thorn.runtime import AgentID, Runtime

log = logging.getLogger(__name__)

_DEFAULT_AGENT_ID = AgentID("default")


class Gateway:
    """Daemon that routes external events to Thorn agents.

    Parameters:
        runtime: The persistent execution environment.
        sources: Event sources to poll / listen on.
        tools: Tools passed to ``session.prompt(..., tools=...)``
            for every event.  Typically the GitLab tool list.
    """

    def __init__(
        self,
        *,
        runtime: Runtime,
        sources: list[EventSource],
        tools: list[Any] | None = None,
    ) -> None:
        self._runtime = runtime
        self._sources = sources
        self._tools = list(tools or [])
        self._stop_event: asyncio.Event | None = None
        self._source_tasks: list[asyncio.Task[None]] = []

    async def run(self) -> None:
        """Enter the runtime context and run all sources until shutdown.

        Installs signal handlers on POSIX; falls back to
        ``KeyboardInterrupt`` on Windows.
        """
        self._stop_event = asyncio.Event()

        async with self._runtime:
            self._install_signal_handlers()

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

    def _resolve_agent(self, event: IncomingEvent) -> Agent:
        """Map an event to the agent instance that should handle it.

        Routing logic (in priority order):

        1. If the event carries an explicit ``agent_id``, use that.
        2. Look for a pre-configured coordinator agent in the runtime
           store.  For the single-coordinator vertical slice, the first
           (and only) persisted agent is used.
        3. Fall back to a bare ``Agent`` with the default ID.

        Future multi-project support would match event metadata (e.g.
        ``project_id``) to the appropriate project-scoped coordinator.
        """
        if event.agent_id is not None:
            return self._runtime.get_or_create_agent(event.agent_id)

        persisted_ids = self._runtime.sessions.list_agent_ids()
        if persisted_ids:
            return self._runtime.get_or_create_agent(persisted_ids[0])

        return self._runtime.get_or_create_agent(_DEFAULT_AGENT_ID)

    async def _handle_event(self, event: IncomingEvent) -> None:
        """Route a single event to the appropriate agent and session.

        Tools are determined by the agent's class-level ``tools``
        declaration (via ``Agent._collect_tools``), not passed
        explicitly by the gateway.

        The agent's lock serializes all event handling for a given
        agent, preventing concurrent mutations to its workspace,
        session state, and MEMORY.md.
        """
        log.info(
            "Handling event from %s (session=%s)",
            event.source, event.session_key,
        )
        agent = self._resolve_agent(event)

        async with agent.lock:
            self._runtime.save_agent(agent)
            session = self._runtime.get_or_create_session(agent, event.session_key)
            try:
                await session.prompt(event.content)
            except Exception:
                log.exception(
                    "Agent failed for event (source=%s, session=%s)",
                    event.source, event.session_key,
                )
                return

            self._runtime.save_session(session)
            log.info(
                "Event handled (source=%s, session=%s)",
                event.source, event.session_key,
            )

    async def _stop_when_sources_done(self) -> None:
        """Set the stop event when all source tasks have completed.

        In production, polling sources loop forever and never return
        from ``start()``, so this only fires on signal-driven shutdown
        (where tasks are cancelled) or when a finite source is used
        (e.g. in tests).  This ensures ``run()`` terminates cleanly on
        all platforms without relying on platform-specific signal
        behavior.
        """
        await asyncio.gather(*self._source_tasks, return_exceptions=True)
        if self._stop_event is not None:
            self._stop_event.set()

    async def shutdown(self) -> None:
        """Stop all sources and cancel background tasks."""
        log.info("Gateway shutting down ...")
        for source in self._sources:
            try:
                await source.stop()
            except Exception:
                log.exception("Error stopping source %s", type(source).__name__)

        for task in self._source_tasks:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

        self._source_tasks.clear()
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
    "Gateway",
]
