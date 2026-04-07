"""Gateway daemon orchestrator.

The ``Gateway`` owns a :class:`~thorn.runtime.Runtime`, a list of
:class:`EventSource` instances, and a tool list.  It is the top-level
object that ``thorn serve`` creates to run the agent daemon.

The gateway's only job is to find/create the right agent for each
incoming event, prompt it with a description of what happened, and
save the agent afterward.  The *agent* decides what actions to take
using its tools.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import signal
import sys
from typing import Any

from thorn.gateway._event import EventSource, IncomingEvent
from thorn.runtime import Runtime

log = logging.getLogger(__name__)


class Gateway:
    """Daemon that routes external events to Thorn agents.

    Parameters:
        runtime: The persistent execution environment.
        sources: Event sources to poll / listen on.
        tools: Tools passed to ``agent.prompt(..., tools=...)``
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

            log.info(
                "Gateway started with %d source(s)", len(self._sources),
            )

            try:
                if sys.platform != "win32":
                    await self._stop_event.wait()
                else:
                    # On Windows, signal handlers are not reliable in
                    # asyncio; we rely on KeyboardInterrupt instead.
                    await asyncio.gather(*self._source_tasks)
            except (KeyboardInterrupt, asyncio.CancelledError):
                pass
            finally:
                await self.shutdown()

    async def _handle_event(self, event: IncomingEvent) -> None:
        """Route a single event to the appropriate agent."""
        log.info(
            "Handling event from %s (session=%s)",
            event.source, event.session_key,
        )
        agent = self._runtime.get_or_create_agent(event.session_key)
        try:
            await agent.prompt(event.content, tools=self._tools)
        except Exception:
            log.exception(
                "Agent failed for event (source=%s, session=%s)",
                event.source, event.session_key,
            )
            return

        self._runtime.save_agent(agent)
        log.info(
            "Event handled (source=%s, session=%s)",
            event.source, event.session_key,
        )

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
