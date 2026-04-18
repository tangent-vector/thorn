"""Per-agent scheduler for session prompt execution.

The :class:`AgentScheduler` owns the concurrency discipline for a
single agent: at most one prompt in flight per session, and at most
*N* prompts in flight across all of that agent's sessions (the
agent-level concurrency cap).  It does not own prompt construction,
session persistence, or event ingestion; those are supplied as
injected callables so the scheduler can be used and tested in
isolation from the rest of the runtime.

High-level model
----------------

The scheduler's state machine is the one described in the Session
Inbox Abstraction plan::

    Post --> Lock(session) --> Semaphore(agent cap) --> build+prompt
                                                            |
                                          v-------------<---+
                                          check inbox again
                                          |   |
                                     pending   empty
                                          |   |
                                    (loop)    (idle; wait for kick)

In this module the per-session lock is not an explicit
:class:`asyncio.Lock`; it is realized structurally.  We run exactly
one long-lived drain task per registered session, so a session's
prompt rounds are serialized by construction.  This costs one idle
asyncio task per registered session, which is bounded because the
runtime only registers sessions with non-empty inboxes on startup
(or on first event arrival thereafter).  A future optimization can
collapse idle drivers if that overhead ever matters.

The wake signal from :meth:`AgentScheduler.submit` to a driver is an
:class:`asyncio.Event`.  The drain loop clears the event *before* it
inspects the inbox, so a kick that arrives between the inbox read
and the idle wait is still observed.  See
:meth:`_SessionDriver._drain_loop` for the exact ordering.

What the scheduler does NOT do
------------------------------

- **Prompt construction.**  The injected :data:`PromptDispatcher`
  receives the session and its inbox and is responsible for building
  whatever prompt shape is appropriate.  The "prompt-format" work
  item in the plan ships a real dispatcher; tests and intermediate
  integrations can pass any async callable.

- **Status transitions on notifications.**  Marking an item
  ``in_progress``, ``handled``, or ``errored`` is the agent's job
  (via the ``update_inbox_item`` tool, a later work item).  The
  scheduler only reads ``prompt_pending()`` to decide whether a
  round is needed; it never mutates notification state itself.

- **Progress guarantees.**  A session whose dispatcher spins without
  the agent making progress (no items transitioning out of pending/
  in_progress) will loop indefinitely here.  The N-strikes guard is
  a distinct work item that layers on top of the scheduler by
  watching transitions between rounds.

- **Persistence.**  Agent identity and session history persistence
  are orthogonal.  An optional :data:`SessionSaver` callback is
  invoked after each successful round to match the gateway's
  existing save-after-prompt discipline without coupling the
  scheduler to :class:`SessionStore`.

Error policy
------------

A dispatcher exception is logged and the drain loop continues.  The
notification state is untouched, so the next loop iteration will see
the same inbox contents and re-invoke the dispatcher -- which, if
the failure is transient, will proceed.  If the failure is
persistent, the session loops; the N-strikes guard evicts the
oldest item eventually.  A :class:`SessionSaver` exception is
similarly logged and the loop continues; the scheduler does not
treat persistence failures as prompt failures.

Shutdown
--------

:meth:`AgentScheduler.shutdown` stops accepting new submissions,
nudges idle drivers to exit, waits up to the given timeout for
in-flight rounds to complete, then cancels any still-running
dispatchers.  ``timeout=None`` waits indefinitely; ``timeout=0``
cancels immediately.  Cancellation propagates into the dispatcher
call, which is expected to unwind the in-flight LLM call via normal
asyncio cancellation rules.  Items left in ``in_progress`` state by
a cancelled dispatcher are the startup sweep's concern on the next
process entry.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import TYPE_CHECKING, Awaitable, Callable

from thorn.runtime._inbox import SessionInbox

if TYPE_CHECKING:
    from thorn.core._agent import Agent
    from thorn.core._session import Session
    from thorn.runtime._session import SessionKey


log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Injection points
# ---------------------------------------------------------------------------

PromptDispatcher = Callable[["Session", SessionInbox], Awaitable[None]]
"""Callable that runs one prompt round for a session.

Invoked by the scheduler when a session's inbox has at least one
prompt-pending notification.  The dispatcher is responsible for
constructing the prompt text (e.g. via the Phase 1 prompt-format
module) and awaiting ``session.prompt(...)``.  Any tool calls the
agent makes during that prompt may mutate the inbox via the
``update_inbox_item`` tool; the scheduler will re-check the inbox
after the dispatcher returns and loop as needed.

Return-normally indicates the round completed; the scheduler will
then call :data:`SessionSaver` (if one was supplied) and re-examine
the inbox.  Raising any exception aborts this round only; the
scheduler logs and loops.
"""

SessionSaver = Callable[["Session"], Awaitable[None]]
"""Callable that persists a session after a completed prompt round.

Typically bound to ``Runtime.save_session``.  Called after each
successful :data:`PromptDispatcher` return; not called if the
dispatcher raised.  Exceptions are logged; the scheduler continues.
"""


DEFAULT_AGENT_CONCURRENCY = 3
"""Default agent-level concurrency cap.

Matches the plan's recommended baseline.  Gateway-level
configuration and per-agent overrides are plumbed in by the
``Runtime`` / ``Gateway`` layers; this scheduler module just picks
the value up from its constructor.
"""


# ---------------------------------------------------------------------------
# AgentScheduler
# ---------------------------------------------------------------------------

class AgentScheduler:
    """Schedule prompt rounds for all sessions under a single agent.

    One scheduler per :class:`Agent`.  Constructed by the runtime on
    agent load; ``submit(session, inbox)`` is the entry point that
    the gateway (and later the runtime's startup sweep) calls to say
    "this session may have work."  The scheduler is idempotent on
    repeated submits: the second call re-uses the existing driver
    for that session and simply re-signals it.

    All scheduler state is in-process and async-only.  Durability
    lives one layer lower in the :class:`~thorn.runtime.DurableQueue`
    primitives -- when the process dies, the scheduler evaporates,
    and the next start rebuilds from the on-disk inboxes.
    """

    def __init__(
        self,
        *,
        agent: "Agent",
        prompt_dispatcher: PromptDispatcher,
        concurrency: int = DEFAULT_AGENT_CONCURRENCY,
        save_session: SessionSaver | None = None,
    ) -> None:
        if concurrency < 1:
            raise ValueError(
                f"AgentScheduler concurrency must be >= 1, got {concurrency!r}"
            )
        self._agent = agent
        self._dispatcher = prompt_dispatcher
        self._save_session = save_session
        self._concurrency = concurrency
        self._semaphore = asyncio.Semaphore(concurrency)
        self._drivers: dict["SessionKey", _SessionDriver] = {}
        self._drivers_lock = asyncio.Lock()
        self._closed = False

    @property
    def agent(self) -> "Agent":
        """The agent this scheduler serves."""
        return self._agent

    @property
    def concurrency(self) -> int:
        """The agent-level concurrency cap this scheduler was constructed with."""
        return self._concurrency

    @property
    def is_closed(self) -> bool:
        """True once :meth:`shutdown` has been called."""
        return self._closed

    def session_keys(self) -> list["SessionKey"]:
        """Return the session keys currently known to this scheduler.

        Ordering is insertion order.  Does not synchronize with the
        internal driver map; treat the result as a best-effort
        snapshot.  Intended for observability and tests.
        """
        return list(self._drivers.keys())

    async def submit(
        self,
        session: "Session",
        inbox: SessionInbox,
    ) -> None:
        """Register (if needed) and wake the driver for *session*.

        Safe to call repeatedly for the same session: the first call
        creates the driver and starts its drain task; subsequent
        calls just re-signal.  The first call's *inbox* is the one
        the driver will use forever -- later submits for the same
        session key silently ignore the *inbox* argument, since a
        session's inbox identity is fixed.

        Raises :class:`RuntimeError` if the scheduler has been
        closed (via :meth:`shutdown`).
        """
        if self._closed:
            raise RuntimeError(
                f"AgentScheduler for {self._agent.id!r} is closed; "
                "cannot submit new work"
            )
        if session.key is None:
            raise ValueError(
                "Cannot submit a session without a key to the scheduler"
            )
        async with self._drivers_lock:
            driver = self._drivers.get(session.key)
            if driver is None:
                driver = _SessionDriver(
                    session=session,
                    inbox=inbox,
                    semaphore=self._semaphore,
                    dispatcher=self._dispatcher,
                    save_session=self._save_session,
                )
                self._drivers[session.key] = driver
                driver.start()
        driver.kick()

    async def shutdown(self, *, timeout: float | None = None) -> None:
        """Stop all drivers and wait for in-flight prompts to finish.

        Behavior:

        - Sets the scheduler's closed flag.  Subsequent
          :meth:`submit` calls raise :class:`RuntimeError`.
        - Wakes every driver's idle-wait, so drivers currently
          sleeping on a kick observe the closed flag and exit
          their loops cleanly.
        - Waits up to *timeout* seconds for any currently-running
          dispatcher to return normally.
        - After the timeout, cancels all remaining driver tasks.
          Cancellation propagates into the dispatcher (and into the
          LLM call it awaits), which is expected to unwind via
          standard asyncio semantics.

        ``timeout=None`` waits indefinitely (no hard cancel).
        ``timeout=0`` skips the grace period and cancels
        immediately.  Idempotent: a second ``shutdown`` call returns
        immediately after flipping the flag.
        """
        if self._closed:
            return
        self._closed = True

        # Snapshot drivers under the lock so shutdown does not race
        # a last-moment submit that is mid-flight.
        async with self._drivers_lock:
            drivers = list(self._drivers.values())

        if not drivers:
            return

        # Flip each driver's closed flag and wake any idle-wait so
        # they re-check and exit on their own terms.  Drivers that
        # are mid-prompt will reach the top-of-loop check after
        # their current dispatcher returns (or be cancelled below).
        for driver in drivers:
            driver.request_stop()

        tasks = [driver.task for driver in drivers if driver.task is not None]
        if not tasks:
            return

        if timeout is not None and timeout <= 0:
            # Immediate-cancel mode.
            for task in tasks:
                task.cancel()
        else:
            # Grace period: wait for tasks to finish on their own.
            try:
                await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True),
                    timeout=timeout,
                )
                return
            except asyncio.TimeoutError:
                for task in tasks:
                    if not task.done():
                        task.cancel()

        # Await cancellation completion.  We swallow CancelledError
        # here because we originated the cancel and the caller of
        # shutdown() expects a clean return.
        with contextlib.suppress(asyncio.CancelledError):
            await asyncio.gather(*tasks, return_exceptions=True)


# ---------------------------------------------------------------------------
# _SessionDriver (internal)
# ---------------------------------------------------------------------------

class _SessionDriver:
    """A long-lived drain task bound to one session.

    Owns an :class:`asyncio.Event` used by the enclosing
    :class:`AgentScheduler` to signal "check the inbox again" and
    an :class:`asyncio.Task` running :meth:`_drain_loop`.  The driver
    is created on first submit for a session and lives until the
    scheduler's ``shutdown`` cancels (or awaits) it.
    """

    __slots__ = (
        "_session", "_inbox", "_semaphore", "_dispatcher", "_save_session",
        "_wake", "_task", "_stop",
    )

    def __init__(
        self,
        *,
        session: "Session",
        inbox: SessionInbox,
        semaphore: asyncio.Semaphore,
        dispatcher: PromptDispatcher,
        save_session: SessionSaver | None,
    ) -> None:
        self._session = session
        self._inbox = inbox
        self._semaphore = semaphore
        self._dispatcher = dispatcher
        self._save_session = save_session
        self._wake = asyncio.Event()
        self._task: asyncio.Task[None] | None = None
        self._stop = False

    @property
    def task(self) -> asyncio.Task[None] | None:
        return self._task

    def start(self) -> None:
        """Launch the drain task.  Idempotent after the first call."""
        if self._task is not None:
            return
        session_key = self._session.key
        agent_id = self._session.agent.id
        self._task = asyncio.create_task(
            self._drain_loop(),
            name=f"thorn-session-driver:{agent_id}:{session_key}",
        )

    def kick(self) -> None:
        """Signal the drain loop that there may be new work."""
        self._wake.set()

    def request_stop(self) -> None:
        """Ask the drain loop to exit at its next quiescent point."""
        self._stop = True
        # Waking the event ensures an idle driver observes the stop
        # flag and falls through its while-loop; a mid-round driver
        # will see the flag when it comes back to the top of the
        # loop.
        self._wake.set()

    async def _drain_loop(self) -> None:
        """Process prompt rounds for this session until stopped.

        Loop invariant: at the top of each iteration we have not yet
        observed any pending work for this iteration; we clear the
        wake event *before* reading the inbox, so any kick that
        happens between our read and our wait is still observed.
        """
        try:
            while not self._stop:
                # Guaranteed yield per iteration.  Without it, a
                # dispatcher that completes synchronously (e.g. one
                # that raises before any real await, or a very fast
                # test stub) combined with an uncontended semaphore
                # acquire would make this loop run tail-to-head with
                # zero yield points, starving shutdown cancellation
                # and any cooperative observers on the same loop.
                await asyncio.sleep(0)
                # Consume any pending signal BEFORE checking the
                # inbox.  If a kick happens after we read an empty
                # inbox and before we call wait(), the set persists
                # and wait() returns immediately.  Reversing the
                # order would let such a kick be silently dropped.
                self._wake.clear()
                pending = self._inbox.prompt_pending()
                if pending:
                    await self._run_one_round()
                    continue
                # Idle until somebody kicks us (or shutdown fires).
                await self._wake.wait()
        except asyncio.CancelledError:
            # Shutdown's hard-cancel path.  Re-raise so the task
            # reports cancelled; callers of shutdown() swallow the
            # error intentionally.
            raise
        except Exception:
            # Defensive: never let an unexpected exception silently
            # kill the driver without a log line.  Dispatcher and
            # save_session exceptions are handled inside
            # _run_one_round and never reach here.
            log.exception(
                "Session driver for agent=%r session=%r crashed unexpectedly",
                self._session.agent.id, self._session.key,
            )

    async def _run_one_round(self) -> None:
        """Invoke the dispatcher once under the agent-level cap."""
        async with self._semaphore:
            try:
                await self._dispatcher(self._session, self._inbox)
            except asyncio.CancelledError:
                # Shutdown tore through the dispatcher.  Propagate
                # so the drain task ends.
                raise
            except Exception:
                log.exception(
                    "Prompt dispatcher raised for agent=%r session=%r; "
                    "leaving inbox state untouched and looping",
                    self._session.agent.id, self._session.key,
                )
                return

        if self._save_session is None:
            return
        try:
            await self._save_session(self._session)
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception(
                "save_session callback raised for agent=%r session=%r; "
                "continuing drain loop",
                self._session.agent.id, self._session.key,
            )


__all__ = [
    "AgentScheduler",
    "DEFAULT_AGENT_CONCURRENCY",
    "PromptDispatcher",
    "SessionSaver",
]
