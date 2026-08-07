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
  whatever prompt shape is appropriate. Production supplies the inbox
  prompt dispatcher; tests and intermediate integrations can pass any
  async callable.

- **Status transitions on notifications.**  Claiming focused work
  and completing or parking it is the agent's job (via the focused
  inbox tools).  The scheduler only reads ``prompt_pending()`` to
  decide whether a round is needed; it never mutates notification
  state itself.

- **Progress guarantees (optional).**  By default the scheduler
  enforces an N-strikes forward-progress policy: after
  :data:`DEFAULT_PROGRESS_STRIKES` consecutive rounds in which no
  inbox item transitioned out of pending/in_progress, the scheduler
  invokes an injected :data:`ProgressEvictor` to evict the oldest
  item.  The default evictor is :func:`default_progress_evictor`,
  which marks the oldest item ``errored`` via the dispatch machinery
  (so any RSVP recipient is notified).  Tests and alternative
  integrations can pass ``progress_evictor=None`` to disable the
  guard, matching the previous behaviour of looping forever on a
  non-making-progress session.  ``progress_strikes`` controls the
  threshold; a value of 0 disables the guard as well.

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

:class:`~thorn.core.errors.ProviderUnavailableError` is treated
specially: it signals that the LLM provider is unreachable /
unresponsive, which is not attributable to any particular session
or agent.  The driver does *not* increment its forward-progress
strike counter on this exception, so a provider outage cannot
cause notifications to be evicted-as-errored.  The driver
additionally sleeps for
:data:`DEFAULT_PROVIDER_UNAVAILABLE_BACKOFF` (plus jitter) before
looping back, releasing the agent-level concurrency slot for the
duration of the wait so a dead provider cannot starve other
schedulers sharing the same cap.

When a :class:`~thorn.runtime.ProviderHealthMonitor` is wired in
(via the ``health_monitor`` constructor argument), each driver
also awaits :meth:`ProviderHealthMonitor.wait_until_healthy`
*before* calling the dispatcher.  That gates every session in
the gateway behind the shared circuit-breaker state, so a
provider outage observed by one session pauses every other
session sharing the monitor until either the provider recovers
or the session in question is selected as the next probe.  Round
outcomes are reported back to the monitor: a normal completion
calls :meth:`ProviderHealthMonitor.report_success`, a
:class:`~thorn.core.errors.ProviderUnavailableError` calls
:meth:`ProviderHealthMonitor.report_failure`.  Other exceptions
are not reported -- they are session-/agent-attributable and
should not influence cross-session health.

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
import os
import random
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from thorn.core.errors import LoopNoProgressError, ProviderUnavailableError
from thorn.runtime._dispatch import DispatchError, apply_handling_transition
from thorn.runtime._inbox import SessionInbox
from thorn.runtime._notification import NotificationStatus
from thorn.runtime._provider_health import ProviderHealthMonitor
from thorn.runtime._provider_state import PROVIDER_UNAVAILABLE_METADATA_KEY

if TYPE_CHECKING:
    from thorn.core._agent import Agent
    from thorn.core._session import Session
    from thorn.runtime._address import AddressBook
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
agent makes during that prompt may mutate the inbox via the focused
inbox tools; the scheduler will re-check the inbox
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

ProgressEvictor = Callable[[SessionInbox], Awaitable[None]]
"""Callable invoked when a session stalls for :data:`DEFAULT_PROGRESS_STRIKES`
consecutive rounds.

The evictor is responsible for unblocking the session by removing at
least one item from ``prompt_pending()``; the canonical
implementation is :func:`default_progress_evictor`, which marks the
oldest item ``errored`` via the dispatch machinery so any RSVP
recipient is notified.

Exceptions raised by the evictor are logged by the scheduler and
the strike counter is left untouched, so the next round retries.
"""


DEFAULT_AGENT_CONCURRENCY = 3
"""Default agent-level concurrency cap.

Three is a conservative baseline that permits useful cross-session work
without allowing one agent to fan out without bound. Gateway-level
configuration and per-agent overrides are plumbed in by the
``Runtime`` / ``Gateway`` layers; this scheduler module just picks
the value up from its constructor.
"""

DEFAULT_PROGRESS_STRIKES = 3
"""Default N-strikes threshold for the forward-progress guarantee.

A session that completes this many rounds in a row without any inbox
item transitioning out of pending/in_progress has its oldest item
evicted (marked ``errored``).  The counter resets as soon as progress
is observed or an eviction succeeds.

A value of 0 (or a ``None`` evictor) disables the guarantee; the
scheduler will then loop indefinitely on stuck sessions, which is
sometimes useful for tests and for integrations that want to supply
their own progress policy out of band.
"""


DEFAULT_PROVIDER_UNAVAILABLE_BACKOFF = 30.0
"""Baseline post-round sleep when a round exits with a provider outage.

When the dispatcher exits with :class:`ProviderUnavailableError`,
the driver sleeps this many seconds (plus a uniform jitter of up
to :data:`DEFAULT_PROVIDER_UNAVAILABLE_BACKOFF_JITTER`) before
looping back to the top and trying again.

Kept above the inner retry cap so that a session that has already
spent its per-call transient/rate-limit budget does not immediately
re-enter the same retry spiral.  Configurable via the
``THORN_PROVIDER_UNAVAILABLE_BACKOFF`` env var for operators who
need to tune the post-outage cooldown.
"""


DEFAULT_PROVIDER_UNAVAILABLE_BACKOFF_JITTER = 15.0
"""Random jitter (seconds) added on top of the provider-outage backoff.

Exists for the usual thundering-herd reason: when a provider
recovers, we do not want every stalled session on the gateway to
come back in lockstep and immediately knock it over again.
Configurable via ``THORN_PROVIDER_UNAVAILABLE_BACKOFF_JITTER``.
"""


def _provider_unavailable_backoff() -> float:
    """Return the post-outage backoff delay for the current process.

    Reads ``THORN_PROVIDER_UNAVAILABLE_BACKOFF`` and
    ``THORN_PROVIDER_UNAVAILABLE_BACKOFF_JITTER`` at call time so
    tests can monkeypatch the environment per-case without
    restarting the interpreter.  The result is
    ``base + uniform(0, jitter)``, clamped to a non-negative
    value.
    """
    base_raw = os.environ.get("THORN_PROVIDER_UNAVAILABLE_BACKOFF")
    jitter_raw = os.environ.get("THORN_PROVIDER_UNAVAILABLE_BACKOFF_JITTER")
    try:
        base = float(base_raw) if base_raw else DEFAULT_PROVIDER_UNAVAILABLE_BACKOFF
    except ValueError:
        base = DEFAULT_PROVIDER_UNAVAILABLE_BACKOFF
    try:
        jitter = (
            float(jitter_raw) if jitter_raw
            else DEFAULT_PROVIDER_UNAVAILABLE_BACKOFF_JITTER
        )
    except ValueError:
        jitter = DEFAULT_PROVIDER_UNAVAILABLE_BACKOFF_JITTER
    base = max(0.0, base)
    jitter = max(0.0, jitter)
    return base + random.uniform(0, jitter)


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
        progress_strikes: int = DEFAULT_PROGRESS_STRIKES,
        progress_evictor: ProgressEvictor | None = None,
        health_monitor: ProviderHealthMonitor | None = None,
    ) -> None:
        if concurrency < 1:
            raise ValueError(
                f"AgentScheduler concurrency must be >= 1, got {concurrency!r}"
            )
        if progress_strikes < 0:
            raise ValueError(
                "AgentScheduler progress_strikes must be >= 0, got "
                f"{progress_strikes!r}"
            )
        self._agent = agent
        self._dispatcher = prompt_dispatcher
        self._save_session = save_session
        self._concurrency = concurrency
        self._progress_strikes = progress_strikes
        self._progress_evictor = progress_evictor
        self._health_monitor = health_monitor
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
            if driver is not None and driver.is_done:
                log.warning(
                    "AgentScheduler driver for agent=%r session=%r had "
                    "stopped unexpectedly; replacing it for new work",
                    self._agent.id, session.key,
                )
                driver = None
            if driver is None:
                driver = _SessionDriver(
                    session=session,
                    inbox=inbox,
                    semaphore=self._semaphore,
                    dispatcher=self._dispatcher,
                    save_session=self._save_session,
                    progress_strikes=self._progress_strikes,
                    progress_evictor=self._progress_evictor,
                    health_monitor=self._health_monitor,
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
        "_progress_strikes", "_progress_evictor", "_health_monitor",
        "_stall_count", "_wake", "_task", "_stop",
    )

    def __init__(
        self,
        *,
        session: "Session",
        inbox: SessionInbox,
        semaphore: asyncio.Semaphore,
        dispatcher: PromptDispatcher,
        save_session: SessionSaver | None,
        progress_strikes: int,
        progress_evictor: ProgressEvictor | None,
        health_monitor: ProviderHealthMonitor | None = None,
    ) -> None:
        self._session = session
        self._inbox = inbox
        self._semaphore = semaphore
        self._dispatcher = dispatcher
        self._save_session = save_session
        self._progress_strikes = progress_strikes
        self._progress_evictor = progress_evictor
        self._health_monitor = health_monitor
        self._stall_count = 0
        self._wake = asyncio.Event()
        self._task: asyncio.Task[None] | None = None
        self._stop = False

    @property
    def task(self) -> asyncio.Task[None] | None:
        return self._task

    @property
    def is_done(self) -> bool:
        return self._task is not None and self._task.done()

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
        """Invoke the dispatcher once under the agent-level cap.

        After the dispatcher returns (or raises a non-cancellation
        exception), updates the forward-progress strike counter
        based on whether any item transitioned out of prompt-pending
        view during the round.  When the counter reaches the
        configured threshold and an evictor is installed, the
        evictor is invoked to unblock the session.

        Progress is defined as *at least one item* from the set of
        pre-round prompt-pending IDs being absent from the post-round
        set.  Additions to the inbox during the round do not count as
        progress -- only closures do -- because the guarantee is
        specifically about the session's ability to clear the items
        it was already shown.  A dispatcher that raises an "ordinary"
        exception (anything other than
        :class:`ProviderUnavailableError`) is treated as making no
        progress; a dispatcher that raises
        :class:`ProviderUnavailableError` is *not* treated as a
        strike against the session, because the failure is not
        attributable to the session's content or behavior.

        When a :class:`ProviderHealthMonitor` is wired in, the
        round first awaits :meth:`ProviderHealthMonitor.wait_until_healthy`
        *before* acquiring the agent's concurrency semaphore.  This
        avoids tying up an agent slot while the monitor's cooldown
        elapses, and it means that during a degraded period the
        round only proceeds when either the provider is believed
        healthy again or this driver was nominated as the next
        probe.  After the dispatcher returns, the round reports
        the outcome to the monitor: a clean return calls
        :meth:`ProviderHealthMonitor.report_success`, a
        :class:`ProviderUnavailableError` calls
        :meth:`ProviderHealthMonitor.report_failure`, and any other
        exception is *not* reported (it is session-attributable, not
        provider-attributable).
        """
        # Wait for the monitor outside the semaphore so a degraded
        # provider does not pin agent-level concurrency slots while
        # the cooldown elapses.  Returns immediately when the
        # monitor is healthy or when this caller is the next probe.
        if self._health_monitor is not None:
            await self._health_monitor.wait_until_healthy()

        dispatcher_raised = False
        provider_unavailable = False
        provider_unavailable_error: ProviderUnavailableError | None = None
        force_progress_eviction = False
        async with self._semaphore:
            before_ids = frozenset(
                item.id for item in self._inbox.prompt_pending()
            )
            try:
                await self._dispatcher(self._session, self._inbox)
            except asyncio.CancelledError:
                # Shutdown tore through the dispatcher.  Propagate
                # so the drain task ends.
                raise
            except ProviderUnavailableError as exc:
                # The LLM provider was unreachable for the whole
                # inner retry budget.  Do not blame the session:
                # leave the strike counter alone, then persist
                # provider-unavailable metadata after the monitor has
                # observed the failure.
                log.warning(
                    "Prompt dispatcher reported provider unavailable for "
                    "agent=%r session=%r after %d attempt(s); not counting "
                    "as a stall. Reason: %s",
                    self._session.agent.id, self._session.key,
                    exc.attempts, exc,
                )
                dispatcher_raised = True
                provider_unavailable = True
                provider_unavailable_error = exc
            except LoopNoProgressError as exc:
                log.warning(
                    "Prompt dispatcher reported no progress for agent=%r "
                    "session=%r after %d tool round(s): %s",
                    self._session.agent.id, self._session.key,
                    exc.rounds, exc,
                )
                dispatcher_raised = True
                force_progress_eviction = True
            except Exception:
                log.exception(
                    "Prompt dispatcher raised for agent=%r session=%r; "
                    "leaving inbox state untouched and looping",
                    self._session.agent.id, self._session.key,
                )
                dispatcher_raised = True

            after_ids = frozenset(
                item.id for item in self._inbox.prompt_pending()
            )
            closed_out = before_ids - after_ids
            if closed_out:
                self._stall_count = 0
            elif not provider_unavailable:
                # A provider outage means the session never got a
                # real chance to make progress -- do not advance
                # the strike counter toward eviction.
                self._stall_count += 1
                if force_progress_eviction:
                    self._stall_count = max(
                        self._stall_count,
                        self._progress_strikes,
                    )

            await self._maybe_evict_for_progress()

        if not dispatcher_raised:
            self._clear_provider_unavailable_metadata()
        if not dispatcher_raised and self._save_session is not None:
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

        # Report round outcome to the health monitor (when wired).
        # We deliberately treat any non-provider-unavailable exception
        # as "do not report": those are session-/agent-attributable
        # and should not influence cross-session pacing.  A clean
        # return is treated as a successful real call -- in production
        # the prompt dispatcher always issues at least one LLM round
        # trip, so a clean return is reliable evidence the provider
        # responded.
        if self._health_monitor is not None:
            if provider_unavailable:
                await self._health_monitor.report_failure()
            elif not dispatcher_raised:
                await self._health_monitor.report_success()

        if provider_unavailable_error is not None:
            self._record_provider_unavailable_metadata(provider_unavailable_error)
            await self._save_provider_unavailable_metadata()

        # Provider-unavailable cooldown is deliberately outside the
        # semaphore block so a stuck provider does not hold the
        # agent-level concurrency slot and starve sibling sessions.
        # When a monitor is wired we skip this per-driver sleep:
        # the monitor's wait_until_healthy at the top of the next
        # round is now the cross-session-coordinated equivalent of
        # this back-off, so doing both would double-pace the driver
        # and lose the coordination benefit.
        if provider_unavailable and self._health_monitor is None:
            delay = _provider_unavailable_backoff()
            log.info(
                "Provider-unavailable backoff for agent=%r session=%r: "
                "sleeping %.1fs before next round",
                self._session.agent.id, self._session.key, delay,
            )
            try:
                await asyncio.sleep(delay)
            except asyncio.CancelledError:
                raise

    def _clear_provider_unavailable_metadata(self) -> None:
        self._session.metadata.pop(PROVIDER_UNAVAILABLE_METADATA_KEY, None)

    def _record_provider_unavailable_metadata(
        self,
        exc: ProviderUnavailableError,
    ) -> None:
        snapshot = (
            self._health_monitor.snapshot()
            if self._health_monitor is not None else None
        )
        payload: dict[str, Any] = {
            "state": "waiting_on_provider",
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "attempts": exc.attempts,
            "reason": str(exc),
        }
        if snapshot is not None:
            payload["provider_health"] = {
                "state": snapshot.state.value,
                "recent_failure_count": snapshot.recent_failure_count,
                "seconds_until_probe": snapshot.seconds_until_probe,
                "probe_in_flight": snapshot.probe_in_flight,
                "consecutive_probe_failures": (
                    snapshot.consecutive_probe_failures
                ),
            }
        self._session.metadata[PROVIDER_UNAVAILABLE_METADATA_KEY] = payload

    async def _save_provider_unavailable_metadata(self) -> None:
        if self._save_session is None:
            return
        try:
            await self._save_session(self._session)
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception(
                "save_session callback raised while recording provider "
                "unavailable state for agent=%r session=%r; continuing "
                "drain loop",
                self._session.agent.id, self._session.key,
            )

    async def _maybe_evict_for_progress(self) -> None:
        """Invoke the installed evictor if the strike threshold has been hit.

        No-ops when the guarantee is disabled (threshold of 0 or no
        installed evictor) or the counter is below the threshold.
        If the evictor raises, the error is logged and the strike
        counter is intentionally left in its elevated state so the
        next round retries eviction; if the evictor returns
        normally, the counter resets to give the session a fresh
        attempt on its remaining items.
        """
        if self._progress_evictor is None:
            return
        if self._progress_strikes <= 0:
            return
        if self._stall_count < self._progress_strikes:
            return
        log.warning(
            "Progress guarantee: session agent=%r key=%r has stalled for %d "
            "rounds; invoking evictor",
            self._session.agent.id, self._session.key, self._stall_count,
        )
        try:
            await self._progress_evictor(self._inbox)
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception(
                "Progress evictor raised for agent=%r session=%r; "
                "strike counter left at %d and will retry next round",
                self._session.agent.id, self._session.key, self._stall_count,
            )
            return
        self._stall_count = 0


# ---------------------------------------------------------------------------
# Default evictor
# ---------------------------------------------------------------------------

_EVICTION_REASON = (
    "Evicted by progress guarantee: this session did not close out any "
    "inbox items across multiple prompt rounds."
)
"""Canned ``error_reason`` attached to evicted notifications.

Kept as a module-level constant rather than f-stringed with the
strike count because the count is a scheduler-level policy knob, not
information intrinsic to this item's failure.  Operators diagnosing
a stuck session should look at logs for the exact count."""


def default_progress_evictor(
    address_book: "AddressBook",
    *,
    health_monitor: ProviderHealthMonitor | None = None,
) -> ProgressEvictor:
    """Build a :data:`ProgressEvictor` that errors the oldest inbox item.

    The returned callable, when invoked with a :class:`SessionInbox`,
    marks the inbox's oldest remaining pending/in_progress item as
    ``errored`` via :func:`apply_handling_transition`.  That routes
    the item through the normal handling-dispatch machinery, so:

    - items with an RSVP target are forwarded to that target with
      the canned eviction reason as ``error_reason``;
    - items without an RSVP are moved to the inbox's ``errored/``
      directory for operator inspection.

    When the inbox is unexpectedly empty (e.g. a concurrent handler
    drained it between the scheduler's check and the evictor's
    invocation), the evictor returns without action.

    A :class:`DispatchError` from step 2 of dispatch is swallowed
    (with a warning): step 1 has already persisted the errored
    status, so the item is gone from ``prompt_pending()`` and the
    stall is resolved; the startup sweep will reconcile the
    abandoned step 2 on the next runtime entry.

    Monitor-awareness: when ``health_monitor`` is supplied and its
    state is not :attr:`ProviderHealthState.HEALTHY`, the evictor
    no-ops with a debug log line.  This is defense in depth on top
    of the scheduler's per-driver suppression of stall-counter
    increments during ``ProviderUnavailableError``: even if some
    other code path elevated the stall counter (e.g. an unrelated
    exception that happened to coincide with a degraded provider),
    we still suppress eviction while the provider is known to be
    unhealthy.  Operators see no notifications evicted-as-errored
    purely because the LLM was down.
    """

    async def _evict(inbox: SessionInbox) -> None:
        if health_monitor is not None and not health_monitor.is_healthy:
            # Defense in depth: never evict while the provider is
            # known degraded.  The session driver already declines
            # to advance the strike counter on
            # ProviderUnavailableError, but other exception paths
            # could in principle elevate the counter while a
            # provider outage is in progress; this guard ensures we
            # never blame a session for the gateway-wide outage.
            log.info(
                "default_progress_evictor: provider monitor reports %s; "
                "skipping eviction on inbox %r until provider recovers",
                health_monitor.state.value, inbox.address,
            )
            return
        pending = inbox.prompt_pending()
        if not pending:
            log.debug(
                "default_progress_evictor: inbox %r is empty; nothing to evict",
                inbox.address,
            )
            return
        oldest = pending[0]
        log.warning(
            "default_progress_evictor: marking oldest item %s on inbox %r "
            "as errored to restore forward progress",
            oldest.id, inbox.address,
        )
        try:
            apply_handling_transition(
                inbox,
                oldest.id,
                NotificationStatus.ERRORED,
                address_book=address_book,
                error_reason=_EVICTION_REASON,
            )
        except DispatchError as exc:
            log.warning(
                "default_progress_evictor: step-2 dispatch failed for "
                "evicted item %s: %s; status has been recorded and the "
                "next startup sweep will reconcile",
                oldest.id, exc,
            )

    return _evict


__all__ = [
    "AgentScheduler",
    "DEFAULT_AGENT_CONCURRENCY",
    "DEFAULT_PROGRESS_STRIKES",
    "DEFAULT_PROVIDER_UNAVAILABLE_BACKOFF",
    "DEFAULT_PROVIDER_UNAVAILABLE_BACKOFF_JITTER",
    "PromptDispatcher",
    "ProgressEvictor",
    "SessionSaver",
    "default_progress_evictor",
]
