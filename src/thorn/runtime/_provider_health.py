"""Gateway-wide LLM provider health monitor (circuit breaker).

The :class:`ProviderHealthMonitor` is a process-wide circuit
breaker around the LLM provider.  Its job is to coordinate every
session's reaction to a provider outage so that:

- Sessions do not each independently spend a full per-call retry
  budget on the same outage; once one session has reported the
  provider as unavailable, others wait until the monitor believes
  it is back rather than piling on.
- A single "probe" attempt is allowed when the cooldown elapses,
  so we can detect recovery without hitting the recovering provider
  with a thundering herd.
- The forward-progress evictor (see
  :func:`~thorn.runtime._scheduler.default_progress_evictor`) does
  not punish sessions for stalls accrued while the provider was
  known to be degraded.

Why a state machine rather than a simple "is healthy" boolean
-------------------------------------------------------------

A boolean would race: in the moment between a successful call and
recovery being noted, a second session could observe ``healthy``
and try to attempt a real call before we know if the recovery
holds.  Modelling the transition explicitly via Healthy / Degraded
states with an in-flight probe slot lets us serialize that first
recovery attempt and gate everyone else on its outcome.

Design choices worth flagging
-----------------------------

- **Sliding-window failure threshold.**  We trip into ``DEGRADED``
  when we see ``failure_threshold`` failures within a rolling
  ``failure_window_seconds`` window.  An older "total failure
  count" approach would punish long-running daemons that
  accumulate the occasional blip across days; the windowed
  approach keeps the trip threshold meaningful regardless of
  uptime.

- **Exponential cooldown growth, capped + jittered.**  Each probe
  failure doubles the cooldown (subject to ``max_cooldown_seconds``)
  to avoid hammering an upstream that is genuinely down.  A small
  uniform jitter on top spreads probes from sibling gateway
  instances or restarted processes that would otherwise come back
  in lockstep.

- **At most one probe in flight.**  The first ``wait_until_healthy``
  caller after the cooldown elapses takes the probe slot and
  returns; everyone else keeps waiting.  This keeps the recovery
  attempt singular even when many sessions are queued up behind
  the monitor.  If the probe succeeds (``report_success``) the
  monitor flips to Healthy and releases all waiters; if it fails
  (``report_failure``) we grow the cooldown and the next available
  prober gets the next slot.

- **Synchronous reporters do not exist.**  Both ``report_success``
  and ``report_failure`` are coroutine methods.  They acquire a
  short-lived asyncio lock to mutate state; making them sync would
  force callers to either schedule the work as a task (losing
  ordering) or accept blocking on a thread lock that the rest of
  the runtime does not use.
"""

from __future__ import annotations

import asyncio
import logging
import os
import random
import time as time_mod
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Deque

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# State enum
# ---------------------------------------------------------------------------

class ProviderHealthState(str, Enum):
    """Current believed state of the LLM provider.

    We deliberately keep the visible state-space small: callers
    that need finer detail (e.g. "degraded for how long?") use the
    :class:`ProviderHealthSnapshot` accessor instead of branching
    on the enum.
    """

    HEALTHY = "healthy"
    """No reason to suspect the provider is down."""

    DEGRADED = "degraded"
    """Recent failures pushed us across the threshold; sessions
    that ask :meth:`ProviderHealthMonitor.wait_until_healthy` will
    block until either we believe the provider is back or they are
    nominated as the next probe."""


# ---------------------------------------------------------------------------
# Defaults (env-tunable)
# ---------------------------------------------------------------------------

_DEFAULT_FAILURE_THRESHOLD = 3
_DEFAULT_FAILURE_WINDOW_SECONDS = 120.0
_DEFAULT_BASE_COOLDOWN_SECONDS = 30.0
_DEFAULT_MAX_COOLDOWN_SECONDS = 300.0
_DEFAULT_COOLDOWN_JITTER_SECONDS = 15.0


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        log.warning(
            "Ignoring %s=%r: not a float; using default %r",
            name, raw, default,
        )
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        log.warning(
            "Ignoring %s=%r: not an int; using default %r",
            name, raw, default,
        )
        return default


# ---------------------------------------------------------------------------
# Snapshot (observability accessor)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ProviderHealthSnapshot:
    """Immutable view of the monitor's current state.

    Returned by :meth:`ProviderHealthMonitor.snapshot` for status
    pages, log lines, and tests.  Times are reported in seconds
    relative to the monitor's clock (``time.monotonic`` by
    default), so they are useful for "how long until the next
    probe?" arithmetic but not for absolute wall-clock display.
    """

    state: ProviderHealthState
    recent_failure_count: int
    """Number of failures within the current sliding window."""
    seconds_until_probe: float
    """How long callers must still wait before another probe slot
    opens.  Always ``0`` when ``state`` is ``HEALTHY``."""
    probe_in_flight: bool
    """True iff a session is currently acting as the recovery
    probe.  Other waiters are blocked on its outcome."""
    consecutive_probe_failures: int
    """Number of consecutive probe failures since the monitor
    entered ``DEGRADED``.  Resets on a successful probe."""


# ---------------------------------------------------------------------------
# Monitor
# ---------------------------------------------------------------------------

class ProviderHealthMonitor:
    """Process-wide circuit breaker for LLM provider health.

    Construct one per gateway and share it across every
    :class:`~thorn.runtime.AgentScheduler`.  Schedulers call
    :meth:`wait_until_healthy` before each prompt round and
    :meth:`report_success` / :meth:`report_failure` based on the
    outcome.  Monitors are safe to share across asyncio tasks but
    are not designed for cross-thread / cross-process use; one per
    process is the intended pattern.

    Parameters mirror the :class:`ProviderHealthSnapshot` fields
    plus rate-limit constants:

    Args:
        failure_threshold: Number of failures within
            ``failure_window_seconds`` required to trip from
            ``HEALTHY`` to ``DEGRADED``.  Higher values are more
            tolerant of intermittent blips.
        failure_window_seconds: Width of the sliding window over
            which failures accumulate.  Failures older than this
            no longer count toward the trip threshold.
        base_cooldown_seconds: Initial cooldown after entering
            ``DEGRADED``; the time before the first probe slot
            opens.
        max_cooldown_seconds: Upper bound on the cooldown after
            exponential growth.  Prevents run-away wait times
            during long outages.
        cooldown_jitter_seconds: Maximum uniform jitter added on
            top of the computed cooldown.  Spreads probe attempts
            from sibling clients that would otherwise probe
            simultaneously.
        clock: Monotonic clock used for timestamps and cooldown
            arithmetic.  Defaults to :func:`time.monotonic`; tests
            inject deterministic clocks.

    All numeric parameters can be overridden via the
    ``THORN_PROVIDER_HEALTH_*`` environment variables on
    :meth:`from_env`-constructed monitors (operators tune these
    without redeploying).
    """

    def __init__(
        self,
        *,
        failure_threshold: int = _DEFAULT_FAILURE_THRESHOLD,
        failure_window_seconds: float = _DEFAULT_FAILURE_WINDOW_SECONDS,
        base_cooldown_seconds: float = _DEFAULT_BASE_COOLDOWN_SECONDS,
        max_cooldown_seconds: float = _DEFAULT_MAX_COOLDOWN_SECONDS,
        cooldown_jitter_seconds: float = _DEFAULT_COOLDOWN_JITTER_SECONDS,
        clock: Callable[[], float] = time_mod.monotonic,
    ) -> None:
        if failure_threshold < 1:
            raise ValueError(
                "failure_threshold must be >= 1, got "
                f"{failure_threshold!r}",
            )
        if failure_window_seconds <= 0:
            raise ValueError(
                "failure_window_seconds must be > 0, got "
                f"{failure_window_seconds!r}",
            )
        if base_cooldown_seconds < 0:
            raise ValueError(
                "base_cooldown_seconds must be >= 0, got "
                f"{base_cooldown_seconds!r}",
            )
        if max_cooldown_seconds < base_cooldown_seconds:
            raise ValueError(
                "max_cooldown_seconds must be >= base_cooldown_seconds; "
                f"got {max_cooldown_seconds!r} < {base_cooldown_seconds!r}",
            )
        if cooldown_jitter_seconds < 0:
            raise ValueError(
                "cooldown_jitter_seconds must be >= 0, got "
                f"{cooldown_jitter_seconds!r}",
            )

        self._failure_threshold = failure_threshold
        self._failure_window_seconds = failure_window_seconds
        self._base_cooldown_seconds = base_cooldown_seconds
        self._max_cooldown_seconds = max_cooldown_seconds
        self._cooldown_jitter_seconds = cooldown_jitter_seconds
        self._clock = clock

        self._state = ProviderHealthState.HEALTHY
        self._failure_times: Deque[float] = deque()
        self._cooldown_until: float = 0.0
        self._consecutive_probe_failures: int = 0
        self._probe_in_flight: bool = False
        # ``_wake_event`` is replaced (not just cleared) on every
        # state transition so that waiters captured under the old
        # event always observe the wake even if a fresh waiter
        # arrives between us setting the event and replacing it.
        # Lock-protected; see ``_replace_wake_event``.
        self._wake_event: asyncio.Event = asyncio.Event()
        self._wake_event.set()
        self._lock = asyncio.Lock()

    # -- factories ---------------------------------------------------------

    @classmethod
    def from_env(
        cls,
        *,
        clock: Callable[[], float] = time_mod.monotonic,
    ) -> "ProviderHealthMonitor":
        """Construct a monitor configured from environment variables.

        Recognised variables (all optional):

        - ``THORN_PROVIDER_HEALTH_FAILURE_THRESHOLD`` (int)
        - ``THORN_PROVIDER_HEALTH_FAILURE_WINDOW`` (float seconds)
        - ``THORN_PROVIDER_HEALTH_BASE_COOLDOWN`` (float seconds)
        - ``THORN_PROVIDER_HEALTH_MAX_COOLDOWN`` (float seconds)
        - ``THORN_PROVIDER_HEALTH_COOLDOWN_JITTER`` (float seconds)
        """
        return cls(
            failure_threshold=_env_int(
                "THORN_PROVIDER_HEALTH_FAILURE_THRESHOLD",
                _DEFAULT_FAILURE_THRESHOLD,
            ),
            failure_window_seconds=_env_float(
                "THORN_PROVIDER_HEALTH_FAILURE_WINDOW",
                _DEFAULT_FAILURE_WINDOW_SECONDS,
            ),
            base_cooldown_seconds=_env_float(
                "THORN_PROVIDER_HEALTH_BASE_COOLDOWN",
                _DEFAULT_BASE_COOLDOWN_SECONDS,
            ),
            max_cooldown_seconds=_env_float(
                "THORN_PROVIDER_HEALTH_MAX_COOLDOWN",
                _DEFAULT_MAX_COOLDOWN_SECONDS,
            ),
            cooldown_jitter_seconds=_env_float(
                "THORN_PROVIDER_HEALTH_COOLDOWN_JITTER",
                _DEFAULT_COOLDOWN_JITTER_SECONDS,
            ),
            clock=clock,
        )

    # -- read-only accessors ----------------------------------------------

    @property
    def state(self) -> ProviderHealthState:
        """Current monitor state.

        Reads are not lock-protected; callers that need a coherent
        view across multiple fields should use :meth:`snapshot`.
        """
        return self._state

    @property
    def is_healthy(self) -> bool:
        """``True`` iff :meth:`state` is :attr:`ProviderHealthState.HEALTHY`."""
        return self._state == ProviderHealthState.HEALTHY

    def snapshot(self) -> ProviderHealthSnapshot:
        """Return an immutable view of the current state."""
        now = self._clock()
        seconds_until_probe = max(0.0, self._cooldown_until - now)
        if self._state == ProviderHealthState.HEALTHY:
            seconds_until_probe = 0.0
        return ProviderHealthSnapshot(
            state=self._state,
            recent_failure_count=len(self._failure_times),
            seconds_until_probe=seconds_until_probe,
            probe_in_flight=self._probe_in_flight,
            consecutive_probe_failures=self._consecutive_probe_failures,
        )

    # -- coordination -----------------------------------------------------

    async def wait_until_healthy(self) -> None:
        """Block until the monitor permits the caller to attempt a round.

        Returns immediately when the state is ``HEALTHY``.  In
        ``DEGRADED``, the caller waits until either:

        - the state transitions back to ``HEALTHY`` (e.g. another
          session probed successfully), or
        - the cooldown elapses and no other probe is in flight, in
          which case this caller is nominated as the probe and
          returned to so it can attempt the next call.

        Cancellation:  the call is fully cooperative; an
        :class:`asyncio.CancelledError` while waiting is propagated
        to the caller after releasing any internal resources.  A
        cancelled probe-claimant does not deadlock the monitor: the
        probe slot is only marked occupied at the moment of
        return-to-caller, after which the caller is responsible for
        eventually calling :meth:`report_success` /
        :meth:`report_failure`.  If the caller never reports
        because of cancellation, the next ``wait_until_healthy``
        observes the timeout has elapsed and claims a fresh probe
        slot; we deliberately do not hold ownership across awaits
        between state acquisition and ``report_*``.
        """
        while True:
            async with self._lock:
                if self._state == ProviderHealthState.HEALTHY:
                    return
                now = self._clock()
                if (
                    not self._probe_in_flight
                    and now >= self._cooldown_until
                ):
                    # Nominate this caller as the recovery probe.
                    # The slot stays held until report_*; we do not
                    # auto-release on caller cancellation because
                    # the caller is expected to either report or be
                    # promptly torn down (in which case the next
                    # wait_until_healthy will reclaim the slot once
                    # the cooldown re-elapses).
                    self._probe_in_flight = True
                    return
                # Capture the current event before releasing the
                # lock so we observe wakes that arrive after this
                # snapshot.  ``_wake_event`` is replaced by every
                # state-changing operation, so a wake racing our
                # release sets THIS event, not the next one.
                event = self._wake_event
                if self._probe_in_flight:
                    # A probe is in flight; wait without a deadline
                    # for it to report back.  Using a zero (or
                    # near-zero) timeout here would busy-spin
                    # because the cooldown has already elapsed --
                    # the only meaningful wake while a probe is in
                    # flight is the report_success / report_failure
                    # call itself.
                    timeout: float | None = None
                else:
                    # Cooldown is still ticking: wait for either an
                    # explicit wake or for the cooldown to elapse.
                    timeout = max(0.0, self._cooldown_until - now)
            try:
                await asyncio.wait_for(event.wait(), timeout=timeout)
            except asyncio.TimeoutError:
                # Cooldown elapsed; loop back and try to claim the
                # probe slot under the lock.
                pass

    # -- reporting --------------------------------------------------------

    async def report_success(self) -> None:
        """Record that a real prompt round completed successfully.

        Effects depend on current state:

        - In ``HEALTHY``: clears the failure window so a long-ago
          blip plus one fresh failure does not immediately trip the
          breaker.  No-op for waiters.
        - In ``DEGRADED``: transitions back to ``HEALTHY``, clears
          the failure history, resets the cooldown bookkeeping, and
          releases all waiters.  Whether the caller was the probe
          or a non-probe successful round (e.g. a slow round that
          completed after a transient blip but before another
          session reported failure) does not change the outcome --
          a successful real call is the strongest possible
          recovery signal.
        """
        async with self._lock:
            self._failure_times.clear()
            self._probe_in_flight = False
            if self._state == ProviderHealthState.DEGRADED:
                self._state = ProviderHealthState.HEALTHY
                self._consecutive_probe_failures = 0
                self._cooldown_until = 0.0
                log.info(
                    "ProviderHealthMonitor: recovered to HEALTHY",
                )
            self._replace_wake_event()

    async def report_failure(self) -> None:
        """Record a :class:`ProviderUnavailableError`-class failure.

        Effects depend on current state:

        - In ``HEALTHY``: timestamp the failure and, if it pushes
          the rolling window count to ``failure_threshold``,
          transition to ``DEGRADED`` with the base cooldown.
          Waiters that arrive during the cooldown will block.
        - In ``DEGRADED`` (this is necessarily a probe failure
          because non-probe waiters are gated by
          :meth:`wait_until_healthy`): increment the consecutive
          probe failure count and grow the cooldown
          exponentially up to ``max_cooldown_seconds``.

        In both cases, all current waiters are woken so they can
        re-evaluate.  In ``DEGRADED`` they will simply observe the
        new cooldown and go back to sleep on a fresh event.
        """
        async with self._lock:
            now = self._clock()
            self._failure_times.append(now)
            self._evict_failures_older_than(now - self._failure_window_seconds)
            self._probe_in_flight = False
            if self._state == ProviderHealthState.HEALTHY:
                if len(self._failure_times) >= self._failure_threshold:
                    self._state = ProviderHealthState.DEGRADED
                    self._consecutive_probe_failures = 0
                    cooldown = self._compute_cooldown(0)
                    self._cooldown_until = now + cooldown
                    log.warning(
                        "ProviderHealthMonitor: tripped to DEGRADED after "
                        "%d failures within %.0fs; first probe in %.1fs",
                        len(self._failure_times),
                        self._failure_window_seconds,
                        cooldown,
                    )
            else:
                self._consecutive_probe_failures += 1
                cooldown = self._compute_cooldown(
                    self._consecutive_probe_failures,
                )
                self._cooldown_until = now + cooldown
                log.warning(
                    "ProviderHealthMonitor: probe #%d failed; next probe "
                    "in %.1fs",
                    self._consecutive_probe_failures, cooldown,
                )
            self._replace_wake_event()

    # -- internals --------------------------------------------------------

    def _evict_failures_older_than(self, cutoff: float) -> None:
        """Drop failure timestamps that have aged out of the window.

        Called only under ``self._lock``.  Walks from the left of
        the deque since failures are appended in monotonic-clock
        order.
        """
        failures = self._failure_times
        while failures and failures[0] < cutoff:
            failures.popleft()

    def _compute_cooldown(self, attempt: int) -> float:
        """Return the next cooldown duration in seconds.

        Uses ``base * 2**attempt``, capped at ``max_cooldown_seconds``,
        plus a uniform jitter of up to ``cooldown_jitter_seconds``.
        Both inputs are clamped at 0 to keep the result non-negative
        even when an operator misconfigures the env-var floats.
        """
        capped = min(
            self._base_cooldown_seconds * (2 ** attempt),
            self._max_cooldown_seconds,
        )
        capped = max(0.0, capped)
        if self._cooldown_jitter_seconds > 0:
            jitter = random.uniform(0, self._cooldown_jitter_seconds)
        else:
            jitter = 0.0
        return capped + jitter

    def _replace_wake_event(self) -> None:
        """Wake all waiters and install a fresh event.

        Setting+replacing rather than just clearing ensures that
        any waiter that had captured the old reference is woken
        exactly once and that new waiters arriving after the
        transition use the new (initially-unset) event.  Called
        only under ``self._lock``.
        """
        old = self._wake_event
        self._wake_event = asyncio.Event()
        old.set()


__all__ = [
    "ProviderHealthMonitor",
    "ProviderHealthSnapshot",
    "ProviderHealthState",
]
