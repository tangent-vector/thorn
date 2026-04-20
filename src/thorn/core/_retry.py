"""Bounded retry utilities."""

from __future__ import annotations

import os
import random
from collections.abc import Iterator
from dataclasses import dataclass

from thorn.core.errors import LoopLimitError


def bound_retries(max_attempts: int = 3) -> Iterator[int]:
    """Yield attempt indices ``0 .. max_attempts-1``, then raise.

    Use with ``for``/``break`` -- ``break`` out on success and the
    generator exits cleanly.  If the loop exhausts all attempts without
    a ``break``, :class:`LoopLimitError` is raised automatically::

        for attempt in bound_retries(3):
            do_work()
            if validate():
                break  # success -- no exception

        # If we get here without breaking, LoopLimitError is raised.
    """
    for i in range(max_attempts):
        yield i
    raise LoopLimitError(
        f"Failed after {max_attempts} attempts",
        rounds=max_attempts,
    )


# ---------------------------------------------------------------------------
# RetryPolicy for provider-level retries
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RetryPolicy:
    """Parameters for exponential backoff with full jitter.

    The delay for attempt ``k`` (zero-based) is

        uniform(0, min(base * 2**k, cap))

    which is the "full jitter" scheme recommended by the AWS
    architecture blog for workloads where thundering-herd avoidance
    matters more than worst-case latency.  The cap prevents
    unbounded growth on long-running outages; the full-jitter
    spread avoids synchronized retries across many sessions that
    started their backoff at roughly the same wall-clock moment.

    Attributes:
        base: Seconds of baseline delay.  The ``k=0`` attempt
            waits ``uniform(0, base)``.
        cap: Upper bound on the delay, in seconds.  After enough
            doublings ``base * 2**k`` exceeds ``cap`` and is
            clamped.
        max_rate_limit_retries: Maximum number of
            :class:`RateLimitError` retries before giving up on a
            single request.
        max_transient_retries: Maximum number of
            :class:`TransientProviderError` retries before giving
            up on a single request.  Transport-level transients
            have their own budget separate from the agent-loop
            ``max_failures`` cap so that a bad network blip does
            not poison the cap meant for non-transient errors.
        retry_after_jitter: Extra random jitter (seconds) added on
            top of any server-supplied ``Retry-After`` wait, so that
            multiple clients coming off the same rate-limit window
            do not re-hit simultaneously.
    """

    base: float = 1.0
    cap: float = 60.0
    max_rate_limit_retries: int = 8
    max_transient_retries: int = 8
    retry_after_jitter: float = 1.0

    # -- factories ---------------------------------------------------------

    @classmethod
    def from_env(cls, *, prefix: str = "THORN_PROVIDER_RETRY_") -> "RetryPolicy":
        """Build a policy from ``<prefix>*`` environment variables.

        Recognised suffixes:

        - ``BASE``  (float, seconds): baseline delay.
        - ``CAP``   (float, seconds): upper bound on the delay.
        - ``MAX_RATE_LIMIT_RETRIES`` (int).
        - ``MAX_TRANSIENT_RETRIES``  (int).
        - ``RETRY_AFTER_JITTER`` (float, seconds).

        Unset variables fall back to the dataclass defaults.
        Invalid values raise :class:`ValueError` so misconfiguration
        is surfaced at startup rather than silently ignored.
        """
        return cls(
            base=_env_float(prefix + "BASE", cls.base),
            cap=_env_float(prefix + "CAP", cls.cap),
            max_rate_limit_retries=_env_int(
                prefix + "MAX_RATE_LIMIT_RETRIES",
                cls.max_rate_limit_retries,
            ),
            max_transient_retries=_env_int(
                prefix + "MAX_TRANSIENT_RETRIES",
                cls.max_transient_retries,
            ),
            retry_after_jitter=_env_float(
                prefix + "RETRY_AFTER_JITTER",
                cls.retry_after_jitter,
            ),
        )

    # -- derived values ----------------------------------------------------

    def backoff_delay(
        self,
        attempt: int,
        *,
        retry_after: float | None = None,
    ) -> float:
        """Return the number of seconds to sleep before attempt ``attempt``.

        ``attempt`` is zero-based: the first retry after an initial
        failure is ``attempt == 0``.  When ``retry_after`` is
        supplied (from a server header), the result is the maximum
        of (retry_after + small jitter, full-jitter backoff) so the
        server's minimum wait is always honoured but we still
        stagger ties between clients whose headers came back at the
        same instant.
        """
        capped = min(self.base * (2 ** attempt), self.cap)
        # Full jitter: spread uniformly in [0, capped] rather than
        # using the deterministic value so two clients that hit the
        # same outage do not retry in lockstep.
        jittered = random.uniform(0, capped) if capped > 0 else 0.0
        if retry_after is None:
            return jittered
        bounded_retry_after = max(0.0, retry_after)
        jitter = (
            random.uniform(0, self.retry_after_jitter)
            if self.retry_after_jitter > 0 else 0.0
        )
        return max(bounded_retry_after + jitter, jittered)


# ---------------------------------------------------------------------------
# Env-var parsing helpers
# ---------------------------------------------------------------------------


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(
            f"environment variable {name} must be a float, got {raw!r}"
        ) from exc


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(
            f"environment variable {name} must be an int, got {raw!r}"
        ) from exc


__all__ = [
    "RetryPolicy",
    "bound_retries",
]
