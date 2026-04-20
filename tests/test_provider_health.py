"""Tests for :class:`thorn.runtime.ProviderHealthMonitor`.

Focus areas, mapping to the design contract:

- Initial state is Healthy and ``wait_until_healthy`` is non-blocking.
- The sliding window trips to Degraded only when ``failure_threshold``
  failures fall within ``failure_window_seconds``; older failures roll
  off and stop counting.
- While Degraded, ``wait_until_healthy`` blocks until either:

  * the cooldown elapses and the caller is nominated as the next
    probe, or
  * another path reports success and we return to Healthy.

- Only one probe slot is open at a time, and a probe failure grows the
  cooldown exponentially up to the configured cap, with bounded jitter.
- ``snapshot()`` reports a coherent view of the current state.

Tests use an injected fake clock plus :func:`asyncio.timeout` so they
do not rely on real wall-clock waits.  The single exception is the
"new wake event after transition" pattern, which we exercise with a
near-zero base cooldown so the timeout path runs in microseconds.
"""

from __future__ import annotations

import asyncio
import os
import unittest

from thorn.runtime import (
    ProviderHealthMonitor,
    ProviderHealthSnapshot,
    ProviderHealthState,
)


class _FakeClock:
    """Monotonic-style clock whose value is set explicitly by tests."""

    def __init__(self, start: float = 1000.0) -> None:
        self._now = start

    def __call__(self) -> float:
        return self._now

    def advance(self, dt: float) -> None:
        self._now += dt


def _quick_monitor(
    *,
    failure_threshold: int = 2,
    failure_window: float = 60.0,
    base: float = 0.0,
    cap: float = 0.0,
    jitter: float = 0.0,
    clock: _FakeClock | None = None,
) -> tuple[ProviderHealthMonitor, _FakeClock | None]:
    """Build a monitor with deterministic (non-blocking) cooldowns.

    A fake clock is only injected when the caller explicitly passes
    one.  Tests that exercise the real ``wait_until_healthy`` path
    must let the monitor use the default ``time.monotonic`` clock so
    the cooldown arithmetic and the asyncio timer share a frame of
    reference -- otherwise a frozen fake clock would make the
    wait-loop spin every time the real-time ``wait_for`` fires.
    """
    kwargs: dict = dict(
        failure_threshold=failure_threshold,
        failure_window_seconds=failure_window,
        base_cooldown_seconds=base,
        max_cooldown_seconds=max(cap, base),
        cooldown_jitter_seconds=jitter,
    )
    if clock is not None:
        kwargs["clock"] = clock
    return ProviderHealthMonitor(**kwargs), clock


class TestConstruction(unittest.TestCase):
    """Argument validation and ``from_env`` plumbing."""

    def test_default_state_is_healthy(self) -> None:
        m = ProviderHealthMonitor()
        self.assertEqual(m.state, ProviderHealthState.HEALTHY)
        self.assertTrue(m.is_healthy)

    def test_invalid_threshold_rejected(self) -> None:
        with self.assertRaises(ValueError):
            ProviderHealthMonitor(failure_threshold=0)

    def test_invalid_window_rejected(self) -> None:
        with self.assertRaises(ValueError):
            ProviderHealthMonitor(failure_window_seconds=0)

    def test_invalid_cooldowns_rejected(self) -> None:
        with self.assertRaises(ValueError):
            ProviderHealthMonitor(base_cooldown_seconds=-1.0)
        with self.assertRaises(ValueError):
            ProviderHealthMonitor(
                base_cooldown_seconds=10.0,
                max_cooldown_seconds=5.0,
            )
        with self.assertRaises(ValueError):
            ProviderHealthMonitor(cooldown_jitter_seconds=-0.1)

    def test_from_env_uses_overrides(self) -> None:
        env_vars = {
            "THORN_PROVIDER_HEALTH_FAILURE_THRESHOLD": "7",
            "THORN_PROVIDER_HEALTH_FAILURE_WINDOW": "11.5",
            "THORN_PROVIDER_HEALTH_BASE_COOLDOWN": "2.5",
            "THORN_PROVIDER_HEALTH_MAX_COOLDOWN": "20.0",
            "THORN_PROVIDER_HEALTH_COOLDOWN_JITTER": "0.5",
        }
        try:
            for k, v in env_vars.items():
                os.environ[k] = v
            m = ProviderHealthMonitor.from_env()
        finally:
            for k in env_vars:
                os.environ.pop(k, None)
        # The constructor doesn't expose the raw values, but we can
        # poke at the private fields for an obvious smoke test --
        # this is the only way to verify env wiring without making
        # the values part of the public contract.
        self.assertEqual(m._failure_threshold, 7)
        self.assertAlmostEqual(m._failure_window_seconds, 11.5)
        self.assertAlmostEqual(m._base_cooldown_seconds, 2.5)
        self.assertAlmostEqual(m._max_cooldown_seconds, 20.0)
        self.assertAlmostEqual(m._cooldown_jitter_seconds, 0.5)

    def test_from_env_falls_back_on_garbage(self) -> None:
        os.environ["THORN_PROVIDER_HEALTH_FAILURE_THRESHOLD"] = "not-an-int"
        try:
            m = ProviderHealthMonitor.from_env()
        finally:
            os.environ.pop(
                "THORN_PROVIDER_HEALTH_FAILURE_THRESHOLD", None,
            )
        # Falls back to the default; we just check it constructed.
        self.assertEqual(m.state, ProviderHealthState.HEALTHY)


class TestHealthyFastPath(unittest.IsolatedAsyncioTestCase):
    """While Healthy, ``wait_until_healthy`` must be non-blocking."""

    async def test_wait_returns_immediately(self) -> None:
        m, _ = _quick_monitor()
        async with asyncio.timeout(1.0):
            await m.wait_until_healthy()

    async def test_success_in_healthy_clears_failures(self) -> None:
        m, _ = _quick_monitor(failure_threshold=3)
        await m.report_failure()
        await m.report_failure()
        self.assertEqual(m.snapshot().recent_failure_count, 2)
        await m.report_success()
        self.assertEqual(m.snapshot().recent_failure_count, 0)
        # Two more failures should not yet trip the breaker.
        await m.report_failure()
        await m.report_failure()
        self.assertEqual(m.state, ProviderHealthState.HEALTHY)


class TestThresholdTripping(unittest.IsolatedAsyncioTestCase):
    """Sliding-window threshold logic."""

    async def test_trips_at_threshold(self) -> None:
        m, _ = _quick_monitor(failure_threshold=3)
        await m.report_failure()
        await m.report_failure()
        self.assertEqual(m.state, ProviderHealthState.HEALTHY)
        await m.report_failure()
        self.assertEqual(m.state, ProviderHealthState.DEGRADED)

    async def test_failures_outside_window_do_not_count(self) -> None:
        clock = _FakeClock()
        m, _ = _quick_monitor(
            failure_threshold=3, failure_window=10.0, clock=clock,
        )
        await m.report_failure()
        await m.report_failure()
        # Push these two well past the window.
        clock.advance(11.0)
        await m.report_failure()
        # Only one failure inside the window now.
        self.assertEqual(m.state, ProviderHealthState.HEALTHY)
        self.assertEqual(m.snapshot().recent_failure_count, 1)


class TestProbeCoordination(unittest.IsolatedAsyncioTestCase):
    """Single-probe-at-a-time and recovery semantics."""

    async def test_probe_slot_after_cooldown(self) -> None:
        m, _ = _quick_monitor(
            failure_threshold=2, base=0.01, cap=0.01, jitter=0.0,
        )
        await m.report_failure()
        await m.report_failure()
        # The first wait after the (tiny) cooldown should claim the
        # probe slot and return rather than blocking forever.
        async with asyncio.timeout(1.0):
            await m.wait_until_healthy()
        self.assertTrue(m.snapshot().probe_in_flight)

    async def test_only_one_probe_at_a_time(self) -> None:
        m, _ = _quick_monitor(
            failure_threshold=2, base=0.01, cap=0.01, jitter=0.0,
        )
        await m.report_failure()
        await m.report_failure()
        # First waiter claims the probe slot.
        await m.wait_until_healthy()
        self.assertTrue(m.snapshot().probe_in_flight)
        # Second waiter should block while the probe is in flight.
        second = asyncio.create_task(m.wait_until_healthy())
        await asyncio.sleep(0.05)
        self.assertFalse(second.done())
        # Probe success unblocks everyone.
        await m.report_success()
        async with asyncio.timeout(1.0):
            await second

    async def test_probe_failure_grows_cooldown(self) -> None:
        m, _ = _quick_monitor(
            failure_threshold=1, base=0.5, cap=10.0, jitter=0.0,
        )
        await m.report_failure()
        first = m.snapshot().seconds_until_probe
        # base * 2**0 = 0.5
        self.assertAlmostEqual(first, 0.5, places=2)
        # Simulate the probe being claimed and failing.
        m._probe_in_flight = True
        await m.report_failure()
        second = m.snapshot().seconds_until_probe
        # base * 2**1 = 1.0
        self.assertAlmostEqual(second, 1.0, places=2)
        m._probe_in_flight = True
        await m.report_failure()
        third = m.snapshot().seconds_until_probe
        # base * 2**2 = 2.0
        self.assertAlmostEqual(third, 2.0, places=2)

    async def test_cooldown_capped_at_max(self) -> None:
        m, _ = _quick_monitor(
            failure_threshold=1, base=1.0, cap=2.0, jitter=0.0,
        )
        await m.report_failure()
        for _ in range(10):
            m._probe_in_flight = True
            await m.report_failure()
        snapshot = m.snapshot()
        # base * 2**11 would overflow the cap; we should be at most
        # the configured maximum.
        self.assertLessEqual(snapshot.seconds_until_probe, 2.0 + 1e-6)

    async def test_jitter_does_not_exceed_bound(self) -> None:
        m, _ = _quick_monitor(
            failure_threshold=1, base=0.1, cap=0.1, jitter=0.05,
        )
        # Trip and inspect cooldown across several attempts.
        for _ in range(20):
            await m.report_failure()
            snap = m.snapshot()
            # capped + jitter ∈ [0.1, 0.15].
            self.assertGreaterEqual(snap.seconds_until_probe, 0.1 - 1e-6)
            self.assertLessEqual(snap.seconds_until_probe, 0.15 + 1e-6)
            # Reset probe slot so the next failure is treated as a
            # fresh probe rather than a non-probe failure (the
            # latter would keep growing the cooldown attempt
            # counter, which we don't care about here).
            m._probe_in_flight = True

    async def test_success_recovers(self) -> None:
        m, _ = _quick_monitor(
            failure_threshold=1, base=10.0, cap=10.0, jitter=0.0,
        )
        await m.report_failure()
        self.assertEqual(m.state, ProviderHealthState.DEGRADED)

        # Even though the cooldown is large, a reported success
        # should immediately flip back to Healthy and release any
        # waiters.
        waiter = asyncio.create_task(m.wait_until_healthy())
        await asyncio.sleep(0.05)
        self.assertFalse(waiter.done())
        await m.report_success()
        async with asyncio.timeout(1.0):
            await waiter
        self.assertEqual(m.state, ProviderHealthState.HEALTHY)
        self.assertEqual(m.snapshot().consecutive_probe_failures, 0)


class TestSnapshot(unittest.IsolatedAsyncioTestCase):
    """``snapshot()`` returns a coherent view of internal state."""

    async def test_initial_snapshot(self) -> None:
        m, _ = _quick_monitor()
        snap = m.snapshot()
        self.assertIsInstance(snap, ProviderHealthSnapshot)
        self.assertEqual(snap.state, ProviderHealthState.HEALTHY)
        self.assertEqual(snap.recent_failure_count, 0)
        self.assertEqual(snap.seconds_until_probe, 0.0)
        self.assertFalse(snap.probe_in_flight)
        self.assertEqual(snap.consecutive_probe_failures, 0)

    async def test_snapshot_reports_window_count(self) -> None:
        m, _ = _quick_monitor(failure_threshold=5)
        await m.report_failure()
        await m.report_failure()
        snap = m.snapshot()
        self.assertEqual(snap.recent_failure_count, 2)
        self.assertEqual(snap.state, ProviderHealthState.HEALTHY)

    async def test_snapshot_reports_probe_in_flight(self) -> None:
        m, _ = _quick_monitor(
            failure_threshold=1, base=0.0, cap=0.0, jitter=0.0,
        )
        await m.report_failure()
        await m.wait_until_healthy()
        snap = m.snapshot()
        self.assertEqual(snap.state, ProviderHealthState.DEGRADED)
        self.assertTrue(snap.probe_in_flight)


if __name__ == "__main__":
    unittest.main()
