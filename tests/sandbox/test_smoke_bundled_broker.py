"""Opt-in smoke test for :class:`BundledBrokerSupervisor` against a
real OCI runtime.

Skipped unless ``podman`` (or ``docker``) is on PATH AND the test is
selected by its marker; even then the test exercises the full
"bring up the OneCLI broker stack, fetch the admin key, tear it down"
round-trip, which:

* pulls the OneCLI + Postgres images on first run (~couple of minutes
  on a slow link, ~seconds when cached),
* boots Postgres + OneCLI (~10s),
* tears them back down with ``--volumes`` so no operator-visible
  artefacts survive.

This is the only test in the suite that exercises the bundled-broker
end-to-end against a real ``docker compose`` invocation.  All other
supervisor logic (compose argv, port parsing, key acquisition,
shutdown idempotency) is covered by :mod:`tests.test_bundled_broker`
which uses an in-process fake.

Run with::

    uv run pytest -m requires_podman tests/sandbox/test_smoke_bundled_broker.py
    uv run pytest -m requires_docker tests/sandbox/test_smoke_bundled_broker.py
"""

from __future__ import annotations

import asyncio
import shutil

import httpx
import pytest

from thorn.gateway._bundled_broker import (
    BundledBrokerSupervisor,
    list_bundled_broker_stacks,
    shutdown_bundled_broker_stack,
)


_PODMAN_AVAILABLE = shutil.which("podman") is not None
_DOCKER_AVAILABLE = shutil.which("docker") is not None


def _force_runtime_factory(binary: str):
    """Build a ``compose_runtime_factory`` that forces a specific runtime.

    The supervisor's default factory prefers podman over docker; the
    smoke test wants to exercise *exactly* the runtime that the
    pytest marker selected, so each test pins the runtime explicitly
    rather than relying on the auto-detection order.
    """

    path = shutil.which(binary)
    if path is None:
        raise RuntimeError(f"{binary} not on PATH")

    def _factory() -> tuple[str, tuple[str, ...]]:
        return binary, (path, "compose")

    return _factory


async def _exercise(binary: str) -> None:
    supervisor = BundledBrokerSupervisor(
        compose_runtime_factory=_force_runtime_factory(binary),
        # Real OneCLI + Postgres bring-up can take a while on cold
        # cache; the supervisor's default 60s health budget is
        # already generous, but we extend a touch here to absorb
        # CI-runner cold-start variance.
        health_timeout_s=180.0,
    )
    config = await supervisor.start()
    try:
        # The synthesised config carries a real admin URL and minted
        # API key.  Hit ``/api/health`` once with a real httpx call
        # to confirm we are talking to the running OneCLI (not just
        # the supervisor's own health-poll loop).
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"{config.admin_url}/api/health")
            assert resp.status_code == 200, resp.text

        # Sanity-check that the supervisor recorded a project name
        # under the bundled prefix; the cleanup helpers (``thorn
        # broker status`` / ``down``) rely on this.
        assert supervisor.project_name is not None
        assert supervisor.project_name.startswith("thorn-broker-")
    finally:
        await supervisor.shutdown()

    # After a clean shutdown, ``compose ls`` must not surface our
    # project anymore -- proves ``--volumes --remove-orphans`` did
    # actually take the stack down (vs e.g. exiting silently after a
    # docker daemon hiccup).
    stacks = await list_bundled_broker_stacks()
    assert all(s.project_name != supervisor.project_name for s in stacks)


@pytest.mark.requires_podman
@pytest.mark.skipif(not _PODMAN_AVAILABLE, reason="podman not on PATH")
def test_bundled_broker_supervisor_real_podman() -> None:
    asyncio.run(_exercise("podman"))


@pytest.mark.requires_docker
@pytest.mark.skipif(not _DOCKER_AVAILABLE, reason="docker not on PATH")
def test_bundled_broker_supervisor_real_docker() -> None:
    asyncio.run(_exercise("docker"))


@pytest.mark.requires_docker
@pytest.mark.skipif(not _DOCKER_AVAILABLE, reason="docker not on PATH")
def test_orphan_cleanup_helpers_round_trip() -> None:
    """``thorn broker status`` / ``down`` find and clean stacks.

    Brings up a supervisor, intentionally drops the supervisor
    reference WITHOUT calling ``shutdown`` so the stack is orphaned,
    and then exercises the orphan-cleanup helpers (which is exactly
    what ``thorn broker status`` and ``thorn broker down`` are built
    on).  Belt-and-braces the kill -9 case the plan called out.
    """

    async def _run() -> None:
        supervisor = BundledBrokerSupervisor(
            compose_runtime_factory=_force_runtime_factory("docker"),
            health_timeout_s=180.0,
        )
        await supervisor.start()
        project = supervisor.project_name
        assert project is not None
        # Deliberately do NOT call shutdown; emulate the kill -9
        # case the plan called out as the reason for the orphan
        # cleanup helpers.

        try:
            stacks = await list_bundled_broker_stacks()
            ours = [s for s in stacks if s.project_name == project]
            assert ours, f"orphan stack {project!r} not found in {stacks!r}"
            await shutdown_bundled_broker_stack(ours[0])

            # Confirm the cleanup actually swept it.
            stacks_after = await list_bundled_broker_stacks()
            assert all(s.project_name != project for s in stacks_after)
        finally:
            # Belt and braces: if the test failed mid-flight, ensure
            # we still attempt a teardown so the developer's docker
            # state isn't left polluted.
            stacks = await list_bundled_broker_stacks()
            for s in stacks:
                if s.project_name == project:
                    try:
                        await shutdown_bundled_broker_stack(s)
                    except Exception:
                        pass

    asyncio.run(_run())
