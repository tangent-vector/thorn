"""Opt-in smoke test for a real OCI runtime (Phase B).

Skipped unless ``podman`` (or ``docker``) is on PATH.  Verifies the
end-to-end happy path against the real runtime: the
:class:`PodmanAdapter` / :class:`DockerAdapter` can talk to the
runtime, ``image_exists`` works, ``run`` + ``inspect`` + ``stop`` +
``rm`` round-trip, and ``list_containers`` filters correctly.

This intentionally does *not* drive ``ContainerDaemonHost`` against
a real ``thorn-sandbox`` image -- that requires either the operator
to have built it or this test to build it, both of which would slow
the smoke run from ~3s to ~3min.  The goal here is "the adapter
isn't subtly broken on this machine"; the deeper "ContainerDaemonHost
manages a real container correctly" question is covered by the
fake-adapter tests in ``test_container.py`` which exercise the same
control-flow.

Run with:  pytest -m requires_podman tests/sandbox/test_smoke_real_oci.py
       or:  pytest -m requires_docker tests/sandbox/test_smoke_real_oci.py
"""

from __future__ import annotations

import asyncio
import shutil
import subprocess
import uuid

import pytest

from thorn.sandbox import (
    ContainerSpec,
    DockerAdapter,
    PodmanAdapter,
)

_PODMAN_AVAILABLE = shutil.which("podman") is not None
_DOCKER_AVAILABLE = shutil.which("docker") is not None
_BUSYBOX_IMAGE = "docker.io/library/busybox:latest"


@pytest.fixture
def container_name() -> str:
    """Unique per-test container name to avoid cross-test interference."""
    return f"thorn-smoke-{uuid.uuid4().hex[:12]}"


async def _smoke(adapter, name: str) -> None:
    # Pull the tiny busybox image if it isn't already cached.  The
    # smoke test runs in an environment where the operator opted in,
    # so a one-time pull at first run is acceptable; subsequent runs
    # short-circuit on ``image_exists``.
    if not await adapter.image_exists(_BUSYBOX_IMAGE):
        # Adapter has no ``pull`` verb (the host workflow is build-then-
        # use, not pull); shelling out directly is fine for this smoke
        # test, whose point is "does the adapter talk to the runtime
        # at all on this host?", not "does the adapter wrap pull?".
        subprocess.run(
            [adapter.binary_path, "pull", _BUSYBOX_IMAGE],
            check=True,
            capture_output=True,
        )
    assert await adapter.image_exists(_BUSYBOX_IMAGE)

    spec = ContainerSpec(
        image=_BUSYBOX_IMAGE,
        name=name,
        command=("sleep", "60"),
    )
    cid = await adapter.run(spec)
    assert cid

    try:
        state = await adapter.inspect(name)
        assert state is not None
        assert state.running, f"unexpected state: {state}"
        assert state.name.endswith(name)

        listed = await adapter.list_containers(name_prefix="thorn-smoke-")
        assert any(c.name.endswith(name) for c in listed), listed
    finally:
        await adapter.stop(name, timeout_s=2)
        await adapter.remove(name, force=True)

    # Post-teardown: inspect should report absence (None) and listing
    # should no longer return our container.
    assert await adapter.inspect(name) is None
    listed = await adapter.list_containers(name_prefix="thorn-smoke-")
    assert not any(c.name.endswith(name) for c in listed)


@pytest.mark.requires_podman
@pytest.mark.skipif(not _PODMAN_AVAILABLE, reason="podman not on PATH")
def test_podman_adapter_smoke(container_name: str) -> None:
    asyncio.run(_smoke(PodmanAdapter(), container_name))


@pytest.mark.requires_docker
@pytest.mark.skipif(not _DOCKER_AVAILABLE, reason="docker not on PATH")
def test_docker_adapter_smoke(container_name: str) -> None:
    asyncio.run(_smoke(DockerAdapter(), container_name))
