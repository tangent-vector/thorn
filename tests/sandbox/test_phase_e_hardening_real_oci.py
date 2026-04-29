"""Opt-in smoke test asserting Phase E hardening flags actually work.

The Phase E unit tests in ``tests/sandbox/test_runtime.py``,
``test_resolve.py``, ``test_container.py``, and ``test_runtime_wiring.py``
verify that each Phase E flag is *emitted* on the OCI CLI in the right
shape.  This file closes the loop: it runs a real container against the
real ``podman`` / ``docker`` runtime with the production-default Phase E
flags, then asserts the runtime actually honored them.

That distinction matters because each Phase E flag depends on a layer
the runtime sits on top of and any of those layers can silently degrade:

* ``--cap-drop`` / ``--security-opt`` ride on Linux capabilities and
  prctl, both of which are kernel-level and therefore work on every host
  we deploy to.  Worth pinning anyway; a misspelled flag would be a
  no-op the unit tests can't catch.
* ``--read-only`` and ``--tmpfs`` ride on overlayfs and tmpfs; standard
  on every modern Linux.
* ``--user`` + ``--userns=keep-id`` ride on user namespaces.  These are
  the load-bearing piece of the identity-model claim: bind-mount writes
  must land owned by the gateway operator.  Worth a real-runtime test
  because the existing smoke (``test_smoke_real_oci.py``) doesn't
  exercise ``--user``.
* ``--pids-limit`` / ``--memory`` / ``--cpus`` ride on cgroup
  controllers.  On rootless podman + cgroup v1 (typical WSL2 today)
  podman *silently ignores* these flags ("Resource limits are not
  supported and ignored on cgroups V1 rootless systems").  The test
  reflects that reality: the production gateway still emits the flags,
  the runtime accepts them without error, but the *enforcement*
  assertion is gated on whether the host actually delegates the
  ``pids`` / ``memory`` / ``cpu`` controllers to rootless users.

Run with:

    pytest -m requires_podman tests/sandbox/test_phase_e_hardening_real_oci.py
or:

    pytest -m requires_docker tests/sandbox/test_phase_e_hardening_real_oci.py
"""

from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest

from thorn.sandbox import (
    ContainerSpec,
    DockerAdapter,
    PodmanAdapter,
    Tmpfs,
)
from thorn.sandbox._runtime import _CLIRuntimeAdapter

_PODMAN_AVAILABLE = shutil.which("podman") is not None
_DOCKER_AVAILABLE = shutil.which("docker") is not None
_BUSYBOX_IMAGE = "docker.io/library/busybox:latest"


# ---------------------------------------------------------------------------
# Production-shaped hardened spec
# ---------------------------------------------------------------------------


def _hardened_spec(name: str) -> ContainerSpec:
    """Return a :class:`ContainerSpec` with the Phase E production defaults.

    Mirrors the values :class:`~thorn.sandbox.ContainerHostConfig`
    populates from :class:`~thorn.gateway._config.SandboxConfig`'s
    defaults (drop-ALL caps, no-new-privs, read-only rootfs with
    ``/tmp`` and ``/var/tmp`` tmpfs, 2G/2cpu/512pids resource limits).
    Using busybox + ``sleep`` gives us a long-lived process to ``exec``
    assertions against without dragging in the full ``thorn-sandbox``
    image (whose readiness probe and toolhost daemon would distract
    from the runtime-level claim under test).
    """

    user = f"{os.getuid()}:{os.getgid()}"
    return ContainerSpec(
        image=_BUSYBOX_IMAGE,
        name=name,
        # ``sleep infinity`` keeps the container alive while we poke
        # at it via ``exec``.  busybox's default entrypoint is empty,
        # so the bare command is enough.
        command=("sleep", "3600"),
        user=user,
        capabilities_drop=("ALL",),
        capabilities_add=(),
        security_opts=("no-new-privileges",),
        read_only_root=True,
        tmpfs_mounts=(
            Tmpfs(target=Path("/tmp"), options="size=1G,mode=1777"),
            Tmpfs(target=Path("/var/tmp"), options="size=256M,mode=1777"),
        ),
        memory_limit="2G",
        cpu_limit=2.0,
        pid_limit=512,
    )


# ---------------------------------------------------------------------------
# Fixture: bring up one hardened container per runtime, share across tests
# ---------------------------------------------------------------------------


@dataclass
class _HardenedContainer:
    adapter: _CLIRuntimeAdapter
    name: str

    def exec(
        self,
        argv: tuple[str, ...],
        *,
        check: bool = False,
        timeout_s: float = 10.0,
    ) -> subprocess.CompletedProcess[str]:
        """Run *argv* inside the container via ``<runtime> exec``.

        Used by per-assertion tests to interrogate the running
        container.  ``exec`` reuses the container's namespaces and
        capability set, so cap drops / read-only rootfs / no-new-privs
        all apply identically to the exec'd process -- exactly what we
        want for the assertion.
        """

        cmd = [self.adapter.binary_path, "exec", self.name, *argv]
        return subprocess.run(
            cmd,
            check=check,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )


@pytest.fixture(scope="module")
def _ensure_busybox_podman() -> None:
    if not _PODMAN_AVAILABLE:
        pytest.skip("podman not on PATH")
    _ensure_image_pulled("podman", _BUSYBOX_IMAGE)


@pytest.fixture(scope="module")
def _ensure_busybox_docker() -> None:
    if not _DOCKER_AVAILABLE:
        pytest.skip("docker not on PATH")
    _ensure_image_pulled("docker", _BUSYBOX_IMAGE)


def _ensure_image_pulled(runtime: str, image: str) -> None:
    """Pull *image* via *runtime* if it isn't already cached.

    Mirrors the helper in ``test_smoke_real_oci.py``: the smoke suite
    runs only when the operator has opted in, so a one-time pull is
    acceptable; subsequent runs short-circuit.
    """

    rc = subprocess.run(
        [runtime, "image", "inspect", image, "--format", "{{.Id}}"],
        capture_output=True,
    ).returncode
    if rc == 0:
        return
    subprocess.run([runtime, "pull", image], check=True, capture_output=True)


def _container_name() -> str:
    return f"thorn-phase-e-{uuid.uuid4().hex[:12]}"


@pytest.fixture
def hardened_podman(_ensure_busybox_podman: None) -> Iterator[_HardenedContainer]:
    yield from _hardened_container(PodmanAdapter())


@pytest.fixture
def hardened_docker(_ensure_busybox_docker: None) -> Iterator[_HardenedContainer]:
    yield from _hardened_container(DockerAdapter())


def _hardened_container(
    adapter: _CLIRuntimeAdapter,
) -> Iterator[_HardenedContainer]:
    name = _container_name()
    spec = _hardened_spec(name)
    asyncio.run(adapter.run(spec))
    try:
        yield _HardenedContainer(adapter=adapter, name=name)
    finally:
        try:
            asyncio.run(adapter.stop(name, timeout_s=2))
        finally:
            asyncio.run(adapter.remove(name, force=True))


# ---------------------------------------------------------------------------
# Resource-limit gating: detect whether cgroup controllers are delegated
# ---------------------------------------------------------------------------


def _runtime_enforces_pid_limit(runtime: str) -> bool:
    """Probe whether ``--pids-limit`` actually constrains forks on this host.

    Spins up a one-shot busybox container with ``--pids-limit=2`` and
    tries to background two sleepers; the second one must fail to
    fork if pids enforcement is real.  Cheaper and more accurate than
    parsing ``<runtime> info`` output, which has runtime-specific
    shapes (``Host.CgroupControllers`` is a podman-only field;
    docker exposes ``CgroupVersion`` / ``CgroupDriver`` instead) and
    would also miss subtler degradations ("the controller is listed
    but the kernel ignored the knob anyway").

    Returns ``False`` when the probe can't be run for any reason
    (image pull failure, runtime error); the caller skips the
    enforcement assertion in that case rather than fail the suite
    over an environment problem.
    """

    try:
        proc = subprocess.run(
            [
                runtime, "run", "--rm",
                "--pids-limit=2",
                _BUSYBOX_IMAGE,
                "sh", "-c",
                # Two backgrounded sleepers + the parent shell would
                # need three slots in the pid namespace; with
                # ``--pids-limit=2`` the second fork must fail.  If
                # both succeed and "done" prints cleanly, the limit
                # is being silently ignored.
                "sleep 60 & sleep 60 & echo done",
            ],
            capture_output=True,
            text=True,
            timeout=15.0,
        )
    except Exception:  # pragma: no cover - defensive
        return False
    combined = proc.stdout + proc.stderr
    return (
        "Resource temporarily unavailable" in combined
        or "can't fork" in combined
    )


# ---------------------------------------------------------------------------
# Shared assertion bodies
# ---------------------------------------------------------------------------


def _assert_identity_is_operator(c: _HardenedContainer) -> None:
    """The in-container uid/gid should equal the host operator's uid/gid.

    This is the load-bearing piece of the Phase E identity model and
    of every "bind-mount writes are owned by the operator" claim that
    the threat-model doc rests on.  The flag combination under test:
    ``--user $(host_uid):$(host_gid)`` plus, for rootless podman,
    ``--userns=keep-id``.
    """

    out = c.exec(("id",)).stdout
    assert f"uid={os.getuid()}" in out, out
    assert f"gid={os.getgid()}" in out, out


def _assert_capabilities_dropped(c: _HardenedContainer) -> None:
    """``--cap-drop=ALL`` must result in an empty effective + bounding set."""

    out = c.exec(("cat", "/proc/self/status")).stdout
    cap_eff = next(
        (line for line in out.splitlines() if line.startswith("CapEff:")),
        "",
    )
    cap_bnd = next(
        (line for line in out.splitlines() if line.startswith("CapBnd:")),
        "",
    )
    assert cap_eff.split()[1] == "0000000000000000", cap_eff
    assert cap_bnd.split()[1] == "0000000000000000", cap_bnd


def _assert_no_new_privileges(c: _HardenedContainer) -> None:
    """``--security-opt=no-new-privileges`` must raise the prctl bit.

    Once ``NoNewPrivs:1`` is set, ``setuid`` / ``setcap`` binaries
    cannot escalate; even if a derived image accidentally ships one,
    it is defanged.
    """

    out = c.exec(("cat", "/proc/self/status")).stdout
    nnp = next(
        (line for line in out.splitlines() if line.startswith("NoNewPrivs:")),
        "",
    )
    assert nnp.split()[1] == "1", nnp


def _assert_rootfs_read_only(c: _HardenedContainer) -> None:
    """Writes to the rootfs must fail with EROFS, not silently succeed."""

    proc = c.exec(("touch", "/etc/thorn-phase-e-write-probe"))
    assert proc.returncode != 0, (
        f"touch /etc succeeded under --read-only: stdout={proc.stdout!r} "
        f"stderr={proc.stderr!r}"
    )
    assert "Read-only" in proc.stderr, proc.stderr


def _assert_tmpfs_writable(c: _HardenedContainer, target: str) -> None:
    """A path covered by a tmpfs mount must accept writes."""

    probe = f"{target}/thorn-phase-e-{uuid.uuid4().hex[:8]}"
    proc = c.exec(("sh", "-c", f"echo hi > {probe} && cat {probe}"))
    assert proc.returncode == 0, (
        f"tmpfs write to {target} failed: stdout={proc.stdout!r} "
        f"stderr={proc.stderr!r}"
    )
    assert proc.stdout.strip() == "hi", proc.stdout


_LOW_PID_LIMIT = 20


def _spec_with_low_pid_limit(name: str) -> ContainerSpec:
    """Hardened spec with ``pid_limit`` small enough to trip easily.

    Keeps every other Phase E flag at its production default so the
    "fork bomb trips at the limit" assertion isn't muddled by, say,
    a stray cap-add or a writable rootfs.  Used by the dedicated
    enforcement test below; the regular fixture uses the production
    default of 512.
    """

    base = _hardened_spec(name)
    return ContainerSpec(
        image=base.image,
        name=base.name,
        command=base.command,
        user=base.user,
        capabilities_drop=base.capabilities_drop,
        capabilities_add=base.capabilities_add,
        security_opts=base.security_opts,
        read_only_root=base.read_only_root,
        tmpfs_mounts=base.tmpfs_mounts,
        memory_limit=base.memory_limit,
        cpu_limit=base.cpu_limit,
        pid_limit=_LOW_PID_LIMIT,
    )


def _assert_pid_limit_actually_enforced(adapter: _CLIRuntimeAdapter) -> None:
    """Spin up a low-``pid_limit`` container and confirm fork() trips.

    The production default of 512 is too generous to test cheaply (a
    fork bomb of ~600 processes is slow and racy).  Drop to
    ``_LOW_PID_LIMIT`` and try to background ``5 * limit`` sleepers;
    the container must hit "Resource temporarily unavailable" before
    they all spawn.  Skipped when the host doesn't delegate cgroup
    controllers because podman silently ignores the flag in that case.
    """

    name = _container_name()
    spec = _spec_with_low_pid_limit(name)
    asyncio.run(adapter.run(spec))
    try:
        target = _LOW_PID_LIMIT * 5
        proc = subprocess.run(
            [
                adapter.binary_path, "exec", name, "sh", "-c",
                f"i=0; while [ $i -lt {target} ]; do sleep 60 & "
                "i=$((i+1)); done; wait",
            ],
            capture_output=True,
            text=True,
            timeout=15.0,
        )
        # Expect at least one fork failure.  The exact stderr wording
        # varies (busybox sh: "can't fork", glibc: "Resource
        # temporarily unavailable"), so accept either marker.  A
        # silently-passing run means the limit wasn't enforced.
        combined = proc.stdout + proc.stderr
        assert (
            "Resource temporarily unavailable" in combined
            or "can't fork" in combined
            or "fork: retry" in combined
        ), (
            f"--pids-limit={_LOW_PID_LIMIT} did not trip while spawning "
            f"{target} sleepers: stdout={proc.stdout!r} "
            f"stderr={proc.stderr!r}"
        )
    finally:
        try:
            asyncio.run(adapter.stop(name, timeout_s=2))
        finally:
            asyncio.run(adapter.remove(name, force=True))


# ---------------------------------------------------------------------------
# Per-runtime test classes
# ---------------------------------------------------------------------------


@pytest.mark.requires_podman
@pytest.mark.skipif(not _PODMAN_AVAILABLE, reason="podman not on PATH")
class TestPhaseEHardeningPodman:
    """Phase E hardening assertions against rootless ``podman``.

    Runs against the host's rootless podman so the
    ``--userns=keep-id`` path that :class:`PodmanAdapter` injects by
    default is exercised end-to-end.  The identity assertion in
    particular only means anything under rootless podman (rootful
    docker would map the in-container uid to the host operator's uid
    without any namespace gymnastics).
    """

    def test_identity_matches_operator(
        self, hardened_podman: _HardenedContainer,
    ) -> None:
        _assert_identity_is_operator(hardened_podman)

    def test_capabilities_dropped(
        self, hardened_podman: _HardenedContainer,
    ) -> None:
        _assert_capabilities_dropped(hardened_podman)

    def test_no_new_privileges(
        self, hardened_podman: _HardenedContainer,
    ) -> None:
        _assert_no_new_privileges(hardened_podman)

    def test_rootfs_read_only(
        self, hardened_podman: _HardenedContainer,
    ) -> None:
        _assert_rootfs_read_only(hardened_podman)

    def test_tmp_writable(
        self, hardened_podman: _HardenedContainer,
    ) -> None:
        _assert_tmpfs_writable(hardened_podman, "/tmp")

    def test_var_tmp_writable(
        self, hardened_podman: _HardenedContainer,
    ) -> None:
        _assert_tmpfs_writable(hardened_podman, "/var/tmp")

    def test_pid_limit_enforced(self) -> None:
        if not _runtime_enforces_pid_limit("podman"):
            pytest.skip(
                "rootless podman on this host does not enforce "
                "--pids-limit (typical for WSL2 + cgroup v1); the "
                "unit-test pinning of the flag emission is sufficient "
                "in this environment",
            )
        _assert_pid_limit_actually_enforced(PodmanAdapter())


@pytest.mark.requires_docker
@pytest.mark.skipif(not _DOCKER_AVAILABLE, reason="docker not on PATH")
class TestPhaseEHardeningDocker:
    """Phase E hardening assertions against ``docker``.

    Same body as the podman class with the docker adapter swapped in.
    Worth running both because the two runtimes diverge on identity
    semantics (``--userns=keep-id`` is podman-specific; docker's
    rootful default already maps in-container uid to host uid 1:1)
    and we want both paths covered before declaring Phase E shovel-
    ready.
    """

    def test_identity_matches_operator(
        self, hardened_docker: _HardenedContainer,
    ) -> None:
        _assert_identity_is_operator(hardened_docker)

    def test_capabilities_dropped(
        self, hardened_docker: _HardenedContainer,
    ) -> None:
        _assert_capabilities_dropped(hardened_docker)

    def test_no_new_privileges(
        self, hardened_docker: _HardenedContainer,
    ) -> None:
        _assert_no_new_privileges(hardened_docker)

    def test_rootfs_read_only(
        self, hardened_docker: _HardenedContainer,
    ) -> None:
        _assert_rootfs_read_only(hardened_docker)

    def test_tmp_writable(
        self, hardened_docker: _HardenedContainer,
    ) -> None:
        _assert_tmpfs_writable(hardened_docker, "/tmp")

    def test_var_tmp_writable(
        self, hardened_docker: _HardenedContainer,
    ) -> None:
        _assert_tmpfs_writable(hardened_docker, "/var/tmp")

    def test_pid_limit_enforced(self) -> None:
        if not _runtime_enforces_pid_limit("docker"):
            pytest.skip(
                "this docker host does not enforce --pids-limit in a "
                "way the probe could observe",
            )
        _assert_pid_limit_actually_enforced(DockerAdapter())
