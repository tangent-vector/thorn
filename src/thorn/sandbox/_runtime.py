"""OCI-runtime adapters: a thin shell over the ``podman``/``docker`` CLIs.

Phase B drives container lifecycle by shelling out to the
operator-installed OCI tool.  The set of verbs we need is small --
``run -d``, ``stop``/``kill``, ``rm``, ``inspect``, ``image inspect``,
``build``, ``ps`` -- and the brain never needs anything fancier than
parsing the CLI's JSON output.  That keeps our dependency surface to
zero and lets a single fake implementation cover the entire test
matrix.

The :class:`OCIRuntimeAdapter` protocol is the seam between
:class:`~thorn.sandbox.ContainerDaemonHost` and the actual runtime.
Two real implementations ship by default:

* :class:`PodmanAdapter` -- the rootless-friendly default; emits
  podman-specific flags (``--userns=keep-id``) where they help.
* :class:`DockerAdapter` -- equivalent flags translated to docker's
  conventions.

A third implementation, :class:`FakeOCIRuntimeAdapter`, lives in this
module too because it is the spine of the Phase-B unit tests: it
records every call, lets tests script ``inspect`` outputs, and never
shells out.

Selection at runtime is driven by ``gateway.json`` (the
``sandbox.oci_runtime`` field, decided in
:func:`select_oci_runtime`).  No environment-variable override --
config is the only knob, so post-hoc diagnosis from a saved
``gateway.json`` always agrees with what actually ran.
"""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Literal, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes shared across implementations
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Mount:
    """One bind-mount entry for ``--mount type=bind,...``.

    Source is a host path; target is the in-container path.
    ``read_only`` toggles the ``ro`` mount option; both ``podman``
    and ``docker`` accept the same flag spelling here so there is no
    per-runtime branching at this level.
    """

    source: Path
    target: Path
    read_only: bool = False


@dataclass(frozen=True)
class Tmpfs:
    """One tmpfs mount entry for ``--tmpfs <target>[:<options>]``.

    Phase E uses this to keep `/tmp` and `/var/tmp` writable when the
    container's rootfs is mounted read-only.  *target* is the
    in-container mount point; *options* is the comma-separated string
    accepted by both runtimes (e.g. ``"size=1G,mode=1777"``).  Empty
    options yield ``--tmpfs <target>`` with the runtime's defaults.
    """

    target: Path
    options: str = ""


@dataclass(frozen=True)
class ContainerSpec:
    """Everything :meth:`OCIRuntimeAdapter.run` needs to launch a container.

    Phase B established the spec's shape.  Phase E adds optional
    hardening fields (capability drops, security opts, read-only
    rootfs + tmpfs, resource limits) as new fields with sensible
    defaults (no-op when unset), so callers that don't care about
    hardening continue to work unchanged while the
    :class:`~thorn.sandbox.ContainerDaemonHost` and the resolver fill
    them in for the production sandbox.

    All fields the adapter consumes live here -- the
    :class:`OCIRuntimeAdapter` surface deliberately remains a single
    ``run`` method that takes one of these.
    """

    image: str
    name: str
    mounts: tuple[Mount, ...] = ()
    env: tuple[tuple[str, str], ...] = ()
    user: str | None = None
    workdir: str | None = None
    entrypoint: tuple[str, ...] | None = None
    command: tuple[str, ...] = ()
    extra_run_args: tuple[str, ...] = ()

    # Phase E hardening fields.  Each field's default is the no-op
    # ("don't emit this flag at all") so existing test fixtures and
    # callers that don't care about hardening continue to pass.  The
    # production gateway populates them via
    # :class:`~thorn.sandbox.ContainerHostConfig`, whose defaults
    # encode the conservative-by-default policy.

    capabilities_drop: tuple[str, ...] = ()
    """Capabilities to remove from the container's bounding set.

    Each entry becomes ``--cap-drop=<value>``; the special token
    ``"ALL"`` drops every capability the runtime would otherwise grant
    (the recommended Phase-E default).  Combined with a non-root
    ``user``, "drop ALL" leaves the container with no capabilities at
    all -- escalation paths via cap bugs require defeating both the
    user-id check and the cap bounding set.
    """

    capabilities_add: tuple[str, ...] = ()
    """Capabilities to grant after :attr:`capabilities_drop` is applied.

    Each entry becomes ``--cap-add=<value>``.  Used for the rare agent
    that legitimately needs a specific cap (e.g. ``CAP_NET_RAW`` for
    ping).  Adding caps is a deliberate departure from "drop ALL" and
    should be justified per agent.
    """

    security_opts: tuple[str, ...] = ()
    """Values to pass through ``--security-opt=<value>``.

    Phase E's default population includes ``"no-new-privileges"`` so
    setuid binaries cannot escalate even if they end up in a derived
    image.  Operators with their own needs (e.g. an AppArmor profile,
    a custom seccomp profile) can extend this surface from
    ``gateway.json`` / ``agent.json``.
    """

    read_only_root: bool = False
    """Mount the container's rootfs read-only (``--read-only``).

    Pairs with :attr:`tmpfs_mounts` to give the container scratch
    space at well-known paths while keeping the rootfs immutable.
    Phase E defaults this to ``True`` for the production sandbox;
    agents that need a writable rootfs (rare; typically dogfooding
    or unusual tool ecosystems) opt out via
    ``agent.json sandbox.read_only_root: false``.
    """

    tmpfs_mounts: tuple[Tmpfs, ...] = ()
    """In-container tmpfs mount points.

    When :attr:`read_only_root` is true, scratch directories
    (``/tmp``, ``/var/tmp``) need to live on tmpfs so tools that
    write there continue to work.  Phase E populates ``/tmp`` (1 GiB)
    and ``/var/tmp`` (256 MiB) by default.
    """

    memory_limit: str | None = None
    """``--memory`` value (e.g. ``"2G"``); ``None`` means unlimited.

    Hard cgroup memory cap.  Phase E defaults to ``"2G"`` for the
    production sandbox so a leaking tool OOMs the container before
    crowding the host.
    """

    cpu_limit: float | None = None
    """``--cpus`` value (fractional CPU count); ``None`` means unlimited.

    Cgroup CPU quota expressed as the maximum fraction of host CPU
    seconds per second the container may consume.  Phase E defaults
    to ``2.0``.
    """

    pid_limit: int | None = None
    """``--pids-limit`` value; ``None`` means unlimited.

    Hard cap on the number of processes the container's pid namespace
    may hold.  Phase E defaults to ``512``: roomy for shell + git +
    Python + a couple of MCP servers, tight enough that a fork bomb
    bounces off the limit before nuking the host.
    """


@dataclass(frozen=True)
class ContainerState:
    """Minimal subset of ``inspect`` output the host actually consumes.

    Maps to docker/podman ``inspect`` fields: ``status`` corresponds to
    ``State.Status`` (``"running"``, ``"exited"``, etc.); ``running``
    is the boolean form (``State.Running``); ``exit_code`` is
    ``State.ExitCode`` when the container has exited.  ``raw`` keeps
    the underlying dict in case callers want to look at fields we
    haven't surfaced yet.
    """

    name: str
    status: str
    running: bool
    exit_code: int | None
    raw: dict[str, object] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Adapter protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class OCIRuntimeAdapter(Protocol):
    """How :class:`ContainerDaemonHost` talks to the underlying OCI tool.

    Methods are deliberately CRUD-shaped rather than verb-shaped so a
    fake implementation can record calls without modeling the full CLI
    surface.  All errors propagate as :class:`OCIRuntimeError` (or a
    subclass) so callers can pattern-match on a single hierarchy
    instead of distinguishing between podman vs docker exit codes.
    """

    @property
    def name(self) -> Literal["podman", "docker"]:
        """Stable identifier for the underlying runtime."""
        ...

    async def image_exists(self, image: str) -> bool:
        """Return ``True`` if the local image cache holds *image*."""
        ...

    async def run(self, spec: ContainerSpec) -> str:
        """Start a detached container per *spec*; return its container ID.

        Implementations build the equivalent of ``<runtime> run -d
        --name <name> ...`` and return the container ID emitted on
        stdout.  The container is *running* (or about to be) when
        this returns; readiness probes belong to the caller.
        """
        ...

    async def inspect(self, name: str) -> ContainerState | None:
        """Return container state, or ``None`` if no such container exists."""
        ...

    async def stop(self, name: str, *, timeout_s: float = 10.0) -> None:
        """Stop a running container, waiting up to *timeout_s* before SIGKILL.

        Idempotent: calling on a non-existent or already-stopped
        container is a silent no-op so callers can use this in
        teardown paths without status-checking first.
        """
        ...

    async def remove(self, name: str, *, force: bool = False) -> None:
        """Remove a container by name; idempotent."""
        ...

    async def build(
        self,
        *,
        context: Path,
        dockerfile: Path,
        tag: str,
        build_args: dict[str, str] | None = None,
    ) -> None:
        """Build *dockerfile* in *context*, tagging the result *tag*."""
        ...

    async def list_containers(
        self, *, name_prefix: str | None = None,
    ) -> list[ContainerState]:
        """Return container states, optionally filtered by name prefix."""
        ...


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class OCIRuntimeError(RuntimeError):
    """Base exception for OCI-runtime CLI failures.

    Carries the underlying command, exit code, and (truncated) stderr
    so callers can render a useful message without having to plumb
    those through manually.  Subclasses:

    * :class:`OCIRuntimeNotFound` -- the configured runtime binary is
      not on ``PATH``.
    * :class:`OCIImageMissing` -- the image asked for is not in the
      local cache.
    """

    def __init__(
        self,
        message: str,
        *,
        command: tuple[str, ...] | None = None,
        exit_code: int | None = None,
        stderr: str | None = None,
    ) -> None:
        super().__init__(message)
        self.command = command
        self.exit_code = exit_code
        self.stderr = stderr


class OCIRuntimeNotFound(OCIRuntimeError):
    """Raised when the requested OCI runtime binary is not on ``PATH``."""


class OCIImageMissing(OCIRuntimeError):
    """Raised by callers when a required image is absent from the cache."""


# ---------------------------------------------------------------------------
# CLI base implementation
# ---------------------------------------------------------------------------


class _CLIRuntimeAdapter:
    """Shared scaffolding for :class:`PodmanAdapter` / :class:`DockerAdapter`.

    The CLI surface for the verbs we use overlaps almost entirely
    between podman and docker, so this base implementation handles:

    * locating the binary on ``PATH`` (raising
      :class:`OCIRuntimeNotFound` with a useful message),
    * shelling out via ``asyncio.create_subprocess_exec`` and
      capturing both streams,
    * uniform error wrapping into :class:`OCIRuntimeError`,
    * shared ``run`` argument-builder used by both subclasses.

    Subclasses contribute only the runtime-specific bits (``--userns``
    handling, network defaults, anything that diverges).
    """

    binary: str = ""  # overridden by subclasses

    def __init__(self, binary: str | None = None) -> None:
        resolved = binary or self.binary
        if not resolved:
            raise ValueError("OCI runtime binary name must be set")
        path = shutil.which(resolved)
        if path is None:
            raise OCIRuntimeNotFound(
                f"OCI runtime binary {resolved!r} is not on PATH; install "
                f"{resolved} or set sandbox.oci_runtime in gateway.json to "
                "another supported runtime",
            )
        self._binary_path = path
        self._binary_name = resolved

    @property
    def binary_path(self) -> str:
        return self._binary_path

    async def _run_cli(
        self,
        args: Iterable[str],
        *,
        check: bool = True,
        capture_stderr: bool = True,
    ) -> tuple[int, str, str]:
        cmd = (self._binary_path, *args)
        logger.debug("oci: running %s", _format_argv_for_log(cmd))
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE if capture_stderr else None,
        )
        stdout_b, stderr_b = await proc.communicate()
        stdout = stdout_b.decode("utf-8", errors="replace")
        stderr = stderr_b.decode("utf-8", errors="replace") if stderr_b else ""
        if check and proc.returncode != 0:
            raise OCIRuntimeError(
                f"{self._binary_name} {args[0] if args else ''} failed "
                f"with exit code {proc.returncode}: {stderr.strip()[:500]}",
                command=cmd,
                exit_code=proc.returncode,
                stderr=stderr,
            )
        return proc.returncode or 0, stdout, stderr

    # -- OCIRuntimeAdapter shape -------------------------------------------

    @property
    def name(self) -> Literal["podman", "docker"]:  # pragma: no cover
        raise NotImplementedError

    async def image_exists(self, image: str) -> bool:
        # Both podman and docker exit non-zero for an unknown image; we
        # use ``image inspect`` (rather than ``inspect``) because the
        # latter is overloaded with containers in podman.
        rc, _stdout, _stderr = await self._run_cli(
            ("image", "inspect", image, "--format", "{{.Id}}"),
            check=False,
        )
        return rc == 0

    async def run(self, spec: ContainerSpec) -> str:
        args = list(self._build_run_args(spec))
        _rc, stdout, _stderr = await self._run_cli(args)
        return stdout.strip().splitlines()[-1] if stdout.strip() else ""

    async def inspect(self, name: str) -> ContainerState | None:
        rc, stdout, _stderr = await self._run_cli(
            ("inspect", name, "--type", "container"),
            check=False,
        )
        if rc != 0:
            return None
        try:
            payload = json.loads(stdout)
        except json.JSONDecodeError as exc:
            raise OCIRuntimeError(
                f"{self._binary_name} inspect emitted non-JSON output: {exc}",
            ) from exc
        if not isinstance(payload, list) or not payload:
            return None
        entry = payload[0]
        return _container_state_from_inspect(name, entry)

    async def stop(self, name: str, *, timeout_s: float = 10.0) -> None:
        rc, _stdout, stderr = await self._run_cli(
            ("stop", "-t", str(int(timeout_s)), name),
            check=False,
        )
        if rc == 0:
            return
        # podman/docker return non-zero for "no such container"; treat
        # that as success (we're idempotent in teardown), surface
        # anything else.
        if "no such container" in stderr.lower() or "not found" in stderr.lower():
            return
        raise OCIRuntimeError(
            f"{self._binary_name} stop {name} failed: {stderr.strip()}",
            exit_code=rc,
            stderr=stderr,
        )

    async def remove(self, name: str, *, force: bool = False) -> None:
        args = ["rm"]
        if force:
            args.append("-f")
        args.append(name)
        rc, _stdout, stderr = await self._run_cli(tuple(args), check=False)
        if rc == 0:
            return
        if "no such container" in stderr.lower() or "not found" in stderr.lower():
            return
        raise OCIRuntimeError(
            f"{self._binary_name} rm {name} failed: {stderr.strip()}",
            exit_code=rc,
            stderr=stderr,
        )

    async def build(
        self,
        *,
        context: Path,
        dockerfile: Path,
        tag: str,
        build_args: dict[str, str] | None = None,
    ) -> None:
        args: list[str] = ["build", "-t", tag, "-f", str(dockerfile)]
        for key, value in (build_args or {}).items():
            args.extend(["--build-arg", f"{key}={value}"])
        args.append(str(context))
        await self._run_cli(tuple(args))

    async def list_containers(
        self, *, name_prefix: str | None = None,
    ) -> list[ContainerState]:
        args = ["ps", "-a", "--format", "{{json .}}"]
        rc, stdout, _stderr = await self._run_cli(tuple(args), check=False)
        if rc != 0:
            return []
        states: list[ContainerState] = []
        for line in stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            name = (
                entry.get("Names")
                or entry.get("Name")
                or entry.get("Id")
                or ""
            )
            if isinstance(name, list):
                name = name[0] if name else ""
            if name_prefix and not str(name).startswith(name_prefix):
                continue
            inspect_entry = await self.inspect(str(name))
            if inspect_entry is not None:
                states.append(inspect_entry)
        return states

    # -- Helpers -----------------------------------------------------------

    def _build_run_args(self, spec: ContainerSpec) -> Iterable[str]:
        yield "run"
        yield "-d"
        yield "--name"
        yield spec.name
        for mount in spec.mounts:
            yield "--mount"
            # Docker's ``--mount`` parser only accepts ``readonly`` (a
            # bare flag) and rejects the ``-v``-style ``ro``/``rw``
            # options outright; passing ``rw`` here yields
            # ``invalid field 'rw' must be a key=value pair``.  Podman
            # accepts both spellings, so emitting ``readonly`` only
            # when actually read-only (and omitting any read-write
            # option, since ``rw`` is the runtime default) is the one
            # form both runtimes parse identically.
            spec_str = f"type=bind,source={mount.source},target={mount.target}"
            if mount.read_only:
                spec_str += ",readonly"
            yield spec_str
        for tmpfs in spec.tmpfs_mounts:
            yield "--tmpfs"
            if tmpfs.options:
                yield f"{tmpfs.target}:{tmpfs.options}"
            else:
                yield str(tmpfs.target)
        for key, value in spec.env:
            yield "-e"
            yield f"{key}={value}"
        if spec.user is not None:
            yield "--user"
            yield spec.user
        if spec.workdir is not None:
            yield "--workdir"
            yield spec.workdir
        # Phase E hardening flags.  Order chosen so the meaningful
        # operator-readable flags appear early in the assembled argv
        # (caps, security-opts, resource limits, read-only, tmpfs)
        # before any escape-hatch ``extra_run_args`` -- which makes
        # the rendered command line easy to scan for "what protections
        # is this container running with?"  Per-flag emission is
        # uniform across podman and docker for the verbs Phase E
        # uses.
        for cap in spec.capabilities_drop:
            yield f"--cap-drop={cap}"
        for cap in spec.capabilities_add:
            yield f"--cap-add={cap}"
        for opt in spec.security_opts:
            yield f"--security-opt={opt}"
        if spec.read_only_root:
            yield "--read-only"
        if spec.memory_limit is not None:
            yield "--memory"
            yield spec.memory_limit
        if spec.cpu_limit is not None:
            yield "--cpus"
            yield str(spec.cpu_limit)
        if spec.pid_limit is not None:
            yield "--pids-limit"
            yield str(spec.pid_limit)
        # Adapter-specific defaults are injected before the operator's
        # ``extra_run_args`` so that an operator-provided flag wins via
        # the OCI runtime's "last value wins" convention for repeated
        # flags.  Today this hook only matters for podman
        # (``--userns=keep-id``); kept on the base class so docker can
        # opt in too if a future correctness fix requires it.
        for extra in self._runtime_specific_default_run_args(spec):
            yield extra
        for extra in spec.extra_run_args:
            yield extra
        # ``docker run --entrypoint`` accepts only a single executable
        # string -- the JSON-array form documented for Dockerfiles and
        # ``podman run`` is rejected by docker, which tries to exec
        # the literal JSON ('["python", "-m", "thorn.toolhost"]') as
        # a binary path and fails with ``executable file not found``.
        # The portable form is to pass the first entrypoint element
        # to ``--entrypoint`` and prepend the rest to ``command``;
        # both runtimes accept this and execute it as expected.
        entrypoint_extra: tuple[str, ...] = ()
        if spec.entrypoint is not None:
            if not spec.entrypoint:
                raise ValueError(
                    "ContainerSpec.entrypoint must be a non-empty argv "
                    "tuple when provided"
                )
            yield "--entrypoint"
            yield spec.entrypoint[0]
            entrypoint_extra = tuple(spec.entrypoint[1:])
        yield spec.image
        for arg in entrypoint_extra:
            yield arg
        for arg in spec.command:
            yield arg

    def _runtime_specific_default_run_args(
        self, spec: ContainerSpec,
    ) -> Iterable[str]:
        """Per-runtime defaults inserted before ``extra_run_args``.

        Hook for subclasses to contribute flags that a particular
        runtime needs in order for the spec's other fields to mean
        what they say.  The base implementation returns nothing;
        :class:`PodmanAdapter` overrides to default
        ``--userns=keep-id`` so that ``spec.user`` lines up 1:1 with
        the host operator's uid even on rootless podman.

        The hook receives *spec* so subclasses can suppress their
        default when the operator has already specified an equivalent
        flag in ``extra_run_args`` (avoids a duplicated flag where the
        runtime's "last value wins" rule would matter).
        """
        return ()


def _format_argv_for_log(argv: tuple[str, ...]) -> str:
    """Return a debug-log command line with env values redacted.

    Container specs carry broker proxy URLs and placeholder credential
    env vars through ``-e``/``--env`` entries.  The command line is
    useful for diagnosing runtime flags, but the env values are not;
    redact all env values uniformly so debug logging cannot leak a
    freshly minted broker access token.
    """
    formatted: list[str] = []
    redact_next_env = False
    for arg in argv:
        if redact_next_env:
            formatted.append(_redact_env_assignment(arg))
            redact_next_env = False
            continue
        if arg in {"-e", "--env"}:
            formatted.append(arg)
            redact_next_env = True
            continue
        if arg.startswith("--env="):
            formatted.append(
                f"--env={_redact_env_assignment(arg.removeprefix('--env='))}"
            )
            continue
        formatted.append(arg)
    return " ".join(formatted)


def _redact_env_assignment(value: str) -> str:
    key, sep, _raw_value = value.partition("=")
    if not sep:
        return value
    return f"{key}=<redacted>"


def _container_state_from_inspect(
    name: str, entry: dict[str, object],
) -> ContainerState:
    state = entry.get("State", {}) if isinstance(entry, dict) else {}
    if not isinstance(state, dict):
        state = {}
    status_raw = state.get("Status") or state.get("status") or "unknown"
    running_raw = state.get("Running") or state.get("running")
    exit_code_raw = state.get("ExitCode")
    if exit_code_raw is None:
        exit_code_raw = state.get("exit_code")
    return ContainerState(
        name=name,
        status=str(status_raw),
        running=bool(running_raw),
        exit_code=(
            int(exit_code_raw) if isinstance(exit_code_raw, int) else None
        ),
        raw=entry,
    )


# ---------------------------------------------------------------------------
# Concrete adapters
# ---------------------------------------------------------------------------


class PodmanAdapter(_CLIRuntimeAdapter):
    """Adapter for the ``podman`` CLI.

    Phase B's preferred runtime: rootless by default, no daemon to
    manage.  Defaults ``--userns=keep-id`` into every container's run
    args because rootless podman, *without* ``keep-id``, maps
    ``--user $(host_uid)`` to a sub-uid (because rootless podman's
    default mapping puts in-container uid 0 at the host operator's
    uid, so in-container uid N != 0 lands on host sub-uid). With
    ``keep-id``, in-container uids that match host operator's uid map
    1:1 instead, which is what every Phase-B/D/E claim about
    "bind-mount writes land owned by the operator" actually requires.

    The default is suppressed when the operator has already set
    ``--userns=...`` in :attr:`ContainerSpec.extra_run_args`, on the
    "operator knows what they're doing" principle.
    """

    binary = "podman"

    @property
    def name(self) -> Literal["podman"]:
        return "podman"

    def _runtime_specific_default_run_args(
        self, spec: ContainerSpec,
    ) -> Iterable[str]:
        # If the operator has already specified a userns mode, don't
        # second-guess them: rootful podman, custom userns layouts,
        # and tests that intentionally exercise non-keep-id mappings
        # all live here.
        for arg in spec.extra_run_args:
            if arg == "--userns" or arg.startswith("--userns="):
                return ()
        return ("--userns=keep-id",)


class DockerAdapter(_CLIRuntimeAdapter):
    """Adapter for the ``docker`` CLI.

    Functionally equivalent to :class:`PodmanAdapter` for the verbs we
    use; kept as a separate class so future divergences (build cache,
    network defaults, ``--userns`` handling) land in the obvious place.
    """

    binary = "docker"

    @property
    def name(self) -> Literal["docker"]:
        return "docker"


# ---------------------------------------------------------------------------
# Fake adapter for tests
# ---------------------------------------------------------------------------


@dataclass
class _FakeContainerRecord:
    """Internal bookkeeping for :class:`FakeOCIRuntimeAdapter`."""

    spec: ContainerSpec
    state: ContainerState


class FakeOCIRuntimeAdapter:
    """In-memory :class:`OCIRuntimeAdapter` for unit tests.

    Records every call so tests can assert wiring (the right name, the
    right mounts, the right entrypoint), and lets the test script
    ``inspect`` results so the two-stage readiness probe of
    :class:`ContainerDaemonHost` can be exercised without a real
    container ever existing.

    Defaults are friendly: every container starts in ``running`` /
    ``Running=True`` so the simple "happy path" flow works out of the
    box.  Tests that need a particular sequence (start->exited,
    start->stuck-in-created, ...) override via :meth:`set_state`.
    """

    def __init__(
        self,
        *,
        name: Literal["podman", "docker"] = "podman",
        present_images: Iterable[str] = (),
    ) -> None:
        self._name = name
        self._present_images: set[str] = set(present_images)
        self._containers: dict[str, _FakeContainerRecord] = {}
        self.run_calls: list[ContainerSpec] = []
        self.stop_calls: list[str] = []
        self.remove_calls: list[str] = []
        self.build_calls: list[tuple[Path, Path, str]] = []
        self.inspect_calls: list[str] = []

    @property
    def name(self) -> Literal["podman", "docker"]:
        return self._name

    # -- Test helpers ------------------------------------------------------

    def add_image(self, image: str) -> None:
        """Mark *image* as present in the local cache."""
        self._present_images.add(image)

    def remove_image(self, image: str) -> None:
        """Mark *image* as absent from the local cache."""
        self._present_images.discard(image)

    def set_state(
        self,
        name: str,
        *,
        status: str = "running",
        running: bool = True,
        exit_code: int | None = None,
    ) -> None:
        """Override the ``inspect`` result for a previously-run container."""
        record = self._containers.get(name)
        if record is None:
            raise KeyError(f"unknown container {name!r}")
        record.state = ContainerState(
            name=name, status=status, running=running, exit_code=exit_code,
            raw=dict(record.state.raw),
        )

    def container_spec(self, name: str) -> ContainerSpec:
        """Return the spec a container was started with (for assertions)."""
        return self._containers[name].spec

    def is_running(self, name: str) -> bool:
        record = self._containers.get(name)
        return bool(record and record.state.running)

    # -- OCIRuntimeAdapter implementation ----------------------------------

    async def image_exists(self, image: str) -> bool:
        return image in self._present_images

    async def run(self, spec: ContainerSpec) -> str:
        if spec.image not in self._present_images:
            raise OCIImageMissing(
                f"fake runtime: image {spec.image!r} not present",
            )
        self.run_calls.append(spec)
        cid = f"fake-{len(self._containers) + 1:04d}"
        self._containers[spec.name] = _FakeContainerRecord(
            spec=spec,
            state=ContainerState(
                name=spec.name, status="running", running=True,
                exit_code=None, raw={"Id": cid},
            ),
        )
        return cid

    async def inspect(self, name: str) -> ContainerState | None:
        self.inspect_calls.append(name)
        record = self._containers.get(name)
        return record.state if record is not None else None

    async def stop(self, name: str, *, timeout_s: float = 10.0) -> None:
        self.stop_calls.append(name)
        record = self._containers.get(name)
        if record is None:
            return
        record.state = ContainerState(
            name=name, status="exited", running=False, exit_code=0,
            raw=dict(record.state.raw),
        )

    async def remove(self, name: str, *, force: bool = False) -> None:
        self.remove_calls.append(name)
        self._containers.pop(name, None)

    async def build(
        self,
        *,
        context: Path,
        dockerfile: Path,
        tag: str,
        build_args: dict[str, str] | None = None,
    ) -> None:
        self.build_calls.append((context, dockerfile, tag))
        self._present_images.add(tag)

    async def list_containers(
        self, *, name_prefix: str | None = None,
    ) -> list[ContainerState]:
        states: list[ContainerState] = []
        for name, record in self._containers.items():
            if name_prefix and not name.startswith(name_prefix):
                continue
            states.append(record.state)
        return states


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------


def select_oci_runtime(
    configured: Literal["podman", "docker"] | None,
    *,
    podman_factory=PodmanAdapter,
    docker_factory=DockerAdapter,
) -> OCIRuntimeAdapter:
    """Pick an :class:`OCIRuntimeAdapter` based on ``gateway.json`` config.

    *configured* is the operator-supplied value of
    ``sandbox.oci_runtime`` from ``gateway.json``; the only knob.

    * When *configured* is ``"podman"`` or ``"docker"``, we instantiate
      that runtime's adapter and let the constructor raise
      :class:`OCIRuntimeNotFound` if the binary is missing -- the
      operator picked it explicitly, so a missing binary is a
      configuration error worth surfacing loudly with the field name
      in the message.
    * When *configured* is ``None`` (omitted from config), we
      auto-detect: prefer podman, fall back to docker.  If neither is
      on ``PATH`` we raise :class:`OCIRuntimeNotFound` with a
      remediation hint.

    No environment-variable override -- per the Phase-B plan,
    configuration is the only knob so post-hoc diagnosis from a saved
    ``gateway.json`` matches what actually ran.

    The ``*_factory`` parameters exist for tests to inject a
    :class:`FakeOCIRuntimeAdapter` without monkey-patching the module
    namespace.
    """
    if configured == "podman":
        return podman_factory()
    if configured == "docker":
        return docker_factory()
    if configured is not None:
        raise ValueError(
            f"sandbox.oci_runtime must be 'podman', 'docker', or null; "
            f"got {configured!r}",
        )

    # Auto-detect: prefer podman.
    if shutil.which("podman") is not None:
        return podman_factory()
    if shutil.which("docker") is not None:
        return docker_factory()
    raise OCIRuntimeNotFound(
        "No OCI runtime found on PATH.  Install podman (preferred) or "
        "docker, or set sandbox.oci_runtime in gateway.json explicitly.",
    )


__all__ = [
    "ContainerSpec",
    "ContainerState",
    "DockerAdapter",
    "FakeOCIRuntimeAdapter",
    "Mount",
    "OCIImageMissing",
    "OCIRuntimeAdapter",
    "OCIRuntimeError",
    "OCIRuntimeNotFound",
    "PodmanAdapter",
    "Tmpfs",
    "select_oci_runtime",
]
