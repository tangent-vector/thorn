"""``ContainerDaemonHost``: Phase-B per-agent sandbox container.

Drops in for :class:`~thorn.toolhost._host.SubprocessDaemonHost` and
hosts the same ``thorn-toolhost`` daemon, only inside an OCI container.
The brain still talks to the daemon over the same Unix-domain socket
as Phase A; the socket lives on a bind-mounted host directory the
container also sees, so neither the protocol nor the executor needs
any modifications.

Key shapes:

* :class:`ContainerHostConfig` -- everything the host needs to
  provision the container for one agent: image, container name,
  host-side bind-mount sources, env passthrough, user, and the OCI
  adapter to drive.
* :class:`ContainerDaemonHost` -- the :class:`DaemonHost`
  implementation; ``start`` does a two-stage readiness probe (image
  present + container in ``running`` state), ``stop`` is best-effort
  and idempotent.

Container-side layout is fixed: ``/agent/home`` for the agent's
home, ``/agent/workspace`` for its workspace, ``/agent/control`` for
the rendezvous (the socket and toolhost log).  The daemon's argv is
synthesized from those paths, so the brain-side and daemon-side
"agent home" paths differ literally (host ``.../agents/X/home`` vs
container ``/agent/home``) but refer to the same files via the bind
mount.

Two readiness budgets, intentionally asymmetric:

* ``container_ready_timeout_s`` (default 30s) covers cold-cache image
  load and process exec; this is where the *host* declares the
  container "running".  Generous because the variance is wide.
* The brain-side socket-reachable poll is the executor's
  ``connect_timeout_s`` (default 5s); by then the container is
  already up, so all we're waiting on is the daemon process binding
  its socket inside.

Distinct error kinds (:class:`ContainerNotReadyError`,
:class:`ContainerStartTimeoutError`,
:class:`SandboxImageMissingError`) make it obvious from the
exception alone which stage failed -- no log scraping required.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from thorn.sandbox._runtime import (
    ContainerSpec,
    Mount,
    OCIImageMissing,
    OCIRuntimeAdapter,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class SandboxImageMissingError(RuntimeError):
    """Raised when the sandbox image is not in the local cache.

    Phase B's policy is hard-fail: image absence blocks gateway
    startup so the operator can run ``thorn sandbox build`` once
    rather than discovering the failure mid-event-handling.  The
    exception message includes the exact image tag and the
    remediation command.
    """


class ContainerNotReadyError(RuntimeError):
    """Raised when the container exited before reaching ``running``.

    Distinct from :class:`ContainerStartTimeoutError` because the
    failure modes call for different operator response: an immediate
    exit usually points at a busted image or argv, while a timeout
    points at a slow host or runaway init.
    """


class ContainerStartTimeoutError(RuntimeError):
    """Raised when the container did not reach ``running`` within budget.

    Carries the elapsed time plus the last-observed ``status`` so
    operators can tell whether the container was still ``created`` /
    ``starting`` (likely host overload) vs. flapping between states.
    """


# ---------------------------------------------------------------------------
# Container-side path layout
# ---------------------------------------------------------------------------

CONTAINER_HOME_DIR = "/agent/home"
CONTAINER_WORKSPACE_DIR = "/agent/workspace"
CONTAINER_CONTROL_DIR = "/agent/control"
CONTAINER_RUNTIME_DIR = "/opt/thorn-runtime"
CONTAINER_SOCKET_PATH = f"{CONTAINER_CONTROL_DIR}/toolhost.sock"
CONTAINER_LOG_PATH = f"{CONTAINER_CONTROL_DIR}/toolhost.log"

CONTAINER_BROKER_CA_PATH = "/etc/thorn/onecli-ca.pem"
"""Container-side path where the broker's CA certificate is mounted.

Phase D: when the gateway is configured with a broker block and an
agent has registered with the broker, every outbound HTTPS request
from the sandbox container goes through the broker's HTTPS-MITM
proxy.  The container needs the broker's CA cert installed so the
TLS handshake succeeds; we mount the host-side cert read-only at
this fixed location and point the standard CA-bundle env vars
(``SSL_CERT_FILE``, ``REQUESTS_CA_BUNDLE``, ``NODE_EXTRA_CA_CERTS``,
``GIT_SSL_CAINFO``) at it.

The path is hard-coded rather than configurable for the same reason
as the other container-side paths above: the daemon's environment
inside the sandbox is uniform across agents, and operators never
have to think about it.
"""

NO_PROXY_DEFAULT = "localhost,127.0.0.1,::1,/agent/control"
"""Default ``NO_PROXY`` list for sandbox containers.

Loopback addresses plus the toolhost control directory (the
control-plane unix socket lives here, not over TCP, but defending
against well-meaning HTTP clients that mistakenly resolve
``localhost`` through the proxy is cheap).  Exposed as a constant
so tests can assert the precise value without duplicating it.
"""


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class ContainerHostConfig:
    """Per-agent provisioning recipe for :class:`ContainerDaemonHost`.

    Carries enough information to translate Phase-A
    :class:`~thorn.toolhost._host.SubprocessDaemonHostConfig` semantics
    into a container: the host-side bind-mount sources (where the
    agent's data actually lives), the image to run, the env passthrough
    list, and timeouts for the host-level readiness probe.

    The container-side paths are intentionally not configurable here:
    they're hard-coded under ``/agent/`` (see module-level
    ``CONTAINER_*`` constants) so the daemon's argv is always the same
    inside the container regardless of how the host directories are
    laid out.  Operators don't need to think about them; debugging a
    "wrong path" inside the container is the same problem at every
    site.
    """

    agent_id: str
    container_name: str
    image: str
    adapter: OCIRuntimeAdapter

    host_home_dir: Path
    host_workspace_dir: Path
    host_control_dir: Path

    env_passthrough: tuple[str, ...] = ()
    """Names of host env vars to forward into the container.

    Operator-controlled allow-list; the host process's value of each
    name is read at container start.  Names that are unset on the host
    are silently skipped.  Phase D will move credentials off this
    surface; the mechanism stays for non-sensitive values
    (``LANG``, proxy hints, debug toggles).
    """

    extra_env: tuple[tuple[str, str], ...] = ()
    """Literal env entries to add (not from the host process)."""

    broker_proxy_url: str | None = None
    """Phase D: HTTPS proxy URL of the OneCLI broker.

    When set, the sandbox container is configured (env vars +
    CA mount) so all outbound HTTPS traffic flows through this
    proxy.  The URL embeds Basic auth (``http://x:<token>@host:port/``)
    so OneCLI's ``Proxy-Authorization`` extractor identifies the
    agent without any per-tool customization.  ``None`` means the
    broker is disabled / not bound for this agent and the sandbox
    runs with no proxy interception.
    """

    broker_ca_host_path: Path | None = None
    """Host-side path to the broker's CA certificate (PEM).

    Required when ``broker_proxy_url`` is set; mounted read-only
    into the container at :data:`CONTAINER_BROKER_CA_PATH`.  The
    gateway populates this from the file the
    :class:`~thorn.gateway.BrokerClient` cached on disk after
    fetching ``/api/gateway/ca`` at startup.
    """

    broker_placeholder_env: tuple[tuple[str, str], ...] = ()
    """Phase D: env entries that replace literal credentials.

    Each entry is a ``(name, placeholder_value)`` pair.  The
    placeholder is the same nonsensical string the brain-side
    audit invariant verified is in ``ServiceCredential`` placeholder
    state -- it satisfies tools that require a non-empty token
    while carrying no real auth material; the broker injects the
    real credential into the upstream request based on
    host/path matching.  Independent of ``extra_env`` so the
    placeholder set is auditable and clearly distinguishable from
    operator-supplied env in logs and tests.
    """

    dev_mount_runtime: Path | None = None
    """Source path for a R/O bind-mount of the framework's source tree.

    When set, the framework code at *dev_mount_runtime* is mounted
    read-only at ``/opt/thorn-runtime`` inside the container and
    ``PYTHONPATH`` is set so the in-container ``python -m
    thorn.toolhost`` picks it up over whatever lives in the image.
    Default ``None`` keeps containers using the bundled, baked-in
    framework code.
    """

    user: str | None = None
    """``--user`` flag passed to the runtime (e.g. ``"1000:1000"``).

    When ``None``, defaults to the gateway process's effective UID/GID
    so files written through bind-mounts land with the operator's
    ownership and no ``chown`` is ever required.  Set to a literal
    string to override (e.g. for tests or unusual setups).
    """

    extra_run_args: tuple[str, ...] = ()
    """Extra arguments passed verbatim to ``<runtime> run`` (after the
    standard mount/env/user flags, before the image).  Use cases:
    ``--userns=keep-id`` for rootless podman, future Phase-F
    ``--cap-drop`` flags."""

    egress_network: str | None = None
    """Phase D: name of the OCI network the container joins.

    When set, ``--network <name>`` is prepended to ``extra_run_args``
    so the runtime (podman or docker) connects the container to
    *only* that network rather than the default bridge.  The
    operator is expected to have created the network with whatever
    isolation properties they want (typically ``--internal`` so
    there is no NAT to the host) and to have connected the broker
    to it; the resulting topology is "broker reachable, nothing
    else" without Thorn touching iptables itself.  ``None`` keeps
    the Phase-B default (whatever the runtime's default network is).
    """

    container_ready_timeout_s: float = 30.0
    """Stage-one budget: how long to wait for the container to reach
    ``running`` after ``run -d`` returns.  Generous on purpose --
    cold-cache image extraction has wide variance."""

    container_ready_poll_s: float = 0.1
    """Polling interval inside the stage-one readiness probe."""

    entrypoint: tuple[str, ...] = field(
        default_factory=lambda: (
            "python",
            "-m",
            "thorn.toolhost",
        )
    )
    """In-container entrypoint argv (excluding daemon-specific flags
    that we add at start time)."""

    max_concurrency: int = 8


# ---------------------------------------------------------------------------
# Host implementation
# ---------------------------------------------------------------------------


class ContainerDaemonHost:
    """Per-agent OCI container hosting a ``thorn-toolhost`` instance.

    Drop-in for :class:`~thorn.toolhost._host.SubprocessDaemonHost`:
    same ``socket_path`` / ``start`` / ``stop`` surface, just executes
    the daemon inside an OCI container instead of a host subprocess.
    Brain-side code stays identical; the only thing that changes is
    where the daemon's filesystem view comes from.

    Two-stage readiness contract on ``start``:

    1. *Image check* -- ``adapter.image_exists(image)`` must return
       ``True``; otherwise we raise :class:`SandboxImageMissingError`
       with a remediation command.  Phase B explicitly does **not**
       auto-build or auto-pull; see the plan's "Behavior when the
       image is missing" decision.
    2. *Container running* -- after ``adapter.run`` returns, we poll
       ``adapter.inspect`` until either the container reports
       ``Running=True`` (success), enters a terminal state without
       running (raise :class:`ContainerNotReadyError` with the exit
       code), or the timeout elapses (raise
       :class:`ContainerStartTimeoutError`).

    The brain's socket-reachable poll is *separate* and lives in
    :class:`~thorn.toolhost.DaemonToolExecutor`; that distinction
    keeps each stage's timeout matched to what it's actually waiting
    on.

    Teardown is best-effort and idempotent: ``stop`` issues a
    ``stop`` then a ``remove --force`` against the adapter and unlinks
    the host-side socket file.  Errors are logged but never surfaced;
    a brain in shutdown wants the next state, not a recovery path.
    """

    def __init__(self, config: ContainerHostConfig) -> None:
        self._config = config
        self._started: bool = False
        self._container_id: str | None = None

    @property
    def socket_path(self) -> Path:
        return self._config.host_control_dir / "toolhost.sock"

    @property
    def container_name(self) -> str:
        return self._config.container_name

    @property
    def image(self) -> str:
        return self._config.image

    @property
    def adapter(self) -> OCIRuntimeAdapter:
        return self._config.adapter

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        if self._started:
            raise RuntimeError(
                "ContainerDaemonHost.start called on an already-running host; "
                "call stop first"
            )

        # Stage zero: image presence.  Hard-fail with a remediation
        # command so the operator does not have to grep our docs to
        # find out which command to run.
        if not await self._config.adapter.image_exists(self._config.image):
            raise SandboxImageMissingError(
                f"sandbox image {self._config.image!r} is not present in "
                f"the local {self._config.adapter.name} cache.  "
                f"Run `thorn sandbox build` (default tag), or set "
                f"`sandbox.image` in gateway.json to an image that has "
                f"been built/pulled, then restart the gateway.",
            )

        # Make sure the host-side control dir exists *before* the
        # container mounts it; otherwise podman/docker may invent the
        # path with root ownership and the in-container daemon (running
        # as the host UID) will not be able to bind its socket.
        self._config.host_control_dir.mkdir(parents=True, exist_ok=True)

        # Pre-clean any stale socket file from a prior unclean shutdown.
        # The daemon does this itself on bind, but doing it here too
        # surfaces permission problems immediately.
        with contextlib.suppress(FileNotFoundError):
            os.unlink(self.socket_path)

        # Pre-remove any prior container by the same name (e.g. left
        # behind by a crashed gateway).  Idempotent on the adapter side.
        await self._config.adapter.remove(
            self._config.container_name, force=True,
        )

        spec = self._build_container_spec()
        try:
            cid = await self._config.adapter.run(spec)
        except OCIImageMissing as exc:
            # Race: the image disappeared between the check and the run
            # (concurrent ``image rm``, e.g.).  Translate to the host
            # exception type so callers don't have to know about the
            # adapter layer.
            raise SandboxImageMissingError(str(exc)) from exc

        self._container_id = cid
        self._started = True

        try:
            await self._wait_for_running()
        except Exception:
            # If readiness fails, tear down so we don't leak the
            # container; the original exception propagates.
            await self.stop()
            raise

    async def stop(self) -> None:
        if not self._started and self._container_id is None:
            return
        try:
            with contextlib.suppress(Exception):
                await self._config.adapter.stop(self._config.container_name)
            with contextlib.suppress(Exception):
                await self._config.adapter.remove(
                    self._config.container_name, force=True,
                )
        finally:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(self.socket_path)
            self._started = False
            self._container_id = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _wait_for_running(self) -> None:
        """Stage-one readiness probe: poll until container is running.

        Distinct from the brain-side socket-reachable poll; this one
        only cares whether the container itself reached the
        ``running`` state.  A terminal non-running state (``exited``,
        ``stopped``, ``dead``) raises :class:`ContainerNotReadyError`
        immediately rather than waiting out the timeout, since further
        polling cannot help.
        """
        deadline = time.monotonic() + self._config.container_ready_timeout_s
        last_status = "unknown"
        while time.monotonic() < deadline:
            state = await self._config.adapter.inspect(
                self._config.container_name,
            )
            if state is None:
                # Container vanished (manual removal between run and
                # inspect, or the runtime is misbehaving).  Surface as
                # a not-ready error rather than waiting it out.
                raise ContainerNotReadyError(
                    f"container {self._config.container_name!r} "
                    "disappeared after start",
                )
            last_status = state.status
            if state.running:
                return
            if state.status in {"exited", "stopped", "dead", "removing"}:
                raise ContainerNotReadyError(
                    f"container {self._config.container_name!r} reached "
                    f"terminal state {state.status!r} "
                    f"(exit_code={state.exit_code}) before becoming ready",
                )
            await asyncio.sleep(self._config.container_ready_poll_s)

        raise ContainerStartTimeoutError(
            f"container {self._config.container_name!r} did not reach "
            f"running state within "
            f"{self._config.container_ready_timeout_s}s "
            f"(last status: {last_status!r})",
        )

    def _build_container_spec(self) -> ContainerSpec:
        cfg = self._config

        mounts: list[Mount] = [
            Mount(source=cfg.host_home_dir, target=Path(CONTAINER_HOME_DIR)),
            Mount(
                source=cfg.host_workspace_dir,
                target=Path(CONTAINER_WORKSPACE_DIR),
            ),
            Mount(
                source=cfg.host_control_dir,
                target=Path(CONTAINER_CONTROL_DIR),
            ),
        ]
        if cfg.dev_mount_runtime is not None:
            mounts.append(
                Mount(
                    source=cfg.dev_mount_runtime,
                    target=Path(CONTAINER_RUNTIME_DIR),
                    read_only=True,
                ),
            )

        # Broker integration is opt-in: the gateway only populates
        # the ``broker_*`` fields after a successful per-agent
        # registration with OneCLI.  When set, the container gets a
        # R/O bind-mount of the broker CA at the well-known path so
        # the in-container TLS stack trusts the broker's MITM cert,
        # and a curated env block that wires every common HTTP
        # client through the proxy.
        if cfg.broker_proxy_url is not None:
            if cfg.broker_ca_host_path is None:
                raise ValueError(
                    "ContainerHostConfig: broker_proxy_url is set but "
                    "broker_ca_host_path is None; both must be supplied "
                    "together so the in-container TLS stack trusts the "
                    "broker's MITM CA",
                )
            mounts.append(
                Mount(
                    source=cfg.broker_ca_host_path,
                    target=Path(CONTAINER_BROKER_CA_PATH),
                    read_only=True,
                ),
            )

        env: list[tuple[str, str]] = []
        if cfg.dev_mount_runtime is not None:
            env.append(("PYTHONPATH", CONTAINER_RUNTIME_DIR))
        # Force unbuffered stdout/stderr so the toolhost log surfaces
        # in real time when the operator is tail-ing it.
        env.append(("PYTHONUNBUFFERED", "1"))

        for name in cfg.env_passthrough:
            value = os.environ.get(name)
            if value is None:
                logger.debug(
                    "container host: env var %r is unset on host; skipping",
                    name,
                )
                continue
            env.append((name, value))
        env.extend(cfg.extra_env)

        if cfg.broker_proxy_url is not None:
            env.extend(_broker_env_entries(cfg.broker_proxy_url))
            env.extend(cfg.broker_placeholder_env)

        command: list[str] = [
            "--socket",
            CONTAINER_SOCKET_PATH,
            "--agent-id",
            cfg.agent_id,
            "--max-concurrency",
            str(cfg.max_concurrency),
            "--home",
            CONTAINER_HOME_DIR,
            "--workspace-root",
            CONTAINER_WORKSPACE_DIR,
            "--log-file",
            CONTAINER_LOG_PATH,
        ]

        user = cfg.user if cfg.user is not None else _current_uid_gid()

        # Phase D: when an ``egress_network`` is configured, prepend
        # ``--network <name>`` so the runtime attaches the container
        # to that network instead of the default bridge.  Stays
        # before ``extra_run_args`` so an operator can still append
        # additional flags (``--userns=keep-id`` etc.) without
        # having to know about the egress wiring.
        run_args: tuple[str, ...] = cfg.extra_run_args
        if cfg.egress_network is not None:
            run_args = ("--network", cfg.egress_network) + run_args

        return ContainerSpec(
            image=cfg.image,
            name=cfg.container_name,
            mounts=tuple(mounts),
            env=tuple(env),
            user=user,
            entrypoint=cfg.entrypoint,
            command=tuple(command),
            extra_run_args=run_args,
        )


def _broker_env_entries(proxy_url: str) -> list[tuple[str, str]]:
    """Env entries that route container traffic through the broker.

    Both upper- and lower-case forms of ``HTTP[S]_PROXY`` /
    ``NO_PROXY`` are emitted because client libraries are
    inconsistent: ``curl`` honors the lowercase forms first while
    most Python and Node clients honor the uppercase forms;
    ``git`` reads only ``http_proxy`` / ``https_proxy``.  Setting
    both is harmless and avoids per-tool surprises.

    The four CA-bundle env vars cover the common stacks: OpenSSL
    (``SSL_CERT_FILE``), Python ``requests``
    (``REQUESTS_CA_BUNDLE``), Node TLS
    (``NODE_EXTRA_CA_CERTS``), and Git
    (``GIT_SSL_CAINFO``).  Tools using other TLS stacks (Go's
    ``crypto/tls``, Rust's ``rustls`` defaults) need image-baked
    trust anchors, addressed when those stacks become a
    constraint.
    """
    return [
        ("HTTPS_PROXY", proxy_url),
        ("HTTP_PROXY", proxy_url),
        ("https_proxy", proxy_url),
        ("http_proxy", proxy_url),
        ("NO_PROXY", NO_PROXY_DEFAULT),
        ("no_proxy", NO_PROXY_DEFAULT),
        ("SSL_CERT_FILE", CONTAINER_BROKER_CA_PATH),
        ("REQUESTS_CA_BUNDLE", CONTAINER_BROKER_CA_PATH),
        ("NODE_EXTRA_CA_CERTS", CONTAINER_BROKER_CA_PATH),
        ("GIT_SSL_CAINFO", CONTAINER_BROKER_CA_PATH),
    ]


def _current_uid_gid() -> str | None:
    """Return ``"<uid>:<gid>"`` for the current process, or ``None`` on Windows."""
    if not hasattr(os, "geteuid"):  # pragma: no cover - Windows
        return None
    return f"{os.geteuid()}:{os.getegid()}"


def derive_container_name(agent_id: str, *, prefix: str = "thorn-agent-") -> str:
    """Synthesize a container name from an agent id.

    Container names must be DNS-label-ish (``[a-zA-Z0-9_.-]+`` for
    both runtimes), and our agent ids are already URL-safe per
    :func:`thorn.runtime._paths.safe_dirname`.  Replace any leftover
    chars defensively so an unusual id never trips the runtime.
    """
    safe = "".join(
        c if (c.isalnum() or c in "_.-") else "_"
        for c in agent_id
    )
    return f"{prefix}{safe}"


__all__ = [
    "CONTAINER_BROKER_CA_PATH",
    "CONTAINER_CONTROL_DIR",
    "CONTAINER_HOME_DIR",
    "CONTAINER_LOG_PATH",
    "CONTAINER_RUNTIME_DIR",
    "CONTAINER_SOCKET_PATH",
    "CONTAINER_WORKSPACE_DIR",
    "ContainerDaemonHost",
    "ContainerHostConfig",
    "ContainerNotReadyError",
    "ContainerStartTimeoutError",
    "NO_PROXY_DEFAULT",
    "SandboxImageMissingError",
    "derive_container_name",
]
