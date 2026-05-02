"""Per-`thorn serve`-process supervisor for the bundled OneCLI broker.

This module owns the "make `thorn serve` Just Work" lifecycle for the
OneCLI credential broker:

* Brings up a dedicated OneCLI + Postgres compose stack with anonymous
  Docker volumes (no bind mounts to anything under ``<agency_home>``
  or ``<workspace_root>``) on a per-process compose project name.
* Mints / fetches the broker's admin API key against OneCLI's
  unauthenticated single-user-mode endpoints
  (``GET /api/user/api-key`` and ``POST /api/user/api-key/regenerate``).
* Synthesises an in-memory :class:`BrokerConfig` whose ``admin_url``
  points at the host-published admin port, whose ``proxy_url`` points
  at the sandbox-facing compose service DNS name, and whose admin key
  reflects the minted key.  Everything downstream
  (:class:`~thorn.gateway._broker.BrokerClient`,
  :func:`~thorn.gateway._broker.register_agent_with_broker`, the CA
  fetch + bind-mount flow) consumes this config unmodified.
* On graceful shutdown: ``<oci> compose -p <project> down --volumes
  --remove-orphans`` so the stack and all volumes evaporate without
  leaving operator-visible artefacts on disk.

Why per-process and fully transient?  ``<agency_home>`` is meant to
hold only state an operator would happily check into git (gateway
config, agent identities, journals, sessions).  Binaries, secrets,
and per-process compose state explicitly do not belong there.  The
existing per-agent broker registration flow already creates
everything fresh on each startup
(:meth:`~thorn.gateway._gateway.Gateway._register_broker_bindings`)
and tears it down on shutdown
(:meth:`~thorn.gateway._gateway.Gateway._teardown_broker_bindings`),
so paying ~10s of cold-start cost in exchange for never having to
worry about broker persistence is the right trade for the
single-VM, hours-to-days-uptime deployment shape this is aimed at.

Compose project / network / port choices: see the YAML at
``src/thorn/gateway/_resources/broker.compose.yml`` and the
header docstring there for the full rationale.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import secrets
import shutil
import time
from collections.abc import Awaitable, Callable
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from typing import Literal

import httpx

from thorn.core._credentials import ServiceCredential
from thorn.gateway._config import BrokerConfig, BundledBrokerImageConfig
from thorn.gateway._resources_helper import (
    BUNDLED_BROKER_COMPOSE_FILENAME,
    materialize_bundled_broker_compose,
)

log = logging.getLogger(__name__)

_BUNDLED_BROKER_PROJECT_PREFIX = "thorn-broker"
"""Compose project-name prefix used for every bundled-broker stack.

Suffixed with a per-process random tag so concurrent ``thorn serve``
runs on the same host get isolated stacks; also recognisable to
``thorn broker status`` / ``thorn broker down`` for orphan cleanup.
"""

_DEFAULT_HEALTH_TIMEOUT_S = 60.0
"""Default budget for OneCLI's ``/api/health`` to come up after
``compose up -d``.  60s covers the cold-image-pull + Postgres
init + Next.js boot path; once images are cached this is closer to
10s and the supervisor returns much sooner."""

_DEFAULT_HEALTH_POLL_INTERVAL_S = 1.0
"""How often to poll OneCLI's health endpoint while waiting for it
to come up.  1s is a fine default: docker-compose's own ``--wait``
already gates on its container-level healthcheck (when present),
so by the time we get here OneCLI is already serving requests in
the typical case and the first poll succeeds."""

_SANDBOX_PROXY_URL = "http://onecli:10255"
"""Broker proxy URL as seen from containers on the compose broker network."""


class BundledBrokerError(RuntimeError):
    """Raised when the bundled broker fails to come up or exits noisily.

    Distinct from :class:`~thorn.gateway._broker.BrokerError` (which
    covers admin-API call failures against an already-running broker)
    so the two failure modes show up in tracebacks under names that
    point at the right remediation: a ``BundledBrokerError`` means
    "the OCI runtime could not stand the stack up", a
    ``BrokerError`` means "the stack is up but the admin API
    rejected our request".
    """


@dataclass(frozen=True)
class _BundledBrokerEndpoints:
    """Discovered host endpoints for a brought-up bundled stack."""

    admin_host: str
    admin_port: int
    proxy_host: str
    proxy_port: int


def _generate_compose_project_name() -> str:
    """Pick a per-process compose project name with the bundled prefix.

    A short random suffix (8 hex chars) is enough to avoid
    collisions across overlapping ``thorn serve`` runs on one host
    while keeping the compose project name short enough for
    podman/docker's logging.  The recognisable prefix is what
    ``thorn broker`` cleanup commands match on.
    """
    return f"{_BUNDLED_BROKER_PROJECT_PREFIX}-{secrets.token_hex(4)}"


async def _default_subprocess_runner(
    argv: tuple[str, ...],
    env: dict[str, str],
    timeout_s: float,
) -> tuple[int, str, str]:
    """Default ``subprocess_runner`` for :class:`BundledBrokerSupervisor`.

    Spawns the given argv with the given environment, captures
    stdout/stderr, decodes them as UTF-8 (replacement on error),
    and returns ``(rc, stdout, stderr)``.  Raises
    :class:`asyncio.TimeoutError` on timeout (the supervisor maps
    this to a :class:`BundledBrokerError`).

    Extracted so unit tests can substitute a record-and-replay fake
    instead of mocking out :mod:`asyncio.subprocess`.
    """
    proc = await asyncio.create_subprocess_exec(
        *argv,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env,
    )
    try:
        stdout_b, stderr_b = await asyncio.wait_for(
            proc.communicate(), timeout=timeout_s,
        )
    except asyncio.TimeoutError:
        proc.kill()
        with _suppressed():
            await proc.wait()
        raise
    return (
        proc.returncode or 0,
        stdout_b.decode("utf-8", errors="replace"),
        stderr_b.decode("utf-8", errors="replace"),
    )


def _detect_compose_runtime() -> tuple[str, tuple[str, ...]]:
    """Detect the local compose runtime and return ``(name, base_argv)``.

    Returns the binary name (``"docker"`` or ``"podman"``) and the
    argv prefix to use for compose subcommands -- typically
    ``("docker", "compose")`` or ``("podman", "compose")``, matching
    the modern subcommand layout that both runtimes ship.

    Auto-detection mirrors :func:`select_oci_runtime` in
    :mod:`thorn.sandbox._runtime`: prefer podman for parity with the
    sandbox runtime selection (so the same OCI runtime is used for
    both broker and sandbox), fall back to docker.

    Raises :class:`BundledBrokerError` when neither runtime is
    available -- the operator's response is mechanical (install one
    or set ``sandbox.backend = "subprocess"`` to opt out of the
    container path).
    """
    for binary in ("podman", "docker"):
        path = shutil.which(binary)
        if path is not None:
            return binary, (path, "compose")
    raise BundledBrokerError(
        "No OCI runtime with a 'compose' subcommand found on PATH "
        "(checked podman and docker).  Install one (podman is "
        "preferred), or set sandbox.backend to 'subprocess' in "
        "gateway.json to opt out of containerised sandboxing and "
        "the bundled broker entirely.",
    )


class BundledBrokerSupervisor:
    """Lifecycle owner for a per-process bundled OneCLI broker stack.

    Single-shot: instantiate, ``await start()`` once, ``await
    shutdown()`` once.  The instance is not reusable after shutdown.

    All state held here is in-memory only.  Nothing is written to
    ``<agency_home>`` or ``<workspace_root>``; nothing is written to
    the host filesystem at all beyond what the OCI runtime materialises
    as part of its compose project (anonymous Docker volumes that
    ``compose down --volumes`` removes on shutdown).
    """

    def __init__(
        self,
        *,
        images: BundledBrokerImageConfig | None = None,
        bind_host: str = "127.0.0.1",
        health_timeout_s: float = _DEFAULT_HEALTH_TIMEOUT_S,
        health_poll_interval_s: float = _DEFAULT_HEALTH_POLL_INTERVAL_S,
        compose_runtime_factory: (
            "Callable[[], tuple[str, tuple[str, ...]]] | None"
        ) = None,
        http_client_factory: (
            "Callable[[], httpx.Client] | None"
        ) = None,
        subprocess_runner: (
            "Callable[[tuple[str, ...], dict[str, str], float], "
            "Awaitable[tuple[int, str, str]]] | None"
        ) = None,
    ) -> None:
        """Build a supervisor.

        *bind_host* sets the host interface compose publishes the
        OneCLI ports on; the default ``127.0.0.1`` keeps the broker
        admin API off the host's external interfaces, which is the
        right policy for single-VM Mode A deployments.  Operators
        who need to expose the admin dashboard for debugging can
        flip this to ``0.0.0.0`` -- the supervisor itself does not
        impose a security boundary, but it picks the safer default.

        *images* carries optional gateway-config image overrides for
        the bundled OneCLI and Postgres services.  When a reference
        is absent, the supervisor leaves the matching compose env var
        to the host environment / compose default path.

        *compose_runtime_factory* is a test seam: production callers
        leave it ``None`` and the supervisor auto-detects podman /
        docker.  Tests inject a fake that returns a record-and-replay
        argv prefix.

        *http_client_factory* is a test seam for the OneCLI admin-key
        acquisition step: production leaves it ``None`` and the
        supervisor uses a default :class:`httpx.Client`; tests pass
        a factory that returns a client with a mock transport.
        """
        self._images = images or BundledBrokerImageConfig()
        self._bind_host = bind_host
        self._health_timeout_s = health_timeout_s
        self._health_poll_interval_s = health_poll_interval_s
        self._compose_runtime_factory = (
            compose_runtime_factory or _detect_compose_runtime
        )
        self._http_client_factory = http_client_factory
        # Default to spawning a real subprocess; tests inject a fake
        # that records argv/env and returns scripted (rc, stdout, stderr).
        self._subprocess_runner = subprocess_runner or _default_subprocess_runner

        self._project_name: str | None = None
        """Set once :meth:`start` picks a compose project name."""
        self._compose_argv_prefix: tuple[str, ...] | None = None
        """Cached ``(<binary>, "compose", "-p", <project>, "-f", <yaml>)``
        prefix; populated in :meth:`start` and reused across compose
        verbs (``port``, ``down``)."""
        self._compose_runtime_name: str | None = None
        self._materialize_stack: ExitStack | None = None
        """Holds the open ``importlib.resources.as_file`` context so
        the compose YAML stays on disk for the lifetime of the
        bundled stack."""
        self._endpoints: _BundledBrokerEndpoints | None = None
        self._broker_config: BrokerConfig | None = None
        self._admin_api_key: ServiceCredential | None = None
        """Literal admin API key minted by :meth:`start`, retained
        in process memory only.  Exposed via :attr:`admin_api_key`
        so the gateway hands it to the broker client without it
        ever entering the persisted ``BrokerConfig``."""
        self._started = False
        self._shutdown_called = False

    # ------------------------------------------------------------------
    # Public read-only accessors (post-start)
    # ------------------------------------------------------------------

    @property
    def project_name(self) -> str | None:
        """Compose project name picked at :meth:`start`, if any.

        Useful to ``thorn broker status`` and to gateway code that
        wants to derive the per-project network name (``<project>
        _thorn-broker``) for sandbox container ``--network`` flags.
        """
        return self._project_name

    @property
    def egress_network_name(self) -> str | None:
        """Per-project egress network name for sandbox containers.

        Compose names project-scoped networks
        ``<project>_<service-network-key>``; the bundled compose
        defines ``thorn-broker`` as the internal sandbox-facing
        network, so the resolved name is
        ``<project>_thorn-broker``.  Returns ``None`` before
        :meth:`start` runs.
        """
        if self._project_name is None:
            return None
        return f"{self._project_name}_thorn-broker"

    @property
    def broker_config(self) -> BrokerConfig | None:
        """The synthesised :class:`BrokerConfig`, if :meth:`start` ran."""
        return self._broker_config

    @property
    def admin_api_key(self) -> ServiceCredential | None:
        """The minted admin API key, if :meth:`start` ran.

        The literal value lives only in process memory; it is never
        written into :class:`BrokerConfig` (which is intended to be
        persistable on disk) or otherwise serialised.  The gateway
        passes this directly to the broker client when constructing
        it for a bundled-broker startup.
        """
        return self._admin_api_key

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> BrokerConfig:
        """Bring the broker stack up and return its :class:`BrokerConfig`.

        Steps:

        1. Pick a unique compose project name.
        2. Materialise the bundled compose YAML to a real on-disk
           path (handles both editable installs and zipped wheels).
        3. ``<oci> compose -p <project> -f <yaml> up -d --wait`` with
           ``ONECLI_ADMIN_PORT=0`` / ``ONECLI_PROXY_PORT=0`` so docker
           picks free ports.
        4. Read the bound ports back via ``<oci> compose port``.
        5. Wait for OneCLI's ``/api/health`` to return 200.
        6. Acquire (or mint) the admin API key via OneCLI's
           single-user-mode HTTP endpoints.
        7. Synthesise and stash a :class:`BrokerConfig` for the
           gateway to consume.

        Raises :class:`BundledBrokerError` when any step fails.
        Idempotent only in the trivial sense -- a second call
        without an intervening :meth:`shutdown` raises
        :class:`RuntimeError`.
        """
        if self._started:
            raise RuntimeError(
                "BundledBrokerSupervisor.start() may only be called once "
                "per instance; create a new supervisor for a fresh stack",
            )
        self._started = True

        runtime_name, runtime_argv = self._compose_runtime_factory()
        self._compose_runtime_name = runtime_name

        project = _generate_compose_project_name()
        self._project_name = project

        self._materialize_stack = ExitStack()
        try:
            compose_path = self._materialize_stack.enter_context(
                materialize_bundled_broker_compose(),
            )
        except Exception as exc:
            self._materialize_stack.close()
            self._materialize_stack = None
            raise BundledBrokerError(
                f"Could not materialise bundled broker compose file "
                f"({BUNDLED_BROKER_COMPOSE_FILENAME}): {exc}",
            ) from exc

        # Build and cache the argv prefix that every later compose
        # verb (port, down) reuses.  Keeping this in one place makes
        # ``-p`` / ``-f`` flag handling a single concern.
        self._compose_argv_prefix = (
            *runtime_argv, "-p", project, "-f", str(compose_path),
        )

        log.info(
            "Bundled broker: bringing up compose project %r (runtime=%s, "
            "compose=%s)",
            project, runtime_name, compose_path,
        )

        try:
            await self._compose_up()
            self._endpoints = await self._discover_endpoints()
            await self._wait_for_health()
            admin_key = await self._acquire_admin_api_key()
        except BaseException:
            await self._safe_compose_down()
            raise

        # The synthesised config is presented to the rest of the
        # gateway as ``mode="bundled"``: it carries the discovered
        # admin/proxy URLs but deliberately does NOT carry the
        # literal admin API key (the key lives only on the
        # supervisor's :attr:`admin_api_key` attribute, in process
        # memory).  Bundled mode signals to the gateway that the
        # admin key comes from the supervisor rather than from
        # ``os.environ[admin_api_key_env_var]``.
        #
        # We sidestep the schema validator (which requires the
        # bundled mode to leave admin_url/proxy_url empty) via
        # ``model_construct``: the validator's invariant is for
        # operator-written configs, not for our post-startup
        # synthesised one.
        self._admin_api_key = ServiceCredential(admin_key)
        self._broker_config = BrokerConfig.model_construct(
            mode="bundled",
            enabled=True,
            admin_url=f"http://{self._endpoints.admin_host}:{self._endpoints.admin_port}",
            admin_api_key_env_var=None,
            # The gateway's admin traffic uses the host-published
            # endpoint above, but sandbox containers join the
            # compose-scoped broker network where ``onecli:10255`` is
            # the routable proxy address.  A host-loopback proxy URL
            # would resolve to the sandbox container itself.
            proxy_url=_SANDBOX_PROXY_URL,
            ca_certificate_path=None,
            bundled_images=self._images,
        )
        log.info(
            "Bundled broker: ready (admin=%s, proxy=%s, network=%s)",
            self._broker_config.admin_url,
            self._broker_config.proxy_url,
            self.egress_network_name,
        )
        return self._broker_config

    async def shutdown(self) -> None:
        """Tear down the bundled stack.

        Idempotent: safe to call multiple times.  Best-effort:
        failures during ``compose down`` are logged but do not
        propagate, so a hung compose teardown cannot block the
        gateway from exiting cleanly.

        Removes the anonymous volumes (``--volumes``) so no broker
        state survives the shutdown -- this is the whole point of
        the per-process / fully-transient design.
        """
        if self._shutdown_called:
            return
        self._shutdown_called = True
        await self._safe_compose_down()

    # ------------------------------------------------------------------
    # Compose driver (private)
    # ------------------------------------------------------------------

    async def _compose_up(self) -> None:
        """Run ``<oci> compose ... up -d --wait``."""
        env = self._compose_env()
        # ``--wait`` blocks until each service's healthcheck reports
        # healthy (or until we hit compose's own internal timeout, at
        # which point it exits non-zero and we surface a clear error).
        # Postgres has a HEALTHCHECK in the compose YAML; OneCLI does
        # not, so the supervisor follows up with its own /api/health
        # poll in :meth:`_wait_for_health`.  ``--wait`` is still
        # useful here because it ensures Postgres is up before we
        # start probing OneCLI.
        await self._run_compose_capturing(
            ("up", "-d", "--wait"), env=env, timeout_s=180.0,
        )

    async def _discover_endpoints(self) -> _BundledBrokerEndpoints:
        """Read the bound host ports back via ``<oci> compose port``.

        ``compose port <service> <container-port>`` prints
        ``<host>:<port>`` (or empty when the port isn't published).
        We always bind to ``self._bind_host`` so the host is known;
        the port, however, is whatever docker / podman picked when
        we passed ``ONECLI_ADMIN_PORT=0`` / ``ONECLI_PROXY_PORT=0``.
        """
        admin_addr = await self._compose_port("onecli", 10254)
        proxy_addr = await self._compose_port("onecli", 10255)
        admin_host, admin_port = _split_compose_port_output(admin_addr)
        proxy_host, proxy_port = _split_compose_port_output(proxy_addr)
        endpoints = _BundledBrokerEndpoints(
            admin_host=admin_host,
            admin_port=admin_port,
            proxy_host=proxy_host,
            proxy_port=proxy_port,
        )
        log.debug("Bundled broker: discovered endpoints: %s", endpoints)
        return endpoints

    async def _compose_port(self, service: str, container_port: int) -> str:
        rc, stdout, stderr = await self._run_compose_capturing(
            ("port", service, str(container_port)),
            env=self._compose_env(),
            timeout_s=10.0,
            check=False,
        )
        if rc != 0:
            raise BundledBrokerError(
                f"compose port {service}/{container_port} failed "
                f"(exit {rc}): {stderr.strip()[:300]}",
            )
        return stdout.strip()

    async def _safe_compose_down(self) -> None:
        """Best-effort ``compose down --volumes --remove-orphans``."""
        if (
            self._compose_argv_prefix is None
            or self._project_name is None
        ):
            return
        try:
            await self._run_compose_capturing(
                ("down", "--volumes", "--remove-orphans"),
                env=self._compose_env(),
                timeout_s=60.0,
                check=False,
            )
            log.info(
                "Bundled broker: compose project %r torn down",
                self._project_name,
            )
        except Exception:
            log.exception(
                "Bundled broker: error tearing down compose project %r; "
                "you may need to clean it up manually with `thorn broker "
                "down` or `%s compose -p %s down --volumes`",
                self._project_name,
                self._compose_runtime_name or "docker",
                self._project_name,
            )
        finally:
            if self._materialize_stack is not None:
                # Releasing the importlib.resources context lets the
                # temporary on-disk extraction (when the wheel is
                # zipped) be cleaned up; for editable / unzipped
                # installs this is a no-op.
                try:
                    self._materialize_stack.close()
                except Exception:
                    log.debug(
                        "Bundled broker: error closing compose-yaml "
                        "materialisation context; continuing",
                        exc_info=True,
                    )
                self._materialize_stack = None

    async def _run_compose_capturing(
        self,
        verbs: tuple[str, ...],
        *,
        env: dict[str, str],
        timeout_s: float,
        check: bool = True,
    ) -> tuple[int, str, str]:
        """Run ``compose ... <verbs>`` and capture stdout/stderr."""
        if self._compose_argv_prefix is None:
            raise RuntimeError(
                "internal error: compose invoked before start() set the prefix",
            )
        argv = (*self._compose_argv_prefix, *verbs)
        log.debug("Bundled broker: running %s", " ".join(argv))
        try:
            rc, stdout, stderr = await self._subprocess_runner(
                argv, env, timeout_s,
            )
        except asyncio.TimeoutError:
            raise BundledBrokerError(
                f"compose {' '.join(verbs)} timed out after {timeout_s}s",
            ) from None
        if check and rc != 0:
            raise BundledBrokerError(
                f"compose {' '.join(verbs)} failed (exit {rc}): "
                f"{stderr.strip()[:500]}",
            )
        return rc, stdout, stderr

    def _compose_env(self) -> dict[str, str]:
        """Build the env dict passed to every compose subprocess.

        Inherits the host env (so any operator-set
        ``ONECLI_POSTGRES_PASSWORD`` etc. flows through) and forces
        the bundled-supervisor-specific overrides:

        * ``ONECLI_ADMIN_PORT=0`` and ``ONECLI_PROXY_PORT=0`` so
          docker picks free host ports.
        * ``ONECLI_BIND_HOST`` to the supervisor's chosen bind
          interface (default ``127.0.0.1``).
        * ``ONECLI_NEXTAUTH_SECRET=""`` to keep OneCLI in its
          single-user "local" mode (the admin-key acquisition flow
          relies on this).
        """
        env = dict(os.environ)
        env["ONECLI_ADMIN_PORT"] = "0"
        env["ONECLI_PROXY_PORT"] = "0"
        env["ONECLI_BIND_HOST"] = self._bind_host
        env["ONECLI_NEXTAUTH_SECRET"] = ""
        env.update(self._images.compose_env_overrides())
        return env

    # ------------------------------------------------------------------
    # OneCLI-side bring-up
    # ------------------------------------------------------------------

    async def _wait_for_health(self) -> None:
        """Poll OneCLI's ``/api/health`` until it returns 200 or we time out.

        ``compose up --wait`` ensures Postgres is healthy by the time
        we get here, but OneCLI itself does not declare a healthcheck
        in the compose YAML (its Next.js boot is fast enough that one
        does not really earn its keep), so we do the wait ourselves.
        """
        if self._endpoints is None:
            raise RuntimeError(
                "internal error: _wait_for_health called before endpoints discovered",
            )
        url = (
            f"http://{self._endpoints.admin_host}:"
            f"{self._endpoints.admin_port}/api/health"
        )
        deadline = time.monotonic() + self._health_timeout_s
        last_error: str | None = None
        while time.monotonic() < deadline:
            try:
                response = await asyncio.to_thread(self._http_get, url)
            except Exception as exc:
                last_error = repr(exc)
            else:
                if response.status_code == 200:
                    log.debug(
                        "Bundled broker: /api/health 200 OK after wait",
                    )
                    return
                last_error = (
                    f"HTTP {response.status_code}: {response.text[:200]}"
                )
            await asyncio.sleep(self._health_poll_interval_s)
        raise BundledBrokerError(
            f"OneCLI /api/health did not return 200 within "
            f"{self._health_timeout_s}s; last error: {last_error}",
        )

    async def _acquire_admin_api_key(self) -> str:
        """Fetch (or mint) the OneCLI admin API key.

        OneCLI's ``/api/user/api-key`` endpoint short-circuits to
        the implicit ``local-admin`` user when ``NEXTAUTH_SECRET`` is
        unset (the bundled-supervisor single-user-mode invariant), so
        no ``Authorization`` header is required.

        Two cases:

        * The user already has a key (``GET`` returns 200): parse
          ``apiKey`` out of the JSON body and use it.
        * The user does not yet have a key (``GET`` returns 404):
          ``POST /api/user/api-key/regenerate`` to mint one and parse
          ``apiKey`` out of the response.

        This is the path documented in the plan.  Any other status
        code is surfaced as :class:`BundledBrokerError` -- we do not
        retry, because all the transient-error remediation has
        already happened in :meth:`_wait_for_health`.
        """
        if self._endpoints is None:
            raise RuntimeError(
                "internal error: _acquire_admin_api_key called before endpoints discovered",
            )
        admin_base = (
            f"http://{self._endpoints.admin_host}:"
            f"{self._endpoints.admin_port}"
        )
        get_url = f"{admin_base}/api/user/api-key"
        regen_url = f"{admin_base}/api/user/api-key/regenerate"

        get_response = await asyncio.to_thread(self._http_get, get_url)
        if get_response.status_code == 200:
            return _parse_api_key_response(get_response.text, source="GET")
        if get_response.status_code != 404:
            raise BundledBrokerError(
                f"OneCLI GET /api/user/api-key returned unexpected "
                f"HTTP {get_response.status_code}: "
                f"{get_response.text[:300]}",
            )

        log.info(
            "Bundled broker: no admin API key yet; minting via POST "
            "/api/user/api-key/regenerate",
        )
        post_response = await asyncio.to_thread(self._http_post, regen_url)
        if post_response.status_code != 200:
            raise BundledBrokerError(
                f"OneCLI POST /api/user/api-key/regenerate returned "
                f"HTTP {post_response.status_code}: "
                f"{post_response.text[:300]}",
            )
        return _parse_api_key_response(post_response.text, source="POST")

    def _http_get(self, url: str) -> httpx.Response:
        """Synchronous HTTP GET; runs on a worker thread.

        Uses the injected client factory if provided (test seam), else
        a default :class:`httpx.Client` per call.  Per-call clients are
        fine here because admin-key acquisition is two requests at
        most over the supervisor's lifetime.
        """
        client = (
            self._http_client_factory()
            if self._http_client_factory is not None
            else httpx.Client(timeout=10.0)
        )
        with client:
            return client.get(url)

    def _http_post(self, url: str) -> httpx.Response:
        client = (
            self._http_client_factory()
            if self._http_client_factory is not None
            else httpx.Client(timeout=10.0)
        )
        with client:
            return client.post(url)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_api_key_response(body_text: str, *, source: Literal["GET", "POST"]) -> str:
    """Pull the ``apiKey`` field out of an OneCLI admin-API response.

    Both ``GET /api/user/api-key`` and ``POST /api/user/api-key/regenerate``
    return JSON of shape ``{"apiKey": "oc_..."}``.  Surface a clear
    error if either field is missing; do *not* echo the body, since
    it carries the secret in the success case.
    """
    try:
        body = json.loads(body_text)
    except json.JSONDecodeError as exc:
        raise BundledBrokerError(
            f"OneCLI {source} /api/user/api-key returned non-JSON body "
            f"(prefix: {body_text[:100]!r})",
        ) from exc
    if not isinstance(body, dict) or "apiKey" not in body:
        raise BundledBrokerError(
            f"OneCLI {source} /api/user/api-key response did not include "
            "an 'apiKey' field; check the OneCLI version against the "
            "Thorn-supported range",
        )
    api_key = body["apiKey"]
    if not isinstance(api_key, str) or not api_key:
        raise BundledBrokerError(
            f"OneCLI {source} /api/user/api-key returned an empty / "
            "non-string apiKey field",
        )
    return api_key


def _split_compose_port_output(raw: str) -> tuple[str, int]:
    """Parse compose's ``host:port`` (or ``[ipv6]:port``) output.

    Returns ``(host, port)``.  Raises :class:`BundledBrokerError`
    when the value is not parseable -- typically the signal that
    compose returned an empty string because the service didn't
    expose the requested container port.
    """
    raw = raw.strip()
    if not raw:
        raise BundledBrokerError(
            "compose port returned empty output; the service may not be "
            "exposing the expected container port",
        )
    # Take the first non-empty line: compose can emit a trailing
    # newline plus warning lines on some podman versions.
    line = next((ln for ln in raw.splitlines() if ln.strip()), "")
    if not line:
        raise BundledBrokerError(
            f"compose port returned only whitespace: {raw!r}",
        )
    if line.startswith("["):
        # IPv6 form: [::1]:12345
        host_end = line.find("]")
        if host_end == -1 or len(line) <= host_end + 2 or line[host_end + 1] != ":":
            raise BundledBrokerError(
                f"compose port emitted unparseable IPv6 form: {line!r}",
            )
        host = line[1:host_end]
        port_str = line[host_end + 2:]
    else:
        host_part, _, port_str = line.rpartition(":")
        if not host_part or not port_str:
            raise BundledBrokerError(
                f"compose port emitted unparseable output: {line!r}",
            )
        host = host_part
    try:
        port = int(port_str)
    except ValueError as exc:
        raise BundledBrokerError(
            f"compose port emitted non-integer port in {line!r}",
        ) from exc
    return host, port


# ---------------------------------------------------------------------------
# Orphan discovery / cleanup helpers (used by ``thorn broker`` CLI)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BundledBrokerStackInfo:
    """One entry from :func:`list_bundled_broker_stacks`.

    Carries enough context for ``thorn broker status`` to render a
    human-readable line and for ``thorn broker down`` to issue the
    tear-down command against the right runtime.
    """

    project_name: str
    """Compose project name (always begins with the bundled prefix)."""

    runtime_name: Literal["podman", "docker"]
    """Which OCI runtime returned this stack -- so cleanup uses the
    same one rather than guessing."""

    status: str
    """Status string as reported by ``compose ls`` (e.g. ``"running(2)"``).
    Free-form -- different runtimes / versions emit slightly different
    shapes; we surface verbatim rather than try to parse."""


async def list_bundled_broker_stacks() -> list[BundledBrokerStackInfo]:
    """Return every compose project on the host that matches our prefix.

    Probes both podman and docker (skipping any that aren't on PATH),
    runs ``<runtime> compose ls --all --format json``, and filters
    the result to projects whose name begins with the bundled prefix
    (``thorn-broker-``).  Used by ``thorn broker status`` /
    ``thorn broker down`` to surface stacks that survived a non-graceful
    gateway shutdown (kill -9, OOM, etc.).

    Best-effort: a runtime that errors out on ``compose ls`` is logged
    at debug and skipped, not propagated -- the operator-facing CLI
    should still produce useful output even when one of the two
    runtimes is mid-upgrade or otherwise broken.
    """
    stacks: list[BundledBrokerStackInfo] = []
    for binary in ("podman", "docker"):
        path = shutil.which(binary)
        if path is None:
            continue
        try:
            stacks.extend(await _list_stacks_for_runtime(path, binary))
        except Exception as exc:
            log.debug(
                "Bundled broker: %s compose ls failed: %s",
                binary, exc,
            )
    return stacks


async def _list_stacks_for_runtime(
    binary_path: str,
    binary_name: Literal["podman", "docker"],
) -> list[BundledBrokerStackInfo]:
    proc = await asyncio.create_subprocess_exec(
        binary_path, "compose", "ls", "--all", "--format", "json",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout_b, stderr_b = await proc.communicate()
    if proc.returncode != 0:
        raise BundledBrokerError(
            f"{binary_name} compose ls failed (exit {proc.returncode}): "
            f"{stderr_b.decode('utf-8', errors='replace').strip()[:200]}",
        )
    try:
        rows = json.loads(stdout_b.decode("utf-8", errors="replace"))
    except json.JSONDecodeError as exc:
        raise BundledBrokerError(
            f"{binary_name} compose ls emitted non-JSON output: {exc}",
        ) from exc
    if not isinstance(rows, list):
        return []
    matches: list[BundledBrokerStackInfo] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        name = row.get("Name") or row.get("name")
        if not isinstance(name, str):
            continue
        if not name.startswith(f"{_BUNDLED_BROKER_PROJECT_PREFIX}-"):
            continue
        status = row.get("Status") or row.get("status") or ""
        matches.append(BundledBrokerStackInfo(
            project_name=name,
            runtime_name=binary_name,
            status=str(status),
        ))
    return matches


async def shutdown_bundled_broker_stack(stack: BundledBrokerStackInfo) -> None:
    """Best-effort ``compose down --volumes --remove-orphans`` for *stack*.

    Used by ``thorn broker down`` to clean up orphaned stacks.  Raises
    :class:`BundledBrokerError` on failure so the CLI can render a
    meaningful exit code.
    """
    binary_path = shutil.which(stack.runtime_name)
    if binary_path is None:
        raise BundledBrokerError(
            f"Cannot tear down {stack.project_name!r}: runtime "
            f"{stack.runtime_name!r} is no longer on PATH",
        )
    proc = await asyncio.create_subprocess_exec(
        binary_path, "compose", "-p", stack.project_name,
        "down", "--volumes", "--remove-orphans",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout_b, stderr_b = await proc.communicate()
    if proc.returncode != 0:
        raise BundledBrokerError(
            f"{stack.runtime_name} compose -p {stack.project_name} down "
            f"failed (exit {proc.returncode}): "
            f"{stderr_b.decode('utf-8', errors='replace').strip()[:300]}",
        )


@contextmanager
def _suppressed():
    """Context manager that swallows any exception.

    Lightweight local helper used by :meth:`_run_compose_capturing`
    when reaping a killed subprocess; the wait can race the kill in
    edge cases and we don't want a secondary exception to mask the
    real (timeout) one.
    """
    try:
        yield
    except Exception:
        pass


__all__ = [
    "BundledBrokerError",
    "BundledBrokerStackInfo",
    "BundledBrokerSupervisor",
    "list_bundled_broker_stacks",
    "shutdown_bundled_broker_stack",
]
