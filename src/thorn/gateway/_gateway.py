"""Gateway daemon orchestrator.

The ``Gateway`` owns a :class:`~thorn.runtime.Runtime`, a list of
:class:`EventSource` instances, and a tool list.  It is the top-level
object that ``thorn serve`` creates to run the agent daemon.

High-level flow
---------------

The gateway is the *router* of the actor model: it translates incoming
events into durable notifications posted to a session's inbox, then
submits that session to a per-agent scheduler.  The scheduler (one
per agent) owns the concurrency discipline -- one prompt in flight
per session, up to *N* in flight per agent.  Prompt construction,
agent-level tool use, and inbox item close-out all live below the
gateway in :mod:`thorn.runtime`.

Startup:

1. Enter the runtime context.
2. Rebuild the :class:`~thorn.runtime.InFlightIndex` from the on-disk
   state of every queue, so source-level dedup is correct before any
   source starts polling.
3. Run :func:`~thorn.runtime.run_startup_sweep` to heal any
   crash-leftover state (``in_progress`` reverts, stuck ``handled``/
   ``errored`` items re-dispatched, temp files cleaned up).
4. Eagerly load every persisted agent and build an
   :class:`~thorn.runtime.AgentScheduler` for each.  Agents live for
   the lifetime of the process, so paying their load cost up front
   is cheaper than doing it on the first-event hot path and makes
   the steady-state predictable.
5. Start event sources and wait for shutdown.

Per-event handling is :meth:`_handle_event`: resolve the agent, ensure
its scheduler exists, lazily create the session and its inbox, post a
:class:`~thorn.runtime.NotificationSpec` to the inbox, and submit the
session to the scheduler.  No ``agent.lock``: concurrency within an
agent is the scheduler's job now.

Shutdown:

1. Stop sources (no new events).
2. Gracefully shut down every scheduler with a bounded grace period.
   Schedulers cancel in-flight dispatchers on timeout; the startup
   sweep on the next process entry reconciles any resulting
   ``in_progress`` items.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import logging
import os
import signal
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

from thorn.core._agent import Agent
from thorn.core._session import Session
from thorn.gateway._broker import (
    BrokerBinding,
    BrokerClient,
    BrokerError,
    admin_api_key_from_env,
    register_agent_with_broker,
)
from thorn.gateway._bundled_broker import (
    BundledBrokerError,
    BundledBrokerSupervisor,
)
from thorn.gateway._config import (
    BrokerConfig,
    BundledBrokerImageConfig,
    GatewayConfig,
)
from thorn.gateway._event import (
    EventSource,
    FormattedEvent,
    RawIncomingEvent,
)
from thorn.gateway._formatter import (
    FormatterDelivery,
    FormatterDrop,
    NotificationFormatter,
)
from thorn.gateway._heartbeat import (
    gateway_heartbeat_path,
    gateway_heartbeat_timestamp,
    write_gateway_heartbeat,
)
from thorn.gateway._peer import PeerRegistry
from thorn.gateway._trigger_policy import (
    SourceTriggerPolicy,
    TriggerAuthorizationPolicy,
)
from thorn.runtime import (
    DEFAULT_AGENT_CONCURRENCY,
    AgentID,
    AgentScheduler,
    NotificationSpec,
    PromptDispatcher,
    ProviderHealthMonitor,
    ProviderHealthSnapshot,
    Runtime,
    SessionAddress,
    SessionInbox,
    SessionKey,
    default_progress_evictor,
    inbox_prompt_dispatcher,
    rebuild_in_flight_index,
    run_startup_sweep,
)

log = logging.getLogger(__name__)


def _default_broker_client_factory(
    config: GatewayConfig,
    bundled_supervisor: "BundledBrokerSupervisor | None",
) -> BrokerClient:
    """Build a :class:`BrokerClient` for the gateway's broker config.

    For ``mode='bundled'`` the supervisor minted the admin API key in
    process memory and exposes it via :attr:`admin_api_key`; that
    literal is what we hand to the client.

    For ``mode='external'`` the operator named the env var holding
    the key in ``broker.admin_api_key_env_var``;
    :func:`admin_api_key_from_env` reads it and surfaces a clear
    error when the var is unset.
    """
    if config.broker is None:
        raise BrokerError(
            "Cannot build a broker client: gateway.json has no "
            "broker block (or it is None at this point in startup)."
        )
    if config.broker.mode == "bundled":
        if bundled_supervisor is None or bundled_supervisor.admin_api_key is None:
            raise BrokerError(
                "Bundled broker mode requires a started supervisor "
                "exposing the admin API key, but none is available.  "
                "This indicates a wiring bug in gateway startup."
            )
        return BrokerClient(
            config.broker, admin_api_key=bundled_supervisor.admin_api_key,
        )
    return BrokerClient(
        config.broker, admin_api_key=admin_api_key_from_env(config.broker),
    )


class BundledBrokerSupervisorFactory(Protocol):
    """Factory seam for constructing the bundled-broker supervisor."""

    def __call__(
        self,
        *,
        images: BundledBrokerImageConfig,
    ) -> BundledBrokerSupervisor:
        """Return a supervisor configured for the given image overrides."""
        ...


def _render_git_extra_headers(
    extra_headers: tuple[tuple[str, str], ...],
) -> str:
    """Render ``(host, header_value)`` pairs as a minimal gitconfig file.

    The resulting INI drives ``git``'s per-URL ``http.extraHeader``
    mechanism: any ``git`` command hitting ``https://<host>/<anything>``
    sends the configured header unconditionally (including on the
    initial unauthenticated request, which is what we need so the
    broker has a header to override before any credential
    negotiation starts).  We emit one section per host rather than
    a single catch-all ``[http]`` section so only the forges the
    operator wired up receive the placeholder header; unrelated
    ``https://...`` git traffic is untouched.

    The global ``http.proxyAuthMethod = basic`` entry makes libcurl
    pre-send ``Proxy-Authorization`` for the broker proxy URL's
    ``x:<aoc_token>`` userinfo.  Without it, Git waits for a 407
    challenge and OneCLI falls back to tunnel mode, so it cannot MITM
    HTTPS and rewrite the placeholder ``Authorization`` header.

    Hosts are deduplicated (first-occurrence-wins) in case two
    services register git HTTPS routing for the same host.  The
    output is newline-terminated so subsequent edits (should any
    ever be made) don't leave the file without a trailing newline.
    """
    if not extra_headers:
        return ""
    seen: set[str] = set()
    lines: list[str] = []
    lines.append("[http]")
    lines.append("    proxyAuthMethod = basic")
    for host, header_value in extra_headers:
        if host in seen:
            continue
        seen.add(host)
        lines.append(f'[http "https://{host}/"]')
        lines.append(f"    extraHeader = {header_value}")
    if not lines:
        return ""
    return "\n".join(lines) + "\n"


_DEFAULT_AGENT_ID = AgentID("default")

DEFAULT_SHUTDOWN_TIMEOUT_SECONDS = 30.0
"""Default grace period handed to :meth:`AgentScheduler.shutdown`.

Long enough to let an in-flight LLM call finish cleanly in most
real-world cases; short enough that a stuck scheduler is not a
deployment incident.  Tests override this with a small value.
"""


class Gateway:
    """Daemon that routes external events to Thorn agents.

    Parameters:
        runtime: The persistent execution environment.  The gateway
            registers session inboxes on its :class:`AddressBook` and
            shares its :class:`InFlightIndex` with every queue it
            creates.
        sources: Event sources to poll / listen on.
        agent_concurrency: Per-agent concurrency cap for the
            :class:`~thorn.runtime.AgentScheduler` of each agent.
            Defaults to :data:`~thorn.runtime.DEFAULT_AGENT_CONCURRENCY`.
        prompt_dispatcher: Override the scheduler's
            :data:`~thorn.runtime.PromptDispatcher`.  Tests use this to
            substitute a dispatcher that closes out inbox items
            automatically; production callers should leave the default
            :func:`~thorn.runtime.inbox_prompt_dispatcher`.
        shutdown_timeout: Grace period (seconds) given to each
            scheduler on gateway shutdown.  ``None`` waits
            indefinitely; ``0`` cancels immediately.  Defaults to
            :data:`DEFAULT_SHUTDOWN_TIMEOUT_SECONDS`.
        health_monitor: Shared :class:`ProviderHealthMonitor` for
            gateway-wide LLM provider circuit-breaking.  When
            supplied, the same instance is wired into every
            scheduler created by the gateway, so a provider outage
            observed by any session pauses every session sharing
            this gateway until the provider recovers.  When
            ``None`` (the default), the gateway constructs one
            from environment variables via
            :meth:`ProviderHealthMonitor.from_env`; pass an
            explicit instance for tests or to share a monitor
            across multiple gateways in the same process.
        heartbeat_interval_s: Seconds between writes to the
            operator-facing gateway heartbeat file.
    """

    def __init__(
        self,
        *,
        runtime: Runtime,
        sources: list[EventSource],
        agent_concurrency: int = DEFAULT_AGENT_CONCURRENCY,
        prompt_dispatcher: PromptDispatcher | None = None,
        shutdown_timeout: float | None = DEFAULT_SHUTDOWN_TIMEOUT_SECONDS,
        health_monitor: ProviderHealthMonitor | None = None,
        gateway_config: GatewayConfig | None = None,
        broker_client_factory: (
            "Callable[[GatewayConfig, BundledBrokerSupervisor | None], BrokerClient] | None"
        ) = None,
        bundled_broker_supervisor_factory: (
            BundledBrokerSupervisorFactory | None
        ) = None,
        heartbeat_interval_s: float = 5.0,
    ) -> None:
        self._runtime = runtime
        self._sources = sources
        self._agent_concurrency = agent_concurrency
        self._prompt_dispatcher: PromptDispatcher = (
            prompt_dispatcher or inbox_prompt_dispatcher
        )
        self._shutdown_timeout = shutdown_timeout
        # Construct the monitor up-front (rather than lazily on
        # first scheduler) so callers can ``snapshot()`` even before
        # any agent has been seen.  ``from_env`` returns a default-
        # configured monitor when no THORN_PROVIDER_HEALTH_* vars
        # are set, so this is also a safe no-config-required path.
        self._health_monitor: ProviderHealthMonitor = (
            health_monitor or ProviderHealthMonitor.from_env()
        )

        # Phase D: optional broker integration.  When *gateway_config*
        # carries a ``broker`` block (and ``broker.enabled``), startup
        # registers each loaded agent with the broker, swaps the
        # in-memory credentials for placeholders, and stashes the
        # resulting :class:`BrokerBinding` for the per-agent sandbox
        # executor to consume.  When the broker is disabled (or
        # *gateway_config* is None, the test-only path), the gateway
        # behaves exactly as in Phase B (env injection + audit not
        # enforced).  *broker_client_factory* is an injection seam
        # for tests; production callers leave it None.
        self._gateway_config = gateway_config
        self._broker_client_factory = (
            broker_client_factory or _default_broker_client_factory
        )
        # The supervisor factory is an injection seam: tests pass a
        # fake supervisor that doesn't shell out to ``docker compose``.
        # Production callers leave it ``None`` and the gateway
        # instantiates the real :class:`BundledBrokerSupervisor` on
        # demand only when ``broker.mode == "bundled"``.  The factory
        # receives the resolved bundled-image config so tests can
        # keep the supervisor fake while still checking propagation.
        self._bundled_broker_supervisor_factory = (
            bundled_broker_supervisor_factory or BundledBrokerSupervisor
        )
        self._bundled_broker_supervisor: BundledBrokerSupervisor | None = None
        self._broker_client: BrokerClient | None = None
        self._broker_bindings: dict[AgentID, BrokerBinding] = {}

        self._stop_event: asyncio.Event | None = None
        self._source_tasks: list[asyncio.Task[None]] = []
        self._heartbeat_task: asyncio.Task[None] | None = None
        self._heartbeat_interval_s = heartbeat_interval_s
        self._heartbeat_started_at: str | None = None
        self._schedulers: dict[AgentID, AgentScheduler] = {}
        self._inboxes: dict[SessionAddress, SessionInbox] = {}
        self._started = False

        # Build the peer-aware notification pipeline from the config.
        # When *gateway_config* is None (test path), an empty registry
        # plus a permissive policy is constructed so the gateway is
        # still usable in unit tests that drive ``_handle_event``
        # directly.  Production callers always pass a config; the
        # secure defaults the config schema enforces are what give
        # the policy real teeth.
        peers = list(gateway_config.peers) if gateway_config is not None else []
        self._peer_registry = PeerRegistry(peers)

        # Per-source policy: today the only knob is the structural
        # carve-out toggle on ``ForgeSpec``.  Future service categories
        # that grow their own per-instance trigger knobs will plug in
        # here the same way.  The dict is keyed on the event ``source``
        # string the source itself stamps on every ``RawIncomingEvent``
        # (e.g. ``"github"``, ``"gitlab"``).
        #
        # We key on the **resolved** forge list (operator-declared +
        # synthesized for project fork URLs that lacked an explicit
        # forge entry), not on ``gateway_config.forges`` directly.
        # Walking only the operator-declared list would silently fall
        # back to the default policy for forges that were synthesised
        # from project URLs -- a latent bug that bit our peer-validation
        # code in the same way before this refactor moved peer
        # cross-config validation into the resolver.
        source_policies: dict[str, SourceTriggerPolicy] = {}
        if gateway_config is not None:
            from thorn.gateway._config import _resolve_forges_and_projects

            resolved_forges, _resolved_projects = _resolve_forges_and_projects(
                gateway_config,
            )
            for forge in resolved_forges:
                # Forges whose `name` collides on the event-source
                # string would step on each other; in v1 there is one
                # source-string per forge type, so the per-forge knob
                # is best-effort when an operator has multiple GitHub
                # entries.  The "right" granularity for that case is
                # per-service-instance keying, which is on the deferred
                # list and tracked in the plan doc.
                source_policies[forge.type or forge.name] = SourceTriggerPolicy(
                    deliver_structural_from_non_peers=(
                        forge.deliver_structural_from_non_peers
                    ),
                )

        self._trigger_policy = TriggerAuthorizationPolicy(
            self._peer_registry,
            source_policies=source_policies,
        )
        self._formatter = NotificationFormatter(
            peer_registry=self._peer_registry,
            policy=self._trigger_policy,
        )

        # Make the registry visible to agent-side tools running inside
        # this runtime: ``thorn.tools.peers`` looks up
        # ``get_context().runtime.peer_registry``, and the runtime is
        # the natural place for shared agency-level state.  The
        # gateway is the single owner of the registry's lifecycle, so
        # we install it here rather than asking each tool / runtime
        # consumer to thread it through.
        self._runtime.peer_registry = self._peer_registry

    @property
    def health_monitor(self) -> ProviderHealthMonitor:
        """The shared provider-health monitor wired into every scheduler.

        Exposed for tests and operator status pages -- callers can
        ``snapshot()`` it to render current state without reaching
        into per-scheduler internals.
        """
        return self._health_monitor

    def health_snapshot(self) -> ProviderHealthSnapshot:
        """Convenience accessor: ``self.health_monitor.snapshot()``.

        Provided so the common observability case does not require
        importing :class:`ProviderHealthSnapshot` just to call
        ``.snapshot()``.
        """
        return self._health_monitor.snapshot()

    async def run(self) -> None:
        """Enter the runtime context and run all sources until shutdown.

        Installs signal handlers on POSIX; falls back to
        ``KeyboardInterrupt`` on Windows.
        """
        self._stop_event = asyncio.Event()

        async with self._runtime:
            self._install_signal_handlers()

            try:
                await self._startup()

                for source in self._sources:
                    task = asyncio.create_task(
                        source.start(self._handle_event),
                    )
                    self._source_tasks.append(task)

                if self._source_tasks:
                    asyncio.create_task(
                        self._stop_when_sources_done(),
                    )

                self._start_heartbeat()

                log.info(
                    "Gateway started with %d source(s)", len(self._sources),
                )

                try:
                    await self._stop_event.wait()
                except (KeyboardInterrupt, asyncio.CancelledError):
                    pass
            finally:
                # ``finally`` covers both the normal exit path (sources
                # finished or stop_event set by signal) and the
                # startup-failure path: if ``_startup`` raises after
                # the bundled-broker supervisor has already brought
                # the stack up, ``shutdown`` is what tears it back
                # down so we don't leak compose projects on the host.
                await self.shutdown()

    # ------------------------------------------------------------------
    # Startup
    # ------------------------------------------------------------------

    async def _startup(self) -> None:
        """One-time startup: sweep, in-flight index rebuild, agent load.

        Order matters:

        1. Rebuild the in-flight index *first* so the sweep's
           ``delete`` calls update the index we hand out to queues
           later.
        2. Run the sweep, which mutates the filesystem (and the
           rebuilt index) to restore invariants.
        3. Pre-load every persisted agent and stand up its scheduler.
        4. Activate every session whose inbox still has
           ``prompt_pending`` work, so drivers resume without having
           to wait for a fresh incoming event.  The session
           activation pass is what closes the "sweep leaves items on
           disk but no driver runs" loop: if an inbox has anything in
           ``{pending, in_progress}``, the session goes on the
           scheduler's work list immediately.
        """
        if self._started:
            return

        rebuilt = rebuild_in_flight_index(self._runtime.paths)
        # Merge the rebuilt contents into the runtime's existing
        # index object so any previously-wired consumers keep their
        # reference stable.  The runtime seeds ``in_flight_index``
        # with an empty instance; the merge is effectively a
        # replacement in practice.
        self._runtime.in_flight_index.clear()
        self._runtime.in_flight_index.bulk_add(rebuilt.snapshot())

        report = run_startup_sweep(
            self._runtime.paths,
            self._runtime.address_book,
            in_flight_index=self._runtime.in_flight_index,
        )
        if (
            report.session_handled_dispatched
            or report.session_errored_dispatched
            or report.session_confirmed_cleaned
            or report.service_in_progress_reverted
            or report.service_confirmed_cleaned
            or report.service_misplaced
            or report.temp_files_removed
            or report.dispatch_skipped_unresolved
        ):
            log.info("Startup sweep completed: %s", report)

        # Load every persisted agent and run the per-service account
        # validation pass before handing the agent to a scheduler.
        # Validation has to happen here -- not just at the CLI surface
        # -- because the gateway is the single point that owns the
        # agent objects the schedulers (and broker registration, and
        # sandbox executor materialisation) actually use.  An agent
        # whose accounts are still ``UntypedAccountConfig`` instances
        # would crash service-driven code paths (e.g.
        # :meth:`BrokerableService.broker_credential_plans`) on first
        # use, so we surface configuration errors here at startup
        # rather than mid-flight.
        from thorn.core._account import validate_agent_accounts

        for agent_id in self._runtime.sessions.list_agent_ids():
            agent = self._runtime.get_or_create_agent(agent_id)
            validate_agent_accounts(agent, self._runtime.get_service)
            self._ensure_scheduler_for_agent(agent)

        self._warn_if_planned_egress_allowlist_configured()

        # Bundled-broker bring-up runs *before* broker registration
        # and *before* sandbox-executor preload.  The supervisor
        # synthesises the broker URLs/key into ``gateway_config.broker``
        # in place, then patches the runtime's sandbox config with
        # the per-project egress network so per-agent container
        # launches join the broker's network rather than the host
        # default.  Both effects must be visible by the time
        # :meth:`_register_broker_bindings` and
        # :meth:`_preload_sandbox_executors` run, which is why this
        # is the first thing past the warning above.  No-op when the
        # broker mode is "external" (the operator brought up their
        # own OneCLI) or when the broker block is absent / disabled.
        await self._maybe_start_bundled_broker()

        # Phase D: register every loaded agent with the broker before
        # the sandbox executor preload runs, so the per-agent
        # container can be built with broker bindings from the start
        # (no later mutation of HTTPS_PROXY / placeholders).  No-op
        # when the broker block is absent or disabled.
        await self._register_broker_bindings()

        # Install the binding lookup *after* the registration loop
        # has populated ``self._broker_bindings``.  Doing it here
        # (rather than in ``__init__``) means the runtime sees a
        # stable dict for the rest of the gateway's lifetime;
        # ``_preload_sandbox_executors`` below is the first call site
        # that materialises sandbox executors, and it consults the
        # lookup once per agent at construction time.
        self._runtime.set_sandbox_broker_binding_lookup(
            self.broker_binding_for,
        )

        # Phase B: kick the per-agent sandbox executor's
        # ``start()`` now (rather than lazily on first ``invoke``)
        # so any image-missing / image-broken / runtime-misconfigured
        # failures surface during gateway startup.  The operator
        # then sees the failure on the same console they ran ``thorn
        # serve`` from, rather than on the first webhook delivery
        # minutes later.  Failure is hard: we let the exception
        # escape so ``thorn serve`` exits with a non-zero status.
        await self._preload_sandbox_executors()

        activated = await self._activate_sessions_with_work()
        if activated:
            log.info(
                "Startup activation: submitted %d session(s) with pending work",
                activated,
            )

        self._started = True

    async def _preload_sandbox_executors(self) -> None:
        """Start every agent's sandbox executor up-front.

        Runs after schedulers are wired but before sessions are
        activated.  For the subprocess backend this just spawns the
        ``thorn-toolhost`` subprocess; for the container backend it
        provisions the per-agent OCI container, so any
        ``SandboxImageMissingError`` (or other startup failure) lands
        on the gateway's console rather than on the first incoming
        event.

        Failures propagate.  Phase B's stance is hard-fail: if the
        operator's sandbox is misconfigured we want ``thorn serve``
        to exit non-zero with the original exception, not run with a
        partially-degraded agency where some agents work and others
        don't.

        Starts run in parallel via :func:`asyncio.gather` so a slow
        cold image extraction does not serialize across agents.
        """
        executors = []
        for agent_id, scheduler in self._schedulers.items():
            agent = scheduler.agent
            executor = self._runtime.get_or_create_sandbox_executor(agent)
            if executor is None:
                continue
            executors.append((agent_id, executor))
        if not executors:
            return
        log.info(
            "Preloading sandbox executors for %d agent(s)", len(executors),
        )
        # ``return_exceptions=True`` lets us log every failure even when
        # multiple agents fail, but we still raise the first one so
        # ``thorn serve`` exits non-zero.
        results = await asyncio.gather(
            *(ex.start() for _aid, ex in executors),
            return_exceptions=True,
        )
        first_exc: BaseException | None = None
        for (agent_id, _executor), result in zip(executors, results):
            if isinstance(result, BaseException):
                log.error(
                    "Sandbox executor preload failed for agent %s: %s",
                    agent_id, result,
                )
                if first_exc is None:
                    first_exc = result
        if first_exc is not None:
            raise first_exc

    async def _activate_sessions_with_work(self) -> int:
        """Submit every session with ``prompt_pending`` work to its scheduler.

        Walks every session inbox on disk via
        :meth:`AgencyPaths.iter_session_inbox_locations`.  For each
        inbox whose ``prompt_pending`` view is non-empty, loads the
        session, registers the canonical
        :class:`~thorn.runtime.SessionInbox` in the address book, and
        submits ``(session, inbox)`` to the agent's scheduler so the
        driver picks up where it left off.

        Returns the number of sessions activated.

        Inboxes holding only ``handled`` / ``errored`` items (which
        the sweep could not dispatch because the RSVP target was
        unresolved) are skipped: a driver wake-up would not help them
        move forward.  Empty inbox directories are also skipped.
        """
        activated = 0
        for agent_id, session_key, inbox_dir in (
            self._runtime.paths.iter_session_inbox_locations()
        ):
            address = SessionAddress(agent_id, session_key)
            # Cheap probe: a fresh SessionInbox does no I/O until we
            # call ``prompt_pending``.  The instance is discarded if
            # we skip; if we don't, ``_ensure_inbox`` constructs the
            # canonical one that goes into the address book and is
            # shared with the driver.
            probe = SessionInbox(
                inbox_dir,
                address,
                in_flight_index=self._runtime.in_flight_index,
            )
            if not probe.prompt_pending():
                continue

            if not self._runtime.sessions.agent_exists(agent_id):
                # A session directory without a persisted agent file
                # is an inconsistency the sweep can't repair.  Log
                # and skip rather than silently fabricating an agent.
                log.warning(
                    "Skipping activation of session %s: agent %s is not "
                    "persisted", address, agent_id,
                )
                continue

            agent = self._runtime.get_or_create_agent(agent_id)
            scheduler = self._ensure_scheduler_for_agent(agent)

            ws = self._runtime.paths.session_workspace(agent_id, session_key)
            ws.mkdir(parents=True, exist_ok=True)
            session = self._runtime.get_or_create_session(
                agent, session_key,
                workspace_root=ws,
                logical_agent_workspace_path=(
                    self._runtime.paths.agent_workspace_mount(agent_id)
                ),
            )
            _, inbox = self._ensure_inbox(agent, session_key)

            # ``scheduler.submit`` returns promptly: it registers the
            # driver (spawning its task) and kicks it if needed, but
            # does not await the prompt round itself.  Submitting
            # sequentially is fine for any realistic number of
            # sessions-with-work, and keeps startup deterministic.
            await scheduler.submit(session, inbox)
            activated += 1
        return activated

    # ------------------------------------------------------------------
    # Agent / scheduler / inbox lifecycle
    # ------------------------------------------------------------------

    def _ensure_scheduler_for_agent(self, agent: Agent) -> AgentScheduler:
        """Return the scheduler for *agent*, creating one on first sight.

        The scheduler is wired with:

        - the injected :data:`PromptDispatcher` (normally the
          inbox-aware default);
        - an async ``save_session`` adapter around
          :meth:`Runtime.save_session`;
        - the canonical :func:`default_progress_evictor` so stalled
          sessions get their oldest item evicted at the N-strikes
          threshold;
        - the gateway's shared :class:`ProviderHealthMonitor`,
          which gates every prompt round on the breaker's state and
          also suppresses eviction while the provider is degraded.

        Scheduler instances are kept in the gateway's
        ``_schedulers`` map keyed by :class:`AgentID`.  On shutdown
        the gateway shuts every scheduler down with a bounded grace
        period.
        """
        if agent.id is None:
            raise ValueError("Cannot build a scheduler for an agent without an id")
        existing = self._schedulers.get(agent.id)
        if existing is not None:
            return existing
        scheduler = AgentScheduler(
            agent=agent,
            prompt_dispatcher=self._prompt_dispatcher,
            concurrency=self._agent_concurrency,
            save_session=self._save_session_async,
            progress_evictor=default_progress_evictor(
                self._runtime.address_book,
                health_monitor=self._health_monitor,
            ),
            health_monitor=self._health_monitor,
        )
        self._schedulers[agent.id] = scheduler

        # Sandbox-executor materialisation is deferred to
        # ``_preload_sandbox_executors`` (during startup) or to the
        # first ``ToolExecutor`` lookup on demand.  The previous
        # implementation eagerly cached the executor here as
        # defensive bookkeeping against a "race" between concurrent
        # prompt rounds, but the dict-based cache in
        # :meth:`Runtime.get_or_create_sandbox_executor` runs on a
        # single asyncio thread without ``await`` points, so there
        # is no actual race to defend against.  Skipping the eager
        # call here is what allows Phase D's broker-binding lookup
        # to be consulted at construction time -- if we cached the
        # executor before ``_register_broker_bindings`` had run,
        # the binding would be lost on the cache hit at preload.
        return scheduler

    async def _save_session_async(self, session: Session) -> None:
        """Async adapter around :meth:`Runtime.save_session`.

        Kept as a method (not a lambda) so the scheduler's own
        exception logging points at a real qualified name.  The
        underlying save is synchronous and small; we do not move it
        to a worker thread because the scheduler already serializes
        per-session, so back-to-back writes don't contend.
        """
        self._runtime.save_session(session)

    def _ensure_inbox(
        self,
        agent: Agent,
        session_key: SessionKey,
    ) -> tuple[SessionAddress, SessionInbox]:
        """Return the :class:`SessionInbox` for ``(agent, session_key)``.

        On first call for a given ``(agent, session_key)`` pair, we:

        - Construct the inbox pointing at
          :meth:`AgencyPaths.session_inbox_dir`, wiring in the
          runtime-wide :class:`InFlightIndex`.
        - Register it in the :class:`AddressBook` under the
          corresponding :class:`SessionAddress` so that step-2
          dispatch (from handling transitions, progress eviction,
          and the startup sweep) can resolve the target.

        Subsequent calls return the same inbox.  A concurrent
        startup-sweep-created inbox is silently replaced so that the
        long-lived gateway instance becomes the canonical registrant.
        """
        if agent.id is None:
            raise ValueError("Cannot build an inbox for an agent without an id")
        address = SessionAddress(agent.id, session_key)
        existing = self._inboxes.get(address)
        if existing is not None:
            return address, existing

        inbox_dir = self._runtime.paths.session_inbox_dir(
            agent.id, session_key,
        )
        inbox = SessionInbox(
            inbox_dir, address,
            in_flight_index=self._runtime.in_flight_index,
        )
        self._inboxes[address] = inbox

        book = self._runtime.address_book
        # The sweep may have registered an ephemeral inbox at this
        # address already (when recovering stuck handled/errored
        # items).  Replace it with our canonical, gateway-owned
        # instance so the rest of the runtime holds a single source
        # of truth.  The two instances point at the same directory,
        # so there is no concurrency hazard from the swap.
        previous = book.get(address)
        if previous is inbox:
            return address, inbox
        if previous is not None:
            book.unregister(address)
        book.register(address, inbox)
        return address, inbox

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    def _resolve_agent(self, event: FormattedEvent) -> Agent:
        """Map an event to the agent instance that should handle it.

        Routing logic (in priority order):

        1. If the event carries an explicit ``agent_id``, use that.
        2. Look for a pre-configured coordinator agent in the runtime
           store.  This remains a single-coordinator fallback for
           events from sources that are not tied to an agent account.
        3. Fall back to a bare ``Agent`` with the default ID.

        Agent-owned account sources should stamp ``agent_id`` before
        handing events to the gateway.  Project, repository, issue,
        and change-request identity belongs in the session key within
        that owning agent rather than in a cross-agent coordinator
        lookup.
        """
        if event.agent_id is not None:
            return self._get_or_load_validated_agent(event.agent_id)

        persisted_ids = self._runtime.sessions.list_agent_ids()
        if persisted_ids:
            return self._get_or_load_validated_agent(persisted_ids[0])

        return self._get_or_load_validated_agent(_DEFAULT_AGENT_ID)

    def _get_or_load_validated_agent(self, agent_id: AgentID) -> Agent:
        """Return the gateway-owned validated agent instance for *agent_id*.

        ``SessionStore.load_agent`` intentionally deserializes accounts
        as ``UntypedAccountConfig`` because service schemas are not known
        at parse time.  Gateway startup validates persisted agents before
        schedulers use them, so routing must prefer that in-memory
        scheduler instance over a fresh store load.  Otherwise forge
        tools can see untyped accounts even though startup validation
        succeeded.
        """
        existing_scheduler = self._schedulers.get(agent_id)
        if existing_scheduler is not None:
            return existing_scheduler.agent

        agent = self._runtime.get_or_create_agent(agent_id)
        if not self._started and not self._schedulers:
            # Unit tests and diagnostic callers sometimes ask the
            # router what it *would* pick before gateway startup has
            # registered services and validated accounts.  The real
            # event path runs after startup, where either the scheduler
            # branch above returns the validated instance or the
            # validation below surfaces a configuration error.
            return agent

        from thorn.core._account import validate_agent_accounts

        validate_agent_accounts(agent, self._runtime.get_service)
        return agent

    async def _handle_event(self, event: RawIncomingEvent) -> None:
        """Apply the trigger-authorization policy and dispatch on success.

        This is the source-facing entry point.  Sources construct a
        :class:`RawIncomingEvent` and hand it to this callback; the
        gateway runs it through the
        :class:`~thorn.gateway._formatter.NotificationFormatter`,
        which consults the peer registry and the
        :class:`~thorn.gateway._trigger_policy.TriggerAuthorizationPolicy`
        to decide whether to deliver, deliver-with-banner, or drop.
        Dropped events are logged at INFO and discarded; delivered
        events are routed through :meth:`_dispatch_formatted`.

        Drops are *terminal* (see the threat-model docs and the
        plan's open question #8): the source's mark-read / mark-done
        behaviour fires uniformly for delivered, deduped, and
        dropped events alike, so a non-peer event the policy
        rejects does not stay in the source's "pending" set forever
        and there is no replay path that could later inject it.
        """
        result = self._formatter.process(event)
        if isinstance(result, FormatterDrop):
            return
        assert isinstance(result, FormatterDelivery)
        await self._dispatch_formatted(result.event)

    async def _dispatch_formatted(self, event: FormattedEvent) -> None:
        """Post a formatted event to the right inbox and wake its scheduler.

        Does *not* execute the prompt directly.  The session driver
        inside the scheduler picks up the posted notification and
        runs a prompt round at its own pace, under the per-agent
        concurrency cap and per-session serialization guarantee.

        The session workspace is derived from ``AgencyPaths``::

            paths.session_workspace(agent_id, session_key)

        which expands to ``<workspace_root>/<agent_id>/<session_key>/``.
        The directory is pre-created here so the agent always starts
        with a valid working directory.

        Agent identity and session metadata are persisted *before*
        the notification is posted, so ``session_exists`` reports
        the session as soon as an event has arrived for it -- even if
        the first prompt has not yet run (or has crashed).  This is
        a semantic shift from the pre-inbox gateway, where
        persistence was gated on a successful prompt.

        Source-level deduplication: if the event carries an
        ``external_key`` that is already recorded in the runtime's
        :class:`InFlightIndex`, the event is silently dropped and no
        session state is touched.  This is the gateway-side half of
        the contract described in :mod:`thorn.runtime._in_flight_index`
        -- sources set stable, namespaced keys on events they emit and
        the gateway enforces at-most-one-in-flight-per-key across the
        whole agency.  Sources may (and typically do) call back to
        their external platform regardless of dedup (for example, the
        GitLab source marks a TODO as done after every post, so that
        the *platform* also stops resurfacing the underlying entity).
        """
        log.info(
            "Handling event from %s (session=%s)",
            event.source, event.session_key,
        )

        if event.external_key is not None:
            if event.external_key in self._runtime.in_flight_index:
                log.info(
                    "Dropping duplicate event from %s "
                    "(external_key=%s already in flight)",
                    event.source, event.external_key,
                )
                return

        agent = self._resolve_agent(event)

        # Persist identity on first sight so the resolved agent is
        # discoverable by a subsequent run even before any prompt has
        # completed.  Idempotent for already-persisted agents.
        if not self._runtime.sessions.agent_exists(agent.id):
            self._runtime.save_agent(agent)

        scheduler = self._ensure_scheduler_for_agent(agent)

        ws = self._runtime.paths.session_workspace(
            agent.id, event.session_key,
        )
        ws.mkdir(parents=True, exist_ok=True)

        session = self._runtime.get_or_create_session(
            agent, event.session_key,
            workspace_root=ws,
            logical_agent_workspace_path=(
                self._runtime.paths.agent_workspace_mount(agent.id)
            ),
        )
        # Persist session metadata up front so the session directory
        # exists before any prompt round runs.  The scheduler's saver
        # will overwrite ``session.json`` after each successful round
        # with an updated ``last_active`` timestamp.
        self._runtime.save_session(session)

        address, inbox = self._ensure_inbox(agent, event.session_key)

        spec = NotificationSpec(
            source=event.source,
            content=event.content,
            target=address,
            metadata=dict(event.metadata),
            external_key=event.external_key,
        )
        notification = inbox.post(spec)
        log.info(
            "Posted %s to %s (source=%s, session=%s)",
            notification.id, address, event.source, event.session_key,
        )
        await scheduler.submit(session, inbox)

    # ------------------------------------------------------------------
    # Operator heartbeat
    # ------------------------------------------------------------------

    def _start_heartbeat(self) -> None:
        if self._heartbeat_task is not None:
            return
        self._heartbeat_started_at = gateway_heartbeat_timestamp()
        self._write_heartbeat(status="running")
        self._heartbeat_task = asyncio.create_task(
            self._heartbeat_loop(),
            name="thorn-gateway-heartbeat",
        )

    async def _heartbeat_loop(self) -> None:
        while True:
            await asyncio.sleep(max(1.0, self._heartbeat_interval_s))
            self._write_heartbeat(status="running")

    async def _stop_heartbeat(self) -> None:
        task = self._heartbeat_task
        self._heartbeat_task = None
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        if self._heartbeat_started_at is not None:
            self._write_heartbeat(status="stopped")

    def _write_heartbeat(self, *, status: str) -> None:
        try:
            write_gateway_heartbeat(
                gateway_heartbeat_path(self._runtime.paths.home_root),
                self._heartbeat_payload(status=status),
            )
        except Exception:
            log.exception("Failed to write gateway heartbeat")

    def _heartbeat_payload(self, *, status: str) -> dict[str, Any]:
        updated_at = gateway_heartbeat_timestamp()
        payload: dict[str, Any] = {
            "status": status,
            "pid": os.getpid(),
            "started_at": self._heartbeat_started_at,
            "updated_at": updated_at,
            "heartbeat_interval_s": self._heartbeat_interval_s,
            "provider_health": self._provider_health_payload(),
            "sources": [
                source.status_snapshot().to_json()
                for source in self._sources
            ],
            "broker": self._broker_status_payload(),
            "sandbox": self._sandbox_status_payload(),
        }
        if status == "stopped":
            payload["stopped_at"] = updated_at
        return payload

    def _provider_health_payload(self) -> dict[str, Any]:
        snapshot = self.health_snapshot()
        return {
            "state": snapshot.state.value,
            "recent_failure_count": snapshot.recent_failure_count,
            "seconds_until_probe": snapshot.seconds_until_probe,
            "probe_in_flight": snapshot.probe_in_flight,
            "consecutive_probe_failures": (
                snapshot.consecutive_probe_failures
            ),
        }

    def _broker_status_payload(self) -> dict[str, Any]:
        config = self._gateway_config
        broker = config.broker if config is not None else None
        supervisor = self._bundled_broker_supervisor
        return {
            "enabled": broker.enabled if broker is not None else False,
            "mode": broker.mode if broker is not None else None,
            "binding_count": len(self._broker_bindings),
            "bundled_project": (
                supervisor.project_name if supervisor is not None else None
            ),
            "bundled_network": (
                supervisor.egress_network_name
                if supervisor is not None
                else None
            ),
        }

    def _sandbox_status_payload(self) -> dict[str, Any]:
        config = self._runtime.sandbox_config
        return {
            "executor_enabled": self._runtime.sandbox_executor_enabled,
            "backend": config.backend if config is not None else None,
            "egress_network": (
                config.egress_network if config is not None else None
            ),
        }

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    async def _stop_when_sources_done(self) -> None:
        """Set the stop event when all source tasks have completed.

        In production, polling sources loop forever and never return
        from ``start()``, so this only fires on signal-driven
        shutdown (where tasks are cancelled) or when a finite source
        is used (e.g. in tests).  This ensures ``run()`` terminates
        cleanly on all platforms without relying on platform-specific
        signal behavior.
        """
        await asyncio.gather(*self._source_tasks, return_exceptions=True)
        if self._stop_event is not None:
            self._stop_event.set()

    async def shutdown(self) -> None:
        """Stop all sources, then drain / cancel every scheduler."""
        log.info("Gateway shutting down ...")
        for source in self._sources:
            try:
                await source.stop()
            except Exception:
                log.exception(
                    "Error stopping source %s", type(source).__name__,
                )

        for task in self._source_tasks:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        self._source_tasks.clear()

        # Schedulers: graceful shutdown with a bounded grace period.
        # ``asyncio.gather`` lets the grace period run in parallel
        # across every agent, so the total shutdown time is bounded
        # by ``shutdown_timeout`` rather than growing linearly with
        # agent count.  Each scheduler internally cancels its
        # drivers on timeout.
        schedulers = list(self._schedulers.values())
        self._schedulers.clear()
        if schedulers:
            await asyncio.gather(
                *(
                    s.shutdown(timeout=self._shutdown_timeout)
                    for s in schedulers
                ),
                return_exceptions=True,
            )
        # Clear the binding lookup before broker teardown so any
        # late callers (e.g. a sandbox executor that survives
        # scheduler shutdown) see ``None`` rather than dangling
        # references to deleted bindings.
        self._runtime.set_sandbox_broker_binding_lookup(None)

        # Phase D: tear down broker registrations after schedulers
        # have drained so any in-flight tool calls had their proxy
        # routing intact for the duration.  Bundled compose teardown is
        # in the ``finally`` because compose owns the actual stack
        # lifecycle; a stale or dead admin endpoint must not leave the
        # containers running.
        try:
            await self._teardown_broker_bindings()
        finally:
            # Bundled broker tear-down runs *after* per-agent broker
            # bindings have been unwound -- the DELETE calls in
            # _teardown_broker_bindings still need the broker to be
            # serving the admin API.  Best-effort: a hung ``compose
            # down`` should not block exit; the supervisor logs a
            # remediation hint when it cannot tear the stack down so
            # operators know to run ``thorn broker down`` manually.
            await self._maybe_shutdown_bundled_broker()

        self._inboxes.clear()
        await self._stop_heartbeat()
        log.info("Gateway stopped.")

    # ------------------------------------------------------------------
    # Phase D: broker registration / teardown
    # ------------------------------------------------------------------

    def _warn_if_planned_egress_allowlist_configured(self) -> None:
        """Log when ``planned_egress_allowlist`` is non-empty.

        Direct per-host allow-list enforcement remains Phase D R3.
        The active security boundary is only the sandbox's OCI
        network membership, so planned exceptions need to be noisy at
        startup rather than silently masquerading as an enforced
        policy.
        """
        config = self._runtime.sandbox_config
        if config is None:
            return
        if not config.planned_egress_allowlist:
            return
        rendered = ", ".join(
            f"{e.host}:{e.port}" for e in config.planned_egress_allowlist
        )
        log.warning(
            "sandbox.planned_egress_allowlist is configured (%s), but "
            "it has no runtime effect today. Thorn does not enforce "
            "per-host direct egress yet (Phase D open question R3); "
            "container outbound is restricted only by "
            "sandbox.egress_network membership and the attached OCI "
            "network topology.",
            rendered,
        )

    def broker_binding_for(
        self,
        agent_id: AgentID,
    ) -> BrokerBinding | None:
        """Return the broker binding for *agent_id*, if any.

        Public read-only accessor consumed by the per-agent sandbox
        executor (Phase D work item 6) so it can inject the proxy
        URL, CA mount, and placeholder env entries when building the
        container.  ``None`` when the broker is disabled, *agent_id*
        is unknown, or the agent's registration was skipped.
        """
        return self._broker_bindings.get(agent_id)

    async def _maybe_start_bundled_broker(self) -> None:
        """Bring up the bundled broker stack when configured.

        Runs only when:

        * A :class:`GatewayConfig` is present (test paths that pass
          ``gateway_config=None`` have no broker shape to honour).
        * The broker block is set with ``mode == "bundled"`` and
          ``enabled``.
        * The resolved sandbox backend is ``container`` (a bundled
          broker without a container is a configuration mistake;
          handled in the schema validator, but we hard-fail here too
          as a defence in depth).

        On success, mutates the in-memory ``gateway_config.broker``
        so it carries the supervisor-discovered URLs/key (the
        downstream broker code reads from this object), and sets
        the runtime's ``sandbox.egress_network`` to the supervisor's
        per-project network so per-agent container launches join the
        right Docker network.

        Failures propagate -- bringing the broker up is a precondition
        for the rest of startup, and we want ``thorn serve`` to exit
        non-zero with a clear error rather than run with the broker
        silently absent.  Best-effort cleanup of the partial stack
        happens in :meth:`shutdown`, which is invoked from
        :meth:`run`'s ``finally`` block on any startup failure.
        """
        config = self._gateway_config
        if config is None or config.broker is None or not config.broker.enabled:
            return
        if config.broker.mode != "bundled":
            return

        sandbox_config = self._runtime.sandbox_config
        sandbox_backend = (
            sandbox_config.backend if sandbox_config is not None else None
        )
        if sandbox_backend != "container":
            # The schema validator on GatewayConfig already drops the
            # bundled-broker default when sandbox.backend resolves to
            # subprocess.  Reaching here means the operator wrote
            # ``broker.mode = "bundled"`` *explicitly* alongside
            # ``sandbox.backend = "subprocess"``.  Hard-fail rather
            # than silently no-op: this combination is incoherent
            # (no container to inject the proxy into) and a soft
            # warning would let the operator believe the broker is
            # active when it isn't.
            raise BundledBrokerError(
                "broker.mode='bundled' requires sandbox.backend='container'; "
                f"resolved sandbox backend is {sandbox_backend or 'subprocess'!r}.  "
                "Either set sandbox.backend='container' (or omit the sandbox "
                "block to inherit the secure default), or set "
                "broker.enabled=false to opt out of broker integration entirely.",
            )

        log.info("Bringing up bundled OneCLI broker (this may take ~10s) ...")
        images = config.broker.bundled_images
        if not isinstance(images, BundledBrokerImageConfig):
            images = BundledBrokerImageConfig.model_validate(images)
        supervisor = self._bundled_broker_supervisor_factory(images=images)
        self._bundled_broker_supervisor = supervisor
        synthesized = await supervisor.start()

        # Mutate the gateway_config.broker in place so the existing
        # _register_broker_bindings code path -- which reaches for
        # ``config.broker.admin_url`` / ``admin_api_key`` /
        # ``proxy_url`` -- picks up the supervisor's values without
        # any further wiring.  The mutation is safe: gateway startup
        # is single-threaded and no consumer reads ``config.broker``
        # before this point.
        config.broker = synthesized

        # Patch the runtime's sandbox config with the supervisor's
        # per-project Docker network so that per-agent sandbox
        # container launches join it (and therefore reach the broker
        # by service DNS while having no NAT to the host network).
        # Also a single-threaded mutation; happens before any sandbox
        # executor materialises.
        if (
            sandbox_config is not None
            and supervisor.egress_network_name is not None
        ):
            sandbox_config.egress_network = supervisor.egress_network_name
            log.info(
                "Bundled broker: sandbox containers will join network %r",
                supervisor.egress_network_name,
            )

    async def _register_broker_bindings(self) -> None:
        """Register every loaded agent with the broker.

        No-op when:

        * *gateway_config* is missing the broker block, or it has
          ``enabled: false`` -- the operator has not opted in to
          broker integration.
        * The agency's sandbox backend resolves to ``subprocess``
          -- the in-process daemon shares the host's network stack
          and credentials, so there is no container to inject
          ``HTTPS_PROXY`` / placeholder env vars into.  Registering
          would swap real credentials for placeholders in memory
          but the subprocess daemon would still be the one making
          HTTPS calls, hitting auth failures everywhere.  We log a
          warning so an operator who configured both broker and
          subprocess backend at once notices the mismatch.

        Hard-fails (raises) when any agent's registration fails: a
        partial broker enrolment is worse than not enrolling at all
        because the audit invariant would be violated and the
        operator might not notice.
        """
        config = self._gateway_config
        if config is None or config.broker is None or not config.broker.enabled:
            return

        sandbox_config = self._runtime.sandbox_config
        sandbox_backend = (
            sandbox_config.backend if sandbox_config is not None else None
        )
        if sandbox_backend != "container":
            log.warning(
                "Broker is enabled in gateway.json but the sandbox "
                "backend resolves to %r; skipping broker registration "
                "because subprocess-mode tools share the host network "
                "and would not pick up the proxy / CA / placeholder "
                "env injection.  Set sandbox.backend to 'container' "
                "(or remove the broker block) to make this "
                "consistent.",
                sandbox_backend or "subprocess (no sandbox block)",
            )
            return

        if not self._schedulers:
            return

        ca_path = self._resolve_broker_ca_path(config.broker)

        client = self._broker_client_factory(
            config, self._bundled_broker_supervisor,
        )
        try:
            # Fetch the broker's CA once at startup and persist it
            # to the resolved path.  This is the canonical CA
            # acquisition path: no operator-side volume wiring
            # required, no assumption about how the broker is
            # deployed (compose / standalone / remote).  Re-fetching
            # on every gateway start also picks up CA rotations
            # automatically.  Run in a worker thread because httpx
            # is sync and the file write blocks; also gives us a
            # clean place to log what happened.
            await asyncio.to_thread(
                self._fetch_and_persist_broker_ca, client, ca_path,
            )
            log.info("Broker: cached CA certificate at %s", ca_path)

            for agent_id, scheduler in self._schedulers.items():
                agent = scheduler.agent
                # Registration is sync (a small handful of HTTP calls
                # over a single connection) -- run in a worker thread
                # so it does not block the event loop, but keep the
                # logic itself synchronous since it is much easier to
                # reason about in that form.
                binding = await asyncio.to_thread(
                    register_agent_with_broker,
                    client=client, agent=agent,
                    service_lookup=self._runtime.get_service,
                    ca_certificate_path=str(ca_path),
                )
                # Render a per-agent gitconfig file when the binding
                # declares any ``http.<url>.extraHeader`` entries
                # (e.g. GitHub / GitLab git HTTPS routing).  The file
                # lives outside the agent's bind-mounted home so the
                # agent can't edit it, and the sandbox runtime
                # mounts it read-only at the fixed
                # ``GIT_CONFIG_GLOBAL`` path.  The updated binding
                # (with ``git_config_path`` populated) is what gets
                # stashed in ``_broker_bindings`` so the
                # sandbox-launch lookup sees the path too.
                binding = self._render_agent_gitconfig(agent_id, binding)
                self._broker_bindings[agent_id] = binding
                log.info(
                    "Broker: registered agent %s as %s with %d secret(s)",
                    agent_id, binding.agent_id, len(binding.secret_ids),
                )
        except Exception:
            # A registration failure mid-loop leaves us with some
            # agents registered with the broker.  Best-effort cleanup
            # on the way out so the next startup attempt is on a
            # clean slate.
            await self._teardown_broker_bindings(client_override=client)
            raise
        self._broker_client = client

    def _resolve_broker_ca_path(self, broker: BrokerConfig) -> Path:
        """Resolve where the broker CA certificate should live on disk.

        When the operator set ``broker.ca_certificate_path`` in
        ``gateway.json``, that wins.  Otherwise we derive a default
        under the agency home (``<agency_home>/onecli-ca.pem``):
        the agency home is already a directory the gateway owns
        with appropriate permissions, so we don't add a new
        operator-facing concept just to find a place to drop a
        public certificate.
        """
        if broker.ca_certificate_path is not None:
            return Path(broker.ca_certificate_path)
        return self._runtime.paths.home_root / "onecli-ca.pem"

    @staticmethod
    def _fetch_and_persist_broker_ca(
        client: BrokerClient,
        ca_path: Path,
    ) -> None:
        """Fetch the CA via the admin API and write it to *ca_path*.

        Synchronous; the caller dispatches it through
        :func:`asyncio.to_thread`.  Creates parent directories as
        needed.  The CA cert is a public artefact (the
        certificate's whole point is to be widely distributed for
        validation) so file mode is plain ``0o644`` -- secret-grade
        permissions would be cargo cult.
        """
        ca_bytes = client.fetch_ca_certificate()
        ca_path.parent.mkdir(parents=True, exist_ok=True)
        ca_path.write_bytes(ca_bytes)
        ca_path.chmod(0o644)

    def _render_agent_gitconfig(
        self,
        agent_id: Any,
        binding: BrokerBinding,
    ) -> BrokerBinding:
        """Write *binding*'s gitconfig file and return an updated binding.

        Returns *binding* unchanged when it has no
        ``git_extra_headers`` entries -- the runtime reads
        ``git_config_path`` as ``None`` in that case and skips the
        sandbox-side mount entirely.  Otherwise writes the rendered
        INI to ``<agent_framework_dir>/sandbox/gitconfig`` (mode
        ``0o644``; the file ships placeholders, not real credentials,
        so loosened perms are fine).

        The file is overwritten on every registration: gateway
        startup always lands with a freshly-generated placeholder,
        even if a stale file from a prior run is present.  Teardown
        unlinks it so an operator auditing the tree sees no
        ambient files between runs.
        """
        if not binding.git_extra_headers:
            return binding
        sandbox_dir = self._runtime.paths.agent_sandbox_dir(agent_id)
        sandbox_dir.mkdir(parents=True, exist_ok=True)
        config_path = sandbox_dir / "gitconfig"
        config_path.write_text(_render_git_extra_headers(binding.git_extra_headers))
        config_path.chmod(0o644)
        return dataclasses.replace(binding, git_config_path=str(config_path))

    async def _teardown_broker_bindings(
        self,
        *,
        client_override: BrokerClient | None = None,
    ) -> None:
        """Best-effort delete of every registered agent + its secrets."""
        client = client_override if client_override is not None else self._broker_client
        if client is None:
            return

        bindings = list(self._broker_bindings.items())
        self._broker_bindings.clear()

        for agent_id, binding in bindings:
            # Unlink the rendered gitconfig, if any, before we free
            # the broker-side state.  The file is a one-shot
            # per-startup artefact, not part of the agent's
            # long-lived state, so leaving it behind would be
            # misleading.  Best-effort: a race with a manual
            # removal is a no-op, anything else gets logged.
            if binding.git_config_path is not None:
                try:
                    Path(binding.git_config_path).unlink(missing_ok=True)
                except OSError as exc:
                    log.warning(
                        "Broker: failed to unlink agent %s gitconfig "
                        "at %s: %s",
                        agent_id, binding.git_config_path, exc,
                    )
            try:
                await asyncio.to_thread(client.delete_agent, binding.agent_id)
                for secret_id in binding.secret_ids:
                    await asyncio.to_thread(client.delete_secret, secret_id)
            except Exception as exc:
                log.warning(
                    "Broker: teardown failed for agent %s: %s",
                    agent_id, exc,
                )

        try:
            await asyncio.to_thread(client.close)
        except Exception:
            log.exception("Broker: error closing client during teardown")
        if client_override is None:
            self._broker_client = None

    async def _maybe_shutdown_bundled_broker(self) -> None:
        """Tear down the bundled broker stack if we brought one up.

        Idempotent and best-effort: ``BundledBrokerSupervisor.shutdown``
        already swallows compose-down errors and logs them; this
        wrapper exists to (a) clear the supervisor reference so a
        re-entry from a signal-driven double-shutdown is a no-op, and
        (b) make the call site in :meth:`shutdown` self-documenting.
        """
        supervisor = self._bundled_broker_supervisor
        if supervisor is None:
            return
        self._bundled_broker_supervisor = None
        try:
            await supervisor.shutdown()
        except Exception:
            log.exception(
                "Bundled broker: unexpected error during supervisor "
                "shutdown; the compose stack may be orphaned.  Run "
                "`thorn broker down` to clean up.",
            )

    def _install_signal_handlers(self) -> None:
        """Register SIGINT/SIGTERM to trigger a clean shutdown on POSIX."""
        if sys.platform == "win32":
            return
        loop = asyncio.get_running_loop()
        assert self._stop_event is not None
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, self._stop_event.set)


__all__ = [
    "DEFAULT_SHUTDOWN_TIMEOUT_SECONDS",
    "Gateway",
]
