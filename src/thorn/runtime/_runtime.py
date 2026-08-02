"""Runtime: the persistent execution environment for Thorn.

The ``Runtime`` is the central service object that manages the provider,
event sink, workspace configuration, and a session store.  It produces
``ExecutionContext`` instances for individual operations and manages the
lifecycle of agent instances and their sessions.

Used as an async context manager, it sets up the ambient
``ExecutionContext`` so that ``agent.prompt()`` works automatically::

    async with runtime:
        result = await agent.prompt("do something")

Every Thorn deployment -- ``thorn run``, ``thorn chat``, or the
gateway daemon -- creates a ``Runtime``.  For one-shot ``thorn run``
the overhead is negligible: it is essentially what the CLI does today,
just structured through a uniform abstraction.
"""

from __future__ import annotations

import contextlib
import contextvars
import hashlib
import os
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Protocol, TypeVar, runtime_checkable

from thorn.core._agent import Agent
from thorn.core._context import (
    EventSink,
    ExecutionContext,
    NullEventSink,
    reset_context,
    set_context,
)
from thorn.core._provider import LLMConfig, LLMProvider, load_provider_from_config
from thorn.core._read_file_history import (
    SESSION_LEDGER_V1_READ_FILE_REUSE_POLICY,
    ReadFileReusePolicy,
)
from thorn.core._service import Service
from thorn.core._session import Session
from thorn.core._validation_convergence import ValidationConvergencePolicy
from thorn.runtime._address import AddressBook
from thorn.runtime._in_flight_index import InFlightIndex
from thorn.runtime._paths import AgencyPaths
from thorn.runtime._session import AgentID, SessionKey
from thorn.runtime._store import SessionStore

if TYPE_CHECKING:
    from thorn.core._context import StatusProvider
    from thorn.core._context_ledger import ContextBudgetPolicy
    from thorn.core._executor import ToolExecutor
    from thorn.core._file_access import FileAccessPolicy
    from thorn.core._prompt_trace import PromptTraceRecorder
    from thorn.core._validation_tracker import ValidationTracker
    from thorn.gateway._config import SandboxConfig
    from thorn.sandbox._runtime import OCIRuntimeAdapter
    from thorn.toolhost._host import DaemonHost

_S = TypeVar("_S", bound=Service)

_UNIX_SOCKET_PATH_BUDGET_BYTES = 100
"""Conservative budget for Unix-domain socket path strings.

Linux's ``sockaddr_un.sun_path`` limit is typically 108 bytes including
the trailing NUL, and other Unix-like systems have similarly small
limits.  Keeping Thorn-generated fallback paths below 100 bytes leaves
headroom for platform-specific accounting without making callers reason
about the exact kernel ABI.
"""


def _toolhost_socket_path_for_subprocess(default_socket_path: Path) -> Path:
    """Return a subprocess-safe socket path for the toolhost daemon.

    The natural socket location lives under the agent control directory
    so all per-agent runtime artifacts are easy to inspect together.
    Deep gateway workspaces can exceed the Unix socket path limit,
    though, causing the daemon to crash before it can even write logs.
    For those cases, keep the log in the control dir but put the
    rendezvous socket in a short, user-private temp directory.
    """
    if len(os.fsencode(default_socket_path)) <= _UNIX_SOCKET_PATH_BUDGET_BYTES:
        return default_socket_path

    digest = hashlib.sha256(
        os.fsencode(default_socket_path),
    ).hexdigest()[:24]
    uid = getattr(os, "getuid", lambda: 0)()
    temp_root = Path(tempfile.gettempdir()) / f"thorn-toolhost-{uid}"
    temp_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    with contextlib.suppress(Exception):
        temp_root.chmod(0o700)
    socket_dir = temp_root / digest
    socket_dir.mkdir(mode=0o700, exist_ok=True)
    with contextlib.suppress(Exception):
        socket_dir.chmod(0o700)
    return socket_dir / "s.sock"


@runtime_checkable
class SandboxBrokerBinding(Protocol):
    """Minimal shape the runtime needs to wire a sandbox to a broker.

    A structural match for :class:`thorn.gateway._broker.BrokerBinding`
    so the gateway can hand its binding objects to the runtime
    without the runtime importing from ``thorn.gateway`` at runtime
    (which would create a layering inversion: the gateway depends on
    the runtime, not the other way around).

    Carries exactly the pieces the
    :class:`~thorn.sandbox._container.ContainerHostConfig` consumes:
    the proxy URL (with embedded Basic auth), the host-side path of
    the broker's CA certificate, the placeholder env entries that
    replace literal credentials inside the sandbox, and (optional)
    the host-side path to a per-agent gitconfig that routes git
    HTTPS through the broker.

    Annotated as plain attributes (rather than ``@property``) so that
    a ``@dataclass(frozen=True)`` exposing the same field names is
    automatically structurally compatible at runtime under
    :func:`isinstance` -- ``runtime_checkable`` Protocols match
    attribute presence, and dataclass attributes are simpler to
    satisfy from concrete implementations than properties.
    """

    proxy_url: str
    ca_certificate_path: str
    placeholder_env: tuple[tuple[str, str], ...]
    git_extra_headers: tuple[tuple[str, str], ...]
    git_config_path: str | None


SandboxBrokerBindingLookup = Callable[
    [AgentID], "SandboxBrokerBinding | None",
]
"""Callable the gateway installs on the runtime to expose bindings.

Returning ``None`` for an agent means "no binding registered yet"
(broker disabled, or the agent was created post-startup before the
broker registration loop has run); the runtime then builds the
sandbox without any broker wiring.  This degrades gracefully: the
sandbox runs, but with no credential injection -- the audit
invariant catches this on the brain side before it can ship a
literal credential into the container.
"""


class Runtime:
    """Persistent execution environment for Thorn.

    Manages provider configuration, event sinks, workspace settings,
    and a session store.  Acts as a factory for ``ExecutionContext``
    instances used by the agent loop, and manages the lifecycle of
    agent instances and their sessions.

    Use as an async context manager to set the ambient
    ``ExecutionContext`` for the duration of the block::

        async with runtime:
            result = await agent.prompt("hello")
    """

    def __init__(
        self,
        *,
        provider: LLMProvider,
        event_sink: EventSink | None = None,
        workspace_root: Path,
        global_ignores: FileAccessPolicy | None = None,
        context_window: int | None = None,
        context_budget_policy: ContextBudgetPolicy | None = None,
        read_file_observation_policy: ReadFileReusePolicy = (
            SESSION_LEDGER_V1_READ_FILE_REUSE_POLICY
        ),
        read_file_reuse_policy: ReadFileReusePolicy | None = None,
        validation_convergence_policy: ValidationConvergencePolicy = (
            ValidationConvergencePolicy.BASELINE
        ),
        session_store: SessionStore | None = None,
        validation_tracker: ValidationTracker | None = None,
        status_providers: list[StatusProvider] | None = None,
        paths: AgencyPaths | None = None,
        address_book: AddressBook | None = None,
        in_flight_index: InFlightIndex | None = None,
        sandbox_executor_enabled: bool = False,
        sandbox_config: SandboxConfig | None = None,
        oci_runtime_adapter: OCIRuntimeAdapter | None = None,
        sandbox_broker_binding_lookup: (
            SandboxBrokerBindingLookup | None
        ) = None,
        subprocess_tool_workspace_root: Path | None = None,
        provider_config: LLMConfig | None = None,
        prompt_trace_recorder: PromptTraceRecorder | None = None,
    ) -> None:
        self.provider = provider
        self.provider_config = provider_config
        self.prompt_trace_recorder = prompt_trace_recorder
        self._configured_provider_cache: dict[str, LLMProvider] = {}
        self.event_sink: EventSink = event_sink or NullEventSink()
        self.workspace_root = workspace_root
        self.global_ignores = global_ignores
        self.context_window = context_window
        self.context_budget_policy = context_budget_policy
        self.read_file_observation_policy = read_file_observation_policy
        self.read_file_reuse_policy = read_file_reuse_policy
        self.validation_convergence_policy = validation_convergence_policy
        self.status_providers: list[StatusProvider] = list(status_providers or [])
        if validation_tracker is not None:
            self.status_providers.append(validation_tracker)

        if paths is None:
            paths = AgencyPaths(
                home_root=workspace_root / ".thorn",
                workspace_root=workspace_root,
            )
        self.paths = paths

        # Fail loudly if we're running against a pre-Phase-A on-disk
        # layout.  No silent migration -- see the plan's "No automatic
        # migration" section.
        paths.raise_if_legacy_layout()

        if session_store is None:
            session_store = SessionStore(paths)
        self.sessions = session_store

        # Address book and in-flight index are both always present so
        # that consumers (dispatchers, tools, sweep) can assume they
        # exist without null-checking.  Callers that run a sweep at
        # startup may replace ``in_flight_index`` with one rebuilt
        # from the filesystem; we keep fields mutable on purpose.
        self.address_book: AddressBook = address_book or AddressBook()
        self.in_flight_index: InFlightIndex = (
            in_flight_index or InFlightIndex()
        )

        # Peer registry: populated by the gateway from the
        # operator-declared peer list once it has parsed
        # ``gateway.json``.  Defaults to an empty registry so that
        # tests / direct ``Runtime`` users (the in-process CLI paths)
        # do not have to construct one explicitly.  An empty registry
        # means "no peers" which causes the trigger-authorization
        # policy to drop conversational events from every actor;
        # that is the right strict-default for non-gateway code
        # paths (which are not subject to a peer model at all and
        # generally do not run the formatter).
        from thorn.gateway._peer import PeerRegistry as _PeerRegistry
        self.peer_registry: _PeerRegistry = _PeerRegistry([])

        self._services: dict[str, Service] = {}

        self._context: ExecutionContext | None = None
        self._context_token: contextvars.Token[ExecutionContext] | None = None

        self._sandbox_executor_enabled = sandbox_executor_enabled
        self._sandbox_executors: dict[AgentID, ToolExecutor] = {}
        self._sandbox_config: SandboxConfig | None = sandbox_config
        self._subprocess_tool_workspace_root = subprocess_tool_workspace_root
        self._oci_runtime_adapter: OCIRuntimeAdapter | None = oci_runtime_adapter
        # Lazily-resolved adapter cache so multiple agents on a
        # container backend share one OCI runtime client (each adapter
        # only instantiates ``shutil.which`` lookups, but resolving it
        # once means the agency-wide ``oci_runtime`` decision is made
        # at most once per process).
        self._oci_runtime_adapter_resolved: bool = (
            oci_runtime_adapter is not None
        )

        # Phase D: the gateway installs a callback so per-agent
        # ``ContainerHostConfig`` construction can pick up the broker
        # binding (proxy URL + CA path + placeholder env).  Default
        # ``None`` keeps the subprocess and container backends working
        # in setups without a broker (CLI, tests, single-shot
        # ``thorn run``).
        self._sandbox_broker_binding_lookup: (
            SandboxBrokerBindingLookup | None
        ) = sandbox_broker_binding_lookup

    # -- Service registry ---------------------------------------------------

    def register_service(self, service: Service) -> None:
        """Register a named service in the agency.

        Raises :class:`ValueError` if a service with the same name is
        already registered.
        """
        if service.name in self._services:
            raise ValueError(
                f"Service {service.name!r} is already registered"
            )
        self._services[service.name] = service

    def get_service(self, name: str) -> Service:
        """Look up a service by name.

        Raises :class:`KeyError` if no service with that name exists.
        """
        try:
            return self._services[name]
        except KeyError:
            registered = ", ".join(sorted(self._services)) or "(none)"
            raise KeyError(
                f"No service named {name!r}. "
                f"Registered services: {registered}"
            ) from None

    def get_services_by_type(self, service_type: type[_S]) -> list[_S]:
        """Return all registered services of the given type."""
        return [
            s for s in self._services.values()
            if isinstance(s, service_type)
        ]

    # -- Context management -------------------------------------------------

    def create_context(
        self,
        *,
        system_prompts: list[str] | None = None,
    ) -> ExecutionContext:
        """Create an ``ExecutionContext`` from this runtime's configuration.

        The context inherits the runtime's provider, event sink, workspace
        root, and other ambient settings.  An optional list of system
        prompts can be supplied for the context.
        """
        return ExecutionContext(
            provider=self.provider,
            event_sink=self.event_sink,
            workspace_root=self.workspace_root,
            global_ignores=self.global_ignores,
            context_window=self.context_window,
            context_budget_policy=self.context_budget_policy,
            read_file_observation_policy=self.read_file_observation_policy,
            read_file_reuse_policy=self.read_file_reuse_policy,
            validation_convergence_policy=self.validation_convergence_policy,
            system_prompts=list(system_prompts or []),
            status_providers=self.status_providers,
            agency_root_directory=self.workspace_root,
            runtime=self,
            prompt_trace_recorder=self.prompt_trace_recorder,
        )

    def provider_for_agent(self, agent: Agent) -> LLMProvider:
        """Return the effective provider for *agent*.

        Most agents use the runtime-wide provider.  Agents loaded from
        ``agent.json`` may carry an ``llm_config`` override; in that case
        the override is merged with the agency default and cached by its
        non-secret configuration key.
        """
        return self._provider_for_llm_overrides(getattr(agent, "llm_config", None))

    def provider_for_session(self, session: Session) -> LLMProvider:
        """Return the effective provider for *session*.

        The layering model is agency config, then agent config, then
        session config.  Thorn does not create session-specific LLM config
        yet, but keeping this as the prompt-time entry point makes that
        future extension local to session construction and persistence.
        """
        return self._provider_for_llm_overrides(
            getattr(session.agent, "llm_config", None),
            getattr(session, "llm_config", None),
        )

    def _provider_for_llm_overrides(self, *raw_configs: Any) -> LLMProvider:
        effective_config = self.provider_config
        has_override = False

        for raw_config in raw_configs:
            override = self._coerce_llm_config(raw_config)
            if override is None or not override.has_operator_config():
                continue

            has_override = True
            if effective_config is None:
                effective_config = LLMConfig()
            effective_config = effective_config.merged_with(override)

        if not has_override:
            return self.provider
        if effective_config is None or not effective_config.has_operator_config():
            return self.provider

        cache_key = effective_config.cache_key()
        cached = self._configured_provider_cache.get(cache_key)
        if cached is not None:
            return cached

        provider = load_provider_from_config(effective_config)
        self._configured_provider_cache[cache_key] = provider
        return provider

    @staticmethod
    def _coerce_llm_config(raw_config: Any) -> LLMConfig | None:
        if raw_config is None:
            return None
        if isinstance(raw_config, LLMConfig):
            return raw_config
        return LLMConfig.model_validate(raw_config)

    @property
    def context(self) -> ExecutionContext:
        """The ambient ``ExecutionContext`` set by ``async with runtime:``.

        Raises ``RuntimeError`` if the runtime is not being used as a
        context manager.
        """
        if self._context is None:
            raise RuntimeError(
                "Runtime.context is only available inside 'async with runtime:'"
            )
        return self._context

    async def __aenter__(self) -> Runtime:
        self._context = self.create_context()
        self._context_token = set_context(self._context)
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        await self.shutdown_sandbox_executors()
        await self._close_providers()

        if self._context_token is not None:
            reset_context(self._context_token)
            self._context_token = None
        self._context = None

    async def shutdown_sandbox_executors(self) -> None:
        """Tear down every cached per-agent sandbox executor.

        ``Gateway.shutdown`` calls this before broker teardown so sandbox
        containers detach from broker-owned networks before compose tries to
        remove them.  ``Runtime.__aexit__`` also calls it, making the helper
        idempotent for non-gateway CLI paths.
        """
        executors = list(self._sandbox_executors.values())
        self._sandbox_executors.clear()
        for executor in executors:
            try:
                await executor.aclose()
            except Exception:
                # Best-effort cleanup: keep tearing down the rest even
                # if one daemon refuses to die cleanly.  The next
                # process start will recreate its socket.
                pass

    async def _close_providers(self) -> None:
        providers = [
            self.provider,
            *self._configured_provider_cache.values(),
        ]
        self._configured_provider_cache.clear()

        closed_provider_ids: set[int] = set()
        for provider in providers:
            provider_id = id(provider)
            if provider_id in closed_provider_ids:
                continue
            closed_provider_ids.add(provider_id)
            try:
                await provider.aclose()
            except Exception:
                pass

    # -- Sandbox executor pool ---------------------------------------------

    def set_sandbox_broker_binding_lookup(
        self, lookup: SandboxBrokerBindingLookup | None,
    ) -> None:
        """Install (or clear) the broker-binding lookup callback.

        The gateway calls this once startup-time broker registration
        has populated its bindings dict, so subsequent
        :meth:`get_or_create_sandbox_executor` calls can pick up the
        right binding for each agent.  Passing ``None`` clears the
        callback (used during shutdown / teardown so we never keep a
        reference to a closed broker session).

        Pre-existing cached executors are *not* rebuilt: the
        gateway's startup ordering ensures schedulers exist (and
        broker bindings get registered) *before* any sandbox
        executor is materialised, so a late install is the
        steady-state path for the in-memory pool.
        """
        self._sandbox_broker_binding_lookup = lookup

    @property
    def sandbox_config(self) -> SandboxConfig | None:
        """The agency-wide :class:`SandboxConfig`, or ``None`` if absent.

        Exposed read-only so the gateway (and tests) can inspect
        the resolved sandbox block without reaching into the
        private ``_sandbox_config`` attribute.  ``None`` matches the
        constructor default and means the operator did not write a
        ``sandbox`` block in ``gateway.json`` -- the runtime keeps
        Phase-A subprocess defaults in that case.
        """
        return self._sandbox_config

    @property
    def sandbox_executor_enabled(self) -> bool:
        """Whether per-agent sandbox executors are enabled.

        When ``True`` (the gateway / CLI default), each agent gets its
        own :class:`~thorn.toolhost.DaemonToolExecutor`, lazily started
        on first use.  When ``False`` (the test default), every venue
        runs in-process and no daemon subprocesses are ever spawned.
        """
        return self._sandbox_executor_enabled

    def _build_sandbox_executor(self, agent: Agent) -> ToolExecutor:
        """Construct a :class:`DaemonToolExecutor` for *agent*.

        The agency's ``sandbox.backend`` (with per-agent override) picks
        between the Phase-A subprocess host and the Phase-B container
        host; everything downstream of that choice (the executor, the
        socket protocol, the brain-side tool-call shape) is identical.

        Factored out so tests / subclasses can override it with a
        stub executor (e.g. one driven over an in-memory socket pair)
        without having to subclass and override the lookup method too.
        """
        from thorn.sandbox._resolve import resolve_sandbox_config
        from thorn.toolhost._executor import (
            DaemonExecutorConfig,
            DaemonToolExecutor,
        )

        if agent.id is None:
            raise ValueError(
                "Cannot start a sandbox executor for an agent without an id"
            )

        resolved = resolve_sandbox_config(
            self._sandbox_config,
            getattr(agent, "sandbox_override", None),
        )

        control_dir = self.paths.agent_control_dir(agent.id)
        control_dir.mkdir(parents=True, exist_ok=True)
        default_socket_path = self.paths.agent_toolhost_socket(agent.id)
        if resolved.backend == "subprocess":
            socket_path = _toolhost_socket_path_for_subprocess(
                default_socket_path,
            )
        else:
            socket_path = default_socket_path
        log_path = control_dir / "toolhost.log"

        config_kwargs: dict[str, Any] = {}
        if resolved.backend == "container":
            config_kwargs["connect_timeout_s"] = resolved.container_ready_timeout_s
            config_kwargs["handshake_timeout_s"] = resolved.container_ready_timeout_s

        tool_workspace_root = self.paths.agent_workspace_mount(agent.id)
        if (
            resolved.backend == "subprocess"
            and self._subprocess_tool_workspace_root is not None
        ):
            tool_workspace_root = self._subprocess_tool_workspace_root

        config = DaemonExecutorConfig(
            socket_path=socket_path,
            agent_id=str(agent.id),
            home_path=self.paths.agent_home_mount(agent.id),
            workspace_root=tool_workspace_root,
            log_path=log_path,
            **config_kwargs,
        )

        host = self._build_daemon_host(
            agent=agent, resolved=resolved,
            socket_path=socket_path, log_path=log_path,
            tool_workspace_root=tool_workspace_root,
        )
        return DaemonToolExecutor(config, host=host)

    def _build_daemon_host(
        self,
        *,
        agent: Agent,
        resolved: Any,
        socket_path: Path,
        log_path: Path,
        tool_workspace_root: Path,
    ) -> DaemonHost:
        """Pick the right :class:`DaemonHost` for *agent*'s resolved config.

        Branches on ``resolved.backend``: ``"subprocess"`` keeps the
        Phase-A in-process daemon, ``"container"`` provisions a per-agent
        OCI container.  All path/control-dir wiring is shared so callers
        do not need to know which branch ran.
        """
        from thorn.toolhost._host import (
            SubprocessDaemonHost,
            SubprocessDaemonHostConfig,
        )

        assert agent.id is not None  # validated by caller

        if resolved.backend == "subprocess":
            host_config = SubprocessDaemonHostConfig(
                socket_path=socket_path,
                agent_id=str(agent.id),
                home_path=self.paths.agent_home_mount(agent.id),
                workspace_root=tool_workspace_root,
                log_path=log_path,
            )
            return SubprocessDaemonHost(host_config)

        # Container backend.
        from thorn.sandbox._container import (
            ContainerDaemonHost,
            ContainerHostConfig,
            derive_container_name,
        )

        adapter = self._resolve_oci_runtime_adapter(resolved)

        control_dir = self.paths.agent_control_dir(agent.id)
        host_home = self.paths.agent_home_mount(agent.id)
        host_workspace = self.paths.agent_workspace_mount(agent.id)
        host_home.mkdir(parents=True, exist_ok=True)
        host_workspace.mkdir(parents=True, exist_ok=True)

        dev_mount = self._dev_mount_runtime_path(resolved)

        # Phase D: pick up the broker binding for this agent, if the
        # gateway has registered one.  ``None`` is the broker-disabled
        # path (single-shot ``thorn run``, tests, gateway with no
        # broker block); the container then runs without proxy
        # interception.  Subprocess backend never consults this
        # lookup -- broker integration is conditional on the
        # container backend by design (the in-process daemon shares
        # the host's network stack and credentials, so broker
        # injection has nothing to attach to).
        binding: SandboxBrokerBinding | None = None
        if self._sandbox_broker_binding_lookup is not None:
            binding = self._sandbox_broker_binding_lookup(agent.id)

        broker_proxy_url: str | None = None
        broker_ca_host_path: Path | None = None
        broker_placeholder_env: tuple[tuple[str, str], ...] = ()
        git_config_host_path: Path | None = None
        if binding is not None:
            broker_proxy_url = binding.proxy_url
            broker_ca_host_path = Path(binding.ca_certificate_path)
            broker_placeholder_env = binding.placeholder_env
            if binding.git_config_path is not None:
                git_config_host_path = Path(binding.git_config_path)

        # Phase E: surface tmpfs scratch mounts whenever the rootfs
        # is read-only.  ``DEFAULT_TMPFS_MOUNTS`` covers ``/tmp`` and
        # ``/var/tmp`` with sane sizes; operators with unusual needs
        # can extend the surface later via a dedicated config field
        # rather than having to disable read-only entirely.
        from thorn.sandbox._container import DEFAULT_TMPFS_MOUNTS

        tmpfs_mounts = (
            DEFAULT_TMPFS_MOUNTS if resolved.read_only_root else ()
        )

        container_config = ContainerHostConfig(
            agent_id=str(agent.id),
            container_name=derive_container_name(str(agent.id)),
            image=resolved.image,
            adapter=adapter,
            host_home_dir=host_home,
            host_workspace_dir=host_workspace,
            host_control_dir=control_dir,
            env_passthrough=tuple(resolved.env_passthrough),
            extra_env=tuple(resolved.extra_env),
            broker_proxy_url=broker_proxy_url,
            broker_ca_host_path=broker_ca_host_path,
            broker_placeholder_env=broker_placeholder_env,
            git_config_host_path=git_config_host_path,
            egress_network=resolved.egress_network,
            dev_mount_runtime=dev_mount,
            container_ready_timeout_s=resolved.container_ready_timeout_s,
            capabilities_drop=resolved.capabilities_drop,
            capabilities_add=resolved.capabilities_add,
            security_opts=resolved.security_opts,
            read_only_root=resolved.read_only_root,
            tmpfs_mounts=tmpfs_mounts,
            memory_limit=resolved.memory_limit,
            cpu_limit=resolved.cpu_limit,
            pid_limit=resolved.pid_limit,
        )
        return ContainerDaemonHost(container_config)

    def _resolve_oci_runtime_adapter(self, resolved: Any) -> OCIRuntimeAdapter:
        """Lazily resolve and cache the agency's OCI runtime adapter.

        Resolution happens at most once per runtime so per-agent host
        construction does not pay the ``shutil.which`` cost on every
        call.  Tests can pre-inject an adapter via ``oci_runtime_adapter``
        in the ``Runtime`` constructor; that value short-circuits this
        method entirely.
        """
        if self._oci_runtime_adapter_resolved and self._oci_runtime_adapter is not None:
            return self._oci_runtime_adapter
        from thorn.sandbox._runtime import select_oci_runtime
        adapter = select_oci_runtime(resolved.oci_runtime)
        self._oci_runtime_adapter = adapter
        self._oci_runtime_adapter_resolved = True
        return adapter

    def _dev_mount_runtime_path(self, resolved: Any) -> Path | None:
        """Resolve the source-tree path for the ``dev_mount_runtime`` toggle.

        When the agency opts into ``dev_mount_runtime``, we mount the
        live ``thorn`` package source tree into the container so the
        in-container daemon picks up local edits without a rebuild.
        Returns ``None`` when the package's source path cannot be
        resolved (e.g. running from a wheel install) so the toggle
        gracefully degrades to "no-op" rather than failing the start.
        """
        if not resolved.dev_mount_runtime:
            return None
        try:
            from thorn import __file__ as thorn_file
        except Exception:
            return None
        return Path(thorn_file).resolve().parent.parent  # .../src/

    def get_or_create_sandbox_executor(
        self, agent: Agent,
    ) -> ToolExecutor | None:
        """Return the per-agent sandbox executor, creating it lazily.

        Returns ``None`` when ``sandbox_executor_enabled`` is ``False``
        (the test default), so call sites can use the result directly
        as ``ExecutionContext.sandbox_executor`` without conditional
        wiring.

        The executor itself is started lazily on first :meth:`invoke`;
        this method only constructs the wrapper and caches it on the
        runtime so subsequent agent rounds reuse the same daemon.
        """
        if not self._sandbox_executor_enabled:
            return None
        if agent.id is None:
            return None
        existing = self._sandbox_executors.get(agent.id)
        if existing is not None:
            return existing
        executor = self._build_sandbox_executor(agent)
        self._sandbox_executors[agent.id] = executor
        return executor

    async def shutdown_sandbox_executor(self, agent_id: AgentID) -> None:
        """Tear down a single per-agent sandbox executor, if any.

        The gateway calls this if it ever wants to retire a per-agent
        daemon mid-run (e.g. on an agent-removed admin event).  The
        normal teardown path is :meth:`__aexit__`, which closes every
        executor in the pool.
        """
        executor = self._sandbox_executors.pop(agent_id, None)
        if executor is None:
            return
        try:
            await executor.aclose()
        except Exception:
            pass

    # -- Agent lifecycle ----------------------------------------------------

    def create_agent(
        self,
        agent_class: type[Agent] = Agent,
        *,
        id: AgentID | str | None = None,
        name: str | None = None,
        workspace: Path | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Agent:
        """Create a new agent instance with identity fields populated.

        When *id* is ``None``, a UUID-based ID is generated.
        When *name* is ``None``, the ID is used as the display name.
        When *workspace* is ``None``, the agent's per-agent workspace
        mount (``<workspace_root>/agents/<id>/workspace/``) is used.
        The agent's home is always the mounted ``home/`` subtree.
        """
        if id is None:
            id = AgentID(str(uuid.uuid4()))
        elif not isinstance(id, AgentID):
            id = AgentID(id)

        home = self.paths.agent_home_mount(id)
        if workspace is None:
            workspace = self.paths.agent_workspace_mount(id)

        return agent_class(
            id=id,
            name=name if name is not None else str(id),
            workspace=workspace,
            home=home,
            metadata=metadata or {},
        )

    def get_or_create_agent(
        self,
        id: AgentID | str,
        agent_class: type[Agent] = Agent,
        *,
        name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Agent:
        """Retrieve a persisted agent, or create a new one if not found."""
        if not isinstance(id, AgentID):
            id = AgentID(id)
        if self.sessions.agent_exists(id):
            return self.sessions.load_agent(id)
        return self.create_agent(agent_class, id=id, name=name, metadata=metadata)

    def save_agent(self, agent: Agent) -> None:
        """Persist agent identity to disk."""
        self.sessions.save_agent(agent)

    # -- Session lifecycle --------------------------------------------------

    def get_or_create_session(
        self,
        agent: Agent,
        key: SessionKey | str,
        *,
        workspace_root: Path | None = None,
        logical_agent_workspace_path: Path | None = None,
    ) -> Session:
        """Retrieve a persisted session, or create a new one if not found.

        The session is scoped under the given agent instance.

        *workspace_root* and *logical_agent_workspace_path* are applied
        **only when creating** a new session.  Existing sessions retain
        the values they were created with; passing different values on
        a subsequent load is a no-op so later events cannot silently
        drift the session's working tree or context-walk upper bound.
        """
        if not isinstance(key, SessionKey):
            key = SessionKey(key)
        if agent.id is None:
            raise ValueError("Cannot manage sessions for an agent without an id")
        if self.sessions.session_exists(agent.id, key):
            return self.sessions.load_session(agent, key)
        now = datetime.now(timezone.utc)
        return Session(
            agent=agent,
            key=key,
            created_at=now,
            last_active=now,
            workspace_root=workspace_root,
            logical_agent_workspace_path=logical_agent_workspace_path,
        )

    def save_session(self, session: Session) -> None:
        """Persist a session, updating its ``last_active`` timestamp."""
        session.touch()
        self.sessions.save_session(session)


__all__ = [
    "Runtime",
]
