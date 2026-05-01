"""Integration tests: ``Runtime`` picks the right ``DaemonHost``.

Verifies that the agency-wide ``SandboxConfig`` + per-agent
``sandbox_override`` actually reach :meth:`Runtime._build_sandbox_executor`
and that it constructs the right host class without trying to start
anything (no real subprocesses, no real containers).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from thorn.core._agent import Agent
from thorn.gateway._config import AgentSandboxOverride, SandboxConfig
from thorn.runtime import Runtime
from thorn.runtime._paths import AgencyPaths
from thorn.runtime._session import AgentID
from thorn.sandbox import (
    ContainerDaemonHost,
    FakeOCIRuntimeAdapter,
    derive_container_name,
)
from thorn.toolhost._host import SubprocessDaemonHost


class _StubProvider:
    """Bare provider stub so Runtime can be constructed in tests."""


def _make_runtime(
    tmp_path: Path,
    *,
    sandbox_config: SandboxConfig | None = None,
    oci_adapter: Any | None = None,
) -> Runtime:
    paths = AgencyPaths(
        home_root=tmp_path / "home",
        workspace_root=tmp_path / "ws",
    )
    paths.home_root.mkdir(parents=True, exist_ok=True)
    paths.workspace_root.mkdir(parents=True, exist_ok=True)
    return Runtime(
        provider=_StubProvider(),  # type: ignore[arg-type]
        workspace_root=paths.workspace_root,
        paths=paths,
        sandbox_executor_enabled=True,
        sandbox_config=sandbox_config,
        oci_runtime_adapter=oci_adapter,
    )


def _make_agent(
    *,
    override: AgentSandboxOverride | None = None,
    aid: str = "agent-x",
) -> Agent:
    agent = Agent(id=AgentID(aid), name=aid)
    if override is not None:
        agent.sandbox_override = override
    return agent


class TestSubprocessDefault:
    def test_no_sandbox_config_uses_subprocess_host(self, tmp_path: Path) -> None:
        runtime = _make_runtime(tmp_path)
        agent = _make_agent()
        executor = runtime.get_or_create_sandbox_executor(agent)
        assert executor is not None
        assert isinstance(executor.host, SubprocessDaemonHost)

    def test_subprocess_backend_explicit(self, tmp_path: Path) -> None:
        runtime = _make_runtime(
            tmp_path, sandbox_config=SandboxConfig(backend="subprocess"),
        )
        agent = _make_agent()
        executor = runtime.get_or_create_sandbox_executor(agent)
        assert executor is not None
        assert isinstance(executor.host, SubprocessDaemonHost)


class TestContainerBackend:
    def test_container_backend_uses_container_host(
        self, tmp_path: Path,
    ) -> None:
        adapter = FakeOCIRuntimeAdapter(present_images=["thorn-sandbox:test"])
        runtime = _make_runtime(
            tmp_path,
            sandbox_config=SandboxConfig(image="thorn-sandbox:test"),
            oci_adapter=adapter,
        )
        agent = _make_agent()
        executor = runtime.get_or_create_sandbox_executor(agent)
        assert executor is not None
        assert isinstance(executor.host, ContainerDaemonHost)
        host = executor.host
        assert host.image == "thorn-sandbox:test"
        assert host.container_name == derive_container_name("agent-x")
        assert host.adapter is adapter

    def test_per_agent_image_override_wins(self, tmp_path: Path) -> None:
        adapter = FakeOCIRuntimeAdapter(
            present_images=["thorn-sandbox:base", "rust-sandbox:1"],
        )
        runtime = _make_runtime(
            tmp_path,
            sandbox_config=SandboxConfig(image="thorn-sandbox:base"),
            oci_adapter=adapter,
        )
        agent = _make_agent(
            override=AgentSandboxOverride(image="rust-sandbox:1"),
        )
        executor = runtime.get_or_create_sandbox_executor(agent)
        assert executor is not None
        assert isinstance(executor.host, ContainerDaemonHost)
        assert executor.host.image == "rust-sandbox:1"

    def test_env_passthrough_is_combined(self, tmp_path: Path) -> None:
        adapter = FakeOCIRuntimeAdapter(present_images=["t:1"])
        runtime = _make_runtime(
            tmp_path,
            sandbox_config=SandboxConfig(
                image="t:1", env_passthrough=["LANG"],
            ),
            oci_adapter=adapter,
        )
        agent = _make_agent(
            override=AgentSandboxOverride(env_passthrough=["RUST_LOG"]),
        )
        executor = runtime.get_or_create_sandbox_executor(agent)
        assert isinstance(executor.host, ContainerDaemonHost)
        cfg = executor.host._config  # type: ignore[attr-defined]
        assert cfg.env_passthrough == ("LANG", "RUST_LOG")

    def test_per_agent_subprocess_backend_overrides_container(
        self, tmp_path: Path,
    ) -> None:
        adapter = FakeOCIRuntimeAdapter(present_images=["t:1"])
        runtime = _make_runtime(
            tmp_path,
            sandbox_config=SandboxConfig(image="t:1"),
            oci_adapter=adapter,
        )
        agent = _make_agent(
            override=AgentSandboxOverride(backend="subprocess"),
        )
        executor = runtime.get_or_create_sandbox_executor(agent)
        assert isinstance(executor.host, SubprocessDaemonHost)


class TestPhaseEHardeningWiring:
    """Phase E: the runtime translates the resolved hardening fields
    into the matching :class:`ContainerHostConfig` fields, and turns
    on the canonical tmpfs scratch mounts when ``read_only_root`` is
    enabled.  These tests confirm the wiring stays put across
    refactors."""

    def test_default_agency_block_lands_conservative_hardening(
        self, tmp_path: Path,
    ) -> None:
        # ``SandboxConfig(image="t:1")`` implicitly carries every
        # Phase-E default (ALL caps dropped, no-new-privileges,
        # read-only rootfs, 2G/2cpu/512pids).  The runtime should
        # surface those into ``ContainerHostConfig`` and wire up the
        # default tmpfs scratch mounts.
        adapter = FakeOCIRuntimeAdapter(present_images=["t:1"])
        runtime = _make_runtime(
            tmp_path,
            sandbox_config=SandboxConfig(image="t:1"),
            oci_adapter=adapter,
        )
        agent = _make_agent()
        executor = runtime.get_or_create_sandbox_executor(agent)
        assert isinstance(executor.host, ContainerDaemonHost)
        cfg = executor.host._config  # type: ignore[attr-defined]

        assert cfg.capabilities_drop == ("ALL",)
        assert cfg.capabilities_add == ()
        assert "no-new-privileges" in cfg.security_opts
        assert cfg.read_only_root is True
        assert cfg.memory_limit == "2G"
        assert cfg.cpu_limit == 2.0
        assert cfg.pid_limit == 512
        # Default tmpfs scratch mounts at /tmp and /var/tmp.
        from pathlib import Path as _Path
        targets = {tmpfs.target for tmpfs in cfg.tmpfs_mounts}
        assert _Path("/tmp") in targets
        assert _Path("/var/tmp") in targets

    def test_per_agent_disables_readonly_clears_tmpfs(
        self, tmp_path: Path,
    ) -> None:
        # When an agent opts out of read-only rootfs (typical for
        # dogfooding), the runtime should also drop the tmpfs
        # scratch mounts -- they're only there to keep canonical
        # scratch paths writable when the rootfs is locked down.
        adapter = FakeOCIRuntimeAdapter(present_images=["t:1"])
        runtime = _make_runtime(
            tmp_path,
            sandbox_config=SandboxConfig(image="t:1"),
            oci_adapter=adapter,
        )
        agent = _make_agent(
            override=AgentSandboxOverride(read_only_root=False),
        )
        executor = runtime.get_or_create_sandbox_executor(agent)
        assert isinstance(executor.host, ContainerDaemonHost)
        cfg = executor.host._config  # type: ignore[attr-defined]
        assert cfg.read_only_root is False
        assert cfg.tmpfs_mounts == ()

    def test_per_agent_resource_limit_overrides_propagate(
        self, tmp_path: Path,
    ) -> None:
        adapter = FakeOCIRuntimeAdapter(present_images=["t:1"])
        runtime = _make_runtime(
            tmp_path,
            sandbox_config=SandboxConfig(image="t:1"),
            oci_adapter=adapter,
        )
        agent = _make_agent(
            override=AgentSandboxOverride(
                memory_limit="32G", cpu_limit=12.0, pid_limit=4096,
            ),
        )
        executor = runtime.get_or_create_sandbox_executor(agent)
        cfg = executor.host._config  # type: ignore[attr-defined]
        assert cfg.memory_limit == "32G"
        assert cfg.cpu_limit == 12.0
        assert cfg.pid_limit == 4096

    def test_per_agent_caps_add_propagates(self, tmp_path: Path) -> None:
        adapter = FakeOCIRuntimeAdapter(present_images=["t:1"])
        runtime = _make_runtime(
            tmp_path,
            sandbox_config=SandboxConfig(image="t:1"),
            oci_adapter=adapter,
        )
        agent = _make_agent(
            override=AgentSandboxOverride(capabilities_add=["NET_RAW"]),
        )
        executor = runtime.get_or_create_sandbox_executor(agent)
        cfg = executor.host._config  # type: ignore[attr-defined]
        assert cfg.capabilities_add == ("NET_RAW",)
        # Drop list is still ALL (agency default).
        assert cfg.capabilities_drop == ("ALL",)


class TestBrokerBindingLookup:
    """Phase D: the runtime threads the gateway-installed binding
    lookup into ``ContainerHostConfig`` at executor-construction
    time.

    The lookup is *only* consulted on the container backend.  The
    subprocess backend does not get broker wiring because the
    in-process daemon shares the host's network and credentials, so
    proxy interception has nothing to attach to.
    """

    def _make_binding(
        self,
        *,
        proxy_url: str = "http://x:tok@broker:8443/",
        ca_path: str = "/host/path/to/ca.pem",
        placeholder_env: tuple[tuple[str, str], ...] = (
            ("GITHUB_TOKEN", "thorn-broker-placeholder-1"),
        ),
    ) -> Any:
        from dataclasses import dataclass

        @dataclass(frozen=True)
        class _Binding:
            proxy_url: str
            ca_certificate_path: str
            placeholder_env: tuple[tuple[str, str], ...]
            git_extra_headers: tuple[tuple[str, str], ...] = ()
            git_config_path: str | None = None

        return _Binding(
            proxy_url=proxy_url,
            ca_certificate_path=ca_path,
            placeholder_env=placeholder_env,
        )

    def test_container_backend_consults_lookup(self, tmp_path: Path) -> None:
        adapter = FakeOCIRuntimeAdapter(present_images=["t:1"])
        runtime = _make_runtime(
            tmp_path,
            sandbox_config=SandboxConfig(image="t:1"),
            oci_adapter=adapter,
        )
        agent = _make_agent(aid="agent-bound")
        binding = self._make_binding(proxy_url="http://x:abc@broker:9999/")
        runtime.set_sandbox_broker_binding_lookup(
            lambda agent_id: binding if str(agent_id) == "agent-bound" else None,
        )

        executor = runtime.get_or_create_sandbox_executor(agent)
        assert isinstance(executor.host, ContainerDaemonHost)
        cfg = executor.host._config  # type: ignore[attr-defined]
        assert cfg.broker_proxy_url == "http://x:abc@broker:9999/"
        assert cfg.broker_ca_host_path == Path("/host/path/to/ca.pem")
        assert cfg.broker_placeholder_env == (
            ("GITHUB_TOKEN", "thorn-broker-placeholder-1"),
        )

    def test_container_backend_no_lookup_means_no_broker(
        self, tmp_path: Path,
    ) -> None:
        adapter = FakeOCIRuntimeAdapter(present_images=["t:1"])
        runtime = _make_runtime(
            tmp_path,
            sandbox_config=SandboxConfig(image="t:1"),
            oci_adapter=adapter,
        )
        agent = _make_agent()
        executor = runtime.get_or_create_sandbox_executor(agent)
        assert isinstance(executor.host, ContainerDaemonHost)
        cfg = executor.host._config  # type: ignore[attr-defined]
        assert cfg.broker_proxy_url is None
        assert cfg.broker_ca_host_path is None
        assert cfg.broker_placeholder_env == ()

    def test_lookup_returning_none_yields_no_broker_wiring(
        self, tmp_path: Path,
    ) -> None:
        adapter = FakeOCIRuntimeAdapter(present_images=["t:1"])
        runtime = _make_runtime(
            tmp_path,
            sandbox_config=SandboxConfig(image="t:1"),
            oci_adapter=adapter,
        )
        agent = _make_agent()
        runtime.set_sandbox_broker_binding_lookup(lambda agent_id: None)

        executor = runtime.get_or_create_sandbox_executor(agent)
        assert isinstance(executor.host, ContainerDaemonHost)
        cfg = executor.host._config  # type: ignore[attr-defined]
        assert cfg.broker_proxy_url is None
        assert cfg.broker_placeholder_env == ()

    def test_subprocess_backend_ignores_lookup(self, tmp_path: Path) -> None:
        runtime = _make_runtime(
            tmp_path,
            sandbox_config=SandboxConfig(backend="subprocess"),
        )
        agent = _make_agent()
        called: list[AgentID] = []

        def lookup(agent_id: AgentID):
            called.append(agent_id)
            return self._make_binding()

        runtime.set_sandbox_broker_binding_lookup(lookup)

        executor = runtime.get_or_create_sandbox_executor(agent)
        assert isinstance(executor.host, SubprocessDaemonHost)
        assert called == [], (
            "Subprocess backend must not consult the broker binding "
            "lookup -- broker integration is conditional on the "
            "container backend"
        )


class TestEgressNetworkPlumbing:
    """Phase D: the resolved ``egress_network`` reaches the
    per-agent ``ContainerHostConfig`` via ``_build_daemon_host``."""

    def test_egress_network_propagates_from_sandbox_config(
        self, tmp_path: Path,
    ) -> None:
        adapter = FakeOCIRuntimeAdapter(present_images=["t:1"])
        runtime = _make_runtime(
            tmp_path,
            sandbox_config=SandboxConfig(
                image="t:1", egress_network="thorn-broker",
            ),
            oci_adapter=adapter,
        )
        agent = _make_agent()
        executor = runtime.get_or_create_sandbox_executor(agent)
        assert isinstance(executor.host, ContainerDaemonHost)
        cfg = executor.host._config  # type: ignore[attr-defined]
        assert cfg.egress_network == "thorn-broker"


class TestNoAgentId:
    def test_no_id_means_no_executor(self, tmp_path: Path) -> None:
        runtime = _make_runtime(tmp_path)
        agent = Agent(name="anonymous")
        assert runtime.get_or_create_sandbox_executor(agent) is None
