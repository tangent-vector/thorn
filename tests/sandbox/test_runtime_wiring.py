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


class TestNoAgentId:
    def test_no_id_means_no_executor(self, tmp_path: Path) -> None:
        runtime = _make_runtime(tmp_path)
        agent = Agent(name="anonymous")
        assert runtime.get_or_create_sandbox_executor(agent) is None
