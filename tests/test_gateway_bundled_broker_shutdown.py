"""Regression coverage for bundled broker shutdown ordering."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from thorn.core._credentials import ServiceCredential
from thorn.core._provider import MockProvider
from thorn.gateway._broker import BrokerBinding
from thorn.gateway._config import (
    BrokerConfig,
    BundledBrokerImageConfig,
    GatewayConfig,
    SandboxConfig,
)
from thorn.gateway._gateway import Gateway
from thorn.runtime import AgentID, Runtime


class _FakeBundledSupervisor:
    def __init__(self) -> None:
        self.egress_network_name = "thorn-broker-fake_thorn-broker"
        self.shutdown_calls = 0
        self._broker_config = BrokerConfig.model_construct(
            mode="bundled",
            enabled=True,
            admin_url="http://127.0.0.1:54321",
            admin_api_key_env_var=None,
            proxy_url="http://onecli:10255",
            ca_certificate_path=None,
            bundled_images=BundledBrokerImageConfig(),
        )
        self._admin_api_key = ServiceCredential("oc_fake")

    @property
    def admin_api_key(self) -> ServiceCredential:
        return self._admin_api_key

    async def start(self) -> BrokerConfig:
        return self._broker_config

    async def shutdown(self) -> None:
        self.shutdown_calls += 1


class _DeadAdminBrokerClient:
    def __init__(self) -> None:
        self.close_calls = 0

    def delete_agent(self, agent_id: str) -> None:
        raise httpx.ConnectError(
            "admin endpoint gone",
            request=httpx.Request(
                "DELETE", f"http://127.0.0.1:1/api/agents/{agent_id}",
            ),
        )

    def delete_secret(self, secret_id: str) -> None:
        raise AssertionError(
            f"delete_secret should not run after agent failure: {secret_id}",
        )

    def close(self) -> None:
        self.close_calls += 1


@pytest.mark.asyncio
async def test_gateway_shutdown_still_stops_bundled_broker_when_admin_delete_fails(
    tmp_path: Path,
) -> None:
    runtime = Runtime(
        provider=MockProvider(),
        workspace_root=tmp_path,
        sandbox_config=SandboxConfig(backend="container"),
    )
    config = GatewayConfig(
        sandbox=SandboxConfig(backend="container"),
        broker=BrokerConfig(mode="bundled", enabled=True),
    )
    supervisor = _FakeBundledSupervisor()
    gateway = Gateway(
        runtime=runtime,
        sources=[],
        gateway_config=config,
        bundled_broker_supervisor_factory=lambda **_: supervisor,
    )
    broker_client = _DeadAdminBrokerClient()

    await gateway._maybe_start_bundled_broker()
    gateway._broker_client = broker_client
    gateway._broker_bindings[AgentID("agent-with-stale-admin")] = BrokerBinding(
        agent_id="onecli-agent-id",
        secret_ids=("onecli-secret-id",),
        access_token=ServiceCredential("aoc_fake"),
        proxy_url="http://x:aoc_fake@onecli:10255",
        ca_certificate_path=str(tmp_path / "onecli-ca.pem"),
        placeholder_env=(),
    )

    await gateway.shutdown()

    assert supervisor.shutdown_calls == 1
    assert broker_client.close_calls == 1
    assert gateway._broker_bindings == {}
