"""Focused tests for gateway routing to validated coordinator agents."""

from __future__ import annotations

from pathlib import Path

import pytest

from thorn.core._account import UntypedAccountConfig
from thorn.core._provider import MockProvider
from thorn.gateway import EventKind, Gateway, RawIncomingEvent, SandboxConfig
from thorn.gateway._bootstrap import bootstrap_coordinator
from thorn.gateway._config import instantiate_services, load_gateway_config
from thorn.runtime import AgencyPaths, AgentID, Runtime, SessionKey
from thorn.tools.forge import GitHubAccountConfig


@pytest.mark.asyncio
async def test_routing_reuses_startup_validated_agent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GITHUB_TOKEN", "fake-token")
    agent_id = AgentID("coord")
    bootstrap_coordinator(
        agency_home=tmp_path / "home",
        agency_workspace=tmp_path / "workspace",
        agent_id=str(agent_id),
        project_name="proj",
        project_url="https://github.com/example/proj",
    )
    config = load_gateway_config(tmp_path / "home")
    config.sandbox = SandboxConfig(backend="subprocess")
    config.broker = None

    paths = AgencyPaths.for_gateway(
        tmp_path / "home",
        tmp_path / "workspace",
    )
    runtime = Runtime(
        provider=MockProvider(),
        workspace_root=paths.workspace_root,
        paths=paths,
    )
    for service in instantiate_services(config):
        runtime.register_service(service)

    gateway = Gateway(
        runtime=runtime,
        sources=[],
        gateway_config=config,
    )
    async with runtime:
        await gateway._startup()

    fresh_agent = runtime.sessions.load_agent(agent_id)
    assert isinstance(
        fresh_agent.accounts.accounts[0],
        UntypedAccountConfig,
    )

    event = RawIncomingEvent(
        source="github",
        session_key=SessionKey("github/1/issue/1"),
        kind=EventKind.STRUCTURAL,
    )
    routed_agent = gateway._resolve_agent(event)

    assert routed_agent.id == agent_id
    assert isinstance(
        routed_agent.accounts.accounts[0],
        GitHubAccountConfig,
    )
