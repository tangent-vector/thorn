"""Executable checks for the documented gateway configuration examples."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path

import pytest

from thorn.core._account import validate_agent_accounts
from thorn.core._agent import Agent
from thorn.core._service import Service
from thorn.gateway._config import (
    GatewayConfig,
    instantiate_services,
    load_gateway_config,
)
from thorn.runtime._serializer import JsonSessionSerializer
from thorn.tools.forge import (
    GitHubAccountConfig,
    GitLabAccountConfig,
    ProjectService,
)

EXAMPLES_ROOT = Path(__file__).resolve().parents[1] / "examples" / "gateway"


def _example_names() -> list[str]:
    return sorted(
        path.name
        for path in EXAMPLES_ROOT.iterdir()
        if (path / "gateway.json").is_file()
    )


def _load_example_config(example_name: str) -> GatewayConfig:
    return load_gateway_config(EXAMPLES_ROOT / example_name)


def _service_lookup(services: Iterable[Service]) -> Callable[[str], Service]:
    services_by_name = {service.name: service for service in services}

    def lookup(service_name: str) -> Service:
        try:
            return services_by_name[service_name]
        except KeyError as exc:
            raise KeyError(service_name) from exc

    return lookup


def _load_and_validate_example_agents(
    example_name: str,
    services: Iterable[Service],
) -> list[Agent]:
    agents_root = EXAMPLES_ROOT / example_name / "agents"
    if not agents_root.is_dir():
        return []

    serializer = JsonSessionSerializer()
    agents: list[Agent] = []
    lookup = _service_lookup(services)
    for identity_path in sorted(agents_root.glob("*/agent.json")):
        agent = serializer.load_agent(identity_path)
        validate_agent_accounts(agent, lookup)
        agents.append(agent)
    return agents


@pytest.mark.parametrize("example_name", _example_names())
def test_gateway_example_configs_load_and_resolve(
    example_name: str,
) -> None:
    config = _load_example_config(example_name)

    services = instantiate_services(config)
    _load_and_validate_example_agents(example_name, services)


@pytest.mark.parametrize(
    ("example_name", "account_type", "service_name", "kind", "env_var_name"),
    [
        ("github-pat", GitHubAccountConfig, "github", "pat", "GITHUB_TOKEN"),
        (
            "gitlab-pat",
            GitLabAccountConfig,
            "gitlab",
            "gitlab-pat",
            "GITLAB_TOKEN",
        ),
    ],
)
def test_public_pat_examples_validate_agent_account_shapes(
    example_name: str,
    account_type: type[GitHubAccountConfig | GitLabAccountConfig],
    service_name: str,
    kind: str,
    env_var_name: str,
) -> None:
    services = instantiate_services(_load_example_config(example_name))
    agents = _load_and_validate_example_agents(example_name, services)

    assert len(agents) == 1
    assert agents[0].accounts is not None
    accounts = agents[0].accounts.accounts
    assert len(accounts) == 1

    account = accounts[0]
    assert isinstance(account, account_type)
    assert account.service == service_name
    assert len(account.credentials) == 1
    assert account.credentials[0].kind == kind
    assert account.credentials[0].env_var_name == env_var_name


@pytest.mark.parametrize("example_name", ["github-pat", "gitlab-pat"])
def test_public_pat_examples_use_secure_gateway_defaults(
    example_name: str,
) -> None:
    config = _load_example_config(example_name)

    assert config.sandbox is not None
    assert config.sandbox.backend == "container"
    assert config.broker is not None
    assert config.broker.mode == "bundled"


def test_subprocess_opt_out_example_disables_broker_default() -> None:
    config = _load_example_config("subprocess-opt-out")

    assert config.sandbox is not None
    assert config.sandbox.backend == "subprocess"
    assert config.broker is None


def test_bundled_broker_mirror_example_pins_images() -> None:
    config = _load_example_config("bundled-broker-mirrors")

    assert config.broker is not None
    assert config.broker.mode == "bundled"
    assert config.broker.bundled_images.onecli == (
        "registry.example.com/team/mirror/onecli:2026-05-01"
    )
    assert config.broker.bundled_images.postgres == (
        "registry.example.com/team/mirror/postgres:18-alpine"
    )


def test_self_hosted_gitlab_example_uses_explicit_native_id() -> None:
    config = _load_example_config("self-hosted-gitlab-native-id")
    services = instantiate_services(config)

    project_service = next(
        service
        for service in services
        if isinstance(service, ProjectService)
    )
    assert project_service.native_id == "264873"

    agents = _load_and_validate_example_agents(
        "self-hosted-gitlab-native-id",
        services,
    )
    assert len(agents) == 1
    assert agents[0].accounts is not None
    account = agents[0].accounts.accounts[0]
    assert isinstance(account, GitLabAccountConfig)
    assert account.service == "gitlab-primary"
