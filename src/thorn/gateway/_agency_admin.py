"""Operator helpers for agency configuration commands."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from thorn.core._account import validate_agent_accounts
from thorn.gateway._config import (
    AGENCY_CONFIG_FILENAMES,
    AgencyConfigFile,
    GatewayConfig,
    ResolvedProject,
    _discover_agency_config_file,
    _parse_agency_config_file,
    _resolve_forges_and_projects,
    instantiate_services,
)
from thorn.runtime import AgencyPaths, AgentID
from thorn.runtime._store import SessionStore

PREFERRED_AGENCY_CONFIG_FILENAME = "agency.yaml"


@dataclass(frozen=True)
class AgencyServiceSummary:
    """Resolved service entry shown by ``thorn agency show``."""

    name: str
    service_type: str

    def to_json(self) -> dict[str, str]:
        return {"name": self.name, "type": self.service_type}


@dataclass(frozen=True)
class AgencyForkSummary:
    """Resolved project fork entry shown by ``thorn agency show``."""

    name: str
    forge_name: str
    native_id: str
    clone_url: str

    def to_json(self) -> dict[str, str]:
        return {
            "name": self.name,
            "forge": self.forge_name,
            "native_id": self.native_id,
            "clone_url": self.clone_url,
        }


@dataclass(frozen=True)
class AgencyProjectSummary:
    """Resolved project entry shown by ``thorn agency show``."""

    name: str
    forks: tuple[AgencyForkSummary, ...]

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "forks": [fork.to_json() for fork in self.forks],
        }


@dataclass(frozen=True)
class AgencyInitResult:
    """Paths created or selected by ``thorn agency init``."""

    agency_home: Path
    workspace_root: Path
    config_file: Path

    def to_json(self) -> dict[str, str]:
        return {
            "agency_home": str(self.agency_home),
            "workspace": str(self.workspace_root),
            "config_file": str(self.config_file),
        }


@dataclass(frozen=True)
class AgencyConfigSummary:
    """Resolved, script-friendly view of one agency configuration."""

    agency_home: Path
    config_file: AgencyConfigFile
    workspace_root: Path | None
    agent_ids: tuple[AgentID, ...]
    services: tuple[AgencyServiceSummary, ...]
    projects: tuple[AgencyProjectSummary, ...]
    peer_ids: tuple[str, ...]
    sandbox_backend: str | None
    broker_mode: str | None
    broker_enabled: bool | None

    def to_json(self) -> dict[str, Any]:
        return {
            "agency_home": str(self.agency_home),
            "config_file": str(self.config_file.path),
            "workspace": (
                str(self.workspace_root) if self.workspace_root is not None else None
            ),
            "agents": [{"id": str(agent_id)} for agent_id in self.agent_ids],
            "services": [service.to_json() for service in self.services],
            "projects": [project.to_json() for project in self.projects],
            "peers": [{"id": peer_id} for peer_id in self.peer_ids],
            "sandbox": {"backend": self.sandbox_backend},
            "broker": {
                "mode": self.broker_mode,
                "enabled": self.broker_enabled,
            },
        }


def _existing_config_files(agency_home: Path) -> list[Path]:
    return [
        agency_home / filename
        for filename in AGENCY_CONFIG_FILENAMES
        if (agency_home / filename).is_file()
    ]


def initialize_agency_home(
    *,
    agency_home: Path,
    workspace_root: Path,
) -> AgencyInitResult:
    """Create a minimal agency config and directory layout.

    The generated config deliberately contains only the workspace path.
    Projects, forges, peers, and agent identities are left for follow-on
    commands or direct edits.
    """
    agency_home = agency_home.expanduser().resolve()
    workspace_root = workspace_root.expanduser().resolve()

    if agency_home.exists() and not agency_home.is_dir():
        raise NotADirectoryError(f"Agency path is not a directory: {agency_home}")

    existing_config_files = _existing_config_files(agency_home)
    if existing_config_files:
        found_names = ", ".join(path.name for path in existing_config_files)
        raise FileExistsError(
            f"Agency home {agency_home} already contains agency "
            f"configuration: {found_names}"
        )

    raw_config = {"workspace": str(workspace_root)}
    GatewayConfig.model_validate(raw_config)

    agency_home.mkdir(parents=True, exist_ok=True)
    workspace_root.mkdir(parents=True, exist_ok=True)
    (agency_home / "agents").mkdir(parents=True, exist_ok=True)

    config_path = agency_home / PREFERRED_AGENCY_CONFIG_FILENAME
    config_path.write_text(
        yaml.safe_dump(raw_config, sort_keys=False),
        encoding="utf-8",
    )
    return AgencyInitResult(
        agency_home=agency_home,
        workspace_root=workspace_root,
        config_file=config_path,
    )


def summarize_agency_config(agency_home: Path) -> AgencyConfigSummary:
    """Load and validate an agency config, returning a compact summary."""
    agency_home = agency_home.expanduser().resolve()
    gateway_config, config_file = _load_gateway_config_with_file(agency_home)
    forge_specs, resolved_projects = _resolve_forges_and_projects(gateway_config)
    services = instantiate_services(gateway_config)
    service_lookup = {service.name: service for service in services}

    workspace_root = gateway_config.resolve_workspace(agency_home)
    store_workspace_root = workspace_root or agency_home
    store = SessionStore(
        AgencyPaths.for_gateway(
            agency_dir=agency_home,
            workspace_dir=store_workspace_root,
        )
    )
    agent_ids = tuple(store.list_agent_ids())
    for agent_id in agent_ids:
        agent = store.load_agent(agent_id)
        validate_agent_accounts(agent, service_lookup.__getitem__)

    return AgencyConfigSummary(
        agency_home=agency_home,
        config_file=config_file,
        workspace_root=workspace_root,
        agent_ids=agent_ids,
        services=tuple(
            AgencyServiceSummary(name=forge.name, service_type=forge.type)
            for forge in forge_specs
        ),
        projects=_resolved_projects_to_summary(resolved_projects),
        peer_ids=tuple(peer.id for peer in gateway_config.peers),
        sandbox_backend=(
            gateway_config.sandbox.backend
            if gateway_config.sandbox is not None
            else None
        ),
        broker_mode=(
            gateway_config.broker.mode if gateway_config.broker is not None else None
        ),
        broker_enabled=(
            gateway_config.broker.enabled if gateway_config.broker is not None else None
        ),
    )


def _load_gateway_config_with_file(
    agency_home: Path,
) -> tuple[GatewayConfig, AgencyConfigFile]:
    config_file = _discover_agency_config_file(agency_home)
    raw = _parse_agency_config_file(config_file)
    try:
        return GatewayConfig.model_validate(raw), config_file
    except ValidationError as exc:
        raise ValueError(str(exc)) from exc


def _resolved_projects_to_summary(
    resolved_projects: Iterable[ResolvedProject],
) -> tuple[AgencyProjectSummary, ...]:
    projects: list[AgencyProjectSummary] = []
    for project in resolved_projects:
        projects.append(
            AgencyProjectSummary(
                name=project.name,
                forks=tuple(
                    AgencyForkSummary(
                        name=fork.name,
                        forge_name=fork.forge_name,
                        native_id=fork.native_id,
                        clone_url=fork.clone_url,
                    )
                    for fork in project.forks
                ),
            )
        )
    return tuple(projects)
