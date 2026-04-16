"""Gateway configuration: loading services from ``.thorn/gateway.json``.

The gateway configuration file declares forges and projects.  Forges
are external platforms that host version-controlled repositories;
projects are logical software projects with one or more forks hosted
on those forges.

The **new** config format uses top-level ``"forges"`` and
``"projects"`` arrays::

    {
      "forges": [
        {
          "name": "github-com",
          "type": "github",
          "base_url": "https://api.github.com",
          "poll_interval": 30
        }
      ],
      "projects": [
        {
          "name": "tiny-talk",
          "forge": "github-com",
          "native_id": "tangent-vector/tiny-talk",
          "clone_url": "https://github.com/tangent-vector/tiny-talk.git",
          "default_branch": "main"
        }
      ]
    }

The **legacy** format uses a flat ``"services"`` array and is still
accepted for backward compatibility during the transition period.

Event sources are **no longer configured explicitly**.  They are
inferred at startup from agent accounts on registered forges (see
:func:`infer_event_sources`).

String values that begin with ``$`` are treated as environment
variable references and expanded at load time, keeping secrets out of
the config file itself.  Per the design convention, only genuinely
secret values (tokens, private keys) should use ``$ENV_VAR``; all
other configuration should be literal.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from thorn.core._service import Service
from thorn.gateway._event import EventSource

log = logging.getLogger(__name__)

GATEWAY_CONFIG_FILENAME = "gateway.json"


# ---------------------------------------------------------------------------
# $ENV_VAR expansion
# ---------------------------------------------------------------------------


def expand_env_vars(data: Any) -> Any:
    """Recursively expand ``$ENV_VAR`` references in string values.

    A string value whose entire content matches ``$NAME`` is replaced
    with ``os.environ[NAME]``.  Non-string values and strings that do
    not start with ``$`` are returned unchanged.

    Dicts and lists are traversed recursively; all other types pass
    through unmodified.

    Raises :class:`ValueError` when a referenced variable is not set.
    """
    if isinstance(data, str):
        if data.startswith("$"):
            var_name = data[1:]
            value = os.environ.get(var_name)
            if value is None:
                raise ValueError(
                    f"Environment variable {var_name!r} "
                    f"(referenced as {data!r}) is not set"
                )
            return value
        return data
    if isinstance(data, dict):
        return {k: expand_env_vars(v) for k, v in data.items()}
    if isinstance(data, list):
        return [expand_env_vars(item) for item in data]
    return data


# ---------------------------------------------------------------------------
# Configuration models -- new format
# ---------------------------------------------------------------------------


class ForgeSpec(BaseModel):
    """One entry in the ``"forges"`` array of ``gateway.json``."""

    name: str
    type: str = Field(description="Forge backend: 'github' or 'gitlab'")
    base_url: str = Field(
        default="",
        description="API base URL (literal, not an env var reference)",
    )
    poll_interval: int = Field(
        default=30,
        ge=5,
        description="Seconds between event polling cycles",
    )


class ForkSpec(BaseModel):
    """A single fork of a project, hosted on a forge.

    Each fork identifies a forge-specific repository.  The ``name``
    becomes the git remote name when the project is cloned locally.
    """

    forge: str = Field(description="Name of the forge hosting this fork")
    native_id: str = Field(
        description="Forge-native identifier (owner/repo for GitHub, numeric for GitLab)",
    )
    name: str = Field(
        default="",
        description="Local name for this fork / git remote name (e.g. 'upstream', 'origin')",
    )
    clone_url: str = Field(
        default="",
        description="HTTPS clone URL override (derived from forge when empty)",
    )


class ProjectSpec(BaseModel):
    """One entry in the ``"projects"`` array of ``gateway.json``.

    A project has one or more forks.  When ``forks`` is non-empty,
    the first fork is the canonical upstream.  When ``forks`` is
    empty, the legacy single-fork fields (``forge``, ``native_id``,
    ``clone_url``) are used to construct a single implicit fork.
    """

    name: str
    default_branch: str = Field(default="main")

    forks: list[ForkSpec] = Field(
        default_factory=list,
        description="Explicit list of forks (new format)",
    )

    forge: str = Field(
        default="",
        description="(Legacy) Name of the forge this project is hosted on",
    )
    native_id: str = Field(
        default="",
        description="(Legacy) Forge-native identifier",
    )
    clone_url: str = Field(
        default="",
        description="(Legacy) HTTPS clone URL for the repository",
    )

    def resolved_forks(self) -> list[ForkSpec]:
        """Return the effective fork list, synthesizing from legacy fields if needed."""
        if self.forks:
            return list(self.forks)
        if self.forge and self.native_id:
            return [ForkSpec(
                forge=self.forge,
                native_id=self.native_id,
                name="upstream",
                clone_url=self.clone_url,
            )]
        return []

    @property
    def primary_forge(self) -> str:
        """Name of the forge hosting the primary (first) fork."""
        forks = self.resolved_forks()
        if forks:
            return forks[0].forge
        return self.forge


# ---------------------------------------------------------------------------
# Configuration models -- legacy format
# ---------------------------------------------------------------------------


class ServiceSpec(BaseModel):
    """One entry in the ``"services"`` array of ``gateway.json`` (legacy)."""

    name: str
    type: str
    config: dict[str, Any] = {}


# ---------------------------------------------------------------------------
# Unified config model
# ---------------------------------------------------------------------------


class GatewayConfig(BaseModel):
    """Top-level model for ``.thorn/gateway.json``.

    Supports both the **new** format (``forges`` + ``projects``) and
    the **legacy** format (flat ``services`` array).  When ``forges``
    is non-empty, the new format takes precedence.
    """

    forges: list[ForgeSpec] = []
    projects: list[ProjectSpec] = []

    services: list[ServiceSpec] = Field(
        default_factory=list,
        description="Legacy flat service list (deprecated)",
    )

    @property
    def is_new_format(self) -> bool:
        return bool(self.forges)


# ---------------------------------------------------------------------------
# Service type registry (used by legacy instantiation path)
# ---------------------------------------------------------------------------

_SERVICE_TYPE_REGISTRY: dict[str, Any] = {}
_BUILTINS_REGISTERED = False


def _register_service_type(
    type_key: str,
    factory: Any,
) -> None:
    """Register a factory for a service type key.

    *factory* is called as ``factory(config, service_name=name)``
    where *config* is a validated instance of the service class's
    ``Config`` model.
    """
    _SERVICE_TYPE_REGISTRY[type_key] = factory


def _ensure_builtin_types() -> None:
    """Lazily register built-in service types on first use."""
    global _BUILTINS_REGISTERED  # noqa: PLW0603
    if _BUILTINS_REGISTERED:
        return
    _BUILTINS_REGISTERED = True

    from thorn.gateway.sources._github import GitHubNotificationsSource
    from thorn.gateway.sources._gitlab import GitLabTODOsSource
    from thorn.tools._github_connection import GitHubConnectionConfig
    from thorn.tools.forge import (
        GitHubForgeService,
        GitLabForgeService,
        GitLabForgeServiceConfig,
        ProjectService,
        ProjectServiceConfig,
    )

    def _make_gitlab_forge(
        spec_config: dict[str, Any],
        *,
        service_name: str,
    ) -> GitLabForgeService:
        cfg = GitLabForgeServiceConfig(**spec_config)
        return GitLabForgeService(cfg, service_name=service_name)

    def _make_github_forge(
        spec_config: dict[str, Any],
        *,
        service_name: str,
    ) -> GitHubForgeService:
        cfg = GitHubConnectionConfig(**spec_config)
        return GitHubForgeService(cfg, service_name=service_name)

    _register_service_type(
        "gitlab",
        lambda config, service_name: _make_gitlab_forge(
            config, service_name=service_name,
        ),
    )
    _register_service_type(
        "github",
        lambda config, service_name: _make_github_forge(
            config, service_name=service_name,
        ),
    )

    def _make_project(
        spec_config: dict[str, Any], *, service_name: str,
    ) -> ProjectService:
        cfg = ProjectServiceConfig(**spec_config)
        return ProjectService(cfg, service_name=service_name)

    _register_service_type("project", _make_project)

    def _make_gitlab_events(
        spec_config: dict[str, Any], *, service_name: str,
    ) -> GitLabTODOsSource:
        cfg = GitLabTODOsSource.Config(**spec_config)
        return GitLabTODOsSource(cfg, service_name=service_name)

    _register_service_type("gitlab-events", _make_gitlab_events)

    def _make_github_events(
        spec_config: dict[str, Any], *, service_name: str,
    ) -> GitHubNotificationsSource:
        cfg = GitHubNotificationsSource.Config(**spec_config)
        return GitHubNotificationsSource(cfg, service_name=service_name)

    _register_service_type("github-events", _make_github_events)


# ---------------------------------------------------------------------------
# New-format instantiation
# ---------------------------------------------------------------------------


def instantiate_new_format(config: GatewayConfig) -> list[Service]:
    """Create :class:`Service` instances from the new-format gateway config.

    Instantiates :class:`ForgeHostService` from ``config.forges`` and
    :class:`ProjectService` from ``config.projects``.  No event sources
    are created here — those are inferred separately via
    :func:`infer_event_sources`.
    """
    from thorn.tools._github_connection import GitHubConnectionConfig, GitHubPatAuth
    from thorn.tools.forge import (
        GitHubForgeService,
        GitLabForgeService,
        GitLabForgeServiceConfig,
        ProjectService,
        ProjectServiceConfig,
    )

    services: list[Service] = []

    for forge_spec in config.forges:
        if forge_spec.type == "gitlab":
            cfg = GitLabForgeServiceConfig(url=forge_spec.base_url, token="")
            svc = GitLabForgeService(cfg, service_name=forge_spec.name)
        elif forge_spec.type == "github":
            base_url = forge_spec.base_url or "https://api.github.com"
            cfg_gh = GitHubConnectionConfig(
                base_url=base_url,
                auth=GitHubPatAuth(token=""),
            )
            svc = GitHubForgeService(cfg_gh, service_name=forge_spec.name)
        else:
            raise ValueError(
                f"Unknown forge type {forge_spec.type!r} for forge {forge_spec.name!r}. "
                f"Supported types: 'github', 'gitlab'."
            )
        log.info(
            "Instantiated %s %r (type=%s)",
            type(svc).__name__, forge_spec.name, forge_spec.type,
        )
        services.append(svc)

    for proj_spec in config.projects:
        resolved = proj_spec.resolved_forks()
        if resolved:
            from thorn.tools.forge import ForkConfig as _ForkConfig

            fork_configs = [
                _ForkConfig(
                    forge=f.forge,
                    native_id=f.native_id,
                    name=f.name or ("upstream" if i == 0 else f"fork-{i}"),
                    clone_url=f.clone_url,
                )
                for i, f in enumerate(resolved)
            ]
            proj_cfg = ProjectServiceConfig(
                forks=fork_configs,
                default_branch=proj_spec.default_branch,
            )
        else:
            proj_cfg = ProjectServiceConfig(
                forge=proj_spec.forge,
                native_id=proj_spec.native_id,
                path=proj_spec.name,
                clone_url=proj_spec.clone_url,
                default_branch=proj_spec.default_branch,
            )
        proj_svc = ProjectService(proj_cfg, service_name=proj_spec.name)
        log.info("Instantiated ProjectService %r (forge=%s)", proj_spec.name, proj_spec.primary_forge)
        services.append(proj_svc)

    return services


# ---------------------------------------------------------------------------
# Event source inference
# ---------------------------------------------------------------------------


class _ForgeProjectInfo:
    """Per-forge project information used during event source inference."""

    __slots__ = ("repositories", "native_id_to_project_name")

    def __init__(self) -> None:
        self.repositories: list[str] = []
        self.native_id_to_project_name: dict[str, str] = {}


def infer_event_sources(
    config: GatewayConfig,
    agents: list[Any],
) -> list[EventSource]:
    """Create event sources for each (agent, forge) pair.

    For each agent that has a :class:`ForgeAccountConfig` on a forge
    declared in *config*, an appropriate event source is created:

    - **GitHub**: A per-repository poller for each project fork on
      this forge, authenticated with the agent's credentials.
    - **GitLab**: A TODOs poller authenticated with the agent's
      credentials (TODOs are user-scoped, no per-repo enumeration).

    The ``poll_interval`` comes from the forge spec in the config.
    Project names are threaded through to the event sources so that
    session keys use project-name-based routing.
    """
    from thorn.core._account import AgentAccountsConfig

    forge_specs_by_name: dict[str, ForgeSpec] = {
        f.name: f for f in config.forges
    }

    forge_project_info: dict[str, _ForgeProjectInfo] = {}
    for proj in config.projects:
        for fork in proj.resolved_forks():
            info = forge_project_info.setdefault(fork.forge, _ForgeProjectInfo())
            info.repositories.append(fork.native_id)
            info.native_id_to_project_name[fork.native_id] = proj.name

    sources: list[EventSource] = []

    for agent in agents:
        accounts: AgentAccountsConfig | None = getattr(agent, "accounts", None)
        if accounts is None:
            continue

        for acct in accounts.forge_accounts:
            forge_spec = forge_specs_by_name.get(acct.forge)
            if forge_spec is None:
                log.warning(
                    "Agent %r has account on forge %r which is not in gateway config; skipping.",
                    getattr(agent, "name", "?"), acct.forge,
                )
                continue

            info = forge_project_info.get(forge_spec.name, _ForgeProjectInfo())
            source = _create_event_source_for_account(
                forge_spec=forge_spec,
                account=acct,
                agent=agent,
                repositories=info.repositories,
                native_id_to_project_name=info.native_id_to_project_name,
            )
            if source is not None:
                sources.append(source)

    return sources


def _create_event_source_for_account(
    *,
    forge_spec: ForgeSpec,
    account: Any,
    agent: Any,
    repositories: list[str],
    native_id_to_project_name: dict[str, str],
) -> EventSource | None:
    """Create a single event source for an agent's account on a forge."""
    from thorn.core._account import ForgeAccountConfig

    assert isinstance(account, ForgeAccountConfig)
    agent_name = getattr(agent, "name", None) or getattr(agent, "id", "unknown")

    if forge_spec.type == "github":
        return _create_github_source(
            forge_spec=forge_spec,
            account=account,
            agent_name=str(agent_name),
            repositories=repositories,
            native_id_to_project_name=native_id_to_project_name,
        )

    if forge_spec.type == "gitlab":
        return _create_gitlab_source(
            forge_spec=forge_spec,
            account=account,
            agent_name=str(agent_name),
            native_id_to_project_name=native_id_to_project_name,
        )

    log.warning(
        "No event source strategy for forge type %r (forge=%r, agent=%r)",
        forge_spec.type, forge_spec.name, agent_name,
    )
    return None


def _create_github_source(
    *,
    forge_spec: ForgeSpec,
    account: Any,
    agent_name: str,
    repositories: list[str],
    native_id_to_project_name: dict[str, str],
) -> EventSource | None:
    """Create a GitHub repository events source for one (agent, forge) pair.

    Polls each repository in *repositories* using the agent's
    credentials.  If there are no repositories on this forge, no
    source is created.
    """
    if not repositories:
        log.info(
            "No projects on forge %r for agent %r; skipping GitHub source.",
            forge_spec.name, agent_name,
        )
        return None

    from thorn.tools._github_connection import GitHubAppAuth, GitHubPatAuth

    creds = account.credentials
    if isinstance(creds, GitHubPatAuth):
        auth_block = {"kind": "pat", "token": creds.token}
    elif isinstance(creds, GitHubAppAuth):
        auth_block = {
            "kind": "app",
            "app_id": creds.app_id,
            "installation_id": creds.installation_id,
            "private_key_pem": creds.private_key_pem,
        }
    else:
        log.warning(
            "Unsupported credential type %s for GitHub forge %r",
            type(creds).__name__, forge_spec.name,
        )
        return None

    from thorn.gateway.sources._github import (
        GitHubNotificationsSource,
        GitHubNotificationsSourceConfig,
    )

    base_url = forge_spec.base_url or "https://api.github.com"
    repo = repositories[0]
    project_name = native_id_to_project_name.get(repo, "")
    source_name = f"{agent_name}-{forge_spec.name}-events"

    cfg = GitHubNotificationsSourceConfig(
        base_url=base_url,
        auth=auth_block,  # type: ignore[arg-type]
        repository=repo,
        poll_interval=forge_spec.poll_interval,
        project_name=project_name,
    )
    source = GitHubNotificationsSource(cfg, service_name=source_name)
    log.info(
        "Inferred GitHub event source %r (repo=%s, project=%s, agent=%s)",
        source_name, repo, project_name or "(unknown)", agent_name,
    )
    return source


def _create_gitlab_source(
    *,
    forge_spec: ForgeSpec,
    account: Any,
    agent_name: str,
    native_id_to_project_name: dict[str, str],
) -> EventSource | None:
    """Create a GitLab TODOs source for one (agent, forge) pair.

    GitLab TODOs are user-scoped, so no repository list is needed.
    The *native_id_to_project_name* mapping (keyed by stringified
    GitLab project ID) is passed through to the source so that
    session keys use project-name-based routing.
    """
    from thorn.core._account import GitLabCredentials

    creds = account.credentials
    if not isinstance(creds, GitLabCredentials):
        log.warning(
            "Unsupported credential type %s for GitLab forge %r",
            type(creds).__name__, forge_spec.name,
        )
        return None

    from thorn.gateway.sources._gitlab import GitLabSourceConfig, GitLabTODOsSource

    source_name = f"{agent_name}-{forge_spec.name}-events"
    cfg = GitLabSourceConfig(
        url=forge_spec.base_url,
        token=creds.token,
        poll_interval=forge_spec.poll_interval,
        project_id_to_name=native_id_to_project_name,
    )
    source = GitLabTODOsSource(cfg, service_name=source_name)
    log.info(
        "Inferred GitLab event source %r (agent=%s)",
        source_name, agent_name,
    )
    return source


# ---------------------------------------------------------------------------
# Legacy instantiation
# ---------------------------------------------------------------------------


def load_gateway_config(thorn_dir: Path) -> GatewayConfig:
    """Load and parse ``gateway.json`` from the given ``.thorn/`` directory.

    Raises :class:`FileNotFoundError` if the config file does not exist.
    """
    config_path = thorn_dir / GATEWAY_CONFIG_FILENAME
    if not config_path.is_file():
        raise FileNotFoundError(
            f"Gateway configuration file not found: {config_path}\n"
            "Run 'thorn serve bootstrap' to create one, or write it manually."
        )
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    return GatewayConfig.model_validate(raw)


def instantiate_services(config: GatewayConfig) -> list[Service]:
    """Create :class:`Service` instances from a gateway configuration.

    Handles both legacy and new format.  For the new format, delegates
    to :func:`instantiate_new_format`.  For the legacy format, uses
    the service type registry.
    """
    if config.is_new_format:
        return instantiate_new_format(config)

    _ensure_builtin_types()

    services: list[Service] = []
    for spec in config.services:
        expanded = expand_env_vars(spec.config)

        if spec.type in _SERVICE_TYPE_REGISTRY:
            factory = _SERVICE_TYPE_REGISTRY[spec.type]
            service = factory(expanded, service_name=spec.name)
        else:
            from thorn.gateway.sources import get_registered_source

            source_class = get_registered_source(spec.type)
            config_instance = source_class.Config(**expanded)
            service = source_class(config_instance, service_name=spec.name)

        log.info(
            "Instantiated %s %r (type=%s)",
            type(service).__name__, spec.name, spec.type,
        )
        services.append(service)
    return services


def instantiate_sources(config: GatewayConfig) -> list[EventSource]:
    """Create :class:`EventSource` instances from a gateway configuration.

    This is the backward-compatible entry point.  It calls
    :func:`instantiate_services` and filters to only return event
    sources.
    """
    all_services = instantiate_services(config)
    return [s for s in all_services if isinstance(s, EventSource)]


__all__ = [
    "ForkSpec",
    "ForgeSpec",
    "GATEWAY_CONFIG_FILENAME",
    "GatewayConfig",
    "ProjectSpec",
    "ServiceSpec",
    "expand_env_vars",
    "infer_event_sources",
    "instantiate_new_format",
    "instantiate_services",
    "instantiate_sources",
    "load_gateway_config",
]
