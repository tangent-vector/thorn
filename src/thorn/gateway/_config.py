"""Gateway configuration: loading services from ``.thorn/gateway.json``.

The gateway configuration file declares forges and projects.  Forges
are external platforms that host version-controlled repositories;
projects are logical software projects with one or more forks hosted
on those forges.

The on-disk format uses top-level typed arrays — currently
``"forges"`` and ``"projects"``::

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
          "forks": [
            {
              "forge": "github-com",
              "native_id": "tangent-vector/tiny-talk",
              "name": "upstream",
              "clone_url": "https://github.com/tangent-vector/tiny-talk.git"
            }
          ]
        }
      ]
    }

Future plug-in service categories will be added as additional typed
arrays alongside ``forges:`` and ``projects:`` (for example, a
``messaging_services:`` array).  Heterogeneous arrays — those whose
entries can be one of several backends keyed by ``"type"`` — are
instantiated through :class:`ServiceTypeRegistry`.  Today the only
heterogeneous array is ``forges:`` (``github`` vs. ``gitlab``);
``projects:`` is uniform and instantiates :class:`ProjectService`
directly.

Event sources are **not** configured explicitly.  They are inferred
at startup from agent accounts on registered forges (see
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
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from thorn.core._service import Service
from thorn.gateway._event import EventSource

log = logging.getLogger(__name__)

GATEWAY_CONFIG_FILENAME = "gateway.json"

_GITHUB_DEFAULT_API_BASE = "https://api.github.com"


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
# Configuration models
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


class GatewayConfig(BaseModel):
    """Top-level model for ``.thorn/gateway.json``.

    Future plug-in service categories will appear as additional typed
    array fields (for example, ``messaging_services: list[...]``).
    """

    forges: list[ForgeSpec] = Field(default_factory=list)
    projects: list[ProjectSpec] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Service type registry
# ---------------------------------------------------------------------------


SpecToConfigDict = Callable[[BaseModel], dict[str, Any]]
"""Callable that maps a typed-array spec model into kwargs for the
service's ``Config`` model.  See :class:`ServiceTypeRegistry`.
"""


class ServiceTypeRegistry:
    """Maps ``(category, type_key)`` → service constructor.

    A "category" groups :class:`Service` subclasses that share a typed
    array in ``gateway.json`` — for example, all forge backends share
    the ``forges:`` array under category ``"forge"``.

    Registration pairs a :class:`Service` subclass with the Pydantic
    ``Config`` model that gets instantiated for it, plus a
    ``spec_to_config`` callable that translates a typed-array entry
    (e.g. a :class:`ForgeSpec`) into kwargs for that ``Config`` model.
    The translation step is per-type because different forge backends
    expose different config shapes (``GitHubConnectionConfig`` carries
    a discriminated ``auth`` block, ``GitLabForgeServiceConfig`` carries
    ``url``/``token``); pushing the translation into the registration
    keeps :func:`instantiate_services` free of any per-type dispatch.
    """

    def __init__(self) -> None:
        self._entries: dict[
            tuple[str, str],
            tuple[type[Service], type[BaseModel], SpecToConfigDict],
        ] = {}

    def register(
        self,
        category: str,
        type_key: str,
        service_cls: type[Service],
        config_cls: type[BaseModel],
        *,
        spec_to_config: SpecToConfigDict,
    ) -> None:
        """Register a service backend under ``(category, type_key)``.

        Subsequent registrations with the same key replace the prior
        entry (which keeps tests and customisation simple).
        """
        self._entries[(category, type_key)] = (
            service_cls, config_cls, spec_to_config,
        )

    def known_types(self, category: str) -> list[str]:
        """Return the registered type keys for *category*, sorted."""
        return sorted(k for c, k in self._entries if c == category)

    def instantiate(
        self,
        category: str,
        type_key: str,
        *,
        spec: BaseModel,
        name: str,
    ) -> Service:
        """Build a service instance for the given typed-array entry.

        Raises :class:`ValueError` when *type_key* is not registered
        under *category*.
        """
        entry = self._entries.get((category, type_key))
        if entry is None:
            raise ValueError(
                f"Unknown {category} type {type_key!r} for entry {name!r}. "
                f"Known {category} types: {self.known_types(category)}"
            )
        service_cls, config_cls, spec_to_config = entry
        config = config_cls(**spec_to_config(spec))
        return service_cls(config, service_name=name)


_REGISTRY: ServiceTypeRegistry | None = None


def get_service_type_registry() -> ServiceTypeRegistry:
    """Return the process-wide :class:`ServiceTypeRegistry` (lazy init).

    Built-in registrations are added on first access; this avoids
    import cycles between :mod:`thorn.gateway._config`,
    :mod:`thorn.tools.forge`, and the forge service modules.
    """
    global _REGISTRY  # noqa: PLW0603
    if _REGISTRY is not None:
        return _REGISTRY
    _REGISTRY = ServiceTypeRegistry()
    _register_builtin_forges(_REGISTRY)
    return _REGISTRY


def _gitlab_forge_spec_to_config(spec: BaseModel) -> dict[str, Any]:
    """Translate a :class:`ForgeSpec` for a GitLab forge into config kwargs."""
    assert isinstance(spec, ForgeSpec)
    if not spec.base_url:
        raise ValueError(
            f"GitLab forge entry {spec.name!r} requires a non-empty "
            "'base_url' (e.g. 'https://gitlab.example.com/api/v4'). "
            "Edit gateway.json to set forges[].base_url for this entry."
        )
    return {"url": spec.base_url, "token": ""}


def _github_forge_spec_to_config(spec: BaseModel) -> dict[str, Any]:
    """Translate a :class:`ForgeSpec` for a GitHub forge into config kwargs.

    The token is intentionally empty; per-agent credentials come from
    :class:`~thorn.core._account.ForgeAccountConfig` at the call site
    (see :meth:`ForgeHostService.authenticated_client`).
    """
    assert isinstance(spec, ForgeSpec)
    return {
        "base_url": spec.base_url or _GITHUB_DEFAULT_API_BASE,
        "auth": {"kind": "pat", "token": ""},
    }


def _register_builtin_forges(registry: ServiceTypeRegistry) -> None:
    """Register the built-in ``"forge"`` backends."""
    from thorn.tools._github_connection import GitHubConnectionConfig
    from thorn.tools.forge import (
        GitHubForgeService,
        GitLabForgeService,
        GitLabForgeServiceConfig,
    )

    registry.register(
        "forge", "gitlab",
        GitLabForgeService, GitLabForgeServiceConfig,
        spec_to_config=_gitlab_forge_spec_to_config,
    )
    registry.register(
        "forge", "github",
        GitHubForgeService, GitHubConnectionConfig,
        spec_to_config=_github_forge_spec_to_config,
    )


# ---------------------------------------------------------------------------
# Loading & instantiation
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

    For each entry in ``config.forges``, the appropriate
    :class:`~thorn.tools.forge.ForgeHostService` is built via the
    :class:`ServiceTypeRegistry`.  Each entry in ``config.projects`` is
    materialised directly as a :class:`~thorn.tools.forge.ProjectService`.

    Event sources are **not** created here — see
    :func:`infer_event_sources`.
    """
    from thorn.tools.forge import (
        ForkConfig,
        ProjectService,
        ProjectServiceConfig,
    )

    registry = get_service_type_registry()

    services: list[Service] = []

    for forge_spec in config.forges:
        service = registry.instantiate(
            "forge", forge_spec.type,
            spec=forge_spec, name=forge_spec.name,
        )
        log.info(
            "Instantiated %s %r (type=%s)",
            type(service).__name__, forge_spec.name, forge_spec.type,
        )
        services.append(service)

    for proj_spec in config.projects:
        resolved = proj_spec.resolved_forks()
        if resolved:
            fork_configs = [
                ForkConfig(
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
        log.info(
            "Instantiated ProjectService %r (forge=%s)",
            proj_spec.name, proj_spec.primary_forge,
        )
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

    - **GitHub**: A notifications poller authenticated with the
      agent's PAT (notifications are user-scoped, no per-repo
      enumeration needed).
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
    native_id_to_project_name: dict[str, str],
) -> EventSource | None:
    """Create a GitHub notifications source for one (agent, forge) pair.

    The Notifications API is user-scoped (like GitLab TODOs), so no
    repository list is needed.  Only PAT credentials are supported;
    GitHub App installation tokens cannot access the Notifications API.
    """
    from thorn.tools._github_connection import GitHubPatAuth

    creds = account.credentials
    if not isinstance(creds, GitHubPatAuth):
        log.warning(
            "GitHub notifications require a PAT; credential type %s "
            "for forge %r is not supported (agent=%r). "
            "GitHub App installation tokens cannot access the Notifications API.",
            type(creds).__name__, forge_spec.name, agent_name,
        )
        return None

    from thorn.gateway.sources._github import (
        GitHubNotificationsSource,
        GitHubNotificationsSourceConfig,
    )

    base_url = forge_spec.base_url or _GITHUB_DEFAULT_API_BASE
    source_name = f"{agent_name}-{forge_spec.name}-events"

    cfg = GitHubNotificationsSourceConfig(
        token=creds.token,
        base_url=base_url,
        poll_interval=forge_spec.poll_interval,
        native_id_to_project_name=native_id_to_project_name,
    )
    source = GitHubNotificationsSource(cfg, service_name=source_name)
    log.info(
        "Inferred GitHub notifications source %r (agent=%s)",
        source_name, agent_name,
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

    if not forge_spec.base_url:
        # Catch the same misconfiguration that ``instantiate_services``
        # rejects, so that startup fails before the first poll cycle.
        raise ValueError(
            f"GitLab forge entry {forge_spec.name!r} requires a non-empty "
            "'base_url' (e.g. 'https://gitlab.example.com/api/v4') so that "
            "the inferred TODOs event source has a host to poll. "
            "Edit gateway.json to set forges[].base_url for this entry."
        )

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


__all__ = [
    "ForgeSpec",
    "ForkSpec",
    "GATEWAY_CONFIG_FILENAME",
    "GatewayConfig",
    "ProjectSpec",
    "ServiceTypeRegistry",
    "expand_env_vars",
    "get_service_type_registry",
    "infer_event_sources",
    "instantiate_services",
    "load_gateway_config",
]
