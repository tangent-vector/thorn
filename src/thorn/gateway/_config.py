"""Gateway configuration: loading services from ``.thorn/gateway.json``.

The gateway configuration file declares which services to instantiate
at startup.  Services include forge connections (``"gitlab"``,
``"github"``), project definitions (``"project"``), and event sources
(``"gitlab-events"``, ``"github-events"``).

Each service entry specifies a ``type`` (looked up in the service type
registry) and a ``config`` dict validated through the service class's
``Config`` model.

String values in ``config`` that begin with ``$`` are treated as
environment variable references and expanded at load time, keeping
secrets out of the config file itself.

Example ``gateway.json``::

    {
      "services": [
        {
          "name": "gitlab-master",
          "type": "gitlab",
          "config": { "url": "$GITLAB_URL", "token": "$GITLAB_TOKEN" }
        },
        {
          "name": "lace",
          "type": "project",
          "config": {
            "forge": "gitlab-master",
            "native_id": "214768",
            "path": "lace/lace",
            "clone_url": "https://gitlab-master.nvidia.com/lace/lace.git",
            "default_branch": "main"
          }
        },
        {
          "name": "gitlab-poller",
          "type": "gitlab-events",
          "config": { "url": "$GITLAB_URL", "token": "$GITLAB_TOKEN" }
        }
      ]
    }

A ``github`` forge service uses :class:`~thorn.tools._github_connection.GitHubConnectionConfig` — for example a GitHub App installation::

    {
      "name": "my-gh",
      "type": "github",
      "config": {
        "base_url": "$GITHUB_URL",
        "auth": {
          "kind": "app",
          "app_id": "$GITHUB_APP_ID",
          "installation_id": "$GITHUB_APP_INSTALLATION_ID",
          "private_key_pem": "$GITHUB_APP_PRIVATE_KEY"
        }
      }
    }
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel

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
# Configuration models
# ---------------------------------------------------------------------------


class ServiceSpec(BaseModel):
    """One entry in the ``"services"`` array of ``gateway.json``."""

    name: str
    type: str
    config: dict[str, Any] = {}


class GatewayConfig(BaseModel):
    """Top-level model for ``.thorn/gateway.json``."""

    services: list[ServiceSpec] = []


# ---------------------------------------------------------------------------
# Service type registry
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
# Loading and instantiation
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

    Handles all service types: forge connections (``"gitlab"``,
    ``"github"``), projects (``"project"``), and event sources
    (``"gitlab-events"``, ``"github-events"``).

    For backward compatibility, the legacy type keys ``"gitlab"`` and
    ``"github"`` (without ``-events`` suffix) are also tried against
    the event source registry when no match is found in the new
    service type registry.
    """
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
    "GATEWAY_CONFIG_FILENAME",
    "GatewayConfig",
    "ServiceSpec",
    "expand_env_vars",
    "instantiate_services",
    "instantiate_sources",
    "load_gateway_config",
]
