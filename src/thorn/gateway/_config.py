"""Gateway configuration: loading services from ``.thorn/gateway.json``.

The gateway configuration file declares which event sources to
instantiate at startup, replacing the previous hard-coded GitLab
source.  Each service entry specifies a ``type`` (looked up in the
source registry) and a ``config`` dict validated through the source
class's ``Config`` model.

String values in ``config`` that begin with ``$`` are treated as
environment variable references and expanded at load time, keeping
secrets out of the config file itself.

Example ``gateway.json``::

    {
      "services": [
        {
          "name": "my-gitlab",
          "type": "gitlab",
          "config": {
            "url": "$GITLAB_URL",
            "token": "$GITLAB_TOKEN"
          }
        }
      ]
    }
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel

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


def instantiate_sources(config: GatewayConfig) -> list[EventSource]:
    """Create :class:`EventSource` instances from a gateway configuration.

    For each service entry, looks up the source class in the registry,
    expands ``$ENV_VAR`` references in the config dict, validates
    through the source class's ``Config`` model, and constructs the
    source instance.
    """
    from thorn.gateway.sources import get_registered_source

    sources: list[EventSource] = []
    for spec in config.services:
        source_class = get_registered_source(spec.type)
        expanded = expand_env_vars(spec.config)
        config_instance = source_class.Config(**expanded)
        source = source_class(config_instance)
        log.info(
            "Instantiated %s source %r (type=%s)",
            type(source).__name__, spec.name, spec.type,
        )
        sources.append(source)
    return sources


__all__ = [
    "GATEWAY_CONFIG_FILENAME",
    "GatewayConfig",
    "ServiceSpec",
    "expand_env_vars",
    "instantiate_sources",
    "load_gateway_config",
]
