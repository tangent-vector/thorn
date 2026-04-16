"""thorn.gateway -- Daemon infrastructure and event sources.

This package provides the ``Gateway`` daemon orchestrator and the
event-source abstraction that feeds external notifications (GitLab
TODOs, future webhook endpoints, etc.) into Thorn agents.
"""

from thorn.gateway._agents import ProjectCoordinator
from thorn.gateway._config import (
    ForgeSpec,
    GatewayConfig,
    ProjectSpec,
    ServiceSpec,
    expand_env_vars,
    infer_event_sources,
    instantiate_new_format,
    instantiate_services,
    instantiate_sources,
    load_gateway_config,
)
from thorn.gateway._event import EventSource, IncomingEvent
from thorn.gateway._gateway import Gateway

__all__ = [
    "EventSource",
    "ForgeSpec",
    "Gateway",
    "GatewayConfig",
    "IncomingEvent",
    "ProjectCoordinator",
    "ProjectSpec",
    "ServiceSpec",
    "expand_env_vars",
    "infer_event_sources",
    "instantiate_new_format",
    "instantiate_services",
    "instantiate_sources",
    "load_gateway_config",
]
