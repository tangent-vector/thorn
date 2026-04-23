"""thorn.gateway -- Daemon infrastructure and event sources.

This package provides the ``Gateway`` daemon orchestrator and the
event-source abstraction that feeds external notifications (GitLab
TODOs, future webhook endpoints, etc.) into Thorn agents.
"""

from thorn.gateway._agents import ProjectCoordinator
from thorn.gateway._config import (
    AgentSandboxOverride,
    ForgeSpec,
    ForkSpec,
    GatewayConfig,
    ProjectSpec,
    SandboxConfig,
    ServiceTypeRegistry,
    expand_env_vars,
    get_service_type_registry,
    infer_event_sources,
    instantiate_services,
    load_gateway_config,
)
from thorn.gateway._event import EventSource, IncomingEvent
from thorn.gateway._gateway import Gateway

__all__ = [
    "AgentSandboxOverride",
    "EventSource",
    "ForgeSpec",
    "ForkSpec",
    "Gateway",
    "GatewayConfig",
    "IncomingEvent",
    "ProjectCoordinator",
    "ProjectSpec",
    "SandboxConfig",
    "ServiceTypeRegistry",
    "expand_env_vars",
    "get_service_type_registry",
    "infer_event_sources",
    "instantiate_services",
    "load_gateway_config",
]
