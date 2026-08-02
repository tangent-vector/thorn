"""thorn.gateway -- Daemon infrastructure and event sources.

This package provides the ``Gateway`` daemon orchestrator and the
event-source abstraction that feeds external notifications (GitLab
TODOs, future webhook endpoints, etc.) into Thorn agents.
"""

from thorn.gateway._actor import ActorIdentity
from thorn.gateway._agents import (
    GatewayAgent,
    LeanProjectCoordinator,
    ProjectCoordinator,
)
from thorn.gateway._broker import (
    AgentRegistration,
    BrokerClient,
    BrokerError,
    HeaderInjection,
    ParamInjection,
    SecretRegistration,
)
from thorn.gateway._config import (
    AgentSandboxOverride,
    BrokerConfig,
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
from thorn.gateway._envelope import PeerStatus, wrap_external
from thorn.gateway._event import (
    ContextItem,
    ContextItemKind,
    EventKind,
    EventSource,
    FormattedEvent,
    IncomingEvent,
    RawIncomingEvent,
)
from thorn.gateway._formatter import (
    FormatterDelivery,
    FormatterDrop,
    FormatterResult,
    NotificationFormatter,
)
from thorn.gateway._gateway import Gateway
from thorn.gateway._peer import PeerAccount, PeerKind, PeerRegistry, PeerSpec
from thorn.gateway._trigger_policy import (
    Deliver,
    DeliverWithBanner,
    Drop,
    SourceTriggerPolicy,
    TriggerAuthorizationPolicy,
    TriggerDecision,
    UnknownActorPolicyMode,
)

__all__ = [
    "ActorIdentity",
    "AgentRegistration",
    "AgentSandboxOverride",
    "BrokerClient",
    "BrokerConfig",
    "BrokerError",
    "ContextItem",
    "ContextItemKind",
    "Deliver",
    "DeliverWithBanner",
    "Drop",
    "EventKind",
    "EventSource",
    "FormattedEvent",
    "FormatterDelivery",
    "FormatterDrop",
    "FormatterResult",
    "ForgeSpec",
    "ForkSpec",
    "Gateway",
    "GatewayConfig",
    "HeaderInjection",
    "IncomingEvent",
    "NotificationFormatter",
    "ParamInjection",
    "PeerAccount",
    "PeerKind",
    "PeerRegistry",
    "PeerSpec",
    "PeerStatus",
    "GatewayAgent",
    "LeanProjectCoordinator",
    "ProjectCoordinator",
    "ProjectSpec",
    "RawIncomingEvent",
    "SandboxConfig",
    "SecretRegistration",
    "ServiceTypeRegistry",
    "SourceTriggerPolicy",
    "TriggerAuthorizationPolicy",
    "TriggerDecision",
    "UnknownActorPolicyMode",
    "expand_env_vars",
    "get_service_type_registry",
    "infer_event_sources",
    "instantiate_services",
    "load_gateway_config",
    "wrap_external",
]
