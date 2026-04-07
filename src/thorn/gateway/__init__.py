"""thorn.gateway -- Daemon infrastructure and event sources.

This package provides the ``Gateway`` daemon orchestrator and the
event-source abstraction that feeds external notifications (GitLab
TODOs, future webhook endpoints, etc.) into Thorn agents.
"""

from thorn.gateway._event import EventSource, IncomingEvent
from thorn.gateway._gateway import Gateway

__all__ = [
    "EventSource",
    "Gateway",
    "IncomingEvent",
]
