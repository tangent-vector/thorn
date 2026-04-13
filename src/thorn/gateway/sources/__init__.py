"""thorn.gateway.sources -- Concrete event source implementations.

Provides a lightweight registry mapping string type keys (e.g.
``"gitlab"``) to :class:`~thorn.gateway._event.EventSource` subclasses.
The gateway configuration loader uses this registry to instantiate
sources from ``.thorn/gateway.json``.
"""

from __future__ import annotations

from thorn.gateway._event import EventSource
from thorn.gateway.sources._github import (
    GitHubNotificationsSource,
    GitHubNotificationsSourceConfig,
)
from thorn.gateway.sources._gitlab import GitLabSourceConfig, GitLabTODOsSource

# ---------------------------------------------------------------------------
# Source registry
# ---------------------------------------------------------------------------

_SOURCE_REGISTRY: dict[str, type[EventSource]] = {}


def register_source(type_key: str, source_class: type[EventSource]) -> None:
    """Register an event source class under the given type key.

    Raises :class:`ValueError` if *type_key* is already registered.
    """
    if type_key in _SOURCE_REGISTRY:
        raise ValueError(f"Source type {type_key!r} is already registered")
    _SOURCE_REGISTRY[type_key] = source_class


def get_registered_source(type_key: str) -> type[EventSource]:
    """Look up a registered source class by type key.

    Raises :class:`KeyError` if the type is not registered.
    """
    if type_key not in _SOURCE_REGISTRY:
        registered = ", ".join(sorted(_SOURCE_REGISTRY)) or "(none)"
        raise KeyError(
            f"Unknown event source type {type_key!r}. "
            f"Registered types: {registered}"
        )
    return _SOURCE_REGISTRY[type_key]


# Register built-in sources
register_source("github", GitHubNotificationsSource)
register_source("gitlab", GitLabTODOsSource)

__all__ = [
    "GitHubNotificationsSource",
    "GitHubNotificationsSourceConfig",
    "GitLabSourceConfig",
    "GitLabTODOsSource",
    "get_registered_source",
    "register_source",
]
