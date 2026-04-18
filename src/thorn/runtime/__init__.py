"""thorn.runtime -- Agent lifecycle, persistence, and the Runtime container.

This package sits between ``thorn.core`` (agent primitives) and
``thorn.gateway`` (daemon / event-source infrastructure), providing:

- ``Runtime``: the persistent execution environment that every Thorn
  deployment creates.  Owns the ambient ``ExecutionContext`` (via
  async context manager) and the session store.
- ``AgentID``: typed identifier for agent instances within a runtime.
- ``SessionKey``: typed identifier for persistent agent sessions.
- ``SessionStore``: filesystem-backed store for persisting agents and
  sessions.
- ``SessionSerializer`` / ``JsonSessionSerializer``: pluggable
  serialization strategy (JSON initial implementation; future Markdown
  implementation planned).
"""

from thorn.runtime._address import (
    Address,
    AddressBook,
    ServiceAddress,
    SessionAddress,
)
from thorn.runtime._inbox import SessionInbox
from thorn.runtime._in_flight_index import InFlightIndex, rebuild_in_flight_index
from thorn.runtime._lock import SessionLockError, session_lock
from thorn.runtime._notification import (
    Notification,
    NotificationSpec,
    NotificationStatus,
)
from thorn.runtime._notification_queue import (
    ArrivalKind,
    DrainOutcome,
    DrainResult,
    NotificationHandler,
    NotificationQueue,
)
from thorn.runtime._paths import AgencyPaths, safe_dirname, unsafe_dirname
from thorn.runtime._queue import DurableQueue
from thorn.runtime._runtime import Runtime
from thorn.runtime._scheduler import (
    DEFAULT_AGENT_CONCURRENCY,
    AgentScheduler,
    PromptDispatcher,
    SessionSaver,
)
from thorn.runtime._session import AgentID, SessionKey
from thorn.runtime._store import SessionStore
from thorn.runtime._serializer import (
    JsonSessionSerializer,
    SessionSerializer,
    deserialize_history,
    serialize_history,
)

__all__ = [
    "Address",
    "AddressBook",
    "AgencyPaths",
    "AgentID",
    "AgentScheduler",
    "ArrivalKind",
    "DEFAULT_AGENT_CONCURRENCY",
    "DrainOutcome",
    "DrainResult",
    "DurableQueue",
    "InFlightIndex",
    "JsonSessionSerializer",
    "Notification",
    "NotificationHandler",
    "NotificationQueue",
    "NotificationSpec",
    "NotificationStatus",
    "PromptDispatcher",
    "Runtime",
    "ServiceAddress",
    "SessionAddress",
    "SessionInbox",
    "SessionKey",
    "SessionLockError",
    "SessionSaver",
    "SessionSerializer",
    "SessionStore",
    "deserialize_history",
    "rebuild_in_flight_index",
    "safe_dirname",
    "serialize_history",
    "session_lock",
    "unsafe_dirname",
]
