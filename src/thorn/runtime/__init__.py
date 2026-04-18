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
from thorn.runtime._lock import SessionLockError, session_lock
from thorn.runtime._notification import (
    Notification,
    NotificationSpec,
    NotificationStatus,
)
from thorn.runtime._paths import AgencyPaths
from thorn.runtime._queue import DurableQueue
from thorn.runtime._runtime import Runtime
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
    "DurableQueue",
    "JsonSessionSerializer",
    "Notification",
    "NotificationSpec",
    "NotificationStatus",
    "Runtime",
    "ServiceAddress",
    "SessionAddress",
    "SessionKey",
    "SessionLockError",
    "SessionSerializer",
    "SessionStore",
    "deserialize_history",
    "serialize_history",
    "session_lock",
]
