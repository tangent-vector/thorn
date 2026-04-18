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
from thorn.runtime._dispatch import (
    DispatchError,
    apply_handling_transition,
    dispatch_step_two,
)
from thorn.runtime._inbox import SessionInbox
from thorn.runtime._inbox_tools import (
    INBOX_TOOLS,
    list_inbox_items,
    read_inbox_item,
    update_inbox_item,
)
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
from thorn.runtime._prompt_format import (
    build_inbox_prompt,
    inbox_prompt_dispatcher,
    summarize_notification_content,
)
from thorn.runtime._queue import DurableQueue
from thorn.runtime._runtime import Runtime
from thorn.runtime._scheduler import (
    DEFAULT_AGENT_CONCURRENCY,
    DEFAULT_PROGRESS_STRIKES,
    AgentScheduler,
    PromptDispatcher,
    ProgressEvictor,
    SessionSaver,
    default_progress_evictor,
)
from thorn.runtime._session import AgentID, SessionKey
from thorn.runtime._store import SessionStore
from thorn.runtime._sweep import SweepReport, run_startup_sweep
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
    "DEFAULT_PROGRESS_STRIKES",
    "DispatchError",
    "DrainOutcome",
    "DrainResult",
    "DurableQueue",
    "INBOX_TOOLS",
    "InFlightIndex",
    "JsonSessionSerializer",
    "Notification",
    "NotificationHandler",
    "NotificationQueue",
    "NotificationSpec",
    "NotificationStatus",
    "ProgressEvictor",
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
    "SweepReport",
    "apply_handling_transition",
    "build_inbox_prompt",
    "default_progress_evictor",
    "deserialize_history",
    "dispatch_step_two",
    "inbox_prompt_dispatcher",
    "list_inbox_items",
    "read_inbox_item",
    "rebuild_in_flight_index",
    "run_startup_sweep",
    "safe_dirname",
    "serialize_history",
    "session_lock",
    "summarize_notification_content",
    "unsafe_dirname",
    "update_inbox_item",
]
