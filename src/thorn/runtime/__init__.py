"""thorn.runtime -- Agent lifecycle, persistence, and the Runtime container.

This package sits between ``thorn.core`` (agent primitives) and
``thorn.gateway`` (daemon / event-source infrastructure), providing:

- ``Runtime``: the persistent execution environment that every Thorn
  deployment creates.  Owns the ambient ``ExecutionContext`` (via
  async context manager) and the session store.
- ``SessionKey``: typed identifier for persistent agent sessions.
- ``SessionStore``: filesystem-backed store for persisting agents.
- ``SessionSerializer`` / ``JsonSessionSerializer``: pluggable
  serialization strategy (JSON initial implementation; future Markdown
  implementation planned).
"""

from thorn.runtime._runtime import Runtime
from thorn.runtime._session import SessionKey
from thorn.runtime._store import SessionStore
from thorn.runtime._serializer import (
    JsonSessionSerializer,
    SessionSerializer,
    deserialize_history,
    serialize_history,
)

__all__ = [
    "Runtime",
    "SessionKey",
    "SessionStore",
    "SessionSerializer",
    "JsonSessionSerializer",
    "deserialize_history",
    "serialize_history",
]
