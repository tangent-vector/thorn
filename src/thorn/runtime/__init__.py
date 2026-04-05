"""thorn.runtime -- Session lifecycle, persistence, and the Runtime container.

This package sits between ``thorn.core`` (agent primitives) and
``thorn.gateway`` (daemon / event-source infrastructure), providing:

- ``Runtime``: the persistent execution environment that every Thorn
  deployment creates.  Factory for ``ExecutionContext`` instances and
  owner of the session store.
- ``Session`` / ``SessionKey``: a single agent conversation context with
  lifecycle metadata.
- ``SessionStore``: filesystem-backed store for persisting sessions.
- ``SessionSerializer`` / ``JsonSessionSerializer``: pluggable
  serialization strategy (JSON initial implementation; future Markdown
  implementation planned).
"""

from thorn.runtime._runtime import Runtime
from thorn.runtime._session import Session, SessionKey
from thorn.runtime._store import SessionStore
from thorn.runtime._serializer import (
    JsonSessionSerializer,
    SessionSerializer,
    deserialize_history,
    serialize_history,
)

__all__ = [
    "Runtime",
    "Session",
    "SessionKey",
    "SessionStore",
    "SessionSerializer",
    "JsonSessionSerializer",
    "deserialize_history",
    "serialize_history",
]
