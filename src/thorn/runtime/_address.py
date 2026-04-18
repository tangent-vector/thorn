"""Addresses for durable queues in an agency.

An ``Address`` identifies the *destination* of a notification: the
durable queue that should receive it.  Two kinds exist:

- ``SessionAddress`` -- targets the inbox of a specific agent session.
  Rendered as ``session:<agent-id>/<session-key>``.
- ``ServiceAddress`` -- targets the notification queue of a registered
  service (e.g. a forge client, an event source).  Rendered as
  ``service:<service-name>``.

Addresses are stored and compared as structured values.  The canonical
string form is a serialization concern -- used in logs, persisted
files, and the on-the-wire notation -- never for in-memory comparison.

``AddressBook`` is a small in-memory registry that maps an ``Address``
to the :class:`~thorn.runtime._queue.DurableQueue` that backs it.  The
``Runtime`` populates the book at startup as agents, sessions, and
services come online.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from thorn.runtime._session import AgentID, SessionKey

if TYPE_CHECKING:
    from thorn.runtime._queue import DurableQueue


class Address:
    """Abstract base for queue addresses.

    Concrete subclasses are frozen dataclasses that render to a
    canonical ``<kind>:<body>`` string and can be parsed back from it.
    Instances are hashable and usable as dict keys.

    Agent IDs and service names are assumed not to contain ``/`` or
    ``:``; session keys may contain ``/`` (partitioning in ``parse``
    uses the first ``/`` only).
    """

    def __str__(self) -> str:  # pragma: no cover - abstract
        raise NotImplementedError

    @staticmethod
    def parse(text: str) -> Address:
        """Parse a canonical ``<kind>:<body>`` string into an ``Address``.

        Raises ``ValueError`` for any input that does not match one of
        the supported grammars.
        """
        kind, sep, body = text.partition(":")
        if not sep:
            raise ValueError(f"Address missing ':' separator: {text!r}")
        if kind == "session":
            agent_raw, slash, key_raw = body.partition("/")
            if not slash or not agent_raw or not key_raw:
                raise ValueError(
                    f"Session address must be 'session:<agent-id>/<session-key>': {text!r}"
                )
            return SessionAddress(AgentID(agent_raw), SessionKey(key_raw))
        if kind == "service":
            if not body:
                raise ValueError(
                    f"Service address must be 'service:<service-name>': {text!r}"
                )
            return ServiceAddress(body)
        raise ValueError(f"Unknown address kind {kind!r} in {text!r}")


@dataclass(frozen=True)
class SessionAddress(Address):
    """Address of a session's inbox.

    The canonical string form is ``session:<agent-id>/<session-key>``.
    Agent IDs must not contain ``/``; session keys may.
    """

    agent_id: AgentID
    session_key: SessionKey

    def __post_init__(self) -> None:
        if "/" in str(self.agent_id):
            raise ValueError(
                f"Agent ID must not contain '/': {self.agent_id!r}"
            )
        if ":" in str(self.agent_id):
            raise ValueError(
                f"Agent ID must not contain ':': {self.agent_id!r}"
            )
        if not str(self.session_key):
            raise ValueError("Session key must be non-empty")

    def __str__(self) -> str:
        return f"session:{self.agent_id}/{self.session_key}"


@dataclass(frozen=True)
class ServiceAddress(Address):
    """Address of a registered service's notification queue.

    The canonical string form is ``service:<service-name>``.
    Service names must not contain ``/`` or ``:``.
    """

    service_name: str

    def __post_init__(self) -> None:
        if not self.service_name:
            raise ValueError("Service name must be non-empty")
        if "/" in self.service_name or ":" in self.service_name:
            raise ValueError(
                f"Service name must not contain '/' or ':': {self.service_name!r}"
            )

    def __str__(self) -> str:
        return f"service:{self.service_name}"


class AddressBook:
    """In-memory registry mapping ``Address`` values to ``DurableQueue``s.

    The ``Runtime`` owns a single instance and populates it as agents,
    sessions, and services become available.  The book is the single
    point of truth for routing a notification by address.

    This class is not thread-safe on its own -- callers must serialize
    access if registration happens concurrently with resolution.
    Typical usage populates the book during startup (single-threaded)
    and then treats it as read-mostly.
    """

    def __init__(self) -> None:
        self._entries: dict[Address, DurableQueue] = {}

    def register(self, address: Address, queue: DurableQueue) -> None:
        """Register *queue* as the resolver for *address*.

        Raises ``ValueError`` if *address* is already registered.  Use
        ``unregister`` + ``register`` to replace a binding deliberately.
        """
        if address in self._entries:
            raise ValueError(f"Address already registered: {address}")
        self._entries[address] = queue

    def unregister(self, address: Address) -> None:
        """Remove the binding for *address*.

        Raises ``KeyError`` if *address* is not registered.
        """
        del self._entries[address]

    def resolve(self, address: Address) -> DurableQueue:
        """Return the queue for *address*.

        Raises ``LookupError`` if *address* is not registered.  Use
        :meth:`get` for a non-raising variant.
        """
        try:
            return self._entries[address]
        except KeyError:
            raise LookupError(f"No queue registered for address: {address}") from None

    def get(self, address: Address) -> DurableQueue | None:
        """Return the queue for *address*, or ``None`` if not registered."""
        return self._entries.get(address)

    def __contains__(self, address: object) -> bool:
        return address in self._entries

    def __len__(self) -> int:
        return len(self._entries)

    def addresses(self) -> list[Address]:
        """Return a list of all currently-registered addresses."""
        return list(self._entries.keys())


__all__ = [
    "Address",
    "AddressBook",
    "ServiceAddress",
    "SessionAddress",
]
