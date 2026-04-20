"""EventBus and listener subscriptions for execution events.

The :class:`EventBus` is itself an :class:`EventSink` implementation, so
it can be installed wherever a runtime expects a sink.  Internally it
fans every event out to a list of registered listeners (each itself an
:class:`EventSink`), each carrying a ``ScopeFilter`` predicate so it
only sees events whose scope chain matches.

The motivating use case is multi-session execution under a shared
:class:`~thorn.runtime.Runtime`: the runtime owns one bus, and each
caller (a CLI command, a chat REPL, a daemon-side log writer, ...)
subscribes a listener whose filter is scoped to the session it cares
about.  Tagging is already done for free by
:func:`~thorn.core._agent._run_session_prompt`, which sets
``session_key`` on the agent scope's metadata; the helper
:func:`in_session` builds a filter that matches that tag anywhere up
the scope chain.

Design notes:

- The bus is a *broadcast-with-filter* model rather than a
  topic/queue model.  This keeps the wire protocol identical to
  :class:`EventSink` (every method already accepts a ``Scope``) and
  defers all filtering policy to the listener side, where it is easy
  to compose (any predicate over the scope chain is admissible).
- A misbehaving listener should not be able to break unrelated
  listeners or, worse, the agent loop that produced the event.  All
  listener invocations are wrapped in a try/except by default; the
  exception is logged and dropped.  *strict* mode is available for
  tests that want listener errors to surface immediately.
- Listeners are awaited sequentially, in subscription order.  Parallel
  fanout via :func:`asyncio.gather` was considered but rejected for
  Phase 3: most listeners are quick (console writes, JSONL appends),
  parallelism would obscure exception ordering, and serial dispatch
  matches what the existing :class:`CompositeEventSink` already does.
- Subscriptions are usable as context managers so callers can write
  ``with bus.subscribe(listener, scope_filter=...): ...`` and have the
  unsubscribe happen on every exit path.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable

from thorn.core._context import EventSink, Scope

if TYPE_CHECKING:
    from thorn.core._provider import ResponseChunk
    from thorn.runtime._session import SessionKey


log = logging.getLogger(__name__)


ScopeFilter = Callable[["Scope | None"], bool]
"""Predicate that decides whether a listener wants a given scope's events.

Filters take the same ``Scope | None`` value the event-sink methods
receive.  Returning ``True`` means deliver the event; ``False`` means
skip this listener for this event.
"""


def accept_all(scope: Scope | None) -> bool:
    """A :data:`ScopeFilter` that accepts every event regardless of scope."""
    return True


def in_session(session_key: "SessionKey | str") -> ScopeFilter:
    """Build a filter that matches events tagged with *session_key*.

    A scope chain is considered "in" *session_key* if any link in the
    chain (from the event's scope up through every ``outer``) carries
    ``session_key`` in its metadata equal to the requested key.

    The session key is set on the agent scope by
    :func:`~thorn.core._agent._run_session_prompt`, so any event
    emitted while running inside a session prompt -- including tool
    calls and skill scopes pushed under the agent scope -- will match.

    Events with ``scope=None`` (e.g. runtime-level events emitted
    outside any session) never match a session filter; subscribe a
    second unfiltered listener if you also want those.
    """
    target = str(session_key)

    def match(scope: Scope | None) -> bool:
        current = scope
        while current is not None:
            if current.metadata.get("session_key") == target:
                return True
            current = current.outer
        return False

    return match


class Subscription:
    """A handle representing one listener's registration on an :class:`EventBus`.

    Returned by :meth:`EventBus.subscribe`.  The handle is also a
    context manager so callers can scope the subscription to a block::

        with bus.subscribe(listener, scope_filter=in_session(key)):
            ...

    :meth:`unsubscribe` is idempotent; calling it on an already
    unsubscribed handle (e.g. after the context-manager exit) is a
    no-op.
    """

    def __init__(
        self,
        bus: "EventBus",
        listener: EventSink,
        scope_filter: ScopeFilter,
    ) -> None:
        self._bus = bus
        self._listener = listener
        self._scope_filter = scope_filter
        self._active = True

    @property
    def listener(self) -> EventSink:
        return self._listener

    @property
    def scope_filter(self) -> ScopeFilter:
        return self._scope_filter

    @property
    def active(self) -> bool:
        return self._active

    def unsubscribe(self) -> None:
        """Remove this listener from the bus.  Safe to call more than once."""
        if not self._active:
            return
        self._active = False
        self._bus._remove(self)

    def __enter__(self) -> "Subscription":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        self.unsubscribe()


class EventBus(EventSink):
    """An :class:`EventSink` that fans events out to filtered listeners.

    Event producers (the agent loop, tools, the gateway) are unaware
    that they are publishing to a bus -- the bus is just an
    :class:`EventSink` to them.  Consumers register via
    :meth:`subscribe` with an optional :data:`ScopeFilter` describing
    which events they care about.

    *strict*: when ``True``, exceptions raised by listeners propagate
    out of the event-sink call.  Default is ``False`` -- exceptions
    are logged and swallowed so a buggy listener cannot disrupt the
    agent loop or sibling listeners.  Use ``strict=True`` only in
    tests.
    """

    def __init__(self, *, strict: bool = False) -> None:
        self._subscriptions: list[Subscription] = []
        self._strict = strict

    def subscribe(
        self,
        listener: EventSink,
        *,
        scope_filter: ScopeFilter = accept_all,
    ) -> Subscription:
        """Register *listener* to receive events matching *scope_filter*.

        Returns a :class:`Subscription` handle.  The same listener may
        be subscribed multiple times with different filters; each
        subscription is independent.
        """
        sub = Subscription(self, listener, scope_filter)
        self._subscriptions.append(sub)
        return sub

    def _remove(self, sub: Subscription) -> None:
        try:
            self._subscriptions.remove(sub)
        except ValueError:
            pass

    @property
    def subscription_count(self) -> int:
        """Number of currently registered subscriptions (test helper)."""
        return len(self._subscriptions)

    async def _dispatch(
        self,
        event_scope: Scope | None,
        method_name: str,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Fan-out helper used by every event-sink method override.

        *event_scope* is the ``Scope`` associated with the event (used
        for filtering only); *args*/*kwargs* are the verbatim arguments
        to forward to each matching listener's method.  The parameter
        name avoids collision with the ``scope=`` keyword that several
        :class:`EventSink` methods accept.

        Iterates over a snapshot of the subscription list so a listener
        unsubscribing itself mid-fanout cannot corrupt iteration.
        """
        for sub in list(self._subscriptions):
            if not sub.active:
                continue
            if not sub.scope_filter(event_scope):
                continue
            method = getattr(sub.listener, method_name)
            try:
                await method(*args, **kwargs)
            except Exception:
                if self._strict:
                    raise
                log.exception(
                    "EventBus listener %r raised in %s; dropping event",
                    sub.listener, method_name,
                )

    async def on_response_chunk(
        self,
        chunk: "ResponseChunk",
        scope: Scope | None = None,
    ) -> None:
        await self._dispatch(scope, "on_response_chunk", chunk, scope=scope)

    async def on_status(
        self,
        message: str,
        scope: Scope | None = None,
    ) -> None:
        await self._dispatch(scope, "on_status", message, scope=scope)

    async def on_scope_enter(self, scope: Scope) -> None:
        await self._dispatch(scope, "on_scope_enter", scope)

    async def on_scope_exit(
        self, scope: Scope, *, duration_s: float | None = None,
    ) -> None:
        await self._dispatch(
            scope, "on_scope_exit", scope, duration_s=duration_s,
        )

    async def on_tool_start(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        scope: Scope | None = None,
    ) -> None:
        await self._dispatch(
            scope, "on_tool_start", name, arguments, scope=scope,
        )

    async def on_tool_end(
        self,
        name: str,
        *,
        duration_s: float | None = None,
        error: str | None = None,
        scope: Scope | None = None,
    ) -> None:
        await self._dispatch(
            scope, "on_tool_end", name,
            duration_s=duration_s, error=error, scope=scope,
        )

    async def on_completion_end(
        self,
        *,
        duration_s: float | None = None,
        usage: dict[str, int] | None = None,
        scope: Scope | None = None,
    ) -> None:
        await self._dispatch(
            scope, "on_completion_end",
            duration_s=duration_s, usage=usage, scope=scope,
        )

    async def on_advisory(
        self,
        source: str,
        content: str,
        *,
        scope: Scope | None = None,
    ) -> None:
        await self._dispatch(
            scope, "on_advisory", source, content, scope=scope,
        )


__all__ = [
    "EventBus",
    "ScopeFilter",
    "Subscription",
    "accept_all",
    "in_session",
]
