"""Tests for ``thorn.core._event_bus`` — :class:`EventBus`, subscriptions, filters."""

from __future__ import annotations

from typing import Any

import pytest

from thorn.core._context import EventSink, Scope
from thorn.core._event_bus import (
    EventBus,
    Subscription,
    accept_all,
    in_session,
)
from thorn.core._provider import TextChunk


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

class _RecordingSink(EventSink):
    """An :class:`EventSink` that records every method call it receives.

    Used by the bus tests to assert which listener saw which events.
    Each entry is a ``(method_name, args, kwargs)`` triple.
    """

    def __init__(self) -> None:
        self.events: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

    async def on_response_chunk(
        self, chunk: Any, scope: Scope | None = None,
    ) -> None:
        self.events.append(("on_response_chunk", (chunk,), {"scope": scope}))

    async def on_status(
        self, message: str, scope: Scope | None = None,
    ) -> None:
        self.events.append(("on_status", (message,), {"scope": scope}))

    async def on_scope_enter(self, scope: Scope) -> None:
        self.events.append(("on_scope_enter", (scope,), {}))

    async def on_scope_exit(
        self, scope: Scope, *, duration_s: float | None = None,
    ) -> None:
        self.events.append(
            ("on_scope_exit", (scope,), {"duration_s": duration_s}),
        )

    async def on_tool_start(
        self, name: str, arguments: dict[str, Any],
        *, scope: Scope | None = None,
    ) -> None:
        self.events.append(
            ("on_tool_start", (name, arguments), {"scope": scope}),
        )

    async def on_tool_end(
        self, name: str, *, duration_s: float | None = None,
        error: str | None = None, scope: Scope | None = None,
    ) -> None:
        self.events.append((
            "on_tool_end", (name,),
            {"duration_s": duration_s, "error": error, "scope": scope},
        ))

    async def on_completion_end(
        self, *, duration_s: float | None = None,
        usage: dict[str, int] | None = None,
        scope: Scope | None = None,
    ) -> None:
        self.events.append((
            "on_completion_end", (),
            {"duration_s": duration_s, "usage": usage, "scope": scope},
        ))

    async def on_advisory(
        self, source: str, content: str,
        *, scope: Scope | None = None,
    ) -> None:
        self.events.append(
            ("on_advisory", (source, content), {"scope": scope}),
        )


class _RaisingSink(EventSink):
    """An :class:`EventSink` whose every method raises a chosen exception.

    Used to verify that the bus's exception handling honours its
    ``strict`` flag.
    """

    def __init__(self, exc: Exception) -> None:
        self._exc = exc

    async def on_response_chunk(
        self, chunk: Any, scope: Scope | None = None,
    ) -> None:
        raise self._exc

    async def on_status(
        self, message: str, scope: Scope | None = None,
    ) -> None:
        raise self._exc


# ---------------------------------------------------------------------------
# accept_all
# ---------------------------------------------------------------------------

class TestAcceptAll:
    def test_accepts_none_scope(self):
        assert accept_all(None) is True

    def test_accepts_any_scope(self):
        scope = Scope(description="x", metadata={"session_key": "abc"})
        assert accept_all(scope) is True


# ---------------------------------------------------------------------------
# in_session
# ---------------------------------------------------------------------------

class TestInSession:
    def test_matches_direct_session_key(self):
        scope = Scope(description="agent", metadata={"session_key": "s1"})
        assert in_session("s1")(scope) is True

    def test_matches_session_key_in_outer_chain(self):
        outer = Scope(description="agent", metadata={"session_key": "s1"})
        inner = Scope(description="tool", outer=outer)
        assert in_session("s1")(inner) is True

    def test_does_not_match_different_key(self):
        scope = Scope(description="agent", metadata={"session_key": "s2"})
        assert in_session("s1")(scope) is False

    def test_does_not_match_none_scope(self):
        # Events emitted outside any session (e.g. runtime-level
        # advisories) should not get routed to a session-scoped
        # listener.
        assert in_session("s1")(None) is False

    def test_does_not_match_scope_without_session_key(self):
        scope = Scope(description="bare", metadata={})
        assert in_session("s1")(scope) is False

    def test_accepts_session_key_object(self):
        # The factory stringifies its argument so any SessionKey-like
        # value (NewType wrapper around str) is accepted.
        from thorn.runtime._session import SessionKey
        scope = Scope(description="agent", metadata={"session_key": "s1"})
        assert in_session(SessionKey("s1"))(scope) is True


# ---------------------------------------------------------------------------
# EventBus subscription lifecycle
# ---------------------------------------------------------------------------

class TestEventBusLifecycle:
    def test_starts_with_no_subscriptions(self):
        bus = EventBus()
        assert bus.subscription_count == 0

    def test_subscribe_returns_active_subscription(self):
        bus = EventBus()
        sink = _RecordingSink()
        sub = bus.subscribe(sink)
        assert isinstance(sub, Subscription)
        assert sub.active is True
        assert sub.listener is sink
        assert bus.subscription_count == 1

    def test_unsubscribe_removes_from_bus(self):
        bus = EventBus()
        sub = bus.subscribe(_RecordingSink())
        sub.unsubscribe()
        assert sub.active is False
        assert bus.subscription_count == 0

    def test_unsubscribe_is_idempotent(self):
        bus = EventBus()
        sub = bus.subscribe(_RecordingSink())
        sub.unsubscribe()
        sub.unsubscribe()
        assert bus.subscription_count == 0

    def test_subscription_as_context_manager(self):
        bus = EventBus()
        sink = _RecordingSink()
        with bus.subscribe(sink) as sub:
            assert sub.active is True
            assert bus.subscription_count == 1
        assert sub.active is False
        assert bus.subscription_count == 0

    def test_same_listener_can_subscribe_multiple_times(self):
        # Each subscription is independent; useful when the same
        # listener wants events from multiple disjoint scopes.
        bus = EventBus()
        sink = _RecordingSink()
        sub1 = bus.subscribe(sink, scope_filter=in_session("a"))
        sub2 = bus.subscribe(sink, scope_filter=in_session("b"))
        assert bus.subscription_count == 2
        sub1.unsubscribe()
        assert bus.subscription_count == 1
        sub2.unsubscribe()
        assert bus.subscription_count == 0


# ---------------------------------------------------------------------------
# EventBus fanout
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestEventBusFanout:
    async def test_unfiltered_listener_sees_every_event(self):
        bus = EventBus()
        sink = _RecordingSink()
        bus.subscribe(sink)

        scope = Scope(description="x", metadata={"session_key": "s1"})
        await bus.on_status("hello", scope=scope)
        await bus.on_response_chunk(TextChunk(text="world"), scope=scope)
        await bus.on_scope_enter(scope)

        assert [e[0] for e in sink.events] == [
            "on_status", "on_response_chunk", "on_scope_enter",
        ]

    async def test_filter_screens_non_matching_events(self):
        bus = EventBus()
        sink = _RecordingSink()
        bus.subscribe(sink, scope_filter=in_session("s1"))

        s1 = Scope(description="agent", metadata={"session_key": "s1"})
        s2 = Scope(description="agent", metadata={"session_key": "s2"})
        await bus.on_status("a", scope=s1)
        await bus.on_status("b", scope=s2)
        await bus.on_status("c", scope=s1)

        messages = [e[1][0] for e in sink.events]
        assert messages == ["a", "c"]

    async def test_two_listeners_receive_independently(self):
        bus = EventBus()
        a_sink = _RecordingSink()
        b_sink = _RecordingSink()
        bus.subscribe(a_sink, scope_filter=in_session("a"))
        bus.subscribe(b_sink, scope_filter=in_session("b"))

        scope_a = Scope(description="agent", metadata={"session_key": "a"})
        scope_b = Scope(description="agent", metadata={"session_key": "b"})
        await bus.on_status("for-a", scope=scope_a)
        await bus.on_status("for-b", scope=scope_b)

        assert [e[1][0] for e in a_sink.events] == ["for-a"]
        assert [e[1][0] for e in b_sink.events] == ["for-b"]

    async def test_filter_walks_outer_chain(self):
        # Tool-scope events should still reach a session-scoped
        # listener because the session_key tag lives on an outer
        # scope (the agent scope).
        bus = EventBus()
        sink = _RecordingSink()
        bus.subscribe(sink, scope_filter=in_session("s1"))

        agent_scope = Scope(
            description="agent", metadata={"session_key": "s1"},
        )
        tool_scope = Scope(description="tool", outer=agent_scope)
        await bus.on_tool_start("my_tool", {"x": 1}, scope=tool_scope)

        assert len(sink.events) == 1
        method, args, kwargs = sink.events[0]
        assert method == "on_tool_start"
        assert args == ("my_tool", {"x": 1})
        assert kwargs == {"scope": tool_scope}

    async def test_listener_unsubscribed_during_fanout_misses_remaining_events(
        self,
    ):
        # A listener that unsubscribes itself mid-event shouldn't
        # corrupt iteration; it should also stop receiving subsequent
        # events on later bus calls.
        bus = EventBus()
        a_sink = _RecordingSink()
        b_sink = _RecordingSink()
        a_sub = bus.subscribe(a_sink)

        class SelfUnsubscribingSink(EventSink):
            async def on_response_chunk(self, chunk, scope=None): ...
            async def on_status(self, message, scope=None):
                a_sub.unsubscribe()

        bus.subscribe(SelfUnsubscribingSink())
        bus.subscribe(b_sink)

        await bus.on_status("first", scope=None)
        # A receives this round (snapshot), b receives it.
        await bus.on_status("second", scope=None)
        # A is gone; b still here.

        assert [e[1][0] for e in a_sink.events] == ["first"]
        assert [e[1][0] for e in b_sink.events] == ["first", "second"]

    async def test_runtime_level_event_skipped_by_session_listener(self):
        # Events without a session-tagged scope should not reach a
        # session-scoped listener; they should still reach an
        # unfiltered listener.
        bus = EventBus()
        all_sink = _RecordingSink()
        s1_sink = _RecordingSink()
        bus.subscribe(all_sink)
        bus.subscribe(s1_sink, scope_filter=in_session("s1"))

        await bus.on_status("runtime msg", scope=None)

        assert [e[1][0] for e in all_sink.events] == ["runtime msg"]
        assert s1_sink.events == []


# ---------------------------------------------------------------------------
# EventBus exception handling
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestEventBusErrorHandling:
    async def test_listener_exception_is_swallowed_by_default(self, caplog):
        bus = EventBus()
        bad = _RaisingSink(RuntimeError("boom"))
        good = _RecordingSink()
        bus.subscribe(bad)
        bus.subscribe(good)

        # Should not raise; good still receives the event.
        with caplog.at_level("ERROR", logger="thorn.core._event_bus"):
            await bus.on_status("ping")

        assert any("boom" in r.message or "boom" in str(r.exc_info)
                   for r in caplog.records)
        assert [e[1][0] for e in good.events] == ["ping"]

    async def test_strict_mode_propagates_listener_exception(self):
        bus = EventBus(strict=True)
        bus.subscribe(_RaisingSink(ValueError("nope")))
        with pytest.raises(ValueError, match="nope"):
            await bus.on_status("ping")

    async def test_one_listener_exception_does_not_block_subsequent_listeners(
        self, caplog,
    ):
        bus = EventBus()
        sink_before = _RecordingSink()
        sink_after = _RecordingSink()
        bus.subscribe(sink_before)
        bus.subscribe(_RaisingSink(RuntimeError("middle blew up")))
        bus.subscribe(sink_after)

        with caplog.at_level("ERROR", logger="thorn.core._event_bus"):
            await bus.on_status("hi")

        assert [e[1][0] for e in sink_before.events] == ["hi"]
        assert [e[1][0] for e in sink_after.events] == ["hi"]


# ---------------------------------------------------------------------------
# EventSink contract -- bus is itself usable wherever a sink is expected.
# ---------------------------------------------------------------------------

class TestEventBusIsAnEventSink:
    def test_is_event_sink_subclass(self):
        assert issubclass(EventBus, EventSink)

    def test_can_be_used_where_event_sink_is_required(self):
        # A trivial type-safety smoke test: the runtime accepts any
        # EventSink for ``event_sink=`` and downstream code only
        # speaks the sink protocol.  If this constructor accepts our
        # bus, the rest follows from the abstract method overrides.
        from pathlib import Path
        from thorn.core._provider import MockProvider
        from thorn.runtime import Runtime

        bus = EventBus()
        rt = Runtime(
            provider=MockProvider(),
            event_sink=bus,
            workspace_root=Path("/tmp"),
        )
        assert rt.event_sink is bus
