"""Tests for thorn.runtime -- Session, SessionStore, serialization, and Runtime."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from thorn.core._agent import Agent
from thorn.core._context import ExecutionContext
from thorn.core._history import (
    CollapseState,
    HistoryTree,
    ToolCallNode,
    TurnNode,
    UserPromptNode,
)
from thorn.core._messages import (
    AssistantMessage,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)
from thorn.core._provider import MockProvider
from thorn.runtime import (
    JsonSessionSerializer,
    Runtime,
    Session,
    SessionKey,
    SessionStore,
    deserialize_history,
    serialize_history,
)


# ---------------------------------------------------------------------------
# Agent.name and Agent.metadata (Phase 3 additions)
# ---------------------------------------------------------------------------


class TestAgentFields:
    def test_default_name_is_none(self):
        agent = Agent()
        assert agent.name is None

    def test_default_metadata_is_empty(self):
        agent = Agent()
        assert agent.metadata == {}

    def test_explicit_name(self):
        agent = Agent(name="reviewer")
        assert agent.name == "reviewer"

    def test_explicit_metadata(self):
        agent = Agent(metadata={"project": "thorn"})
        assert agent.metadata == {"project": "thorn"}

    def test_name_in_str_representation(self):
        agent = Agent(name="reviewer")
        assert str(agent) == "Agent('reviewer')"

    def test_str_without_name(self):
        agent = Agent()
        assert str(agent) == "Agent"

    def test_subclass_str_with_name(self):
        class Developer(Agent):
            pass

        agent = Developer(name="alice")
        assert str(agent) == "Developer('alice')"

    def test_kwargs_still_work(self):
        agent = Agent(name="test", module="parser")
        assert agent.name == "test"
        assert agent.module == "parser"  # type: ignore[attr-defined]

    def test_name_available_in_prompt_templates(self):
        class Greeter(Agent):
            system_prompts = ["Hello, {name}!"]

        agent = Greeter(name="world")
        rendered = agent._render_system_prompts()
        assert rendered == ["Hello, world!"]


# ---------------------------------------------------------------------------
# SessionKey
# ---------------------------------------------------------------------------


class TestSessionKey:
    def test_is_str_subclass(self):
        key = SessionKey("my-session")
        assert isinstance(key, str)
        assert isinstance(key, SessionKey)

    def test_str_operations(self):
        key = SessionKey("gitlab:issue:42")
        assert key.startswith("gitlab:")
        assert "issue" in key

    def test_equality_with_str(self):
        key = SessionKey("abc")
        assert key == "abc"

    def test_as_dict_key(self):
        key = SessionKey("k1")
        d: dict[SessionKey, int] = {key: 1}
        assert d[SessionKey("k1")] == 1


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------


class TestSession:
    def test_creation_with_defaults(self):
        agent = Agent(name="test")
        session = Session(key=SessionKey("s1"), agent=agent)
        assert session.key == "s1"
        assert session.agent is agent
        assert session.metadata == {}
        assert isinstance(session.created_at, datetime)
        assert isinstance(session.last_active, datetime)

    def test_creation_with_metadata(self):
        agent = Agent()
        session = Session(
            key=SessionKey("s2"),
            agent=agent,
            metadata={"issue_iid": 42},
        )
        assert session.metadata == {"issue_iid": 42}

    def test_touch_updates_last_active(self):
        agent = Agent()
        session = Session(key=SessionKey("s3"), agent=agent)
        old_ts = session.last_active
        import time
        time.sleep(0.01)
        session.touch()
        assert session.last_active > old_ts

    def test_timestamps_are_utc(self):
        session = Session(key=SessionKey("s4"), agent=Agent())
        assert session.created_at.tzinfo is not None
        assert session.last_active.tzinfo is not None


# ---------------------------------------------------------------------------
# History serialization / deserialization
# ---------------------------------------------------------------------------


def _make_simple_history() -> HistoryTree:
    """Build a HistoryTree with a user prompt and one assistant turn."""
    tree = HistoryTree()
    tree.append_user_prompt("What is 2+2?")
    tree.append_turn(
        AssistantMessage(
            content="The answer is 4.",
            tool_calls=[],
        ),
        [],
    )
    return tree


def _make_history_with_tool_calls() -> HistoryTree:
    """Build a history with tool calls and results."""
    tree = HistoryTree()
    tree.append_user_prompt("Read the file foo.py")
    tree.append_turn(
        AssistantMessage(
            content="I'll read that file.",
            tool_calls=[
                ToolCall(
                    call_id="tc_1",
                    name="read_file",
                    arguments='{"path": "foo.py"}',
                ),
            ],
        ),
        [
            ToolResultMessage(
                call_id="tc_1",
                content="def hello(): pass",
            ),
        ],
    )
    tree.append_turn(
        AssistantMessage(content="The file contains a hello function."),
        [],
    )
    return tree


def _make_history_with_collapsed_nodes() -> HistoryTree:
    """Build a history with some nodes in collapsed state."""
    tree = _make_history_with_tool_calls()
    # Collapse the first turn
    turn = tree.nodes[1]
    assert isinstance(turn, TurnNode)
    turn.collapse_state = CollapseState.COLLAPSED
    # Detail-collapse the tool call
    if turn.tool_call_nodes:
        turn.tool_call_nodes[0].detail_collapsed = True
    return tree


def _make_history_with_error_result() -> HistoryTree:
    """Build a history with an error tool result."""
    tree = HistoryTree()
    tree.append_user_prompt("Delete temp files")
    tree.append_turn(
        AssistantMessage(
            content="Deleting...",
            tool_calls=[
                ToolCall(
                    call_id="tc_err",
                    name="delete_file",
                    arguments='{"path": "/tmp/nope"}',
                ),
            ],
        ),
        [
            ToolResultMessage(
                call_id="tc_err",
                content="Permission denied",
                is_error=True,
            ),
        ],
    )
    return tree


class TestHistorySerialization:
    def test_empty_history_roundtrip(self):
        tree = HistoryTree()
        data = serialize_history(tree)
        assert data == []
        restored = deserialize_history(data)
        assert restored.nodes == []

    def test_simple_history_roundtrip(self):
        tree = _make_simple_history()
        data = serialize_history(tree)
        restored = deserialize_history(data)

        assert len(restored.nodes) == 2
        assert isinstance(restored.nodes[0], UserPromptNode)
        assert restored.nodes[0].message.content == "What is 2+2?"
        assert isinstance(restored.nodes[1], TurnNode)
        assert restored.nodes[1].assistant_content == "The answer is 4."

    def test_tool_call_roundtrip(self):
        tree = _make_history_with_tool_calls()
        data = serialize_history(tree)
        restored = deserialize_history(data)

        assert len(restored.nodes) == 3
        turn = restored.nodes[1]
        assert isinstance(turn, TurnNode)
        assert len(turn.tool_call_nodes) == 1

        tcn = turn.tool_call_nodes[0]
        assert tcn.tool_call.name == "read_file"
        assert tcn.tool_call.call_id == "tc_1"
        assert json.loads(tcn.tool_call.arguments) == {"path": "foo.py"}
        assert tcn.result.content == "def hello(): pass"
        assert not tcn.result.is_error

    def test_collapse_state_preserved(self):
        tree = _make_history_with_collapsed_nodes()
        data = serialize_history(tree)
        restored = deserialize_history(data)

        turn = restored.nodes[1]
        assert isinstance(turn, TurnNode)
        assert turn.collapse_state == CollapseState.COLLAPSED
        assert turn.tool_call_nodes[0].detail_collapsed is True

    def test_error_result_roundtrip(self):
        tree = _make_history_with_error_result()
        data = serialize_history(tree)
        restored = deserialize_history(data)

        turn = restored.nodes[1]
        assert isinstance(turn, TurnNode)
        tcn = turn.tool_call_nodes[0]
        assert tcn.result.is_error is True
        assert tcn.result.content == "Permission denied"

    def test_intrinsic_salience_preserved(self):
        tree = HistoryTree()
        tree.append_user_prompt("hi", intrinsic_salience=0.5)
        tree.append_turn(
            AssistantMessage(content="hello"),
            [],
            intrinsic_salience=0.7,
        )

        data = serialize_history(tree)
        restored = deserialize_history(data)

        assert restored.nodes[0].intrinsic_salience == 0.5
        assert restored.nodes[1].intrinsic_salience == 0.7

    def test_serialized_format_is_readable_json(self):
        tree = _make_simple_history()
        data = serialize_history(tree)
        json_str = json.dumps(data, indent=2)
        reparsed = json.loads(json_str)
        assert reparsed == data

    def test_rendered_messages_match_after_roundtrip(self):
        tree = _make_history_with_tool_calls()
        original_messages = tree.render()
        data = serialize_history(tree)
        restored = deserialize_history(data)
        restored_messages = restored.render()

        assert len(original_messages) == len(restored_messages)
        for orig, rest in zip(original_messages, restored_messages):
            assert type(orig) is type(rest)
            assert orig.role == rest.role
            if hasattr(orig, "content"):
                assert orig.content == rest.content

    def test_multiple_tool_calls_in_single_turn(self):
        tree = HistoryTree()
        tree.append_user_prompt("Do two things")
        tree.append_turn(
            AssistantMessage(
                content="I'll do both.",
                tool_calls=[
                    ToolCall(call_id="c1", name="tool_a", arguments='{"x": 1}'),
                    ToolCall(call_id="c2", name="tool_b", arguments='{"y": 2}'),
                ],
            ),
            [
                ToolResultMessage(call_id="c1", content="result_a"),
                ToolResultMessage(call_id="c2", content="result_b"),
            ],
        )

        data = serialize_history(tree)
        restored = deserialize_history(data)

        turn = restored.nodes[1]
        assert isinstance(turn, TurnNode)
        assert len(turn.tool_call_nodes) == 2
        assert turn.tool_call_nodes[0].tool_call.name == "tool_a"
        assert turn.tool_call_nodes[1].tool_call.name == "tool_b"

    def test_user_prompt_collapse_state_roundtrip(self):
        tree = HistoryTree()
        node = tree.append_user_prompt("x" * 3000)
        node.collapse_state = CollapseState.COLLAPSED

        data = serialize_history(tree)
        restored = deserialize_history(data)
        assert restored.nodes[0].collapse_state == CollapseState.COLLAPSED


# ---------------------------------------------------------------------------
# JsonSessionSerializer
# ---------------------------------------------------------------------------


class TestJsonSessionSerializer:
    def test_save_and_load_roundtrip(self, tmp_path: Path):
        agent = Agent(name="test-agent", metadata={"role": "coder"})
        agent._history = _make_history_with_tool_calls()

        session = Session(
            key=SessionKey("test-key"),
            agent=agent,
            metadata={"project_id": 123},
        )

        serializer = JsonSessionSerializer()
        serializer.save(session, tmp_path)

        assert (tmp_path / "session.json").exists()
        assert (tmp_path / "history.json").exists()

        restored = serializer.load(tmp_path)
        assert restored.key == SessionKey("test-key")
        assert isinstance(restored.key, SessionKey)
        assert restored.agent.name == "test-agent"
        assert restored.agent.metadata == {"role": "coder"}
        assert restored.metadata == {"project_id": 123}
        assert len(restored.agent._history.nodes) == 3

    def test_timestamps_preserved(self, tmp_path: Path):
        ts = datetime(2025, 6, 15, 12, 30, 0, tzinfo=timezone.utc)
        session = Session(
            key=SessionKey("ts-test"),
            agent=Agent(),
            created_at=ts,
            last_active=ts,
        )

        serializer = JsonSessionSerializer()
        serializer.save(session, tmp_path)
        restored = serializer.load(tmp_path)

        assert restored.created_at == ts
        assert restored.last_active == ts

    def test_session_json_is_human_readable(self, tmp_path: Path):
        session = Session(key=SessionKey("readable"), agent=Agent(name="bot"))
        serializer = JsonSessionSerializer()
        serializer.save(session, tmp_path)

        content = (tmp_path / "session.json").read_text(encoding="utf-8")
        assert "\n" in content
        parsed = json.loads(content)
        assert parsed["key"] == "readable"
        assert parsed["agent_name"] == "bot"

    def test_history_json_is_human_readable(self, tmp_path: Path):
        agent = Agent()
        agent._history = _make_simple_history()
        session = Session(key=SessionKey("hist"), agent=agent)

        serializer = JsonSessionSerializer()
        serializer.save(session, tmp_path)

        content = (tmp_path / "history.json").read_text(encoding="utf-8")
        assert "\n" in content
        parsed = json.loads(content)
        assert isinstance(parsed, list)
        assert len(parsed) == 2

    def test_empty_history(self, tmp_path: Path):
        session = Session(key=SessionKey("empty"), agent=Agent())
        serializer = JsonSessionSerializer()
        serializer.save(session, tmp_path)
        restored = serializer.load(tmp_path)
        assert len(restored.agent._history.nodes) == 0

    def test_agent_class_name_stored(self, tmp_path: Path):
        class CustomAgent(Agent):
            pass

        session = Session(
            key=SessionKey("cls"),
            agent=CustomAgent(name="custom"),
        )

        serializer = JsonSessionSerializer()
        serializer.save(session, tmp_path)

        content = json.loads(
            (tmp_path / "session.json").read_text(encoding="utf-8")
        )
        assert content["agent_class"] == "CustomAgent"

    def test_agent_class_resolved_on_load(self, tmp_path: Path):
        class ResolvableAgent(Agent):
            pass

        session = Session(
            key=SessionKey("resolve"),
            agent=ResolvableAgent(name="r"),
        )
        serializer = JsonSessionSerializer()
        serializer.save(session, tmp_path)

        restored = serializer.load(tmp_path)
        assert type(restored.agent).__name__ == "ResolvableAgent"
        assert isinstance(restored.agent, ResolvableAgent)

    def test_unknown_agent_class_falls_back_to_base(self, tmp_path: Path):
        session = Session(key=SessionKey("fallback"), agent=Agent(name="fb"))
        serializer = JsonSessionSerializer()
        serializer.save(session, tmp_path)

        session_path = tmp_path / "session.json"
        data = json.loads(session_path.read_text(encoding="utf-8"))
        data["agent_class"] = "NoSuchAgent"
        session_path.write_text(
            json.dumps(data, indent=2), encoding="utf-8",
        )

        restored = serializer.load(tmp_path)
        assert type(restored.agent) is Agent
        assert restored.agent.name == "fb"

    def test_missing_history_file(self, tmp_path: Path):
        session = Session(key=SessionKey("no-hist"), agent=Agent())
        serializer = JsonSessionSerializer()
        serializer.save(session, tmp_path)
        (tmp_path / "history.json").unlink()

        restored = serializer.load(tmp_path)
        assert len(restored.agent._history.nodes) == 0


# ---------------------------------------------------------------------------
# SessionStore
# ---------------------------------------------------------------------------


class TestSessionStore:
    def test_save_and_load(self, tmp_path: Path):
        store = SessionStore(tmp_path)
        agent = Agent(name="stored")
        agent._history = _make_simple_history()
        session = Session(key=SessionKey("k1"), agent=agent)

        store.save(session)
        assert store.exists("k1")

        loaded = store.load("k1")
        assert loaded.key == "k1"
        assert loaded.agent.name == "stored"
        assert len(loaded.agent._history.nodes) == 2

    def test_exists_returns_false_for_missing(self, tmp_path: Path):
        store = SessionStore(tmp_path)
        assert not store.exists("nonexistent")

    def test_load_missing_raises_key_error(self, tmp_path: Path):
        store = SessionStore(tmp_path)
        with pytest.raises(KeyError, match="nonexistent"):
            store.load("nonexistent")

    def test_list_keys_empty(self, tmp_path: Path):
        store = SessionStore(tmp_path)
        assert store.list_keys() == []

    def test_list_keys_returns_sorted(self, tmp_path: Path):
        store = SessionStore(tmp_path)
        for name in ["charlie", "alice", "bob"]:
            store.save(Session(key=SessionKey(name), agent=Agent()))

        keys = store.list_keys()
        assert keys == [
            SessionKey("alice"),
            SessionKey("bob"),
            SessionKey("charlie"),
        ]

    def test_delete_removes_session(self, tmp_path: Path):
        store = SessionStore(tmp_path)
        store.save(Session(key=SessionKey("del-me"), agent=Agent()))
        assert store.exists("del-me")

        store.delete("del-me")
        assert not store.exists("del-me")

    def test_delete_nonexistent_is_noop(self, tmp_path: Path):
        store = SessionStore(tmp_path)
        store.delete("ghost")

    def test_overwrite_existing_session(self, tmp_path: Path):
        store = SessionStore(tmp_path)
        store.save(
            Session(key=SessionKey("ow"), agent=Agent(name="v1")),
        )
        store.save(
            Session(key=SessionKey("ow"), agent=Agent(name="v2")),
        )
        loaded = store.load("ow")
        assert loaded.agent.name == "v2"

    def test_list_keys_on_nonexistent_root(self, tmp_path: Path):
        store = SessionStore(tmp_path / "no-such-dir")
        assert store.list_keys() == []

    def test_session_key_type_preserved(self, tmp_path: Path):
        store = SessionStore(tmp_path)
        store.save(Session(key=SessionKey("typed"), agent=Agent()))
        keys = store.list_keys()
        assert all(isinstance(k, SessionKey) for k in keys)

    def test_custom_serializer(self, tmp_path: Path):
        call_log: list[str] = []

        class TrackingSerializer:
            def save(self, session: Session, directory: Path) -> None:
                call_log.append(f"save:{session.key}")
                (directory / "marker.txt").write_text("saved", encoding="utf-8")

            def load(self, directory: Path) -> Session:
                call_log.append("load")
                return Session(key=SessionKey("custom"), agent=Agent())

        store = SessionStore(tmp_path, serializer=TrackingSerializer())
        store.save(Session(key=SessionKey("cs"), agent=Agent()))
        store.load("cs")

        assert "save:cs" in call_log
        assert "load" in call_log


# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------


class TestRuntime:
    def _make_runtime(self, tmp_path: Path, **kwargs: Any) -> Runtime:
        return Runtime(
            provider=MockProvider(),
            workspace_root=tmp_path,
            **kwargs,
        )

    def test_creation_with_defaults(self, tmp_path: Path):
        rt = self._make_runtime(tmp_path)
        assert rt.provider is not None
        assert rt.workspace_root == tmp_path
        assert rt.sessions is not None

    def test_default_session_store_location(self, tmp_path: Path):
        rt = self._make_runtime(tmp_path)
        expected = tmp_path / ".thorn" / "sessions"
        assert rt.sessions.root == expected

    def test_custom_session_store(self, tmp_path: Path):
        custom_store = SessionStore(tmp_path / "custom-sessions")
        rt = self._make_runtime(tmp_path, session_store=custom_store)
        assert rt.sessions is custom_store

    def test_create_context(self, tmp_path: Path):
        rt = self._make_runtime(
            tmp_path,
            workspace_instructions="Be helpful.",
        )
        ctx = rt.create_context(system_prompts=["Extra prompt."])

        assert isinstance(ctx, ExecutionContext)
        assert ctx.provider is rt.provider
        assert ctx.workspace_root == tmp_path
        assert ctx.workspace_instructions == "Be helpful."
        assert "Extra prompt." in ctx.system_prompts

    def test_create_context_inherits_ask_user_handler(self, tmp_path: Path):
        async def handler(q: str) -> str:
            return "answer"

        rt = self._make_runtime(tmp_path, ask_user_handler=handler)
        ctx = rt.create_context()
        assert ctx.ask_user_handler is handler

    def test_create_session_with_auto_key(self, tmp_path: Path):
        rt = self._make_runtime(tmp_path)
        session = rt.create_session()

        assert isinstance(session.key, SessionKey)
        assert len(session.key) > 0
        assert session.agent.name == str(session.key)

    def test_create_session_with_explicit_key(self, tmp_path: Path):
        rt = self._make_runtime(tmp_path)
        session = rt.create_session("my-key")

        assert session.key == SessionKey("my-key")

    def test_create_session_with_str_key_becomes_session_key(self, tmp_path: Path):
        rt = self._make_runtime(tmp_path)
        session = rt.create_session("str-key")
        assert isinstance(session.key, SessionKey)

    def test_create_session_with_custom_agent(self, tmp_path: Path):
        rt = self._make_runtime(tmp_path)
        agent = Agent(name="custom")
        session = rt.create_session("s1", agent=agent)

        assert session.agent is agent

    def test_create_session_with_metadata(self, tmp_path: Path):
        rt = self._make_runtime(tmp_path)
        session = rt.create_session("s1", metadata={"issue": 42})
        assert session.metadata == {"issue": 42}

    def test_save_and_get_or_create_session(self, tmp_path: Path):
        rt = self._make_runtime(tmp_path)
        session = rt.create_session("persistent")
        session.agent._history = _make_simple_history()
        rt.save_session(session)

        retrieved = rt.get_or_create_session("persistent")
        assert retrieved.key == "persistent"
        assert len(retrieved.agent._history.nodes) == 2

    def test_get_or_create_creates_when_missing(self, tmp_path: Path):
        rt = self._make_runtime(tmp_path)
        session = rt.get_or_create_session("new-key")
        assert session.key == "new-key"
        assert len(session.agent._history.nodes) == 0

    def test_save_session_updates_last_active(self, tmp_path: Path):
        rt = self._make_runtime(tmp_path)
        session = rt.create_session("ts-test")
        old_ts = session.last_active
        import time
        time.sleep(0.01)
        rt.save_session(session)
        assert session.last_active > old_ts

    def test_create_context_system_prompts_are_independent(self, tmp_path: Path):
        rt = self._make_runtime(tmp_path)
        ctx1 = rt.create_context(system_prompts=["a"])
        ctx2 = rt.create_context(system_prompts=["b"])
        ctx1.system_prompts.append("c")
        assert "c" not in ctx2.system_prompts


# ---------------------------------------------------------------------------
# Top-level re-exports
# ---------------------------------------------------------------------------


class TestReExports:
    def test_runtime_importable_from_thorn(self):
        from thorn import Runtime, Session, SessionKey
        assert Runtime is not None
        assert Session is not None
        assert SessionKey is not None

    def test_runtime_importable_from_thorn_runtime(self):
        from thorn.runtime import (
            Runtime,
            Session,
            SessionKey,
            SessionStore,
            SessionSerializer,
            JsonSessionSerializer,
        )
        assert all(cls is not None for cls in [
            Runtime, Session, SessionKey, SessionStore,
            SessionSerializer, JsonSessionSerializer,
        ])
