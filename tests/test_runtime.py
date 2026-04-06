"""Tests for thorn.runtime -- Agent persistence, SessionStore, serialization, and Runtime."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from thorn.core._agent import Agent
from thorn.core._context import ExecutionContext, get_context
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
    SessionKey,
    SessionStore,
    deserialize_history,
    serialize_history,
)


# ---------------------------------------------------------------------------
# Agent persistence fields (key, created_at, last_active, touch)
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

    def test_default_key_is_none(self):
        agent = Agent()
        assert agent.key is None

    def test_explicit_key(self):
        key = SessionKey("gitlab:issue:42")
        agent = Agent(key=key)
        assert agent.key == key
        assert isinstance(agent.key, SessionKey)

    def test_default_created_at_is_none(self):
        agent = Agent()
        assert agent.created_at is None

    def test_explicit_created_at(self):
        ts = datetime(2025, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
        agent = Agent(created_at=ts)
        assert agent.created_at == ts

    def test_default_last_active_is_none(self):
        agent = Agent()
        assert agent.last_active is None

    def test_explicit_last_active(self):
        ts = datetime(2025, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
        agent = Agent(last_active=ts)
        assert agent.last_active == ts

    def test_touch_sets_last_active(self):
        agent = Agent()
        assert agent.last_active is None
        agent.touch()
        assert agent.last_active is not None
        assert isinstance(agent.last_active, datetime)

    def test_touch_updates_last_active(self):
        ts = datetime(2020, 1, 1, tzinfo=timezone.utc)
        agent = Agent(last_active=ts)
        agent.touch()
        assert agent.last_active > ts

    def test_all_persistence_fields_together(self):
        key = SessionKey("test-key")
        ts = datetime(2025, 1, 1, tzinfo=timezone.utc)
        agent = Agent(
            name="reviewer",
            key=key,
            created_at=ts,
            last_active=ts,
            metadata={"role": "code-review"},
        )
        assert agent.name == "reviewer"
        assert agent.key == key
        assert agent.created_at == ts
        assert agent.last_active == ts
        assert agent.metadata == {"role": "code-review"}


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
    turn = tree.nodes[1]
    assert isinstance(turn, TurnNode)
    turn.collapse_state = CollapseState.COLLAPSED
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
# JsonSessionSerializer (now saves/loads Agent directly)
# ---------------------------------------------------------------------------


class TestJsonSessionSerializer:
    def test_save_and_load_roundtrip(self, tmp_path: Path):
        agent = Agent(
            name="test-agent",
            metadata={"role": "coder"},
            key=SessionKey("test-key"),
            created_at=datetime(2025, 6, 15, 12, 0, 0, tzinfo=timezone.utc),
            last_active=datetime(2025, 6, 15, 13, 0, 0, tzinfo=timezone.utc),
        )
        agent._history = _make_history_with_tool_calls()

        serializer = JsonSessionSerializer()
        serializer.save(agent, tmp_path)

        assert (tmp_path / "session.json").exists()
        assert (tmp_path / "history.json").exists()

        restored = serializer.load(tmp_path)
        assert restored.key == SessionKey("test-key")
        assert isinstance(restored.key, SessionKey)
        assert restored.name == "test-agent"
        assert restored.metadata == {"role": "coder"}
        assert len(restored._history.nodes) == 3

    def test_timestamps_preserved(self, tmp_path: Path):
        ts = datetime(2025, 6, 15, 12, 30, 0, tzinfo=timezone.utc)
        agent = Agent(
            key=SessionKey("ts-test"),
            created_at=ts,
            last_active=ts,
        )

        serializer = JsonSessionSerializer()
        serializer.save(agent, tmp_path)
        restored = serializer.load(tmp_path)

        assert restored.created_at == ts
        assert restored.last_active == ts

    def test_session_json_is_human_readable(self, tmp_path: Path):
        agent = Agent(name="bot", key=SessionKey("readable"))
        serializer = JsonSessionSerializer()
        serializer.save(agent, tmp_path)

        content = (tmp_path / "session.json").read_text(encoding="utf-8")
        assert "\n" in content
        parsed = json.loads(content)
        assert parsed["key"] == "readable"
        assert parsed["agent_name"] == "bot"

    def test_history_json_is_human_readable(self, tmp_path: Path):
        agent = Agent(key=SessionKey("hist"))
        agent._history = _make_simple_history()

        serializer = JsonSessionSerializer()
        serializer.save(agent, tmp_path)

        content = (tmp_path / "history.json").read_text(encoding="utf-8")
        assert "\n" in content
        parsed = json.loads(content)
        assert isinstance(parsed, list)
        assert len(parsed) == 2

    def test_empty_history(self, tmp_path: Path):
        agent = Agent(key=SessionKey("empty"))
        serializer = JsonSessionSerializer()
        serializer.save(agent, tmp_path)
        restored = serializer.load(tmp_path)
        assert len(restored._history.nodes) == 0

    def test_agent_class_name_stored(self, tmp_path: Path):
        class CustomAgent(Agent):
            pass

        agent = CustomAgent(name="custom", key=SessionKey("cls"))

        serializer = JsonSessionSerializer()
        serializer.save(agent, tmp_path)

        content = json.loads(
            (tmp_path / "session.json").read_text(encoding="utf-8")
        )
        assert content["agent_class"] == "CustomAgent"

    def test_agent_class_resolved_on_load(self, tmp_path: Path):
        class ResolvableAgent(Agent):
            pass

        agent = ResolvableAgent(name="r", key=SessionKey("resolve"))
        serializer = JsonSessionSerializer()
        serializer.save(agent, tmp_path)

        restored = serializer.load(tmp_path)
        assert type(restored).__name__ == "ResolvableAgent"
        assert isinstance(restored, ResolvableAgent)

    def test_unknown_agent_class_falls_back_to_base(self, tmp_path: Path):
        agent = Agent(name="fb", key=SessionKey("fallback"))
        serializer = JsonSessionSerializer()
        serializer.save(agent, tmp_path)

        session_path = tmp_path / "session.json"
        data = json.loads(session_path.read_text(encoding="utf-8"))
        data["agent_class"] = "NoSuchAgent"
        session_path.write_text(
            json.dumps(data, indent=2), encoding="utf-8",
        )

        restored = serializer.load(tmp_path)
        assert type(restored) is Agent
        assert restored.name == "fb"

    def test_missing_history_file(self, tmp_path: Path):
        agent = Agent(key=SessionKey("no-hist"))
        serializer = JsonSessionSerializer()
        serializer.save(agent, tmp_path)
        (tmp_path / "history.json").unlink()

        restored = serializer.load(tmp_path)
        assert len(restored._history.nodes) == 0

    def test_none_timestamps_roundtrip(self, tmp_path: Path):
        agent = Agent(key=SessionKey("no-ts"))
        assert agent.created_at is None
        assert agent.last_active is None

        serializer = JsonSessionSerializer()
        serializer.save(agent, tmp_path)
        restored = serializer.load(tmp_path)
        assert restored.created_at is None
        assert restored.last_active is None

    def test_none_key_roundtrip(self, tmp_path: Path):
        agent = Agent(name="keyless")
        serializer = JsonSessionSerializer()
        serializer.save(agent, tmp_path)
        restored = serializer.load(tmp_path)
        assert restored.key is None
        assert restored.name == "keyless"


# ---------------------------------------------------------------------------
# SessionStore
# ---------------------------------------------------------------------------


class TestSessionStore:
    def test_save_and_load(self, tmp_path: Path):
        store = SessionStore(tmp_path)
        agent = Agent(name="stored", key=SessionKey("k1"))
        agent._history = _make_simple_history()

        store.save(agent)
        assert store.exists("k1")

        loaded = store.load("k1")
        assert loaded.key == "k1"
        assert loaded.name == "stored"
        assert len(loaded._history.nodes) == 2

    def test_exists_returns_false_for_missing(self, tmp_path: Path):
        store = SessionStore(tmp_path)
        assert not store.exists("nonexistent")

    def test_load_missing_raises_key_error(self, tmp_path: Path):
        store = SessionStore(tmp_path)
        with pytest.raises(KeyError, match="nonexistent"):
            store.load("nonexistent")

    def test_save_without_key_raises(self, tmp_path: Path):
        store = SessionStore(tmp_path)
        agent = Agent(name="no-key")
        with pytest.raises(ValueError, match="without a key"):
            store.save(agent)

    def test_list_keys_empty(self, tmp_path: Path):
        store = SessionStore(tmp_path)
        assert store.list_keys() == []

    def test_list_keys_returns_sorted(self, tmp_path: Path):
        store = SessionStore(tmp_path)
        for name in ["charlie", "alice", "bob"]:
            store.save(Agent(key=SessionKey(name)))

        keys = store.list_keys()
        assert keys == [
            SessionKey("alice"),
            SessionKey("bob"),
            SessionKey("charlie"),
        ]

    def test_delete_removes_session(self, tmp_path: Path):
        store = SessionStore(tmp_path)
        store.save(Agent(key=SessionKey("del-me")))
        assert store.exists("del-me")

        store.delete("del-me")
        assert not store.exists("del-me")

    def test_delete_nonexistent_is_noop(self, tmp_path: Path):
        store = SessionStore(tmp_path)
        store.delete("ghost")

    def test_overwrite_existing_session(self, tmp_path: Path):
        store = SessionStore(tmp_path)
        store.save(Agent(name="v1", key=SessionKey("ow")))
        store.save(Agent(name="v2", key=SessionKey("ow")))
        loaded = store.load("ow")
        assert loaded.name == "v2"

    def test_list_keys_on_nonexistent_root(self, tmp_path: Path):
        store = SessionStore(tmp_path / "no-such-dir")
        assert store.list_keys() == []

    def test_session_key_type_preserved(self, tmp_path: Path):
        store = SessionStore(tmp_path)
        store.save(Agent(key=SessionKey("typed")))
        keys = store.list_keys()
        assert all(isinstance(k, SessionKey) for k in keys)

    def test_custom_serializer(self, tmp_path: Path):
        call_log: list[str] = []

        class TrackingSerializer:
            def save(self, agent: Agent, directory: Path) -> None:
                call_log.append(f"save:{agent.key}")
                (directory / "marker.txt").write_text("saved", encoding="utf-8")

            def load(self, directory: Path) -> Agent:
                call_log.append("load")
                return Agent(key=SessionKey("custom"))

        store = SessionStore(tmp_path, serializer=TrackingSerializer())
        store.save(Agent(key=SessionKey("cs")))
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

    def test_create_agent_with_auto_key(self, tmp_path: Path):
        rt = self._make_runtime(tmp_path)
        agent = rt.create_agent()

        assert isinstance(agent.key, SessionKey)
        assert len(agent.key) > 0
        assert agent.name == str(agent.key)
        assert agent.created_at is not None
        assert agent.last_active is not None

    def test_create_agent_with_explicit_key(self, tmp_path: Path):
        rt = self._make_runtime(tmp_path)
        agent = rt.create_agent("my-key")

        assert agent.key == SessionKey("my-key")

    def test_create_agent_with_str_key_becomes_session_key(self, tmp_path: Path):
        rt = self._make_runtime(tmp_path)
        agent = rt.create_agent("str-key")
        assert isinstance(agent.key, SessionKey)

    def test_create_agent_with_custom_name(self, tmp_path: Path):
        rt = self._make_runtime(tmp_path)
        agent = rt.create_agent("s1", name="reviewer")
        assert agent.name == "reviewer"
        assert agent.key == SessionKey("s1")

    def test_create_agent_with_metadata(self, tmp_path: Path):
        rt = self._make_runtime(tmp_path)
        agent = rt.create_agent("s1", metadata={"issue": 42})
        assert agent.metadata == {"issue": 42}

    def test_save_and_get_or_create_agent(self, tmp_path: Path):
        rt = self._make_runtime(tmp_path)
        agent = rt.create_agent("persistent")
        agent._history = _make_simple_history()
        rt.save_agent(agent)

        retrieved = rt.get_or_create_agent("persistent")
        assert retrieved.key == "persistent"
        assert len(retrieved._history.nodes) == 2

    def test_get_or_create_creates_when_missing(self, tmp_path: Path):
        rt = self._make_runtime(tmp_path)
        agent = rt.get_or_create_agent("new-key")
        assert agent.key == "new-key"
        assert len(agent._history.nodes) == 0

    def test_save_agent_updates_last_active(self, tmp_path: Path):
        rt = self._make_runtime(tmp_path)
        agent = rt.create_agent("ts-test")
        old_ts = agent.last_active
        import time
        time.sleep(0.01)
        rt.save_agent(agent)
        assert agent.last_active > old_ts

    def test_create_context_system_prompts_are_independent(self, tmp_path: Path):
        rt = self._make_runtime(tmp_path)
        ctx1 = rt.create_context(system_prompts=["a"])
        ctx2 = rt.create_context(system_prompts=["b"])
        ctx1.system_prompts.append("c")
        assert "c" not in ctx2.system_prompts


# ---------------------------------------------------------------------------
# Runtime as async context manager
# ---------------------------------------------------------------------------


class TestRuntimeContextManager:
    def _make_runtime(self, tmp_path: Path, **kwargs: Any) -> Runtime:
        return Runtime(
            provider=MockProvider(),
            workspace_root=tmp_path,
            **kwargs,
        )

    @pytest.mark.asyncio
    async def test_sets_ambient_context(self, tmp_path: Path):
        rt = self._make_runtime(tmp_path)
        async with rt:
            ctx = get_context()
            assert ctx is rt.context
            assert ctx.provider is rt.provider
            assert ctx.workspace_root == tmp_path

    @pytest.mark.asyncio
    async def test_clears_context_on_exit(self, tmp_path: Path):
        rt = self._make_runtime(tmp_path)
        async with rt:
            pass

        with pytest.raises(RuntimeError):
            rt.context

    @pytest.mark.asyncio
    async def test_context_property_outside_block_raises(self, tmp_path: Path):
        rt = self._make_runtime(tmp_path)
        with pytest.raises(RuntimeError, match="only available inside"):
            rt.context

    @pytest.mark.asyncio
    async def test_returns_self(self, tmp_path: Path):
        rt = self._make_runtime(tmp_path)
        async with rt as entered:
            assert entered is rt

    @pytest.mark.asyncio
    async def test_context_cleaned_up_on_exception(self, tmp_path: Path):
        rt = self._make_runtime(tmp_path)
        with pytest.raises(ValueError):
            async with rt:
                raise ValueError("boom")

        with pytest.raises(RuntimeError):
            rt.context


# ---------------------------------------------------------------------------
# Top-level re-exports
# ---------------------------------------------------------------------------


class TestReExports:
    def test_runtime_importable_from_thorn(self):
        from thorn import Runtime, SessionKey
        assert Runtime is not None
        assert SessionKey is not None

    def test_session_not_importable_from_thorn(self):
        import thorn
        assert not hasattr(thorn, "Session")

    def test_runtime_importable_from_thorn_runtime(self):
        from thorn.runtime import (
            Runtime,
            SessionKey,
            SessionStore,
            SessionSerializer,
            JsonSessionSerializer,
        )
        assert all(cls is not None for cls in [
            Runtime, SessionKey, SessionStore,
            SessionSerializer, JsonSessionSerializer,
        ])

    def test_session_not_importable_from_thorn_runtime(self):
        import thorn.runtime
        assert not hasattr(thorn.runtime, "Session")
