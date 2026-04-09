"""Tests for thorn.runtime -- Agent/Session persistence, SessionStore, serialization, and Runtime."""

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
from thorn.core._session import Session
from thorn.runtime import (
    AgentID,
    JsonSessionSerializer,
    Runtime,
    SessionKey,
    SessionStore,
    deserialize_history,
    serialize_history,
)


# ---------------------------------------------------------------------------
# Agent identity fields (id, workspace, name, metadata)
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

    def test_default_id_is_none(self):
        agent = Agent()
        assert agent.id is None

    def test_explicit_id(self):
        aid = AgentID("agent-42")
        agent = Agent(id=aid)
        assert agent.id == aid
        assert isinstance(agent.id, AgentID)

    def test_default_workspace_is_none(self):
        agent = Agent()
        assert agent._workspace is None

    def test_explicit_workspace(self, tmp_path: Path):
        agent = Agent(workspace=tmp_path)
        assert agent.workspace == tmp_path

    def test_all_identity_fields_together(self):
        aid = AgentID("test-id")
        agent = Agent(
            id=aid,
            name="reviewer",
            metadata={"role": "code-review"},
        )
        assert agent.id == aid
        assert agent.name == "reviewer"
        assert agent.metadata == {"role": "code-review"}


# ---------------------------------------------------------------------------
# Session fields (key, created_at, last_active, touch, metadata)
# ---------------------------------------------------------------------------


class TestSessionFields:
    def _make_agent(self) -> Agent:
        return Agent(id=AgentID("test-agent"), name="test")

    def test_default_key_is_none(self):
        session = Session(agent=self._make_agent())
        assert session.key is None

    def test_explicit_key(self):
        key = SessionKey("gitlab:issue:42")
        session = Session(agent=self._make_agent(), key=key)
        assert session.key == key
        assert isinstance(session.key, SessionKey)

    def test_default_created_at_is_none(self):
        session = Session(agent=self._make_agent())
        assert session.created_at is None

    def test_explicit_created_at(self):
        ts = datetime(2025, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
        session = Session(agent=self._make_agent(), created_at=ts)
        assert session.created_at == ts

    def test_default_last_active_is_none(self):
        session = Session(agent=self._make_agent())
        assert session.last_active is None

    def test_explicit_last_active(self):
        ts = datetime(2025, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
        session = Session(agent=self._make_agent(), last_active=ts)
        assert session.last_active == ts

    def test_touch_sets_last_active(self):
        session = Session(agent=self._make_agent())
        assert session.last_active is None
        session.touch()
        assert session.last_active is not None
        assert isinstance(session.last_active, datetime)

    def test_touch_updates_last_active(self):
        ts = datetime(2020, 1, 1, tzinfo=timezone.utc)
        session = Session(agent=self._make_agent(), last_active=ts)
        session.touch()
        assert session.last_active > ts

    def test_default_metadata_is_empty(self):
        session = Session(agent=self._make_agent())
        assert session.metadata == {}

    def test_explicit_metadata(self):
        session = Session(
            agent=self._make_agent(), metadata={"context": "issue-42"},
        )
        assert session.metadata == {"context": "issue-42"}

    def test_session_references_agent(self):
        agent = self._make_agent()
        session = Session(agent=agent)
        assert session.agent is agent

    def test_empty_history_on_creation(self):
        session = Session(agent=self._make_agent())
        assert len(session._history.nodes) == 0

    def test_all_session_fields_together(self):
        agent = self._make_agent()
        key = SessionKey("test-key")
        ts = datetime(2025, 1, 1, tzinfo=timezone.utc)
        session = Session(
            agent=agent,
            key=key,
            created_at=ts,
            last_active=ts,
            metadata={"role": "code-review"},
        )
        assert session.agent is agent
        assert session.key == key
        assert session.created_at == ts
        assert session.last_active == ts
        assert session.metadata == {"role": "code-review"}


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
# AgentID
# ---------------------------------------------------------------------------


class TestAgentID:
    def test_is_str_subclass(self):
        aid = AgentID("my-agent")
        assert isinstance(aid, str)
        assert isinstance(aid, AgentID)

    def test_str_operations(self):
        aid = AgentID("dev:agent:1")
        assert aid.startswith("dev:")
        assert "agent" in aid

    def test_equality_with_str(self):
        aid = AgentID("abc")
        assert aid == "abc"

    def test_as_dict_key(self):
        aid = AgentID("a1")
        d: dict[AgentID, int] = {aid: 1}
        assert d[AgentID("a1")] == 1


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


def _make_history_with_mixed_collapse_states() -> HistoryTree:
    """Build a multi-turn history exercising all collapse state combinations.

    NOTE: the compaction design is expected to change; this helper will
    need updating when that happens.

    Nodes:
      [0] UserPromptNode  -- expanded (short content)
      [1] TurnNode        -- expanded, with one detail-collapsed tool call
      [2] UserPromptNode  -- expanded (second user message)
      [3] TurnNode        -- fully collapsed (summary only)
      [4] UserPromptNode  -- expanded (third user message)
      [5] TurnNode        -- expanded, all tool calls expanded
    """
    tree = HistoryTree()

    tree.append_user_prompt("Read foo.py and bar.py")

    tree.append_turn(
        AssistantMessage(
            content="I'll read both files.",
            tool_calls=[
                ToolCall(call_id="tc_1", name="read_file", arguments='{"path": "foo.py"}'),
                ToolCall(call_id="tc_2", name="read_file", arguments='{"path": "bar.py"}'),
            ],
        ),
        [
            ToolResultMessage(call_id="tc_1", content="def hello(): pass"),
            ToolResultMessage(call_id="tc_2", content="class Bar:\n    x = 1"),
        ],
    )
    turn1 = tree.nodes[1]
    assert isinstance(turn1, TurnNode)
    turn1.tool_call_nodes[0].detail_collapsed = True

    tree.append_user_prompt("Now edit foo.py to add a docstring")

    tree.append_turn(
        AssistantMessage(
            content="I'll add a docstring.",
            tool_calls=[
                ToolCall(call_id="tc_3", name="edit_file",
                         arguments='{"path": "foo.py", "edits": [{"old": "def hello(): pass", "new": "def hello():\\n    \\"Greet.\\"\\n    pass"}]}'),
            ],
        ),
        [
            ToolResultMessage(call_id="tc_3", content="Applied 1 edit(s) to foo.py"),
        ],
    )
    turn2 = tree.nodes[3]
    assert isinstance(turn2, TurnNode)
    turn2.collapse_state = CollapseState.COLLAPSED

    tree.append_user_prompt("Looks good, now run the tests")

    tree.append_turn(
        AssistantMessage(
            content="Running tests now.",
            tool_calls=[
                ToolCall(call_id="tc_4", name="run_shell", arguments='{"command": "pytest"}'),
            ],
        ),
        [
            ToolResultMessage(call_id="tc_4", content="3 passed"),
        ],
    )

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
# JsonSessionSerializer -- agent identity (save_agent / load_agent)
# ---------------------------------------------------------------------------


class TestJsonSessionSerializerAgent:
    def test_save_and_load_agent_roundtrip(self, tmp_path: Path):
        agent = Agent(
            id=AgentID("test-agent"),
            name="bot",
            metadata={"role": "coder"},
        )

        serializer = JsonSessionSerializer()
        agent_path = tmp_path / "agent.json"
        serializer.save_agent(agent, agent_path)

        assert agent_path.exists()

        restored = serializer.load_agent(agent_path)
        assert restored.id == AgentID("test-agent")
        assert isinstance(restored.id, AgentID)
        assert restored.name == "bot"
        assert restored.metadata == {"role": "coder"}

    def test_agent_json_is_human_readable(self, tmp_path: Path):
        agent = Agent(id=AgentID("readable"), name="bot")
        serializer = JsonSessionSerializer()
        agent_path = tmp_path / "agent.json"
        serializer.save_agent(agent, agent_path)

        content = agent_path.read_text(encoding="utf-8")
        assert "\n" in content
        parsed = json.loads(content)
        assert parsed["id"] == "readable"
        assert parsed["name"] == "bot"

    def test_agent_class_name_stored(self, tmp_path: Path):
        class CustomAgent(Agent):
            pass

        agent = CustomAgent(id=AgentID("cls"), name="custom")
        serializer = JsonSessionSerializer()
        agent_path = tmp_path / "agent.json"
        serializer.save_agent(agent, agent_path)

        content = json.loads(agent_path.read_text(encoding="utf-8"))
        assert content["agent_class"] == "CustomAgent"

    def test_agent_class_resolved_on_load(self, tmp_path: Path):
        class ResolvableAgent(Agent):
            pass

        agent = ResolvableAgent(id=AgentID("resolve"), name="r")
        serializer = JsonSessionSerializer()
        agent_path = tmp_path / "agent.json"
        serializer.save_agent(agent, agent_path)

        restored = serializer.load_agent(agent_path)
        assert type(restored).__name__ == "ResolvableAgent"
        assert isinstance(restored, ResolvableAgent)

    def test_unknown_agent_class_falls_back_to_base(self, tmp_path: Path):
        agent = Agent(id=AgentID("fallback"), name="fb")
        serializer = JsonSessionSerializer()
        agent_path = tmp_path / "agent.json"
        serializer.save_agent(agent, agent_path)

        data = json.loads(agent_path.read_text(encoding="utf-8"))
        data["agent_class"] = "NoSuchAgent"
        agent_path.write_text(
            json.dumps(data, indent=2), encoding="utf-8",
        )

        restored = serializer.load_agent(agent_path)
        assert type(restored) is Agent
        assert restored.name == "fb"

    def test_none_id_roundtrip(self, tmp_path: Path):
        agent = Agent(name="no-id")
        serializer = JsonSessionSerializer()
        agent_path = tmp_path / "agent.json"
        serializer.save_agent(agent, agent_path)
        restored = serializer.load_agent(agent_path)
        assert restored.id is None
        assert restored.name == "no-id"


# ---------------------------------------------------------------------------
# JsonSessionSerializer -- session data (save_session / load_session)
# ---------------------------------------------------------------------------


class TestJsonSessionSerializerSession:
    def _make_agent(self) -> Agent:
        return Agent(id=AgentID("test-agent"), name="test")

    def test_save_and_load_session_roundtrip(self, tmp_path: Path):
        agent = self._make_agent()
        session = Session(
            agent=agent,
            key=SessionKey("test-key"),
            created_at=datetime(2025, 6, 15, 12, 0, 0, tzinfo=timezone.utc),
            last_active=datetime(2025, 6, 15, 13, 0, 0, tzinfo=timezone.utc),
            metadata={"context": "issue-42"},
        )
        session._history = _make_history_with_tool_calls()

        serializer = JsonSessionSerializer()
        session_dir = tmp_path / "session_dir"
        serializer.save_session(session, session_dir)

        assert (session_dir / "session.json").exists()
        assert (session_dir / "history.json").exists()

        restored = serializer.load_session(session_dir, agent)
        assert restored.key == SessionKey("test-key")
        assert isinstance(restored.key, SessionKey)
        assert restored.agent is agent
        assert restored.metadata == {"context": "issue-42"}
        assert len(restored._history.nodes) == 3

    def test_timestamps_preserved(self, tmp_path: Path):
        agent = self._make_agent()
        ts = datetime(2025, 6, 15, 12, 30, 0, tzinfo=timezone.utc)
        session = Session(
            agent=agent,
            key=SessionKey("ts-test"),
            created_at=ts,
            last_active=ts,
        )

        serializer = JsonSessionSerializer()
        session_dir = tmp_path / "session_dir"
        serializer.save_session(session, session_dir)
        restored = serializer.load_session(session_dir, agent)

        assert restored.created_at == ts
        assert restored.last_active == ts

    def test_session_json_is_human_readable(self, tmp_path: Path):
        agent = self._make_agent()
        session = Session(agent=agent, key=SessionKey("readable"))
        serializer = JsonSessionSerializer()
        session_dir = tmp_path / "session_dir"
        serializer.save_session(session, session_dir)

        content = (session_dir / "session.json").read_text(encoding="utf-8")
        assert "\n" in content
        parsed = json.loads(content)
        assert parsed["key"] == "readable"

    def test_history_json_is_human_readable(self, tmp_path: Path):
        agent = self._make_agent()
        session = Session(agent=agent, key=SessionKey("hist"))
        session._history = _make_simple_history()

        serializer = JsonSessionSerializer()
        session_dir = tmp_path / "session_dir"
        serializer.save_session(session, session_dir)

        content = (session_dir / "history.json").read_text(encoding="utf-8")
        assert "\n" in content
        parsed = json.loads(content)
        assert isinstance(parsed, list)
        assert len(parsed) == 2

    def test_empty_history(self, tmp_path: Path):
        agent = self._make_agent()
        session = Session(agent=agent, key=SessionKey("empty"))
        serializer = JsonSessionSerializer()
        session_dir = tmp_path / "session_dir"
        serializer.save_session(session, session_dir)
        restored = serializer.load_session(session_dir, agent)
        assert len(restored._history.nodes) == 0

    def test_missing_history_file(self, tmp_path: Path):
        agent = self._make_agent()
        session = Session(agent=agent, key=SessionKey("no-hist"))
        serializer = JsonSessionSerializer()
        session_dir = tmp_path / "session_dir"
        serializer.save_session(session, session_dir)
        (session_dir / "history.json").unlink()

        restored = serializer.load_session(session_dir, agent)
        assert len(restored._history.nodes) == 0

    def test_none_timestamps_roundtrip(self, tmp_path: Path):
        agent = self._make_agent()
        session = Session(agent=agent, key=SessionKey("no-ts"))
        assert session.created_at is None
        assert session.last_active is None

        serializer = JsonSessionSerializer()
        session_dir = tmp_path / "session_dir"
        serializer.save_session(session, session_dir)
        restored = serializer.load_session(session_dir, agent)
        assert restored.created_at is None
        assert restored.last_active is None

    def test_none_key_roundtrip(self, tmp_path: Path):
        agent = self._make_agent()
        session = Session(agent=agent)
        serializer = JsonSessionSerializer()
        session_dir = tmp_path / "session_dir"
        serializer.save_session(session, session_dir)
        restored = serializer.load_session(session_dir, agent)
        assert restored.key is None

    def test_compacted_history_render_survives_roundtrip(self, tmp_path: Path):
        """Save a session with mixed collapse states, reload it, and
        verify that rendered messages match exactly.

        NOTE: this test encodes current compaction/collapse rendering
        behavior.  The compaction design is expected to change; update
        this test when that happens.
        """
        agent = self._make_agent()
        session = Session(agent=agent, key=SessionKey("compacted"))
        session._history = _make_history_with_mixed_collapse_states()

        rendered_before = session._history.render()

        serializer = JsonSessionSerializer()
        session_dir = tmp_path / "session_dir"
        serializer.save_session(session, session_dir)
        restored = serializer.load_session(session_dir, agent)

        rendered_after = restored._history.render()

        assert len(rendered_after) == len(rendered_before)
        for orig, loaded in zip(rendered_before, rendered_after):
            assert type(orig) is type(loaded)
            assert orig.role == loaded.role

            if isinstance(orig, UserMessage):
                assert orig.content == loaded.content
            elif isinstance(orig, AssistantMessage):
                assert orig.content == loaded.content
                assert len(orig.tool_calls) == len(loaded.tool_calls)
                for tc_orig, tc_loaded in zip(orig.tool_calls, loaded.tool_calls):
                    assert tc_orig.call_id == tc_loaded.call_id
                    assert tc_orig.name == tc_loaded.name
                    assert tc_orig.arguments == tc_loaded.arguments
            elif isinstance(orig, ToolResultMessage):
                assert orig.call_id == loaded.call_id
                assert orig.content == loaded.content
                assert orig.is_error == loaded.is_error

        nodes = restored._history.nodes
        assert len(nodes) == 6

        turn1 = nodes[1]
        assert isinstance(turn1, TurnNode)
        assert turn1.collapse_state == CollapseState.EXPANDED
        assert turn1.tool_call_nodes[0].detail_collapsed is True
        assert turn1.tool_call_nodes[1].detail_collapsed is False

        turn2 = nodes[3]
        assert isinstance(turn2, TurnNode)
        assert turn2.collapse_state == CollapseState.COLLAPSED

        turn3 = nodes[5]
        assert isinstance(turn3, TurnNode)
        assert turn3.collapse_state == CollapseState.EXPANDED
        assert turn3.tool_call_nodes[0].detail_collapsed is False


# ---------------------------------------------------------------------------
# SessionStore -- agent identity operations
# ---------------------------------------------------------------------------


class TestSessionStoreAgent:
    def test_save_and_load_agent(self, tmp_path: Path):
        store = SessionStore(tmp_path)
        agent = Agent(id=AgentID("a1"), name="stored")

        store.save_agent(agent)
        assert store.agent_exists("a1")

        loaded = store.load_agent("a1")
        assert loaded.id == AgentID("a1")
        assert loaded.name == "stored"

    def test_agent_exists_returns_false_for_missing(self, tmp_path: Path):
        store = SessionStore(tmp_path)
        assert not store.agent_exists("nonexistent")

    def test_load_missing_agent_raises_key_error(self, tmp_path: Path):
        store = SessionStore(tmp_path)
        with pytest.raises(KeyError, match="nonexistent"):
            store.load_agent("nonexistent")

    def test_save_agent_without_id_raises(self, tmp_path: Path):
        store = SessionStore(tmp_path)
        agent = Agent(name="no-id")
        with pytest.raises(ValueError, match="without an id"):
            store.save_agent(agent)

    def test_list_agent_ids_empty(self, tmp_path: Path):
        store = SessionStore(tmp_path)
        assert store.list_agent_ids() == []

    def test_list_agent_ids_returns_sorted(self, tmp_path: Path):
        store = SessionStore(tmp_path)
        for name in ["charlie", "alice", "bob"]:
            store.save_agent(Agent(id=AgentID(name), name=name))

        ids = store.list_agent_ids()
        assert ids == [AgentID("alice"), AgentID("bob"), AgentID("charlie")]

    def test_delete_agent_removes_identity(self, tmp_path: Path):
        store = SessionStore(tmp_path)
        store.save_agent(Agent(id=AgentID("del-me"), name="doomed"))
        assert store.agent_exists("del-me")

        store.delete_agent("del-me")
        assert not store.agent_exists("del-me")

    def test_delete_nonexistent_agent_is_noop(self, tmp_path: Path):
        store = SessionStore(tmp_path)
        store.delete_agent("ghost")

    def test_overwrite_existing_agent(self, tmp_path: Path):
        store = SessionStore(tmp_path)
        store.save_agent(Agent(id=AgentID("ow"), name="v1"))
        store.save_agent(Agent(id=AgentID("ow"), name="v2"))
        loaded = store.load_agent("ow")
        assert loaded.name == "v2"

    def test_list_agent_ids_on_nonexistent_root(self, tmp_path: Path):
        store = SessionStore(tmp_path / "no-such-dir")
        assert store.list_agent_ids() == []

    def test_agent_id_type_preserved(self, tmp_path: Path):
        store = SessionStore(tmp_path)
        store.save_agent(Agent(id=AgentID("typed"), name="t"))
        ids = store.list_agent_ids()
        assert all(isinstance(i, AgentID) for i in ids)

    def test_loaded_agent_has_workspace_from_convention(self, tmp_path: Path):
        store = SessionStore(tmp_path)
        agent = Agent(id=AgentID("dev-1"), name="Dev")
        store.save_agent(agent)

        loaded = store.load_agent(AgentID("dev-1"))
        assert loaded.workspace == tmp_path / "dev-1"


# ---------------------------------------------------------------------------
# SessionStore -- session operations
# ---------------------------------------------------------------------------


class TestSessionStoreSession:
    def _make_agent(self, agent_id: str = "a1") -> Agent:
        return Agent(id=AgentID(agent_id), name="test")

    def test_save_and_load_session(self, tmp_path: Path):
        store = SessionStore(tmp_path)
        agent = self._make_agent()
        session = Session(
            agent=agent,
            key=SessionKey("s1"),
            created_at=datetime(2025, 6, 15, tzinfo=timezone.utc),
            last_active=datetime(2025, 6, 15, tzinfo=timezone.utc),
        )
        session._history = _make_simple_history()

        store.save_session(session)
        assert store.session_exists(agent.id, "s1")

        loaded = store.load_session(agent, "s1")
        assert loaded.key == SessionKey("s1")
        assert loaded.agent is agent
        assert len(loaded._history.nodes) == 2

    def test_session_exists_returns_false_for_missing(self, tmp_path: Path):
        store = SessionStore(tmp_path)
        assert not store.session_exists("a1", "nonexistent")

    def test_load_missing_session_raises_key_error(self, tmp_path: Path):
        store = SessionStore(tmp_path)
        agent = self._make_agent()
        with pytest.raises(KeyError, match="nonexistent"):
            store.load_session(agent, "nonexistent")

    def test_save_session_without_key_raises(self, tmp_path: Path):
        store = SessionStore(tmp_path)
        agent = self._make_agent()
        session = Session(agent=agent)
        with pytest.raises(ValueError, match="without a key"):
            store.save_session(session)

    def test_save_session_without_agent_id_raises(self, tmp_path: Path):
        store = SessionStore(tmp_path)
        agent = Agent(name="no-id")
        session = Session(agent=agent, key=SessionKey("s1"))
        with pytest.raises(ValueError, match="without an id"):
            store.save_session(session)

    def test_list_session_keys_empty(self, tmp_path: Path):
        store = SessionStore(tmp_path)
        assert store.list_session_keys("a1") == []

    def test_list_session_keys_returns_sorted(self, tmp_path: Path):
        store = SessionStore(tmp_path)
        agent = self._make_agent()
        for name in ["charlie", "alice", "bob"]:
            session = Session(agent=agent, key=SessionKey(name))
            store.save_session(session)

        keys = store.list_session_keys(agent.id)
        assert keys == [
            SessionKey("alice"),
            SessionKey("bob"),
            SessionKey("charlie"),
        ]

    def test_delete_session(self, tmp_path: Path):
        store = SessionStore(tmp_path)
        agent = self._make_agent()
        store.save_session(Session(agent=agent, key=SessionKey("del-me")))
        assert store.session_exists(agent.id, "del-me")

        store.delete_session(agent.id, "del-me")
        assert not store.session_exists(agent.id, "del-me")

    def test_delete_nonexistent_session_is_noop(self, tmp_path: Path):
        store = SessionStore(tmp_path)
        store.delete_session("a1", "ghost")

    def test_overwrite_existing_session(self, tmp_path: Path):
        store = SessionStore(tmp_path)
        agent = self._make_agent()

        s1 = Session(
            agent=agent,
            key=SessionKey("ow"),
            metadata={"version": "v1"},
        )
        store.save_session(s1)

        s2 = Session(
            agent=agent,
            key=SessionKey("ow"),
            metadata={"version": "v2"},
        )
        store.save_session(s2)

        loaded = store.load_session(agent, "ow")
        assert loaded.metadata == {"version": "v2"}

    def test_session_key_type_preserved(self, tmp_path: Path):
        store = SessionStore(tmp_path)
        agent = self._make_agent()
        store.save_session(Session(agent=agent, key=SessionKey("typed")))
        keys = store.list_session_keys(agent.id)
        assert all(isinstance(k, SessionKey) for k in keys)

    def test_custom_serializer(self, tmp_path: Path):
        call_log: list[str] = []

        class TrackingSerializer:
            def save_agent(self, agent: Agent, path: Path) -> None:
                call_log.append(f"save_agent:{agent.id}")
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("{}", encoding="utf-8")

            def load_agent(self, path: Path) -> Agent:
                call_log.append("load_agent")
                return Agent(id=AgentID("custom"), name="custom")

            def save_session(self, session: Session, directory: Path) -> None:
                call_log.append(f"save_session:{session.key}")
                (directory / "marker.txt").write_text("saved", encoding="utf-8")

            def load_session(self, directory: Path, agent: Agent) -> Session:
                call_log.append("load_session")
                return Session(agent=agent, key=SessionKey("custom"))

        store = SessionStore(tmp_path, serializer=TrackingSerializer())
        agent = Agent(id=AgentID("cs"), name="cs")
        store.save_agent(agent)
        store.load_agent("cs")
        session = Session(agent=agent, key=SessionKey("sk"))
        store.save_session(session)
        store.load_session(agent, "sk")

        assert "save_agent:cs" in call_log
        assert "load_agent" in call_log
        assert "save_session:sk" in call_log
        assert "load_session" in call_log


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
        expected = tmp_path / ".thorn" / "agents"
        assert rt.sessions.root == expected

    def test_custom_session_store(self, tmp_path: Path):
        custom_store = SessionStore(tmp_path / "custom-agents")
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

    def test_create_agent_with_auto_id(self, tmp_path: Path):
        rt = self._make_runtime(tmp_path)
        agent = rt.create_agent()

        assert isinstance(agent.id, AgentID)
        assert len(agent.id) > 0
        assert agent.name == str(agent.id)

    def test_create_agent_with_explicit_id(self, tmp_path: Path):
        rt = self._make_runtime(tmp_path)
        agent = rt.create_agent(id="my-id")

        assert agent.id == AgentID("my-id")
        assert isinstance(agent.id, AgentID)

    def test_create_agent_with_str_id_becomes_agent_id(self, tmp_path: Path):
        rt = self._make_runtime(tmp_path)
        agent = rt.create_agent(id="str-id")
        assert isinstance(agent.id, AgentID)

    def test_create_agent_with_custom_name(self, tmp_path: Path):
        rt = self._make_runtime(tmp_path)
        agent = rt.create_agent(id="a1", name="reviewer")
        assert agent.name == "reviewer"
        assert agent.id == AgentID("a1")

    def test_create_agent_with_metadata(self, tmp_path: Path):
        rt = self._make_runtime(tmp_path)
        agent = rt.create_agent(id="a1", metadata={"issue": 42})
        assert agent.metadata == {"issue": 42}

    def test_create_agent_assigns_workspace(self, tmp_path: Path):
        rt = self._make_runtime(tmp_path)
        agent = rt.create_agent(id="a1")
        assert agent.workspace is not None

    def test_save_and_get_or_create_agent(self, tmp_path: Path):
        rt = self._make_runtime(tmp_path)
        agent = rt.create_agent(id="persistent", name="bot")
        rt.save_agent(agent)

        retrieved = rt.get_or_create_agent("persistent")
        assert retrieved.id == AgentID("persistent")
        assert retrieved.name == "bot"

    def test_loaded_agent_has_workspace(self, tmp_path: Path):
        rt = self._make_runtime(tmp_path)
        agent = rt.create_agent(id="a1")
        original_workspace = agent.workspace
        rt.save_agent(agent)

        loaded = rt.get_or_create_agent("a1")
        assert loaded.workspace is not None
        assert loaded.workspace == original_workspace

    def test_get_or_create_creates_when_missing(self, tmp_path: Path):
        rt = self._make_runtime(tmp_path)
        agent = rt.get_or_create_agent("new-id")
        assert agent.id == AgentID("new-id")

    def test_get_or_create_session(self, tmp_path: Path):
        rt = self._make_runtime(tmp_path)
        agent = rt.create_agent(id="a1")
        session = rt.get_or_create_session(agent, "s1")
        assert session.key == SessionKey("s1")
        assert session.agent is agent
        assert session.created_at is not None
        assert session.last_active is not None

    def test_get_or_create_session_loads_existing(self, tmp_path: Path):
        rt = self._make_runtime(tmp_path)
        agent = rt.create_agent(id="a1")

        session = rt.get_or_create_session(agent, "s1")
        session._history = _make_simple_history()
        rt.save_session(session)

        reloaded = rt.get_or_create_session(agent, "s1")
        assert reloaded.key == SessionKey("s1")
        assert len(reloaded._history.nodes) == 2

    def test_save_session_updates_last_active(self, tmp_path: Path):
        rt = self._make_runtime(tmp_path)
        agent = rt.create_agent(id="a1")
        ts = datetime(2020, 1, 1, tzinfo=timezone.utc)
        session = Session(agent=agent, key=SessionKey("s1"), last_active=ts)
        rt.save_session(session)
        assert session.last_active > ts

    def test_get_or_create_session_without_agent_id_raises(self, tmp_path: Path):
        rt = self._make_runtime(tmp_path)
        agent = Agent(name="no-id")
        with pytest.raises(ValueError, match="without an id"):
            rt.get_or_create_session(agent, "s1")

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
        from thorn import AgentID, Runtime, SessionKey
        assert Runtime is not None
        assert SessionKey is not None
        assert AgentID is not None

    def test_session_not_importable_from_thorn(self):
        import thorn
        assert not hasattr(thorn, "Session")

    def test_runtime_importable_from_thorn_runtime(self):
        from thorn.runtime import (
            AgentID,
            Runtime,
            SessionKey,
            SessionStore,
            SessionSerializer,
            JsonSessionSerializer,
        )
        assert all(cls is not None for cls in [
            AgentID, Runtime, SessionKey, SessionStore,
            SessionSerializer, JsonSessionSerializer,
        ])

    def test_session_not_importable_from_thorn_runtime(self):
        import thorn.runtime
        assert not hasattr(thorn.runtime, "Session")

    def test_agent_id_importable_from_thorn_runtime(self):
        from thorn.runtime import AgentID
        assert AgentID is not None

    def test_session_importable_from_thorn_core(self):
        from thorn.core import Session
        assert Session is not None
