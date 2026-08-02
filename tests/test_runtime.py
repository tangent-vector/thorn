"""Tests for thorn.runtime -- Agent/Session persistence, SessionStore, serialization, and Runtime."""

from __future__ import annotations

import json
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from thorn.core._agent import Agent
from thorn.core._context import ExecutionContext, get_context
from thorn.core._external_content import ExternalContentPeerStatus
from thorn.core._history import (
    ArchiveMarkerNode,
    CollapseState,
    HistoryTree,
    HousekeepingNode,
    TurnNode,
    UserPromptNode,
)
from thorn.core._messages import (
    AssistantMessage,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)
from thorn.core._provider import (
    LLMConfig,
    LLMModelConfig,
    LLMProviderType,
    MockProvider,
    OpenAIProvider,
    OpenAIProviderSettings,
    load_provider_from_config,
)
from thorn.core._service import Service
from thorn.core._session import Session
from thorn.core._validation_convergence import ValidationConvergencePolicy
from thorn.runtime import (
    AgentID,
    HandlingPhase,
    JsonSessionSerializer,
    NotificationID,
    Runtime,
    SessionKey,
    SessionStore,
    WorkingSet,
    deserialize_history,
    serialize_history,
)
from thorn.runtime._paths import AgencyPaths

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
    def test_is_not_str_subclass(self):
        # SessionKey deliberately stopped deriving from `str` so that
        # callers must commit to either a structured (component-based)
        # view or a string-formatted view, never both.  See
        # `src/thorn/runtime/_session.py` for the rationale.
        key = SessionKey("my-session")
        assert isinstance(key, SessionKey)
        assert not isinstance(key, str)

    def test_str_form_joins_components(self):
        key = SessionKey("gitlab/issue/42")
        assert str(key) == "gitlab/issue/42"
        assert key.components == ("gitlab", "issue", "42")

    def test_components_form_round_trips(self):
        key = SessionKey(("gitlab", "issue", "42"))
        assert str(key) == "gitlab/issue/42"
        assert SessionKey(str(key)) == key

    def test_repr_is_unambiguous(self):
        assert repr(SessionKey("abc")) == "SessionKey('abc')"
        assert repr(SessionKey("a/b")) == "SessionKey('a/b')"

    def test_equality_to_other_session_key_only(self):
        key = SessionKey("abc")
        assert key == SessionKey("abc")
        # Plain strings are no longer equal to a SessionKey (the
        # whole point of dropping the str-subclass shape).
        assert key != "abc"

    def test_as_dict_key(self):
        key = SessionKey("k1")
        d: dict[SessionKey, int] = {key: 1}
        assert d[SessionKey("k1")] == 1

    def test_iter_and_len(self):
        key = SessionKey("a/b/c")
        assert list(key) == ["a", "b", "c"]
        assert len(key) == 3

    def test_construction_validation(self):
        # Empty string -> error (no components).
        with pytest.raises(ValueError):
            SessionKey("")
        # Empty iterable -> error.
        with pytest.raises(ValueError):
            SessionKey(())
        # Empty component (e.g. trailing or doubled slash) -> error.
        with pytest.raises(ValueError):
            SessionKey("a//b")
        with pytest.raises(ValueError):
            SessionKey(("a", "", "b"))
        # `/` inside a literal component (iterable form) -> error.
        with pytest.raises(ValueError):
            SessionKey(("a/b", "c"))


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
            ToolResultMessage(
                call_id="tc_2",
                content="class Bar:\n    x = 1",
                external_content_peer_statuses=frozenset(
                    {ExternalContentPeerStatus.UNKNOWN}
                ),
            ),
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

    def test_archive_marker_roundtrip(self):
        tree = HistoryTree()
        marker = ArchiveMarkerNode(
            archived_at=datetime(2026, 4, 8, 22, 10, 0, tzinfo=timezone.utc),
            summary="Investigated issue #6 and opened MR",
            node_count=12,
            journal_date="2026-04-08",
        )
        tree.nodes.append(marker)
        tree.append_user_prompt("continue")

        data = serialize_history(tree)
        restored = deserialize_history(data)

        assert len(restored.nodes) == 2
        rm = restored.nodes[0]
        assert isinstance(rm, ArchiveMarkerNode)
        assert rm.archived_at == datetime(2026, 4, 8, 22, 10, 0, tzinfo=timezone.utc)
        assert rm.summary == "Investigated issue #6 and opened MR"
        assert rm.node_count == 12
        assert rm.journal_date == "2026-04-08"

    def test_archive_marker_serialized_format(self):
        tree = HistoryTree()
        marker = ArchiveMarkerNode(
            archived_at=datetime(2026, 4, 8, 22, 10, 0, tzinfo=timezone.utc),
            summary="archived",
            node_count=5,
            journal_date="2026-04-08",
        )
        tree.nodes.append(marker)

        data = serialize_history(tree)
        assert len(data) == 1
        assert data[0]["type"] == "archive_marker"
        assert data[0]["node_count"] == 5
        assert data[0]["journal_date"] == "2026-04-08"
        assert "archived_at" in data[0]

    def test_housekeeping_node_roundtrip(self):
        tree = HistoryTree()
        tree.append_user_prompt("start")
        hk = HousekeepingNode(inner_nodes=[
            UserPromptNode(UserMessage(content="housekeeping prompt")),
            TurnNode(
                assistant_content="journaled everything",
                tool_call_nodes=[],
            ),
        ])
        tree.nodes.append(hk)
        tree.append_user_prompt("next task")

        data = serialize_history(tree)
        restored = deserialize_history(data)

        assert len(restored.nodes) == 3
        assert isinstance(restored.nodes[0], UserPromptNode)
        rh = restored.nodes[1]
        assert isinstance(rh, HousekeepingNode)
        assert len(rh.inner_nodes) == 2
        assert isinstance(rh.inner_nodes[0], UserPromptNode)
        assert rh.inner_nodes[0].message.content == "housekeeping prompt"
        assert isinstance(rh.inner_nodes[1], TurnNode)
        assert rh.inner_nodes[1].assistant_content == "journaled everything"

    def test_housekeeping_node_serialized_format(self):
        hk = HousekeepingNode(inner_nodes=[
            UserPromptNode(UserMessage(content="prompt")),
        ])
        tree = HistoryTree()
        tree.nodes.append(hk)

        data = serialize_history(tree)
        assert len(data) == 1
        assert data[0]["type"] == "housekeeping"
        assert len(data[0]["inner_nodes"]) == 1
        assert data[0]["inner_nodes"][0]["type"] == "user_prompt"

    def test_mixed_tree_roundtrip(self):
        """Full roundtrip with all four node types."""
        tree = HistoryTree()

        marker = ArchiveMarkerNode(
            archived_at=datetime(2026, 4, 7, 12, 0, 0, tzinfo=timezone.utc),
            summary="old content",
            node_count=20,
            journal_date="2026-04-07",
        )
        tree.nodes.append(marker)

        tree.append_user_prompt("continue working")
        tree.append_turn(
            AssistantMessage(
                content="reading file",
                tool_calls=[ToolCall(call_id="c1", name="read_file", arguments='{"path": "x.py"}')],
            ),
            [ToolResultMessage(call_id="c1", content="file content")],
        )

        hk = HousekeepingNode(inner_nodes=[
            UserPromptNode(UserMessage(content="please journal")),
            TurnNode(assistant_content="done", tool_call_nodes=[]),
        ])
        tree.nodes.append(hk)

        tree.append_user_prompt("final task")

        data = serialize_history(tree)
        restored = deserialize_history(data)

        assert len(restored.nodes) == 5
        assert isinstance(restored.nodes[0], ArchiveMarkerNode)
        assert isinstance(restored.nodes[1], UserPromptNode)
        assert isinstance(restored.nodes[2], TurnNode)
        assert isinstance(restored.nodes[3], HousekeepingNode)
        assert isinstance(restored.nodes[4], UserPromptNode)

        original_messages = tree.render()
        restored_messages = restored.render()
        assert len(original_messages) == len(restored_messages)
        for orig, rest in zip(original_messages, restored_messages):
            assert type(orig) is type(rest)
            assert orig.role == rest.role

    def test_rendered_messages_match_after_archive_marker_roundtrip(self):
        tree = HistoryTree()
        marker = ArchiveMarkerNode(
            archived_at=datetime(2026, 4, 8, 10, 0, 0, tzinfo=timezone.utc),
            summary="test",
            node_count=3,
            journal_date="2026-04-08",
        )
        tree.nodes.append(marker)
        tree.append_user_prompt("hi")

        original = tree.render()
        data = serialize_history(tree)
        restored = deserialize_history(data)
        restored_msgs = restored.render()

        assert len(original) == len(restored_msgs)
        assert original[0].content == restored_msgs[0].content


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
        # The parent directory name is the source of truth for the
        # agent ID, so place the file under ``<id>/agent.json``.
        agent_dir = tmp_path / "test-agent"
        agent_dir.mkdir()
        agent_path = agent_dir / "agent.json"
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
        agent_path = tmp_path / "readable.json"
        serializer.save_agent(agent, agent_path)

        content = agent_path.read_text(encoding="utf-8")
        assert "\n" in content
        parsed = json.loads(content)
        # ``id`` is no longer written to the JSON: the path stem
        # encodes it.  Only the human-facing ``name`` is stored
        # explicitly so the file remains readable on its own.
        assert "id" not in parsed
        assert parsed["name"] == "bot"

    def test_agent_class_name_stored(self, tmp_path: Path):
        class CustomAgent(Agent):
            pass

        agent = CustomAgent(id=AgentID("cls"), name="custom")
        serializer = JsonSessionSerializer()
        agent_path = tmp_path / "cls.json"
        serializer.save_agent(agent, agent_path)

        content = json.loads(agent_path.read_text(encoding="utf-8"))
        assert content["agent_class"] == "CustomAgent"

    def test_agent_class_resolved_on_load(self, tmp_path: Path):
        class ResolvableAgent(Agent):
            pass

        agent = ResolvableAgent(id=AgentID("resolve"), name="r")
        serializer = JsonSessionSerializer()
        agent_path = tmp_path / "resolve.json"
        serializer.save_agent(agent, agent_path)

        restored = serializer.load_agent(agent_path)
        assert type(restored).__name__ == "ResolvableAgent"
        assert isinstance(restored, ResolvableAgent)

    def test_unknown_agent_class_falls_back_to_base(self, tmp_path: Path):
        agent = Agent(id=AgentID("fallback"), name="fb")
        serializer = JsonSessionSerializer()
        agent_path = tmp_path / "fallback.json"
        serializer.save_agent(agent, agent_path)

        data = json.loads(agent_path.read_text(encoding="utf-8"))
        data["agent_class"] = "NoSuchAgent"
        agent_path.write_text(
            json.dumps(data, indent=2), encoding="utf-8",
        )

        restored = serializer.load_agent(agent_path)
        assert type(restored) is Agent
        assert restored.name == "fb"

    def test_sandbox_override_roundtrips(self, tmp_path: Path):
        from thorn.gateway._config import AgentSandboxOverride

        override = AgentSandboxOverride(
            image="thorn-sandbox-rust:dev",
            env_passthrough=["RUST_LOG"],
            extra_env={"CARGO_HOME": "/agent/home/.cargo"},
            container_ready_timeout_s=60.0,
        )
        agent = Agent(
            id=AgentID("rust-agent"),
            name="rusty",
            sandbox_override=override,
        )
        serializer = JsonSessionSerializer()
        agent_dir = tmp_path / "rust-agent"
        agent_dir.mkdir()
        agent_path = agent_dir / "agent.json"
        serializer.save_agent(agent, agent_path)

        on_disk = json.loads(agent_path.read_text(encoding="utf-8"))
        assert on_disk["sandbox"]["image"] == "thorn-sandbox-rust:dev"
        assert on_disk["sandbox"]["env_passthrough"] == ["RUST_LOG"]
        assert on_disk["sandbox"]["extra_env"] == {
            "CARGO_HOME": "/agent/home/.cargo",
        }
        assert on_disk["sandbox"]["container_ready_timeout_s"] == 60.0
        assert "backend" not in on_disk["sandbox"]

        restored = serializer.load_agent(agent_path)
        restored_override = getattr(restored, "sandbox_override", None)
        assert isinstance(restored_override, AgentSandboxOverride)
        assert restored_override.image == "thorn-sandbox-rust:dev"
        assert restored_override.env_passthrough == ["RUST_LOG"]
        assert restored_override.extra_env == {
            "CARGO_HOME": "/agent/home/.cargo",
        }
        assert restored_override.container_ready_timeout_s == 60.0
        assert restored_override.backend is None

    def test_llm_config_roundtrips(self, tmp_path: Path):
        llm_config = LLMConfig(
            provider=OpenAIProviderSettings(
                type=LLMProviderType.OPENAI,
                api_url="https://llm.example/v1",
                api_key_env_var="THORN_LLM_KEY",
            ),
            model=LLMModelConfig(
                name="agent-model",
                options={"reasoning_effort": "high"},
            ),
        )
        agent = Agent(
            id=AgentID("model-agent"),
            name="model-agent",
            llm_config=llm_config,
        )
        serializer = JsonSessionSerializer()
        agent_dir = tmp_path / "model-agent"
        agent_dir.mkdir()
        agent_path = agent_dir / "agent.json"
        serializer.save_agent(agent, agent_path)

        on_disk = json.loads(agent_path.read_text(encoding="utf-8"))
        assert on_disk["llm"]["provider"]["api_url"] == "https://llm.example/v1"
        assert on_disk["llm"]["provider"]["api_key_env_var"] == "THORN_LLM_KEY"
        assert on_disk["llm"]["model"]["name"] == "agent-model"
        assert on_disk["llm"]["model"]["options"]["reasoning_effort"] == "high"

        restored = serializer.load_agent(agent_path)
        restored_llm_config = getattr(restored, "llm_config", None)
        assert restored_llm_config == llm_config

    def test_no_sandbox_override_omits_block(self, tmp_path: Path):
        agent = Agent(id=AgentID("plain"), name="plain")
        serializer = JsonSessionSerializer()
        agent_dir = tmp_path / "plain"
        agent_dir.mkdir()
        agent_path = agent_dir / "agent.json"
        serializer.save_agent(agent, agent_path)
        on_disk = json.loads(agent_path.read_text(encoding="utf-8"))
        assert "sandbox" not in on_disk
        restored = serializer.load_agent(agent_path)
        assert getattr(restored, "sandbox_override", None) is None

    def test_load_derives_id_from_parent_directory(self, tmp_path: Path):
        """``load_agent`` derives the AgentID from the file's parent dir.

        This is the single-source-of-truth design: the on-disk path
        encodes the ID (Phase-A layout puts it in the containing
        directory name), the JSON body holds only the human-facing
        ``name``.  Even an in-memory agent without a saved-from ID
        gets a well-defined ID once it has been persisted (because
        the persistence step picks the file path).
        """
        agent = Agent(name="display-only")
        serializer = JsonSessionSerializer()
        agent_dir = tmp_path / "dir-derived"
        agent_dir.mkdir()
        agent_path = agent_dir / "agent.json"
        serializer.save_agent(agent, agent_path)
        restored = serializer.load_agent(agent_path)
        assert restored.id == AgentID("dir-derived")
        assert restored.name == "display-only"


# ---------------------------------------------------------------------------
# JsonSessionSerializer -- agent accounts (save/load with credentials)
# ---------------------------------------------------------------------------


class TestJsonSessionSerializerAgentAccounts:
    """Cover the on-disk shape of agent accounts: a flat
    ``"accounts"`` array of objects, each carrying a ``service``
    discriminator and a ``credentials`` list.  Per-service fields
    (``git_user_name`` etc.) live alongside ``service`` and survive
    parse-time validation via :class:`UntypedAccountConfig`.
    """

    def test_save_and_load_accounts_roundtrip(self, tmp_path: Path):
        from thorn.core._account import AgentAccountsConfig, UntypedAccountConfig
        from thorn.core._credentials import Credential

        accounts = AgentAccountsConfig(accounts=[
            UntypedAccountConfig(
                service="my-forge",
                credentials=[Credential(
                    kind="gitlab-pat",
                    env_var_name="MY_GL_TOKEN",
                )],
                git_user_name="bot",
                git_user_email="bot@thorn",
            ),
        ])
        agent = Agent(
            id=AgentID("acct-test"),
            name="bot",
            metadata={},
            accounts=accounts,
        )

        serializer = JsonSessionSerializer()
        agent_path = tmp_path / "acct-test.json"
        serializer.save_agent(agent, agent_path)

        restored = serializer.load_agent(agent_path)

        restored_accounts = getattr(restored, "accounts", None)
        assert restored_accounts is not None
        assert len(restored_accounts.accounts) == 1

        acct = restored_accounts.accounts[0]
        assert acct.service == "my-forge"
        # Service-specific extras flow through UntypedAccountConfig
        # (the parse-time shape, before the gateway's per-service
        # validation pass swaps in a typed AccountConfig).
        dump = acct.model_dump()
        assert dump["git_user_name"] == "bot"
        assert dump["git_user_email"] == "bot@thorn"
        assert acct.credentials == [
            Credential(kind="gitlab-pat", env_var_name="MY_GL_TOKEN"),
        ]

    def test_accounts_written_to_json(self, tmp_path: Path):
        """The on-disk JSON shape is a flat ``accounts`` array."""
        from thorn.core._account import AgentAccountsConfig, UntypedAccountConfig
        from thorn.core._credentials import Credential

        accounts = AgentAccountsConfig(accounts=[
            UntypedAccountConfig(
                service="gl",
                credentials=[Credential(
                    kind="gitlab-pat",
                    env_var_name="GL_TOKEN",
                )],
            ),
        ])
        agent = Agent(id=AgentID("a"), name="a", accounts=accounts)

        serializer = JsonSessionSerializer()
        path = tmp_path / "a.json"
        serializer.save_agent(agent, path)

        data = json.loads(path.read_text(encoding="utf-8"))
        assert "accounts" in data
        assert isinstance(data["accounts"], list)
        assert len(data["accounts"]) == 1
        assert data["accounts"][0]["service"] == "gl"
        assert data["accounts"][0]["credentials"] == [
            {"kind": "gitlab-pat", "env_var_name": "GL_TOKEN", "name": None},
        ]

    def test_no_accounts_key_when_empty(self, tmp_path: Path):
        """Agents without accounts should not have an 'accounts' key in JSON."""
        agent = Agent(id=AgentID("plain"), name="plain", metadata={})

        serializer = JsonSessionSerializer()
        path = tmp_path / "agent.json"
        serializer.save_agent(agent, path)

        data = json.loads(path.read_text(encoding="utf-8"))
        assert "accounts" not in data

    def test_load_credentials_list_shape(self, tmp_path: Path):
        """The deserializer accepts the new ``credentials: [...]`` shape
        and preserves per-service extras unchanged for the gateway's
        eager validation pass."""
        from thorn.core._account import UntypedAccountConfig
        from thorn.core._credentials import Credential

        agent_data = {
            "agent_class": "Agent",
            "name": "shape-test",
            "metadata": {},
            "accounts": [
                {
                    "service": "gl",
                    "credentials": [
                        {"kind": "gitlab-pat", "env_var_name": "MY_GL_TOKEN"},
                    ],
                    "git_user_name": "bot",
                    "git_user_email": "bot@thorn",
                },
            ],
        }
        path = tmp_path / "shape-test.json"
        path.write_text(json.dumps(agent_data), encoding="utf-8")

        serializer = JsonSessionSerializer()
        restored = serializer.load_agent(path)

        acct = restored.accounts.accounts[0]
        assert isinstance(acct, UntypedAccountConfig)
        assert acct.credentials == [
            Credential(kind="gitlab-pat", env_var_name="MY_GL_TOKEN"),
        ]

    def test_load_preserves_per_service_extra_fields(
        self, tmp_path: Path,
    ):
        """Service-specific fields (``git_user_name`` etc.) are not
        recognised by the base :class:`AccountConfig`; the
        deserializer must use :class:`UntypedAccountConfig` so those
        fields survive intact for the gateway's per-service
        validation pass to consume."""
        agent_data = {
            "agent_class": "Agent",
            "name": "no-strip",
            "metadata": {},
            "accounts": [
                {
                    "service": "gl",
                    "credentials": [
                        {"kind": "gitlab-pat", "env_var_name": "TOK"},
                    ],
                    "git_user_name": "$NOT_AN_ENV_VAR",
                    "git_user_email": "$ALSO_NOT",
                    "custom_field": "preserved",
                },
            ],
        }
        path = tmp_path / "no-strip.json"
        path.write_text(json.dumps(agent_data), encoding="utf-8")

        serializer = JsonSessionSerializer()
        restored = serializer.load_agent(path)

        acct = restored.accounts.accounts[0]
        # Unknown / per-service extras are preserved verbatim;
        # values are NOT subjected to ``$ENV_VAR`` expansion (the
        # framework dropped that magic in favour of explicit
        # ``env_var_name`` on credentials).
        dump = acct.model_dump()
        assert dump["git_user_name"] == "$NOT_AN_ENV_VAR"
        assert dump["git_user_email"] == "$ALSO_NOT"
        assert dump["custom_field"] == "preserved"

    def test_load_multiple_credentials_on_one_account(
        self, tmp_path: Path,
    ):
        """An account may carry multiple credentials of distinct kinds
        in the same list; the loader keeps all of them."""
        from thorn.core._credentials import Credential

        agent_data = {
            "agent_class": "Agent",
            "name": "multi-cred",
            "metadata": {},
            "accounts": [
                {
                    "service": "gh",
                    "credentials": [
                        {"kind": "pat", "env_var_name": "GH_PAT"},
                        {
                            "kind": "app", "name": "release-app",
                            "env_var_name": "GH_APP_KEY",
                        },
                    ],
                },
            ],
        }
        path = tmp_path / "multi-cred.json"
        path.write_text(json.dumps(agent_data), encoding="utf-8")

        serializer = JsonSessionSerializer()
        restored = serializer.load_agent(path)

        creds = restored.accounts.accounts[0].credentials
        assert creds == [
            Credential(kind="pat", env_var_name="GH_PAT"),
            Credential(
                kind="app", name="release-app", env_var_name="GH_APP_KEY",
            ),
        ]

    def test_load_without_accounts_field_leaves_agent_unconfigured(
        self, tmp_path: Path,
    ):
        """An agent JSON without an ``accounts`` key loads without one.

        We no longer emit a deprecation warning for legacy metadata
        keys (``metadata.git_user_name`` etc.); the broader cleanup
        of agent identity / git-service promotion is deferred and
        will be handled in a follow-up redesign.  This test pins
        down the pared-back behaviour so the deprecation tooling
        doesn't accidentally come back.
        """
        agent_data = {
            "agent_class": "Agent",
            "name": "no-accounts",
            "metadata": {"project": "my-proj"},
        }
        path = tmp_path / "no-accounts.json"
        path.write_text(json.dumps(agent_data), encoding="utf-8")

        serializer = JsonSessionSerializer()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            restored = serializer.load_agent(path)

        assert restored.metadata["project"] == "my-proj"
        assert not hasattr(restored, "accounts") or restored.accounts is None

        deprecation_warnings = [
            w for w in caught if issubclass(w.category, DeprecationWarning)
        ]
        assert deprecation_warnings == []


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

    def test_llm_config_roundtrips(self, tmp_path: Path):
        agent = self._make_agent()
        llm_config = LLMConfig(
            model=LLMModelConfig(
                name="session-model",
                options={"temperature": 0.4},
            ),
        )
        session = Session(
            agent=agent,
            key=SessionKey("session-model"),
            llm_config=llm_config,
        )

        serializer = JsonSessionSerializer()
        session_dir = tmp_path / "session_dir"
        serializer.save_session(session, session_dir)

        on_disk = json.loads(
            (session_dir / "session.json").read_text(encoding="utf-8")
        )
        assert on_disk["llm"]["model"]["name"] == "session-model"
        assert on_disk["llm"]["model"]["options"] == {"temperature": 0.4}

        restored = serializer.load_session(session_dir, agent)
        assert restored.llm_config == llm_config

    def test_working_set_roundtrips(self, tmp_path: Path):
        agent = self._make_agent()
        working_set = WorkingSet(
            phase=HandlingPhase.INSPECT,
            focused_inbox_item_id=NotificationID("01FOCUS"),
            objective="Investigate the notification before acting.",
        )
        session = Session(
            agent=agent,
            key=SessionKey("working-set"),
            working_set=working_set,
        )

        serializer = JsonSessionSerializer()
        session_dir = tmp_path / "session_dir"
        serializer.save_session(session, session_dir)

        on_disk = json.loads(
            (session_dir / "session.json").read_text(encoding="utf-8")
        )
        assert on_disk["working_set"] == {
            "phase": "inspect",
            "focused_inbox_item_id": "01FOCUS",
            "objective": "Investigate the notification before acting.",
        }

        restored = serializer.load_session(session_dir, agent)
        assert restored.working_set == working_set

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

    def test_workspace_root_roundtrip(self, tmp_path: Path):
        agent = self._make_agent()
        ws = tmp_path / "checkout" / "repo"
        session = Session(
            agent=agent,
            key=SessionKey("ws-test"),
            workspace_root=ws,
        )

        serializer = JsonSessionSerializer()
        session_dir = tmp_path / "session_dir"
        serializer.save_session(session, session_dir)

        raw = json.loads((session_dir / "session.json").read_text())
        assert raw["workspace_root"] == str(ws)

        restored = serializer.load_session(session_dir, agent)
        assert restored.workspace_root == ws

    def test_workspace_root_none_omitted_from_json(self, tmp_path: Path):
        agent = self._make_agent()
        session = Session(agent=agent, key=SessionKey("no-ws"))

        serializer = JsonSessionSerializer()
        session_dir = tmp_path / "session_dir"
        serializer.save_session(session, session_dir)

        raw = json.loads((session_dir / "session.json").read_text())
        assert "workspace_root" not in raw

        restored = serializer.load_session(session_dir, agent)
        assert restored.workspace_root is None

    def test_legacy_session_without_workspace_root_loads(self, tmp_path: Path):
        """Sessions persisted before workspace_root was added still load."""
        agent = self._make_agent()
        session_dir = tmp_path / "legacy"
        session_dir.mkdir()
        (session_dir / "session.json").write_text(json.dumps({
            "key": "old-session",
            "created_at": None,
            "last_active": None,
            "metadata": {},
        }))
        (session_dir / "history.json").write_text("[]")

        serializer = JsonSessionSerializer()
        restored = serializer.load_session(session_dir, agent)
        assert restored.workspace_root is None
        assert restored.key == SessionKey("old-session")
        assert restored.working_set == WorkingSet()

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
                assert (
                    orig.external_content_peer_statuses
                    == loaded.external_content_peer_statuses
                )

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
# JsonSessionSerializer -- atomic-write crash safety
# ---------------------------------------------------------------------------


class TestJsonSessionSerializerAtomicWrites:
    """Cancel-mid-save / crash-mid-save must leave on-disk state intact.

    The graceful-shutdown work item requires that each persisted
    file is written atomically: a reader that sees the file always
    sees either its old contents or its new contents, never a torn
    partial write.  These tests simulate the failure modes that
    matter in practice:

    - a permission / OSError raised during the final ``os.replace``
      (stand-in for any interrupt that prevents the rename from
      completing);
    - repeated saves after a failure, to verify the store can
      recover without operator intervention.

    The sidecar temp files used by the atomic writer are allowed
    to be left behind after a failure; we assert they do not
    contaminate the live files.
    """

    def test_agent_save_failure_preserves_old_contents(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        serializer = JsonSessionSerializer()
        path = tmp_path / "agent.json"

        # Seed a known-good record.
        initial = Agent(
            id=AgentID("a1"), name="initial", metadata={"v": 1},
        )
        serializer.save_agent(initial, path)
        original_bytes = path.read_bytes()

        # Patch os.replace in the serializer module to simulate a
        # crash between writing the sidecar and renaming it into
        # place.  The live file must remain unchanged.
        import thorn.runtime._serializer as serializer_mod

        def boom(src: str | Path, dst: str | Path) -> None:
            raise OSError("simulated rename failure")

        monkeypatch.setattr(serializer_mod.os, "replace", boom)

        updated = Agent(
            id=AgentID("a1"), name="updated", metadata={"v": 2},
        )
        with pytest.raises(OSError):
            serializer.save_agent(updated, path)

        # Live file untouched.
        assert path.read_bytes() == original_bytes
        # Loading still yields the initial record.
        loaded = serializer.load_agent(path)
        assert loaded.name == "initial"

    def test_agent_save_after_failure_recovers(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        serializer = JsonSessionSerializer()
        path = tmp_path / "agent.json"

        initial = Agent(id=AgentID("a1"), name="initial")
        serializer.save_agent(initial, path)

        import thorn.runtime._serializer as serializer_mod

        call_count = {"n": 0}
        real_replace = serializer_mod.os.replace

        def flaky_replace(src: str | Path, dst: str | Path) -> None:
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise OSError("first-time failure")
            real_replace(src, dst)

        monkeypatch.setattr(serializer_mod.os, "replace", flaky_replace)

        next_attempt = Agent(id=AgentID("a1"), name="second")
        with pytest.raises(OSError):
            serializer.save_agent(next_attempt, path)

        # Second call succeeds; the live file now reflects the new
        # state and the stale sidecar is gone (because the
        # successful replace consumed it).
        third = Agent(id=AgentID("a1"), name="third")
        serializer.save_agent(third, path)

        loaded = serializer.load_agent(path)
        assert loaded.name == "third"
        assert not any(
            p.name.startswith(".tmp-")
            for p in path.parent.iterdir()
        )

    def test_session_save_failure_preserves_history(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # history.json must survive a failed save of either file.
        # We stash a known history.json, then force the rename of
        # history.json to fail on the next save; the live file must
        # remain at its original contents.
        serializer = JsonSessionSerializer()
        directory = tmp_path / "session"
        agent = Agent(id=AgentID("a1"), name="bot")

        session = Session(agent=agent, key=SessionKey("s1"))
        session._history.append_user_prompt("hello world")
        serializer.save_session(session, directory)

        history_path = directory / "history.json"
        original_history_bytes = history_path.read_bytes()

        # Patch os.replace so any call with history.json as dst
        # raises; session.json replace is allowed through.
        import thorn.runtime._serializer as serializer_mod
        real_replace = serializer_mod.os.replace

        def selective_replace(src: str | Path, dst: str | Path) -> None:
            if str(dst).endswith("history.json"):
                raise OSError("history rename failed")
            real_replace(src, dst)

        monkeypatch.setattr(serializer_mod.os, "replace", selective_replace)

        # Mutate in-memory state and try to save.
        session._history.append_user_prompt("second turn")
        with pytest.raises(OSError):
            serializer.save_session(session, directory)

        # history.json is still the old value -- the in-memory
        # mutation never reached disk.
        assert history_path.read_bytes() == original_history_bytes

    def test_atomic_writer_leaves_no_tmp_on_success(
        self, tmp_path: Path,
    ) -> None:
        # Happy path: a successful save leaves no sidecar files
        # behind.  Critical for operators so they can trust that a
        # lingering .tmp-* file indicates an earlier failure.
        serializer = JsonSessionSerializer()
        directory = tmp_path / "session"
        agent = Agent(id=AgentID("a1"), name="bot")

        session = Session(agent=agent, key=SessionKey("s1"))
        serializer.save_session(session, directory)

        sidecars = [
            p for p in directory.iterdir()
            if p.name.startswith(".tmp-")
        ]
        assert sidecars == []


# ---------------------------------------------------------------------------
# SessionStore -- agent identity operations
# ---------------------------------------------------------------------------


class TestSessionStoreAgent:
    def test_save_and_load_agent(self, tmp_path: Path):
        store = SessionStore(AgencyPaths(home_root=tmp_path, workspace_root=tmp_path))
        agent = Agent(id=AgentID("a1"), name="stored")

        store.save_agent(agent)
        assert store.agent_exists("a1")

        loaded = store.load_agent("a1")
        assert loaded.id == AgentID("a1")
        assert loaded.name == "stored"

    def test_agent_exists_returns_false_for_missing(self, tmp_path: Path):
        store = SessionStore(AgencyPaths(home_root=tmp_path, workspace_root=tmp_path))
        assert not store.agent_exists("nonexistent")

    def test_load_missing_agent_raises_key_error(self, tmp_path: Path):
        store = SessionStore(AgencyPaths(home_root=tmp_path, workspace_root=tmp_path))
        with pytest.raises(KeyError, match="nonexistent"):
            store.load_agent("nonexistent")

    def test_save_agent_without_id_raises(self, tmp_path: Path):
        store = SessionStore(AgencyPaths(home_root=tmp_path, workspace_root=tmp_path))
        agent = Agent(name="no-id")
        with pytest.raises(ValueError, match="without an id"):
            store.save_agent(agent)

    def test_list_agent_ids_empty(self, tmp_path: Path):
        store = SessionStore(AgencyPaths(home_root=tmp_path, workspace_root=tmp_path))
        assert store.list_agent_ids() == []

    def test_list_agent_ids_returns_sorted(self, tmp_path: Path):
        store = SessionStore(AgencyPaths(home_root=tmp_path, workspace_root=tmp_path))
        for name in ["charlie", "alice", "bob"]:
            store.save_agent(Agent(id=AgentID(name), name=name))

        ids = store.list_agent_ids()
        assert ids == [AgentID("alice"), AgentID("bob"), AgentID("charlie")]

    def test_delete_agent_removes_identity(self, tmp_path: Path):
        store = SessionStore(AgencyPaths(home_root=tmp_path, workspace_root=tmp_path))
        store.save_agent(Agent(id=AgentID("del-me"), name="doomed"))
        assert store.agent_exists("del-me")

        store.delete_agent("del-me")
        assert not store.agent_exists("del-me")

    def test_delete_nonexistent_agent_is_noop(self, tmp_path: Path):
        store = SessionStore(AgencyPaths(home_root=tmp_path, workspace_root=tmp_path))
        store.delete_agent("ghost")

    def test_overwrite_existing_agent(self, tmp_path: Path):
        store = SessionStore(AgencyPaths(home_root=tmp_path, workspace_root=tmp_path))
        store.save_agent(Agent(id=AgentID("ow"), name="v1"))
        store.save_agent(Agent(id=AgentID("ow"), name="v2"))
        loaded = store.load_agent("ow")
        assert loaded.name == "v2"

    def test_list_agent_ids_on_nonexistent_root(self, tmp_path: Path):
        store = SessionStore(
            AgencyPaths(
                home_root=tmp_path / "no-such-dir",
                workspace_root=tmp_path / "no-such-dir",
            ),
        )
        assert store.list_agent_ids() == []

    def test_agent_id_type_preserved(self, tmp_path: Path):
        store = SessionStore(AgencyPaths(home_root=tmp_path, workspace_root=tmp_path))
        store.save_agent(Agent(id=AgentID("typed"), name="t"))
        ids = store.list_agent_ids()
        assert all(isinstance(i, AgentID) for i in ids)

    def test_loaded_agent_has_workspace_from_convention(self, tmp_path: Path):
        paths = AgencyPaths(home_root=tmp_path, workspace_root=tmp_path)
        store = SessionStore(paths)
        agent = Agent(id=AgentID("dev-1"), name="Dev")
        store.save_agent(agent)

        loaded = store.load_agent(AgentID("dev-1"))
        assert loaded.workspace == paths.agent_workspace_mount(AgentID("dev-1"))
        assert loaded.home == paths.agent_home_mount(AgentID("dev-1"))


# ---------------------------------------------------------------------------
# SessionStore -- session operations
# ---------------------------------------------------------------------------


class TestSessionStoreSession:
    def _make_agent(self, agent_id: str = "a1") -> Agent:
        return Agent(id=AgentID(agent_id), name="test")

    def test_save_and_load_session(self, tmp_path: Path):
        store = SessionStore(AgencyPaths(home_root=tmp_path, workspace_root=tmp_path))
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
        store = SessionStore(AgencyPaths(home_root=tmp_path, workspace_root=tmp_path))
        assert not store.session_exists("a1", "nonexistent")

    def test_load_missing_session_raises_key_error(self, tmp_path: Path):
        store = SessionStore(AgencyPaths(home_root=tmp_path, workspace_root=tmp_path))
        agent = self._make_agent()
        with pytest.raises(KeyError, match="nonexistent"):
            store.load_session(agent, "nonexistent")

    def test_save_session_without_key_raises(self, tmp_path: Path):
        store = SessionStore(AgencyPaths(home_root=tmp_path, workspace_root=tmp_path))
        agent = self._make_agent()
        session = Session(agent=agent)
        with pytest.raises(ValueError, match="without a key"):
            store.save_session(session)

    def test_save_session_without_agent_id_raises(self, tmp_path: Path):
        store = SessionStore(AgencyPaths(home_root=tmp_path, workspace_root=tmp_path))
        agent = Agent(name="no-id")
        session = Session(agent=agent, key=SessionKey("s1"))
        with pytest.raises(ValueError, match="without an id"):
            store.save_session(session)

    def test_list_session_keys_empty(self, tmp_path: Path):
        store = SessionStore(AgencyPaths(home_root=tmp_path, workspace_root=tmp_path))
        assert store.list_session_keys("a1") == []

    def test_list_session_keys_returns_sorted(self, tmp_path: Path):
        store = SessionStore(AgencyPaths(home_root=tmp_path, workspace_root=tmp_path))
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
        store = SessionStore(AgencyPaths(home_root=tmp_path, workspace_root=tmp_path))
        agent = self._make_agent()
        store.save_session(Session(agent=agent, key=SessionKey("del-me")))
        assert store.session_exists(agent.id, "del-me")

        store.delete_session(agent.id, "del-me")
        assert not store.session_exists(agent.id, "del-me")

    def test_delete_nonexistent_session_is_noop(self, tmp_path: Path):
        store = SessionStore(AgencyPaths(home_root=tmp_path, workspace_root=tmp_path))
        store.delete_session("a1", "ghost")

    def test_overwrite_existing_session(self, tmp_path: Path):
        store = SessionStore(AgencyPaths(home_root=tmp_path, workspace_root=tmp_path))
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
        store = SessionStore(AgencyPaths(home_root=tmp_path, workspace_root=tmp_path))
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

        store = SessionStore(
            AgencyPaths(home_root=tmp_path, workspace_root=tmp_path),
            serializer=TrackingSerializer(),
        )
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
        custom_store = SessionStore(
            AgencyPaths(
                home_root=tmp_path / "custom-agents",
                workspace_root=tmp_path / "custom-agents",
            ),
        )
        rt = self._make_runtime(tmp_path, session_store=custom_store)
        assert rt.sessions is custom_store

    def test_create_context(self, tmp_path: Path):
        rt = self._make_runtime(tmp_path)
        ctx = rt.create_context(system_prompts=["Extra prompt."])

        assert isinstance(ctx, ExecutionContext)
        assert ctx.provider is rt.provider
        assert ctx.workspace_root == tmp_path
        assert "Extra prompt." in ctx.system_prompts

    def test_create_context_preserves_validation_convergence_policy(
        self,
        tmp_path: Path,
    ):
        runtime = self._make_runtime(
            tmp_path,
            validation_convergence_policy=(
                ValidationConvergencePolicy.ACTION_EPOCH_V1
            ),
        )

        context = runtime.create_context()

        assert (
            context.validation_convergence_policy
            is ValidationConvergencePolicy.ACTION_EPOCH_V1
        )

    def test_provider_for_agent_uses_agent_llm_override(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ):
        monkeypatch.setenv("AGENCY_LLM_KEY", "agency-secret")
        monkeypatch.setenv("AGENT_LLM_KEY", "agent-secret")

        agency_llm_config = LLMConfig(
            provider=OpenAIProviderSettings(
                type=LLMProviderType.OPENAI,
                api_url="https://agency.example/v1",
                api_key_env_var="AGENCY_LLM_KEY",
            ),
            model=LLMModelConfig(
                name="agency-model",
                options={"max_tokens": 4096},
            ),
        )
        runtime = Runtime(
            provider=load_provider_from_config(agency_llm_config),
            provider_config=agency_llm_config,
            workspace_root=tmp_path,
        )
        agent = Agent(
            id=AgentID("agent"),
            llm_config=LLMConfig(
                provider=OpenAIProviderSettings(
                    type=LLMProviderType.OPENAI,
                    api_url="https://agent.example/v1",
                    api_key_env_var="AGENT_LLM_KEY",
                ),
                model=LLMModelConfig(
                    name="agent-model",
                    options={"reasoning_effort": "medium"},
                ),
            ),
        )

        provider = runtime.provider_for_agent(agent)
        assert isinstance(provider, OpenAIProvider)
        assert provider.config.api_url == "https://agent.example/v1"
        assert provider.config.api_key == "agent-secret"
        assert provider.config.model_name == "agent-model"
        assert provider.config.model_options == {
            "max_tokens": 4096,
            "reasoning_effort": "medium",
        }
        assert runtime.provider_for_agent(agent) is provider

    def test_provider_for_agent_without_override_uses_runtime_provider(
        self, tmp_path: Path,
    ):
        runtime = self._make_runtime(tmp_path)
        agent = Agent(id=AgentID("plain"))

        assert runtime.provider_for_agent(agent) is runtime.provider

    def test_provider_for_agent_model_override_inherits_gateway_provider(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ):
        monkeypatch.setenv("AGENCY_LLM_KEY", "agency-secret")
        agency_llm_config = LLMConfig(
            provider=OpenAIProviderSettings(
                type=LLMProviderType.OPENAI,
                api_url="https://agency.example/v1",
                api_key_env_var="AGENCY_LLM_KEY",
            ),
            model=LLMModelConfig(name="agency-model"),
        )
        runtime = Runtime(
            provider=load_provider_from_config(agency_llm_config),
            provider_config=agency_llm_config,
            workspace_root=tmp_path,
        )
        agent = Agent(
            id=AgentID("agent"),
            llm_config=LLMConfig(
                model=LLMModelConfig(name="agent-model"),
            ),
        )

        provider = runtime.provider_for_agent(agent)
        assert isinstance(provider, OpenAIProvider)
        assert provider.config.api_url == "https://agency.example/v1"
        assert provider.config.api_key == "agency-secret"
        assert provider.config.model_name == "agent-model"

    def test_provider_for_session_applies_session_override_last(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ):
        monkeypatch.setenv("AGENCY_LLM_KEY", "agency-secret")
        agency_llm_config = LLMConfig(
            provider=OpenAIProviderSettings(
                type=LLMProviderType.OPENAI,
                api_url="https://agency.example/v1",
                api_key_env_var="AGENCY_LLM_KEY",
            ),
            model=LLMModelConfig(
                name="agency-model",
                options={"temperature": 0.1, "max_tokens": 4096},
            ),
        )
        runtime = Runtime(
            provider=load_provider_from_config(agency_llm_config),
            provider_config=agency_llm_config,
            workspace_root=tmp_path,
        )
        agent = Agent(
            id=AgentID("agent"),
            llm_config=LLMConfig(
                model=LLMModelConfig(
                    name="agent-model",
                    options={"reasoning_effort": "medium"},
                ),
            ),
        )
        session = Session(
            agent=agent,
            key=SessionKey("session-model"),
            llm_config=LLMConfig(
                model=LLMModelConfig(
                    name="session-model",
                    options={"temperature": 0.4},
                ),
            ),
        )

        provider = runtime.provider_for_session(session)

        assert isinstance(provider, OpenAIProvider)
        assert provider.config.api_url == "https://agency.example/v1"
        assert provider.config.api_key == "agency-secret"
        assert provider.config.model_name == "session-model"
        assert provider.config.model_options == {
            "max_tokens": 4096,
            "reasoning_effort": "medium",
            "temperature": 0.4,
        }

    @pytest.mark.asyncio
    async def test_context_exit_closes_runtime_and_cached_providers(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ):
        monkeypatch.setenv("AGENCY_LLM_KEY", "agency-secret")
        agency_llm_config = LLMConfig(
            provider=OpenAIProviderSettings(
                type=LLMProviderType.OPENAI,
                api_url="https://agency.example/v1",
                api_key_env_var="AGENCY_LLM_KEY",
            ),
            model=LLMModelConfig(name="agency-model"),
        )
        runtime = Runtime(
            provider=load_provider_from_config(agency_llm_config),
            provider_config=agency_llm_config,
            workspace_root=tmp_path,
        )
        agent = Agent(
            id=AgentID("agent"),
            llm_config=LLMConfig(model=LLMModelConfig(name="agent-model")),
        )
        runtime.provider_for_agent(agent)

        closed_model_names: list[str] = []

        async def record_close(provider: OpenAIProvider) -> None:
            closed_model_names.append(provider.config.model_name)

        monkeypatch.setattr(OpenAIProvider, "aclose", record_close)

        async with runtime:
            pass

        assert sorted(closed_model_names) == ["agency-model", "agent-model"]

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

    def test_get_or_create_session_with_workspace_root(self, tmp_path: Path):
        rt = self._make_runtime(tmp_path)
        agent = rt.create_agent(id="a1")
        ws = tmp_path / "workspaces" / "repo1"
        session = rt.get_or_create_session(agent, "s1", workspace_root=ws)
        assert session.workspace_root == ws

    def test_get_or_create_session_ignores_workspace_on_load(self, tmp_path: Path):
        """Workspace root is set only at creation time; loading an
        existing session uses the persisted value, not the new argument.
        """
        rt = self._make_runtime(tmp_path)
        agent = rt.create_agent(id="a1")
        ws_original = tmp_path / "workspaces" / "original"
        session = rt.get_or_create_session(agent, "s1", workspace_root=ws_original)
        rt.save_session(session)

        ws_different = tmp_path / "workspaces" / "different"
        reloaded = rt.get_or_create_session(agent, "s1", workspace_root=ws_different)
        assert reloaded.workspace_root == ws_original

    def test_create_context_system_prompts_are_independent(self, tmp_path: Path):
        rt = self._make_runtime(tmp_path)
        ctx1 = rt.create_context(system_prompts=["a"])
        ctx2 = rt.create_context(system_prompts=["b"])
        ctx1.system_prompts.append("c")
        assert "c" not in ctx2.system_prompts

    def test_create_context_sets_runtime_reference(self, tmp_path: Path):
        rt = self._make_runtime(tmp_path)
        ctx = rt.create_context()
        assert ctx.runtime is rt


# ---------------------------------------------------------------------------
# Runtime service registry
# ---------------------------------------------------------------------------


class TestRuntimeServiceRegistry:
    def _make_runtime(self, tmp_path: Path) -> Runtime:
        return Runtime(
            provider=MockProvider(),
            workspace_root=tmp_path,
        )

    def _make_service(self, name: str) -> "StubService":
        return StubService(name)

    def test_register_and_get_service(self, tmp_path: Path):
        rt = self._make_runtime(tmp_path)
        svc = self._make_service("my-service")
        rt.register_service(svc)
        assert rt.get_service("my-service") is svc

    def test_get_missing_service_raises_key_error(self, tmp_path: Path):
        rt = self._make_runtime(tmp_path)
        with pytest.raises(KeyError, match="no-such"):
            rt.get_service("no-such")

    def test_duplicate_registration_raises(self, tmp_path: Path):
        rt = self._make_runtime(tmp_path)
        rt.register_service(self._make_service("dup"))
        with pytest.raises(ValueError, match="already registered"):
            rt.register_service(self._make_service("dup"))

    def test_get_services_by_type(self, tmp_path: Path):
        rt = self._make_runtime(tmp_path)
        svc_a = self._make_service("a")
        svc_b = self._make_service("b")
        rt.register_service(svc_a)
        rt.register_service(svc_b)
        results = rt.get_services_by_type(StubService)
        assert set(results) == {svc_a, svc_b}

    def test_get_services_by_type_empty(self, tmp_path: Path):
        rt = self._make_runtime(tmp_path)
        assert rt.get_services_by_type(StubService) == []

    def test_empty_registry_on_creation(self, tmp_path: Path):
        rt = self._make_runtime(tmp_path)
        assert rt.get_services_by_type(StubService) == []

    def test_key_error_message_lists_registered(self, tmp_path: Path):
        rt = self._make_runtime(tmp_path)
        rt.register_service(self._make_service("alpha"))
        rt.register_service(self._make_service("beta"))
        with pytest.raises(KeyError, match="alpha.*beta"):
            rt.get_service("gamma")


class StubService(Service):
    """Minimal Service subclass for testing the registry."""

    Config = type("Config", (), {})  # type: ignore[assignment]

    def __init__(self, svc_name: str) -> None:
        self._name = svc_name

    @property
    def name(self) -> str:
        return self._name


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
    async def test_ambient_context_has_runtime_ref(self, tmp_path: Path):
        rt = self._make_runtime(tmp_path)
        async with rt:
            ctx = get_context()
            assert ctx.runtime is rt

    @pytest.mark.asyncio
    async def test_runtime_propagated_through_push_scope(self, tmp_path: Path):
        rt = self._make_runtime(tmp_path)
        async with rt:
            ctx = get_context()
            child = ctx.push_scope("test")
            assert child.runtime is rt

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
# Runtime package exports
# ---------------------------------------------------------------------------


class TestReExports:
    def test_session_not_importable_from_thorn(self):
        import thorn
        assert not hasattr(thorn, "Session")

    def test_runtime_importable_from_thorn_runtime(self):
        from thorn.runtime import (
            AgentID,
            JsonSessionSerializer,
            Runtime,
            SessionKey,
            SessionSerializer,
            SessionStore,
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
