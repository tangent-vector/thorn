"""Session serialization: protocol and JSON implementation.

The ``SessionSerializer`` protocol defines how agents are persisted
to and restored from a directory on disk.  The protocol is designed to
accommodate future serialization formats (notably, a Markdown-based
format where agents can read and self-edit their own history for
compaction/summarization).

The initial ``JsonSessionSerializer`` writes human-readable, formatted
JSON -- not minified single-line output -- because the Markdown
serializer constraint requires that output be agent-editable, and the
JSON serializer should set the same expectation.

File layout within a session directory::

    <session-dir>/
        session.json      -- agent metadata and persistence fields
        history.json      -- conversation history (HistoryTree nodes)
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from thorn.core._agent import Agent
from thorn.core._history import (
    CollapseState,
    HistoryTree,
    HistoryNode,
    ToolCallNode,
    TurnNode,
    UserPromptNode,
)
from thorn.core._messages import (
    ToolCall,
    ToolResultMessage,
    UserMessage,
)
from thorn.runtime._session import SessionKey


_SESSION_FILE = "session.json"
_HISTORY_FILE = "history.json"


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class SessionSerializer(Protocol):
    """Pluggable strategy for persisting and restoring agents.

    Implementations write to / read from a directory that the
    ``SessionStore`` manages.  The directory is guaranteed to exist
    before ``save`` is called.

    The format must produce human-readable, agent-editable output
    (constraint from the future Markdown serializer goal).
    """

    def save(self, agent: Agent, directory: Path) -> None:
        """Persist *agent* into *directory*."""
        ...

    def load(self, directory: Path) -> Agent:
        """Restore an agent from *directory*."""
        ...


# ---------------------------------------------------------------------------
# HistoryTree <-> JSON helpers
# ---------------------------------------------------------------------------

def _serialize_tool_call_node(node: ToolCallNode) -> dict[str, Any]:
    return {
        "call_id": node.tool_call.call_id,
        "name": node.tool_call.name,
        "arguments": node.tool_call.arguments,
        "result_content": node.result.content,
        "result_is_error": node.result.is_error,
        "detail_collapsed": node.detail_collapsed,
        "intrinsic_salience": node.intrinsic_salience,
    }


def _serialize_node(node: HistoryNode) -> dict[str, Any]:
    if isinstance(node, UserPromptNode):
        return {
            "type": "user_prompt",
            "content": node.message.content,
            "collapse_state": node.collapse_state.value,
            "intrinsic_salience": node.intrinsic_salience,
        }
    if isinstance(node, TurnNode):
        return {
            "type": "turn",
            "assistant_content": node.assistant_content,
            "collapse_state": node.collapse_state.value,
            "intrinsic_salience": node.intrinsic_salience,
            "tool_calls": [
                _serialize_tool_call_node(tcn)
                for tcn in node.tool_call_nodes
            ],
        }
    raise TypeError(f"Unknown node type: {type(node).__name__}")


def _deserialize_tool_call_node(data: dict[str, Any]) -> ToolCallNode:
    tc = ToolCall(
        call_id=data["call_id"],
        name=data["name"],
        arguments=data["arguments"],
    )
    result = ToolResultMessage(
        call_id=data["call_id"],
        content=data["result_content"],
        is_error=data.get("result_is_error", False),
    )
    node = ToolCallNode(
        tc,
        result,
        intrinsic_salience=data.get("intrinsic_salience", 0.8),
    )
    node.detail_collapsed = data.get("detail_collapsed", False)
    return node


def _deserialize_node(data: dict[str, Any]) -> HistoryNode:
    node_type = data["type"]

    if node_type == "user_prompt":
        node = UserPromptNode(
            UserMessage(content=data["content"]),
            intrinsic_salience=data.get("intrinsic_salience", 1.0),
        )
        state = data.get("collapse_state", "expanded")
        node.collapse_state = CollapseState(state)
        return node

    if node_type == "turn":
        tool_call_nodes = [
            _deserialize_tool_call_node(tc_data)
            for tc_data in data.get("tool_calls", [])
        ]
        node = TurnNode(
            assistant_content=data.get("assistant_content", ""),
            tool_call_nodes=tool_call_nodes,
            intrinsic_salience=data.get("intrinsic_salience", 1.0),
        )
        state = data.get("collapse_state", "expanded")
        node.collapse_state = CollapseState(state)
        return node

    raise ValueError(f"Unknown history node type: {node_type!r}")


def serialize_history(history: HistoryTree) -> list[dict[str, Any]]:
    """Convert a ``HistoryTree`` to a JSON-serializable list of dicts."""
    return [_serialize_node(node) for node in history.nodes]


def deserialize_history(data: list[dict[str, Any]]) -> HistoryTree:
    """Reconstruct a ``HistoryTree`` from its serialized representation."""
    tree = HistoryTree()
    tree.nodes = [_deserialize_node(item) for item in data]
    return tree


# ---------------------------------------------------------------------------
# Agent class resolution
# ---------------------------------------------------------------------------

def _resolve_agent_class(class_name: str) -> type[Agent]:
    """Look up an Agent subclass by name, falling back to base Agent."""
    if class_name == "Agent":
        return Agent
    cls = Agent._registry.get(class_name)
    if cls is not None:
        return cls
    return Agent


# ---------------------------------------------------------------------------
# JSON serializer
# ---------------------------------------------------------------------------

class JsonSessionSerializer:
    """Persists agents as formatted, human-readable JSON files.

    Produces two files in the session directory:

    - ``session.json``: agent metadata and persistence fields
    - ``history.json``: the full conversation history tree
    """

    def save(self, agent: Agent, directory: Path) -> None:
        session_data: dict[str, Any] = {
            "key": str(agent.key) if agent.key is not None else None,
            "agent_class": type(agent).__name__,
            "agent_name": agent.name,
            "agent_metadata": agent.metadata,
            "created_at": agent.created_at.isoformat() if agent.created_at else None,
            "last_active": agent.last_active.isoformat() if agent.last_active else None,
        }
        session_path = directory / _SESSION_FILE
        session_path.write_text(
            json.dumps(session_data, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        history_data = serialize_history(agent._history)
        history_path = directory / _HISTORY_FILE
        history_path.write_text(
            json.dumps(history_data, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    def load(self, directory: Path) -> Agent:
        session_path = directory / _SESSION_FILE
        session_data = json.loads(session_path.read_text(encoding="utf-8"))

        agent_cls = _resolve_agent_class(session_data.get("agent_class", "Agent"))

        key_raw = session_data.get("key")
        key = SessionKey(key_raw) if key_raw is not None else None

        created_raw = session_data.get("created_at")
        created_at = datetime.fromisoformat(created_raw) if created_raw else None

        last_raw = session_data.get("last_active")
        last_active = datetime.fromisoformat(last_raw) if last_raw else None

        agent = agent_cls(
            name=session_data.get("agent_name"),
            metadata=session_data.get("agent_metadata"),
            key=key,
            created_at=created_at,
            last_active=last_active,
        )

        history_path = directory / _HISTORY_FILE
        if history_path.exists():
            history_data = json.loads(history_path.read_text(encoding="utf-8"))
            agent._history = deserialize_history(history_data)

        return agent


__all__ = [
    "JsonSessionSerializer",
    "SessionSerializer",
    "deserialize_history",
    "serialize_history",
]
