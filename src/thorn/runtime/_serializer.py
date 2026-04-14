"""Session serialization: protocol and JSON implementation.

The ``SessionSerializer`` protocol defines how agents and sessions are
persisted to and restored from disk.  The protocol is designed to
accommodate future serialization formats (notably, a Markdown-based
format where agents can read and self-edit their own history for
compaction/summarization).

The initial ``JsonSessionSerializer`` writes human-readable, formatted
JSON -- not minified single-line output -- because the Markdown
serializer constraint requires that output be agent-editable, and the
JSON serializer should set the same expectation.

Agent identity is stored separately from session data:

    <agent-id>.json       -- agent class, name, metadata, id
    <session-dir>/
        session.json      -- session timestamps, metadata
        history.json      -- conversation history (HistoryTree nodes)
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from thorn.core._agent import Agent
from thorn.core._history import (
    AdvisoryNode,
    ArchiveMarkerNode,
    CollapseState,
    HistoryNode,
    HistoryTree,
    HousekeepingNode,
    ToolCallNode,
    TurnNode,
    UserPromptNode,
)
from thorn.core._messages import (
    ToolCall,
    ToolResultMessage,
    UserMessage,
)
from thorn.core._session import Session
from thorn.runtime._session import AgentID, SessionKey


_SESSION_FILE = "session.json"
_HISTORY_FILE = "history.json"


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class SessionSerializer(Protocol):
    """Pluggable strategy for persisting and restoring agents and sessions.

    Implementations write to / read from paths and directories that the
    ``SessionStore`` manages.  Directories are guaranteed to exist
    before ``save_session`` is called.

    The format must produce human-readable, agent-editable output
    (constraint from the future Markdown serializer goal).
    """

    def save_agent(self, agent: Agent, path: Path) -> None:
        """Persist agent identity to a JSON file at *path*."""
        ...

    def load_agent(self, path: Path) -> Agent:
        """Restore agent identity from *path*."""
        ...

    def save_session(self, session: Session, directory: Path) -> None:
        """Persist *session* into *directory*."""
        ...

    def load_session(self, directory: Path, agent: Agent) -> Session:
        """Restore a session from *directory*, owned by *agent*."""
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


def _serialize_advisory_node(node: AdvisoryNode) -> dict[str, Any]:
    return {
        "source": node.source,
        "content": node.content,
        "collapse_state": node.collapse_state.value,
        "intrinsic_salience": node.intrinsic_salience,
    }


def _deserialize_advisory_node(data: dict[str, Any]) -> AdvisoryNode:
    state = data.get("collapse_state", "expanded")
    return AdvisoryNode(
        source=data["source"],
        content=data["content"],
        collapse_state=CollapseState(state),
        intrinsic_salience=data.get("intrinsic_salience", 0.2),
    )


def _serialize_node(node: HistoryNode) -> dict[str, Any]:
    if isinstance(node, UserPromptNode):
        return {
            "type": "user_prompt",
            "content": node.message.content,
            "collapse_state": node.collapse_state.value,
            "intrinsic_salience": node.intrinsic_salience,
        }
    if isinstance(node, TurnNode):
        result: dict[str, Any] = {
            "type": "turn",
            "assistant_content": node.assistant_content,
            "collapse_state": node.collapse_state.value,
            "intrinsic_salience": node.intrinsic_salience,
            "tool_calls": [
                _serialize_tool_call_node(tcn)
                for tcn in node.tool_call_nodes
            ],
        }
        if node.advisory_nodes:
            result["advisories"] = [
                _serialize_advisory_node(adv)
                for adv in node.advisory_nodes
            ]
        return result
    if isinstance(node, ArchiveMarkerNode):
        return {
            "type": "archive_marker",
            "archived_at": node.archived_at.isoformat(),
            "summary": node.summary,
            "node_count": node.node_count,
            "journal_date": node.journal_date,
        }
    if isinstance(node, HousekeepingNode):
        return {
            "type": "housekeeping",
            "inner_nodes": [_serialize_node(inner) for inner in node.inner_nodes],
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
        advisory_nodes = [
            _deserialize_advisory_node(adv_data)
            for adv_data in data.get("advisories", [])
        ]
        node = TurnNode(
            assistant_content=data.get("assistant_content", ""),
            tool_call_nodes=tool_call_nodes,
            advisory_nodes=advisory_nodes or None,
            intrinsic_salience=data.get("intrinsic_salience", 1.0),
        )
        state = data.get("collapse_state", "expanded")
        node.collapse_state = CollapseState(state)
        return node

    if node_type == "archive_marker":
        return ArchiveMarkerNode(
            archived_at=datetime.fromisoformat(data["archived_at"]),
            summary=data["summary"],
            node_count=data["node_count"],
            journal_date=data["journal_date"],
        )

    if node_type == "housekeeping":
        inner = [_deserialize_node(d) for d in data.get("inner_nodes", [])]
        return HousekeepingNode(inner_nodes=inner)

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
    """Persists agents and sessions as formatted, human-readable JSON files.

    Agent identity is stored as a single JSON file (agent class, name,
    metadata, ID).  Session data is stored in a directory with two files
    (``session.json`` for timestamps/metadata, ``history.json`` for the
    conversation history tree).
    """

    def save_agent(self, agent: Agent, path: Path) -> None:
        agent_data: dict[str, Any] = {
            "id": str(agent.id) if agent.id is not None else None,
            "agent_class": type(agent).__name__,
            "name": agent.name,
            "metadata": agent.metadata,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(agent_data, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    def load_agent(self, path: Path) -> Agent:
        agent_data = json.loads(path.read_text(encoding="utf-8"))

        agent_cls = _resolve_agent_class(agent_data.get("agent_class", "Agent"))

        id_raw = agent_data.get("id")
        agent_id = AgentID(id_raw) if id_raw is not None else None

        return agent_cls(
            id=agent_id,
            name=agent_data.get("name"),
            metadata=agent_data.get("metadata"),
        )

    def save_session(self, session: Session, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)

        session_data: dict[str, Any] = {
            "key": str(session.key) if session.key is not None else None,
            "created_at": session.created_at.isoformat() if session.created_at else None,
            "last_active": session.last_active.isoformat() if session.last_active else None,
            "metadata": session.metadata,
        }
        if session.workspace_root is not None:
            session_data["workspace_root"] = str(session.workspace_root)
        session_path = directory / _SESSION_FILE
        session_path.write_text(
            json.dumps(session_data, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        history_data = serialize_history(session._history)
        history_path = directory / _HISTORY_FILE
        history_path.write_text(
            json.dumps(history_data, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    def load_session(self, directory: Path, agent: Agent) -> Session:
        session_path = directory / _SESSION_FILE
        session_data = json.loads(session_path.read_text(encoding="utf-8"))

        key_raw = session_data.get("key")
        key = SessionKey(key_raw) if key_raw is not None else None

        created_raw = session_data.get("created_at")
        created_at = datetime.fromisoformat(created_raw) if created_raw else None

        last_raw = session_data.get("last_active")
        last_active = datetime.fromisoformat(last_raw) if last_raw else None

        ws_raw = session_data.get("workspace_root")
        workspace_root = Path(ws_raw) if ws_raw is not None else None

        session = Session(
            agent=agent,
            key=key,
            created_at=created_at,
            last_active=last_active,
            metadata=session_data.get("metadata", {}),
            workspace_root=workspace_root,
        )

        history_path = directory / _HISTORY_FILE
        if history_path.exists():
            history_data = json.loads(history_path.read_text(encoding="utf-8"))
            session._history = deserialize_history(history_data)

        return session


__all__ = [
    "JsonSessionSerializer",
    "SessionSerializer",
    "deserialize_history",
    "serialize_history",
]
