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

    <agent-id>.json       -- agent class, name, metadata, id, accounts
    <session-dir>/
        session.json      -- session timestamps, metadata
        history.json      -- conversation history (HistoryTree nodes)
"""

from __future__ import annotations

import json
import logging
import os
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

# Prefix used for sidecar temp files during atomic writes.  Matches
# the convention used by :class:`DurableQueue` so that both systems
# surface their in-flight writes in recognisably named files.
_ATOMIC_TEMP_PREFIX = ".tmp-"


def _atomic_write_text(path: Path, payload: str) -> None:
    """Write *payload* to *path* atomically.

    Uses the write-to-sidecar + :func:`os.replace` pattern: the
    target is first written in full to ``<dir>/.tmp-<name>``, then
    replaced into place.  Guarantees:

    - A crash (or task cancellation) before the ``os.replace`` call
      leaves the previous contents of *path* intact.  The partial
      sidecar is orphaned but recoverable by operator cleanup or a
      future sweep; importantly, any reader seeing *path* sees either
      the old full value or the new full value, never a truncation.
    - A crash during the ``os.replace`` call is not possible on POSIX
      or Windows: the rename is atomic.

    Used for both session/history files and agent-identity files so
    that graceful-shutdown scenarios (cancel mid-save) cannot leave
    the store in a torn state.  The sidecar is placed in the same
    directory as *path* so that the rename is guaranteed to be on
    the same filesystem.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.parent / f"{_ATOMIC_TEMP_PREFIX}{path.name}"
    temp_path.write_text(payload, encoding="utf-8")
    os.replace(temp_path, path)


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


log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Agent accounts serialization
# ---------------------------------------------------------------------------


def _deserialize_accounts(
    raw_accounts: list[dict[str, Any]] | dict[str, Any],
) -> Any:
    """Parse the ``"accounts"`` value from an agent JSON file.

    The on-disk shape is a flat JSON array of account objects, each
    of which has a ``service`` discriminator and a ``credentials``
    list.  This helper performs only *parse-time* validation: each
    entry is wrapped in an :class:`UntypedAccountConfig` so that
    per-service fields survive intact for the gateway's
    validation pass to consume against the right typed
    :class:`AccountConfig` subclass.

    No environment-variable expansion happens here: credentials in
    the new shape carry an explicit ``env_var_name`` field and the
    literal value is read from ``os.environ`` only at the points
    where it is actually needed (broker registration, direct
    authentication).
    """
    from thorn.core._account import AgentAccountsConfig, UntypedAccountConfig

    if not isinstance(raw_accounts, list):
        raise ValueError(
            "Agent JSON 'accounts' must be a JSON array of account "
            f"objects, got {type(raw_accounts).__name__}.  See the "
            "agent identity-file documentation for the expected shape."
        )

    # Build UntypedAccountConfig instances explicitly: the
    # AgentAccountsConfig.accounts field is typed as list[AccountConfig]
    # so per-service fields would otherwise be stripped by base-class
    # validation.  The gateway's validation pass replaces each of
    # these with a typed concrete AccountConfig subclass once the
    # services are known.
    typed_entries = [
        UntypedAccountConfig.model_validate(entry) for entry in raw_accounts
    ]
    # Bypass the field-level revalidation that would otherwise downgrade
    # each UntypedAccountConfig back to its declared base shape and
    # strip the per-service ``extra`` fields.  ``model_construct``
    # writes the list straight through.
    return AgentAccountsConfig.model_construct(accounts=typed_entries)


def _serialize_accounts(agent: Agent) -> list[dict[str, Any]] | None:
    """Serialize the agent's accounts to a JSON-safe list.

    Returns ``None`` when the agent has no accounts configured, so
    the key can be omitted from the output JSON entirely.

    Each entry is dumped through its own concrete model (rather
    than via the container's ``model_dump``) so that per-service
    extras carried by an :class:`UntypedAccountConfig` -- and
    extras carried by typed subclasses with ``model_config =
    extra="allow"`` -- survive the round-trip.  Dumping through
    the parent ``AgentAccountsConfig.accounts: list[AccountConfig]``
    strips extras, because Pydantic only serialises fields it
    knows about on the declared element type.

    Each credential entry's ``env_var_name`` (and other fields) is
    written through unchanged; literal secret values are never on
    the agent state, so there is nothing to redact at serialization
    time either.
    """
    from thorn.core._account import AgentAccountsConfig

    accounts: AgentAccountsConfig | None = getattr(agent, "accounts", None)
    if accounts is None or not accounts.accounts:
        return None
    return [entry.model_dump(mode="json") for entry in accounts.accounts]


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
        """Persist agent identity atomically.

        Writes the agent JSON via :func:`_atomic_write_text` so a
        mid-write crash or cancellation leaves any previous
        identity file intact.  Callers can therefore rely on the
        post-crash state always being a complete, loadable agent
        record (either the previous one or the new one, never
        torn).

        The agent's ``id`` is intentionally **not** written: the
        identity file's path stem encodes it via
        :func:`safe_dirname`, and on load the ``AgentID`` is derived
        from that stem.  This collapses the historical ``id`` /
        ``name`` duplication into the single human-facing ``name``.
        """
        agent_data: dict[str, Any] = {
            "name": agent.name,
            "agent_class": type(agent).__name__,
            "metadata": agent.metadata,
        }
        accounts_data = _serialize_accounts(agent)
        if accounts_data is not None:
            agent_data["accounts"] = accounts_data
        # Sandbox overrides ride alongside accounts as an optional
        # ``sandbox`` block.  Only written when the agent actually
        # carries an override; absent block means "use the agency
        # default".
        sandbox_override = getattr(agent, "sandbox_override", None)
        if sandbox_override is not None:
            sandbox_dict = sandbox_override.model_dump(
                mode="json", exclude_none=True,
            )
            if sandbox_dict:
                agent_data["sandbox"] = sandbox_dict
        llm_config = getattr(agent, "llm_config", None)
        if llm_config is not None:
            from thorn.core._provider import LLMConfig
            if not isinstance(llm_config, LLMConfig):
                llm_config = LLMConfig.model_validate(llm_config)
            llm_dict = llm_config.model_dump(mode="json", exclude_none=True)
            if llm_dict:
                agent_data["llm"] = llm_dict
        _atomic_write_text(
            path,
            json.dumps(agent_data, indent=2, ensure_ascii=False) + "\n",
        )

    def load_agent(self, path: Path) -> Agent:
        from thorn.runtime._paths import unsafe_dirname

        agent_data = json.loads(path.read_text(encoding="utf-8"))

        agent_cls = _resolve_agent_class(agent_data.get("agent_class", "Agent"))

        # Derive the AgentID from the parent directory name so that
        # the store's directory layout is the single source of truth
        # for agent identity.  Under the Phase-A layout each agent
        # lives at ``<agents_root>/<safe-agent-id>/agent.json``, so
        # the containing directory name carries the ID.  When the
        # JSON also carries an explicit ``name``, prefer that for the
        # human label; otherwise fall back to the path-derived value.
        path_id = AgentID(unsafe_dirname(path.parent.name))
        name = agent_data.get("name") or str(path_id)

        metadata = agent_data.get("metadata") or {}

        kwargs: dict[str, Any] = {}
        raw_accounts = agent_data.get("accounts")
        if raw_accounts is not None:
            kwargs["accounts"] = _deserialize_accounts(raw_accounts)

        raw_sandbox = agent_data.get("sandbox")
        if raw_sandbox is not None:
            from thorn.gateway._config import AgentSandboxOverride
            kwargs["sandbox_override"] = AgentSandboxOverride.model_validate(
                raw_sandbox,
            )

        raw_llm = agent_data.get("llm")
        if raw_llm is not None:
            from thorn.core._provider import LLMConfig
            kwargs["llm_config"] = LLMConfig.model_validate(raw_llm)

        return agent_cls(
            id=path_id,
            name=name,
            metadata=metadata,
            **kwargs,
        )

    def save_session(self, session: Session, directory: Path) -> None:
        """Persist session metadata and history atomically.

        Each of ``session.json`` and ``history.json`` is written
        via :func:`_atomic_write_text`, so a crash or cancellation
        mid-save leaves both files at their previous contents (or
        both at the new contents, if the save completed).  The two
        files are not collectively atomic -- a crash between the
        two replaces could leave ``session.json`` updated but
        ``history.json`` still on the previous version -- which is
        acceptable because neither file's contents depend on the
        other's value for correctness (``session.json`` carries
        timestamps; ``history.json`` carries the conversation
        tree).  The guarantee we need for graceful shutdown is
        that each file is internally well-formed, and that holds.
        """
        directory.mkdir(parents=True, exist_ok=True)

        session_data: dict[str, Any] = {
            "key": str(session.key) if session.key is not None else None,
            "created_at": session.created_at.isoformat() if session.created_at else None,
            "last_active": session.last_active.isoformat() if session.last_active else None,
            "metadata": session.metadata,
        }
        if session.workspace_root is not None:
            session_data["workspace_root"] = str(session.workspace_root)
        if session.logical_agent_workspace_path is not None:
            session_data["logical_agent_workspace_path"] = str(
                session.logical_agent_workspace_path,
            )
        llm_config = getattr(session, "llm_config", None)
        if llm_config is not None:
            from thorn.core._provider import LLMConfig
            if not isinstance(llm_config, LLMConfig):
                llm_config = LLMConfig.model_validate(llm_config)
            llm_dict = llm_config.model_dump(mode="json", exclude_none=True)
            if llm_dict:
                session_data["llm"] = llm_dict
        _atomic_write_text(
            directory / _SESSION_FILE,
            json.dumps(session_data, indent=2, ensure_ascii=False) + "\n",
        )

        history_data = serialize_history(session._history)
        _atomic_write_text(
            directory / _HISTORY_FILE,
            json.dumps(history_data, indent=2, ensure_ascii=False) + "\n",
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

        logical_ws_raw = session_data.get("logical_agent_workspace_path")
        logical_agent_workspace_path = (
            Path(logical_ws_raw) if logical_ws_raw is not None else None
        )

        raw_llm = session_data.get("llm")
        llm_config = None
        if raw_llm is not None:
            from thorn.core._provider import LLMConfig
            llm_config = LLMConfig.model_validate(raw_llm)

        session = Session(
            agent=agent,
            key=key,
            created_at=created_at,
            last_active=last_active,
            metadata=session_data.get("metadata", {}),
            workspace_root=workspace_root,
            logical_agent_workspace_path=logical_agent_workspace_path,
            llm_config=llm_config,
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
