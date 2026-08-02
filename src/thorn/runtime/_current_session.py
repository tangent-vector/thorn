"""Helpers for resolving the ambient agent session in runtime tools."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from thorn.core._context import get_context
from thorn.runtime._session import AgentID, SessionKey

if TYPE_CHECKING:
    from thorn.core._agent import Agent
    from thorn.core._session import Session
    from thorn.runtime._runtime import Runtime


@dataclass(frozen=True)
class CurrentSessionRuntime:
    """Runtime plus typed identity for the current agent session."""

    runtime: Runtime
    agent: Agent
    agent_id: AgentID
    session_key: SessionKey
    session: Session | None


def current_session_runtime(tool_label: str) -> CurrentSessionRuntime | str:
    """Resolve runtime, agent ID, and session key for an agent tool.

    Returns a user-facing error string on failure so tool bodies can
    surface configuration/scope problems directly to the model.
    """
    try:
        ctx = get_context()
    except RuntimeError:
        return (
            f"Error: no active execution context. {tool_label} must be "
            "called from within an agent prompt."
        )

    runtime = ctx.runtime
    if runtime is None:
        return f"Error: no runtime is available. {tool_label} require a Runtime."

    agent = ctx.agent
    if agent is None or agent.id is None:
        return (
            "Error: no agent is bound to the current scope. "
            f"{tool_label} require an agent."
        )

    session_key = _session_key_from_scope(ctx.scope)
    if session_key is None:
        return (
            f"Error: no session is active. {tool_label} can only be used "
            "inside a session prompt."
        )

    return CurrentSessionRuntime(
        runtime=runtime,
        agent=agent,
        agent_id=agent.id,
        session_key=session_key,
        session=_session_from_scope(ctx.scope),
    )


def _session_key_from_scope(scope: object) -> SessionKey | None:
    current = scope
    while current is not None:
        metadata = getattr(current, "metadata", {})
        key = metadata.get("session_key")
        if key is not None:
            return SessionKey(str(key))
        current = getattr(current, "outer", None)
    return None


def _session_from_scope(scope: object) -> "Session | None":
    current = scope
    while current is not None:
        metadata = getattr(current, "metadata", {})
        session = metadata.get("session")
        if session is not None:
            return session
        current = getattr(current, "outer", None)
    return None


__all__ = [
    "CurrentSessionRuntime",
    "current_session_runtime",
]
