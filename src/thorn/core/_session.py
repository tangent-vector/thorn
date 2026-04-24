"""Session: a conversation thread owned by an agent instance.

A ``Session`` carries the ``HistoryTree``, timestamps, and session-scoped
metadata for a single conversation thread.  Multiple sessions can exist
under one ``Agent`` instance (e.g., one per GitLab issue).

Sessions are the unit of conversation persistence; agent instances are
the unit of identity persistence.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from thorn.core._agent import Agent
    from thorn.core._context_injection import SeedContent
    from thorn.core._file_access import FileAccessRule
    from thorn.runtime._session import SessionKey


class Session:
    """A conversation thread owned by a specific agent instance.

    Owns the ``HistoryTree``, timestamps, and session-scoped metadata.
    References the ``Agent`` instance it belongs to for persona
    configuration (system prompts, tools, file access, context seeds).

    The ``prompt`` accessor works identically to the ``Agent.prompt``
    accessor — ``session.prompt("text")`` and ``session.prompt[T]("text")``
    both work.

    Attributes:
        workspace_root: When set, overrides both ``agent.workspace`` and
            the ambient ``ExecutionContext.workspace_root`` for this
            session's turns.  Used for file tools, ``AGENTS.md``
            loading, and ``RelativeTo.WORKSPACE`` policy resolution.
            Immutable after the session's first persisted turn —
            subsequent events reuse the stored value.
        logical_agent_workspace_path: The upper bound of the
            workspace-side context-gathering walk for this session's
            prompts.  In gateway mode this is the agent's
            ``workspace`` mount (one level above every session
            workspace under that agent); in CLI mode it is the
            outermost enclosing project directory of the session
            workspace, picked at startup by
            :func:`thorn.runtime._project_detection.pick_logical_agent_workspace_path_for_cli_session`.
            Carried per-session because a single CLI agent identity
            may serve sessions rooted in different logical projects
            (the missing ``agency -> agent -> ??? -> session`` rung
            from the design plan, which is *deferred* — we just
            smuggle the value through the Session for now).
    """

    def __init__(
        self,
        *,
        agent: Agent,
        key: SessionKey | None = None,
        created_at: datetime | None = None,
        last_active: datetime | None = None,
        metadata: dict[str, Any] | None = None,
        workspace_root: Path | None = None,
        logical_agent_workspace_path: Path | None = None,
    ) -> None:
        from thorn.core._history import HistoryTree

        self.agent = agent
        self.key: SessionKey | None = key
        self._history: HistoryTree = HistoryTree()
        self.created_at: datetime | None = created_at
        self.last_active: datetime | None = last_active
        self.metadata: dict[str, Any] = metadata if metadata is not None else {}
        self.workspace_root: Path | None = workspace_root
        self.logical_agent_workspace_path: Path | None = (
            logical_agent_workspace_path
        )

    def touch(self) -> None:
        """Update ``last_active`` to the current UTC time."""
        self.last_active = datetime.now(timezone.utc)

    @property
    def prompt(self) -> _SessionPromptAccessor:
        """Accessor for ``session.prompt("...")`` and ``session.prompt[T]("...")``."""
        return _SessionPromptAccessor(self)


class _SessionPromptAccessor:
    """Provides ``session.prompt("...")`` (text) and ``session.prompt[T]("...")`` (structured)."""

    __slots__ = ("_session",)

    def __init__(self, session: Session) -> None:
        self._session = session

    def __getitem__(self, result_type: type) -> _TypedSessionPrompt:
        return _TypedSessionPrompt(self._session, result_type)

    async def __call__(
        self,
        text: str,
        *,
        tools: list[Any] | None = None,
        system: str | None = None,
        file_access: list[FileAccessRule] | None = None,
        recommended_context: list[SeedContent] | None = None,
    ) -> str:
        from thorn.core._agent import _run_session_prompt

        return await _run_session_prompt(
            session=self._session,
            text=text,
            result_type=str,
            extra_tools=tools,
            extra_system=system,
            extra_file_access=file_access,
            recommended_context=recommended_context,
        )


class _TypedSessionPrompt:
    """Callable returned by ``session.prompt[T]`` for structured results."""

    __slots__ = ("_session", "_result_type")

    def __init__(self, session: Session, result_type: type) -> None:
        self._session = session
        self._result_type = result_type

    async def __call__(
        self,
        text: str,
        *,
        tools: list[Any] | None = None,
        system: str | None = None,
        file_access: list[FileAccessRule] | None = None,
        recommended_context: list[SeedContent] | None = None,
    ) -> Any:
        from thorn.core._agent import _run_session_prompt

        return await _run_session_prompt(
            session=self._session,
            text=text,
            result_type=self._result_type,
            extra_tools=tools,
            extra_system=system,
            extra_file_access=file_access,
            recommended_context=recommended_context,
        )
