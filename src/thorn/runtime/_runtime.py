"""Runtime: the persistent execution environment for Thorn.

The ``Runtime`` is the central service object that manages the provider,
event sink, workspace configuration, and a session store.  It produces
``ExecutionContext`` instances for individual operations and manages the
lifecycle of agent instances and their sessions.

Used as an async context manager, it sets up the ambient
``ExecutionContext`` so that ``agent.prompt()`` works automatically::

    async with runtime:
        result = await agent.prompt("do something")

Every Thorn deployment -- ``thorn run``, ``thorn chat``, or the
gateway daemon -- creates a ``Runtime``.  For one-shot ``thorn run``
the overhead is negligible: it is essentially what the CLI does today,
just structured through a uniform abstraction.
"""

from __future__ import annotations

import contextvars
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from thorn.core._agent import Agent
from thorn.core._context import (
    AskUserHandler,
    EventSink,
    ExecutionContext,
    NullEventSink,
    reset_context,
    set_context,
)
from thorn.core._provider import LLMProvider
from thorn.core._session import Session
from thorn.runtime._session import AgentID, SessionKey
from thorn.runtime._store import SessionStore

if TYPE_CHECKING:
    from thorn.core._file_access import FileAccessPolicy
    from thorn.core._validation_tracker import ValidationTracker


class Runtime:
    """Persistent execution environment for Thorn.

    Manages provider configuration, event sinks, workspace settings,
    and a session store.  Acts as a factory for ``ExecutionContext``
    instances used by the agent loop, and manages the lifecycle of
    agent instances and their sessions.

    Use as an async context manager to set the ambient
    ``ExecutionContext`` for the duration of the block::

        async with runtime:
            result = await agent.prompt("hello")
    """

    def __init__(
        self,
        *,
        provider: LLMProvider,
        event_sink: EventSink | None = None,
        workspace_root: Path,
        workspace_instructions: str | None = None,
        global_ignores: FileAccessPolicy | None = None,
        ask_user_handler: AskUserHandler | None = None,
        context_window: int | None = None,
        session_store: SessionStore | None = None,
        validation_tracker: ValidationTracker | None = None,
    ) -> None:
        self.provider = provider
        self.event_sink: EventSink = event_sink or NullEventSink()
        self.workspace_root = workspace_root
        self.workspace_instructions = workspace_instructions
        self.global_ignores = global_ignores
        self.ask_user_handler = ask_user_handler
        self.context_window = context_window
        self.validation_tracker = validation_tracker

        if session_store is None:
            agents_root = workspace_root / ".thorn" / "agents"
            session_store = SessionStore(agents_root)
        self.sessions = session_store

        self._context: ExecutionContext | None = None
        self._context_token: contextvars.Token[ExecutionContext] | None = None

    # -- Context management -------------------------------------------------

    def create_context(
        self,
        *,
        system_prompts: list[str] | None = None,
    ) -> ExecutionContext:
        """Create an ``ExecutionContext`` from this runtime's configuration.

        The context inherits the runtime's provider, event sink, workspace
        root, and other ambient settings.  An optional list of system
        prompts can be supplied for the context.
        """
        return ExecutionContext(
            provider=self.provider,
            event_sink=self.event_sink,
            workspace_root=self.workspace_root,
            workspace_instructions=self.workspace_instructions,
            global_ignores=self.global_ignores,
            ask_user_handler=self.ask_user_handler,
            context_window=self.context_window,
            system_prompts=list(system_prompts or []),
            validation_tracker=self.validation_tracker,
        )

    @property
    def context(self) -> ExecutionContext:
        """The ambient ``ExecutionContext`` set by ``async with runtime:``.

        Raises ``RuntimeError`` if the runtime is not being used as a
        context manager.
        """
        if self._context is None:
            raise RuntimeError(
                "Runtime.context is only available inside 'async with runtime:'"
            )
        return self._context

    async def __aenter__(self) -> Runtime:
        self._context = self.create_context()
        self._context_token = set_context(self._context)
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        if self._context_token is not None:
            reset_context(self._context_token)
            self._context_token = None
        self._context = None

    # -- Agent lifecycle ----------------------------------------------------

    def create_agent(
        self,
        agent_class: type[Agent] = Agent,
        *,
        id: AgentID | str | None = None,
        name: str | None = None,
        workspace: Path | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Agent:
        """Create a new agent instance with identity fields populated.

        When *id* is ``None``, a UUID-based ID is generated.
        When *name* is ``None``, the ID is used as the display name.
        When *workspace* is ``None``, a directory under the runtime's
        agents root is used.
        """
        if id is None:
            id = AgentID(str(uuid.uuid4()))
        elif not isinstance(id, AgentID):
            id = AgentID(id)

        if workspace is None:
            workspace = self.sessions.root / str(id)

        return agent_class(
            id=id,
            name=name if name is not None else str(id),
            workspace=workspace,
            metadata=metadata or {},
        )

    def get_or_create_agent(
        self,
        id: AgentID | str,
        agent_class: type[Agent] = Agent,
        *,
        name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Agent:
        """Retrieve a persisted agent, or create a new one if not found."""
        if not isinstance(id, AgentID):
            id = AgentID(id)
        if self.sessions.agent_exists(id):
            return self.sessions.load_agent(id)
        return self.create_agent(agent_class, id=id, name=name, metadata=metadata)

    def save_agent(self, agent: Agent) -> None:
        """Persist agent identity to disk."""
        self.sessions.save_agent(agent)

    # -- Session lifecycle --------------------------------------------------

    def get_or_create_session(
        self,
        agent: Agent,
        key: SessionKey | str,
    ) -> Session:
        """Retrieve a persisted session, or create a new one if not found.

        The session is scoped under the given agent instance.
        """
        if not isinstance(key, SessionKey):
            key = SessionKey(key)
        if agent.id is None:
            raise ValueError("Cannot manage sessions for an agent without an id")
        if self.sessions.session_exists(agent.id, key):
            return self.sessions.load_session(agent, key)
        now = datetime.now(timezone.utc)
        return Session(
            agent=agent,
            key=key,
            created_at=now,
            last_active=now,
        )

    def save_session(self, session: Session) -> None:
        """Persist a session, updating its ``last_active`` timestamp."""
        session.touch()
        self.sessions.save_session(session)


__all__ = [
    "Runtime",
]
