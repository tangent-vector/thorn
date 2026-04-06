"""Runtime: the persistent execution environment for Thorn.

The ``Runtime`` is the central service object that replaces the ad-hoc
``ExecutionContext`` construction scattered across CLI commands.  It
holds the provider, event sink, workspace configuration, and a session
store, and can produce ``ExecutionContext`` instances for individual
operations.

Used as an async context manager, it sets up the ambient
``ExecutionContext`` so that ``agent.prompt()`` works automatically::

    async with runtime:
        result = await agent.prompt("do something")

Every Thorn deployment -- ``thorn run``, ``thorn chat``, or a future
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
from thorn.runtime._session import SessionKey
from thorn.runtime._store import SessionStore

if TYPE_CHECKING:
    from thorn.core._file_access import FileAccessPolicy
    from thorn.core._validation_tracker import ValidationTracker


class Runtime:
    """Persistent execution environment for Thorn.

    Manages provider configuration, event sinks, workspace settings,
    and a session store.  Acts as a factory for ``ExecutionContext``
    instances used by the agent loop.

    Use as an async context manager to set the ambient
    ``ExecutionContext`` for the duration of the block::

        async with runtime:
            result = await agent.prompt("hello")

    Attributes:
        provider: LLM provider for completion requests.
        event_sink: Sink that receives streaming execution events.
        workspace_root: Resolved absolute path to the workspace.
        workspace_instructions: Contents of the workspace ``AGENTS.md``
            file, if any.
        global_ignores: Ceiling policy from ``.aiignore`` /
            ``.thornignore``.
        sessions: Filesystem-backed session store.
        ask_user_handler: Callback for the ``ask_user`` tool.
        context_window: Effective context window budget in tokens.
        validation_tracker: Shared validation status tracker.
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
            sessions_root = workspace_root / ".thorn" / "sessions"
            session_store = SessionStore(sessions_root)
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
        key: SessionKey | str | None = None,
        *,
        name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Agent:
        """Create a new agent with persistence fields populated.

        When *key* is ``None``, a UUID-based key is generated.
        When *name* is ``None``, the key is used as the display name.
        """
        if key is None:
            key = SessionKey(str(uuid.uuid4()))
        elif not isinstance(key, SessionKey):
            key = SessionKey(key)

        now = datetime.now(timezone.utc)
        return Agent(
            key=key,
            name=name if name is not None else str(key),
            metadata=metadata or {},
            created_at=now,
            last_active=now,
        )

    def get_or_create_agent(
        self,
        key: SessionKey | str,
        *,
        name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Agent:
        """Retrieve a persisted agent, or create a new one if not found."""
        if not isinstance(key, SessionKey):
            key = SessionKey(key)
        if self.sessions.exists(key):
            return self.sessions.load(key)
        return self.create_agent(key, name=name, metadata=metadata)

    def save_agent(self, agent: Agent) -> None:
        """Persist an agent, updating its ``last_active`` timestamp."""
        agent.touch()
        self.sessions.save(agent)


__all__ = [
    "Runtime",
]
