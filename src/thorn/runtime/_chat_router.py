"""Chat-style multi-turn prompt routing through :class:`AgentScheduler`.

Phase 4 of the CLI/gateway unification routes the ``thorn chat`` REPL
through :class:`~thorn.runtime._scheduler.AgentScheduler`.  Each user
input becomes a notification posted to the session's inbox; the
scheduler invokes a :data:`~thorn.runtime._scheduler.PromptDispatcher`
that calls ``session.prompt(...)``; the dispatcher resolves a per-turn
:class:`asyncio.Future` so the REPL can ``await`` the agent's reply
synchronously.

This module provides :class:`ChatPromptRouter`, which owns the
per-turn future queue and exposes a
:data:`~thorn.runtime._scheduler.PromptDispatcher` for the scheduler
plus a high-level :meth:`turn` method for the REPL.

Why a class with a queue rather than a per-turn dispatcher
----------------------------------------------------------

:class:`AgentScheduler` binds its ``prompt_dispatcher`` at construction
time: the scheduler runs for the whole REPL session and we cannot swap
the dispatcher between turns.  We also cannot reuse
:func:`~thorn.runtime._cli_dispatcher.make_cli_prompt_dispatcher`
directly because it is one-shot -- its captured future fires exactly
once.

The natural shape for "fixed dispatcher, fresh result-future per turn"
is a small router object that carries a FIFO queue of pending futures.
:meth:`turn` enqueues a future, posts the notification, and awaits the
future.  The dispatcher (called once per round by the scheduler) pops
the next future and resolves it with the round's result.  The queue is
unbounded but in practice never holds more than one entry: the REPL
awaits its own future before reading the next input, so producers are
serialised at the user-input step.

Why catch ``SkillError`` / ``ThornError`` inside the dispatcher
---------------------------------------------------------------

The pre-Phase-4 chat REPL called ``runtime.save_session`` after every
turn regardless of whether the prompt raised, because
``session.prompt`` mutates history in place even on failure.
:class:`AgentScheduler` skips its ``save_session`` callback when the
dispatcher raises.  To preserve the save-on-error behaviour we catch
the recoverable Thorn exceptions inside the dispatcher, hand them to
the awaiting future via :meth:`asyncio.Future.set_exception`, and
return normally so the scheduler still saves.  Cancellation
(``asyncio.CancelledError``) and other non-recoverable exceptions
propagate so shutdown semantics are unaffected.

**Open question.**  The exception classes this module treats as
"recoverable, still save" are hard-coded (``SkillError`` and the
broader ``ThornError`` base).  A different driver -- e.g. a future
IPC client that wants every non-cancellation error to be reportable
without losing history -- would want to extend or replace that set.
The right configurability shape is not yet obvious (a constructor
parameter? a subclass override? a broader contract on the scheduler
itself about when to save?), so the policy is left as-is here and
flagged for revisit once a second caller actually shows up.

Mismatch between queued futures and dispatched rounds
-----------------------------------------------------

A round dispatched without a queued future (e.g. an autonomously
enqueued notification, which the chat REPL never produces today but
the inbox model permits) is processed normally; the result is silently
discarded and no future is touched.  A queued future without a matching
notification (e.g. a ``turn`` call whose post failed before reaching
the inbox) hangs the awaiter forever and is a programming error in
``turn``; the implementation here is careful to enqueue the future
*before* posting and to never short-circuit past the post.
"""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from typing import TYPE_CHECKING, Any

from thorn.core.errors import SkillError, ThornError
from thorn.runtime._inbox import SessionInbox
from thorn.runtime._notification import NotificationSpec
from thorn.runtime._scheduler import AgentScheduler, PromptDispatcher

if TYPE_CHECKING:
    from thorn.core._session import Session
    from thorn.runtime._address import SessionAddress


log = logging.getLogger(__name__)


class ChatPromptRouter:
    """Route an interactive REPL's turns through an :class:`AgentScheduler`.

    Construct once per REPL session.  Hand :attr:`dispatcher` to the
    :class:`AgentScheduler` constructor, then call :meth:`turn` once
    per user input.

    *target* is the :class:`SessionAddress` notifications will be
    posted to; typically the address of the chat session's own inbox.

    *extra_tools* and *extra_system* are forwarded to
    ``session.prompt(...)`` on every turn, mirroring
    :func:`~thorn.runtime._cli_dispatcher.make_cli_prompt_dispatcher`.

    *source* is the notification ``source`` field, defaulting to
    ``"user"`` since the REPL turn is the human's input.  Exposed as
    a parameter so test scaffolding can override it.
    """

    def __init__(
        self,
        *,
        target: "SessionAddress",
        extra_tools: list[Any] | None = None,
        extra_system: str | None = None,
        source: str = "user",
    ) -> None:
        self._target = target
        self._extra_tools = extra_tools
        self._extra_system = extra_system
        self._source = source
        # Producers (``turn``) append to the right; the dispatcher
        # pops from the left.  A ``deque`` is sufficient because the
        # router never blocks on enqueue (the queue is unbounded) and
        # never blocks on dequeue (a missing future is treated as a
        # benign autonomous round; see module docstring).
        self._pending: deque[asyncio.Future[Any]] = deque()

    @property
    def dispatcher(self) -> PromptDispatcher:
        """The :data:`PromptDispatcher` to wire into :class:`AgentScheduler`."""
        return self._dispatch

    @property
    def pending_count(self) -> int:
        """Number of futures awaiting a round result (test helper)."""
        return len(self._pending)

    async def turn(
        self,
        *,
        scheduler: AgentScheduler,
        session: "Session",
        inbox: SessionInbox,
        prompt_text: str,
    ) -> Any:
        """Post *prompt_text* and await the matching round's result.

        Order of operations matters:

        1. Create the result future and enqueue it.  This must happen
           before the post so that the dispatcher (which may run as
           soon as the post completes if the scheduler's drain task is
           already idle) can pop the future on its first round.
        2. Post the notification.
        3. Kick the scheduler.  ``submit`` is idempotent and
           inexpensive; calling it on every turn keeps the chat REPL
           agnostic to whether the driver is freshly registered or
           long-lived.
        4. Await the future.  The dispatcher resolves it via
           :meth:`asyncio.Future.set_result` on success or
           :meth:`asyncio.Future.set_exception` on a recoverable
           failure; in either case the awaiter unblocks here.
        """
        loop = asyncio.get_running_loop()
        result_future: asyncio.Future[Any] = loop.create_future()
        self._pending.append(result_future)
        try:
            inbox.post(NotificationSpec(
                source=self._source,
                content=prompt_text,
                target=self._target,
            ))
        except BaseException:
            # We never reached the dispatcher, so nothing will ever pop
            # this future.  Drop it eagerly to keep the queue accurate
            # for the next turn.
            try:
                self._pending.remove(result_future)
            except ValueError:
                pass
            raise
        await scheduler.submit(session, inbox)
        return await result_future

    async def _dispatch(
        self,
        session: "Session",
        inbox: SessionInbox,
    ) -> None:
        """The :data:`PromptDispatcher` the scheduler invokes per round.

        See the module docstring for the rationale on auto-deleting
        the processed item and on swallowing recoverable Thorn
        exceptions to preserve save-on-error semantics.
        """
        items = inbox.prompt_pending()
        if not items:
            log.debug(
                "ChatPromptRouter: no items pending for agent=%r session=%r",
                session.agent.id, session.key,
            )
            return
        item = items[0]
        result_future = self._pending.popleft() if self._pending else None
        try:
            result = await session.prompt(
                item.content,
                tools=self._extra_tools,
                system=self._extra_system,
            )
        except (SkillError, ThornError) as exc:
            # Recoverable: hand the exception to the awaiter, but
            # return normally so the scheduler runs ``save_session``
            # for us (history was mutated in place by ``session.prompt``
            # even though it raised).
            self._delete_item_silently(inbox, item.id)
            if result_future is not None and not result_future.done():
                result_future.set_exception(exc)
            return
        except BaseException as exc:
            # Cancellation, KeyboardInterrupt, programming errors, etc.
            # Notify the awaiter (so it doesn't hang on shutdown) and
            # let the scheduler observe the raise -- which means
            # ``save_session`` is skipped, matching the prior chat
            # behaviour for non-Thorn exceptions (the REPL would have
            # let them propagate up to ``asyncio.run`` and exit).
            self._delete_item_silently(inbox, item.id)
            if result_future is not None and not result_future.done():
                result_future.set_exception(exc)
            raise
        self._delete_item_silently(inbox, item.id)
        if result_future is not None and not result_future.done():
            result_future.set_result(result)

    @staticmethod
    def _delete_item_silently(inbox: SessionInbox, item_id: Any) -> None:
        """Delete an inbox item, ignoring a concurrent removal.

        Mirrors the same defensive pattern used by
        :func:`~thorn.runtime._cli_dispatcher.make_cli_prompt_dispatcher`
        so that progress accounting in the scheduler's drain loop sees
        a closed-out set even if a parallel handler beat us to the
        delete.
        """
        try:
            inbox.delete(item_id)
        except KeyError:
            pass


__all__ = ["ChatPromptRouter"]
