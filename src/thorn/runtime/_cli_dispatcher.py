"""CLI prompt dispatcher for one-shot, request/reply CLI commands.

The CLI commands (currently ``thorn run``; ``thorn chat`` will follow
in Phase 4 of the CLI/gateway unification) post their user input as a
single notification on a session inbox and expect the result of the
agent's response back synchronously, in contrast with the gateway's
fire-and-forget pattern.

This module provides :func:`make_cli_prompt_dispatcher`, a factory
that produces a :data:`~thorn.runtime._scheduler.PromptDispatcher`
paired with an :class:`asyncio.Future` the caller can await for the
round's outcome.

Why a factory and not a notification-attached future
----------------------------------------------------

The future is in-process and not serialisable.  Attaching it to a
:class:`~thorn.runtime._notification.NotificationSpec` would either
pollute that type with a non-serialisable field or smuggle it through
the metadata dict, both of which would couple the durable-queue layer
to caller-side request/reply state.  Keeping the future inside the
dispatcher's closure keeps the notification surface clean and makes
the request/reply scope a local concern of the caller that
constructed the dispatcher.

"Round complete" semantics
--------------------------

For ``thorn run``, "round complete" is defined as "the dispatcher
returned from ``session.prompt``".  The future resolves at that
point.  Notifications the agent might autonomously enqueue during the
round (it currently cannot, but the inbox model permits it) are left
in the inbox; the scheduler shutdown discards them along with the
session.  This is the simplest viable answer to Open Question 1 in
the unification plan and is appropriate for one-shot CLI commands;
the chat REPL refactor in Phase 4 will need to revisit it.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from thorn.runtime._inbox import SessionInbox
from thorn.runtime._scheduler import PromptDispatcher

if TYPE_CHECKING:
    from thorn.core._session import Session


log = logging.getLogger(__name__)


def make_cli_prompt_dispatcher(
    *,
    result_future: "asyncio.Future[Any]",
    extra_tools: list[Any] | None = None,
    extra_system: str | None = None,
) -> PromptDispatcher:
    """Build a :data:`PromptDispatcher` that reports its result via *result_future*.

    Intended for one-shot CLI commands (``thorn run``) that post a
    single notification, run one prompt round, and want the round's
    text result returned synchronously.  *result_future* must be a
    fresh future the caller will ``await`` exactly once.

    *extra_tools* and *extra_system* are forwarded to
    ``session.prompt(tools=..., system=...)`` on every invocation, so
    a long-lived dispatcher (multi-turn chat in Phase 4) can carry a
    fixed tool set and system prompt for the whole REPL.

    Behaviour per invocation:

    - If the inbox is empty, returns silently.  Matches the
      :func:`~thorn.runtime._prompt_format.inbox_prompt_dispatcher`
      no-op convention so the scheduler can re-poll without surprise.
    - Otherwise takes the oldest pending item, calls
      ``await session.prompt(item.content, tools=..., system=...)``,
      and removes the item from the inbox so the driver's
      ``_run_one_round`` accounting records the round as making
      progress (``closed_out`` includes the item's id).
    - Resolves *result_future* with the prompt's return value, or
      with the raised exception if the prompt raised.

    Resolution is one-shot.  A second invocation that finds another
    item in the inbox runs the prompt as usual but does not touch the
    already-resolved future.  This is defensive against unexpected
    re-entry; for ``thorn run`` exactly one notification is ever
    posted and the dispatcher runs exactly once.
    """

    async def dispatch(session: "Session", inbox: SessionInbox) -> None:
        items = inbox.prompt_pending()
        if not items:
            log.debug(
                "cli_prompt_dispatcher: no items pending for agent=%r session=%r",
                session.agent.id, session.key,
            )
            return
        item = items[0]
        try:
            result = await session.prompt(
                item.content,
                tools=extra_tools,
                system=extra_system,
            )
        except BaseException as exc:
            # Remove the item before re-raising so the driver's
            # progress accounting sees a closed_out set; otherwise it
            # would treat the round as a stall and eventually evict
            # the item via the progress guarantee.  The future carries
            # the exception so the awaiting caller learns about it
            # exactly once (set_exception on an already-done future
            # would itself raise).
            try:
                inbox.delete(item.id)
            except KeyError:
                pass
            if not result_future.done():
                result_future.set_exception(exc)
            raise
        try:
            inbox.delete(item.id)
        except KeyError:
            pass
        if not result_future.done():
            result_future.set_result(result)

    return dispatch


__all__ = ["make_cli_prompt_dispatcher"]
