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
import json
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from thorn.core._loop import ToolRoundTermination
from thorn.core._messages import ToolCall, ToolResultMessage
from thorn.runtime._address import AddressBook
from thorn.runtime._dispatch import apply_handling_transition
from thorn.runtime._inbox import SessionInbox
from thorn.runtime._notification import (
    InboxCompletionRationale,
    Notification,
    NotificationID,
    NotificationStatus,
)
from thorn.runtime._prompt_format import summarize_notification_content
from thorn.runtime._scheduler import PromptDispatcher
from thorn.runtime._working_set import HandlingPhase, WorkingSet

if TYPE_CHECKING:
    from thorn.core._session import Session


log = logging.getLogger(__name__)


@dataclass(frozen=True)
class _FreshDirectCompletionPolicy:
    """End a fresh direct prompt after its durable successful closeout."""

    inbox: SessionInbox
    notification_id: NotificationID

    def evaluate(
        self,
        *,
        assistant_text: str,
        tool_calls: tuple[ToolCall, ...],
        result_messages: tuple[ToolResultMessage, ...],
        session: Any,
    ) -> ToolRoundTermination | None:
        if len(tool_calls) != 1:
            return None
        [tool_call] = tool_calls
        if tool_call.name != "complete_focused_work":
            return None

        matching_results = [
            result
            for result in result_messages
            if result.call_id == tool_call.call_id
        ]
        if len(matching_results) != 1 or matching_results[0].is_error:
            return None

        try:
            self.inbox.get(self.notification_id)
        except KeyError:
            pass
        else:
            return None
        if session.working_set != WorkingSet():
            return None

        try:
            arguments = json.loads(tool_call.arguments)
            completion = InboxCompletionRationale.from_json(
                arguments["completion"],
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            return None
        if completion.validation_errors():
            return None

        final_text = assistant_text.strip()
        if not final_text:
            final_text = completion.to_display_text()
        return ToolRoundTermination(text=final_text)


def establish_fresh_cli_direct_focus(
    *,
    session: "Session",
    inbox: SessionInbox,
    notification: Notification,
    address_book: AddressBook,
) -> NotificationID | None:
    """Focus the sole fresh direct request before its first provider call.

    A fresh ``thorn run`` invocation already knows which notification
    contains the user's one-shot request.  Making the model rediscover,
    read, and focus that same item spends several provider rounds without
    adding information.  This helper performs the equivalent durable
    inbox and working-set transition when the session is still at clean
    intake and *notification* is its only prompt-pending item.

    Returns the focused notification ID when the transition was applied,
    or ``None`` when the preconditions do not hold.  Callers must persist
    the mutated session before submitting it to the scheduler.  Resumed
    sessions deliberately do not call this helper: their existing focus
    and queue state must remain agent-controlled.
    """
    working_set = session.working_set
    if working_set.phase is not HandlingPhase.INTAKE:
        return None
    if working_set.focused_inbox_item_id is not None:
        return None
    pending_items = inbox.prompt_pending()
    if len(pending_items) != 1:
        return None
    [pending_item] = pending_items
    if pending_item.id != notification.id:
        return None
    if pending_item.status is not NotificationStatus.PENDING:
        return None

    apply_handling_transition(
        inbox,
        notification.id,
        NotificationStatus.IN_PROGRESS,
        address_book=address_book,
    )
    session.working_set = WorkingSet(
        phase=HandlingPhase.INSPECT,
        focused_inbox_item_id=notification.id,
        objective=summarize_notification_content(notification),
    )
    return notification.id


def _reset_framework_focus(
    session: "Session",
    notification_id: NotificationID | None,
) -> None:
    """Avoid persisting focus on a direct item the dispatcher deletes."""
    if notification_id is None:
        return
    if session.working_set.focused_inbox_item_id != notification_id:
        return
    session.working_set = WorkingSet()


def make_cli_prompt_dispatcher(
    *,
    result_future: "asyncio.Future[Any]",
    extra_system: str | None = None,
    framework_focused_item_id: NotificationID | None = None,
    terminal_completion_item_id: NotificationID | None = None,
) -> PromptDispatcher:
    """Build a :data:`PromptDispatcher` that reports its result via *result_future*.

    Intended for one-shot CLI commands (``thorn run``) that post a
    single notification, run one prompt round, and want the round's
    text result returned synchronously.  *result_future* must be a
    fresh future the caller will ``await`` exactly once.

    *extra_system* is forwarded to ``session.prompt(system=...)`` on
    every invocation, so the caller can carry a fixed
    invocation-mode steering prompt (e.g. "you are executing a
    single non-interactive request") for the dispatcher's lifetime.
    Tool registration is *not* a dispatcher concern -- the agent's
    role declares its tool set via :meth:`Agent._collect_tools` and
    the per-prompt context-gathering pipeline contributes any
    additional MCP / skill tools.

    *framework_focused_item_id* identifies a notification focused by
    :func:`establish_fresh_cli_direct_focus`.  If the model returns or
    raises without closing that item through the inbox tools, the
    dispatcher resets the working set before deleting the direct item.
    This keeps a later explicit resume internally consistent.

    *terminal_completion_item_id* enables a fresh-direct terminal policy
    for that exact notification.  A successful, single-call
    ``complete_focused_work`` round can then return its accompanying text
    or structured completion rationale without another provider request.

    Behaviour per invocation:

    - If the inbox is empty, returns silently.  Matches the
      :func:`~thorn.runtime._prompt_format.inbox_prompt_dispatcher`
      no-op convention so the scheduler can re-poll without surprise.
    - Otherwise takes the oldest pending item, calls
      ``await session.prompt(item.content, system=...)``, and
      removes the item from the inbox so the driver's
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
            prompt_kwargs: dict[str, Any] = {"system": extra_system}
            if terminal_completion_item_id == item.id:
                prompt_kwargs["tool_round_terminal_policy"] = (
                    _FreshDirectCompletionPolicy(
                        inbox=inbox,
                        notification_id=item.id,
                    )
                )
            result = await session.prompt(item.content, **prompt_kwargs)
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
            _reset_framework_focus(session, framework_focused_item_id)
            if not result_future.done():
                result_future.set_exception(exc)
            raise
        try:
            inbox.delete(item.id)
        except KeyError:
            pass
        _reset_framework_focus(session, framework_focused_item_id)
        if not result_future.done():
            result_future.set_result(result)

    return dispatch


__all__ = [
    "establish_fresh_cli_direct_focus",
    "make_cli_prompt_dispatcher",
]
