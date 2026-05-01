"""Single source of truth for which built-in tools live in which venue.

The brain (``thorn.core._func._known_builtin_tools``) and the toolhost
daemon (``thorn.toolhost._server.build_default_registry``) historically
each maintained their own ad-hoc lists of "the tools we know about".
That arrangement let drift accumulate silently: when a new toolset was
added on the brain side without updating the daemon (or vice versa),
sandbox-bound calls would fail with ``unknown_tool`` only at first
invocation in production.

This module fixes the layering by publishing the canonical lists in one
place.  ``IN_PROCESS_TOOLS`` is the set of tools that need brain-side
state (the runtime, peer registry, agency credentials, inbox, ...) and
must execute in the gateway/CLI process.  ``SANDBOXED_TOOLS`` is the
set of tools that the toolhost daemon dispatches inside the agent's
container -- they take untrusted arguments, perform filesystem / shell
work, and never reach into runtime state.

Both lists are flat, deduplicated, and ordered for stable iteration.
The brain consumes their union; the daemon consumes only the sandboxed
slice.  Adding a new tool means appending to exactly one of these
lists, with no parallel update on the daemon side.
"""

from __future__ import annotations

from typing import Any

from thorn.core._journal import JOURNAL_TOOLS
from thorn.core._tools import ALL_BUILTIN_TOOLS, run_shell, write_file
from thorn.runtime._inbox_tools import INBOX_TOOLS
from thorn.tools.forge import FORGE_TOOLS
from thorn.tools.peers import PEER_TOOLS


def _dedup(items: list[Any]) -> list[Any]:
    """Return *items* with duplicates removed, preserving first-seen order.

    Built-in toolset constants overlap (e.g. ``ALL_BUILTIN_TOOLS``
    contains ``read_file``, but a future toolset might too); keeping
    the catalog deduplicated avoids registering the same tool twice
    in the daemon dispatch table.
    """
    seen: set[Any] = set()
    out: list[Any] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


IN_PROCESS_TOOLS: list[Any] = _dedup(
    [
        *INBOX_TOOLS,
        *FORGE_TOOLS,
        *PEER_TOOLS,
    ]
)
"""Tools that must execute in the brain process.

These reach into runtime state -- the inbox, the agency-managed forge
clients and credential broker, the peer registry -- that the toolhost
daemon's process does not have.  They never go through the daemon
dispatch path.
"""


SANDBOXED_TOOLS: list[Any] = _dedup(
    [
        *ALL_BUILTIN_TOOLS,
        run_shell,
        write_file,
        *JOURNAL_TOOLS,
    ]
)
"""Tools that the toolhost daemon dispatches inside the agent's sandbox.

These take agent-supplied arguments and perform filesystem / shell
work; the sandbox is what bounds the blast radius of a malicious or
confused tool argument.  Nothing in this list may rely on brain-side
runtime state (no ``ctx.runtime`` access, no agency lookups).
"""


ALL_BUILTIN_TOOL_FUNCTIONS: list[Any] = _dedup(
    [*IN_PROCESS_TOOLS, *SANDBOXED_TOOLS]
)
"""Union of every built-in tool the brain knows about.

Used by :func:`thorn.core._func._known_builtin_tools` to drive the
``_prepare_tools`` allowlist.  Anything not in this list is rejected
as "not a registered Thorn tool" when an agent tries to declare it.
"""


__all__ = [
    "IN_PROCESS_TOOLS",
    "SANDBOXED_TOOLS",
    "ALL_BUILTIN_TOOL_FUNCTIONS",
]
