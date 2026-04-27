"""The default agent role backing ``thorn run`` and ``thorn chat``.

A bare :class:`~thorn.core._agent.Agent` instance carries no tools.
The local coding agent is the role that the CLI commands instantiate
under :data:`thorn._cli.CLI_AGENT_ID` so that the agent itself owns
the standard "useful coding agent" tool set rather than having the
CLI smuggle it in through a per-invocation ``extra_tools`` channel
on the prompt dispatcher.

Keeping the tool kit on the role class means it flows through the
ordinary :meth:`Agent._collect_tools` MRO walk -- the same path used
by every other role -- which lets the dispatcher / router stay
agnostic about what tools any particular session actually has.
"""

from __future__ import annotations

from typing import Any, ClassVar

from thorn.core._agent import Agent
from thorn.core._tools import FILE_WRITING, run_shell
from thorn.tools.git import GIT_TOOLS


class LocalCodingAgent(Agent):
    """Default agent role for interactive and one-shot CLI use.

    Bundles the file-I/O, shell, and git tool sets that any useful
    coding agent needs to operate against a workspace.  No system
    prompts are declared on the role itself -- per-invocation steering
    (one-shot vs. REPL chat vs. ...) is supplied by the caller via
    the dispatcher's ``extra_system`` argument, since that is a
    property of *how* the prompt is being driven rather than *who*
    the agent is.
    """

    tools: ClassVar[list[Any]] = [FILE_WRITING, run_shell, GIT_TOOLS]


__all__ = ["LocalCodingAgent"]
