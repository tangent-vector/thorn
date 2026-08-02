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


class LocalCodingAgent(Agent):
    """Default agent role for interactive and one-shot CLI use.

    Bundles the file-I/O and shell tool sets that any useful coding
    agent needs to operate against a workspace.  Git operations are
    performed via ``run_shell`` invoking ``git`` directly, rather than
    through a dedicated git tool API; this keeps the agent's surface
    aligned with how a human collaborator would drive the same
    repository (and avoids duplicating ``git``'s CLI as a parallel
    tool API that we would have to keep in lockstep with the real
    binary).

    No system prompts are declared on the role itself -- per-invocation
    steering (one-shot vs. REPL chat vs. ...) is supplied by the
    caller via the dispatcher's ``extra_system`` argument, since that
    is a property of *how* the prompt is being driven rather than
    *who* the agent is.
    """

    tools: ClassVar[list[Any]] = [FILE_WRITING, run_shell]


__all__ = ["LocalCodingAgent"]
