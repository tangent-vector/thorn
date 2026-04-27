"""First-class agent roles bundled with Thorn.

This package collects :class:`~thorn.core._agent.Agent` subclasses that
ship as part of the framework rather than being defined per-deployment
in operator code.  It currently holds the local coding agent that
backs ``thorn run`` / ``thorn chat``; future built-in roles
(maintenance, research, etc.) will land alongside it.

The longer-term direction is for agent roles to be predominantly
data-driven rather than represented as Python classes.  Until that
mechanism exists the class-per-role pattern stays, but new role
definitions belong here -- not buried inside CLI or gateway internals
-- so the eventual migration to data-driven roles has a single
inventory to consult.
"""

from __future__ import annotations

from thorn.agents._local_coding_agent import LocalCodingAgent

__all__ = [
    "LocalCodingAgent",
]
