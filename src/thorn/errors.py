"""Exception hierarchy for thorn (re-export from thorn.core.errors).

This module is a backward-compatibility shim.  The canonical definitions
now live in ``thorn.core.errors``.
"""

from thorn.core.errors import (
    AgentFailureError,
    LoopLimitError,
    ProviderError,
    RateLimitError,
    SkillError,
    ThornError,
)

__all__ = [
    "AgentFailureError",
    "LoopLimitError",
    "ProviderError",
    "RateLimitError",
    "SkillError",
    "ThornError",
]
