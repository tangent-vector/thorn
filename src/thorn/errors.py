"""Exception hierarchy for thorn (re-export from thorn.core.errors).

This module is a backward-compatibility shim.  The canonical definitions
now live in ``thorn.core.errors``.
"""

from thorn.core.errors import (
    AgentFailureError,
    LoopLimitError,
    LoopRepetitionError,
    ProviderError,
    ProviderUnavailableError,
    RateLimitError,
    SkillError,
    ThornError,
    TransientProviderError,
)

__all__ = [
    "AgentFailureError",
    "LoopLimitError",
    "LoopRepetitionError",
    "ProviderError",
    "ProviderUnavailableError",
    "RateLimitError",
    "SkillError",
    "ThornError",
    "TransientProviderError",
]
