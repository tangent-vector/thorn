"""Exception hierarchy for thorn (re-export from thorn.core.errors).

This module is a backward-compatibility shim.  The canonical definitions
now live in ``thorn.core.errors``.
"""

from thorn.core.errors import (
    AgentFailureError,
    LoopLimitError,
    LoopNoProgressError,
    LoopRepetitionError,
    ProviderError,
    ProviderFailureKind,
    ProviderUnavailableError,
    RateLimitError,
    SkillError,
    ThornError,
    TransientProviderError,
)

__all__ = [
    "AgentFailureError",
    "LoopLimitError",
    "LoopNoProgressError",
    "LoopRepetitionError",
    "ProviderError",
    "ProviderFailureKind",
    "ProviderUnavailableError",
    "RateLimitError",
    "SkillError",
    "ThornError",
    "TransientProviderError",
]
